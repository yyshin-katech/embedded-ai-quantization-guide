#!/usr/bin/env python3
# detr_sq_export.py (HOST, emb-ai venv) — ONNX-level SmoothQuant on DETR's 95 Gemms, then
# the SAME symmetric QDQ recipe that let the accuracy-valid sym engine build (detr_sym_export.py).
#
# WHY manual ONNX-level SmoothQuant:
#   * stage2 §4.4 proved SmoothQuant on DETR but ONLY on the torch fake-quant path (Design X);
#     it never produced an ONNX, so it was never measured on-device (ONNX->TRT).
#   * modelopt.onnx is NOT importable in this venv (onnxslim missing, [onnx] extra) -> we do it by hand.
#
# THE MATH (all 95 DETR Gemms are transA=0, transB=0 -> Y = A · B, weight B = [K, N]):
#   contraction dim K = A.shape[-1] = B.shape[0] (weight ROWS).
#   per-input-channel scale over K:  s_k = a_k^alpha / w_k^(1-alpha)
#       a_k = per-channel(K) activation abs-max from calibration (100 tail images)
#       w_k = per-row(axis0) weight abs-max
#   migrate:  A'[:,k] = A[:,k] / s_k   (inject Mul(A, 1/s), broadcast on last axis)
#             B'[k,:] = B[k,:] * s_k   (scale weight ROW k)
#   -> A'·B' = sum_k (A[:,k]/s_k)(B[k,:]*s_k) = A·B   EXACT for any s_k>0 (s cancels).
#   The benefit is entirely on the activation side: per-tensor activation quant now sees a
#   compressed range (outlier channels divided down). With per-channel weights the weight-side
#   cost is minimal. This is stage2 §4.4's "activation granularity is the lever" — on-device.
#
# alpha=1.0 is stage2 §4.4's DETR-best (66.6% recovery > alpha=0.5 49.8%); we build both.
import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "detr_accuracy", "scripts"))
from detr_prep import preprocess                                   # shared, byte-identical
from pycocotools.coco import COCO
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--tag", required=True)                            # e.g. a10, a05
args = ap.parse_args()
ALPHA = args.alpha

SRC   = "_workspace/stage2/detr_sim.onnx"
WDIR  = "_workspace/stage3_jetson_detr_sq"
SMOOTH = os.path.join(WDIR, f"detr_sq_smoothed_fp32_{args.tag}.onnx")
DST    = os.path.join(WDIR, f"detr_int8_sq_{args.tag}.onnx")
ANN = "_workspace/coco/annotations/instances_val2017.json"
IMG = "_workspace/coco/val2017"
N_CALIB = 100
CLIP_LO, CLIP_HI, DEAD = 1e-2, 1e2, 1e-8
os.makedirs(WDIR, exist_ok=True)

coco = COCO(ANN)
img_ids = sorted(coco.getImgIds())
calib_ids = img_ids[-N_CALIB:]                                     # tail, disjoint from head-1000 eval
paths = [os.path.join(IMG, coco.loadImgs(i)[0]["file_name"]) for i in calib_ids]
print(f"[{args.tag}] alpha={ALPHA} | calib {len(paths)} (COCO val tail {N_CALIB})", flush=True)

# ---------------------------------------------------------------- load + map the 95 Gemms
m = onnx.load(SRC)
g = m.graph
inits = {i.name: i for i in g.initializer}
gemms = [n for n in g.node if n.op_type == "Gemm"]
assert len(gemms) == 95, len(gemms)
for n in gemms:                                                    # confirm the transB=0 premise
    at = {a.name: a.i for a in n.attribute}
    assert at.get("transA", 0) == 0 and at.get("transB", 0) == 0, (n.name, at)
    assert n.input[1] in inits, ("weight not initializer", n.name)

# unique activation tensors feeding Gemm input0 (14 are shared by 2 Gemms)
uniq_in = sorted({n.input[0] for n in gemms})
print(f"  Gemms={len(gemms)}  unique Gemm-input0 tensors={len(uniq_in)}", flush=True)

# ---------------------------------------------------------------- collect per-channel(K) activation abs-max
# add each unique Gemm-input0 as a graph output, run 100 calib images on CUDA, running max.
mm = onnx.load(SRC)
existing_out = {o.name for o in mm.graph.output}
for t in uniq_in:
    if t not in existing_out:
        mm.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
tmp = os.path.join(WDIR, "_detr_with_gemm_inputs.onnx")
onnx.save(mm, tmp, save_as_external_data=True, location="_detr_gemm_inputs_ext.bin",
          all_tensors_to_one_file=True, convert_attribute=False)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(tmp, providers=providers)
print("  stats session providers:", sess.get_providers(), flush=True)

absmax = {t: None for t in uniq_in}                                # per-unique-tensor [K] running abs-max
for idx, p in enumerate(paths):
    outs = sess.run(uniq_in, {"pixel_values": preprocess(p)})
    for t, arr in zip(uniq_in, outs):
        a = np.abs(arr.reshape(-1, arr.shape[-1])).max(axis=0)     # [K]
        absmax[t] = a if absmax[t] is None else np.maximum(absmax[t], a)
    if (idx + 1) % 25 == 0:
        print(f"    calib {idx+1}/{len(paths)}", flush=True)

# ---------------------------------------------------------------- compute s_k per Gemm, inject Mul + scale weight rows
mul_nodes = {}                                                     # gemm.name -> Mul NodeProto
sq_stats = []
for n in gemms:
    a_k = absmax[n.input[0]].astype(np.float64)                    # [K] activation abs-max
    W = numpy_helper.to_array(inits[n.input[1]]).astype(np.float64)  # [K, N]
    K = W.shape[0]
    assert a_k.shape[0] == K, (n.name, a_k.shape, W.shape)
    w_k = np.abs(W).max(axis=1)                                    # [K] per-row weight abs-max
    dead = (a_k < DEAD) | (w_k < DEAD)
    s = np.power(a_k, ALPHA) / np.power(w_k, 1.0 - ALPHA)
    s = np.clip(s, CLIP_LO, CLIP_HI)
    s[dead] = 1.0                                                  # skip dead / degenerate channels
    r = (1.0 / s).astype(np.float32)                               # activation multiplier

    # inject Mul(input0, r) -> new tensor; rewire this Gemm's input0
    rname = f"sq_recip_{n.name}"
    g.initializer.append(numpy_helper.from_array(r, name=rname))
    mout = f"sq_in_{n.name}"
    mul_nodes[n.name] = helper.make_node("Mul", [n.input[0], rname], [mout], name=f"SQMul_{n.name}")
    n.input[0] = mout

    # scale weight ROW k by s_k  (transB=0: B'[k,:] = B[k,:]*s_k)
    Wp = (W * s[:, None]).astype(np.float32)
    g.initializer.remove(inits[n.input[1]])
    newW = numpy_helper.from_array(Wp, name=n.input[1])
    g.initializer.append(newW); inits[n.input[1]] = newW

    pre = float(a_k.max()); post = float((a_k / s).max())          # per-tensor activation abs-max: before vs after
    sq_stats.append({"gemm": n.name, "K": int(K), "N": int(W.shape[1]),
                     "act_absmax_pre": round(pre, 4), "act_absmax_post": round(post, 4),
                     "compression": round(pre / post, 3) if post > 0 else None,
                     "s_min": round(float(s.min()), 4), "s_max": round(float(s.max()), 4),
                     "n_clamped": int(((s <= CLIP_LO + 1e-12) | (s >= CLIP_HI - 1e-9)).sum()),
                     "n_dead": int(dead.sum())})

# rebuild node list with each Mul inserted immediately before its Gemm (topological)
newnodes = []
for n in list(g.node):
    if n.op_type == "Gemm" and n.name in mul_nodes:
        newnodes.append(mul_nodes[n.name])
    newnodes.append(n)
del g.node[:]
g.node.extend(newnodes)
onnx.checker.check_model(m, full_check=False) if os.path.getsize(SRC) < 2e9 else None
onnx.save(m, SMOOTH, save_as_external_data=True, location=f"detr_sq_smoothed_{args.tag}_ext.bin",
          all_tensors_to_one_file=True, convert_attribute=False)
print(f"  wrote smoothed fp32 -> {SMOOTH}", flush=True)

# ---------------------------------------------------------------- FP32 EXACTNESS GATE (silent-wrong guard)
# SmoothQuant is numerically exact (s cancels). Verify original vs smoothed on 2 images.
# MUST run on CPUExecutionProvider: the CUDA EP uses TF32 matmul on Ampere, and TF32's
# 10-bit mantissa is sensitive to SmoothQuant's operand rescaling (activation ÷s, weight ×s)
# -> ~1e-2 spurious diff that is a precision artifact, NOT a wiring bug. CPU fp32 is the
# authoritative reference (gives ~1e-4, pure accumulation-reorder residual).
s0 = ort.InferenceSession(SRC,    providers=["CPUExecutionProvider"])
s1 = ort.InferenceSession(SMOOTH, providers=["CPUExecutionProvider"])
onames = [o.name for o in g.output]
max_abs = 0.0; max_rel = 0.0
for p in paths[:2]:
    x = {"pixel_values": preprocess(p)}
    r0 = {o.name: v for o, v in zip(s0.get_outputs(), s0.run(None, x))}
    r1 = {o.name: v for o, v in zip(s1.get_outputs(), s1.run(None, x))}
    for name in ["logits", "pred_boxes"]:
        d = np.abs(r0[name] - r1[name])
        max_abs = max(max_abs, float(d.max()))
        denom = np.abs(r0[name]).max()
        max_rel = max(max_rel, float(d.max() / denom) if denom > 0 else 0.0)
print(f"  EXACTNESS  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}", flush=True)
assert max_abs < 1e-2, f"SmoothQuant not FP32-exact (max_abs={max_abs}) -> silent-wrong risk"
print("  SMOOTH_EXACT_OK", flush=True)

# ---------------------------------------------------------------- quantize_static: identical sym recipe
class Reader(CalibrationDataReader):
    def __init__(self, paths): self.it = iter(paths)
    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {"pixel_values": preprocess(p)}

print("=== quantize_static: SmoothQuant + symmetric QInt8 QDQ (zp=0), QuantizeBias=False, per-channel ===", flush=True)
quantize_static(
    SMOOTH, DST,
    calibration_data_reader=Reader(paths),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["Conv", "Gemm"],                         # Mul stays FP (not listed)
    nodes_to_exclude=["/model/backbone/model/conv1/Conv"],
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False},
)
print("wrote", DST, round(os.path.getsize(DST) / 1e6, 1), "MB", flush=True)

# ---------------------------------------------------------------- verify sym invariants hold (case C/B cleared)
mq = onnx.load(DST); gq = mq.graph
qinits = {i.name: i for i in gq.initializer}
zp_total = zp_nonzero = int32_dq = 0
for n in gq.node:
    if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) >= 3 and n.input[2] in qinits:
        zp_total += 1
        if np.any(numpy_helper.to_array(qinits[n.input[2]]) != 0):
            zp_nonzero += 1
    if n.op_type == "DequantizeLinear" and n.input[0] in qinits and \
       numpy_helper.to_array(qinits[n.input[0]]).dtype == np.int32:
        int32_dq += 1
n_mul = sum(1 for n in gq.node if n.op_type == "Mul" and n.name.startswith("SQMul_"))
print(f"VERIFY  zp_total={zp_total}  zp_nonzero={zp_nonzero}  int32_bias_DQ={int32_dq}  SQMul_kept={n_mul}", flush=True)
assert zp_nonzero == 0 and int32_dq == 0, "sym invariants broken -> case C/B would fire"

# ---------------------------------------------------------------- record smoothing metadata (SSOT input)
comp = sorted(sq_stats, key=lambda d: -(d["compression"] or 0))
meta = {
    "alpha": ALPHA, "tag": args.tag, "n_gemms": len(gemms), "n_calib": N_CALIB,
    "onnx_smoothed_bytes": os.path.getsize(SMOOTH), "onnx_qdq_bytes": os.path.getsize(DST),
    "verify": {"zp_total": zp_total, "zp_nonzero": zp_nonzero, "int32_bias_DQ": int32_dq, "SQMul_kept": n_mul},
    "exactness": {"max_abs_diff": max_abs, "max_rel_diff": max_rel},
    "clip": [CLIP_LO, CLIP_HI],
    "compression_summary": {
        "max": comp[0], "median_ratio": round(float(np.median([d["compression"] for d in sq_stats if d["compression"]])), 3),
        "n_gemms_compressed_ge_2x": int(sum(1 for d in sq_stats if (d["compression"] or 0) >= 2.0)),
    },
    "top5_compressed": comp[:5],
    "per_gemm": sq_stats,
}
OUT = f"experiments/stage3_tensorrt/jetson_ondevice/detr_smoothquant/results/sq_export_meta_{args.tag}.json"
json.dump(meta, open(OUT, "w"), indent=2)
print("wrote", OUT, flush=True)
print(f"  compression: max {comp[0]['compression']}x ({comp[0]['gemm']}), median {meta['compression_summary']['median_ratio']}x, "
      f">=2x in {meta['compression_summary']['n_gemms_compressed_ge_2x']}/{len(gemms)} Gemms", flush=True)
print(f"SQ_EXPORT_OK [{args.tag}]", flush=True)
