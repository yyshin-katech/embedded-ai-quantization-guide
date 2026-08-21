#!/usr/bin/env python3
# orin_detr_map.py (ON BOARD, ~/orin_bench) — dump raw DETR outputs for each saved .plan
# engine over a COCO subset, so the HOST can compute pycocotools mAP (the board has no
# pycocotools). Mirrors accuracy/orin_accuracy.py (tensorrt + cuda.bindings.runtime,
# batch1, execute_async_v3) but DETR has 4 outputs — we keep only logits [1,100,92] and
# pred_boxes [1,100,4], and MUST .copy() each into a preallocated array (the host output
# buffer is reused every iteration — the stage5 zero-copy aliasing lesson).
#
# The INT8 engines under test:
#   detr_gpu_int8_sym.plan   = explicit symmetric QDQ (Conv+Gemm), real calib scales  -> ACCURACY-VALID
#   detr_gpu_int8_implicit.plan = trtexec --int8 auto dynamic-range                    -> accuracy NOT claimed
# Preprocess = detr_prep.preprocess (force-resize to fixed 800x1066, ImageNet norm),
# byte-identical to the host-side calibration.
import argparse, json, os, sys
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detr_prep import preprocess

TRT_NP = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
          trt.DataType.INT32: np.int32, trt.DataType.INT8: np.int8}
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def ck(ret):
    err = ret[0] if isinstance(ret, tuple) else ret
    rest = ret[1:] if isinstance(ret, tuple) else ()
    if int(err) != 0:
        raise RuntimeError("CUDA error %d" % int(err))
    return rest


def load_engine(path):
    with open(path, "rb") as f:
        data = f.read()
    eng = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(data)
    if eng is None:
        raise RuntimeError("deserialize failed: %s" % path)
    return eng


def run_engine(path, paths):
    eng = load_engine(path)
    ctx = eng.create_execution_context()
    ios = []
    for i in range(eng.num_io_tensors):
        nm = eng.get_tensor_name(i)
        mode = eng.get_tensor_mode(nm)
        ios.append((nm, mode))
    in_name = [nm for nm, m in ios if m == trt.TensorIOMode.INPUT][0]
    out_names = [nm for nm, m in ios if m == trt.TensorIOMode.OUTPUT]

    in_shape = tuple(eng.get_tensor_shape(in_name))            # (1,3,800,1066)
    in_dt = TRT_NP[eng.get_tensor_dtype(in_name)]
    in_nbytes = int(np.prod(in_shape)) * np.dtype(in_dt).itemsize
    d_in = ck(cudart.cudaMalloc(in_nbytes))[0]
    ctx.set_tensor_address(in_name, int(d_in))

    # allocate every output (TRT needs all addresses); remember logits/boxes by shape
    outs = {}      # name -> (d_ptr, host_buf, shape, dt)
    logits_name = boxes_name = None
    for nm in out_names:
        osh = tuple(ctx.get_tensor_shape(nm))
        odt = TRT_NP[eng.get_tensor_dtype(nm)]
        nb = int(np.prod(osh)) * np.dtype(odt).itemsize
        dptr = ck(cudart.cudaMalloc(nb))[0]
        ctx.set_tensor_address(nm, int(dptr))
        hb = np.empty(osh, dtype=odt)
        outs[nm] = (dptr, hb, osh, odt, nb)
        if osh[-2:] == (100, 92):
            logits_name = nm
        elif osh[-2:] == (100, 4):
            boxes_name = nm
    assert logits_name and boxes_name, "could not find logits/boxes outputs: %s" % [outs[n][2] for n in out_names]

    N = len(paths)
    all_logits = np.empty((N, 100, 92), dtype=np.float32)
    all_boxes = np.empty((N, 100, 4), dtype=np.float32)
    H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for i, p in enumerate(paths):
        x = preprocess(p).astype(in_dt, copy=False)
        host_in = np.ascontiguousarray(x)
        ck(cudart.cudaMemcpy(int(d_in), host_in.ctypes.data, in_nbytes, H2D))
        if not ctx.execute_async_v3(0):
            raise RuntimeError("execute_async_v3 failed at img %d" % i)
        ck(cudart.cudaDeviceSynchronize())
        for nm in (logits_name, boxes_name):
            dptr, hb, osh, odt, nb = outs[nm]
            ck(cudart.cudaMemcpy(hb.ctypes.data, int(dptr), nb, D2H))
        all_logits[i] = outs[logits_name][1].reshape(100, 92).astype(np.float32)   # .copy via astype
        all_boxes[i] = outs[boxes_name][1].reshape(100, 4).astype(np.float32)
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, N), flush=True)

    ck(cudart.cudaFree(d_in))
    for nm in out_names:
        ck(cudart.cudaFree(outs[nm][0]))
    return all_logits, all_boxes, list(in_shape), str(in_dt.__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines-dir", default=os.path.expanduser("~/orin_bench/engines"))
    ap.add_argument("--img-dir", default=os.path.expanduser("~/orin_bench/coco_sub"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/orin_bench/coco_sub/manifest.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/orin_bench/detr_accuracy"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(open(args.manifest))               # [{image_id, file_name}, ...]
    paths = [os.path.join(args.img_dir, m["file_name"]) for m in manifest]
    img_ids = [int(m["image_id"]) for m in manifest]
    print("eval images:", len(paths), flush=True)

    engines = [
        ("gpu_fp32", "detr_gpu_fp32.plan", True),
        ("gpu_fp16", "detr_gpu_fp16.plan", True),
        ("gpu_int8_sym", "detr_gpu_int8_sym.plan", True),        # explicit symmetric QDQ -> accuracy-valid
        ("gpu_int8_implicit", "detr_gpu_int8_implicit.plan", False),  # --int8 auto-range -> NOT claimed
    ]
    meta_engines = []
    for tag, fn, acc_valid in engines:
        p = os.path.join(args.engines_dir, fn)
        if not os.path.exists(p):
            print("MISSING", p, flush=True)
            continue
        print("[%s] %s  (accuracy_valid=%s)" % (tag, fn, acc_valid), flush=True)
        lg, bx, ish, idt = run_engine(p, paths)
        np.savez_compressed(os.path.join(args.out, "detr_%s_raw.npz" % tag),
                            logits=lg, boxes=bx, img_ids=np.array(img_ids, dtype=np.int64))
        meta_engines.append({"tag": tag, "engine": fn, "accuracy_valid": acc_valid,
                             "in_shape": ish, "in_dtype": idt, "n": len(paths)})

    meta = {"board": "NVIDIA Jetson AGX Orin Developer Kit (64GB)",
            "trt": trt.__version__, "cuda_python": "12.9.7", "nvpmodel": "MAXN",
            "n_eval": len(paths), "preprocess": "force-resize 800x1066, ImageNet norm (detr_prep.py)",
            "engines": meta_engines,
            "note": "int8_sym built from detr_int8_sym.onnx (symmetric QInt8 QDQ, Conv+Gemm, "
                    "QuantizeBias=False, exclude backbone conv1); int8_implicit = trtexec --int8 auto-range"}
    json.dump(meta, open(os.path.join(args.out, "orin_detr_meta.json"), "w"), indent=2)
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
