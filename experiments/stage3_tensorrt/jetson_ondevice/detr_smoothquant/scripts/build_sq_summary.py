#!/usr/bin/env python3
# build_sq_summary.py (HOST) — single-source-of-truth for the on-device SmoothQuant DETR run.
# mAP/recovery/divergence come from analyze_detr_sq_map.py (detr_sq_map_summary.json);
# FP32 + sym reference latency/bytes/mAP come from the COMMITTED stage3 DETR-accuracy SSOT
# (detr_accuracy/results/detr_accuracy_summary.json, 58ba518); the NEW sq_a10/sq_a05
# latency + engine bytes are THIS session's trtexec builds (from raw/detr_sq_build_*.log,
# identical flags). Compression/verify/exactness come from the export meta. All ratios
# recomputed here so the report never hard-codes arithmetic.
import json, os

BASE = "experiments/stage3_tensorrt/jetson_ondevice"
SQD  = os.path.join(BASE, "detr_smoothquant/results")
MAP  = json.load(open(os.path.join(SQD, "detr_sq_map_summary.json")))
ACC  = json.load(open(os.path.join(BASE, "detr_accuracy/results/detr_accuracy_summary.json")))  # committed 58ba518
MA10 = json.load(open(os.path.join(SQD, "sq_export_meta_a10.json")))
MA05 = json.load(open(os.path.join(SQD, "sq_export_meta_a05.json")))

# THIS session's trtexec builds (GPU Compute median ms, engine bytes) — raw/detr_sq_build_*.log
SQ_BUILD = {"sq_a10": (11.3433, 47052636), "sq_a05": (11.4084, 47092836)}

# reference rows straight from the committed accuracy SSOT (same board/flags/subset)
ref = {}
for tag in ("gpu_fp32", "gpu_int8_sym"):
    e = ACC["engines"][tag]
    ref[tag] = {"label": e["label"], "latency_ms": e["latency_ms"], "engine_bytes": e["engine_bytes"],
                "engine_mib": e["engine_mib"], "mAP": e["mAP"], "mAP_s": e["mAP_s"]}

engines = {}
for tag in ("gpu_fp32", "gpu_int8_sym"):
    engines[tag] = dict(ref[tag], accuracy_valid=True)
for tag, meta in (("sq_a10", MA10), ("sq_a05", MA05)):
    lat, by = SQ_BUILD[tag]
    m = MAP[tag]
    engines[tag] = {
        "label": f"INT8 SmoothQuant α={meta['alpha']} + sym QDQ (accuracy-valid)",
        "latency_ms": lat, "engine_bytes": by, "engine_mib": round(by / 1048576, 2),
        "mAP": m["mAP"], "mAP50": m["mAP50"], "mAP75": m["mAP75"],
        "mAP_s": m["mAP_s"], "mAP_m": m["mAP_m"], "mAP_l": m["mAP_l"],
        "accuracy_valid": True,
        "smoothquant": {"alpha": meta["alpha"], "n_gemms": meta["n_gemms"],
                        "compression_max": meta["compression_summary"]["max"]["compression"],
                        "compression_max_gemm": meta["compression_summary"]["max"]["gemm"],
                        "compression_median": meta["compression_summary"]["median_ratio"],
                        "n_ge_2x": meta["compression_summary"]["n_gemms_compressed_ge_2x"],
                        "exactness_max_abs": meta["exactness"]["max_abs_diff"],
                        "verify": meta["verify"], "onnx_qdq_bytes": meta["onnx_qdq_bytes"]},
    }

fp32 = engines["gpu_fp32"]["mAP"]; sym = engines["gpu_int8_sym"]["mAP"]
fp32_lat = engines["gpu_fp32"]["latency_ms"]; sym_lat = engines["gpu_int8_sym"]["latency_ms"]
gap = round(fp32 - sym, 4)
R = MAP["_recovery"]

derived = {"collapse_gap": gap}
for tag in ("sq_a10", "sq_a05"):
    e = engines[tag]
    derived[tag] = {
        "recovery_pct_of_gap": R[tag]["recovery_pct_of_gap"],
        "maps_recovery_pct_of_gap": R[tag]["maps_recovery_pct_of_gap"],
        "delta_vs_sym_abs": R[tag]["delta_vs_sym_abs"],
        "delta_vs_fp32_pct": R[tag]["delta_vs_fp32_pct"],
        "speedup_vs_fp32": round(fp32_lat / e["latency_ms"], 3),
        "latency_vs_sym_ms": round(e["latency_ms"] - sym_lat, 4),
    }

summary = {
    "board": "NVIDIA Jetson AGX Orin Developer Kit (64GB), JetPack 6.2.1, TensorRT 10.3.0, MAXN",
    "task": "On-device SmoothQuant on DETR (facebook/detr-resnet-50): does activation-granularity "
            "smoothing recover the accuracy-valid explicit-sym INT8 collapse (58ba518: 0.4237->0.2383)?",
    "method": {
        "smoothquant": "ONNX-level, manual. All 95 DETR Gemms are transB=0 (Y=A·B, weight [K,N]); "
                       "per-input-channel(K) scale s_k = a_k^α / w_k^(1-α); inject Mul(A,1/s) + scale "
                       "weight rows (axis0) by s -> A'·B'=A·B exact (FP32 gate max_abs<1e-2, CPU EP).",
        "why_manual": "modelopt.onnx not importable (onnxslim missing); stage2 §4.4 SmoothQuant lived only "
                      "on the torch fake-quant path (Design X) and never produced an ONNX -> never on-device.",
        "quant": "identical sym recipe as detr_int8_sym.onnx: QDQ QInt8 act+weight, per_channel, "
                 "op_types_to_quantize=[Conv,Gemm], QuantizeBias=False, ActivationSymmetric+WeightSymmetric, "
                 "exclude backbone conv1. The ONLY delta vs the committed sym engine is the SmoothQuant Muls.",
        "eval": "COCO val2017 head 1000, fixed 800x1066, pycocotools bbox mAP (board dumps raw logits/boxes). "
                "FP32/sym reproduced from committed npz with the SAME postprocess (0.4237/0.2383 -> no drift).",
        "alphas": "a10=1.0 (stage2 §4.4 DETR-best), a05=0.5.",
    },
    "engines": engines,
    "recovery": {"fp32_mAP": fp32, "sym_mAP": sym, "gap": gap,
                 "fp32_mAP_s": R["fp32_mAP_s"], "sym_mAP_s": R["sym_mAP_s"], **derived},
    "divergence_vs_fp32": MAP["_divergence_vs_fp32"],
    "sq_vs_sym_absmax": MAP["_sq_vs_sym_absmax"],
    "stage2_crossval": {
        "stage2_recovery_pct": 59.9, "stage2_path": "torch fake-quant (Design X), per-tensor INT8 on ALL "
            "linears incl. attention Q/K/V/out; gap 0.0908 (0.4209->0.3301) recovered +0.0544 to 0.3845 (§4.4)",
        "ondevice_recovery_pct_a10": derived["sq_a10"]["recovery_pct_of_gap"],
        "verdict": "The stage2 §4.4 59.9% recovery does NOT transfer to the on-device ONNX->TRT path: "
                   "SmoothQuant-on-Gemms recovers only ~9% of the (larger) on-device collapse. SmoothQuant is "
                   "still directionally correct (α=1.0 > α=0.5, matching stage2's α ordering; mAP_s partially "
                   "recovers) but modest. Root cause = op coverage: the ONLY buildable on-device INT8 engine "
                   "leaves attention MatMul + LayerNorm + Softmax in FP16 (forced by the case-C parser + "
                   "quantized-constant builder walls, stage2 §4.5), so INT8 is Gemm-only and smoothing the "
                   "Gemm inputs is not the dominant residual-error lever there. The collapse is small-object "
                   "dominated (mAP_s 0.2179->0.0336) and SmoothQuant barely moves it (->0.0449).",
    },
    "headline": f"On-device SmoothQuant (α=1.0, 95 Gemms, activation compression up to "
                f"{engines['sq_a10']['smoothquant']['compression_max']}× / median "
                f"{engines['sq_a10']['smoothquant']['compression_median']}×) recovers only "
                f"{derived['sq_a10']['recovery_pct_of_gap']}% of the DETR INT8 collapse "
                f"({sym}→{engines['sq_a10']['mAP']} of the {gap} gap) — an order of magnitude below stage2 "
                f"§4.4's 59.9% (torch fake-quant, all ops). Nearly latency-neutral (+{derived['sq_a10']['latency_vs_sym_ms']} "
                f"ms vs sym). The lever's effectiveness is path/op-coverage dependent; on the only buildable "
                f"on-device path (attention left FP16), Gemm-input smoothing is not enough.",
    "caveats": [
        "Absolute mAP not comparable to stage2 (fixed-800x1066 vs dynamic shape; TRT vs ORT torch; 1000 vs 5000 img). "
        "Only the RELATIVE recovery-of-gap is the result.",
        "1000-image head subset (stage1 함정 0 inflation applies); same-bundle relative deltas only.",
        "SmoothQuant is FP32-exact by construction (s cancels); the on-device benefit comes purely from tighter "
        "per-tensor activation quant after outlier migration — verified the engines are real INT8 (logits_corr "
        "~0.982 vs FP32) and genuinely differ from the sym engine (logits_absmax ~16 vs sym).",
        "Jetson is NVIDIA edge silicon, not one of the three automotive vendors (TI/Qualcomm/Renesas).",
    ],
}

OUT = os.path.join(SQD, "detr_sq_summary.json")
json.dump(summary, open(OUT, "w"), indent=2, ensure_ascii=False)
print("wrote", OUT)
for tag, e in engines.items():
    extra = ""
    if tag.startswith("sq_"):
        d = derived[tag]
        extra = "  recovery=%.1f%% Δvs_sym=%+.4f  (+%.4f ms vs sym)" % (
            d["recovery_pct_of_gap"], d["delta_vs_sym_abs"], d["latency_vs_sym_ms"])
    print("  %-14s lat=%8.4f ms  %6.2f MiB  mAP=%.4f mAP_s=%.4f%s"
          % (tag, e["latency_ms"], e["engine_mib"], e["mAP"], e["mAP_s"], extra))
print("  gap=%.4f  a10 recovery=%.1f%%  a05 recovery=%.1f%%  (stage2 §4.4 torch=59.9%%)"
      % (gap, derived["sq_a10"]["recovery_pct_of_gap"], derived["sq_a05"]["recovery_pct_of_gap"]))
