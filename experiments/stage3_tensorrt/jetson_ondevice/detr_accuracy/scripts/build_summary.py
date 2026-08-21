#!/usr/bin/env python3
# build_summary.py (HOST) — assemble the single-source-of-truth summary for the DETR
# accuracy-valid INT8 follow-on. Accuracy comes from analyze_detr_map.py's output
# (detr_map_summary.json); latency + engine bytes for FP32/FP16/implicit are the values
# already COMMITTED in the stage3 DETR latency SSOT (detr/results/detr_summary.json,
# commit 9ef2a58, identical trtexec flags --warmUp=2000 --duration=10 --iterations=200
# --avgRuns=100); the NEW int8_sym latency/bytes come from this session's build. All
# derived ratios are recomputed here so the report never hard-codes arithmetic.
import json, os

BASE = "experiments/stage3_tensorrt/jetson_ondevice"
ACC = json.load(open(os.path.join(BASE, "detr_accuracy/results/detr_map_summary.json")))
LAT = json.load(open(os.path.join(BASE, "detr/results/detr_summary.json")))  # committed 9ef2a58

# latency (GPU Compute median ms) + engine bytes, batch1 MAXN, identical flags
LATENCY = {
    "gpu_fp32":         (LAT["gpu_fp32"]["gpu_compute_median_ms"],     LAT["gpu_fp32"]["engine_bytes"]),
    "gpu_fp16":         (LAT["gpu_fp16"]["gpu_compute_median_ms"],     LAT["gpu_fp16"]["engine_bytes"]),
    "gpu_int8_sym":     (11.002,                                       46859132),   # THIS session's build
    "gpu_int8_implicit":(LAT["gpu_int8_implicit"]["gpu_compute_median_ms"], LAT["gpu_int8_implicit"]["engine_bytes"]),
}
LABEL = {
    "gpu_fp32": "FP32", "gpu_fp16": "FP16",
    "gpu_int8_sym": "INT8 explicit-sym (accuracy-valid)",
    "gpu_int8_implicit": "INT8 implicit --int8 (auto-range, accuracy NOT claimed)",
}

engines = {}
for tag in ["gpu_fp32", "gpu_fp16", "gpu_int8_sym", "gpu_int8_implicit"]:
    lat, by = LATENCY[tag]
    a = ACC[tag]
    engines[tag] = {
        "label": LABEL[tag], "latency_ms": lat, "engine_bytes": by,
        "engine_mib": round(by / 1024 / 1024, 2),
        "mAP": a["mAP"], "mAP50": a["mAP50"], "mAP75": a["mAP75"],
        "mAP_s": a["mAP_s"], "mAP_m": a["mAP_m"], "mAP_l": a["mAP_l"],
        "accuracy_valid": a["accuracy_valid"],
    }

f32_lat = engines["gpu_fp32"]["latency_ms"]; f16_lat = engines["gpu_fp16"]["latency_ms"]
f32_map = engines["gpu_fp32"]["mAP"];        f32_maps = engines["gpu_fp32"]["mAP_s"]
sym = engines["gpu_int8_sym"]; imp = engines["gpu_int8_implicit"]

derived = {
    "fp16_speedup_vs_fp32":     round(f32_lat / f16_lat, 3),
    "sym_speedup_vs_fp32":      round(f32_lat / sym["latency_ms"], 3),
    "implicit_speedup_vs_fp32": round(f32_lat / imp["latency_ms"], 3),
    "sym_vs_fp16_speedup":      round(f16_lat / sym["latency_ms"], 3),
    "implicit_vs_sym_speedup":  round(sym["latency_ms"] / imp["latency_ms"], 3),
    "sym_map_drop_abs":  round(sym["mAP"] - f32_map, 4),
    "sym_map_drop_pct":  round((sym["mAP"] - f32_map) / f32_map * 100, 1),
    "sym_maps_drop_pct": round((sym["mAP_s"] - f32_maps) / f32_maps * 100, 1),
    "implicit_map_drop_abs": round(imp["mAP"] - f32_map, 4),
    "implicit_map_drop_pct": round((imp["mAP"] - f32_map) / f32_map * 100, 1),
    "fp16_map_delta_abs":    round(engines["gpu_fp16"]["mAP"] - f32_map, 4),
}

summary = {
    "board": "NVIDIA Jetson AGX Orin Developer Kit (64GB), JetPack 6.2.1, TensorRT 10.3.0, MAXN",
    "task": "DETR (facebook/detr-resnet-50) accuracy-valid explicit-QDQ INT8 on-device — close the "
            "'정확도 미측정' caveat the stage3 DETR latency run (9ef2a58) carried",
    "eval": {"dataset": "COCO val2017 head 1000 (sorted img_ids)", "n": 1000,
             "preprocess": "force-resize 800x1066 (fixed ONNX shape) + ImageNet norm",
             "metric": "pycocotools bbox mAP (host); board dumps raw logits/boxes"},
    "case_c_bypass": {
        "stage3_9ef2a58_explicit_result": "BUILD FAILED — trtexec direct parser, node 0: "
            "'Assertion failed: shiftIsAllZeros(zeroPoint)' (case C, zp!=0 1085/1485 + 149 INT32 bias DQ)",
        "fix_1_symmetric": "ActivationSymmetric+WeightSymmetric (zp=0 everywhere) clears case C; "
            "QuantizeBias=False clears case B (INT32 bias DQ)",
        "fix_2_op_restrict": "op_types_to_quantize=[Conv,Gemm] — default (quantize every op) parsed but the "
            "BUILDER rejected quantized-constant-in-self-attn (qdqGraphOptimizer::matchQuantizedConstantPluginOrDQ): "
            "'Quantized constant is only allowed before DQ or PLUGIN_V2 or kPLUGIN_V3'",
        "exclude": "backbone conv1 (case D stem 3ch7x7, parse-ok/build-fail Error10)",
        "onnx": "detr_int8_sym.onnx", "onnx_bytes": 43408891,
        "verify": {"zp_total": 688, "zp_nonzero": 0, "int32_bias_DQ": 0},
        "result": "explicit-QDQ INT8 now BUILDS (44.69 MiB engine, 11.002 ms) — the case-C fix is a "
                  "TOOLCHAIN unlock (buildable), tested next for whether it is also an ACCURACY unlock",
    },
    "implicit_mechanism": {
        "trt_log": "Calibrator is not being used. Users must provide dynamic range for all tensors "
                   "that are not Int32 or Bool.",
        "meaning": "--int8 without a calibration cache -> TRT has no data-derived activation ranges. It still "
                   "runs INT8 kernels (the implicit engine is FASTER than FP16, 9.43<13.28 ms, and SMALLER, "
                   "58.76<81.43 MiB — so INT8 kernels ARE active; it is NOT a pure FP16 fallback), but on "
                   "UNCONTROLLED (non-data-derived) ranges. Therefore its accuracy is NOT claimed: here it "
                   "happens to land near FP16 (0.4073, −3.9%), but the companion ResNet50-DLA implicit "
                   "collapsed to 0.017 on the same uncontrolled path. Only an explicit, calibrated QDQ engine "
                   "gives an accuracy you can stand behind.",
    },
    "engines": engines,
    "derived": derived,
    "divergence_vs_fp32": ACC["_divergence_vs_fp32"],
    "divergence_note": "global Pearson corr over all 100x92 logits is dominated by the no-object logit "
                       "dimension, so it is nearly identical for sym (0.982) and implicit (0.980) despite a "
                       "0.17 mAP gap — a weak proxy. pycocotools mAP is the authoritative metric.",
    "stage2_crossval": {
        "stage2_fp32": 0.4207, "stage2_int8": 0.2402, "stage2_drop_pct": -42.9, "stage2_maps_drop_pct": -77,
        "stage2_path": "ORT-QDQ INT8, DYNAMIC shape, CUDA EP, COCO val 5000 (commit 41dc49e / stage2 §4.5)",
        "onboard_drop_pct": derived["sym_map_drop_pct"], "onboard_maps_drop_pct": derived["sym_maps_drop_pct"],
        "verdict": "on-device explicit-sym INT8 (TRT, fixed shape, 1000 img) reproduces the stage2 collapse "
                   "(−43.8% vs −42.9%; mAP_s −84.6% vs −77%) — cross-validates. Absolute mAP is NOT comparable "
                   "(fixed-vs-dynamic shape, TRT-vs-ORT, 1000-vs-5000 img); only the RELATIVE collapse is.",
    },
    "headline": "The case-C symmetric re-export makes explicit-QDQ INT8 BUILDABLE on trtexec (was build-failed "
                "in 9ef2a58), but the accuracy-valid engine REPRODUCES the stage2 DETR collapse "
                "(0.4237→0.2383, −43.8%; mAP_s −84.6%). The fix is a toolchain unlock, NOT an accuracy rescue — "
                "the lever remains activation granularity (SmoothQuant, §4.4), exactly as stage2 §4.5 concluded.",
}

OUT = os.path.join(BASE, "detr_accuracy/results/detr_accuracy_summary.json")
json.dump(summary, open(OUT, "w"), indent=2, ensure_ascii=False)
print("wrote", OUT)
for tag, e in engines.items():
    print("  %-18s lat=%7.4f ms  %6.2f MiB  mAP=%.4f mAP_s=%.4f  acc_valid=%s"
          % (tag, e["latency_ms"], e["engine_mib"], e["mAP"], e["mAP_s"], e["accuracy_valid"]))
print("  derived:", json.dumps(derived, ensure_ascii=False))
