#!/usr/bin/env python3
"""SSOT assembler for the DX-M1 DETR axis. Reads the raw per-artifact JSONs (accuracy,
dxbenchmark FPS/latency, on-device dx_engine E2E latency, and the dxbenchmark profiler)
and emits ONE consolidated detr_dxm1_summary.json + a human-readable digest. Every number
the report/doc cites comes from here so there is a single source of truth.

Headline: on DX-M1 the dx_com auto-partition puts the CNN backbone (+ the first encoder
self-attention, through its softmax) on the NPU at INT8, and the ENTIRE remainder of the
transformer on the HOST CPU at FP32 (via ORT). So DETR "INT8" is near-lossless -- NOT
because DX-M1 quantises transformers well, but because it declines to quantise the
transformer at all. The price is a 910 ms host-CPU FP32 transformer -> 1.04 s E2E,
host-CPU-compute-bound (a THIRD regime vs ResNet50 NPU-bound / yolo26n D2H-bound)."""
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
PROF = os.path.join(ROOT, "prof")

KEY = re.compile(r'^(.*?)\[((?:npu|cpu)_\d+)\]\[(Device_-?\d+)\]\[Job_(\d+)\]$')


def load(p):
    return json.load(open(p))


def stage_p50(profiler_path, stage):
    d = load(profiler_path)
    a = []
    for k, v in d.items():
        m = KEY.match(k)
        if m and m.group(1) == stage:
            rec = v[0] if isinstance(v, list) and v else v
            if isinstance(rec, dict) and 'start' in rec:
                a.append((rec['end'] - rec['start']) / 1e6)
    return round(st.median(a), 2) if a else None


def main():
    acc = load(os.path.join(RES, "detr_map_summary.json"))
    bench_full = load(os.path.join(PROF, "bench_useort_summary.json"))["results"][0]
    bench_npu = load(os.path.join(PROF, "bench_npuonly_summary.json"))["results"][0]
    e2e = load(os.path.join(RES, "detr_npu_raw_lat.json"))
    prof = os.path.join(PROF, "profiler_useort.json")

    # regime stage decomposition of the full (use-ort) pipeline, per-frame p50 (ms)
    cpu_task = stage_p50(prof, "CPU Task")
    npu_infer = stage_p50(prof, "Inference Core 0")
    d2h = stage_p50(prof, "D2H")
    h2d = stage_p50(prof, "H2D")
    lat_full = bench_full["Latency"]["mean"]

    summary = {
        "model": "facebook/detr-resnet-50 (detr_2out.onnx, 2-output: logits+pred_boxes)",
        "board": "lab Raspberry Pi 5 (Cortex-A76 x4) + DEEPX DX-M1 (M.2, 3-core INT8)",
        "toolchain": "dx_com 2.4.0 (x86, calib=ema 100 in-domain COCO PNG) / DXRT 3.4.2 / dx_engine cp313",
        "eval": "COCO val2017 first 500 (fixed 800x1066 force-resize, bit-identical npy across FP32/NPU)",

        "accuracy": {
            "fp32_mAP": acc["fp32"]["mAP"], "npu_int8_mAP": acc["npu_int8"]["mAP"],
            "delta_mAP_fp32_minus_npu": acc["delta_fp32_minus_npu"]["mAP"],
            "rel_mAP_drop_pct": acc["rel_mAP_drop_pct"],
            "fp32_mAP_s": acc["fp32"]["mAP_s"], "npu_int8_mAP_s": acc["npu_int8"]["mAP_s"],
            "delta_mAP_s": acc["delta_fp32_minus_npu"]["mAP_s"],
            "logits_corr": acc["divergence_npu_vs_fp32"]["logits_corr"],
            "boxes_corr": acc["divergence_npu_vs_fp32"]["boxes_corr"],
            "verdict": "near-lossless (backbone-INT8 error only; transformer FP32 in BOTH ref and NPU)",
        },

        "partition": {
            "graph_nodes": 708,
            "npu_group": "CNN backbone (ResNet50) + first encoder self-attn through its Softmax -> INT8 wbit8/abit8",
            "npu_outputs": [
                "/model/Transpose_output_0  FLOAT [1,850,256]  (backbone feature map, 850=25x34 tokens)",
                "/model/encoder/layers.0/self_attn/Softmax_output_0  FLOAT [1,8,850,850]  (first attention map)",
            ],
            "npu_input_bytes": 2558400, "npu_output_bytes": 23990400,
            "cpu_group": "rest of transformer (enc layers 0(post)-5 + all 6 decoder layers + FFNs + heads) -> HOST CPU FP32 via ORT",
            "cpu_memoryops_marked": 2,
            "note": "compiler auto-cut: unquantizable dynamic attention forces the transformer onto the host; this is WHY there is no INT8 collapse -- the transformer is never INT8",
        },

        "latency_ms": {
            "full_pipeline_e2e_dxbench": lat_full,
            "full_pipeline_e2e_dxengine_p50": e2e["lat_p50_ms"],
            "npu_only_e2e_dxbench": bench_npu["Latency"]["mean"],
            "npu_inference_only": bench_full["NPU Inference Time"]["mean"],
        },
        "throughput_fps": {
            "full_pipeline": bench_full["FPS"],
            "npu_only_transformer_skipped": bench_npu["FPS"],
            "host_transformer_slowdown_x": round(bench_npu["FPS"] / bench_full["FPS"], 1),
        },
        "regime_decomposition_p50_ms": {
            "host_cpu_transformer_fp32": cpu_task,
            "npu_int8_inference": npu_infer,
            "d2h_24MB_attention_handoff": d2h,
            "h2d_image": h2d,
            "host_cpu_pct_of_e2e": round(cpu_task / lat_full * 100, 1) if cpu_task else None,
            "npu_pct_of_e2e": round(npu_infer / lat_full * 100, 1) if npu_infer else None,
            "regime": "host-CPU-compute-bound (3rd DX-M1 regime: vs ResNet50 NPU-compute-bound / yolo26n PCIe-D2H-bound)",
        },

        "cross_model_dxm1": {
            "resnet50_cnn_cls": {"fp32->int8": "0.7620->0.7660 (~0)", "regime": "NPU-compute-bound (3-core 2.19x)"},
            "yolo26n_cnn_det": {"fp32->int8": "0.448->0.439 (-2.0%)", "regime": "PCIe-D2H-bound (2.82MB head, flat 1.0x)"},
            "detr_transformer_det": {"fp32->int8": f'{acc["fp32"]["mAP"]}->{acc["npu_int8"]["mAP"]} ({-acc["rel_mAP_drop_pct"]:+.1f}%)',
                                     "regime": "host-CPU-compute-bound (transformer FP32 on host, NPU 4%)"},
        },

        "caveats": [
            "500-img subset, batch1, prebuilt .dxnn; RELATIVE FP32->INT8 delta only (absolute mAP not comparable to stage2 dynamic-shape 0.4207).",
            "Near-lossless is NOT evidence DX-M1 quantises transformers -- the compiler keeps the transformer FP32 on the host, so this axis does not test transformer-INT8 on DX-M1.",
            "host-CPU-bound is a Pi5-A76 property (single-thread ORT FP32 transformer); a faster host or a toolchain that quantised attention would move the ceiling.",
            "same rn50/yolo26n D2H-bound caveat: PCIe Gen2x1 on Pi5 (DX-M1 native Gen3x4).",
        ],
    }

    out = os.path.join(RES, "detr_dxm1_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
