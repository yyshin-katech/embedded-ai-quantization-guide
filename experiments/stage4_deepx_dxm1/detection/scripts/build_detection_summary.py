#!/usr/bin/env python3
"""Assemble the DX-M1 detection axis SSOT from the per-variant artifacts:
  results/map_{fp32,npu_ppe,npu_coco}.json        (pycocotools, same scorer/subset)
  results/predictions_npu_{ppe,coco}_lat.json     (dx_engine host-timed latency)
  raw/{coco_prof,e2e_ppe_prof}/profiler.json      (dxbenchmark stage decomposition)

Emits results/detection_summary.json — the single source for the report/callout:
accuracy decomposition (pure-quant vs calibration-domain) + latency + regime.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "crossover", "scripts"))
import analyze_profiler as ap  # noqa: E402


def load(p):
    return json.load(open(p))


def prof(path, label):
    """Reuse the crossover analyzer for identical p50/core-dist semantics."""
    import io
    import contextlib
    sys.argv = ["x", path, label]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ap.main()
    d = json.loads(buf.getvalue())
    # core job distribution from the per-core inference counts
    dist = {c.split()[-1]: d["inference_per_core"][c]["n"]
            for c in d["inference_per_core"]}
    return {
        "fps": round(d["fps"], 2),
        "H2D_p50_ms": round(d["stages"]["H2D"]["p50_ms"], 3),
        "D2H_p50_ms": round(d["stages"]["D2H"]["p50_ms"], 3),
        "Inference_p50_ms": round(d["inference_all_cores"]["p50_ms"], 3),
        "d2h_over_infer": round(d["stages"]["D2H"]["p50_ms"]
                                / d["inference_all_cores"]["p50_ms"], 2),
        "core_job_dist": dist,
        "output_bytes_raw_head": 2822400,
        "input_bytes": 1228800,
        "regime": "D2H-bound",
    }


def main():
    R = os.path.join(ROOT, "results")
    fp32 = load(os.path.join(R, "map_fp32.json"))
    ppe = load(os.path.join(R, "map_npu_ppe.json"))
    coco = load(os.path.join(R, "map_npu_coco.json"))
    lat_ppe = load(os.path.join(R, "predictions_npu_ppe_lat.json"))
    lat_coco = load(os.path.join(R, "predictions_npu_coco_lat.json"))

    def r4(x):
        return round(float(x), 4)

    pure_quant = r4(fp32["mAP_50_95"] - coco["mAP_50_95"])   # FP32 - OptionB
    calib_dom = r4(coco["mAP_50_95"] - ppe["mAP_50_95"])     # OptionB - OptionA
    maps_drop = r4(fp32["mAP_s"] - coco["mAP_s"])            # small-object quant hit

    out = {
        "model": "yolo26n (ultralytics), 640x640 letterbox, COCO val2017 first-500",
        "eval": {"n_images": 500, "scorer": "pycocotools bbox, same subset imgIds",
                 "conf_thresh": 0.001, "decode": "shared [1,300,6] xyxy@640 (identical FP32/INT8)"},
        "accuracy_mAP_50_95": {
            "fp32_ref": fp32["mAP_50_95"],
            "int8_ppe_vendor_calib": ppe["mAP_50_95"],
            "int8_coco_indomain_calib": coco["mAP_50_95"],
        },
        "decomposition": {
            "pure_quantization_fp32_minus_coco": pure_quant,
            "pure_quantization_rel_pct": round(100 * pure_quant / fp32["mAP_50_95"], 1),
            "calibration_domain_coco_minus_ppe": calib_dom,
            "small_object_quant_hit_mAP_s": maps_drop,
            "small_object_rel_pct": round(100 * maps_drop / fp32["mAP_s"], 1),
            "verdict": "detection INT8 near-lossless on DX-M1; calibration domain within noise",
        },
        "full_metrics": {"fp32": fp32, "int8_ppe": ppe, "int8_coco": coco},
        "latency_pi_batch1_hosttimed_ms": {
            "int8_ppe_p50": lat_ppe["lat_p50_ms"],
            "int8_coco_p50": lat_coco["lat_p50_ms"],
            "note": "FP32 ref ran x86 CPU (mAP reference only, not a Pi latency comparand)",
        },
        "regime": {
            "int8_coco_optionB": prof(os.path.join(ROOT, "raw", "coco_prof", "profiler.json"), "coco_e2e"),
            "int8_ppe_optionA": prof(os.path.join(ROOT, "raw", "e2e_ppe_prof", "profiler.json"), "ppe_e2e"),
            "finding": ("dx_com cuts the graph at the raw multi-scale head (6 conv outputs, "
                        "2,822,400 B); decode+NMS run host-side. The 'end2end' [1,300,6] is a "
                        "host convenience, NOT an on-NPU reduction -> D2H still transfers the full "
                        "2.82 MB raw head every frame -> D2H-bound, core scaling 1.00x, core job "
                        "dist ~473/27/2. NMS-folding does NOT flip the regime (confirms+deepens "
                        "the crossover raw-head yolo26n: 472/28/2, D2H 21.81 ms)."),
        },
        "caveats": [
            "500-image val2017 subset -> relative deltas only; absolute mAP not a literature comparand.",
            "prebuilt/PTQ, batch1, Pi5 PCIe Gen2x1 -> D2H-bound is a Pi5-link property (native Gen3x4 would move the wall).",
            "latency host-timed via dx_engine (includes host decode/NMS), not pure NPU compute.",
        ],
    }
    op = os.path.join(R, "detection_summary.json")
    json.dump(out, open(op, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"-> {op}", file=sys.stderr)


if __name__ == "__main__":
    main()
