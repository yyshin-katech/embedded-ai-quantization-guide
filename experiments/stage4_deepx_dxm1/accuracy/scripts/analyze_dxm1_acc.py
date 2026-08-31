#!/usr/bin/env python3
# Cross-compare DX-M1 NPU pred_cls against the SAME-A76 CPU baselines, over the SAME
# 1000-image bundle (rpi_sub_u8.npy = tv.npy[:1000]) the committed Orin/CPU-proxy used.
#
# The decisive setup: the same physical Cortex-A76 is BOTH the stage4 CPU-fallback proxy
# AND the DX-M1 host, so per-image argmax is a clean subtraction of CPU vs NPU on one board.
#
# The (b) prediction-agreement thread so far:
#   stage4  same MLAS SDOT kernel  (Jetson<->Pi5)   -> 1000/1000 (bit-identical)
#   stage4  different CPU kernel   (Jetson<->x86)    ->  958/1000
#   stage3  CPU<->accelerator, SAME QDQ scales       ->  961/1000  (TRT INT8 vs MLAS INT8)
#   HERE    CPU<->accelerator, scales CANNOT match   ->  ???
# On DX-M1 the same-scale cross-kernel leg is IMPOSSIBLE: dx_com HARD-REJECTS external QDQ
# at compile (GraphStructureError: 106 isolated nodes) -- finding (a). So the NPU runs its
# OWN native PTQ scales, and npu_int8-vs-cpu_int8 confounds kernel + scales + where
# preprocessing is quantized (NPU folds+quantizes div/normalize in-graph; CPU does float
# preprocessing host-side). We record all three legs and flag the confounds.
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # .../accuracy
RESULTS = os.path.join(HERE, "results")


def load(path):
    with open(path) as f:
        return json.load(f)


def agree(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return int((a == b).sum()), len(a)


def main():
    gts = np.load(os.path.join(RESULTS, "rpi_labels.npy")).astype(np.int64)
    n = len(gts)

    arms = {}
    for tag, fn in [("npu_native", "npu_native.json"),
                    ("cpu_fp32", "cpu_fp32.json"),
                    ("cpu_int8", "cpu_int8.json")]:
        d = load(os.path.join(RESULTS, fn))
        arms[tag] = {"pred": np.asarray(d["pred_cls"], dtype=np.int64),
                     "top1": d["top1"], "engine": d["engine"],
                     "device": d.get("device"),
                     "accuracy_valid": d.get("accuracy_valid", True),
                     "input_repr": d.get("input_repr")}

    # sanity: top-1 recomputes from pred vs gts
    for tag, r in arms.items():
        rec = float((r["pred"] == gts).mean())
        assert abs(rec - r["top1"]) < 1e-9, (tag, rec, r["top1"])

    out = {"n_eval": n, "board": "Raspberry Pi 5 (Cortex-A76) + DEEPX DX-M1",
           "top1": {t: arms[t]["top1"] for t in arms},
           "accuracy_valid": {t: arms[t]["accuracy_valid"] for t in arms},
           "input_repr": {t: arms[t]["input_repr"] for t in arms},
           "external_qdq_on_npu": "REJECTED at compile: GraphStructureError "
                                  "(106 isolated nodes) -> no .dxnn produced (finding a)",
           "agreements": {}}

    def rec_agree(name, a, b, note=None):
        m, tot = agree(a, b)
        out["agreements"][name] = {"agree": m, "n": tot, "frac": round(m / tot, 4)}
        if note:
            out["agreements"][name]["note"] = note
        return m, tot

    # ---- clean intra-CPU quantization drop (same MLAS family, same host-side preprocessing) ----
    rec_agree("cpu_int8__vs__cpu_fp32", arms["cpu_int8"]["pred"], arms["cpu_fp32"]["pred"],
              note="clean: same silicon+kernel family, only quantization differs")
    # ---- does the NPU INT8 (native PTQ) track the float reference? ----
    rec_agree("npu_int8__vs__cpu_fp32", arms["npu_native"]["pred"], arms["cpu_fp32"]["pred"],
              note="NPU INT8 vs CPU FP32: quantization + kernel + preprocessing-location")
    # ---- HEADLINE (b): two INT8 realizations, DIFFERENT scales (external QDQ can't compile) ----
    rec_agree("npu_int8__vs__cpu_int8", arms["npu_native"]["pred"], arms["cpu_int8"]["pred"],
              note="CPU<->accelerator INT8; scales DIFFER (NPU native PTQ vs ORT QDQ) "
                   "-- NOT the stage3 same-scale 961/1000 leg")

    # ---- cross-run reproducibility: my A76 CPU arms (ORT 1.29.0) vs the committed
    #      stage4 Pi5 CPU-proxy (ORT 1.28.0), SAME board, SAME bundle, different session.
    #      Same MLAS SDOT kernel across ORT versions -> expect bit-identical (100%),
    #      proving the DX-M1 legs are per-image (not just top-1) comparable to committed data. ----
    committed = os.path.abspath(os.path.join(
        HERE, "..", "..", "stage5_infrastructure", "cpu_proxy", "raw"))
    try:
        old_fp32 = np.asarray(load(os.path.join(committed, "rpi5_fp32.json"))["pred_cls"], np.int64)
        old_int8 = np.asarray(load(os.path.join(committed, "rpi5_int8.json"))["pred_cls"], np.int64)
        m_fp32, _ = agree(arms["cpu_fp32"]["pred"], old_fp32[:n])
        m_int8, _ = agree(arms["cpu_int8"]["pred"], old_int8[:n])
        out["cross_run_checks"] = {
            "note": "this session ORT 1.29.0 vs committed stage4 Pi5 ORT 1.28.0, same A76+bundle",
            "cpu_fp32__vs__committed_rpi5_fp32": {"agree": m_fp32, "n": n},
            "cpu_int8__vs__committed_rpi5_int8": {"agree": m_int8, "n": n},
        }
    except FileNotFoundError:
        out["cross_run_checks"] = {"note": "committed rpi5 raw not found; skipped"}

    # ---- disagreement breakdown on the two-INT8-kernel headline: who's right ----
    a = arms["npu_native"]["pred"]; b = arms["cpu_int8"]["pred"]
    diff = np.where(a != b)[0]
    a_right = int((a[diff] == gts[diff]).sum())
    b_right = int((b[diff] == gts[diff]).sum())
    both_wrong = int(((a[diff] != gts[diff]) & (b[diff] != gts[diff])).sum())
    out["headline_disagreement"] = {
        "pair": "npu_native (DX-M1 INT8) vs cpu_int8 (A76 MLAS SDOT)",
        "n_disagree": int(len(diff)),
        "npu_correct_there": a_right,
        "mlas_correct_there": b_right,
        "both_wrong_there": both_wrong,
        "net_top1_delta_x1000": int(a_right - b_right),
        "first_indices": diff[:15].tolist(),
    }

    with open(os.path.join(RESULTS, "accuracy_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # pretty print
    print("== DX-M1 accuracy axis (n=%d, same 1000-image bundle as committed Orin/CPU-proxy) ==" % n)
    print("== finding (a): external QDQ on NPU -> %s" % out["external_qdq_on_npu"])
    print("== top-1 ==")
    for t in ["cpu_fp32", "cpu_int8", "npu_native"]:
        print("  %-11s %.4f  [%s]" % (t, arms[t]["top1"], arms[t]["engine"]))
    print("== agreements (/%d) ==" % n)
    for k, r in out["agreements"].items():
        print("  %-28s %4d/%d  (%.1f%%)" % (k, r["agree"], r["n"], 100 * r["frac"]))
    if "cpu_int8__vs__committed_rpi5_int8" in out.get("cross_run_checks", {}):
        cr = out["cross_run_checks"]
        print("== cross-run (ORT 1.29 vs committed 1.28, same A76+bundle) ==")
        print("  cpu_fp32 %d/%d   cpu_int8 %d/%d"
              % (cr["cpu_fp32__vs__committed_rpi5_fp32"]["agree"], n,
                 cr["cpu_int8__vs__committed_rpi5_int8"]["agree"], n))
    hd = out["headline_disagreement"]
    print("== headline disagreement: %s ==" % hd["pair"])
    print("  n_disagree=%d  NPU-right=%d  MLAS-right=%d  both-wrong=%d  net=%+d"
          % (hd["n_disagree"], hd["npu_correct_there"], hd["mlas_correct_there"],
             hd["both_wrong_there"], hd["net_top1_delta_x1000"]))
    print("wrote", os.path.join(RESULTS, "accuracy_summary.json"))


if __name__ == "__main__":
    main()
