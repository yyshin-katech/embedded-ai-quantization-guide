#!/usr/bin/env python3
# Cross-compare on-device TRT pred_cls (Jetson AGX Orin iGPU/DLA) against the
# Jetson A78AE CPU-proxy (MLAS SDOT) pred_cls, over the SAME 1000-image bundle.
#
# The headline: does the on-device TRT INT8 kernel agree with MLAS INT8 on the SAME
# silicon? stage4 found INT8 predictions are integer-kernel-path dependent among CPUs
# (Jetson<->Pi5 100% same MLAS SDOT; Jetson<->x86 958/1000 different kernel). Here we
# cross a much bigger datapath boundary: CPU integer kernel (MLAS) vs GPU/DLA integer
# kernel (TensorRT), on one board, with identical QDQ scales (same resnet50_int8_qdq.onnx).
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # .../accuracy
RESULTS = os.path.join(HERE, "results")
CPU_RAW = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "stage5_infrastructure", "cpu_proxy", "raw"))


def load(path):
    with open(path) as f:
        return json.load(f)


def agree(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return int((a == b).sum()), len(a)


def main():
    gts = np.load(os.path.join(RESULTS, "rpi_labels.npy")).astype(np.int64)
    n = len(gts)

    dev = {}
    for tag in ["gpu_fp32", "gpu_fp16", "gpu_int8", "dla_fp16", "dla_int8"]:
        d = load(os.path.join(RESULTS, "rn50_%s_accuracy.json" % tag))
        dev[tag] = {"pred": np.asarray(d["pred_cls"], dtype=np.int64),
                    "top1": d["top1"], "accuracy_valid": d["accuracy_valid"]}

    cpu_fp32 = load(os.path.join(CPU_RAW, "jetson_orin_a78ae_fp32.json"))
    cpu_int8 = load(os.path.join(CPU_RAW, "jetson_orin_a78ae_int8.json"))
    cpu = {"fp32": {"pred": np.asarray(cpu_fp32["pred_cls"], dtype=np.int64),
                    "top1": cpu_fp32["accuracy"]},
           "int8": {"pred": np.asarray(cpu_int8["pred_cls"], dtype=np.int64),
                    "top1": cpu_int8["accuracy"]}}

    # verify top-1 recomputes from pred vs gts (sanity)
    for tag, r in dev.items():
        rec = float((r["pred"] == gts).mean())
        assert abs(rec - r["top1"]) < 1e-9, (tag, rec, r["top1"])

    out = {"n_eval": n, "board": "Jetson AGX Orin",
           "on_device_top1": {t: dev[t]["top1"] for t in dev},
           "cpu_proxy_top1": {"fp32": cpu["fp32"]["top1"], "int8": cpu["int8"]["top1"]},
           "agreements": {}}

    def rec_agree(name, a, b):
        m, tot = agree(a, b)
        out["agreements"][name] = {"agree": m, "n": tot, "frac": round(m / tot, 4)}
        return m, tot

    # ---- FP32 sanity: deterministic across platforms (stage4 said 100%) ----
    rec_agree("igpu_fp32__vs__cpu_fp32", dev["gpu_fp32"]["pred"], cpu["fp32"]["pred"])
    # ---- HEADLINE: TRT INT8 (GPU) vs MLAS INT8 (CPU), same silicon, same QDQ scales ----
    rec_agree("igpu_int8__vs__cpu_int8", dev["gpu_int8"]["pred"], cpu["int8"]["pred"])
    # ---- internal: does on-device INT8 match its own FP32? (lossless-at-1000?) ----
    rec_agree("igpu_int8__vs__igpu_fp32", dev["gpu_int8"]["pred"], dev["gpu_fp32"]["pred"])
    rec_agree("igpu_fp16__vs__igpu_fp32", dev["gpu_fp16"]["pred"], dev["gpu_fp32"]["pred"])
    rec_agree("dla_fp16__vs__igpu_fp16", dev["dla_fp16"]["pred"], dev["gpu_fp16"]["pred"])
    rec_agree("dla_fp16__vs__igpu_fp32", dev["dla_fp16"]["pred"], dev["gpu_fp32"]["pred"])
    # cross-precision reference: TRT INT8 vs CPU FP32 (the "gold" labels-ish)
    rec_agree("igpu_int8__vs__cpu_fp32", dev["gpu_int8"]["pred"], cpu["fp32"]["pred"])

    # ---- disagreement breakdown on the headline pair: who's right where they differ ----
    a = dev["gpu_int8"]["pred"]; b = cpu["int8"]["pred"]
    diff = np.where(a != b)[0]
    a_right = int((a[diff] == gts[diff]).sum())
    b_right = int((b[diff] == gts[diff]).sum())
    both_wrong = int(((a[diff] != gts[diff]) & (b[diff] != gts[diff])).sum())
    out["headline_disagreement"] = {
        "pair": "igpu_int8 (TRT) vs cpu_int8 (MLAS SDOT)",
        "n_disagree": int(len(diff)),
        "trt_correct_there": a_right,
        "mlas_correct_there": b_right,
        "both_wrong_there": both_wrong,
        "first_indices": diff[:15].tolist(),
    }

    with open(os.path.join(RESULTS, "accuracy_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # pretty print
    print("== on-device top-1 (n=%d) ==" % n)
    for t in ["gpu_fp32", "gpu_fp16", "gpu_int8", "dla_fp16", "dla_int8"]:
        v = "" if dev[t]["accuracy_valid"] else "  [implicit auto-range, accuracy NOT claimed]"
        print("  %-9s %.4f%s" % (t, dev[t]["top1"], v))
    print("== cpu-proxy (A78AE MLAS) top-1 ==")
    print("  fp32 %.4f   int8 %.4f" % (cpu["fp32"]["top1"], cpu["int8"]["top1"]))
    print("== agreements (/%d) ==" % n)
    for k, r in out["agreements"].items():
        print("  %-32s %4d/%d  (%.1f%%)" % (k, r["agree"], r["n"], 100 * r["frac"]))
    hd = out["headline_disagreement"]
    print("== headline disagreement: %s ==" % hd["pair"])
    print("  n_disagree=%d  TRT-right=%d  MLAS-right=%d  both-wrong=%d"
          % (hd["n_disagree"], hd["trt_correct_there"], hd["mlas_correct_there"], hd["both_wrong_there"]))
    print("wrote", os.path.join(RESULTS, "accuracy_summary.json"))


if __name__ == "__main__":
    main()
