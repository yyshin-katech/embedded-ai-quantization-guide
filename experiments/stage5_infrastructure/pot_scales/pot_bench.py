#!/usr/bin/env python3
# Run one ONNX model over the shared 1000-image ResNet50 bundle on ORT's CPU
# ExecutionProvider and dump per-image predictions PLUS a digest of the raw logits.
#
# This is the measurement half of the power-of-two (PoT) scale experiment. The
# baseline it is compared against is ../cpu_proxy/raw/*.json, so:
#   * the preprocessing below replicates cpu_proxy/rpi_bench.py bit-for-bit
#     (u8 NHWC -> /255 -> NCHW -> ImageNet norm, float32), and
#   * the JSON keys are a superset of that script's, so pot_agree.py can compare a
#     new PoT run directly against the existing baseline runs.
#
# Why the logits digest and not just top-1: C2 measures top-1 prediction agreement,
# which is what the paper reports, but the PoT claim is stronger than that -- it is
# *bit-identical outputs*. A run can agree 1000/1000 on argmax while still differing
# in the logits. logits_md5 settles that without shipping a 4 MB array; --save-logits
# keeps the array when the digests disagree and you need to see by how much.
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(u8_nhwc):
    """Identical to cpu_proxy/rpi_bench.py preprocess(). Do not 'improve' this."""
    x = u8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x, dtype=np.float32)


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") or line.startswith("Model"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True,
                    help="fp32 | int8 | int8_pot   (label only)")
    ap.add_argument("--data", required=True,
                    help="dir holding rpi_sub_u8.npy + rpi_labels.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--soc", default="rpi5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = ORT default (all cores). Pin it if you are chasing "
                         "FP32 bit-identity: reduction order is thread-count dependent.")
    ap.add_argument("--save-logits", metavar="PATH",
                    help="also write the (n,1000) float32 logits as .npy")
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    u8 = np.load(os.path.join(args.data, "rpi_sub_u8.npy"))[: args.n]
    gts = np.load(os.path.join(args.data, "rpi_labels.npy"))[: args.n].astype(np.int64)
    X = preprocess(u8)
    one = X[0:1]

    for _ in range(args.warmup):
        sess.run(None, {in_name: one})
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: one})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)

    logits = np.empty((args.n, 1000), dtype=np.float32)
    for i in range(args.n):
        out = sess.run(None, {in_name: X[i:i + 1]})[0]
        logits[i] = np.asarray(out, dtype=np.float32).reshape(-1)
    preds = logits.argmax(axis=1).astype(np.int64)
    acc = float((preds == gts).mean())

    digest = hashlib.md5(np.ascontiguousarray(logits).tobytes()).hexdigest()
    if args.save_logits:
        np.save(args.save_logits, logits)

    res = {
        "model": "resnet50",
        "soc": args.soc,
        "precision": args.precision,
        "model_file": os.path.basename(args.model),
        "latency_ms": float(np.median(ts)),
        "latency_p95_ms": float(np.percentile(ts, 95)),
        "accuracy": acc,
        "n_eval": int(args.n),
        "provider": "CPUExecutionProvider",
        "ort_version": ort.__version__,
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "intra_op_threads": (args.threads if args.threads > 0 else "default(all)"),
        "warmup": args.warmup,
        "iters": args.iters,
        "pred_cls": preds.tolist(),
        "logits_md5": digest,
        "notes": "CPU-only, batch1, single-input latency; preprocess=crop_tv NCHW "
                 "ImageNet-norm (identical to cpu_proxy/rpi_bench.py)",
    }
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("[%s] %s: median=%.4fms p95=%.4fms top1=%.4f logits_md5=%s "
          "(n=%d, ort=%s, threads=%s)"
          % (platform.machine(), args.precision, res["latency_ms"],
             res["latency_p95_ms"], acc, digest, args.n, ort.__version__,
             res["intra_op_threads"]))


if __name__ == "__main__":
    main()
