#!/usr/bin/env python3
# Portable CPU-only ONNX benchmark — LOW-MEMORY variant (for 2GB boards, no swap).
# Identical results to rpi_bench.py: same preprocess math, same single-input latency,
# same one-at-a-time top-1. Difference: never materializes the full (n,3,224,224)
# float32 array (~602MB for n=1000). Instead mmaps the uint8 cache and preprocesses
# ONE image at a time. Predictions are bit-identical (preprocess is elementwise).
import argparse, json, os, platform, time
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(u8_nhwc):
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
    ap.add_argument("--precision", required=True)      # fp32 | int8 (label only)
    ap.add_argument("--data", required=True)           # dir with rpi_sub_u8.npy + rpi_labels.npy
    ap.add_argument("--out", required=True)
    ap.add_argument("--soc", default="rpi5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)  # 0 = ORT default (all cores)
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # mmap the uint8 cache (never fully loaded into RAM); labels are tiny.
    u8 = np.load(os.path.join(args.data, "rpi_sub_u8.npy"), mmap_mode="r")[: args.n]
    gts = np.load(os.path.join(args.data, "rpi_labels.npy"))[: args.n].astype(np.int64)
    one = preprocess(np.ascontiguousarray(u8[0:1]))     # single image, reused for latency

    # --- latency: warmup then timed single-input loop ---
    for _ in range(args.warmup):
        sess.run(None, {in_name: one})
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: one})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    median_ms = float(np.median(ts))
    p95_ms = float(np.percentile(ts, 95))

    # --- accuracy: top-1 over the n subset, lazy per-image preprocess (low memory) ---
    preds = np.empty(args.n, dtype=np.int64)
    for i in range(args.n):
        xi = preprocess(np.ascontiguousarray(u8[i:i + 1]))
        out = sess.run(None, {in_name: xi})[0]
        preds[i] = int(np.asarray(out).reshape(-1).argmax())
    acc = float((preds == gts).mean())

    res = {
        "model": "resnet50",
        "soc": args.soc,
        "precision": args.precision,
        "latency_ms": median_ms,
        "latency_p95_ms": p95_ms,
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
        "notes": "CPU-only, batch1, single-input latency; preprocess=crop_tv NCHW ImageNet-norm; lowmem lazy per-image",
    }
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"[{platform.machine()}] {args.precision}: "
          f"median={median_ms:.4f}ms p95={p95_ms:.4f}ms top1={acc:.4f} "
          f"(n={args.n}, ort={ort.__version__}, threads={res['intra_op_threads']})")


if __name__ == "__main__":
    main()
