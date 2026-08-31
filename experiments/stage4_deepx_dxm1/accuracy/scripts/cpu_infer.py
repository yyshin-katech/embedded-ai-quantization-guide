#!/usr/bin/env python3
# A76 CPU baseline for the DX-M1 accuracy axis (runs ON the Pi, in the cpu-venv w/ ORT).
#
# Same physical Cortex-A76 that hosts the DX-M1 is also the stage4 CPU-fallback proxy.
# We run ResNet50 over the SAME 1000-image bundle (rpi_sub_u8.npy = tv.npy[:1000], the
# exact bundle behind the committed Orin/CPU-proxy numbers) so per-image pred_cls is
# 1:1 comparable to the NPU run and to prior stages.
#
# Preprocess replicates cpu_proxy/rpi_bench.py EXACTLY: /255, NHWC->NCHW, ImageNet norm.
# Two arms:
#   fp32  -> resnet50_fp32.onnx          (native float reference)
#   int8  -> resnet50_int8_qdq.onnx      (external ORT QDQ, MLAS SDOT integer kernel)
import argparse, json, os, time
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(u8_nhwc):
    x = u8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)          # dir with rpi_sub_u8.npy + rpi_labels.npy
    ap.add_argument("--out", required=True)           # output json path
    ap.add_argument("--tag", required=True)           # cpu_fp32 | cpu_int8
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    u8 = np.load(os.path.join(args.data, "rpi_sub_u8.npy"))[:args.n]
    gts = np.load(os.path.join(args.data, "rpi_labels.npy"))[:args.n].astype(np.int64)
    X = preprocess(u8)
    n = X.shape[0]

    preds = np.empty(n, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n):
        o = sess.run(None, {in_name: X[i:i + 1]})[0]
        preds[i] = int(np.asarray(o).reshape(-1).argmax())
    dt = time.perf_counter() - t0

    acc = float((preds == gts).mean())
    rec = {"tag": args.tag, "engine": os.path.basename(args.model), "device": "A76 CPU (ORT CPUEP)",
           "top1": acc, "n_eval": n, "accuracy_valid": True,
           "in_name": in_name, "threads": args.threads,
           "wall_s_total": round(dt, 3), "wall_ms_per_img": round(1000 * dt / n, 3),
           "ort_version": ort.__version__, "pred_cls": preds.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print("[%-9s] top1=%.4f (n=%d) %.3f ms/img  in=%s  ort=%s -> %s"
          % (args.tag, acc, n, rec["wall_ms_per_img"], in_name, ort.__version__, args.out))


if __name__ == "__main__":
    main()
