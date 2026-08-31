#!/usr/bin/env python3
"""A76 CPU baseline for yolo26n (FP32 ONNX, ORT CPUExecutionProvider).
Mirrors the DX-M1 NPU dxrun measurement: batch-1 latency + thread sweep.
Usage: cpu_bench.py <model.onnx> <threads> <warmup> <iters>
"""
import sys, time, json, statistics as st
import numpy as np, onnxruntime as ort

model, threads, warm, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
so = ort.SessionOptions()
so.intra_op_num_threads = threads
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
inp = s.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
x = np.random.rand(*shape).astype(np.float32)
name = inp.name
for _ in range(warm):
    s.run(None, {name: x})
ts = []
for _ in range(iters):
    t0 = time.perf_counter()
    s.run(None, {name: x})
    ts.append((time.perf_counter() - t0) * 1000.0)
ts.sort()
p50 = st.median(ts)
out = {"model": model.split("/")[-1], "threads": threads, "warmup": warm, "iters": iters,
       "lat_p50_ms": round(p50, 4), "lat_mean_ms": round(st.mean(ts), 4),
       "lat_p90_ms": round(ts[min(len(ts)-1, int(len(ts)*0.9))], 4),
       "lat_min_ms": round(ts[0], 4), "lat_max_ms": round(ts[-1], 4),
       "fps_single_stream": round(1000.0 / p50, 2)}
print(json.dumps(out))
