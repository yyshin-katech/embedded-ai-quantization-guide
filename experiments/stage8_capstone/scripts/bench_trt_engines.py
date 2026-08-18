#!/usr/bin/env python
"""BEVDet TRT 엔진(FP32/FP16/INT8) 지연 + 출력 일치도 벤치.
커스텀 플러그인(libmmdeploy_tensorrt_ops.so) 로드 후 각 엔진 deserialize.
init 가중치라 절대 예측은 무의미 → 지연(가중치 무관)과 FP32 대비 출력 편차가 지표.
"""
import os, sys, ctypes, json, numpy as np
os.environ.setdefault("CUDA_DEVICE", "0")
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import tensorrt as trt

PLUGIN = os.path.expanduser("~/capstone-bev/mmdeploy-bevdet/build/lib/libmmdeploy_tensorrt_ops.so")
SAMPLE = os.path.expanduser("~/capstone-bev/BEVDet/work_dirs/capstone/bench_sample.npz")
BASE = os.path.expanduser("~/capstone-bev/BEVDet/work_dirs/capstone")
ENGINES = {"FP32": "trtbevdet.engine", "FP16": "trtbevdet_fp16.engine",
           "INT8": "trtbevdet_int8.engine"}
N_WARMUP, N_ITER = 15, 60

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")
ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)
print("plugin loaded:", PLUGIN)

data = np.load(SAMPLE)
inputs_np = {k: data[k] for k in data.files}
print("sample inputs:", {k: v.shape for k, v in inputs_np.items()})

TRT_NP = {trt.float32: np.float32, trt.int32: np.int32, trt.float16: np.float16}

def run_engine(path):
    with open(path, "rb") as f:
        rt = trt.Runtime(logger)
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    n = engine.num_bindings
    bufs, host_out, bindings = {}, {}, [None] * n
    # set input shapes first
    for i in range(n):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            ctx.set_binding_shape(i, inputs_np[name].shape)
    for i in range(n):
        name = engine.get_binding_name(i)
        shape = tuple(ctx.get_binding_shape(i))
        dt = TRT_NP[engine.get_binding_dtype(i)]
        nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
        dptr = cuda.mem_alloc(nbytes)
        bindings[i] = int(dptr)
        if engine.binding_is_input(i):
            host = np.ascontiguousarray(inputs_np[name].astype(dt))
            cuda.memcpy_htod(dptr, host)
            bufs[name] = dptr
        else:
            host_out[name] = (np.empty(shape, dtype=dt), dptr)
    # warmup
    for _ in range(N_WARMUP):
        ctx.execute_v2(bindings)
    cuda.Context.synchronize()
    # timed
    start, end = cuda.Event(), cuda.Event()
    times = []
    for _ in range(N_ITER):
        start.record()
        ctx.execute_v2(bindings)
        end.record(); end.synchronize()
        times.append(start.time_till(end))
    times = np.array(sorted(times))
    # fetch outputs (D2H)
    outs = {}
    for name, (h, d) in host_out.items():
        cuda.memcpy_dtoh(h, d)
        outs[name] = h.copy()
    lat = dict(p50=float(np.median(times)), mean=float(times.mean()),
               p90=float(times[int(0.9*len(times))]), min=float(times.min()))
    return lat, outs, os.path.getsize(path)

results = {}
ref_outs = None
for tag, fn in ENGINES.items():
    path = os.path.join(BASE, fn)
    if not os.path.exists(path):
        print(f"[skip] {tag}: {fn} 없음"); continue
    lat, outs, size = run_engine(path)
    entry = dict(latency_ms=lat, engine_bytes=size)
    if tag == "FP32":
        ref_outs = outs
    if ref_outs is not None and tag != "FP32":
        # FP32 대비 출력 편차(양자화 오차 전파)
        diffs = {}
        for k in ref_outs:
            a, b = ref_outs[k].astype(np.float64), outs[k].astype(np.float64)
            denom = np.abs(a).max() + 1e-9
            diffs[k] = dict(max_abs=float(np.abs(a-b).max()),
                            rel_max=float(np.abs(a-b).max()/denom),
                            corr=float(np.corrcoef(a.ravel(), b.ravel())[0,1]))
        entry["vs_fp32"] = diffs
    results[tag] = entry
    print(f"\n=== {tag} ===  engine={size/1e6:.1f} MB")
    print(f"  latency p50={lat['p50']:.3f}ms mean={lat['mean']:.3f} p90={lat['p90']:.3f} min={lat['min']:.3f}")
    if "vs_fp32" in entry:
        for k, d in entry["vs_fp32"].items():
            print(f"  vs FP32 {k}: max_abs={d['max_abs']:.4g} rel_max={d['rel_max']:.4g} corr={d['corr']:.5f}")

OUTJSON = os.environ.get("BENCH_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "trt_ladder.json"))
with open(OUTJSON, "w") as f:
    json.dump(results, f, indent=2)
print("\nSAVED:", OUTJSON)
os._exit(0)
