#!/usr/bin/env python3
# On-device accuracy harness for Jetson AGX Orin.
#
# Runs the SAVED TensorRT .plan engines (built by the solo sweep, commit 00fd97d)
# over the SAME 1000-image subset the CPU-proxy (Jetson A78AE, commit 49e30ff) used,
# so per-image pred_cls is 1:1 comparable. The question this closes:
#   the RTX/CPU-proxy INT8 accuracy — does it hold on Orin *silicon*, where INT8 runs
#   on the TensorRT integer datapath (iGPU/DLA), NOT on the MLAS SDOT CPU kernel?
# stage4 proved INT8 predictions are integer-kernel-path dependent among CPUs
# (Jetson<->Pi5 bit-identical same MLAS SDOT; Jetson<->x86 958/1000 different kernel).
# This extends that to CPU(MLAS) vs accelerator(TRT) on ONE board.
#
# Preprocess replicates cpu_proxy/rpi_bench.py EXACTLY (÷255, NHWC->NCHW, ImageNet norm),
# and the INT8 engine was built from resnet50_int8_qdq.onnx — the SAME QDQ ONNX the CPU
# proxy fed to ORT — so quantization scales are identical and the ONLY difference under
# test is the integer kernel datapath.
import argparse, json, os
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

TRT_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT8:  np.int8,
}


def preprocess(u8_nhwc):
    x = u8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x, dtype=np.float32)


def ck(ret):
    """cuda-python returns (err, *rest); raise on error, return rest tuple."""
    err = ret[0] if isinstance(ret, tuple) else ret
    rest = ret[1:] if isinstance(ret, tuple) else ()
    if int(err) != 0:
        raise RuntimeError("CUDA error %d" % int(err))
    return rest


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path):
    with open(path, "rb") as f:
        data = f.read()
    rt = trt.Runtime(TRT_LOGGER)
    eng = rt.deserialize_cuda_engine(data)
    if eng is None:
        raise RuntimeError("deserialize failed: %s" % path)
    return eng


def run_engine(path, X, gts):
    eng = load_engine(path)
    ctx = eng.create_execution_context()
    n = X.shape[0]

    in_name = out_name = None
    for i in range(eng.num_io_tensors):
        nm = eng.get_tensor_name(i)
        if eng.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            in_name = nm
        else:
            out_name = nm

    in_shape = tuple(eng.get_tensor_shape(in_name))
    if any(d < 0 for d in in_shape):
        in_shape = (1, 3, 224, 224)
        ctx.set_input_shape(in_name, in_shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    in_dt  = TRT_NP[eng.get_tensor_dtype(in_name)]
    out_dt = TRT_NP[eng.get_tensor_dtype(out_name)]
    in_nbytes  = int(np.prod(in_shape))  * np.dtype(in_dt).itemsize
    out_nbytes = int(np.prod(out_shape)) * np.dtype(out_dt).itemsize

    d_in  = ck(cudart.cudaMalloc(in_nbytes))[0]
    d_out = ck(cudart.cudaMalloc(out_nbytes))[0]
    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))

    Xc = X.astype(in_dt, copy=False)
    host_out = np.empty(out_shape, dtype=out_dt)
    preds = np.empty(n, dtype=np.int64)
    H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for i in range(n):
        host_in = np.ascontiguousarray(Xc[i:i + 1])
        ck(cudart.cudaMemcpy(int(d_in), host_in.ctypes.data, in_nbytes, H2D))
        if not ctx.execute_async_v3(0):           # default stream
            raise RuntimeError("execute_async_v3 failed at img %d" % i)
        ck(cudart.cudaDeviceSynchronize())
        ck(cudart.cudaMemcpy(host_out.ctypes.data, int(d_out), out_nbytes, D2H))
        preds[i] = int(np.asarray(host_out).reshape(-1).argmax())

    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    acc = float((preds == gts).mean())
    return preds, acc, list(in_shape), list(out_shape), str(in_dt.__name__), str(out_dt.__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines-dir", default=os.path.expanduser("~/orin_bench/engines"))
    ap.add_argument("--data", default="/home/katech/cpu_bench/data")
    ap.add_argument("--out", default=os.path.expanduser("~/orin_bench/accuracy"))
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    u8 = np.load(os.path.join(args.data, "rpi_sub_u8.npy"))[:args.n]
    gts = np.load(os.path.join(args.data, "rpi_labels.npy"))[:args.n].astype(np.int64)
    X = preprocess(u8)

    # tag -> (engine file, accuracy-valid?)  DLA INT8 = implicit --int8 auto-range (NOT accuracy-valid)
    engines = [
        ("gpu_fp32", "rn50_gpu_fp32.plan", True),
        ("gpu_fp16", "rn50_gpu_fp16.plan", True),
        ("gpu_int8", "rn50_gpu_int8.plan", True),   # explicit QDQ (stage3 real scales)
        ("dla_fp16", "rn50_dla_fp16.plan", True),
        ("dla_int8", "rn50_dla_int8.plan", False),  # implicit auto-range: latency-valid only
    ]
    results = {}
    for tag, fn, acc_valid in engines:
        p = os.path.join(args.engines_dir, fn)
        if not os.path.exists(p):
            print("MISSING", p)
            continue
        preds, acc, ish, osh, idt, odt = run_engine(p, X, gts)
        rec = {"tag": tag, "engine": fn, "top1": acc, "n_eval": args.n,
               "accuracy_valid": acc_valid, "in_shape": ish, "out_shape": osh,
               "in_dtype": idt, "out_dtype": odt, "pred_cls": preds.tolist()}
        results[tag] = rec
        with open(os.path.join(args.out, "rn50_%s_accuracy.json" % tag), "w") as f:
            json.dump(rec, f)
        vflag = "" if acc_valid else "  [implicit auto-range: accuracy NOT claimed]"
        print("[%-8s] top1=%.4f (n=%d) in=%s/%s out=%s/%s%s"
              % (tag, acc, args.n, ish, idt, osh, odt, vflag))

    meta = {"board": "NVIDIA Jetson AGX Orin Developer Kit (64GB)",
            "trt": trt.__version__, "n_eval": args.n,
            "cuda_python": "12.9.7", "nvpmodel": "MAXN",
            "engines": [r["tag"] for r in results.values()],
            "note": "batch1; preprocess=crop_tv NCHW ImageNet-norm identical to cpu_proxy/rpi_bench.py; "
                    "gpu_int8 built from resnet50_int8_qdq.onnx (same QDQ ONNX the CPU proxy used)"}
    with open(os.path.join(args.out, "orin_accuracy_meta.json"), "w") as f:
        json.dump(meta, f)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
