#!/usr/bin/env python3
# DX-M1 NPU on-device accuracy runner (runs ON the Pi, in venv-dx-runtime w/ dx_engine).
#
# Loads a compiled ResNet50 .dxnn, runs it over the SAME 1000-image bundle the CPU
# baseline and the committed Orin/CPU-proxy used, dumps per-image pred_cls + top-1.
#
# The input contract is QUERIED from the engine (get_input_tensors_info / get_input_size),
# never assumed -- a dx_com-compiled classifier may take float32 NCHW (no preprocessing
# folded, the clean case for CPU<->NPU comparison) or uint8 NHWC (normalize folded in).
# We build the matching representation from the raw u8 bundle and RECORD which one we fed
# (input_repr) so the (b) comparison's preprocessing path is fully transparent.
import argparse, json, os, time
import numpy as np
from dx_engine import InferenceEngine

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_nchw(u8_nhwc):
    x = u8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x, dtype=np.float32)


def dtype_of(info0):
    """Normalize the queried dtype (string or np.dtype) to a numpy dtype."""
    d = info0.get("dtype") if isinstance(info0, dict) else getattr(info0, "dtype", None)
    s = str(d).lower()
    if "uint8" in s or s == "u8":
        return np.uint8
    if "float16" in s or "half" in s:
        return np.float16
    if "float" in s or "f32" in s or "single" in s:
        return np.float32
    if "int8" in s:
        return np.int8
    try:
        return np.dtype(d).type
    except Exception:
        return np.float32


def shape_of(info0):
    s = info0.get("shape") if isinstance(info0, dict) else getattr(info0, "shape", None)
    return tuple(int(x) for x in s) if s is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)         # .dxnn
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)           # npu_native | npu_extqdq
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--accuracy-valid", type=int, default=1)
    args = ap.parse_args()

    ie = InferenceEngine(args.model)
    try:
        info = ie.get_input_tensors_info()
    except Exception as e:
        info = None
        print("get_input_tensors_info() failed:", e)
    try:
        in_size = ie.get_input_size()
    except Exception:
        in_size = None
    try:
        oinfo = ie.get_output_tensors_info()
    except Exception:
        oinfo = None
    print("== input contract ==")
    print("  info:", info)
    print("  get_input_size:", in_size)
    print("  output info:", oinfo)

    info0 = info[0] if isinstance(info, (list, tuple)) and info else info
    in_dt = dtype_of(info0) if info0 is not None else np.float32
    in_shape = shape_of(info0) if info0 is not None else None

    u8 = np.load(os.path.join(args.data, "rpi_sub_u8.npy"))[:args.n]      # (n,224,224,3) u8
    gts = np.load(os.path.join(args.data, "rpi_labels.npy"))[:args.n].astype(np.int64)
    n = u8.shape[0]

    # decide representation from (dtype, layout)
    nhwc = in_shape is not None and len(in_shape) == 4 and in_shape[-1] == 3
    nchw = in_shape is not None and len(in_shape) == 4 and in_shape[1] == 3
    if np.dtype(in_dt) == np.uint8:
        src = u8 if (nhwc or not nchw) else np.transpose(u8, (0, 3, 1, 2))
        repr_tag = "raw_uint8_" + ("nhwc" if (nhwc or not nchw) else "nchw")
        src = np.ascontiguousarray(src)
    else:
        xn = preprocess_nchw(u8)                                          # (n,3,224,224) f32
        if nhwc:
            src = np.ascontiguousarray(np.transpose(xn, (0, 2, 3, 1)).astype(in_dt))
            repr_tag = "norm_%s_nhwc" % np.dtype(in_dt).name
        else:
            src = np.ascontiguousarray(xn.astype(in_dt))
            repr_tag = "norm_%s_nchw" % np.dtype(in_dt).name
    print("== feeding repr:", repr_tag, "dtype", np.dtype(in_dt).name, "per-img shape", src[0:1].shape)

    preds = np.empty(n, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n):
        buf = np.ascontiguousarray(src[i:i + 1])
        out = ie.run([buf])
        logits = np.asarray(out[0]).reshape(-1)
        preds[i] = int(logits.argmax())
    dt = time.perf_counter() - t0

    acc = float((preds == gts).mean())
    rec = {"tag": args.tag, "engine": os.path.basename(args.model), "device": "DEEPX DX-M1 NPU",
           "top1": acc, "n_eval": n, "accuracy_valid": bool(args.accuracy_valid),
           "input_repr": repr_tag, "in_dtype": np.dtype(in_dt).name,
           "in_shape": list(in_shape) if in_shape else None,
           "wall_s_total": round(dt, 3), "wall_ms_per_img": round(1000 * dt / n, 3),
           "pred_cls": preds.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f)
    vflag = "" if args.accuracy_valid else "  [accuracy NOT claimed]"
    print("[%-11s] top1=%.4f (n=%d) %.3f ms/img repr=%s%s -> %s"
          % (args.tag, acc, n, rec["wall_ms_per_img"], repr_tag, vflag, args.out))


if __name__ == "__main__":
    main()
