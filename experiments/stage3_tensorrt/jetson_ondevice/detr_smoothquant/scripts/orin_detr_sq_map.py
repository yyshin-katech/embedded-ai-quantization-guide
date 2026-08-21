#!/usr/bin/env python3
# orin_detr_sq_map.py (ON BOARD, ~/orin_bench) — dump raw DETR outputs for the SmoothQuant
# INT8 engines, reusing run_engine() from orin_detr_map.py (batch1 execute_async_v3, the
# .copy() zero-copy-aliasing guard). Host then computes pycocotools mAP.
#
# Engines under test (built by trtexec --int8 --fp16 from the SmoothQuant+sym ONNX):
#   detr_gpu_int8_sq_a10.plan = SmoothQuant alpha=1.0 (stage2 §4.4 DETR-best) + symmetric QDQ
#   detr_gpu_int8_sq_a05.plan = SmoothQuant alpha=0.5 + symmetric QDQ
# Both are accuracy-valid (explicit calibrated symmetric QDQ on Conv+Gemm); the ONLY delta
# vs the committed detr_gpu_int8_sym.plan is the ONNX-level SmoothQuant activation migration.
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin_detr_map import run_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines-dir", default=os.path.expanduser("~/orin_bench/engines"))
    ap.add_argument("--img-dir", default=os.path.expanduser("~/orin_bench/coco_sub"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/orin_bench/coco_sub/manifest.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/orin_bench/detr_smoothquant"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(open(args.manifest))
    paths = [os.path.join(args.img_dir, m["file_name"]) for m in manifest]
    img_ids = [int(m["image_id"]) for m in manifest]
    print("eval images:", len(paths), flush=True)

    engines = [("sq_a10", "detr_gpu_int8_sq_a10.plan"),
               ("sq_a05", "detr_gpu_int8_sq_a05.plan")]
    meta_engines = []
    for tag, fn in engines:
        p = os.path.join(args.engines_dir, fn)
        if not os.path.exists(p):
            print("MISSING", p, flush=True); continue
        print("[%s] %s" % (tag, fn), flush=True)
        lg, bx, ish, idt = run_engine(p, paths)
        np.savez_compressed(os.path.join(args.out, "detr_%s_raw.npz" % tag),
                            logits=lg, boxes=bx, img_ids=np.array(img_ids, dtype=np.int64))
        meta_engines.append({"tag": tag, "engine": fn, "in_shape": ish, "in_dtype": idt, "n": len(paths)})

    import tensorrt as trt
    meta = {"board": "NVIDIA Jetson AGX Orin Developer Kit (64GB)", "trt": trt.__version__,
            "nvpmodel": "MAXN", "n_eval": len(paths),
            "preprocess": "force-resize 800x1066, ImageNet norm (detr_prep.py)",
            "engines": meta_engines,
            "note": "SmoothQuant (ONNX-level, 95 Gemms, per-input-channel s=a^alpha/w^(1-alpha)) + "
                    "symmetric QInt8 QDQ (Conv+Gemm, QuantizeBias=False, exclude conv1). "
                    "a10=alpha 1.0 (stage2 §4.4 DETR-best), a05=alpha 0.5. ONLY delta vs "
                    "detr_gpu_int8_sym.plan is the SmoothQuant activation migration."}
    json.dump(meta, open(os.path.join(args.out, "orin_detr_sq_meta.json"), "w"), indent=2)
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
