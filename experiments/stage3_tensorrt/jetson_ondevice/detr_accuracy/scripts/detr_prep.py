#!/usr/bin/env python3
# detr_prep.py — shared, byte-identical preprocessing for BOTH the host-side
# symmetric-QDQ calibration AND the on-board eval. The stage3 DETR engines are
# built from the FIXED-shape ONNX `detr_sim.onnx` (pixel_values [1,3,800,1066]),
# so every image is force-resized to exactly 800x1066 (aspect ratio NOT preserved).
# This distorts boxes vs stage2's dynamic-shape run — hence absolute mAP is not
# comparable to stage2 (0.4207); only the same-preprocessing FP32→INT8 *relative*
# delta is the result. DETR normalisation = ImageNet mean/std after ÷255,
# matching transformers DetrImageProcessor defaults.
import numpy as np
from PIL import Image

H, W = 800, 1066                                   # fixed ONNX input (matches detr_sim.onnx)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    """JPEG path -> (1,3,800,1066) float32 NCHW, contiguous."""
    im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)  # PIL size = (W,H)
    a = np.asarray(im, dtype=np.float32) / 255.0                          # HWC in [0,1]
    a = (a - MEAN) / STD                                                  # ImageNet norm
    a = a.transpose(2, 0, 1)[None]                                        # 1,C,H,W
    return np.ascontiguousarray(a, dtype=np.float32)
