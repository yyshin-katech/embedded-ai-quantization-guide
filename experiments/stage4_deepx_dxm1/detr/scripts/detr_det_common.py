#!/usr/bin/env python3
"""Shared DETR preprocessing for the DX-M1 axis — force-resize to the fixed ONNX
input 800x1066 (aspect ratio NOT preserved, matching stage3 detr_prep.py) so every
image maps exactly onto detr_sim.onnx's pixel_values [1,3,800,1066].

The SAME uint8 [800,1066,3] RGB pixels feed both backends:
  * FP32 reference (onnxruntime, x86)  -> to_fp32_nchw() applies Div/255 + ImageNet
    normalize + transpose EXPLICITLY.
  * INT8 NPU (dx_engine, Pi)           -> those same Div/normalize/transpose steps are
    FOLDED into the .dxnn by dx_com; the caller supplies the raw uint8 RGB.
So the only difference between the two paths is quantization (same rationale as the
detection axis' eval_u8.npy). DETR boxes are resolution-normalised, so the fixed-resize
does not enter the box math — postprocess scales by each image's ORIGINAL W,H.
"""
import numpy as np
from PIL import Image

H, W = 800, 1066                                     # fixed ONNX input (matches detr_sim.onnx)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_u8(path):
    """JPEG path -> (800,1066,3) uint8 RGB, force-resize (aspect NOT preserved)."""
    im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)   # PIL size = (W,H)
    return np.asarray(im, dtype=np.uint8)


def to_fp32_nchw(u8):
    """(800,1066,3) uint8 RGB -> (1,3,800,1066) float32 NCHW: Div/255 + ImageNet norm."""
    a = u8.astype(np.float32) / 255.0
    a = (a - MEAN) / STD
    a = a.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(a, dtype=np.float32)
