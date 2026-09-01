#!/usr/bin/env python3
"""Shared decode for the DX-M1 YOLO26n detection accuracy axis.

The FP32 reference (onnxruntime, x86) and the INT8 NPU (dx_engine, DX-M1) emit the
SAME end2end tensor: output0 [1,300,6] = [x1,y1,x2,y2,score,class] in 640x640
letterbox-pixel space, class 0..79 (contiguous COCO). NMS is folded into the graph.
Because both backends share this exact format, decode is identical → any mAP gap is
quantization (INT8) vs FP32, cleanly isolated (mirrors the accuracy axis' top-1 method).

Letterbox: centered pad to 640, pad value 114, PIL BILINEAR. Verified on COCO val
000000000139.jpg (living room) → person/chair/potted-plant/dining-table/tv/clock/vase.
"""
import numpy as np
# NOTE: PIL is imported lazily inside letterbox() so this module also imports on the
# Pi's dx-runtime venv (numpy-only, no PIL) where only decode()/COCO80_TO_91 are used.

S = 640
PAD = 114

# contiguous YOLO class 0..79 -> COCO category_id (1..90, non-contiguous)
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def letterbox(pil_rgb):
    """PIL RGB image -> (canvas uint8 NHWC RGB [640,640,3], r, left, top, w0, h0)."""
    from PIL import Image
    w0, h0 = pil_rgb.size
    r = min(S / h0, S / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    rz = np.asarray(pil_rgb.resize((nw, nh), Image.BILINEAR))
    canvas = np.full((S, S, 3), PAD, np.uint8)
    left, top = (S - nw) // 2, (S - nh) // 2
    canvas[top:top + nh, left:left + nw] = rz
    return canvas, r, left, top, w0, h0


def decode(out, image_id, r, left, top, w0, h0, conf=0.001):
    """out [300,6] xyxy@640 -> list of COCO det dicts in original-image xywh."""
    dets = []
    for x1, y1, x2, y2, sc, cls in out:
        if sc < conf:
            continue
        # un-letterbox to original pixels
        ox1 = (x1 - left) / r
        oy1 = (y1 - top) / r
        ox2 = (x2 - left) / r
        oy2 = (y2 - top) / r
        ox1 = min(max(ox1, 0.0), w0)
        ox2 = min(max(ox2, 0.0), w0)
        oy1 = min(max(oy1, 0.0), h0)
        oy2 = min(max(oy2, 0.0), h0)
        bw, bh = ox2 - ox1, oy2 - oy1
        if bw <= 0 or bh <= 0:
            continue
        dets.append({
            "image_id": int(image_id),
            "category_id": int(COCO80_TO_91[int(cls)]),
            "bbox": [round(float(ox1), 3), round(float(oy1), 3),
                     round(float(bw), 3), round(float(bh), 3)],
            "score": round(float(sc), 5),
        })
    return dets
