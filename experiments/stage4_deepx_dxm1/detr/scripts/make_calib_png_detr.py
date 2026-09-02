#!/usr/bin/env python3
"""dx_com calibration set for DETR: force-resize COCO val2017[start:start+n] to 800x1066
and save as PNG for the config-path compile (default_loader). DISJOINT from the eval
subset (val2017[0:500]) so calibration never sees eval data (mirrors make_calib_png_det.py).

Saved RGB via PIL -> dx_com's loader reads BGR (cv2) then the config's convertColor
BGR2RGB restores true RGB, matching the FP32 reference and the eval .npy. Pre-resized to
800x1066 so the config needs no resize; div/normalize/transpose are folded by dx_com.
"""
import argparse
import os

from PIL import Image

from detr_det_common import load_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="/home/yuyeong/embedded-ai-quantization-guide/_workspace/coco")
    ap.add_argument("--start", type=int, default=500, help="disjoint from eval [0:500]")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    img_dir = os.path.join(args.coco, "val2017")
    files = sorted(os.listdir(img_dir))[args.start: args.start + args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in files:
        u8 = load_u8(os.path.join(img_dir, fn))            # (800,1066,3) uint8 RGB
        Image.fromarray(u8).save(os.path.join(args.out_dir, os.path.splitext(fn)[0] + ".png"))
    print(f"saved {len(files)} calib PNGs (val2017[{args.start}:{args.start+len(files)}]) -> {args.out_dir}")


if __name__ == "__main__":
    main()
