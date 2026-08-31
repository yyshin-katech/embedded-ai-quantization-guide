#!/usr/bin/env python3
# Regenerate the 100-image PNG calibration set dx_com's default_loader reads.
#
# dx_com calibrates PTQ from image files via a config default_loader (NOT the python
# DataLoader path, which folded a WRONG default normalize -> top1=0). We therefore dump
# raw-pixel PNGs and let native_cfg.json apply div/255 + ImageNet normalize at calibration
# and FOLD them into the .dxnn (the compiler skips convertColor/expandDim; see
# raw/native_compile.log). The calib split is DISJOINT from the 1000-image eval bundle.
#
# Source: calib_u8.npy = tv.npy[calib_idx] (u8 NHWC RGB), a non-eval ImageNet-val slice.
# We save PNGs via cv2.imwrite, which expects BGR, so we pass RGB->BGR; native_cfg's
# convertColor BGR2RGB then restores RGB at load -> calibration sees RGB, matching the
# RGB uint8 fed at runtime (mean/std are RGB-order). This round-trip is why convertColor
# is in the config even though it is not folded into the runtime graph.
import argparse, os
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-npy", required=True)   # calib_u8.npy (u8 NHWC RGB)
    ap.add_argument("--out", required=True)          # calib_png dir
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    u8 = np.load(args.calib_npy)[:args.n]            # (n,224,224,3) RGB u8
    for i in range(u8.shape[0]):
        bgr = cv2.cvtColor(u8[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.out, "calib_%04d.png" % i), bgr)
    print("wrote %d PNGs -> %s" % (u8.shape[0], args.out))


if __name__ == "__main__":
    main()
