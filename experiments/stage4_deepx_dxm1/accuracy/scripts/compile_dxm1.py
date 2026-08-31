#!/usr/bin/env python3
# DX-M1 compile driver (runs on x86 AI-LAP in dxcom-venv; dx_com is x86-only).
#
# Two arms for the accuracy axis:
#   native  : FP32 ONNX (resnet50_fp32.onnx) + dx_com's OWN PTQ calibration (ema/minmax)
#             -> the honest DEEPX-native INT8 path.
#   extqdq  : the pre-quantized ORT QDQ ONNX (resnet50_int8_qdq.onnx) fed straight in
#             -> probes whether dx_com honors / ignores / rejects external QDQ scales
#             (the Qualcomm-HTP parallel).
#
# PITFALL (empirical): the python DataLoader path below does NOT let you keep preprocessing
# out of the graph. dx_com ALWAYS folds div/normalize into the .dxnn (runtime input becomes
# uint8 NHWC, get_input_size=150528), and via the DataLoader path it folded a WRONG default
# normalize -> on-device top1=0. The ACCURACY-VALID native build was produced via the CLI
# CONFIG path instead (see native_cfg.json + raw/native_compile.log): raw-uint8 PNG calib
# with explicit div/255 + ImageNet normalize preprocessings, which the compiler folds
# correctly (it prints "provide uint8 HWC input directly"). This script is retained as the
# external-QDQ rejection driver / reference; for the native arm use:
#   dxcom -m resnet50_fp32.onnx -c native_cfg.json -o OUTDIR --opt_level 1
# and feed raw uint8 NHWC at runtime (npu_infer.py auto-detects the contract).
import argparse, os, sys, time, traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dx_com

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class CalibNCHW(Dataset):
    """Yields normalized [3,224,224] float32 tensors (DataLoader adds the batch dim).
    Same preprocessing as rpi_bench.py / cpu_infer.py: /255, HWC->CHW, ImageNet norm."""
    def __init__(self, u8_path, n):
        u8 = np.load(u8_path)[:n]                       # (n,224,224,3) u8
        self.u8 = u8
    def __len__(self):
        return self.u8.shape[0]
    def __getitem__(self, i):
        x = self.u8[i].astype(np.float32) / 255.0      # HWC
        x = np.transpose(x, (2, 0, 1))                 # CHW
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)          # calib_u8.npy (native arm only)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", required=True)           # resnet50_native | resnet50_extqdq
    ap.add_argument("--calib-method", default="ema")   # ema | minmax | iqr
    ap.add_argument("--calib-num", type=int, default=100)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--export-html", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = CalibNCHW(args.calib, args.calib_num)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    # sanity: show one batch shape
    b0 = next(iter(dl))
    print("[compile] calib batch shape:", tuple(b0.shape), b0.dtype,
          "min/max %.3f/%.3f" % (float(b0.min()), float(b0.max())))
    print("[compile] model=%s name=%s method=%s num=%d opt=%d"
          % (os.path.basename(args.model), args.name, args.calib_method, args.calib_num, args.opt_level))

    t0 = time.perf_counter()
    try:
        dx_com.compile(
            model=args.model,
            output_dir=args.outdir,
            dataloader=dl,
            calibration_method=args.calib_method,
            calibration_num=args.calib_num,
            opt_level=args.opt_level,
            output_name=args.name,
            export_html=args.export_html,
        )
    except Exception:
        dt = time.perf_counter() - t0
        print("[compile] FAILED after %.1fs" % dt)
        traceback.print_exc()
        sys.exit(3)
    dt = time.perf_counter() - t0

    # locate produced .dxnn
    prod = [f for f in os.listdir(args.outdir) if f.endswith(".dxnn")]
    print("[compile] OK in %.1fs -> %s" % (dt, prod))
    for f in prod:
        p = os.path.join(args.outdir, f)
        print("   %s  %d bytes" % (f, os.path.getsize(p)))


if __name__ == "__main__":
    main()
