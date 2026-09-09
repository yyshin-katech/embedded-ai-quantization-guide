#!/usr/bin/env python3
"""Synthetic fixed-compute / swept-output models for the DX-M1 regime transition curve.

Design (variable isolation): a FIXED convolutional trunk provides fixed NPU compute
(fixed profiler "Inference" p50), a 4-channel bottleneck decouples the head's compute
from its output size, and a final 1x1 "expander" conv 4->C_out produces an output of
[1, C_out, 14, 14].  Output bytes ~= 784 * C_out (fp32), while the expander's own compute
(4 * C_out * 196 MACs) stays negligible vs the trunk.  Sweeping C_out therefore sweeps the
D2H (device->host) transfer size at ~constant compute -> it walks the model across the
compute-bound <-> D2H-bound boundary that the crossover axis (274bffa) proved exists at the
extremes but did not trace.

Runs on x86 in dxcom-venv (dx_com is x86-only); emits ONNX + compiled .dxnn per point.
The .dxnn are copied to the Pi5 host and benchmarked with dxrun/dxbenchmark.

Accuracy is irrelevant on this axis (random weights, random calib) -- only latency/regime
are claimed, exactly like the BEV capstone init-weight caveat.  dxrun -b/dxbenchmark
synthesize their own input, so no real data is needed at runtime.
"""
import argparse, os, sys, json, time, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dx_com

SPATIAL = 14          # head spatial H=W
CBOT = 4              # bottleneck channels (decouples head compute from output size)
BYTES_PER_ELEM = 4    # nominal fp32; actual moved bytes read from the runtime on-device


def cbr(ci, co, s):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Trunk(nn.Module):
    """224x224x3 -> (heavy body @28x28) -> 14x14xCMID -> 1x1 conv to CBOT ch.

    body_depth sets the FIXED compute level (T).  We size T so that Inference > H2D
    (input is fixed 150 KB -> H2D ~0.86 ms), otherwise the serial H2D would mask the
    compute-bound regime at small output.  Target Inference ~2.5 ms (resnet50-like)."""
    def __init__(self, cmid=256, body_depth=8):
        super().__init__()
        layers = [cbr(3, 64, 2),       # 112
                  cbr(64, 128, 2),     # 56
                  cbr(128, cmid, 2)]   # 28
        for _ in range(body_depth):
            layers.append(cbr(cmid, cmid, 1))  # heavy 28x28 body (462 MMACs each @cmid=256)
        layers.append(cbr(cmid, cmid, 2))      # 14
        self.stem = nn.Sequential(*layers)
        self.bottleneck = nn.Conv2d(cmid, CBOT, 1, 1, 0, bias=True)  # -> [1,CBOT,14,14]

    def forward(self, x):
        return self.bottleneck(self.stem(x))


class Net(nn.Module):
    def __init__(self, cout, cmid=256, body_depth=8):
        super().__init__()
        self.trunk = Trunk(cmid, body_depth)
        self.head = nn.Conv2d(CBOT, cout, 1, 1, 0, bias=True)  # expander -> [1,cout,14,14]

    def forward(self, x):
        return self.head(self.trunk(x))


def count_macs(cmid=256, body_depth=8):
    m = 0
    m += 3 * 64 * 9 * 112 * 112
    m += 64 * 128 * 9 * 56 * 56
    m += 128 * cmid * 9 * 28 * 28
    m += body_depth * (cmid * cmid * 9 * 28 * 28)
    m += cmid * cmid * 9 * 14 * 14
    return m


class RandCalib(Dataset):
    """Random normalized [3,224,224] float tensors; accuracy is not a claim on this axis."""
    def __init__(self, n):
        self.n = n
        g = np.random.RandomState(0)
        self.data = g.standard_normal((n, 3, 224, 224)).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i])


def export_onnx(cout, path, body_depth=8):
    net = Net(cout, body_depth=body_depth).eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = net(dummy)
    torch.onnx.export(net, dummy, path, input_names=["input"], output_names=["output"],
                      opset_version=13, dynamo=False)
    return tuple(y.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--couts", required=True, help="comma-separated output-channel counts")
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--dxnn-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--calib-num", type=int, default=20)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--body-depth", type=int, default=8,
                    help="# of heavy 28x28 conv layers = FIXED compute level T (constant across sweep)")
    args = ap.parse_args()
    print(f"[design] body_depth={args.body_depth}  fixed-trunk MACs={count_macs(body_depth=args.body_depth)/1e9:.2f} G "
          f"(resnet50~4.1G); output swept via head expander only", flush=True)

    for d in (args.onnx_dir, args.dxnn_dir, args.log_dir):
        os.makedirs(d, exist_ok=True)

    ds = RandCalib(args.calib_num)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    couts = [int(c) for c in args.couts.split(",") if c.strip()]

    manifest = []
    for cout in couts:
        name = f"tr_c{cout}"
        onnx_path = os.path.join(args.onnx_dir, name + ".onnx")
        outdir = os.path.join(args.dxnn_dir, name)
        os.makedirs(outdir, exist_ok=True)
        nominal_bytes = cout * SPATIAL * SPATIAL * BYTES_PER_ELEM
        print(f"\n==== {name}  cout={cout}  nominal_out={nominal_bytes} B "
              f"({nominal_bytes/1024:.1f} KB) ====", flush=True)
        oshape = export_onnx(cout, onnx_path, body_depth=args.body_depth)
        print(f"[onnx] {onnx_path}  out_shape={oshape}  size={os.path.getsize(onnx_path)} B",
              flush=True)

        rec = {"name": name, "cout": cout, "out_shape": list(oshape),
               "nominal_out_bytes": nominal_bytes, "onnx_bytes": os.path.getsize(onnx_path),
               "body_depth": args.body_depth,
               "trunk_gmacs": round(count_macs(body_depth=args.body_depth) / 1e9, 3)}
        logpath = os.path.join(args.log_dir, name + "_compile.log")
        t0 = time.perf_counter()
        # capture dx_com stdout+stderr (progress + any NPU/CPU split structure) to a log file
        old1, old2 = os.dup(1), os.dup(2)
        lf = open(logpath, "w")
        os.dup2(lf.fileno(), 1)
        os.dup2(lf.fileno(), 2)
        ok = True
        try:
            dx_com.compile(model=onnx_path, output_dir=outdir, dataloader=dl,
                           calibration_method="minmax", calibration_num=args.calib_num,
                           opt_level=args.opt_level, output_name=name, export_html=False)
        except Exception:
            ok = False
            traceback.print_exc()
        finally:
            os.dup2(old1, 1); os.dup2(old2, 2)
            os.close(old1); os.close(old2)
            lf.close()
        dt = time.perf_counter() - t0
        prod = [f for f in os.listdir(outdir) if f.endswith(".dxnn")]
        rec["compile_ok"] = ok and bool(prod)
        rec["compile_s"] = round(dt, 1)
        rec["dxnn"] = prod
        rec["dxnn_bytes"] = (os.path.getsize(os.path.join(outdir, prod[0])) if prod else 0)
        # scan compile log for the graph-partition structure (NPU vs CPU groups)
        try:
            txt = open(logpath, errors="ignore").read()
            import re
            groups = re.findall(r'(\d+)\s+(NPU|CPU)\s+group', txt)
            rec["groups"] = groups
        except Exception:
            rec["groups"] = []
        print(f"[compile] ok={rec['compile_ok']} {dt:.1f}s -> {prod} "
              f"({rec['dxnn_bytes']} B) groups={rec['groups']}", flush=True)
        manifest.append(rec)

    mpath = os.path.join(args.dxnn_dir, "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\n[manifest] {mpath}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
