#!/usr/bin/env python3
# Finding (a): does dx_com honor / ignore / REJECT externally-supplied QDQ scales?
#
# The Qualcomm-HTP parallel: on Hexagon HTP a BYO ORT-QDQ model COMPILED fine but the
# runtime SILENTLY ignored the external scales -> top-1 collapse 0.75->0.005. Here we ask
# the same of DEEPX dx_com and get the OPPOSITE failure mode: a LOUD compile-time rejection.
#
# We compile two topologically-identical arms:
#   FP32 twin : resnet50_fp32.onnx        (dx_com's own PTQ -> OK)
#   QDQ arm   : resnet50_int8_qdq.onnx    (external ORT QDQ fed straight in)
# print each op-inventory (only Q/DQ nodes differ), attempt the QDQ compile, and walk the
# exception chain (dx_com wraps the real cause as a generic 'contact DEEPX' InternalError).
import argparse, os, sys, traceback
import onnx
import dx_com


def inventory(path):
    m = onnx.load(path)
    c = {}
    for n in m.graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return dict(sorted(c.items()))


def chain(e):
    out, seen, cur, lvl = [], set(), e, 0
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append("[level %d] %s: %s" % (lvl, type(cur).__name__, str(cur).splitlines()[0] if str(cur) else ""))
        cur = cur.__cause__ or cur.__context__
        lvl += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--qdq", required=True)
    ap.add_argument("--calib-png", required=True)   # for a real dataloader-free config compile
    ap.add_argument("--outdir", default="extqdq_probe")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("# dx_com external-QDQ rejection probe")
    print("# compiler:", getattr(dx_com, "version", "?"), "| dx-com", getattr(dx_com, "__version__", "?"))
    print("FP32 twin op-inventory (COMPILES OK):", inventory(args.fp32))
    print("QDQ arm  op-inventory (PREPARE-FAILS):", inventory(args.qdq))
    print("delta = only QuantizeLinear/DequantizeLinear inserted; Conv/Gemm/Add/Relu identical")
    print()
    print("=== FAILURE (chained) ===")
    try:
        # config-less minimal attempt: dx_com still parses graph structure before calibration,
        # which is where the external QDQ branches trip GraphStructureError.
        dx_com.compile(model=args.qdq, output_dir=args.outdir, calibration_num=1)
        print("!! UNEXPECTED: QDQ arm compiled (no rejection)")
    except Exception as e:
        for line in chain(e):
            print(line)
        sys.exit(0)


if __name__ == "__main__":
    main()
