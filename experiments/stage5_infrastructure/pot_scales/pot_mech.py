#!/usr/bin/env python3
# Mechanism probe for the PoT NO-GO: why does forcing every QDQ scale to a power of
# two make cross-device INT8 agreement WORSE (baseline 958/1000 -> 869/1000 ceil)
# instead of restoring it to n/n the way Chen 2026 saw on a single GPU?
#
# pot_agree.py delivers the verdict (FAIL). This script delivers the reason, from the
# raw logits. Two readings of the failure, both named in README.md's residual risks:
#   (1) PoT enlarges the raw cross-kernel divergence -- the float32 requant epilogue
#       diverges MORE between the x86 (no-VNNI MLAS) and A76 (SDOT MLAS) kernels.
#   (2) PoT leaves the per-element divergence about the same size but pushes more
#       decisions onto a knife edge (more near-exact .5 ties), so the SAME cross-kernel
#       noise flips more argmax decisions. This is risk #2 in the runbook.
# They are separated by measuring, per precision, BOTH the cross-device logit
# divergence magnitude AND the top1-top2 decision-margin distribution, then locating
# the flipped images on that margin axis. If flips sit at tiny margins and PoT has
# more tiny-margin images, (2) dominates; if PoT's |Dlogit| is itself larger, (1) does.
#
# Inputs: (n,1000) float32 .npy logits from pot_bench.py --save-logits, one per
# (device, precision). Every pair is compared at MATCHED precision only -- comparing
# x86/int8 against pi/int8_pot would confound the rewrite with the ISA.
import argparse
import json
import os

import numpy as np


def load(path):
    # promote to f64 so the subtraction and the tie counting are not themselves lossy
    return np.load(path).astype(np.float64)


def margins(logits):
    """Per-image gap between the top-1 and top-2 logit -- how borderline the argmax is.
    A small margin means a small cross-kernel perturbation can flip the prediction."""
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]


def summ(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {
        "max": float(x.max()),
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
    }


TAUS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def analyze(precision, a_x86, a_pi):
    n = len(a_x86)
    top_x, top_p = a_x86.argmax(1), a_pi.argmax(1)
    flip = top_x != top_p
    d = np.abs(a_x86 - a_pi)          # per-element cross-device |Dlogit|
    perimg = d.max(1)                 # worst single element per image
    m_x = margins(a_x86)              # decision margin on x86 (reference device)
    return {
        "precision": precision,
        "n": int(n),
        "agree": int((~flip).sum()),
        "differ": int(flip.sum()),
        "elem_abs_diff": summ(d.reshape(-1)),        # size of the cross-kernel noise
        "perimg_max_abs_diff": summ(perimg),
        "margin_x86_all": summ(m_x),                 # how borderline decisions are
        "borderline_le_tau": {("%.0e" % t): int((m_x < t).sum()) for t in TAUS},
        "flip_margin_x86": summ(m_x[flip]),          # margins where flips happen
        "noflip_margin_x86": summ(m_x[~flip]),       # margins where they don't
        # of the flips, how many are explained by a borderline x86 margin
        "flips_with_margin_le_1e-2": int((m_x[flip] < 1e-2).sum()) if flip.any() else 0,
        "flips_with_margin_le_1e-1": int((m_x[flip] < 1e-1).sum()) if flip.any() else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x86-int8", required=True)
    ap.add_argument("--pi-int8", required=True)
    ap.add_argument("--x86-pot", required=True)
    ap.add_argument("--pi-pot", required=True)
    ap.add_argument("--pot-label", default="int8_pot",
                    help="label for the PoT arm (int8_pot ceil / int8_potn nearest)")
    ap.add_argument("--out", help="write the full report as JSON")
    args = ap.parse_args()

    base = analyze("int8", load(args.x86_int8), load(args.pi_int8))
    pot = analyze(args.pot_label, load(args.x86_pot), load(args.pi_pot))

    def line(label, base_v, pot_v):
        print("  %-30s baseline %-16s  PoT %-16s" % (label, base_v, pot_v))

    print("=" * 78)
    print("PoT MECHANISM PROBE  (x86 no-VNNI MLAS  vs  A76 SDOT MLAS, matched precision)")
    print("=" * 78)
    print("cross-device top-1 agreement out of %d:" % base["n"])
    line("agree / differ", "%d / %d" % (base["agree"], base["differ"]),
         "%d / %d" % (pot["agree"], pot["differ"]))
    print()
    print("(1) is the cross-kernel logit divergence LARGER under PoT?")
    line("|Dlogit| per-element  mean", "%.4g" % base["elem_abs_diff"]["mean"],
         "%.4g" % pot["elem_abs_diff"]["mean"])
    line("|Dlogit| per-element  p99", "%.4g" % base["elem_abs_diff"]["p99"],
         "%.4g" % pot["elem_abs_diff"]["p99"])
    line("|Dlogit| per-element  max", "%.4g" % base["elem_abs_diff"]["max"],
         "%.4g" % pot["elem_abs_diff"]["max"])
    line("per-image worst |Dlogit| p50", "%.4g" % base["perimg_max_abs_diff"]["p50"],
         "%.4g" % pot["perimg_max_abs_diff"]["p50"])
    print()
    print("(2) does PoT push more decisions onto a knife edge (smaller margins)?")
    line("median top1-top2 margin", "%.4g" % base["margin_x86_all"]["p50"],
         "%.4g" % pot["margin_x86_all"]["p50"])
    for t in TAUS:
        k = "%.0e" % t
        line("images with margin < %s" % k,
             str(base["borderline_le_tau"][k]), str(pot["borderline_le_tau"][k]))
    print()
    print("where do the flips sit on the margin axis? (flips at tiny margins => risk #2)")
    line("median margin of FLIPPED imgs",
         "%.4g" % base["flip_margin_x86"]["p50"] if base["flip_margin_x86"] else "-",
         "%.4g" % pot["flip_margin_x86"]["p50"] if pot["flip_margin_x86"] else "-")
    line("median margin of AGREED imgs",
         "%.4g" % base["noflip_margin_x86"]["p50"] if base["noflip_margin_x86"] else "-",
         "%.4g" % pot["noflip_margin_x86"]["p50"] if pot["noflip_margin_x86"] else "-")
    line("flips with x86 margin < 1e-1",
         "%d / %d" % (base["flips_with_margin_le_1e-1"], base["differ"]),
         "%d / %d" % (pot["flips_with_margin_le_1e-1"], pot["differ"]))
    print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"baseline": base, "pot": pot}, f, indent=2)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
