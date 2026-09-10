#!/usr/bin/env python3
# Pairwise cross-platform agreement over prediction dumps, and the PoT go/no-go gate.
#
# Reads any mix of:
#   * ../cpu_proxy/raw/*.json          (the C2 baseline runs; pred_cls, no logits_md5)
#   * pot_bench.py output              (pred_cls + logits_md5)
# and prints, for every pair, the top-1 prediction agreement out of n -- the same
# metric the paper's C2 table reports -- plus, when both sides carry a logits digest,
# whether the raw outputs are bit-identical.
#
# THE GATE. The baseline, from ../cpu_proxy/README.md (SSOT):
#     fp32  imx8mn(A53) <-> pi5(A76)   1000/1000        int8  965/1000
#     fp32  imx8mn(A53) <-> x86        1000/1000        int8  961/1000
#     fp32  x86         <-> pi5(A76)   1000/1000        int8  958/1000
# PoT succeeds on a pair iff that pair reaches n/n on INT8. Anything short of n/n --
# even 999/1000 -- is a NO-GO for the cross-device determinism claim: the point of
# power-of-two scales is that the epilogue becomes a shift, which either is or is not
# exactly reproduced.
#
# WHAT THE GATE SCORES, and what it must not. Only pairs where BOTH runs are PoT runs
# (precision "int8_pot") are scored. The baseline int8 <-> int8 pairs above are the
# control: they are *known* to sit at 958-965/1000 and can never reach n/n, so scoring
# them would make --gate fail unconditionally no matter how well PoT worked. They are
# printed for contrast and excluded. Cross-precision pairs (a board's own int8 vs. its
# int8_pot) measure the rewrite's prediction drift on one device -- informational, and
# the place to read the accuracy cost -- never a gate.
#
# Usage:
#   python pot_agree.py a.json b.json [c.json ...]
#   python pot_agree.py --baseline ../cpu_proxy/raw --glob 'results/*_pot.json'
import argparse
import glob as globmod
import itertools
import json
import os
import sys

import numpy as np


def is_pot(precision):
    """A PoT run is one produced from a power-of-two-rewritten model, which the runbook
    labels `int8_pot`. Anything else that starts with int8 is a baseline run."""
    return precision.startswith("int8") and "pot" in precision


def load(path):
    with open(path) as f:
        d = json.load(f)
    if "pred_cls" not in d:
        raise SystemExit("%s has no pred_cls -- not a prediction dump" % path)
    d["_path"] = path
    d["_tag"] = "%s/%s" % (d.get("soc", "?"), d.get("precision", "?"))
    d["_pred"] = np.asarray(d["pred_cls"], dtype=np.int64)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="*", help="prediction dumps to compare")
    ap.add_argument("--baseline", help="directory of baseline dumps to include")
    ap.add_argument("--glob", action="append", default=[],
                    help="glob of further dumps (repeatable)")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero unless every INT8 pair agrees n/n")
    args = ap.parse_args()

    paths = list(args.jsons)
    if args.baseline:
        paths += sorted(globmod.glob(os.path.join(args.baseline, "*.json")))
    for g in args.glob:
        paths += sorted(globmod.glob(g))
    paths = list(dict.fromkeys(paths))
    if len(paths) < 2:
        ap.error("need at least two prediction dumps (got %d)" % len(paths))

    runs = [load(p) for p in paths]
    n = len(runs[0]["_pred"])
    for r in runs:
        if len(r["_pred"]) != n:
            raise SystemExit("%s has %d predictions, expected %d -- the runs must "
                             "cover the same image bundle in the same order"
                             % (r["_path"], len(r["_pred"]), n))

    print("%-34s %-12s %-9s %s" % ("run", "top-1", "md5", "file"))
    for r in runs:
        print("%-34s %-12.4f %-9s %s"
              % (r["_tag"], r.get("accuracy", float("nan")),
                 (r.get("logits_md5") or "-")[:8], os.path.basename(r["_path"])))

    print()
    print("%-13s %-54s %14s %8s %s"
          % ("precision", "pair", "agreement", "differ", "logits"))
    gated, control = [], []
    for a, b in itertools.combinations(runs, 2):
        pa, pb = a.get("precision", "?"), b.get("precision", "?")
        same = int((a["_pred"] == b["_pred"]).sum())
        ma, mb = a.get("logits_md5"), b.get("logits_md5")
        if ma and mb:
            bits = "identical" if ma == mb else "differ"
        else:
            bits = "n/a"
        pair = "%s <-> %s" % (a["_tag"], b["_tag"])
        print("%-13s %-54s %8d/%-5d %8d %s"
              % (pa if pa == pb else "%s|%s" % (pa, pb), pair, same, n, n - same, bits))
        if pa == pb and pa.startswith("int8"):
            (gated if is_pot(pa) else control).append((pair, same, bits))

    print()
    if control:
        print("baseline INT8 pairs (control -- expected to disagree, NOT gated):")
        for pair, same, _ in control:
            print("   %-54s %d/%d (%d differ)" % (pair, same, n, n - same))
        print()

    failures = [(pair, same) for pair, same, _ in gated if same != n]
    if not gated:
        print("GATE: N/A -- no int8_pot <-> int8_pot pair in this comparison.")
        print("      Feed pot_bench.py runs labelled --precision int8_pot from two or")
        print("      more targets; the baseline dumps alone cannot satisfy the gate.")
        if args.gate:
            sys.exit(2)
        return
    if failures:
        print("GATE: FAIL -- %d of %d PoT INT8 pair(s) short of %d/%d:"
              % (len(failures), len(gated), n, n))
        for pair, same in failures:
            print("   %-54s %d/%d (%d differ)" % (pair, same, n, n - same))
        if args.gate:
            sys.exit(1)
        return
    print("GATE: PASS -- all %d PoT INT8 pair(s) agree %d/%d." % (len(gated), n, n))
    bits = set(b for _, _, b in gated)
    if bits == set(["identical"]):
        print("      logits bit-identical on every gated pair -- full GO.")
    elif "differ" in bits:
        print("      but logits DIFFER on at least one pair: predictions agree while")
        print("      the raw outputs do not. Partial success -- report it as such.")
    else:
        print("      logits digests unavailable on at least one side (baseline dumps")
        print("      carry none), so bit-identity is unproven -- top-1 agreement only.")


if __name__ == "__main__":
    main()
