#!/usr/bin/env python3
# Rewrite every QuantizeLinear / DequantizeLinear scale in a QDQ ONNX graph to the
# nearest power of two:  s -> 2^round(log2 s).
#
# WHY. INT32 accumulation in an INT8 GEMM is exact, so two integer kernels cannot
# disagree there. The divergence measured in the paper's C2 (958-965/1000 top-1
# agreement across CPUs that do not share an integer kernel) is introduced by the
# re-quantization *epilogue*
#
#       out_int8 = round(acc_int32 * M) + zp,      M = (s_a * s_w) / s_out
#
# where M is an arbitrary float that each kernel approximates its own way (fixed-point
# multiplier + shift, FP32 multiply, FMA order, rounding mode). Force every scale to a
# power of two and M = 2^(k_a + k_w - k_out) becomes an exact power of two -- a pure
# arithmetic shift, which every kernel must implement identically. Chen (2026) showed
# this restores bit-identical agreement for two INT8 GEMM kernels on ONE GPU; whether
# it holds across *physical* device boundaries is what this directory tests.
#
# WHAT IT TOUCHES.
#   * activation scales: replaced (there is no stored tensor -- the scale IS the grid)
#   * weight scales:     replaced, and the int8 weight initializer is re-quantized
#                        against the new scale so the represented real values move as
#                        little as possible. Without that step the scale change --
#                        up to 2x under the default `ceil`, up to sqrt(2) under
#                        `nearest` -- silently rescales every weight.
#   Per-channel (vector) scales are handled elementwise, honouring the DQ `axis`.
#
# SCOPE LIMIT. Only the top-level `model.graph` is rewritten. Q/DQ nodes inside a
#   nested subgraph -- an If/Loop/Scan body -- are NOT touched. Rewriting those
#   correctly needs an outer-scope initializer chain (a subgraph inherits initializers
#   by name and may shadow them) and a (scope, name) key for the shared-weight guard:
#   more machinery than the target graph justifies, and every line of it is a new place
#   for a silent miss. So they are *detected* instead -- counted into the report,
#   warned about, re-checked after the write, and fatal. One unrewritten scale makes M
#   a non-power-of-two and voids the experiment, so the tool refuses to write a model
#   it cannot honestly call PoT rather than quietly under-covering it.
#
# OUTPUT INVARIANT. --out is written via a `.partial` scratch file and moved into place
#   only after check() passes, so *a file existing at --out means it verified*. That
#   holds without anyone reading the exit code, which is the point: a non-PoT model
#   left under a PoT name is this experiment's silent-wrong failure mode.
#
# ASSUMPTION (checked, not assumed): the target graph is symmetric, zero_point == 0
# everywhere -- which is how resnet50_int8_qdq.onnx is built
# (see ../../stage3_tensorrt/t02_latency_3point.py: ActivationSymmetric/WeightSymmetric,
#  QuantizeBias=False). Non-zero zero_points are reported and left arithmetically
#  correct, but they add a second divergence source that PoT does not remove.
#
# Usage:
#   python pot_rewrite.py --in resnet50_int8_qdq.onnx --out resnet50_int8_pot.onnx
#   python pot_rewrite.py --in model.onnx --dry-run          # report only, write nothing
#   python pot_rewrite.py --selftest                         # numpy only, no onnx needed
import argparse
import json
import os
import sys

import numpy as np

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")
# float32 powers of two are exact for these exponents; stay clear of subnormals.
K_MIN, K_MAX = -126, 127


# --------------------------------------------------------------------------- math
def pot_exp(s, mode="ceil"):
    """Power-of-two exponent for every entry of s (float array).

    mode="ceil"    : k = ceil(log2 s)  ->  s' >= s. The grid only ever gets coarser,
                     so nothing that fitted in the int range before can fall outside
                     it now. Costs up to one bit of resolution. Default.
    mode="nearest" : k = round(log2 s) ->  s' within a factor sqrt(2) of s. Closest
                     grid, but where s' < s the int range no longer reaches the
                     original extremes and the largest-magnitude values saturate.
    """
    s = np.asarray(s, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError("non-finite scale")
    if np.any(s <= 0):
        raise ValueError("non-positive scale: %r" % (s[s <= 0][:4],))
    lg = np.log2(s)
    if mode == "ceil":
        k = np.ceil(lg).astype(np.int64)
    elif mode == "nearest":
        k = np.rint(lg).astype(np.int64)
    else:
        raise ValueError("unknown rounding mode %r" % (mode,))
    return np.clip(k, K_MIN, K_MAX)


def pot_scale(s, dtype=np.float32, mode="ceil"):
    """s -> 2^k, exactly representable in float32."""
    return np.exp2(pot_exp(s, mode).astype(np.float64)).astype(dtype)


def requantize(q, zp, s_old, s_new, axis, qmin, qmax):
    """Re-quantize an integer weight tensor so it represents the same real values
    under the new scale.  real = (q - zp) * s ;  q' = clip(round(real / s') + zp).
    `axis` is the per-channel axis, or None for a scalar scale."""
    q = np.asarray(q)
    real_dtype = np.float64
    s_old = np.asarray(s_old, dtype=real_dtype)
    s_new = np.asarray(s_new, dtype=real_dtype)
    zp = np.asarray(zp, dtype=real_dtype)
    if axis is not None and s_old.ndim == 1 and s_old.size > 1:
        shape = [1] * q.ndim
        shape[axis] = s_old.size
        s_old = s_old.reshape(shape)
        s_new = s_new.reshape(shape)
        if zp.ndim == 1 and zp.size == s_old.size:
            zp = zp.reshape(shape)
    real = (q.astype(real_dtype) - zp) * s_old
    raw = np.rint(real / s_new) + zp
    q_new = np.clip(raw, qmin, qmax).astype(q.dtype)
    n_sat = int(np.count_nonzero(raw != np.clip(raw, qmin, qmax)))
    back = (q_new.astype(real_dtype) - zp) * s_new
    err = np.abs(back - real)
    denom = np.maximum(np.abs(real), 1e-12)
    return q_new, float(err.max()), float((err / denom).max()), n_sat


# ------------------------------------------------------------------------- onnx io
def _load_onnx():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        sys.exit("onnx is not installed in this interpreter.\n"
                 "  pip install onnx        (>=1.14; 1.18.0 is the repo's pinned build)\n"
                 "Run --selftest to exercise the arithmetic without onnx.")
    return onnx, numpy_helper


def _qrange(dtype):
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _axis_of(node, default=1):
    for a in node.attribute:
        if a.name == "axis":
            return int(a.i)
    return default


def _subgraph_qdq(graph):
    """Q/DQ nodes living inside a nested subgraph (an If/Loop/Scan body).

    See the SCOPE LIMIT note in the header: this tool rewrites the top-level graph
    only. A Q/DQ node in a subgraph would keep its float scale while every top-level
    check still passed -- exactly the silent-miss this experiment cannot afford -- so
    find them and refuse. Returns a sorted list of "<path>/<node>" strings.
    Detection only: it walks nested bodies to any depth but rewrites nothing."""
    found = []

    def walk(g, path):
        for n in g.node:
            for a in n.attribute:
                subs = []
                one = getattr(a, "g", None)
                # protobuf hands back a default-empty GraphProto, never None, so test
                # for actual nodes rather than for the attribute's presence.
                if one is not None and getattr(one, "node", None):
                    subs.append((one, "%s/%s:%s" % (path, n.op_type, a.name)))
                for i, sg in enumerate(getattr(a, "graphs", None) or []):
                    if getattr(sg, "node", None):
                        subs.append((sg, "%s/%s:%s[%d]"
                                     % (path, n.op_type, a.name, i)))
                for sg, p in subs:
                    for sn in sg.node:
                        if sn.op_type in QDQ_OPS:
                            # node names are optional in ONNX; fall back to the op type
                            found.append("%s/%s"
                                         % (p, getattr(sn, "name", "") or sn.op_type))
                    walk(sg, p)

    walk(graph, "")
    return sorted(set(found))


def rewrite(model, mode="ceil", verbose=True):
    """Rewrite in place. Returns a report dict."""
    onnx, numpy_helper = _load_onnx()
    g = model.graph
    inits = {t.name: t for t in g.initializer}

    # ---- 1. every scale initializer feeding a Q/DQ node -----------------------
    #  A scale this loop cannot reach -- one arriving as a graph input, as the output
    #  of a Constant node, or living inside an If/Loop/Scan subgraph -- is NOT
    #  rewritten. M then stays a non-power-of-two and the experiment is void, so the
    #  misses are counted into the report, re-checked after the write, and fatal.
    scale_names, zp_names, skipped = set(), set(), set()
    for n in g.node:
        if n.op_type in QDQ_OPS and len(n.input) >= 2:
            if n.input[1] in inits:
                scale_names.add(n.input[1])
            else:
                skipped.add(n.input[1])
                if verbose:
                    print("  [warn] %s: scale %r is not an initializer (dynamic) "
                          "-- NOT rewritten" % (n.op_type, n.input[1]))
            if len(n.input) >= 3 and n.input[2] in inits:
                zp_names.add(n.input[2])

    #  ...and the Q/DQ nodes that loop cannot see at all, because they sit in a nested
    #  subgraph. Out of scope to rewrite (header), so they must at least be visible.
    subgraph_qdq = _subgraph_qdq(g)
    if subgraph_qdq and verbose:
        print("  [warn] %d Q/DQ node(s) live inside a nested If/Loop/Scan subgraph; "
              "this tool rewrites the top-level graph only -- NOT rewritten: %s"
              % (len(subgraph_qdq), subgraph_qdq[:4]))

    old_scales, new_scales, exps = {}, {}, {}
    for name in sorted(scale_names):
        s_old = numpy_helper.to_array(inits[name])
        s_new = pot_scale(s_old, dtype=s_old.dtype, mode=mode)
        old_scales[name] = s_old
        new_scales[name] = s_new
        exps[name] = pot_exp(s_old, mode)

    # ---- 2. non-zero zero_points are a second divergence source ---------------
    nonzero_zp = []
    for name in sorted(zp_names):
        z = numpy_helper.to_array(inits[name])
        if np.any(np.asarray(z) != 0):
            nonzero_zp.append(name)

    # ---- 3. re-quantize weight tensors against their new scale ----------------
    #  A weight DQ is a DequantizeLinear whose data input is an integer initializer.
    #  `inits[wname]` is mutated in place, so a weight shared by two DQ nodes must be
    #  converted exactly once -- a second pass would re-quantize the already-converted
    #  tensor from old to new scale again and silently halve/double it.
    wmax_abs, wmax_rel, n_weights, n_sat_total = 0.0, 0.0, 0, 0
    done_weights, shared_weights = set(), set()
    for n in g.node:
        if n.op_type != "DequantizeLinear" or len(n.input) < 2:
            continue
        wname, sname = n.input[0], n.input[1]
        if wname not in inits or sname not in new_scales:
            continue
        q = numpy_helper.to_array(inits[wname])
        if not np.issubdtype(q.dtype, np.integer):
            continue
        if wname in done_weights:
            shared_weights.add(wname)
            if verbose:
                print("  [warn] weight %r feeds more than one DequantizeLinear; "
                      "re-quantized once, against %r" % (wname, sname))
            continue
        done_weights.add(wname)
        zp = np.zeros((), dtype=q.dtype)
        if len(n.input) >= 3 and n.input[2] in inits:
            zp = numpy_helper.to_array(inits[n.input[2]])
        qmin, qmax = _qrange(q.dtype)
        axis = _axis_of(n) if np.asarray(old_scales[sname]).size > 1 else None
        q_new, a, r, nsat = requantize(q, zp, old_scales[sname], new_scales[sname],
                                       axis, qmin, qmax)
        t = numpy_helper.from_array(q_new, wname)
        inits[wname].CopyFrom(t)
        wmax_abs = max(wmax_abs, a)
        wmax_rel = max(wmax_rel, r)
        n_sat_total += nsat
        n_weights += 1

    # ---- 4. write the new scales ---------------------------------------------
    for name, s_new in new_scales.items():
        inits[name].CopyFrom(numpy_helper.from_array(s_new, name))

    shifts = np.concatenate([np.atleast_1d(
        np.log2(np.asarray(new_scales[n], dtype=np.float64) /
                np.asarray(old_scales[n], dtype=np.float64)).ravel())
        for n in new_scales]) if new_scales else np.zeros(1)
    all_k = np.concatenate([np.atleast_1d(exps[n]).ravel() for n in exps]) \
        if exps else np.zeros(1, dtype=np.int64)

    return {
        "rounding": mode,
        "scale_initializers_rewritten": len(new_scales),
        "scale_values_rewritten": int(sum(np.asarray(v).size
                                          for v in new_scales.values())),
        "weight_tensors_requantized": n_weights,
        "weight_values_saturated": n_sat_total,
        "scales_not_initializer": sorted(skipped),
        "qdq_in_subgraphs": subgraph_qdq,
        "weights_shared_by_several_dq": sorted(shared_weights),
        "max_abs_log2_scale_shift": float(np.abs(shifts).max()),
        "max_weight_abs_error": wmax_abs,
        "max_weight_rel_error": wmax_rel,
        "exponent_range": [int(all_k.min()), int(all_k.max())],
        "nonzero_zero_points": nonzero_zp,
    }


def check(path):
    """Re-open a written model and assert every Q/DQ scale is exactly 2^k.

    Returns (values_checked, not_a_power_of_two, not_an_initializer, in_subgraphs).
    The last two lists matter as much as the second: a scale this function cannot read,
    and a Q/DQ node it is not allowed to reach, are both scales the rewrite did not
    touch, and one such scale voids the whole experiment. Reporting only `bad` would
    let either pass as a clean run."""
    onnx, numpy_helper = _load_onnx()
    m = onnx.load(path)
    inits = {t.name: t for t in m.graph.initializer}
    bad, unreadable, total = [], [], 0
    for n in m.graph.node:
        if n.op_type in QDQ_OPS and len(n.input) >= 2:
            if n.input[1] not in inits:
                unreadable.append(n.input[1])
                continue
            s = np.asarray(numpy_helper.to_array(inits[n.input[1]]), dtype=np.float64)
            total += s.size
            k = np.rint(np.log2(s))
            if not np.allclose(s, np.exp2(k), rtol=0, atol=0):
                bad.append(n.input[1])
    return total, sorted(set(bad)), sorted(set(unreadable)), _subgraph_qdq(m.graph)


# ---------------------------------------------------------------------- selftest
def selftest():
    print("== pot_rewrite selftest (numpy only) ==")
    s = np.array([1.0, 0.5, 0.75, 1.5, 0.0001, 3.0, 2.0 ** -20], dtype=np.float32)
    # round(log2 .75) = round(-0.415) = 0 -> 1.0 ; round(log2 1.5) = round(.585) = 1 -> 2.0
    assert np.array_equal(pot_exp(s, "nearest"),
                          np.array([0, -1, 0, 1, -13, 2, -20], dtype=np.int64))
    # ceil(log2 .75) = 0 ; ceil(log2 1.5) = 1 ; ceil(log2 1e-4) = -13
    assert np.array_equal(pot_exp(s, "ceil"),
                          np.array([0, -1, 0, 1, -13, 2, -20], dtype=np.int64))
    s2 = np.array([0.3, 0.6, 1.2], dtype=np.float32)     # ceil and nearest differ here
    assert np.array_equal(pot_exp(s2, "nearest"), np.array([-2, -1, 0]))
    assert np.array_equal(pot_exp(s2, "ceil"), np.array([-1, 0, 1]))
    assert np.array_equal(pot_scale(s, mode="nearest"),
                          np.exp2(pot_exp(s, "nearest").astype(np.float64)).astype(np.float32))
    print("  ok  pot_exp / pot_scale, both rounding modes, on 10 known values")

    # nearest never moves a scale by more than sqrt(2); ceil only ever coarsens
    rng = np.random.RandomState(0)
    s = np.exp(rng.uniform(np.log(1e-6), np.log(1e2), 10000)).astype(np.float32)
    sh_n = np.log2(pot_scale(s, mode="nearest").astype(np.float64) / s.astype(np.float64))
    sh_c = np.log2(pot_scale(s, mode="ceil").astype(np.float64) / s.astype(np.float64))
    assert np.abs(sh_n).max() <= 0.5 + 1e-6, np.abs(sh_n).max()
    assert sh_c.min() >= -1e-6 and sh_c.max() <= 1.0 + 1e-6, (sh_c.min(), sh_c.max())
    print("  ok  10k random scales: |nearest| <= 0.5 (%.6f); ceil in [0, 1] (%.6f)"
          % (np.abs(sh_n).max(), sh_c.max()))

    # M = (s_a * s_w) / s_out is exactly a power of two after the rewrite
    for mode in ("ceil", "nearest"):
        a, w, o = (pot_scale(np.float32(v), mode=mode) for v in (0.037, 0.0021, 0.11))
        M = float(a) * float(w) / float(o)
        assert M == 2.0 ** round(np.log2(M)), (mode, M)
    print("  ok  M = (s_a*s_w)/s_out is an exact power of two in both modes")

    # per-channel re-quantization round-trip; ceil must never saturate
    q = rng.randint(-127, 128, size=(8, 4, 3, 3)).astype(np.int8)
    s_old = np.abs(rng.uniform(1e-4, 1e-1, size=8)).astype(np.float32)
    zp = np.zeros((), dtype=np.int8)
    s_new = pot_scale(s_old, mode="ceil")
    q2, a_err, r_err, nsat = requantize(q, zp, s_old, s_new, 0, -128, 127)
    assert q2.shape == q.shape and q2.dtype == q.dtype
    assert nsat == 0, nsat
    bound = float((s_new / 2.0).max()) * (1 + 1e-6)   # half a step of the NEW grid
    assert a_err <= bound, (a_err, bound)
    print("  ok  ceil requantize: 0 saturated, max abs err %.3e <= half-step %.3e"
          % (a_err, bound))

    # nearest can saturate: where s' < s the int range no longer reaches the extremes
    s_new_n = pot_scale(s_old, mode="nearest")
    _, a_n, _, nsat_n = requantize(q, zp, s_old, s_new_n, 0, -128, 127)
    assert nsat_n > 0 and a_n > a_err, (nsat_n, a_n, a_err)
    print("  ok  nearest requantize saturates %d of %d weights (err %.3e > ceil's %.3e)"
          % (nsat_n, q.size, a_n, a_err))

    # a scale-only rewrite (no requantization) is far worse than either
    naive = np.abs((q.astype(np.float64) *
                    (s_new - s_old).reshape(8, 1, 1, 1))).max()
    assert naive > a_err, (naive, a_err)
    print("  ok  requantized err %.3e beats scale-only err %.3e (%.0fx)"
          % (a_err, naive, naive / max(a_err, 1e-30)))

    # clipping saturates rather than wrapping
    q3 = np.full((4,), 127, dtype=np.int8)
    q4, _, _, n4 = requantize(q3, np.zeros((), np.int8), np.float32(1.0),
                              np.float32(0.5), None, -128, 127)
    assert np.all(q4 == 127) and n4 == 4, (q4, n4)
    print("  ok  saturating requantize clips to the int8 range (no wraparound)")
    print("selftest ok")


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", help="input QDQ ONNX")
    ap.add_argument("--out", dest="dst", help="output ONNX (omit with --dry-run)")
    ap.add_argument("--rounding", choices=("ceil", "nearest"), default="ceil",
                    help="ceil (default, never saturates) or nearest (closest grid, "
                         "but saturates the extremes wherever the scale shrinks)")
    ap.add_argument("--report", help="write the report JSON here")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse and report, write nothing")
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the arithmetic with numpy only")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.src:
        ap.error("--in is required (or use --selftest)")
    if not args.dry_run and not args.dst:
        ap.error("--out is required unless --dry-run")

    onnx, _ = _load_onnx()
    model = onnx.load(args.src)
    rep = rewrite(model, mode=args.rounding)
    rep["input"] = os.path.basename(args.src)

    if args.dry_run:
        rep["output"] = None
    else:
        #  Write to a scratch path and move it into place only once check() has
        #  passed. The invariant that buys is: a file AT --out is a file that
        #  verified. Saving straight to --out would leave a model that is not PoT
        #  sitting under a name that says it is, and any next step that does not
        #  read the exit code would happily benchmark it -- the exact silent-wrong
        #  this tool exists to prevent, just moved one step downstream.
        tmp = args.dst + ".partial"
        onnx.checker.check_model(model)
        onnx.save(model, tmp)
        try:
            total, bad, unreadable, in_sub = check(tmp)
        except BaseException:
            _unlink(tmp)
            raise
        rep["output"] = os.path.basename(args.dst)
        rep["checked_scale_values"] = total
        rep["non_pot_after_write"] = bad
        rep["scales_not_initializer_after_write"] = unreadable
        rep["qdq_in_subgraphs_after_write"] = in_sub
        why = None
        if bad:
            why = ("%d scale initializers are not powers of two after the rewrite: %s"
                   % (len(bad), bad[:5]))
        elif unreadable:
            why = ("%d Q/DQ scales are not initializers, so they were not rewritten "
                   "and M is not a power of two: %s" % (len(unreadable), unreadable[:5]))
        elif in_sub:
            why = ("%d Q/DQ node(s) live inside a nested subgraph, which this tool "
                   "does not rewrite (see SCOPE LIMIT), so M is not a power of two "
                   "there: %s" % (len(in_sub), in_sub[:5]))
        if why:
            _unlink(tmp)
            stale = ("\n  WARNING: %s already existed and is left as it was -- it is "
                     "from an EARLIER run, not this one." % args.dst
                     if os.path.exists(args.dst) else "")
            sys.exit("FAILED: %s\n  nothing written to %s%s" % (why, args.dst, stale))
        os.replace(tmp, args.dst)   # atomic where the platform allows it

    print(json.dumps(rep, indent=2, sort_keys=True))
    if rep["scales_not_initializer"]:
        print("\n[warn] %d Q/DQ scale(s) do not come from an initializer and were NOT\n"
              "       rewritten. M is not a power of two for those nodes -- the "
              "experiment\n       is void until they are resolved."
              % len(rep["scales_not_initializer"]))
    if rep["qdq_in_subgraphs"]:
        print("\n[warn] %d Q/DQ node(s) sit inside a nested If/Loop/Scan subgraph. This\n"
              "       tool rewrites the top-level graph only (SCOPE LIMIT), so their\n"
              "       scales are untouched and M is not a power of two there. Writing a\n"
              "       model in this state is refused."
              % len(rep["qdq_in_subgraphs"]))
    if rep["weight_values_saturated"]:
        print("\n[warn] %d weight values saturated at the int8 limits. Re-run with\n"
              "       --rounding ceil to eliminate this (at up to one bit of grid "
              "resolution)." % rep["weight_values_saturated"])
    if rep["nonzero_zero_points"]:
        print("\n[warn] %d zero_point initializers are non-zero. PoT scales remove the\n"
              "       multiplier's divergence but not the zero-point add; expect this\n"
              "       graph to fall short of bit-identity."
              % len(rep["nonzero_zero_points"]))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=True)
        print("\nreport -> %s" % args.report)


if __name__ == "__main__":
    main()
