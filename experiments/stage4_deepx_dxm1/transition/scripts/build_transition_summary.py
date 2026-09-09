#!/usr/bin/env python3
"""Reduce the DX-M1 transition sweep raw (raw/corescale.csv + raw/tr_c*/analyzed_n{1,2,3}.json
+ dxnn/manifest.json) into results/transition_summary.json -- the SSOT for the report,
study-guide callout, and paper.

Design recap: a FIXED heavy conv trunk (body_depth=18, ~2.7 ms Inference) provides constant
NPU compute; a 4-ch bottleneck + 1x1 expander head sweeps the output (D2H) size at ~constant
compute.  Sweeping cout therefore walks the model from compute-bound (Inference > D2H,
core-scaling ~=3x, jobs even) to D2H-bound (D2H > Inference, scaling ~=1x, one core hogs the
shared PCIe link) -- tracing the transition curve the crossover axis (274bffa) proved exists
at the extremes (resnet50 vs yolo26n) but did not trace.

Stage convention (matches crossover axis): D2H p50 is intrinsic (nearly identical at 1c/3c),
Inference p50 read at 1 core (uncontended -- the 3-core Inference inflates in the transition
zone from output-handoff backpressure, a throughput artifact, not extra compute).
"""
import json, csv, os, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
RES = os.path.join(HERE, "results")
MANIFEST = os.path.join(HERE, "dxnn", "manifest.json")
os.makedirs(RES, exist_ok=True)

COUTS = [5, 20, 41, 82, 163, 327, 490, 653, 980, 1276, 2551, 5102]


def load_analyzed(cout, ncores):
    return json.load(open(os.path.join(RAW, f"tr_c{cout}", f"analyzed_n{ncores}.json")))


def stage_p50(d, name):
    s = d["stages"].get(name)
    return s["p50_ms"] if s else None


def main():
    man = {r["cout"]: r for r in json.load(open(MANIFEST))}
    cs = {int(r["cout"]): r for r in csv.DictReader(open(os.path.join(RAW, "corescale.csv")))}

    rows = []
    for c in COUTS:
        d1 = load_analyzed(c, 1)
        d3 = load_analyzed(c, 3)
        inf1 = d1["inference_all_cores"]["p50_ms"]      # clean fixed compute (1 core)
        inf3 = d3["inference_all_cores"]["p50_ms"]      # 3-core Inference (inflates in the
        #                                                 transition zone: output-handoff
        #                                                 backpressure, a throughput artifact)
        d2h1 = stage_p50(d1, "D2H")
        h2d1 = stage_p50(d1, "H2D")
        d2h3 = stage_p50(d3, "D2H")
        # 3-core per-core job distribution (starvation signature)
        ipc = d3["inference_per_core"]
        jobs = [ipc[k]["n"] for k in sorted(ipc.keys())]
        r = cs[c]
        s31 = float(r["scale_3c_1c"])
        # single-inference regime = which stage limits ONE inference
        single = "compute-bound" if inf1 >= d2h1 else "D2H-bound"
        # throughput-scaling regime from 3c/1c
        if s31 >= 2.4:
            scaling = "near-linear"
        elif s31 <= 1.05:
            scaling = "flat"
        else:
            scaling = "sub-linear"
        rows.append({
            "name": f"tr_c{c}", "cout": c,
            "out_bytes": man[c]["nominal_out_bytes"],
            "out_shape": man[c]["out_shape"],
            "body_depth": man[c].get("body_depth"),
            "trunk_gmacs": man[c].get("trunk_gmacs"),
            "groups": man[c].get("groups"),
            "inference_p50_1c_ms": round(inf1, 3),
            "inference_p50_3c_ms": round(inf3, 3),
            "h2d_p50_1c_ms": round(h2d1, 3),
            "d2h_p50_1c_ms": round(d2h1, 3),
            "d2h_p50_3c_ms": round(d2h3, 3),
            "fps_1c": float(r["fps_1c"]), "fps_2c": float(r["fps_2c"]), "fps_3c": float(r["fps_3c"]),
            "scale_3c_1c": s31, "scale_2c_1c": float(r["scale_2c_1c"]),
            "jobs_3c": jobs, "jobs_3c_pct": [round(j / sum(jobs) * 100) for j in jobs],
            "single_inference_regime": single,
            "throughput_scaling_regime": scaling,
        })

    # fixed-compute check: spread of 1-core Inference across the sweep
    infs = [x["inference_p50_1c_ms"] for x in rows]
    h2ds = [x["h2d_p50_1c_ms"] for x in rows]
    obytes = [x["out_bytes"] for x in rows]
    sweep_x = round(max(obytes) / min(obytes), 1)      # output-size sweep span (~1020x)

    # linear fit D2H(bytes) over the linear region (cout<=1276) for the intrinsic crossover
    lin = [(x["out_bytes"], x["d2h_p50_1c_ms"]) for x in rows if x["cout"] <= 1276]
    n = len(lin); sx = sum(b for b, _ in lin); sy = sum(y for _, y in lin)
    sxx = sum(b * b for b, _ in lin); sxy = sum(b * y for b, y in lin)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)      # ms per byte
    intercept = (sy - slope * sx) / n
    inf_mean = round(st.mean(infs), 3)
    # intrinsic crossover: D2H(bytes) == Inference  -> single inference becomes D2H-bound
    xover_bytes = (inf_mean - intercept) / slope
    # link-saturation onset: D2H == Inference/ncores -> 3 cores can no longer all be fed
    NCORES = 3
    onset_bytes = (inf_mean / NCORES - intercept) / slope

    summary = {
        "axis": "DX-M1 regime transition curve (compute-bound <-> D2H-bound)",
        "closes": "crossover axis (274bffa) caveat #3: proved 2 regimes exist at the extremes "
                  "(resnet50 compute-bound / yolo26n D2H-bound) but did not trace the transition",
        "environment": {
            "host": "Raspberry Pi 5 (Cortex-A76 x4, Debian 13)",
            "npu": "DEEPX DX-M1 (3 cores @1000MHz), PCIe Gen2 x1",
            "runtime": "DXRT v3.4.2 (dxbenchmark), FW v2.7.4",
            "method": "fixed heavy conv trunk (body_depth=18, ~2.7ms Inference, all NPU / 0 CPU "
                      "group) + 4-ch bottleneck + 1x1 expander head sweeps output(D2H) size at "
                      "constant compute; dxbenchmark synthesizes input (latency/regime only, "
                      "random weights -> no accuracy claim)",
            "core_map": "-n 1 = 1 core (NPU_0) | -n 4 = 2 cores (NPU_0/1) | -n 0 = 3 cores (NPU_ALL)",
        },
        "fixed_compute_check": {
            "inference_p50_1c_ms_min": min(infs), "inference_p50_1c_ms_max": max(infs),
            "inference_p50_1c_ms_mean": inf_mean,
            "spread_pct": round((max(infs) - min(infs)) / st.mean(infs) * 100, 1),
            "h2d_p50_1c_ms_min": min(h2ds), "h2d_p50_1c_ms_max": max(h2ds),
            "out_bytes_min": min(obytes), "out_bytes_max": max(obytes),
            "output_sweep_x": sweep_x,
            "note": "Inference and H2D ~constant across the %gx output-size sweep -> compute and "
                    "input transfer are the FIXED variables; only D2H (output) is swept "
                    "(D2H itself spans ~126x in time, less than output size because a fixed "
                    "~0.14ms transfer-overhead floor dominates at small output)." % sweep_x,
        },
        "transition": {
            "compute_bound_plateau": "cout<=163 (out<=125KB): scaling ~2.98x near-linear, jobs 33/33/33",
            "transition_band": "out ~250KB-1MB: scaling 1.99x->1.21x, core2 progressively starved",
            "d2h_bound": "out>=~2MB: scaling ~1.00x flat, one core hogs link (93/6/0 -> 98/2/0, "
                         "matches crossover-axis yolo26n 472/28/2)",
            "intrinsic_crossover_out_bytes": round(xover_bytes),
            "intrinsic_crossover_note": "D2H p50 == fixed Inference (%.2fms) -> a SINGLE inference "
                                        "becomes D2H-bound (~%.2f MB output)" % (inf_mean, xover_bytes / 1048576),
            "link_saturation_onset_out_bytes": round(onset_bytes),
            "link_saturation_note": "D2H == Inference/%d -> %d cores can no longer all be fed by the "
                                    "one shared PCIe link, 3c/1c scaling starts breaking (~%.0f KB)"
                                    % (NCORES, NCORES, onset_bytes / 1024),
            "band_width_x": round(xover_bytes / onset_bytes, 2),
            "band_width_note": "transition band spans ~%.1fx in output size ~= N_cores (%d): N cores "
                               "share one D2H link, so multi-core throughput saturates at 1/N the "
                               "per-core D2H that bounds a single inference" % (xover_bytes / onset_bytes, NCORES),
            "d2h_linear_fit": {"ms_per_MB": round(slope * 1048576, 3), "intercept_ms": round(intercept, 3),
                               "region": "cout<=1276 (out<=1MB); D2H turns super-linear above ~2MB"},
        },
        "headline": {
            "scaling_span": "3c/1c core-scaling traces 2.98x (compute-bound) -> 1.00x (D2H-bound) "
                            "monotonically across a %gx output-size sweep at FIXED compute" % sweep_x,
            "vs_crossover_axis": "refines 274bffa: the regime boundary is not a knife-edge but a "
                                 "~3x-wide band (= N_cores), because output size sets D2H and N cores "
                                 "share one PCIe link",
        },
        "rows": rows,
    }
    out = os.path.join(RES, "transition_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"[summary] {out}")
    print(f"  fixed compute: Inference 1c {min(infs)}-{max(infs)}ms (mean {inf_mean}, spread "
          f"{summary['fixed_compute_check']['spread_pct']}%) over {sweep_x}x output-size sweep")
    print(f"  intrinsic crossover (D2H=Inference): {round(xover_bytes)} B ({xover_bytes/1048576:.2f} MB)")
    print(f"  link-saturation onset (D2H=Inference/3): {round(onset_bytes)} B ({onset_bytes/1024:.0f} KB)")
    print(f"  transition band width: {summary['transition']['band_width_x']}x (~N_cores=3)")
    print(f"\n  {'cout':>5} {'outKB':>6} {'Inf1c':>6} {'D2H1c':>7} {'3c/1c':>6} {'jobs%':>10} {'single':>13} {'scaling':>11}")
    for x in rows:
        print(f"  {x['cout']:>5} {x['out_bytes']//1024:>6} {x['inference_p50_1c_ms']:>6} "
              f"{x['d2h_p50_1c_ms']:>7} {x['scale_3c_1c']:>6} {'/'.join(map(str,x['jobs_3c_pct'])):>10} "
              f"{x['single_inference_regime']:>13} {x['throughput_scaling_regime']:>11}")


if __name__ == "__main__":
    main()
