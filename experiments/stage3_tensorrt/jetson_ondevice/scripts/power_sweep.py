#!/usr/bin/env python3
"""nvpmodel power-mode sweep for Jetson AGX Orin.

RUN AS ROOT (nvpmodel -m requires sudo):
    sudo python3 ~/orin_bench/power_sweep.py

For each nvpmodel power mode {MAXN, 50W, 30W, 15W} measures the two INT8
champions -- iGPU INT8 (explicit QDQ) and DLA0 INT8 (implicit --int8) -- with
the same sustained-trtexec + tegrastats method as ppw.py: GPU-compute median
latency, throughput, board-total steady power, perf/watt. The solo 5-engine
sweep was MAXN-only; this maps the perf/watt curve across power budgets, which
is what an automotive/embedded deployment actually runs under. Restores MAXN at
the end and chowns outputs back to katech.
"""
import subprocess, time, re, json, os, signal

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HOME = "/home/katech"                 # script runs as root; ~ would be /root
ENG  = f"{HOME}/orin_bench/engines"
DUR  = 20

MODES = [(0, "MAXN"), (3, "50W"), (2, "30W"), (1, "15W")]
ENGINES = [
    ("gpu_int8", f"{ENG}/rn50_gpu_int8.plan", ""),   # iGPU, explicit QDQ
    ("dla_int8", f"{ENG}/rn50_dla_int8.plan", "0"),  # DLA0, implicit --int8
]

rail_re = {
    "gpu_soc": re.compile(r"VDD_GPU_SOC (\d+)mW"),
    "cpu_cv":  re.compile(r"VDD_CPU_CV (\d+)mW"),
    "sys_5v0": re.compile(r"VIN_SYS_5V0 (\d+)mW"),
}
gr3d_re = re.compile(r"GR3D_FREQ (\d+)%")

def med(xs):
    xs = sorted(xs); n = len(xs)
    return (xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2) if n else float("nan")

def measure(plan, dla, label):
    teg_log = f"{HOME}/orin_bench/raw/teg_pm_{label}.log"
    teg_f = open(teg_log, "w")
    teg = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=teg_f,
                           stderr=subprocess.STDOUT)
    time.sleep(1.5)
    cmd = [TRTEXEC, f"--loadEngine={plan}", f"--duration={DUR}",
           "--warmUp=2000", "--iterations=100"]
    if dla != "":
        cmd += [f"--useDLACore={dla}", "--allowGPUFallback"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    time.sleep(0.5)
    teg.send_signal(signal.SIGINT); teg.wait(); teg_f.close()

    thr = gct = None
    for ln in r.stdout.splitlines():
        m = re.search(r"Throughput:\s*([\d.]+)\s*qps", ln)
        if m: thr = float(m.group(1))
        m = re.search(r"GPU Compute Time:.*median = ([\d.]+) ms", ln)
        if m: gct = float(m.group(1))

    samples = []
    with open(teg_log) as f:
        for ln in f:
            row = {}; ok = True
            for k, rr in rail_re.items():
                mm = rr.search(ln)
                if not mm: ok = False; break
                row[k] = int(mm.group(1))
            if not ok: continue
            row["total"] = row["gpu_soc"] + row["cpu_cv"] + row["sys_5v0"]
            g = gr3d_re.search(ln)
            row["gr3d"] = int(g.group(1)) if g else -1
            samples.append(row)
    idle_floor = min((s["total"] for s in samples), default=0)
    busy = [s for s in samples if s["total"] > idle_floor*1.20] or samples
    steady = med([s["total"] for s in busy])
    peak = max((s["total"] for s in samples), default=float("nan"))
    return {
        "throughput_qps": thr, "gpu_compute_ms": gct, "exit": r.returncode,
        "power_steady_mw": steady, "power_peak_mw": peak,
        "power_idle_floor_mw": idle_floor,
        "gr3d_busy_median_pct": med([s["gr3d"] for s in busy]),
        "inf_per_s_per_w": round(thr/(steady/1000.0), 3) if (thr and steady) else None,
        "n_samples": len(samples), "n_busy": len(busy),
    }

results = []
for mid, mname in MODES:
    subprocess.run(["nvpmodel", "-m", str(mid)])
    time.sleep(6)  # let DVFS caps settle
    q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
    active = "MAXN" if "MAXN" in q else (re.search(r"NV Power Mode:\s*(\S+)", q) or [None, "?"])[1]
    print(f"\n### mode {mname} (id {mid}) active_query={active!r}")
    for ename, plan, dla in ENGINES:
        r = measure(plan, dla, f"{mname}_{ename}")
        r.update({"mode_id": mid, "mode": mname, "engine": ename})
        results.append(r)
        print(f"  {ename}: {r['gpu_compute_ms']} ms  {r['throughput_qps']} qps  "
              f"{round(r['power_steady_mw']/1000,2)} W  {r['inf_per_s_per_w']} inf/s/W  exit={r['exit']}")

subprocess.run(["nvpmodel", "-m", "0"])  # restore MAXN
print("\n### restored MAXN")
out_path = f"{HOME}/orin_bench/results/power_sweep.json"
json.dump({"model": "torchvision ResNet50", "batch": 1, "duration_s": DUR,
           "note": "iGPU INT8=explicit QDQ (accuracy-valid); DLA INT8=implicit --int8 (latency/power-valid). No jetson_clocks (DVFS active), matches MAXN baseline method.",
           "results": results}, open(out_path, "w"), indent=2)
subprocess.run(["chown", "-R", "katech:katech",
                f"{HOME}/orin_bench/results", f"{HOME}/orin_bench/raw"])
print("saved:", out_path)
