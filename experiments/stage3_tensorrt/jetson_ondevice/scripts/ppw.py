#!/usr/bin/env python3
"""Perf-per-watt harness for Jetson AGX Orin.
Runs a prebuilt TensorRT engine under sustained load via trtexec while sampling
tegrastats, then computes steady/peak board power and inferences-per-second-per-watt.

Usage: ppw.py <engine.plan> <label> [duration_s] [dla_core]
  dla_core: "0"/"1" to load on a DLA core (engine must have been BUILT for DLA),
            omit/"" for GPU.
Power rails parsed (AGX Orin): VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 = board total.
"""
import subprocess, time, re, sys, json, signal, os

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HOME = os.path.expanduser("~")
PLAN = sys.argv[1]
LABEL = sys.argv[2]
DUR = int(sys.argv[3]) if len(sys.argv) > 3 else 30
DLA = sys.argv[4] if len(sys.argv) > 4 else ""

teg_log = f"{HOME}/orin_bench/raw/teg_{LABEL}.log"
teg_f = open(teg_log, "w")
teg = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=teg_f,
                       stderr=subprocess.STDOUT)
time.sleep(1.5)  # let tegrastats spin up; captures a little idle head

cmd = [TRTEXEC, f"--loadEngine={PLAN}", f"--duration={DUR}",
       "--warmUp=2000", "--iterations=100"]
if DLA != "":
    cmd += [f"--useDLACore={DLA}", "--allowGPUFallback"]

t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
t1 = time.time()
time.sleep(0.5)
teg.send_signal(signal.SIGINT)
teg.wait()
teg_f.close()

# ---- parse trtexec ----
thr = gct = e2e = None
for ln in r.stdout.splitlines():
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", ln)
    if m: thr = float(m.group(1))
    m = re.search(r"GPU Compute Time:.*median = ([\d.]+) ms", ln)
    if m: gct = float(m.group(1))
    m = re.search(r"^\[.*\]\s*\[I\]\s*Latency:.*median = ([\d.]+) ms", ln)
    if m: e2e = float(m.group(1))

# ---- parse tegrastats power window ----
rail_re = {
    "gpu_soc": re.compile(r"VDD_GPU_SOC (\d+)mW"),
    "cpu_cv":  re.compile(r"VDD_CPU_CV (\d+)mW"),
    "sys_5v0": re.compile(r"VIN_SYS_5V0 (\d+)mW"),
}
gr3d_re = re.compile(r"GR3D_FREQ (\d+)%")
samples = []
with open(teg_log) as f:
    for ln in f:
        row = {}
        ok = True
        for k, rr in rail_re.items():
            mm = rr.search(ln)
            if not mm: ok = False; break
            row[k] = int(mm.group(1))
        if not ok: continue
        row["total"] = row["gpu_soc"] + row["cpu_cv"] + row["sys_5v0"]
        g = gr3d_re.search(ln)
        row["gr3d"] = int(g.group(1)) if g else -1
        samples.append(row)

# steady window = samples under load, by POWER threshold (robust for DLA, whose
# compute doesn't register on GR3D/GPU-3D counter). idle_floor self-calibrates.
def med(xs):
    xs = sorted(xs); n = len(xs)
    return (xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2) if n else float("nan")

idle_floor = min((s["total"] for s in samples), default=0)
thresh = idle_floor * 1.20
busy = [s for s in samples if s["total"] > thresh] or samples

steady_total = med([s["total"] for s in busy])
peak_total   = max((s["total"] for s in samples), default=float("nan"))
steady_gpu   = med([s["gpu_soc"] for s in busy])
steady_cpu   = med([s["cpu_cv"] for s in busy])
steady_5v0   = med([s["sys_5v0"] for s in busy])
steady_gr3d  = med([s["gr3d"] for s in busy])  # ~99% on GPU engines, ~0% on DLA

out = {
    "label": LABEL, "plan": PLAN, "dla_core": DLA, "duration_s": DUR,
    "wall_s": round(t1 - t0, 3), "exit": r.returncode,
    "throughput_qps": thr, "gpu_compute_ms": gct, "e2e_latency_ms": e2e,
    "power_steady_mw": steady_total, "power_peak_mw": peak_total,
    "power_steady_gpu_soc_mw": steady_gpu, "power_steady_cpu_cv_mw": steady_cpu,
    "power_steady_sys_5v0_mw": steady_5v0, "power_idle_floor_mw": idle_floor,
    "gr3d_busy_median_pct": steady_gr3d,
    "n_samples": len(samples), "n_busy": len(busy),
    "inf_per_s_per_w": (thr / (steady_total/1000.0)) if (thr and steady_total) else None,
    "tegrastats_log": teg_log,
}
res_path = f"{HOME}/orin_bench/results/ppw_{LABEL}.json"
with open(res_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("saved:", res_path)
# tail of trtexec stderr if it failed
if r.returncode != 0:
    print("=== trtexec FAILED, tail ===")
    print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
