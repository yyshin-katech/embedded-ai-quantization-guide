#!/usr/bin/env python3
"""Concurrent multi-device load harness for Jetson AGX Orin.

Launches N trtexec processes SIMULTANEOUSLY (each pinned to the iGPU or a DLA
core), samples tegrastats across the whole window, then reports per-engine and
aggregate throughput, board power, and parallel-scaling efficiency. This closes
the "iGPU+DLA 동시부하(진짜 병렬 오프로드)는 미측정" caveat from the solo
5-engine sweep (ppw.py): solo runs prove each backend's ceiling one at a time,
this proves whether they add up when run together.

Usage: concurrent.py <label> <duration_s> <spec> [<spec> ...]
  spec = <engine.plan>:<device>   device in {gpu, dla0, dla1}

All processes get identical --duration so their sustained windows overlap; the
per-engine throughput trtexec reports is therefore its rate WHILE the others are
also loaded. Power window = board-total samples above idle_floor*1.20 (same
self-calibrating threshold as ppw.py, robust to DLA not registering on GR3D).
"""
import subprocess, time, re, sys, json, signal, os

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HOME = os.path.expanduser("~")
LABEL = sys.argv[1]
DUR = int(sys.argv[2])
SPECS = sys.argv[3:]

teg_log = f"{HOME}/orin_bench/raw/teg_conc_{LABEL}.log"
teg_f = open(teg_log, "w")
teg = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=teg_f,
                       stderr=subprocess.STDOUT)
time.sleep(1.5)  # let tegrastats spin up; captures a little idle head

# ---- launch every engine at once (concurrency happens HERE) ----
procs = []
for spec in SPECS:
    path, dev = spec.rsplit(":", 1)
    cmd = [TRTEXEC, f"--loadEngine={path}", f"--duration={DUR}",
           "--warmUp=3000", "--iterations=100"]
    if dev == "dla0":
        cmd += ["--useDLACore=0", "--allowGPUFallback"]
    elif dev == "dla1":
        cmd += ["--useDLACore=1", "--allowGPUFallback"]
    elif dev != "gpu":
        raise SystemExit(f"bad device in spec: {spec}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    procs.append({"spec": spec, "dev": dev, "path": path, "proc": p})

t0 = time.time()
for pr in procs:               # all were Popen'd already → they ran concurrently
    pr["out"], _ = pr["proc"].communicate()
    pr["exit"] = pr["proc"].returncode
t1 = time.time()
time.sleep(0.5)
teg.send_signal(signal.SIGINT)
teg.wait()
teg_f.close()

def parse(out):
    thr = gct = None
    for ln in out.splitlines():
        m = re.search(r"Throughput:\s*([\d.]+)\s*qps", ln)
        if m: thr = float(m.group(1))
        m = re.search(r"GPU Compute Time:.*median = ([\d.]+) ms", ln)
        if m: gct = float(m.group(1))
    return thr, gct

per = []
agg_thr = 0.0
for pr in procs:
    thr, gct = parse(pr["out"])
    per.append({"spec": pr["spec"], "dev": pr["dev"],
                "throughput_qps": thr, "gpu_compute_ms": gct,
                "exit": pr["exit"]})
    if thr: agg_thr += thr

# ---- parse tegrastats power window (same rails/logic as ppw.py) ----
rail_re = {
    "gpu_soc": re.compile(r"VDD_GPU_SOC (\d+)mW"),
    "cpu_cv":  re.compile(r"VDD_CPU_CV (\d+)mW"),
    "sys_5v0": re.compile(r"VIN_SYS_5V0 (\d+)mW"),
}
gr3d_re = re.compile(r"GR3D_FREQ (\d+)%")
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

def med(xs):
    xs = sorted(xs); n = len(xs)
    return (xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2) if n else float("nan")

idle_floor = min((s["total"] for s in samples), default=0)
thresh = idle_floor * 1.20
busy = [s for s in samples if s["total"] > thresh] or samples
steady_total = med([s["total"] for s in busy])
peak_total   = max((s["total"] for s in samples), default=float("nan"))
steady_gr3d  = med([s["gr3d"] for s in busy])

out = {
    "label": LABEL, "duration_s": DUR, "n_engines": len(SPECS), "specs": SPECS,
    "wall_s": round(t1 - t0, 3),
    "per_engine": per,
    "aggregate_throughput_qps": round(agg_thr, 3),
    "power_steady_mw": steady_total, "power_peak_mw": peak_total,
    "power_idle_floor_mw": idle_floor, "gr3d_busy_median_pct": steady_gr3d,
    "aggregate_inf_per_s_per_w": round(agg_thr / (steady_total/1000.0), 3)
        if (agg_thr and steady_total) else None,
    "n_samples": len(samples), "n_busy": len(busy),
    "tegrastats_log": teg_log,
}
res_path = f"{HOME}/orin_bench/results/conc_{LABEL}.json"
with open(res_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("saved:", res_path)
for pr in procs:
    if pr["exit"] != 0:
        print(f"=== {pr['spec']} FAILED exit={pr['exit']}, tail ===")
        print("\n".join(pr["out"].splitlines()[-15:]))
