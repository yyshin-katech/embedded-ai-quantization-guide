#!/usr/bin/env python3
"""
DETR-on-Orin trtexec build+bench harness (runs ON the Jetson AGX Orin board).

New-model axis over the ResNet50 latency ladder (same board, same trtexec flags,
so latency is 1:1 comparable to the ResNet50 numbers in results/summary.json).
Model = facebook/detr-resnet-50 (transformer detector: CNN backbone + encoder/
decoder with LayerNorm/Softmax/MatMul/Gather/Where...).

Configs (batch1, GPU Compute median, device event-timed; H2D/D2H excluded):
  gpu_fp32           detr_sim.onnx                                             (weight-valid latency)
  gpu_fp16           detr_sim.onnx  --fp16                                     (weight-valid latency)
  gpu_int8_explicit  detr_int8.onnx --int8 --fp16                              (explicit QDQ; stage3 case B/C probe)
  gpu_int8_implicit  detr_sim.onnx  --int8 --fp16                              (auto-range; LATENCY-ONLY)
  dla_fp16           detr_sim.onnx  --fp16 --useDLACore=0 --allowGPUFallback   (+ placement)
  dla_int8_implicit  detr_sim.onnx  --int8 --fp16 --useDLACore=0 --allowGPUFallback (+ placement; LATENCY-ONLY)

Notes / caveats (kept identical to the ResNet50 run for comparability):
  * Latency = trtexec "GPU Compute Time" median (device event-timed, batch1, MAXN).
  * "explicit" INT8 = ORT-exported QDQ (detr_int8.onnx: zp!=0 in 1085/1485 Q/DQ +
    INT32 bias DequantizeLinear on all 149 Conv/Gemm) -> direct trtexec parse is
    expected to hit stage3 case C (zero-point!=0) / case B (INT32 bias DQ). If it
    fails, the error tail is captured as a confirming finding and we fall back to
    the implicit number for the INT8 latency rung.
  * "implicit" INT8 uses trtexec auto dynamic-range (no calibration file) -> the
    number is LATENCY-ONLY; accuracy is NOT claimed (same caveat as DLA INT8).
  * Accuracy is NOT measured on-board (COCO val2017 not present). It is cited from
    the stage2 RTX 3080 run: FP32 mAP 0.4207 -> INT8 0.2402 (-42.9%).
"""
import subprocess, json, os, re

HOME  = os.path.expanduser("~")
BENCH = os.path.join(HOME, "orin_bench")
TX    = "/usr/src/tensorrt/bin/trtexec"
ONNX  = os.path.join(BENCH, "onnx")
ENG   = os.path.join(BENCH, "engines")
RAW   = os.path.join(BENCH, "detr_raw")
RES   = os.path.join(BENCH, "detr_results")
for d in (ENG, RAW, RES):
    os.makedirs(d, exist_ok=True)

FP32  = os.path.join(ONNX, "detr_sim.onnx")
INT8Q = os.path.join(ONNX, "detr_int8.onnx")
# Same timing flags as the ResNet50 run -> 1:1 latency comparability.
FLAGS = ["--warmUp=2000", "--duration=10", "--iterations=200", "--avgRuns=100"]

CONFIGS = [
    # tag,               onnx,  extra flags,                                                    dla?
    ("gpu_fp32",          FP32,  [],                                                             False),
    ("gpu_fp16",          FP32,  ["--fp16"],                                                     False),
    ("gpu_int8_explicit", INT8Q, ["--int8", "--fp16"],                                           False),
    ("gpu_int8_implicit", FP32,  ["--int8", "--fp16"],                                           False),
    ("dla_fp16",          FP32,  ["--fp16", "--useDLACore=0", "--allowGPUFallback"],             True),
    ("dla_int8_implicit", FP32,  ["--int8", "--fp16", "--useDLACore=0", "--allowGPUFallback"],   True),
]


def run(cmd, logpath):
    with open(logpath, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def parse_perf(logpath):
    gpu_median = throughput = None
    with open(logpath, errors="ignore") as f:
        for line in f:
            m = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", line)
            if m:
                gpu_median = float(m.group(1))
            m = re.search(r"Throughput: ([\d.]+) qps", line)
            if m:
                throughput = float(m.group(1))
    return gpu_median, throughput


def parse_errors(logpath, limit=15):
    errs = []
    with open(logpath, errors="ignore") as f:
        for line in f:
            if re.search(r"\[E\]|ERROR|error:|[Ff]ailed|shiftIsAllZeros|could not|[Uu]nsupported|Assertion", line):
                errs.append(line.rstrip())
    return errs[-limit:]


def parse_placement(vlog):
    dla = gpu = foreign = 0
    with open(vlog, errors="ignore") as f:
        for line in f:
            if "[DlaLayer]" in line:
                dla += 1
            elif "[GpuLayer]" in line:
                gpu += 1
            if "successfully offloaded to DLA" in line:
                foreign += 1
    return {"dla_layers": dla, "gpu_layers": gpu, "foreign_nodes_on_dla": foreign}


def main():
    results = {}
    for tag, onnx, extra, is_dla in CONFIGS:
        plan = os.path.join(ENG, "detr_%s.plan" % tag)
        log  = os.path.join(RAW, "detr_%s.log" % tag)
        cmd  = [TX, "--onnx=%s" % onnx] + extra + ["--saveEngine=%s" % plan] + FLAGS
        print("[build+time] %-18s %s" % (tag, " ".join(extra) or "(fp32)"), flush=True)
        rc = run(cmd, log)
        gpu_median, thr = parse_perf(log)
        entry = {
            "tag": tag, "onnx": os.path.basename(onnx), "flags": extra,
            "returncode": rc, "built": rc == 0 and os.path.exists(plan),
            "gpu_compute_median_ms": gpu_median, "throughput_qps": thr,
            "engine_bytes": os.path.getsize(plan) if os.path.exists(plan) else None,
        }
        if rc != 0 or gpu_median is None:
            entry["errors_tail"] = parse_errors(log)
        if is_dla:
            vlog = os.path.join(RAW, "detr_%s_verbose.log" % tag)
            vcmd = [TX, "--onnx=%s" % onnx] + extra + ["--skipInference", "--verbose"]
            vrc  = run(vcmd, vlog)
            entry["placement"] = parse_placement(vlog)
            entry["placement"]["verbose_rc"] = vrc
        results[tag] = entry
        print("   -> built=%-5s gpu_median=%s ms  thr=%s qps  eng=%s B%s" % (
            entry["built"], gpu_median, thr, entry["engine_bytes"],
            ("  placement=%s" % entry.get("placement")) if is_dla else ""), flush=True)

    out = os.path.join(RES, "detr_summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nWROTE", out, flush=True)


if __name__ == "__main__":
    main()
