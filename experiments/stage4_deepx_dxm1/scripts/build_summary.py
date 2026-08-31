#!/usr/bin/env python3
"""Build the consolidated CPU-vs-NPU comparison SSOT and the YOLOV5S summary
from the raw per-run result files. Run from results/ or pass its path.

    python3 build_summary.py [results_dir]

Writes results/cpu_npu_comparison.json and results/y5s_summary.json.
"""
import json, os, re, sys, statistics as st

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + "/../results"
R = os.path.abspath(R)
RAW = os.path.abspath(R + "/../raw")
def L(f): return json.load(open(os.path.join(R, f)))


def build_comparison():
    npu = L("dxrun_summary.json")
    seq, thr = npu["latency_mode_seq_1core"], npu["throughput_mode_async"]
    cpu = {t: L(f"cpu_yolo26n_t{t}.json") for t in (1, 2, 4)}
    pw = {k: L(f"power_{k}.json") for k in ("idle", "cpu_load", "npu_load")}

    npu_fps = thr["3core"]["fps"]          # async, all cores
    npu_lat = seq["e2e_latency_ms"]        # seq single-stream E2E
    npu_comp = seq["npu_processing_ms"]    # pure NPU core compute
    cpu_lat = cpu[4]["lat_p50_ms"]         # 4-thread (all A76 cores)
    cpu_fps = cpu[4]["fps_single_stream"]
    P_idle, P_cpu, P_npu = (pw[k]["watt_mean"] for k in ("idle", "cpu_load", "npu_load"))

    def rd(x, n=4): return round(x, n)
    card = {}
    for w in (3.0, 5.0):
        card[f"card_{int(w)}W"] = {
            "npu_total_power_w": rd(P_npu + w, 3),
            "npu_energy_per_inf_j": rd((P_npu + w) / npu_fps),
            "npu_perf_per_watt": rd(npu_fps / (P_npu + w), 2),
            "cpu_over_npu_energy_ratio": rd((P_cpu / cpu_fps) / ((P_npu + w) / npu_fps), 2),
            "npu_over_cpu_ppw_ratio": rd((npu_fps / (P_npu + w)) / (cpu_fps / P_cpu), 2),
        }
    out = {
        "model": "yolo26n (COCO, 640x640) — CPU FP32 ONNX vs NPU INT8 .dxnn",
        "board": "Raspberry Pi 5 (Cortex-A76 x4) + DEEPX DX-M1 (M.2, PCIe Gen2 x1)",
        "note": "Same physical board: A76 is both the stage4 CPU-fallback proxy and the DX-M1 host. "
                "CPU=FP32 (native best), NPU=INT8 (accelerator is INT8-only) — honest deployment choice.",
        "npu": {"engine": "INT8 .dxnn", "seq_latency_e2e_ms": npu_lat,
                "npu_compute_ms": npu_comp, "async_throughput_fps": npu_fps},
        "cpu": {"engine": "FP32 ONNX ORT CPUEP 4-thread", "latency_p50_ms": cpu_lat,
                "throughput_fps": cpu_fps,
                "thread_scaling": {t: {"lat_ms": cpu[t]["lat_p50_ms"],
                                       "fps": cpu[t]["fps_single_stream"]} for t in (1, 2, 4)}},
        "offload_gain": {
            "latency_e2e_x": rd(cpu_lat / npu_lat, 2),
            "latency_vs_npu_compute_x": rd(cpu_lat / npu_comp, 2),
            "throughput_x": rd(npu_fps / cpu_fps, 2)},
        "power_host_board_w": {"idle": P_idle, "cpu_load": P_cpu, "npu_load": P_npu,
                               "cpu_over_npu_x": rd(P_cpu / P_npu, 2),
                               "note": "Pi5 internal PMIC rails only; DX-M1 card draws from EXT5V "
                                       "UPSTREAM of these rails => NOT captured (metering gap)."},
        "energy_and_ppw": {
            "cpu_energy_per_inf_j": rd(P_cpu / cpu_fps),
            "cpu_perf_per_watt": rd(cpu_fps / P_cpu, 2),
            "npu_energy_per_inf_j_hostside": rd(P_npu / npu_fps),
            "npu_perf_per_watt_hostside": rd(npu_fps / P_npu, 2),
            "npu_over_cpu_ppw_hostside_x": rd((npu_fps / P_npu) / (cpu_fps / P_cpu), 2),
            "total_system_with_card_tdp": card,
            "card_tdp_source": "DEEPX DX-M1 spec: 2W min / ~3W typ / 5W max, 25 TOPS INT8 "
                               "(deepx.ai; Radxa AICore DX-M1M '25 TOPS for 3W')."}}
    json.dump(out, open(os.path.join(R, "cpu_npu_comparison.json"), "w"), indent=2)
    return out


def parse_dxrun(path):
    """Pull NPU-proc/latency/FPS averages from a dxrun -s/-b stdout capture."""
    t = open(path).read()
    def g(label):
        m = re.search(re.escape(label) + r"[^:]*:\s*([\d.]+)", t)
        return float(m.group(1)) if m else None
    return {"npu_processing_ms": g("NPU Processing Time"),
            "latency_ms": g("Latency"),
            "fps": g("FPS")}


def build_y5s():
    d = os.path.join(RAW, "dxrun")
    out = {"model": "YOLOV5S_1.dxnn (INT8, prebuilt) — heavier second model, confirms host-bound",
           "seq_1core_latency": parse_dxrun(os.path.join(d, "y5s_lat_seq.txt")),
           "async_1core": parse_dxrun(os.path.join(d, "y5s_thr_1core.txt")),
           "async_3core": parse_dxrun(os.path.join(d, "y5s_thr_3core.txt"))}
    a1, a3 = out["async_1core"]["fps"], out["async_3core"]["fps"]
    out["core_scaling_3v1"] = round(a3 / a1, 3) if a1 else None
    out["note"] = ("Even lighter NPU compute (~2.6ms) than yolo26n (8.86ms) yet LOWER throughput "
                   "(~41 vs 91 fps) and flat/declining cores => host/IO-bound (big raw output tensor).")
    json.dump(out, open(os.path.join(R, "y5s_summary.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    c = build_comparison()
    y = build_y5s()
    print(json.dumps({"comparison": c, "y5s": y}, indent=2))
