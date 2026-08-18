#!/usr/bin/env python3
"""Qualcomm AI Hub: compile ResNet50 (INT8 QDQ + FP32) to QCS8550 Hexagon NPU,
profile on real device, dump raw profiles + a compute-unit/offload summary.
Token is read from ~/.qai_hub/client.ini (outside repo) — never hardcoded here."""
import qai_hub as hub
import json, os, sys, traceback

OUT = "/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/qaihub_out"
os.makedirs(OUT, exist_ok=True)
DEV_NAME = "QCS8550 (Proxy)"
ASSETS = {
    "int8": "/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_int8_qdq.onnx",
    "fp32": "/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_fp32.onnx",
}
INPUT_SPECS = {"input": (1, 3, 224, 224)}
RUNTIME = "--target_runtime qnn_context_binary"

def log(m):
    print(m, flush=True)

def summarize_profile(prof):
    """Pull latency + compute-unit breakdown from an AI Hub profile dict."""
    out = {}
    es = prof.get("execution_summary", {})
    out["estimated_inference_time_us"] = es.get("estimated_inference_time")
    out["estimated_peak_memory_range_bytes"] = es.get("estimated_inference_peak_memory_range")
    out["first_load_time_us"] = es.get("first_load_time")
    out["compile_memory_range"] = es.get("compile_memory_range")
    # per-layer compute unit tally
    detail = prof.get("execution_detail", []) or []
    tally = {}
    time_by_cu = {}
    for layer in detail:
        cu = layer.get("compute_unit", "?")
        tally[cu] = tally.get(cu, 0) + 1
        t = layer.get("execution_time", 0) or 0
        time_by_cu[cu] = time_by_cu.get(cu, 0) + t
    out["layer_count"] = len(detail)
    out["layers_by_compute_unit"] = tally
    out["time_us_by_compute_unit"] = time_by_cu
    if detail:
        npu = tally.get("NPU", 0)
        out["npu_layer_fraction"] = round(npu / len(detail), 4)
        tt = sum(time_by_cu.values()) or 1
        out["npu_time_fraction"] = round(time_by_cu.get("NPU", 0) / tt, 4)
    return out

def main():
    device = hub.Device(DEV_NAME)
    log(f"=== device: {DEV_NAME} ===")

    # 1) submit compile jobs (both precisions, parallel server-side)
    cjobs = {}
    for prec, path in ASSETS.items():
        cj = hub.submit_compile_job(
            model=path, device=device, name=f"rn50_{prec}_qnn",
            input_specs=INPUT_SPECS, options=RUNTIME,
        )
        cjobs[prec] = cj
        log(f"[compile submit] {prec}: id={cj.job_id} url={cj.url}")

    # 2) wait compiles
    compiled = {}
    comp_status = {}
    for prec, cj in cjobs.items():
        st = cj.wait()
        ok = bool(getattr(st, "success", False))
        comp_status[prec] = {"success": ok, "code": str(getattr(st, "code", "")), "job_id": cj.job_id, "url": cj.url}
        log(f"[compile done] {prec}: success={ok} code={getattr(st,'code','')}")
        if ok:
            compiled[prec] = cj.get_target_model()

    # 3) profile each compiled model on the real device
    profiles = {}
    for prec, model in compiled.items():
        try:
            pj = hub.submit_profile_job(model=model, device=device, name=f"rn50_{prec}_profile")
            log(f"[profile submit] {prec}: id={pj.job_id} url={pj.url}")
            st = pj.wait()
            ok = bool(getattr(st, "success", False))
            log(f"[profile done] {prec}: success={ok}")
            prof = pj.download_profile()
            with open(f"{OUT}/profile_{prec}_raw.json", "w") as f:
                json.dump(prof, f, indent=2, default=str)
            profiles[prec] = {"job_id": pj.job_id, "url": pj.url, "success": ok,
                              "summary": summarize_profile(prof)}
            log(f"[profile summary] {prec}: {json.dumps(profiles[prec]['summary'], ensure_ascii=False)}")
        except Exception as e:
            log(f"[profile ERROR] {prec}: {type(e).__name__}: {e}")
            traceback.print_exc()

    result = {"device": DEV_NAME, "runtime": RUNTIME,
              "compile": comp_status, "profile": profiles}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    log("=== DONE ===")
    log(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
