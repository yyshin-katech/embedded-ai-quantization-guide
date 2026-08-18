#!/usr/bin/env python3
"""Compile+profile the AI-Hub-cleaned INT8 QDQ ResNet50 on QCS8550 Hexagon NPU."""
import qai_hub as hub
import json, os, traceback

OUT = "/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/qaihub_out"
os.makedirs(OUT, exist_ok=True)
DEV_NAME = "QCS8550 (Proxy)"
MODEL = "/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_int8_qdq_aihub.onnx"
INPUT_SPECS = {"input": (1, 3, 224, 224)}
RUNTIME = "--target_runtime qnn_context_binary"

def log(m): print(m, flush=True)

def summarize_profile(prof):
    out = {}
    es = prof.get("execution_summary", {})
    out["estimated_inference_time_us"] = es.get("estimated_inference_time")
    out["estimated_peak_memory_range"] = es.get("estimated_inference_peak_memory_range")
    out["first_load_time_us"] = es.get("first_load_time")
    detail = prof.get("execution_detail", []) or []
    tally, time_by_cu = {}, {}
    for layer in detail:
        cu = layer.get("compute_unit", "?")
        tally[cu] = tally.get(cu, 0) + 1
        t = layer.get("execution_time", 0) or 0
        time_by_cu[cu] = time_by_cu.get(cu, 0) + t
    out["layer_count"] = len(detail)
    out["layers_by_compute_unit"] = tally
    out["time_us_by_compute_unit"] = time_by_cu
    if detail:
        out["npu_layer_fraction"] = round(tally.get("NPU", 0) / len(detail), 4)
        tt = sum(time_by_cu.values()) or 1
        out["npu_time_fraction"] = round(time_by_cu.get("NPU", 0) / tt, 4)
    return out

def main():
    device = hub.Device(DEV_NAME)
    log(f"=== INT8(clean) -> {DEV_NAME} ===")
    cj = hub.submit_compile_job(model=MODEL, device=device, name="rn50_int8clean_qnn",
                                input_specs=INPUT_SPECS, options=RUNTIME)
    log(f"[compile submit] int8: id={cj.job_id} url={cj.url}")
    st = cj.wait()
    ok = bool(getattr(st, "success", False))
    log(f"[compile done] int8: success={ok} code={getattr(st,'code','')}")
    res = {"device": DEV_NAME, "runtime": RUNTIME, "model": MODEL,
           "compile": {"success": ok, "code": str(getattr(st,'code','')), "job_id": cj.job_id, "url": cj.url}}
    if ok:
        model = cj.get_target_model()
        pj = hub.submit_profile_job(model=model, device=device, name="rn50_int8clean_profile")
        log(f"[profile submit] int8: id={pj.job_id} url={pj.url}")
        pst = pj.wait()
        pok = bool(getattr(pst, "success", False))
        log(f"[profile done] int8: success={pok}")
        prof = pj.download_profile()
        with open(f"{OUT}/profile_int8_raw.json", "w") as f:
            json.dump(prof, f, indent=2, default=str)
        res["profile"] = {"job_id": pj.job_id, "url": pj.url, "success": pok,
                          "summary": summarize_profile(prof)}
        log(f"[profile summary] int8: {json.dumps(res['profile']['summary'], ensure_ascii=False)}")
    with open(f"{OUT}/summary_int8.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    log("=== INT8 DONE ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}"); traceback.print_exc(); raise
