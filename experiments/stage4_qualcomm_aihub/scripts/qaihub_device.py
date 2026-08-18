#!/usr/bin/env python3
"""Compile+profile ResNet50 (fp32 + AI-Hub-clean int8) on a given AI Hub device.
Usage: qaihub_device.py "<device name>" <slug>"""
import qai_hub as hub
import json, os, sys, traceback

OUT = "/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/qaihub_out"
os.makedirs(OUT, exist_ok=True)
DEV_NAME = sys.argv[1]
SLUG = sys.argv[2]
ASSETS = {
    "fp32": "/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_fp32.onnx",
    "int8": "/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_int8_qdq_aihub.onnx",
}
INPUT_SPECS = {"input": (1, 3, 224, 224)}
RUNTIME = "--target_runtime qnn_context_binary"

def log(m): print(m, flush=True)

def summarize(prof):
    es = prof.get("execution_summary", {})
    detail = prof.get("execution_detail", []) or []
    cu = {}
    cyc = 0
    for l in detail:
        c = l.get("compute_unit", "?"); cu[c] = cu.get(c, 0) + 1
        cyc += l.get("execution_cycles") or 0
    return {
        "estimated_inference_time_us": es.get("estimated_inference_time"),
        "first_load_time_us": es.get("first_load_time"),
        "layer_count": len(detail),
        "layers_by_compute_unit": cu,
        "npu_layer_fraction": round(cu.get("NPU", 0) / len(detail), 4) if detail else None,
        "sum_execution_cycles": cyc,
    }

def main():
    device = hub.Device(DEV_NAME)
    log(f"=== {DEV_NAME} ({SLUG}) ===")
    res = {"device": DEV_NAME, "runtime": RUNTIME, "compile": {}, "profile": {}}
    cjobs = {}
    for prec, path in ASSETS.items():
        cj = hub.submit_compile_job(model=path, device=device, name=f"rn50_{prec}_{SLUG}",
                                    input_specs=INPUT_SPECS, options=RUNTIME)
        cjobs[prec] = cj
        log(f"[compile submit] {prec}: id={cj.job_id} url={cj.url}")
    compiled = {}
    for prec, cj in cjobs.items():
        st = cj.wait(); ok = bool(getattr(st, "success", False))
        res["compile"][prec] = {"success": ok, "code": str(getattr(st,'code','')), "job_id": cj.job_id, "url": cj.url}
        log(f"[compile done] {prec}: success={ok} code={getattr(st,'code','')}")
        if ok: compiled[prec] = cj.get_target_model()
    for prec, model in compiled.items():
        try:
            pj = hub.submit_profile_job(model=model, device=device, name=f"rn50_{prec}_{SLUG}_prof")
            log(f"[profile submit] {prec}: id={pj.job_id} url={pj.url}")
            st = pj.wait(); ok = bool(getattr(st, "success", False))
            prof = pj.download_profile()
            with open(f"{OUT}/profile_{SLUG}_{prec}_raw.json", "w") as f:
                json.dump(prof, f, indent=2, default=str)
            res["profile"][prec] = {"job_id": pj.job_id, "url": pj.url, "success": ok, "summary": summarize(prof)}
            log(f"[profile summary] {prec}: {json.dumps(res['profile'][prec]['summary'], ensure_ascii=False)}")
        except Exception as e:
            log(f"[profile ERROR] {prec}: {type(e).__name__}: {e}"); traceback.print_exc()
    with open(f"{OUT}/summary_{SLUG}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    log(f"=== {SLUG} DONE ===")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}"); traceback.print_exc(); raise
