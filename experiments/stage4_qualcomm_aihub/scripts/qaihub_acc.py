#!/usr/bin/env python3
"""On-device numerical correctness: run the compiled INT8 ResNet50 on QCS8550 NPU
for the same 200 inputs, compare NPU top-1 / prediction-agreement vs ORT-CPU & GT."""
import qai_hub as hub
import numpy as np, json, os, traceback

ACC = "/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/acc"
COMPILE_JOB = "jglxmv22g"   # int8 clean compile on QCS8550
DEV = "QCS8550 (Proxy)"

def log(m): print(m, flush=True)

def main():
    xs = np.load(f"{ACC}/inputs.npy")          # (N,1,3,224,224) f32
    lab = np.load(f"{ACC}/labels.npy")
    ort_pred = np.load(f"{ACC}/ort_pred.npy")
    N = xs.shape[0]
    model = hub.get_job(COMPILE_JOB).get_target_model()
    log(f"got compiled model from {COMPILE_JOB}: {model}")
    inputs = {"input": [xs[i] for i in range(N)]}
    ij = hub.submit_inference_job(model=model, device=hub.Device(DEV),
                                  name="rn50_int8_acc200", inputs=inputs)
    log(f"[inference submit] id={ij.job_id} url={ij.url}")
    st = ij.wait()
    log(f"[inference done] success={bool(getattr(st,'success',False))}")
    out = ij.download_output_data()
    key = list(out.keys())[0]
    arrs = out[key]                            # list of (1,1000) arrays
    npu_pred = np.array([int(np.asarray(a).reshape(-1).argmax()) for a in arrs])
    np.save(f"{ACC}/npu_pred.npy", npu_pred)
    res = {
        "device": DEV, "n": int(N), "output_key": key,
        "npu_top1_vs_gt": round(float((npu_pred == lab).mean()), 4),
        "ort_top1_vs_gt": round(float((ort_pred == lab).mean()), 4),
        "npu_vs_ort_agreement": round(float((npu_pred == ort_pred).mean()), 4),
        "npu_ort_disagree_count": int((npu_pred != ort_pred).sum()),
        "inference_job": ij.url,
    }
    with open(f"{ACC}/acc_summary.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log("=== ACC RESULT ===")
    log(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}"); traceback.print_exc(); raise
