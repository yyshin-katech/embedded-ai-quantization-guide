#!/usr/bin/env python3
"""FP32(->fp16) on-device accuracy over the full 200 imgs on QCS8550 HTP (NCHW).
Real vendor-NPU top-1 datapoint (trustworthy path: no foreign QDQ import)."""
import qai_hub as hub
import numpy as np, json
ACC="/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/acc"
FP32_COMPILE="jpyx31r05"
DEV="QCS8550 (Proxy)"
def log(m): print(m, flush=True)
xs=np.load(f"{ACC}/inputs.npy"); lab=np.load(f"{ACC}/labels.npy"); ort=np.load(f"{ACC}/ort_pred.npy")
N=xs.shape[0]
model=hub.get_job(FP32_COMPILE).get_target_model()
ij=hub.submit_inference_job(model=model, device=hub.Device(DEV), name="rn50_fp32_acc200",
                            inputs={"input":[xs[i] for i in range(N)]})
log(f"[fp32 acc200] {ij.url}"); ij.wait()
out=ij.download_output_data(); k=list(out.keys())[0]
pred=np.array([int(np.asarray(a).reshape(-1).argmax()) for a in out[k]])
np.save(f"{ACC}/fp32_pred200.npy", pred)
res={"n":int(N),"device":DEV,
 "fp32_htp_top1_vs_gt":round(float((pred==lab).mean()),4),
 "ort_top1_vs_gt":round(float((ort==lab).mean()),4),
 "fp32_vs_ort_agreement":round(float((pred==ort).mean()),4),
 "fp32_distinct":int(len(np.unique(pred))),
 "inference_job":ij.url}
json.dump(res, open(f"{ACC}/fp32_acc200.json","w"), indent=2)
log("=== FP32 acc200 ==="); log(json.dumps(res,indent=2))
