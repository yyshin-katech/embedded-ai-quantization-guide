#!/usr/bin/env python3
"""Discriminate the collapse cause: run the SAME 20 NCHW imgs on QCS8550 HTP with
FP32(->fp16) vs INT8 compiled models. If FP32 recovers ~0.75 and INT8 stays ~0,
the fault is HTP's QDQ interpretation, not the input path."""
import qai_hub as hub
import numpy as np, json, traceback
ACC="/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/acc"
FP32_COMPILE="jpyx31r05"   # fp32 compile on QCS8550
INT8_COMPILE="jglxmv22g"   # int8 clean compile on QCS8550
DEV="QCS8550 (Proxy)"
def log(m): print(m, flush=True)
xs=np.load(f"{ACC}/inputs.npy")[:20]        # (20,1,3,224,224) NCHW f32
lab=np.load(f"{ACC}/labels.npy")[:20]
ort=np.load(f"{ACC}/ort_pred.npy")[:20]
dev=hub.Device(DEV)
def run(compile_job, name):
    model=hub.get_job(compile_job).get_target_model()
    ij=hub.submit_inference_job(model=model, device=dev, name=name,
                                inputs={"input":[xs[i] for i in range(20)]})
    log(f"[{name}] {ij.url}"); ij.wait()
    out=ij.download_output_data(); k=list(out.keys())[0]
    pred=np.array([int(np.asarray(a).reshape(-1).argmax()) for a in out[k]])
    return pred, ij.url
fp32_pred,fp32_url=run(FP32_COMPILE,"acc20_fp32_nchw")
int8_pred,int8_url=run(INT8_COMPILE,"acc20_int8_nchw")
res={
 "n":20,
 "ort_top1":round(float((ort==lab).mean()),3),
 "fp32_htp_top1":round(float((fp32_pred==lab).mean()),3),
 "fp32_vs_ort":round(float((fp32_pred==ort).mean()),3),
 "int8_htp_top1":round(float((int8_pred==lab).mean()),3),
 "int8_vs_ort":round(float((int8_pred==ort).mean()),3),
 "fp32_distinct":int(len(np.unique(fp32_pred))),
 "int8_distinct":int(len(np.unique(int8_pred))),
 "fp32_url":fp32_url,"int8_url":int8_url,
}
json.dump(res, open(f"{ACC}/fp32_vs_int8_acc20.json","w"), indent=2)
log("=== FP32 vs INT8 on HTP (20 imgs, NCHW) ==="); log(json.dumps(res,indent=2))
