#!/usr/bin/env python3
"""Isolate the collapse cause: run 20 imgs on QCS8550 INT8 NPU with NCHW vs NHWC input layout."""
import qai_hub as hub
import numpy as np, json
ACC="/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/acc"
def log(m): print(m, flush=True)
xs=np.load(f"{ACC}/inputs.npy")[:20]        # (20,1,3,224,224)
lab=np.load(f"{ACC}/labels.npy")[:20]
ort=np.load(f"{ACC}/ort_pred.npy")[:20]
model=hub.get_job("jglxmv22g").get_target_model()   # int8 compiled
dev=hub.Device("QCS8550 (Proxy)")

def run(name, arrs):
    ij=hub.submit_inference_job(model=model, device=dev, name=name, inputs={"input":arrs})
    log(f"[{name}] {ij.url}"); ij.wait()
    out=ij.download_output_data(); k=list(out.keys())[0]
    pred=np.array([int(np.asarray(a).reshape(-1).argmax()) for a in out[k]])
    return pred

# NCHW (as-is)
p_nchw=run("layout_nchw",[xs[i] for i in range(20)])
# NHWC (transpose channel to last): (1,3,224,224)->(1,224,224,3)
xnhwc=np.transpose(xs,(0,1,3,4,2))
p_nhwc=run("layout_nhwc",[xnhwc[i] for i in range(20)])

res={
 "ort_top1":round(float((ort==lab).mean()),3),
 "nchw_top1":round(float((p_nchw==lab).mean()),3),
 "nchw_vs_ort":round(float((p_nchw==ort).mean()),3),
 "nhwc_top1":round(float((p_nhwc==lab).mean()),3),
 "nhwc_vs_ort":round(float((p_nhwc==ort).mean()),3),
}
json.dump(res, open(f"{ACC}/layout_test.json","w"), indent=2)
log("=== LAYOUT TEST ==="); log(json.dumps(res,indent=2))
