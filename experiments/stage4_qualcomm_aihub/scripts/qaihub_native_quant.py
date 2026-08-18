#!/usr/bin/env python3
"""The RIGHT path for on-device INT8 fidelity: let AI Hub quantize the FP32 ONNX
(HTP-native QDQ) instead of importing a foreign ORT-QDQ. quantize->compile->profile
->inference(200). Contrast vs foreign-QDQ collapse (0.005) to close finding #2."""
import qai_hub as hub
import numpy as np, json, os, traceback
ACC="/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/acc"
OUT="/tmp/claude-1000/-home-yuyeong-embedded-ai-quantization-guide/c0519c3e-a0e0-4508-aecd-4800fb5a5976/scratchpad/qaihub_out"
FP32_ONNX="/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage3/resnet50_fp32.onnx"
DEV="QCS8550 (Proxy)"
RUNTIME="--target_runtime qnn_context_binary"
EXISTING_QUANTIZE_JOB="jp1jz48kp"  # reuse in-flight quantize job (avoid re-quantizing); set None to submit fresh
def log(m): print(m, flush=True)
def summarize(prof):
    es=prof.get("execution_summary",{}); det=prof.get("execution_detail",[]) or []
    cu={}; cyc=0
    for l in det:
        c=l.get("compute_unit","?"); cu[c]=cu.get(c,0)+1; cyc+=l.get("execution_cycles") or 0
    return {"estimated_inference_time_us":es.get("estimated_inference_time"),
            "layer_count":len(det),"layers_by_compute_unit":cu,
            "npu_layer_fraction":round(cu.get("NPU",0)/len(det),4) if det else None,
            "sum_execution_cycles":cyc}
def main():
    xs=np.load(f"{ACC}/inputs.npy"); lab=np.load(f"{ACC}/labels.npy"); ort=np.load(f"{ACC}/ort_pred.npy")
    N=xs.shape[0]; dev=hub.Device(DEV)
    res={"device":DEV,"path":"AI Hub native quantize (INT8 w+a)"}
    # 1) AI Hub native quantize (reuse in-flight job if provided)
    if EXISTING_QUANTIZE_JOB:
        qj=hub.get_job(EXISTING_QUANTIZE_JOB); log(f"[quantize REUSE] {qj.url}")
    else:
        calib={"input":[xs[i] for i in range(100)]}
        qj=hub.submit_quantize_job(model=FP32_ONNX, calibration_data=calib,
                                   weights_dtype=hub.QuantizeDtype.INT8,
                                   activations_dtype=hub.QuantizeDtype.INT8,
                                   name="rn50_aihub_int8_quant")
        log(f"[quantize] {qj.url}")
    qst=qj.wait()
    res["quantize"]={"job_id":qj.job_id,"url":qj.url,"success":bool(getattr(qst,'success',False))}
    qmodel=qj.get_target_model()
    # 2) compile
    cj=hub.submit_compile_job(model=qmodel, device=dev, name="rn50_aihubq_qnn",
                              input_specs={"input":(1,3,224,224)}, options=RUNTIME)
    log(f"[compile] {cj.url}"); cst=cj.wait()
    res["compile"]={"job_id":cj.job_id,"url":cj.url,"success":bool(getattr(cst,'success',False))}
    cmodel=cj.get_target_model()
    # 3) profile
    pj=hub.submit_profile_job(model=cmodel, device=dev, name="rn50_aihubq_prof")
    log(f"[profile] {pj.url}"); pj.wait(); prof=pj.download_profile()
    json.dump(prof, open(f"{OUT}/profile_aihubq_raw.json","w"), indent=2, default=str)
    res["profile"]={"job_id":pj.job_id,"url":pj.url,"summary":summarize(prof)}
    log(f"[profile summary] {json.dumps(res['profile']['summary'],ensure_ascii=False)}")
    # 4) inference 200
    ij=hub.submit_inference_job(model=cmodel, device=dev, name="rn50_aihubq_acc200",
                                inputs={"input":[xs[i] for i in range(N)]})
    log(f"[inference] {ij.url}"); ij.wait()
    out=ij.download_output_data(); k=list(out.keys())[0]
    pred=np.array([int(np.asarray(a).reshape(-1).argmax()) for a in out[k]])
    np.save(f"{ACC}/aihubq_pred200.npy", pred)
    res["accuracy"]={"n":int(N),"inference_job":ij.url,
        "int8_htp_top1_vs_gt":round(float((pred==lab).mean()),4),
        "ort_top1_vs_gt":round(float((ort==lab).mean()),4),
        "int8_vs_ort_agreement":round(float((pred==ort).mean()),4),
        "int8_distinct":int(len(np.unique(pred)))}
    json.dump(res, open(f"{OUT}/summary_aihubq.json","w"), indent=2, ensure_ascii=False, default=str)
    log("=== AI-HUB-NATIVE INT8 DONE ==="); log(json.dumps(res, ensure_ascii=False, indent=2, default=str))
if __name__=="__main__":
    try: main()
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}"); traceback.print_exc(); raise
