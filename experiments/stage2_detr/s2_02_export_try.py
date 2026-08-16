#!/usr/bin/env python3
# s2_02_export_try.py — 2단계 §4.2 검증: DETR ONNX export 시도/실패 채집.
# 문서 §4.2가 grid_sample 중심 카탈로그인데 DETR엔 grid_sample이 없다 →
# DETR에서 '실제로' 무엇이 어느 경로에서 깨지는지 원문 로그로 채집한다.
import inspect, os, torch
from transformers import DetrForObjectDetection

os.makedirs("_workspace/stage2", exist_ok=True)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
dummy = torch.randn(1, 3, 800, 1066)

sig = inspect.signature(torch.onnx.export)
print("torch", torch.__version__,
      "| torch.onnx.export dynamo 기본값 =", sig.parameters.get("dynamo").default)

def attempt(tag, **kw):
    shown = {k: v for k, v in kw.items()}
    print(f"\n### ATTEMPT: {tag}  kw={shown}", flush=True)
    try:
        torch.onnx.export(model, (dummy,), f"_workspace/stage2/{tag}.onnx",
                          input_names=["pixel_values"],
                          output_names=["logits", "pred_boxes"], **kw)
        sz = os.path.getsize(f"_workspace/stage2/{tag}.onnx") / 1e6
        print(f">>> RESULT[{tag}]: OK  ({sz:.1f} MB)")
    except Exception as e:
        print(f">>> RESULT[{tag}]: FAIL :: {type(e).__name__}: {str(e)[:700]}")

attempt("detr_legacy_op11", opset_version=11, do_constant_folding=True, dynamo=False)  # 문서 §4.2(A)
attempt("detr_legacy_op17", opset_version=17, do_constant_folding=True, dynamo=False)  # legacy 상향
attempt("detr_dynamo_op17", opset_version=17, dynamo=True)                            # 문서 §4.2(B)
attempt("detr_default_op11", opset_version=11)                                        # 문서 case A는 dynamo 무지정
print("\nEXPORT_TRY_DONE")
