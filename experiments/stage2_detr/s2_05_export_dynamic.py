#!/usr/bin/env python3
# s2_05_export_dynamic.py — 2단계 COCO mAP 실측용 동적 shape export.
# 고정 [1,3,800,1066] export는 COCO의 다양한 종횡비를 못 받는다 →
# legacy(dynamo=False) opset17 + dynamic_axes(H,W)로 단일파일 재export.
import torch, os, numpy as np
from transformers import DetrForObjectDetection
import onnxruntime as ort

os.makedirs("_workspace/stage2", exist_ok=True)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
dummy = torch.randn(1, 3, 800, 1066)

out = "_workspace/stage2/detr_dyn.onnx"
torch.onnx.export(
    model, (dummy,), out,
    input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
    opset_version=17, do_constant_folding=True, dynamo=False,
    dynamic_axes={"pixel_values": {0: "b", 2: "h", 3: "w"},
                  "logits": {0: "b"}, "pred_boxes": {0: "b"}})
print("detr_dyn.onnx =", round(os.path.getsize(out) / 1e6, 1), "MB")

# 동적 shape 검증: 서로 다른 크기 3종이 다 통과하는지
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
for hw in [(800, 1066), (800, 1201), (750, 800)]:
    r = s.run(None, {"pixel_values": np.random.randn(1, 3, *hw).astype(np.float32)})
    print(hw, "-> logits", r[0].shape, "boxes", r[1].shape)
print("DYN_EXPORT_OK")
