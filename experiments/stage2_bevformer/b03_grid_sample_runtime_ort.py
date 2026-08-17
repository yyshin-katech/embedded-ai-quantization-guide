#!/usr/bin/env python3
"""
b03 — grid_sample 런타임(ORT 1.23.2) 거동: 4D vs 5D, CUDA vs CPU.

§4.6.1 표 단정 검증:
  - 4D GridSample: CUDA EP 에서 실제 실행되는가
  - 5D GridSample(opset20): CUDA EP 커널이 있는가 / 없으면 CPU로 "조용히" 폴백하는가
  - 5D dynamo(opset17, out-of-spec) 모델은 애초에 로드되는가 (onnx.checker + ORT load)

CUDA-only providers 로 강제 → 커널 없으면 에러(= CUDA 미지원 증명).
[CUDA,CPU] → 성공하되 폴백 경고(= 조용한 CPU 폴백 증명).
"""
import os, warnings, numpy as np
import onnx, onnxruntime as ort
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
print(f"ORT {ort.__version__}  providers={ort.get_available_providers()}\n")


def export_4d_control():
    class M(nn.Module):
        def forward(self, f, g):
            return F.grid_sample(f, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    p = os.path.join(OUT, "_gs4d_op17.onnx")
    torch.onnx.export(M(), (torch.randn(1, 8, 16, 16), torch.rand(1, 10, 10, 2) * 2 - 1),
                      p, opset_version=17, dynamo=False)
    return p


def checker(path):
    try:
        onnx.checker.check_model(onnx.load(path))
        return "checker=PASS"
    except Exception as e:
        return f"checker=FAIL({type(e).__name__}: {str(e)[:120]})"


def run(path, providers, feeds):
    """세션 생성 + 1회 추론. (성공여부, 메시지)."""
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 1  # INFO: 노드 배치/폴백 경고 노출
        sess = ort.InferenceSession(path, so, providers=providers)
        got = [p for p in sess.get_providers()]
        out = sess.run(None, feeds)
        return True, f"run OK  active_providers={got}  out_shape={out[0].shape}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:220]}"


def main():
    # ---- 4D control ----
    p4 = export_4d_control()
    f4 = {"feat" if False else i.name: v for i, v in zip(
        onnx.load(p4).graph.input,
        [np.random.randn(1, 8, 16, 16).astype(np.float32),
         (np.random.rand(1, 10, 10, 2).astype(np.float32) * 2 - 1)])}
    print("### 4D GridSample (opset17) — 양성 대조군")
    print("  ", checker(p4))
    print("   CUDA-only :", run(p4, ["CUDAExecutionProvider"], f4)[1])
    print("   CPU-only  :", run(p4, ["CPUExecutionProvider"], f4)[1])

    # ---- 5D legacy opset20 (표준 유효) ----
    p5 = os.path.join(OUT, "_gs5d_legacy_op20.onnx")
    if os.path.exists(p5):
        g = onnx.load(p5).graph
        f5 = {g.input[0].name: np.random.randn(1, 4, 6, 8, 8).astype(np.float32),
              g.input[1].name: (np.random.rand(1, 5, 8, 8, 3).astype(np.float32) * 2 - 1)}
        print("\n### 5D GridSample (legacy opset20, 표준 유효)")
        print("  ", checker(p5))
        ok_c, msg_c = run(p5, ["CUDAExecutionProvider"], f5)
        print("   CUDA-only :", msg_c, "" if ok_c else "  <-- CUDA 커널 부재 증명")
        print("   CUDA+CPU  :", run(p5, ["CUDAExecutionProvider", "CPUExecutionProvider"], f5)[1])
        print("   CPU-only  :", run(p5, ["CPUExecutionProvider"], f5)[1])
    else:
        print("\n(5D legacy opset20 파일 없음 — b02 먼저 실행)")

    # ---- 5D dynamo opset17 (out-of-spec) ----
    pd = os.path.join(OUT, "_gs5d_dynamo_op17.onnx")
    print("\n### 5D GridSample (dynamo opset17, out-of-spec 여부)")
    if os.path.exists(pd):
        print("  ", checker(pd))
        gd = onnx.load(pd).graph
        fd = {gd.input[0].name: np.random.randn(1, 4, 6, 8, 8).astype(np.float32),
              gd.input[1].name: (np.random.rand(1, 5, 8, 8, 3).astype(np.float32) * 2 - 1)}
        print("   CPU-only  :", run(pd, ["CPUExecutionProvider"], fd)[1])
    else:
        print("   (b02 는 dynamo op17 5D를 보존하지 않음 — 필요시 재생성)")


if __name__ == "__main__":
    main()
