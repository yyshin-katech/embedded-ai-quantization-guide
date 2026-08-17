# bench/backends/qnn.py  (Qualcomm QCS8550 — ONNX Runtime QNN EP)
# 학습가이드 §4-4 stub 그대로. 실측 검증(2026-08-17): QNN 양자화 유틸은 x86에서 돌지만
# HTP(NPU) 실행엔 Snapdragon 디바이스가 필요 → 이 머신(RTX dGPU)에선 device required.
# 설계상 stub(NaN+notes)이라 정정 없음.
from .base import Backend, BenchResult


class QNNBackend(Backend):
    soc_name = "qcs8550"

    def build(self, onnx_path, precision, calib_path):
        raise NotImplementedError("QNN context binary 생성 — QNN SDK/디바이스 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="device required (QCS8550, ONNX Runtime QNN EP)",
        )
