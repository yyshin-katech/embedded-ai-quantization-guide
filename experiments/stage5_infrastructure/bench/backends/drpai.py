# bench/backends/drpai.py  (Renesas RZ/V2H — rzv_drp-ai_tvm : feed-forward only)
# 학습가이드 §4-4 stub 그대로. 실측 검증(2026-08-17): DRP-AI TVM 컴파일엔 보드+툴체인 필요.
# 2단계 BEVFormer에서 확인했듯 grid_sample/제어흐름은 여기서 실패시켜 CI가 잡게 하는 설계.
# 설계상 stub(NaN+notes)이라 정정 없음.
from .base import Backend, BenchResult


class DRPAIBackend(Backend):
    soc_name = "rzv2h"

    def build(self, onnx_path, precision, calib_path):
        raise NotImplementedError("DRP-AI TVM compile hook — 보드 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="board required (RZ/V2H, feed-forward only)",
        )
