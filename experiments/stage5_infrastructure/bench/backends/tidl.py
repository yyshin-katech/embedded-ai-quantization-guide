# bench/backends/tidl.py  (TI TDA4VM — edgeai-tidl-tools)
# 학습가이드 §4-4 stub 그대로. 실측 검증(2026-08-17): 이 stub는 설계상 '보드/에뮬 부재 시
# NaN + notes' 라 정정할 것이 없다 — 하드웨어 없이도 CI가 도는 게 목적. 4단계(멀티 SoC)에서
# edgeai-tidl-tools x86 에뮬로 build/run의 NotImplementedError를 채우는 것이 후속 과제.
from .base import Backend, BenchResult


class TIDLBackend(Backend):
    soc_name = "tda4vm"

    def build(self, onnx_path, precision, calib_path):
        raise NotImplementedError("TIDL compile hook — TDA4VM/PC-emulation 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="board/emulation required (TDA4VM, edgeai-tidl-tools)",
        )
