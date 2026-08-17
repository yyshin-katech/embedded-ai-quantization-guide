# bench/backends/base.py
# 학습가이드 §4-2 "공통 인터페이스"의 검증본(2026-08-17, AI-LAP/RTX3080).
# 이 파일은 순수 파이썬(추상 클래스 + dataclass + 타이머)이라 문서 코드 그대로 실행된다 —
# GPU/SDK 의존이 없어 정정할 것이 없었다. trt.py(§4-3)만 실측에서 정정됨(pycuda→polygraphy).
from __future__ import annotations
import abc
import json
import pathlib
import time
from dataclasses import dataclass, asdict


@dataclass
class BenchResult:
    """모든 백엔드가 반환해야 하는 공통 결과 스키마.

    report/ 와 tests/ 는 이 필드 이름에만 의존한다.
    필드를 바꾸면 report·회귀테스트·baseline CSV가 전부 영향을 받으므로,
    스키마 변경은 decision_log에 남긴다.

    주의(§4-3 검증): `accuracy`는 mAP(검출)든 top-1(분류)든 0~1 실수면 된다.
    이 검증 인스턴스는 ResNet50/ImageNet top-1을 넣는다 — BEVFormer INT8은
    2단계에서 '유효 export 경로 없음(포크 필요)'으로 범위 밖이라, 실제로 RTX에서
    빌드·측정 가능한 분류 모델로 하네스를 관통시켰다. 스키마는 동일하다.
    """
    model: str            # 예: "resnet50"
    soc: str              # 예: "rtx", "orin", "tda4vm", "qcs8550", "rzv2h"
    precision: str        # "fp32" | "fp16" | "int8"
    latency_ms: float     # 대표값(중앙값 권장)
    peak_mem_mb: float    # 추론 중 peak memory
    accuracy: float       # mAP 또는 top-1 등 (0~1)
    engine_build_s: float # 빌드 소요(참고용)
    latency_p95_ms: float = float("nan")  # 안전 마진 판단용(선택)
    trt_version: str = ""                 # 재현성: 어떤 TRT로 뽑았나
    notes: str = ""                       # 실패/특이사항

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(abc.ABC):
    """모든 SoC 백엔드의 공통 인터페이스.

    새 SoC 추가 = 이 클래스를 상속한 파일 하나 추가.
    report/ 와 CI는 이 계약에만 의존한다.

    계약(4):
      build(onnx, precision, calib) -> None   # ONNX를 엔진으로
      run(inputs) -> np.ndarray                # 1회 추론
      measure(...) -> BenchResult              # latency/mem/acc 측정
      collect(result) -> pathlib.Path          # 결과를 results/*.json으로 봉인
    """
    soc_name: str = "unknown"

    def __init__(self):
        self._build_s: float = 0.0   # build()가 채운다

    @abc.abstractmethod
    def build(self, onnx_path: str, precision: str, calib_path: str | None) -> None:
        """ONNX → 백엔드 엔진으로 컴파일/빌드. 빌드 소요를 self._build_s에 저장."""
        ...

    @abc.abstractmethod
    def run(self, inputs) -> "np.ndarray":
        """1회 추론. 출력 텐서 반환."""
        ...

    @abc.abstractmethod
    def measure(self, model: str, precision: str,
                loader, evaluator, warmup: int = 20, iters: int = 200) -> BenchResult:
        """latency / peak_mem / accuracy 를 측정해 BenchResult로 반환."""
        ...

    # --- collect: 결과를 디스크에 봉인 (모든 백엔드 공통, 오버라이드 불필요) ---
    def collect(self, result: BenchResult, out_dir: str = "results") -> pathlib.Path:
        """BenchResult를 results/{model}__{soc}__{precision}.json 으로 저장.

        파일명 규약이 report/·tests/의 파싱 기준이므로 여기서 단일화한다.
        """
        out = pathlib.Path(out_dir)
        out.mkdir(exist_ok=True)
        fn = out / f"{result.model}__{result.soc}__{result.precision}.json"
        fn.write_text(json.dumps(result.to_dict(), indent=2))
        return fn

    # --- 공통 유틸 (모든 백엔드가 재사용) ---
    @staticmethod
    def _timeit(fn, warmup: int, iters: int) -> tuple[float, float]:
        """(median_ms, p95_ms) 반환. GPU면 각 백엔드가 fn 안에서 동기화를 넣는다.

        검증 노트(§4-3): 문서 원안의 TRT 백엔드는 fn 안에서 pycuda
        `Context.synchronize()`로 동기화했다. 이 검증본은 polygraphy `TrtRunner.infer`가
        내부에서 execute+stream sync를 끝내고 반환하므로 fn=self.run 만으로 충분하다.
        _timeit 자체는 백엔드 무관이라 문서 그대로 둔다.
        """
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        p95 = samples[min(int(len(samples) * 0.95), len(samples) - 1)]
        return median, p95
