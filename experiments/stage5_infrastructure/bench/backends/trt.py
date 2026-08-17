# bench/backends/trt.py
# 학습가이드 §4-3 "구체 구현 TRTBackend"의 실측 검증·정정본
# (2026-08-17, AI-LAP / RTX 3080, TensorRT 10.16.1.11, polygraphy 0.50.3).
#
# 문서 원안 대비 정정 3건(실측 근거는 ../harness_constraints.md):
#   (1) pycuda 부재 → polygraphy TrtRunner로 대체.
#       원안은 `import pycuda.driver as cuda; import pycuda.autoinit`을 쓰지만
#       정본 venv(~/emb-ai)에 pycuda가 없다(3단계에서 trtexec 부재를 확인한 것과 같은 결).
#       polygraphy(설치돼 있음)의 TrtRunner가 디바이스 메모리 할당·H2D/D2H·stream sync를
#       내부에서 처리하므로 run()이 짧아지고, 3단계 지연 측정과 같은 경로가 된다.
#       (pycuda를 직접 깔면 원안도 동작하지만, 정본 스택엔 없다.)
#   (2) INT8 캘리브레이터 배선. 원안은 `config.int8_calibrator = MyCalibrator(...)`를
#       주석으로 '지면상 생략' → 그대로면 INT8 엔진이 스케일 없이 빌드돼 무의미하다.
#       여기서 IInt8EntropyCalibrator2(3단계 t04와 동일)를 실제로 붙인다.
#   (3) peak_mem: 원안의 engine.device_memory_size는 TRT 10.x에서 deprecated →
#       device_memory_size_v2를 우선 사용(어느 쪽이 먹혔는지 notes에 남긴다).
import time

import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import (
    network_from_onnx_path, engine_from_network, CreateConfig, Calibrator, TrtRunner,
)

from .base import Backend, BenchResult


class TRTBackend(Backend):
    soc_name = "rtx"  # 데스크톱 dGPU. Orin이면 "orin"으로.

    def __init__(self, in_name="input", calib_feed=None, calib_cache=None,
                 workspace_gb=6):
        """calib_feed: () -> iterable of {in_name: np.ndarray}  (INT8 캘리브용, 없으면 INT8 스킵)
        문서 §4-5의 loader/evaluator처럼 데이터 의존부는 주입한다(하네스는 데이터 무지)."""
        super().__init__()
        self.engine = None
        self._runner = None
        self._in_name = in_name
        self._calib_feed = calib_feed
        self._calib_cache = calib_cache
        self._workspace_gb = workspace_gb
        self._mem_attr = ""      # 어떤 device_memory API가 먹혔는지 기록

    def build(self, onnx_path: str, precision: str, calib_path: str | None) -> None:
        t0 = time.perf_counter()
        flags = {}
        if precision == "fp16":
            flags["fp16"] = True
        elif precision == "int8":
            # implicit 캘리브레이션: fp16 병용으로 INT8 커널 없는 층은 FP16 폴백(문서 §4-3/3단계 t04).
            flags["int8"] = True
            flags["fp16"] = True
            if self._calib_feed is not None:
                flags["calibrator"] = Calibrator(
                    data_loader=self._calib_feed(),
                    cache=self._calib_cache,
                    BaseClass=trt.IInt8EntropyCalibrator2,   # 문서 실습3의 그 API(10.16서 생존)
                )
        config = CreateConfig(
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: self._workspace_gb << 30},
            **flags,
        )
        # TensorRT 10.x 표준 빌드 경로(build_serialized_network를 polygraphy가 감쌈).
        self.engine = engine_from_network(network_from_onnx_path(onnx_path), config=config)
        if self.engine is None:
            raise RuntimeError("engine build returned None (미지원 op이거나 config 오류)")
        self._runner = TrtRunner(self.engine)
        self._runner.activate()      # 컨텍스트 생성(디바이스 메모리 확보)
        self._build_s = time.perf_counter() - t0

    def run(self, inputs: np.ndarray) -> np.ndarray:
        # polygraphy TrtRunner.infer가 set_tensor_address/execute_async_v3/stream.sync를 내부 수행.
        # (문서 원안의 pycuda mem_alloc/memcpy_htod/execute_async_v3/synchronize 블록을 대체)
        out = self._runner.infer({self._in_name: np.ascontiguousarray(inputs)})
        return list(out.values())[0]

    def _peak_mem_mb(self) -> float:
        """엔진 디바이스 메모리(근사, MB). TRT 10.x: device_memory_size는 deprecated →
        device_memory_size_v2 우선. Orin에선 tegrastats 병행 권장(문서 §4-3)."""
        for attr in ("device_memory_size_v2", "device_memory_size"):
            v = getattr(self.engine, attr, None)
            if v:
                self._mem_attr = attr
                return v / (1024 ** 2)
        return float("nan")

    def measure(self, model, precision, loader, evaluator,
                warmup=20, iters=200) -> BenchResult:
        sample = loader.one_batch()  # 대표 입력 1개(latency용)

        def _once():
            self.run(sample)         # polygraphy가 내부 동기화 → 별도 Context.synchronize 불필요

        median, p95 = self._timeit(_once, warmup, iters)

        # 정확도: 검증셋 전체를 돌려 evaluator가 top-1(또는 mAP) 계산
        # 정정(4): polygraphy TrtRunner.infer는 호스트 출력버퍼를 재사용한다(zero-copy).
        #   run()의 반환은 그 버퍼의 view라, 그대로 리스트에 모으면 5000개 전부가 마지막
        #   추론 결과를 가리켜 argmax가 동일해진다(실측 acc 0.0014 = 1/1000 우연). 문서 §4-3의
        #   추상 measure() 계약("preds를 모은 뒤 평가")이 zero-copy 러너와 만나면 무음 오답이 된다.
        #   → 각 추론 직후 .copy()로 스냅샷(1000 float ×5000 = 20MB, 지연경로엔 영향 없음).
        preds = [self.run(x).copy() for x in loader.eval_set()]
        acc = evaluator.compute_acc(preds, loader.gts())

        peak_mem = self._peak_mem_mb()
        note = f"polygraphy TrtRunner; mem_api={self._mem_attr}"
        if precision == "int8" and self._calib_feed is None:
            note += "; INT8 no-calibrator(스케일 없음 — 무의미)"

        res = BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=round(median, 4), latency_p95_ms=round(p95, 4),
            peak_mem_mb=round(peak_mem, 1), accuracy=round(acc, 4),
            engine_build_s=round(self._build_s, 1),
            trt_version=trt.__version__,   # 재현성: 어떤 TRT로 뽑았는지 봉인
            notes=note,
        )
        self._runner.deactivate()          # 컨텍스트 해제(다음 precision 빌드 전 메모리 회수)
        return res
