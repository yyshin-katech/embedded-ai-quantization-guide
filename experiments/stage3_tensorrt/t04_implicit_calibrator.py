"""
t04 — 실습3 검증: implicit INT8 캘리브레이션이 TRT 10.16에서 여전히 되나.

문서 §2.2/실습3은 implicit(IInt8EntropyCalibrator2)을 "10.1 deprecated → 11.0 제거,
정본 10.16에선 살아있음"으로 서술. 여기서 **QDQ 없는 FP32 ONNX + polygraphy Calibrator
(IInt8EntropyCalibrator2)** 로 INT8 엔진을 실제로 빌드해 확인한다.

explicit(t02)과의 대비:
- implicit은 캘리브레이터가 빌드 중 스케일 결정 → 그래프에 Q/DQ 불필요.
- TRT가 레이어별 정밀도를 자동 결정하므로 stem 커널 부재도 자동 폴백(explicit은 수동 제외 필요).
결과 → t04.json (+ 캘리브 캐시, deprecation 경고 캡처).
"""
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stderr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t3_common as C  # noqa: E402
from t02_latency_3point import FP32_ONNX, INPUT  # noqa: E402

RESULT = os.path.join(HERE, "t04.json")
CACHE = os.path.join(C.WS, "t04_calib.cache")


def calib_gen(tv, idx):
    for j in idx:
        yield {INPUT: C.preprocess_nchw(tv[j][None])}


def main():
    import tensorrt as trt
    from polygraphy.backend.trt import (
        network_from_onnx_path, engine_from_network, CreateConfig, Calibrator, save_engine,
    )

    C.liveness()
    tv, labels = C.load_tv_cache()
    idx = list(C.calib_indices(200))

    # IInt8EntropyCalibrator2를 명시적으로 base class로 — 문서 실습3의 그 API.
    calibrator = Calibrator(
        data_loader=calib_gen(tv, idx),
        cache=CACHE,
        BaseClass=trt.IInt8EntropyCalibrator2,
    )
    cfg = CreateConfig(
        memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 6 << 30},
        int8=True, fp16=True,   # fp16 병용: implicit도 INT8 커널 없는 층은 FP16 폴백
        calibrator=calibrator,
    )

    print("[build] implicit INT8 (IInt8EntropyCalibrator2, calib 200장) — deprecation 경고 캡처")
    warn_buf = io.StringIO()
    dep_warnings = []
    with warnings.catch_warnings(record=True) as wlist, redirect_stderr(warn_buf):
        warnings.simplefilter("always")
        eng_path = os.path.join(C.WS, "t04_int8_implicit.engine")
        engine = engine_from_network(network_from_onnx_path(FP32_ONNX), config=cfg)
        save_engine(engine, eng_path)
        for w in wlist:
            if "deprecat" in str(w.message).lower():
                dep_warnings.append(str(w.message)[:120])
    stderr_txt = warn_buf.getvalue()
    dep_in_stderr = [ln for ln in stderr_txt.splitlines()
                     if "deprecat" in ln.lower() and "calibrat" in ln.lower()][:3]

    x1 = C.preprocess_nchw(tv[0][None])
    lat = C.bench_latency(engine, {INPUT: x1}, iters=300, warmup=80)
    top1, ntot = C.evaluate_top1(engine, tv, labels, np.arange(0, 5000), INPUT, batch=1)
    layer_txt = C.engine_layer_info(engine)
    int8_lines = sum(1 for ln in layer_txt.splitlines()
                     if ("int8" in ln.lower() or "imma" in ln.lower()))

    out = {
        "path": "implicit calibration (IInt8EntropyCalibrator2)",
        "trt_version": trt.__version__,
        "calibrator_class": "IInt8EntropyCalibrator2",
        "build_ok": True,
        "cache_written": os.path.exists(CACHE),
        "cache_bytes": os.path.getsize(CACHE) if os.path.exists(CACHE) else 0,
        "deprecation_warnings_py": dep_warnings,
        "deprecation_in_trt_log": dep_in_stderr,
        "latency_ms": lat,
        "top1": round(top1, 4),
        "eval_n": ntot,
        "int8_kernel_lines": int(int8_lines),
        "engine_bytes": os.path.getsize(eng_path),
    }
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  build_ok=True | calib cache={out['cache_bytes']}B "
          f"| int8커널줄={int8_lines}")
    print(f"  p50={lat['p50']:.3f}ms thrpt={lat['throughput']:.0f}/s | top1={top1*100:.2f}%")
    print(f"  deprecation(py)={len(dep_warnings)} trt_log={len(dep_in_stderr)}")
    print("→", RESULT)


if __name__ == "__main__":
    main()
