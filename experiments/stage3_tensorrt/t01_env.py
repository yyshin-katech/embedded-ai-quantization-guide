"""
t01 — Stage 3 환경 실측 캡처.

검증 대상(문서 §3.1 "0단계에서 설치 가정" · 실습1/2 "trtexec ..." 전제):
  1. trtexec가 정본 pip 스택에 실재하는가? → 부재 확정 + 대체 경로 확인.
  2. TRT 10.16의 implicit 캘리브레이터(IInt8*Calibrator2)가 아직 있는가? (버전 경계)
  3. explicit(STRONGLY_TYPED)/INT8/FP16/DLA 빌더 능력 introspection.
  4. modelopt.onnx QDQ export 경로 가용성.
결과 → t01_env.json.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t3_common as C  # noqa: E402

res = {}

# --- GPU 라이브니스(필수) ---
res["gpu"] = C.liveness()

# --- trtexec 전수조사 ---
trtexec_hits = []
for root in ["/home/yuyeong/emb-ai", "/usr", "/opt", "/usr/local/cuda"]:
    try:
        out = subprocess.run(
            ["find", root, "-name", "trtexec", "-type", "f"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        if out:
            trtexec_hits += out.splitlines()
    except Exception:
        pass
res["trtexec_on_PATH"] = shutil.which("trtexec")
res["trtexec_filesystem_hits"] = trtexec_hits
res["trtexec_present"] = bool(trtexec_hits or res["trtexec_on_PATH"])

# --- TensorRT Python API introspection ---
import tensorrt as trt  # noqa: E402

res["tensorrt_version"] = trt.__version__
res["implicit_calibrators"] = {
    "IInt8EntropyCalibrator2": hasattr(trt, "IInt8EntropyCalibrator2"),
    "IInt8MinMaxCalibrator": hasattr(trt, "IInt8MinMaxCalibrator"),
    "IInt8EntropyCalibrator": hasattr(trt, "IInt8EntropyCalibrator"),
    "IInt8LegacyCalibrator": hasattr(trt, "IInt8LegacyCalibrator"),
}
bf = trt.BuilderFlag
res["builder_flags"] = {
    k: hasattr(bf, k) for k in
    ["INT8", "FP16", "BF16", "FP8", "INT4", "OBEY_PRECISION_CONSTRAINTS",
     "PREFER_PRECISION_CONSTRAINTS", "STRICT_TYPES"]
}
res["strongly_typed_flag"] = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")

# DLA 코어 수 — RTX 3080(dGPU)이면 0, Jetson Orin이면 2. 실습5 out-of-scope 하드 증거.
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
res["num_DLA_cores"] = int(builder.num_DLA_cores)
res["platform_has_fast_int8"] = bool(builder.platform_has_fast_int8)
res["platform_has_fast_fp16"] = bool(builder.platform_has_fast_fp16)

# --- polygraphy 대체 빌드 경로 가용성 ---
try:
    from polygraphy.backend.trt import (  # noqa: F401
        network_from_onnx_path, engine_from_network, CreateConfig, TrtRunner, Calibrator,
    )
    import polygraphy
    res["polygraphy_build_path"] = {"available": True, "version": polygraphy.__version__}
except Exception as e:
    res["polygraphy_build_path"] = {"available": False, "error": str(e)}

# --- modelopt.onnx QDQ export 경로 ---
try:
    import modelopt.onnx.quantization  # noqa: F401
    res["modelopt_onnx"] = {"importable": True}
except Exception as e:
    res["modelopt_onnx"] = {"importable": False, "error": str(e).splitlines()[-1]}

# --- onnxruntime (대칭 QDQ 생성 경로) ---
try:
    import onnxruntime as ort
    res["onnxruntime_version"] = ort.__version__
except Exception as e:
    res["onnxruntime_version"] = "ERR:" + str(e)

out_path = os.path.join(HERE, "t01_env.json")
with open(out_path, "w") as f:
    json.dump(res, f, indent=2)

print("=== t01 환경 실측 ===")
print("TensorRT      :", res["tensorrt_version"])
print("trtexec 실재  :", res["trtexec_present"], "| PATH:", res["trtexec_on_PATH"],
      "| fs hits:", len(res["trtexec_filesystem_hits"]))
print("implicit calib:", res["implicit_calibrators"])
print("STRONGLY_TYPED:", res["strongly_typed_flag"], "| INT8/FP16 fast:",
      res["platform_has_fast_int8"], res["platform_has_fast_fp16"])
print("num_DLA_cores :", res["num_DLA_cores"], "(0=dGPU, DLA 실습 범위 밖)")
print("polygraphy    :", res["polygraphy_build_path"])
print("modelopt.onnx :", res["modelopt_onnx"])
print("→", out_path)
