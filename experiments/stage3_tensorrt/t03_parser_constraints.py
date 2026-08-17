"""
t03 — §2.2.1 정밀화: TRT 10.16 파서/빌더 제약을 직접 절제.

1단계 §2.2.1은 ResNet18을 ORT TensorRT-EP 경유로 관측해 "하드 블로커는 zero_point≠0
하나뿐, INT32 bias DQ는 2차 증상"으로 판정했다. 여기서는 ResNet50 QDQ 변형들을
**polygraphy로 TRT 10.16 파서·빌더에 직접** 먹여 파서 레벨 제약과 빌더 레벨 제약을
분리한다.

케이스(전부 ResNet50, stem 제외 여부만 D에서 변주):
  A sym QInt8 · bias 미양자화 · stem 제외      → 기대 parse✅ build✅ (t02 정본)
  B sym QInt8 · bias 양자화(INT32 DQ) · stem 제외 → INT32 DQ 파서 제약 시험
  C asym QUInt8(zp≠0) · bias 미양자화 · stem 제외  → zero_point 파서 제약 시험
  D sym QInt8 · bias 미양자화 · stem 포함(conv1 양자화) → 빌더 커널 제약 시험
  E detr_int8.onnx (2단계 ORT 실제 export)        → 실전 혼합(INT32+비대칭) 시험

결과 → t03.json. 각 케이스에 parse_ok / build_ok / stage / err 요약 + zero_point 통계.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t3_common as C  # noqa: E402
from t02_latency_3point import CalibReader, FP32_ONNX, INT8_ONNX, INPUT  # noqa: E402

RESULT = os.path.join(HERE, "t03.json")
B_ONNX = os.path.join(C.WS, "rn50_B_sym_bias.onnx")
Cc_ONNX = os.path.join(C.WS, "rn50_C_asym.onnx")
D_ONNX = os.path.join(C.WS, "rn50_D_stem_included.onnx")
DETR = os.path.join(C.REPO, "_workspace", "stage2", "detr_int8.onnx")


def make_variant(out_path, activation_sym, quantize_bias, exclude_stem, act_type):
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    if os.path.exists(out_path):
        print(f"  [skip] {os.path.basename(out_path)} 존재")
        return
    pre = os.path.join(C.WS, "rn50_pre_tmp.onnx")
    quant_pre_process(FP32_ONNX, pre)
    extra = {"WeightSymmetric": True, "QuantizeBias": quantize_bias}
    extra["ActivationSymmetric"] = activation_sym
    kw = {}
    if exclude_stem:
        kw["nodes_to_exclude"] = ["/conv1/Conv"]
    quantize_static(
        pre, out_path,
        calibration_data_reader=CalibReader(200),
        quant_format=QuantFormat.QDQ,
        activation_type=act_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options=extra,
        **kw,
    )


def zp_stats(onnx_path):
    """activation zero_point(=Q/DQ의 3번째 입력) 중 int8/uint8의 비영 비율, dtype 집합."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    dtypes = set()
    nonzero = 0
    total = 0
    for n in m.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) >= 3:
            t = inits.get(n.input[2])
            if t is None:
                continue
            arr = numpy_helper.to_array(t)
            dtypes.add(str(arr.dtype))
            if "int32" in str(arr.dtype):
                continue  # bias는 별도
            total += 1
            if np.any(arr != 0):
                nonzero += 1
    return {"zp_dtypes": sorted(dtypes),
            "act_zp_nonzero_frac": round(nonzero / max(total, 1), 3),
            "act_zp_count": total}


def try_parse_build(onnx_path, do_build=True):
    """parse → build 순차 시도. 실패 지점(stage)과 에러 요약 반환."""
    import tensorrt as trt
    from polygraphy.backend.trt import network_from_onnx_path, engine_from_network, CreateConfig
    out = {"parse_ok": False, "build_ok": False, "stage": None, "err": None}
    try:
        # polygraphy network_from_onnx_path는 eager — 호출 즉시 파싱(실패 시 raise).
        loader = network_from_onnx_path(onnx_path)
        net = loader() if callable(loader) else loader  # (builder, network, parser)
        out["parse_ok"] = True
    except Exception as e:
        out["stage"] = "parse"
        out["err"] = _short(e)
        return out
    if not do_build:
        return out
    try:
        cfg = CreateConfig(
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 6 << 30},
            int8=True, fp16=True,
        )
        engine_from_network(net, config=cfg)
        out["build_ok"] = True
    except Exception as e:
        out["stage"] = "build"
        out["err"] = _short(e)
    return out


def _short(e):
    s = str(e)
    for key in ["DequantizeLayer can only run", "Could not find any implementation",
                "zero", "UINT8", "kUINT8", "Could not parse ONNX", "Invalid Engine"]:
        i = s.find(key)
        if i >= 0:
            return s[max(0, i - 40):i + 120].replace("\n", " ")
    return s[:160].replace("\n", " ")


def main():
    C.liveness()
    from onnxruntime.quantization import QuantType
    print("[gen] 변형 생성 (B/C/D)")
    make_variant(B_ONNX, activation_sym=True, quantize_bias=True,
                 exclude_stem=True, act_type=QuantType.QInt8)
    make_variant(Cc_ONNX, activation_sym=False, quantize_bias=False,
                 exclude_stem=True, act_type=QuantType.QUInt8)
    make_variant(D_ONNX, activation_sym=True, quantize_bias=False,
                 exclude_stem=False, act_type=QuantType.QInt8)

    cases = [
        ("A_sym_nobias_exstem", INT8_ONNX, True),
        ("B_sym_bias_exstem", B_ONNX, False),      # 파서에서 걸릴 것 → build 생략
        ("C_asym_qu8_exstem", Cc_ONNX, True),
        ("D_sym_nobias_incstem", D_ONNX, True),
        ("E_detr_int8_ORT", DETR, False),          # 실전 export, 파서만
    ]
    out = {"note": "TRT 10.16 직접 파서/빌더 제약 절제 (ResNet50 + DETR)", "cases": {}}
    for name, path, do_build in cases:
        if not os.path.exists(path):
            out["cases"][name] = {"skipped": "onnx 없음: " + path}
            print(f"  {name}: onnx 없음")
            continue
        z = zp_stats(path)
        r = try_parse_build(path, do_build=do_build)
        out["cases"][name] = {**r, **z}
        print(f"  {name:24s} parse={r['parse_ok']} build={r['build_ok']} "
              f"stage={r['stage']} | act_zp≠0={z['act_zp_nonzero_frac']} "
              f"dtypes={z['zp_dtypes']}")
        if r["err"]:
            print(f"      err: {r['err']}")

    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)
    print("→", RESULT)


if __name__ == "__main__":
    main()
