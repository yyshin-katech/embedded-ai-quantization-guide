"""
t02 — 실습1 헤드라인: FP32 / FP16 / INT8 지연·정확도 3점.

문서 실습1은 전부 `trtexec --onnx=... --fp16/--int8`로 쓰여 있으나 정본 스택에 trtexec가
없다(t01). 여기서는 **동일 3점을 polygraphy Python API로** 빌드해 재현한다.

모델: torchvision ResNet50 (IMAGENET1K_V1, 공개 top-1 76.13%) — 1단계 ResNet 계열 연속성 +
export 리스크 최소. 배치=1(지연의 정본 지표). INT8은 explicit QDQ(ORT quantize_static,
대칭 QInt8 — §2.2.1대로 zero_point=0)로 만들어 TRT 파서가 그대로 먹는다.

산출물: t02_*.engine, resnet50_fp32.onnx, resnet50_int8_qdq.onnx, t02.json
(재실행 시 존재 산출물은 건너뜀).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t3_common as C  # noqa: E402

FP32_ONNX = os.path.join(C.WS, "resnet50_fp32.onnx")
INT8_ONNX = os.path.join(C.WS, "resnet50_int8_qdq.onnx")
INPUT = "input"
RESULT = os.path.join(HERE, "t02.json")


def export_fp32():
    if os.path.exists(FP32_ONNX):
        print("  [skip] fp32 onnx 존재")
        return
    import torch
    import torchvision
    print("  ResNet50(V1) export → ONNX (dynamo=False, opset17)")
    m = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(1, 3, 224, 224)
    # torch 2.11 기본 dynamo=True(2단계 실측) → 고전 exporter 강제
    torch.onnx.export(
        m, dummy, FP32_ONNX,
        input_names=[INPUT], output_names=["logits"],
        opset_version=17, dynamo=False,
    )


class CalibReader:
    """ORT quantize_static용 — calib200을 배치1로 흘려줌."""
    def __init__(self, n=200):
        self.tv, _ = C.load_tv_cache()
        self.idx = list(C.calib_indices(min(n, 200)))
        self._it = iter(self.idx)

    def get_next(self):
        try:
            j = next(self._it)
        except StopIteration:
            return None
        x = C.preprocess_nchw(self.tv[j][None])  # (1,3,224,224)
        return {INPUT: x}

    def rewind(self):
        self._it = iter(self.idx)


def quantize_int8():
    if os.path.exists(INT8_ONNX):
        print("  [skip] int8 qdq onnx 존재")
        return
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = os.path.join(C.WS, "resnet50_fp32_pre.onnx")
    quant_pre_process(FP32_ONNX, pre)
    print("  ORT quantize_static → 대칭 QInt8 QDQ (per-channel, zero_point=0)")
    quantize_static(
        pre, INT8_ONNX,
        calibration_data_reader=CalibReader(200),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # stem(3ch 7×7) conv1은 제외 — Q/DQ가 INT8을 강제하나 TRT 10.16에 해당 융합
        # (conv1+relu+maxpool) INT8 커널이 없어 build 실패(Error Code 10). 실전 표준.
        nodes_to_exclude=["/conv1/Conv"],
        # 대칭(zp=0)만으로는 부족 — ORT 기본 QuantizeBias=True가 INT32 bias DQ를 삽입,
        # TRT 10.16 파서가 이를 하드 거부(INT32 DQ 불가). t03에서 절제로 검증. 여기선 끔.
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False},
    )


def main():
    C.liveness()
    print("[1] FP32 ONNX export")
    export_fp32()
    print("[2] INT8 QDQ ONNX (ORT, 대칭)")
    quantize_int8()

    tv, labels = C.load_tv_cache()
    x1 = C.preprocess_nchw(tv[0][None])  # 배치1 지연 입력
    eval_idx = np.arange(0, 5000)        # 정확도 subset (배치1 루프)

    specs = [
        ("fp32", FP32_ONNX, "fp32"),
        ("fp16", FP32_ONNX, "fp16"),
        # 순수 int8는 스템 conv1 융합 블록의 INT8 커널 부재로 빌드 실패(Error Code 10).
        # 실전 처방 = INT8 + FP16 폴백(trtexec의 `--int8 --fp16`). 이게 실제 배포 구성.
        ("int8", INT8_ONNX, "int8_fp16"),
    ]
    out = {"model": "resnet50_IMAGENET1K_V1", "eval_n": int(len(eval_idx)),
           "public_fp32_top1": 0.7613, "results": {}}

    for name, onnx_path, prec in specs:
        print(f"[3] build+bench {name}  (onnx={os.path.basename(onnx_path)})")
        eng_path = os.path.join(C.WS, f"t02_{name}.engine")
        engine = C.build_engine(onnx_path, prec, save_path=eng_path)
        lat = C.bench_latency(engine, {INPUT: x1}, iters=300, warmup=80)
        top1, ntot = C.evaluate_top1(engine, tv, labels, eval_idx, INPUT, batch=1)
        # int8 커널 증거
        layer_txt = C.engine_layer_info(engine)
        int8_kernels = sum(1 for ln in layer_txt.splitlines()
                           if ("int8" in ln.lower() or "imma" in ln.lower()))
        out["results"][name] = dict(
            latency_ms=lat, top1=round(top1, 4), eval_n=ntot,
            int8_kernel_lines=int(int8_kernels),
            engine_bytes=os.path.getsize(eng_path),
        )
        print(f"    p50={lat['p50']:.3f}ms p90={lat['p90']:.3f}ms "
              f"thrpt={lat['throughput']:.0f}/s | top1={top1*100:.2f}% "
              f"| int8_kernel_lines={int8_kernels}")
        del engine

    # 요약 배수
    r = out["results"]
    base = r["fp32"]["latency_ms"]["p50"]
    for name in r:
        r[name]["speedup_vs_fp32"] = round(base / r[name]["latency_ms"]["p50"], 2)

    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== 3점 요약 ===")
    for name in ["fp32", "fp16", "int8"]:
        d = r[name]
        print(f"  {name:5s}  p50 {d['latency_ms']['p50']:.3f}ms  "
              f"×{d['speedup_vs_fp32']:<4}  top1 {d['top1']*100:.2f}%  "
              f"engine {d['engine_bytes']//1024}KB")
    print("→", RESULT)


if __name__ == "__main__":
    main()
