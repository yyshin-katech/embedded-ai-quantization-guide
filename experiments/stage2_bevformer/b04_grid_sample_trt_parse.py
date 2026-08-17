#!/usr/bin/env python3
"""
b04 — TensorRT 10.16 OnnxParser 로 GridSample 4D vs 5D 파싱.

§4.6.1 / #8 단정 검증: TensorRT native GridSample 은 4D(rank-4)만, 5D 볼류메트릭은 파싱 실패(issue #3890).
trtexec 바이너리가 없어 tensorrt.OnnxParser(파이썬 API)로 직접 파싱하고 에러 원문을 채집한다.
"""
import os, json
import tensorrt as trt

OUT = os.path.dirname(os.path.abspath(__file__))
print(f"TensorRT {trt.__version__}\n")


def parse(path):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    try:
        network = builder.create_network(0)  # TRT10: explicit batch 기본
    except Exception:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(path, "rb") as f:
        ok = parser.parse(f.read())
    errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
    return ok, errs


def main():
    results = {}
    cases = [
        ("4D (opset17)", "_gs4d_op17.onnx"),
        ("5D volumetric (legacy opset20)", "_gs5d_legacy_op20.onnx"),
        ("5D volumetric (dynamo opset20)", "_gs5d_dynamo_op20.onnx"),
    ]
    for label, fn in cases:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print(f"### {label}: (파일 없음 {fn})")
            continue
        ok, errs = parse(p)
        print(f"### {label}")
        print(f"   parse ok = {ok}")
        for e in errs:
            print(f"   ERR: {e}")
        results[label] = {"file": fn, "parse_ok": ok, "errors": errs}
        print()
    with open(os.path.join(OUT, "b04_trt_parse_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("저장: b04_trt_parse_result.json")


if __name__ == "__main__":
    main()
