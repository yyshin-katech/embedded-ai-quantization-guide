# TRT 10.16 직접 파서·빌더 제약 — 로그 원문과 설계 규칙

2단계 `onnx_export_failures.md`의 3단계판. **export 실패**가 아니라 **엔진 빌드 실패**를 다룬다.
문서 §2.2.1(1단계 ResNet18을 ORT TensorRT-EP로 관측)이 "하드 블로커는 `zero_point≠0`
하나뿐, INT32 bias DQ는 2차 증상"으로 판정한 것을, **같은 QDQ를 polygraphy로 TRT 10.16
파서·빌더에 직접** 먹여 절제(`t03_parser_constraints.py`)한 실측 로그다.

- 환경: RTX 3080 / TensorRT **10.16.1.11** / polygraphy 0.50.3 / onnxruntime 1.23.2 (`t01_env.json`)
- 모델: torchvision **ResNet50** QDQ 변형 4종(A~D) + 2단계 **DETR** `detr_int8.onnx`(E)
- 결과 JSON: `t03.json` — 케이스별 `parse_ok / build_ok / stage / act_zp_nonzero_frac / zp_dtypes`

---

## 0. 결론 먼저 — 파서 제약과 빌더 제약은 별개 축

| 실패 축 | 언제 | 신호 | 처방 |
|---|---|---|---|
| **파서(parse)** | `network_from_onnx_path` 즉시 | `INVALID_NODE`, `shiftIsAllZeros`, `only activation datatypes` | QDQ를 대칭·bias 미양자화로 재생성 |
| **빌더(build)** | `engine_from_network` 중 | `Error Code 10: Could not find any implementation`, `Invalid Engine` | 해당 노드 INT8 제외(`nodes_to_exclude`) |

직접 파서에는 **두 개의 독립적 하드 파서 블로커**가 있다 — (1) INT32 bias DQ, (2) `zero_point≠0`.
그와 **직교하는 별도 축**으로 빌더 레벨 커널 부재(stem)가 있다. §2.2.1(ORT-EP 경로)이 본 것은
(2) 하나뿐이었는데, 그건 ORT-EP가 파서에 넘기기 전 (1)을 자체 그래프 최적화로 흡수하기 때문이다
(아래 §5에서 경로 차이로 설명). **경로 한정 정밀화이지 반증이 아니다.**

---

## 5개 케이스 절제 결과 (`t03.json`)

| 케이스 | 구성 | parse | build | 실패단계 | act zp≠0 | zp dtypes |
|---|---|:---:|:---:|---|:---:|---|
| **A** | 대칭 QInt8 · bias 미양자화 · stem 제외 | ✅ | ✅ | — | 0.0 | `[int8]` |
| **B** | 대칭 QInt8 · **bias 양자화(INT32 DQ)** · stem 제외 | ❌ | — | parse | 0.0 | `[int32, int8]` |
| **C** | **비대칭 QUInt8(zp≠0)** · bias 미양자화 · stem 제외 | ❌ | — | parse | 0.213 | `[int8, uint8]` |
| **D** | 대칭 QInt8 · bias 미양자화 · **stem 포함(conv1 INT8)** | ✅ | ❌ | build | 0.0 | `[int8]` |
| **E** | DETR `detr_int8.onnx`(2단계 ORT 실제 export) | ❌ | — | parse | 0.831 | `[int32, int8]` |

A만 통과 = **정본 처방**(대칭 QInt8 + `QuantizeBias=False` + stem 제외). B/C/E는 파서에서, D는 빌더에서 죽는다.

---

## A — 통과(정본 처방)

```
대칭 QInt8, per-channel, QuantizeBias=False, nodes_to_exclude=["/conv1/Conv"]
→ parse_ok=True, build_ok=True, act_zp_nonzero_frac=0.0, zp_dtypes=[int8]
```

activation zero_point 293개 전부 0, dtype은 int8만. 파서·빌더 모두 통과. 이 엔진이 `t02` INT8 3점의 그 엔진.

---

## B — INT32 bias DQ가 **독립적** 파서 하드 블로커 (대칭인데도)

구성: `ActivationSymmetric=True`(zp=0) 인데 `QuantizeBias=True`만 켬 → bias가 INT32 DQ로 삽입됨.
zp는 전부 0(`act_zp_nonzero_frac=0.0`)이라 §2.2.1의 "zp≠0" 블로커와 **무관**한데도 파서가 죽는다.

```
[W] onnxOpImporters.cpp:1703: For zero_point with type int32 TensorRT will use INT8 instead.
[E] IDequantizeLayer::setPrecision: Error Code 3: API Usage Error (Parameter check failed,
    condition: isQuantized(dataType) || (uint8QDQSupported() && dataType == DataType::kUINT8).
    A DequantizeLayer can only run in DataType::kINT8, DataType::kFP8, DataType::kFP4,
    or DataType::kINT4 precision In setPrecision at .../dequantizeLayer.cpp:131)
[E] ITensor::getDimensions: Error Code 3: API Usage Error (fc.bias_DequantizeLinear:
    only activation datatypes allowed as input to this layer.
    In checkActivation at .../dequantizeLayer.cpp:173)
[E] In node 0 with name: fc.bias_DequantizeLinear and operator: DequantizeLinear (parseNode):
    INVALID_NODE: Invalid Node - fc.bias_DequantizeLinear
```

**해석:** 직접 파서는 홀로 선 INT32 DequantizeLinear를 dtype 검사(`only activation datatypes allowed`)에서
막는다 — 이건 모델과 무관한 순수 타입 규칙이다. `QuantizeBias=False`는 직접 파서에서는
**선택이 아니라 필수**. (§2.2.1의 "2차 증상" 판정은 ORT-EP 경로에 한정 — §5 참조.)

---

## C — `zero_point≠0`가 파서 하드 블로커 (§2.2.1과 일치)

구성: `ActivationSymmetric=False` + `QUInt8` → activation zp가 비영(0.213이 비영). 첫 conv에서 죽는다.

```
[W] onnxOpImporters.cpp:1695: TensorRT supports QuantizeLinear/DequantizeLinear with UINT8
    zero_point only on DLA (version >= 3.16). Defaulting to INT8 instead.
    To import as UINT8, set the kIMPORT_UINT8_QUANTIZATION flag.
[E] onnxOpImporters.cpp:1738 In function QuantDequantLinearHelper:
    [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
[E] In node 61 with name: /layer1/layer1.0/downsample/downsample.0/Conv_output_0_QuantizeLinear
    and operator: QuantizeLinear (QuantDequantLinearHelper): INVALID_NODE:
    Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
```

**해석:** `shiftIsAllZeros(zeroPoint)` 단언 실패 = §2.2.1이 ORT-EP에서 본 것과 동일한 블로커.
직접 파서에서도 유효 → **경로 무관 하드 블로커**. 처방: `ActivationSymmetric=True`(대칭 QInt8).

---

## D — stem INT8 커널 부재는 **빌더** 실패 (파서는 통과)

구성: A와 동일하되 `nodes_to_exclude`에서 `/conv1/Conv`를 뺌 → stem conv1도 INT8 QDQ.
zp 전부 0, dtype int8만 → **파서는 통과**. 그런데 빌더가 stem 융합블록에서 죽는다.

```
[E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
    (Could not find any implementation for node
     onnx::Conv_497_quantized + /conv1/Conv + PWN(/relu/Relu) + /maxpool/MaxPool.
     In computeCosts at .../optimizer.cpp:4265)
[!] Invalid Engine. Please ensure the engine was built correctly
```

**해석:** 융합블록 `conv1(3ch 7×7) + relu + maxpool`에 대한 INT8 tactic이 카탈로그에 없다.
나머지 53개 conv는 전부 INT8을 받았는데(로그의 requested-precision 맵) 이 stem만 구현이 없다.
파서 제약과 **완전히 다른 축** — QDQ는 완벽히 유효한데 커널이 없는 것. 처방: `nodes_to_exclude=["/conv1/Conv"]`.
(implicit 캘리브 경로(`t04`)는 TRT가 층별 정밀도를 자동 결정하므로 이 stem을 자동으로 FP16에 남겨 이 실패가 안 난다.)

---

## E — 실제 모델(DETR)은 두 블로커를 동시에, 파서는 첫 것에서 abort

`detr_int8.onnx`(2단계 ORT `quantize_static` 산출, 비대칭 QUInt8 + INT32 bias)를 그대로 먹임.
`act_zp_nonzero_frac=0.831`(대부분 비영) + `zp_dtypes=[int32, int8]`(bias INT32도 존재) — B와 C의 블로커를 **둘 다** 가짐.

```
[E] onnxOpImporters.cpp:1738 In function QuantDequantLinearHelper:
    [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
[E] In node 0 with name: /model/Tile_output_0_DequantizeLinear and operator: DequantizeLinear
    (QuantDequantLinearHelper): INVALID_NODE:
    Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
```

**해석:** 파서는 그래프 순서상 **먼저 만난 블로커**(node 0의 zp≠0)에서 즉시 멈춘다.
INT32 bias 문제(B)는 이 노드를 넘겼어야 보였을 것 — 즉 로그가 한 번에 한 블로커만 보여주므로
"에러 하나 고쳤더니 다음 게 나온다"가 정상. ORT 기본 설정으로 뽑은 QDQ는 직접 파서에 그대로는 못 들어간다.

---

## 5. 왜 §2.2.1(ORT-EP)과 직접 파서가 갈리나 — 경로 차이

§2.2.1은 ResNet18을 **ONNX Runtime의 TensorRT Execution Provider**로 돌려 관측했고,
"대칭이면 bias INT32 DQ가 있어도 성공(2차 증상)"이라고 정당하게 측정했다. 여기(B)선 대칭인데도 실패한다.
차이는 **모델이 아니라 경로**다:

- **ORT TensorRT-EP**: 파서에 넘기기 전 ORT가 자체 그래프 최적화(상수 접기·QDQ 재배치)로
  INT32 bias DQ를 conv에 흡수/제거한다 → TRT 파서는 **홀로 선 INT32 DQ를 아예 못 본다**.
  그래서 §2.2.1 case C(대칭+bias)가 통과했다.
- **직접 파서**(polygraphy / trtexec / ModelOpt→엔진): 원본 ONNX 그래프를 그대로 파싱 →
  홀로 선 INT32 DequantizeLinear를 dtype 검사에서 하드 거부(B).

근거: B의 에러가 `only activation datatypes allowed as input`(순수 dtype 검사, 모델 독립)라는 점.
따라서 §2.2.1의 결론은 **ORT-EP 경로에서 유효**하고, 직접 파서 경로에서는 `QuantizeBias=False`가
필수로 **강화**된다. 삭제·반전이 아니라 경로 병기(가법적 정밀화).

---

## 설계 규칙 (직접 파서로 INT8 엔진 만들 때)

1. **activation은 대칭 QInt8** (`ActivationSymmetric=True`). 비대칭 QUInt8은 `shiftIsAllZeros` 파서 단언에서 죽는다(C·E).
2. **bias는 양자화하지 말 것** (`QuantizeBias=False`). 직접 파서는 INT32 bias DQ를 독립적으로 거부(B) — ORT-EP와 달리 선택 아님 필수.
3. **stem(첫 conv)은 INT8에서 제외** (`nodes_to_exclude=["/conv1/Conv"]`). 파서는 통과하나 빌더가 INT8 tactic 부재로 죽는다(D). 파서 제약과 별개 축.
4. **INT8은 항상 FP16과 병용**(`int8=True, fp16=True`). INT8 커널 없는 층을 FP32 대신 FP16으로 폴백해 실전 배포 구성(trtexec `--int8 --fp16`)과 일치.
5. **ORT 기본 설정 QDQ는 직접 파서에 그대로 못 넣는다**(E). ORT 기본은 비대칭 QUInt8 + bias 양자화 → 위 1·2를 명시적으로 꺼야 한다.
6. implicit 캘리브(`t04`)는 QDQ 없이 TRT가 층별 정밀도를 자동 결정 → stem 자동 폴백으로 D가 안 나지만, **10.1에서 deprecated**(explicit로 대체) — 신규 파이프라인은 explicit QDQ 권장.
