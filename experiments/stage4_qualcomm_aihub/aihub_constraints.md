# Qualcomm AI Hub — 실측 제약·설계 규칙 (Hexagon HTP, ResNet50)

> 4단계 멀티 SoC의 **Qualcomm 벤더-NPU 축**을 보드 없이 클라우드 실기기로 실측한 기록.
> 모델 = torchvision ResNet50(3·5단계 자산). 디바이스 = QCS8550(Proxy) · SA8775P ADP(자동차).
> 런타임 = `--target_runtime qnn_context_binary` (ONNX → Hexagon HTP context binary).
> qai_hub 0.54.0 / QAIRT SDK 2.45.0 / HTP v73.

로그 원문은 각 스크립트의 `*.log` 및 `results/`·`raw/` JSON에 있다. 절대값은 상대 관계로만 읽는다(§ 캐비앗).

---

## 헤드라인 실측 (지연 · offload — 신뢰)

`submit_compile_job` → `submit_profile_job`. 그래프 구조를 충실히 보존하므로 정확도 함정과 무관하게 유효.

| 디바이스 · 정밀도 | on-device 지연 | INT8 배속 | NPU offload | execution_cycles | 레이어 |
|---|---:|---:|---:|---:|---:|
| QCS8550 · FP32(→HTP fp16) | 1864 µs | — | 125/125 (100%) | 4,677,822 | 125 |
| QCS8550 · INT8 QDQ | 1052 µs | **×1.77** | 128/128 (100%) | 3,754,903 | 128 |
| SA8775P ADP · FP32(→HTP fp16) | 3056 µs | — | 125/125 (100%) | 6,192,577 | 125 |
| SA8775P ADP · INT8 QDQ | 1505 µs | **×2.03** | 128/128 (100%) | 4,462,570 | 128 |

- **두 디바이스 모두 100% NPU offload** — 깨끗한 CNN이라 폴백 0. §06 목표("Offloaded ≈ Total, subgraph 최소")를 벤더 실기기에서 정량 달성.
- INT8 배속은 execution_cycles로 교차 확증(FP32 사이클 > INT8 사이클, 두 디바이스 공통).
- 자동차 SA8775P HTP가 프록시보다 ~1.6× 느리지만(사이클도 더 많음) INT8/FP32 관계는 유지.
- 잡: 컴파일 FP32 `jpyx31r05` · INT8정리본 `jglxmv22g` · SA8775P FP32 `j5w76e34g`/INT8 `jg9mnlym5`.
  프로파일 QCS8550 FP32 `jp8xyqmqg`/INT8 `j5674yznp` · SA8775P FP32 `jp41rld2p`/INT8 `jpyx31j05`.

---

## 발견 #1 — AI Hub 프론트엔드는 ORT/TRT보다 엄격 (value_info ↔ IO 충돌 거부)

ORT 양자화기가 shape-inference로 넣은 출력 텐서 `logits`가 `graph.value_info`와 `graph.output`에 **동시에** 존재한다(ONNX 스펙 위반). ORT·TensorRT는 관대해 통과하지만 AI Hub 컴파일은 실패한다.

```
# 원본 INT8 QDQ 컴파일 (jp2wyk06p)
❌ FAILED  Tensors {'logits'} occur in value_info but also in model IO.
```

**수정** (`scripts/clean_valueinfo_for_aihub.py`) — IO와 충돌하는 value_info만 제거. 계산 그래프 불변:

```python
io = set(i.name for i in g.input) | set(o.name for o in g.output)
g.value_info = [v for v in g.value_info if v.name not in io]   # 122 → 121
onnx.checker.check_model(m)   # PASS
```

- FP32 ONNX는 value_info=0이라 무영향. 문제는 **양자화기가 산출한 QDQ ONNX**에 국한.
- 정리본 컴파일 `jglxmv22g` → SUCCESS, 100% offload.
- **설계 규칙:** 다른 툴체인(ORT/TRT)에서 통과하던 QDQ ONNX라도 AI Hub 제출 전 value_info↔IO 충돌을 한 번 걸러라.

---

## 발견 #2 — 외부 QDQ INT8 = on-device 정확도의 조용한 붕괴 (silent-wrong)

컴파일·프로파일이 통과(100% offload)해도 **on-device 수치가 맞는다는 뜻은 아니다.** QCS8550에서 200장으로 on-device 예측을 ORT-CPU·정답과 대조:

| 경로 (QCS8550, 200장) | on-device top-1 | ORT 일치율 | distinct |
|---|---:|---:|---:|
| ORT-CPU (기준) | 0.750 | — | 181 |
| FP32(→HTP fp16) | 0.745 | 0.96 | 183 |
| INT8 · 외부 ORT-QDQ | **0.005** | 0.005 | 35 |
| INT8 · AI Hub 자체 quantize | **0.735** | 0.94 | 184 |

- 20장 대조군도 동일: FP32 0.75 / INT8-외부 0.0 → **스케일 무관한 구조적 붕괴**.
- FP32(fp16)가 충실하다는 것은 **입력(NCHW)·전처리·업로드·출력파싱·정확도 하네스가 모두 정상**이라는 증거 → 붕괴는 외부 QDQ 임포트 특유.
- 예측이 35개 클래스에 집중(409/862/818/506/723). exit 0·정상 shape 출력이라 **조용하다**.
- **범인:** ORT 양자화기의 QDQ scale을 AI Hub HTP 임포트가 존중하지 않음. **동일 ONNX가 4단계 CPU 프록시의 x86 CPUEP에선 top-1 0.753**(1,000장, `experiments/stage5_infrastructure/cpu_proxy/results/resnet50__x86_cpu__int8.json`)인데 HTP에선 0.005 — 같은 자산이 런타임에 따라 정상↔붕괴로 갈리므로 범인은 QDQ ONNX가 아니라 HTP 임포트다.
- **올바른 경로 = AI Hub 자체 `submit_quantize_job`** (HTP-native QDQ 생성). 대조 결과는 위 표 마지막 행:
  top-1 **0.735** · ORT 일치 0.94 · distinct 184 — 외부 QDQ의 0.005 붕괴가 FP32(fp16) 0.745·ORT-CPU 0.750에 **근접 회복**한다. 붕괴가 임포트 특유였음을 확정. 부수 실측: native-quant는 **더 leaner**(748 µs · 1,985,339 cyc · 127층 100% NPU)라 외부-QDQ INT8(1052 µs · 3,754,903 cyc · 128층)보다 **빠르다** — HTP-native 양자화가 더 최적화된 그래프를 만든다(외부 QDQ는 임포트만 될 뿐 최적 스케줄이 아님).
- 잡: FP32-200 `j5qyq627g` · INT8외부-200 `j5mmxn675` · 20장 감별 FP32 `j5mmxn3y5`/INT8 `jgor62v4g`.
  AI Hub quantize `jp1jz48kp` → compile `jgnnv69vg` → profile `jpyx314r5` → inference `jglxmv0eg`.

---

## 발견 #3 — FP32 ONNX도 HTP에선 fp16으로 실행 (native fp32 없음)

HTP는 fp16/int8 네이티브다. FP32 모델을 올리면 그래프 첫머리에 자동 변환 노드가 삽입된다(프로파일 execution_detail 실측):

```json
{"name": "QNN_DATATYPE_FLOAT_32_converted_input_QNN_DATATYPE_FLOAT_16",
 "compute_unit": "NPU", "execution_cycles": 35758}
```

- 이 실측의 "FP32" 행은 엄밀히 **fp16-on-HTP**. 그래서 FP32 top-1도 ORT-CPU(진짜 fp32)와 완전 동일이 아니라 0.745(−0.005).
- **설계 규칙:** "FP32 대비 INT8 배속"은 "fp16 대비 int8 배속"으로 읽어라. HTP엔 fp32 경로가 없다.

---

## 발견 #4 — 입력 인터페이스는 엄격 NCHW (NHWC 레이아웃 가설 기각)

붕괴(#2)의 첫 가설은 "HTP가 channel-last(NHWC)를 기대한다"였다. 두 경로로 반증:

```python
# 컴파일 잡의 선언 입력
job.target_shapes = {'input': ((1, 3, 224, 224), 'float32')}   # NCHW

# NHWC (1,224,224,3) 피드 시도 (jgj7nve1g)
❌ FAILED  Cannot assign data from unexpected shape.
           Expected [1, 3, 224, 224], got [1, 224, 224, 3].
```

- AI Hub는 **선언된 NCHW를 엄격 검증**한다. `input_specs={"input":(1,3,224,224)}`로 컴파일했으므로 인터페이스도 NCHW.
- 내 NCHW 피드가 올바른 형상이었고, 붕괴 원인은 레이아웃이 아니라 #2의 외부-QDQ 임포트.
- **설계 규칙:** 추론 입력은 **컴파일 시 선언한 레이아웃 그대로** 공급. `inputs={"<입력텐서명>": [arr, ...]}`의 키는 모델 실제 입력명과 일치해야 한다(ResNet50은 `"input"`).

---

## 범위 밖 (정직한 폴백)

- **TI TDA4VM · Renesas RZ/V2H:** AI Hub는 Qualcomm 전용 → 이 실측은 §06 세 벤더 중 Qualcomm 축만 채운다. TI/Renesas는 보드·툴체인(TIDL/DRP-AI TVM) 대기.
- **다른 Snapdragon 부품:** AI Hub 디바이스 팜의 다른 자동차 SoC(SA8295P 등)는 미실행 — 필요 시 `scripts/qaihub_device.py "<device>" <slug>`로 동일 절차 확장.

---

## 캐비앗 (절대값 비교 금지)

- 절대 지연은 on-device `estimated_inference_time`(HTP 스케줄러 추정), 배치1 — 다른 단계의 wall-clock/event-timed와 1:1 비교 불가.
- top-1은 200장 서브셋이라 부풀림 가능(1단계 함정 0). **상대 관계**(FP32 충실 vs INT8 외부-QDQ 붕괴 vs AI Hub-native 회복)만 유효.
- "QCS8550 (Proxy)"는 프록시 디바이스, "SA8775P ADP"는 실제 자동차 보드. 디바이스 **간** 절대속도보다 디바이스 **내** INT8/FP32 관계가 논점.

---

## 재현

```bash
# 0) 격리 venv + 인증 (토큰은 repo 밖 ~/.qai_hub/client.ini)
python -m venv ~/qaihub-venv && source ~/qaihub-venv/bin/activate
pip install qai-hub
qai-hub configure --api_token <AI_HUB_TOKEN>

# 1) 외부 QDQ 정리 (발견 #1)
python scripts/clean_valueinfo_for_aihub.py resnet50_int8_qdq.onnx resnet50_int8_qdq_aihub.onnx

# 2) 디바이스별 compile+profile (지연·offload)
python scripts/qaihub_device.py "QCS8550 (Proxy)" qcs8550
python scripts/qaihub_device.py "SA8775P ADP" sa8775p

# 3) on-device 정확도 진단 (발견 #2)
python scripts/qaihub_acc.py                 # 외부 QDQ INT8 200장 → 붕괴
python scripts/qaihub_fp32_acc200.py         # FP32(fp16) 200장 → 충실
python scripts/qaihub_native_quant.py        # AI Hub 자체 quantize → 회복

# 4) 레이아웃 반증 (발견 #4)
python scripts/qaihub_layout_test.py
```
