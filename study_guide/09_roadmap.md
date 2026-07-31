# 9. 12주 학습 로드맵 — 워밍업부터 인프라화까지

> 원본 가이드 매핑: "12주 로드맵(표: 주차 / 할 일 / 산출물)" · 예상 소요: 12주(주당 8~15시간) · 선행 조건: [01_environment_setup.md](01_environment_setup.md) 완료 권장

이 문서는 앞선 8개 단계 문서([01](01_environment_setup.md)~[08](08_capstone.md))와 다음 [함정 문서](10_pitfalls.md)를 **하나의 12주 일정**으로 엮는다. 각 주차는 "무엇을 읽고 → 무엇을 만들고 → 어떻게 완료를 판정하는가"로 구성된다.

> 💡 팁: 로드맵은 이론 정리보다 **일정·산출물·완료 판정**이 핵심이므로, stage-guide-writing 규약 중 "2) 배경 이론" 섹션은 생략한다.

> ⚠️ 정본 버전 스택(2026-07 기준, 이 스터디의 기준값): **CUDA 12.8 / onnx 1.18.0 (IR 11) / onnxruntime-gpu 1.23.2 / TensorRT 10.16.x LTS / ExecuTorch 1.3.x**. 이 조합으로 산출물을 재현한다. ONNX export는 이 스택의 IR 상한 때문에 **opset ≤ 23**을 지킨다. 상위 버전을 쓰면 op 지원·플러그인 API가 달라질 수 있으니, 다르게 갈 거면 [01_environment_setup.md](01_environment_setup.md)의 버전 표를 함께 갱신하라.

---

## 0) 이 단계에서 무엇을·왜 하는가

임베디드 AI 양자화는 배울 것이 많다: 양자화 이론, Transformer 양자화, TensorRT, 그리고 TIDL/QNN/DRP-AI 같은 이종 SoC까지. 아무 순서로나 건드리면 "이론은 아는데 칩에서 안 돌아가는" 상태에 빠지기 쉽다.

이 로드맵은 두 가지를 보장한다.

1. **의존성 순서 보장** — 캘리브레이션도 모르는 채 TensorRT INT8을 돌리면 정확도가 왜 떨어지는지 해석할 수 없다. 이론 → dGPU 실습 → 실제 SoC 순서를 강제한다.
2. **매주 손에 잡히는 산출물** — 주차마다 파일 하나(`layer_sensitivity.csv`, `onnx_export_failures.md` 등)를 남긴다. 이 파일들이 곧 포트폴리오이자 [08_capstone.md](08_capstone.md)의 재료가 된다.

**왜 "일 단위에 가까운" 계획인가.** 학습 로드맵이 실패하는 가장 흔한 이유는 "이번 주에 TensorRT를 한다" 같은 **덩어리 목표**다. 덩어리는 착수 비용이 크고, 막히면 통째로 멈춘다. 그래서 이 문서는 각 주를 **읽을 문서 · 하루 단위 작업 · 산출물 · 완료 판정 · 예상 시간 · 막혔을 때 대응**으로 쪼갠다. 매일 "오늘 끝낼 한 조각"이 보이면 로드맵은 굴러간다.

> 🔴 함정 미리보기: 여기서 만드는 산출물들은 [10_pitfalls.md](10_pitfalls.md)의 5대 함정과 1:1로 연결된다. 예를 들어 `layer_sensitivity.csv`는 "함정 1: 캘리브레이션 데이터가 전부다"를 몸으로 겪는 과정이다. 로드맵을 돌리다 막히면 즉시 함정 문서를 펴라.

---

## 1) 학습 목표 & 완료 체크리스트

이 12주를 마치면 다음을 할 수 있어야 한다.

- [ ] FP32 모델을 PTQ로 INT8 양자화하고, 레이어별 정확도 민감도를 근거로 mixed-precision을 설계할 수 있다.
- [ ] Transformer(ViT/DETR 계열)를 ONNX로 export하고, 실패 op를 진단·우회할 수 있다.
- [ ] TensorRT로 INT8 엔진을 빌드하고 Nsight로 병목을 프로파일링, custom plugin까지 작성할 수 있다.
- [ ] 최소 1종의 실제 SoC(TIDL/QNN/DRP-AI 중)에 배포하고 offload 비율을 해석할 수 있다.
- [ ] 회귀 하네스와 design rules 문서로 팀이 재현 가능한 파이프라인을 만들 수 있다.

> 💡 팁: 이 목록은 "졸업 요건"이다. 매주 완료 판정을 통과하면 자동으로 채워진다.

---

## 2) 선행 지식 지도 — 어느 주차에 무엇이 필요한가

각 주차는 **선행 개념이 준비되어 있어야** 헛돌지 않는다. 자기 배경을 이 표에 대조해, 부족한 칸을 시작 전에 메운다. "필요 수준"은 최소 요구치다.

| 주차 | 반드시 있어야 할 선행 지식 | 필요 수준 | 모자라면 먼저 볼 곳 |
|------|---------------------------|-----------|--------------------|
| 0 (워밍업) | Python, `pip`/venv, 기본 CLI | 낮음 | [01_environment_setup.md](01_environment_setup.md) |
| 1~2 | 선형대수 기초, NumPy, PyTorch inference, 정확도 metric | 중 | [03_quantization_theory.md](03_quantization_theory.md) 2)절 |
| 3~5 | Transformer 구조(attention/LayerNorm), ONNX 개념, opset | 중~상 | [04_transformer_quantization.md](04_transformer_quantization.md) 2)절 |
| 6~8 | 1~2주 산출물, CUDA 개념, C++ 읽기(포인터/메모리), CMake | 상 | [05_tensorrt.md](05_tensorrt.md), [함정 5](10_pitfalls.md#함정-5--c를-피하지-마라) |
| 9~11 | ONNX Runtime EP 개념, 3~5주 산출물, 크로스컴파일 감각 | 상 | [06_multi_soc.md](06_multi_soc.md) |
| 12 | 앞 산출물 전부, Make/CI 기초, 회귀 테스트 개념 | 중 | [07_infrastructure.md](07_infrastructure.md) |

> 💡 팁: **C++는 6주차 전에 "읽을 수 있는" 수준이면 충분하다.** 처음부터 쓸 줄 알 필요는 없다. 8주차에 벤더 템플릿의 `enqueue` 한 함수만 고치는 데서 시작한다([함정 5](10_pitfalls.md#함정-5--c를-피하지-마라)).

> ⚠️ 주의: 1~2주 산출물(`layer_sensitivity.csv`)과 3~5주 산출물(`onnx_export_failures.md`)이 없으면 6주차 이후가 공중에 뜬다. **산출물은 스킵해도 다음 단계 입력이므로 반드시 남긴다.**

---

## 3) 환경·도구 준비

전 주차 공통 환경은 [01_environment_setup.md](01_environment_setup.md)에서 이미 구축했다고 가정한다. 로드맵 시작 전 아래를 한 번에 확인한다.

```bash
# 로드맵 시작 전 환경 스모크 테스트 (Ubuntu 22.04 + RTX)
nvidia-smi                                  # GPU/드라이버 인식 확인 (CUDA 12.8 런타임 호환 드라이버)
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # PyTorch+CUDA
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"  # ORT + EP 목록
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"          # TensorRT 10.16.x 확인
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi        # 컨테이너에서 GPU 접근
```

기대 출력 예:
```
NVIDIA-SMI ...  Driver Version: 5xx.xx  CUDA Version: 12.8
2.x.x+cu128 True
1.23.2 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
TensorRT 10.16.x.x
```

- `torch.cuda.is_available()` → `True`
- ORT provider 목록에 `CUDAExecutionProvider`(및 설치 시 `TensorrtExecutionProvider`)가 보여야 한다.
- `tensorrt.__version__`이 `10.16.x`로 나와야 한다(플러그인 API가 여기 묶여 있다).

하나라도 실패하면 로드맵을 시작하지 말고 [01_environment_setup.md](01_environment_setup.md)로 돌아간다.

> ⚠️ 주의: onnxruntime-gpu는 PyPI 기본 빌드의 CUDA 계열이 릴리스마다 바뀐다. **정본 스택은 CUDA 12.x 빌드(1.23.2)**다. **1.27 이상은 PyPI 기본이 CUDA 13**이라, 그걸 깔면 드라이버/런타임이 어긋나 `TensorrtExecutionProvider`가 안 잡힐 수 있다. provider 목록이 비면 [01_environment_setup.md](01_environment_setup.md)의 설치 명령(`pip install "onnxruntime-gpu<1.27"`)으로 재설치한다. ([ONNX Runtime 설치 문서](https://onnxruntime.ai/docs/install/))

> ⚠️ 주의: 4단계(멀티 SoC)는 실제 보드가 있어야 완료 판정을 통과한다. 보드가 없다면 9~11주는 "컴파일/오프로드 리포트까지"로 범위를 좁히고, 실측 latency는 클라우드 devkit이나 [06_multi_soc.md](06_multi_soc.md)에 안내된 에뮬레이터로 대체한다.

---

## 4) 단계별 실습 — 주차별 상세 계획

각 주차 표기 규칙:
- **읽기**: 그 주에 정독할 단계 문서.
- **일 단위 작업**: 체크박스(끝내야 넘어감). 하루~반나절 조각으로 쪼갬.
- **산출물**: 그 주가 남기는 파일. 리포지토리에 커밋.
- **완료 판정**: 이 조건을 만족하면 다음 주로.
- **예상 시간**: 주당 총량(주당 8~15시간 기준).
- **막혔을 때**: 그 주 특유의 벽과 우회로.

권장 진도: **주당 8~15시간**(이론 30% / 실습 60% / 문서화 10%).

> 💡 팁: 아래 "일 단위 작업"은 **주 5일 × 하루 1.5~3시간** 가정이다. 주말 몰아서 하는 사람은 하루치 2~3개를 묶어 처리하되, 순서(읽기 → 실습 → 문서화)는 유지한다.

---

### 워밍업 (0주, 선택) — 난이도 사다리 Lv.1~4

> 읽기: [02_deployment_ladder.md](02_deployment_ladder.md)
> 목적: "온디바이스 배포"가 무엇인지 손으로 먼저 겪어, 이후 이론이 붕 뜨지 않게 한다.
> 예상 시간: 6~10시간(하루~이틀). 경험자는 스킵.

**Day 1 — Lv.1~2 (클릭 배포 + 로컬 변환)**
- [ ] **Lv.1 — Edge Impulse**: 웹 스튜디오에서 소형 분류 모델을 학습→양자화→배포. 클릭만으로 INT8 배포 경험. (2025년 Qualcomm 인수 후 Dragonwing/Hexagon 타깃 지원)
- [ ] **Lv.2 — LiteRT(구 TFLite)**: 같은/유사 모델을 `.tflite`로 변환, PC에서 인터프리터로 실행.

**Day 2 — Lv.3~4 (표준 런타임 감각)**
- [ ] **Lv.3 — ONNX Runtime**: ONNX로 export 후 `CPUExecutionProvider`/`CUDAExecutionProvider`로 실행, EP 개념 체득.
- [ ] **Lv.4 — ExecuTorch**: PyTorch 모델을 `.pte`로 export해 실행. (ExecuTorch 1.0 GA — 2025-10 PyTorch Conf 발표, 정본 스택 1.3.x. `to_edge`→`to_executorch` 흐름으로 `.pte` 직렬화)
- [ ] 각 단계에서 **동일 입력에 대한 FP32 vs INT8 출력/latency**를 메모.

**산출물**: `fp32_int8_notes.md` — 4개 런타임에서 같은 모델의 FP32/INT8 정확도·지연 비교 메모(표 1개면 충분).

예시 표 형태(채워야 할 골격):
```
| 런타임        | 모델        | FP32 acc | INT8 acc | FP32 ms | INT8 ms | 한 줄 소감 |
|--------------|-------------|----------|----------|---------|---------|-----------|
| ONNX Runtime | resnet18    |   ...    |   ...    |   ...   |   ...   | EP 바꾸니 x배 |
| ExecuTorch   | 동일        |   ...    |   ...    |   ...   |   ...   | .pte 크기 ... |
```

**완료 판정**: 4개 런타임 중 **최소 2개**에서 온디바이스(또는 PC) INT8 추론이 돌고, FP32 대비 정확도 손실/속도 이득을 한 문장으로 설명할 수 있다.

**막혔을 때**
- Edge Impulse 계정/보드 이슈로 Lv.1이 막히면 → **건너뛰고 Lv.3(ONNX Runtime)부터**. 워밍업의 핵심은 Lv.3~4다.
- ExecuTorch export 에러 → 최신 튜토리얼([ExecuTorch 문서](https://pytorch.org/executorch/))의 `to_edge`/`to_executorch` 예제를 **그대로** 복붙해 먼저 성공시킨 뒤 내 모델로 교체.

> 💡 팁: 워밍업은 "느낌 잡기"다. 완벽히 하지 말고 하루~이틀 안에 끝내라. **경험자는 스킵**한다(아래 [학습자 유형별 변형](#학습자-유형별-변형) 참고).

---

### 1~2주 — 양자화 이론 + ResNet PTQ 실습

> 읽기: [03_quantization_theory.md](03_quantization_theory.md)
> 목적: scale/zero-point, symmetric vs asymmetric, per-tensor vs per-channel, calibration의 의미를 **코드로** 이해한다.
> 예상 시간: 주당 10~14시간(1주 이론 5~6h + 2주 실습 8~10h).

#### 1주 (이론 집중, ~5~6h)

- [ ] **Day 1 (1h)** — 양자화 수식(affine mapping, `q = round(x/scale) + zero_point`) 손으로 유도. `scale = (max-min)/(qmax-qmin)` 도 함께.
- [ ] **Day 2 (1h)** — symmetric/asymmetric, per-tensor/per-channel 차이를 표로 정리(아래 골격 채우기).
- [ ] **Day 3 (1.5h)** — PTQ vs QAT, static vs dynamic 개념 정리. (참고: CNN은 static, Transformer/RNN은 dynamic이 일반적 권장 — [10_pitfalls.md](10_pitfalls.md) 함정 2에서 검증)
- [ ] **Day 4 (1.5h)** — 서베이 논문 훑기: Gholami et al. (2021, arXiv:2103.13630), Nagel et al. (2021, arXiv:2106.08295). "요약 5줄"만 자기 노트에 남긴다.
- [ ] **Day 5 (0.5h)** — [10_pitfalls.md](10_pitfalls.md) 함정 1·2를 미리 1회 통독(2주 실습 전 예방주사).

1주 정리 표 골격:
```
|              | per-tensor | per-channel |
|--------------|-----------|-------------|
| symmetric    |   언제?    |    언제?     |
| asymmetric   |   언제?    |    언제?     |
```

#### 2주 (ResNet PTQ 실습, ~8~10h)

- [ ] **Day 1 (2h)** — ResNet-50(torchvision 사전학습) FP32 정확도 측정(baseline). 검증셋·전처리를 **하나의 함수**로 고정(함정 2 예방).
- [ ] **Day 2 (2h)** — ONNX로 export 후 onnxruntime static PTQ로 INT8 양자화(대표성 있는 캘리브 셋 100~500장). `QuantFormat.QDQ`, `per_channel=True` 권장.
- [ ] **Day 3~4 (3~4h)** — **레이어별 민감도 스윕**: 레이어를 하나씩 FP32로 되돌리며(또는 하나씩만 INT8) 정확도 변화를 기록. 자동화 루프로 CSV에 append.
- [ ] **Day 5 (1~2h)** — 민감도 상위 레이어를 FP32로 유지하는 mixed-precision 안 설계 + 그 정확도 재측정.

민감도 스윕 자동화 골격(그대로 확장 가능):
```python
# sensitivity_sweep.py — 레이어를 하나씩 FP32로 유지하며 top-1 변화를 CSV로 append
# 실행: python3 sensitivity_sweep.py  (정본 스택: onnxruntime-gpu 1.23.2, CUDA 12.8)
import csv
# baseline_acc, int8_all_acc 는 사전 측정값
rows = []
for layer in candidate_layers:            # 후보 레이어 목록(민감할 법한 conv/downsample 등)
    acc = eval_with_layer_kept_fp32(layer)  # 해당 레이어만 FP32로 유지한 혼합 정밀 정확도
    rows.append({
        "layer_name": layer,
        "dtype": "fp32-kept",
        "top1_acc": round(acc, 4),
        "delta_vs_fp32": round(acc - baseline_acc, 4),
        "keep_fp32": (acc - int8_all_acc) > 0.005,   # 0.5%p 이상 회복되면 유지 후보
    })
with open("layer_sensitivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer_name","dtype","top1_acc","delta_vs_fp32","keep_fp32"])
    w.writeheader(); w.writerows(rows)
print(f"wrote layer_sensitivity.csv: {len(rows)} rows")
```

기대 출력(예):
```
wrote layer_sensitivity.csv: 18 rows
```

**산출물**: `layer_sensitivity.csv` — 컬럼 예: `layer_name, dtype, top1_acc, delta_vs_fp32, keep_fp32(bool)`.

**완료 판정**: INT8 전체 양자화 정확도, mixed-precision 정확도, FP32 baseline **3개 수치**가 CSV에 있고, "어느 레이어가 왜 민감한가"를 한 문단으로 설명 가능.

**막혔을 때**
- INT8 정확도가 **비정상적으로 낮음(예: 10%p+ 하락)** → 십중팔구 전처리 불일치다. [10_pitfalls.md](10_pitfalls.md) **함정 2**의 `preprocess_parity.py`부터 돌려라. 민감도 스윕은 그 다음이다.
- 특정 조건에서만 낮음 → **함정 1**의 `calib_coverage.py`로 캘리브 range 커버리지 점검.
- 스윕이 너무 느림 → 후보 레이어를 **residual block 단위**로 묶어 굵게 스윕한 뒤, 민감 구간만 레이어 단위로 좁힌다(1시간 룰).

> 🔴 함정 연결: 여기서 캘리브 셋을 대충 고르면 정확도가 이상하게 나온다. [10_pitfalls.md](10_pitfalls.md) **함정 1(캘리브레이션 대표성)**·**함정 2(전처리 불일치)**를 2주차에 반드시 읽어라.

---

### 3~5주 — Transformer 양자화 (SmoothQuant, ViT/DETR)

> 읽기: [04_transformer_quantization.md](04_transformer_quantization.md)
> 목적: activation outlier 때문에 CNN 방식이 통하지 않는 Transformer를, export 실패까지 감당하며 양자화한다.
> 예상 시간: 3주 합계 30~40시간(주당 10~14h).

#### 3주 (export & baseline, ~10~12h)

- [ ] **Day 1 (2h)** — ViT 또는 DETR 계열 모델 선정, FP32 baseline 측정. 입력 전처리 상수(mean/std/resize) 고정.
- [ ] **Day 2~3 (4h)** — PyTorch → ONNX export. `opset≥17` 권장, dynamic axes 설정. **export가 깨지는 op를 전부 기록**(에러 메시지 원문 그대로).
- [ ] **Day 4 (3h)** — `onnxruntime`/Polygraphy로 ONNX 수치 검증(FP32 단계에서 PyTorch와 출력 일치 확인). `polygraphy run model.onnx --onnxrt --atol 1e-3 --rtol 1e-3`.
- [ ] **Day 5 (2h)** — 실패 op를 `onnx_export_failures.md` 초안 표로 정리(op / 에러 / 임시 우회).

#### 4주 (activation outlier 대응, ~10~12h)

- [ ] **Day 1 (2h)** — activation 분포 히스토그램으로 outlier 확인(어느 레이어에서 값이 튀는지).
- [ ] **Day 2~3 (5h)** — SmoothQuant(Xiao et al., 2022, arXiv:2211.10438) 적용 — activation 난이도를 weight로 이전. `alpha` 스윕(0.5 근처).
- [ ] **Day 4~5 (4h)** — per-channel weight quant + 적절한 calibration으로 INT8 양자화, SmoothQuant 전/후 정확도 비교.

#### 5주 (ViT 전용 기법 비교, ~8~10h)

- [ ] **Day 1~3 (5h)** — FQ-ViT / PTQ4ViT / RepQ-ViT 중 1~2개 시도, LayerNorm·Softmax·GELU 처리 차이 비교.
- [ ] **Day 4~5 (3~4h)** — 기법별 정확도/지연 비교표 작성 + `onnx_export_failures.md`에 op 인벤토리(아래) 추가.

op 인벤토리 스니펫(함정 3의 1차 증거):
```python
# op_inventory.py — 모델이 쓰는 op를 세어 백엔드 화이트리스트와 대조
import onnx, collections
m = onnx.load("model.onnx")
c = collections.Counter(n.op_type for n in m.graph.node)
for op, cnt in c.most_common():
    print(f"{op:22s} x{cnt}")
```
기대 출력(예):
```
MatMul                 x48
Add                    x37
LayerNormalization     x25
Softmax                x12
Gather                 x8
...
```

**산출물**: `onnx_export_failures.md` — export/런타임에서 실패한 op, 에러 메시지, 우회법(op 대체·opset 상향·custom)을 표로. (이 파일은 [10_pitfalls.md](10_pitfalls.md) 함정 3의 1차 증거다.)

**완료 판정**: Transformer가 INT8로 export→실행되고, SmoothQuant 적용 전/후 정확도 차이를 수치로 제시. export 실패 op가 최소 1건 이상 문서화됨(없다면 더 어려운 모델로).

**막혔을 때**
- `torch.onnx.export`가 특정 op에서 죽음 → **opset을 한 단계 올려** 재시도 → 그래도 안 되면 그 op를 동등 표현으로 분해(예: 커스텀 `GELU` → `Erf` 기반 표준 GELU)하거나, export 시 `dynamo=True`/`dynamo=False` 경로를 바꿔 시도. 실패 원문은 반드시 문서에 남긴다.
- SmoothQuant 후에도 정확도 회복이 미미 → `alpha` 재스윕(0.4~0.7). LayerNorm/Softmax는 보통 INT8이 아니라 **FP16로 유지**하는 게 안전(함정 3 예방).
- 1시간 룰: 한 op에서 막히면 그 op만 FP16로 두고 **파이프라인을 먼저 완주**한 뒤 되돌아온다.

> ⚠️ 주의: "ONNX export 성공"은 아직 칩 동작을 뜻하지 않는다. 5주차가 [10_pitfalls.md](10_pitfalls.md) **함정 3**의 출발점이다.

---

### 6~8주 — TensorRT 완주 (INT8 / DLA / custom plugin)

> 읽기: [05_tensorrt.md](05_tensorrt.md)
> 목적: dGPU에서 TensorRT INT8 엔진을 완성하고, 프로파일링과 custom plugin까지 밀어붙인다. (정본: TensorRT 10.16.x LTS)
> 예상 시간: 3주 합계 32~42시간(주당 11~14h). 8주(plugin)는 난이도 급상승 — 시간 여유 확보.

#### 6주 (INT8 엔진 빌드, ~10~12h)

- [ ] **Day 1 (2h)** — ONNX → TensorRT 엔진 빌드(`trtexec` 먼저, 그 다음 Python API), FP16 먼저 성공시키고 INT8로.
- [ ] **Day 2~3 (5h)** — INT8 calibrator 구현(entropy/minmax), calibration cache 생성·재사용. **calibrator 입력 전처리 = 추론 전처리** 확인(함정 2).
- [ ] **Day 4~5 (4h)** — Polygraphy로 TRT vs ONNXRuntime 출력 정합성 비교. `polygraphy run model.onnx --trt --onnxrt --atol 1e-2 --rtol 1e-2`.

빠른 시작(복붙):
```bash
# FP16 → INT8 엔진 빌드 (TensorRT 10.16.x). 먼저 FP16이 되는지부터 확인.
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.plan   # FP16 baseline
trtexec --onnx=model.onnx --int8 --calib=calib.cache \
        --saveEngine=model_int8.plan --verbose                  # INT8(캐시 있으면 재사용)
```

#### 7주 (프로파일링 & 최적화, ~10~12h)

- [ ] **Day 1~2 (4h)** — Nsight Systems로 타임라인 분석, layer fusion/병목 파악.
- [ ] **Day 3 (3h)** — DLA(가능 시) 오프로드 실험, GPU-only와 비교. (dGPU 데스크톱에는 DLA가 없다 — Jetson Orin에서만. 없으면 "GPU-only 최적화"로 대체하고 리포트에 명시.)
- [ ] **Day 4~5 (4h)** — 배치·스트림·워크스페이스 튜닝으로 throughput 개선, 수치 기록.

#### 8주 (custom plugin, ~12~14h)

- [ ] **Day 1 (2h)** — 벤더/커뮤니티 **IPluginV3 예제**를 그대로 빌드·등록·실행(내 코드 넣기 전). (TensorRT 10.x에서 IPluginV2는 폐기, `IPluginV3` = `IPluginV3OneCore`+`IPluginV3OneBuild`+`IPluginV3OneRuntime` 다중상속, creator는 `IPluginCreatorV3One`.)
- [ ] **Day 2~4 (6h)** — TRT가 지원하지 않는 op 1개를 골라 **C++ custom plugin** 작성·등록·검증. `enqueue`(연산 본체)만 교체하는 데서 시작.
- [ ] **Day 5 (4h)** — plugin 포함 엔진의 정확도/지연을 baseline과 비교, `compute-sanitizer`로 메모리 오류 점검.

**산출물**: `orin_perf_report.md` — dGPU(및 가능 시 Jetson Orin) 기준 FP32/FP16/INT8 latency·throughput·정확도 표 + Nsight 스크린샷/요약 + custom plugin 결과.

**완료 판정**: INT8 엔진이 목표 정확도 손실(예: top-1 −1~2%p 이내)을 지키며 FP32 대비 명확한 속도 이득을 보이고, custom plugin이 포함된 엔진이 정상 동작.

**막혔을 때**
- INT8 빌드는 되는데 정확도가 무너짐 → calibrator 입력 전처리를 의심(함정 2). calibration cache를 지우고 **추론과 동일한 전처리 배치**로 재생성.
- 특정 op에서 빌드 실패("no implementation"/"unsupported") → `polygraphy inspect capability --with-partitioning model.onnx`로 미지원 op를 특정([함정 3](10_pitfalls.md#함정-3--onnx-export-성공--칩에서-동작)). plugin 대상인지 판단.
- C++ plugin 빌드 벽 → **커뮤니티 최소 예제 저장소를 먼저 성공**시킨다(예: [TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example)). 내 op는 그 위에서 `enqueue`만 교체.

> 🔴 함정 연결: custom plugin·런타임 통합은 전부 C++다. [10_pitfalls.md](10_pitfalls.md) **함정 5(C++를 피하지 마라)**를 8주차에 읽고, 겁먹지 말고 통과하라.

---

### 9~11주 — 멀티 SoC 확장 (TIDL / QNN / DRP-AI)

> 읽기: [06_multi_soc.md](06_multi_soc.md)
> 목적: dGPU 밖의 이종 가속기에서 "지원 op 부분집합 + CPU/ARM fallback"이라는 현실을 직접 겪는다.
> 예상 시간: 3주 합계 28~40시간. 보드 유무에 따라 편차 큼.

> ⚠️ 주의: 3개 타깃 전부에 보드를 갖추기는 현실적으로 어렵다. **최소 1개 타깃을 실측까지**, 나머지는 컴파일/오프로드 리포트까지를 기본 목표로 잡는다.

#### 9주 (TI TIDL, ~9~13h)

- [ ] **Day 1~2 (4h)** — `edgeai-tidl-tools`로 모델 컴파일, "Runtimes Graphviz"로 subgraph 분할 시각화 확인. C7x-MMA offload vs ARM fallback 파악.
- [ ] **Day 3~4 (4h)** — `deny_list`로 특정 op를 강제 ARM 실행(ONNX는 op명 예: `'MaxPool, Add'`, TFLite는 레이어 코드 예: `'1, 2'`)시켜 subgraph 분할이 성능에 미치는 영향 관찰.
- [ ] **Day 5 (2h)** — offload 비율·subgraph 개수를 로그에서 추출해 매트릭스 행 채우기.

#### 10주 (Qualcomm QNN, ~9~13h)

- [ ] **Day 1~2 (4h)** — ONNX Runtime **QNN EP**로 실행, `GetCapability` 기반 subgraph 분할과 CPU fallback 관찰(예: Loop/If 미지원, dynamic shape 미지원).
- [ ] **Day 3 (2h)** — `session.disable_cpu_ep_fallback="1"`로 "전부 HTP에 올라가는가"를 강제 검증(안 올라가면 예외 → 미지원 지점 특정).
- [ ] **Day 4~5 (4h)** — HTP(NPU) offload 비율과 fallback op 목록 기록, 매트릭스 행 채우기.

#### 11주 (Renesas DRP-AI + 매트릭스 종합, ~9~13h)

- [ ] **Day 1~2 (4h)** — DRP-AI TVM으로 RZ/V2H 타깃 컴파일·실행(보드/에뮬레이터).
- [ ] **Day 3~5 (5h)** — **4-target 성능 매트릭스** 완성: dGPU(TensorRT) + TIDL + QNN + DRP-AI. 최소 1개 타깃 실측.

**산출물**: `four_target_matrix.md` — 행=타깃, 열=`op 지원율/offload 비율/latency/정확도/치명적 미지원 op`. 최소 1개 타깃은 실측치, 나머지는 컴파일 결과.

**완료 판정**: 각 타깃에서 "얼마나 가속기에 올라갔고(offload %), 무엇이 CPU/ARM으로 떨어졌는지"를 수치로 제시. fallback이 성능을 어떻게 갉아먹는지 한 문단으로 설명.

**막혔을 때**
- 보드가 없다 → 클라우드 devkit/에뮬레이터로 **컴파일 리포트까지만** 목표를 좁힌다(완료 판정을 "실측 1 + 컴파일 2"로 재정의).
- 가속기에 올렸는데 더 느림 → latency 보기 전에 **offload 비율·subgraph 개수부터**([함정 4](10_pitfalls.md#함정-4--fallback-지옥-subgraph가-쪼개지면-fp32보다-느리다)). `< 50% offload / > 10 subgraph`면 fallback 지옥.
- op가 자꾸 CPU로 떨어짐 → 그 op를 모델 앞/뒤로 몰거나 전·후처리로 빼서 subgraph를 통짜로 만든다.

> 🔴 함정 연결: subgraph가 잘게 쪼개지면 FP32보다 느려질 수 있다. [10_pitfalls.md](10_pitfalls.md) **함정 4(fallback 지옥)**를 9주차부터 옆에 두고 offload 비율부터 확인하라.

---

### 12주 — 인프라화 + 문서화

> 읽기: [07_infrastructure.md](07_infrastructure.md), [08_capstone.md](08_capstone.md)
> 목적: 지금까지의 실습을 팀이 재현 가능한 파이프라인·문서로 승격한다.
> 예상 시간: 10~14h.

- [ ] **Day 1~2 (4h)** — 양자화→컴파일→검증을 스크립트/Makefile/CI로 자동화(`make quantize`, `make build`, `make verify`).
- [ ] **Day 3 (3h)** — **회귀 하네스** 구축: 모델·타깃별 정확도/지연을 자동 측정하고 기준선 대비 회귀를 감지(예: top-1 하락, latency 상승 시 실패로 종료).
- [ ] **Day 4 (2h)** — calibration 셋 버전 관리 및 전처리 파라미터(mean/std) 단일 소스화(함정 2의 팀 차원 예방책).
- [ ] **Day 5 (2~3h)** — `design_rules.md` 작성: "이 팀에서 양자화할 때 지켜야 할 규칙"(대표성 있는 캘리브, 전처리 일치, op 화이트리스트, fallback 임계치 등). 각 규칙 옆에 [10_pitfalls.md](10_pitfalls.md)의 함정 번호를 근거로 링크.

`make regress` 기대 동작(골격):
```bash
# make regress → 전 타깃 회귀 리포트 생성 후, 기준선 밑돌면 비정상 종료
$ make regress
[regress] resnet50  int8  top1=75.9% (base 76.1%, Δ-0.2)  latency=1.8ms (base 1.9)  PASS
[regress] vit_b16   int8  top1=80.1% (base 81.4%, Δ-1.3)  latency=3.2ms            FAIL(>1.0%p)
make: *** [regress] Error 1
```

**산출물**: `design_rules.md` + 회귀 하네스(코드 + README). [08_capstone.md](08_capstone.md)의 캡스톤 산출물과 통합.

**완료 판정**: `make regress`(또는 동등 명령) 한 줄로 전 타깃 회귀 리포트가 생성되고, 새 모델을 넣었을 때 design rules만 따르면 재현되는 상태.

**막혔을 때**
- CI에서 GPU가 안 잡힘 → 로컬 `make regress`를 먼저 통과시키고, CI는 "CPU/컴파일 검증만" 단계적 도입.
- 회귀 임계치 설정이 애매 → 처음엔 관대하게(top-1 −1.0%p, latency +10%) 잡고 점차 조인다.

> 💡 팁: `design_rules.md`의 각 규칙은 [10_pitfalls.md](10_pitfalls.md)의 5대 함정에 대한 "우리 팀의 예방책"으로 쓰면 자연스럽다.

---

## 5) 예시 / 결과 해석

### 주차 → 문서 → 산출물 → 함정 매핑표 (마스터)

이 표 하나로 "지금 몇 주차이고, 무엇을 읽고, 무엇을 남기며, 어디서 막힐지"를 한눈에 본다.

| 주차 | 읽는 문서 | 핵심 할 일 | 산출물 | 완료 판정 요지 | 관련 함정 |
|------|-----------|-----------|--------|----------------|-----------|
| 0 (선택) | [02_deployment_ladder.md](02_deployment_ladder.md) | 난이도 사다리 Lv.1~4 | `fp32_int8_notes.md` | 2개+ 런타임 INT8 성공 | — |
| 1~2 | [03_quantization_theory.md](03_quantization_theory.md) | 이론 + ResNet PTQ | `layer_sensitivity.csv` | FP32/INT8/mixed 3수치 | [1](10_pitfalls.md#함정-1--캘리브레이션-데이터가-전부다), [2](10_pitfalls.md#함정-2--전처리-불일치-meanstd가-어긋난다) |
| 3~5 | [04_transformer_quantization.md](04_transformer_quantization.md) | SmoothQuant, ViT/DETR | `onnx_export_failures.md` | SmoothQuant 전/후 + 실패 op 1건+ | [3](10_pitfalls.md#함정-3--onnx-export-성공--칩에서-동작) |
| 6~8 | [05_tensorrt.md](05_tensorrt.md) | TensorRT INT8/DLA/plugin | `orin_perf_report.md` | INT8 정확도 유지 + plugin 동작 | [5](10_pitfalls.md#함정-5--c를-피하지-마라) |
| 9~11 | [06_multi_soc.md](06_multi_soc.md) | TIDL/QNN/DRP-AI | `four_target_matrix.md` | offload % + fallback 해석 | [4](10_pitfalls.md#함정-4--fallback-지옥-subgraph가-쪼개지면-fp32보다-느리다) |
| 12 | [07](07_infrastructure.md), [08](08_capstone.md) | 인프라화·문서화 | `design_rules.md` + 회귀 하네스 | `make regress` 한 줄 재현 | 전부 |

> 💡 팁: 정본 산출물명은 **고정**이다 — `layer_sensitivity.csv`(1~2주), `onnx_export_failures.md`(3~5주), `design_rules.md`(12주). 파일명을 바꾸면 [10_pitfalls.md](10_pitfalls.md)·[07_infrastructure.md](07_infrastructure.md)의 상호참조가 깨진다.

### 완료 판정 자가 점검 (매주 금요일 5분)

- [ ] 이번 주 산출물 파일이 리포지토리에 커밋되었다.
- [ ] baseline(FP32) 대비 수치 비교가 최소 1개 있다.
- [ ] 막힌 부분을 `블로커.md`(개인 메모)에 1줄이라도 적었다.
- [ ] 다음 주 읽을 문서를 미리 열어봤다.
- [ ] (해당 주) 연결된 함정 문서의 해당 절을 1회 읽었다.

---

## 학습자 유형별 변형

한 사람이 모두 초심자는 아니다. 배경에 따라 진입점을 조정한다.

| 유형 | 조정 | 절약되는 기간 |
|------|------|---------------|
| **완전 입문** | 0주 워밍업 Lv.1부터 전부. 1~2주 이론에 시간 더 배분. | — (풀 코스) |
| **PyTorch/ML 경험, 배포 처음** | 워밍업은 **Lv.3(ONNX Runtime)부터**. Lv.1~2 스킵. | ~0.5주 |
| **ONNX/양자화 경험자** | 워밍업 **전체 스킵**, 1주 이론은 속독 후 2주 실습으로. | ~1.5주 |
| **TensorRT 실무 경험자** | 워밍업·1~2주 스킵 가능. 6~8주는 custom plugin/DLA 등 **약점만** 골라. 대신 3~5주(Transformer)·9~11주(멀티 SoC)에 집중. | ~3주 |
| **특정 SoC(TI/Qualcomm/Renesas) 종사자** | 자기 SoC는 9~11주에서 심화, 나머지 타깃은 비교용으로 가볍게. TensorRT는 필수 통과(면접 단골). | 상황별 |
| **논문/연구 중심(양자화 알고리즘 관심)** | 1주 이론 + 4주 SmoothQuant/ViT 기법에 무게. 다만 **6~8주 TensorRT·9~11주 SoC를 스킵하지 말 것** — "칩에서 안 도는 알고리즘"은 임베디드에서 무의미. 산출물로 이론을 증명. | 상황별 |
| **면접이 코앞(2~3주 남음)** | 아래 [압축 코스](#압축-코스-6주-경험자용)를 다시 절반으로: 1~2주 실습 + TensorRT INT8 + 멀티 SoC 1종 컴파일. 산출물 3개만이라도 확보. | — |

각 유형의 구체 진입점:

- **TensorRT 경험자**: 6주 INT8 빌드는 복습이니 반나절로 끝내고, **8주 custom plugin(IPluginV3)에 시간을 몰아라.** 대부분의 TRT 실무자도 plugin은 안 짜봤다 — 여기가 차별점. 그리고 3~5주 Transformer 양자화(activation outlier)는 CNN만 해본 사람에게 새 영역이다.
- **임베디드 처음(펌웨어/HW 배경)**: 1~2주 이론에 시간을 넉넉히. 대신 C++/CMake/크로스컴파일은 강점이므로 8주·9~11주가 상대적으로 수월하다. 약점은 "정확도 metric·데이터 파이프라인"이니 함정 1·2를 두 번 읽어라.
- **논문 중심**: SmoothQuant를 "적용"에서 그치지 말고 `alpha` 스윕 곡선을 그려 `onnx_export_failures.md` 옆에 첨부. 단, **반드시 TensorRT로 한 번은 칩(엔진)까지 내려라.** 이론↔배포 격차가 [10_pitfalls.md](10_pitfalls.md) 함정 3의 핵심이다.

> 💡 팁: 스킵하더라도 **각 단계의 산출물 파일은 남겨라.** 포트폴리오는 "무엇을 아는가"가 아니라 "무엇을 만들었는가"로 평가된다.

### 압축 코스 (6주, 경험자용)

시간이 없고 배경이 있다면 아래 6주 코스를 쓴다. 각 주는 "핵심 작업 + 반드시 남길 산출물 + 완료 판정"까지 명시한다.

| 주 | 핵심 작업 | 산출물 | 완료 판정 | 예상 시간 |
|----|-----------|--------|-----------|-----------|
| 1 | 이론 속독(반나절) + ResNet PTQ + 민감도 스윕 | `layer_sensitivity.csv` | FP32/INT8/mixed 3수치 + 민감 레이어 설명 | 12~15h |
| 2 | Transformer export + SmoothQuant(전/후) | `onnx_export_failures.md` | 실패 op 1건+ 문서화, SmoothQuant Δacc | 12~15h |
| 3 | TensorRT FP16→INT8 + Nsight 병목 1개 | (perf 노트 초안) | INT8 정확도 유지 + 속도 이득 수치 | 12~15h |
| 4 | TensorRT custom plugin(IPluginV3, C++) | `orin_perf_report.md` | plugin 포함 엔진 정상 + reference 대비 오차 내 | 14~16h |
| 5 | 멀티 SoC 1종 실측(TIDL/QNN/DRP-AI 중) | `four_target_matrix.md` | offload % + fallback 해석 | 12~15h |
| 6 | 회귀 하네스 + design rules | `design_rules.md` + 하네스 | `make regress` 한 줄 재현 | 10~14h |

압축 코스 운용 규칙:
- **깊이 대신 완결.** 각 주 "완료 판정"만 통과하면 다음 주로. 완벽주의 금지(1시간 룰 엄수).
- **4주(plugin)와 5주(SoC 실측)가 병목.** 여기서 하루씩 밀릴 각오를 하고, 앞 3주를 타이트하게.
- 압축 코스도 **산출물 6개는 그대로**다. 이게 포트폴리오의 골격.

> ⚠️ 주의: 압축 코스는 "경험자"용이다. 완전 입문이 6주로 밀면 함정 1·2에서 조용히 무너진 채로 진도만 나간다. 자신 없으면 12주 풀 코스가 결국 더 빠르다.

---

## 6) 막혔을 때 대응 (Troubleshooting)

로드맵은 자주 막힌다. 막힘의 종류별 대응:

| 증상 | 먼저 볼 곳 | 대응 |
|------|-----------|------|
| INT8 정확도가 이상하게 낮음 | [10_pitfalls.md](10_pitfalls.md) 함정 1·2 | 캘리브 대표성/전처리 mean·std 일치부터 확인(`preprocess_parity.py`→`calib_coverage.py`) |
| ONNX export가 깨짐 | [04](04_transformer_quantization.md), [10](10_pitfalls.md) 함정 3 | opset 상향, 문제 op 대체/분해, `onnx_export_failures.md`에 기록 |
| 칩에서 export는 됐는데 안 돌거나 느림 | [10_pitfalls.md](10_pitfalls.md) 함정 3·4 | `polygraphy inspect capability`로 미지원 op 특정 → offload 비율·subgraph 개수 확인 |
| TensorRT 엔진 빌드 실패 | [05_tensorrt.md](05_tensorrt.md) | Polygraphy로 op 단위 격리, plugin 필요 여부 판단 |
| C++ 벽에 부딪힘 | [10_pitfalls.md](10_pitfalls.md) 함정 5 | 커뮤니티 최소 예제부터. plugin 템플릿 복사 후 `enqueue`만 교체 |
| provider 목록에 TRT/CUDA EP가 없음 | [01_environment_setup.md](01_environment_setup.md) | onnxruntime-gpu **CUDA 12.x 빌드**로 재설치(정본 스택) |

**막힘 대응 3원칙**
1. **1시간 룰**: 한 문제에 1시간 이상 막히면 우회로(다른 op/다른 기법/mixed-precision)를 먼저 확보하고, 막힌 지점은 `블로커.md`에 남긴다. 로드맵을 멈추지 않는다.
2. **격리 후 재현**: 전체 파이프라인이 아니라 **한 레이어/한 op**로 최소 재현 케이스를 만든다(Polygraphy·단일 op ONNX).
3. **증거 남기기**: 실패는 삭제하지 말고 산출물 문서에 기록. 이 기록이 [10_pitfalls.md](10_pitfalls.md)의 실전 근거이자 면접 이야깃거리다.

> ⚠️ 주의: "완료 판정을 못 넘었는데 다음 주로 넘어가기"는 눈덩이가 된다. 대신 **범위를 좁혀** 판정을 통과시켜라(예: 3개 타깃 → 1개 타깃 실측 + 2개 컴파일).

---

## 7) 산출물(Deliverables)

12주가 끝나면 아래가 리포지토리에 있어야 한다(각 단계 문서의 산출물 총합).

- [ ] `fp32_int8_notes.md` (0주, 선택)
- [ ] `layer_sensitivity.csv` (1~2주)
- [ ] `onnx_export_failures.md` (3~5주)
- [ ] `orin_perf_report.md` (6~8주)
- [ ] `four_target_matrix.md` (9~11주)
- [ ] `design_rules.md` + 회귀 하네스 코드 (12주)
- [ ] `README.md` — 위 산출물을 링크한 인덱스(자기 리포지토리 기준)

> 💡 팁: 이 산출물 세트 자체가 임베디드 AI 취업 포트폴리오의 골격이다. [08_capstone.md](08_capstone.md)의 캡스톤과 겹치도록 설계했으니 중복 작업을 피하라.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [ONNX Runtime 설치](https://onnxruntime.ai/docs/install/) — onnxruntime-gpu의 CUDA 빌드 계열·설치 명령(정본 스택 CUDA 12.x 확인).
- [ONNX Runtime 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — static/dynamic PTQ, EP별 대칭성 요구사항.
- [NVIDIA TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/) — INT8 calibration, custom plugin. (정본 10.16.x LTS)
- [TensorRT 10.16.0 릴리스 노트](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html) — IPluginV2 제거·IPluginV3 전환, 지원 매트릭스.
- [ExecuTorch](https://pytorch.org/executorch/) — 워밍업 Lv.4. (정본 1.3.x, `to_edge`→`to_executorch`→`.pte`)
- [Edge Impulse Docs](https://docs.edgeimpulse.com/) — 워밍업 Lv.1.
- [LiteRT](https://ai.google.dev/edge/litert) — 워밍업 Lv.2.
- [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) — TIDL 컴파일·`deny_list`·Runtimes Graphviz.
- [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — QNN offload/fallback, `disable_cpu_ep_fallback`.
- [DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) — Renesas RZ/V2H.

### 논문
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295
- Xiao et al. (2022), *SmoothQuant*, arXiv:2211.10438

> 각 단계의 상세 참고문헌은 해당 단계 문서([03](03_quantization_theory.md)~[07](07_infrastructure.md))를 참조.

---

## 9) 다음 단계

로드맵을 따라가다 막히는 지점은 대부분 5개 함정 중 하나다. 다음 문서에서 각 함정을 증상→원인→예방→디버깅으로 파헤친다.

→ 다음: [10. 함정 5개](10_pitfalls.md)
← 이전: [8. 캡스톤 프로젝트](08_capstone.md)
