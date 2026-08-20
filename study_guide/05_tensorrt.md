# 3. TensorRT로 첫 완주

> 원본 가이드 매핑: "3단계 — TensorRT로 첫 완주 (2~3주)" · 예상 소요: 2~3주 · 선행 조건: [0단계 환경 준비](01_environment_setup.md) 완료(CUDA/드라이버/TensorRT 설치), [1단계 양자화 이론](03_quantization_theory.md)의 `layer_sensitivity.csv` 산출물
>
> 실행 환경: **Ubuntu 22.04 LTS + NVIDIA RTX GPU(dGPU)**. Orin 보드가 있으면 실측이 이상적이고, 없으면 RTX에서 90%를 진행할 수 있다(DLA·실제 보드 latency만 Orin 필요). 다른 SoC와도 개념은 90% 동일하다.

---

## 0) 이 단계에서 무엇을·왜 하는가

이론(1·2단계)에서 "양자화가 왜 정확도를 떨어뜨리는가"를 배웠다면, 이 단계는 **실제 하드웨어 위에서 엔진을 빌드하고 latency와 정확도를 숫자로 재는 첫 완주**다. 임베디드 AI 엔지니어가 실무에서 가장 많이 만지는 도구가 TensorRT이고, 면접에서도 "trtexec로 뭘 봤냐 / DLA fallback을 어떻게 잡았냐 / custom plugin을 짜봤냐"를 묻는다.

핵심 메시지 3가지:

1. **profiling의 절반은 `trtexec`에서 끝난다.** 엔진 하나 빌드하고 per-layer 프로파일을 뽑는 것만으로 병목의 대부분이 보인다. 그런데 대부분의 사람은 `trtexec`가 뱉는 숫자(Throughput / Latency / Enqueue Time / GPU Compute Time / H2D·D2H)를 **읽을 줄 모른다.** 이 문서는 그 출력을 한 줄씩 해석하는 데 지면을 크게 쓴다.
2. **정확도 디버깅은 `polygraphy`가 필수품이다.** "INT8로 바꿨더니 mAP가 떨어졌다"를 레이어 단위로 어디서 깨졌는지 찾는다. 특히 `polygraphy debug precision`으로 **"몇 번째 레이어까지 고정밀로 돌리면 정확도가 회복되는가"를 이분탐색(bisect)** 하는 절차가 실무의 핵심이다.
3. **DLA와 custom plugin이 진짜 차별점이다.** GPU-only 빌드는 누구나 한다. DLA fallback을 0으로 만들고, 미지원 op를 위한 plugin을 직접 짜는 능력이 임베디드 특화 역량이다.

> 💡 **왜 GPU에서 INT8을 하려면 결국 TensorRT인가 (1단계 실측):** [1단계](03_quantization_theory.md)에서 만든 INT8 QDQ ONNX를 ONNX Runtime **CUDA EP**로 돌리면 FP32보다 **오히려 느리다**(1.33 → 1.81 ms). CUDA EP에는 QDQ INT8 conv 커널이 없어 DQ로 되돌려 FP로 계산하기 때문이다. 같은 모델을 **TensorRT로 제대로 태우면 0.51 ms**(FP32 0.96 ms 대비 **1.86×**)가 나온다. 즉 **GPU에서 INT8 이득을 보려면 TensorRT 엔진 빌드(또는 TensorRT EP)가 사실상 필수**다 — 이 단계가 존재하는 이유다. 단, **아무 QDQ 모델이나 되는 건 아니다**(2.2.1). *실측: RTX 3060 · ResNet18 · batch=1 · 워밍업 20 + 60회 p50 · ORT 1.23.2. 출처: [1단계 실행 로그 8장](../logs/stage1_quantization_log.html) + [재실행 보고서 10절](../logs/stage1_real_imagenet_report.html).*

> ⚠️ **버전 경계 (2026-07 기준, 반드시 먼저 읽을 것)**
> TensorRT는 2026년 상반기 **11.x**가 나오면서 API가 크게 바뀌었다. 이 단계는 **두 세계**를 다루되, **정본(주 경로)은 10.x LTS**다.
> - **TensorRT 10.x (주 경로)**: JetPack(Orin)에 탑재되는 계열이자 이 스터디의 정본. `trtexec --int8 --fp16 --best`, `IInt8EntropyCalibrator2`, implicit quantization이 **그대로 동작**한다(단, 10.1부터 캘리브레이터는 deprecated 경고). **그리고 DLA는 10.x에서만 지원된다** — 11.0/11.1에는 DLA 지원이 아예 없다([Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)). Orin으로 넘어갈 것이라면 10.x가 사실상 강제된다.
> - **TensorRT 11.x (참고)**: RTX 데스크톱 최신. **모든 네트워크가 strongly-typed**여서 `--fp16/--int8/--best/--bf16/--fp8/--int4/--calib` 플래그가 **전부 제거**됐고, 정밀도는 빌드가 아니라 **모델에 미리 심어서**(ModelOpt AutoCast / QDQ) 결정한다. DLA도 빠졌다.
> 자신이 어느 버전인지 `trtexec --version`으로 먼저 확인하고, 각 실습의 **버전 태그**를 따라가라.
>
> 📌 **이 스터디의 정본(canonical) 경로 = TensorRT 10.x LTS 고정**([0단계 결정](01_environment_setup.md) 참조). 아래 실습의 **주 경로는 10.16.x LTS**(`trtexec --int8 --fp16` 등이 그대로 동작)이고, **11.x 변경사항은 "앞으로 이렇게 바뀐다"는 참고**로 각 실습의 `2-B`/버전 주의 콜아웃에 병기한다. Orin JetPack도 10.x 계열이라 10.x로 배워두면 데스크톱·보드가 일치한다.
> - 왜 하필 **10.16.x LTS**인가: 10.16은 **Long-Term Support 라인**으로, 여기서 deprecated된 API도 **2027-03까지 유지**된다([TensorRT 10.16 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html)). 즉 학습·포트폴리오 기간 내내 명령/코드가 안 깨진다. `--int8 --fp16`이 동작하고 DLA가 살아있는 **가장 안정적인 축**이다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] `trtexec --version`으로 내 TensorRT 메이저 버전(10.x인지 11.x인지) 확인 — 정본은 **10.16.x LTS**
- [ ] YOLO(YOLO11/YOLOv8)를 ONNX로 export하고 `trtexec`로 엔진을 빌드
- [ ] 동일 모델을 **FP32 / FP16 / INT8** 3가지로 빌드하고 **latency + mAP 3점 비교표** 작성
- [ ] `trtexec` 출력에서 **Throughput / Latency(min·median·p90·p99) / Enqueue Time / GPU Compute Time / H2D·D2H**를 각각 무슨 뜻인지 말로 설명
- [ ] `--dumpProfile`의 **per-layer 표**를 읽고 상위 5개 레이어가 시간의 몇 %인지 계산
- [ ] `--timingCacheFile`로 빌드 시간을 단축하고, `--saveEngine`/`--loadEngine`으로 엔진 직렬화·역직렬화
- [ ] `polygraphy inspect model`로 그래프 구조 확인, `polygraphy run --trt --onnxrt --atol/--rtol`로 프레임워크 간 출력 비교
- [ ] `polygraphy debug precision --mode bisect`로 **"몇 번째 레이어까지 FP16로 올리면 정확도가 회복되는가"** 를 이분탐색
- [ ] `IInt8EntropyCalibrator2`(10.x) **또는** ModelOpt PTQ(11.x)로 INT8 캘리브레이션 직접 수행 + **캘리브 캐시 저장/재사용**
- [ ] TensorRT Model Optimizer로 **QDQ ONNX** export → `trtexec` 빌드까지 1회 연계
- [ ] [1단계](03_quantization_theory.md)에서 만든 ORT QDQ ONNX를 **TRT 호환 설정(`QInt8` 대칭)으로 다시 뽑아** 파싱 성공 + INT8 tactic 확인 → **무음 폴백 배제**(2.2.1 / 실습 4-(C))
- [ ] (Orin 보드 있으면) 동일 모델을 **GPU-only / DLA-only / 하이브리드** 3가지로 빌드
- [ ] (Orin 보드 있으면) **DLA fallback 레이어 목록 추출 → 유발 op 치환 → fallback 0** 도전
- [ ] TensorRT **custom plugin(IPluginV3)** C++ 골격을 이해하고 최소 빌드(identity plugin)
- [ ] Nsight Systems로 커널 단위 타임라인 1회 캡처하고 병목 유형(전송 vs 연산)을 판정
- [ ] 산출물 `latency_accuracy_matrix.csv`, `polygraphy_diff.md`, (보드 시) `dla_fallback.md` 작성

---

## 2) 배경 이론 / 개념

### 2.1 TensorRT가 하는 일

ONNX/프레임워크 그래프를 받아 → **레이어 융합(fusion)**, **정밀도 선택**, **커널 auto-tuning(tactic 선택)**, **메모리 재사용**을 거쳐 특정 GPU/DLA에 최적화된 **엔진(.engine/.plan)** 을 만든다.

직관적으로 왜 이렇게 하나:
- **fusion**: `Conv → Bias → ReLU`를 하나의 커널로 합치면, 중간 결과를 글로벌 메모리에 썼다 읽는 왕복이 사라진다. 임베디드에서 병목은 연산량(FLOPs)이 아니라 **메모리 대역폭**인 경우가 많아서, fusion이 곧 속도다.
- **tactic 선택(auto-tuning)**: 같은 Conv라도 GPU/입력크기에 따라 최적 CUDA 커널이 다르다. TensorRT는 후보 커널들을 **실제로 타이밍해 보고** 가장 빠른 것을 고른다. 그래서 **빌드가 느리다**(그 대신 `--timingCacheFile`로 캐시하면 다음 빌드가 빨라진다).
- **정밀도 선택**: FP32/FP16/INT8 중 레이어별로 어떤 걸 쓸지 결정한다. 10.x는 `--fp16/--int8/--best` 플래그로, 11.x는 모델에 심긴 정보로 결정한다.

엔진은 **빌드한 GPU 아키텍처·TensorRT 버전에 종속적**이라 **타깃에서 빌드**하는 게 원칙이다(RTX에서 빌드한 엔진은 Orin에서 안 돌아간다). 이식이 꼭 필요하면 `--versionCompatible`(버전 호환 엔진) 같은 옵션이 있으나, 아키텍처 자체(SM 버전)가 다르면 여전히 재빌드가 안전하다.

### 2.2 Implicit vs Explicit quantization (버전 경계의 핵심)

| 구분 | Implicit (10.x) | Explicit / QDQ (10.x·11.x) |
|------|-----------------|-----------------------------|
| 스케일 결정 | 빌드 시 캘리브레이터가 결정 | 모델 그래프의 Q/DQ 노드가 이미 보유 |
| 대표 API | `IInt8EntropyCalibrator2` | ModelOpt / pytorch-quantization의 QDQ export |
| 정밀도 제어 | `trtexec --int8 --fp16` 플래그 | strong typing(모델이 곧 정밀도) |
| 11.x 지원 | **제거됨** | **표준 방식** |
| 정확도 | 자동이라 편하지만 최적 아님 | QAT와 결합 시 최고 |

**왜 이 방향으로 통일됐나:** implicit은 "빌드할 때 캘리브레이션 데이터를 흘려서 스케일을 정한다"라서, **엔진을 빌드하는 순간까지 정밀도가 확정되지 않는다.** 이는 재현성·디버깅·부분 정밀도 지정을 어렵게 한다. explicit(QDQ)은 **모델 그래프 안에 Q/DQ(Quantize/DeQuantize) 노드가 스케일을 들고 있어**, 엔진 빌더는 그저 그 정보를 읽어 인접 레이어에 fuse할 뿐이다. "정밀도는 모델의 속성"이 되어 재현성이 좋다.

NVIDIA는 10.1부터 implicit(entropy calibrator)을 deprecated 처리했고, 11.0에서 관련 `trtexec` 플래그를 없앴다. **엔진 빌드가 아니라 "모델에 Q/DQ를 심는 단계"에서 정밀도가 결정되는 방향**으로 통일됐다. 정본(10.16.x LTS)에서는 **둘 다 동작**하므로, implicit로 감을 잡고 explicit/QDQ로 실무 감각을 익히는 순서가 좋다.

> 💡 **QDQ가 엔진에서 어떻게 INT8이 되나:** ONNX의 `... → DQ → Conv → Q → ...` 패턴을 TensorRT가 보면, DQ와 Q를 Conv 안으로 **접어(fuse)** Conv를 **INT8 텐서코어 커널**로 실행한다. 그래서 QDQ 모델을 빌드하면 `--int8` 플래그 없이도(11.x) 또는 있어도(10.x) INT8 실행이 나온다. `--dumpProfile`에서 해당 Conv의 tactic 이름에 `int8`/`imma`(integer matrix-multiply-accumulate)가 뜨면 성공이다.

#### 2.2.1 🔴 외부에서 만들어 온 QDQ ONNX가 TRT에서 그대로 빌드된다는 보장은 없다 (1단계 실측)

"QDQ면 어디서 만들든 TensorRT가 읽는다"는 **틀렸다.** TRT ONNX 파서는 Q/DQ에 **두 가지 하드 제약**을 건다([onnx-tensorrt 10.16-GA `operators.md`](https://github.com/onnx/onnx-tensorrt/blob/10.16-GA/docs/operators.md)):

| 노드 | 파서가 받는 양자화 타입 | 제약 |
|------|------------------------|------|
| `DequantizeLinear` | **INT8 / FP8 / FP4 / INT4** (INT32 없음) | `x_zero_point`가 **반드시 0** |
| `QuantizeLinear` | 입력 FP32 / FP16 / BF16 | `y_zero_point`가 **반드시 0** |

즉 TensorRT는 **대칭(symmetric) 양자화만** 받는다. 이 두 제약은 **10.x와 11.x가 동일**하다(같은 문서의 `main`=11.1 브랜치도 문구가 같다) — 버전을 올려도 해결되지 않는다는 뜻이다.

가장 흔하게 걸리는 사례가 **1단계에서 ONNX Runtime `quantize_static`으로 만든 QDQ ONNX**다. ORT의 x86 권장 설정(`activation_type=QUInt8` + `ActivationSymmetric=False`)을 그대로 들고 오면 파서가 그래프를 **통째로 거부**한다:

```text
[ERROR] ITensor::getDimensions: Error Code 4: API Usage Error
        (conv1.weight_bias_DequantizeLinear: input has type Int32 but must have type
         FP8, FP4, Int4, or Int8. In checkType at nodeBase.cpp:455)
[ERROR] ModelImporter.cpp:149: ERROR: In function parseNode:
        [6] Invalid Node - conv1.weight_bias_DequantizeLinear
[ERROR] [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
```

에러는 **두 종류**가 뜨지만, **고쳐야 할 것은 하나뿐이다.**

- ① **bias INT32 DQ.** ORT는 `QuantizeBias`가 기본 `True`라 bias를 **INT32로 양자화하고 DQ를 붙인다**(ORT 1.23.2 `quantize.py` docstring: *"Default is True which quantizes floating-point biases and it solely inserts a DeQuantizeLinear node"*). 위 표대로 TRT 파서의 DQ 지원 타입에 INT32는 없다. ResNet18에서는 conv/fc **21개 전부**에서 났다.
- ② **non-zero zero-point.** `QUInt8` 비대칭이라 activation의 `zero_point ≠ 0` → `shiftIsAllZeros` assertion. **← 이쪽이 유일한 하드 블로커다**(근거는 아래 절제 실험).

🔴 **무서운 건 에러가 아니라 그 다음이다.** ORT의 TensorRT EP는 파싱에 실패해도 **예외를 던지지 않고 조용히 폴백**한다. 노드를 하나도 못 가져가면서 파티셔닝 오버헤드까지 얹혀 **FP32보다 3배 느려진다** — 즉 **"INT8로 바꿨는데 왜 느리지?"의 정체가 정확도 문제가 아니라 파싱 실패**인 경우가 있다.

| 설정 | TRT p50 | vs FP32(TRT) | top-1 (50k) | Δ vs FP32 | McNemar vs FP32 |
|------|---------|--------------|-------------|-----------|-----------------|
| FP32 | 0.96 ms | 1.00× (기준) | 68.74% | — | — |
| INT8 `QUInt8` 비대칭 (1단계 4.3 기본) | **3.06 ms** | **0.31× (느려짐)** | 68.62% | −0.12%p | p=0.061 n.s. |
| INT8 `QInt8` 대칭 (`ActivationSymmetric=True`) | **0.51 ms** | **1.86×** | 68.33% | −0.41%p | **p=5.0e-8 유의** |
| INT8 대칭 + `QuantizeBias=False` | **0.51 ms** | **1.86×** | 68.33% | −0.41%p | **p=5.0e-8 유의** |

> 실측 환경: **RTX 3060 · ResNet18 · batch=1 · 워밍업 20 + 60회 p50 · ORT 1.23.2 TensorRT EP**. **top-1은 ImageNet val 50,000장 전량**을 동일 전처리 캐시(`squash`)로 평가한 값이고, 짝지어진 표본이라 McNemar를 쓴다. 다른 모델·해상도에 이 배수를 그대로 옮기지 말 것. 원 로그: [1단계 실행 로그 8장](../logs/stage1_quantization_log.html)(메커니즘) + [재실행 보고서 9~10절](../logs/stage1_real_imagenet_report.html)(위 수치).

**같은 INT8인데 설정 하나로 6배(3.06 → 0.51 ms) 차이**가 난다. 다만 **그 대가는 무료가 아니다.**

🔴 **정정 — 대칭 강제의 정확도 대가는 유의하다.** 1단계 1차 실행(큐레이션 1,000장)에서는 −0.4%p / p=0.5224로 "유의하지 않음"이라고 적었다. **50,000장으로 다시 재니 유의하다**: 비대칭 68.62% → 대칭 68.33%, **−0.29%p, McNemar p=9.2e-5**(FP32 기준으로는 −0.41%p, p=5.0e-8). 1,000장에는 0.3%p를 잡을 검정력이 없었던 것이다([10_pitfalls.md](10_pitfalls.md) 함정 0).

메커니즘은 분명하다 — zero-point를 0으로 묶으면 **post-ReLU처럼 한쪽(≥0)만 쓰는 분포에서 표현 구간의 절반을 버린다.** 비대칭 `QUInt8`은 zero-point를 분포 하단(실측 `[0, 173]`)으로 옮겨 256단계를 전부 양의 영역에 쓰지만, 대칭 `QInt8`은 −128~127 중 실질적으로 0~127만 쓰게 된다. **INT8이 사실상 INT7이 되는 것**이고, 그 대가가 0.29%p다.

그래도 **1.86× 속도를 0.29%p로 사는 거래**이므로 대개 남는 장사다(폴백 상태로 두면 오히려 3배 느리다). 판단 기준:

| 정확도 예산 | 판단 |
|------------|------|
| > 1%p 여유 | 고민 없이 대칭으로 뒤집는다. |
| 0.3~1%p | 대칭으로 가되 **−0.3%p를 예산에 계상**한다. 여유가 빠듯하면 부분 FP16 혼합(2.4)으로 되찾는다. |
| < 0.3%p | 대칭 PTQ만으로는 예산을 넘길 수 있다. **QAT**([03 6절](03_quantization_theory.md))로 대칭 제약 아래에서 재학습하는 것이 정공법이다. |

즉 **"어차피 유의하지 않으니 공짜"라는 서술은 틀렸다.** 속도를 위해 정확도를 얼마 지불하는지 알고 뒤집어야 한다.

**그런데 "설정 하나"란 정확히 무엇인가 — ①인가 ②인가?** 에러가 두 종류 뜨니 "원인이 둘"로 읽히지만, **2×2 절제 실험**으로 갈라 보면 파싱 성공/실패를 가르는 변수는 **activation zero-point가 0인가** 하나뿐이다.

| case | 설정 | INT32 bias DQ | act zero-point | TRT p50 | CUDA p50 | top-1 (50k) | 판정 |
|------|------|---------------|----------------|---------|----------|-------------|------|
| A | `QUInt8` 비대칭 + bias 양자화 O | **21개** | `[0, 173]` | 3.06 ms | 1.81 ms | 68.62% | 🔴 폴백(실패) |
| B | `QUInt8` 비대칭 + bias 양자화 **X** | **0개** | `[0, 173]` | 2.97 ms | 1.80 ms | 68.62% | 🔴 **폴백(실패)** |
| C | `QInt8` 대칭 + bias 양자화 O | **21개** | `[0, 0]` | **0.51 ms** | 2.11 ms | 68.33% | ✅ **성공** |
| D | `QInt8` 대칭 + bias 양자화 X | 0개 | `[0, 0]` | 0.51 ms | 1.99 ms | 68.33% | ✅ 성공 |

> 판정 기준: **TRT p50 < CUDA p50 × 0.8** 이면 TRT가 실제로 그래프를 가져간 것. 폴백이면 TRT가 CUDA보다 **오히려 느려지는데**(파티셔닝 오버헤드), 그 역전 자체가 무음 폴백의 지표다 — 절대 시간을 몰라도 **두 EP를 나란히 재면 폴백을 잡아낼 수 있다**는 뜻이라, 실무에서 쓸 만한 판정법이다. 단 **GPU를 단독 점유하고 재야** 한다(다른 작업과 병렬로 재면 두 값이 함께 출렁여 판정이 오염된다).
> 실측 환경: RTX 3060 · ResNet18 · batch=1 · **워밍업 20 + 60회 p50** · 같은 캘리브 200장/MinMax/per-channel. top-1은 ImageNet val 50,000장.

- **B가 실패** → INT32 bias DQ를 **없애도 소용없다.**
- **C가 성공** → INT32 bias DQ가 **있어도 문제없다.**
- 따라서 ①은 **②로 Q/DQ 융합이 깨진 뒤 홀로 남은 bias DQ가 내는 2차 증상**이다. 파서는 `DQ(int32 bias) → Conv` 패턴을 통째로 접을 때는 INT32 bias를 받아들이고, 융합이 깨져 DQ가 홀로 남을 때만 타입 검사에 걸린다. **에러 메시지 개수를 원인 개수로 세면 안 된다**는 교훈이기도 하다 — 첫 줄에 뜬 에러(①)를 붙잡고 `QuantizeBias`만 만졌다면 case B에 갇혀 하루를 날렸을 것이다.
- **두 축이 정확도에서도 완전히 분리된다.** `top-1` 열을 보라 — 값이 **대칭/비대칭 축만 따라 움직이고** `QuantizeBias`에는 전혀 반응하지 않는다. 50,000장에서 **C와 D의 예측은 한 장도 다르지 않았고**(0장 불일치), A와 B는 1장만 달랐다. 즉 **`QuantizeBias`는 정확도에 영향이 없다** — 파싱·그래프 정리 관점에서만 논할 옵션이다.
- 실무 처방: **`activation_type=QInt8` + `ActivationSymmetric=True`, 이 하나면 된다.** `QuantizeBias=False`는 **필수가 아니라 선택**이고(속도 0.51 ms 동일, 정확도 완전 동일), 대칭 전환의 정확도 대가 −0.29%p는 **`QuantizeBias`로 되찾을 수 없다.** 구체적인 재양자화 코드와 검증 절차는 **실습 4-(C)** 에 있다. **단 이 "선택"은 ORT TensorRT EP 경로 한정이다** — 아래 3단계 실측(직접 파서 절제)에서 보듯, 같은 QDQ ONNX를 `polygraphy`/`trtexec`의 **직접 파서**로 빌드하면 INT32 bias DQ가 독립 하드 블로커가 되어 `QuantizeBias=False`가 **필수**로 바뀐다.

> 💡 **ModelOpt QDQ는 왜 이 문제가 없나:** ModelOpt는 *"generates new ONNX models with QDQ nodes **following TensorRT rules**"* 라고 문서에 명시돼 있고, 산출물을 곧바로 `trtexec --onnx=quant.onnx`로 빌드하는 것을 표준 경로로 제시한다([ModelOpt ONNX Quantization](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)). 즉 **삽입 규칙 자체가 위 표의 제약(대칭·허용 dtype)에 맞춰져 있다.** 반대로 ORT `quantize_static`은 **기본 타깃이 x86 CPU**라, 같은 QDQ 포맷이라도 TRT가 못 받는 형태를 만들어 낸다. **"QDQ = 공용어"는 포맷 얘기지 호환 보장이 아니다.** 실습 4가 "권장 경로"인 실질적 이유가 이것이다.
>
> ⚠️ 확인 필요: ModelOpt가 **bias를 어떻게 처리하는지**(FP로 남기는지, 다른 방식으로 접는지)는 공개 문서에 명시가 없다(2026-08 기준 [ONNX Quantization 가이드](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)·[qdq_utils API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.quantization.qdq_utils.html) 모두 언급 없음). 자신의 산출물에서 직접 세어 보는 게 확실하다 — `python -c "import onnx; m=onnx.load('x.quant.onnx'); print(sum(1 for n in m.graph.node if n.op_type=='DequantizeLinear' and 'bias' in n.name))"`.

> 🔴 **3단계 실측 정밀화 — 직접 TRT 파서엔 하드 블로커가 "하나"가 아니라 "둘"이다 (2026-08-17 · RTX 3080 · TensorRT 10.16.1.11 · [리포트](../logs/stage3_tensorrt_report.html) · 로그 원문 [`parser_constraints.md`](../experiments/stage3_tensorrt/parser_constraints.md)):** 위 2×2 절제는 **ORT TensorRT EP** 경로(RTX 3060·ResNet18)에서 잰 것이고, 그 경로에선 "zero-point≠0 하나뿐"이 맞다. 그런데 같은 종류의 QDQ ONNX를 **polygraphy/`trtexec`의 직접 ONNX 파서**(정본 TRT 10.16.1.11)에 물리면 축이 하나 더 드러난다. ResNet50으로 5케이스를 절제했다([`t03`](../experiments/stage3_tensorrt/t03_parser_constraints.py)):

| 케이스 | 구성 | parse | build | 실패 지점·블로커 |
|---|---|:---:|:---:|---|
| A | 대칭 `QInt8` · bias 양자화 off · stem 제외 | ✅ | ✅ | — (**정본 처방**) |
| B | 대칭 `QInt8` · **bias INT32 양자화 on** · stem 제외 | ❌ | — | **파서**: INT32 bias DQ 거부(대칭 zp=0인데도) |
| C | **비대칭 `QUInt8`** · stem 제외 | ❌ | — | **파서**: `shiftIsAllZeros`(zp≠0, act zp 21.3%가 비영) |
| D | 대칭 `QInt8` · bias off · **stem(conv1) 포함** | ✅ | ❌ | **빌더**: stem 융합블록 INT8 커널 부재(Error Code 10) |
| E | 2단계 DETR `detr_int8.onnx`(ORT 산출) 실제 | ❌ | — | **파서**: zp≠0 + INT32 bias 동시(act zp 83.1%가 비영) |

- **B가 핵심 정밀화 지점이다.** §2.2.1(ORT-EP)에선 "INT32 bias DQ는 ②의 2차 증상, 대칭이면 남아 있어도 무해"(case C 성공)였는데, **직접 파서에선 대칭(zp=0)이어도 INT32 bias DQ 하나만으로 파싱이 죽는다** — `IDequantizeLayer::setPrecision: … A DequantizeLayer can only run in DataType::kINT8, kFP8, kFP4, or kINT4 precision` → `INVALID_NODE: Invalid Node - fc.bias_DequantizeLinear`. 즉 **직접 파서에선 INT32 bias DQ가 독립 하드 블로커**다.
- **왜 갈리나(경로 차이 — 반전이 아니라 병기).** ORT TensorRT EP는 그래프를 파서에 넘기기 **전에** 자체 QDQ 정리를 거쳐 `DQ(int32 bias)→Conv` 패턴을 흡수한다 — 그래서 EP 경로의 파서는 이 DQ를 애초에 안 본다(§2.2.1 case C가 통과한 이유). 반면 polygraphy/`trtexec`는 ONNX를 **날것으로** 파서에 던지므로 홀로 남은 INT32 bias DQ를 그대로 만나 거부한다. **§2.2.1의 ORT-EP 결론은 그 경로 안에서 그대로 유효하며, 이 절은 그것을 뒤집는 게 아니라 "직접 파서 경로"를 병기해 정밀화한다.** 실무 규칙: **ORT `quantize_static`으로 QDQ를 만들어 직접 파서로 빌드하려면 대칭 `QInt8`에 더해 `QuantizeBias=False`도 필수**다(둘 다 있어야 case A). ModelOpt QDQ(실습 4)를 쓰면 애초에 TRT 규칙대로 삽입돼 이 문제를 우회한다.
- **파서 실패와 빌더 실패는 별개 축이다(D).** stem conv1(3ch 7×7 Conv+relu+maxpool 융합)은 QDQ가 정상이라 **파싱은 통과**하지만, 그 융합 패턴의 INT8 커널이 없어 **빌드 단계**에서 죽는다 — `Could not find any implementation for node … + /conv1/Conv + PWN(/relu/Relu) + /maxpool/MaxPool`. 처방은 QDQ 수정이 아니라 **그 노드만 INT8 대상에서 제외**(`nodes_to_exclude=["/conv1/Conv"]`)해 TRT가 나머지 conv는 INT8로 융합하게 두는 것이다.
- **E는 실모델 확인.** 2단계에서 만든 DETR INT8 ONNX(ORT 비대칭 기본)를 직접 파서에 물리면 **첫 DQ 노드에서** zp≠0으로 즉사한다(`/model/Tile_output_0_DequantizeLinear` … `shiftIsAllZeros(zeroPoint)`) — 2단계에서 관측한 "ORT 기본 QDQ를 TRT가 못 받는다"의 파서 레벨 근거다.

### 2.3 DLA (Deep Learning Accelerator)

Orin/Xavier에 있는 **고정 기능 가속기**. 특징:

- **FP16·INT8만** 지원(FP32 불가).
- 지원 op가 GPU보다 **훨씬 좁다.** CNN(Conv/Deconv/Pool/Activation/ElementWise/Scale/LRN/Concat/Resize/Slice/Shuffle/Reduce 등)은 대체로 되지만, **Softmax는 Orin에서만**(Xavier 불가), **MatMul/attention/transformer 계열은 통째로 DLA에서 안 돈다**([DLA Supported Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html)). 게다가 **동적 shape 불가** — profile의 min=opt=max여야 한다.
- 미지원 레이어는 `--allowGPUFallback`으로 GPU에 떨어뜨린다 → 이걸 **GPU-DLA 파티셔닝**이라 부른다.
- 🔴 **함정:** fallback이 많으면 DLA↔GPU 사이 **메모리 복사 오버헤드**가 커져서 **오히려 느려진다.** 경험칙으로 연산의 **20~30% 이상이 GPU로 fallback되면 DLA를 쓰는 이점이 사라진다.** 그래서 "fallback 0 만들기"가 목표가 된다.
- ⚠️ **DLA는 TensorRT 10.x 전용 개념이다.** 11.0/11.1은 DLA를 지원하지 않으므로, DLA 실습은 **10.x(JetPack) + Orin/Xavier 보드**에서만 의미가 있다.

> **🔬 실측 (온디바이스): Jetson AGX Orin에서 위 DLA 이론을 실제로 돌려봄** — JetPack 6.2.1 · TRT **10.3.0.30** · **2× NVDLA v2** · ResNet50(3단계와 동일 ONNX). 이론 한 줄("20~30% 폴백이면 이점 소멸")의 실제 위치와, 이론엔 없던 규칙 하나("DLA는 INT8 전용기")를 실측으로 채운다.
>
> - **실 trtexec가 실존한다(3단계 반전).** 정본 pip 휠(`tensorrt-cu12`)엔 `trtexec` 실행파일이 **없어** 3·5단계는 polygraphy Python API로 우회했다. Jetson엔 JetPack 동봉 **`/usr/src/tensorrt/bin/trtexec`**(배너 `v100300`)가 실존 → 위 DLA 명령을 **그대로** 관통한다. 단 **정정**: 이 트림의 빌드전용 플래그는 `--buildOnly`가 아니라 **`--skipInference`**(전자는 `[E] Unknown option`으로 exit 1).
> - **폴백은 이론의 "20~30%" 경계 훨씬 아래였다.** ResNet50는 **DLA-후보 2/2 오프로드**(ForeignNode 2개 = conv 백본 `/conv1/Conv‥/layer4.2/relu_2` **120층** + `/fc/Gemm`+bias), GPU 폴백은 **compute 2층뿐** — `GlobalAveragePool`(REDUCE: 빌드 경고 `DLA cores do not support AVG Reduce operation`)과 flatten(SHUFFLE). 폴백이 최소라 DLA가 제대로 이긴다(이론의 붕괴 조건에 안 걸림).
> - **DLA는 INT8 전용기다(이론에 없던 규칙).** 레이어 배치가 **INT8·FP16 완전 동일**한데 **DLA FP16이 DLA INT8보다 13.87× 느리다**(17.73 vs **1.28 ms**) → 원인은 배치가 아니라 순수 NVDLA v2 데이터패스(INT8 MAC 처리량 ≫ FP16). **DLA에 올릴 거면 반드시 INT8.** DLA FP16은 iGPU FP32(1.94 ms)보다도 9배 느린 최악 조합이다. (iGPU와 정반대 — 작은 Ampere iGPU는 INT8≈FP16으로 0.984.)
> - **오프로드가 수치로 증명된다.** `tegrastats`의 GR3D(GPU-3D) 사용률이 iGPU INT8 **95%** → DLA **3~16%**로 붕괴 → 연산이 GPU가 아니라 DLA에서 실제로 돈다(GPU가 비므로 다른 모델/헤드 병렬 가능 — **단 조건부**, 아래 후속 실측이 정량화). 단 **DLA는 GR3D에 거의 안 잡히므로** 전력 하네스의 부하검출은 GR3D가 아니라 전력 임계(idle×1.20)로 자기보정해야 한다.
> - **DLA INT8 = 성능/와트 챔피언.** **51.29 inf/s/W**로 iGPU INT8(33.16)의 **1.547×**, 전력은 **0.511×(절반, 15.19 W)**. 지연만 1.262× 느릴 뿐 → 전력·GPU-여유가 목적이면 DLA INT8, 순수 최저지연이면 iGPU INT8.
> - **후속 실측 — iGPU∥DLA는 공짜 병렬이 아니다 + 전력이 리더를 뒤집는다.** (a) `trtexec`를 iGPU·DLA에 **동시**에 띄우면 합산 처리량이 이상적 합의 **60.8%뿐** — DLA가 27%로 붕괴(지연 1.28→4.75 ms). 범인은 위 "폴백 2층"(GlobalAveragePool+flatten)이 **포화된 iGPU 큐 뒤에 직렬화**되는 것 → **진짜 병렬은 DLA 폴백 0층일 때만** 성립. 반대로 **DLA0+DLA1(GPU 유휴)은 87.0%로 깨끗이 확장·성능/와트 66.07로 전 구성 최고**. (b) `nvpmodel`로 **MAXN→50W**로 조이면 iGPU는 **−29.4%** 급락하나 DLA는 **−2.9%**뿐 → **50W에서 DLA가 iGPU를 +8.8% 추월**(성능/와트는 전 예산에서 DLA ×1.47~1.55 우월). 임베디드·자동차처럼 전력이 조여 있을수록 DLA 이득↑.
>
> 📄 전체 실측·SVG·판정: [`../logs/stage3_jetson_orin_ondevice_report.html`](../logs/stage3_jetson_orin_ondevice_report.html)(solo 5-엔진) · [`../logs/stage3_jetson_orin_concurrent_power_report.html`](../logs/stage3_jetson_orin_concurrent_power_report.html)(동시부하·전력스윕 후속) · 데이터·스크립트·제약: [`../experiments/stage3_tensorrt/jetson_ondevice/`](../experiments/stage3_tensorrt/jetson_ondevice/) · 멀티-SoC 플랫폼 관점(iGPU vs DLA 성능/와트): [4단계 §2-3](06_multi_soc.md)
> **캐비앗:** 지연 = event-timed·batch1·MAXN → 타 단계(polygraphy·wall-clock)와 1:1 비교 불가, 상대만. **DLA INT8은 `trtexec --int8` 암묵 캘리브(자동 레인지)라 지연·전력만 유효, 정확도 미주장**(정확도는 명시적 QDQ인 iGPU INT8·3단계 RTX·4단계 CPU 프록시에서 확립). 전력은 보드 총합(캐리어 오버헤드 포함). 후속 실측의 30W/15W는 재부팅 게이트라 미측정(값이 50W와 동일 → 리포트에 회색 행으로 병기), GPU-폴백 직렬화 결론은 모델 의존.

### 2.4 이 단계에서 도구가 겹치지 않게: trtexec / polygraphy / ModelOpt / Nsight 역할 분담

| 도구 | 한 문장 역할 | 주 산출물 |
|------|-------------|-----------|
| `trtexec` | 엔진 빌드 + latency/throughput 벤치마크 + per-layer 프로파일 | `.engine`, latency 표 |
| `polygraphy` | **정확도** 디버깅(프레임워크 간 출력 비교, 어느 레이어가 깨지는지 이분탐색) | mismatch 레이어 목록 |
| ModelOpt | 모델에 **Q/DQ를 심는**(PTQ/QAT) 단계 → QDQ ONNX | `*.quant.onnx` |
| Nsight Systems | 커널·메모리 복사 **타임라인**(전송 vs 연산 병목 판정) | `*.nsys-rep` |

"속도는 trtexec/Nsight, 정확도는 polygraphy, 정밀도 심기는 ModelOpt"로 외우면 헷갈리지 않는다.

---

## 3) 환경·도구 준비

### 3.1 TensorRT 설치 확인 (0단계에서 설치 가정)

```bash
# TensorRT 버전 확인 — 정본은 10.16.x LTS. 10.x인지 11.x인지가 이 단계 전체의 분기점
trtexec --version                      # trtexec가 PATH에 없으면 아래 경로 시도
/usr/src/tensorrt/bin/trtexec --version
dpkg -l | grep -i tensorrt             # deb 설치 시 패키지 버전
python3 -c "import tensorrt as trt; print(trt.__version__)"   # Python 바인딩 버전
```

예상 출력(정본 10.16.x 예):

```text
[I] TensorRT version: 10.16.0
...
&&&& PASSED TensorRT.trtexec [TensorRT v101600] ...
```

- `TensorRT v101600` = 10.16.0. 앞 두 자리(`10`)가 메이저 → **`--int8/--fp16/--best`가 살아있는 10.x 계열**임을 뜻한다.
- `v110000` 이상이면 11.x → 정밀도 플래그가 없으니 실습 2-B/4로 간다.

> ⚠️ 확인 필요: 위에서 나온 버전을 웹의 [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html)와 대조해 EOL/알려진 버그를 확인하라. 2026-07 기준 **정본은 10.16.x LTS**(10.16에서 deprecated된 API는 2027-03까지 유지)이며, 11.x는 최신이나 정밀도 플래그·DLA가 없다. Orin JetPack은 통상 **TensorRT 10.x**를 탑재한다.

> 🔴 **3단계 실측 헤드라인 (2026-08-17 · RTX 3080 · TensorRT 10.16.1.11 pip 휠 · [리포트](../logs/stage3_tensorrt_report.html)):** `pip install tensorrt-cu12`로 깐 **정본 휠에는 `trtexec` 실행파일이 아예 없다** — `trtexec_on_PATH=null`, 파일시스템 0건([`t01_env.json`](../experiments/stage3_tensorrt/t01_env.json)). 위 `trtexec --version`은 **deb/JetPack 설치 기준**이고, 0단계를 pip 경로로 따라온 독자는 이 단계의 거의 모든 `trtexec …` 명령을 그대로는 못 쓴다. 실행 가능한 대체 경로는 **polygraphy 0.50.3의 Python API**(`network_from_onnx_path`로 파싱 → `engine_from_network(…, config=CreateConfig(int8=…, fp16=…))`으로 빌드)이며, 이 문서의 3점 빌드·파서 절제·implicit 실측은 전부 그 경로로 냈다([`experiments/stage3_tensorrt/`](../experiments/stage3_tensorrt/README.md)). 버전 확인만큼은 pip 휠에서도 되는 **`python3 -c "import tensorrt as trt; print(trt.__version__)"`**(실측 `10.16.1.11`)로 한다. 아래 절과 실습의 `trtexec …` 블록은 **deb/JetPack 독자용 정본으로 유지**하고, pip 독자를 위한 polygraphy 등가 실측을 각 지점에 병기한다.

### 3.2 부가 도구 설치

```bash
# ultralytics (YOLO export용), onnx, onnxruntime-gpu
#    🔴 '-U'로 onnx/onnxruntime-gpu를 통째로 올리지 마라. 0단계에서 못 박은 정본
#       (onnx 1.18.0 / ORT 1.23.2)이 최신(onnx 1.22.0=IR 13, ORT 1.28.0=CUDA 13)으로
#       올라가면서 ONNX 로드와 CUDA EP가 동시에 깨진다. 핀을 그대로 유지한다.
pip install -U ultralytics
pip install "onnx==1.18.0" "onnxruntime-gpu<1.27"
#    🔴 onnxscript 필수: torch 2.11의 torch.onnx.export는 기본이 dynamo=True이고 그 경로가
#       onnxscript를 요구한다(실습 4의 QAT export 등). 없으면 "No module named 'onnxscript'".
#       onnxscript 0.7.1은 onnx>=1.17만 요구하므로 위 1.18.0 핀은 그대로 유지된다.
pip install onnxscript

# Polygraphy (정확도/레이어 디버깅) — NVIDIA index 병용
python3 -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
polygraphy --version                   # 설치 확인 (정본 0.50.3, 0단계 실측)

# TensorRT Model Optimizer (QDQ export / PTQ / QAT)
pip install -U nvidia-modelopt[all]    # import 이름은 modelopt (modelopt.onnx / modelopt.torch)
python3 -c "import modelopt; print(modelopt.__version__)"

# pycuda (직접 캘리브레이터 구현 시 버퍼 관리에 사용) — 선택
pip install pycuda
```

각 패키지가 이 단계에서 하는 일:
- `ultralytics`: YOLO `.pt` → ONNX export, `.engine` 검증(mAP)까지 한 CLI로.
- `polygraphy`: `inspect`/`run`/`debug`/`data` 서브커맨드. 정확도 디버깅 전용.
- `nvidia-modelopt`: `modelopt.onnx.quantization`(ONNX PTQ), `modelopt.torch.quantization`(QAT). QDQ ONNX를 만든다.
- `pycuda`: 파이썬 캘리브레이터에서 디바이스 버퍼 할당/H2D 복사(`mem_alloc`, `memcpy_htod`).

> 💡 팁: 버전 충돌이 잦다. `pip install` 후 `python3 -c "import tensorrt, onnxruntime, modelopt"`가 한 번에 통과하는지 확인하고, 안 되면 컨테이너(`nvcr.io/nvidia/tensorrt`)로 격리하는 편이 빠르다. 컨테이너 태그는 정본에 맞춰 **10.16.x 계열**을 고른다.

> 🔴 **실측 함정 — `modelopt.onnx`는 import 자체가 실패한다(2026-08-17, RTX 3080):** 위 `nvidia-modelopt`가 깔려 `import modelopt`(버전 문자열)·`modelopt.torch`(2단계 §4.4 SmoothQuant에서 쓴 경로)는 **정상**인데, `import modelopt.onnx.quantization`은 **`Please install optional \`\`[onnx]\`\` dependencies.`로 죽는다**([`t01_env.json`](../experiments/stage3_tensorrt/t01_env.json)의 `modelopt_onnx.importable=false`) — `[onnx]` 엑스트라(예: `onnxslim`)가 채워지지 않은 것이다. **처방**: ONNX PTQ 경로가 필요하면 `pip install "nvidia-modelopt[onnx]"`(또는 최소 `pip install onnxslim`)로 보충한 뒤 `python3 -c "import modelopt.onnx.quantization"`가 통과하는지 재확인한다. 이 단계의 INT8 실측은 그 대신 **ORT `quantize_static` QDQ + polygraphy 빌드**(실습1·4-(C) 경로)로 냈고, `modelopt.torch` PTQ는 2단계에서 이미 검증했으므로 여기선 ONNX-측 공백만 메우면 된다.

---

## 4) 단계별 실습

### 실습 1 — YOLO detector: FP32 → FP16 → INT8, latency/mAP 3점 비교

#### (1) YOLO를 ONNX로 export

```bash
# YOLO11n(또는 yolov8n) COCO 사전학습 가중치를 ONNX로 export
# opset>=13 권장(INT8 QDQ 요구). simplify로 그래프 정리.
yolo export model=yolo11n.pt format=onnx opset=13 imgsz=640 simplify=True
# 결과: yolo11n.onnx  (동적 배치가 필요하면 dynamic=True 추가)
```

```bash
# ONNX 그래프 구조/입출력 이름·shape 확인 (polygraphy inspect)
polygraphy inspect model yolo11n.onnx --show layers | head -n 40
```

`polygraphy inspect model` 출력에서 봐야 할 것:
- 맨 위 **입력/출력 텐서 이름과 shape**(예: `images [1,3,640,640] float32`, `output0 [1,84,8400]`). 뒤에서 `polygraphy run`이나 캘리브레이터가 이 이름을 정확히 써야 한다.
- `--show layers`는 노드 나열, `--show layers attrs`까지 주면 각 노드의 attribute(커널 크기·stride 등)도 나온다. **DLA 친화성**(2.3)을 판단할 때 이 attribute를 본다.

#### (2-A) TensorRT 10.x (정본) — 플래그로 3점 빌드

```bash
# FP32 (기준선). --saveEngine으로 엔진 저장, --dumpProfile로 per-layer 시간
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n_fp32.engine \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed \
        --timingCacheFile=yolo11n.timing.cache

# FP16
trtexec --onnx=yolo11n.onnx --fp16 --saveEngine=yolo11n_fp16.engine \
        --dumpProfile --separateProfileRun \
        --timingCacheFile=yolo11n.timing.cache

# INT8 (implicit). 캘리브레이터 없이 --int8만 주면 임의 스케일 → 정확도 안 봄, latency만 참고
# 정확한 INT8은 실습 3의 캘리브레이터 또는 QDQ 모델을 쓴다.
trtexec --onnx=yolo11n.onnx --int8 --saveEngine=yolo11n_int8.engine \
        --dumpProfile --separateProfileRun \
        --timingCacheFile=yolo11n.timing.cache

# --best = FP32/FP16/INT8 모든 정밀도를 켜고 레이어별로 가장 빠른 것을 자동 선택(정확도 보장 아님)
trtexec --onnx=yolo11n.onnx --best --saveEngine=yolo11n_best.engine \
        --timingCacheFile=yolo11n.timing.cache
```

각 플래그의 의미:
- `--saveEngine=<file>` : 빌드된 엔진을 파일로 **직렬화**(serialize). 이후 `--loadEngine`으로 재빌드 없이 로드한다.
- `--dumpProfile` : 벤치 실행 후 **레이어별 실행 시간 표**를 출력(아래 (3)에서 해석).
- `--separateProfileRun` : 벤치마크 실행과 프로파일 실행을 **분리**. 프로파일링 훅이 latency를 오염시키지 않게 하여, latency 수치와 per-layer 프로파일을 **둘 다 신뢰**할 수 있게 한다.
- `--profilingVerbosity=detailed` : 레이어 이름/tactic까지 상세히. `layer_names_only`(기본)보다 정보가 많아 병목 레이어 특정에 유리.
- `--timingCacheFile=<file>` : tactic 타이밍 결과를 캐시. **첫 빌드는 캐시를 만들고, 이후 FP16/INT8/best 빌드가 이 캐시를 재사용**해 빌드가 눈에 띄게 빨라진다(같은 그래프의 반복 타이밍을 건너뛴다).
- `--best` : FP32/FP16/INT8 **모든 정밀도를 후보로 켜서** 레이어별 최속 정밀도 선택. 정확도는 보장하지 않으니 **속도 상한을 보는 용도**.

> 🔴 함정: `--int8`만 주고 캘리브레이션 데이터를 안 주면 TensorRT가 **동적 범위를 임의(또는 미지정)로 잡아** mAP가 크게 깨진다. latency 비교용으로만 쓰고, **정확도 숫자는 실습 3의 캘리브레이션 엔진으로** 낸다.

> 💡 **빌드 옵션 심화 — workspace / sparsity**
> - `--memPoolSize=workspace:2048` : 빌더가 tactic 탐색·중간 버퍼에 쓸 수 있는 **workspace 상한을 2048 MB**로. 너무 작으면 빠른 tactic을 못 골라 `no implementation` 에러나 느린 엔진이 된다. 크게 주면 더 빠른 tactic을 찾을 여지가 생긴다(메모리 여유 내에서). DLA용은 `--memPoolSize=dlaSRAM:1,dlaLocalDRAM:...`처럼 pool별로 준다.
> - `--sparsity=enable` : 가중치가 **2:4 구조적 sparsity** 패턴이면 sparse 텐서코어를 사용. 단, 아무 모델이나 빨라지지 않는다 — 가중치가 실제로 2:4 패턴이어야 한다. `--sparsity=force`는 "패턴이라 가정하고" 켜지만, 그건 먼저 `polygraphy surgeon prune`으로 가중치를 2:4로 다시 쓴 뒤 `--sparsity=enable`을 쓰는 게 정석이다(force는 정확도를 깨뜨릴 수 있음).

#### 엔진 직렬화/역직렬화 (한 번 빌드, 여러 번 로드)

```bash
# 저장한 엔진을 재빌드 없이 로드해서 벤치만 다시 (빌드 5~10분 → 로드 1초)
trtexec --loadEngine=yolo11n_int8.engine \
        --dumpProfile --separateProfileRun
```

- `.engine`(=`.plan`)은 **직렬화된 바이너리**다. Python에서는 `runtime.deserialize_cuda_engine(f.read())`로 로드한다.
- 🔴 **직렬화 엔진의 이식성:** 엔진은 (a)빌드한 **GPU 아키텍처(SM 버전)**, (b)**TensorRT 버전**, (c)일부 빌드 플래그에 종속적이다. RTX(예: SM 8.9)에서 만든 엔진은 Orin(SM 8.7)에서 로드되지 않는다 → **타깃에서 재빌드**가 원칙. 그래서 CI에서 "엔진을 아티팩트로 굽는" 잡은 **타깃 GPU별로** 돌린다.

> 🟩 **3단계 실측 — trtexec 없이 polygraphy로 3점 빌드 (2026-08-17 · RTX 3080 · ResNet50 `IMAGENET1K_V1` · batch=1 · [리포트](../logs/stage3_tensorrt_report.html) · [`t02`](../experiments/stage3_tensorrt/t02_latency_3point.py)):** 위 `trtexec --onnx=… {--fp16|--int8}` 3점을 pip 스택(trtexec 부재)에선 polygraphy Python API로 등가 재현한다 — `network_from_onnx_path(onnx)`로 파싱 → `engine_from_network(…, config=CreateConfig(fp16=…, int8=…))`으로 빌드. YOLO 대신 **1단계 연속선상의 ResNet50**으로 export 리스크를 없앴고, 지연은 워밍업 후 p50, 정확도는 ImageNet val 5,000장.

| 구성 (polygraphy `CreateConfig`) | p50 (ms) | vs FP32 | top-1 | 엔진 | INT8 커널줄 |
|---|---|:---:|---|---|:---:|
| FP32 | 1.6615 | ×1.00 | 76.88% | 122.3 MiB | 0 |
| FP16 (`fp16=True`) | **0.8459** | **×1.96** | 76.88%(동일) | 49.2 MiB | 0 |
| INT8+FP16 (`int8=True, fp16=True`, QDQ) | **0.7843** | **×2.12** | 76.36%(−0.52%p) | 25.5 MiB | **74** |

- **FP16은 사실상 공짜**(top-1 완전 동일, ×1.96). **INT8은 ×2.12에 −0.52%p**이고, 엔진 레이어 덤프에 **INT8 커널이 74줄** 잡혀 Q/DQ가 실제 INT8 GEMM으로 융합됐음이 확인된다(무음 폴백이면 이 줄 수가 0인 것과 대비된다 — 2.2.1의 CUDA-EP 교차 판정과 같은 신호).
- **왜 "INT8+FP16"인가:** 순수 `int8=True`만 주면 stem conv1 융합블록의 INT8 커널 부재로 **빌드가 실패**한다(2.2.1 실측 case D). 실전 배포 구성이자 안전한 기본값은 **`int8=True, fp16=True`**(= trtexec `--int8 --fp16`) — INT8 커널이 없는 층만 FP16으로 자동 폴백시켜 빌드를 통과시키면서 나머지는 INT8로 융합한다.
- top-1 서브셋(5,000장) 76.88%는 공개 FP32 76.13%보다 부풀려짐(1단계 함정 0). 논점은 **배수와 정확도 순위**(FP32=FP16 ≥ INT8)이지 절대값이 아니다.

#### (2-B) TensorRT 11.x (참고) — strongly-typed(플래그 제거됨)

> ⚠️ **11.x에서는 `--fp16/--int8/--best/--bf16/--fp8/--int4/--calib`가 전부 제거됐다.** 정밀도는 빌드가 아니라 **모델에 미리 심어서** 결정한다. **이 스터디의 정본은 10.16.x LTS이므로 이 절은 "미래 대비 참고"다.**
> - **FP16/혼합정밀도**: ModelOpt **AutoCast**로 FP16/BF16 캐스트를 넣은 ONNX를 만든 뒤 빌드.
> - **INT8**: 실습 4의 ModelOpt PTQ로 **QDQ ONNX**를 만든 뒤 빌드.

```bash
# 11.x: 정밀도 플래그 없이 빌드 (엔진 정밀도는 입력 ONNX가 결정)
trtexec --onnx=yolo11n.onnx           --saveEngine=yolo11n_fp32.engine --dumpProfile --separateProfileRun
trtexec --onnx=yolo11n_autocast.onnx  --saveEngine=yolo11n_fp16.engine --dumpProfile --separateProfileRun
trtexec --onnx=yolo11n.quant.onnx     --saveEngine=yolo11n_int8.engine --dumpProfile --separateProfileRun
```

> 📌 `--dumpProfile --separateProfileRun --profilingVerbosity=detailed`와 `--saveEngine/--loadEngine/--timingCacheFile/--memPoolSize`는 **10.x·11.x 공통으로 살아있다.** 사라진 건 **정밀도 지정 플래그**뿐이다. 즉 벤치·프로파일·직렬화 워크플로우는 두 버전에서 동일하게 배워두면 된다.

#### (3) trtexec 출력 전체 해석 (이 단계의 핵심)

`trtexec`를 돌리면 마지막에 **Performance summary** 블록이 나온다. 정본 10.16.x의 대표 형태(값은 예시):

```text
[I] === Performance summary ===
[I] Throughput: 512.34 qps
[I] Latency: min = 1.71 ms, max = 2.98 ms, mean = 1.94 ms, median = 1.90 ms,
             percentile(90%) = 2.11 ms, percentile(95%) = 2.24 ms, percentile(99%) = 2.55 ms
[I] Enqueue Time: min = 0.42 ms, max = 1.10 ms, mean = 0.55 ms, median = 0.50 ms, ...
[I] H2D Latency: min = 0.18 ms, mean = 0.21 ms, ...
[I] GPU Compute Time: min = 1.30 ms, max = 2.40 ms, mean = 1.52 ms, median = 1.48 ms,
                      percentile(99%) = 2.05 ms
[I] D2H Latency: min = 0.05 ms, mean = 0.06 ms, ...
[I] Total Host Walltime: 3.0012 s
[I] Total GPU Compute Time: 3.0405 s
```

각 줄이 **정확히 무엇을 재는가** (NVIDIA [Command-Line Programs](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/reference/command-line-programs.html) 정의):

| 항목 | 정의 | 무엇을 알려주나 |
|------|------|----------------|
| **Throughput (qps)** | (총 추론 수) / **Total Host Walltime**. | 실제 처리량. **1/GPU_Compute_mean보다 크게 낮으면 GPU가 놀고 있다**(호스트 오버헤드/전송 병목). |
| **Latency (Host Latency)** | **H2D + GPU Compute + D2H**. 한 번 추론의 end-to-end. | 전처리 복사까지 포함한 "체감 latency". |
| **Enqueue Time** | 호스트가 추론을 **enqueue**하는 시간(H2D/D2H CUDA API 호출, 호스트 휴리스틱, 커널 런치). | **CPU가 GPU를 못 따라가는지**의 지표. Enqueue > GPU Compute면 **CPU 바운드**(런치 오버헤드). → CUDA Graph 고려. |
| **H2D Latency** | 입력 텐서 **Host→Device** 복사 시간. | 입력이 크거나 PCIe가 느리면 커진다. |
| **GPU Compute Time** | 순수 **커널 실행** 시간(min/max/mean/median/percentile). | **정밀도 비교의 핵심 지표.** FP16/INT8로 줄어드는 건 이 값. |
| **D2H Latency** | 출력 텐서 **Device→Host** 복사 시간. | 출력이 크면(예: seg mask) 커진다. |
| **Total Host Walltime** | 벤치 전체 벽시계 시간. | Throughput 분모. |
| **Total GPU Compute Time** | 모든 추론의 GPU Compute 합. | Walltime과 비슷하면 GPU가 꽉 찬 것. |

**percentile을 왜 보나 (min/median만 보면 안 되는 이유):**
- `percentile(99%) = 2.55 ms`는 "**추론의 99%가 2.55 ms 안에 끝난다**"는 뜻. 나머지 1%는 그보다 느리다(꼬리 latency).
- 임베디드 실시간 시스템은 **평균이 아니라 최악(꼬리)** 이 마감시한(deadline)을 어기는지가 중요하다. 카메라 30fps면 프레임당 33.3 ms 예산인데, **p99가 예산을 넘으면 100프레임당 1프레임씩 드롭**한다. 그래서 **p90/p99를 deadline과 비교**한다.
- median vs mean: median ≪ mean이면 **가끔 튀는 스파이크**(예: 다른 프로세스와의 GPU 경합, 클럭 스로틀)가 있다는 신호.

**해석 결정 트리(빠른 진단):**
1. `Throughput`이 `1000/GPU_Compute_mean_ms`(이론 최대 qps)의 **70% 미만**이면 → GPU가 논다. 원인은 (a)`Enqueue Time`이 크다(CPU 바운드) 또는 (b)`H2D+D2H`가 크다(전송 바운드).
2. `Enqueue Time mean > GPU Compute mean` → **CPU 런치 오버헤드 병목**. `--useCudaGraph`로 커널 런치를 묶거나, 배치를 키운다.
3. `H2D + D2H > GPU Compute의 상당 부분` → **전송 병목**. 입력 전처리를 GPU로 옮기거나 `--noDataTransfer`로 순수 compute를 따로 재서 확인.
4. 위 다 아니고 그냥 `GPU Compute`가 크다 → **연산 자체가 병목**. `--dumpProfile`로 어느 레이어인지 찾는다(아래).

```bash
# trtexec 출력에서 성능 관련 줄만 뽑기
trtexec --loadEngine=yolo11n_int8.engine --dumpProfile --separateProfileRun 2>&1 \
  | grep -Ei "Throughput|Latency|Enqueue|GPU Compute|H2D|D2H|Walltime|percentile"

# 순수 GPU compute만 격리(전송 제외)해서 재기 — 전송 병목 여부 판정
trtexec --loadEngine=yolo11n_int8.engine --noDataTransfer \
        --dumpProfile --separateProfileRun 2>&1 | grep -Ei "GPU Compute|Throughput"
```

- `--noDataTransfer` : H2D/D2H를 **끄고** 순수 커널만 잰다. 이걸 켰을 때 Throughput이 크게 오르면 → 아까 병목은 **전송**이었다는 확증.
- `--useCudaGraph` : 반복 추론을 CUDA Graph로 캡처해 **런치 오버헤드를 제거**. Enqueue 바운드일 때 효과.
- `--useSpinWait` : 완료 대기를 sleep 대신 **busy-wait**로. latency 측정의 지터를 줄이지만 CPU를 태운다(측정 정밀도용).

#### (4) `--dumpProfile` 레이어별 시간 읽는 법

`--dumpProfile`을 켜면 Performance summary 앞/뒤에 **레이어별 프로파일 표**가 나온다(값은 예시):

```text
[I] === Profile (per layer) ===
[I]   Layer                                        Time(ms)   Avg(ms)   Time(%)
[I]   /model.0/conv/Conv + PWN(Sigmoid,Mul)          452.10     0.452     29.8
[I]   /model.2/m.0/cv2/conv/Conv                     210.44     0.210     13.9
[I]   /model.9/cv2/conv/Conv                         180.02     0.180     11.9
[I]   ... (하위 레이어들) ...
[I]   Total                                         1517.00     1.517    100.0
```

읽는 법:
- **`Time(%)`가 큰 상위 5개가 병목**이다. 위 예에서 상위 3개가 이미 ~55%. 최적화는 **여기부터** 손댄다.
- 레이어 이름의 **`+`** 는 **fusion**을 뜻한다. `Conv + PWN(Sigmoid,Mul)`는 Conv와 SiLU가 한 커널로 합쳐진 것 — TensorRT가 잘 융합했다는 좋은 신호.
- **정밀도 검증법:** FP16/INT8 빌드에서 상위 레이어의 `Time`이 FP32 대비 **안 줄었다면**, 그 레이어가 저정밀 tactic을 **못 골랐다**는 뜻(2.2). `--profilingVerbosity=detailed`로 tactic 이름을 보면 `int8`/`fp16`/`imma`가 붙었는지 확인할 수 있다. INT8인데 tactic이 `float`면 그 레이어는 INT8로 안 도는 것 → polygraphy로 추적(실습 2).

> 💡 팁: per-layer 표를 CSV로 저장하려면 `--exportProfile=prof.json`을 함께 준다. JSON을 파싱해 `Time(%)` 내림차순으로 정렬하면 "Top-N 병목 레이어" 목록이 곧 최적화 백로그가 된다.

```bash
# 프로파일을 JSON으로 내보내고, 상위 시간 레이어를 뽑기
trtexec --loadEngine=yolo11n_int8.engine --dumpProfile --separateProfileRun \
        --exportProfile=yolo11n_int8_prof.json
python3 - <<'PY'
import json
p = json.load(open("yolo11n_int8_prof.json"))
rows = [r for r in p if "percentage" in r or "averageMs" in r]
rows.sort(key=lambda r: r.get("percentage", 0), reverse=True)
for r in rows[:8]:
    print(f"{r.get('percentage',0):5.1f}%  {r.get('averageMs',0):7.3f}ms  {r.get('name','')[:60]}")
PY
```

#### (5) mAP 측정

ultralytics로 각 엔진의 mAP를 COCO val에서 측정한다.

```bash
# TensorRT 엔진을 직접 검증 (ultralytics가 .engine 로드 지원)
yolo val model=yolo11n_fp32.engine data=coco.yaml imgsz=640
yolo val model=yolo11n_fp16.engine data=coco.yaml imgsz=640
yolo val model=yolo11n_int8.engine data=coco.yaml imgsz=640   # 캘리브레이션된 엔진으로
```

> ⚠️ 주의: `coco.yaml`/COCO val2017 이미지가 로컬에 있어야 한다(ultralytics가 자동 다운로드 시도). 데이터가 무거우면 **val 부분셋(예: 500장)** 으로 상대 비교만 해도 학습 목적에는 충분하다. mAP는 **절대값보다 FP32 대비 하락폭**을 본다.

---

### 실습 2 — polygraphy로 FP32(ONNX-Runtime) vs INT8(TRT) 레이어별 비교 + `debug precision` 이분탐색

INT8 mAP가 떨어졌을 때 "어느 레이어에서 오차가 폭발하는가"를 찾는다. 3단계로 좁힌다: **(전체 비교) → (레이어별 마킹) → (자동 이분탐색)**.

#### (a) 네트워크 최종 출력만 비교 — 문제가 있는지 없는지부터

```bash
# ONNX-Runtime(FP32, golden) vs TensorRT(빌드된 엔진) 최종 출력 비교
polygraphy run yolo11n.onnx --onnxrt --trt \
  --save-results golden.json \
  --atol 1e-2 --rtol 1e-2
```

- `--onnxrt` : ONNX-Runtime를 **기준(golden)** 으로 실행.
- `--trt` : 같은 입력을 TensorRT로 실행.
- `--atol/--rtol` : 허용 절대/상대 오차. YOLO는 후처리 전 raw 출력이라 `1e-2` 정도가 현실적. 너무 타이트하면 항상 FAIL이 뜬다.
- 결과: 텐서별 `PASSED`/`FAILED`와 최대 절대·상대 오차. **여기서 PASS면 굳이 아래로 안 내려가도 된다.**

#### (b) 레이어별 비교 — 어디쯤에서 어긋나는지 대략 잡기

```bash
# 모든 텐서를 출력으로 mark 해서 어느 레이어부터 어긋나는지
#   --trt-outputs/--onnx-outputs mark all → 전 텐서 비교, --fail-fast로 첫 mismatch에서 멈춤
polygraphy run yolo11n.onnx --onnxrt --trt \
  --trt-outputs mark all \
  --onnx-outputs mark all \
  --atol 1e-2 --rtol 1e-2 \
  --fail-fast
```

```bash
# (참고) 저장한 결과를 나중에 텐서별로 diff
polygraphy run yolo11n.onnx --onnxrt --save-results onnx_out.json
polygraphy run yolo11n.onnx --trt    --save-results trt_out.json
polygraphy data diff onnx_out.json trt_out.json      # 두 결과 텐서별 차이 요약
```

읽는 법 & 함정:
- `FAILED`가 **처음 뜨는 텐서**가 "오차가 태어난 곳"에 가깝다. 그 텐서를 만든 레이어가 용의자.
- 🔴 **함정:** `--trt-outputs mark all`을 켜면 **레이어 융합/타이밍/포맷 제약이 달라져** 엔진 자체가 바뀌고, 그 바람에 원래 있던 오차가 **가려지거나 새 오차가 생길 수** 있다(fusion이 깨지면서 정밀도 경로가 달라짐). 그래서 mark all은 "대략 어디쯤"을 잡는 용도이고, 정밀 추적은 (c)의 이분탐색으로 넘어간다.
- custom plugin이 낀 그래프는 ONNX-Runtime에 해당 op가 없어 비교가 안 될 수 있다 → plugin 구간을 제외하고 앞/뒤로 나눠 비교.

#### (c) `polygraphy debug precision` — "몇 번째 레이어까지 고정밀이면 회복되나"를 이분탐색

이게 실무의 결정타다. "이 레이어들만 FP16로 남기면 정확도가 산다"를 **자동 이분탐색**으로 찾는다.

```bash
# 1) golden 입력/출력을 먼저 저장 (FP32 ONNX-Runtime 기준)
polygraphy run yolo11n.onnx --onnxrt \
  --save-inputs net_input.json \
  --save-outputs onnx_res.json

# 2) debug precision: 저정밀(예: fp16) 엔진을 만들되, "앞쪽 N개 레이어를 고정밀로" 올려가며
#    --check가 통과할 때까지 N을 이분탐색으로 좁힌다
polygraphy debug precision yolo11n.onnx \
  --mode bisect \
  --fp16 \
  --no-remove-intermediate \
  --check polygraphy run polygraphy_debug.engine --trt \
          --load-inputs net_input.json \
          --load-outputs onnx_res.json \
          --atol 1e-2 --rtol 1e-2
```

작동 방식과 각 인자:
- `--mode bisect` : **이분탐색**. "앞에서 N개 레이어를 고정밀로 올리면 통과"하는 최소 N을 log(레이어수) 번의 시도로 찾는다(`linear`는 하나씩, 느리지만 확실).
- `--fp16` : 나머지 레이어를 돌릴 **낮은 정밀도**. INT8 문제를 볼 땐 이 자리에 맞는 저정밀 설정을 준다(도구 버전에 따라 저정밀 기준 플래그가 다르니 `polygraphy debug precision --help`로 확인).
- `--no-remove-intermediate` : 중간 산출물(`polygraphy_debug.engine`)을 지우지 않고 남겨 **분석**에 쓴다.
- `--check <cmd>` : 각 시도마다 만들어진 `polygraphy_debug.engine`을 이 명령으로 검증. **exit code 0이면 통과**, 아니면 실패로 본다(정확도를 golden과 비교하는 `polygraphy run`을 넣는다). 더 세밀한 판정은 `--fail-regex "..."`로 출력 문자열을 매칭할 수도 있다.
- 결과 메시지: **`To achieve acceptable accuracy, try running the first N layer(s) in higher precision`** 형태로 나온다. 즉 "앞쪽 N개는 FP16(고정밀)로 두고 나머지를 저정밀로" 하면 통과라는 뜻.

> 🔴 **함정(실무에서 자주 막히는 지점):** `debug precision`이 알려주는 **`N`은 TensorRT 내부 레이어 인덱스**라서, ONNX 노드 이름과 1:1로 안 붙는다([TensorRT #4616](https://github.com/NVIDIA/TensorRT/issues/4616) 참고). N을 ONNX 노드로 되돌리려면 `polygraphy inspect model`의 레이어 순서, `--exportProfile`의 레이어 목록, `--onnx-outputs mark all` 비교를 **교차 대조**해야 한다. 그래서 실전에서는 "정확한 노드"보다 "**앞부분/뒷부분/특정 블록**"이라는 구간 감각을 얻는 데 쓰고, 그 구간을 (b)로 좁혀 확정한다.

> 🔗 이 결과가 **mixed precision 근거**가 된다. "앞쪽 N개(=민감 레이어)를 FP16로 유지"라는 결론을 [1단계](03_quantization_theory.md)의 **`layer_sensitivity.csv`**(양자화 민감도)와 대조해, 두 근거가 같은 레이어를 가리키면 확신을 갖고 그 레이어만 고정밀로 고정한다(실습 4 참조).

---

### 실습 3 — INT8 캘리브레이터 직접 구현 (TensorRT 10.x, `IInt8EntropyCalibrator2`)

> ⚠️ **버전 주의:** `IInt8EntropyCalibrator2`는 **10.1부터 deprecated**, **11.0에서 관련 경로가 제거**됐다. **정본 10.16.x LTS에서는 여전히 동작**(deprecated 경고만)하며, deprecated API는 2027-03까지 유지된다. 11.x거나 신규 프로젝트라면 실습 4(ModelOpt PTQ/QDQ)로 대체하라. 그래도 "캘리브레이터가 내부에서 뭘 하는가"를 이해하려면 한 번은 짜볼 가치가 있다.

> 🟩 **3단계 실측 — implicit이 정본 10.16에서 여전히 빌드된다 (2026-08-17 · RTX 3080 · ResNet50 · [`t04`](../experiments/stage3_tensorrt/t04_implicit_calibrator.py)):** 위 `IInt8EntropyCalibrator2`(QDQ 없는 FP32 ONNX + 캘리브 200장)로 INT8 엔진 **빌드 성공**(캐시 5,776B). deprecation 경고는 **Python 바인딩 레벨 134건**(그중 `Superseded by explicit quantization`=TensorRT **10.1** 표시 **8건** → 문서 §2.2의 "10.1부터 deprecated"를 실측으로 확증, 나머지 126건은 config 플래그의 strong-typing/10.12 경고), **TRT 로그 자체엔 0건**. 즉 "deprecated=제거"가 아니라 정본 LTS에선 경고만 뜨고 정상 동작한다.

| INT8 경로(동일 ResNet50) | p50 (ms) | top-1 | INT8 커널줄 | 제어성 |
|---|---|---|:---:|---|
| explicit QDQ (실습1) | 0.7843 | 76.36% | 74 | 층별 명시(권장) |
| implicit calib (실습3) | **0.7074** | **76.80%** | 57 | TRT 자동(deprecated) |

- 이 모델에선 implicit이 **더 빠르고(×2.35 vs explicit ×2.12) 더 정확했다**(76.80% vs 76.36%). 이유는 TRT가 지연 최소화 목표로 **층별 정밀도를 자동 선택**(INT8을 57층만)한 반면, explicit QDQ는 감싼 층을 **전부 INT8로 강제**(74층)했기 때문이다.
- ⚠️ **그래도 신규는 explicit 권장.** 위 수치는 **이 모델에서 우연히** 유리했을 뿐이고, implicit은 (a) deprecated(10.1)라 제거 예정이고 (b) 어느 층을 INT8로 둘지 **제어할 수 없다**. 재현성·이식성·1단계 sensitivity 연동이 필요한 실전에선 explicit QDQ(실습1·4)가 맞다.

**캘리브레이터가 하는 일(직관):** INT8은 실수 텐서를 정수로 매핑하는데, 그 스케일 `s = max_abs / 127`을 정하려면 "이 텐서 값이 보통 어디까지 커지나"를 알아야 한다. 캘리브레이터는 **대표 입력 배치를 실제로 흘려보내며 각 텐서의 분포(히스토그램)를 모아** 동적 범위를 추정한다. `EntropyCalibrator2`는 그 범위를 **KL divergence(정보 손실)를 최소화**하도록 고른다(단순 min/max보다 outlier에 강함). 이 콜백이 배치를 공급하는 게 우리가 짤 코드다.

최소 구현(파이썬):

```python
# int8_calibrator.py  (TensorRT 10.x)
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA 컨텍스트 초기화

class YoloEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """대표 이미지 배치를 공급해 INT8 동적 범위를 추정한다."""
    def __init__(self, calib_files, input_shape=(1, 3, 640, 640), cache="calib.cache"):
        super().__init__()                     # 반드시 부모 생성자 호출
        self.cache_file = cache
        self.batch = np.empty(input_shape, dtype=np.float32)
        self.files = calib_files               # 전처리된 .npy 또는 이미지 경로 리스트
        self.idx = 0
        self.d_input = cuda.mem_alloc(self.batch.nbytes)  # 디바이스 버퍼 1개(입력 1개 가정)

    def get_batch_size(self):
        return self.batch.shape[0]

    def get_batch(self, names):
        # 데이터가 소진되면 None을 반환해야 캘리브레이션이 끝난다
        if self.idx >= len(self.files):
            return None
        img = np.load(self.files[self.idx]).astype(np.float32)  # 사전 전처리(리사이즈/정규화) 가정
        self.batch[...] = img
        cuda.memcpy_htod(self.d_input, self.batch)              # H2D 복사
        self.idx += 1
        return [int(self.d_input)]                              # 디바이스 포인터 리스트 반환

    def read_calibration_cache(self):
        # 캐시가 있으면 재계산 생략 → 빌드 가속 (분포 재수집을 통째로 건너뜀)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

콜백 4개가 각각 언제 불리나(생명주기):
1. `get_batch_size()` → 빌더가 배치 크기 확인(한 번).
2. `get_batch(names)` → **소진될 때까지 반복 호출**. `names`는 입력 텐서 이름 리스트(다중 입력이면 순서대로 포인터를 돌려준다). `None`을 반환하는 순간 캘리브레이션 종료.
3. `read_calibration_cache()` → 빌드 **시작 시** 호출. 캐시가 있으면 2번(데이터 흘리기)을 **통째로 건너뛴다** → 두 번째 빌드부터 매우 빠름.
4. `write_calibration_cache(cache)` → 분포 수집이 끝나면 호출. 결과 스케일을 `calib.cache`로 저장.

배치 스트림을 여러 장으로 (batch>1) 공급하려면 `input_shape=(8,3,640,640)`처럼 배치 차원을 키우고, `get_batch`에서 8장을 쌓아 한 번에 복사한다(GPU 메모리 여유 내에서 배치를 키우면 캘리브레이션이 빨라진다).

```python
# build_int8.py  — 위 캘리브레이터로 INT8 엔진 빌드 (TensorRT 10.x)
import glob, tensorrt as trt
from int8_calibrator import YoloEntropyCalibrator

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)
with open("yolo11n.onnx", "rb") as f:
    assert parser.parse(f.read()), "ONNX parse 실패"

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)                          # implicit INT8 활성
# (선택) workspace 상한 지정 — 큰 모델에서 tactic 탐색 여유 확보
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB
config.int8_calibrator = YoloEntropyCalibrator(
    sorted(glob.glob("calib_data/*.npy"))                      # 500~1000장 권장
)
serialized = builder.build_serialized_network(network, config)
with open("yolo11n_int8_calib.engine", "wb") as f:
    f.write(serialized)
print("INT8 엔진 빌드 완료 → yolo11n_int8_calib.engine")
```

빌드 로그에서 확인할 것:
- `Calibrated ... using ... entropy` 류 메시지가 뜨고, `calib.cache` 파일이 생성된다.
- 로그에 `Missing scale and zero-point for tensor ...`가 뜨면 → 그 텐서에 스케일이 안 잡힌 것(캘리브 데이터가 그 경로를 안 지났거나 op 미지원). mixed precision으로 그 레이어를 FP16 고정하거나 캘리브 데이터를 보강한다.

> 💡 팁: 캘리브레이션 이미지는 **학습/검증 분포를 대표**해야 한다(주간·야간·근접·원거리 골고루). YOLO 계열은 통상 **1000장 이상**을 권장하고, 너무 적으면 mAP가 눈에 띄게 떨어진다. `calib.cache`가 생기면 다음 빌드부터 재사용된다(데이터·전처리를 바꾸면 캐시를 **지우고** 다시 만들어야 한다 — 안 그러면 옛 스케일을 재사용해 버린다).

---

### 실습 4 — TensorRT Model Optimizer로 PTQ/QAT + QDQ ONNX export (10.x·11.x 공통, 권장)

> ⚠️ 확인 필요: 아래 명령/인자는 [NVIDIA Model Optimizer 저장소](https://github.com/NVIDIA/Model-Optimizer)와 `examples/onnx_ptq` 예제 기준(2026-07)이다. 자신의 `modelopt.__version__`에 맞는 README를 재확인하라(인자명은 릴리스마다 소폭 변한다).

**왜 ModelOpt인가:** 실습 3의 캘리브레이터는 **implicit**(엔진 빌드 시 스케일 결정)이라 11.x에서 죽는다. ModelOpt는 **explicit** — 스케일을 계산해 **ONNX 그래프에 Q/DQ 노드로 박아** 준다. 그 QDQ ONNX는 10.x·11.x **양쪽에서 동일하게** 빌드된다. 즉 정본(10.16)에서도 미래(11.x)에서도 안 깨지는 경로다.

#### (A) ONNX PTQ — 가장 간단한 경로

```bash
# 1) 캘리브레이션 데이터를 numpy로 준비 (전처리 완료된 입력 텐서 묶음)
#    shape 예: (N, 3, 640, 640) float32 → calib.npy 로 저장
python3 - <<'PY'
import numpy as np, glob
xs = [np.load(p) for p in sorted(glob.glob("calib_data/*.npy"))]
np.save("calib.npy", np.stack(xs).astype("float32"))
PY

# 2) ONNX 모델에 INT8 PTQ 적용 → Q/DQ 노드가 삽입된 ONNX 출력
python3 -m modelopt.onnx.quantization \
    --onnx_path=yolo11n.onnx \
    --quantize_mode=int8 \
    --calibration_data=calib.npy \
    --calibration_method=entropy \
    --output_path=yolo11n.quant.onnx
# quantize_mode 대안: fp8, int4 (하드웨어 지원 확인 필요). INT8은 opset>=13 요구(자동 업그레이드).
# calibration_method: int8/fp8은 entropy(기본)|max, int4는 awq_clip(기본)|rtn_dq 등.
```

각 인자:
- `--quantize_mode=int8` : 정본 경로. `fp8`(Hopper/Ada 이상), `int4`(가중치 전용, LLM 계열)도 있으나 하드웨어·op 지원을 먼저 확인.
- `--calibration_data=calib.npy` : 대표 입력 텐서 묶음(numpy). YOLO는 전처리(리사이즈/정규화)까지 끝난 `(N,3,640,640)` float32.
- `--calibration_method=entropy` : INT8 기본. outlier가 심한 activation이 많으면 `max`가 나을 때도 있으니 둘 다 재본다.
- `--output_path=yolo11n.quant.onnx` : Q/DQ가 삽입된 결과 ONNX. 생략하면 `.quant` 접미사로 자동 저장.

```bash
# 3) QDQ ONNX를 TensorRT로 빌드 (엔진이 Q/DQ를 인접 레이어에 fuse → INT8 텐서코어 실행)
#    10.x(정본): --int8 붙여도 되고 안 붙여도 QDQ가 있으면 INT8 실행. 11.x: 플래그 없이.
trtexec --onnx=yolo11n.quant.onnx --int8 --saveEngine=yolo11n_int8_qdq.engine \
        --dumpProfile --separateProfileRun --profilingVerbosity=detailed
```

- 빌드 후 `--dumpProfile`에서 Conv tactic에 `int8`/`imma`가 붙었으면 QDQ가 제대로 fuse된 것. `float` tactic만 보이면 Q/DQ 배치가 어긋난 것(그 레이어 앞뒤에 Q/DQ가 짝이 안 맞음).

#### (B) PyTorch QAT → QDQ export (정확도 회복이 필요할 때)

`modelopt.torch.quantization`으로 모델에 fake-quant을 삽입(`mtq.quantize`)하고 몇 epoch fine-tune한 뒤 ONNX로 export하면, fake-quant이 **Q/DQ 노드로 변환**되어 나온다. 개념 골격:

```python
# qat_sketch.py  (개념 골격 — 데이터로더/학습 루프는 프로젝트에 맞게)
import torch
import modelopt.torch.quantization as mtq

model = load_pretrained_yolo()                 # nn.Module

# 1) 양자화 설정(INT8) 선택 후 fake-quant 삽입 + 캘리브레이션
def forward_loop(m):
    for imgs, _ in calib_loader:               # 대표 배치 몇 개
        m(imgs.cuda())
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

# 2) 짧게 fine-tune (fake-quant 상태로 몇 epoch) — 정확도 회복
train_a_few_epochs(model)                      # 기존 학습 루프 재사용, lr 작게

# 3) QDQ가 박힌 ONNX로 export → trtexec로 빌드(위 (A)-3과 동일)
torch.onnx.export(model, dummy_input, "yolo11n.qat.onnx", opset_version=13)
```

QAT는 PTQ로 회복 안 되는 정밀도 민감 태스크에서 쓴다(2단계 Transformer 양자화와 연결). 구버전 `pytorch-quantization`도 동일 개념(QDQ export)이지만 ModelOpt로 통합되는 추세다.

#### (C) 1단계에서 만든 ORT QDQ ONNX를 가져올 때 — TRT 호환 설정으로 다시 뽑기

[1단계 4.3](03_quantization_theory.md)의 `quantize_ptq.py`로 만든 INT8 QDQ ONNX는 **그대로 가져오면 TensorRT가 파싱에 실패하고 무음 폴백한다**(2.2.1). 3단계로 넘어오기 전에 **타깃을 TensorRT로 바꿔 다시 양자화**해야 한다. 고칠 원인은 **activation zero-point 하나**이고, 그걸 위해 바꾸는 줄이 두 개다.

```python
# quantize_trt.py — 1단계 quantize_ptq.py를 TensorRT 타깃으로 다시 뽑는다
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType, QuantFormat
from calib_reader import ImageNetCalibReader          # 1단계 4.2 그대로 재사용

quantize_static(
    model_input="resnet18_fp32.onnx",
    model_output="resnet18_int8_trt.onnx",
    calibration_data_reader=ImageNetCalibReader("imagenet/val", input_name="input", limit=200),
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QInt8,     # ← (a) QUInt8(비대칭)에서 변경. 이게 없으면 (b)가 무효(zp=127)
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    extra_options={
        "WeightSymmetric": True,
        "ActivationSymmetric": True,     # ← (b) activation zero_point = 0 — 유일한 하드 블로커(2.2.1 ②) 해소
        # "QuantizeBias": False,         # ← (c) 선택. bias DQ 21개가 사라질 뿐, 파싱 성공의 조건은 아니다
    },
)
```

- **(a)와 (b)는 한 세트지만, 고치는 원인은 하나다.** ORT의 `ActivationSymmetric=True`는 *"int8·int16이면 zero-point가 0, **uint8·uint16이면 zero-point가 127·32767**"* 로 동작한다(ORT 1.23.2 `quantize.py`의 `get_qdq_config` docstring). 즉 `QUInt8`인 채로 `ActivationSymmetric`만 켜면 zero-point가 **127**이 되어 여전히 `shiftIsAllZeros`에 걸린다. `activation_type`까지 `QInt8`로 바꿔야 비로소 0이 된다 — **두 줄을 함께 바꿔야 하는 이유는 그것 하나뿐**이고, 목표는 어디까지나 `zero_point = 0`이다.
- **(c) `QuantizeBias: False`는 선택적 정리이지 해법이 아니다.** 기본값 `True`는 bias를 INT32로 양자화하고 DQ를 붙이는데, 이걸 끄면 ResNet18 기준 bias DQ **21개**가 사라진다. 다만 2.2.1의 절제 실험대로 **(a)(b)만으로 이미 빌드가 통과하고**(bias DQ가 21개 남아 있는 채로), (c)를 더해도 p50은 **0.51 ms로 동일**하다. 반대로 **(c)만 켜고 (a)(b)를 빼면 여전히 파싱에 실패한다**(절제 실험 case B). 그래프를 깔끔하게 하고 싶을 때 쓰는 옵션으로 보면 된다.
  - **정확도 영향은 "비슷하다"가 아니라 정확히 0이다.** 50,000장에서 (a)(b) 단독(case C)과 (a)(b)+(c)(case D)의 **예측이 한 장도 다르지 않았다**(top-1 68.33% 동일, 0장 불일치). bias를 INT32로 접든 FP로 남기든 TRT가 융합한 뒤의 계산 결과가 같다는 뜻이다. 따라서 **(c)를 정확도 카드로 쓰지 마라** — 대칭 전환으로 잃은 −0.29%p는 (c)로 되찾을 수 없고, 되찾는 방법은 부분 FP16 혼합(2.4)이나 QAT뿐이다.
- 참고로 ONNX Runtime 공식 문서도 **"quantization on GPU only supports S8S8"**(activation·weight 모두 signed int8)이라고 못 박고 있다([Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). 1단계의 `QUInt8` 권장은 **x86 CPU/VNNI 한정**이다.

빌드 전에 **파싱이 실제로 성공하는지** 먼저 확인한다 — 무음 폴백을 막는 가장 확실한 방법이다.

```bash
# (1) trtexec로 먼저 빌드해 본다. 파싱 에러는 여기서 화면에 그대로 뜬다(EP처럼 숨지 않는다).
trtexec --onnx=resnet18_int8_trt.onnx --int8 --saveEngine=resnet18_int8_trt.engine \
        --dumpProfile --separateProfileRun --profilingVerbosity=detailed 2>&1 | tee build_int8.log

# (2) 파싱 실패 신호 — 하나라도 걸리면 위 (a)(b)가 제대로 들어갔는지 다시 확인
grep -Ei "Non-zero zero point|must have type|Invalid Node|Unsupported" build_int8.log

# (3) Q/DQ가 실제로 INT8 커널로 fuse됐는지 — tactic 이름에 int8/imma가 보여야 성공(2.2 팁)
grep -Ei "int8|imma" build_int8.log | head
```

> ⚠️ **`trtexec`가 없다면 (pip 설치 경로).** `pip install tensorrt-cu12`로 깐 휠에는 **`trtexec` 바이너리가 들어 있지 않다**(`polygraphy`만 온다). 0단계를 pip 경로로 따라온 독자는 위 3커맨드를 못 쓴다. 그럴 땐 **같은 모델을 TRT EP와 CUDA EP로 각각 재서 부호로 판정**하면 된다 — 파싱에 성공하면 TRT가 CUDA보다 뚜렷이 빠르고, 폴백하면 비슷하거나 더 느리다.
>
> ```python
> # trt_vs_cuda.py — trtexec 없이 파싱 성공 여부를 판정한다
> import sys, time, numpy as np, onnxruntime as ort
>
> path = sys.argv[1]
> x = np.random.rand(1, 3, 224, 224).astype(np.float32)
>
> def p50(provs):
>     so = ort.SessionOptions(); so.log_severity_level = 2   # 파서 에러를 보려면 3~4로 올리지 말 것
>     s = ort.InferenceSession(path, so, providers=provs)
>     nm = s.get_inputs()[0].name
>     for _ in range(20):                       # 워밍업 (TRT는 여기서 엔진을 빌드한다)
>         s.run(None, {nm: x})
>     ts = []
>     for _ in range(60):
>         t0 = time.perf_counter()
>         s.run(None, {nm: x})
>         ts.append((time.perf_counter() - t0) * 1e3)
>     return float(np.percentile(ts, 50))
>
> trt  = p50(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
> cuda = p50(["CUDAExecutionProvider", "CPUExecutionProvider"])
> print(f"TRT p50={trt:.2f} ms  CUDA p50={cuda:.2f} ms  → "
>       f"{'파싱 성공' if trt < cuda * 0.8 else '🔴 폴백(파싱 실패)'}")
> ```
>
> 실측 예: 폴백 모델 `TRT 3.06 / CUDA 1.81` → 🔴, 고친 모델 `TRT 0.51 / CUDA 2.11` → 파싱 성공. **판정 근거는 절대값이 아니라 두 EP의 대소**다(폴백 상태의 절대값은 3~5 ms대에서 출렁인다). 이 방식으로 2.2.1의 2×2 절제 실험을 그대로 재현할 수 있다.

> 🔴 **함정 — ORT TensorRT EP는 파싱에 실패해도 예외를 안 던진다.** `providers=["TensorrtExecutionProvider", ...]`로 세션을 만들면 `get_providers()`에는 TRT가 그대로 보이는데 실제로는 전부 CUDA/CPU로 폴백해 있다. 그래서 **위처럼 `trtexec`로 먼저 빌드해 보는 절차**가 필요하다. 굳이 ORT EP로 가야 한다면 `SessionOptions`에 `log_severity_level=2`를 주고 로그에서 (2)의 문자열을 찾아라. [0단계 3-4-a](01_environment_setup.md)의 `libnvinfer.so.10` 무음 CPU 폴백과 **같은 계열**의 함정이다 — 임베디드에서 "조용히 느려지는" 실패가 "요란하게 죽는" 실패보다 훨씬 비싸다.

> 🔗 **1단계 산출물 연결 (mixed precision 근거):** [1단계](03_quantization_theory.md)에서 만든 **`layer_sensitivity.csv`** 를 mixed precision 근거로 쓴다. 민감도 상위 레이어(양자화 시 오차가 큰 레이어)는 INT8에서 제외하고 FP16으로 남긴다. 실습 2-(c)의 `debug precision` 결과("앞쪽 N개를 고정밀로")와 이 CSV가 **같은 레이어를 가리키면** 그 레이어를 고정밀로 고정한다.
> - 10.x(정본): `--precisionConstraints=obey --layerPrecisions=<layer>:fp16` 로 특정 레이어 정밀도 고정.
> - 11.x: 위 플래그가 제거됐으므로, ModelOpt에서 해당 레이어를 **양자화 대상에서 제외**(Q/DQ 미삽입)하거나 AutoCast로 FP16 유지하도록 지정한다.
> 이렇게 "민감 레이어만 고정밀"로 두는 것이 mixed precision의 실전 형태다. 산출물 `mixed_precision_plan.md`에 "어느 레이어를, 어떤 근거(sensitivity CSV / debug precision)로, FP16로 남겼고, mAP가 얼마나 회복됐는지"를 적는다.

```bash
# 10.x: 특정 레이어를 FP16으로 고정하며 INT8 빌드 (mixed precision)
trtexec --onnx=yolo11n.quant.onnx --int8 --fp16 \
        --precisionConstraints=obey \
        --layerPrecisions=/model.9/cv2/conv/Conv:fp16 \
        --saveEngine=yolo11n_mixed.engine --dumpProfile --separateProfileRun
```

- `--precisionConstraints=obey` : "내가 지정한 레이어 정밀도를 **반드시 지켜라**"(무시하면 에러). `prefer`는 "되도록"이라 조용히 무시될 수 있어 검증엔 `obey`가 낫다.
- `--layerPrecisions=<name>:fp16` : 그 레이어를 FP16으로. 이름은 `--profilingVerbosity=detailed`의 per-layer 표에서 그대로 복사한다.

---

### 실습 5 — DLA 파티셔닝 (GPU-only / DLA-only / 하이브리드)

> 🟨 **보드 필요 구분:** 아래는 **Orin/Xavier 보드 + TensorRT 10.x(JetPack)에서만** 실행/실측이 의미 있다(DLA 하드웨어 + DLA를 지원하는 TRT 필요). RTX 데스크톱에는 DLA가 없어 `--useDLACore`가 에러다. **또한 TensorRT 11.0/11.1은 DLA 자체를 지원하지 않는다** — DLA는 정본 10.x 전용 실습이다. **보드가 없으면 이 실습은 개념 학습으로 남기고, 산출물 `dla_fallback.md`에 '보드 필요 — 미실행'으로 명시**한다.

> 🟥 **3단계 실측 — 이 머신(RTX 3080)엔 DLA가 없다(범위 밖, 정직한 폴백):** TRT introspection에서 **`num_DLA_cores=0`**([`t01_env.json`](../experiments/stage3_tensorrt/t01_env.json)) — dGPU라 DLA 코어가 물리적으로 없다. 따라서 이 실습은 **하드웨어 범위 밖**(2단계 BEVFormer 전체 INT8과 같은 처리)이며, 아래 `trtexec --useDLACore` 명령은 Orin/Xavier에서만 유효하다. 이 머신 기준 산출물 `dla_fallback.md`는 '보드 필요 — 미실행'으로 남긴다.

```bash
# (a) GPU-only (기준선) — Orin에서
trtexec --onnx=yolo11n.onnx --fp16 --saveEngine=yolo11n_gpu.engine \
        --dumpProfile --separateProfileRun

# (b) DLA-only 지향 + GPU fallback 허용 (DLA는 FP16/INT8만)
trtexec --onnx=yolo11n.onnx --fp16 \
        --useDLACore=0 --allowGPUFallback \
        --saveEngine=yolo11n_dla.engine \
        --dumpProfile --separateProfileRun 2>&1 | tee dla_build.log

# (c) 하이브리드: 위와 동일하되, fallback 레이어를 로그에서 읽어 어떤 op가 GPU로 갔는지 분석
```

- `--useDLACore=0` : DLA 코어 0에 올린다(Orin은 DLA 코어가 2개, 0/1). DLA는 **FP16/INT8만** 되므로 `--fp16`(또는 INT8 QDQ)이 필수.
- `--allowGPUFallback` : DLA가 **못 도는 레이어를 GPU로** 떨어뜨려 빌드가 실패하지 않게 한다. 이걸 빼면 미지원 op에서 빌드가 막힌다.

fallback 레이어 목록 추출:

```bash
# 빌드 로그에서 DLA로 간 레이어 / GPU로 fallback된 레이어를 구분
grep -Ei "DLA|GPU|fallback|running on|device" dla_build.log | head -n 80

# 엔진의 레이어별 device 배치를 정밀하게 덤프 (레이어 정보까지)
trtexec --loadEngine=yolo11n_dla.engine --dumpLayerInfo \
        --exportLayerInfo=dla_layers.json
python3 - <<'PY'
import json
info = json.load(open("dla_layers.json"))
layers = info.get("Layers", info if isinstance(info, list) else [])
on_gpu = [l for l in layers if isinstance(l, dict) and "gpu" in str(l).lower()]
print(f"총 레이어 {len(layers)}개 중 GPU fallback 의심 {len(on_gpu)}개")
PY
```

- `--dumpLayerInfo` / `--exportLayerInfo=<json>` : 엔진의 레이어별 정보(정밀도·device 등)를 덤프. fallback을 로그 grep보다 정확히 센다.

읽는 법 & 개선 루프(원본 가이드의 "fallback 0 만들기 = HW-aware redesign 미니 버전"):

1. 로그/레이어 정보에서 **`GPU` 표시된 레이어 = fallback**. 이들이 전체의 20~30%를 넘으면 DLA 이득이 사라진다.
2. fallback을 유발하는 대표 op를 파악한다. DLA 미지원의 대표 사례(2.3):
   - **attention/MatMul 계열** → DLA 불가(통째로 GPU).
   - **Softmax** → Xavier 불가(Orin만). Xavier면 이것도 fallback.
   - **동적 shape** → DLA 불가(min=opt=max 강제).
   - **큰 커널/stride, grouped/dilated deconv, 특수 activation**(예: 커스텀 activation) → 제약 위반 시 fallback.
3. 모델을 **DLA 친화 op로 치환**한다(예: 미지원 upsample → 지원되는 resize 형태로, 커스텀 activation → ReLU/LeakyReLU/Clipped ReLU 등 DLA 지원 목록 내로, 동적 shape → 고정 shape). 이게 **HW-aware redesign의 축소판**이다.
4. 다시 빌드해 fallback이 줄었는지 확인 → **fallback 0**을 목표로 반복. `dla_fallback.md`에 **치환 전/후 fallback 수와 latency 변화**를 표로 남긴다.

> 🔴 함정: fallback을 "허용"만 해두고 방치하면 DLA↔GPU 텐서 복사가 매 레이어 경계마다 생겨 GPU-only보다 느려진다. **DLA는 "쪼개서 조금"이 아니라 "덩어리째 DLA"** 여야 이득이다. 로그에서 DLA subgraph가 **여러 조각으로 쪼개졌으면**(중간중간 GPU가 낌) 경계 복사가 그만큼 늘어난 것 → 그 경계의 op를 없애 subgraph를 하나로 합치는 게 핵심.

---

### 실습 6 — Custom Plugin 골격 (IPluginV3, C++) — *진짜 차별점*

> ⚠️ **버전 경계(중요):** TensorRT는 **10부터 `IPluginV2` 계열을 deprecated**, **11.0에서 `IPluginV2`/`IPluginV2DynamicExt`/`IPluginCreator`/`addPluginV2()`를 전면 제거**했다(2026-07 기준 확인). **현행 인터페이스는 `IPluginV3`** 이며, `IPluginCreatorV3One`으로 등록하고 `INetworkDefinition::addPluginV3()`로 추가한다. 정본 10.16.x에서도 **신규 플러그인은 `IPluginV3`로 작성**하는 게 맞다(V2는 deprecated). 인터넷의 옛 `IPluginV2DynamicExt` 예제는 11.x에서 컴파일되지 않으니 **`IPluginV3`로 시작**하라.

`IPluginV3`는 하나의 모놀리식 클래스가 아니라 **세 capability 인터페이스의 조합**이다. 왜 쪼갰나: 빌드 시에만 필요한 것(형상/타입/포맷)과 런타임에만 필요한 것(커널 실행)을 분리하면, TensorRT가 **필요한 시점에 필요한 인터페이스만** 얻어 갈 수 있다(직렬화된 엔진을 로드할 땐 Build capability가 없어도 됨).

| Capability | 인터페이스 | 대표 메서드 | 역할 |
|-----------|-----------|-----------|------|
| Core | `IPluginV3OneCore` | `getPluginName`, `getPluginVersion`, `getPluginNamespace` | 식별(레지스트리 키) |
| Build | `IPluginV3OneBuild` | `getNbOutputs`, `getOutputShapes`, `getOutputDataTypes`, `supportsFormatCombination`, `getFormatCombinationLimit`, `configurePlugin` | 빌드 시 형상/타입/포맷 결정 |
| Runtime | `IPluginV3OneRuntime` | `enqueue`, `getWorkspaceSize`, `onShapeChange`, `attachToContext`, `getFieldsToSerialize` | 실제 커널 실행 + 직렬화 |

메서드별 역할(왜 각각 필요한가):
- `getOutputShapes(...)` : 입력 shape로부터 **출력 shape를 기호(symbolic)로** 계산. 동적 shape에서 출력 크기를 TensorRT가 알아야 버퍼를 잡는다.
- `getOutputDataTypes(...)` : 출력 dtype 결정(보통 입력 dtype 따라감).
- `supportsFormatCombination(pos, io, ...)` : "이 위치(pos)의 텐서가 이 (dtype, 메모리 포맷) 조합을 지원하는가?"를 답한다. 지원하는 조합만 `true` → TensorRT가 그 안에서 최적을 고른다.
- `getFormatCombinationLimit()` : TensorRT가 tactic당 타이밍할 포맷 조합 **개수 상한**. 기본으로 충분하나, 조합이 많은 플러그인은 늘린다.
- `configurePlugin(...)` : 실제 선택된 입출력 형상/포맷을 통보받아 내부 상태를 준비(빌드 단계).
- `enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream)` : **핵심.** 실제 CUDA 커널을 이 stream에 launch. 반환 0=성공.
- `getWorkspaceSize(...)` : 커널이 필요로 하는 scratch 메모리 크기. TensorRT가 미리 잡아 `enqueue`의 `workspace`로 준다.
- `onShapeChange(...)` : 실행 중 입력 shape가 바뀌면 통보(동적 shape 대응).
- `attachToContext(...)` : 실행 컨텍스트에 붙을 때(예: cuBLAS 핸들 확보) 훅.
- `getFieldsToSerialize()` : 엔진에 **저장할 플러그인 속성**을 반환 → 나중에 `createPlugin`이 그걸로 복원.

C++ 골격(예: deformable attention 같은 미지원 op를 위한 최소 형태). **시그니처는 정본 10.16.x [Adding Custom Layers Using the C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html) 기준**:

```cpp
// DeformAttnPlugin.h  (TensorRT 10.x, IPluginV3)
#include <NvInfer.h>
using namespace nvinfer1;

// 세 capability를 한 클래스에서 구현하는 형태
class DeformAttnPlugin : public IPluginV3, public IPluginV3OneCore,
                         public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    // --- IPluginV3: capability 노출 (kBUILD/kRUNTIME 아니면 kCORE) ---
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        try {
            if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
            if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
            return static_cast<IPluginV3OneCore*>(this);   // 기본 = Core
        } catch (...) {}
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new DeformAttnPlugin(*this); }

    // --- Core: 식별 (레지스트리 키) ---
    const char* getPluginName()      const noexcept override { return "DeformAttn"; }
    const char* getPluginVersion()   const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }

    // --- Build: 출력 개수/형상/타입/지원 포맷 ---
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override {
        outputs[0] = inputs[0];          // 예시: 입력과 동일 형상
        return 0;                        // 0 = 성공
    }
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        outputTypes[0] = inputTypes[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        // 예: 모든 입출력이 FP16 + LINEAR일 때만 지원
        return inOut[pos].desc.type == DataType::kHALF &&
               inOut[pos].desc.format == TensorFormat::kLINEAR;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        return 0;
    }

    // --- Runtime: 커널 실행 + workspace + 직렬화 ---
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        // TODO: 여기서 실제 CUDA 커널(deformable attention)을 stream에 launch
        // launch_deform_attn(inputs[0], outputs[0], inputDesc[0].dims, stream);
        return 0;                        // 0 = 성공
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
        return 0;                        // scratch 필요 없으면 0
    }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
                          PluginTensorDesc const* out, int32_t nbOutputs) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override { return clone(); }

    // 엔진에 저장할 속성(없으면 빈 컬렉션). 복원은 Creator의 createPlugin에서.
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        static PluginFieldCollection fc{0, nullptr};
        return &fc;
    }
};
```

플러그인 등록(레지스트리)과 로드:

```cpp
// DeformAttnCreator.h  — 레지스트리에 등록되는 팩토리
class DeformAttnCreator : public IPluginCreatorV3One {
public:
    const char* getPluginName()      const noexcept override { return "DeformAttn"; }
    const char* getPluginVersion()   const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override {
        static PluginFieldCollection fc{0, nullptr};
        return &fc;                      // 플러그인 attribute 이름 목록
    }
    // 빌드/역직렬화 시 TensorRT가 호출해 플러그인 인스턴스를 만든다. phase로 빌드/런타임 구분.
    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override {
        return new DeformAttnPlugin();   // fc에서 attribute를 읽어 생성자에 넘기는 게 일반형
    }
};

// 이 매크로 한 줄로 플러그인 레지스트리에 자동 등록된다.
REGISTER_TENSORRT_PLUGIN(DeformAttnCreator);
```

빌드·로드:

```bash
# .so 로 컴파일 (경로는 환경에 맞게)
g++ -shared -fPIC -o libdeform_attn.so DeformAttnPlugin.cpp \
    -I/usr/include/x86_64-linux-gnu -I/usr/local/cuda/include \
    -L/usr/lib/x86_64-linux-gnu -lnvinfer -L/usr/local/cuda/lib64 -lcudart

# trtexec에서 플러그인 로드하며 빌드/실행
trtexec --onnx=model_with_deform_attn.onnx \
        --staticPlugins=./libdeform_attn.so \
        --saveEngine=model.engine
```

- `--staticPlugins=<.so>`(구 `--plugins`) : 빌드/실행 전에 이 공유 라이브러리를 로드해 레지스트리에 플러그인을 등록시킨다. ONNX의 커스텀 op가 이 플러그인으로 해석된다.

> 💡 팁: **처음부터 deformable attention을 다 짜지 말고**, "입력을 그대로 통과시키는 identity plugin"을 먼저 등록→빌드→로드까지 성공시켜라. 위 골격에서 `enqueue`가 입력을 출력으로 `cudaMemcpyAsync` 복사만 하면 identity다. 빌드 파이프라인이 뚫린 뒤 `enqueue`의 커널만 교체하면 된다. 전체 예제는 [TensorRT GitHub](https://github.com/NVIDIA/TensorRT)의 plugin 샘플과 [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)를 참고. ModelOpt 예제도 `--staticPlugins`로 커스텀 op 플러그인을 로드하는 형태를 보여준다([onnx_ptq 예제](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)).

---

### 실습 7 — Nsight Systems로 커널 단위 병목 찾기 (선택, 심화)

`trtexec`의 요약 숫자로 "전송 vs 연산" 병목을 판정했다면, Nsight Systems는 그 병목을 **타임라인에서 눈으로** 확인·확정한다.

```bash
# trtexec 실행을 Nsight Systems로 프로파일 → 커널 타임라인/메모리 복사 확인
nsys profile -o yolo11n_int8_nsys \
     --trace=cuda,nvtx,osrt \
     trtexec --loadEngine=yolo11n_int8.engine --dumpProfile --separateProfileRun --iterations=100
# 결과 yolo11n_int8_nsys.nsys-rep 를 nsys-ui(GUI)로 열어 커널별 시간·H2D/D2H 복사 확인
```

- `--trace=cuda,nvtx,osrt` : CUDA 런타임/커널, NVTX 마커(TensorRT가 레이어 이름을 NVTX로 심는다), OS 런타임을 추적.
- `--iterations=100` : 충분한 반복으로 정상 구간(steady state)을 캡처(초기 워밍업 제외).

**타임라인에서 병목 찾는 절차(GUI):**
1. **CUDA HW 행**을 펼쳐 **Kernels 레인**과 **Memory 레인(H2D/D2H)** 을 나란히 본다.
2. **연산 병목:** 한 프레임 구간에서 Kernels 레인이 빈틈없이 꽉 차 있고 그 안에 **유독 긴 커널 막대**가 있으면, 그 커널이 병목. NVTX 마커로 어느 레이어인지 이름을 확인 → 실습 1-(4)의 per-layer 표와 대조.
3. **전송 병목:** Memory 레인의 H2D/D2H 막대가 **Kernels 레인과 겹치지 않고**(직렬화되어) compute를 기다리게 만들면, 파이프라인이 전송에 막힌 것. → 입력 전처리를 GPU로 옮기거나(H2D 감소), 출력을 줄이거나, 스트림을 겹쳐 double-buffering.
4. **런치(CPU) 병목:** CPU(osrt/CUDA API) 행에서 커널 런치 사이 **간격이 크고** GPU가 그 사이 놀면 → Enqueue 바운드. `--useCudaGraph`로 런치를 묶는다.
5. **DLA(보드):** DLA 사용 시 DLA 레인과 GPU 레인 사이 **복사 구간**이 반복되면 fallback 경계 문제(실습 5).

커널 **내부**(warp occupancy, 메모리 병목, register 압박)까지 파려면 Nsight **Compute**(`ncu`)를 쓴다:

```bash
# 특정 커널 하나를 깊게 (occupancy/메모리 처리량 등). 커널이 많으므로 -c 로 개수 제한.
ncu -o yolo11n_ncu -c 20 --set full \
    trtexec --loadEngine=yolo11n_int8.engine --iterations=1
```

읽는 법 요약: 타임라인에서 (1) **memcpy(H2D/D2H)가 compute를 가리는지**(파이프라인 문제), (2) **특정 커널이 비정상적으로 긴지**(그 레이어가 tactic을 잘못 골랐거나 미지원 정밀도), (3) DLA 사용 시 **DLA↔GPU 복사 구간**을 본다.

---

## 5) 예시 / 결과 해석

### 5.1 latency/mAP 3점 비교표 (예시 — 실제 값은 각자 환경에서 채움)

| 정밀도 | 엔진 | GPU Compute (mean) | p99 | 상대 속도 | mAP50-95 | FP32 대비 |
|--------|------|--------------------|-----|-----------|----------|-----------|
| FP32 | `yolo11n_fp32.engine` | 예: 6.0 ms | 예: 7.1 ms | 1.0x | 예: 0.39 | 기준 |
| FP16 | `yolo11n_fp16.engine` | 예: 2.4 ms | 예: 2.9 ms | ~2.5x | 예: 0.389 | ~0.001 ↓ |
| INT8(calib/QDQ) | `yolo11n_int8_qdq.engine` | 예: 1.8 ms | 예: 2.3 ms | ~3.3x | 예: 0.37 | 약 0.5~2%p ↓ |

해석 가이드:
- **FP16은 거의 공짜 점심**이다: 속도 크게 상승, 정확도 하락 미미. 임베디드 1차 최적화는 항상 FP16부터.
- **INT8의 정확도 하락은 캘리브레이션 품질에 좌우**된다. 문헌상 YOLO 계열 static INT8는 통상 **속도 ~1.5~3.3x**, **mAP50-95 하락 약 0.5~2%p(캘리브레이션·데이터가 나쁘면 3~7%p까지)** 로 보고된다. 하락이 크면 → 캘리브레이션 이미지 수/대표성 점검 → QAT(실습 4B) 고려.
- **mean만 보지 말고 p99도 넣어라.** 실시간 시스템은 p99가 프레임 예산(30fps=33.3ms)을 넘는지가 진짜 합격/불합격 기준이다(4-(3) 참조).
- 숫자는 GPU(예: RTX)와 Orin이 **절대값이 다르다.** 학습 목적에서는 **정밀도 간 상대 비율**과 **정확도 하락폭**에 집중하라.

### 5.2 trtexec 출력 판독 예시 (무엇을 보고 무슨 결론을 내렸나)

가상의 INT8 엔진 출력에서:

```text
Throughput: 300 qps
Latency: mean = 3.1 ms, percentile(99%) = 5.8 ms
Enqueue Time: mean = 3.4 ms
GPU Compute Time: mean = 1.9 ms
H2D Latency: mean = 0.9 ms
D2H Latency: mean = 0.3 ms
```

판독:
- 이론 최대 qps = 1000/1.9 ≈ **526 qps**인데 실측은 **300 qps** → GPU가 논다(효율 57%).
- `Enqueue mean(3.4) > GPU Compute mean(1.9)` → **CPU 런치 바운드**가 원인. 결론: `--useCudaGraph`로 런치를 묶거나 배치를 키운다.
- `H2D(0.9)`도 compute의 절반 → 부차적 전송 병목. 전처리를 GPU로 옮기면 추가 이득.
- `p99(5.8) ≫ mean(3.1)` → 꼬리가 김. 다른 프로세스와 GPU 경합 또는 클럭 스로틀 의심 → `nvidia-smi -q -d CLOCK`으로 스로틀 확인, `--useSpinWait`로 측정 지터 제거 후 재측정.

### 5.3 polygraphy 레이어 비교 해석 (예시)

- 대부분 레이어 PASS인데 **특정 Concat/Sigmoid 직후부터 FAIL**이 시작된다면, 그 앞 레이어(예: outlier가 큰 activation)가 INT8 스케일에 안 맞는 것 → 그 레이어를 FP16으로 고정(mixed precision, `layer_sensitivity.csv`와 대조).
- `debug precision --mode bisect` 결과가 "**앞쪽 30개 레이어를 고정밀로**"라고 나오면, 그 30개가 대략 backbone 초입(입력 통계가 불안정한 구간)일 확률이 높다. `--layerPrecisions`로 그 구간만 FP16 고정 후 mAP 재측정 → 회복되면 확정.
- 최종 출력만 조금 어긋나는 정도면 mAP 영향이 작을 수 있으니 **실제 mAP로 판단**한다(오차 ≠ 정확도 하락).

### 5.4 DLA 결과 해석 (보드 시)

| 빌드 | 실행 위치 | fallback 레이어 | 상대 latency | 판단 |
|------|-----------|-----------------|--------------|------|
| GPU-only | 전부 GPU | - | 1.0x | 기준 |
| DLA + fallback 많음 | DLA 일부 + GPU 다수 | 예: 35% | 예: 1.2x(느려짐) | 🔴 복사 오버헤드 |
| DLA 최적화(fallback↓) | 대부분 DLA | 예: 0~5% | 예: 0.9x + GPU 여유 | ✅ 전력/GPU 부하 이득 |

DLA의 진짜 가치는 순수 latency보다 **GPU를 비워 다른 태스크(예: 다중 카메라)를 병렬로 돌릴 수 있다는 것**과 **전력 효율**이다. `dla_fallback.md`에는 "op 치환 전/후 fallback 수, 그리고 GPU가 얼마나 비었는지(다른 태스크 여유)"를 함께 적으면 포트폴리오로서 강하다.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| `trtexec: command not found` (pip 설치) | pip 휠 `tensorrt-cu12`엔 **trtexec 바이너리가 없다**(polygraphy만 옴, 3단계 실측 [`t01`](../experiments/stage3_tensorrt/t01_env.json)) | 빌드·벤치를 **polygraphy Python API**(`network_from_onnx_path`+`engine_from_network`)로. 버전은 `python3 -c "import tensorrt as trt; print(trt.__version__)"`. deb/JetPack이면 `/usr/src/tensorrt/bin/trtexec` |
| `trtexec: unrecognized option '--int8'` | **TensorRT 11.x**(정밀도 플래그 제거됨) | 정본 10.16.x로 맞추거나, 실습 2-B/4로 전환. INT8은 ModelOpt QDQ, FP16은 AutoCast로 모델에 심기 |
| `import modelopt.onnx…` → `Please install optional [onnx] dependencies` | `nvidia-modelopt[all]`인데도 `[onnx]` 엑스트라(onnxslim) 미충족(3단계 실측 [`t01`](../experiments/stage3_tensorrt/t01_env.json)) | `pip install "nvidia-modelopt[onnx]"`(또는 `pip install onnxslim`) 후 재확인. `modelopt.torch`는 영향 없음(2단계 §4.4에서 사용) |
| `IInt8EntropyCalibrator2` import/동작 실패 | 11.x에서 제거 / 10.1+ deprecated | 정본 10.16.x에서 사용(경고만), 신규는 ModelOpt PTQ(실습 4A) |
| INT8인데 mAP가 폭락 | 캘리브레이션 데이터 부족·비대표 / `--int8`만 주고 캘리브 생략 / 옛 `calib.cache` 재사용 | 대표 이미지 1000장+로 재캘리브(캐시 삭제 후), QDQ/QAT 사용 |
| `input has type Int32 but must have type FP8, FP4, Int4, or Int8` + `Invalid Node - <name>_bias_DequantizeLinear` | **아래 zero-point 행에서 파생되는 2차 증상.** `zero_point ≠ 0`으로 Q/DQ 융합이 깨진 뒤 **홀로 남은 bias DQ**가 타입 검사에 걸린 것이다. ORT는 `QuantizeBias` 기본 `True`라 bias를 INT32로 양자화해 DQ를 붙이는데, 융합이 정상이면 TRT는 이걸 그대로 받아들인다(절제 실험 case C: bias DQ 21개인 채로 빌드 성공) | **아래 zero-point 행을 먼저 고치고 재시도** — 그것만으로 이 에러도 같이 사라진다. 🔴 `extra_options={"QuantizeBias": False}`는 **이 에러의 해법이 아니다**: bias DQ를 0개로 만들어도 zero-point가 그대로면 여전히 실패·폴백한다(절제 실험 case B). 애초에 ModelOpt PTQ로 QDQ를 만들면 회피 → **2.2.1 / 실습 4-(A)·(C)** |
| `Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported` | 비대칭 양자화라 `zero_point ≠ 0`. TRT는 Q/DQ에 **대칭(zp=0)만** 허용 → **이 행이 진짜 하드 블로커다**(2×2 절제 실험으로 확인, 2.2.1) | `activation_type=QuantType.QInt8` **+** `extra_options={"ActivationSymmetric": True}`로 재양자화. 🔴 `QUInt8`인 채로 `ActivationSymmetric`만 켜면 **zp=127**이라 여전히 실패 → **실습 4-(C)** |
| INT8 QDQ 모델인데 TRT가 **에러 없이 FP32보다 느림** | 파서가 그래프를 통째로 거부 → ORT TensorRT EP가 **무음 폴백**(`get_providers()`엔 TRT가 그대로 보임). 실측 0.96 → **3.06 ms**(RTX 3060·ResNet18·batch=1·p50) | 같은 ONNX를 `trtexec`로 빌드해 파싱 에러를 눈으로 확인(EP와 달리 에러가 뜬다), ORT면 `log_severity_level=2`. **같은 모델을 CUDA EP로도 재서 TRT가 더 느리면 폴백 확정.** 원인은 십중팔구 위 zero-point 행 → **2.2.1** |
| `--fp16` 줬는데 여전히 FP32로 빌드됨 | 해당 GPU/레이어가 FP16 tactic이 없거나 정밀도 제약이 무시 | `--profilingVerbosity=detailed`로 tactic 확인, `--precisionConstraints=obey --layerPrecisions`로 강제(10.x) 또는 모델 캐스트(11.x) |
| `no implementation` / `no tactic` | workspace 부족 또는 미지원 포맷 | `--memPoolSize=workspace:4096`로 상한↑, 입력 IO 포맷 조정, 미지원 op면 plugin |
| Throughput이 이론치의 절반 | Enqueue 바운드(CPU 런치) 또는 H2D/D2H 병목 | Enqueue>Compute면 `--useCudaGraph`/배치↑, 전송이면 `--noDataTransfer`로 확인 후 전처리 GPU 이동 |
| p99가 mean보다 훨씬 큼 | GPU 경합 / 클럭 스로틀 / 워밍업 미제외 | `--iterations`↑로 steady state, `nvidia-smi`로 스로틀 확인, `--useSpinWait` |
| plugin `.so`가 옛 `IPluginV2` 기반이라 11.x에서 링크/등록 실패 | 11.0에서 V2 계열 제거 | `IPluginV3`+`IPluginCreatorV3One`으로 포팅(실습 6) |
| `--useDLACore` 무시/에러 | RTX엔 DLA 없음 / **11.x는 DLA 미지원** / JetPack 런타임 문제 | 정본 10.x + Orin에서만 실행. JetPack/L4T 버전과 TensorRT DLA 런타임 호환 확인 |
| DLA 켰는데 GPU-only보다 느림 | fallback 과다로 DLA↔GPU 복사 폭증 | `--dumpLayerInfo`로 fallback 레이어 특정 → 유발 op 치환(실습 5), subgraph를 덩어리로 |
| polygraphy 비교가 항상 FAIL | atol/rtol이 너무 타이트 / 동적 shape·전처리 불일치 | `--atol 1e-2 --rtol 1e-2`로 완화, 입력 shape·정규화 동일하게, `--save-inputs`로 golden 입력 고정 |
| `debug precision`의 N을 ONNX 노드로 못 붙임 | N은 TRT 내부 레이어 인덱스(노드와 1:1 아님) | `inspect model`·`exportProfile`·`mark all` 교차 대조로 구간 추정, 그 구간을 (b)로 좁힘 |
| 엔진이 다른 장비에서 로드 안 됨 | 엔진은 GPU 아키텍처·TRT 버전 종속 | 타깃에서 직접 빌드(RTX↔Orin 엔진 호환 불가) |
| `Could not find any implementation for node` | 그래프에 미지원 op | op 치환, 또는 custom plugin(실습 6) 작성 |

---

## 7) 산출물 (Deliverables)

이 단계가 끝나면 다음이 남아야 한다(포트폴리오/면접 근거):

- [ ] `latency_accuracy_matrix.csv` — FP32/FP16/INT8 × {GPU compute mean, **p99**, throughput, mAP50-95, FP32 대비 하락폭}
- [ ] `yolo11n_{fp32,fp16,int8_qdq}.engine` — 3점 엔진(빌드 로그 + `--exportProfile` JSON 포함)
- [ ] `trtexec_readout.md` — 한 엔진의 Throughput/Latency/Enqueue/GPU Compute/H2D·D2H를 줄별로 해석하고 병목 결론까지 적은 문서(4-(3) 형식)
- [ ] `polygraphy_diff.md` — FP32 vs INT8 레이어별 비교 요약, mismatch 시작 레이어, `debug precision`의 "앞쪽 N개 고정밀" 결론과 해석
- [ ] `calib_data/` + `calib.cache`(10.x) 또는 `yolo11n.quant.onnx`(ModelOpt) — 캘리브레이션 산출물
- [ ] `mixed_precision_plan.md` — [1단계](03_quantization_theory.md) `layer_sensitivity.csv` + `debug precision` 결과를 근거로 "어느 레이어를 FP16로 남겼는가"와 그 mAP 회복폭
- [ ] (보드 시) `dla_fallback.md` — GPU-only/DLA/하이브리드 비교표, fallback 레이어 목록(`--exportLayerInfo`), op 치환 전/후 fallback 수(없으면 '보드 필요 — 미실행'으로 명시)
- [ ] (심화) `deform_attn_plugin/` — IPluginV3 골격 소스(+Creator+REGISTER) + 빌드된 `.so`(최소 identity라도)
- [ ] (선택) `*.nsys-rep` — Nsight Systems 타임라인 캡처 1건 + 병목 유형 판정 메모

> 🟩 **3단계 실측 산출물 (이 저장소, 2026-08-17 · RTX 3080):** 위 체크리스트는 학습자용 목표이고, **이 문서의 실측 검증**은 아래로 남겼다 — 실측 리포트 [`logs/stage3_tensorrt_report.html`](../logs/stage3_tensorrt_report.html), 파서/빌더 제약 로그 원문+설계규칙 [`experiments/stage3_tensorrt/parser_constraints.md`](../experiments/stage3_tensorrt/parser_constraints.md), 재현 스크립트·JSON [`experiments/stage3_tensorrt/`](../experiments/stage3_tensorrt/README.md)(`t01`~`t04`). trtexec 부재 → polygraphy 3점(INT8 ×2.12), 직접 파서의 2개 하드 블로커, implicit 생존이 여기서 재현된다.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [NVIDIA TensorRT 문서(전체)](https://docs.nvidia.com/deeplearning/tensorrt/) — 개발자 가이드·API 레퍼런스 진입점
- [TensorRT 10.16 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html) — **정본 LTS**. deprecated API를 2027-03까지 유지
- [TensorRT Release Notes(최신 인덱스)](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html) — 버전별 변경·EOL 확인
- [Command-Line Programs (trtexec, 10.16)](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/reference/command-line-programs.html) — **trtexec 플래그·성능 지표(Throughput/Latency/Enqueue/GPU Compute/H2D·D2H) 정의**
- [Migrating trtexec from 10.x to 11.x](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html) — **제거된 정밀도 플래그(`--fp16/--int8/--best/--bf16/--fp8/--int4/--calib`)** 목록
- [onnx-tensorrt `operators.md` (10.16-GA)](https://github.com/onnx/onnx-tensorrt/blob/10.16-GA/docs/operators.md) — **TRT ONNX 파서가 받는 op·타입·제약의 1차 출처.** `DequantizeLinear`은 INT8/FP8/FP4/INT4만 받고 `x_zero_point`는 0이어야 한다([`main`=11.1 브랜치](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)도 동일) → 2.2.1
- [ONNX Runtime — Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — 1단계 QDQ 산출물을 가져올 때의 dtype 분기 근거. **"quantization on GPU only supports S8S8"** → 실습 4-(C)
- [Polygraphy — Debugging Accuracy(how-to)](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md) — layerwise 비교·모델 축소 결정 트리
- [Polygraphy debug reduce/precision 예제](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/02_reducing_failing_onnx_models) — `--mode bisect/linear`, `--check`, `--fail-regex` 실사용
- [Polygraphy(도구 루트)](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) — 정확도/레이어 디버깅 도구(2026-07 기준 0.49.x)
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html) · [Adding Custom Layers Using the C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html) — **IPluginV3 C++ 시그니처·등록**
- [IPluginV3 (Python API, 10.16)](https://docs.nvidia.com/deeplearning/tensorrt/10.16.1/_static/python-api/infer/Plugin/IPluginV3.html) · [10.x→11.x C++ 마이그레이션](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html) — **IPluginV2 제거·IPluginV3 전환**
- [Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html) · [DLA Supported Layers and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) — DLA 지원 범위/제약(Softmax Orin 전용, attention 불가, 동적 shape 불가)
- [jetson_dla_tutorial (NVIDIA-AI-IOT)](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial) — Orin DLA 실습 튜토리얼
- [TensorRT Model Optimizer(ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) · [onnx_ptq 예제](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq) — PTQ/QAT + QDQ ONNX export (`nvidia-modelopt`)
- [modelopt.onnx.quantization.quantize (API)](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.quantization.quantize.html) — `--quantize_mode/--calibration_method` 등 인자 레퍼런스
- [ModelOpt Changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html) — 버전별 인자/기능 변화 확인
- [ultralytics Export 문서](https://docs.ultralytics.com/modes/export) — YOLO ONNX export 옵션
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) — 커널 단위 타임라인 프로파일러
- [NVIDIA/TensorRT (GitHub)](https://github.com/NVIDIA/TensorRT) — 플러그인/샘플 소스

### 논문
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, arXiv:1712.05877 — INT8 정수 추론의 원조(캘리브레이션/스케일 이해의 기반)
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295 — PTQ/QAT·per-tensor vs per-channel 실전 정리
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630 — 양자화 기법 서베이

> ⚠️ 확인 필요: 위 버전·플래그·API는 **2026-07 기준(정본 TensorRT 10.16.x LTS)** 이다. 특히 (1) `modelopt.onnx.quantization` 인자명, (2) `IPluginV3` 세부 메서드 시그니처, (3) Polygraphy `debug precision` 옵션은 릴리스마다 변할 수 있으니 자신의 설치 버전 문서(`--help`)로 재확인하라.

---

## 9) 다음 단계

TensorRT로 NVIDIA 위에서 첫 완주를 마쳤다면, 이제 **하나의 모델을 여러 SoC로** 내보내며 벤더별 op 지원의 벽을 체험할 차례다.

➡️ [4단계 — 멀티 SoC (TIDL / QNN / DRP-AI)](06_multi_soc.md)
⬅️ 이전: [2단계 — Transformer 양자화](04_transformer_quantization.md)
