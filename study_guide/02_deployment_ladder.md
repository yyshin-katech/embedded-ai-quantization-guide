# 0.5. 쉬운 방법부터 — 배포 난이도 사다리

> 원본 가이드 매핑: "0.5단계 — 쉬운 방법부터: 배포 난이도 사다리"
> 예상 소요: Lv.1~4 합쳐 2~3주 (레벨당 반나절~며칠)
> 선행 조건: [0단계 환경 준비](01_environment_setup.md) 완료 (Ubuntu 22.04 + NVIDIA RTX GPU, Python/conda, Docker)

---

## 0) 이 단계에서 무엇을·왜 하는가

본 로드맵의 목적지인 **벤더 툴체인**(TensorRT / TI TIDL / Qualcomm QNN / Renesas DRP-AI)은 임베디드 AI에서 **가장 난이도가 높은 축**에 속한다. 처음부터 여기에 뛰어들면 "빌드가 안 된다 / op가 안 올라간다 / 보드가 없다"에서 몇 주를 태우고 정작 **"모델을 임베디드에 올린다"는 감각**을 얻지 못한다.

그래서 이 단계에서는 **난이도 사다리(ladder)** 를 오른다. 낮은 난이도의 툴로 **`변환 → 양자화 → 배포 → fallback(미지원 op 처리)`** 이라는 동일한 뼈대를 **4번 반복**한다. 반복할 때마다 추상화가 한 겹씩 벗겨지고, Lv.5(벤더 툴체인)에 도달할 즈음엔 파이프라인 전체가 손에 익는다.

- **왜 사다리인가 (직관)**: 각 레벨은 앞 레벨의 "확대판"이다. 같은 사진을 배율만 높여 보는 것과 같다. Lv.1(노코드)에서 **한 번의 클릭 뒤에 숨어 있던** `데이터→학습→INT8→배포`가, Lv.4에서는 **한 줄씩 손으로** 드러난다. 특히 ExecuTorch에서 겪는 **"export가 깨지는 경험"** 은 [2단계 Transformer 양자화](04_transformer_quantization.md)의 예고편이고, ONNX Runtime의 **EP 교체**는 [4단계 멀티 SoC](06_multi_soc.md)의 미니어처다. 이 두 연결을 각 레벨에서 의식적으로 확인하는 것이 이 단계의 진짜 목표다.
- **왜 지금인가 (Why now)**: 이론([1단계](03_quantization_theory.md))에 들어가기 전에 "완주 경험"을 만들어 두면, 이후 이론·최적화가 추상론이 아니라 **내가 이미 돌려본 것의 원리**로 읽힌다. 예를 들어 [1단계](03_quantization_theory.md)에서 배울 `scale`·`zero_point` 공식은, Lv.2에서 이미 `inp["quantization"]`로 만져 본 그 숫자다. **먼저 손으로 만지고 나중에 원리를 붙이는** 순서가 이 로드맵의 설계 의도다.

> 💡 팁: **TensorRT/ONNX 경험자는 Lv.3부터** 시작해도 된다. 완전 초심자는 Lv.1부터 순서대로. 단, "완주 경험"이 목적이므로 최소 한 레벨은 **끝까지(런타임 실행·수치 측정까지)** 통과할 것 — 중간에 멈추면 감각이 남지 않는다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] **Lv.1 Edge Impulse**: 웹 Studio에서 프로젝트 생성 → 데이터 수집 → Impulse 설계 → 학습 → INT8 TFLite 자동 양자화 → 배포까지 원클릭 완주. BYOM(ONNX 업로드) 경로로 내 모델도 올려본다.
- [ ] **Lv.2 LiteRT(구 TFLite)**: `TFLiteConverter`로 **PTQ 3종**(dynamic range / float16 / full INT8)을 **직접** 켜고 크기·정확도·속도를 표로 비교. `representative_dataset`을 손으로 작성한다. 인터프리터로 추론.
- [ ] **Lv.3 ONNX Runtime**: PyTorch → ONNX export 후 **EP 한 줄 교체**(CPU → CUDA → TensorRT)로 latency를 측정. **워밍업 + 반복측정 + p50/p95** 벤치 하네스를 작성하고 `provider_options`를 설정한다.
- [ ] **Lv.4 ExecuTorch**: `torch.export` → `to_edge_transform_and_lower`(XNNPACK) → `to_executorch()` → `.pte` 실행 최소 예제를 **단계별 산출물과 함께** 통과. 미지원 op에서 export가 깨지는 실제 에러를 관찰.
- [ ] **Qualcomm AI Hub**: `qai-hub`로 모델을 업로드해 **클라우드의 실제 단말**에서 컴파일(`submit_compile_job`) → 프로파일(`submit_profile_job`) → 결과 조회까지 lifecycle을 돈다(보드 없이 QNN 감각 체험).
- [ ] 4번의 반복에서 공통 뼈대(`변환→양자화→배포→fallback`)를 스스로 서술할 수 있다.

---

## 2) 배경 이론 / 개념 — 난이도 사다리 한눈에

각 레벨은 "추상화 수준"과 "직접 만지는 범위"가 다르다. 위로 갈수록 자동화가 걷히고 벤더 네이티브에 가까워진다.

| Lv | 도구 | 난이도 | 입력 | 핵심 학습 포인트 | 보드 필요? | 검증 버전 (2026-07 기준) |
|----|------|--------|------|------------------|-----------|--------------------------|
| 1 | **Edge Impulse** | ★☆☆☆☆ | 데이터 or 모델(BYOM) | 노코드로 전 과정 자동화, INT8 자동 양자화, QNN/Hexagon 배포 | 배포 시 (RPi/Arduino/폰), 학습은 클라우드 | `edge-impulse-cli` 최신 |
| 2 | **LiteRT** (구 TFLite) | ★★☆☆☆ | Keras/TF | PTQ 3종을 **직접** 켠다(`representative_dataset`), 인터프리터 추론 | RPi 있으면 좋음(PC로도 가능) | `ai-edge-litert` 최신 |
| 3 | **ONNX Runtime** | ★★★☆☆ | PyTorch/TF→ONNX | **EP 교체 한 줄**로 백엔드 스위칭(CPU/CUDA/TensorRT/QNN) | 불필요(RTX PC) | `onnxruntime-gpu` 1.23.2 (CUDA 12 wheel) + `onnx` 1.18.0 |
| 4 | **ExecuTorch** | ★★★★☆ | PyTorch | `torch.export` 직행, 미지원 op에서 export 붕괴 체험 | 불필요(PC)~선택(폰/MCU) | `executorch` 1.3.x |
| 5 | **벤더 툴체인** | ★★★★★ | 벤더별 상이 | 최대 성능·최대 난이도 = **본 가이드 [1~5단계](03_quantization_theory.md)** | 대부분 필요 | (각 단계 문서 참조) |

> 💡 팁: **공통 뼈대**를 매 레벨에서 의식적으로 찾아라 —
> `① 원본 모델 → ② (변환) 배포 포맷 → ③ (양자화) INT8 → ④ (배포) 런타임에서 실행 → ⑤ (fallback) 미지원 연산 처리`.
> Lv.1은 ①~⑤를 전부 숨기고, Lv.4는 대부분 드러낸다. 아래는 레벨별로 각 단계가 "어디에 있는지"를 매핑한 것이다.

| 뼈대 단계 | Lv.1 Edge Impulse | Lv.2 LiteRT | Lv.3 ONNX Runtime | Lv.4 ExecuTorch |
|-----------|-------------------|-------------|-------------------|-----------------|
| ② 변환 | Studio가 자동 | `TFLiteConverter` | `torch.onnx.export` | `torch.export` + `to_edge_*` |
| ③ 양자화 | `Quantized(int8)` 클릭 | `optimizations`/`representative_dataset` | (EP 옵션 `trt_int8_enable`) | PT2E(`prepare/convert_pt2e`) |
| ④ 배포/실행 | 펌웨어/라이브러리 | `Interpreter`/`CompiledModel` | `InferenceSession` | `.pte` + `Runtime` |
| ⑤ fallback | Studio가 처리 | (해당 op를 float로) | `providers` 우선순위 리스트 | partitioner 미할당 → CPU |

> ⚠️ 주의: **양자화·NPU 최적화의 최종 성능은 벤더 네이티브(Lv.5)가 앞선다.** 사다리는 "감각과 속도"를 얻는 훈련이지, 프로덕션 최적해가 아니다. 예컨대 Lv.3의 TensorRT EP는 편의성(한 줄 교체)을 위해 세밀한 커널 튜닝·레이어 융합 제어를 포기한다 — 그 제어권을 되찾는 것이 [3단계 TensorRT](05_tensorrt.md)다.

---

## 3) 환경·도구 준비

[0단계](01_environment_setup.md)에서 만든 conda 환경을 재사용한다. 레벨별로 **가상환경을 분리**하면 의존성 충돌(특히 TF vs PyTorch, 그리고 onnxruntime CPU판 vs GPU판)을 피할 수 있다.

```bash
# 레벨별 독립 conda 환경 (권장) — 이름은 자유
conda create -y -n ladder-litert  python=3.11   # Lv.2 (LiteRT/TF)
conda create -y -n ladder-onnx     python=3.11   # Lv.3 (ONNX Runtime)
conda create -y -n ladder-et       python=3.11   # Lv.4 (ExecuTorch), qai-hub도 여기서
```

```bash
# Lv.1 Edge Impulse CLI — Node.js 20 LTS 필요 (v16+). npm 전역 설치
#  (Ubuntu에 Node가 없다면 nvm 또는 nodesource로 20.x 설치 후)
node -v                                 # v20.x 이상인지 확인
npm install -g edge-impulse-cli         # edge-impulse-daemon/uploader/runner 등 포함 (2026-07 기준 최신)
edge-impulse-daemon --version           # 설치 확인
```

> 💡 팁: `npm install -g edge-impulse-cli`는 여러 실행파일을 한 번에 깐다 — `edge-impulse-daemon`(보드↔Studio 프록시), `edge-impulse-uploader`(데이터 업로드), `edge-impulse-runner`(배포 모델 로컬 실행), `edge-impulse-data-forwarder`(센서 스트리밍). 뒤 실습에서 daemon과 uploader를 쓴다.

```bash
# Lv.2 LiteRT (구 TensorFlow Lite. 2024년 개명) — 런타임 전용 경량 패키지
conda activate ladder-litert
pip install ai-edge-litert              # 인터프리터/런타임 (2026-07 기준 최신)
pip install tensorflow                  # TFLiteConverter(변환기)는 TF 본체에 포함
```

```bash
# Lv.3 ONNX Runtime (GPU) — 이 스터디는 CUDA 12.8 스택에 맞춰 CUDA 12 대응 wheel 사용
#   (2026-07 기준 정본은 onnxruntime-gpu 1.23.2. PyPI 기본 wheel은 1.27부터 CUDA 13이므로
#    상한 '<1.27'로 CUDA 12 라인에 묶는다. onnx도 1.18.0으로 고정해야 ORT의 IR 11 상한과
#    맞는다 — 무제한 설치는 1.22.0(IR 13)을 깔아 로드가 깨진다. 근거는 0단계 2절 참조)
conda activate ladder-onnx
pip install torch torchvision           # PyTorch (CUDA 빌드는 pytorch.org 인덱스 사용)
pip install "onnx==1.18.0" "onnxruntime-gpu<1.27"   # ONNX + GPU 런타임(CPU/CUDA/TensorRT EP 포함)
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# 예상 출력:
#   1.23.2
#   ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# ↑ 세 EP가 모두 보이면 OK. CUDA/TensorRT가 없으면 CPU판이 깔린 것 — 6)절 참조.
```

```bash
# Lv.4 ExecuTorch — PyTorch 네이티브 온디바이스 런타임 (v1.0 GA 2025-10)
conda activate ladder-et
pip install executorch                  # (2026-07 기준 1.3.x, XNNPACK 등 백엔드 포함)
python -c "import executorch; print(executorch.__version__)"
# 예상 출력: 1.3.x
```

```bash
# Qualcomm AI Hub — 클라우드 실단말 컴파일/프로파일링
pip install "qai-hub[torch]"            # PyTorch 트레이싱/export 포함 설치
# 가입 후 https://app.aihub.qualcomm.com 의 Account에서 API 토큰 발급받아:
qai-hub configure --api_token <YOUR_API_TOKEN>
python -c "import qai_hub as hub; print(len(hub.get_devices()), 'devices available')"
# 예상 출력: (토큰이 유효하면) 예: '80 devices available'  ← 숫자는 시점에 따라 다름
```

> ⚠️ 확인 필요: LiteRT의 신규 권장 추론 경로는 `CompiledModel` API이며(델리게이트를 손으로 다루지 않고 NPU 가속에 접근), 예전 `Interpreter` API는 "하위 호환용으로 유지"된다고 명시돼 있다(출처: [LiteRT inference 문서](https://developers.google.com/edge/litert/inference)). 아래 Lv.2 실습은 학습용으로 익숙한 **`Interpreter` API**를 먼저 보이고, `CompiledModel` 경로도 병기한다. 최신 프로덕션에서는 `CompiledModel`을 우선 검토할 것.

> ⚠️ 확인 필요: `onnxruntime-gpu`의 **TensorRT EP**는 실행 시 시스템에 TensorRT/cuDNN이 맞는 버전으로 있어야 동작한다. 이 스터디의 정본은 **TensorRT 10.16.x LTS**(2026-03 릴리스, 이 버전에서 deprecated된 API는 2027-03까지 유지되는 LTS 라인)다. 버전 호환 매트릭스는 [ONNX Runtime TensorRT EP 문서](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)에서 설치 시점 기준으로 재확인.

---

## 4) 단계별 실습

### Lv.1 — Edge Impulse (노코드/로우코드, ★☆☆☆☆)

**흐름**: 계정 → 프로젝트 → 데이터 → Impulse 설계 → 학습 → **INT8 배포**. 전 과정이 웹 UI([studio.edgeimpulse.com](https://studio.edgeimpulse.com))에서 일어난다. 2025년 Qualcomm 인수 이후 **Dragonwing + Hexagon NPU(QNN) 배포**가 공식 지원된다.

Lv.1의 목적은 "결과물"이 아니라 **완주 감각**이다. 여기서 자동으로 지나가는 각 화면이 Lv.2~4에서 손으로 재현할 대상이므로, **화면 전환마다 "이건 뼈대의 어느 단계인가"를 메모**하며 진행한다.

#### (실행 흐름) 데이터부터 INT8 배포까지 — 화면 순서

1. **계정·프로젝트 생성**: [studio.edgeimpulse.com](https://studio.edgeimpulse.com) 가입 → 좌상단 **"Create new project"** → 이름 입력. 프로젝트 대시보드가 열린다. (뼈대 ①: 원본 모델의 "빈 그릇"이 만들어진 상태)
2. **데이터 수집(Data acquisition)**: 좌측 메뉴 `Data acquisition`. 두 경로 중 하나 —
   - **웹 업로드**: `Upload data` 버튼으로 파일(오디오 wav/이미지 등)을 끌어 놓기. 자동으로 train/test로 분할된다.
   - **CLI 스트리밍**: 보드/PC를 Studio에 연결해 센서 데이터를 실시간 수집.
     ```bash
     # 보드/PC를 Studio 프로젝트에 연결 (시리얼 장치 프록시)
     edge-impulse-daemon
     # 프롬프트에서: 계정 로그인 → 연결할 프로젝트 선택 → 디바이스 이름 지정
     # 연결되면 Studio의 'Devices' 탭에 초록불로 나타난다.
     ```
     ```bash
     # (대안) 이미 있는 파일을 CLI로 일괄 업로드 — 라벨은 디렉토리명으로 자동 부여
     edge-impulse-uploader --category training  data/train/yes/*.wav
     edge-impulse-uploader --category testing   data/test/yes/*.wav
     ```
3. **Impulse 설계(Create impulse)**: 좌측 `Impulse design > Create impulse`. 좌→우 파이프라인을 조립한다:
   `Input(시간창 크기·stride) → Processing block(전처리: 예 MFCC/Spectrogram/Image) → Learning block(분류/회귀 등) → Output(클래스)`. `Save Impulse` 클릭.
4. **특징 생성(Generate features)**: Processing 블록 페이지에서 `Generate features` → 특징 공간 산점도가 그려진다(클래스가 잘 분리되면 학습이 쉽다는 신호).
5. **학습(Start training)**: Learning 블록 페이지에서 epoch·learning rate·모델 아키텍처를 고르고 **`Start training`**. 학습은 **클라우드 GPU**에서 돌아 로컬 RTX가 필요 없다. 끝나면 **혼동행렬 + 검증 정확도 + on-device 성능 추정(RAM/Flash/추론시간)**이 표시된다. (뼈대 ①→ 학습 완료)
6. **INT8 배포(Deployment)**: 좌측 `Deployment` 탭 → 타깃 선택.
   - **라이브러리**: `C++ library` / `Arduino library` / `WebAssembly` 등.
   - **보드 펌웨어**: 지원 보드를 고르면 **원클릭 빌드**로 `.bin`/`.uf2` 펌웨어 생성.
   - 화면 하단 **`Model optimizations`에서 `Quantized (int8)` 선택** → 빌드 시 **INT8 TFLite로 자동 PTQ**되어 산출물에 포함된다. `Unoptimized (float32)`와 나란히 놓고 추정 RAM/추론시간을 비교해 보라(여기서 이미 "INT8이 작고 빠르다"가 숫자로 보인다 — 뼈대 ③).
   - Qualcomm 타깃(Dragonwing/Android)에서는 **QNN 델리게이트**로 Hexagon NPU에 라우팅(뼈대 ⑤가 자동으로 처리됨).
7. **보드에서 실행**: 생성한 펌웨어를 플래시하거나, 라이브러리를 프로젝트에 링크. 로컬에서 빠르게 확인하려면:
   ```bash
   # 배포된 모델을 PC/보드에서 즉시 실행 (별도 빌드 없이 감각 확인)
   edge-impulse-runner
   ```

#### (예상 결과) 소형 키워드 인식(예: "yes/no") 기준

| 항목 | Unoptimized (float32) | Quantized (int8) |
|------|-----------------------|------------------|
| 모델 크기(Flash) | 예: ~80 KB | 예: ~22 KB (≈1/4) |
| 예상 RAM | 예: ~12 KB | 예: ~8 KB |
| 검증 정확도 | 예: 96.5% | 예: 96.0% (미세 하락) |
| 추론시간(Cortex-M 추정) | 예: ~40 ms | 예: ~12 ms |

> 수치는 예시다. Studio의 on-device 성능 추정치는 배포 타깃(Cortex-M4/M7/RPi/Dragonwing)에 따라 달라진다.

#### BYOM (Bring Your Own Model) — 내 ONNX를 올리는 경로 상세

이미 학습된 모델(예: [Lv.3](#lv3--onnx-runtime-ep-한-줄-교체-)에서 만든 `resnet18.onnx`)이 있으면 Studio가 프로파일링·양자화·타깃 빌드를 대신 처리한다.

1. 대시보드 우상단 또는 `Deployment > Upload your model` 진입.
2. **모델 파일 업로드**: 지원 포맷 **SavedModel(.zip) · ONNX(.onnx) · TFLite/LiteRT(.tflite) · scikit-learn(.pkl)**.
3. **입력/출력 스펙 지정**: 입력 텐서 형상(예 `1,3,224,224`)과 스케일링(0–255 → 0–1 등), 출력 타입(분류/회귀)을 폼에서 지정. 여기서 형상이 틀리면 프로파일이 실패하니 export 때의 `input_names`/shape와 일치시킨다.
4. **프로파일 & 양자화**: 업로드하면 Studio가 대상 하드웨어별 추론시간·메모리를 추정하고, `int8` 옵션 선택 시 대신 양자화한다. **Dragonwing에서는 양자화 모델이 Hexagon NPU로 가속**된다.
5. **배포**: 이후는 위 6~7단계와 동일(라이브러리/펌웨어).

> 💡 팁: BYOM은 Lv.1과 Lv.3을 잇는 다리다. Lv.3에서 만든 `resnet18.onnx`를 그대로 올려 보면, "같은 모델이 노코드 파이프라인에서는 어떻게 취급되는지"를 대조할 수 있다. **한 아티팩트, 두 경로**를 경험하는 것이 사다리의 요령이다.

> 🔴 함정: BYOM 업로드 시 **전처리(정규화)를 모델 밖에서 하고 있었다면** Studio 폼에 그 스케일링을 명시해야 결과가 맞는다. "정확도가 랜덤 수준"이면 대개 입력 스케일/채널 순서(NCHW vs NHWC) 불일치다.

---

### Lv.2 — LiteRT (구 TFLite) + PTQ 3종 직접 켜기 (★★☆☆☆)

Lv.1이 한 번의 클릭 뒤에 숨긴 **양자화**를, 여기서는 **세 가지 방식**으로 직접 켜서 서로 비교한다. 2024년 TFLite가 **LiteRT**로 개명됐다. 변환기(`TFLiteConverter`)는 TF 본체에, 경량 런타임은 `ai-edge-litert`에 있다.

**세 가지 PTQ의 직관 (왜 3종인가)**

| 방식 | 무엇을 양자화하나 | 크기 | 대표 타깃 | representative_dataset |
|------|------------------|------|-----------|------------------------|
| **Dynamic range** | 가중치=INT8, **activation은 float(런타임에 동적 양자화)** | ~1/4 | CPU (가장 간단) | **불필요** |
| **Float16** | 가중치=FP16, activation=FP32(첫 추론 전 FP32로 업샘플) | ~1/2 | **GPU** | 불필요 |
| **Full INT8** | 가중치·activation **모두 INT8**(정수만) | ~1/4 | MCU/NPU/DSP | **필요(캘리브레이션)** |

- **Dynamic range**가 가장 쉽다(플래그 하나). 그러나 activation이 여전히 float라 정수-only 하드웨어(MCU/NPU)에는 못 올린다.
- **Float16**은 정확도 손실이 거의 없고 GPU에서 유리하지만, INT8만큼 작지도 빠르지도 않다.
- **Full INT8**만이 진짜 "정수 전용" 모델이다 — 대신 activation 범위를 알아야 하므로 **`representative_dataset`(캘리브레이션)** 이 필수다. 이 셋의 차이를 손으로 만든 표로 확인하는 것이 Lv.2의 핵심이며, 이론은 [1단계](03_quantization_theory.md)에서 정밀하게 다룬다.

#### (a) 모델 준비 + PTQ 3종 변환

`representative_dataset`이 캘리브레이션의 축소판이다 — 실제 입력 분포에서 뽑은 소량 샘플로 activation의 min/max 범위를 추정한다.

```python
# lv2_convert.py — Keras 모델 → FP32 / dynamic-range / float16 / full-INT8 .tflite (4종)
import numpy as np
import tensorflow as tf

# 0) 예시 모델 (실제로는 학습된 모델을 로드). MNIST용 소형 CNN.
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])
# (학습 생략: 데모용. 실제로는 model.fit(...)으로 학습된 가중치 사용)

def save(name, tflite_bytes):
    with open(name, "wb") as f:
        f.write(tflite_bytes)
    print(f"  {name:26s} {len(tflite_bytes)/1024:7.1f} KB")

# 1) FP32 baseline
conv = tf.lite.TFLiteConverter.from_keras_model(model)
save("model_fp32.tflite", conv.convert())

# 2) Dynamic range PTQ — 플래그 하나. representative_dataset 불필요.
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]              # 가중치만 INT8, activation은 동적
save("model_dynamic.tflite", conv.convert())

# 3) Float16 PTQ — GPU 친화. representative_dataset 불필요.
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]             # 가중치 FP16
save("model_fp16.tflite", conv.convert())

# 4) Full INT8 PTQ — representative_dataset로 activation 범위 캘리브레이션
#    (학습/평가셋에서 뽑은 수백 샘플이면 충분)
calib_images = np.random.rand(200, 28, 28, 1).astype("float32")  # 데모용 랜덤

def representative_data_gen():
    # 각 yield는 '모델 입력 하나'를 담은 리스트. 배치 1로 200개 흘려보낸다.
    for img in tf.data.Dataset.from_tensor_slices(calib_images).batch(1).take(200):
        yield [img]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]                       # 양자화 스위치 ON
conv.representative_dataset = representative_data_gen                  # 캘리브레이션 데이터
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # full-INT8 강제
conv.inference_input_type = tf.int8                                   # 입출력도 정수로
conv.inference_output_type = tf.int8
save("model_int8.tflite", conv.convert())

# 예상 출력 (크기는 모델에 따라 다름):
#   model_fp32.tflite            ~13.0 KB
#   model_dynamic.tflite          ~5.0 KB
#   model_fp16.tflite             ~7.0 KB
#   model_int8.tflite             ~4.0 KB
```

> 🔴 함정: `representative_dataset`을 주지 않고 `optimizations=[DEFAULT]`만 켜면 그건 **full INT8이 아니라 dynamic range**다(activation은 여전히 float). MCU/NPU처럼 정수-only 하드웨어에 올리려면 반드시 `supported_ops=[...INT8]` + `representative_dataset` + `inference_input/output_type=tf.int8` 3종 세트가 필요하다. 이 차이가 임베디드 배포 실패의 단골 원인이다 — [1단계 양자화 이론](03_quantization_theory.md)에서 정확히 다룬다.

> 🔴 함정: `representative_data_gen`은 **`yield [img]`처럼 리스트**로 내보내야 한다(모델 입력이 여러 개면 원소도 여러 개). `yield img`로 바로 텐서를 주면 `ValueError`가 난다. 또 dtype은 `float32`여야 한다(정수화는 변환기가 내부적으로 수행).

#### (b) 추론 — 인터프리터 vs CompiledModel

**(b-1) Interpreter API** (학습용·하위 호환) — Raspberry Pi에서도 동일 코드가 돈다(`ai-edge-litert`는 ARM 휠 제공).

```python
# lv2_infer.py — LiteRT 인터프리터로 추론 (PC/RPi 공통)
import numpy as np
from ai_edge_litert.interpreter import Interpreter   # (구 tf.lite.Interpreter 대체)

def run(path, x):
    itp = Interpreter(model_path=path)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    # INT8 모델이면 입력을 양자화 스케일로 정수화 (역양자화 공식의 역함수)
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]          # ← [1단계]에서 배울 scale/zero_point가 여기 있다
        x = np.round(x / scale + zp).astype(np.int8)
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    y = itp.get_tensor(out["index"])
    # 출력이 INT8이면 다시 float로 역양자화
    if out["dtype"] == np.int8:
        oscale, ozp = out["quantization"]
        y = (y.astype(np.float32) - ozp) * oscale
    return y

sample = np.random.rand(1, 28, 28, 1).astype("float32")
for m in ["model_fp32.tflite", "model_dynamic.tflite", "model_fp16.tflite", "model_int8.tflite"]:
    print(f"{m:22s} -> argmax={int(np.argmax(run(m, sample)))}")

# 예상 출력 (argmax는 랜덤 가중치라 무의미하지만, 4종 모두 에러 없이 도는지가 관건):
#   model_fp32.tflite      -> argmax=7
#   model_dynamic.tflite   -> argmax=7
#   model_fp16.tflite      -> argmax=7
#   model_int8.tflite      -> argmax=7
```

**(b-2) CompiledModel API** (신규 권장 경로) — 델리게이트를 손으로 다루지 않고 가속기(CPU/GPU/NPU)를 고른다.

```python
# lv2_infer_compiled.py — LiteRT CompiledModel 경로 (신규 권장)
import numpy as np
from ai_edge_litert import interpreter as litert   # 패키지 최신 버전 기준

# CompiledModel은 accelerator를 문자열로 고른다 ('cpu' / 'gpu' / 'npu' 등, 지원 시)
cm = litert.CompiledModel("model_fp32.tflite", accelerator="cpu")
inputs = [np.random.rand(1, 28, 28, 1).astype("float32")]
outputs = cm.run(inputs)
print("CompiledModel out shape:", outputs[0].shape)   # 예상: (1, 10)
```

> ⚠️ 확인 필요: `CompiledModel`의 정확한 임포트 경로·생성자 인자(`accelerator` 키명 등)는 `ai-edge-litert` 버전에 따라 다르다. 설치한 버전의 [LiteRT inference 문서](https://developers.google.com/edge/litert/inference)에서 시그니처를 재확인하고, 안 맞으면 학습용으로는 (b-1) `Interpreter`로 진행할 것.

#### (c) PTQ 3종 비교 — 이 단계의 핵심 산출물

```bash
# 파일 크기 한눈에 비교 (INT8/dynamic은 ~1/4, fp16은 ~1/2로 줄어드는지)
ls -lh model_fp32.tflite model_dynamic.tflite model_fp16.tflite model_int8.tflite
```

정확도(테스트셋 accuracy)와 추론 시간(`time.perf_counter()`로 워밍업 후 N회 평균)을 각각 측정해 아래 [5)절 표](#5-예시--결과-해석)의 **4행 표**를 내 값으로 채운다. 측정 스니펫:

```python
# lv2_bench.py — 워밍업 후 반복측정으로 추론시간 (모델 하나당)
import time, numpy as np
from ai_edge_litert.interpreter import Interpreter

def latency_ms(path, n=200):
    itp = Interpreter(model_path=path); itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    x = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(20):                      # warm-up
        itp.set_tensor(inp["index"], x); itp.invoke()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        itp.set_tensor(inp["index"], x); itp.invoke()
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return ts.mean(), np.percentile(ts, 50), np.percentile(ts, 95)

for m in ["model_fp32.tflite", "model_dynamic.tflite", "model_fp16.tflite", "model_int8.tflite"]:
    avg, p50, p95 = latency_ms(m)
    print(f"{m:22s} avg={avg:.3f} p50={p50:.3f} p95={p95:.3f} ms")
```

> 💡 팁: **PC(x86)에서는 INT8이 반드시 빠르지 않다.** 데스크톱 CPU는 float 처리가 매우 최적화돼 있어, INT8 이득은 오히려 **ARM/MCU/NPU**에서 두드러진다. 가능하면 이 벤치를 **Raspberry Pi에서도** 돌려 두 환경의 상대속도가 뒤집히는 것을 관찰하라 — 이것이 "타깃 하드웨어에서 측정하라"는 임베디드 철칙의 첫 체감이다.

---

### Lv.3 — ONNX Runtime: EP 한 줄 교체 (★★★☆☆)

[4단계 멀티 SoC](06_multi_soc.md)의 미니어처. **모델은 그대로 두고 Execution Provider(EP)만 바꿔** 백엔드를 스위칭한다. Lv.2의 "PTQ 방식을 바꾼다"와 달리, 여기서는 "실행 엔진을 바꾼다" — 모델 파일은 하나(`resnet18.onnx`)로 고정이다.

#### (a) PyTorch → ONNX export

```python
# lv3_export.py — torchvision 모델을 ONNX로 내보내기
import torch, torchvision

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "resnet18.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # 배치 가변
    opset_version=17,
)
print("saved resnet18.onnx")
# (선택) 그래프 유효성 검사
import onnx; onnx.checker.check_model("resnet18.onnx"); print("onnx check: OK")
# 예상 출력:
#   saved resnet18.onnx
#   onnx check: OK
```

#### (b) EP 벤치 하네스 — 워밍업 + 반복측정 + p50/p95 통계

Lv.3의 진짜 산출물은 "숫자 하나"가 아니라 **믿을 수 있는 벤치 하네스**다. 아래 하네스는 (1) 최초 엔진 빌드를 제외하는 **워밍업**, (2) 이상치에 강한 **p50/p95 분위수**, (3) EP별 `provider_options` 설정을 모두 담는다. 바뀌는 것은 `providers` 리스트뿐 — 이 "한 줄 교체"가 핵심이다.

```python
# lv3_bench.py — CPU vs CUDA vs TensorRT EP를 워밍업+반복측정으로 비교
import time, numpy as np, onnxruntime as ort

MODEL = "resnet18.onnx"
x = {"input": np.random.rand(1, 3, 224, 224).astype("float32")}

def bench(name, providers, warmup=15, iters=100):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(MODEL, sess_options=so, providers=providers)
    active = sess.get_providers()[0]                 # 실제로 최우선 배치된 EP
    for _ in range(warmup):                          # TensorRT는 최초에 엔진 빌드로 매우 느림 → 반드시 제외
        sess.run(None, x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, x)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    print(f"{name:9s} active={active:26s} "
          f"p50={np.percentile(ts,50):6.2f}  p95={np.percentile(ts,95):6.2f}  "
          f"avg={ts.mean():6.2f}  min={ts.min():6.2f} ms")

# --- CPU ---
bench("CPU", ["CPUExecutionProvider"])

# --- CUDA (provider_options로 device 지정) ---
bench("CUDA", [("CUDAExecutionProvider", {"device_id": 0}),
               "CPUExecutionProvider"])

# --- TensorRT (FP16 + 엔진 캐시로 재빌드 회피). 미지원 노드는 CUDA→CPU로 fallback ---
bench("TensorRT", [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,          # FP16 커널 허용 (정확도 허용 범위면 큰 이득)
        "trt_engine_cache_enable": True,  # 빌드한 엔진을 디스크에 캐시
        "trt_engine_cache_path": "./trt_cache",
        "trt_max_workspace_size": 2 * 1024**3,   # 2 GB
        # "trt_int8_enable": True,        # INT8까지 가려면 캘리브레이션 캐시 필요 (여기선 생략)
    }),
    ("CUDAExecutionProvider", {"device_id": 0}),  # TRT가 못 맡는 노드 fallback
    "CPUExecutionProvider",                         # 그래도 안 되면 CPU
])

# 예상 출력 (RTX 데스크톱, 값은 GPU/드라이버에 따라 다름):
#   CPU       active=CPUExecutionProvider        p50= 28.40  p95= 33.10  avg= 29.02  min= 26.9 ms
#   CUDA      active=CUDAExecutionProvider       p50=  2.90  p95=  3.40  avg=  3.01  min=  2.7 ms
#   TensorRT  active=TensorrtExecutionProvider   p50=  1.45  p95=  1.80  avg=  1.52  min=  1.3 ms
```

> 💡 팁: `providers`는 **우선순위 리스트**다. `["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]`처럼 주면 앞 EP가 못 맡는 노드를 뒤 EP가 **fallback**으로 처리한다 — 공통 뼈대의 ⑤(fallback)가 여기서 명시적으로 드러난다. 이 "안 되면 뒤로 넘긴다"는 발상이 [4단계](06_multi_soc.md)에서 SoC별 델리게이트/파티셔너로 확장된다.

> 💡 팁: **왜 p95까지 보나.** avg만 보면 어쩌다 튄 한 번의 지연이 묻힌다. 임베디드/서빙에서는 "최악에 가까운" p95/p99가 사용자 체감을 좌우하므로, 벤치는 처음부터 분위수로 보고하는 습관을 들인다.

> ⚠️ 주의: **TensorRT EP의 첫 세션은 엔진 빌드로 수 초~수십 초 걸린다**(정상). `trt_engine_cache_enable=True` + `trt_engine_cache_path`를 주면 다음 실행부터 캐시를 재사용해 빨라진다. 입력 shape가 바뀌면 엔진을 다시 빌드하므로, 벤치는 shape를 고정하고 워밍업을 충분히 준다.

> ⚠️ 주의: **QNN EP**(Qualcomm Hexagon)와 **OpenVINO EP**는 기본 `onnxruntime-gpu` 휠에 없다. QNN EP는 별도 패키지(예: `onnxruntime-qnn`)로 설치하며 보통 **QNN SDK + ARM64 환경**이 필요하다. RTX 데스크톱에서는 CPU/CUDA/TensorRT로 감각을 익히고, QNN은 아래 **Qualcomm AI Hub**로 대체 체험한다.

> ⚠️ 확인 필요: QNN EP 설치 명령은 시점에 따라 달라진다. 2026-07 검색 기준 나이틀리 피드(`onnxruntime-qnn`)가 안내되나, 정식 채널/패키지명은 [ONNX Runtime 설치 문서](https://onnxruntime.ai/docs/install/)와 [QNN EP 문서](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)에서 설치 시점 기준 재확인 필요.

---

### Lv.4 — ExecuTorch: PyTorch 네이티브 (★★★★☆)

Meta의 온디바이스 런타임. **v1.0 GA는 2025-10**, 2026-07 기준 최신은 **v1.3.x**. 기본 런타임 풋프린트가 매우 작고(수십 KB급), 백엔드가 12개+(XNNPACK/QNN/CoreML/Ethos-U/Vulkan/OpenVINO 등). ONNX를 우회하고 **`torch.export`로 직행**하는 것이 특징이다 — Lv.3처럼 중간 IR(ONNX) 파일을 만들지 않고 PyTorch 그래프를 곧장 `.pte`로 굳힌다.

#### (a) export → 로우어링 → .pte — 단계별 산출물

파이프라인은 3단계이고, **각 단계가 서로 다른 객체를 산출**한다. 이 "객체가 무엇으로 바뀌는가"를 이해하는 것이 Lv.4의 절반이다.

| 단계 | 함수 | 산출물(타입) | 무엇이 결정되나 |
|------|------|--------------|-----------------|
| ① 그래프 캡처 | `torch.export.export(model, inputs)` | `ExportedProgram` | 동적 shape·제어흐름이 **여기서** 확정. 미지원 op면 **여기서 깨진다** |
| ② Edge 변환+로우어링 | `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])` | `EdgeProgramManager` | 어떤 노드를 XNNPACK로 넘기고 어떤 노드를 CPU에 남길지(파티셔닝=뼈대 ⑤) |
| ③ 직렬화 | `.to_executorch()` → `.buffer` | `ExecutorchProgramManager` / bytes | 최종 `.pte` 바이트. 파일로 쓰면 끝 |

```python
# lv4_export.py — PyTorch → ExecuTorch .pte (XNNPACK 백엔드), 단계별 산출물 관찰
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)

# ① torch.export로 그래프 캡처 → ExportedProgram
exported = export(model, sample_inputs)
print("① ExportedProgram nodes:", len(list(exported.graph.nodes)))

# ② Edge IR 변환 + XNNPACK로 로우어링 → EdgeProgramManager, 이어서 ③ .to_executorch()
et_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

# ③ .pte 직렬화 (buffer = 최종 바이트)
with open("mobilenet_v2.pte", "wb") as f:
    f.write(et_program.buffer)

import os
print("③ mobilenet_v2.pte:", round(os.path.getsize("mobilenet_v2.pte")/1024/1024, 2), "MB")
# 예상 출력:
#   ① ExportedProgram nodes: ~180   (모델·버전에 따라 다름)
#   ③ mobilenet_v2.pte: ~13.9 MB    (FP32. INT8 PT2E를 걸면 ~1/4로 줄어듦)
```

> 💡 팁 (산출물 크기의 의미): FP32 `.pte`는 원본 가중치를 거의 그대로 담아 ~14 MB다. 아래 (c)의 **PT2E INT8** 흐름을 얹으면 ~3.5 MB 수준으로 떨어진다 — 여기서 "양자화가 곧 파일 크기"라는 뼈대 ③이 눈에 보인다.

#### (b) .pte 실행 (Python 런타임 바인딩)

```python
# lv4_run.py — 저장한 .pte를 Python에서 로드·실행
import torch
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program("mobilenet_v2.pte")
method = program.load_method("forward")
outputs = method.execute([torch.randn(1, 3, 224, 224)])
print("output shape:", outputs[0].shape)   # 예상: torch.Size([1, 1000])
```

#### (c) (선택) INT8까지 — PT2E 양자화 흐름

XNNPACK은 CPU 백엔드이며 INT8을 지원한다. 여기까지 가려면 export **전에** PT2E(`prepare_pt2e`/`convert_pt2e`)로 그래프를 양자화한다. 개념 골격:

```python
# lv4_quantize.py (골격) — PT2E INT8 후 XNNPACK 로우어링
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

# 1) export_for_training 그래프 → quantizer 부착 → prepare
captured = export(model, sample_inputs).module()
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
prepared = prepare_pt2e(captured, quantizer)

# 2) 캘리브레이션 (대표 입력 몇 배치 흘리기 — Lv.2 representative_dataset의 PyTorch판)
for _ in range(50):
    prepared(torch.randn(1, 3, 224, 224))

# 3) convert → 다시 export → XNNPACK 로우어링 → .pte
quantized = convert_pt2e(prepared)
et_int8 = to_edge_transform_and_lower(
    export(quantized, sample_inputs),
    partitioner=[XnnpackPartitioner()],
).to_executorch()
open("mobilenet_v2_int8.pte", "wb").write(et_int8.buffer)
```

> ⚠️ 확인 필요: PT2E 양자화 API의 임포트 경로는 버전에 따라 이동해 왔다(`torch.ao...` ↔ `torchao...`). 설치한 버전의 [ExecuTorch Quantization / XNNPACK 문서](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html)에서 정확한 경로를 재확인할 것. (c)는 (a)/(b)를 먼저 통과한 뒤 도전하는 보너스다.

> 🔴 함정 (핵심 학습 포인트): **`torch.export`가 미지원 연산/동적 제어흐름에서 깨진다.** 이 "export가 깨지는 경험"이 Lv.4의 진짜 목적이며, [2단계 Transformer 양자화](04_transformer_quantization.md)에서 만날 "그래프 캡처 실패 → 모델 리라이팅"의 예고편이다. 일부러 깨뜨려 보라:

```python
# lv4_break.py — data-dependent 제어흐름은 torch.export를 깨뜨린다
import torch
from torch.export import export

class Dynamic(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:          # ← 입력 값에 따라 분기: 그래프로 굳힐 수 없음
            return x * 2
        return x - 1

export(Dynamic(), (torch.randn(4),))
# 예상 에러 (요지):
#   torch._dynamo.exc.UserError: Could not guard on data-dependent expression ...
#   (Dynamic control flow: `if x.sum() > 0` depends on tensor values)
# 해결 방향: torch.cond 로 분기를 명시적 op로 바꾸거나, 분기를 모델 밖으로 빼낸다.
```

> 커스텀 CUDA op, 일부 동적 shape, `.item()`/`.tolist()` 같은 data-dependent 접근도 같은 계열의 실패를 낸다. **에러 메시지에서 "data-dependent" 또는 "guard"를 발견하면 그게 신호**다 — 이 패턴 인식이 [2단계](04_transformer_quantization.md)의 사전 학습이다.

> ⚠️ 확인 필요: `executorch.runtime.Runtime` Python 런타임 API는 버전에 따라 시그니처가 바뀔 수 있다. 실행부 예제는 설치한 버전의 [ExecuTorch Getting Started](https://docs.pytorch.org/executorch/stable/getting-started.html) / [Export & Lowering 문서](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)에서 재확인.

---

### Qualcomm AI Hub — 보드 없이 실단말 프로파일링

QNN 트랙의 **"HTP(Hexagon) 실행은 보드가 필요하다"** 는 장벽을, **보드 구매 없이** 상당 부분 대체한다. 모델을 업로드하면 **클라우드에 있는 실제 단말**(Galaxy, Dragonwing 등)에서 컴파일·프로파일링·정확도 검증을 돌려준다.

**lifecycle (3잡 구조)**: `submit_compile_job`(모델→타깃 바이너리) → `submit_profile_job`(실단말에서 latency/메모리 측정) → (선택) `submit_inference_job`(실단말에서 실제 출력 받아 정확도 검증). 각 잡은 비동기라 `.wait()`로 완료를 기다리고 결과를 조회한다.

```python
# aihub_profile.py — 디바이스 선택 → 컴파일 → 실단말 프로파일 → 결과 조회 (풀 lifecycle)
import torch, torchvision
import qai_hub as hub

# 0) 사용 가능한 실단말 목록 조회 → 원하는 디바이스 고르기
for d in hub.get_devices():
    print(d.name)                    # 예: 'Samsung Galaxy S24 (Family)', 'QCS6490 (Proxy)' ...
device = hub.Device("Samsung Galaxy S24 (Family)")

# 1) 모델을 TorchScript로 트레이스 (또는 torch.export 산출물/ONNX도 가능)
model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
traced = torch.jit.trace(model, torch.rand(1, 3, 224, 224))

# 2) 타깃 단말용으로 컴파일 (클라우드) — input_specs 키는 '모델 입력 이름'과 일치
compile_job = hub.submit_compile_job(
    model=traced,
    input_specs=dict(image=(1, 3, 224, 224)),
    device=device,
)
compile_job.wait()                                  # 컴파일 완료 대기
target_model = compile_job.get_target_model()       # 컴파일 산출물(디바이스용 바이너리)

# 3) 컴파일 결과를 실단말에서 프로파일링 → latency/메모리/NPU 사용 리포트
profile_job = hub.submit_profile_job(model=target_model, device=device)
profile_job.wait()
print("profile dashboard:", profile_job.url)        # 웹에서 per-layer 타이밍/시각화 확인

# 4) 프로파일 결과를 코드에서 조회 (추정 추론시간 등)
prof = profile_job.download_profile()               # dict 형태의 상세 리포트
est_us = prof["execution_summary"]["estimated_inference_time"]  # 마이크로초
print(f"estimated inference: {est_us/1000:.2f} ms on {device.name}")

# 예상 출력 (요지):
#   Samsung Galaxy S24 (Family)
#   ... (디바이스 목록) ...
#   profile dashboard: https://app.aihub.qualcomm.com/jobs/xxxxxxxx/
#   estimated inference: 1.83 ms on Samsung Galaxy S24 (Family)
```

> 💡 팁: **정확도까지 보고 싶으면** `submit_inference_job(model=target_model, inputs={"image": [...]}, device=device)`로 실단말 출력을 받아 PyTorch 기준 출력과 비교한다(`.download_output_data()`). 이것이 "양자화 후 실제 단말에서 정확도가 유지되는가"를 보드 없이 검증하는 방법이다.

> 💡 팁: `hub.Device("...(Family)")`는 계열 중 가용 단말을 자동 배정하고, `hub.Device(attributes="qualcomm-snapdragon-8-elite")`처럼 속성으로도 고를 수 있다. **Dragonwing 계열**이 있으면 QNN/Hexagon 경로 latency를 보드 없이 근사 측정 가능 — Lv.3에서 못 만진 QNN EP의 대체 체험이다.

> ⚠️ 확인 필요: 예제의 `input_specs` 키·`Device` 문자열·`download_profile()`의 결과 딕셔너리 키(`execution_summary`/`estimated_inference_time`)는 시점에 따라 달라질 수 있다. 실제 사용 시 [AI Hub 문서](https://app.aihub.qualcomm.com/docs/)의 최신 시그니처로 맞추고(입력 이름과 일치해야 함), 무료 티어의 잡 수/큐 대기 시간은 시점에 따라 다르다.

---

## 5) 예시 / 결과 해석

각 레벨에서 아래 표를 **직접 채우는 것**이 산출물이다(수치는 예시 형태 — 실제 값은 여러분 환경에서 측정).

**Lv.2 — PTQ 3종 비교 (LiteRT, MNIST 소형 CNN 기준)**

| 항목 | FP32 | Dynamic range | Float16 | Full INT8 |
|------|------|---------------|---------|-----------|
| 파일 크기 | 예: 13 KB | 예: ~5 KB (≈1/4) | 예: ~7 KB (≈1/2) | 예: ~4 KB (≈1/4) |
| Top-1 정확도 | 예: 98.9% | 예: 98.9% | 예: 98.9% | 예: 98.5% |
| 추론시간(PC x86) | 예: 0.30 ms | 예: 0.28 ms | 예: 0.31 ms | 예: 0.29 ms |
| 추론시간(RPi ARM) | 예: 1.8 ms | 예: 1.3 ms | 예: 1.6 ms | 예: 0.9 ms |
| 정수-only HW 배포 | ✗ | ✗ (activation float) | ✗ | ✓ |

**Lv.3 — EP별 latency (동일 ResNet-18, RTX PC, 워밍업 후 p50/p95)**

| Provider | p50 | p95 | 비고 |
|----------|-----|-----|------|
| CPUExecutionProvider | 예: 28 ms | 예: 33 ms | baseline |
| CUDAExecutionProvider | 예: 2.9 ms | 예: 3.4 ms | GPU 가속 (약 10배) |
| TensorrtExecutionProvider | 예: 1.45 ms | 예: 1.8 ms | 최초 엔진 빌드 후 최속(warm-up·캐시 필수) |

**해석 포인트**
- Lv.2: **"정수화로 용량↓, 대신 정확도 미세 손실"**은 표 어디서나 보이지만, **속도 이득은 하드웨어를 탄다** — PC(x86)에선 차이가 작고 RPi(ARM)에서 벌어진다. 그리고 **정수-only 하드웨어에 올릴 수 있는 건 Full INT8뿐**이라는 마지막 행이 임베디드 관점의 핵심이다.
- Lv.3: **모델·코드 거의 그대로 두고 EP만 바꿔** 성능이 수십 배 차이. 이 "한 줄 교체"가 [4단계](06_multi_soc.md)에서 SoC별 EP/툴로 확장된다.
- 공통: TensorRT/AI Hub는 **최초 컴파일이 느리고 이후 빠르다** → 프로파일링은 반드시 warm-up 후, avg가 아니라 p50/p95로 측정.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| `edge-impulse-daemon: command not found` | 전역 npm bin이 PATH에 없음 / Node 미설치 | `node -v`로 v16+ 확인, `npm install -g edge-impulse-cli` 재실행, `npm root -g`의 상위 `bin`을 PATH에 추가 |
| BYOM 업로드 후 정확도가 랜덤 수준 | 입력 스케일/채널순서(NCHW↔NHWC) 불일치, 전처리 누락 | 업로드 폼에 정규화(0–255→0–1 등)와 입력 형상을 export 때와 동일하게 지정 |
| Lv.2 "INT8인데 activation이 float" | `representative_dataset` 없이 `Optimize.DEFAULT`만 켬 (=dynamic range) | `supported_ops=[TFLITE_BUILTINS_INT8]` + `representative_dataset` + `inference_input/output_type=tf.int8` 3종 세트 |
| `representative_dataset`에서 `ValueError` | `yield img`로 텐서를 직접 반환 / dtype이 float32 아님 | `yield [img]`(리스트로 감싸기), 샘플 dtype `float32` 확인 |
| `ai_edge_litert` import 에러 | TFLite 구 패키지와 혼동 | `pip install ai-edge-litert` (구 `tflite-runtime`/`tensorflow.lite` 아님). 개명 사실 유의 |
| INT8 모델이 PC에서 안 빨라짐 | x86 CPU는 float가 이미 최적화됨 | 정상. 속도 이득은 ARM/MCU/NPU에서 확인 — 같은 벤치를 RPi에서 재측정 |
| `ort.get_available_providers()`에 CUDA/TensorRT 없음 | `onnxruntime`(CPU판) 설치됨 / CUDA·TensorRT 런타임 불일치 | `pip uninstall onnxruntime onnxruntime-gpu` 후 `onnxruntime-gpu`만 재설치, CUDA 12.x·드라이버 확인 |
| TensorRT EP 첫 실행이 매우 느림 | 최초 엔진 빌드(정상 동작) | `trt_engine_cache_enable=True`+`trt_engine_cache_path`로 캐시, warm-up 반복 후 측정. shape 고정 |
| TensorRT EP인데 `active`가 CUDA로 뜸 | 해당 노드를 TRT가 못 맡아 fallback | 정상 동작(뼈대 ⑤). 로그에서 어떤 op가 빠졌는지 확인, 필요시 opset/모델 단순화 |
| Lv.4 `torch.export` 예외(`data-dependent`/`guard`) | 미지원 op·동적 제어흐름(`if x.sum()>0`, `.item()` 등) | **의도된 학습 포인트**. `torch.cond`로 분기 명시화, 분기를 모델 밖으로, 또는 partitioner에서 제외. [2단계](04_transformer_quantization.md) 참조 |
| Lv.4 PT2E 임포트 실패 | API 경로 이동(`torch.ao`↔`torchao`) | 설치 버전의 [XNNPACK 백엔드 문서](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html)에서 정확한 경로 확인 |
| `qai_hub` 401/인증 오류 | API 토큰 미설정 | `qai-hub configure --api_token <TOKEN>` (토큰은 AI Hub 계정 페이지에서 발급) |
| AI Hub 컴파일/프로파일 잡이 오래 큐잉 | 무료 티어 큐 대기 | 정상. 잡은 비동기이므로 `.wait()`로 대기하거나 `job.url` 대시보드에서 상태 확인 |

---

## 7) 산출물 (Deliverables)

이 단계가 남겨야 할 것:

- [ ] `study_guide/artifacts/lv2/` : `model_fp32.tflite`, `model_dynamic.tflite`, `model_fp16.tflite`, `model_int8.tflite`, `lv2_convert.py`, `lv2_infer.py`, `lv2_bench.py`
- [ ] `study_guide/artifacts/lv3/` : `resnet18.onnx`, `lv3_export.py`, `lv3_bench.py`, (엔진 캐시 `trt_cache/`)
- [ ] `study_guide/artifacts/lv4/` : `mobilenet_v2.pte`, `lv4_export.py`, `lv4_run.py`, (선택 `lv4_quantize.py`/`mobilenet_v2_int8.pte`, `lv4_break.py`)
- [ ] **비교 노트 1장** (`ladder_notes.md`): 위 [5)절](#5-예시--결과-해석)의 표 2개(PTQ 3종, EP별 latency)를 **내 측정값**으로 채운 것 + "공통 뼈대(변환→양자화→배포→fallback)를 각 레벨에서 어디가 담당했는지"를 [2)절의 매핑 표](#2-배경-이론--개념--난이도-사다리-한눈에)를 참고해 3~5줄로 서술.
- [ ] (선택) Edge Impulse 프로젝트 링크, Qualcomm AI Hub profile job URL, 그리고 **BYOM으로 `resnet18.onnx`를 올린 프로파일 결과**(Lv.1↔Lv.3 다리).

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [Edge Impulse Documentation](https://docs.edgeimpulse.com/) — 노코드 학습·배포, BYOM, QNN 가속 (2025 Qualcomm 인수)
- [Edge Impulse Studio](https://studio.edgeimpulse.com) — 프로젝트/데이터/Impulse/배포 웹 UI
- [edge-impulse-cli (npm)](https://www.npmjs.com/package/edge-impulse-cli) — CLI/daemon/uploader/runner (2026-07 기준 최신)
- [LiteRT (구 TFLite) 공식](https://ai.google.dev/edge/litert) · [ai-edge-litert (PyPI)](https://pypi.org/project/ai-edge-litert/) — 런타임 (2026-07 기준 최신)
- [LiteRT Post-training dynamic range quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quant) — dynamic range(4x, activation float)
- [LiteRT Post-training float16 quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant) — float16(2x, GPU 타깃)
- [LiteRT Post-training integer quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant) — full INT8 + `representative_dataset`. ⚠️ 구 링크 `ai.google.dev/…/post_training_quantization`은 **Google OAuth 무한 리다이렉트에 걸린다**(2026-08-06 실측: `accounts.google.com/o/oauth2/…`로 302, `curl: (47) Maximum (50) redirects followed`) — 위 `developers.google.com` 주소를 쓸 것.
- [LiteRT inference (Interpreter / CompiledModel)](https://developers.google.com/edge/litert/inference) — 추론 API 두 경로
- [tf.lite.TFLiteConverter API](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter) — 변환기 레퍼런스
- [ONNX Runtime 설치](https://onnxruntime.ai/docs/install/) · [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) — EP 교체, `onnxruntime-gpu` 1.23.2(CUDA 12)
- [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) — `provider_options`, 엔진 캐시, FP16/INT8 · [CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [TensorRT 10.16 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html) — LTS 라인(2026-03), 정본 버전
- [ExecuTorch Getting Started](https://docs.pytorch.org/executorch/stable/getting-started.html) · [Export & Lowering](https://docs.pytorch.org/executorch/stable/using-executorch-export.html) · [XNNPACK Backend](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html) — `torch.export`→`.pte`, PT2E 양자화
- [ExecuTorch GitHub Releases](https://github.com/pytorch/executorch/releases) — v1.0 GA 2025-10, 최신 v1.3.x
- [Introducing ExecuTorch 1.0 (PyTorch blog)](https://pytorch.org/blog/introducing-executorch-1-0/) — 백엔드 12개+, 풋프린트/멀티모달
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) · [문서](https://app.aihub.qualcomm.com/docs/) · [qai-hub Get Started](https://aihub.qualcomm.com/get-started) — 클라우드 실단말 프로파일링
- [qai_hub.submit_compile_job](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html) · [Profiling Models 예제](https://app.aihub.qualcomm.com/docs/hub/profile_examples.html) — compile/profile lifecycle

### 논문
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, arXiv:1712.05877 — TFLite INT8 PTQ의 이론적 배경
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient Neural Network Inference*, arXiv:2103.13630 — 양자화 전반 서베이
- Nachin et al. (2026), *ExecuTorch — A Unified PyTorch Solution to Run AI Models On-Device*, arXiv:[2605.08195](https://arxiv.org/abs/2605.08195) — ExecuTorch 설계 백서 (2026-05 등록, 2026-07 접근 확인)

---

## 9) 다음 단계

사다리로 "완주 감각"을 얻었다면, 이제 **왜 정수화가 되는지·어디서 정확도가 새는지**를 정면으로 배운다. Lv.2에서 손으로 만진 `scale`/`zero_point`, Lv.4에서 관찰한 "export 붕괴"가 다음 두 문서에서 원리로 이어진다.

➡️ **[1단계 — 양자화 이론](03_quantization_theory.md)**
⬅️ 이전: **[0단계 — 환경 준비](01_environment_setup.md)**
