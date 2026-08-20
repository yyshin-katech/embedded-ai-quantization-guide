# 4. 멀티 SoC 확장 (TIDL / QNN / DRP-AI)

> **원본 가이드 매핑**: "4단계 — 멀티 SoC 확장 (3~4주)" · 같은 ONNX 하나를 여러 벤더 백엔드에 밀어넣는 multi-target deployment pipeline.
> **예상 소요**: 3~4주 (툴체인 3개 × 설치·컴파일·로그 해석)
> **선행 조건**: [3단계 TensorRT](05_tensorrt.md) 완료. INT8 PTQ/QDQ ONNX 개념 숙지. Docker 사용 가능한 Ubuntu 22.04 + RTX GPU 데스크톱.
> **실행 환경 원칙**: 세 툴체인 모두 **컴파일·에뮬레이션은 x86_64(이 데스크톱)에서 가능**하고, **실제 타겟 실행만 보드가 필요**하다. 보드 없이도 "컴파일이 되는가 / 얼마나 offload되는가"까지 전부 검증할 수 있다.
> **정본 버전 스택(이 커리큘럼 공통)**: CUDA 12.8 / onnx 1.18.0 / onnxruntime-gpu 1.23.2 / TensorRT 10.16.x LTS. **단, 이 단계의 세 벤더 툴은 각자 자체 런타임/버전을 쓴다** — TIDL은 TI가 포크한 onnxruntime, QNN EP는 QAIRT 번들, DRP-AI는 자체 TVM 빌드. 정본 스택은 "2·3단계에서 ONNX를 만든 데스크톱 환경"의 기준이고, 여기서는 그 ONNX를 각 벤더 툴에 넘긴다.

---

## 0) 이 단계에서 무엇을·왜 하는가

3단계까지는 NVIDIA 한 벤더의 스택(TensorRT)만 다뤘다. 실무의 임베디드 AI 채용 공고(JD)에 거의 항상 나오는 문구가 **"multi-target deployment pipeline"** 이다. 하나의 학습된 모델을 TI, Qualcomm, Renesas 등 서로 다른 SoC에 배포할 수 있어야 한다는 뜻이다.

이 단계의 목표는 **"같은 ONNX 하나"** 를 세 개의 벤더 백엔드(TI TIDL, Qualcomm QNN, Renesas DRP-AI)에 각각 밀어넣어 보고, **툴체인이 달라도 문제의 구조는 동일하다**는 것을 몸으로 익히는 것이다.

> 💡 **핵심 통찰 (이 단계 전체를 관통하는 한 문장)**
> ONNX 표준 op → **벤더가 지원하는 부분집합** → 그중 **INT8로 가능한 부분집합** → 나머지는 **CPU fallback** → **fallback을 최소화하는 것이 성능 최적화의 전부**다.
> 벤더가 셋이든 다섯이든 이 사다리는 똑같다. 툴 이름과 로그 형식만 다르다.

**왜 FP32 그대로는 안 되는가 (직관).** 데스크톱 GPU는 FP32/FP16 연산기가 넘쳐서 양자화가 "선택"이지만, 엣지 SoC의 가속기(NPU/DSP/MAC 어레이)는 **전력·면적을 아끼려고 정수 연산기만** 박아 둔 경우가 많다. 그래서 FP32 텐서를 주면 가속기가 받을 자료형이 아예 없다 → 컴파일러가 그 노드를 CPU로 돌리거나(fallback), 아예 매핑 거부(에러)를 한다. 즉 **양자화는 "정확도 튜닝"이기 이전에 "이 하드웨어에서 돌리기 위한 입장권"** 이다.

세 백엔드 모두 공통적으로:
1. FP32 ONNX를 그대로 못 돌린다. 가속기(NPU/DSP)는 대부분 **INT8 전용**이다 → 양자화가 선행되어야 한다.
2. ONNX op의 **부분집합만** 하드웨어 가속한다 → 나머지는 ARM Cortex-A CPU로 떨어진다(fallback/delegation).
3. **fallback이 많으면** subgraph가 잘게 쪼개지고, 가속기↔CPU 데이터 이동 오버헤드로 **성능이 붕괴**한다.

**왜 subgraph 쪼개짐이 그렇게 나쁜가 (직관).** 가속기와 CPU는 물리적으로 다른 메모리 영역을 본다. 그래프가 `[가속]→[CPU]→[가속]→[CPU]` 로 4토막 나면, 매 경계마다 텐서를 가속기 메모리에서 CPU 메모리로(또는 반대로) **복사하고, 자료형을 변환(dequant/quant)하고, 두 엔진을 번갈아 깨워야** 한다. 이 왕복 비용이 실제 연산보다 커지는 순간이 흔하다. 이건 3단계 TensorRT의 "layer 사이 reformatting/CPU 왕복"과 **정확히 같은 병목**이다 — 벤더만 바뀌었을 뿐.

그래서 이 단계에서 진짜 배워야 할 기술은 "컴파일 명령어"가 아니라 **"컴파일 로그에서 어떤 레이어가 가속되고 어떤 게 CPU로 떨어졌는지 읽는 법"** 이다. 명령어는 리포에서 복사하면 되지만, 로그 해석은 사람이 해야 한다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] 2단계에서 export/정리한 INT8 QDQ ONNX(또는 FP32 ONNX + 캘리브레이션 세트)를 입력으로 준비한다. → [2단계 ONNX export 함정](04_transformer_quantization.md) 참조
- [ ] **TI TIDL**: `edgeai-tidl-tools`를 Docker로 셋업하고, x86 host emulation으로 컴파일 + 추론까지 돌린다.
- [ ] TIDL 컴파일 로그에서 **"Offloaded Nodes / Total Nodes"와 subgraph 개수**를 읽고, fallback 비율을 계산한다.
- [ ] TIDL 컴파일 옵션(`tensor_bits`, `accuracy_level`, `calibration_frames/iterations`, `mixed_precision_factor`, `*_16bit_names_list`, `deny_list`, `debug_level`)의 **의미를 각각 설명**할 수 있다.
- [ ] **Qualcomm QNN**: `onnxruntime`의 QNN EP 유틸(`qnn_preprocess_model` + `get_qnn_qdq_config` + `quantize`)로 QDQ 모델을 만든다.
- [ ] QNN EP provider_options(`backend_path`, `htp_performance_mode`, `htp_graph_finalization_optimization_mode`)와 context binary 캐시, `disable_cpu_ep_fallback`의 역할을 이해한다.
- [ ] QNN EP 세션을 만들거나(가능 시), 보드가 없으면 **Qualcomm AI Hub**(`qai_hub`)로 실기기 프로파일을 대체한다.
- [ ] **Renesas DRP-AI**: `rzv_drp-ai_tvm`으로 `compile_onnx_model_quant.py`를 실행해 INT8 컴파일을 시도한다(RZ/V2H·V2N은 `--mera2`).
- [ ] DRP-AI 캘리브레이션 **전처리(mean/std)를 학습과 정확히 일치**시켰는지 코드로 검증한다(실무 최다 실수).
- [ ] 세 백엔드 각각의 **INT8 요구사항·미지원 op·fallback 로그 위치**를 비교표로 정리한다(산출물).
- [ ] "fallback 비율부터 확인"을 **정량 지표(offload%, subgraph 수)** 로 계산해 백엔드 우선순위를 정한다.

---

## 2) 배경 이론 / 개념 — 세 백엔드를 한 장으로

세 툴체인은 이름·CLI·로그가 다르지만 아래 표처럼 **같은 자리에 같은 개념**이 있다. 이 표를 먼저 머리에 넣고 각 절을 읽으면 훨씬 빠르다.

| 개념 | TI TIDL | Qualcomm QNN | Renesas DRP-AI |
|------|---------|--------------|----------------|
| 가속기 | C7x-MMA (DSP+MMA) | Hexagon **HTP** (NPU) | **DRP-AI3** MAC 어레이 |
| 양자화 요구 | 8/16-bit/mixed 지원 | HTP는 **양자화 모델 권장**(fp16도 가능) | DRP-AI3 MAC은 **INT8 연산만** |
| 컴파일러 진입점 | `edgeai-tidl-tools` (ONNX-RT/TFLite-RT/DLR delegate) | `onnxruntime` **QNN EP** (QAIRT) | `rzv_drp-ai_tvm` (**TVM 기반**, EdgeCortix MERA) |
| 런타임 정체 | TI 포크 onnxruntime의 `TIDLCompilationProvider`/`TIDLExecutionProvider` | 표준 onnxruntime의 `QNNExecutionProvider` | TVM `graph_executor` + DRP-AI 런타임 |
| 지원 op 범위 | TIDL 지원 op 목록 | ONNX op **부분집합**(Loop/If·동적 shape 미지원) | feed-forward 전용(RNN/LSTM/GRU 불가) |
| 미지원 op 처리 | ARM Cortex-A로 fallback | CPU EP로 fallback | **CPU(TVM) 위임** |
| fallback 신호 | subgraph 개수 ↑, Offloaded Nodes ↓ | 콘솔 CPU 노드 경고 / `disable_cpu_ep_fallback` 로드 에러 | CPU 위임 노드 ↑ → 느려짐 |
| x86에서 되는 것 | Docker로 컴파일 + **host emulation 추론** | **양자화 유틸(x86_64 전용)** + 컴파일 | 컴파일(quantize + translate) |
| 보드 필요한 것 | 타겟 실측 latency | 실기기 추론(또는 AI Hub 대체) | 타겟 실행 |

> ⚠️ **주의 — 양자화 도구가 x86 전용인 경우가 많다**
> QNN 양자화 유틸은 "onnx 패키지 ARM64 설치 문제로 **x86_64에서만 지원**"된다고 명시돼 있다. TIDL/DRP-AI도 컴파일은 x86 호스트에서 한다. 즉 **이 단계의 대부분은 보드 없이 데스크톱에서 진행**하고, 마지막 실측만 보드로 넘긴다.

### 2-1. 벤더별 지원 op 경향 비교표 (개념 지도)

정확한 지원 op는 각 벤더 문서가 정본이지만(버전마다 바뀐다), **"무엇이 잘 되고 무엇이 잘 깨지는가"의 경향**은 세 벤더가 놀랄 만큼 비슷하다. 이 경향을 알고 있으면 2단계 export 단계에서 미리 지뢰를 피할 수 있다.

| op 계열 | 일반적 지원도 | TIDL | QNN(HTP) | DRP-AI3 | 실무 메모 |
|---------|--------------|------|----------|---------|-----------|
| Conv / DepthwiseConv / Gemm/MatMul | ★★★ 핵심 가속 대상 | 가속 | 가속 | 가속 | 이게 가속기의 존재 이유. 여기서 fallback 나면 뭔가 잘못됨 |
| BatchNorm / Relu / Clip / Add | ★★★ | 대개 Conv에 fuse | 가속 | 가속 | fuse되면 로그에서 개별 노드로 안 보일 수 있음 |
| MaxPool / AvgPool / GlobalAvgPool | ★★☆ | 대개 가속 | 가속 | 대개 가속 | 특정 kernel/stride 조합이 파티션을 끊는 경우가 있어 `deny_list` 후보 1순위 |
| Resize / Upsample | ★★☆ 모드 의존 | 모드/스케일 제약 | 모드 제약 | 제약 | `nearest` vs `linear`, align_corners에 민감 |
| Softmax / LayerNorm / GELU | ★★☆ 위치 의존 | 지원 확대 중 | 지원 확대 중 | 제한적 | Transformer의 단골 fallback 지점 → [4단계 Transformer 양자화](04_transformer_quantization.md) |
| Transpose / Reshape / Concat / Slice | ★★☆ | shape 연산은 종종 CPU | 종종 CPU | 종종 CPU | "shape 마사지" op가 가속 구간을 토막 내는 주범 |
| **Loop / If / Scan (control flow)** | ✗ 광범위 미지원 | fallback | **미지원(명시)** | 미지원(feed-forward만) | export에서 반드시 제거 |
| **RNN / LSTM / GRU** | ✗ | 제한/미지원 | 제한 | **미지원(명시)** | 순환 구조는 이 세 백엔드 대상 아님 |
| 동적 shape (dynamic axes) | ✗ 대체로 고정 필요 | 고정 필요 | **미지원(명시)** | 고정 필요 | export 시 batch=1 고정 권장 |

> 💡 **팁 — 위 표의 "★★☆" 이하 행이 곧 fallback 후보 목록이다.** 2단계에서 ONNX를 export할 때 이 행들(특히 control flow·동적 shape·shape 마사지 op)을 미리 정리해 두면, 세 벤더 어디에 넣어도 subgraph가 덜 쪼개진다. "어느 벤더든 좋아하는 그래프"는 **Conv/MatMul 위주의, batch 고정, control flow 없는 feed-forward** 다.

### 2-2. 🔬 실측 (부분·프록시): ARM Cortex-A 폴백 바닥값 — Pi 5(A76) + i.MX8M Nano(A53) + Jetson Orin(A78AE)

> **이건 NPU 실측이 아니다.** 위 세 벤더 NPU(TIDL/QNN/DRP-AI)는 보드가 없어 아직 못 돌린다(4단계 본 과제, 아래 4-A~C). 하지만 세 SoC가 **공통으로 가진 것** — 미지원 op가 떨어지는 **ARM Cortex-A 폴백 경로**(§2 표 "미지원 op 처리" 행) — 은 **오늘 측정할 수 있다**. **Raspberry Pi 5**(Cortex-A76)를 프록시로 삼아, 3·5단계 자산인 **ResNet50 INT8 QDQ ONNX**를 순수 `CPUExecutionProvider`로 관통시켜 봤다. 즉 **"모든 op가 CPU로 폴백됐을 때의 바닥값"** — offload%가 0%로 떨어진 최악의 경우다. 이어 **dotprod 없는 ARM**(NXP **i.MX8M Nano** EVK의 Cortex-A53)을 한 점 더 실측해, **"부호를 가르는 게 ISA 계열(ARM/x86)인가, dot-product 명령 유무인가"까지 갈랐다**(아래 3-사분면). 마지막으로 **dotprod 있는 두 번째 ARM**(Jetson AGX Orin의 Cortex-**A78AE** — 자동차 등급 코어)을 한 점 더 얹어 Pi 5 A76 패턴이 우연이 아님을 확증했고, 이 점이 아래 크로스플랫폼 발견(**같은 SDOT 커널이면 INT8 예측이 비트 동일**)의 결정적 증거가 된다.

**헤드라인: 같은 INT8 그래프인데 CPU ISA가 양자화 이득의 부호를 뒤집는다 — 그리고 부호를 가르는 건 ISA 계열이 아니라 dot-product 명령의 유무다.**

| 플랫폼 (CPU) | FP32 지연 | INT8 지연 | vs 자기 FP32 | INT8 dot-product |
|---|--:|--:|:--|:--|
| **Pi 5** (Cortex-A76) | 144.95 ms | **79.08 ms** | **×1.83 빠름 ✓** | `asimddp`(SDOT) **있음** |
| **Jetson AGX Orin** (Cortex-A78AE) | 38.47 ms | **18.22 ms** | **×2.11 빠름 ✓** | `asimddp`(SDOT) **있음** |
| **i.MX8M Nano** (Cortex-A53) | 680.20 ms | **1123.02 ms** | **1.65× 느림 ✗** | `asimddp` **없음**(ARMv8.0) |
| x86 dev-host (i9-10900K) | 9.28 ms | **16.34 ms** | **1.76× 느림 ✗** | VNNI **없음**(AVX2까지) |

- ONNX Runtime CPU 커널(MLAS)은 **정수 dot-product 명령**이 있으면 INT8을 가속하고 없으면 못 한다 — ARM `SDOT`(`asimddp`, ARMv8.2+) vs x86 `VPDPBUSD`(VNNI). **A76엔 있고, A53(ARMv8.0)·Comet Lake엔 없다.** dot-product 없는 경로는 스칼라/일반 SIMD로 떨어지고 그 위에 `QuantizeLinear`/`DequantizeLinear` 노드 비용까지 얹혀 **FP32보다 되레 느려진다**. 같은 그래프·같은 런타임인데 **ISA의 INT8 명령 하나**가 부호를 가른다.
- **핵심: 부호를 가르는 건 "ARM이냐 x86이냐"가 아니라 dotprod 유무다.** Pi 5의 A76과 i.MX8M Nano의 A53은 **같은 ARMv8 계열**인데 부호가 반대다 — A53(dotprod ×)이 x86(VNNI ×)과 같은 "느려짐" 부호를 낸다. 즉 통념 **"ARM이면 INT8이 유리"는 ARM 안에서 반박**된다. 결정 인자는 코어가 **ARMv8.2의 dot-product 확장을 구현했는지** 하나다.
- **함의:** NPU는 INT8 전용 MAC이라 INT8이 항상 이긴다(존재 이유). 부호 반전은 **폴백된 부분**에서 벌어진다 → **폴백이 많을수록 A-코어의 dotprod 유무가 최종 성능을 좌우**한다. dotprod 있는 A76·**A78AE**(여러 최신 오토모티브 SoC의 A-코어; AE=Automotive Enhanced, 위 Jetson으로 직접 실측)면 폴백조차 INT8이 유리하지만, **dotprod 없는 구형 A53(위 i.MX8M Nano로 직접 실측)이면 폴백 INT8이 되레 독**이다. 그래서 이 단계에서 offload%만큼이나 **"폴백이 어느 A-코어(ARMv8.0 vs 8.2+)로 떨어지는가"**도 봐야 한다.
- **덤: 크로스플랫폼 예측 동일성은 FP32에선 무조건, INT8에선 "정수 dot-product 커널이 같을 때만" 성립.** 같은 1,000장 예측을 네 플랫폼(ORT 3버전 1.17.1/1.23.2/1.28.0) 1:1 대조하니 **FP32는 여섯 쌍 모두 1000/1000(100%)** 완전 일치. INT8은 갈리는데, **갈림의 기준은 ISA 계열이 아니라 "같은 정수커널을 쓰는가"다** — 둘 다 MLAS `SDOT` 경로인 **Jetson A78AE ↔ Pi 5 A76은 1000/1000(100%, 비트 동일)**, 경로가 다른 쌍은 ~96%(imx↔pi5·imx↔jetson 965 · x86↔pi5·x86↔jetson 958 · imx↔x86 961). **결정적 대조**: 서로 **다른** ORT 버전(Jetson 1.23.2 ↔ Pi 5 1.28.0)인데도 INT8이 1000/1000인 반면, **같은** ORT 버전(Jetson ↔ x86, 둘 다 1.23.2)인데도 958로 갈린다 → 갈림은 ORT 버전도 ISA 계열도 아닌 **정수 dot-product 커널 경로**다. **INT8 정확도를 서로 다른 타겟에서 비트 단위로 기대하면 안 된다 — 단 같은 정수커널(예: SDOT)을 공유하면 비트 동일이 보장된다.**
- **실물 저사양 보드의 벽 2건**(i.MX8M Nano, 2GB no-swap·ORT 1.17.1 — x86 개발기에선 안 보이는 벽): **(a)** FP32가 602MB float32 배열에서 **OOM(SIGKILL)** → 이미지 1장씩 lazy 전처리하는 `rpi_bench_lowmem.py`로 해소(전처리 elementwise라 예측 비트 동일); **(b)** INT8 QDQ의 `opset_import`가 미사용 `ai.onnx.ml v5`를 선언해 **ORT 1.17.1(상한 opset 4)이 로드 거부** → 미사용 opset 항목 strip(415 노드 전부 기본 도메인이라 연산 그래프 불변). **엣지에선 "메모리 상한"과 "런타임의 opset 상한"이 실물 보드에서만 드러나는 두 벽**이다.

> 📄 전체 실측·그림·재현: [`../logs/stage4_arm_cpu_fallback_report.html`](../logs/stage4_arm_cpu_fallback_report.html)(Pi 5·x86 원 발견) · [`../logs/stage4_imx8mn_a53_report.html`](../logs/stage4_imx8mn_a53_report.html)(A53 — 3-사분면 완성·벽 2건) · [`../logs/stage4_jetson_agx_orin_a78ae_report.html`](../logs/stage4_jetson_agx_orin_a78ae_report.html)(A78AE — 2번째 dotprod-ARM·같은-SDOT 비트동일) · 데이터·스크립트: [`../experiments/stage5_infrastructure/cpu_proxy/`](../experiments/stage5_infrastructure/cpu_proxy/)
> **캐비앗:** Pi 5·i.MX8M Nano·Jetson Orin은 여기선 **CPU 폴백 프록시**로만 쓴 것이다(Jetson의 실제 가속기 iGPU/DLA 실측은 바로 아래 §2-3). Nano엔 NPU가 없다(i.MX8M **Plus**에만 탑재; 가속 수치 전이 불가). 절대 지연·top-1은 CPUEP·wall-clock·배치1·1,000장 서브셋·ORT 버전 상이(1.17.1/1.23.2/1.28.0) 기준 → **상대 관계(부호·배율·예측 일치율)만 유효**. 세 자동차 벤더 NPU 실측은 보드 확보 시 4-A~C에서(그때 이 CPU 바닥값이 "가속기가 이겨야 할 최소선"으로 대조축이 된다). Jetson 축은 다음 §2-3에서 그 '최소선을 넘는' 가속기 실측으로 이어진다.

### 2-3. 🔬 실측 (온디바이스): Jetson AGX Orin 가속기 — iGPU 정밀도 사다리 · DLA 오프로드 · 성능/와트

> **이건 실제 벤더 가속기 실측이다 — §2-2의 짝.** §2-2가 **같은 Jetson 실리콘의 CPU(A78AE) 바닥값**(offload 0%, "가속기가 이겨야 할 최소선")이었다면, 여기서는 **그 최소선을 넘는** — 같은 보드의 **Ampere iGPU + 2× NVDLA v2** 가속기를 실측했다. 모델·ONNX는 §2-2·3·5단계와 **동일한 ResNet50**, 도구는 JetPack 동봉 **실 `trtexec`**(3·5단계의 pip-휠 trtexec 부재를 온디바이스가 해소). NVIDIA Jetson은 세 자동차 벤더(TI/Qualcomm/Renesas) 밖이지만, **"iGPU vs 전용 가속기(DLA)"의 오프로드·성능/와트 트레이드오프를 실기기로 정량화**한 첫 지점이다.

| 엔진 | 백엔드 | 지연 ms | vs FP32 | steady W | GR3D | inf/s/W |
|---|---|--:|:--|--:|--:|--:|
| iGPU FP32 | iGPU | 1.9375 | ×1.00 | 42.26 | 98% | 12.19 |
| iGPU FP16 | iGPU | 1.0293 | ×1.88 | 36.20 | 96% | 26.77 |
| iGPU INT8 | iGPU | 1.0132 | ×1.91 | 29.70 | 95% | 33.16 |
| DLA FP16 | DLA0 | 17.7344 | ×0.11 | 15.69 | 3% | 3.59 |
| **DLA INT8** | DLA0 | **1.2783** | ×1.52 | **15.19** | 16% | **51.29** |

- **DLA INT8 = 성능/와트 챔피언(51.29 inf/s/W).** iGPU INT8(33.16)의 **1.547×**, 전력은 **0.511×(절반, 15.19 W)**. 지연만 1.262× 느릴 뿐 → **전력·발열·GPU-여유가 목적이면 DLA INT8**(오토모티브 상시가동에 결정적), **순수 최저지연이면 iGPU INT8**. 게다가 DLA가 도는 동안 GPU가 비어 다른 모델/헤드를 병렬로 얹을 수 있다 — **단 조건부**(DLA 서브그래프에 GPU-폴백이 없을 때만; [§2-4](#2-4--실측-온디바이스-igpudla-동시부하--nvpmodel-전력-스윕)에서 동시부하로 정량화).
- **오프로드가 수치로 증명된다(§5 "offload%부터 확인" 원칙의 실증).** `tegrastats`의 GR3D(GPU-3D)가 iGPU INT8 **95%** → DLA **3~16%**로 붕괴 = 연산이 GPU가 아니라 DLA에서 실제로 돈다. 배치는 **DLA-후보 2/2 오프로드**(ForeignNode 2개 = conv 백본 120층 + `/fc/Gemm`), GPU 폴백은 compute 2층뿐(AVG-pool·flatten). §4-A 함정("offload% 높음 **+** subgraph 개수 낮음"이 목표)을 DLA가 정확히 만족 — ForeignNode 2개라 조각남 없음.
- **DLA는 INT8 전용기([3단계 §2.3](05_tensorrt.md#23-dla-deep-learning-accelerator) 실측과 동일).** 레이어 배치가 INT8·FP16 완전 동일한데 **DLA FP16이 13.87× 느리다**(17.73 vs 1.28 ms, 순수 NVDLA v2 데이터패스). DLA에 올릴 거면 무조건 INT8.
- **작은 Ampere iGPU는 INT8 ≈ FP16(지연비 0.984).** batch1 ResNet50이 커널 launch·메모리 대역에 묶여 INT8 연산 이득이 거의 상쇄된다 — 3단계 **RTX 3080은 0.927**(INT8 8% 빠름)이었다. **정밀도 이득도 GPU 규모에 의존**(단 전력·엔진크기는 INT8이 확실히 이김).

> 📄 전체 실측·SVG·판정: [`../logs/stage3_jetson_orin_ondevice_report.html`](../logs/stage3_jetson_orin_ondevice_report.html) · 데이터·스크립트·제약: [`../experiments/stage3_tensorrt/jetson_ondevice/`](../experiments/stage3_tensorrt/jetson_ondevice/) · TensorRT-측 DLA 이론+실측: [3단계 §2.3](05_tensorrt.md#23-dla-deep-learning-accelerator)
> **캐비앗:** 지연 = `trtexec` event-timed·batch1·MAXN → 타 단계(wall-clock)와 1:1 비교 불가, 상대만. **DLA INT8은 `--int8` 암묵 캘리브(자동 레인지)라 지연·전력만 유효, 정확도 미주장**(정확도는 명시적 QDQ인 iGPU INT8·3단계 RTX·§2-2 CPU 프록시에서 확립). 전력 = 보드 총합(캐리어 오버헤드 6.7~8.7 W 포함). iGPU+DLA 동시부하(진짜 병렬 오프로드)는 [§2-4](#2-4--실측-온디바이스-igpudla-동시부하--nvpmodel-전력-스윕)에서 측정.

### 2-4. 🔬 실측 (온디바이스): iGPU∥DLA 동시부하 · nvpmodel 전력 스윕

> **§2-3이 남긴 두 캐비앗을 실측으로 닫는다.** §2-3은 iGPU·DLA를 **한 번에 하나씩(solo)** 재 각 백엔드의 천장만 봤다. 실제 멀티-SoC 배치의 두 질문 — **①** "DLA가 GPU를 비우니 다른 워크로드를 병렬로 얹으면 처리량이 더해지는가"(진짜 병렬 오프로드), **②** "자동차/임베디드가 실제로 도는 전력 예산에서 리더가 바뀌는가" — 을 같은 ResNet50 INT8으로 잰다. 도구는 §2-3과 동일한 실 `trtexec`, 전력은 `tegrastats` 보드-합.

**① 동시부하 — iGPU∥DLA는 공짜 병렬이 아니다** (`trtexec` N개를 iGPU/DLA0/DLA1에 **동시** 기동, 20초 지속, 합산 qps = 각 프로세스 Throughput의 합, 스케일링 = 합산 ÷ 이상적 solo 합; solo 기준선 gpu 980.7 / dla0 778.8 / dla1 774.5 qps)

| 구성 (동시) | 합산 qps | 이상적 합 | 스케일링 | 전력 W | 성능/와트 |
|---|--:|--:|--:|--:|--:|
| iGPU + DLA0 | 1069.5 | 1759.6 | **60.8%** | 29.72 | 35.99 |
| **DLA0 + DLA1** (GPU 유휴) | **1351.5** | 1553.3 | **87.0%** | 20.46 | **66.07** |
| iGPU + DLA0 + DLA1 | 1197.2 | 2534.1 | **47.2%** | 30.51 | 39.24 |

- **iGPU를 섞으면 DLA가 붕괴한다(60.8%).** iGPU+DLA0 합산은 이상적 합의 60.8%뿐 — iGPU는 87.6% 유지하는데 **DLA0만 27.1%로 붕괴**(지연 1.28→4.75 ms, **3.71×** 팽창). 범인은 [§2-3](#2-3--실측-온디바이스-jetson-agx-orin-가속기--igpu-정밀도-사다리--dla-오프로드--성능와트)에서 본 **DLA 폴백 2층**(GlobalAveragePool+flatten)이 **포화된 iGPU 큐 뒤에 직렬화**되는 것. 큐잉 시그니처(지연 팽창)·비대칭 붕괴(DLA만)가 이를 확증. → **§2-3 §5 "offload% 높음 + subgraph 적음"이 목표라도, GPU-폴백이 1층이라도 있으면 진짜 병렬은 깨진다.**
- **두 카메라면 DLA0+DLA1(87.0%·성능/와트 66.07 최고).** GPU를 비워두고 두 스트림을 두 DLA에 나누면 깨끗이 확장하고 전 구성 최고 성능/와트 — 폴백 2층이 **유휴 GPU에서 즉시** 처리되기 때문. iGPU까지 섞은 3개 전부(47.2%)는 iGPU가 **두 DLA의 폴백을 동시에 목 조르므로** 오히려 후퇴.
- **설계 규칙: 진짜 iGPU∥DLA 병렬은 DLA 서브그래프의 GPU-폴백이 0층일 때만.** ResNet50은 말단 pool 때문에 깨진다 → DLA에 얹을 모델은 말단 pool/reshape를 DLA 지원 연산으로 재작성하거나, 아예 두 DLA로 나눠라.

**② nvpmodel 전력 스윕 — 예산이 조이면 리더가 뒤집힌다** (전력 모드별 두 INT8 챔피언; 유효 행 MAXN·50W. 30W/15W는 재부팅 게이트라 폐기 = 값이 50W와 동일)

| 모드 | iGPU INT8 | DLA INT8 |
|---|---|---|
| MAXN | **983.9 qps** · 1.014 ms · 29.60 W · 33.24 inf/s/W | 778.8 qps · 1.278 ms · 15.09 W · 51.61 |
| 50W | 694.97 qps · 1.436 ms · 19.34 W · 35.94 | **756.19 qps** · 1.317 ms · 14.29 W · **52.90** |

- **50W에서 DLA가 iGPU를 +8.8% 추월.** iGPU는 MAXN→50W로 처리량이 **−29.4%** 급락(전압/클럭 민감)하나, **DLA는 −2.9%**뿐(고정 함수 데이터패스라 이미 ≈14–15 W에서 돌아 예산 축소에 둔감) → MAXN에선 iGPU가 앞서지만 50W에선 순위가 뒤집힌다.
- **성능/와트는 전 예산에서 DLA 우월(×1.55 MAXN, ×1.47 50W).** 임베디드·자동차처럼 전력이 조여 있을수록 DLA로 얹는 이득이 커진다 — §2-3의 "상시가동엔 DLA INT8" 결론을 전력 축에서 재확인.

> 📄 전체 실측·SVG·판정: [`../logs/stage3_jetson_orin_concurrent_power_report.html`](../logs/stage3_jetson_orin_concurrent_power_report.html)(동시부하·전력스윕) · solo 5-엔진: [`../logs/stage3_jetson_orin_ondevice_report.html`](../logs/stage3_jetson_orin_ondevice_report.html) · 데이터·스크립트: [`../experiments/stage3_tensorrt/jetson_ondevice/`](../experiments/stage3_tensorrt/jetson_ondevice/)(`concurrent.py`·`power_sweep.py`·`conc_*.json`·`power_sweep.json`)
> **캐비앗:** 절대 지연·처리량·전력 = MAXN(DVFS)·batch1·wall-clock 지속 → 상대만(`jetson_clocks` 미사용). DLA INT8은 `--int8` 암묵 캘리브라 지연·전력만 유효, 정확도 미주장(§2-3과 동일). 30W/15W는 재부팅 게이트라 폐기(값이 50W와 동일 = 미전환) — 삭제 않고 리포트에 회색 행으로 병기. GPU-폴백 직렬화 결론은 ResNet50의 특정 폴백 2층에 기인 → **모델 의존**.

---

## 3) 환경·도구 준비 — 공통 입력 만들기

세 백엔드의 공통 입력은 **① ONNX 모델 ② 캘리브레이션 이미지 폴더**다. 먼저 이걸 고정해 두면 세 툴에 그대로 재사용할 수 있다.

```bash
# 작업 루트 (이 데스크톱 홈 하위, 경로는 각자 환경에 맞게)
export WORK=$HOME/multi_soc && mkdir -p $WORK && cd $WORK

# 2단계 산출물에서 가져오거나, 검증용으로 표준 분류 모델을 받는다.
#   - INT8 QDQ ONNX가 이미 있으면 그걸 쓰고,
#   - 없으면 FP32 ONNX + 캘리브레이션 이미지로 각 툴에서 PTQ를 돌린다.
mkdir -p $WORK/model $WORK/calib
# 예: resnet18-v1-7.onnx 를 $WORK/model/ 에 둔다 (2단계에서 export한 모델로 대체 가능)
# calib/ 에는 대표 분포를 담은 수십 장(20~100장)을 넣는다. 클래스 분포를 고르게.
```

세 툴 모두 캘리브레이션 이미지를 학습과 동일하게 전처리해서 넣어야 하므로, **전처리 파라미터(mean/std/resize/색공간)를 한 파일에 못박아 두고 세 툴에 재사용**하는 것을 강력히 권한다. 아래를 `$WORK/preprocess.py`로 저장해 두고 세 절에서 import한다.

```python
# $WORK/preprocess.py — 세 백엔드가 공유하는 단일 전처리 정의 (학습과 반드시 일치)
import numpy as np
from PIL import Image

# 학습에 쓴 값과 "바이트 단위로" 동일해야 한다. 여기가 어긋나면 정확도가 조용히 무너진다.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet 표준(예시). 네 학습값으로 교체!
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224  # 입력 해상도. 모델 입력 shape과 일치시킬 것

def load_chw(path: str) -> np.ndarray:
    """이미지 1장 → (3,224,224) float32, NCHW용. resize/normalize를 학습과 동일하게."""
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))  # 색공간 RGB 고정
    x = np.asarray(img, dtype=np.float32) / 255.0               # [0,1] 스케일
    x = (x - MEAN) / STD                                        # 채널별 정규화
    return np.transpose(x, (2, 0, 1)).copy()                    # HWC → CHW
```

> 💡 **팁 — 캘리브레이션 세트는 "학습 분포의 축소판"**
> 세 툴 모두 PTQ 캘리브레이션에 수십 장이면 충분하다(DRP-AI 기본 흐름도 "few dozen sample images", TIDL 기본 `calibration_frames`도 20). 하지만 **전처리(resize/normalize/mean-std)가 학습과 어긋나면 정확도가 조용히 무너진다.** 특히 DRP-AI에서 가장 자주 터진다(6절 참조). 위처럼 전처리를 **한 곳에 정의**해 세 툴이 공유하면 이 실수를 구조적으로 막는다.

각 백엔드별 설치는 해당 절에서 다룬다. 셋 다 **Docker 또는 전용 SDK**를 쓰므로, 호스트 Python 환경을 오염시키지 않도록 각각 격리하는 것을 권장한다.

> 🔴 **함정 — 세 벤더의 onnxruntime을 한 venv에 섞지 마라**
> TIDL은 `onnxruntime`을 **TI 포크**(TIDLCompilationProvider/TIDLExecutionProvider 포함)로 덮어쓴다. QNN EP는 표준 `onnxruntime`(+QAIRT)을 쓴다. 3단계에서 쓴 `onnxruntime-gpu 1.23.2`도 또 다른 빌드다. **이 셋을 같은 venv에 pip install하면 서로 덮어써서 "provider를 못 찾음" 에러가 난다.** 반드시 `TIDL(Docker)` / `.qnn` venv / `.drpai` 를 분리하라.

---

## 4) 단계별 실습

### 4-A. TI TIDL — `edgeai-tidl-tools` (x86 host emulation)

TI의 Jacinto/AM 계열(TDA4VM, J721E, J721S2, J784S4, AM62A 등)에 배포할 때 쓰는 스택이다. C7x-MMA 가속기용 아티팩트를 만들고, **x86에서 host emulation으로 추론까지 검증**할 수 있는 게 강점이다. "보드 없이 offload 비율을 본다"는 이 단계의 핵심 검증을 가장 깔끔하게 해 준다.

#### 설치 (Docker 권장)

```bash
cd $WORK
# (2026-07 기준) 최신 태그: 11_02_17_00 (SDK 11.2.0/11.1 호환 패치),
#   최신 stable: 11_02_16_00 (SDK 11.2.1, ONNX Runtime 1.23). 타겟 SDK 버전에 맞는 태그를 고른다.
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tools
git checkout 11_02_16_00   # docs/sdk_version_compatibility_table.md 에서 보드 SDK에 맞는 태그 확인 후 지정

# 호스트 의존성 (Ubuntu 22.04)
sudo apt-get install -y libyaml-cpp-dev libglib2.0-dev python3-setuptools

# 방법 1) 스크립트 셋업 — SOC를 지정해야 한다 (예: AM62A, J721S2, J721E, J784S4)
export SOC=am62a
source ./setup.sh                      # 툴/모델/의존성 다운로드 (TI 포크 onnxruntime_tidl 설치 포함)
# (설치 세부는 리포 최신 README 기준으로 재확인. 아래 Docker 방법이 재현성이 더 좋다.)

# 방법 2) Docker — 의존성 자동 처리 (권장)
#   scripts/docker/ 의 지침을 따른다.
ls scripts/docker/                     # Dockerfile / build/run 스크립트 위치 확인
```

> ⚠️ **확인 필요**: 릴리스마다 셋업 스크립트 위치가 바뀐다(`setup.sh`, `scripts/setup/`, `setup_env.sh <SOC>` 등). `11_02_04_00` 이후로 리포 구조가 크게 재편되었다. `11_02_16_00` 기준으로 위 흐름을 확인했으나, **다른 태그를 쓰면 그 태그의 `README.md`를 그대로 따를 것.** SOC 환경변수는 소문자(`am62a`)/대문자(`AM62A`) 표기가 릴리스별로 다를 수 있으니 README 예시에 맞춘다. (출처: [edgeai-tidl-tools README](https://github.com/TexasInstruments/edgeai-tidl-tools))

> 💡 **팁 — 왜 표준 pip onnxruntime로는 안 되는가**
> TIDL 컴파일은 TI가 포크한 onnxruntime(`onnxruntime_tidl`)이 제공하는 **`TIDLCompilationProvider`(임포트/컴파일용)** 와 **`TIDLExecutionProvider`(가속 추론용)** 를 쓴다. PyPI 표준 onnxruntime에는 이 provider가 없다. `setup.sh`/Docker가 이 포크를 설치해 주므로, **TIDL 작업은 반드시 이 환경 안에서** 한다.

#### 컴파일 + host emulation 추론

TIDL 컴파일은 "ONNX Runtime에 **TIDL delegate/EP를 붙여** 캘리브레이션+파티셔닝하는" 방식이다. 리포의 예제 스크립트를 그대로 쓰는 것이 가장 안전하다.

```bash
cd $WORK/edgeai-tidl-tools/examples/osrt_python/ort   # (태그에 따라 examples/ 하위 경로 상이)
# 컴파일(캘리브레이션 + 아티팩트 생성) → 이어서 추론(host emulation)
python3 onnxrt_ep.py -c    # compile: 캘리브레이션 후 model-artifacts/ 생성
python3 onnxrt_ep.py       # infer:   생성된 아티팩트로 x86 emulation 추론
#   -c 는 "compile" 모드. 빼면 이미 생성된 아티팩트로 추론만 한다.
#   TIDL_TOOLS_PATH / SOC 환경변수가 설정돼 있어야 예제가 툴을 찾는다.
```

핵심은 예제 안의 **컴파일 옵션 딕셔너리**다. 여기서 8/16-bit, 캘리브레이션, deny_list(강제 CPU), 디버그 레벨을 제어한다. 아래는 `model_compilation.md`(2026-07 확인) 기준으로 **각 키의 의미를 주석으로 못박은** 완결 버전이다.

```python
# TIDL 컴파일 옵션 (delegate/compile options). 키 이름은 model_compilation.md 기준(2026-07 확인).
compile_options = {
    "tidl_tools_path": os.environ["TIDL_TOOLS_PATH"],  # TIDL 툴 설치 경로(필수). setup.sh가 export
    "artifacts_folder": "./model-artifacts",           # 산출물 폴더(필수, 비어 있어야 함)

    # ── 정밀도 ─────────────────────────────────────────────
    "tensor_bits": 8,          # 8/16/32. 32는 float이라 "타겟 실행 불가"(host emul 검증용). 8-bit 우선!
    "accuracy_level": 1,       # 0/1/2/9, 기본 1. 클수록 정확도 우선(=캘리브레이션 더 많이/느리게).
                               #   0 = 가장 빠른 컴파일(간이 캘리브레이션, 정확도 손실 감수)
                               #   1 = 기본. calibration_frames/iterations 가 여기서 적용됨
                               #   2/9 = 더 정밀(고급 캘리브레이션). 정확도 갭이 클 때만.

    # ── 캘리브레이션 (accuracy_level=1일 때 적용) ─────────────
    "advanced_options:calibration_frames": 25,      # 캘리브레이션에 쓸 이미지 수 (기본 20). ↑ 정확도/컴파일시간
    "advanced_options:calibration_iterations": 50,  # 캘리브레이션 반복 (기본 50). ↑ 정확도/컴파일시간

    # ── 혼합 정밀도 (8-bit로 정확도 갭이 크면 자동 승격) ──────────
    #   value = (mixed precision 허용 latency) / (8-bit inference latency)
    #   예: 1.2 → "8-bit 대비 1.2배 느려지는 것까지 허용"하며, 그 예산 안에서 정확도가 가장 좋아지는
    #       레이어들을 자동으로 16-bit로 올린다. -1이면 자동 혼합 비활성.
    "advanced_options:mixed_precision_factor": -1,
    # 특정 레이어만 수동으로 16-bit로 올릴 때(자동 대신 손으로 지정):
    # "advanced_options:output_feature_16bit_names_list": "conv1,layer4.1.conv2",  # 이 노드들의 "출력 텐서"를 16bit
    # "advanced_options:params_16bit_names_list": "layer4.1.conv2",                # 이 노드들의 "가중치"를 16bit

    # ── fallback 제어 & 디버그 ───────────────────────────────
    "deny_list:layer_type": "",   # 이 op 타입을 강제로 ARM(CPU)로. 예: "MaxPool". (allow_list와 동시 사용 불가)
    # "deny_list:layer_name": "resnetv15_pool1_fwd",   # 이름으로 특정 노드만 CPU 고정
    "debug_level": 1,             # 0=끔, 1=파티셔닝/subgraph 로그+레이어 사이클, 2=상세,
                                  #   3=고정소수점 레이어 트레이스 덤프, 4=고정+부동 트레이스, 5=상세+트레이스
}
```

각 옵션을 **언제 만지는지** 한눈에:

| 옵션 | 언제 만지나 | 방향 |
|------|------------|------|
| `tensor_bits` | 항상 먼저 8로 | 8 실패 시에만 16 |
| `accuracy_level` | 8-bit 정확도가 아쉬울 때 | 1→2로 올림(느려짐 감수) |
| `calibration_frames/iterations` | 정확도가 캘리브레이션 부족처럼 보일 때 | ↑ (예: 20→50, 50→100) |
| `mixed_precision_factor` | 일부 레이어만 정밀도 문제일 때 | 1.1~1.3부터 |
| `*_16bit_names_list` | 문제 레이어를 이미 특정했을 때 | 그 레이어만 지정 |
| `deny_list` | 특정 op가 파티션을 계속 끊을 때 | 그 op를 CPU로 격리(경계 정리) |
| `debug_level` | 로그가 부족할 때 | 1로 시작, 필요 시 3 이상 |

> 💡 **팁 — 8-bit 우선, 갭 보이면 상향**
> 먼저 `tensor_bits=8`로 전체를 돌리고, 정확도 갭이 큰 레이어만 `output_feature_16bit_names_list`/`params_16bit_names_list`로 16-bit 승격하거나 `mixed_precision_factor`로 자동 혼합한다. 처음부터 16-bit로 가면 속도 이점을 통째로 버리는 것. **순서: 8-bit 전체 → (갭) 자동 혼합(factor) → (특정 레이어 확정 시) 수동 16bit 리스트.**

#### PTQ가 실패할 때 — QAT 또는 pre-quantized로 우회

TIDL의 내부 PTQ 캘리브레이션이 정확도를 못 맞추면 두 갈래다:
1. **QAT**(2단계에서 학습): 학습 시점에 양자화를 반영해 재학습. 근본 해결이지만 학습 파이프라인이 필요.
2. **Pre-quantized 입력**: 이미 **QDQ 노드가 박힌 ONNX**(또는 pre-quantized TFLite)를 넣어 **TIDL 자체 캘리브레이션을 우회**한다. TIDL은 QDQ의 scale/zero-point를 그대로 받아들인다. 2단계에서 만든 QDQ ONNX가 있으면 이 경로가 가장 빠르다.

> 💡 **직관 — 왜 pre-quantized가 통하나**
> TIDL 내부 PTQ는 "스스로 scale을 추정"하는 과정이다. 이미 QDQ 노드가 박혀 있으면 scale/zero-point가 **그래프에 명시**돼 있으므로 TIDL은 추정을 건너뛰고 그 값을 신뢰한다. 즉 **양자화 파라미터의 주도권을 네가 쥔다** → 2·3단계에서 검증한 양자화를 그대로 이식.

#### fallback/offload 로그 읽기 (이 절의 핵심)

컴파일이 끝나면 콘솔과 아티팩트에 **어떤 노드가 C7x로 offload되고 어떤 게 ARM으로 fallback됐는지**가 나온다. `debug_level=1`이면 "parsing summary table"(C7x로 offload된 노드 수 vs CPU에서 오픈소스 런타임으로 돌 노드 수)이 출력된다.

```text
# 컴파일 로그 말미 예시 (실제 값은 모델마다 다름)
Final number of subgraphs created are : 3, - Offloaded Nodes - 196, Total Nodes - 236
```

이 한 줄을 **토큰별로** 읽는다:
- **`Total Nodes - 236`** = 이 ONNX 그래프의 전체 노드 수(파티셔닝 대상). 분모.
- **`Offloaded Nodes - 196`** = C7x-MMA로 간 노드 수. → offload 비율 = 196/236 ≈ **83%**. 나머지 40개(236−196)는 ARM Cortex-A로 fallback.
- **`Final number of subgraphs created are : 3`** = 그래프가 **3토막**으로 파티셔닝됐다는 뜻. **이 숫자가 커질수록 나쁘다.** 3이면 "가속 구간 사이에 CPU 구간이 끼어들어 그래프가 3조각" 났다는 뜻 → 매 경계에서 C7x↔ARM 데이터 이동 오버헤드가 (최소) 2번 발생.
- 이상적 목표: **`Offloaded ≈ Total`(비율 100%에 근접) 이면서 `subgraphs = 1`.** 둘 다 봐야 한다 — offload 90%여도 subgraph가 8개면 데이터 왕복으로 느리다.

아티팩트 폴더에 남는 파일(로그로 안 보이는 세부는 여기서 확인):
- `tempDir/runtimes_visualization.svg` — 모델이 가속기와 CPU로 **어떻게 분할됐는지** 그림. **파티션이 어디서 끊겼는지 눈으로 찾는 가장 빠른 방법.** 브라우저로 열어 "빨간(CPU)/파란(TIDL)" 경계 노드를 본다.
- `tempDir/*.layer_info.txt` — TIDL 레이어 ↔ 원본 모델 레이어 매핑(어느 원본 레이어가 어느 TIDL 레이어가 됐는지).
- `tempDir/*.html` — 레이어 단위 상세 뷰어(사이클/정밀도 등).
- `allowedNode.txt` — 가속 허용된 노드 목록. **이 파일이 생성되지 않으면** 파티셔닝 자체가 실패했거나 미지원 op가 과다하다는 강한 신호(6절 참조).
- `*_net.bin` / `*_io_1.bin` — 실제 타겟에 올릴 컴파일된 네트워크/IO 바이너리.

> 🔴 **함정 — subgraph가 잘게 쪼개지면 성능이 붕괴한다**
> offload 비율이 높아도 subgraph가 5~10개로 쪼개지면, TensorRT의 "레이어 사이 CPU 왕복"과 똑같은 병목이 생긴다. **"Offloaded % 높음 + subgraph 개수 낮음"** 이 목표다. 특정 op 하나가 파티션을 계속 쪼갠다면, `runtimes_visualization.svg`에서 그 경계 op를 찾아 `deny_list`로 아예 ARM에 고정하거나(경계를 한 군데로 몰아 정리), 모델 그래프를 수정해 제거/치환한다. **역설적으로, 문제 op를 CPU로 "더 확실히" 보내면 subgraph 수가 줄어 전체는 빨라질 수 있다.**

---

### 4-B. Qualcomm QNN — `onnxruntime` QNN EP

Qualcomm Snapdragon/Dragonwing 계열의 Hexagon **HTP**(NPU)에 배포한다. ONNX Runtime의 **QNN Execution Provider**를 쓰면 양자화·컴파일·추론을 파이썬 하나로 묶을 수 있다. QNN EP는 내부적으로 Qualcomm AI Runtime(**QAIRT**, 구 QNN SDK)을 감싼다.

#### 설치

```bash
cd $WORK
python3 -m venv .qnn && source .qnn/bin/activate
# onnxruntime-qnn: QNN(QAIRT) EP 패키지. (2026-07 기준) 양자화 유틸은 x86_64에서만 동작.
#   HTP 실기기 실행 자체는 Qualcomm 디바이스(예: Windows on Snapdragon ARM64)에서 이뤄지지만,
#   "양자화 + QDQ 생성"은 이 x86 데스크톱에서 한다.
pip install onnxruntime-qnn
```

> ⚠️ **확인 필요**: `onnxruntime-qnn` 배포 채널은 플랫폼에 따라 다르다. 공식 문서는 "pip install onnxruntime-qnn"이 **Windows ARM64(Qualcomm NPU)** 대상이라고 명시한다. x86 리눅스에서는 **양자화 유틸만** 사용하고(그건 x86에서 동작), 실기기 HTP 실행/프로파일은 아래 AI Hub로 대체하는 구성이 현실적이다. (출처: [ONNX Runtime QNN EP 문서](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html))

#### 양자화 (x86에서) — HTP는 양자화 모델을 선호

HTP 백엔드는 FP32를 (fp16 fallback 없이는) 효율적으로 못 돌린다. 정수 가속을 받으려면 8/16-bit로 먼저 양자화한다. QNN EP 전용 유틸이 "QNN이 좋아하는 형태의 QDQ"를 만들어 준다. 아래는 세 함수(`qnn_preprocess_model` → `get_qnn_qdq_config` → `quantize`)를 **끝까지 이어붙인 완결 코드**다.

```python
# qnn_quantize.py  — x86_64에서 실행 (양자화 유틸은 x86 전용)
import os, glob
import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)
import sys; sys.path.insert(0, os.environ.get("WORK", "."))
from preprocess import load_chw   # 3절에서 만든 공유 전처리 (학습과 동일)

input_model_path  = "model/resnet18-v1-7.onnx"
preproc_model_path = "model/model.preproc.onnx"
output_model_path  = "model/model.qdq.onnx"

# ── CalibrationDataReader: calib/ 이미지를 "모델 입력 이름"에 맞춰 하나씩 공급 ──
class DataReader(CalibrationDataReader):
    def __init__(self, model_path, calib_dir="calib"):
        sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = sess.get_inputs()[0].name           # 모델의 실제 입력 텐서 이름
        self.files = sorted(glob.glob(os.path.join(calib_dir, "*")))
        self.i = 0
    def get_next(self):
        if self.i >= len(self.files):
            return None                                        # 끝나면 None → 캘리브레이션 종료
        x = load_chw(self.files[self.i])[None, ...].astype(np.float32)  # (1,3,224,224)
        self.i += 1
        return {self.input_name: x}
    def rewind(self):
        self.i = 0

# 1) 전처리: QNN이 다루기 쉬운 형태로 그래프 정리(불필요 노드 제거/폴딩). 바뀌었는지 bool 반환.
model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
model_to_quantize = preproc_model_path if model_changed else input_model_path

# 2) QNN용 QDQ 설정 생성. HTP 권장 조합: 활성값 uint16 + 가중치 uint8.
my_data_reader = DataReader(model_to_quantize)
qnn_config = get_qnn_qdq_config(
    model_to_quantize,
    my_data_reader,
    activation_type=QuantType.QUInt16,  # HTP: 16-bit 활성값 (uint16). 순수 8bit면 QUInt8
    weight_type=QuantType.QUInt8,       # HTP: 8-bit 가중치 (uint8)
)

# 3) 양자화 실행 → QDQ ONNX 저장
quantize(model_to_quantize, output_model_path, qnn_config)
print("saved", output_model_path)
```

> 💡 **팁 — 8-bit부터, 갭 보이면 활성값을 16-bit로**
> TIDL과 동일 전략. 순수 8/8(uint8/uint8)로 먼저 시도하고, 정확도가 부족하면 활성값을 `QUInt16`으로 올린다(위 예처럼 uint16 act + uint8 weight가 HTP의 흔한 스윗스팟). QNN op는 대개 **uint8/uint16** 자료형을 지원하므로 이 두 조합이 표준이다.

> 💡 **팁 — `qnn_preprocess_model`을 건너뛰지 마라**
> 이 전처리는 그래프를 QNN이 좋아하는 형태로 정리(예: 폴딩, 특정 패턴 치환)해서 **양자화·파티셔닝 품질을 올린다.** `model_changed`가 `True`면 반드시 정리된 모델을 다음 단계에 넘겨야 한다(위 코드의 삼항 연산이 그 역할).

#### 추론 세션 (QNN EP) + context binary

Qualcomm 디바이스에서 실행하는 코드다. `disable_cpu_ep_fallback`을 켜면 **CPU로 떨어지는 순간 에러**가 나므로, fallback 발생 여부를 즉시 잡아낼 수 있다.

```python
# qnn_infer.py  — Qualcomm 디바이스(HTP)에서 실행
import onnxruntime

options = onnxruntime.SessionOptions()
# CPU fallback을 금지 → 미지원 op가 있으면 로드시 에러로 알려준다 (fallback 탐지용)
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# context binary(사전 컴파일된 그래프 캐시) 생성 → 다음 실행부터 초기화(HTP finalize) 시간 단축
options.add_session_config_entry("ep.context_enable", "1")      # 컨텍스트 덤프 켜기
options.add_session_config_entry("ep.context_embed_mode", "1")  # 1=바이너리를 onnx 안에 임베드, 0=별도 파일
options.add_session_config_entry("ep.context_file_path", "./model_ctx.onnx")  # 미지정 시 <model>_ctx.onnx

session = onnxruntime.InferenceSession(
    "model/model.qdq.onnx",
    sess_options=options,
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_path": "libQnnHtp.so",   # HTP(NPU). CPU 테스트는 "libQnnCpu.so", GPU는 "libQnnGpu.so"
                                          #   (Windows면 QnnHtp.dll / QnnCpu.dll)
        "htp_performance_mode": "burst",  # burst/balanced/default/high_performance/
                                          #   power_saver/sustained_high_performance
        "htp_graph_finalization_optimization_mode": "3",  # 0(기본)~3. 클수록 최적화↑(finalize 준비시간↑)
        # "soc_model": "<SoC 번호>",         # 타겟 SoC 지정(문자열). AOT 컴파일 시 유용
        # "htp_arch": "<아키텍처 번호>",       # HTP 아키텍처 버전
        # "vtcm_mb": "8",                     # VTCM 크기(MB, 문자열)
        # "enable_htp_fp16_precision": "1",  # (기본 1) FP 연산을 fp16으로. 양자화 안 된 부분 가속
        # "offload_graph_io_quantization": "1", # (기본 1) 그래프 입출력 quant/dequant도 HTP로
        # "device_id": "0",                  # 디바이스 식별자(기본 "0")
    }],
)
```

각 provider option을 **언제 쓰나**:

| 옵션 | 역할 | 실무 기본값 |
|------|------|-------------|
| `backend_path` | 어느 Qualcomm 백엔드로 갈지(HTP/CPU/GPU) | `libQnnHtp.so` (실기기), 검증은 `libQnnCpu.so` |
| `htp_performance_mode` | 클럭/전력 정책 | 벤치는 `burst`, 상시는 `sustained_high_performance` |
| `htp_graph_finalization_optimization_mode` | 그래프 finalize 최적화 강도 | 배포용은 `3`(준비 오래, 실행 빠름) |
| `enable_htp_fp16_precision` | 비양자화 FP를 fp16로 | 기본 1 유지 |
| `offload_graph_io_quantization` | 입출력 quant도 가속기에서 | 기본 1 유지 |

> ⚠️ **주의 — QNN EP가 지원하는 op는 ONNX의 부분집합**
> 공식 문서가 명시: **`Loop`, `If`(control flow)는 미지원이고, 동적 shape(dynamic shapes)도 미지원.** 제어흐름/동적 shape이 들어간 그래프는 그 지점에서 막힌다 → 2단계에서 batch를 고정하고 이런 op가 안 나오도록 export하는 게 최선(참고: [2단계 ONNX export 함정](04_transformer_quantization.md)). 미지원 op는 `disable_cpu_ep_fallback`을 안 켰다면 조용히 CPU로 떨어진다.

> 💡 **팁 — `disable_cpu_ep_fallback=1`은 "fallback 탐지 스위치"다**
> 평소엔 미지원 op가 조용히 CPU로 새서 느려져도 알아채기 어렵다. 이 스위치를 켜면 **CPU로 갈 노드가 하나라도 있으면 세션 로드가 즉시 실패**하므로, "완전 HTP 실행이 되는가"를 이분법으로 확인할 수 있다. 개발 중엔 켜서 fallback 지점을 강제로 드러내고, 정 안 되는 op만 골라 다시 끄고 CPU 허용한다.

> 🔴 **함정 — context binary는 SoC/디바이스 종속**
> HTP context binary는 **아키텍처 종속이라 다른 디바이스로 이식 불가.** SoC가 바뀌면 다시 생성해야 한다(그래서 `soc_model`/`htp_arch`로 타겟을 명시하는 옵션이 있다). 캐시를 커밋해두고 재사용할 때 SoC가 같은지 반드시 확인. 다른 SoC의 캐시를 재사용하면 로드 실패하거나(운 나쁘면) 조용히 잘못 동작한다.

#### 보드가 없으면 — Qualcomm AI Hub로 실기기 프로파일 대체

이 데스크톱에는 HTP가 없다. 실기기 latency/정확도가 필요하면 **[Qualcomm AI Hub](https://aihub.qualcomm.com/)** 에 모델을 올려 **클라우드의 실제 Snapdragon 디바이스**에서 컴파일·프로파일·추론을 돌린다. `qai_hub` 파이썬 클라이언트로 3종 잡을 던지는 구조다.

```python
# qai_hub_profile.py  — 이 x86 데스크톱에서 실행. 클라우드의 실기기로 컴파일/프로파일/추론.
import qai_hub as hub
import numpy as np

# 0) 사전: `qai-hub configure --api_token <토큰>` 으로 계정 연결. `qai-hub list-devices`로 기기 확인.
device = hub.Device("Snapdragon 8 Elite QRD")   # list-devices 결과 중 하나로 교체

# 1) 컴파일 잡: ONNX(또는 QDQ ONNX) → 타겟 런타임 바이너리 (하드웨어 인지 최적화 + 수치 검증)
compile_job = hub.submit_compile_job(
    model="model/model.qdq.onnx",
    device=device,
    options="--target_runtime qnn_context_binary",  # QNN context binary로 컴파일(ORT 호환 형태도 가능)
)
compiled_model = compile_job.get_target_model()

# 2) 프로파일 잡: "실제 기기"에서 레이어별 timing·컴퓨트 유닛 사용률·로드/추론 시간 측정
profile_job = hub.submit_profile_job(model=compiled_model, device=device)
print(profile_job.download_profile())   # per-layer latency, NPU/CPU 분배 등

# 3) 추론 잡: 입력 업로드 → 실기기 추론 → 출력 다운로드 (정확도 대조용)
x = np.random.rand(1, 3, 224, 224).astype(np.float32)  # 실제로는 검증셋으로 교체
infer_job = hub.submit_inference_job(model=compiled_model, device=device, inputs={"image": [x]})
outputs = infer_job.download_output_data()
```

> 💡 **팁 — 로컬 검증 → AI Hub 실측, 2단 구성이 표준**
> QNN EP로 로컬에서 "양자화 + 그래프 로드(disable_cpu_ep_fallback로 fallback 0 확인)"까지 검증하고, **latency/정확도 실측만 AI Hub**로 넘긴다. AI Hub 프로파일 잡의 per-layer timing은 TIDL의 `runtimes_visualization.svg`와 같은 목적 — "어디서 시간이 새는가"를 본다. AI Hub는 미리 컴파일된 ONNX Runtime 모델(QNN 바이너리 임베드)도 프로파일할 수 있어, 위 `qnn_infer.py` 산출물과 자연스럽게 이어진다.

#### 🔬 실측 (Qualcomm 축): AI Hub로 Hexagon HTP 실기기 프로파일 — QCS8550 · SA8775P ADP

> **이건 진짜 벤더-NPU 실측이다.** §2-2가 "모든 op가 CPU로 폴백된 바닥값(offload 0%)"이었다면, 여기서는 같은 ResNet50(3·5단계 자산)을 AI Hub 클라우드의 **실제 Snapdragon Hexagon HTP** 두 종에 올려 **on-device 지연 + NPU offload%**를 측정했다. 위 `qai_hub_profile.py` 골격을 그대로 돌린 결과다(`--target_runtime qnn_context_binary`, qai_hub 0.54.0).

**헤드라인: 두 디바이스 모두 100% NPU offload, INT8이 fp16 대비 ×1.77·×2.03.**

| 디바이스 · 정밀도 | on-device 지연 | INT8 배속 | NPU offload | cycles |
|---|--:|:--|:--|--:|
| **QCS8550** (Proxy) · FP32→fp16 | 1864 µs | — | **100%** (125/125) | 4,677,822 |
| **QCS8550** · INT8 QDQ | **1052 µs** | **×1.77** | **100%** (128/128) | 3,754,903 |
| **SA8775P ADP**(자동차) · FP32→fp16 | 3056 µs | — | **100%** (125/125) | 6,192,577 |
| **SA8775P ADP** · INT8 QDQ | **1505 µs** | **×2.03** | **100%** (128/128) | 4,462,570 |

- **깨끗한 CNN이라 폴백 0** — 이 단계의 이상형("Offloaded ≈ Total, subgraph 최소")을 벤더 실기기에서 정량 달성. §2-2 CPU 바닥값 위에 "가속기가 실제로 얼마나 버는가"가 얹힌다. INT8 배속은 execution_cycles로 교차 확증(FP32 사이클 > INT8 사이클).
- **HTP엔 native fp32가 없다** — FP32 ONNX도 그래프 첫머리에 `..._FLOAT_32_converted_..._FLOAT_16` 노드가 삽입돼 **fp16으로 실행**된다. 그래서 "FP32 대비 배속"은 엄밀히 "fp16 대비 int8 배속"이다.
- **🔴 함정 1 — AI Hub 프론트엔드는 ORT/TRT보다 엄격.** ORT 양자화기가 넣은 `logits`가 value_info+graph-IO 양쪽에 있으면(ONNX 스펙 위반) **컴파일이 거부**된다(`Tensors {'logits'} occur in value_info but also in model IO`). ORT·TRT는 통과시킨다. → IO와 충돌하는 value_info를 제거하고 제출(`onnx.checker` PASS, 계산 불변).
- **🔴 함정 2 (silent-wrong) — 외부 QDQ를 지참하면 on-device INT8 정확도가 조용히 붕괴.** compile/profile은 통과(100% offload)해도, ORT가 만든 INT8 QDQ를 HTP로 임포트하면 **on-device top-1이 0.75→0.005로 붕괴**한다(200장). 같은 경로의 **FP32(fp16)는 충실**(0.745, ORT와 96% 일치)하므로 입력·전처리·파싱은 정상 — 범인은 "외부 QDQ scale을 HTP 임포트가 존중하지 않음". exit 0·정상 shape라 조용하다. **올바른 경로는 지참 QDQ가 아니라 AI Hub 자체 `submit_quantize_job`**(HTP-native QDQ): 같은 200장에서 **top-1 0.735·ORT 일치 0.94로 회복**(FP32 0.745·ORT-CPU 0.750에 근접, 외부 QDQ 0.005 붕괴와 대조) → 붕괴가 임포트 특유였음을 확정. 게다가 native-quant는 더 leaner해 **748 µs**(외부-QDQ INT8 1052µs보다 빠름)로, HTP-native 양자화가 더 최적화된 그래프를 만든다.

> 📄 전체 실측·SVG·판정: [`../logs/stage4_qualcomm_aihub_report.html`](../logs/stage4_qualcomm_aihub_report.html) · 데이터·스크립트·설계규칙: [`../experiments/stage4_qualcomm_aihub/`](../experiments/stage4_qualcomm_aihub/)
> **캐비앗:** 절대 지연은 on-device `estimated_inference_time`(HTP 추정)·배치1, top-1은 200장 서브셋 → **상대 관계(INT8 배속·offload·FP32충실 vs INT8붕괴)만 유효**. AI Hub는 Qualcomm 전용이라 이 실측은 세 벤더 중 **Qualcomm 축**만 채운다(TI TDA4VM·Renesas RZ/V2H는 4-A·4-C, 보드/툴체인 대기). "QCS8550 (Proxy)"는 프록시 디바이스, "SA8775P ADP"는 실제 자동차 보드.

---

### 4-C. Renesas DRP-AI — `rzv_drp-ai_tvm`

Renesas RZ/V 계열(RZ/V2H, RZ/V2N에 DRP-AI3; RZ/V2L·V2M·V2MA는 구형 DRP-AI)에 배포한다. **TVM 기반** 컴파일러(**EdgeCortix MERA** 컴파일러 프레임워크로 구동)라 미지원 레이어는 TVM이 CPU로 위임한다. 리포는 최근 **RUHMI(Robust Unified Heterogeneous Model Integration)** 로도 브랜딩되지만, 폴더/스크립트 이름은 여전히 `rzv_drp-ai_tvm` · `compile_onnx_model_quant.py`다.

#### 설치

```bash
cd $WORK
git clone https://github.com/renesas-rz/rzv_drp-ai_tvm.git
cd rzv_drp-ai_tvm
# 설치는 리포의 setup / installation 가이드를 따른다(TVM 빌드 + DRP-AI Translator + DRP-AI Quantizer 설치).
# 아래 환경변수를 export 해야 컴파일 스크립트가 각 도구를 찾는다.
export TVM_ROOT=$WORK/rzv_drp-ai_tvm            # TVM(=DRP-AI TVM) 루트
export SDK=/opt/poky/<ver>/sysroots/...         # RZ/V Linux 패키지에서 생성한 Yocto SDK 경로
export TRANSLATOR=/opt/drp-ai_translator_release # DRP-AI Translator 설치 경로
export QUANTIZER=/opt/drp-ai_quantizer           # DRP-AI Quantizer 설치 경로
```

> ⚠️ **확인 필요**: `SDK`/`TRANSLATOR`/`QUANTIZER`/`TVM_ROOT`의 **실제 설치 경로**는 사용자가 Renesas에서 받은 패키지 버전에 따라 다르다. 위는 튜토리얼의 변수명(`$SDK $TRANSLATOR $QUANTIZER $TVM_ROOT`)을 확인한 것이며, 경로 값은 각자 설치에 맞춰 넣어야 한다. (출처: [rzv_drp-ai_tvm tutorial_RZV2H.md](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/tutorials/tutorial_RZV2H.md))

#### 컴파일 (양자화 → translate) — RZ/V2H/V2N은 DRP-AI Quantizer 필수

DRP-AI3 MAC은 **INT8 연산만** 한다. 따라서 **translation 전에 반드시 양자화**해야 한다. `compile_onnx_model_quant.py`가 캘리브레이션 이미지로 PTQ INT8을 만든 뒤 translate까지 한 번에 처리한다.

```bash
cd $WORK/rzv_drp-ai_tvm/tutorials

# RZ/V2H 예시 (튜토리얼 명령을 그대로 확인함, 2026-07)
python3 compile_onnx_model_quant.py \
    ../model/resnet18-v1-7.onnx \              # (positional) 입력 ONNX 경로
    -o resnet18_onnx \                          # 출력 디렉토리 이름
    -t $SDK \                                   # Yocto SDK 경로
    -d $TRANSLATOR \                            # DRP-AI Translator 경로
    -c $QUANTIZER \                             # DRP-AI Quantizer 경로
    -s 1,3,224,224 \                            # 입력 shape (N,C,H,W). 모델 입력과 일치
    -n 10 \                                     # 캘리브레이션에 쓸 이미지 수 (예: 10장)
    --images $WORK/calib \                      # 캘리브레이션 이미지 폴더 (학습과 동일 전처리로!)
    --mera2                                     # RZ/V2H·V2N 새 흐름(MERA2). V2L/V2M/V2MA는 --mera1
# 참고 인자:
#   -p / --quantization_option "<opts>"  : DRP-AI Quantizer에 넘길 추가 옵션(단, 한 번만 지정 가능)
# ONNX/ExIR 포맷은 V2H/V2N에서 기본이 MERA2, PyTorch(.pt)는 기본 MERA1.
```

각 인자 요약:

| 인자 | 의미 | 비고 |
|------|------|------|
| (positional) | 입력 모델 경로 | ONNX |
| `-o` | 출력 디렉토리 | translate 산출물 + `preprocess/`(전처리 오브젝트) 생성 |
| `-t` | Yocto SDK | RZ/V Linux 패키지에서 생성 |
| `-d` | DRP-AI Translator | 그래프→DRP-AI 매핑 |
| `-c` | DRP-AI Quantizer | INT8 PTQ 수행(V2H/V2N 필수) |
| `-s` | 입력 shape | `1,3,224,224` 처럼 콤마 구분 |
| `-n` | 캘리브레이션 이미지 수 | 수십 장이면 충분 |
| `--images` | 캘리브레이션 폴더 | **학습과 동일 전처리** |
| `--mera2` / `--mera1` | 컴파일 프레임워크 선택 | 기기·포맷에 따라 기본값 상이 |

> 🔴 **함정 (실무 최다 실수) — 캘리브레이션 전처리를 학습과 정확히 일치시켜라**
> `--images` 폴더의 이미지에 적용되는 **mean/std/resize/색공간**이 학습 때와 조금이라도 다르면, **컴파일은 성공하는데 정확도만 조용히 붕괴**한다(에러가 안 나서 더 위험). 튜토리얼 기본 ResNet 전처리는 mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` 인데, **이 값을 반드시 네 학습 파이프라인과 맞춰라.** DRP-AI에서 나는 "정확도 이상"의 절반 이상이 이것이다.

**전처리 일치 검증 코드.** 컴파일에 넣기 전에, 캘리브레이션 파이프라인이 뽑는 텐서가 학습 텐서와 같은지 수치로 확인한다. "눈으로 봤으니 맞겠지"가 이 함정의 시작이다.

```python
# verify_preprocess.py — DRP-AI 캘리브레이션 전처리가 학습과 일치하는지 수치로 검증
import numpy as np, sys, os
sys.path.insert(0, os.environ.get("WORK", "."))
from preprocess import load_chw, MEAN, STD, SIZE   # 3절의 공유 전처리

x = load_chw("calib/sample_0.jpg")   # 캘리브레이션이 실제로 뽑는 텐서
print("shape        :", x.shape)                      # 기대: (3, 224, 224)
print("dtype        :", x.dtype)                       # 기대: float32
print("per-ch mean  :", x.reshape(3, -1).mean(1))      # 정규화 후라 대략 0 근처(입력 분포 따라 다름)
print("per-ch std   :", x.reshape(3, -1).std(1))       # 대략 1 근처
print("min/max      :", x.min(), x.max())              # 정규화 범위 확인(예: -2.1 ~ +2.6)

# 학습 코드의 transform과 "같은 이미지"에 대해 동일 값이 나오는지 대조하는 게 핵심.
# 예: torchvision 파이프라인이 있으면 그 출력과 np.allclose 로 비교.
# assert np.allclose(x, train_transform(img).numpy(), atol=1e-4), "전처리 불일치!"
```

> 💡 **팁 — 색공간·정규화 순서가 은근한 함정**
> `RGB vs BGR`, `/255 먼저 vs mean-std 먼저`, `resize 보간법(bilinear/nearest)`, `[0,1] vs [0,255] 스케일에서 정규화` — 이 조합 중 하나만 어긋나도 값이 통째로 틀어진다. `min/max`가 학습 때와 다른 범위면(예: 학습은 -2~+2인데 여기선 0~1) **정규화가 빠진 것**이다.

#### DRP-AI3 양자화 요구사항 & 구조 제약

- **INT8 전용**: DRP-AI3 MAC은 INT8만 → 양자화 없이는 아예 매핑 불가.
- **모든 노드 입력이 각자 initializer를 가질 것**: DRP-AI3 양자화는 각 노드의 입력(weight/bias 등)이 **자기 initializer**를 갖도록 요구한다. 공유/동적으로 계산되는 입력 구조는 전처리로 풀어(상수 폴딩·복제) 각 입력이 개별 initializer를 갖게 만들어야 한다.
- **feed-forward 전용**: 루프·재귀 계열(**RNN/LSTM/GRU**)은 DRP-AI에 매핑 불가. 시계열/순환 구조는 이 백엔드 대상이 아니다.

> 💡 **직관 — "각 입력이 자기 initializer" 요구가 왜 있나**
> DRP-AI Translator는 컴파일 시점에 각 연산의 가중치를 **정적으로 배치**해야 한다. 두 노드가 같은 텐서를 동적으로 공유하거나 런타임에 계산되는 입력을 쓰면, 정적 배치가 깨져 매핑이 실패한다. 그래서 export/전처리 단계에서 **상수는 상수로 굳혀(fold)** 두는 것이 안전하다.

#### fallback 로그 읽기 (MERA1 vs MERA2)

DRP-AI TVM은 **TVM 기반**이라, DRP-AI가 못 맡는 op는 **TVM이 CPU(Arm Cortex-A)로 위임**한다. 컴파일 산출물/로그에서 **DRP-AI로 간 연산 vs CPU로 위임된 연산**의 비율을 본다. MERA1/MERA2는 관측 도구가 다르다:

| 구분 | MERA1 | MERA2 (v2.7.0~) |
|------|-------|-----------------|
| 대상 기기 | RZ/V2L, V2M, V2MA | RZ/V2H, V2N |
| 그래프 최적화 | 기본 | 강화(YOLOv8 등 개선) |
| 런타임 Python API | 제한 | **지원(V2H/V2N)** |
| IR 뷰어 | 없음 | **있음(V2H/V2N)** — 어느 노드가 DRP-AI/CPU로 갔는지 시각 확인 |
| CPU 위임 관측 | 컴파일 로그의 위임 메시지 | IR 뷰어 + 로그 |

> ⚠️ **확인 필요**: DRP-AI TVM 컴파일 로그의 정확한 "CPU 위임 노드 목록" 출력 형식/파일명은 리포 버전(MERA1/MERA2)에 따라 다르다. `tutorials/` 및 `docs/About_mera.md`의 로그 예시로 재확인할 것. **MERA2(V2H/V2N)라면 IR 뷰어**로 파티션을 시각 확인하는 것이 가장 확실하다. 원칙은 TIDL과 동일: **CPU 위임이 늘수록 느려진다.** (출처: [rzv_drp-ai_tvm docs/About_mera.md](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/About_mera.md))

> 💡 **팁 — 프로파일링 how-to가 리포에 있다**
> `rzv_drp-ai_tvm/how-to/tips/profiling/` 에 레이어별 프로파일 방법이 있다. TIDL의 `debug_level`/HTML 뷰어, QNN의 AI Hub per-layer timing과 **같은 목적** — "가속기에 남은 비율"과 "느린 레이어"를 찾는다.

---

## 5) 예시 / 결과 해석 — "fallback 비율부터 확인" (정량 지표)

세 백엔드를 다 돌렸으면, 제일 먼저 볼 것은 정확도도 latency도 아니고 **offload/fallback 비율**이다. 벤더마다 이름이 다를 뿐 다음 두 숫자를 **반드시 계산**한다:

- **offload 비율** = `가속 노드 / 전체 노드` (목표 ≥ 90%, 이상적 100%)
- **파티션 수(subgraph 수)** = 그래프가 몇 토막 났는가 (목표 1, 나빠도 한 자릿수 초반)

아래처럼 정리한다(값은 예시).

| 백엔드 | 정밀도 | offload(가속) 노드 | fallback(CPU) 노드 | offload 비율 | subgraph/파티션 수 | 판정 |
|--------|--------|--------------------|--------------------|--------------|--------------------|------|
| TIDL (AM62A) | INT8 | 196 / 236 | 40 | **83%** | 3 | 양호 — subgraph 낮음 |
| TIDL (재시도, deny_list 정리) | mixed | 228 / 236 | 8 | **97%** | 1 | 우수 — 단일 subgraph |
| QNN (HTP) | uint16/uint8 | (전부 QNN) | 0 (`disable_cpu_ep_fallback=1`로 강제) | **100%** | 1 | 우수 — CPU 0 |
| DRP-AI (RZ/V2H) | INT8 | 대부분 DRP-AI | 소수 CPU 위임 | (IR 뷰어로 산출) | — | 위임 op 확인 필요 |

해석 원칙 (순서대로):
1. **offload 비율이 낮은 / fallback 비율이 높은 백엔드부터 손본다.** latency 튜닝보다 이게 먼저다. "80% 미만"이면 뭔가 구조적으로 안 맞는 것 — 우선 그 백엔드를 파고든다.
2. **subgraph/파티션 개수를 1에 가깝게.** 개수가 크면 offload %가 높아도 데이터 이동으로 느리다. offload 97%인데 subgraph 8개보다, offload 90%에 subgraph 1개가 대개 더 빠르다.
3. 특정 op 하나가 파티션을 계속 쪼개면 → 그 op를 (a) 그래프에서 제거/치환(2단계로 회귀), (b) `deny_list`로 CPU 고정(경계 정리), (c) 벤더 지원 버전으로 태그 업그레이드.
4. **정확도 갭**은 그 다음. 8-bit로 부족하면 문제 레이어만 16-bit 승격(TIDL `*_16bit_names_list`/`mixed_precision_factor`, QNN 활성값 `QUInt16`). DRP-AI는 8-bit 고정이므로 정확도가 부족하면 QAT나 그래프 수정으로 간다.

> 💡 **팁 — 세 로그의 "같은 자리"를 본다**
> TIDL은 `Offloaded Nodes / Total Nodes` + `Final number of subgraphs`, QNN은 `disable_cpu_ep_fallback`가 잡아주는 로드 에러/경고(또는 AI Hub per-layer), DRP-AI는 TVM의 CPU 위임 노드(MERA2는 IR 뷰어). **이름만 다르고 읽는 목적은 하나: "얼마나 가속기에 남았는가."** 이 두 숫자(offload%·subgraph 수)를 세 백엔드에 대해 채운 표가 곧 산출물이다.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| (공통) 컴파일은 되는데 정확도만 급락 | **캘리브레이션 전처리(mean/std/resize/색공간)가 학습과 불일치** | 전처리를 학습 파이프라인과 **바이트 단위로** 일치. `verify_preprocess.py`로 min/max·per-ch 통계 대조. DRP-AI 최다 원인. |
| (공통) provider를 못 찾음 / 심볼 에러 | 세 벤더 onnxruntime을 **한 venv에 혼재** | TIDL(Docker)/`.qnn`/`.drpai` 환경 분리. 각 환경에서만 해당 툴 실행. |
| TIDL: subgraph가 5~10개로 쪼개짐 | 중간에 미지원/비효율 op가 파티션을 끊음 | `runtimes_visualization.svg`에서 경계 op 찾기 → `deny_list`로 CPU 고정 또는 그래프에서 제거. |
| TIDL: `allowedNode.txt` 미생성 | 파티셔닝 실패/미지원 op 과다 | `debug_level=1`(필요 시 2+)로 parsing summary 확인, 태그를 보드 SDK에 맞는 것으로 재선택. |
| TIDL: PTQ 정확도 미달 | 내부 캘리브레이션 한계 | `calibration_frames/iterations` ↑ 또는 `accuracy_level` ↑; 그래도 부족하면 QAT 재학습 또는 **pre-quantized QDQ ONNX** 입력으로 캘리브레이션 우회. |
| QNN: 모델 로드시 op 관련 에러 | `Loop`/`If`/**동적 shape** 등 QNN EP 미지원 | 제어흐름 제거 + batch 고정으로 export(2단계). 불가피하면 해당 부분만 CPU EP 허용(disable_cpu_ep_fallback 끔). |
| QNN: `quantize`/유틸이 ARM64에서 실패 | 양자화 유틸은 **x86_64 전용** | 양자화는 이 데스크톱(x86)에서, 실행은 디바이스/AI Hub에서. |
| QNN: 매 실행 초기화가 느림 | HTP 그래프 finalize 반복 | **context binary** 생성(`ep.context_enable=1`)해 캐시. 단, SoC 바뀌면 재생성. |
| QNN: 다른 기기에서 context 캐시 로드 실패 | context binary는 **SoC/아키텍처 종속** | 타겟 SoC에서 재생성. `soc_model`/`htp_arch`로 타겟 명시. |
| DRP-AI: translate 단계에서 매핑 실패 | 양자화 안 함 / RNN·LSTM·GRU 포함 | INT8 양자화 선행(V2H/V2N `-c $QUANTIZER`). 순환 구조는 DRP-AI 대상 아님(feed-forward만). |
| DRP-AI: 양자화기 initializer 에러 | 노드 입력이 **자기 initializer 미보유** | DRP-AI3 요구사항 — 각 노드 입력이 개별 initializer를 갖도록 그래프 전처리(상수 폴딩·복제). |
| DRP-AI: 추론이 예상보다 느림 | **CPU(TVM) 위임** 과다 | MERA2 IR 뷰어/로그로 위임 노드 확인, 미지원 op를 지원 op로 치환/제거. |
| DRP-AI: `--mera2`인데 동작 이상 | 기기/포맷과 프레임워크 불일치 | V2L/V2M/V2MA는 `--mera1`, V2H/V2N은 `--mera2`. PyTorch(.pt)는 기본 MERA1임에 유의. |

---

## 7) 산출물 (Deliverables)

이 단계가 끝나면 다음이 남아야 한다:

- `preprocess.py` — 세 백엔드가 공유한 **단일 전처리 정의**(학습과 일치). `verify_preprocess.py` 검증 로그.
- `model/model.qdq.onnx` — QNN용 QDQ ONNX(및 TIDL/DRP-AI 입력으로 재사용한 모델).
- TIDL `model-artifacts/` + `tempDir/runtimes_visualization.svg` + `allowedNode.txt` — 파티셔닝 시각화/허용 노드.
- QNN `model_ctx.onnx` — HTP context binary 캐시(디바이스에서 생성 시). AI Hub 프로파일 리포트(보드 없을 때).
- DRP-AI `resnet18_onnx/`(또는 지정한 `-o` 폴더) — translate 산출물 + `preprocess/` 오브젝트.
- **백엔드 비교표(5절 형식)** — 백엔드별 **offload 비율 + subgraph 수** + INT8 요구사항 + 미지원 op를 한 장으로 정리한 표. **이 표가 JD의 'multi-target deployment pipeline' 결과물의 핵심.**
- 각 백엔드 컴파일 로그에서 **fallback을 유발한 op 리스트**와 그 처리(제거/치환/deny_list) 기록.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [TI edgeai-tidl-tools (GitHub)](https://github.com/TexasInstruments/edgeai-tidl-tools) — TIDL 컴파일/추론 툴·예제. 태그: `11_02_16_00`(stable), `11_02_17_00`(2026-07 patch).
- [TI edgeai-tidl-tools Releases](https://github.com/TexasInstruments/edgeai-tidl-tools/releases) — SDK 버전↔태그 매핑, 지원 플랫폼(J721E/J721S2/J784S4/J722S/AM62A).
- [TI edgeai-tidl-tools model_compilation.md](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/model_compilation.md) — `tensor_bits`/`accuracy_level`/`calibration_*`/`mixed_precision_factor`/`*_16bit_names_list`/`deny_list` 옵션 정의.
- [TI edgeai-tidl-tools debugging.md](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/debugging.md) — `debug_level`(0~5) 의미, parsing summary(offload vs CPU 노드), 레이어 트레이스 덤프.
- [TI TIDL 지원 op 문서 (operators.md)](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/operators.md) — 가속 가능 op 목록. ⚠️ 구 문서 `docs/supported_ops_rts_versions.md`는 **404다** — 태그 `11_00_08_00`까지는 있었으나(21,683 B) `11_02_16_00` 재편에서 삭제됐다. 이 `operators.md`가 후속 문서(28,756 B, 2026-08-06 실측).
- [TI TIDL Quantization 문서 (quantization.md)](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/quantization.md) — 위 재편에서 신설. PTQ, simple vs advanced 캘리브레이션, advanced bias calibration, histogram 기반 range 수집, 자동/수동 혼합정밀도, QAT, 정확도 가이드라인을 벤더 관점에서 한 문서에 정리.
- [TI TVM User's Guide (Compiling Models)](https://software-dl.ti.com/codegen/docs/tvm/tvm_tidl_users_guide/compiling.html) — subgraph 파티셔닝/offload 로그 해석.
- [ONNX Runtime QNN Execution Provider 문서](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — `get_qnn_qdq_config`/`qnn_preprocess_model`/provider_options/context binary/미지원 op(Loop·If·동적 shape).
- [onnxruntime-qnn (GitHub)](https://github.com/onnxruntime/onnxruntime-qnn) — QAIRT EP 패키지·문서.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 클라우드 실기기 컴파일·프로파일(보드 없을 때 실측 대체).
- [Qualcomm AI Hub 문서 (compile/profile 예제)](https://app.aihub.qualcomm.com/docs/) — `qai_hub` 컴파일/프로파일/추론 잡 API.
- [Renesas rzv_drp-ai_tvm (GitHub)](https://github.com/renesas-rz/rzv_drp-ai_tvm) — DRP-AI TVM 컴파일러(RUHMI / EdgeCortix MERA 기반).
- [rzv_drp-ai_tvm RZ/V2H 튜토리얼](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/tutorials/tutorial_RZV2H.md) — `compile_onnx_model_quant.py` 명령·전처리·환경변수.
- [rzv_drp-ai_tvm About_mera.md](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/About_mera.md) — MERA1/MERA2 차이(대상 기기·`--mera2`·IR 뷰어).
- [DRP-AI Quantizer 사용자 매뉴얼](https://www.renesas.com/en/document/mas/drp-ai-quantizer-v101-users-manual) — INT8 양자화 요구사항.
- [Renesas RZ/V2H 제품 페이지](https://www.renesas.com/en/products/rz-v2h) — DRP-AI3 사양.

### 논문 (양자화 이론 복습용)
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Integer-Arithmetic-Only Inference*, arXiv:1712.05877 — INT8 정수 추론의 원조(세 백엔드 모두의 전제).
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295 — PTQ/QAT·per-channel·혼합정밀도 실무 백서.
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient Neural Network Inference*, arXiv:2103.13630 — 양자화 전반 서베이.

---

## 9) 다음 단계

- 이전: **[3단계 — TensorRT](05_tensorrt.md)** (단일 벤더 심화)
- 다음: **[5단계 — 인프라화](07_infrastructure.md)** — 지금까지 손으로 돌린 multi-target 컴파일을 CI/파이프라인으로 자동화하고, 백엔드별 아티팩트·프로파일을 재현 가능하게 묶는다.

> ✅ **이 단계 한 줄 요약**: 툴체인이 셋이어도 문제는 하나다. **ONNX → 벤더 지원 부분집합 → INT8 가능 부분집합 → 나머지 CPU fallback**, 그리고 **fallback 최소화가 전부**다. 세 백엔드에서 `offload 비율`과 `subgraph 수` 두 숫자를 뽑아 비교표를 채우면, 어떤 새 SoC가 와도 이 사다리를 그대로 적용하면 된다.
