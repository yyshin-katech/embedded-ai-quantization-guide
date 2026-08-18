# embedded-ai-quantization-guide

**AI 모델 양자화 → 임베디드 배포 실전 학습 가이드.** 멀티카메라 Transformer 인식 모델을 INT8 양자화하여 멀티 SoC(NVIDIA Orin/Thor · TI Jacinto · Qualcomm · Renesas RZ/V)에 올려 구동하는 전 과정을, **Ubuntu 22.04 + NVIDIA RTX GPU**에서 "읽고 따라 하면 실제로 실행되는" 단계별 문서로 정리했습니다.

> 모든 버전·링크는 2026-07 기준으로 웹 검증했습니다. 실제 설치 시점엔 각 공식 페이지에서 재확인하세요.
>
> ✅ **0단계 환경은 2026-07-31 실제 머신에서 설치·검증 완료**했습니다(Ubuntu 22.04.5 / RTX 3060 / 드라이버 595.84 / `nvcc` 12.8.93). 아래 버전 스택과 [`01_environment_setup.md`](study_guide/01_environment_setup.md)의 예상 출력은 그 **실측값**입니다. 실제로 따라 하며 남긴 커맨드·출력·함정 해결 과정은 [`logs/`](logs/)에 있습니다.
>
> ✅ **0.5단계(배포 사다리)·1단계(양자화 이론)도 2026-08-02 실측 완료**했습니다. 1단계에서는 ORT의 **Entropy 캘리브레이터가 기본값에서 MinMax로 조용히 퇴화**하고(산출 ONNX의 md5까지 동일), ORT 권장 설정(`QUInt8` 비대칭)으로 만든 INT8 QDQ를 **TensorRT가 파싱조차 못 해 FP32보다 3배 느려지는**(3.06 ms vs 0.96 ms) 무음 폴백을 잡아냈습니다 — 전 과정은 [`logs/stage1_quantization_log.html`](logs/stage1_quantization_log.html).
>
> 🔁 **2026-08-06, 1단계를 ImageNet val 50,000장 전량으로 재실행**했습니다. 1차는 클래스당 1장 큐레이션 셋(1,000장)이었는데, 전량으로 다시 재니 **절대 top-1이 평균 +9.77%p 부풀려져 있었고, Δ의 부호가 3건·유의성 판정이 5건 뒤집혔습니다**. FP32는 공개 재현값과 **0.05%p** 안에서 만납니다(69.81% vs 69.758%). 여기서 나온 정정 12건을 [`03`](study_guide/03_quantization_theory.md)·[`05`](study_guide/05_tensorrt.md)·[`10`](study_guide/10_pitfalls.md)에 반영했습니다 — **작은 평가셋으로는 나머지 함정을 진단할 수 없다**는 것이 [10단계 함정 0](study_guide/10_pitfalls.md)으로 새로 들어갔습니다. 실행 로그 [`logs/stage1_real_imagenet_log.html`](logs/stage1_real_imagenet_log.html) · 분석 보고서 [`logs/stage1_real_imagenet_report.html`](logs/stage1_real_imagenet_report.html).
>
> ✅ **2026-08-16, 2단계(Transformer 양자화 지옥)를 RTX 3080에서 실측 완료**했습니다. `facebook/detr-resnet-50`을 ONNX export→INT8 PTQ→**COCO val2017 전량 5,000장** mAP까지 완주하니 초안의 경험적 단정 **3가지가 뒤집혔습니다**: ① export 첫 블로커는 통념의 `grid_sampler`가 아니라 **SDPA**(DETR엔 grid_sample이 아예 없음), ② "특정 op만 FP로 빼면 회복"은 attention matmul 36개를 FP로 남겨도 **+0.36 mAP뿐(사실상 실패)**, ③ 폭락(mAP **0.4207→0.2402, −42.9%**, 작은 객체 **−77%**)의 범인은 attention이 아니라 **망 전체에 분산**(절제로 확인). FP32는 공개값 **42.0**과 일치해 계측을 신뢰할 수 있습니다. 실측 리포트 [`logs/stage2_detr_quantization_report.html`](logs/stage2_detr_quantization_report.html) · 실패 로그 [`experiments/stage2_detr/onnx_export_failures.md`](experiments/stage2_detr/onnx_export_failures.md).
>
> ✅ **2026-08-17, 2단계 §4.6 BEVFormer-tiny(grid_sample/Deformable Attention)까지 실측**했습니다. 정본 venv로는 mmdet3d/mmcv가 안 돌아 **무컴파일 레거시 venv**(프리빌트 휠 + 드라이버 하위호환)를 따로 세우고 `fundamentalvision/BEVFormer`를 완주했습니다. op 단위 단정은 **반전 0건(초안이 맞음)** — grid_sample opset 경계·5D=opset20·ORT 1.23.2 무음 CPU 폴백·TRT rank-4 단언·MSDeformAttn 분해 전부 실측 일치. 대신 실전 함정 **2개**를 새로 채집했습니다: ① 바닐라 mmcv 커스텀 op은 **CPU 텐서로만 유효 export**(CUDA는 출력을 `Constant`로 baked→입력 소실, exit 0인데 **silent-wrong**), ② 전체 모델 export는 grid_sample이 아니라 **`point_sampling`의 `lidar2img` 사영에서 먼저 사망**(`RuntimeError: shape '[1,6,6,1,4,4]'…`). FP32 nuScenes-mini mAP **0.2647**/NDS 0.2667이나 이는 **81샘플·2 scene 고분산 스모크**(문헌비교 불가, 상대 델타만); 전체 INT8은 유효 export 경로가 없어 **범위 밖**(포크 플러그인 재빌드 전제). 실측 리포트 [`logs/stage2_bevformer_quantization_report.html`](logs/stage2_bevformer_quantization_report.html) · 실패 로그 [`experiments/stage2_bevformer/onnx_export_failures.md`](experiments/stage2_bevformer/onnx_export_failures.md).

> ✅ **2026-08-17, 2단계 §4.4 SmoothQuant까지 실측**했습니다. DETR 폭락 리포트(§4.5)가 지목한 "진짜 레버 = activation 양자화 입도"를 **nvidia-modelopt 0.45.0** SmoothQuant로 직접 시험했습니다(COCO val2017 전량 5,000장, torch fake-quant 자기일관 3원). 결과: per-tensor INT8이 **0.4209→0.3301(−0.0908)** 무너진 폭락을, SmoothQuant(α=1.0)가 **0.3845로 gap의 59.9%(+0.0544 mAP)** 되찾습니다 — §4.5의 op-선택 mixed(+0.0036)의 **약 15배**로, "op 선택이 아니라 activation 입도가 레버"라는 §4.5 판정 4를 **실측 확증**합니다. 초안 오류도 정정: modelopt 프리셋 기본 α는 논문의 0.5가 아니라 **1.0**(DETR에선 α=1.0이 66.6% > α=0.5의 49.8%), absmax 감쇠 실측 **3.69×→1.96×**. ⚠️ 절대 mAP는 torch fake-quant 경로라 커밋된 §4.5 ORT QDQ 절대값(0.4207/0.2402)과 **1:1 비교 불가**(상대 관계만 유효). 실측 리포트 [`logs/stage2_smoothquant_report.html`](logs/stage2_smoothquant_report.html) · 재현 [`experiments/stage2_smoothquant/`](experiments/stage2_smoothquant/).

> ✅ **2026-08-17, 3단계(TensorRT로 첫 완주)를 RTX 3080에서 실측 완료**했습니다. `torchvision` ResNet50을 ONNX→TensorRT **10.16.1.11**(pip 휠)로 FP32/FP16/INT8 3점까지 완주하며 초안의 단정 3가지를 정정했습니다: ① 문서가 전부 `trtexec` 명령으로 쓰였으나 정본 pip 휠(`tensorrt-cu12`)에 **trtexec 실행파일이 없어**(PATH·파일시스템 0건) 그대로는 실행 불가 → **polygraphy 0.50.3 Python API**로 동일 결과를 냅니다(FP32 1.66ms→FP16 0.85ms **×1.96**→INT8 0.78ms **×2.12**, −0.52%p·엔진 122→25 MiB). ② 1단계 §2.2.1의 "TensorRT 폴백 원인 = activation zp≠0 하나뿐"은 **ORT TensorRT EP 경로 한정**이었습니다 — polygraphy/trtexec **직접 파서**는 INT32 bias DQ도 **독립 하드 블로커**로 거부(대칭·zp=0인데도 parse 실패)해 블로커가 **둘**입니다(경로 병기 정밀화이지 1단계 반전이 아님 — ORT-EP가 파서 전 bias DQ를 흡수). ③ deprecated된 implicit `IInt8EntropyCalibrator2`가 TRT 10.16에서 **여전히 빌드**(경고만 134건)되고 이 모델선 explicit보다 빠르고 정확하나, 제어성·제거예정 탓에 신규는 explicit 권장. DLA(실습5)는 RTX 3080 dGPU에 `num_DLA_cores=0`이라 **범위 밖**(정직한 폴백). 실측 리포트 [`logs/stage3_tensorrt_report.html`](logs/stage3_tensorrt_report.html) · 파서 제약 로그 [`experiments/stage3_tensorrt/parser_constraints.md`](experiments/stage3_tensorrt/parser_constraints.md).

> ✅ **2026-08-17, 5단계(인프라화 — 벤치 하네스·CI·회귀 게이트)를 RTX 3080에서 실측 완료**했습니다. 문서의 하네스 골격(ABC 백엔드 인터페이스→config 순회→pandas 매트릭스→pytest 회귀 게이트)을 **실행 가능한 형태로 완성**하고 ResNet50/ImageNet으로 관통시켜 **8건**을 정정했습니다(BEVFormer는 2단계 결론상 INT8 유효 export 경로 없음 → 검증은 3단계 자산 ResNet50). 그중 **2건은 exit 0·에러 0으로 통과하며 결과만 조용히 틀리는 무음 오답**이라 실행해야만 드러납니다: ① 정본 pip 휠엔 `pycuda`가 없어 polygraphy `TrtRunner`로 대체하는데, 이 러너가 **호스트 출력버퍼를 재사용(zero-copy)**해 정확도 eval의 5,000개 예측이 **전부 마지막 추론을 가리켜 top-1 0.0014(=1/1000)**로 붕괴 → `.copy()` 한 줄로 **0.7688** 복구, ② `pivot_table`의 기본 `dropna=True`가 stub '보드필요' 회색 행을 조용히 버려 **§5-1의 "회색행 보존" 원칙을 자기위반** → `dropna=False`. 나머지는 pycuda/trtexec 부재(3단계와 같은 결)·INT8 캘리브레이터 미배선·`device_memory_size_v2`·`data.py` 미제공·`EXPLICIT_BATCH` deprecated·pytest-regressions "v3.0+"(실제 최신 **2.11.0**) 정정입니다. 실측 매트릭스는 FP32 1.837ms→FP16 1.0231ms **×1.80**→INT8 0.8628ms **×2.13**(top-1 0.768=3단계 t04 일치)이며, 회귀 게이트는 **의도적 회귀 주입→3 테스트 FAIL→복원 통과**로 실증했습니다. SoC 백엔드 3종(TIDL/QNN/DRP-AI)은 실물 없어 **회색 stub**(4단계 과제). 실측 리포트 [`logs/stage5_infrastructure_report.html`](logs/stage5_infrastructure_report.html) · 제약 로그 [`experiments/stage5_infrastructure/harness_constraints.md`](experiments/stage5_infrastructure/harness_constraints.md).

> 🔬 **2026-08-18, 4단계 ARM Cortex-A 폴백 바닥값을 Raspberry Pi 5로 실측**했습니다(**부분·프록시** — 벤더 NPU가 아니라, 세 SoC가 공통으로 가진 **CPU 폴백 경로**를 Cortex-A76으로 측정). 3·5단계 자산 **ResNet50 INT8 QDQ**를 순수 `CPUExecutionProvider`로 관통시키니 **같은 INT8 그래프인데 CPU ISA가 양자화 이득의 부호를 뒤집습니다**: Pi 5(Cortex-A76, `asimddp`/SDOT **있음**)는 INT8이 **144.95→79.08ms ×1.83 빨라지고**, x86 i9-10900K(Comet Lake, **VNNI 없음**)는 되레 **9.28→16.34ms 1.76× 느려집니다**(dot-product 가속 부재 + quant/dequant 오버헤드). 덤으로 `CPUExecutionProvider`의 크로스플랫폼 예측 동일성은 **FP32에서만**(x86↔ARM 1000/1000=100%) 성립하고 **INT8은 958/1000(95.8%, 42장 상이)** — INT8 정확도를 서로 다른 타겟에서 비트 단위로 기대하면 안 됩니다. ⚠️ Pi는 폴백 프록시일 뿐 자동차 NPU가 아니며(가속 수치 전이 불가), 절대값은 CPUEP·배치1·1,000장 서브셋 기준 **상대 관계만 유효**. NPU 실측은 보드 확보 시 4-A~C 과제(이 CPU 바닥값이 "가속기가 이겨야 할 최소선"). 실측 리포트 [`logs/stage4_arm_cpu_fallback_report.html`](logs/stage4_arm_cpu_fallback_report.html) · 데이터·스크립트 [`experiments/stage5_infrastructure/cpu_proxy/`](experiments/stage5_infrastructure/cpu_proxy/).

> 🔬 **2026-08-18, 같은 4단계의 Qualcomm 벤더-NPU 축을 보드 없이 Qualcomm AI Hub 클라우드 실기기로 실측**했습니다(위 CPU 프록시가 못 준 "가속기가 실제로 얼마나 버는가"를 그 바닥값 위에 얹음). 같은 ResNet50을 Hexagon HTP 두 종(**QCS8550** Proxy · **SA8775P ADP** 자동차 보드)에 `qnn_context_binary`로 올려 **두 디바이스 모두 100% NPU offload**(깨끗한 CNN이라 폴백 0 — §06 이상형 "Offloaded≈Total, subgraph 최소"를 벤더 실기기에서 정량 달성), INT8이 fp16 대비 **×1.77·×2.03**(execution_cycles로 교차확증)을 확인했습니다. 🔴 **무음 오답 1건**: ORT가 만든 **외부 INT8 QDQ를 지참하면** compile/profile은 100% offload로 통과해도 **on-device top-1이 0.75→0.005로 조용히 붕괴**합니다(같은 경로 FP32(→HTP fp16)는 0.745로 충실 → 범인은 "HTP 임포트가 외부 QDQ scale을 존중 안 함", exit 0·정상 shape라 조용함). **올바른 경로 = AI Hub 자체 `submit_quantize_job`**(HTP-native QDQ)로 **top-1 0.735 회복**(외부 QDQ 0.005과 대조, FP32 0.745에 근접), 게다가 748µs로 외부-QDQ INT8(1052µs)보다 **더 빠릅니다**(native 양자화가 더 leaner한 그래프 생성). 그 밖에 AI Hub 프론트엔드가 `value_info↔IO` 충돌을 ORT/TRT보다 엄격 거부·HTP엔 native fp32 없음(fp16 실행)·엄격 NCHW 인터페이스도 실측. ⚠️ AI Hub는 Qualcomm 전용이라 세 벤더 중 **Qualcomm 축**만 채웁니다(TI TDA4VM·Renesas RZ/V2H는 보드/툴체인 대기); 절대값은 배치1·200장 서브셋 기준 **상대 관계만 유효**. 실측 리포트 [`logs/stage4_qualcomm_aihub_report.html`](logs/stage4_qualcomm_aihub_report.html) · 데이터·설계규칙 [`experiments/stage4_qualcomm_aihub/`](experiments/stage4_qualcomm_aihub/).

---

## 🚀 바로 시작

- **가이드 인덱스**: [`study_guide/README.md`](study_guide/README.md) — 여기서 시작하세요.
- **작업 인수인계**: [`HANDOFF.md`](HANDOFF.md) — 다른 PC에서 이어서 할 때. 어디까지 실측 검증됐고, git에 없는 것(데이터 27GB·venv·작업 스크립트)을 어떻게 확보하며, 진행 중인 **QAT 회복 실험**을 어떻게 재실행하는지. 이전 작업 머신의 GPU 하드웨어 고장(Xid 79) 진단도 포함.
- **학습 자료 모음**: [`learning_resources.html`](learning_resources.html) — 기초부터 순서대로 볼 수 있는 **사이트 34곳**(가이드 인용 23 + 보강 8, 링크 실측 검증 완료)과 가이드가 인용한 **논문 19편**의 원문 링크. 논문 PDF는 저작권상 재배포하지 않고 `paper/fetch_papers.py`로 받게 했습니다.
- **HTML로 편하게 보기**(다크 테마 · 진행률 체크박스 · 목차): 저장소를 클론한 뒤 `study_guide/README.html`을 브라우저로 엽니다.
  ```bash
  git clone https://github.com/yyshin-katech/embedded-ai-quantization-guide.git
  # study_guide/README.html 더블클릭 (또는 브라우저로 열기)
  ```
  > HTML을 웹에서 바로 보고 싶으면 GitHub Pages(Settings → Pages → `main` / root)를 켜면 됩니다.

---

## 📚 학습 로드맵

| # | 문서 | 단계 | 핵심 산출물 |
|---|------|------|------------|
| 01 | [환경 준비](study_guide/01_environment_setup.md) | 0 | 검증된 개발 환경 |
| 02 | [배포 난이도 사다리](study_guide/02_deployment_ladder.md) | 0.5 | 첫 온디바이스 배포 경험 |
| 03 | [양자화 이론](study_guide/03_quantization_theory.md) | 1 | `layer_sensitivity.csv` |
| 04 | [Transformer 양자화 지옥](study_guide/04_transformer_quantization.md) | 2 ★ | `onnx_export_failures.md` |
| 05 | [TensorRT로 첫 완주](study_guide/05_tensorrt.md) | 3 | Orin 성능 리포트 |
| 06 | [멀티 SoC 확장](study_guide/06_multi_soc.md) | 4 | 4-target 성능 매트릭스 |
| 07 | [인프라화](study_guide/07_infrastructure.md) | 5 | `design_rules.md`, 회귀 하네스 |
| 08 | [캡스톤 프로젝트](study_guide/08_capstone.md) | 캡스톤 | 공개 리포 + 블로그 |
| 09 | [12주 로드맵](study_guide/09_roadmap.md) | 로드맵 | 학습 스케줄 |
| 10 | [함정 5개 (+ 측정의 함정)](study_guide/10_pitfalls.md) | 함정 | 실무 체크리스트 |

각 문서 구조: `왜 → 체크리스트 → 이론 → 환경 → 실습 → 결과해석 → 트러블슈팅 → 산출물 → 참고문헌 → 다음`. 총 ~8,000줄.

---

## 📌 정본 버전 스택 (2026-07)

| 도구 | 버전 | 비고 |
|---|---|---|
| CUDA | 12.8 라인 고정 | 12/13 분열 회피 |
| PyTorch | `torch 2.11.0+cu128` | 기준선 |
| **onnx** | **`1.18.0` (IR 11)** | 🔴 **반드시 고정.** ORT 1.23.2의 IR 상한이 11 → 최신 onnx(IR 13)는 로드 실패. export 시 opset ≤ 23 |
| onnxruntime-gpu | **`1.23.2` (CUDA 12)** | `pip install "onnxruntime-gpu<1.27"`. 1.27+는 PyPI 기본이 CUDA 13 |
| TensorRT | `tensorrt-cu12==10.16.1.11` (10.x LTS) | 11.x는 `--int8/--fp16` 제거(strongly-typed) + CUDA 13 → 실습 호환 위해 10.x |
| numpy | **`1.26.4` (`numpy<2`)** | `nuscenes-devkit 1.2.0`이 `numpy<2.0.0` 요구 |
| ExecuTorch | 1.3.x | v1.0 GA(2025-10) 이후. 0단계에선 설치 안 함 |

> ⚠️ 경로 A(호스트 pip)에서는 **`libcudnn.so.9`(cuDNN)와 `libnvinfer.so.10`(TensorRT)를 못 찾아 ONNX Runtime이 조용히 CPU로 fallback**하는 함정이 있습니다(둘 다 CUDA Toolkit deb에는 없고, 각각 venv의 `nvidia-cudnn-cu12`/`tensorrt_libs` 패키지 디렉터리에만 있음). 해결법은 `01_environment_setup.md`의 **3-4-a절**에 있습니다 — 건너뛰지 마세요.

정확한 스택은 [`study_guide/01_environment_setup.md`](study_guide/01_environment_setup.md)를 정본으로 따르세요.

---

## 🗂️ 저장소 구조

```
.
├── study_guide/            # 학습 가이드 (MD + HTML)
│   ├── README.md/.html     # 인덱스
│   └── 01_*.md … 10_*.md   # 단계별 문서 (+ 각 .html)
├── logs/                   # 실제 머신에서 따라 해본 실행 로그·분석 (HTML)
│   ├── stage0_setup_log.html        # 0단계 환경 준비 실행 로그
│   ├── stage0.5_ladder_log.html     # 0.5단계 배포 사다리 Lv.1~4 실행 로그
│   ├── lv2_ptq_deep_dive.html       # PTQ 4종(FP32/dynamic/fp16/INT8) 이론·실측 딥다이브
│   ├── stage1_quantization_log.html # 1단계 1차 실습(큐레이션 1,000장) + 정정 10건 근거
│   ├── stage1_real_imagenet_log.html    # 1단계 재실행 — ImageNet val 50,000장 전량
│   └── stage1_real_imagenet_report.html # 재실행 분석 보고서 + 정정 12건 근거
├── learning_resources.html # 학습 사이트 34곳 + 인용 논문 19편 (링크 실측 검증)
├── experiments/            # 실습에서 파생된 진행 중 실험 (가이드 본문과 별개)
│   └── qat_recovery/       # QAT 회복 실험 2팔 — 미완, HANDOFF.md §5 참조
├── paper/                  # 논문 PDF 받는 스크립트 (PDF 자체는 .gitignore — 재배포 안 함)
├── HANDOFF.md              # 다른 PC에서 작업 이어받기 (상태·전송 목록·재실행 절차)
├── guide (1).html          # 원본 기획 문서 (출발점)
├── CLAUDE.md               # 제작에 쓰인 하네스 포인터
└── .claude/                # 에이전트 팀 + 스킬 (제작 하네스)
    ├── agents/             # guide-author, tech-reviewer
    ├── skills/             # research / writing / review / md-to-html / orchestrator
    └── memory/             # 실측 기록 사본 (sudo 암호 마스킹 처리)
```

---

## 🛠️ 어떻게 만들었나 — 하네스(에이전트 팀)

이 가이드는 [Claude Code](https://claude.com/claude-code) 위에서 **다중 에이전트 하네스**로 제작했습니다: `guide-author` 여러 명이 단계별 문서를 병렬 리서치·작성(Fan-out)하고, `tech-reviewer`가 버전 정합성·명령어·링크를 교차 검증(Fan-in)한 뒤, `md-to-html` 스킬(의존성 없는 자체 파이썬 렌더러 [`.claude/skills/md-to-html/scripts/render.py`](.claude/skills/md-to-html/scripts/render.py))로 HTML을 생성합니다. `.claude/`에 그 구성이 그대로 들어 있습니다.

MD가 정본이고 HTML은 파생물입니다. 내용을 고칠 땐 MD를 수정한 뒤 재렌더하세요:
```bash
python3 .claude/skills/md-to-html/scripts/render.py study_guide
```

---

## 📎 원본

이 저장소는 [`guide (1).html`](<guide (1).html>)의 단계 구조를 기반으로, 실행 가능한 명령어·코드·예시·참고문헌으로 확장한 것입니다.

## 📄 라이선스

[MIT License](LICENSE) — 문서·코드 모두 자유롭게 사용·복사·수정·배포·재라이선스할 수 있으며, 저작권 고지와 라이선스 문구만 포함하면 됩니다.
