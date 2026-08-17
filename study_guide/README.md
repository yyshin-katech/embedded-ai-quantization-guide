# AI 모델 양자화 → 임베디드 배포 실전 학습 가이드

멀티카메라 Transformer 인식 모델을 **INT8 양자화**하여 **멀티 SoC(NVIDIA Orin/Thor · TI Jacinto · Qualcomm · Renesas RZ/V)** 에 올려 구동하는 과정을, **읽고 그대로 따라 하면 실행되는** 단계별 실습 문서로 정리한 학습 가이드입니다.

> 원본 기획: `../guide (1).html` (Embedded AI Engineer 실전 학습 가이드 v2)
> 이 `study_guide/`는 그 단계 구조를 **Ubuntu 22.04 + NVIDIA RTX GPU** 환경에서 실행 가능한 명령어·코드·예시·참고문헌으로 확장한 것입니다.

> ✅ **실측 반영 현황** — `01`(0단계)·`02`(0.5단계)·`03`(1단계)은 실제 머신(Ubuntu 22.04.5 / RTX 3060 12GB / 드라이버 595.84)에서 **끝까지 따라 해보고 그 결과로 문서를 정정**했습니다. 예상 출력·실측 표·함정은 모두 그때 나온 실제 값입니다. 전 과정 로그: [`../logs/`](../logs/) (`stage0_setup_log.html`, `stage0.5_ladder_log.html`, `lv2_ptq_deep_dive.html`, `stage1_quantization_log.html`). `04`(2단계)는 **2026-08-16 RTX 3080에서 실측 완료** — DETR을 COCO val2017 전량(5,000장)으로 돌려 초안의 3가지 단정(export 블로커=SDPA·op선택 mixed 실패·분산 손상)을 정정했습니다(리포트 [`../logs/stage2_detr_quantization_report.html`](../logs/stage2_detr_quantization_report.html), 실패 로그 [`../experiments/stage2_detr/onnx_export_failures.md`](../experiments/stage2_detr/onnx_export_failures.md)). **§4.6 BEVFormer-tiny도 2026-08-17 실측**했습니다 — 무컴파일 레거시 venv(프리빌트 휠)로 `fundamentalvision/BEVFormer`를 돌려 grid_sample/MSDeformAttn **op 단정은 반전 0건**(초안이 맞음), 실전 함정 **2건**(mmcv 커스텀 op은 **CPU에서만** 유효 export·전체 export는 grid_sample이 아니라 **`point_sampling`에서 사망**), FP32 nuScenes-mini **mAP 0.2647**(81샘플 스모크·문헌비교 불가·상대 델타만), 전체 INT8은 **범위 밖**(유효 export 없음→포크 필요)을 확인했습니다(리포트 [`../logs/stage2_bevformer_quantization_report.html`](../logs/stage2_bevformer_quantization_report.html), 실패 로그 [`../experiments/stage2_bevformer/onnx_export_failures.md`](../experiments/stage2_bevformer/onnx_export_failures.md)). **§4.4 SmoothQuant도 2026-08-17 실측**했습니다 — nvidia-modelopt 0.45.0으로 DETR을 COCO val 5,000장 재측정해, per-tensor INT8 폭락(0.4209→0.3301)을 SmoothQuant(α=1.0)가 **gap의 59.9%(→0.3845) 회복**함을 확인했습니다(§4.5 op-선택 mixed의 약 15배 → "activation 입도가 레버" 확증). 프리셋 기본 α=1.0(논문 0.5 아님)·absmax 3.69×→1.96× 등 정정 포함, torch fake-quant 경로라 상대 관계만 유효(리포트 [`../logs/stage2_smoothquant_report.html`](../logs/stage2_smoothquant_report.html), 재현 [`../experiments/stage2_smoothquant/`](../experiments/stage2_smoothquant/)). 나머지 문서는 웹 검증 기반이며 아직 실행 검증 전입니다.
>
> 🔁 **2026-08-06 — `03`(1단계)을 ImageNet val 50,000장 전량으로 재실행**했습니다. 1차 실습은 클래스당 1장 큐레이션 셋(1,000장)이었고, 그 절대 top-1이 **평균 +9.77%p 부풀려져 있었습니다**. 전량 재측정으로 **Δ의 부호 3건·유의성 판정 5건이 뒤집혀** `03`·`05`·`10`을 정정했습니다(새 항목: [10단계 함정 0](10_pitfalls.md) — *평가셋이 작으면 나머지 함정을 진단할 수 없다*). 근거: [`stage1_real_imagenet_log.html`](../logs/stage1_real_imagenet_log.html) · [`stage1_real_imagenet_report.html`](../logs/stage1_real_imagenet_report.html). **이 가이드의 수치를 인용할 때는 큐레이션 셋 값이 아닌 50k 값을 쓰세요.**

---

## 🎯 이 가이드가 향하는 곳

```
작은 모델을 실제로 칩(또는 에뮬레이터)에 올려 "깨뜨려보기"
= 정확도(mAP) × 실시간성(latency) × 툴체인 호환성(op 지원)의 3중 제약 탐색
```

①정확도·②실시간성은 책으로 배우지만, **③"툴체인이 그 op를 먹는가"는 직접 깨져봐야** 배웁니다. 이 가이드는 그 "깨져보는 경험"을 낮은 난이도부터 순서대로 쌓도록 설계되어 있습니다.

---

## 🧭 사용법

1. **순서대로** `01` → `10`을 따라갑니다. 각 문서는 `0)왜 → 1)체크리스트 → 2)이론 → 3)환경 → 4)실습 → 5)결과해석 → 6)트러블슈팅 → 7)산출물 → 8)참고문헌 → 9)다음` 구조입니다.
2. **먼저 [01_environment_setup](01_environment_setup.md)** 로 환경(드라이버/CUDA/Docker/TensorRT)을 갖춥니다. 버전 스택의 정본(canonical)이 이 문서에 있습니다.
3. 보드가 없어도 **약 80%는 x86 PC에서** 진행됩니다. 각 문서가 "보드 없이 가능 / 보드 필요"를 구분해 표기합니다.
4. 각 단계의 **산출물(Deliverables)** 을 반드시 남기세요. 뒤 단계와 최종 캡스톤·포트폴리오의 입력이 됩니다.

---

## 📚 문서 인덱스

| # | 문서 | 원본 HTML 단계 | 무엇을 하는가 | 핵심 산출물 |
|---|------|---------------|--------------|------------|
| 01 | [환경 준비](01_environment_setup.md) | 0단계 | 드라이버·CUDA·Docker·nvidia-container-toolkit·TensorRT/ONNX RT/polygraphy 설치, nuScenes mini | 검증된 개발 환경 |
| 02 | [배포 난이도 사다리](02_deployment_ladder.md) | 0.5단계 | Edge Impulse → LiteRT → ONNX Runtime → ExecuTorch → Qualcomm AI Hub | 첫 온디바이스 배포 경험 |
| 03 | [양자화 이론](03_quantization_theory.md) | 1단계 | scale/zero-point 유도, QDQ, 캘리브레이션 4종, QAT/STE, ResNet18 PTQ | `layer_sensitivity.csv` |
| 04 | [Transformer 양자화 지옥](04_transformer_quantization.md) | 2단계 ★핵심 | LayerNorm/Softmax/GELU/attention 붕괴, SmoothQuant, DETR export 실패→우회 | `onnx_export_failures.md` |
| 05 | [TensorRT로 첫 완주](05_tensorrt.md) | 3단계 | trtexec, polygraphy, INT8 캘리브레이터, DLA 파티셔닝, 커스텀 플러그인 | Orin 성능 리포트 |
| 06 | [멀티 SoC 확장](06_multi_soc.md) | 4단계 | 같은 ONNX를 TIDL / QNN / DRP-AI에 배포, fallback 최소화 | 4-target 성능 매트릭스 |
| 07 | [인프라화](07_infrastructure.md) | 5단계 | `bench/` 벤치 하네스, CI 회귀 테스트, design_rules 자동화 | `design_rules.md`, 회귀 하네스 |
| 08 | [캡스톤 프로젝트](08_capstone.md) | 캡스톤 | nuScenes mini BEV detector를 4개 타깃에 배포·비교 | 공개 리포 + 블로그 5편 |
| 09 | [12주 로드맵](09_roadmap.md) | 로드맵 | 주차별 상세 계획 + 학습자 유형별 변형 | 학습 스케줄 |
| 10 | [함정 5개 (+ 측정의 함정)](10_pitfalls.md) | 함정 | 평가셋 검정력(함정 0)·캘리브 대표성·전처리 일치·fallback 지옥 등 + 재현 스니펫 | 실무 체크리스트 |

---

## 🔗 산출물 체인 — 각 단계가 어떻게 이어지는가

```
03 layer_sensitivity.csv ──┐
                           ├──▶ 05/08 mixed precision 근거 (어느 레이어를 FP16으로 남길지)
04 onnx_export_failures.md ┘        │
                                    ▼
06 4-target 결과 ──▶ 07 성능 매트릭스 + design_rules.md ──▶ 08 캡스톤(종합) ──▶ 포트폴리오
```

- **03 → 05/08**: 레이어 민감도가 mixed precision 튜닝의 근거.
- **04 → 08**: export 실패/우회 기록이 캡스톤의 ONNX 변환 단계와 design rules의 원형.
- **06 → 07 → 08**: 백엔드별 결과가 매트릭스로 모이고, 규칙으로 추상화되어 캡스톤에서 종합.

---

## 🗓️ 12주 학습 순서 (요약)

| 주차 | 문서 | 산출물 |
|------|------|--------|
| 워밍업(선택) | [02 사다리](02_deployment_ladder.md) Lv.1~4 | 첫 배포 경험 |
| 1–2주 | [03 이론](03_quantization_theory.md) | `layer_sensitivity.csv` |
| 3–5주 | [04 Transformer](04_transformer_quantization.md) | `onnx_export_failures.md` |
| 6–8주 | [05 TensorRT](05_tensorrt.md) | Orin 성능 리포트 |
| 9–11주 | [06 멀티 SoC](06_multi_soc.md) | 4-target 매트릭스 |
| 12주 | [07 인프라화](07_infrastructure.md) | `design_rules.md` |

> 상세 주차별 체크리스트·완료 판정·압축코스는 [09_roadmap.md](09_roadmap.md) 참고.

---

## ⚙️ 환경 전제

- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA RTX (dGPU)
- **필수**: Docker + NVIDIA Container Toolkit
- **선택(실측용)**: Jetson Orin, TI/Qualcomm/Renesas 개발 보드 — 없으면 host emulation / Qualcomm AI Hub로 상당 부분 대체

> ⚠️ **버전 정합성 주의**: 2026-07 기준 CUDA 12/13 라인과 TensorRT 10.x/11.x가 갈라져 있습니다. 이 스터디는 **호환성이 넓은 라인으로 고정**하는 것을 권장하며, 정확한 버전 스택은 [01_environment_setup.md](01_environment_setup.md)를 정본으로 따르세요.

---

## 📎 비고

- 모든 문서의 버전·링크는 **2026-07 기준**으로 웹 검증했으며, 확정 못한 항목은 각 문서에 `⚠️ 확인 필요`로 표기했습니다. 실제 설치 시점에 각 공식 페이지에서 재확인하세요.
- 이 문서 세트는 `guide (1).html`의 단계 구조를 기반으로 자동 생성 파이프라인(하네스: `guide-author` × N → `tech-reviewer`)으로 작성되었습니다.
