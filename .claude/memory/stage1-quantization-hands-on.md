---
name: stage1-quantization-hands-on
description: "1단계(양자화 이론) 실습 — 큐레이션 1000장 1차(정정 10건) + 진짜 val 50,000장 재실행(정정 12건). 둘 다 커밋(c22ba4c). QAT 회복 실험까지 완료. 작업 디렉터리 ~/stage1-work"
metadata: 
  node_type: memory
  type: project
---

`study_guide/03_quantization_theory.md`(1단계)를 **두 번** 실행했다. venv는 [[stage0-env-installed]]의
`~/emb-ai` 재사용, 작업 디렉터리는 저장소 밖 `~/stage1-work/`(현 머신 [[machine-ai-lap-rtx3080]]에 있음).

**1차 (2026-08-02, 큐레이션 1,000장)** → `logs/stage1_quantization_log.html`, 정정 10건, 커밋 `32f090f`.
데이터가 클래스당 1장이라 FP32 top-1이 78.5%로 부풀려졌다.
**2차 (2026-08-04→06, 진짜 ImageNet val 50,000장)** → `logs/stage1_real_imagenet_log.html` +
`stage1_real_imagenet_report.html`. 데이터는 [[imagenet-val-50k-local]]. 정정 12건 + 죽은 링크 3건 +
`learning_resources.html`까지 반영해 커밋 `c22ba4c`. `10_pitfalls.md` 함정 0(평가셋 검정력)·함정
2-b(전처리)가 이때 신설.

**ORT 1.23.2 캘리브레이션 함정 2가지 (2단계에서도 그대로 걸림):**
- `CalibrationMethod.Entropy`를 `quantize_static`으로 부르면 **MinMax로 조용히 퇴화**(기본 num_bins에서
  탐색 후보가 1개 = 전체 범위). 50k 재실행에서 **산출 .onnx가 md5까지 동일**함을 확인.
- 그 `num_bins`는 `quantize_static`으로 **전달 자체가 불가**(화이트리스트 5키 밖). 우회하려면
  `create_calibrator` 몽키패치(`quantize_v2.py`의 `patch_hist_bins()`).

**TensorRT × ORT QDQ 비호환 — 2×2 절제로 원인 확정:** 하드 블로커는 **activation zero-point ≠ 0
하나뿐**. INT32 bias DQ는 2차 증상. 대칭 `QInt8`이면 정상(TRT 0.51ms = FP32 0.96ms의 1.86×), 비대칭은
FP32보다 3배 느림(무음 폴백). 판별: 같은 모델 CUDA EP p50과 비교해 TRT가 안 빠르면 폴백. 대칭 전환
대가 −0.29%p(p=9.2e-5), `QuantizeBias`는 정확도 영향 0.

**50k에서만 알 수 있었던 것 (문서 반영 완료):**
- 가이드 `preprocess()`(256×256 종횡비 무시)가 **−1.07%p**를 잃음(p=1.6e-14). torchvision 방식으로
  재면 FP32 69.81% = 공개값 69.758%와 일치. **양자화 손실(−0.12%p)보다 9배 큰 영향** — 전처리가 캘리브 선택보다 중요.
- 큐레이션 셋 부풀림 평균 +9.77%p인데 [+8.41, +10.39]로 상수 아님(보정 불가).
- 유의성 판정 **5/13건이 뒤집힘**. 1차의 "INT8 > FP32(+0.40%p)"는 p=0.48 노이즈였고, 정확히는
  "ResNet18 PTQ INT8은 FP32와 통계적으로 구별 안 됨"(50k에서 −0.12%p, p=0.061).
- weight SQNR은 **레이어 민감도 예측력 없음**(Spearman −0.036). per-channel weight-only INT8은 사실상 무손실 —
  실제 손실은 activation 양자화에서.
- MinMax가 CNN에서 최적(클리핑 강도 ↔ top-1 단조).

**QAT 회복 실험: 완료.** 미완이던 2팔 실험을 3080에서 완주(W4A8 손실변형, 97.1% 회복, QAT−대조군
−1.50%p) — [[qat-recovery-experiment]]. `03` §2.5.4에 반영. 2단계(DETR)는 [[stage2-detr-hands-on]].
커밋 전 스캔은 [[repo-is-public-scan-before-commit]], 인수인계는 레포 `HANDOFF.md`.
