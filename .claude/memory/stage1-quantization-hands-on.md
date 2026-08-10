---
name: stage1-quantization-hands-on
description: "1단계(양자화 이론) 실습 — 큐레이션 1000장 1차 실행(정정 10건) + 진짜 val 50,000장 재실행(정정 12건). 둘 다 커밋 완료(c22ba4c). 작업 디렉터리 ~/stage1-work"
metadata: 
  node_type: memory
  type: project
---

`study_guide/03_quantization_theory.md`(1단계)를 이 머신에서 **두 번** 실행했다. venv는
[[stage0-env-installed]]의 `~/emb-ai` 재사용, 작업 디렉터리는 저장소 밖 `~/stage1-work/`.

**1차 (2026-08-02, 큐레이션 1,000장)** → `logs/stage1_quantization_log.html`, 정정 10건 반영 후
커밋 `32f090f` + push 완료. 데이터가 `EliSchwartz/imagenet-sample-images`(클래스당 1장)라
FP32 top-1이 78.5%로 부풀려졌다.

**2차 (2026-08-04, 진짜 ImageNet val 50,000장)** → `logs/stage1_real_imagenet_log.html` +
`logs/stage1_real_imagenet_report.html`. 데이터 확보 경위는 [[imagenet-val-50k-local]].
**2026-08-06에 정정 12건 + 죽은 링크 3건 + `learning_resources.html`까지 반영해 커밋·푸시 완료
(`c22ba4c`).** 아래 "50k에서만 알 수 있었던 것"은 이제 전부 문서에 들어가 있다 —
`10_pitfalls.md` 함정 0(평가셋 검정력)·함정 2-b(전처리)가 이때 신설됐다.

**ORT 1.23.2 캘리브레이션 함정 2가지 (2단계에서도 그대로 걸림):**
- `CalibrationMethod.Entropy`를 `quantize_static`으로 부르면 **MinMax로 조용히 퇴화**한다.
  `EntropyCalibrater` 기본값 `num_bins=128, num_quantized_bins=128` → `get_entropy_threshold`의
  탐색 후보 배열이 `np.zeros(64-64+1)` = 1개(= 전체 범위 = MinMax). 50k 재실행에서
  **산출 .onnx 파일이 md5까지 동일**함을 확인(scale 32/32 동일보다 강한 증거).
- 그 `num_bins`는 `quantize_static`으로 **전달 자체가 불가능**하다. `quantize.py`의
  `calib_extra_options_keys` 화이트리스트가 5개뿐(`CalibTensorRangeSymmetric`/`CalibMovingAverage`/
  `CalibMovingAverageConstant`/`CalibMaxIntermediateOutputs`/`CalibPercentile`).
  우회하려면 `importlib.import_module("onnxruntime.quantization.quantize").create_calibrator`를
  몽키패치해야 한다(`~/stage1-work/quantize_v2.py`의 `patch_hist_bins()`). 주의:
  `import ... .quantize as QZ`는 패키지 `__init__`이 re-export한 **함수**를 잡아서 실패한다.

**TensorRT × ORT QDQ 비호환 — 2×2 절제로 원인 확정 (50k 재실행에서 정밀화):**
하드 블로커는 **activation zero-point ≠ 0 하나뿐**이다. INT32 bias DQ는 2차 증상 —
bias DQ를 0개로 없앤 비대칭 모델(B)은 여전히 폴백하고, bias DQ가 21개 남은 대칭 모델(C)은
정상 실행된다(TRT 0.51ms = FP32 0.96ms 대비 1.86×). 비대칭은 2.97~3.06ms로 **FP32보다 3배 느림**.
판별법: 같은 모델을 CUDA EP로도 돌려 p50 비교 — TRT가 더 빠르지 않으면 무음 폴백.
정확도 대가는 대칭 전환 −0.29%p(p=9.2e-5), `QuantizeBias`는 **정확도 영향 0**(예측 완전 동일).

**50k에서만 알 수 있었던 것 (문서 정정 12건, 보고서 8장에 표 — 반영·커밋 완료):**
- 가이드 `preprocess()`(256×256 종횡비 무시)가 **−1.07%p**를 잃는다(p=1.6e-14). torchvision 방식
  (짧은변 256 + center crop)으로 재면 FP32 69.81% = 공개값 69.758%와 0.05%p 일치.
  **양자화 손실(−0.12%p)보다 9배 큰 영향** — 전처리가 캘리브 방법 선택보다 중요하다.
- 큐레이션 셋 부풀림 평균 +9.77%p인데 범위가 [+8.41, +10.39]로 **상수가 아니라 보정 불가**.
- 유의성 판정 **5/13건이 "유의하지 않음 → 유의"로 뒤집힘** (검정력 차이). 1차의
  "INT8이 FP32보다 낫다(+0.40%p)"는 애초에 p=0.48 노이즈였고, 정확한 서술은
  "ResNet18 PTQ INT8은 FP32와 통계적으로 구별되지 않는다"(50k에서 −0.12%p, p=0.061).
- weight SQNR은 **레이어 민감도 예측력이 없다**(Spearman ρ=−0.036, SQNR 최저 3개와 실제 최악
  3개 교집합 0). per-channel weight-only INT8은 전 레이어 동시에도 −0.044%p로 사실상 무손실 —
  실제 PTQ 손실은 activation 양자화에서 나온다.
- 캘리브 커버리지 200클래스 → 1000클래스는 **유의하지 않음**(+0.03%p, p=0.639).
- MinMax가 CNN에서 최적이라는 1차 결론은 유지·강화됐다(클리핑 강도 ↔ top-1 단조).

**유지되는 1차 결론:** 클리핑 단조 관계, MinMax 최적(CNN 한정), Entropy 기본값 퇴화,
TRT 대칭 요구, opset 다운컨버트 실패가 exit 0으로 통과. 정정 대상은 '수치'와 '유의성 주장'뿐.

**미완:** QAT 회복 실험 — 2팔 설계(QAT vs FP32-finetune 대조군)로 확장했고 [[gpu-xid79-fallen-off-bus]]
때문에 3회 중단됐다. 대조군은 아직 한 번도 못 돌렸다. 상세·재실행 절차는 [[qat-recovery-experiment]].
다른 PC 인수인계는 저장소의 `HANDOFF.md`.
0.5단계까지의 내용은 [[study-guide-project]], 커밋 전 스캔은 [[repo-is-public-scan-before-commit]].
