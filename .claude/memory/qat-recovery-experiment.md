---
name: qat-recovery-experiment
description: "QAT 회복 실험 완료(W4A8 손실변형) — FP32 68.51%→PTQ 4-bit 44.35%(−24.16%p)→QAT 67.81%=97.1% 회복. 대조군(FP32 파인튜닝) +0.80%p, QAT−대조군 −1.50%p가 4-bit 환원불가 대가. §2.5.4·보고서 §11 반영"
metadata: 
  node_type: memory
  type: project
---

`study_guide/03_quantization_theory.md` §2.5가 QAT/STE를 합성 텐서로만 보여주던 것을
**실모델(ResNet18)로 측정**한 실험. 스크립트는 `experiments/qat_recovery/`(레포 안) + `~/stage1-work/`.
**완료**(2026-08-16, AI-LAP/RTX3080 [[machine-ai-lap-rtx3080]] — 구 머신 GPU 고장으로 미완이던 것을 완주).

**왜 손실변형(W4A8)이었나:** 정본 W8A8은 PTQ 손실이 0.06~0.14%p(측정 노이즈급)라 "회복할 것이
없어" 회복률이 1116.7% 같은 무의미한 수가 됐다. 그래서 `QAT_WBITS=4` **한 노브만 오버라이드**해
weight 4-bit로 −24.16%p를 만들고 회복률을 처음으로 의미있게 측정했다.

**2팔 설계(대조군이 핵심):** ImageNet train이 없어 val 50,000장을 쪼개 학습(클래스당 40장 학습/
10장 평가, `rng(0)`, 서로소 assert).
- 팔1 `qat_recovery.py`: FP32 → PTQ(fake-quant, 학습 없음) → QAT(STE).
- 팔2 `qat_control_finetune.py`: **fake-quant만 제거**하고 분할·optimizer·LR·schedule·배치·에폭·
  전처리를 전부 동일하게 맞춘 FP32 파인튜닝. `(bs,ep,wbits)` 일치검증 내장(안 맞으면 실행 거부).

**확정 결과 (회복 계단):**
| 단계 | top-1 | Δ |
|---|---|---|
| FP32 | 68.51% | — |
| PTQ (weight 4-bit) | 44.35% | −24.16%p |
| QAT | 67.81% | **97.1% 회복** |

**대조군 분해:** 동일 조건 FP32 파인튜닝 = **+0.80%p**(순수 추가학습 몫) / **QAT − 대조군 = −1.50%p**
(4-bit 환원불가 대가). → 판정: **"QAT 회복은 진짜지만 공짜는 아니다"** — "QAT=공짜 회복" 통념 반증.

**캐비앗:** val split으로 학습했으므로 **절대 top-1은 문헌 비교 불가**, 상대 관계(회복률·대조군 격차)만 유효.

**반영:** `03_quantization_theory.md` **§2.5.4 신설**(기존 LSQ는 §2.5.5로 승번), 보고서 §11
(`logs/stage1_50k_rerun_reproduction_report.html#s11`). GPU는 3080이라 SW Power Cap 없이 완주
([[gpu-xid79-fallen-off-bus]]는 구 머신 이력). 1단계 전체는 [[stage1-quantization-hands-on]].
