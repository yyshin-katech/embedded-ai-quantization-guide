# QAT 회복 실험 (완결 — W4A8)

**가이드 본문이 아닙니다.** [1단계](../../study_guide/03_quantization_theory.md) §2.5에서 파생된 실습 작업물입니다.
저장소에 둔 이유는 다른 PC에서 이어받을 수 있게 하려는 것입니다 — 전체 맥락은
[`HANDOFF.md` §5](../../HANDOFF.md).

> **상태(2026-08-15):** 이전 머신 GPU 고장으로 미완이던 2팔 실험을 정상 RTX 3080에서 완주했습니다.
> 기본 W8A8은 무손실급이라 회복률이 노이즈에 묻혀 **weight를 4-bit로 낮춘 손실 변형(W4A8)**으로
> 처음으로 회복률을 읽었습니다: **PTQ −24.16%p → QAT 97.1% 회복**, 그러나 동일 학습을 받은
> FP32 대조군보다는 **−1.50%p** 낮습니다. 렌더된 요약은
> [`logs/stage1_50k_rerun_reproduction_report.html` §11](../../logs/stage1_50k_rerun_reproduction_report.html).

## 무엇을 시험하는가

"QAT는 PTQ가 잃은 정확도를 되찾는다"는 주장을 실제로 측정합니다. 가이드 §2.5는 STE를
설명하지만 실습은 합성 텐서로 gradient가 통과하는지만 봅니다.

| 파일 | 팔 | 내용 |
|---|---|---|
| `qat_recovery.py` | 1 | FP32 → PTQ(fake-quant, 학습 없음) → QAT(STE 2에폭) |
| `qat_control_finetune.py` | 2 | **fake-quant만 제거**하고 나머지를 전부 동일하게 맞춘 FP32 파인튜닝 |

## 대조군이 왜 필수인가

팔 1만 보면 QAT가 FP32를 이깁니다(무손실 W8A8·BS=96에서 69.13% vs 68.52%). 하지만 **QAT 팔만
val 40,000장으로 2에폭 추가 학습을 받았습니다.** 그 이득이 '양자화 인식' 때문인지 '그냥 더 학습한
것' 때문인지 구분되지 않습니다 — 1단계 50k 재실행에서 정정한 "INT8이 FP32보다 +0.40%p
낫다"(실제로는 p=0.48 노이즈)와 구조가 같은 오류입니다. 팔 2가 그 '추가 학습' 몫을 떼어냅니다.

> 🔴 **설정에 따라 읽는 값이 다릅니다.**
> - **기본 W8A8**(무손실급): PTQ 손실이 0.06~0.14%p로 측정 노이즈 크기라 회복률은 `1116.7%`
>   같은 무의미한 수가 됩니다 — **읽을 값은 `QAT − 대조군` 격차 하나**입니다.
> - **손실 변형 W4A8**(`QAT_WBITS=4`): PTQ 손실이 −24.16%p로 커져 **회복률이 노이즈 위에서
>   처음 측정**됩니다. 그래도 마지막 판정은 여전히 `QAT − 대조군`입니다.

## 결과 (W4A8 · 2026-08-15 · RTX 3080)

설정: `QAT_WBITS=4` (per-channel 대칭 4-bit weight + per-tensor 비대칭 UINT8 activation),
`BS_TRAIN=48 · BS_EVAL=128 · EPOCHS=2 · CALIB_N=5000`. val 40,000 학습 / 10,000 평가(서로소).
기록물: `qat_recovery_bs48_w4.log` · `qat_control_bs48_w4.log` · `qat_recovery_result_bs48_w4.json`.

**팔 1 — 회복 계단**

| 단계 | top-1 | vs FP32 | vs PTQ |
|---|---|---|---|
| FP32 (학습 없음) | 68.51% | — | — |
| PTQ · weight 4-bit | 44.35% | **−24.16%p** | — |
| QAT ep0 (STE) | 67.59% | −0.92%p | +23.24%p |
| QAT ep1 (STE) | **67.81%** | −0.70%p | **+23.46%p** |

→ PTQ 손실 24.16%p 중 QAT가 **97.1%**(23.46%p)를 되찾았습니다.

**팔 2 — 대조군 분해 (fake-quant만 제거)**

| | top-1 | Δ |
|---|---|---|
| FP32 학습 전 | 68.51% | — (팔 1과 일치 ✅) |
| FP32 + 파인튜닝 (대조군) | 69.31% | **+0.80%p** · 순수 추가학습 |
| QAT 4-bit | 67.81% | −0.70%p |
| **QAT − 대조군** | — | **−1.50%p** · 4-bit 잔여 대가 |

**판정 4가지**
1. **손실 변형 성공** — 노브 하나(weight 8→4bit)로 회복률이 처음 노이즈(±0.05%p) 위에서 측정됐습니다.
2. **§2.5 주장 실증** — "QAT는 PTQ가 잃은 정확도를 되찾는다"가 손실 큰 구간에서 **97.1%**로 참입니다.
3. **진짜 양자화 인식 학습** — 추가 학습만으로는 +0.80%p뿐인데, QAT는 손상 지점(44%)에서 +23.46%p
   올렸습니다. 회복은 일반 파인튜닝이 아니라 양자화 인식 적응의 기여입니다.
4. **정직한 상한** — 동일 학습을 받은 FP32 대조군보다 QAT는 여전히 **−1.50%p** 뒤집니다.
   4-bit weight(+8-bit act)의 환원 불가 용량 비용이며, "QAT = 공짜 회복"이 아닙니다.

## 실행

전제: `~/stage1-work/data/cache/{squash,labels}.npy`(전처리 캐시)와 GPU가 필요합니다.
데이터 준비는 [`HANDOFF.md` §4](../../HANDOFF.md).

```bash
cp *.py ~/stage1-work/ && cd ~/stage1-work
nvidia-smi -L                                                          # GPU 생존 확인 먼저

# 손실 변형 W4A8 — 회복률이 읽히는 설정 (권장)
QAT_WBITS=4 python3 qat_recovery.py         2>&1 | tee qat_recovery_bs48_w4.log   # 팔 1 (~2분)
QAT_WBITS=4 python3 qat_control_finetune.py 2>&1 | tee qat_control_bs48_w4.log    # 팔 2 (~2분)

# 기본 W8A8 — 무손실급, 회복률은 노이즈이니 QAT−대조군 격차만 읽음
# python3 qat_recovery.py         2>&1 | tee qat_recovery_bs48.log
# python3 qat_control_finetune.py 2>&1 | tee qat_control_bs48.log
```

배치·에폭·캘리브 장수·weight 비트폭은 환경변수로 덮어씁니다: `QAT_BS_TRAIN` / `QAT_BS_EVAL` /
`QAT_EPOCHS` / `QAT_CALIB_N` / `QAT_WBITS`.

**두 팔을 반드시 같은 설정으로** 돌리세요. 팔 1이 `qat_recovery_result_bs{N}_w{W}.json`을 쓰고 팔 2가
읽으며, `(bs_train, epochs, wbits)`가 자기 설정과 다르면 실행을 거부합니다.

## 알려진 한계

ImageNet train split이 없어 **val을 쪼개서** 학습합니다(클래스당 40장 학습 / 10장 평가,
`rng(0)`, 서로소 assert). 평가셋 누수는 없지만 val 분포로 학습한 모델이라 **절대 top-1을
문헌값과 비교하면 안 됩니다.** FP32 68.51%도 그래서 공식 69.758%가 아니며, 이 실험의 모든 수는
**상대 관계**(회복률·대조군 격차)로만 유효합니다.

기본 양자화 구성(per-channel 대칭 INT8 weight + per-tensor 비대칭 UINT8 activation)은 PTQ
손실이 무손실급이라 **"회복할 것이 없다"**가 정직한 결론입니다. 회복률을 의미 있게 재려면 PTQ
손실이 큰 변형이 필요합니다 — 이 실험은 그중 **weight 4-bit**(`QAT_WBITS=4`)를 골라 완주했습니다.
남은 후보는 Entropy 정규화(−9.45%p), Percentile 99.9(−6.83%p)이며, 같은 2팔 틀로 재사용할 수
있습니다.
