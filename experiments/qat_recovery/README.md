# QAT 회복 실험 (진행 중)

**가이드 본문이 아닙니다.** [1단계](../../study_guide/03_quantization_theory.md) §2.5에서 파생된 실습 작업물이고, 아직 결론이 없습니다.
저장소에 둔 이유는 다른 PC에서 이어받을 수 있게 하려는 것입니다 — 전체 맥락은
[`HANDOFF.md` §5](../../HANDOFF.md).

## 무엇을 시험하는가

"QAT는 PTQ가 잃은 정확도를 되찾는다"는 주장을 실제로 측정합니다. 가이드 §2.5는 STE를
설명하지만 실습은 합성 텐서로 gradient가 통과하는지만 봅니다.

| 파일 | 팔 | 내용 |
|---|---|---|
| `qat_recovery.py` | 1 | FP32 → PTQ(fake-quant, 학습 없음) → QAT(STE 2에폭) |
| `qat_control_finetune.py` | 2 | **fake-quant만 제거**하고 나머지를 전부 동일하게 맞춘 FP32 파인튜닝 |

## 대조군이 왜 필수인가

팔 1만 보면 QAT가 FP32를 이깁니다(BS=96에서 69.13% vs 68.52%). 하지만 **QAT 팔만 val
40,000장으로 2에폭 추가 학습을 받았습니다.** 그 이득이 '양자화 인식' 때문인지 '그냥 더 학습한
것' 때문인지 구분되지 않습니다 — 1단계 50k 재실행에서 정정한 "INT8이 FP32보다 +0.40%p
낫다"(실제로는 p=0.48 노이즈)와 구조가 같은 오류입니다.

> 🔴 **읽을 값은 `QAT − 대조군` 격차 하나입니다.** 회복률은 읽지 마세요 — 분모(PTQ 손실)가
> 0.06~0.14%p로 측정 노이즈 크기라 `1116.7%` 같은 무의미한 수가 나옵니다.

## 실행

전제: `~/stage1-work/data/cache/{squash,labels}.npy`(전처리 캐시)와 GPU가 필요합니다.
데이터 준비는 [`HANDOFF.md` §4](../../HANDOFF.md).

```bash
cp *.py ~/stage1-work/ && cd ~/stage1-work
nvidia-smi -L                                                       # GPU 생존 확인 먼저

python3 qat_recovery.py         2>&1 | tee qat_recovery_bs48.log    # 팔 1 (~6분)
python3 qat_control_finetune.py 2>&1 | tee qat_control_bs48.log     # 팔 2 (~6분)
```

배치·에폭·캘리브 장수는 환경변수로 덮어씁니다: `QAT_BS_TRAIN` / `QAT_BS_EVAL` /
`QAT_EPOCHS` / `QAT_CALIB_N`.

**두 팔을 반드시 같은 설정으로** 돌리세요. 팔 1이 `qat_recovery_result_bs{N}.json`을 쓰고 팔 2가
읽으며, `(bs_train, epochs)`가 자기 설정과 다르면 실행을 거부합니다.

## 알려진 한계

ImageNet train split이 없어 **val을 쪼개서** 학습합니다(클래스당 40장 학습 / 10장 평가,
`rng(0)`, 서로소 assert). 평가셋 누수는 없지만 val 분포로 학습한 모델이라 **절대 top-1을
문헌값과 비교하면 안 됩니다.** 상대 관계만 유효합니다.

현재 양자화 구성(per-channel 대칭 INT8 weight + per-tensor 비대칭 UINT8 activation)은 PTQ
손실이 무손실급이라 **"회복할 것이 없다"가 정직한 결론에 가깝습니다.** 회복률을 의미 있게
재려면 PTQ 손실이 큰 변형이 필요합니다 — Entropy 정규화(−9.45%p), Percentile 99.9(−6.83%p),
weight 4-bit 중 하나.
