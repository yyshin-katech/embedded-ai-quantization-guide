---
name: qat-recovery-experiment
description: "QAT 회복 실험 미완 — 2팔 설계(QAT vs FP32-finetune 대조군) 확정, 3회 GPU 이탈로 대조군은 한 번도 못 돌림. 회복률 분모가 노이즈 크기(0.06~0.14%p)라 읽을 값은 'QAT − 대조군' 격차 하나"
metadata: 
  node_type: memory
  type: project
---

`study_guide/03_quantization_theory.md` §2.5는 QAT/STE를 설명만 하고 실습은 합성 텐서로
gradient 통과만 본다. val 50,000장([[imagenet-val-50k-local]])이 생겨 주장 자체를 측정하려
만든 실험이 `~/stage1-work/qat_recovery.py` + `qat_control_finetune.py`다. **아직 미완.**

**설계 (2팔이 핵심):** ImageNet train split이 없어 **val을 쪼개** 학습한다 — 클래스당
40장 학습 / 10장 평가, `np.random.default_rng(0)`, 서로소 assert. val 분포로 학습하므로
**절대 top-1은 문헌값과 비교 불가**, 상대 관계만 유효.
- 팔 1 `qat_recovery.py`: FP32 → PTQ(fake-quant, 학습 없음) → QAT(STE 2 에폭)
- 팔 2 `qat_control_finetune.py`: **fake-quant만 제거**하고 분할·optimizer·LR·schedule·
  배치·에폭·전처리를 전부 동일하게 맞춘 FP32 파인튜닝

**왜 대조군이 필수인가:** 팔 1만 보면 QAT가 FP32를 이긴다(BS=96에서 69.13% vs 68.52%,
+0.61%p). 그런데 QAT 팔만 val 40,000장으로 2에폭 추가 학습을 받았으므로 그 이득이
'양자화 인식'인지 '그냥 더 학습한 것'인지 구분되지 않는다 — 50k 재실행에서 정정한
"INT8이 FP32보다 +0.40%p 낫다"(실제로는 p=0.48 노이즈)와 **구조가 같은 오류**다.
**읽을 값은 `QAT − 대조군` 격차 하나뿐이다.** 회복률은 읽지 말 것 — 분모(PTQ 손실)가
0.06~0.14%p로 측정 노이즈 크기라 1116.7% 같은 무의미한 수가 나온다(스크립트에 경고 내장).

**실측된 값 (전부 부분 실행, 평가 10,000장):**
| 설정 | FP32 | PTQ | QAT ep0 | QAT ep1 | 대조군 |
|---|---|---|---|---|---|
| BS=96 / calib 5,000 | 68.52% | 68.46% | 68.89% | 69.13% | ❌ 못 돌림 |
| BS=48 / calib 2,560 | 68.51% | 68.37% | 69.47% | ❌ GPU 사망 | ❌ 못 돌림 |

두 설정은 비교 불가다 — step 수가 416→833으로 2배고, 캘리브 장수도 5,000→2,560으로
달랐다(아래 버그). BS=48 FP32가 68.51%로 재현된 건 분할·전처리·eval 모드가 같다는 확인.

**고친 함정 3가지 (재실행 전 알아야 함):**
1. **`.eval()` 누락** — 대조군에서 빼먹으니 BatchNorm이 running stats 대신 배치 통계를 써
   학습 전 기준선이 68.52% → 68.04%로 어긋났다. 팔 1의 FP32 값과 교차 검증해 잡았다.
   지금은 `model.eval()` + ±0.05%p 가드가 들어가 있다.
2. **하드코딩 상수 → JSON 핸드오프** — 대조군이 `QAT_EP1 = 69.13`을 상수로 물고 있었다.
   배치를 바꾸면 BS=96 값과 BS=48 대조군을 조용히 비교해 오답을 낸다. 이제 팔 1이
   `qat_recovery_result_bs{N}.json`을 쓰고 팔 2가 읽으며, `(bs_train, epochs)`가 자기 설정과
   다르면 **실행을 거부**한다. 3차 실행에서 이 가드가 실제로 작동해 오비교를 막았다.
3. **캘리브 장수가 배치에 묶여 있었다** — `range(0, 20*BS_EVAL, BS_EVAL)`이라 BS_EVAL을
   250→128로 줄이자 관측 이미지가 5,000→2,560장으로 반토막 나 PTQ가 −0.09%p 움직였다
   (배치는 결과를 바꿔선 안 되는 노브다). `CALIB_N=5000`으로 분리 완료.

**막고 있는 것:** [[gpu-xid79-fallen-off-bus]]. 3회 다 이 워크로드에서 죽었다. 배치 반감은
무효한 레버로 판명됐고(전력 −8W뿐), 다음 레버는 `sudo nvidia-smi -pl`로 전력 상한을 내리는
것이다. **GPU 문제를 먼저 처리하지 않으면 이 실험은 계속 중단된다.**

**재실행 절차:** `nvidia-smi -L` 생존 확인 → 두 팔을 **같은 배치로** 순차 실행
(`qat_recovery.py` 먼저, JSON 생성 확인, 그다음 `qat_control_finetune.py`) → 2초 텔레메트리
병행. 팔당 5~7분.

**남은 설계 개선(미착수):** PTQ 손실이 큰 변형으로 바꾸면 회복률 분모가 의미를 갖는다 —
Entropy 정규화(−9.45%p), Percentile 99.9(−6.83%p), weight 4-bit 중 하나. 현재 구성
(per-channel 대칭 INT8 weight + per-tensor 비대칭 UINT8 act)은 PTQ 손실이 애초에 무손실급이라
"회복할 것이 없다"가 정직한 결론에 가깝다.

결과가 나오면 `03_quantization_theory.md` §2.5에 반영할 것. 1단계 전체는
[[stage1-quantization-hands-on]].
