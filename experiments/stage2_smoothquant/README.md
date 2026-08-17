# experiments/stage2_smoothquant — §4.4 SmoothQuant 실측 검증

2단계 §4.4(SmoothQuant)를 **facebook/detr-resnet-50** 실모델에서 완주한 재현 스크립트·로그·JSON. DETR 폭락 리포트(§4.5, [`stage2_detr`](../stage2_detr/))가 지목한 "진짜 레버 = activation 양자화 입도"를 SmoothQuant로 직접 시험해, 초안 §4.4의 **"⚠️ API 확인 필요"**와 **"다음 검증 과제로 남긴다"**를 실측으로 종결한다.

- **머신/환경**: RTX 3080 · Ubuntu 22.04 · venv `~/emb-ai` (torch 2.11.0+cu128, transformers 5.15.0) + **nvidia-modelopt 0.45.0**
- **데이터**: COCO val2017 5,000장(`_workspace/coco/`), 캘리브 100장(균등간격)
- **경로**: **Design X = torch fake-quant 자기일관 3원**(FP32 / per-tensor INT8 `max` / 동일 입도 + SmoothQuant). SmoothQuant→ONNX→ORT(Design Y)는 §4.5의 SDPA/dynamo export 벽·이중 양자화를 다시 만나므로 회피. 목적이 "op-선택 대비 SmoothQuant의 **상대** 회복"이라 in-framework가 깨끗.
- **리포트**: [`logs/stage2_smoothquant_report.html`](../../logs/stage2_smoothquant_report.html)

## 핵심 결과 (실측)

| 구성 (COCO val 5,000장) | mAP | vs FP32 | 회복 |
|---|---|---|---|
| FP32 | 0.4209 | — (커밋 0.4207 ✓) | — |
| INT8_DEFAULT (per-tensor act, `max`) | 0.3301 | −0.0908 | 0% (대조군) |
| INT8_SMOOTHQUANT (α=1.0 프리셋) | 0.3845 | −0.0364 | **+0.0544 (gap의 59.9%)** |

> op-선택 mixed(§4.5, attn 36개 FP)가 **+0.36 mAP**뿐이던 폭락을, SmoothQuant는 **gap의 59.9%**를 되찾는다 → "진짜 레버 = activation 입도"(§4.5 판정 4) 확증.

## 정정한 초안 오류 / 종결한 미결

1. **α 기본값 오귀속** — 초안 "modelopt 기본 α=0.5"는 실제 **프리셋 기본 α=1.0**(문자열 프리셋→빈 kwargs→`smoothquant(alpha=1.0)`, `model_calib.py:954`). 논문 권장 0.5는 dict-override로 지정. (sq_01/sq_04)
2. **dict-override 문법** — `config["algorithm"]={"method":"smoothquant","alpha":a}`는 **유효**(`mtq.quantize`가 str/dict 모두 수용: `model_quant.py:511`, `mode.py:377-380`). sq_04가 α=0.5/1.0에서 서로 다른 mAP로 실증(override_worked=true).
3. **삽화 absmax 수치** — 초안 예시 31.7×→2.96×는 삽화값. 실측 `k_proj` = **3.69×→1.96×**. (sq_03)
4. **"다음 검증 과제"** — §4.5 판정 4의 미결 SmoothQuant를 **완료**로 전환.

## API 실측 (sq_01)

- `INT8_SMOOTHQUANT_CFG` = `{quant_cfg, algorithm:"smoothquant"}`, `INT8_DEFAULT_CFG` = algorithm `"max"`. **두 프리셋은 algorithm만 다르다**(weight per-ch axis 0 · act per-tensor axis null 동일) → SmoothQuant 효과를 단독 격리하는 깨끗한 A/B.
- `mtq.quantize(model, config: dict, forward_loop=None) -> Module`

## 스크립트

| 파일 | 역할 | 산출 |
|---|---|---|
| `sq_common.py` | 공통 모듈(모델·전처리·후처리·COCOeval·캘리브·forward_loop) | — |
| `sq_01_modelopt_api.py` | modelopt 0.45.0 API 기록(프리셋 구조·시그니처) | `sq_01_api.json` |
| `sq_02_triplet_map.py` | 자기일관 3원 mAP (FP32/INT8_DEFAULT/INT8_SMOOTHQUANT) | `sq_02_triplet_map.json` |
| `sq_03_absmax_smooth.py` | 스무딩 전/후 채널 absmax(분리 셋) | `sq_03_absmax.json` |
| `sq_04_alpha_sweep.py` | α∈{0.5,1.0} 스윕(동일 서브셋) + dict-override 실증 | `sq_04_alpha_sweep.json` |

```bash
# 재현 (venv ~/emb-ai, modelopt 0.45.0 필요)
cd <repo>
PYTHONPATH=experiments/stage2_smoothquant ~/emb-ai/bin/python experiments/stage2_smoothquant/sq_01_modelopt_api.py
PYTHONPATH=experiments/stage2_smoothquant ~/emb-ai/bin/python experiments/stage2_smoothquant/sq_02_triplet_map.py --calib=100
PYTHONPATH=experiments/stage2_smoothquant ~/emb-ai/bin/python experiments/stage2_smoothquant/sq_03_absmax_smooth.py
PYTHONPATH=experiments/stage2_smoothquant ~/emb-ai/bin/python experiments/stage2_smoothquant/sq_04_alpha_sweep.py --limit=500
```

## 캐비앗

- 절대 mAP는 **torch fake-quant(modelopt)** 기준 — 커밋된 §4.5 **ORT QDQ 절대값**(0.4207/0.2402)과 양자화 커널이 달라 1:1 비교 불가. 유효 결론은 **같은 경로 안의 상대 관계**(회복분·α 방향·absmax 감쇠).
- 회복%는 이 경로의 per-tensor `max` 대조군 대비. α 최적값은 모델·캘리브 의존.
- INT8_DEFAULT drop(−0.0908)이 §4.5 ORT drop(−0.1805)보다 얕은 것도 **경로 차이**(modelopt fake-quant vs ORT QDQ) — 절대값이 아니라 회복 방향이 논점.
