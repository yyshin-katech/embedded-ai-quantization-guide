---
name: stage2-smoothquant-hands-on
description: 2단계 §4.4 SmoothQuant 완료(DETR): per-tensor INT8 폭락의 59.9%를 SmoothQuant가 회복, 프리셋 기본 α=1.0(논문 0.5 아님)
metadata:
  node_type: memory
  type: project
---

2단계 §4.4(SmoothQuant)를 `facebook/detr-resnet-50` 실모델에서 완주(2026-08-17, RTX 3080). DETR 폭락 리포트(§4.5, [[stage2-detr-hands-on]])가 지목한 "진짜 레버 = activation 양자화 입도"를 SmoothQuant로 직접 시험해 확증.

**경로 선택(Design X = torch fake-quant 자기일관 3원)**: nvidia-modelopt 0.45.0(`~/emb-ai`)로 FP32 / INT8_DEFAULT(per-tensor act, `max`) / INT8_SMOOTHQUANT을 전부 in-framework fake-quant로 측정. SmoothQuant→ONNX→ORT(Design Y)는 §4.5의 SDPA/dynamo export 벽·이중 양자화를 다시 만나므로 회피. 두 프리셋은 **algorithm만 다름**(weight per-ch axis0·act per-tensor 동일) → SmoothQuant 효과만 격리하는 깨끗한 A/B.

**핵심 실측(COCO val2017 전량 5,000장, `sq_02`)**:
- FP32 mAP **0.4209**(커밋 0.4207 재현✓) → INT8_DEFAULT **0.3301**(−0.0908, 대조군) → INT8_SMOOTHQUANT(α=1.0) **0.3845**(−0.0364)
- 회복 = +0.0544 = **폭락 gap의 59.9%**. mAP_s도 0.1255→0.1704(+0.0449). §4.5 op-선택 mixed(+0.0036)의 **약 15배** → "op 선택이 아니라 activation 입도가 레버"(§4.5 판정 4) 확증.

**정정한 초안 오류**:
- α 기본값: 초안 "modelopt 기본 0.5"는 틀림 → 프리셋 기본 **α=1.0**(문자열 프리셋→빈 kwargs→`smoothquant(alpha=1.0)`, `model_calib.py:954`). 논문 권장 0.5는 dict-override로.
- dict-override 유효: `cfg["algorithm"]={"method":"smoothquant","alpha":x}` 동작함(`model_quant.py:511`, `mode.py:377-380`). α 스윕(`sq_04`, eval 500장 별도 서브셋): α=0.5→gap 49.8% 회복, **α=1.0→66.6%**(override_worked=true). LayerNorm-heavy 검출은 full migration(α=1.0)이 유리.
- absmax 삽화값(31.7×→2.96×) → 실측 k_proj **3.69×→1.96×**(`sq_03`).

**캐비앗(불변)**: 절대 mAP는 **torch fake-quant(modelopt)** 경로 → 커밋된 §4.5 **ORT QDQ 절대값**(0.4207/0.2402)과 커널이 달라 1:1 비교 불가(그래서 여기 INT8_DEFAULT drop −0.0908이 §4.5 ORT drop −0.1805보다 얕음). 유효 결론은 같은 경로 안의 **상대 관계**(회복분·α 방향·absmax 감쇠).

**산출물**: `logs/stage2_smoothquant_report.html`(§1~7·SVG), `experiments/stage2_smoothquant/`(sq_common + sq_01~04 스크립트·JSON·README). 문서: `study_guide/04_transformer_quantization.md` §4.4에 실측 반영 + §4.5/5.2/6/7 상호참조 갱신.

**검증**: 자체 팬인(수치 SSOT 1:1·SVG 비례·섹션 무결성) + `tech-reviewer` 독립 팬인 **통과**(🔴1·🟡1 직접 수정·🟢3). 🔴=smooth_check 주석 "30배→3배면 성공" 통념이 실측(3.69→1.96)과 충돌해 오독 유발→모델 의존 명시로 교체, 🟡=§2.2.3 삽화값 LLM/DETR 스코프 병기; 🟢① modelopt 소스 라인 3건(`model_calib.py:954`·`model_quant.py:511`·`mode.py:377-380`)은 설치본 0.45.0 대조로 **라인 단위 실검증 완료**, 🟢③ 리포트 SVG 라벨 클리핑 viewBox 720→800 해소.

이로써 2단계 실기 3종(DETR §4.5·BEVFormer §4.6·SmoothQuant §4.4) 완료. 머신은 [[machine-ai-lap-rtx3080]], BEVFormer는 [[stage2-bevformer-hands-on]].
