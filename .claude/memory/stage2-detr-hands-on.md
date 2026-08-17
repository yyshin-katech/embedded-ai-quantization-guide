---
name: stage2-detr-hands-on
description: "2단계 DETR Transformer INT8 완료(2026-08-16, 커밋 41dc49e 미푸시) — COCO val2017 5,000장. 초안 단정 3건 반전(export 블로커=SDPA·op선택 mixed 실패·손상 분산). 다음=SmoothQuant §4.4·BEVFormer §4.6"
metadata: 
  node_type: memory
  type: project
---

`study_guide/04_transformer_quantization.md`(712줄 초안, 미검증)를 AI-LAP/RTX3080
([[machine-ai-lap-rtx3080]])에서 `facebook/detr-resnet-50`으로 완주(export→INT8 PTQ→COCO val2017
**전량 5,000장** mAP). **커밋 41dc49e**(main, 미푸시). 산출물·재현물은 레포 안.

**초안의 경험적 단정 3건 반전:**
1. **export 첫 블로커 = SDPA**(`aten::scaled_dot_product_attention`, opset14+)지 통념의
   `grid_sampler`가 아니다 — **DETR엔 grid_sample이 아예 없다**(Grid op=0). 4경로 대조로 확정.
2. **op-선택 mixed precision 실패** — attention score matmul 36개를 FP로 빼도 mAP 0.2402→**0.2438
   (+0.36뿐)**. 초안 §4.5의 "문제 op만 제외하면 회복" 반증. 3/4 초안 제외 패턴은 no-op(GELU 없음·
   Softmax/LayerNorm은 ORT QDQ가 애초에 양자화 안 함).
3. **손상은 분산돼 있다** — transformer만 INT8=0.2391, backbone만 INT8=0.2653(sub-additive) →
   범인은 특정 op가 아니라 **activation 양자화 입도**. 진짜 레버는 **SmoothQuant·per-token**.

**폭락 실측(CUDA EP, MinMax calib 100장, per-channel QDQ QInt8):** FP32 **0.4207**(공개값 42.0 일치)
→ INT8 **0.2402(−42.9%)**, 작은 객체 mAP_s **−77%**. Percentile 캘리브는 동적shape·OOM(31GB)·`inf`
**3중 실패** → MinMax만 생존.

**부수 실측:** torch 2.11 export 기본 `dynamo=True`(요청 opset 무시·18 강제·external data 분리,
[[stage0-env-installed]]), timm 필수(없으면 TimmBackbone ImportError).

**산출물:** `logs/stage2_detr_quantization_report.html`(§1~8·SVG), `experiments/stage2_detr/`
(재현 s2_01~s2_10 + `onnx_export_failures.md` 포트폴리오 산출물). 검증: 자체 + tech-reviewer
독립 팬인 7/7 PASS(🔴 0).

**캐비앗:** 절대 mAP는 CUDA EP·MinMax 기준이라 **상대 관계만 유효**.
**다음 과제:** SmoothQuant(§4.4, nvidia-modelopt). BEVFormer-tiny(§4.6)는 **완료**([[stage2-bevformer-hands-on]]) —
op 단정 반전 0·전체 INT8은 포크 필요로 범위 밖. 그 뒤 3단계(TensorRT)~7단계+캡스톤은 아직 웹 검증만 된 상태.
커밋 규약은 [[repo-is-public-scan-before-commit]].
