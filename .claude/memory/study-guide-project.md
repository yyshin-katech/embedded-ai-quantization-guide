---
name: study-guide-project
description: "0.5단계(배포 난이도 사다리) 실습 완료(구 머신) — executorch/torch/torchvision ABI 핀 충돌과 해법, LiteRT CompiledModel API 부재, Lv.2 PTQ 4종 실측. 산출물 venv/디렉터리는 AI-LAP엔 아직 없음"
metadata: 
  node_type: memory
  type: project
---

`study_guide/02_deployment_ladder.md`(0.5단계, Lv.1~4 + Qualcomm AI Hub)를 2026-07-31 구 머신에서
실행하고 `logs/stage0.5_ladder_log.html`에 정리했다. **관련 venv(`~/ladder-litert`·`~/ladder-et`)와
`~/ladder-work`는 구 머신 것이라 현 머신 AI-LAP엔 아직 없다**([[machine-ai-lap-rtx3080]]) — 0.5단계
재실행이 필요하면 재생성. 아래 findings는 머신 무관하게 유효.

**핵심 발견 — executorch/torch/torchvision 3자 버전 핀 충돌:** `pip install executorch`만 하면
`torch==2.13.0`이 깔리는데(제약이 `torch>=2.12.0a0` 하한뿐), executorch pybindings 확장이 더 좁은 ABI로
빌드돼 있어 `ImportError: undefined symbol: _ZN3c104impl3cow23materialize_cow_storage...`로 깨진다.
`torch==2.12.1`로 내리면 이번엔 `torchvision==0.28.0`이 `RuntimeError: operator torchvision::nms does
not exist`로 깨짐 → **`torch==2.12.1` + `torchvision==0.27.1 --no-deps` + `executorch==1.3.1`**
(+`qai-hub==0.53.0`)로 셋 다 동작.
**Why:** pip 제약이 느슨해도(`>=`) 컴파일된 C++ 확장은 정확한 ABI 매칭을 요구 — [[stage0-env-installed]]의
TensorRT `libnvinfer.so.10` 문제와 같은 계열("설치는 성공했지만 조용히 깨지는 실행부").
**How to apply:** executorch 재현/업그레이드 시 `torch`↔`torchvision` 페어링을 먼저 고정
(torch 2.X ↔ torchvision 0.(15+X), 예: 2.12↔0.27, 2.13↔0.28) 후 그 위에 executorch 설치. `pip install
executorch`의 기본 의존성 해석을 그대로 믿지 말 것.

**기타 실측 (상세 `logs/stage0.5_ladder_log.html`):**
- Edge Impulse CLI: 문서의 `edge-impulse-runner`는 `edge-impulse-run-impulse`로 리네임됨. `daemon`/
  `run-impulse`는 `--help` 무시하고 로그인/시리얼 탐색으로 진입 → 자동화 시 `timeout N ... </dev/null`로 감쌀 것.
- `ai-edge-litert==2.1.6`엔 문서가 "확인 필요"로 둔 `CompiledModel` API가 실제로 없음 — 레거시 `Interpreter` 경로만 유효.
- Qualcomm AI Hub(`qai-hub==0.53.0`): 계정 없으면 `hub.get_devices()`가 `~/.qai_hub/client.ini not found`로 즉시 실패(정상, 에러가 가입 경로 안내).

**Lv.2 PTQ 심화(`logs/lv2_ptq_deep_dive.html`):** LiteRT PTQ 4종(FP32/dynamic-range/float16/full-INT8)을
실측 대조. `model_dynamic.tflite`와 `model_int8.tflite`의 per-channel 가중치 scale이 완전히 동일 —
가중치 양자화는 representative_dataset과 무관(min/max로만). full-INT8이 가장 느리게 측정됐는데 이 데모는
overhead-bound라 일반 결론 아님(인용 주의).

0단계는 [[stage0-env-installed]], 1단계는 [[stage1-quantization-hands-on]], 커밋 전 스캔은 [[repo-is-public-scan-before-commit]].
