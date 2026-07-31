## 하네스: AI 양자화 → 임베디드 배포 학습 가이드 제작

**목표:** `guide (1).html`의 단계 구조를 기반으로, Ubuntu 22.04 + NVIDIA RTX 환경에서 실행 가능한 단계별 학습 가이드 MD 세트(`study_guide/`)를 생성/갱신한다.

**트리거:** 양자화·임베디드 가이드 문서의 작성/보완/업데이트/부분수정/재검토 요청 시 `embedded-guide-orchestrator` 스킬을 사용하라. 단순 질문(개념 설명 등)은 직접 응답 가능.

**구성 요소:** 에이전트 `guide-author`(리서치+작성), `tech-reviewer`(검증) / 스킬 `embedded-ai-research`, `stage-guide-writing`, `tech-guide-review`, `md-to-html`(HTML 렌더러) / 오케스트레이터 `embedded-guide-orchestrator`. (상세는 `.claude/agents/`, `.claude/skills/` 참조)

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-07-27 | 초기 구성 (Fan-out/Fan-in + Producer-Reviewer) | 전체 | 단계별 가이드 MD 제작 하네스 신규 구축 |
| 2026-07-28 | MD 심화 재작성 + `md-to-html` 스킬(HTML 렌더러) 추가 | skills/md-to-html, study_guide/* | 각 단계 상세화 및 HTML 버전 제공 요청 |
| 2026-07-31 | 실측 반영 정정 (부분 재실행: author-1 → reviewer-1) | study_guide/01·02·03·04·05·06·07·09·10, README | 0단계를 실제 머신에 설치해보니 정본 핀 `onnxruntime-gpu==1.28.0`이 미존재, `onnx` IR 상한·cuDNN 경로·`numpy<2`·`onnxscript` 누락이 드러남 → 실측 스택(ORT 1.23.2 / onnx 1.18.0)으로 전 문서 통일 |
| 2026-07-31 | 0.5단계 실습 반영: TensorRT EP LD_LIBRARY_PATH 픽스 (3-4-a 보강, 부분수정) | study_guide/01_environment_setup.md·html, README | 0.5단계(배포 사다리) Lv.3 실습 중 `TensorrtExecutionProvider`가 provider 목록엔 나오지만 `libnvinfer.so.10`(venv의 `tensorrt_libs` 패키지, `nvidia/*/lib` 글롭 밖)을 못 찾아 조용히 CPU로 폴백하는 걸 실측 발견(p50 11.83ms=CPU급 → 픽스 후 0.41ms) → cuDNN과 동일한 LD_LIBRARY_PATH 패턴으로 해결, `verify_trt_ep.py` 검증 스크립트·트러블슈팅 행 추가 |
