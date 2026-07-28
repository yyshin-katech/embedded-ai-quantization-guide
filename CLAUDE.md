## 하네스: AI 양자화 → 임베디드 배포 학습 가이드 제작

**목표:** `guide (1).html`의 단계 구조를 기반으로, Ubuntu 22.04 + NVIDIA RTX 환경에서 실행 가능한 단계별 학습 가이드 MD 세트(`study_guide/`)를 생성/갱신한다.

**트리거:** 양자화·임베디드 가이드 문서의 작성/보완/업데이트/부분수정/재검토 요청 시 `embedded-guide-orchestrator` 스킬을 사용하라. 단순 질문(개념 설명 등)은 직접 응답 가능.

**구성 요소:** 에이전트 `guide-author`(리서치+작성), `tech-reviewer`(검증) / 스킬 `embedded-ai-research`, `stage-guide-writing`, `tech-guide-review`, `md-to-html`(HTML 렌더러) / 오케스트레이터 `embedded-guide-orchestrator`. (상세는 `.claude/agents/`, `.claude/skills/` 참조)

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-07-27 | 초기 구성 (Fan-out/Fan-in + Producer-Reviewer) | 전체 | 단계별 가이드 MD 제작 하네스 신규 구축 |
| 2026-07-28 | MD 심화 재작성 + `md-to-html` 스킬(HTML 렌더러) 추가 | skills/md-to-html, study_guide/* | 각 단계 상세화 및 HTML 버전 제공 요청 |
