---
name: embedded-guide-orchestrator
description: "AI 모델 양자화 → 임베디드 배포 학습 가이드(guide (1).html 기반)의 단계별 MD 문서 세트를 만들거나 갱신하는 오케스트레이터. '양자화 임베디드 가이드 작성/보완/업데이트', '단계별 md 다시 작성', '특정 단계만 수정', '가이드 재검토' 등의 요청 시 사용한다. guide-author(리서치+작성)와 tech-reviewer(검증)를 조율하는 Fan-out/Fan-in + Producer-Reviewer 하네스."
---

# embedded-guide-orchestrator — 양자화 임베디드 가이드 하네스

## 목표
`guide (1).html`의 단계 구조를 기반으로, Ubuntu 22.04 + NVIDIA RTX 환경에서 **읽고 따라 하면 실행되는** 단계별 학습 가이드 MD 세트를 `study_guide/`에 생성/갱신하고, 필요 시 각 단계의 HTML 버전(`md-to-html` 스킬)까지 렌더한다.

## 실행 모드
**하이브리드 = Fan-out/Fan-in(서브 에이전트) + Producer-Reviewer.**
단계 문서는 서로 독립이라 `guide-author`를 병렬 서브 에이전트로 팬아웃하고, `tech-reviewer`가 팬인으로 교차 검증한다. 팀 통신이 구조적으로 불필요하므로 `Agent` 도구를 직접 호출한다. 모든 호출에 `model: "opus"`.

## Phase 0: 컨텍스트 확인
1. `study_guide/`와 `_workspace/` 존재 여부 확인.
   - 미존재 → **초기 실행** (아래 Phase 1~4 전체)
   - 존재 + "특정 단계만 수정" → **부분 재실행** (해당 파일 담당 author 1명만 재호출 → reviewer)
   - 존재 + "전체 갱신/새 입력" → 기존 `study_guide/`를 `study_guide_prev/`로 이동 후 초기 실행
2. 원본 `guide (1).html`을 읽어 단계별 섹션 텍스트를 확보.

## Phase 1: 작업 분해 (파일명 맵)
`stage-guide-writing` 스킬의 파일명 맵을 사용. 10개 콘텐츠 문서 + README.
author 배정(1인 1~2문서):

| Author | 담당 파일 | HTML 단계 |
|--------|-----------|-----------|
| author-1 | 01_environment_setup.md | 0 |
| author-2 | 02_deployment_ladder.md | 0.5 |
| author-3 | 03_quantization_theory.md | 1 |
| author-4 | 04_transformer_quantization.md | 2 |
| author-5 | 05_tensorrt.md | 3 |
| author-6 | 06_multi_soc.md | 4 |
| author-7 | 07_infrastructure.md | 5 |
| author-8 | 08_capstone.md | 캡스톤 |
| author-9 | 09_roadmap.md + 10_pitfalls.md | 로드맵·함정 |

## Phase 2: Fan-out (병렬 작성)
- 각 author를 `general-purpose` 서브 에이전트(`model: opus`)로 **한 메시지에 병렬 호출**.
- 각 프롬프트에 포함: 담당 단계명, 해당 HTML 섹션 텍스트, 출력 절대경로, 파일명 맵, "두 스킬(embedded-ai-research, stage-guide-writing)을 먼저 읽어라" 지시.
- 산출물은 `study_guide/`에 직접 저장. 중간 로그는 `_workspace/`.

## Phase 3: Fan-in (검증)
- 모든 author 반환 후 `tech-reviewer`(`general-purpose`, `model: opus`) 1~2명 호출.
- 입력: 완성된 파일 목록 + 원본 HTML + 파일명 맵.
- 출력: `_workspace/review_report.md`. 🔴 치명 이슈는 직접 수정.

## Phase 4: 통합
- 오케스트레이터가 `study_guide/README.md`(인덱스: 학습 순서, 파일 링크, 12주 매핑, 전제조건)를 작성.
- 사용자에게 결과 요약 + `review_report.md`의 남은 🟡/🟢 보고.

## Phase 5: HTML 렌더 (요청 시)
- MD 심화·검증이 끝난 뒤 `md-to-html` 스킬로 각 `*.md` → `*.html` 생성. HTML은 파생물이므로 내용 수정은 항상 MD에서, HTML은 재생성.
- `python3 .claude/skills/md-to-html/scripts/render.py /mnt/h/AI_Model_Embeddings/study_guide`

## 데이터 전달 프로토콜
- **파일 기반**(주): 산출물 `study_guide/`, 중간물 `_workspace/`.
- **반환값 기반**: 각 author/reviewer가 요약만 메인에 반환(본문 재출력 금지).
- 파일명 컨벤션: 최종물은 파일명 맵대로, 리뷰 로그는 `_workspace/review_report.md`.

## 에러 핸들링
- author 실패 시 1회 재시도. 재실패하면 그 문서는 스텁으로 남기고 README에 "미완성" 표기, 사용자에게 보고.
- reviewer가 구조적 결함(섹션 누락)을 보고하면 해당 author만 재호출.
- 상충/불확실 정보는 삭제하지 않고 출처 병기.

## 테스트 시나리오
- **정상 흐름**: 초기 실행 → 9 author 병렬 → 파일 10개 생성 → reviewer 교차검증 → README → 요약 보고.
- **에러 흐름**: author-5(TensorRT) 웹검색 실패 → 1회 재시도 → 공식 URL로 폴백 + "링크 확인 필요" 표기 → reviewer가 플래그 → 사용자 보고.
- **부분 재실행**: "3단계만 다시" → author-5만 재호출 → reviewer가 05 및 그 상호참조만 재검증.
