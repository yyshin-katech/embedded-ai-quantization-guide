---
name: md-to-html
description: "임베디드 AI 양자화 학습 가이드의 Markdown 문서(study_guide/*.md)를 다크 테마 HTML로 변환한다. '각 단계 HTML 버전 만들어줘', 'md를 html로', '보기 좋게 렌더링', '가이드 HTML 생성/갱신' 요청 시 사용. 외부 의존성 없는 자체 파이썬 스크립트(scripts/render.py)로 결정적 변환하며, 원본 guide 디자인(진행률 체크박스·네비게이션·목차)을 재현한다. MD를 새로 렌더하거나 갱신할 때 반드시 이 스킬을 쓴다."
---

# md-to-html — 학습 가이드 Markdown → HTML 렌더러

## 목적
`study_guide/`의 각 단계 MD를 **읽기 편한 HTML**로 변환한다. LLM이 아니라 **결정적 스크립트**로 처리하여 일관성·재현성·저비용을 보장한다.

## 언제 쓰는가
- MD 문서를 (재)작성/심화한 뒤 HTML 버전을 만들거나 갱신할 때.
- "각 단계 HTML로", "보기 좋게", "가이드 페이지 렌더" 등의 요청.

## 왜 스크립트인가
HTML 변환은 규칙적·반복적이다. 에이전트가 매번 손으로 HTML을 쓰면 문서마다 스타일이 어긋나고 토큰이 낭비된다. `scripts/render.py`는 pandoc/markdown 라이브러리 없이(오프라인 안전) 동작하며, 내 문서가 `stage-guide-writing` 규약(표준 표·펜스 코드·`- [ ]` 체크박스)을 따르므로 안정적으로 파싱된다.

## 사용법
```bash
# study_guide/ 전체를 렌더 (기본 경로)
python3 .claude/skills/md-to-html/scripts/render.py /mnt/h/AI_Model_Embeddings/study_guide

# 단일 파일만
python3 .claude/skills/md-to-html/scripts/render.py /mnt/h/AI_Model_Embeddings/study_guide/05_tensorrt.md
```
- 입력 디렉토리의 모든 `*.md`를 같은 이름 `*.html`로 출력한다.
- `README.md` → `README.html`(인덱스 허브). 문서 내 `NN_*.md` / `README.md` 링크는 자동으로 `.html`로 재작성된다.

## 산출 HTML 특징 (원본 guide 재현 + 강화)
- 다크 테마(원본 `guide (1).html` 팔레트 재사용), 반응형, 인쇄 스타일.
- 상단 **진행률 바** + `- [ ]` 체크박스 → 체크 상태 `localStorage` 저장(문서별 키).
- **이전/다음 단계 네비게이션** + 인덱스(README) 링크.
- h2/h3 자동 **목차(TOC)**.
- 코드 블록/표/콜아웃(💡⚠️🔴) 스타일링.

## 렌더가 잘 되게 하는 MD 규약 (작성 시 지킬 것)
- 코드는 펜스(```lang) + 언어 태그. 표는 표준 파이프 표(헤더 구분행 `|---|`).
- 체크박스는 `- [ ]` / `- [x]`. 콜아웃은 `> 💡/⚠️/🔴` 인용구.
- 문서 첫 줄은 `# 제목`(H1) 하나.

## 워크플로우 상의 위치
오케스트레이터(`embedded-guide-orchestrator`)의 **렌더 단계**에서, MD 심화·검증이 끝난 뒤 마지막에 실행한다. MD가 정본이고 HTML은 파생물이므로, 내용 수정은 항상 MD에서 하고 HTML은 재생성한다.
