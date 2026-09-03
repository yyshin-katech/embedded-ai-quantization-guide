# publication/ — 논문화 작업 공간

이 저장소에 축적된 실측 코퍼스(`logs/` 30건 · `experiments/` 9군)를 학술 논문으로 정리하기 위한 작업 폴더.

> ⚠️ 기존 `paper/` 폴더와 다름. `paper/`는 `learning_resources.html`용 참고문헌 수집(fetch) 스크립트 전용이고, 여기 `publication/`은 **우리 실측 결과의 논문화**를 위한 공간이다.

## 산출물

| 파일 | 내용 |
|------|------|
| `publishability_assessment.html` | **투고 가능성 검토서.** 30개 리포트를 기여(C1~C8) 단위로 신규성 채점, 위협 요인·갭 분석·타깃 벤류·추천 논문 구조를 정리. 논문 착수 전 설계도. |
| `paper1_isint8portable.md` | **논문 1 초안 v0.3** (arXiv 타깃, 영어). "Is INT8 Portable?" — INT8 비이식성 3축(속도 부호 C1·수치 C2·배포 C3) + 병목 레짐 C4 + 함정 C8. §2 관련연구·References(BibTeX) **완료** — 문헌 조사 반영, 핵심은 **Chen 2026 동시 발표작**(arXiv:2608.13756/2609.00363, C2 메커니즘을 단일 GPU·LLM에서 선규명)에 대한 정직한 포지셔닝(§5 헤드라인을 "발견"→"물리적 디바이스 간 측정"으로, FP32 비트동일 대조 + power-of-two 완화책 future work). **BibTeX 46건 전량 웹 실검증 완료(2026-09-04, arXiv/publisher 대조)** — 오류 6건(저자명·제목·venue) 수정, `[unverified]` 잔존 0. 인프라 식별자 제외·DEEPX 포함. |
| `paper1_isint8portable.tex` + `refs.bib` | **논문 1 LaTeX 변환본** (arXiv-ready). MD v0.3를 자체완결형 `\documentclass[11pt]{article}`로 손수 전사 — 표 4종 booktabs, 인용 `\citep`(natbib numbers/unsrtnat), 유니코드→수식 전 변환. 정적 린트 통과(비ASCII 0·환경/중괄호 균형·**인용 46=정의 46 정확 일치**·표 열 정합). ⚠️로컬 LaTeX 툴체인 부재로 **테스트 컴파일 불가** → Overleaf/arXiv에서 `pdflatex→bibtex→pdflatex×2` 1회 빌드 필요. 저자=플레이스홀더(제출자, Claude 제외). |

## 검토 요약 (2026-09-01 기준)

- **판정:** 투고할 가치 있음. 현실적 목표는 **측정(measurement)/워크숍 트랙 + arXiv 프리프린트**. 톱티어 본회의는 통계 보강(반복실행·CI) 후 도전.
- **추천 논지(옵션 A):** "INT8은 이식되지 않는다" — 속도 이득의 부호(C1)·수치 출력의 재현성(C2, 헤드라인)·배포 가능성/BYO-QDQ(C3)이 모두 타깃 정수 커널 경로에 좌우됨.
- **최대 강점:** 통제된 크로스플랫폼 측정(7종+ 하드웨어) + 완비된 재현 아티팩트.
- **최대 약점:** 통계적 엄밀성(batch1·단일실행·서브셋·지연 CI 부재), 관련연구 미작성.

상세는 `publishability_assessment.html` 참조.

## 다음 단계

1. 논지 1개 확정(옵션 A) 후 핵심 비교 구성에 반복 실행 + 신뢰구간 재실행(아티팩트 재사용).
2. 관련연구(MLPerf/MLPerf Tiny 등) + Threats to Validity 절 초안.
3. 논문 스캐폴드(초록·개요·그림 목록) 착수.
