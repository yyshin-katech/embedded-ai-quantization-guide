---
name: repo-is-public-scan-before-commit
description: "이 레포 커밋 규약 — public이라 커밋 전 반드시 시크릿 스캔(발견 시 마스킹 후 커밋), main에 직접 커밋(브랜치 X), 푸시는 명시 요청 시에만"
metadata: 
  node_type: memory
  type: feedback
---

`github.com/yyshin-katech/embedded-ai-quantization-guide`는 **public 저장소**(API로 `visibility: public`
확인). `logs/`·터미널 출력 기반 문서는 실제 세션을 옮긴 것이라 비밀정보가 섞일 수 있다.

**① 커밋 전 시크릿 스캔.**
**Why:** 2026-07-31 `logs/stage0_setup_log.html`에서 이 머신의 **실제 sudo 암호가 평문**으로 발견됐고
(`echo '...' | sudo -S`), 2026-08-10 `.claude/memory/` 복사 때도 sudo 암호 2곳이 걸렸다. public에 한 번
푸시되면 히스토리에 영구히 남아 제거에 재작성이 필요하므로 커밋 전에 잡는 게 압도적으로 싸다.
**How to apply:** `grep -rniE "password|비밀번호|ghp_|github_pat|secret|api[_-]?key|token|PRIVATE KEY|Bearer"`
+ 이메일/IP 패턴으로 스캔. 발견 시 **사용자 확인된 방식은 "마스킹 후 커밋"** — 값만 `<암호>`류
플레이스홀더로 바꾸고 파일은 커밋(비밀값은 교육적 가치가 없어 마스킹해도 손실 없음). 발견 사실은 푸시 전 사용자에게 알릴 것.

**② main에 직접 커밋한다(브랜치 X).**
**Why:** 모든 기존 커밋이 main 위에 선형으로 쌓여 있고 CLAUDE.md 변경이력 워크플로가 그 전제로 설계됨.
일반 기본값 "작업 시 브랜치 먼저"는 이 프로젝트에선 틀림.
**How to apply:** 커밋 요청받으면 스캔 → main에 커밋. 커밋 메시지 끝에
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. git identity는 로컬 전용(user=yyshin, email은 이 머신 git config에만 둔다).

**③ 푸시는 명시 요청 시에만.** "커밋"은 커밋까지만 — 푸시는 별도 요청. (예: 41dc49e는 커밋만, 미푸시 — [[stage2-detr-hands-on]].)

메모리의 git 사본(레포 `.claude/memory/`)을 갱신·커밋할 때도 위 스캔이 그대로 적용된다.
