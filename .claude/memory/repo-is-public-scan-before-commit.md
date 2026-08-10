---
name: repo-is-public-scan-before-commit
description: "embedded-ai-quantization-guide 저장소는 public — 실행 로그를 커밋하기 전 반드시 비밀정보 스캔, 발견 시 마스킹 후 커밋(사용자 확인된 방식)"
metadata: 
  node_type: memory
  type: feedback
---

`github.com/yyshin-katech/embedded-ai-quantization-guide`는 **public 저장소**다(API로 확인:
`visibility: public`). 이 저장소의 `logs/`에 올라가는 문서는 실제 터미널 세션을 그대로 옮긴
실행 로그라서, 커밋 전에 비밀정보가 섞여 들어갔는지 확인해야 한다.

**Why:** 2026-07-31 `logs/` 최초 커밋 직전에 `logs/stage0_setup_log.html`에서
`echo '<암호>' | sudo -S -v   # 암호로 자격 캐시` 줄을 발견했다 — [[stage0-env-installed]]에
기록된 이 머신의 **실제 sudo 암호**가 평문으로 들어 있었다. public 저장소에 한 번 푸시되면 나중에
지워도 git 히스토리에 영구히 남고 제거하려면 히스토리 재작성이 필요하므로, 커밋 전에 잡는 것과
후에 잡는 것의 비용 차이가 크다. (이번엔 첫 커밋 전에 잡아서 히스토리 오염 없음.)

**How to apply:** `logs/`나 터미널 출력 기반 문서를 커밋하기 전에
`grep -rniE "password|비밀번호|ghp_|github_pat|secret|api[_-]?key|token|PRIVATE KEY|Bearer "` +
이메일/IP 패턴으로 스캔한다. 발견하면 **사용자가 확인해준 처리 방식은 "마스킹 후 커밋"** —
값을 `<암호>` 같은 플레이스홀더로 바꾸고 파일 자체는 커밋한다(비밀값은 문서의 교육적 가치에
기여하는 바가 없어 마스킹해도 내용 손실이 없다). 파일을 아예 빼거나 `.gitignore`로 돌리는 것보다
이 방식을 선호했다. 단, 발견 사실 자체는 푸시 전에 사용자에게 알릴 것 — 공개 여부와 되돌리기
비용을 알려주고 결정을 받는 게 맞다. 관련: [[repo-git-push-auth]]
