# 작업 메모리 사본

이 가이드를 실제 머신에서 따라 하며 쌓인 **실측 기록**입니다. Claude Code의 로컬 메모리
(`~/.claude/projects/<project>/memory/`)를 저장소로 복사한 것이라, 다른 PC에서 작업을
이어받을 때 맥락이 끊기지 않게 하려는 목적입니다.

정본은 로컬 메모리 쪽이고 이건 사본입니다. 작업 상태 요약은 [`../../HANDOFF.md`](../../HANDOFF.md)를 먼저 보세요.

> ⚠️ 메모리는 **작성 시점의 관찰**입니다. 파일·함수·플래그를 지목한 서술은 지금도 유효한지
> 코드로 확인한 뒤 사실로 인용하세요.

| 파일 | 내용 |
|---|---|
| [`stage0-env-installed.md`](stage0-env-installed.md) | 0단계 환경 실제 설치 결과 — 확정 스택 버전, `LD_LIBRARY_PATH` 픽스 2개(cuDNN·TensorRT), `dynamo=True`가 요구하는 `onnxscript`, opset 다운컨버트 무음 폴백 |
| [`study-guide-project.md`](study-guide-project.md) | 0.5단계 배포 사다리 — `executorch`/`torch`/`torchvision` 3자 ABI 핀 충돌과 해법, LiteRT `CompiledModel` API 부재, Lv.2 PTQ 4종 실측 |
| [`stage1-quantization-hands-on.md`](stage1-quantization-hands-on.md) | 1단계 양자화 이론 2회 실행 — ORT Entropy가 MinMax로 퇴화(산출 md5 동일), TensorRT 폴백 원인은 activation zero-point≠0 하나뿐, 50k 재실행 정정 12건 |
| [`qat-recovery-experiment.md`](qat-recovery-experiment.md) | **미완 작업** — QAT 회복 실험 2팔 설계와 현재 상태. 회복률을 읽지 말아야 하는 이유 |
| [`imagenet-val-50k-local.md`](imagenet-val-50k-local.md) | ImageNet val 50,000장 확보·검증 경위, 라벨 규약, 전처리 2종(`crop_tv` vs `crop_squash`)의 −1.07%p 차이 |
| [`gpu-xid79-fallen-off-bus.md`](gpu-xid79-fallen-off-bus.md) | RTX 3060 Xid 79 3회 재발 진단 — 배치 축소가 무효한 레버라는 실측 반증, SW Power Cap 상시 점등 |
| [`repo-is-public-scan-before-commit.md`](repo-is-public-scan-before-commit.md) | 커밋 전 비밀정보 스캔 절차와 "마스킹 후 커밋" 방침 |

## 공개본에서 손댄 것

저장소가 public이라 복사 시 다음을 처리했습니다.

- **sudo 암호 마스킹** — `stage0-env-installed.md`와 `repo-is-public-scan-before-commit.md`에
  이전 작업 머신의 실제 sudo 암호가 평문으로 있었습니다. `<암호>`로 치환했습니다. 비밀값은
  문서의 교육적 가치에 기여하는 바가 없어 마스킹해도 내용 손실이 없습니다.
- **`repo-git-push-auth.md` 제외** — 그 머신의 git 자격증명 저장 방식(경로·계정)만 담고 있어
  공개 가치가 없고 노출 위험만 있습니다. 새 PC에서는 각자 `git` 인증을 새로 설정하면 됩니다.
- **로컬 메타 제거** — `originSessionId`, `modified` 필드는 로컬 세션 식별자라 지웠습니다.
