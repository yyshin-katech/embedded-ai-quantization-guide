# 작업 메모리 사본

이 가이드를 실제 머신에서 따라 하며 쌓인 **실측 기록**입니다. Claude Code의 로컬 메모리
(`~/.claude/projects/<project>/memory/`)를 저장소로 복사한 것이라, 다른 PC에서 작업을
이어받을 때 맥락이 끊기지 않게 하려는 목적입니다.

정본은 로컬 메모리 쪽이고 이건 사본입니다. 작업 상태 요약은 [`../../HANDOFF.md`](../../HANDOFF.md)를 먼저 보세요.

> ⚠️ 메모리는 **작성 시점의 관찰**입니다. 파일·함수·플래그를 지목한 서술은 지금도 유효한지
> 코드로 확인한 뒤 사실로 인용하세요.

현재 작업 머신은 **AI-LAP (RTX 3080)** 입니다 — 이전 Nuvo-6108GC(RTX 3060)가 GPU 고장으로 교체됐습니다.

| 파일 | 내용 |
|---|---|
| [`machine-ai-lap-rtx3080.md`](machine-ai-lap-rtx3080.md) | 현재 작업 머신(AI-LAP 노트북, RTX 3080) — 죽은 Nuvo/RTX3060 대체, venv·데이터 이관, GPU가 QAT를 300W로 완주 |
| [`stage0-env-installed.md`](stage0-env-installed.md) | 0단계 환경 — 확정 스택 버전, `LD_LIBRARY_PATH` 픽스 2개(cuDNN·TensorRT), `dynamo=True`가 요구하는 `onnxscript`, opset 다운컨버트 무음 폴백 |
| [`study-guide-project.md`](study-guide-project.md) | 0.5단계 배포 사다리 — `executorch`/`torch`/`torchvision` 3자 ABI 핀 충돌과 해법, LiteRT `CompiledModel` API 부재, Lv.2 PTQ 4종 실측 |
| [`stage1-quantization-hands-on.md`](stage1-quantization-hands-on.md) | 1단계 양자화 이론 2회 실행 — ORT Entropy가 MinMax로 퇴화(산출 md5 동일), TensorRT 폴백 원인은 activation zero-point≠0 하나뿐, 50k 재실행 정정 12건 |
| [`imagenet-val-50k-local.md`](imagenet-val-50k-local.md) | ImageNet val 50,000장 확보·검증 경위, 라벨 규약, 전처리 2종(`crop_tv` vs `crop_squash`)의 −1.07%p 차이 |
| [`qat-recovery-experiment.md`](qat-recovery-experiment.md) | **완료** — QAT 회복(W4A8 손실변형): FP32→PTQ 4-bit −24.16%p→QAT 97.1% 회복, QAT−대조군 −1.50%p("공짜 아님") |
| [`stage2-detr-hands-on.md`](stage2-detr-hands-on.md) | 2단계 DETR INT8(커밋 41dc49e) — 초안 단정 3건 반전(export 블로커=SDPA·op선택 mixed 실패·손상 분산), 다음=SmoothQuant §4.4 |
| [`stage2-bevformer-hands-on.md`](stage2-bevformer-hands-on.md) | 2단계 BEVFormer-tiny §4.6 — FP32 nuScenes-mini mAP 0.2647(스모크), op 단정 반전 0·실전 함정 +2(mmcv op CPU-only export·전체 export는 point_sampling에서 사망), 전체 INT8은 포크 필요(범위 밖), 무컴파일 레거시 env 레시피 |
| [`gpu-xid79-fallen-off-bus.md`](gpu-xid79-fallen-off-bus.md) | (구 머신 이력·해소) RTX 3060 Xid 79 3회 재발 진단 — SW Power Cap 상시 점등, 배치 축소 무효. 3080 이관으로 해결 |
| [`repo-is-public-scan-before-commit.md`](repo-is-public-scan-before-commit.md) | 커밋 규약 — 시크릿 스캔, main 직접 커밋, 푸시는 요청 시만 |

## 공개본에서 손댄 것

저장소가 public이라 복사 시 다음을 처리했습니다.

- **sudo 암호 마스킹/제외** — 이전 작업 머신의 실제 sudo 암호가 평문으로 있던 것을 마스킹했고,
  갱신본(2026-08-16)부터는 아예 기록하지 않습니다. 비밀값은 문서의 교육적 가치에 기여하지 않아 손실이 없습니다.
- **git 자격증명 세부 제외** — `repo-git-push-auth.md`(자격증명 저장 방식)와 PAT 저장 위치 언급은
  공개 가치가 없고 노출 위험만 있어 공개본에서 뺐습니다. 새 PC에서는 각자 `git` 인증을 새로 설정하면 됩니다.
- **로컬 메타 제거** — `originSessionId`, `modified` 필드는 로컬 세션 식별자라 지웠습니다.
