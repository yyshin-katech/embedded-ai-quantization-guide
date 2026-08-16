---
name: machine-ai-lap-rtx3080
description: "현재 작업 머신은 AI-LAP 노트북(RTX 3080, 드라이버 595.84) — 반복 이탈하던 Nuvo-6108GC/RTX3060을 대체. venv ~/emb-ai와 ImageNet 27GB는 이관 완료, GPU는 QAT를 300W로 완주(무음 폴백 없음)"
metadata: 
  node_type: memory
  type: project
---

2026-08 현재 이 프로젝트의 실측 작업 머신은 **`AI-LAP`(노트북, NVIDIA RTX 3080, 드라이버 595.84)** 이다
(hostname·`nvidia-smi -L`로 확인). 이전 머신 `yuyeong-Nuvo-6108GC`(산업용 박스, RTX 3060)는 GPU가
Xid 79로 반복 이탈해([[gpu-xid79-fallen-off-bus]]) QAT·2단계 같은 장시간 GPU 워크로드를 완주하지 못했고,
**머신을 옮긴 것이 해결책이었다**(HANDOFF §6 "옮기는 게 해결책" 가설 확인).

**이관된 것(레포 밖, 커밋 대상 아님):**
- venv `~/emb-ai` (14GB) — 정본 스택으로 재설치([[stage0-env-installed]]). activate에 cuDNN·TensorRT
  LD_LIBRARY_PATH 블록 포함. 2단계 DETR도 이 venv로 완주해 스택 유효성 재확인.
- `~/stage1-work` (27GB) — ImageNet val 50,000장 + 전처리 캐시 + 1단계/QAT 스크립트
  ([[imagenet-val-50k-local]], [[stage1-quantization-hands-on]]). `data/val_full` 6.4GB 확인됨.

**아직 없는 것(이 머신에서 미재현):** `~/ladder-work`·`~/emb-ai-work`(0.5단계 산출물·검증 스크립트,
[[study-guide-project]]) — 필요하면 재생성.

**GPU 상태:** RTX 3080은 QAT 2팔을 **300W / 95% util / 70°C로 SW Power Cap 없이 완주**했다
([[qat-recovery-experiment]]). Nuvo/3060에서 상시 걸리던 `0x4`(SW Power Cap)가 여기선 안 뜬다.
그래도 GPU 작업 전 `nvidia-smi -L` 생존 확인 습관은 유지(무음 폴백 판별은 [[stage1-quantization-hands-on]]).

새 PC 인수인계 절차 정본은 레포 `HANDOFF.md`. 커밋 규약은 [[repo-is-public-scan-before-commit]].
