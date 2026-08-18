---
name: stage2-bevformer-hands-on
description: "2단계 §4.6 BEVFormer-tiny 실측 완료(2026-08-17, 커밋 04e1432 푸시완료) — nuScenes-mini FP32 mAP 0.2647. op 단정 반전 0(초안 맞음)·실전 함정 +2(mmcv op CPU-only export·전체 export는 point_sampling에서 사망). 무컴파일 레거시 env 레시피. 전체 INT8은 범위 밖(포크 필요). SmoothQuant §4.4는 후속 완료(커밋됨)"
metadata:
  node_type: memory
  type: project
---

`study_guide/04_transformer_quantization.md` §4.6(BEVFormer-tiny 초안, 미검증)를 AI-LAP/RTX3080
([[machine-ai-lap-rtx3080]])에서 `fundamentalvision/BEVFormer` `bevformer_tiny`로 실측 검증.
DETR([[stage2-detr-hands-on]])의 자매 과제. 커밋 규약 [[repo-is-public-scan-before-commit]].

**무컴파일 레거시 env 레시피 (hard-won, 재사용 자산):** BEVFormer 실모델은 정본 venv(torch 2.11)에서
**안 돈다**(mmcv-full 1.x/mmdet3d는 torch ≤1.13 전제) → 별도 venv `~/bevf-legacy`. 핵심: **프리빌트 휠로
CUDA 소스 컴파일 전부 우회** — mmcv-full **1.7.0**(`download.openmmlab.com/mmcv/dist/cu117/torch1.13`),
mmdet3d **1.0.0rc6**(py3-none-any, `--no-build-isolation`), torch **1.13.1+cu117**. torch cu117이 **driver
595(CUDA 12.8 capable)에서 하위호환으로 CUDA init 성공**(정본 툴킷은 12.8뿐, 11.7 없음). 핀: mmdet 2.28.2,
numpy 1.23.5, numba 0.58.1, opencv 4.8.1.78, onnx 1.14.1, onnxruntime 1.16.3. BEVFormer plugin의
dd3d/detectron2 체인은 **4파일 패치로 우회**(bevformer_tiny엔 dd3d 불필요).

**2-tier 검증:** Tier A = grid_sample/MSDeformAttn op 지뢰를 정본 `~/emb-ai`에서 격리(b01~b05).
Tier B = 레거시 venv 실 mmcv op(b06) + 실모델(b08 mAP, b09 export).

**결과 — op 단정 반전 0(초안이 맞았다) + 실전 함정 +2:**
1. **op 지뢰 5종 전부 실측 일치**: grid_sample 4D=opset16 경계·5D=opset20·ORT 1.23.2 CUDA 5D 무커널→CPU
   조용히 폴백·TRT rank-4 하드단언(`addGridSample ... nbDims==4`)·MSDeformAttn 분해=grid_sample×num_levels.
2. **함정①(초안에 없음)**: 바닐라 mmcv 커스텀 op은 **CPU 텐서로만 유효 export**. CUDA export는 출력을
   Constant로 baked → `value`/`reference_points`가 graph.input에서 소실(exit 0, checker PASS인데 **silent-wrong**).
3. **함정②(초안에 없음)**: **전체 모델 export는 grid_sample이 아니라 `point_sampling`에서 먼저 죽는다** —
   `encoder.py:119` `lidar2img.view(1,B,num_cam,1,4,4)`에서 `RuntimeError: shape '[1,6,6,1,4,4]' is invalid
   for input of size 96`. 원인 3층: lidar2img/can_bus가 img_metas(비텐서)→numpy→new_tensor로 들어오는
   **forward 내부 기능적 입력** + 시간축 재귀(prev_bev stateful) + 그 안쪽 grid_sample. 포크
   (`DerryHub/BEVFormer_tensorrt`)가 이 셋을 텐서화. **바닐라 유효 export 경로 없음.**

**FP32 실측:** BEVFormer-tiny(33.52M, ckpt 643/643) nuScenes **v1.0-mini** val = **mAP 0.2647 / NDS 0.2667**.
🔴 **캐비앗**: **81 keyframes / 2 scene 스모크**(trailer·construction_vehicle·barrier = 0.000). mAP가 공개
full-val 0.252와 가까운 건 우연, NDS는 0.267 vs 0.354로 벌어짐 → **절대값 문헌비교 불가, 상대 델타만**.
실행 3함정: `PYTHONPATH=<repo>` 필수 · test.py non-dist는 `assert False`라 `torch.distributed.launch
--launcher pytorch` 필수 · dataloader `workers_per_gpu=0`(dict_keys pickle 회피).

**전체 INT8 = 범위 밖(정직한 폴백):** 유효 전체모델 ONNX가 안 나옴 → INT8 PTQ 도달 불가. 포크의 커스텀 op
플러그인+텐서화 래퍼 툴체인(TRT 8.5/CUDA 11.6 기준 → 정본 재빌드 필요) 전제. op-단위 양자화 거동은 Tier A+B2가 검증.

**산출물:** `logs/stage2_bevformer_quantization_report.html`(§1~6·per-class SVG),
`experiments/stage2_bevformer/`(onnx_export_failures.md + b01~b09 + b08/b09 결과 JSON). 검증: 자체 + tech-reviewer 팬인(🔴 0, PASS).
**데이터(git 밖):** `~/bevformer_work/BEVFormer/data/nuscenes/`(v1.0-mini 추출 + temporal info pkl),
`data/can_bus/`(7832), `~/bevformer_work/ckpts/bevformer_tiny_epoch_24.pth`(400MB). BEVFormer clone도 git 밖.

**다음 과제:** SmoothQuant(§4.4, per-token activation — DETR·BEVFormer 공통 범인=activation 양자화 입도) →
BEVFormer 전체 INT8은 포크 플러그인 정본 재빌드 후. 이후 3단계(TensorRT)~7단계+캡스톤은 웹 검증만 된 상태.
