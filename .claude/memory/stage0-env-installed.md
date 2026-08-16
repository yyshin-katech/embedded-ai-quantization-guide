---
name: stage0-env-installed
description: "0단계 환경 정본 스택과 가이드 대비 조정 — LD_LIBRARY_PATH 픽스 2개(cuDNN·TensorRT), dynamo=True가 요구하는 onnxscript, opset 다운컨버트 무음 폴백"
metadata: 
  node_type: memory
  type: project
---

`study_guide/01_environment_setup.md`의 0단계 세팅(**경로 A: 호스트 pip venv, CUDA 12**)을 실제로
설치·검증한 기록. 최초 설치는 2026-07-28 구 머신(Nuvo-6108GC / RTX 3060 / Ubuntu 22.04.5)이었고,
**현재 작업 머신 AI-LAP(RTX 3080)에는 venv `~/emb-ai`가 정본 스택으로 재설치돼 있다**
([[machine-ai-lap-rtx3080]]). (sudo 암호는 이 메모리에 남기지 않는다 — 레포가 public이라
[[repo-is-public-scan-before-commit]].)

**경로/위치:** venv `~/emb-ai`(activate 시 venv 번들 `nvidia/*/lib` + `tensorrt_libs`를
LD_LIBRARY_PATH에 넣는 블록 포함). 구 머신엔 작업물 `~/emb-ai-work`(verify_*.py 등)·데이터
`/data/sets/nuscenes`(v1.0-mini, 10 scenes/404)가 있었으나 AI-LAP엔 아직 없음(필요 시 재생성).

**설치된 정본 스택:** nvcc 12.8 · torch 2.11.0+cu128 · onnxruntime-gpu **1.23.2**(CUDA 12) ·
onnx **1.18.0**(IR 11, 최대 opset 23) · tensorrt-cu12 10.16.1.11 · polygraphy 0.50.3 ·
numpy **1.26.4** · **onnxscript 0.7.1 + onnx-ir 0.2.1** · Docker + NVIDIA Container Toolkit.
(2단계 DETR도 이 스택으로 완주 — 유효성 재확인. [[stage2-detr-hands-on]])

**실측 함정 (재현·후속 단계에서 중요):**
1. 가이드 핀 `onnxruntime-gpu==1.28.0`은 존재하지 않음 → PyPI `onnxruntime-gpu<1.27`로 CUDA-12 최신 **1.23.2** 설치.
2. ORT 1.23.2 max IR 11이라 onnx 최신(IR 13) 로드 실패 → **onnx 1.18.0(IR 11) 고정**. export 시 opset/IR 상한 주의.
3. ORT CUDA EP가 `libcudnn.so.9`를 못 찾아 **조용한 CPU 폴백** → activate LD_LIBRARY_PATH 블록으로 해결.
4. **TensorRT EP도 같은 병:** `libnvinfer.so.10`이 `tensorrt_libs/`(=`nvidia/*/lib` 글롭 밖)에 있어
   못 찾고 조용히 CPU 폴백 → activate에 `tensorrt_libs` 경로 추가로 해결(ResNet-18 p50 1.40ms
   CUDA→0.41ms TRT). `01`의 3-4-a절에 반영 완료(`verify_trt_ep.py` 포함).
5. torch 2.11 `torch.onnx.export`는 **기본이 `dynamo=True`**, 그 경로가 `onnxscript`를 요구(없으면
   export 첫 줄에서 ModuleNotFoundError). `opset_version=17` 요청해도 내부 18로 만든 뒤 다운컨버트
   실패 → **조용히 opset 18로 폴백**, 결과가 `.onnx`+`.onnx.data` 2파일로 분리(옮길 때 둘 다). 대안 `dynamo=False`.
6. nuscenes-devkit이 numpy<2 요구 → numpy 1.26.4로 다운그레이드.

0.5단계는 [[study-guide-project]], 1단계는 [[stage1-quantization-hands-on]].
