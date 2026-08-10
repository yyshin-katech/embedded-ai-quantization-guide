---
name: stage0-env-installed
description: Stage 0 환경이 이 머신에 실제 설치 완료됨 — 설치된 스택 버전과 가이드 대비 조정 사항
metadata: 
  node_type: memory
  type: project
---

2026-07-28에 `study_guide/01_environment_setup.md`의 0단계 세팅을 이 머신(Ubuntu 22.04.5, RTX 3060 12GB)에 **경로 A(호스트 pip venv, CUDA 12)** 로 실제 설치 완료. sudo 비밀번호는 `<암호>`(로컬 메모리에만 기록).

> 🔴 **위 줄은 이 머신의 실제 sudo 암호다 — public 저장소로 복사·인용 금지.** 저장소
> `github.com/yyshin-katech/embedded-ai-quantization-guide`는 public이므로, 이 메모리를 git으로
> 복사할 때는 반드시 `<암호>`로 마스킹한다([[repo-is-public-scan-before-commit]]).
> 실제로 2026-08-10 `.claude/memory/` 복사 시 이 줄이 걸려 마스킹 처리했다.

2026-07-31 재검증: 4개 검증 스크립트 모두 통과(CUDA EP 실제 활성, nuScenes 10/404). 드라이버가 595.71.05 → **595.84** 로 올라갔으나 CUDA 상한이 여전히 ≥12.8이라 스택 영향 없음.

**경로/위치:** venv `~/emb-ai` (activate하면 자동으로 venv 번들 `nvidia/*/lib`를 LD_LIBRARY_PATH에 추가하는 블록이 들어감) · 작업/산출물 `~/emb-ai-work` (ENV.md, verify_env.py, verify_cuda_ep.py, load_nuscenes.py, requirements.txt) · 데이터 `/data/sets/nuscenes` (nuScenes v1.0-mini, 10 scenes/404 samples).

**설치된 정본 스택:** nvcc 12.8.93 · torch 2.11.0+cu128 · onnxruntime-gpu **1.23.2**(CUDA 12) · onnx **1.18.0**(IR 11, 최대 opset 23) · tensorrt-cu12 10.16.1.11 · polygraphy 0.50.3 · numpy **1.26.4** · **onnxscript 0.7.1 + onnx-ir 0.2.1**(2026-07-31 추가) · Docker 29.6.2 + NVIDIA Container Toolkit 1.19.1.

**2026-07-31 추가 발견:** torch 2.11의 `torch.onnx.export`는 **기본이 `dynamo=True`** 이고 그 경로가 `onnxscript`를 요구한다. 최초 설치 스택에 빠져 있어서 3·4·5단계 export가 전부 `ModuleNotFoundError: No module named 'onnxscript'`로 첫 줄에서 죽는 상태였음 → `pip install onnxscript`로 해결(onnx 1.18.0/numpy 1.26.4 핀 안 깨짐). 대안은 `dynamo=False`.

**2026-07-31 0.5단계(배포 사다리) 실습 중 추가 발견 — TensorRT EP도 cuDNN과 같은 병이었다:** `ort.get_available_providers()`엔 `TensorrtExecutionProvider`가 나오지만, 실제로 `InferenceSession`을 그 EP로 만들면 `libnvinfer.so.10: cannot open shared object file`로 로드 실패 → 조용히 CPU로 fallback. 원인: `libnvinfer.so.10`이 `tensorrt_cu12_libs` pip 패키지가 까는 `~/emb-ai/lib/python3.10/site-packages/tensorrt_libs/`에 있는데, 기존 activate 블록은 `nvidia/*/lib`만 LD_LIBRARY_PATH에 넣고 있어서 이 경로는 빠져 있었음. → `~/emb-ai/bin/activate`에 `tensorrt_libs` 경로를 동적으로 추가하는 블록을 보강해서 해결(실측 확인: TensorrtExecutionProvider가 실제 활성 EP로 잡히고, ResNet-18 p50 1.40ms(CUDA)→0.41ms(TensorRT)로 개선). **2026-07-31에 `01_environment_setup.md`의 3-4-a절에도 이 fix를 반영 완료** — `_EMBAI_TRTLIBS` 계산·`tensorrt_libs` 진단 명령·`verify_trt_ep.py` 검증 스크립트·트러블슈팅 표 신규 행까지 문서화됨. 섹션 heading은 안 바꿔서 기존 앵커 참조(3곳+)는 안 깨짐.

또한 `torch.onnx.export(..., opset_version=17)`을 실제로 돌리면(dynamo=True 경로) 내부적으로 opset 18로 먼저 만들고 17로 다운컨버트를 시도하다 `onnxscript.version_converter`에서 `RuntimeError`(`No initializer or constant input to node found`)가 나며 실패 → **조용히 opset 18로 폴백**(요청한 17이 아님, 다만 IR 10이라 ORT 1.23.2 로드엔 문제 없음). 그리고 결과 `.onnx` 파일은 **`.onnx` + `.onnx.data`(가중치, 44MB+) 2-파일 구조**로 분리 저장됨(dynamo 익스포터의 기본 동작) — 배포/BYOM 업로드 시 `.data` 파일을 빠뜨리면 로드가 깨지므로 반드시 같이 옮겨야 함.

**가이드 대비 조정(재현·후속 단계에서 중요):**
1. 드라이버 이미 설치돼 3-1·재부팅 생략.
2. 가이드 핀 `onnxruntime-gpu==1.28.0`(CUDA-12 Azure 피드)은 존재하지 않음 → PyPI `onnxruntime-gpu<1.27`로 CUDA-12 최신 **1.23.2** 설치.
3. ORT 1.23.2가 max IR 11이라 onnx 최신(1.22, IR 13) 로드 실패 → **onnx를 1.18.0(IR 11)로 고정**. 3·4단계 ONNX export 시 opset/IR 상한 주의.
4. ORT CUDA EP가 `libcudnn.so.9`(cuDNN9는 CUDA toolkit deb에 없음, torch가 nvidia-cudnn-cu12로 제공)를 못 찾아 조용한 CPU fallback 발생 → activate에 LD_LIBRARY_PATH 블록으로 영구 해결.
5. nuscenes-devkit 1.2.0이 numpy<2 요구 → numpy 2.2.6에서 1.26.4로 다운그레이드(core 스택 정상).

다음은 [[study-guide-project]]의 0.5단계(`02_deployment_ladder.md`).
