---
name: embedded-ai-research
description: "AI 모델 양자화·임베디드 배포 학습 가이드를 쓸 때 필요한 웹 리서치 방법론과 신뢰 출처 모음. TensorRT/TIDL/QNN/DRP-AI/ONNX Runtime/LiteRT/ExecuTorch/Edge Impulse의 최신 버전·op 지원·설치법·논문을 조사하거나, 임베디드 AI 가이드 문서의 '참고 사이트/참고문헌'을 채우거나, 명령어·버전을 검증할 때 반드시 사용한다. 재조사·업데이트·링크 검증 요청에도 사용."
---

# embedded-ai-research — 임베디드 AI 리서치 방법론

## 언제 쓰는가
임베디드 AI 양자화 가이드의 한 단계를 쓰기 전에, 그 단계에서 언급되는 툴/버전/op/논문이 **2026년 7월 기준으로 맞는지** 웹으로 검증할 때. 추측으로 버전·명령어를 쓰지 않기 위한 스킬이다.

## 리서치 원칙
1. **1차 출처 우선** — 벤더 공식 문서 · 공식 GitHub 릴리스 노트 · arXiv 원문. 블로그는 보조로만.
2. **버전을 못 박는다** — "최신"이라 쓰지 말고 실제 버전 번호(예: `TensorRT 10.x`, `CUDA 12.x`)를 확인해 적는다. 확인 시점(`2026-07 기준`)을 병기.
3. **op 지원은 릴리스 노트에서** — "이 op가 지원되는가"는 벤더 릴리스 노트/지원 op 목록에서 확인. 미지원/불안정 사례는 그대로 기록(가이드의 핵심 가치).
4. **검증 못 하면 표기** — 확인 실패 시 `> ⚠️ 확인 필요:` + 출처 후보 URL을 남긴다. 지어내지 않는다.

## 웹 검색 쿼리 팁
- 버전 확인: `"TensorRT" release notes 2026 site:docs.nvidia.com`
- 설치법: `nvidia-container-toolkit install Ubuntu 22.04`
- op 지원: `edgeai-tidl-tools supported operators`, `onnxruntime QNN EP supported ops`
- 논문: arXiv ID로 직접 접근(아래 표) 후 최신 후속 논문 검색.

## 신뢰 출처 맵 (조사 시작점 — 실제 최신 URL은 웹으로 재확인)

### 환경·기반
- NVIDIA CUDA / driver: https://developer.nvidia.com/cuda-downloads , `ubuntu-drivers devices`
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/
- Docker Engine (Ubuntu): https://docs.docker.com/engine/install/ubuntu/
- nuScenes: https://www.nuscenes.org/ , devkit: https://github.com/nutonomy/nuscenes-devkit

### 양자화 이론·툴
- ONNX Runtime 양자화: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- PyTorch 양자화: https://pytorch.org/docs/stable/quantization.html
- NVIDIA TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- Neural Compressor(Intel): https://github.com/intel/neural-compressor

### 배포 사다리 (0.5단계)
- Edge Impulse: https://docs.edgeimpulse.com/  (2025 Qualcomm 인수 → Dragonwing/Hexagon 배포)
- LiteRT (구 TFLite): https://ai.google.dev/edge/litert
- ONNX Runtime EP: https://onnxruntime.ai/docs/execution-providers/
- ExecuTorch: https://pytorch.org/executorch/ , https://github.com/pytorch/executorch  (2025 말 v1.0 GA)
- Qualcomm AI Hub: https://aihub.qualcomm.com/

### TensorRT (3단계)
- TensorRT 문서: https://docs.nvidia.com/deeplearning/tensorrt/
- TensorRT GitHub(플러그인/샘플): https://github.com/NVIDIA/TensorRT
- Polygraphy: https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy
- Nsight Systems/Compute: https://developer.nvidia.com/nsight-systems
- Jetson/Orin: https://developer.nvidia.com/embedded/jetson-orin

### 멀티 SoC (4단계)
- TI TIDL(edgeai-tidl-tools): https://github.com/TexasInstruments/edgeai-tidl-tools , https://github.com/TexasInstruments/edgeai-tensorlab
- Qualcomm QNN via ONNX Runtime: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Qualcomm AI Engine Direct(QNN) SDK: https://qpm.qualcomm.com/ (Qualcomm AI Hub 문서 경유)
- Renesas DRP-AI TVM: https://github.com/renesas-rz/rzv_drp-ai_tvm , https://www.renesas.com/ (RZ/V2H)

### 핵심 논문 (arXiv ID)
| 주제 | 논문 | arXiv |
|------|------|-------|
| 양자화 서베이 | Gholami et al., A Survey of Quantization Methods (2021) | 2103.13630 |
| 양자화 백서 | Nagel et al., A White Paper on NN Quantization (Qualcomm, 2021) | 2106.08295 |
| 정수 추론 원조 | Jacob et al., Integer-Arithmetic-Only Inference (2018) | 1712.05877 |
| activation outlier | Xiao et al., SmoothQuant (2022) | 2211.10438 |
| ViT 양자화 | FQ-ViT (2021) | 2111.13824 |
| ViT 양자화 | PTQ4ViT (2021) | 2111.12293 |
| ViT 양자화 | RepQ-ViT (2022) | 2212.08254 |
| BEV 인식 | BEVFormer (2022) | 2203.17270 |
| BEV 인식 | BEVDet (2021) | 2112.11790 |
| BEV 인식 | PETR (2022) | 2203.05625 |

> arXiv는 `https://arxiv.org/abs/<ID>`로 접근. 참고문헌 인용 시 저자·연도·제목·URL을 모두 표기.

## 인용 포맷 (문서에 넣을 형태)
- 사이트: `[NVIDIA TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/)`
- 논문: `Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630`
