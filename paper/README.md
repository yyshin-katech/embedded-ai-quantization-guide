# 논문 19편 — 가이드 참고문헌 전문(全文)

`study_guide/01~10*.md`의 `## 8) 참고 사이트 & 참고문헌` 섹션, `study_guide/08_capstone.md` §10,
`logs/lv2_ptq_deep_dive.html` §6에서 arXiv 논문을 전부 추출해 내려받은 것이다.
**19편 / 44.7 MB / 331쪽** (2026-08-06 기준).

읽는 순서와 각 논문이 가이드의 어디에 걸리는지는 저장소 루트의
[`learning_resources.html`](../learning_resources.html)에 정리해 두었다.

## 검증 절차

내려받은 뒤 파일 하나하나에 대해:

1. **`%PDF` 매직바이트 확인** — HTTP 200을 주면서 오류 HTML을 뱉는 경우를 걸러내기 위해서다.
2. **쪽수 카운트** — `/Type /Page` 오브젝트를 세서 0쪽(잘린 파일)을 잡는다.
3. **제목 대조** — `arxiv.org/abs/<id>`의 `<meta name="citation_title">`과 비교.
   내가 가이드에서 옮겨 적은 제목 중 2건이 축약형이었고(LSQ, BEVFormer) 원문으로 고쳤다.

`19/19` 전부 통과했다.

## 재현

이 폴더를 지웠거나 저장소에 PDF가 없으면 다음으로 다시 만든다:

```bash
python3 fetch_papers.py     # arXiv에서 19편을 3.2초 간격으로 내려받아 검증
```

arXiv 권고에 따라 요청 간 3.2초를 둔다. 전체 약 1분.

## 목록


### A. 양자화 코어 (5편)

| 파일 | 논문 | 저자 | 인용 문서 | 크기 |
|---|---|---|---|---|
| [`1712.05877_jacob2018-integer-only-inference.pdf`](1712.05877_jacob2018-integer-only-inference.pdf) | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference<br>`arXiv:1712.05877` · CVPR 2018 (arXiv 2017-12) | Jacob 외 | `02_deployment_ladder`, `03_quantization_theory`, `05_tensorrt`, `06_multi_soc`, `08_capstone`, `10_pitfalls`, `logs/lv2_ptq_deep_dive.html` | 0.3 MB / 14p |
| [`1806.08342_krishnamoorthi2018-quantizing-cnn-whitepaper.pdf`](1806.08342_krishnamoorthi2018-quantizing-cnn-whitepaper.pdf) | Quantizing deep convolutional networks for efficient inference: A whitepaper<br>`arXiv:1806.08342` · arXiv 2018-06 | Krishnamoorthi | `logs/lv2_ptq_deep_dive.html` | 0.9 MB / 36p |
| [`2103.13630_gholami2021-quantization-survey.pdf`](2103.13630_gholami2021-quantization-survey.pdf) | A Survey of Quantization Methods for Efficient Neural Network Inference<br>`arXiv:2103.13630` · arXiv 2021-03 (Berkeley) | Gholami 외 | `02_deployment_ladder`, `03_quantization_theory`, `05_tensorrt`, `06_multi_soc`, `07_infrastructure`, `08_capstone`, `09_roadmap`, `10_pitfalls`, `logs/lv2_ptq_deep_dive.html` | 2.2 MB / 33p |
| [`2106.08295_nagel2021-quantization-white-paper.pdf`](2106.08295_nagel2021-quantization-white-paper.pdf) | A White Paper on Neural Network Quantization<br>`arXiv:2106.08295` · arXiv 2021-06 (Qualcomm AI Research) | Nagel 외 | `03_quantization_theory`, `05_tensorrt`, `06_multi_soc`, `07_infrastructure`, `08_capstone`, `09_roadmap`, `10_pitfalls`, `logs/lv2_ptq_deep_dive.html` | 1.8 MB / 27p |
| [`1902.08153_esser2020-learned-step-size-quantization.pdf`](1902.08153_esser2020-learned-step-size-quantization.pdf) | Learned Step Size Quantization<br>`arXiv:1902.08153` · ICLR 2020 | Esser 외 | `03_quantization_theory` | 0.4 MB / 12p |

### B. Transformer / ViT / LLM (4편)

| 파일 | 논문 | 저자 | 인용 문서 | 크기 |
|---|---|---|---|---|
| [`2211.10438_xiao2022-smoothquant.pdf`](2211.10438_xiao2022-smoothquant.pdf) | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models<br>`arXiv:2211.10438` · ICML 2023 (arXiv 2022-11, MIT HAN Lab) | Xiao 외 | `04_transformer_quantization`, `07_infrastructure`, `09_roadmap`, `10_pitfalls` | 5.4 MB / 13p |
| [`2111.13824_lin2021-fq-vit.pdf`](2111.13824_lin2021-fq-vit.pdf) | FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer<br>`arXiv:2111.13824` · IJCAI 2022 | Lin 외 | `04_transformer_quantization` | 3.1 MB / 10p |
| [`2111.12293_yuan2021-ptq4vit.pdf`](2111.12293_yuan2021-ptq4vit.pdf) | PTQ4ViT: Post-training quantization for vision transformers with twin uniform quantization<br>`arXiv:2111.12293` · ECCV 2022 | Yuan 외 | `04_transformer_quantization` | 0.7 MB / 20p |
| [`2212.08254_li2022-repq-vit.pdf`](2212.08254_li2022-repq-vit.pdf) | RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers<br>`arXiv:2212.08254` · ICCV 2023 | Li 외 | `04_transformer_quantization` | 0.7 MB / 10p |

### C. BEV / 3D 검출 (8편)

| 파일 | 논문 | 저자 | 인용 문서 | 크기 |
|---|---|---|---|---|
| [`2010.04159_zhu2020-deformable-detr.pdf`](2010.04159_zhu2020-deformable-detr.pdf) | Deformable DETR: Deformable Transformers for End-to-End Object Detection<br>`arXiv:2010.04159` · ICLR 2021 | Zhu 외 | `08_capstone` | 4.6 MB / 16p |
| [`2008.05711_philion2020-lift-splat-shoot.pdf`](2008.05711_philion2020-lift-splat-shoot.pdf) | Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D<br>`arXiv:2008.05711` · ECCV 2020 | Philion 외 | `08_capstone` | 5.3 MB / 17p |
| [`2112.11790_huang2021-bevdet.pdf`](2112.11790_huang2021-bevdet.pdf) | BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View<br>`arXiv:2112.11790` · arXiv 2021-12 | Huang 외 | `08_capstone` | 0.6 MB / 19p |
| [`2203.05625_liu2022-petr.pdf`](2203.05625_liu2022-petr.pdf) | PETR: Position Embedding Transformation for Multi-View 3D Object Detection<br>`arXiv:2203.05625` · ECCV 2022 | Liu 외 | `08_capstone` | 4.3 MB / 18p |
| [`2203.17270_li2022-bevformer.pdf`](2203.17270_li2022-bevformer.pdf) | BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers<br>`arXiv:2203.17270` · ECCV 2022 | Li 외 | `04_transformer_quantization`, `08_capstone` | 4.7 MB / 20p |
| [`2403.10913_xu2024-defa.pdf`](2403.10913_xu2024-defa.pdf) | DEFA: Efficient Deformable Attention Acceleration via Pruning-Assisted Grid-Sampling and Multi-Scale Parallel Processing<br>`arXiv:2403.10913` · arXiv 2024-03 | Xu 외 | `04_transformer_quantization` | 1.0 MB / 6p |
| [`2505.14022_huang2025-msda-on-npu.pdf`](2505.14022_huang2025-msda-on-npu.pdf) | Towards Efficient Multi-Scale Deformable Attention on NPU<br>`arXiv:2505.14022` · arXiv 2025-05 | Huang 외 | `04_transformer_quantization` | 1.4 MB / 10p |
| [`2502.15488_fq-petr-2025.pdf`](2502.15488_fq-petr-2025.pdf) | FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection<br>`arXiv:2502.15488` · arXiv 2025-02 | Yu 외 | `04_transformer_quantization` | 1.4 MB / 14p |

### D. 데이터셋 / 런타임 (2편)

| 파일 | 논문 | 저자 | 인용 문서 | 크기 |
|---|---|---|---|---|
| [`1903.11027_caesar2020-nuscenes.pdf`](1903.11027_caesar2020-nuscenes.pdf) | nuScenes: A multimodal dataset for autonomous driving<br>`arXiv:1903.11027` · CVPR 2020 | Caesar 외 | `01_environment_setup`, `08_capstone` | 4.8 MB / 16p |
| [`2605.08195_nachin2026-executorch.pdf`](2605.08195_nachin2026-executorch.pdf) | ExecuTorch -- A Unified PyTorch Solution to Run AI Models On-Device<br>`arXiv:2605.08195` · arXiv 2026-05 | Nachin 외 | `02_deployment_ladder` | 1.1 MB / 20p |

## 무결성

| arXiv | MD5 | 바이트 |
|---|---|---|
| `1712.05877` | `ae3ba24db141882575515059abfca2fe` | 264,955 |
| `1806.08342` | `c7853f8a7ff2ad75c9e4935fd5c0da0d` | 878,840 |
| `2103.13630` | `8a4891bf85dd0e15a853e17be63783fd` | 2,234,098 |
| `2106.08295` | `e98475e49fa66a2e65a7c420af5fd9e2` | 1,784,263 |
| `1902.08153` | `601fe34a0b921f764bd1c091b25b024b` | 426,971 |
| `2211.10438` | `2bece97d2e7003e039ee51f4fdae3cab` | 5,378,996 |
| `2111.13824` | `c61f43285bbda2d02834ff824fc30c6f` | 3,082,351 |
| `2111.12293` | `97648deec961dc37ca67e3c772307c95` | 744,541 |
| `2212.08254` | `11f14bf06d60b45299d094aef0f61bfa` | 709,927 |
| `2010.04159` | `64f1d7d04a0cfc8963195a95a67c1928` | 4,566,251 |
| `2008.05711` | `f1a98602a4d2e4c723a02cf6490c7bfd` | 5,319,285 |
| `2112.11790` | `cdcc6abcc7d7ec15f558d8cd2b926559` | 583,501 |
| `2203.05625` | `bc2887487e7c2da4f81398be19c3e642` | 4,345,274 |
| `2203.17270` | `8f89a8eed6222a9831d79559cdeca028` | 4,739,748 |
| `2403.10913` | `f64222b58c0a32598b5de6db83c11b0f` | 1,030,798 |
| `2505.14022` | `c270b8eab1b744e245940b08fd9a2391` | 1,378,512 |
| `2502.15488` | `d8a9cbcdd3d8d33dfd96fec18f541c3a` | 1,414,877 |
| `1903.11027` | `a318b15ad9dcbfa39b05ecf423fe92ea` | 4,788,382 |
| `2605.08195` | `38fdc0b3edc0ad83b64ae9f6a631c29e` | 1,064,619 |

## 라이선스

arXiv PDF의 저작권은 각 저자에게 있고, 배포 조건은 논문별 라이선스(arXiv 비독점 라이선스 또는
CC BY 계열)를 따른다. 이 폴더는 **학습용 로컬 사본**이다. 이 저장소는 public이므로,
PDF를 커밋하는 대신 `.gitignore`에 넣고 `fetch_papers.py`만 커밋하는 편이 안전하다 —
스크립트만 있으면 누구나 동일한 파일을 arXiv에서 직접 얻는다.

