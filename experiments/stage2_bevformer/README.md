# experiments/stage2_bevformer — BEVFormer grid_sample / Deformable Attention 실기 검증

§4.6(`study_guide/04_transformer_quantization.md`)의 BEVFormer-tiny 양자화 단정을 실측으로 검증한 산출물.
DETR( [`../stage2_detr`](../stage2_detr/) )의 자매 실습.

## 왜 2-tier인가
BEVFormer 실모델은 **정본 venv(`~/emb-ai`, torch 2.11)에서 안 돈다** — mmcv-full 1.x / mmdet3d는
torch ≤1.13 + CUDA ≤11.7에서 CUDA op 컴파일이 전제. 그래서:

- **Tier A** = §4.6이 가르치는 단정 대부분(grid_sample opset 경계, 5D 볼류메트릭, MSDeformAttn 분해,
  ORT/TRT 런타임)을 **op 단위 최소 repro**로 정본 venv에서 검증 → **완료(b01~b05)**.
- **Tier B** = 레거시 venv로 BEVFormer-tiny 실모델 export→INT8 PTQ→nuScenes mAP → 상태는
  [`onnx_export_failures.md`](onnx_export_failures.md) 하단 "Tier B 상태" 참조.

결과·로그 원문·design rules는 전부 [`onnx_export_failures.md`](onnx_export_failures.md)에 있다(이 README는 실행법).

## 환경 (Tier A)
```
Ubuntu 22.04 · RTX 3080 10GB · CUDA 12.8 · torch 2.11.0+cu128
onnx 1.18.0(IR 11) · onnxruntime-gpu 1.23.2 · TensorRT 10.16.1.11
source ~/emb-ai/bin/activate
```

## 실행 순서 (Tier A)
```bash
source ~/emb-ai/bin/activate
python b01_grid_sample_4d_opset.py     # 4D grid_sample opset 경계(정확히 16) + dynamo 상향
python b02_grid_sample_5d_export.py    # 5D 볼류메트릭: legacy 16/17/18 실패·20 성공, dynamo out-of-spec
python b03_grid_sample_runtime_ort.py  # ORT: 4D=CUDA OK / 5D=CUDA 커널부재→CPU 조용한 폴백  (b02 이후)
python b04_grid_sample_trt_parse.py    # TensorRT: 4D 파싱 OK / 5D rank-4 단언 실패             (b02 이후)
python b05_msdeformattn_decompose.py   # MSDeformAttn 분해: 1 op→140 노드, GridSample×num_levels
```
> b03/b04는 b02가 남긴 `_gs5d_legacy_op20.onnx`를 재사용한다(먼저 b02 실행). `_*.onnx`는 임시라 커밋 제외.

## 실행 순서 (Tier B — 레거시 venv `~/bevf-legacy`)
```bash
# 레거시 venv: torch 1.13.1+cu117 · mmcv-full 1.7.1(프리빌트 cu117/torch1.13 휠) · numpy 1.23.5 · onnx 1.14.1
~/bevf-legacy/bin/python b06_mmcv_real_op.py   # 실제 mmcv CUDA op 로드/실행 + 실모듈 export(CPU vs CUDA)
```
> b06은 **정본 venv가 아니라 `~/bevf-legacy`** 로 실행한다(torch 1.13 필요). B1(CUDA op이 CUDA 12.8에서 사는가) +
> B2(실모듈 export: CPU=표준분해 / CUDA=silent-wrong 상수 baked)를 검증. 결과는 `onnx_export_failures.md` Tier B 절.

## 파일
| 파일 | 내용 | venv |
|------|------|------|
| `b01_grid_sample_4d_opset.py` + `_result.json` | 4D grid_sample opset 스윕(legacy 9~20 + dynamo) | `~/emb-ai` |
| `b02_grid_sample_5d_export.py` + `_result.json` | 5D 볼류메트릭 export(legacy/dynamo) | `~/emb-ai` |
| `b03_grid_sample_runtime_ort.py` | ORT 4D/5D CUDA vs CPU 런타임(로그만 출력) | `~/emb-ai` |
| `b04_grid_sample_trt_parse.py` + `_result.json` | TensorRT OnnxParser 4D/5D | `~/emb-ai` |
| `b05_msdeformattn_decompose.py` + `_result.json` | MSDeformAttn 표준 op 분해(mmcv 폴백 복사본) | `~/emb-ai` |
| `b06_mmcv_real_op.py` + `_result.json` | **실제 mmcv CUDA op** 로드/실행 + 실모듈 export(CPU/CUDA) | `~/bevf-legacy` |
| `onnx_export_failures.md` | **결과·로그 원문·design rules(핵심 산출물)** | — |

## 핵심 결과 (한 줄)
§4.6 grid_sample/deformable 단정 **반전 0건** — 초안이 전부 맞았고, dynamo 거동 2건을 정밀화하고
인용을 로그 원문(ORT `CUDA kernel not found ... GridSample`, TRT `addGridSample ... nbDims == 4`)으로 강화.
