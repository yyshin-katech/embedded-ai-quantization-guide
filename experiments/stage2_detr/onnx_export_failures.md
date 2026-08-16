# ONNX Export & Quantization Failure Log — DETR (`facebook/detr-resnet-50`)

> 환경(본 실습 실측, 2026-08-16): Ubuntu 22.04 · RTX 3080 · driver 595.84 · CUDA 12.8 · torch 2.11.0+cu128
> transformers 5.15.0 · timm 1.0.28 · onnx 1.18.0 (IR 11) · onnxruntime-gpu 1.23.2 · venv `~/emb-ai`
> 목적: "무엇이 왜 깨졌고 어떻게 우회했는가" = 재사용 가능한 design rules
> 재현: `experiments/stage2_detr/s2_02_export_try.py` … `s2_10_*.py`, 원 로그 `_workspace/stage2/*.log`
> 상위 서사·해석: [2단계 가이드 §4](../../study_guide/04_transformer_quantization.md) · [실측 리포트](../../logs/stage2_detr_quantization_report.html)

이 파일은 2단계 가이드 §7 템플릿을 **DETR 실습 실측으로 채운 실물**이다. 로그는 요약하지 않고 통째로 붙인다.

---

## 요약 표

| # | 단계 | 증상(로그 핵심) | 원인 | 우회/해결 | 상태 |
|---|------|-----------------|------|-----------|------|
| 0 | export(opset11, **legacy** `dynamo=False`) | `aten::scaled_dot_product_attention ... opset 11 is not supported`(v14+) | SDPA symbolic이 opset14부터 | `opset_version=17, dynamo=False` | ✅ DETR 실측 |
| 0b | export(**기본** `dynamo=True`) | 실패 아님. opset이 **18로 고정**되고 external data(`.onnx.data`) 분리 | torch 2.11 기본이 dynamo | 단일파일·opset 고정 원하면 `dynamo=False` | ✅ DETR 실측 |
| 0c | dynamo → opset17 다운컨버트 | `No Adapter To Version $17 for Resize` | version_converter에 Resize 어댑터 부재 | opset 18 그대로 두거나 legacy 사용 | ✅ DETR 실측 |
| 0d | `from_pretrained` (timm 미설치) | `ImportError: TimmBackbone requires the timm library ...` | DETR backbone이 `TimmBackbone` | `pip install timm` | ✅ DETR 실측 |
| 6 | PTQ(all-INT8, per-channel QDQ) | **mAP 0.4207 → 0.2402 폭락(−42.9%)**, 작은 객체 −77% | per-tensor **activation** 양자화가 망 전체에 분산 손상 | op 제외 mixed 실패(+0.36 only); SmoothQuant 필요 | ✅ DETR 실측 |
| 6b | PTQ 회복 — attn matmul FP | mixed 0.2438 (int8 대비 **+0.0036뿐**) | 손상이 "특정 op"가 아님 | op granularity로는 회복 불가 | ✅ DETR 실측 |
| 6c | PTQ 회복 — Percentile 캘리브 | ①inhomogeneous shape ②OOM(31GB) ③`range [inf,inf] not finite` **3중 실패** | 동적 shape + 고해상 activation + attn mask fill | MinMax만 생존 | ✅ DETR 실측 |
| 1 | export(opset11, legacy) | `aten::grid_sampler ... opset 11 is not supported`(v16+) | GridSample 표준화가 opset16 | `opset_version≥16` | ⏳ BEVFormer(DETR엔 grid_sample 없음) |
| 2 | export(opset17, 5D grid) | `GridSample with 5D volumetric input` 미지원 | opset16/17은 4D만 | 5D→4D 분해 / plugin | ⏳ BEVFormer |
| 8 | TRT build(GridSample 5D) | rank-4만 지원(issue #3890) | TRT native 4D 한정 | 5D 분해 / plugin | ⏳ BEVFormer |
| 9 | TIDL compile | `CHECK failed (index)<(current_size_)` | self-attn QDQ 미지원 | attn FP 유지 | ⏳ 하드웨어 미보유 |

`⏳` = DETR 실습 범위 밖(BEVFormer/특정 HW). DETR로 검증 가능한 항목은 전부 `✅`.

---

## 상세 로그 (케이스별)

### Case 0 — SDPA opset 미지원 (DETR 첫 블로커)

- **시도**: `torch.onnx.export(model, (pv,), "detr_legacy_op11.onnx", opset_version=11, dynamo=False)`
  (스크립트 `s2_02_export_try.py`, 로그 `_workspace/stage2/s2_02_export_try.log`)
- **전체 에러 로그(원문)**:
  ```
  >>> RESULT[detr_legacy_op11]: FAIL :: UnsupportedOperatorError: Exporting the
  operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not
  supported. Support for this operator was added in version 14, try exporting with
  this version
  ```
  트레이스 상 SDPA 노드(잘라내지 말 것 — 검색 앵커):
  ```
  aten::scaled_dot_product_attention(%query_states.1, %key_states.1, %value_states.1,
    %attention_mask.5, ...), scope: ... DetrEncoderLayer::layers.0/DetrSelfAttention::self_attn
    # transformers/integrations/sdpa_attention.py:154:0
  ```
- **원인 분석**: 순수 opset 문제. DETR self/cross-attention 18개가 전부 `aten::scaled_dot_product_attention`으로 트레이스되고, 그 symbolic은 opset **14**부터 존재. **grid_sampler가 아니다 — DETR엔 grid_sample이 없다**(그래프 Grid op count = 0). "Transformer export 실패 = grid_sampler"라는 통념이 DETR에선 틀림. 첫 블로커는 SDPA.
- **우회**: `opset_version=17, dynamo=False` → 성공(단일 파일 170.4MB, IR 8). 또는 `dynamo=True`(성공하되 opset 18·external data, Case 0b).
- **재현성/영향**: opset 17 legacy export 성공. 정확도 영향 없음(export는 무손실).

### Case 0b — torch 2.11 기본 `dynamo=True`가 opset·파일형태를 바꿈

- **핵심**: torch 2.11에서 `torch.onnx.export`의 **기본값이 `dynamo=True`**다. `dynamo=`를 생략하면 "낮은 opset으로 legacy 실패를 보려던" 시도조차 dynamo 경로로 돌아 **조용히 성공**해 버려, Case 0의 실패를 재현할 수 없다.
- **실측 4경로 대조**(`s2_02_export_try.py`):

  | 시도 | kwargs | 결과 | 산출물 |
  |------|--------|------|--------|
  | `detr_legacy_op11` | `opset=11, dynamo=False` | **FAIL** (SDPA, Case 0) | — |
  | `detr_legacy_op17` | `opset=17, dynamo=False` | OK | 단일 **170.4 MB** (IR 8) |
  | `detr_dynamo_op17` | `opset=17, dynamo=True` | OK (opset→**18**) | **2.2 MB** + `.onnx.data`(166.5MB) |
  | `detr_default_op11` | `opset=11` (dynamo 생략=True) | OK (opset→**18**) | **2.2 MB** + `.onnx.data` |

- **dynamo가 opset 18을 강제하는 경고(원문)**:
  ```
  W ... Setting ONNX exporter to use operator set version 18 because the requested
  opset_version 17 is a lower version than we have implementations for. Automatic
  version conversion will be performed ... If version conversion is unsuccessful, the
  opset version of the exported model will be kept at 18.
  ```
- **교훈 3가지**:
  1. legacy 실패를 재현하려면 **반드시 `dynamo=False`를 명시**한다(생략 = dynamo).
  2. dynamo 경로는 **요청 opset을 무시**하고 18로 올린다(다운컨버트 실패, Case 0c).
  3. dynamo는 weight를 **external data로 분리**(main 2.2MB + `.onnx.data`)한다. 단일 파일이 필요하면 legacy.

### Case 0c — dynamo의 opset17 다운컨버트 실패

- **전체 에러 로그(원문)**:
  ```
  RuntimeError: /github/workspace/onnx/version_converter/BaseConverter.h:65:
  adapter_lookup: Assertion `false` failed: No Adapter To Version $17 for Resize
  ```
- **원인**: dynamo가 opset 18로 export한 뒤 요청값 17로 되돌리려 version_converter를 돌리는데, `Resize`(backbone interpolate) 어댑터가 없어 실패. 결국 **opset 18로 유지**하고 export 자체는 성공(경고만).
- **우회**: 애초에 `opset_version>=18`로 요청하거나, opset을 통제해야 하면 legacy(`dynamo=False`)를 쓴다.

### Case 0d — timm 미설치 시 로드 실패 (환경 함정)

- **전체 에러 로그(원문)**:
  ```
  ImportError: TimmBackbone requires the timm library but it was not found in your
  environment. Please install it and rerun ...
  ```
- **원인**: `facebook/detr-resnet-50`의 backbone은 `TimmBackbone`이다. config에 `backbone_config`가 있어 네이티브 백본처럼 보여도(transformers 5.15에선 `use_timm_backbone` 키가 config에서 사라짐) 실제 인스턴스는 timm 백본이라, timm 없으면 `from_pretrained`가 곧바로 죽는다.
- **우회**: `pip install timm`(실측 1.0.28). 이 함정은 §3 설치 단계에 반영.

### Case 6 — 전부 INT8에서 mAP 폭락 (핵심 발견, COCO val2017 전량 5,000장)

- **시도**: `quantize_static(..., quant_format=QDQ, activation_type=QInt8, weight_type=QInt8, per_channel=True)` — Conv 54 + MatMul 136 = **190 노드 전부**. 캘리브 = COCO val 앞 100장, MinMax.
  (`s2_06_quantize_dynamic.py` / eval `s2_07_coco_eval.py`, pycocotools `bbox`)
- **관측(COCO val2017 5,000장, CUDA EP)**:

  | 구성 | mAP | mAP50 | mAP75 | mAP_s | mAP_m | mAP_l |
  |------|-----|-------|-------|-------|-------|-------|
  | FP32 | **0.4207** | 0.6231 | 0.4421 | 0.2131 | 0.4595 | 0.6102 |
  | all-INT8 | **0.2402** | 0.4708 | 0.2179 | 0.0487 | 0.2329 | 0.4500 |

  FP32 0.4207이 공개값 **42.0**과 일치 → 계측 파이프라인 신뢰. INT8은 **−0.1805(−42.9%)**, 작은 객체 mAP_s는 0.2131→0.0487(**−77%**)로 가장 크게 무너짐.
- **원인 분석**: 소수 "문제 op"가 아니라 **per-tensor activation 양자화가 망 전체에 분산**된 손상. 절제(Case 6 하위)로 확정.

### Case 6b — mixed(attention score matmul FP)로 회복 시도 → 실패

- **시도**: `nodes_to_exclude`로 activation×activation matmul 36개(`/model/*/self_attn/MatMul(_1)`, `/model/*/encoder_attn/MatMul(_1)`)를 FP로 남기고 나머지 INT8. (`s2_08_quantize_mixed.py`)
- **관측**: mixed mAP **0.2438** — all-INT8(0.2402) 대비 **+0.0036뿐**. 폭락(−0.1805)의 2%도 못 되찾음.
- **절제(범인 위치 분리, `s2_09_quantize_ablation.py`)**:

  | 구성 | mAP | 해석 |
  |------|-----|------|
  | backbone FP · transformer INT8 (`bb_fp`) | 0.2391 | transformer만 INT8인데도 **거의 full 폭락** |
  | transformer FP · backbone INT8 (`tf_fp`) | 0.2653 | backbone만 INT8이어도 크게 폭락 |

  두 절반이 각자 폭락 대부분을 만든다(**sub-additive**, 단일 범인 op 집합 없음).
- **왜 doc 초안의 4개 제외 패턴 중 3개가 no-op이었나**: DETR 그래프엔 **GELU가 없고**(Gelu 0, Erf 0), Softmax(18)·LayerNorm(31)은 존재하지만 **ORT QDQ가 애초에 Conv/MatMul/Gemm만 양자화**하므로 제외할 대상이 아니다. 실제로 효과가 있는 제외는 attention MatMul뿐이고 그마저 +0.36.
- **결론**: op 선택(granularity가 op 단위)으로는 회복 불가. **activation 양자화 방식 자체**(per-tensor→per-token / SmoothQuant §2.2)가 진짜 레버.

### Case 6c — Percentile 캘리브로 회복 시도 → 3중 실패 (MinMax만 생존)

- **가설**: outlier를 clip하면 per-tensor scale이 본체를 되찾아 회복될 것(§2.1.1). Percentile 99.99/99.9 캘리브로 재양자화. (`s2_10_quantize_percentile.py`)
- **실패 3형태(전부 실측)**:
  1. **동적 shape** 모델 → 이미지마다 activation shape가 달라 ORT HistogramCollector가 stack하려다 `ValueError: setting an array element with a sequence ... inhomogeneous shape`.
  2. 고정 shape(1066×800, `do_resize=False`)로 (1)을 우회하면 이미지당 **~3.6GB** → N=8에서 **OOM SIGKILL**(peak 30.9GB/31GB, exit 137).
  3. N=4로 줄이면(peak 19GB) `numpy.histogram`이 attention mask fill 값에서 inf를 만나 `ValueError: autodetected range of [inf, inf] is not finite`.
- **결론**: stock ORT의 Percentile/Entropy(HistogramCollector)는 **DETR 같은 동적·고해상 detection 그래프에서 사실상 못 돈다**. 스칼라 min/max인 **MinMax만 생존** → 위 모든 mAP는 MinMax 기준. 회복은 캘리브 방법이 아니라 SmoothQuant/per-token으로 가야 함.

---

## Design Rules (이 로그에서 도출)

- [x] **"Transformer export 실패 = grid_sampler"는 모델마다 다르다.** DETR의 첫 블로커는 **SDPA**(opset14+). grid_sample은 BEVFormer/deformable 계열의 얘기다.
- [x] torch ≥ 2.9/2.11에서 `torch.onnx.export`는 **기본이 `dynamo=True`**. 재현 가능한 실습·opset 통제·단일 파일이 필요하면 **`dynamo=False`를 명시**한다.
- [x] dynamo 경로는 **요청 opset을 무시**하고 18로 올리며(다운컨버트는 Resize에서 실패) weight를 **external data로 분리**한다.
- [x] `facebook/detr-resnet-50`은 **timm 필수**(TimmBackbone). 설치 목록에 반드시 넣는다.
- [x] **"특정 op만 FP로 빼면 회복"은 DETR에서 반증됨**(+0.36 mAP). 손상이 분산돼 있으면 op granularity가 아니라 **activation 양자화 방식**(per-tensor→per-token/SmoothQuant)을 바꾼다.
- [x] stock ORT **Percentile/Entropy 캘리브는 동적·고해상 detection 그래프에서 못 돈다**(shape·OOM·inf 3중). 회복 레버로 기대하지 말 것.
- [x] detection은 classification보다 양자화에 훨씬 취약하다(1단계 ResNet18 −0.1%p vs DETR −42.9%). **작은 객체 AP가 먼저·가장 크게 무너진다**(−77%).
- [ ] (BEVFormer, 다음 과제) BEV 모델은 처음부터 **opset ≥ 16**, 입력 **shape 고정** export. `grid_sample` 5D는 **TensorRT 4D 한정** → 분해/plugin 전제.
- [ ] (BEVFormer) Deformable Attention은 **GPU=plugin / NPU=구조변경**을 사전 결정. grid_sample/deformable/scatter/gather/dynamic shape를 NPU 타깃 전 사전 점검.

> 💡 에러 로그는 요약하지 말고 통째로. 6개월 뒤 다른 모델에서 같은 에러를 만나면 그 원문이 검색 앵커가 된다.
