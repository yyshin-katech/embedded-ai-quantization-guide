# ONNX Export & Quantization Failure Log — BEVFormer / grid_sample & Deformable Attention

> 환경(본 실습 실측): Ubuntu 22.04 · **RTX 3080 10GB** · driver 595.84 · CUDA 12.8 · torch **2.11.0+cu128**
>       onnx 1.18.0 (IR 11) · onnxruntime-gpu **1.23.2** · **TensorRT 10.16.1.11** · onnxscript 0.7.1
> 목적: §4.6의 grid_sample/Deformable Attention "지뢰" 단정을 **op 단위 최소 재현**으로 검증한다
>       (= 재사용 가능한 design rules). DETR( [`../stage2_detr/onnx_export_failures.md`](../stage2_detr/onnx_export_failures.md) )의 자매 로그.

## 실습 구조 — 왜 op 단위부터인가 (2-tier)

BEVFormer 실모델은 정본 venv(torch 2.11)에서 **돌지 않는다** — mmcv-full 1.x / mmdet3d 계열은
torch ≤1.13 + CUDA ≤11.7에 CUDA op 컴파일이 전제라, 별도 레거시 venv가 필요하다(2026-08-17 선행
점검). 그래서 검증을 두 단계로 나눴다.

- **Tier A (이 로그의 §요약표 #1~5)** — §4.6이 가르치는 단정은 대부분 **개별 op의 export 거동**이다
  (grid_sample opset 경계, 5D 볼류메트릭, MSDeformAttn 분해, TRT rank-4). 이것들은 BEVFormer 전체
  모델·nuScenes 없이 **정본 venv에서 최소 repro**로 그대로 실증된다. 지뢰를 격리해 밟아본 것.
- **Tier B (§Tier B 상태)** — 레거시 venv로 BEVFormer-tiny를 실제 export→INT8 PTQ→nuScenes mAP까지.

이 파일의 **Tier A는 확정**이다(아래 전부 실측). Tier B는 진행 상태를 이 파일 하단에 계속 갱신한다.

---

## 요약 표 (Tier A — 전부 실측 확정)

| # | 단계 | 증상(로그 핵심) | 원인 | 우회/해결 | 상태 |
|---|------|-----------------|------|-----------|------|
| 1 | export grid_sample **4D**, legacy opset≤15 | `aten::grid_sampler ... opset N is not supported. Support ... added in version 16` | ONNX `GridSample` 표준화가 opset 16 | `opset_version≥16`(legacy) | ✅ 실측(b01) |
| 1b | export grid_sample 4D, **dynamo(기본)** req opset 11 | (에러 아님) 요청 11 무시 → **emit 18** | torch 2.11 기본 dynamo는 저-opset 요청을 상향 | opset 통제하려면 `dynamo=False` | ✅ 실측(b01) |
| 2 | export grid_sample **5D**, legacy opset 16/17/18 | `Unsupported: ONNX export of operator GridSample with 5D volumetric input` | opset 16/17 GridSample은 4D 전용 | opset **20**(legacy) / 분해 / plugin | ✅ 실측(b02) |
| 2b | export grid_sample 5D, **dynamo** req opset 17 | (에러 아님) opset **17**에 5D GridSample을 그대로 emit(**out-of-spec**) | dynamo 경로가 5D를 검증 없이 통과 | 다운스트림에서 터지므로 신뢰 말 것 | ✅ 실측(b02) |
| 3 | run grid_sample **5D**(opset20), **ORT CUDA EP** | `[I:onnxruntime ... GetCapability] CUDA kernel not found in registries for Op type: GridSample` → **CPU로 조용히 폴백** | ORT 1.23.2 CUDA EP에 5D GridSample 커널 없음(1.27+에 추가) | 4D 분해 / plugin / ORT 업그레이드 | ✅ 실측(b03) |
| 4 | build grid_sample **5D**(opset20), **TensorRT** | `addGridSample: Error Code 3 ... condition: input.getDimensions().nbDims == 4` | TRT native GridSample은 rank-4 전용(issue #3890) | 5D→4D 분해 / GridSample plugin | ✅ 실측(b04) |
| 5 | export **MSDeformAttn 표준 op 분해**(mmcv 폴백) | (성공) 1개 논리 op → **140 노드**, GridSample **4개(=num_levels)** | 레벨마다 grid_sample 1회 | GPU=plugin(4.6.2-b), NPU=구조변경 | ✅ 실측(b05) |

**한 줄 결론:** §4.6의 grid_sample/deformable 단정은 **반전 0건** — 초안이 전부 맞았다. 다만 두 곳을
**정밀화**했고(1b·2b, 아래), 인용을 **로그 원문 수준**으로 끌어올렸다(3의 ORT 로그, 4의 TRT API 단언).

---

## 상세 로그 (케이스별)

### Case 1 — grid_sample 4D opset 경계는 **정확히 16** (b01)
- **시도**: `F.grid_sample(feat[1,8,16,16], grid[1,10,10,2], mode='bilinear', padding_mode='zeros', align_corners=False)` 를
  `torch.onnx.export(..., opset_version=K, dynamo=False)` 로 K=9,11,13,14,15,16,17,18,20 스윕.
- **결과**: K∈{9,11,13,14,15} 전부 실패, K∈{16,17,18,20} 성공(단일 `GridSample` 노드, emit=요청).
- **에러 원문**(잘라내지 말 것):
  `torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::grid_sampler' to ONNX opset version 11 is not supported. Support for this operator was added in version 16, try exporting with this version`
- **분석**: ONNX `GridSample` 표준화가 opset 16이고 torch symbolic도 정확히 16부터. 경계는 15|16 사이 **딱 하나**.
- **우회**: legacy에서 `opset_version≥16`. (초안 §5.1 "opset 16/17=4D ✅, ≤12 ❌" 및 troubleshooting `opset 11 not supported`와 **완전 일치**.)

### Case 1b — dynamo(torch 2.11 기본)는 요청 opset을 존중하지 않는다 (b01)
- **시도**: 같은 4D 모듈을 `dynamo=True`(기본), `opset_version=11` / `=17` 로 export.
- **결과**: req **11 → emit 18**(자동 상향), req 17 → emit 17. 즉 저-opset 요청은 조용히 18로 끌어올린다.
- **함의**: Case 1의 `opset 11 not supported` 에러는 **legacy(`dynamo=False`) 전용 현상**이다. 정본 torch 2.11의
  **기본 경로(dynamo=True)에선 그 에러가 아예 안 뜨고**, 대신 opset을 통제할 수 없다. → **특정 opset이 필요하면 `dynamo=False`.**
  (DETR에서 SDPA 첫 블로커가 legacy 전용이었던 것과 **같은 구조** — [`../stage2_detr`](../stage2_detr/onnx_export_failures.md) Case 0.)

### Case 2 — grid_sample 5D는 opset 20부터, 16/17/18은 4D 전용 (b02)
- **시도**: 5D 볼류메트릭 `F.grid_sample(feat[1,4,6,8,8], grid[1,5,8,8,3])` 를 legacy opset 16/17/18/20 로 export.
- **결과**: 16/17/18 실패, **20 성공**(GridSample 노드).
- **에러 원문**:
  `torch.onnx.errors.OnnxExporterError: Unsupported: ONNX export of operator GridSample with 5D volumetric input. Please feel free to request support or submit a pull request on PyTorch GitHub`
- **분석**: ONNX 표준이 5D GridSample을 opset **20**에서 추가. torch 2.11 legacy가 이를 반영해 20에서만 통과.
- **우회**: opset 20 export(단, 런타임 지원은 별개 — Case 3/4). 또는 5D→4D 분해 / plugin.

### Case 2b — dynamo는 5D를 opset 17에 **out-of-spec**으로 흘려보낸다 (b02)
- **결과**: `dynamo=True, opset_version=17` 로 5D를 export하면 **에러 없이** opset 17 그래프에 GridSample(5D 입력)을 emit한다.
- **함의**: opset 17 GridSample 스펙엔 5D가 없다 → **표준 위반 모델**. export가 "성공"해도 다운스트림(런타임/파서)에서
  터진다. **export 성공 ≠ 유효 모델.** BEV 모델은 반드시 `onnx.checker` + 실런타임 로드로 검증할 것.

### Case 3 — ORT 1.23.2: 5D GridSample은 CUDA 커널이 없어 **조용히 CPU로 폴백** (b03)
- **시도**: opset20 5D 모델(표준 유효, `onnx.checker` PASS)을 ORT CUDA EP로 로드·추론(`log_severity_level=1`).
- **관측 로그 원문**:
  `2026-08-17 02:10:50 [I:onnxruntime:, cuda_execution_provider.cc:2771 GetCapability] CUDA kernel not found in registries for Op type: GridSample node name: /GridSample`
  → 추론은 **에러 없이 성공**(out shape `(1,4,5,8,8)`), 그러나 GridSample 노드는 CPU에서 실행됨(속도 저하).
- **대조**: **4D** GridSample은 이 로그가 **안 뜬다** → CUDA EP가 정상 수용(GPU 실행). CPU-only도 5D를 지원(ORT 1.23.2 CPU 커널엔 5D 있음).
- **분석**: CUDA EP의 볼류메트릭(3D) GridSample 커널은 ORT **1.27**에서 추가. 정본 1.23.2엔 없어 `GetCapability`가
  노드를 거절 → ORT가 자동 CPU 폴백(에러 없음). **초안 §4.6.1의 문구·로그와 정확히 일치.**
- **함정**: 이 폴백은 `INFO` 레벨 로그 한 줄이 전부라 **놓치기 쉽다**. 5D BEV 샘플링이 "되는 줄 알았는데 느린" 전형.

### Case 4 — TensorRT 10.16: native GridSample은 rank-4 전용, 5D는 파싱 실패 (b04)
- **시도**: `tensorrt.OnnxParser`(파이썬 API, trtexec 미포함)로 4D/5D onnx 파싱.
- **결과**: **4D parse ok=True**, **5D parse ok=False**(legacy·dynamo opset20 둘 다).
- **에러 원문**:
  `[TRT] [E] INetworkDefinition::addGridSample: Error Code 3: API Usage Error (Parameter check failed, condition: input.getDimensions().nbDims == 4. In addGridSample at /_src/optimizer/api/network.cpp:1803)`
- **분석**: TRT importer가 GridSample을 `IGridSampleLayer`로 매핑하는데 이 레이어가 **`nbDims == 4`를 하드 단언**한다.
  초안의 "issue #3890(rank-4만)"을 **API 단언 수준**으로 확정 — 5D는 native로 절대 안 들어간다.
- **부수 관찰**: torch가 bilinear를 GridSample attribute `mode: "linear"`로 export한다(4D/5D 공통).
- **우회**: 5D→4D 분해(레벨별 4D 샘플링) 또는 GridSample plugin(`DerryHub/BEVFormer_tensorrt`).

### Case 5 — MSDeformAttn 표준 op 분해 = mmcv 폴백, grid_sample×레벨 + 노드 폭증 (b05)
- **시도**: mmcv `multi_scale_deformable_attn_pytorch`(순수 PyTorch 폴백)를 **원본 그대로** 옮겨,
  bs1·heads8·embed32·**levels4**·points4·queries100, spatial `[(8,8),(4,4),(2,2),(1,1)]`로 export.
- **결과**: forward 정상(out `(1,100,256)` = [bs, queries, heads×embed]). opset **16/17 성공**, opset 13 실패(grid_sampler<16).
  - **노드 140개**, `GridSample` **4개 = num_levels**(레벨당 1회). 분포: Constant 66 · Reshape 15 · Slice 13 · Concat 10 · Transpose 10 · Shape 9 · Gather 4 · **GridSample 4** · Unsqueeze 4 · Mul 2 …
- **분석**: 커스텀 CUDA 커널 1개(논리적 1 op)를 표준 op로 다시 쓰면 **1 → 140 노드**로 펼쳐지고, 핵심인
  grid_sample이 **레벨 수만큼** 남는다. 초안 §4.6.2(a) "mmcv 폴백이 곧 분해 / opset 16+에서 표준 그래프 / grid_sample 다회 / 노드 폭증"을 **정량 확인**.
- **함의**: 이 분해본은 (1) grid_sample이 4개 남아 **grid_sample 못 받는 NPU엔 여전히 불가**, (2) 노드 폭증으로 느림 →
  GPU 타깃이면 plugin(b), NPU 타깃이면 구조 변경. **분해는 "export가 되게" 하지만 "빠르게/이식되게"는 아니다.**

---

## 초안 대비 정정/정밀화

- **반전(사실 뒤집힘): 0건.** §4.6의 grid_sample/deformable 단정은 전부 실측과 일치.
- **정밀화 2건**:
  - (1b) `grid_sampler opset N not supported` 에러는 **legacy 전용**. torch 2.11 기본 dynamo에선 안 뜨고 opset이 통제 불가로 상향된다.
  - (2b) dynamo는 5D를 **opset 17에 out-of-spec으로** 흘려보낸다 → export 성공이 유효 모델을 보장하지 않음.
- **인용 강화 2건**: (3) ORT `CUDA kernel not found in registries ... GridSample` 로그 원문, (4) TRT `addGridSample ... nbDims == 4` API 단언.

## Design Rules (이 로그에서 도출)
- [ ] BEV 모델 export는 **opset ≥ 16**(4D grid_sample), 5D 볼류메트릭이 있으면 **opset 20** — 그러나 **런타임이 5D를 받는지 별도 확인**.
- [ ] 특정 opset이 필요하면 **`dynamo=False`**. torch 2.11 기본(dynamo)은 요청 opset을 조용히 상향/무시한다.
- [ ] `torch.onnx.export` 성공을 믿지 말 것 — **`onnx.checker` + 실런타임 로드**로 유효성 확인(dynamo가 out-of-spec 모델을 흘려보냄).
- [ ] 정본 스택에서 **5D grid_sample = 분해/plugin 전제**: ORT 1.23.2 CUDA는 5D를 **조용히 CPU 폴백**(INFO 로그 1줄), TensorRT는 **rank-4 단언으로 파싱 실패**.
- [ ] MSDeformAttn을 mmcv 폴백으로 분해하면 export는 되지만 **grid_sample×num_levels + 노드 폭증** — GPU=plugin / NPU=구조변경을 사전 결정.
- [ ] 로그는 **원문 통째로** 남긴다(6개월 뒤 같은 에러의 검색 앵커).

---

## Tier B — BEVFormer-tiny 실모델 (레거시 venv `~/bevf-legacy`)

레거시 venv: python 3.10 · **torch 1.13.1+cu117** · **mmcv-full 1.7.0(프리빌트 cu117/torch1.13 휠)** ·
**mmdet3d 1.0.0rc6**(py3-none-any, ops는 mmcv로 이관) · **mmdet 2.28.2** · numpy **1.23.5**(mmdet3d의 `np.int` 별칭) ·
numba 0.58.1 · opencv-python 4.8.1.78 · onnx 1.14.1 · onnxruntime 1.16.3.
> 무컴파일 성립 조건(hard-won): mmcv/mmdet3d **프리빌트 휠**로 CUDA 소스 빌드를 전부 우회(정본 툴킷은 CUDA 12.8뿐, 11.7 없음).
> mmdet3d는 `--no-build-isolation`(setup.py가 torch import) + numba/opencv/numpy 핀 고정으로 설치. BEVFormer plugin은
> dd3d/detectron2 체인을 4파일 패치로 우회(bevformer_tiny는 dd3d 불필요). 실행: `~/bevf-legacy/bin/python b06_mmcv_real_op.py`.

| # | 단계 | 결과(핵심) | 상태 |
|---|------|-----------|------|
| B1 | 레거시 mmcv CUDA op이 CUDA 12.8/드라이버 595에서 사는가 | **프리빌트 휠로 컴파일 없이 로드·실행 성공**(아래) | ✅ 실측(b06) |
| B2 | 실제 mmcv `MultiScaleDeformableAttention` 모듈 ONNX export | CPU→표준분해 244노드 / **CUDA→silent-wrong(상수 baked)** | ✅ 실측(b06) |
| B3-a | BEVFormer-tiny **FP32** nuScenes-mini val mAP(실모델) | **mAP 0.2647 / NDS 0.2667**(81샘플 스모크, 벤치 아님) | ✅ 실측(b08) |
| B3-b | BEVFormer-tiny **전체 모델** ONNX export | CPU forward OK지만 export가 `point_sampling`에서 사망(아래) | ✅ 실측(b09) |
| B3-c | 전체 모델 INT8 PTQ→INT8 mAP | **도달 불가** — 유효 export 경로 없음(B3-b) → 포크 플러그인 툴체인 필요 | ⛔ 범위 밖 |

### B1 — 레거시 mmcv CUDA op은 **컴파일 없이** CUDA 12.8에서 산다 (b06)
- **핵심**: 정본 툴킷이 **CUDA 12.8뿐**(11.7 없음)이라 mmcv 1.x CUDA op **소스 컴파일은 위험**하다고 봤으나 —
  **프리빌트 휠**(`download.openmmlab.com/mmcv/dist/cu117/torch1.13`)이 존재해 **빌드 자체를 우회**했다.
- torch 1.13.1+cu117이 **드라이버 595(CUDA 12.8 capable)에서 CUDA init 성공**(하위호환) → RTX 3080 matmul OK.
- mmcv 1.7.0 `_ext`(컴파일된 .so) 로드 성공 → `MultiScaleDeformableAttnFunction`(**진짜 CUDA 커널**) 실행:
  out `(1,100,256)`, 순수 PyTorch 분해와 `|Δ|max = 1.4e-06` 일치 = 커널 정상.
- **함의**: "레거시 mmcv = 옛 CUDA 재현 필요"라는 통념과 달리, **프리빌트 휠 + 드라이버 하위호환**으로 정본 머신에서 바로 돈다.

### B2 — 실제 모듈 export: CPU=표준분해, **CUDA=조용히 틀린 그래프** (b06) 🔴
- mmcv 1.7.0 `multi_scale_deform_attn.py`는 **`symbolic`도 `is_in_onnx_export` 가드도 없다**(소스 확인) →
  바닐라 mmcv엔 **커스텀 ONNX 노드/플러그인 경로가 내장돼 있지 않다**(그건 `DerryHub/BEVFormer_tensorrt` 포크가 *추가*하는 것).
- 모듈 forward는 `value.is_cuda` 로 분기(line 351-358): **CUDA→CUDA Function**, **CPU→순수 PyTorch 폴백**.
  이 분기 때문에 export 결과가 디바이스에 따라 갈린다:

  | export 디바이스 | 결과 | 노드 | 그래프 입력 | 판정 |
  |---|---|---|---|---|
  | **CPU** | 표준 op 분해 | **244** (GridSample **4**=num_levels) | `[query, value, reference_points]` (전부 생존) | ✅ 유효 |
  | **CUDA** | export "성공"(exit 0, `onnx.checker` PASS) | **4** (`Constant 1, MatMul 1, Add 2`) | **`[query]` 만** | 🔴 **silent-wrong** |

- **CUDA 경로의 함정**: symbolic이 없는 autograd.Function을 트레이서가 표현하지 못해 **MSDeformAttn 출력 전체를 Constant로 baked** →
  `value`·`reference_points`가 **그래프 입력에서 사라진다**(실측: `graph.input == ['query']`). 남는 건 `output_proj`(MatMul+Add)와
  residual Add뿐. 모델이 **입력을 무시**하지만 **에러 없이 통과**한다(폭탄).
- **결론(§4.6.2 정정/강화)**: "바닐라 mmcv로 BEVFormer MSDeformAttn을 export하는 **유일한 유효 경로 = CPU 폴백 분기 강제**" →
  b05가 본 표준 분해(GridSample×num_levels)가 곧 실모델 export 결과다. **CUDA에서 export하면 조용히 틀린다** —
  초안이 못 짚은 실전 함정(export 성공 ≠ 유효; per-device 분기 주의).

**추가 design rule**:
- [ ] mmcv 커스텀 op 모델은 **반드시 CPU 텐서로 export**(CUDA export는 커스텀 op을 상수로 baked → 입력 소실). export 후 **그래프 입력 목록**을 확인해 `value`/`reference_points` 생존을 검증할 것.
- [ ] 프리빌트 mmcv 휠(cu117/torch1.13)은 **드라이버 하위호환** 덕에 CUDA 12.8 머신에서 컴파일 없이 동작 — 레거시 CUDA 툴킷 재설치 불필요.

### B3-a — 실모델 FP32 mAP (nuScenes-mini val, 81샘플) (b08)
- **환경**: 레거시 venv에서 BEVFormer-tiny(33.52M) + `bevformer_tiny_epoch_24.pth`(643/643 로드) → nuScenes **v1.0-mini** val.
  temporal info pkl은 BEVFormer `tools/create_data.py … --version v1.0-mini`로 생성(val=**81 keyframes / 2 scene**).
- **실행 주의**(막혔던 3곳): ① `create_data.py`는 `PYTHONPATH=<repo>`가 없으면 `No module named 'tools'` ② test.py의
  non-dist 분기가 `assert False`로 막혀 있어 **`torch.distributed.launch --nproc_per_node=1 … --launcher pytorch`** 필수
  ③ dataloader worker가 `dict_keys` pickle 불가로 죽어 **`--cfg-options data.workers_per_gpu=0`** 필요.
- **결과**: **mAP 0.2647 · NDS 0.2667** (mATE 0.857·mASE 0.769·mAOE 1.258·mAVE 0.733·mAAE 0.299).
  per-class: car 0.478·bus 0.576·pedestrian 0.446·truck 0.389·traffic_cone 0.354·motorcycle 0.286·bicycle 0.119,
  **trailer·construction_vehicle·barrier = 0.000**(2개 mini scene에 희소/미검출).
- 🔴 **캐비앗(반드시 병기)**: **81샘플·2 scene·3클래스 0.000의 고분산 스모크**다. mAP 0.2647이 공개 full-val 0.252와
  **가까운 건 우연**이고, NDS는 0.2667 vs 공개 **0.354**로 크게 벌어진다(mini엔 TP-의존 오차/속도추정이 빈약). →
  **절대값은 문헌 비교 불가**, 같은 mini 슬라이스에서의 **상대 FP32↔INT8 델타**만 의미 있다(DETR 캐비앗과 동일 성격).

### B3-b — 전체 모델 export는 `point_sampling`(BEV→카메라 사영)에서 죽는다 (b09) 🔴
- **바닐라 BEVFormer엔 ONNX export 스크립트가 없다**(`tools/`에 onnx/deploy/pth2onnx 전무) — export 경로는
  `DerryHub/BEVFormer_tensorrt` 포크가 **추가**하는 것. 그래서 "전체를 그냥 `torch.onnx.export`" 하면 어디서 막히나 실측.
- **CPU forward는 성공**(B2 규칙대로 CPU 강제): `pts_bbox_head(extract_feat(img), img_metas, prev_bev=None)` →
  `bev_embed (2500,1,256)`. 즉 **연산 자체는 돈다**. 그러나 `torch.onnx.export(opset16, CPU)`는 즉시 실패:
  - **에러 원문**: `RuntimeError: shape '[1, 6, 6, 1, 4, 4]' is invalid for input of size 96`
    at `projects/mmdet3d_plugin/bevformer/modules/encoder.py:119` — `point_sampling()`의
    `lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4)`.
- **원인(구조적)**: `point_sampling`은 forward **내부**에서 카메라 캘리브를 이렇게 재구성한다 —
  `for m in img_metas: lidar2img.append(m['lidar2img'])` → `np.asarray(...)` → `reference_points.new_tensor(...)`.
  즉 **`lidar2img`/`can_bus`는 메타데이터가 아니라 그래프의 기능적 입력**(BEV 격자를 카메라 픽셀로 사영)인데,
  이게 **비텐서 img_metas → numpy 우회 → new_tensor**로 들어온다. eager에선 `B=1`이 상수로 잡히지만,
  **트레이서는 `D,B,num_query = reference_points.size()[:3]`의 파이썬-int 차원을 잘못 바인딩**(B를 num_cam=6으로) →
  `view(1, 6, 6, 1, 4, 4)`(=3456) vs 실제 96(=6×4×4) 불일치. **eager 성공 ≠ export 성공**의 교과서 사례.
- **함의**: BEVFormer 전체 export가 "지옥"인 진짜 이유는 grid_sample op 하나가 아니라 —
  ① **img_metas(lidar2img/can_bus)가 forward 내부 기하 연산의 입력**이라 그래프에서 사라지면 안 되고,
  ② forward_test가 `self.prev_frame_info`로 **시간축 재귀(stateful)** 하며,
  ③ 캘리브를 numpy로 재구성하는 `.view()` 하드코딩이 트레이서와 안 맞는다.
  → 포크의 export 래퍼는 정확히 이 셋을 걷어낸다(**lidar2img/can_bus/prev_bev를 명시적 텐서 입력으로** 승격 + point_sampling 재작성).
  vanilla를 그대로 내보내는 유효 경로는 **없다**.

### B3-c — 전체 모델 INT8 mAP: 범위 밖(정직한 폴백) ⛔
- B3-b로 **유효한 전체 모델 ONNX가 안 나오므로**, 그 위에서 INT8 PTQ→INT8 mAP는 이 타임박스에서 **도달 불가**.
- 전체 모델 INT8은 `DerryHub/BEVFormer_tensorrt`의 **커스텀 op 플러그인 + tensor-화 래퍼** 툴체인이 전제이며,
  그 포크는 TRT 8.5/CUDA 11.6 기준이라 정본(TRT 10.16/CUDA 12.8)에서 **플러그인 재빌드**가 필요(가이드 §4.6.3 "확인 필요"와 일치).
- 따라서 B3의 실모델 산출은 **FP32 mini mAP(B3-a)** + **export 벽의 실측 지점(B3-b)** 까지다. op-단위 INT8/양자화 거동은
  Tier A(b01~b05)와 B2가 **실모델 op로** 이미 검증했다.

**추가 design rule (B3)**:
- [ ] BEVFormer류(멀티뷰 BEV)는 `img_metas`의 **lidar2img/can_bus가 forward 내부 기하 연산의 입력**이다 — export 전
      이들을 **명시적 텐서 입력으로 승격**하고 `point_sampling`류의 `.view(하드코딩 dim)`을 트레이서-안전하게 재작성해야 한다(포크가 하는 일).
- [ ] forward_test의 **시간축 재귀(prev_bev, self.prev_frame_info)** 는 stateful이라 트레이스 불가 — export는 **단일 프레임(prev_bev를 입출력으로 노출)** 으로 풀어야 한다.
- [ ] 실모델 mAP은 데이터 슬라이스를 **명시**하라. nuScenes-mini(81/2scene)는 **스모크**지 벤치가 아니다 — 절대값 문헌비교 금지, 같은 슬라이스 상대 델타만.
- [ ] 레거시 mmdet3d 실행 3종 세트: `PYTHONPATH=<repo>` 필수 · test.py는 `torch.distributed.launch --launcher pytorch`(non-dist 분기 `assert False`) · dataloader는 `workers_per_gpu=0`(`dict_keys` pickle 회피).

## 재현 순서
```bash
# ── Tier A (정본 venv ~/emb-ai, torch 2.11) — op 단위 지뢰 ──
source ~/emb-ai/bin/activate
python b01_grid_sample_4d_opset.py      # 4D opset 경계
python b02_grid_sample_5d_export.py     # 5D export (opset20 산출물 보존)
python b03_grid_sample_runtime_ort.py   # ORT 4D/5D 런타임 (b02 이후)
python b04_grid_sample_trt_parse.py     # TensorRT 파싱 (b02 이후)
python b05_msdeformattn_decompose.py    # MSDeformAttn 분해

# ── Tier B (레거시 venv ~/bevf-legacy, torch 1.13 / mmcv-full 1.7.0 / mmdet3d 1.0.0rc6) — 실모델 ──
~/bevf-legacy/bin/python b06_mmcv_real_op.py     # B1/B2: 실 mmcv MSDeformAttn op(CUDA 12.8 생존 + CPU-only 유효 export)
# B3-a FP32 mini mAP (BEVFormer repo 루트에서):
PYTHONPATH=$(pwd) ~/bevf-legacy/bin/python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes --canbus ./data --extra-tag nuscenes --version v1.0-mini --out-dir ./data/nuscenes
PYTHONPATH=$(pwd) ~/bevf-legacy/bin/python -m torch.distributed.launch --nproc_per_node=1 tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py ckpts/bevformer_tiny_epoch_24.pth \
  --launcher pytorch --eval bbox --cfg-options data.workers_per_gpu=0   # → mAP 0.2647 / NDS 0.2667
# B3-b 전체 모델 export 벽:
PYTHONPATH=$(pwd) ~/bevf-legacy/bin/python b09_fullmodel_export_attempt.py  # → point_sampling lidar2img.view RuntimeError
```
> 산출 JSON: `b08_fp32_mini_map.json`(FP32 mAP), `b09_fullmodel_export_result.json`(export 벽). 로그: `b08_fp32_mini_eval.log`.
