# 8단계 캡스톤 실기 제약·설계규칙 (BEVDet end-to-end, AI-LAP RTX 3080)

794줄 초안(미검증)을 실제로 관통시키며 부딪힌 벽·해법·설계규칙. SSOT = `results/*.json`.
헤드라인: **문서 §3이 제시한 2안(Docker cu116 / blackwell 패치)이 이 머신엔 둘 다 불가 → user-space cu117
툴체인(제3의 길)으로 커스텀 CUDA op을 sudo·Docker 없이 컴파일하고 FP32 파이프라인을 walking skeleton으로
관통**. 절대 정확도는 Baidu-locked 가중치 벽에서 정직하게 폴백. **후속 세션**에서 벽 5(INT8/TRT plugin)까지
관통 — 커스텀 플러그인 툴체인을 조립해 FP32→FP16→INT8 TRT 엔진 사다리를 실측(아래 벽 5).

---

## 벽 1 — 커스텀 CUDA op 컴파일 (torch MAJOR CUDA 불일치)

**증상.** BEVDet dev3.0의 bundled mmdet3d(1.0.0rc4) `bev_pool_v2`(LSS pooling, python fallback 없음)를
빌드하려면 nvcc가 필요한데, 이 머신 유일 toolkit은 **CUDA 12.8**. torch 1.13.1+cu117의
`torch.utils.cpp_extension._check_cuda_version`가 컴파일러 MAJOR(12) ≠ torch build MAJOR(11)에서
**RuntimeError로 hard-error**(minor 불일치는 warning, major는 중단).

**§4.6 BEVFormer와 동일 근본원인.** 레거시 mmcv/mmdet3d 커스텀 CUDA 커널은 torch≤1.13+CUDA≤11.7 계열에 묶임.
문서 §3 초안이 준 길은 (a) Docker `nvcr.io ... cu116` 또는 (b) blackwell 패치뿐 — 이 머신엔 **Docker 없음**,
정본 emb-ai venv(torch 2.11+cu128) **오염 불가** → 둘 다 막힘.

**해법 = user-space cu117 툴체인 조립(제3의 길, sudo·Docker·4.3GB runfile 전부 회피):**

| 조각 | 출처 | 왜 이걸로 |
|------|------|----------|
| nvcc | micromamba `cuda-nvcc=11.7` → V11.7.99 | 완전한 프론트엔드. pip `nvidia-cuda-nvcc-cu11`은 ptxas만 있어 불충분 |
| libcudart | pip `nvidia-cuda-runtime-cu11==11.7.99` → 실제 `libcudart.so.11.0` | torch 번들 `libcudart-<hash>.so`는 SONAME 어긋나 `-lcudart` 링크 불가 |
| Python.h | micromamba `python=3.10` env → CPATH | `python3.10-dev` 미설치·sudo 불요 우회 |
| CUDA_HOME | `~/capstone-bev/cuda-home` 조립 | bin/include/nvvm→cu117 심링크, lib64 실디렉토리에 `libcudart.so`→pip 런타임 |

빌드 env: `CUDA_HOME=~/capstone-bev/cuda-home`, `PATH=$CUDA_HOME/bin:$PATH`,
`TORCH_CUDA_ARCH_LIST=8.6`, `CPATH=~/capstone-bev/pyhdr/include/python3.10` →
`python setup.py build_ext --inplace` → **`bev_pool_v2_ext.*.so` 9,131,040 bytes** (forward/backward export).
→ `scripts/build_bev_pool_v2.sh`

**설계규칙.** 레거시 커스텀 CUDA op을 최신 드라이버 머신에서 빌드할 때, Docker/sudo가 막히면 세 조각
(nvcc·libcudart·Python.h)을 user-space에 각각 조달해 `CUDA_HOME`을 손으로 조립하는 길이 있다. 핵심은
`-lcudart`가 **실제 SONAME `libcudart.so.11.0`**로 해결되게 하는 것(torch 번들 hash-suffix .so로는 안 됨).

---

## 벽 2 — spconv (카메라 전용인데 LiDAR-fusion 검출기를 eager import)

**증상.** 모델 빌드 시 `ModuleNotFoundError: No module named 'spconv'`. BEVDet은 카메라 전용인데도.

**원인.** bundled mmdet3d `models/detectors/__init__.py`가 **모든** 검출기를 eager import → 그중 `DAL`
(LiDAR-camera fusion)이 `spconv_voxelize`→spconv 요구. spconv 의존이 detectors/middle_encoders/roi_heads/ops
**6곳 캐스케이드**.

**해법.** 6개 `__init__` 다중패치보다 격리 legacy env에 `pip install spconv-cu117`(→ spconv-cu117 2.3.6 +
cumm-cu117 0.4.11)가 깨끗. 정본 venv와 격리돼 있어 부작용 없음.

**설계규칙.** OpenMMLab detectors `__init__`은 eager-import-all이라, 쓰지도 않는 모달리티의 의존까지 끌고 온다.
누락 의존은 "이 모델이 그걸 쓴다"가 아니라 "패키지가 그걸 import한다"는 신호 — 설치가 패치보다 싸다.

---

## 벽 3 — create_data가 trainval 하드코딩 (mini 실행 불가)

`tools/create_data_bevdet.py`는 `version='v1.0-trainval'`·`add_ann_adj_info`의
`nuscenes_version='v1.0-trainval'` 하드코딩 + 무거운 `create_groundtruth_database` 호출. 3곳 패치본
`create_data_bevdet_mini.py`로 우회(→ `scripts/create_data_mini_patch.md`). 실행엔 `PYTHONPATH=$PWD`(repo root)
필수 — `python tools/x.py`는 repo root를 sys.path에 안 넣어 `tools.data_converter` import 실패.

**산출:** `bevdetv3-nuscenes_infos_{train,val}.pkl` = **323 train / 81 val** (mini 8/2 scenes).

**초안 오류.** 문서 §4-1은 `bevdetv2-nuscenes_infos_*.pkl`라 적었으나 dev3.0 실제 태그는 **`bevdetv3-nuscenes`**
(config도 v3 참조). v2는 구버전 잔재.

---

## 벽 4 — 정식 가중치 = Baidu-locked (절대 정확도의 정직한 폴백)

**하드 월.** BEVDet-R50 detection 정식 체크포인트는 **Baidu Pan 전용**(헤드리스 환경 접근 불가). WebSearch 2회로
접근 가능한 미러 못 찾음. → 절대 mAP/NDS는 **사용자 사전승인 fall-back 지점**.

**대체.** `init_r50.pth` = `torchvision://resnet50` backbone(ImageNet 실사전학습) + `init_weights()` head(랜덤).
→ FP32 eval(81 val): **mAP 0.0000 / NDS 0.0260**, 전 클래스 AP 0.000. **이건 버그가 아니라 예상값** — backbone만
진짜고 LSS view-transform·BEV encoder·detection head는 랜덤이라 검출 0이 정상.

**무엇이 검증됐나(walking skeleton, 문서 §9 완주 기준):** 툴체인·데이터·**bev_pool_v2 CUDA op(GPU 실행)**·
nuScenes eval 하네스가 exit 0으로 끝까지 관통. 파이프라인 기능적 완결성 확인.
**무엇이 안 됐나:** 절대 mAP 문헌대조(공개 BEVDet-R50 ≈ mAP 0.298/NDS 0.379) — Baidu 가중치 필요.

**latency는 가중치 무관하게 유효.** forward 계산 그래프는 정식 모델과 동일 → FP32 forward **p50 34.06ms**
(batch1, CUDA event-timed, RTX 3080; 재실행 34.29ms 지터 내 일치), peak 420.9 MiB, 44.25M params.
BEVDet 공식 README의 **BEVDet-R50 total 33.3ms(RTX 3090)**와 근접 → 랜덤 head라도 실연산량은 정식과 동일함을 교차확증.

---

## 벽 5 — INT8/TRT plugin (후속 세션 관통 ✅)

문서 §4-4의 INT8 경로 **A1**(`tools/convert_bevdet_to_TRT.py --int8`)은 `TRTBEVPoolv2` 커스텀 **TRT 플러그인**을
TensorRT 8.5 + CUDA 11.x 툴체인으로 빌드해야 하는 벽이었다(정본은 TRT 10.16). 최초 세션엔 "다음 과제"로 폴백했으나,
**후속 세션에서 이 플러그인 툴체인을 user-space에 실제로 조립해 경로 A1을 끝까지 관통**시켰다 —
FP32→FP16→**INT8** TRT 엔진 3종을 빌드하고 지연 사다리를 측정. (경로 A2 ModelOpt QDQ·B DerryHub는 미시도.)

관통은 **6개 벽(W1~W6) 순차 통과**로 구성됐다:

| 벽 | 내용 | 해법 |
|----|------|------|
| **W1** | TensorRT 8.5.3.1 | pip 휠(tensorrt 8.5.3.1)로 legacy env에 설치 |
| **W2** | cuDNN 8.6 | 런타임 `libcudnn.so.8` |
| **W3** | mmdeploy fork bevpoolv2 플러그인 빌드 | 풀 mmdeploy CMake 트리 **우회** — 2개 TU(`trt_bev_pool_kernel.cu`+`trt_bev_pool.cpp`)가 self-contained → user-space cu117 nvcc(`-arch=sm_86 -std=c++14`) + g++(`-std=c++17`) 직접 컴파일·링크(`-lnvinfer`, `-l:libcudart.so.11.0`) → **`libmmdeploy_tensorrt_ops.so` 58,896 B**, `mmdeploy::TRTBEVPoolV2Creator`(bev_pool_v2 v1) 등록. → `scripts/build_trt_plugin.sh` |
| **W4** | pycuda | user-space cu117로 빌드(convert/bench가 요구) |
| **W5** | ONNX export 트레이서 사망 | `torch.onnx.export` 추적 중 nested `QuickCumsumCuda`(symbolic 없는 CUDA autograd.Function)에서 **`RuntimeError: _Map_base::at` + segfault**. → `TRTBEVPoolv2.forward`를 **export 전용 shim**(`feat.new_zeros(depth.shape[0], out_h, out_w, feat.shape[-1])`)으로 monkeypatch. `symbolic`이 `mmdeploy::bev_pool_v2` 노드를 emit하고 **실연산은 추론 시 TRT 플러그인**이 수행하므로 엔진 정확도는 무영향(표준 mmdeploy custom-op export 기법). → `scripts/convert_bevdet_trt.py` |
| **W6** | 엔진 빌드 segfault | convert의 `from_onnx`가 deprecated `builder.build_engine(network, config)` 사용 → TRT 8.5.3에서 이 네트워크에 대해 segfault. **진짜 벽이 아니라 API 문제**였다 — 모던 `builder.build_serialized_network(network, config)`로 교체하니 플러그인 레이어 포함 **완전 빌드**. |

**INT8 캘리브레이션.** ENTROPY_CALIBRATION_2, nuScenes-**mini val 81장**. `create_calib_input_data`가 img는
per-sample, ranks/metas(인덱스)는 sample-0을 재사용 → 플러그인 enqueue 중 OOB 없음. INT32 인덱스 텐서
(interval_starts/lengths 등)는 `Missing scale and zero-point` 경고 = **양자화 안 함(정상)**.

**결과 (지연 사다리, batch1, CUDA event-timed, N=60; SSOT `results/trt_ladder.json`·`results/int8_build.json`):**

| 정밀도 | 지연 p50 | vs FP32 | 엔진 크기 | FP32 대비 출력 corr |
|--------|---------|---------|-----------|---------------------|
| FP32 | 14.680 ms | — | 245.0 MB | — |
| FP16 | 4.905 ms | **×2.99** | 90.1 MB (×2.72↓) | ≥0.9994 (거의 동일) |
| INT8 | 2.630 ms | **×5.58** (FP16 대비 ×1.87) | 46.7 MB (×5.24↓) | 0.985~1.000 (rel_max≤0.22, height head 최저 0.985) |

**교차확증.** TRT FP32 엔진 14.68 ms는 PyTorch eager forward **34.06 ms**(벽 4)보다 2.3× 빠름 — 커널 퓨전·파이썬
오버헤드 제거. 랜덤 head라도 실연산 그래프가 동일하므로 사다리는 **가중치와 무관하게 유효**.

**설계규칙.** ① 레거시 mmdeploy 커스텀 op은 풀 CMake 트리 없이 **해당 op의 TU만 직접 컴파일**하는 게 빠르다
(bevpoolv2는 2 TU self-contained). ② symbolic 있는 custom autograd.Function이라도 **forward가 nested no-symbolic
CUDA op을 호출하면 트레이서가 죽는다** → forward를 shape-only shim으로 갈아끼우면 symbolic만 emit되고 실연산은
런타임 플러그인이 맡는다(export/inference 분리). ③ TRT 8.5의 `build_engine` segfault는 **플러그인 탓이 아닐 수
있다** — deprecated API를 `build_serialized_network`로 바꿔 먼저 배제하라.

---

## 재현 env 요약 (한 번에)

```bash
SP=~/bevf-legacy/lib/python3.10/site-packages
export CUDA_HOME=~/capstone-bev/cuda-home
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
cd ~/capstone-bev/BEVDet && export PYTHONPATH=$PWD
PY=~/bevf-legacy/bin/python   # torch 1.13.1+cu117, mmdet3d 1.0.0rc4, spconv-cu117 2.3.6
```

**INT8/TRT 사다리 재현(벽 5):** 플러그인 .so + TRT/cudnn 런타임을 `LD_LIBRARY_PATH`에 얹는다.

```bash
SP=~/bevf-legacy/lib/python3.10/site-packages
export CUDA_HOME=~/capstone-bev/cuda-home; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$SP/tensorrt:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/curand/lib
export PYTHONPATH=~/capstone-bev/mmdeploy-bevdet:~/capstone-bev/BEVDet
cd ~/capstone-bev/BEVDet
bash scripts/build_trt_plugin.sh                                  # W3: 플러그인 .so
for M in fp32 fp16 int8; do BEVDET_MODE=$M $PY scripts/convert_bevdet_trt.py; done  # W5+W6: 엔진 3종
$PY scripts/dump_bench_sample.py                                  # 벤치 입력 샘플
$PY scripts/bench_trt_engines.py                                  # 지연 사다리 → results/trt_ladder.json
```

**캐비앗(불변):** ① 절대 mAP/NDS는 init 가중치 + mini 81장 → 무의미(파이프라인 검증용). ② latency는 CUDA
event-timed·forward-only·batch1 → 다른 단계와 1:1 비교 불가, 구조·상대만. ③ bev_pool_v2_ext.so는 legacy env
전용 아티팩트(정본 emb-ai 오염 0). ④ INT8/TRT-plugin은 **후속 세션에서 관통**(벽 5) — 단 init 가중치·mini
81장 캘리브라 지연 사다리·엔진 크기·FP32대비 출력편차만 유효(절대 정확도 아님).
