# 8단계 캡스톤 실기 제약·설계규칙 (BEVDet end-to-end, AI-LAP RTX 3080)

794줄 초안(미검증)을 실제로 관통시키며 부딪힌 벽·해법·설계규칙. SSOT = `results/*.json`.
헤드라인: **문서 §3이 제시한 2안(Docker cu116 / blackwell 패치)이 이 머신엔 둘 다 불가 → user-space cu117
툴체인(제3의 길)으로 커스텀 CUDA op을 sudo·Docker 없이 컴파일하고 FP32 파이프라인을 walking skeleton으로
관통**. 절대 정확도는 Baidu-locked 가중치 벽에서 정직하게 폴백.

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

## 벽 5 — INT8/TRT plugin (범위 밖, 정직한 폴백)

문서 §4-4의 INT8 3경로 모두 이 머신에선 벽:
- **A1 (`tools/convert_bevdet_to_TRT.py --int8`)**: `TRTBEVPoolv2` 커스텀 **TRT 플러그인**을 TensorRT 8.5 +
  CUDA 11.x 툴체인으로 빌드해야 함(정본은 TRT 10.16). §4.6 BEVFormer의 "포크 커스텀 op 플러그인 툴체인"과 동일 벽.
- **A2 (ModelOpt QDQ)**: bev_pool_v2가 표준 ONNX op가 아니라 export 자체가 §4-3 벽(커스텀 심볼릭 필요).
- **B (DerryHub plugins)**: 별도 플러그인 레포 + TRT 8.x.

→ **사용자 사전합의대로 범위 밖**(중간 스코프 = 실제 FP32 baseline까지). "TRT-8.5-plugin INT8은 다음 과제"로
정직하게 폴백. 이 머신에서 실측 가능한 INT8 축은 이미 3·5단계(ResNet50 polygraphy)·4단계(HTP)에서 완결.

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

**캐비앗(불변):** ① 절대 mAP/NDS는 init 가중치 + mini 81장 → 무의미(파이프라인 검증용). ② latency는 CUDA
event-timed·forward-only·batch1 → 다른 단계와 1:1 비교 불가, 구조·상대만. ③ bev_pool_v2_ext.so는 legacy env
전용 아티팩트(정본 emb-ai 오염 0). ④ INT8/TRT-plugin은 범위 밖(다음 과제).
