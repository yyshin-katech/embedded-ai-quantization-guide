---
name: stage8-capstone-hands-on
description: "8단계 캡스톤 BEVDet 실측(2026-08-18, AI-LAP RTX 3080, 커밋 64e4c84 푸시완료): 커스텀 CUDA op bev_pool_v2를 sudo·Docker 없이 user-space cu117 툴체인(제3의 길)으로 컴파일→nuScenes-mini FP32 walking skeleton 관통. 정식 가중치 Baidu-locked→init 가중치라 mAP 0.0000(예상값), latency p50 34.06ms(공식 33.3ms 교차확증). INT8/TRT-plugin 범위 밖"
metadata:
  node_type: memory
  type: project
---

8단계 캡스톤(**BEVDet end-to-end**) 실기 검증. 794줄 초안(미검증)을 실제 AI-LAP RTX 3080
([[machine-ai-lap-rtx3080]])에서 관통. **중간 스코프**(사용자 사전합의 = 실제 FP32 baseline까지, TRT-INT8-plugin은
범위 밖, 첫 하드월에서 자동 폴백). 산출물: `logs/stage8_capstone_report.html` ·
`experiments/stage8_capstone/`(results 4 JSON·scripts 4·`capstone_constraints.md`·README) ·
`study_guide/08_capstone.md` §3·§4-1·§4-2·§4-4·§9 🔬 콜아웃 5개(삽입만, git `10 insert/0 delete` 승번 0).

**헤드라인 — 제3의 길(문서 §3 빈칸):** BEVDet `bev_pool_v2`(LSS pooling, python fallback 없음) 커스텀 CUDA op은
torch≤1.13+CUDA≤11.7 요구인데 이 머신 유일 toolkit은 **CUDA 12.8** → `torch._check_cuda_version` MAJOR 불일치
(12≠11) **hard-error**([[stage2-bevformer-hands-on]] §4.6과 동일 근본원인). 문서 §3의 2안(Docker cu116/blackwell
패치)이 **Docker 없고 sudo 없는 머신엔 둘 다 불가** → **user-space cu117 툴체인 조립**(제3의 길)으로 넘음:
- **nvcc** = micromamba `cuda-nvcc=11.7`(V11.7.99, 완전 프론트엔드 — pip `nvidia-cuda-nvcc-cu11`은 ptxas만이라 불충분)
- **libcudart** = pip `nvidia-cuda-runtime-cu11==11.7.99`(실제 `libcudart.so.11.0` — torch 번들 `libcudart-<hash>.so`는
  `-lcudart` 링크 불가)
- **Python.h** = micromamba `python=3.10` 헤더를 `CPATH`로(`python3.10-dev`·sudo 불요 우회)
- **CUDA_HOME** = `~/capstone-bev/cuda-home` 조립(bin/include/nvvm→cu117 심링크, lib64에 `libcudart.so`→pip 런타임)
- `TORCH_CUDA_ARCH_LIST=8.6` + `python setup.py build_ext --inplace` → **`bev_pool_v2_ext.so` 9,131,040 bytes 빌드 성공**.
- **두 번째 벽:** 카메라 전용 BEVDet인데 bundled mmdet3d `detectors/__init__.py`가 LiDAR-fusion 검출기 **DAL을
  eager import**→spconv 요구 → `pip install spconv-cu117`(2.3.6+cumm 0.4.11)로 해소(6개 `__init__` 다중패치보다 깨끗).

**격리 legacy venv `~/bevf-legacy`:** torch **1.13.1+cu117**, mmdet3d **1.0.0rc4**(BEVDet dev3.0 bundled), mmcv-full
1.7.0, spconv-cu117 2.3.6, python 3.10.12. 정본 emb-ai(torch 2.11+cu128) **오염 0** — `bev_pool_v2_ext.so`는 legacy
전용 아티팩트. 재현 env: `CUDA_HOME=~/capstone-bev/cuda-home; PATH=$CUDA_HOME/bin:$PATH;
LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:...; cd ~/capstone-bev/BEVDet; PYTHONPATH=$PWD`(SP=bevf-legacy
site-packages). BEVDet 데이터는 [[stage2-bevformer-hands-on]]의 nuScenes-mini 심링크 재사용.

**데이터(벽 3):** `create_data_bevdet.py`는 `v1.0-trainval` 하드코딩+무거운 `create_groundtruth_database`라 mini 불가 →
3곳 패치본 `create_data_bevdet_mini.py`(`v1.0-mini`×2 + gt_db 주석). 실행엔 `PYTHONPATH=$PWD`(repo root) 필수
(`python tools/x.py`는 sys.path에 repo root 안 넣음). 산출 **`bevdetv3-nuscenes_infos_{train,val}.pkl` = 323 train/81
val**(8/2 scenes). **초안 오류:** 태그가 `bevdetv2`가 아니라 **`bevdetv3-nuscenes`**(dev3.0 실제, config도 v3).

**walking skeleton 완주(문서 §9 기준) + 정직한 폴백:** BEVDet-R50 정식 detection 가중치는 **Baidu Pan 전용**(헤드리스
접근 불가, WebSearch 2회로 미러 못 찾음) → **사용자 사전승인 fall-back 지점**. `init_r50.pth`(torchvision ResNet50
backbone[ImageNet 실사전학습] + `init_weights()` head[랜덤])로 stock `tools/test.py` 관통(81 val):
- **mAP 0.0000 / NDS 0.0260**(전 클래스 AP 0.000)은 **버그 아니라 예상값** — backbone만 진짜, LSS·BEV encoder·head 랜덤.
- **검증됨:** 툴체인·데이터·`bev_pool_v2` CUDA op(GPU 실행)·nuScenes eval 하네스 exit 0 관통(기능적 완결성).
- **안 됨:** 절대 mAP 문헌대조(공개 ≈ mAP 0.298/NDS 0.379) — Baidu 가중치 확보 시 동일 하네스 재실행.
- **latency는 가중치 무관 유효:** FP32 forward **p50 34.0603ms**(batch1, CUDA event-timed, N=30; 재실행 34.2944ms 지터
  내 일치, peak 420.9 MiB, 44.25M params)가 공식 README **BEVDet-R50 total 33.3ms(RTX 3090)**와 근접 → 랜덤 head라도
  실연산량은 정식과 동일 교차확증. `bench_fp32_latency.py` 재실행으로 스크립트 충실성 확인.

**INT8/TRT-plugin 범위 밖(정직한 폴백):** §4-4 3경로 모두 벽 — A1(`convert_bevdet_to_TRT.py --int8`)·B(DerryHub)는
커스텀 TRT 플러그인 `TRTBEVPoolv2`를 TRT 8.5+CUDA 11.x로 빌드(정본은 TRT 10.16, §4.6 포크 플러그인 벽과 동일),
A2(ModelOpt QDQ)는 `bev_pool_v2`가 표준 ONNX op 아니라 export 자체 §4-3 벽. 이 머신 INT8 축은 이미 3·5단계
([[stage3-tensorrt-hands-on]]·[[stage5-infrastructure-hands-on]])·4단계([[stage4-qualcomm-aihub-hands-on]])서 완결 →
BEVDet INT8은 TRT-8.5-plugin 툴체인 확보 시 다음 과제.

**캐비앗(불변):** ① 절대 mAP/NDS는 init 가중치+mini 81장 → 무의미(파이프라인 검증용, 문헌비교 불가). ② latency는
event-timed·forward-only·batch1 → 다른 단계(TRT event-timed/하네스 wall-clock)와 1:1 비교 불가, 구조·상대만. ③
`bev_pool_v2_ext.so` legacy env 전용(emb-ai 오염 0). ④ INT8/TRT-plugin 다음 과제.

**커밋 64e4c84 푸시완료** — 스캔 후 요청으로 수행([[repo-is-public-scan-before-commit]], 스크립트/문서 전건 시크릿 0건).
tech-reviewer 팬인 PASS(🔴0·🟡1해소·🟢2) 후 통합 완료(README·study_guide/README·CLAUDE 변경이력·이 메모리). **남은 과제:** Baidu 가중치 확보 시 절대 mAP 재실행 · BEVDet
INT8은 TRT 8.5+커스텀 플러그인 툴체인 대기 · TI TDA4VM·Renesas RZ/V2H 벤더 NPU는 보드 대기.
