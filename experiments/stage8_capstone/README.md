# experiments/stage8_capstone

8단계 **캡스톤(BEVDet end-to-end)** 실기 검증 산출물. 794줄 초안(미검증)을 실제 AI-LAP RTX 3080에서
관통시켜 정정. **중간 스코프 = 실제 FP32 baseline까지**(사용자 사전합의): user-space cu117 툴체인으로
`bev_pool_v2` 커스텀 CUDA op 컴파일 → nuScenes-mini에서 FP32 파이프라인을 **walking skeleton**(문서 §9 완주
기준)으로 관통. 절대 정확도(Baidu-locked 가중치)는 **정직한 폴백**. **후속 세션**에서 TRT-8.5-plugin INT8까지
관통 — `TRTBEVPoolV2` 커스텀 플러그인을 user-space에 조립(W3)하고 export shim(W5)·`build_serialized_network`(W6)로
FP32→FP16→**INT8** TRT 엔진 사다리를 실측(가중치-무관 지연·엔진 크기·출력편차).

- 모델: **BEVDet-R50** (configs/bevdet/bevdet-r50.py, 44.25M)
- 데이터: nuScenes-**mini** val 81장 (§4.6 BEVFormer가 쓰던 mini 재사용)
- 스택(격리 legacy env `~/bevf-legacy`): torch **1.13.1+cu117**, mmdet3d **1.0.0rc4**(BEVDet dev3.0 bundled),
  mmcv-full 1.7.0, spconv-cu117 2.3.6
- 툴체인: **user-space cu117**(sudo·Docker 없이) — nvcc V11.7.99(micromamba) + libcudart.so.11.0(pip) + CUDA_HOME 조립

전체 서술·SVG·판정은 **`logs/stage8_capstone_report.html`**, 벽·레시피·설계규칙은 **`capstone_constraints.md`**.

## 헤드라인

| 항목 | 실측 | 비고 |
|---|---|---|
| 커스텀 CUDA op 빌드 | **성공** (bev_pool_v2_ext.so 9.13 MB) | 문서 §3 2안(Docker/blackwell) 밖의 제3의 길 |
| create_data (mini) | 323 train / 81 val | 태그 `bevdetv3`(초안 v2 아님) |
| FP32 walking skeleton | **관통(exit 0)** | export→CUDA op(GPU)→eval 하네스 |
| FP32 latency | **p50 34.06 ms** (batch1, event-timed) | 공식 README 33.3ms(3090)와 교차확증 |
| FP32 mAP / NDS | 0.0000 / 0.0260 | init 가중치 → **예상값**(정직한 폴백) |
| **TRT 엔진 3종** | FP32 245 / FP16 90 / **INT8 47 MB** | 후속 세션 관통(W3 플러그인+W5 shim+W6 build API) |
| **INT8 지연 사다리** | **FP32 14.68 → FP16 4.91(×2.99) → INT8 2.63(×5.58) ms** | batch1 event-timed, 가중치-무관 유효 |
| INT8 출력편차 vs FP32 | corr 0.985~1.000 (rel_max≤0.22) | height head 최저 0.985(양자화 오차 전파) |

## 정정 / 발견 요약

1. **§3 제3의 길** — 커스텀 CUDA op을 sudo·Docker 없이 user-space cu117 툴체인으로 컴파일(nvcc·libcudart·Python.h
   3조각 조달 + CUDA_HOME 손조립). torch MAJOR CUDA 불일치 hard-error 우회.
2. **spconv 벽** — 카메라 전용 BEVDet인데 detectors `__init__`이 LiDAR-fusion DAL을 eager import → spconv-cu117 설치.
3. **§4-1 태그 정정** — `bevdetv2` → **`bevdetv3`**(dev3.0 실제 태그), mini 하드코딩 3곳 패치, `PYTHONPATH=$PWD` 필수.
4. **§4-2 정직한 폴백** — 정식 가중치 Baidu-locked → init 가중치로 파이프라인 관통, mAP 0은 예상값. latency는 유효.
5. **§4-4 경로 A1 관통(후속 세션)** — `convert_bevdet_to_TRT.py --int8`을 실제로 통과. W3 플러그인 직접 빌드
   (풀 CMake 우회, 2 TU) + W5 export shim(트레이서 `_Map_base::at` 우회, symbolic만 emit·실연산은 런타임 플러그인)
   + W6 `build_engine`→`build_serialized_network`(segfault는 플러그인이 아니라 deprecated API 탓). INT8 캘리브
   ENTROPY_2·mini 81장. → FP32→FP16→INT8 지연 사다리·엔진 크기·FP32대비 출력편차 실측. (경로 A2/B 미시도.)

## 파일

```
scripts/
  build_bev_pool_v2.sh        # user-space cu117 툴체인 조립 + 커스텀 CUDA op 빌드 (벽 1·2)
  create_data_mini_patch.md   # create_data_bevdet.py → mini 패치 3곳 + 실행 (벽 3)
  run_fp32_baseline.sh        # tools/test.py FP32 eval 관통 (walking skeleton, 벽 4)
  bench_fp32_latency.py       # CUDA event-timed forward latency (재실행 지터 내 일치)
  build_trt_plugin.sh         # [INT8] bevpoolv2 TRT 플러그인 직접 빌드 (풀 CMake 우회, 벽 5-W3)
  convert_bevdet_trt.py       # [INT8] W5 export shim + W6 build_serialized_network 통합 convert 실행기
  dump_bench_sample.py        # [INT8] 벤치 입력(img+5 ranks) npz 덤프
  bench_trt_engines.py        # [INT8] FP32/FP16/INT8 엔진 지연 사다리 + FP32대비 출력편차
results/
  toolchain.json             # 툴체인 조립·버전·빌드 산출 SSOT
  create_data_mini.json      # info pkl 생성 결과 (323/81)
  fp32_baseline_eval.json    # FP32 eval 지표 (mAP/NDS + 정직한 해석)
  fp32_latency.json          # FP32 forward latency (+재현 확인)
  int8_build.json            # [INT8] 벽 W1~W6·플러그인·캘리브·엔진·사다리 비율 SSOT
  trt_ladder.json            # [INT8] FP32/FP16/INT8 지연 p50 + 엔진 크기 + 출력 corr 원자료
```

## 재현 / 캐비앗

재현 env·절차·캐비앗은 `capstone_constraints.md` 참조(FP32 env + INT8/TRT 사다리 env 2블록). 요약: 절대 mAP/NDS는
init 가중치+mini라 무의미(파이프라인 검증용), latency는 event-timed·batch1이라 다른 단계와 1:1 비교 불가(구조·상대만),
bev_pool_v2_ext.so·플러그인 .so·TRT 엔진은 legacy env 전용(정본 emb-ai 오염 0). **INT8/TRT-plugin은 후속 세션에서
관통**(벽 5) — 단 init 가중치·mini 81장이라 지연 사다리·엔진 크기·FP32대비 출력편차만 유효(절대 정확도 아님).
전체 서술은 **`logs/stage8_capstone_int8_report.html`**(INT8/TRT 관통 전용 보고서).
