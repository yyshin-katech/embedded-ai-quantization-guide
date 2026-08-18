# experiments/stage8_capstone

8단계 **캡스톤(BEVDet end-to-end)** 실기 검증 산출물. 794줄 초안(미검증)을 실제 AI-LAP RTX 3080에서
관통시켜 정정. **중간 스코프 = 실제 FP32 baseline까지**(사용자 사전합의): user-space cu117 툴체인으로
`bev_pool_v2` 커스텀 CUDA op 컴파일 → nuScenes-mini에서 FP32 파이프라인을 **walking skeleton**(문서 §9 완주
기준)으로 관통. 절대 정확도(Baidu-locked 가중치)·TRT-8.5-plugin INT8은 **정직한 폴백**.

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
| INT8 / TRT-plugin | 범위 밖 | TRT 8.5 + 커스텀 플러그인 필요(다음 과제) |

## 정정 / 발견 요약

1. **§3 제3의 길** — 커스텀 CUDA op을 sudo·Docker 없이 user-space cu117 툴체인으로 컴파일(nvcc·libcudart·Python.h
   3조각 조달 + CUDA_HOME 손조립). torch MAJOR CUDA 불일치 hard-error 우회.
2. **spconv 벽** — 카메라 전용 BEVDet인데 detectors `__init__`이 LiDAR-fusion DAL을 eager import → spconv-cu117 설치.
3. **§4-1 태그 정정** — `bevdetv2` → **`bevdetv3`**(dev3.0 실제 태그), mini 하드코딩 3곳 패치, `PYTHONPATH=$PWD` 필수.
4. **§4-2 정직한 폴백** — 정식 가중치 Baidu-locked → init 가중치로 파이프라인 관통, mAP 0은 예상값. latency는 유효.
5. **§4-4/§9 범위 밖** — INT8 3경로 모두 TRT 8.5 + 커스텀 플러그인 벽 → 사용자 합의대로 폴백.

## 파일

```
scripts/
  build_bev_pool_v2.sh        # user-space cu117 툴체인 조립 + 커스텀 CUDA op 빌드 (벽 1·2)
  create_data_mini_patch.md   # create_data_bevdet.py → mini 패치 3곳 + 실행 (벽 3)
  run_fp32_baseline.sh        # tools/test.py FP32 eval 관통 (walking skeleton, 벽 4)
  bench_fp32_latency.py       # CUDA event-timed forward latency (재실행 지터 내 일치)
results/
  toolchain.json             # 툴체인 조립·버전·빌드 산출 SSOT
  create_data_mini.json      # info pkl 생성 결과 (323/81)
  fp32_baseline_eval.json    # FP32 eval 지표 (mAP/NDS + 정직한 해석)
  fp32_latency.json          # FP32 forward latency (+재현 확인)
```

## 재현 / 캐비앗

재현 env·절차·캐비앗은 `capstone_constraints.md` 참조. 요약: 절대 mAP/NDS는 init 가중치+mini라 무의미(파이프라인
검증용), latency는 event-timed·batch1이라 다른 단계와 1:1 비교 불가(구조·상대만), bev_pool_v2_ext.so는 legacy env
전용(정본 emb-ai 오염 0), INT8/TRT-plugin은 다음 과제.
