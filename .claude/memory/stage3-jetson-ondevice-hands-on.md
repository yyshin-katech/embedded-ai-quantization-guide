---
name: stage3-jetson-ondevice-hands-on
description: "3·4단계 Jetson AGX Orin 온디바이스 실측(2026-08-20, 데이터 커밋 00fd97d·문서 1a8f073 푸시완료, ResNet50): 실 trtexec 실존(stage3/5 'trtexec 부재→polygraphy' 반전)·DLA 실측(stage3 'DLA 범위 밖' 해소). DLA INT8=성능/와트 챔피언 51.29 inf/s/W(iGPU INT8의 ×1.547·전력 절반)·DLA는 INT8 전용기(FP16 13.87× 느림)·오프로드 GR3D 95%→3~16%로 증명·DLA 후보 2/2·iGPU INT8≈FP16 0.984. DLA INT8 정확도 미주장(--int8 암묵)"
metadata: 
  node_type: memory
  type: project
---

3·5단계는 정본 pip 휠에 `trtexec`가 없어 polygraphy로 우회했고([[stage3-tensorrt-hands-on]]·
[[stage5-infrastructure-hands-on]]), 3단계 DLA는 dGPU라 `num_DLA_cores=0`으로 범위 밖이었다. 사용자 요청 실기
**NVIDIA Jetson AGX Orin Dev Kit(64GB)**(JetPack 6.2.1/L4T R36.4.3 · CUDA 12.6 · **TRT 10.3.0.30** · cuDNN 9.3.0 ·
12× Cortex-A78AE · **2× NVDLA v2** · nvpmodel MAXN, 접속 [[jetson-agx-orin-board-access]])에서 두 한계를 모두 해소.
같은 3·5단계 ResNet50 자산(iGPU INT8 = explicit QDQ, stage3 real scales · [[machine-ai-lap-rtx3080]]가 RTX 대조).
SSOT=`experiments/stage3_tensorrt/jetson_ondevice/results/summary.json`.

**해소 1 — 실 trtexec 실존(stage3/5 반전):** JetPack 동봉 **`/usr/src/tensorrt/bin/trtexec`**(배너 `v100300`)가
실존 → 문서의 원 trtexec 명령을 그대로 관통. 정정: TRT 10.3.0 빌드전용 플래그는 `--buildOnly`가 아니라
**`--skipInference`**(전자는 exit 1 `Unknown option`).

**해소 2 — DLA 5엔진 실측(stage3 'DLA 범위 밖' 해소; timing=trtexec GPU-compute median event-timed·batch1,
power=tegrastats board-total steady):**
| 엔진 | 지연 | vsFP32 | steadyW | GR3D% | inf/s/W |
|---|---|---|---|---|---|
| iGPU FP32 | 1.9375 ms | ×1.00 | 42.26 | 98 | 12.19 |
| iGPU FP16 | 1.0293 ms | ×1.88 | 36.20 | 96 | 26.77 |
| iGPU INT8 | 1.01318 ms | ×1.912 | 29.70 | 95 | 33.16 |
| DLA FP16 | 17.7344 ms | ×0.109 | 15.69 | 3 | 3.59 |
| DLA INT8 | 1.27832 ms | ×1.516 | 15.19 | 16 | **51.29** |

**핵심 판정 4:**
- **DLA INT8 = 성능/와트 챔피언:** 51.29 inf/s/W = iGPU INT8(33.16)의 **1.547×**, 전력 **0.511×(절반, 15.19 vs
  29.70 W)**, 지연만 1.262× 느림 → 전력·GPU-여유 목적이면 DLA INT8, 순수 최저지연이면 iGPU INT8.
- **DLA는 INT8 전용기:** 레이어 배치가 INT8·FP16 완전 동일한데 **DLA FP16이 13.87× 느림**(17.73 vs 1.28 ms) →
  원인은 배치 아닌 순수 NVDLA v2 데이터패스(INT8 MAC 처리량 ≫ FP16). DLA FP16은 iGPU FP32(1.94 ms)보다도 9배
  느린 최악 조합. **DLA에 올릴 거면 반드시 INT8.**
- **오프로드가 수치로 증명:** DLA-후보 **2/2 오프로드** = ForeignNode 2개 [(1) conv 백본 `/conv1/Conv‥/layer4.2/
  relu_2` **120 DLA층**, (2) `/fc/Gemm`+bias ElementWise], GPU 폴백은 compute **2층뿐**(GlobalAveragePool REDUCE —
  DLA에 AVG-reduce 없음 + flatten SHUFFLE), FP16·INT8 동일 → 이론의 "20~30% 폴백이면 이점 소멸" 경계 훨씬 아래.
  tegrastats GR3D가 iGPU 95%→DLA 3~16%로 붕괴해 연산이 DLA에서 실제로 돎을 확증. 단 **DLA는 GR3D에 거의 안
  잡히므로** 전력하네스 부하검출은 GR3D 아닌 전력 임계(idle×1.20)로 자기보정.
- **작은 Ampere iGPU는 INT8≈FP16:** iGPU INT8/FP16 지연비 **0.984**(거의 무이득) — 작은 iGPU는 launch/memory-bound라
  INT8 연산이득이 안 드러남(RTX 3080은 0.927로 조금 벌었음, [[stage3-tensorrt-hands-on]]). iGPU와 DLA 정반대.

**문서 반영(2026-08-20, `embedded-guide-orchestrator` 부분 재실행 — author-5·6 직접 + tech-reviewer 서브에이전트
팬인):** ① `05_tensorrt.md` §2.3 DLA에 🔬 실측 콜아웃(판정 4·실 trtexec·flag 정정) ② `06_multi_soc.md` **신설
§2-3**(가속기 실측: iGPU 정밀도 사다리 + DLA 오프로드/성능·와트) + §2-2 sign-flip 표 **4번째 행 Jetson A78AE**
(캐리오버, [[stage4-arm-cpu-fallback-proxy]] CPU-프록시 데이터 커밋 49e30ff) + 크로스플랫폼 발견 경로-의존 정밀화.
승번 오염 0(05 +11/-0 · 06 +27/-6=라인수정, 섹션삭제 0). HTML 재렌더. **문서 커밋 1a8f073**(main, 푸시완료
00fd97d..1a8f073).

**tech-reviewer 팬인 PASS(🔴0·material🟡0·🟢2):** summary.json + cpu_proxy 8-JSON SSOT 1:1 · 크로스플랫폼 raw
pred_cls 독립 재계산 · 산술 7관계(×2.11·1.547×·0.511×·1.262×·13.87×·0.984·9.15×) · DLA 배치 raw verbose 로그
확인(2 ForeignNode + 2 GPU폴백, 초기 스크래치의 "6 blocks" 오류 없음) · 앵커 실재(05 id
`23-dla-deep-learning-accelerator` · 06 2링크 · 05 역링크) · 캐비앗 병기.

**산출물(데이터 커밋 00fd97d 푸시완료):** `logs/stage3_jetson_orin_ondevice_report.html`(§1~6·지연/전력/성능와트
SVG) · `experiments/stage3_tensorrt/jetson_ondevice/`(scripts · results/summary.json · raw verbose 로그 · README ·
constraints).

**캐비앗(불변):** ① **DLA INT8은 `trtexec --int8` 암묵 캘리브(자동 레인지)라 지연·전력만 유효, 정확도 미주장** —
정확도는 명시적 QDQ인 iGPU INT8·3단계 RTX·4단계 CPU 프록시에서 확립. ② 지연 event-timed·batch1·MAXN → 타
단계(polygraphy·wall-clock)와 1:1 비교 불가, 상대만. ③ 전력=보드 총합(캐리어 오버헤드 6.7~8.7 W·idle floor
7.59 W·peak 42.37 W 포함). ④ iGPU+DLA 동시부하(진짜 병렬 오프로드)는 미측정. ⑤ Jetson은 **NVIDIA edge**이지 세
자동차 벤더(TI/Qualcomm/Renesas) NPU 아님 — [[stage4-qualcomm-aihub-hands-on]]가 Qualcomm HTP 담당, TI/Renesas는
보드 대기.
