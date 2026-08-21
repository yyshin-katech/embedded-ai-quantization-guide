---
name: stage3-jetson-ondevice-hands-on
description: "3·4단계 Jetson AGX Orin 온디바이스 실측(2026-08-20, 데이터 커밋 00fd97d·문서 1a8f073 푸시완료, ResNet50): 실 trtexec 실존(stage3/5 'trtexec 부재→polygraphy' 반전)·DLA 실측(stage3 'DLA 범위 밖' 해소). DLA INT8=성능/와트 챔피언 51.29 inf/s/W(iGPU INT8의 ×1.547·전력 절반)·DLA는 INT8 전용기(FP16 13.87× 느림)·오프로드 GR3D 95%→3~16%로 증명·DLA 후보 2/2·iGPU INT8≈FP16 0.984. DLA INT8 정확도 미주장(--int8 암묵). 후속(2026-08-21, 커밋 c03c174 푸시완료): iGPU∥DLA 동시부하 = GPU-폴백 직렬화로 공짜 병렬 아님(iGPU+DLA0 60.8%·DLA 27% 붕괴, DLA0+DLA1 87.0%가 최적 66.07 inf/s/W)·nvpmodel MAXN→50W 리더 교차(iGPU -29.4% vs DLA -2.9% → 50W서 DLA +8.8%). 후속2(2026-08-21, 커밋 9ef2a58 푸시완료): 새 모델 축 DETR — 무거운 트랜스포머는 iGPU INT8 이득 부활(INT8/FP16=0.710 vs ResNet50 0.984), explicit QDQ INT8 파싱불가(3단계 case C 재현), DLA 트랜스포머 부적합(폴백 404/16조각 → iGPU FP16의 30× 느림, DLA=CNN 가속기). tech-reviewer PASS 🔴0. 후속3 정확도 축(2026-08-21, 커밋 d563439 푸시완료): 저장 .plan을 4단계 CPU-프록시 같은 ResNet50 1000장 번들에 돌려 예측 1:1 — accuracy-valid iGPU INT8 explicit-QDQ top-1 0.762=FP32>CPU MLAS 0.750, INT8 경로의존이 CPU↔가속기 넘음(TRT vs MLAS 961/1000=4단계 x86 대역·≠Pi5 100%), implicit DLA INT8 0.017 붕괴(캐비앗 정량화). tech-reviewer PASS 🔴0·🟡1(line-ref 876→892). 후속4 정확도 축 DETR(2026-08-21, 커밋 대기): 대칭 재양자화로 case C/B + 트랜스포머 self-attn quantized-const 2벽 우회→explicit-sym INT8 빌드가능(44.69MiB·11.0ms), but accuracy-valid INT8이 stage2 폭락 재현(mAP 0.4237→0.2383 −43.8%·mAP_s −84.6%, ORT/dynamic/5000 −42.9%와 교차확증)=case-C 픽스는 툴체인 언락이지 정확도 구제 아님(레버=activation 입도 SmoothQuant §4.4)·implicit 0.4073은 무통제라 미주장(순수 FP16 폴백 아님). tech-reviewer PASS 🔴1·🟡3 전부 fixed"
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

**후속 실측(2026-08-21, `embedded-guide-orchestrator` 부분 재실행 — author-5·6 직접 + tech-reviewer 팬인, 커밋
`c03c174` 푸시완료 `80d0bf6..c03c174`):** 위 캐비앗 ④("동시부하 미측정")를 닫고 nvpmodel 전력 축을 추가.
SSOT=`concurrent_power_summary.json`(6 conc JSON + power_sweep.json에서 파생, 리뷰어 raw 재산출).
- **동시부하 = 공짜 병렬 아님(GPU-폴백 직렬화):** N개 trtexec 동시 실행·20s·INT8·MAXN. **iGPU+DLA0 = ideal 합의
  60.8%뿐** — DLA가 27%로 붕괴(lat 1.28→4.75 ms)하는데 iGPU는 87.6% 유지. 범인=위 해소2의 **GPU-폴백 2층**
  (GlobalAveragePool+flatten)이 포화된 iGPU 큐 뒤에서 직렬화. **DLA0+DLA1(GPU 유휴) = 87.0%(1351.5 qps)로 스케일
  + 전 구성 최고 perf/watt 66.07 inf/s/W**(폴백층이 빈 GPU에서 즉시 실행). 3-way(1197.2) < 2-DLA(1351.5): 바쁜
  GPU가 두 DLA 폴백을 각 -75% 교살. **설계규칙: 진짜 iGPU∥DLA 병렬은 DLA 서브그래프 GPU-폴백 0층일 때만.**
- **nvpmodel 전력 스윕 = 리더 교차:** MAXN→50W에서 **iGPU -29.4% 급락(983.9→695 qps) vs DLA -2.9%뿐 → 50W서
  DLA가 iGPU보다 +8.8% 빠름**. DLA ppw는 iGPU 대비 ×1.55(MAXN)~×1.47(50W) 상시 우위. **30W/15W는 재부팅
  게이트**(온라인 CPU 코어 축소, 사용자 재부팅 거부)라 미측정·값이 50W와 동일 → 리포트에 회색 행으로 폐기 병기.
- 문서 반영: `05_tensorrt.md` §2.3 콜아웃 보강 + `06_multi_soc.md` **신설 §2-4**(승번 오염 0). 산출물
  `logs/stage3_jetson_orin_concurrent_power_report.html`(§1~6·SVG 3종) · 동 디렉토리 scripts +2(`concurrent.py`·
  `power_sweep.py`[root 실행]) · results +8 · raw +14. **tech-reviewer 팬인 PASS(🔴0)**: raw JSON 독립 재산출로
  스케일링 60.78/87.01/47.24%·DLA 붕괴 27.06%·지연 3.713×·전력델타 -29.4/-2.9/+8.8%·ppw ×1.55/×1.47 1:1.

**캐비앗(불변):** ① **DLA INT8은 `trtexec --int8` 암묵 캘리브(자동 레인지)라 지연·전력만 유효, 정확도 미주장** —
정확도는 명시적 QDQ인 iGPU INT8·3단계 RTX·4단계 CPU 프록시에서 확립. ② 지연 event-timed·batch1·MAXN → 타
단계(polygraphy·wall-clock)와 1:1 비교 불가, 상대만. ③ 전력=보드 총합(캐리어 오버헤드 6.7~8.7 W·idle floor
7.59 W·peak 42.37 W 포함). ④ iGPU+DLA 동시부하는 **위 후속 실측에서 측정 완료**(GPU-폴백 직렬화 결론은 ResNet50
GlobalAveragePool 모델 의존). ⑤ Jetson은 **NVIDIA edge**이지 세 자동차 벤더(TI/Qualcomm/Renesas) NPU 아님 —
[[stage4-qualcomm-aihub-hands-on]]가 Qualcomm HTP 담당, TI/Renesas는 보드 대기.

**후속 실측 — 새 모델 축(DETR, 2026-08-21, `embedded-guide-orchestrator` 부분수정 — author 직접 + tech-reviewer
팬인 PASS 🔴0; 커밋 9ef2a58 푸시완료):** 위는 전부 깨끗한 CNN(ResNet50). 모델을 `facebook/detr-resnet-50`
(CNN 백본 + 트랜스포머 enc/dec, 2단계 §4.5 자산 재사용)으로 **바꿔** 같은 보드·같은 trtexec·같은 플래그로 재측정.
SSOT=`experiments/stage3_tensorrt/jetson_ondevice/detr/results/detr_summary.json`. 세 발견이 위 CNN 결론을 모델 축에서
정밀화:
- **① "작은 iGPU는 INT8 무이득(0.984)"은 모델 의존이었다:** DETR iGPU FP32 25.85→FP16 13.28(×1.947)→INT8-impl
  9.43 ms(×2.742). **iGPU INT8/FP16 = 0.710**(FP16보다 ×1.41 빠름) — ResNet50 0.984(무이득)와 부호 반전. 무거운
  트랜스포머는 어텐션 대형 matmul로 **compute-bound**라 INT8 이득 부활. **이득 유무는 iGPU가 아니라 모델 연산강도가 정함.**
  (DETR은 같은 보드 ResNet50보다 지연 ×13.3/×12.9/×9.3 — 엔진 파일크기 비 아님.)
- **② accuracy-valid INT8이 파싱 불가(3단계 case C를 실 트랜스포머가 재현):** ORT `quantize_static(QInt8)` 산
  `detr_int8.onnx`(zp≠0 **1085/1485** Q·DQ + Conv/Gemm **149개 전부** INT32 bias DQ)를 trtexec 직접 파서에 넣으면
  **노드 0에서 0.051s** `Assertion failed: shiftIsAllZeros(zeroPoint)`. 3단계 §2.2.1 **case C(zp≠0)** 가 합성 아닌 실
  트랜스포머 export에서 재현. 같은 보드 ResNet50 INT8이 빌드된 건 대칭(zp=0) export였기 때문 → 같은 도구·다른 export
  정책이 가름. iGPU INT8 9.43 ms는 **implicit(자동 레인지)라 지연 전용**.
- **③ DLA에 트랜스포머 부적합(헤드라인·이론 line180/실습5 line876 "파편화" 실사례):** 같은 recipe가 ResNet50 **2/2**
  → DETR **DLA 326 / GPU폴백 404 / 16조각**. 폴백 404 = 트랜스포머 비-DLA 연산(250 SHUFFLE·34 MATRIX_MULTIPLY[Q·Kᵀ]·
  30 NORMALIZATION[LayerNorm NVDLA v2 미지원]·12 SELECT[마스크 Where]·66 CONSTANT·12 UNARY, 합 404). **16조각은 단순
  파편이 아니라 NVDLA v2 하드 한계**(`DLA supports only 16 subgraphs per DLA core` 경고로 슬롯 포화→나머지 GPU로 밀림,
  리뷰어 발견). 결과 **DLA FP16 398.6 ms = iGPU FP16의 30×**; DLA INT8(78.5 ms)로 켜도(남은 섬서 "DLA=INT8 전용기"
  규칙 유지·×5.08) 여전히 iGPU INT8의 8.3× 느림. **설계규칙: DLA는 DLA 서브그래프 GPU-폴백 ~0일 때만 이득 — DLA는 CNN
  가속기, 트랜스포머 detection 헤드는 iGPU에.** (폴백 55.3%=404/730 → 이론 "20~30% 폴백이면 이점 소멸" 극단 사례.)

DETR 캐비앗: iGPU/DLA INT8 implicit라 지연·크기만 유효·정확도 미주장 → DETR 정확도는 온보드 미측정(COCO 부재), **2단계
RTX 인용** FP32 mAP 0.4207→INT8 0.2402(−42.9%)·mAP_s −77%([[stage2-detr-hands-on]]). 문서 반영: `05_tensorrt.md`
§2.3에 DETR 콜아웃 순수삽입(**git diff 8/0 = 승번 오염 0**, §2.4 헤더 보존)·HTML 재렌더. 산출물
`logs/stage3_jetson_orin_detr_report.html`(§1~6·SVG 3종) · `experiments/stage3_tensorrt/jetson_ondevice/detr/`
(detr_bench.py·detr_summary.json·raw 로그 9·README). tech-reviewer 팬인 PASS: 하드SSOT+verbose 5.9MB op히스토그램
재카운트(404=250+66+34+30+12+12)+case-C 로그 원문+onnx 인트로스펙션(1085/1485·149/149) 전건 1:1, 🟡1(지연비 "무겁다"
오독소지) 리뷰 후 수정. **다음 후보:** DETR 대칭 재양자화로 explicit INT8 정확도유효 엔진 · on-board COCO mAP · SmoothQuant(2단계 §4.4)로 activation 입도 개선 후 재측정.

**후속 실측 — 정확도 축(ResNet50, 2026-08-21, 커밋 d563439 푸시완료, `embedded-guide-orchestrator` 부분수정 — author-5·6 직접 +
tech-reviewer 팬인 PASS 🔴0·🟡1 리뷰어 직접수정):** 위 실측(솔로·동시부하·DETR)이 전부 지연·전력이라 매번
"정확도 미주장"을 달았음 → 이번엔 저장된 `.plan`을 4단계 CPU-프록시([[stage4-arm-cpu-fallback-proxy]] 커밋
49e30ff)가 쓴 **같은 ResNet50 1000장 번들**에 돌려 이미지별 예측 1:1 대조. 결정적 장치: `rn50_gpu_int8.plan`이
CPU 프록시와 **같은 `resnet50_int8_qdq.onnx`**에서 빌드 → QDQ scale 동일 → 유일 변수 = 정수커널 데이터패스.
SSOT=`experiments/stage3_tensorrt/jetson_ondevice/accuracy/results/accuracy_summary.json`.
- **① accuracy-valid INT8 온디바이스 유지 + CPU 커널 이김:** iGPU INT8(explicit QDQ 실 scale) top-1 **0.7620 =
  iGPU FP32**(1000장 무손실), CPU MLAS INT8 **0.7500**보다 높음 → 위 캐비앗의 "정확도는 explicit QDQ iGPU
  INT8에서 확립"을 **온디바이스 입증**. 무음 FP32 폴백 아님(자기 FP32와 예측 57장 상이·지연 1.01≠1.94ms).
- **② INT8 예측 경로의존이 CPU↔가속기 경계 넘음(4단계 확장):** TRT INT8(GPU 정수커널) vs MLAS INT8(CPU SDOT) =
  **961/1000**(39장, 같은 QDQ scale·같은 실리콘). 96.1%는 4단계 Jetson↔x86(958=95.8%)과 **같은 대역**·Jetson↔Pi5
  100%(같은 MLAS SDOT) 아님 → **정수커널 다르면 ~96%, 경계가 CPU↔CPU든 CPU↔가속기든 무관**. 39장 중 TRT 정답
  15·MLAS 정답 3·둘다오답 21(net +12 = 0.762−0.750). FP32는 경로 무관 = iGPU FP32 vs CPU FP32 **1000/1000**
  (4단계 FP32 100% 가속기까지 확장). [[stage4-arm-cpu-fallback-proxy]] 크로스플랫폼 표의 CPU↔가속기 짝.
- **③ implicit DLA INT8 = 0.017 붕괴(캐비앗 정량화):** `rn50_dla_int8.plan`(=`--int8` 캘리브 없음·자동레인지)
  top-1 **0.0170**, **같은 하네스**가 DLA FP16 0.7610을 주므로 하네스버그 아님 → implicit/DLA INT8 "정확도 미주장"
  라벨을 수치 확증. accuracy-valid INT8은 explicit-QDQ iGPU 엔진뿐.
- 부수: iGPU FP16 999/1000·DLA FP16 0.7610(998/1000 vs iGPU FP16)·**"top-1 불변 ≠ 예측 불변"**(INT8 vs 자기 FP32
  57장 플립이나 label 대비 net-neutral, pred_cls가 정직한 산출물).

문서 반영: `05_tensorrt.md` §2.3 콜아웃 **author 순수삽입(8/0)**·`06_multi_soc.md` §2-3 bullet+📄 링크행(2/1).
**tech-reviewer 팬인 PASS(🔴0·🟡1·🟢A~E):** raw pred_cls를 numpy로 독립 재산출 전건 일치(top-1 7종·일치율
7종·헤드라인 39=15+3+21·net+12·first_indices)·SVG 비례(top1×600·flips×9·0-flip sliver 정직)·긴 한글 §2-2 슬러그
bit-identical·캐비앗 4종·헤딩 무renumber; 🟡1 = 기존 DETR 콜아웃 `실습5 line 876`→**892** 정정(876이 `PY`
heredoc 종료자에 착지, 내 +8삽입이 드리프트 8→16행 키움·형제참조 180/187/193 정밀 → 리뷰어 직접수정+05.html
재렌더, **net 05 = 9/1**[+1 delete = 이 수정 한 줄]). 산출물 `logs/stage3_jetson_orin_accuracy_report.html`
(§1~5·SVG 2종) · `experiments/stage3_tensorrt/jetson_ondevice/accuracy/`(`orin_accuracy.py` 온보드 러너
[tensorrt Python API + `cuda.bindings.runtime`·batch1 `execute_async_v3`]·`analyze_accuracy.py` 호스트
교차대조·results[5 per-engine pred_cls + accuracy_summary.json SSOT + meta + rpi_labels.npy]·raw·README).
캐비앗: 절대 top-1은 1000장 서브셋(1단계 함정0 부풀림)이라 상대만·`dla_int8` implicit 정량화용·QDQ scale 3단계
동결. **다음 후보(불변):** DETR 대칭 재양자화 → case-C 우회 → explicit INT8 정확도유효 엔진 · on-board COCO mAP.

**후속 실측 — 정확도 축 DETR(2026-08-21, `embedded-guide-orchestrator` 부분수정 — author 직접 + tech-reviewer 팬인
PASS 🔴1·🟡3 전부 리뷰어 직접수정; 커밋 대기 — 규약상 요청 시만):** 위 후속3은 깨끗한 CNN(ResNet50)이었고 DETR은
후속2에서 explicit QDQ INT8이 case C로 파싱 불가라 "정확도 미측정" 캐비앗을 달았음 → 그 **후속3 "다음 후보"를 실행**:
대칭 재양자화로 case C를 우회해 accuracy-valid INT8 엔진을 빌드하고 온보드 COCO mAP 측정.
SSOT=`experiments/stage3_tensorrt/jetson_ondevice/detr_accuracy/results/detr_accuracy_summary.json`. eval=COCO val2017
head 1000 · fixed 800×1066 · pycocotools bbox mAP(보드는 raw logits/boxes만 dump, 호스트가 mAP).
- **① 두 벽 → 툴체인 언락(빌드 가능):** (1) case C(zp≠0 `shiftIsAllZeros`)+case B(INT32 bias DQ)를 **대칭 재export**로
  제거 — `ActivationSymmetric+WeightSymmetric`(zp=0 전역→C 해소)·`QuantizeBias=False`(INT32 bias DQ 0→B 해소)·conv1
  제외(case D). (2) **트랜스포머 고유 2차 벽(빌더)**: ORT 기본은 모든 op 양자화 → 대칭이어도 빌더가 self-attention의
  `Constant_3_output_0_quantized`를 거부(`qdqGraphOptimizer::matchQuantizedConstantPluginOrDQ`: "Quantized constant is
  only allowed before DQ or PLUGIN_V2/V3") → `op_types_to_quantize=[Conv,Gemm]`(weight-bearing op만; attention matmul·
  LayerNorm·Softmax는 FP16 — 2단계 §4.5 정합). 결과 `detr_int8_sym.onnx`(43.4MB·zp_nonzero=0·int32_bias_DQ=0) →
  **9ef2a58서 build-failed였던 explicit INT8이 빌드됨**(44.69 MiB·11.002 ms).
- **② 헤드라인(가설 반전) — case-C 픽스는 툴체인 언락이지 정확도 구제 아님:** accuracy-valid explicit-sym INT8이
  **stage2 DETR 폭락을 다른 경로로 재현** — FP32 mAP 0.4237→INT8 **0.2383(−43.8%)**·mAP_s 0.2179→**0.0336(−84.6%)**.
  stage2(ORT-QDQ·dynamic·CUDA EP·5000장) 0.4207→0.2402(−42.9%)·mAP_s −77%와 **교차확증**(TRT/fixed/1000 vs
  ORT/dynamic/5000 두 경로가 같은 폭락). **레버는 sym/case-C 픽스가 아니라 activation 양자화 입도**(SmoothQuant,
  2단계 §4.4가 per-tensor gap의 59.9% 회복). FP16 무손실(+0.0006, 0.4243).
- **③ implicit `--int8` 메커니즘 정정(작업 중 자기교정한 정합성 버그):** implicit(캘리브 없음) mAP 0.4073(−3.9%)이
  sym(0.2383)보다 높다고 "implicit이 더 나은 INT8"이 아님 — 캘리브 캐시 없음→**데이터 유래 activation 레인지
  없음(무통제)**이라 정확도 미주장. **단 "순수 FP16 폴백"은 아님**: implicit 지연 9.43 ms < FP16 13.28 ms이고 엔진
  58.76 < 81.43 MiB → INT8 커널이 실제로 돎(순수 FP16 폴백이면 더 느리고 커야 함). 여기선 우연히 FP16 근처(0.4073)에
  착지했을 뿐, 같은 무통제 경로 companion(후속3 ResNet50 DLA implicit)은 **0.017 붕괴**. accuracy 있는 건 explicit·
  calibrated QDQ 엔진뿐. (divergence corr은 약한 프록시: sym 0.982 ≈ implicit 0.980인데 mAP 0.17 격차 — no-object
  logit 차원이 지배 → pycocotools mAP가 authoritative.)

지연 사다리(batch1 GPU-compute median·MAXN·동일 플래그): FP32 25.85 / FP16 13.28(×1.947) / **sym 11.002(×2.35 vs
FP32·×1.207 vs FP16, accuracy-valid)** / implicit 9.43(×2.742, 미주장) ms · 엔진 160.12/81.43/**44.69**/58.76 MiB.
**tech-reviewer 팬인 PASS(🔴1·🟡3·🟢3 전부 fixed·재검증):** raw npz에서 mAP 독립 재계산(`analyze_detr_map.py`)·SSOT
byte-identical 재생성·산술(×1.947·×2.35·×2.742·−43.8%·−84.6%)·SVG 비례(mAP factor 1060·latency factor 18.57)·numstat
8/0+1/0·heading diff 0. 🔴1 = sym 엔진크기 **단위모순**(6곳 "46.9 MiB"[=bytes/1e6 십진MB]가 표·SSOT의 "44.69
MiB"[bytes/1048576]와 자기모순 → 전부 44.69 MiB 통일). 🟡 = README 잔존 "FP16 폴백" 프레이밍·05 콜아웃 raw `<u>`
리터럴 렌더·리포트 §4.4 슬러그 오류, 전부 수정.

문서 반영: `05_tensorrt.md` §2.3 DETR 정확도 콜아웃 **순수삽입(8/0)** + HTML 재렌더(1/0)·**승번 오염 0**(heading 무변).
산출물 `logs/stage3_jetson_orin_detr_accuracy_report.html`(§1~6·SVG 2종) ·
`experiments/stage3_tensorrt/jetson_ondevice/detr_accuracy/`(scripts 6[`detr_sym_export`·`detr_prep`·`prep_coco_sub`·
`orin_detr_map`·`analyze_detr_map`·`build_summary`]·results·raw·README). 캐비앗: 절대 mAP는 fixed 800×1066(vs stage2
dynamic)·TRT vs ORT·1000 vs 5000장이라 stage2와 1:1 비교 불가 — **상대 폭락만** 결과(그 상대 폭락이 stage2
교차확증)·1000장 서브셋(1단계 함정0)·implicit 미주장·Jetson≠자동차 벤더. **이로써 DETR "정확도 미측정" 캐비앗
닫힘.** 다음 후보: DETR에 SmoothQuant/per-token(2단계 §4.4)로 activation 입도 개선 후 온디바이스 재측정(폭락 회복 여부).
