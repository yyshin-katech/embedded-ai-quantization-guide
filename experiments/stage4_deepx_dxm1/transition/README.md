# stage4 — DX-M1 레짐 전이 곡선 (compute-bound ↔ D2H-bound)

크로스오버 축([`../crossover/`](../crossover/))의 **후속·완성 축**. 크로스오버 축은 **모델만 바꿔**(resnet50 vs yolo26n vs YOLOV5S) 두 레짐이 **양 극단에 존재함**을 증명했으나, 캐비앗 #3에 스스로 남긴 한계는 **"전이 곡선은 아님(중간 출력 미측정)"** 이었다 — 두 점만 찍었지 그 사이 곡선을 긋지 못했다. 이 축은 **출력(D2H) 크기 하나만 연속으로 스윕**해 compute-bound에서 D2H-bound로 넘어가는 **전이 곡선 전체를 추적**해 그 캐비앗을 닫는다.

핵심 장치(**변수 격리**): **고정된 무거운 conv 트렁크**(body_depth=18, 8.923 GMACs, Inference ~2.70ms, 전부 NPU)가 연산을 상수로 잡고, **4채널 병목 + 1×1 "확장" 헤드**(`Conv2d(4→C_out, 1×1)`)가 출력 텐서 `[1,C_out,14,14]` fp32 ≈ 784·C_out 바이트만 키운다 → **C_out을 스윕하면 연산은 고정한 채 D2H(출력) 크기만** 3.83KB→3.82MB(1020×)로 움직인다. dxbenchmark는 입력을 합성하므로 **지연·레짐만 유효**(랜덤 weight → 정확도 주장 없음).

리포트: [`../../../logs/stage4_deepx_dxm1_transition_report.html`](../../../logs/stage4_deepx_dxm1_transition_report.html)

---

## 헤드라인 (SSOT: `results/transition_summary.json`)

### (1) 전이 곡선 추적 — 고정 연산에서 출력만 키우니 코어-스케일이 **2.98×→1.00× 단조 하강**

12개 합성 모델, **Inference 1코어 p50 = 2.677–2.809ms**(평균 2.704, spread **4.9%**)로 사실상 고정. H2D도 0.466–0.559ms 고정. 오직 D2H만 스윕:

| C_out | 출력 | Inf 1c | D2H 1c | **3c/1c** | 잡 분포% | single-inf | 스케일링 |
|-------|------|--------|--------|-----------|----------|------------|----------|
| 5 | 3.83 KB | 2.677 | 0.141 | **2.982** | 33/33/33 | compute | near-linear |
| 20 | 15.3 KB | 2.678 | 0.147 | **2.965** | 33/33/33 | compute | near-linear |
| 41 | 31.4 KB | 2.677 | 0.174 | **2.985** | 33/33/33 | compute | near-linear |
| 82 | 62.8 KB | 2.679 | 0.319 | **2.987** | 33/33/33 | compute | near-linear |
| 163 | 124.8 KB | 2.682 | 0.433 | **2.429** | 33/33/33 | compute | near-linear |
| 327 | 250.4 KB | 2.687 | 0.776 | **1.985** | 35/32/33 | compute | sub-linear |
| 490 | 375.2 KB | 2.692 | 1.016 | **1.871** | 33/33/33 | compute | sub-linear |
| 653 | 500.0 KB | 2.700 | 1.367 | **1.595** | 36/31/33 | compute | sub-linear |
| 980 | 750.3 KB | 2.704 | 1.980 | **1.492** | 46/45/9 | compute | sub-linear |
| 1276 | 977.0 KB | 2.711 | 2.429 | **1.211** | 50/48/2 | compute | sub-linear |
| 2551 | 1953.1 KB | 2.748 | 8.892 | **0.996** | 93/6/0 | **D2H** | flat |
| 5102 | 3906.2 KB | 2.809 | 17.708 | **1.000** | 98/2/0 | **D2H** | flat |

→ 연산은 고정인데 **3코어 처리량 이득이 2.98×(near-linear)에서 1.00×(flat)로 매끄럽게 무너진다**. 동시에 3코어 잡 분포가 **33/33/33 → 98/2/0**으로 이동 — 출력이 커질수록 공유 PCIe 링크가 코어 1·2를 굶기고 코어 0이 링크를 독점한다(D2H-bound 축 yolo26n 472/28/2와 같은 시그니처). single-inference 레짐(한 번의 추론을 무엇이 제한하나)은 **D2H p50 = 고정 Inference(2.70ms)** 지점, 즉 출력 **~1.05MB**에서 compute-bound→D2H-bound로 넘어간다. 크로스오버 축이 찍은 두 극점(resnet50 4KB compute-bound / yolo26n 2.82MB D2H-bound) **사이의 곡선을 실제로 그은 것** = 캐비앗 #3 닫힘.

### (2) 레짐 경계는 **칼날(knife-edge)이 아니라 폭 ~3.36× 밴드** — 그 폭 = N_cores

전이에는 **두 개의 특성 문턱**이 있고, 둘의 비가 밴드 폭을 정한다:

| 문턱 | 조건 | 출력 크기 | 무슨 일 |
|------|------|-----------|---------|
| **링크 포화 시작** | D2H = Inference / 3 | **~319 KB** | 3코어를 하나의 링크가 더는 다 못 먹임 → 3c/1c 스케일링이 깨지기 시작 |
| **고유 크로스오버** | D2H = Inference | **~1.05 MB** | 단일 추론이 D2H-bound로 전환 |

→ 밴드 폭 = 1.05MB / 319KB = **3.36× ≈ N_cores(3)**. 우연이 아니다 — **N개 코어가 하나의 D2H 링크를 공유**하므로, 단일 추론을 묶는 per-core D2H의 **1/N** 지점에서 이미 멀티코어 처리량이 포화한다. D2H 선형 구간 기울기 **2.456 ms/MB**(intercept 0.137ms, C_out≤1276)로 두 문턱을 산출. 이는 크로스오버 축(274bffa)을 **반증이 아니라 정밀화**한다 — "regime이 갈린다"는 이진 결론에, 경계는 **폭을 가진 밴드이고 그 폭이 코어 수라는** 정량 구조를 더한다.

---

## 변수 격리 설계 (왜 이 결론이 confound-free인가)

- **고정 변수 = 연산·입력**: 트렁크(body_depth=18, 8.923 GMACs)가 12개 모델에서 **동일** → Inference 1c spread 4.9%·H2D spread ≈고정. 유일하게 움직이는 건 헤드 확장기의 출력 채널.
- **스윕 변수 = 출력(D2H)만**: `[1,C_out,14,14]` fp32 = 784·C_out B. C_out ∈ {5…5102} → 출력 3920…3999968 B(**1020.4×**). D2H 시간은 126×만 늘어남(작은 출력에선 ~0.14ms 전송 오버헤드 바닥이 지배 → 출력 크기보다 완만).
- **전부 NPU(호스트 연산 confound 0)**: 12개 모델 모두 `groups = [["1","NPU"],["0","CPU"]]` = **1 NPU 그룹 / 0 CPU 그룹**. DETR 축(트랜스포머를 호스트 CPU FP32로 자동분할)과 달리 여기 D2H는 **순수 device→host 출력 전송**이지 host-compute가 아니다.

**스테이지 측정 규약(크로스오버 축과 동일)**:
- **D2H p50은 고유값** — 1코어와 3코어가 거의 동일(링크는 코어 수와 무관하게 하나). 표의 D2H는 1코어 값.
- **Inference p50은 1코어(비경합)에서 읽는다** — 3코어 Inference는 **전이 구간(C_out 163–653)에서 3.28–4.34ms로 부풀려진다**(출력 핸드오프 backpressure). 이는 **연산 증가가 아니라 처리량 아티팩트**이므로, "고정 연산"의 근거는 1코어 값이 정본.
- **`-n`은 코어 개수가 아니라 코어 ID** — 카운트 스윕은 `-n 1`(1코어 NPU_0) / `-n 4`(2코어 NPU_0/1) / `-n 0`(3코어 NPU_ALL).

---

## 파일 구성

```
scripts/
  build_transition_models.py  # (x86) 합성 모델 생성: Trunk(cmid=256, body_depth=18) + 4ch 병목
  #                             + 1×1 확장 헤드(4→C_out) → ONNX(opset13) → dx_com 컴파일 → manifest.json
  run_transition_bench.sh     # (Pi) 스윕 하네스: 12 C_out × 3 코어수(-n 1/4/0) dxbenchmark -v
  #                             → raw/<model>/analyzed_n{1,2,3}.json + raw/corescale.csv
  build_transition_summary.py # (호스트) raw+manifest 축약 → SSOT transition_summary.json
results/
  transition_summary.json     # ← SSOT (고정연산 검증·전이·두 문턱·밴드폭·행별 스테이지)
raw/
  corescale.csv               # 12행 fps_1c/2c/3c + scale_3c_1c/2c_1c
  tr_c<cout>/analyzed_n{1,2,3}.json  # 코어수별 스테이지 p50 + per-core 잡 분포
  tr_c<cout>/profiler_n3.json        # 3코어 원시 프로파일러(대용량, gitignore)
dxnn/
  manifest.json               # 12 모델 컴파일 메타(out_shape·nominal_out_bytes·trunk_gmacs·groups)
  tr_c<cout>/*.dxnn           # 컴파일 산출물(~23.4MB×12, gitignore)
onnx/  logs/                   # ONNX·컴파일 로그(대용량, gitignore)
```

프로파일러 분석기(`analyze_profiler.py`)는 크로스오버 축 것을 Pi에서 재사용(스테이지 p50 + per-core 잡 분포 추출).

## 재현

```bash
# --- x86 (AI-LAP, dxcom-venv) — 합성 모델 생성 + 컴파일 ---
emb-ai/bin/python scripts/build_transition_models.py \
  --couts 5,20,41,82,163,327,490,653,980,1276,2551,5102 \
  --onnx-dir onnx --dxnn-dir dxnn --log-dir logs --body-depth 18
#   → onnx/tr_c<cout>.onnx (12개) → dx_com 컴파일 → dxnn/tr_c<cout>/ + dxnn/manifest.json
#   트렁크 고정(body_depth=18, 8.923 GMACs), 헤드 확장기만 C_out 스윕(출력 크기 격리)

# --- Pi 5 온디바이스 (venv-dx-runtime) — 코어수 × 출력 스윕 ---
#   dxnn/ + analyze_profiler.py 를 ~/dxm1_transition/{models,} 로 scp 후:
bash run_transition_bench.sh 5 2      # TIME=5s, WARMUP=2
#   → 12 C_out × {1,2,3}코어 dxbenchmark → raw/<model>/analyzed_n{1,2,3}.json + corescale.csv

# --- 호스트 축약 (SSOT) ---
python scripts/build_transition_summary.py   # raw + manifest → results/transition_summary.json
```

`onnx/`·`dxnn/`·`logs/`·`raw/*/profiler_n3.json`은 **미커밋**(대용량/재생성 정책). SSOT(`transition_summary.json`) + `raw/analyzed_n*.json`·`corescale.csv` + `dxnn/manifest.json`이면 리포트·가이드·논문을 전부 재현.

---

## 캐비앗

1. **합성 랜덤-weight 모델** → **지연·레짐 결론만 유효**, 정확도는 주장하지 않음(입력도 dxbenchmark 합성). 이 축의 목적은 병목 곡선 추적이지 모델 품질이 아니다.
2. **batch1 · Pi5 PCIe Gen2×1** → D2H-bound·두 문턱 위치는 **Pi 5 링크 성질**(네이티브 Gen3×4면 밴드가 오른쪽으로 이동, 미측정). 크로스오버·검출 축과 동일 근본 — **밴드 폭 = N_cores** 관계는 링크 대역과 무관(코어가 링크 공유하는 구조적 사실)하나, 밴드의 **절대 위치**(319KB·1.05MB)는 Gen2×1 대역 의존.
3. **3코어 Inference 부풀림(전이 구간 3.28–4.34ms)은 처리량 아티팩트**(출력 핸드오프 backpressure)이지 연산 증가가 아님 — 고정연산 근거는 1코어 p50(2.70ms, spread 4.9%).
4. **두 문턱은 선형 fit(2.456 ms/MB) 외삽** — D2H는 ~2MB 초과에서 super-linear로 꺾이므로 fit 구간은 C_out≤1276(≤1MB). 문턱은 "D2H=Inference/3" / "D2H=Inference" 정의의 산출값이지 직접 측정점이 아님(밴드 폭 3.36×는 두 외삽의 비).
5. **모델 의존**: "regime을 정하는 건 출력 크기"는 이 툴체인(dx_com이 헤드에서 그래프 절단, decode/NMS는 호스트)과 conv 트렁크 계열에서의 결론 — 크로스오버·검출 축과 정합.
6. **Pi5는 DEEPX DX-M1 호스트일 뿐 자동차 3벤더(TI·Renesas·Qualcomm) 아님** — 이 전이 곡선은 DX-M1 벤더 NPU + Pi5 호스트 조합의 레짐 성질이지, 가이드 06 §2의 자동차 SoC 3벤더 parity 주장이 아니다(크로스오버·검출·DETR 축과 동일 스코프).
