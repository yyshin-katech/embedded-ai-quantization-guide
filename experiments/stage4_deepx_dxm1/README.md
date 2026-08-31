# stage4 — DEEPX DX-M1 벤더 NPU 온디바이스 실측 (Raspberry Pi 5)

4단계(멀티 SoC) 벤더-NPU 축의 **첫 물리 온디바이스 실측**. 랩 Raspberry Pi 5에 M.2로 꽂힌 **DEEPX DX-M1**(3-core, 25 TOPS INT8)으로 지연·처리량·오프로드·전력 4축을 관통했다.

핵심 장치: **같은 Cortex-A76이 4단계 CPU 폴백 프록시([`../stage5_infrastructure/cpu_proxy/`](../stage5_infrastructure/cpu_proxy/), offload 0% 바닥값)이자 DX-M1 호스트** → 같은 모델(`yolo26n` 640)로 같은 보드에서 **CPU FP32 ↔ NPU INT8**을 깨끗이 뺄셈한다.

리포트: [`../../logs/stage4_deepx_dxm1_report.html`](../../logs/stage4_deepx_dxm1_report.html)

---

## 헤드라인 (SSOT: `results/cpu_npu_comparison.json` · `dxrun_summary.json` · `y5s_summary.json`)

| 축 | 실측 | 판정 |
|----|------|------|
| **오프로드(처리량)** | NPU INT8 async 3-core **91.51 fps** vs A76 FP32 4T **8.01 fps** | **×11.42** — 벤더 NPU 압승 |
| **오프로드(지연)** | NPU E2E **29.43ms** vs CPU **124.84ms** (NPU-compute만 **8.86ms**) | E2E **×4.24** / compute **×14.08** |
| **host-bound** | async 1/2/3-core = **91.54 / 91.52 / 91.51 fps** (평평) | 3/1 스케일 **1.00** → PCIe Gen2×1 병목 |
| **perf/watt** | NPU **37.70** vs CPU **1.29** inf/s/W (host-side) | **×29.29**; 카드 TDP 포함해도 ×10–13 |

**정직한 대조**: NPU는 **INT8 전용 가속기**(FP 정밀도 사다리 없음) → 대조는 "CPU FP32(네이티브 최적) ↔ NPU INT8(가속기 유일 경로)"라는 배포 선택 축이다.

---

## 환경

- **호스트**: Raspberry Pi 5 (Cortex-A76 ×4 @2.4GHz, 8GB, Debian 13 trixie, 커널 6.18.34, Python 3.13.5)
- **NPU**: DEEPX DX-M1 (M.2 2280 Key-M, NPU 3-core @1000MHz/750mV, LPDDR5 3.92GiB, 25 TOPS INT8), FW v2.7.4 / RT Driver v2.6.0, `/dev/dxrt0`(0666)
- **PCIe**: DX-M1 네이티브 **Gen3×4**(≈4GB/s)이나 Pi 5에선 **Gen2×1**로 동작 → 유효 대역폭 ~8× 손실 (host-bound의 근본 원인)
- **런타임**: DXRT v3.4.2+d803450 (`dxrun`, `dxbenchmark`, `dxrt-cli`)
- **CPU 대조**: ONNX Runtime 1.29.0 (cp313 aarch64), `CPUExecutionProvider`, intra-op 스레드 스윕 1/2/4

## 모델 (아키텍처 일치 확인)

| 경로 | 파일 | 포맷 | 크기 | 비고 |
|------|------|------|------|------|
| NPU | `yolo26n.dxnn` | INT8 (DEEPX-compiled) | 6,890,550 B | prebuilt, ema calib 100 |
| CPU | `yolo26n.onnx` | FP32 | 9,941,928 B | ORT CPUEP |

둘 다 **base COCO yolo26n @ 640×640** (입력 [1,3,640,640] → 출력 [1,300,6]). 2차 모델 `YOLOV5S_1.dxnn`(INT8 prebuilt)은 host-bound 재확인용.

---

## 파일 구성

```
scripts/
  cpu_bench.py         # A76 CPU FP32 baseline (ORT CPUEP, batch1 지연 + 스레드 스윕)
  psample.py           # Pi5 호스트 전력 샘플러 (vcgencmd pmic_read_adc = Σ I×V, 12레일)
  analyze_profiler.py  # dxbenchmark profiler.json → 스테이지/코어별 요약
  build_summary.py     # 위 결과들 → cpu_npu_comparison.json + y5s_summary.json (SSOT 생성기)
results/
  dxrun_summary.json          # NPU 지연(seq) + 처리량(async 1/2/3core)  ← SSOT
  cpu_yolo26n_t{1,2,4}.json   # CPU 스레드 스윕
  power_{idle,cpu_load,npu_load}.json  # 호스트 전력 3상태
  cpu_npu_comparison.json     # 오프로드·전력·perf/watt 통합  ← SSOT
  y5s_summary.json            # YOLOV5S host-bound 확인  ← SSOT
  yolo26n_npu{0,01,all}.json  # profiler.json 파생(코어별 잡 분포·스테이지)
raw/
  dxrun/{lat_seq,thr_1core,thr_2core,thr_3core}.txt      # yolo26n dxrun stdout
  dxrun/y5s_{lat_seq,thr_1core,thr_3core}.txt            # YOLOV5S dxrun stdout
  yolo26n_npu{0,01,all}/{profiler.json,stdout.txt}       # dxbenchmark 원본
```

## 재현

측정은 **Pi 5 온디바이스**에서 수행. `.dxnn`은 prebuilt(dx-compiler는 x86 전용이라 Pi에서 컴파일 불가).

```bash
# --- Pi 5 위에서 ---
# NPU 지연 (순차 단일스트림)
dxrun -m yolo26n.dxnn -s -t 5 -w 1        # → raw/dxrun/lat_seq.txt
# NPU 처리량 (async, 코어 수별: -n 0=NPU_ALL, 1/2/3=코어 고정)
dxrun -m yolo26n.dxnn -b -n 3 -t 5 -w 1   # → raw/dxrun/thr_3core.txt
# dxbenchmark로 profiler.json (코어별 잡 분포)
dxbenchmark -m yolo26n.dxnn ...           # → raw/yolo26n_npuall/profiler.json

# CPU baseline (같은 보드)
python3 scripts/cpu_bench.py yolo26n.onnx 4 20 100   # → results/cpu_yolo26n_t4.json

# 전력 (부하 스크립트를 백그라운드로 돌리며 병렬 샘플)
python3 scripts/psample.py 12 npu_load    # → results/power_npu_load.json

# --- 호스트(어디서든) ---
python3 scripts/analyze_profiler.py raw/yolo26n_npuall/profiler.json yolo26n_npuall > results/yolo26n_npuall.json
python3 scripts/build_summary.py results   # → cpu_npu_comparison.json + y5s_summary.json
```

`build_summary.py`가 SSOT 2종을 결과 파일들에서 재생성하므로, 보고서 수치는 모두 추적 가능하다.

---

## host-bound 메커니즘 (온보드 프로파일러만으로 자립)

3코어 실행(1024잡)에서 코어별 잡 분포 = **695 / 427 / 2** → 코어2는 사실상 미사용. 스테이지 지연: NPU 코어 연산 8.86~9.07ms인데 **D2H(출력 전송) 21.8ms**가 파이프라인을 지배(큰 raw 출력 텐서를 Gen2×1로 빼는 비용). → 호스트가 한 코어분 일감밖에 못 흘려보냄. YOLOV5S는 연산이 **더 가벼운데도(2.6ms) 처리량이 더 낮음(~41 fps)** = host/IO-bound 확증. **천장은 DX-M1이 아니라 Pi 5의 PCIe Gen2×1.**

## 전력 계측 갭 (정직화)

`pmic_read_adc` 12레일은 전부 **Pi 5 내부**(VDD_CORE·DDR·SYS 등). M.2 DX-M1 카드는 **PMIC 상류 EXT5V**에서 전력을 끌어와 **이 측정에 미포착**. → perf/watt를 두 경계로 보고: (a) 완측 host-side(×29.29), (b) 카드 스펙 TDP(3W typ/5W max) 포함 전체-시스템 상계(×13.1/×9.57). 어느 경계든 NPU perf/watt 10배+ 승. (Orin의 "GR3D%가 DLA 못 봄"과 같은 종류의 한계.)

---

## 캐비앗

1. **NPU는 INT8 전용** — FP 정밀도 사다리는 NPU 위에 없음. 대조는 CPU FP32 ↔ NPU INT8.
2. 절대 지연·처리량·전력은 **배치1·wall-clock·prebuilt .dxnn·Pi 5 단일 호스트** 기준 → 상대 관계만 유효.
3. **host-bound는 Pi 5의 Gen2×1 탓**이지 DX-M1 천장이 아님.
4. **정확도 미측정**(prebuilt .dxnn·라벨셋 부재) → 다음 축.
5. Pi 5는 DEEPX 벤더-NPU 호스트일 뿐 **자동차 3벤더(TI/Qualcomm/Renesas)가 아님** — 수치 전이 불가.
6. DEEPX 다중 호스트 참조표(리포트 §4)는 **외부(비실측)** — 결론은 온보드 프로파일러로 자립, 참조는 보강.
