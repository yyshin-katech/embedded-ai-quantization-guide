# DX-M1 크로스오버 축 — host-bound ↔ NPU-bound

직전 벤치(`../` = `stage4_deepx_dxm1`)의 캐비앗 #3("host-bound은 Pi 5의 PCIe Gen2×1 탓이지
DX-M1 천장이 아님")을 **반대 사례로 닫는다.** 같은 DX-M1·같은 Pi 5·같은 `dxrun`/`dxbenchmark`에서
**모델만 바꿔** 병목 스테이지가 갈리는 크로스오버를 실측.

## 헤드라인 (SSOT: `results/crossover_summary.json`)

| 모델 | 출력 bytes | 연산 p50 | D2H p50 | regime | 코어스케일 3c/1c | 3코어 fps |
|------|-----------|---------|---------|--------|-----------------|-----------|
| **ResNet50** | 4,000 | 2.77ms | **0.11ms** | **compute-bound** | **2.19×** | 1078.93 |
| yolo26n | 2,822,400 | 9.00ms | **21.81ms** | D2H-bound | 1.00× | 91.19 |
| YOLOV5S | 5,483,520 | **2.59ms**(최경량) | 21.14ms | D2H-bound | 1.02× | 41.01 |

- **천장은 모델이 정한다**: resnet50(출력 4KB)은 D2H 무시가능 → NPU 연산이 병목 → 3코어 2.19× near-linear 스케일.
  yolo26n(출력 2.82MB)은 D2H 21.81ms가 연산 9.0ms를 지배 → 코어 늘려도 1.00× 평평.
- **코어 잡 분포가 증거**: resnet50 961/910/1010(3코어 균등) vs yolo26n 472/28/2(공유 PCIe가 코어1·2 굶김).
- **변수 격리(YOLOV5S)**: 연산이 resnet50보다 **가벼운데도**(2.59<2.77ms) 출력이 크면 D2H-bound → 26× 느림.
  ⇒ regime을 정하는 건 연산이 아니라 **출력(D2H) 크기**.
- **buffer-count 대조**: resnet50 bc2→4서 641→1108 급등(NPU 여유), yolo26n 91서 포화(물리적 D2H 벽).

## 환경

- **호스트/NPU**: Raspberry Pi 5 (Cortex-A76 ×4, Debian 13) + DEEPX DX-M1 (3코어 @1000MHz), **PCIe Gen2 X1**
- **런타임**: DXRT v3.4.2+d803450 (`dxrun`, `dxbenchmark`)
- **모델**: resnet50_native.dxnn(정확도 축 재사용) · yolo26n.dxnn(벤치와 동일 6890550 B) · YOLOV5S_1.dxnn(dx_rt 예제)

## `-n` 코어 매핑 (중요)

`-n`은 코어 **개수**가 아니라 코어 **ID**: `0=NPU_ALL · 1=NPU_0 · 2=NPU_1 · 3=NPU_2 · 4=NPU_0/1 · 5=NPU_1/2 · 6=NPU_0/2`.
코어-**카운트** 스윕 = **1코어 `-n 1` · 2코어 `-n 4` · 3코어 `-n 0`**.

## 재현

```bash
# 코어-카운트 스윕 (seq + async 1/2/3코어, --buffer-count 8)
dxrun -m resnet50_native.dxnn -s -t 5 -w 1                      # seq
dxrun -m resnet50_native.dxnn -b -n 1 --buffer-count 8 -t 5 -w 1  # 1코어
dxrun -m resnet50_native.dxnn -b -n 4 --buffer-count 8 -t 5 -w 1  # 2코어
dxrun -m resnet50_native.dxnn -b -n 0 --buffer-count 8 -t 5 -w 1  # 3코어(ALL)
# 프로파일러 (스테이지 분해: 연산 vs D2H)
dxbenchmark --dir <model_dir> -t 5 --warmup 1                   # → profiler.json
python3 scripts/analyze_profiler.py raw/rn50_npuall/profiler.json rn50_npuall > results/rn50_npuall.json
# SSOT 재생성
python3 scripts/build_crossover_summary.py                       # → results/crossover_summary.json
```

## 파일

```
scripts/
  analyze_profiler.py         # profiler.json → 스테이지/코어별 요약 (../scripts와 동일)
  build_crossover_summary.py  # corescale.csv + bufsweep.csv + 3 프로파일 → crossover_summary.json (SSOT)
results/
  crossover_summary.json      # ← SSOT (헤드라인·regime·비율 전부)
  {rn50,y26,y5s}_npuall.json  # analyze_profiler.py 파생(스테이지 p50·코어 잡 분포)
raw/
  corescale/corescale.csv     # seq + async 1/2/3코어 dxrun stdout 파생
  bufsweep/bufsweep.csv       # --buffer-count 2/4/8/16 스윕
  {rn50,y26,y5s}_npuall/       # dxbenchmark 원본: profiler.json + stdout.txt + DXBENCHMARK_*.{html,csv,json}
  {rn50,y26}_seq.txt, *_async_n{1,2,3}.txt  # 개별 dxrun stdout (async_n{1,2,3}=단일코어 NPU_0/1/2, 각 ~493/91 — 코어-카운트 아님, corescale.csv가 정본)
```

## 캐비앗

1. 절대 처리량/지연은 batch1·prebuilt·Pi5 Gen2×1 → **상대 관계만** 유효.
2. host-bound은 **Pi5 Gen2×1** 탓 — 네이티브 Gen3×4면 D2H 벽 위치가 달라질 수 있음(미측정).
3. crossover 결론은 **모델(출력 크기) 의존** — 2 regime의 존재를 증명, 전이 곡선 아님(중간 출력 미측정).
4. YOLOV5S는 입력 512×512라 yolo26n(640×640)과 1:1 아님 — 공유 결론(둘 다 D2H-bound)만 사용.
5. 정확도는 이 축의 논점 아님 → `../accuracy/`에서 이미 닫음. Pi5는 DEEPX 호스트일 뿐 자동차 3벤더 아님.
