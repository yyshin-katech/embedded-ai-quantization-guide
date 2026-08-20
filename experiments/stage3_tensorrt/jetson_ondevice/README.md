# stage3 / jetson_ondevice — Jetson AGX Orin 온디바이스 TensorRT (iGPU + DLA + 성능/전력)

3단계 TensorRT를 **실 보드**(Jetson AGX Orin Dev Kit 64GB)에서 재실행한 온디바이스 실측. RTX 3080 프록시(3·5단계)로는 불가능했던 4가지를 한 번의 브링업으로 관통:

| # | 항목 | RTX 3080에서 불가였던 이유 | 결과 |
|---|---|---|---|
| 1 | **DLA INT8** | RTX엔 DLA 코어 0개("DLA 범위 밖") | 1.28ms·15.2W·**51.3 inf/s/W**(챔피언) |
| 2 | **실 trtexec 온디바이스** | 정본 pip 휠에 trtexec 실행파일 부재 | 5엔진 전부 실 바이너리로 빌드 |
| 3 | **성능/와트(tegrastats)** | dGPU엔 통합 보드 전력 레일 없음 | 피크 42.4W·유휴 7.6W 실측 |
| 4 | **iGPU Ampere INT8** | — | 1.01ms·×1.91, 단 INT8≈FP16 |

리포트: [`logs/stage3_jetson_orin_ondevice_report.html`](../../../logs/stage3_jetson_orin_ondevice_report.html) · 상세 제약: [`constraints.md`](constraints.md)

---

## 환경

- 보드: NVIDIA Jetson AGX Orin Developer Kit (64GB 모듈, RAM 62,841MB), `nvpmodel` **MAXN**
- SoC: Ampere iGPU(2048 CUDA/64 Tensor) · **2× NVDLA v2** · 12× Cortex-A78AE
- SW: JetPack **6.2.1**(L4T R36.4.3) · CUDA **12.6**(V12.6.68) · TensorRT **10.3.0.30** · cuDNN **9.3.0**
- 도구: `/usr/src/tensorrt/bin/trtexec`(JetPack 동봉) · `tegrastats`(sudo 불필요)
- 모델: **3단계와 동일** torchvision ResNet50 ONNX — `resnet50_fp32.onnx`(102MB) / `resnet50_int8_qdq.onnx`(26MB, 명시적 QDQ 293개)

## 방법

- 지연 = `trtexec` **GPU Compute Time median**(디바이스 event-timed, H2D/D2H 제외), batch1 — 3단계 polygraphy event-timed와 방법론 동일.
- 전력 = `tegrastats --interval 100` 보드 총합(`VDD_GPU_SOC`+`VDD_CPU_CV`+`VIN_SYS_5V0`), 30초 지속부하 중앙값.
- 5엔진(iGPU FP32/FP16/INT8 + DLA FP16/INT8) 각각 **독립 실행**(동시부하 없음).

## 재현

```bash
# 보드에서 (~/orin_bench)
TX=/usr/src/tensorrt/bin/trtexec
# --- 엔진 빌드 ---
$TX --onnx=onnx/resnet50_fp32.onnx        --saveEngine=engines/rn50_gpu_fp32.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
$TX --onnx=onnx/resnet50_fp32.onnx  --fp16 --saveEngine=engines/rn50_gpu_fp16.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
$TX --onnx=onnx/resnet50_int8_qdq.onnx --int8 --fp16 --saveEngine=engines/rn50_gpu_int8.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
$TX --onnx=onnx/resnet50_fp32.onnx  --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=engines/rn50_dla_fp16.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
$TX --onnx=onnx/resnet50_fp32.onnx  --int8 --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=engines/rn50_dla_int8.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
# --- 레이어 배치 캡처(선택) ---  ※ --skipInference (NOT --buildOnly)
$TX --onnx=onnx/resnet50_fp32.onnx --int8 --fp16 --useDLACore=0 --allowGPUFallback --skipInference --verbose 2>&1 | tee raw/dla_int8_verbose.log
# --- 성능/와트(지속부하 + tegrastats) ---
python3 scripts/ppw.py engines/rn50_gpu_fp32.plan gpu_fp32 30
python3 scripts/ppw.py engines/rn50_gpu_int8.plan gpu_int8 30
python3 scripts/ppw.py engines/rn50_dla_int8.plan dla_int8 30 0   # 마지막 인자 = DLA 코어
```

`scripts/ppw.py <engine.plan> <label> [duration_s] [dla_core]` — trtexec 지속부하 중 tegrastats를 샘플링해 steady/peak 전력·GR3D·inf/s/W를 `results/ppw_<label>.json`으로 저장. **부하 검출은 전력 임계(idle×1.20)로 자기보정**(DLA는 GR3D에 안 잡힘).

## 결과 (SSOT: `results/summary.json`)

| 엔진 | 백엔드 | 지연 ms | vs FP32 | steady W | peak W | GR3D | inf/s/W | 엔진 MiB |
|---|---|---|---|---|---|---|---|---|
| gpu_fp32 | iGPU | 1.9375 | ×1.00 | 42.26 | 42.37 | 98% | 12.19 | 98.09 |
| gpu_fp16 | iGPU | 1.0293 | ×1.88 | 36.20 | 36.59 | 96% | 26.77 | 49.29 |
| gpu_int8 | iGPU | 1.0132 | ×1.91 | 29.70 | 29.71 | 95% | 33.16 | 26.18 |
| dla_fp16 | DLA0 | 17.7344 | ×0.11 | 15.69 | 15.69 | 3% | 3.59 | 49.14 |
| **dla_int8** | DLA0 | **1.2783** | ×1.52 | **15.19** | 15.29 | 16% | **51.29** | 24.77 |

파생: iGPU INT8/FP16 지연비 **0.984**(INT8≈FP16) · DLA INT8이 DLA FP16보다 **13.87×** 빠름(NVDLA INT8 전용) · DLA INT8 vs iGPU INT8 = 지연 1.262×·전력 0.511×·성능/와트 **1.547×**.

## 산출물

```
jetson_ondevice/
├── README.md                 이 파일
├── constraints.md            발견 6종 + 로그 원문 + 설계 규칙
├── scripts/ppw.py            이식형 성능/와트 하네스
├── results/
│   ├── summary.json          SSOT (5엔진 + 파생 + 배치)
│   └── ppw_{gpu_fp32,gpu_fp16,gpu_int8,dla_fp16,dla_int8}.json
└── raw/
    ├── {gpu_fp32,gpu_fp16,gpu_int8,dla_fp16,dla_int8}.log   빌드+벤치
    └── {dla_int8,dla_fp16}_verbose.log                       레이어 배치
```

## 캐비앗

- 지연 = event-timed·batch1·MAXN → 타 단계(stage5 wall-clock)와 1:1 비교 불가, **상대 관계만**.
- **DLA INT8은 암묵 캘리브(자동 레인지) → 지연·전력만 유효, 정확도 미주장**(정확도는 stage3 RTX·stage4 CPU 프록시). iGPU INT8은 명시적 QDQ라 정확도 유효.
- 전력 = 보드 총합(SYS_5V0 캐리어 오버헤드 6.7~8.7W 포함).
- RTX 3080 비교값(FP32 1.6615/INT8 0.7843ms)은 다른 GPU라 참고용.
- iGPU+DLA 동시부하(진짜 병렬 오프로드)는 미측정 — 다음 과제.

## 관련

- 같은 보드 CPU 축: [`logs/stage4_jetson_agx_orin_a78ae_report.html`](../../../logs/stage4_jetson_agx_orin_a78ae_report.html)
- 3단계 RTX(trtexec 부재→polygraphy·DLA 범위밖): [`logs/stage3_tensorrt_report.html`](../../../logs/stage3_tensorrt_report.html) · [`experiments/stage3_tensorrt/`](../)
- 벤더 NPU(Qualcomm Hexagon): [`logs/stage4_qualcomm_aihub_report.html`](../../../logs/stage4_qualcomm_aihub_report.html)
