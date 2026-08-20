# Jetson AGX Orin 온디바이스 TensorRT — 제약·발견·설계 규칙

측정일 2026-08-20 · Jetson AGX Orin Dev Kit(64GB) · JetPack 6.2.1(L4T R36.4.3) · CUDA 12.6 · **TensorRT 10.3.0.30** · cuDNN 9.3.0 · MAXN.
모델은 3단계와 **동일한** torchvision ResNet50 ONNX(FP32 `resnet50_fp32.onnx`, 명시적-QDQ `resnet50_int8_qdq.onnx`). 지연 = `trtexec` GPU-compute median(디바이스 event-timed, batch1). 전력 = `tegrastats` 보드 총합.

이 파일이 채우는 두 공백:
- **3·5단계**: 정본 pip 휠(`tensorrt-cu12`)에 `trtexec` 실행파일이 **없어** polygraphy Python API로 우회했다. Jetson엔 JetPack 동봉 `/usr/src/tensorrt/bin/trtexec`가 실존 → 원래 명령을 그대로 관통.
- **3단계**: RTX 3080은 DLA 코어가 **0개**라 "DLA 범위 밖". Orin엔 **NVDLA v2 2개**가 있어 실측.

---

## 발견 1 — 실 trtexec 실존, 플래그 1건 정정

정본 pip 휠엔 없던 `trtexec`가 Jetson엔 있다:

```
$ /usr/src/tensorrt/bin/trtexec --version
&&&& RUNNING TensorRT.trtexec [TensorRT v100300]
```

빌드/벤치 명령(5엔진 전부 exit 0):

```bash
TX=/usr/src/tensorrt/bin/trtexec
# iGPU
$TX --onnx=onnx/resnet50_fp32.onnx      --saveEngine=engines/rn50_gpu_fp32.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
$TX --onnx=onnx/resnet50_fp32.onnx --fp16 --saveEngine=engines/rn50_gpu_fp16.plan ...
$TX --onnx=onnx/resnet50_int8_qdq.onnx --int8 --fp16 --saveEngine=engines/rn50_gpu_int8.plan ...   # 명시적 QDQ
# DLA (--allowGPUFallback 필수: pool/flatten은 DLA 불가)
$TX --onnx=onnx/resnet50_fp32.onnx --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=engines/rn50_dla_fp16.plan ...
$TX --onnx=onnx/resnet50_fp32.onnx --int8 --fp16 --useDLACore=0 --allowGPUFallback --saveEngine=engines/rn50_dla_int8.plan ...
```

> **정정**: 이 트림(TRT 10.3.0)의 **빌드전용 플래그는 `--buildOnly`가 아니라 `--skipInference`**.
> `--buildOnly`는 `[E] Unknown option: --buildOnly`로 exit 1. 레이어 배치 캡처는 `--skipInference --verbose`.

---

## 발견 2 — NVDLA는 INT8 전용기 (DLA FP16 = 함정)

| DLA0 | 지연 | vs DLA INT8 | steady W | GR3D |
|---|---|---|---|---|
| INT8 | **1.2783 ms** | 1.00× | 15.19 | 16% |
| FP16 | **17.7344 ms** | **13.87× 느림** | 15.69 | 3% |

**레이어 배치가 동일한데** 지연만 13.87× 벌어진다 → 원인은 레이어 할당이 아니라 순수 **NVDLA v2 데이터패스**(INT8 MAC 처리량 ≫ FP16). **DLA에 올릴 거면 반드시 INT8.** DLA FP16은 iGPU FP32(1.94ms)보다도 9배 느린 최악 조합(단 15.7W로 저전력).

이것은 iGPU와 **정반대**다 — iGPU는 INT8≈FP16(발견 5).

---

## 발견 3 — DLA 오프로드의 계측적 증명 (GR3D 붕괴)

| 엔진 | 백엔드 | GR3D(GPU-3D) | VDD_GPU_SOC |
|---|---|---|---|
| iGPU INT8 | iGPU | **95%** | 21.80 W |
| DLA FP16 | DLA0 | **3%** | 5.98 W |
| DLA INT8 | DLA0 | **16%** | 5.58 W |

DLA 실행 중 GPU-3D 사용률이 3~16%로 붕괴 → **연산이 GPU가 아니라 DLA에서 실제로 돈다**. GPU는 비어 있으므로 다른 모델/헤드를 병렬로 돌릴 수 있다. "가속기 오프로드"가 말이 아니라 수치로 확인됨.

> **주의**: DLA 부하는 GR3D 카운터에 거의 안 잡히므로, 성능/와트 하네스(`ppw.py`)의 "부하 구간" 검출은 GR3D가 아니라 **전력 임계(idle_floor×1.20)** 로 자기보정해야 한다. GR3D 기준이면 DLA 샘플이 전부 버려진다.

---

## 발견 4 — DLA 레이어 배치 (2 ForeignNode, 2/2 오프로드)

`--skipInference --verbose`의 배치 섹션(INT8·FP16 동일):

```
[V] [TRT] Number of DLA node candidates offloaded : 2 out of 2
[V] [TRT] {ForeignNode[/conv1/Conv.../layer4/layer4.2/relu_2/Relu]} successfully offloaded to DLA.
[V] [TRT] {ForeignNode[/fc/Gemm + (Unnamed Layer* 125) [ElementWise]]} successfully offloaded to DLA.
[V] [TRT] [GpuLayer] REDUCE: /avgpool/GlobalAveragePool
[V] [TRT] [GpuLayer] SHUFFLE: reshape_after_/fc/Gemm
```

- **DLA 서브그래프 = 2개 ForeignNode**: ① conv 백본 전체 `/conv1/Conv‥/layer4.2/relu_2`(**120개 DLA 층**) ② `/fc/Gemm + bias`(FC도 DLA에 올라감).
- **DLA-후보 2/2 오프로드**(100%).
- **GPU 폴백 = compute 2층뿐**: `GlobalAveragePool`(REDUCE→POOLING)과 flatten `SHUFFLE`. 폴백 이유는 빌드 경고에 명시:
  ```
  [W] [TRT] /avgpool/GlobalAveragePool: DLA cores do not support AVG Reduce operation.
  [W] [TRT] Layer '/avgpool/GlobalAveragePool' (REDUCE): Unsupported on DLA. Switching ... to GPU.
  [W] [TRT] Layer '/Flatten' (SHUFFLE): Unsupported on DLA. Switching ... to GPU.
  ```
- INT8/FP16 배치 완전 동일(둘 다 2/2·120 DLA·2 GPU) → 발견 2의 13.87× 격차가 배치가 아닌 데이터패스 탓임을 확정.

---

## 발견 5 — 작은 Ampere iGPU는 INT8 ≈ FP16

| iGPU | 지연 | vs FP32 |
|---|---|---|
| FP32 | 1.9375 ms | ×1.00 |
| FP16 | 1.0293 ms | ×1.88 |
| INT8 | 1.0132 ms | ×1.91 |

iGPU INT8/FP16 = **0.984** (INT8이 1.6%밖에 안 빠름). 3단계 **RTX 3080**은 0.927(INT8이 8% 빠름). batch1 ResNet50은 연산량보다 커널 launch·메모리 대역에 묶이는데, Orin의 작은 iGPU(2048 CUDA코어)는 그 바운드가 더 세서 INT8 연산 이득이 거의 상쇄된다. **정밀도 이득도 GPU 규모에 의존.** (단 INT8 엔진 크기 26.18 MiB = FP32의 27%이고, 전력에서는 확실히 이김.)

---

## 발견 6 — 성능/전력 · PSU

| 엔진 | steady W | peak W | inf/s/W |
|---|---|---|---|
| iGPU FP32 | 42.26 | **42.37** | 12.19 |
| iGPU FP16 | 36.20 | 36.59 | 26.77 |
| iGPU INT8 | 29.70 | 29.71 | 33.16 |
| DLA FP16 | 15.69 | 15.69 | 3.59 |
| **DLA INT8** | **15.19** | 15.29 | **51.29** |

- **DLA INT8 = 성능/와트 챔피언(51.29)**: iGPU INT8 대비 지연 1.262× 느리지만 전력은 **0.511×(절반)**, 성능/와트 **1.547×**, 게다가 GPU 프리.
- iGPU 정밀도↓ = 전력↓ 단조(42.3→36.2→29.7W).
- **전원 공급장치**: ResNet50 batch1 실측 피크 **42.37W**, 유휴 **7.59W**. 64GB 모듈 설계 상한 60W(MAXN) → 동봉 19V 어댑터로 충분, 서드파티면 19V·90W급 권장. 배치 확대·iGPU+DLA+CPU 동시부하면 60W 쪽으로 상승.
- 전력 모델(AGX Orin): board-total = `VDD_GPU_SOC` + `VDD_CPU_CV` + `VIN_SYS_5V0`. `VIN_SYS_5V0`(6.7~8.7W)는 캐리어·주변장치 오버헤드. `tegrastats`는 **sudo 불필요**.

---

## 설계 규칙 (재현 시)

1. **`trtexec`는 JetPack 동봉본**(`/usr/src/tensorrt/bin/`)을 쓴다. pip 휠엔 없다. 빌드전용은 `--skipInference`(≠`--buildOnly`).
2. **DLA는 INT8로만 실사용.** FP16 DLA는 13.87× 느림. `--useDLACore=N`엔 `--allowGPUFallback` 필수(pool/flatten은 DLA 불가).
3. **DLA 오프로드 검증은 GR3D로.** 95→3~16% 붕괴 = GPU가 실제로 비었다는 증거. 단 **전력 하네스의 부하검출은 GR3D가 아니라 전력 임계**로(DLA는 GR3D에 안 잡힘).
4. **iGPU INT8 이득은 모델·GPU 규모 의존.** 작은 iGPU + batch1이면 INT8≈FP16일 수 있다. 지연이 목적이면 프로파일로 확인, 전력·엔진크기가 목적이면 INT8이 항상 유리.
5. **각 엔진 독립 실행**(동시부하 금지) — 지연·전력 타이밍 오염 방지.
6. **정확도 스코프**: iGPU INT8은 명시적 QDQ(stage3 실 스케일)라 정확도 유효. **DLA INT8은 `--int8` 암묵 캘리브(자동 레인지)라 지연·전력만 유효, 정확도 미주장**(정확도는 stage3 RTX·stage4 CPU 프록시에서 확립). stage8 "지연은 가중치·캘리브 무관" 원칙과 동일.

---

## 캐비앗 (불변)

- 지연 = event-timed·batch1·MAXN → 타 단계(stage5 wall-clock)와 1:1 비교 불가, 상대만.
- 전력 = 보드 총합(SYS_5V0 오버헤드 포함), 부하 중앙값.
- RTX 3080 비교값(FP32 1.6615/INT8 0.7843ms)은 다른 GPU라 참고용.
- iGPU+DLA 동시부하(진짜 병렬 오프로드)는 미측정 — 다음 과제.
