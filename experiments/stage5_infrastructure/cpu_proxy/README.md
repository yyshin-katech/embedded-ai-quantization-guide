# CPU 프록시 벤치 — ARM Cortex-A 폴백 실측 (Raspberry Pi 5)

4단계(멀티 SoC)의 벤더 NPU(TI TIDL / Qualcomm QNN / Renesas DRP-AI)는 보드가 없어 아직 못 돌린다.
하지만 그 세 SoC가 **공통으로 가진 ARM Cortex-A 폴백 경로**(가속기가 지원 못 하는 op가 떨어지는 그곳)는
**오늘 측정할 수 있다** — **Raspberry Pi 5**(Cortex-A76)를 프록시로.

3·5단계 자산인 **ResNet50 INT8 QDQ ONNX**를 순수 `CPUExecutionProvider`로 관통시켜, 같은 INT8 그래프가
**CPU ISA에 따라 양자화 이득의 부호가 뒤집힌다**는 것을 실측했다. 이것이 이 디렉터리의 헤드라인이다.

> **리포트(그림 포함):** [`../../../logs/stage4_arm_cpu_fallback_report.html`](../../../logs/stage4_arm_cpu_fallback_report.html)
> **상위 가이드:** [`../../../study_guide/06_multi_soc.md`](../../../study_guide/06_multi_soc.md) §2-1 뒤 🔬 콜아웃

---

## 헤드라인 — 양자화 이득의 부호는 CPU ISA가 결정한다

같은 INT8 QDQ ONNX, 같은 ONNX Runtime `CPUExecutionProvider`. 바뀐 건 **CPU뿐**인데 결과의 부호가 반대다.

| 플랫폼 / precision | 지연 median | p95 | vs 자기 FP32 | top-1(1000장) | ORT |
|---|--:|--:|:--|--:|:--|
| Pi5 (A76·dotprod) / fp32 | 144.9519 ms | 152.4119 ms | ×1.00 (기준) | 0.7620 | 1.28.0 |
| **Pi5 (A76·dotprod) / int8** | **79.0827 ms** | 81.2943 ms | **×1.83 빠름 ✓** | 0.7500 | 1.28.0 |
| x86 (i9-10900K·no VNNI) / fp32 | 9.2765 ms | 10.0726 ms | ×1.00 (기준) | 0.7620 | 1.23.2 |
| **x86 (i9-10900K·no VNNI) / int8** | **16.3376 ms** | 20.1000 ms | **×0.57 (1.76× 느림) ✗** | 0.7530 | 1.23.2 |

- **Pi 5 Cortex-A76**엔 `asimddp`(SDOT/UDOT INT8 dot-product)가 있어 INT8 conv가 FP32보다 **1.83× 빠르다**.
- **i9-10900K**(Comet Lake)엔 **VNNI가 없다**(AVX2까지). INT8 경로가 dot-product 가속을 못 받고, 그 위에
  `QuantizeLinear`/`DequantizeLinear` 노드 비용까지 얹혀 **FP32 AVX2 경로보다 되레 1.76× 느려진다**.
- ONNX Runtime의 CPU 커널(MLAS)은 정수 dot-product 명령이 있으면 INT8을 가속하고 없으면 못 한다 —
  ARM `SDOT`(`asimddp`) vs x86 `VPDPBUSD`(AVX-512 VNNI / AVX-VNNI). **ISA의 이 명령 하나가 부호를 가른다.**

> **왜 중요한가:** 4단계 SoC의 NPU/DSP는 INT8 전용 MAC이라 INT8이 항상 이긴다(그게 존재 이유). 부호 반전은
> **"가속기가 못 받아 CPU로 떨어진 부분"**에서 벌어진다 — 즉 **폴백이 많을수록 A-코어의 dotprod 유무가 최종
> 성능을 좌우**한다. dotprod 있는 A76(여러 최신 오토모티브 SoC의 A-코어)이면 폴백조차 INT8이 유리하지만,
> dotprod 없는 구형 A53/A72면 폴백 INT8이 되레 독이다. 이 표가 그 바닥값이다.

---

## 크로스플랫폼 예측 일치 — FP32는 비트 동일, INT8은 아니다

같은 1,000장·같은 모델·같은 `CPUExecutionProvider`로 x86과 ARM의 예측 클래스를 1:1 대조:

| precision | x86 top-1 | ARM top-1 | 예측 일치 | 상이 |
|---|--:|--:|:--|--:|
| fp32 | 0.7620 | 0.7620 | **1000/1000 (100.0%)** | 0 |
| int8 | 0.7530 | 0.7500 | 958/1000 (95.8%) | 42 |

- **FP32는 두 ISA에서 1,000장 전부 같은 예측** — "CPUEP는 크로스플랫폼 동일"이 FP32에선 문자 그대로 성립.
- **INT8은 다르다**: 정수 누산·재양자화 반올림이 `SDOT`(ARM)와 AVX2 정수 경로(x86)에서 미세하게 갈려 **42장이
  뒤집힌다**. 순 top-1 영향은 0.3%p로 작지만, **INT8 정확도를 서로 다른 타겟에서 비트 단위로 기대하면 안 된다**는
  실무 교훈이다. (ORT 버전도 다르나 1.28.0/1.23.2 — FP32가 100% 일치하므로 갈림은 버전이 아니라 ISA의 INT8 경로.)

---

## 왜 `results/`가 아니라 `cpu_proxy/`인가 (부수 발견)

이 CPU 4행을 5단계 벤치 매트릭스(`../bench/results/`)에 봉인하려 하자 커밋된 회귀 golden 테스트가 깨졌고,
그 자체가 하네스 통찰이었다.

- `tests/test_regression.py::test_matrix_matches_golden`은 `dataframe_regression.check(atol=1e-3, rtol=1e-3)`다.
  **결정론적 TRT 지연**(0.8628ms 재현)엔 맞지만, **비결정 CPU wall-clock**(Pi 145ms를 ±0.145ms로 재현 불가 —
  열·스케줄러·부하로 수 ms씩 흔들림)엔 부적합. 즉 이 golden은 **암묵적으로 TRT 결정론 백엔드 전용**이다.
- CPU 4행을 `bench/results/`에 넣으면 `Obtained (10,) vs Expected (6,)` shape 실패. 별도 `cpu_proxy/results/`로
  격리 → **커밋된 5단계 하네스는 6행·초록 유지**. (임계값 테스트 2개는 baseline과 inner-merge라 새 SoC를 애초에
  제외 — 정책상 정상.)

행 스키마는 `bench/results/`와 동일(그대로 옮겨 매트릭스에 붙일 수 있음). `peak_mem_mb`=NaN(미측정),
`engine_build_s`=0.0·`trt_version`=""(TRT 아님, CPU 경로).

---

## 파일 구성

```
cpu_proxy/
├── README.md                              ← 이 문서
├── rpi_bench.py                           ← 이식형 CPU-only 러너 (x86·aarch64 동일 실행)
├── results/                               ← 매트릭스 스키마 4행 (bench/results/와 동일 필드)
│   ├── resnet50__rpi5__fp32.json
│   ├── resnet50__rpi5__int8.json
│   ├── resnet50__x86_cpu__fp32.json
│   └── resnet50__x86_cpu__int8.json
└── raw/                                   ← 러너 원본 출력 (pred_cls 배열·arch·cpu·ort_version 포함)
    ├── rpi5_fp32.json  rpi5_int8.json
    └── x86_fp32.json   x86_int8.json
```

`raw/*.json`이 SSOT다 — `results/*.json`과 리포트·가이드 수치는 전부 여기서 옮겼다. 크로스플랫폼 일치율은
`raw/`의 `pred_cls` 배열을 x86↔ARM 1:1 비교해 산출.

---

## 재현

`rpi_bench.py`는 numpy + onnxruntime만 있으면 x86·aarch64 어디서든 동일하게 돈다. 전처리는 `bench/data.py`의
`_preprocess_nchw`(÷255·NHWC→NCHW·ImageNet 정규화, crop_tv)와 **바이트 동일**하게 복제돼 있다.

```bash
# 입력: <data>/rpi_sub_u8.npy (n,224,224,3 uint8) + rpi_labels.npy (n,)
#       ImageNet val 첫 1,000장 서브셋을 uint8로 저장해 Pi로 전송 (crop 전처리는 러너 내부에서)

# FP32
python3 rpi_bench.py --model resnet50_fp32.onnx     --precision fp32 \
    --data <data> --out raw/rpi5_fp32.json --soc rpi5 --n 1000 --warmup 20 --iters 200
# INT8 (동일 그래프, precision 라벨만 다름 — QDQ ONNX가 INT8 경로를 결정)
python3 rpi_bench.py --model resnet50_int8_qdq.onnx --precision int8 \
    --data <data> --out raw/rpi5_int8.json --soc rpi5 --n 1000 --warmup 20 --iters 200

# x86 대조: 같은 명령을 dev-host에서 --soc x86_cpu 로 실행
```

지연 = `time.perf_counter` wall-clock 단일입력(warmup 20 + iters 200)의 median/p95. 정확도 = 1,000장 top-1
+ 예측 클래스 배열 덤프(크로스플랫폼 일치 계산용). provider는 `["CPUExecutionProvider"]`로 강제.

---

## 캐비앗 (불변)

1. **Pi는 ARM Cortex-A 폴백 프록시일 뿐 자동차 NPU가 아니다** — QCS8550/RZ-V2H/TDA4VM의 가속 수치는 여기서
   전이되지 않는다. 측정한 건 "모든 op가 CPU로 폴백된 바닥값".
2. **절대 지연·top-1은 CPUEP·wall-clock 단일입력·배치1·1,000장 서브셋·ORT 버전 상이 기준** → 상대 관계
   (부호·배율·예측 일치율)만 유효.
3. **Pi↔x86 절대 속도 비교는 논점 아님** — 서로 다른 급의 기계다(4×A76 @2.4GHz vs 10×x86 @3.7GHz). 논점은
   각 플랫폼 **자기 FP32 대비 INT8의 부호**다.
4. 1,000장 서브셋 top-1은 공개 50k보다 부풀려질 수 있음(1단계 함정 0). Pi 열 상태: 런 직후 78.5°C, 스로틀 흔적 없음.
