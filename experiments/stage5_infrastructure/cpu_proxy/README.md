# CPU 프록시 벤치 — ARM Cortex-A 폴백 실측 (Pi 5 A76 + i.MX8M Nano A53)

4단계(멀티 SoC)의 벤더 NPU(TI TIDL / Qualcomm QNN / Renesas DRP-AI)는 보드가 없어 아직 못 돌린다.
하지만 그 세 SoC가 **공통으로 가진 ARM Cortex-A 폴백 경로**(가속기가 지원 못 하는 op가 떨어지는 그곳)는
**오늘 측정할 수 있다** — 실물 ARM 보드를 프록시로.

3·5단계 자산인 **ResNet50 INT8 QDQ ONNX**를 순수 `CPUExecutionProvider`로 관통시켜, 같은 INT8 그래프가
**CPU ISA에 따라 양자화 이득의 부호가 뒤집힌다**는 것을 실측했다. 이것이 이 디렉터리의 헤드라인이다.
프록시는 **세 점(3-사분면)**으로 측정했다 — **Pi 5**(Cortex-A76, dotprod 있음) · **i.MX8M Nano**(Cortex-A53,
dotprod 없는 ARM) · **x86 dev-host**(i9-10900K, VNNI 없음). 특히 A53은 **"부호를 가르는 게 ARM/x86 계열이냐,
dot-product 명령 유무냐"를 ARM 안에서 갈라 준다**(A76·A53은 같은 ARMv8 계열인데 부호가 반대).

> **리포트(그림 포함):** [`../../../logs/stage4_arm_cpu_fallback_report.html`](../../../logs/stage4_arm_cpu_fallback_report.html)(Pi 5·x86 원 발견) ·
> [`../../../logs/stage4_imx8mn_a53_report.html`](../../../logs/stage4_imx8mn_a53_report.html)(A53 — 3-사분면·벽 2건)
> **상위 가이드:** [`../../../study_guide/06_multi_soc.md`](../../../study_guide/06_multi_soc.md) §2-2 🔬 콜아웃

---

## 헤드라인 — 양자화 이득의 부호는 CPU ISA가 결정한다 (그리고 부호를 가르는 건 dotprod 유무다)

같은 INT8 QDQ ONNX, 같은 ONNX Runtime `CPUExecutionProvider`. 바뀐 건 **CPU뿐**인데 결과의 부호가 갈린다.

| 플랫폼 / precision | 지연 median | p95 | vs 자기 FP32 | top-1(1000장) | ORT |
|---|--:|--:|:--|--:|:--|
| Pi5 (A76·dotprod ○) / fp32 | 144.9519 ms | 152.4119 ms | ×1.00 (기준) | 0.7620 | 1.28.0 |
| **Pi5 (A76·dotprod ○) / int8** | **79.0827 ms** | 81.2943 ms | **×1.83 빠름 ✓** | 0.7500 | 1.28.0 |
| i.MX8M Nano (A53·dotprod ✗) / fp32 | 680.2026 ms | 684.4127 ms | ×1.00 (기준) | 0.7620 | 1.17.1 |
| **i.MX8M Nano (A53·dotprod ✗) / int8** | **1123.0230 ms** | 1138.2552 ms | **×0.61 (1.65× 느림) ✗** | 0.7560 | 1.17.1 |
| x86 (i9-10900K·VNNI ✗) / fp32 | 9.2765 ms | 10.0726 ms | ×1.00 (기준) | 0.7620 | 1.23.2 |
| **x86 (i9-10900K·VNNI ✗) / int8** | **16.3376 ms** | 20.1000 ms | **×0.57 (1.76× 느림) ✗** | 0.7530 | 1.23.2 |

- **Pi 5 Cortex-A76**엔 `asimddp`(SDOT/UDOT INT8 dot-product)가 있어 INT8 conv가 FP32보다 **1.83× 빠르다**.
- **i.MX8M Nano Cortex-A53**은 **ARMv8.0-A**라 `asimddp`가 **없다**. 같은 ARM인데도 INT8이 FP32보다 **1.65× 느려진다** —
  dot-product 가속을 못 받고 `Quantize`/`Dequantize` 노드 비용까지 얹힌 결과. **A76과 부호가 반대다.**
- **i9-10900K**(Comet Lake)엔 **VNNI가 없다**(AVX2까지). INT8 경로가 dot-product 가속을 못 받아 **1.76× 느려진다**.
- ONNX Runtime의 CPU 커널(MLAS)은 정수 dot-product 명령이 있으면 INT8을 가속하고 없으면 못 한다 —
  ARM `SDOT`(`asimddp`, ARMv8.2+) vs x86 `VPDPBUSD`(AVX-512 VNNI / AVX-VNNI). **ISA의 이 명령 하나가 부호를 가른다.**
- **핵심 — 부호를 가르는 건 "ARM이냐 x86이냐"가 아니라 dotprod 유무다.** Pi 5의 A76과 i.MX8M Nano의 A53은
  **같은 ARMv8 계열**인데 부호가 반대다: A53(dotprod ✗)이 x86(VNNI ✗)과 같은 "느려짐" 부호를 낸다. 통념
  **"ARM이면 INT8이 유리"는 ARM 안에서 반박**된다. 결정 인자는 코어가 **ARMv8.2 dot-product 확장을 구현했는지** 하나다.

> **왜 중요한가:** 4단계 SoC의 NPU/DSP는 INT8 전용 MAC이라 INT8이 항상 이긴다(그게 존재 이유). 부호 반전은
> **"가속기가 못 받아 CPU로 떨어진 부분"**에서 벌어진다 — 즉 **폴백이 많을수록 A-코어의 dotprod 유무가 최종
> 성능을 좌우**한다. dotprod 있는 A76(여러 최신 오토모티브 SoC의 A-코어)이면 폴백조차 INT8이 유리하지만,
> **dotprod 없는 구형 A53(위 i.MX8M Nano로 직접 실측)이면 폴백 INT8이 되레 독**이다. 이 표가 그 바닥값이다.

---

## 크로스플랫폼 예측 일치 — FP32는 비트 동일, INT8은 아니다 (3-플랫폼)

같은 1,000장·같은 모델·같은 `CPUExecutionProvider`로 세 플랫폼의 예측 클래스를 1:1 대조:

| precision | 대조 쌍 | 예측 일치 | 상이 |
|---|---|:--|--:|
| fp32 | imx8mn(A53) ↔ pi5(A76) | **1000/1000 (100.0%)** | 0 |
| fp32 | imx8mn(A53) ↔ x86 | **1000/1000 (100.0%)** | 0 |
| fp32 | x86 ↔ pi5(A76) | **1000/1000 (100.0%)** | 0 |
| int8 | imx8mn(A53) ↔ pi5(A76) | 965/1000 (96.5%) | 35 |
| int8 | imx8mn(A53) ↔ x86 | 961/1000 (96.1%) | 39 |
| int8 | x86 ↔ pi5(A76) | 958/1000 (95.8%) | 42 |

- **FP32는 세 ISA에서 1,000장 전부 같은 예측** — "CPUEP는 크로스플랫폼 동일"이 FP32에선 문자 그대로 성립.
  ORT 세 버전(1.17.1 / 1.23.2 / 1.28.0)이 서로 다른데도 100%이므로 갈림은 버전이 아니다.
- **INT8은 다르다**: 정수 누산·재양자화 반올림이 `SDOT`(A76)·스칼라(A53)·AVX2 정수 경로(x86)에서 미세하게 갈려
  **35~42장이 뒤집힌다**. 순 top-1 영향은 0.3~0.6%p로 작지만, **INT8 정확도를 서로 다른 타겟에서 비트 단위로
  기대하면 안 된다**는 실무 교훈이 이제 **3점으로 확립**된다.

---

## 실물 저사양 보드의 벽 2건 (i.MX8M Nano — x86 개발기에선 안 보임)

i.MX8M Nano LPDDR4 EVK는 **2GB LPDDR4·swap 0·ORT 1.17.1**이다. x86 dev-host나 Pi 5(8GB)에선 안 나타난 두 벽을
만났고, **둘 다 "결과 불변"을 먼저 증명한 뒤** 해소해서 위 수치를 얻었다.

**(a) FP32 OOM (`rpi_bench.py` → `rpi_bench_lowmem.py`)**
`rpi_bench.py`는 정확도 루프에서 전체 입력을 한 번에 materialize한다 — `X = preprocess(u8)`가 1,000장을
`float32 (1000,3,224,224)` = **약 602MB** 배열로 올려 2GB·no-swap에서 **SIGKILL(rc=137)**. `rpi_bench_lowmem.py`는
uint8 캐시를 `mmap_mode="r"`로 열고 **이미지를 1장씩 lazy 전처리**한다:

```python
u8 = np.load(f"{data}/rpi_sub_u8.npy", mmap_mode="r")[: args.n]   # 디스크에 둔 채 mmap
for i in range(args.n):
    xi = preprocess(np.ascontiguousarray(u8[i:i+1]))              # 한 장만 float32로
    pred_cls.append(int(sess.run(None, {inp: xi})[0].argmax(1)[0]))
```

전처리는 **elementwise**(÷255·정규화·transpose)라 배치 단위든 1장 단위든 **결과가 비트 동일** → 예측 클래스
배열이 `rpi_bench.py`와 완전히 같다(그래서 저메모리 변형이 정당). peak RSS ≈ 602 → 333MB로 내려가 2GB 안에서 완주.

**(b) INT8 opset skew (모델 `opset_import` strip)**
`resnet50_int8_qdq.onnx`는 `opset_import`에 **미사용** `ai.onnx.ml v5`(+ training / com.microsoft / org.pytorch.aten)를
선언한다. 최신 ORT는 무시하지만 **ORT 1.17.1은 `ai.onnx.ml` 상한이 opset 4**라 로드 자체를 거부한다. 실제 노드
415개는 **전부 기본 도메인**(`''`)이므로 미사용 opset 항목을 떼어내 `[('', 17)]`만 남기면 **연산 그래프는 불변**:

```python
import onnx
m = onnx.load("resnet50_int8_qdq.onnx")
used = {n.domain for n in m.graph.node}          # -> {''}  (기본 도메인뿐)
del m.opset_import[:]
m.opset_import.append(onnx.helper.make_operatorsetid("", 17))
onnx.save(m, "resnet50_int8_qdq_op4.onnx")       # 노드 0개 변경, ORT 1.17.1 로드 성공
```

> **교훈:** 엣지에선 **"메모리 상한"과 "런타임의 opset 상한"**이 실물 보드에서만 드러나는 두 벽이다. 둘 다
> **모델·전처리의 의미를 바꾸지 않고**(예측 비트 동일 / 노드 불변) 우회할 수 있어야 측정이 정당하다.

---

## 왜 `results/`가 아니라 `cpu_proxy/`인가 (부수 발견)

이 CPU 행들을 5단계 벤치 매트릭스(`../bench/results/`)에 봉인하려 하자 커밋된 회귀 golden 테스트가 깨졌고,
그 자체가 하네스 통찰이었다.

- `tests/test_regression.py::test_matrix_matches_golden`은 `dataframe_regression.check(atol=1e-3, rtol=1e-3)`다.
  **결정론적 TRT 지연**(0.8628ms 재현)엔 맞지만, **비결정 CPU wall-clock**(Pi 145ms를 ±0.145ms로 재현 불가 —
  열·스케줄러·부하로 수 ms씩 흔들림)엔 부적합. 즉 이 golden은 **암묵적으로 TRT 결정론 백엔드 전용**이다.
- CPU 행을 `bench/results/`에 넣으면 `Obtained (n,) vs Expected (6,)` shape 실패. 별도 `cpu_proxy/results/`로
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
├── rpi_bench_lowmem.py                    ← 저메모리 변형 (mmap + 1장씩 lazy 전처리, 2GB 보드용·예측 비트 동일)
├── results/                               ← 매트릭스 스키마 (bench/results/와 동일 필드)
│   ├── resnet50__rpi5__fp32.json          resnet50__rpi5__int8.json
│   ├── resnet50__imx8mn_a53__fp32.json    resnet50__imx8mn_a53__int8.json
│   └── resnet50__x86_cpu__fp32.json       resnet50__x86_cpu__int8.json
└── raw/                                   ← 러너 원본 출력 (pred_cls 배열·arch·cpu·ort_version 포함)
    ├── rpi5_fp32.json         rpi5_int8.json
    ├── imx8mn_a53_fp32.json   imx8mn_a53_int8.json
    └── x86_fp32.json          x86_int8.json
```

`raw/*.json`이 SSOT다 — `results/*.json`과 리포트·가이드 수치는 전부 여기서 옮겼다. 크로스플랫폼 일치율은
`raw/`의 `pred_cls` 배열을 세 플랫폼 1:1 비교해 산출.

---

## 재현

`rpi_bench.py`(+ 저메모리 `rpi_bench_lowmem.py`)는 numpy + onnxruntime만 있으면 x86·aarch64 어디서든 동일하게
돈다. 전처리는 `bench/data.py`의 `_preprocess_nchw`(÷255·NHWC→NCHW·ImageNet 정규화, crop_tv)와 **바이트 동일**하게
복제돼 있다.

```bash
# 입력: <data>/rpi_sub_u8.npy (n,224,224,3 uint8) + rpi_labels.npy (n,)
#       ImageNet val 첫 1,000장 서브셋을 uint8로 저장해 보드로 전송 (crop 전처리는 러너 내부에서)

# --- Pi 5 (8GB) / x86 dev-host : 표준 러너 ---
python3 rpi_bench.py --model resnet50_fp32.onnx     --precision fp32 \
    --data <data> --out raw/rpi5_fp32.json --soc rpi5 --n 1000 --warmup 20 --iters 200
python3 rpi_bench.py --model resnet50_int8_qdq.onnx --precision int8 \
    --data <data> --out raw/rpi5_int8.json --soc rpi5 --n 1000 --warmup 20 --iters 200
# x86 대조: 같은 명령을 dev-host에서 --soc x86_cpu 로

# --- i.MX8M Nano (2GB no-swap, ORT 1.17.1) : 저메모리 러너 + opset strip ---
# (선행) INT8 모델의 미사용 opset을 떼어 ('',17)만 남긴다 (위 "벽 (b)" 스니펫) → resnet50_int8_qdq_op4.onnx
python3 rpi_bench_lowmem.py --model resnet50_fp32.onnx        --precision fp32 \
    --data <data> --out raw/imx8mn_a53_fp32.json --soc imx8mn_a53 --n 1000 --warmup 20 --iters 200
python3 rpi_bench_lowmem.py --model resnet50_int8_qdq_op4.onnx --precision int8 \
    --data <data> --out raw/imx8mn_a53_int8.json --soc imx8mn_a53 --n 1000 --warmup 20 --iters 200
```

지연 = `time.perf_counter` wall-clock 단일입력(warmup 20 + iters 200)의 median/p95. 정확도 = 1,000장 top-1
+ 예측 클래스 배열 덤프(크로스플랫폼 일치 계산용). provider는 `["CPUExecutionProvider"]`로 강제.

> **보드 접근 주의:** i.MX8M Nano는 sudo·네트워크 설정이 필요할 수 있다. **sudo가 필요한 명령은 저장 암호를
> 쓰지 말고 사용자가 직접 실행**한다(이 세션 규약). 벤치 자체는 root 권한이 필요 없다.

---

## 캐비앗 (불변)

1. **Pi 5·i.MX8M Nano는 ARM Cortex-A 폴백 프록시일 뿐 자동차 NPU가 아니다** — QCS8550/RZ-V2H/TDA4VM의 가속
   수치는 여기서 전이되지 않는다. 측정한 건 "모든 op가 CPU로 폴백된 바닥값(offload 0%)". (특히 i.MX8M **Nano**엔
   NPU가 아예 없다 — NPU는 i.MX8M **Plus**에만 있다.)
2. **절대 지연·top-1은 CPUEP·wall-clock 단일입력·배치1·1,000장 서브셋·ORT 버전 상이 기준** → 상대 관계
   (부호·배율·예측 일치율)만 유효.
3. **세 플랫폼 절대 속도 비교는 논점 아님** — 서로 다른 급의 기계다(4×A76 @2.4GHz vs 4×A53 @1.5GHz vs
   10×x86 @3.7GHz). 논점은 각 플랫폼 **자기 FP32 대비 INT8의 부호**다.
4. 1,000장 서브셋 top-1은 공개 50k보다 부풀려질 수 있음(1단계 함정 0). Pi 열 상태: 런 직후 78.5°C, 스로틀 흔적 없음.
