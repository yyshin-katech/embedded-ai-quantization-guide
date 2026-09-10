# Power-of-two 스케일 — INT8 크로스디바이스 비트동일성 완화책 (스크립트·런북, **미실행**)

> **상태: 스크립트와 런북만 있다. 측정은 아직 하지 않았다.** 이 디렉터리엔 `results/`가 없고,
> 아래 표의 "PoT" 칸은 전부 비어 있다. 논문(§11 Conclusion)이 "untested and a concrete next step"이라고
> 쓴 그 실험의 **실행 가능한 형태**가 여기까지다. 수치를 인용하지 말 것 — 아직 없다.

논문 C2의 헤드라인은 **같은 INT8 QDQ ONNX·같은 scale인데 정수 커널이 다르면 예측이 갈린다**는 것이다
(CPU↔CPU 958~965/1000, CPU↔iGPU 961/1000, CPU↔벤더NPU 939/1000 — FP32는 전부 1000/1000).
Chen 2026이 **단일 GPU·LLM**에서 그 원인을 재양자화 epilogue로 국소화하고 **power-of-two(PoT) 스케일이
커널 간 비트동일성을 복원**함을 보였다. 이 디렉터리가 묻는 것은 하나다:

> **그 완화책이 *물리적 디바이스 경계*를 넘어서도 성립하는가?**

---

## 메커니즘 — 왜 2의 거듭제곱인가

INT8 GEMM의 INT32 누산은 **정확하다**. 두 커널이 갈릴 여지가 없다. 갈리는 곳은 그 뒤의 epilogue다:

```
out_int8 = round(acc_int32 × M) + zp,      M = (s_a · s_w) / s_out
```

`M`은 임의의 실수다. 커널마다 이걸 다르게 근사한다 — 고정소수점 multiplier+shift(ARM MLAS 계열),
FP32 곱(x86 경로), FMA 순서, 반올림 규칙. **`s`를 전부 2의 거듭제곱으로 강제하면**
`M = 2^(k_a + k_w − k_out)`가 되어 **순수 산술 시프트**가 된다. 근사할 것이 남지 않는다.

`resnet50_int8_qdq.onnx`는 이 실험에 이상적이다 — 대칭 QInt8·per-channel·**zero_point가 전부 0**이고
INT32 bias DQ도 없다(`ActivationSymmetric`/`WeightSymmetric`, `QuantizeBias=False`,
[`../../stage3_tensorrt/t02_latency_3point.py`](../../stage3_tensorrt/t02_latency_3point.py)).
즉 epilogue에서 발산 가능한 항이 **`M` 하나뿐**이다. 변수를 하나로 좁힌 상태로 그 하나를 제거한다.

---

## 기준선 (SSOT) — `pot_agree.py`가 그대로 재현한다

| precision | 대조 쌍 | 기준선 | PoT 후 |
|---|---|--:|--:|
| fp32 | 전 6쌍 | **1000/1000** | (대조군, 불변 기대) |
| int8 | imx8mn(A53) ↔ jetson_orin_a78ae(A78AE) | 965/1000 | — |
| int8 | imx8mn(A53) ↔ rpi5(A76) | 965/1000 | — |
| int8 | imx8mn(A53) ↔ x86 | 961/1000 | — |
| int8 | jetson_orin_a78ae(A78AE) ↔ rpi5(A76) | **1000/1000** | (같은 MLAS SDOT 커널 — 원래 일치) |
| int8 | jetson_orin_a78ae(A78AE) ↔ x86 | 958/1000 | — |
| int8 | **rpi5(A76) ↔ x86** | **958/1000** | — ← **주 게이트** |

기준선 출처는 [`../cpu_proxy/raw/*.json`](../cpu_proxy/raw)이고, `pot_agree.py --baseline ../cpu_proxy/raw`가
위 6개 INT8 쌍과 6개 FP32 쌍을 **그대로 재계산**한다(도구 검증 완료 — 논문 §5 표와 1:1 일치).
INT8 6행은 그 도구 출력의 6행과 순서까지 같다.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `pot_rewrite.py` | QDQ 그래프의 **모든** Q/DQ scale을 2의 거듭제곱으로 재작성. weight 초기화자는 새 scale에 맞춰 **재양자화**(scale만 바꾸면 가중치가 조용히 스케일된다 — 기본 `ceil`에선 최대 **2배**, `nearest`에선 최대 √2배). per-channel(`axis`) 처리. **최상위 그래프만** 재작성한다 — 초기화자가 아닌 scale(그래프 입력·`Constant` 노드)과 중첩 서브그래프(If/Loop/Scan 본문) 안의 Q/DQ는 손대지 않으므로 **리포트에 남기고 경고하며 쓰기 후 재검증에서 실패 처리**한다(조용히 빠뜨리느니 쓰기를 거부). `--selftest`로 onnx 없이 산술만 검증 가능. |
| `pot_bench.py` | 한 모델을 1,000장 번들에 돌려 `pred_cls` + **logits md5** 덤프. 전처리는 `../cpu_proxy/rpi_bench.py`와 **바이트 동일**, JSON 키는 그 상위집합이라 기준선과 직접 비교된다. |
| `pot_agree.py` | 덤프들 간 쌍별 top-1 일치 + logits 비트동일 여부 + **go/no-go 게이트** 판정. |

### 왜 logits md5까지 보는가

논문 C2의 지표는 **top-1 예측 일치**다. 그런데 PoT의 주장은 그보다 강한 **비트동일 출력**이다.
argmax는 1000/1000인데 logits는 다를 수 있다. md5가 그 구분을 4MB 배열 없이 끝낸다.
어긋나면 `--save-logits`로 배열을 남겨 얼마나 벌어졌는지 본다.

### 라운딩 모드 — `ceil`(기본) vs `nearest`

`--rounding`으로 고른다. selftest가 실측한 차이:

- `ceil` (`k = ceil(log2 s)`, **기본**): `s' ≥ s`라 격자가 **거칠어지기만** 한다 → 기존에 int8 범위에
  들어가던 값이 밖으로 나갈 일이 없다. 해상도를 최대 1비트 잃는다. **포화 0건.**
- `nearest` (`k = round(log2 s)`): 격자는 가장 가깝지만(√2 이내) `s' < s`인 채널에선 int8 범위가
  원래 극값에 못 미쳐 **가장 큰 가중치가 포화된다**. selftest 합성 케이스에서 288개 중 **20개 포화**,
  최대 오차가 `ceil`의 **18.8배**로 뛰었다(6.170e-02 → 1.157e+00).

비트동일성 자체는 **두 모드 모두** 성립한다(둘 다 `M`이 2의 거듭제곱). 모드 선택은 순전히 **정확도 대가**의
문제다. 그래서 기본을 `ceil`로 두었다.

---

## 런북

### 0. 입력

- 모델: `_workspace/stage3/resnet50_int8_qdq.onnx`
  ([`../../stage3_tensorrt/t02_latency_3point.py`](../../stage3_tensorrt/t02_latency_3point.py)가 생성. 리포지터리에 커밋되지 않는 빌드 산출물)
- 데이터: `<data>/rpi_sub_u8.npy` (n,224,224,3 uint8) + `rpi_labels.npy` (n,) — `../cpu_proxy`와 **같은 번들**
  (같은 이미지·같은 순서여야 비교가 성립한다. `pot_agree.py`가 길이 불일치를 거부한다.)

### 1. PoT 모델 생성 (개발기 1회)

```bash
mkdir -p results                          # 리포트·덤프가 여기로 간다 (커밋 안 함)

python pot_rewrite.py --selftest          # onnx 없이 산술 검증 (먼저)

python pot_rewrite.py \
    --in  _workspace/stage3/resnet50_int8_qdq.onnx \
    --out _workspace/stage3/resnet50_int8_pot.onnx \
    --rounding ceil \
    --report results/pot_rewrite_ceil.json
```

리포트에서 확인할 것: `nonzero_zero_points`가 **빈 배열**(아니면 PoT로 못 지우는 발산원이 남는다),
`scales_not_initializer`·`qdq_in_subgraphs`가 **빈 배열**(하나라도 있으면 그 노드의 `M`은 2의 거듭제곱이
아니고 실험이 무효다), `weight_values_saturated`가 **0**, `scale_initializers_rewritten`이 기대치,
`--out`을 쓴 경우 `non_pot_after_write`·`scales_not_initializer_after_write`·`qdq_in_subgraphs_after_write`가
빈 배열(쓰기 후 재검증 — 셋 중 하나라도 비지 않으면 스크립트가 비정상 종료한다).

> **A53만 추가 단계:** i.MX8M Nano(ORT 1.17.1)는 미사용 opset을 떼야 로드된다.
> `../cpu_proxy/README.md`의 "벽 (b)" 스니펫(노드 0개 변경)을 **두 모델 모두**에 적용해
> `resnet50_int8_qdq_op4.onnx`·`resnet50_int8_pot_op4.onnx`를 만들고, 2단계에서 `SUF=_op4`로 그 쌍을 가리킨다.
>
> ⚠️ **메모리 (미해결):** `pot_bench.py`의 `X = preprocess(u8)`는 `rpi_bench.py`와 똑같이
> `(n,3,224,224)` float32 배열을 통째로 올린다(n=1000이면 **약 602MB**). `../cpu_proxy/README.md` "벽 (a)"가
> 기록한 대로 그 경로는 2GB·무swap A53에서 **SIGKILL(rc=137)**로 죽었고, 그래서
> `rpi_bench_lowmem.py`(uint8 캐시 `mmap_mode="r"` + 이미지 1장씩 lazy 전처리, peak RSS 602→333MB)가 따로 있다.
> **`pot_bench.py`에는 그 lazy 경로가 아직 없다** — A53에서 2단계를 돌리려면 먼저 포팅해야 하고,
> 그 전까진 A53 행을 못 채운다(x86↔A76 **주 게이트는 영향 없음**).

### 2. 각 보드에서 측정

```bash
# 이 디렉터리(experiments/stage5_infrastructure/pot_scales)에서 실행한다.
SOC=rpi5                                  # 이 보드의 라벨. imx8mn_a53 | jetson_orin_a78ae | x86
SUF=                                      # A53만 SUF=_op4 (위 "A53만 추가 단계"), 나머지는 빈 값
MDIR=_workspace/stage3                    # 1단계가 두 모델을 쓴 곳
DATA=<data>                               # rpi_sub_u8.npy + rpi_labels.npy 가 있는 디렉터리
mkdir -p results

# baseline(대조군)과 PoT를 같은 보드·같은 세션에서 연달아 — 스레드 수를 고정한다
for M in int8:resnet50_int8_qdq int8_pot:resnet50_int8_pot; do
  P=${M%%:*}; F=${M##*:}
  python3 pot_bench.py --model $MDIR/${F}${SUF}.onnx --precision $P --data $DATA \
      --out results/${SOC}_${P}.json --soc $SOC --n 1000 \
      --warmup 20 --iters 200 --threads 4
done
```

`--soc`는 `pot_agree.py`가 쌍 이름을 짓는 데만 쓰므로 기준선 덤프와 **같은 라벨**을 써야
`rpi5/int8 <-> rpi5/int8_pot` 같은 교차 쌍이 한 보드로 묶인다.

`--threads`를 명시적으로 고정하는 이유: FP32 대조군의 비트동일성은 리덕션 순서에 의존하고
리덕션 순서는 스레드 수에 의존한다 — 논문 §5가 FP32 1000/1000을 얻은 조건 자체가
"fixed thread count, a single reduction path per target"이다. INT8 게이트를 볼 때 그 변수를 열어둘 이유가 없다.

### 3. 판정

보드들의 `results/`를 한 디렉터리로 모은 뒤:

```bash
python pot_agree.py --glob 'results/*.json' --gate
```

글롭을 `*_int8_pot.json`이 아니라 `results/*.json`으로 잡는 이유: 2단계가 같은 보드에서 뜬
baseline(`int8`) 덤프도 함께 남겼고, **같은 보드의 `int8` ↔ `int8_pot` 교차 쌍이 재작성의
정확도 대가를 읽는 자리**이기 때문이다(같은 커널·같은 이미지·같은 세션이라 남는 변수가 scale 재작성뿐).
게이트는 여전히 `int8_pot` ↔ `int8_pot` 쌍만 채점하므로 이 교차 쌍은 **참고값이지 게이트가 아니다**.
`--glob`은 반복 가능하니 보드별 디렉터리를 따로 넘겨도 된다.

`--baseline ../cpu_proxy/raw`는 **여기에 같이 넣지 말 것.** 그건 SSOT 재현용(위 기준선 표)이고,
2단계가 각 보드의 `int8`을 새로 떴으므로 같은 보드·같은 precision이 두 디렉터리에서 각각 로드돼
태그가 같은 run이 중복 생기고 대조군 표가 자기 자신과의 쌍으로 어지러워진다. 게이트 판정 자체는
영향받지 않지만(중복은 전부 대조군), 읽기 어려워질 뿐이라 두 명령을 따로 돌리는 편이 낫다.

---

## Go / No-Go 게이트

**주 게이트 — x86 ↔ Pi 5(A76), INT8: 958/1000 → 1000/1000 이면 GO.**

- **GO**: dotprod 있는 ARM 커널과 없는 x86 커널이 **같은 정수 결과**를 낸다 ⇒ PoT가 물리적 디바이스
  경계를 넘어서도 성립. 논문 §11의 "untested"가 닫히고, 안전 관련 이중화 컴퓨트에 **실행 가능한 처방**이 생긴다.
- **NO-GO (999/1000도 NO-GO)**: PoT의 요점은 epilogue가 시프트가 되어 **정확히 재현되거나 아니거나**라는
  것이다. 한 장이라도 갈리면 `M` 말고 다른 발산원이 남아 있다는 뜻이고, 그 잔여 원인을 지목하는 것이
  다음 과제가 된다.

부가 게이트: 나머지 INT8 쌍 — A53↔A78AE(965), A53↔A76(965), A53↔x86(961), A78AE↔x86(958) — 도 같은 기준이고,
이미 1000/1000인 A78AE↔A76은 **불변**이어야 한다(퇴행 감시). 단 A53이 들어가는 세 쌍은
2단계의 메모리 벽(위 ⚠️)이 풀리기 전엔 측정 자체가 불가다. 그리고 `logits_md5`가 **일치**해야 완전한 GO다
(argmax만 일치하고 logits가 갈리면 "예측은 같지만 비트동일은 아니다" — 부분 성공으로 따로 보고할 것).

**게이트가 채점하는 쌍 / 채점하지 않는 쌍.** `pot_agree.py`는 **양쪽이 다 PoT인 쌍**(`int8_pot` ↔ `int8_pot`)만
채점한다. 기준선 `int8` ↔ `int8` 쌍은 **재작성 전 대조군**이고 그 값은 PoT의 성패가 아니라 기준선의 성질이다
— 여섯 쌍 중 다섯이 958~965/1000이므로(나머지 하나 A78AE↔A76만 원래 1000/1000) 이들을 채점하면
PoT가 아무리 잘 돼도 `--gate`가 무조건 실패한다. 그래서 함께 출력하되 실패로 세지 않는다.
같은 보드의 `int8` ↔ `int8_pot` 교차 쌍은 **재작성 자체가 예측을 얼마나 흔드는지**(=정확도 대가를 읽는 자리)를
보여주는 참고값이지 게이트가 아니다. PoT 쌍이 하나도 없으면 게이트는 PASS가 아니라 **N/A**(`--gate` 시 exit 2)다.

### 잔여 위험 — GO가 자동이 아닌 이유 둘

1. **float32 epilogue의 2²⁴ 벽.** `M`이 2의 거듭제곱이면 `acc × M`은 float 곱에서 **정확**하다 —
   단 `|acc_int32| < 2²⁴ ≈ 1.68e7`일 때만. ResNet50의 후반 레이어 누산은 이 경계에 근접한다.
   경계를 넘으면 float 곱 epilogue와 고정소수점 시프트 epilogue가 **여전히 갈릴 수 있다**.
2. **타이 브레이킹.** PoT는 `acc × M`을 정확히 `x.5`로 떨어뜨리는 경우를 **오히려 늘린다**.
   round-half-to-even과 round-half-away-from-zero가 다른 두 커널은 그 지점에서 체계적으로 갈린다.
   즉 PoT는 "근사 오차"는 지우지만 "반올림 규칙 불일치"는 지우지 못한다.

둘 다 실측으로만 갈린다. 그래서 이건 확인 실험이 아니라 **진짜 실험**이다.

---

## 확장 — CPU↔가속기 경계 (선택, Orin 필요)

CPU↔CPU가 GO면 다음은 더 센 경계다. `rn50_gpu_int8.plan`은 **같은** `resnet50_int8_qdq.onnx`에서 빌드돼
CPU 프록시와 scale이 동일했고, 그 상태로 MLAS INT8 대비 **961/1000**이었다
([`../../stage3_tensorrt/jetson_ondevice/accuracy`](../../stage3_tensorrt/jetson_ondevice/accuracy)).
PoT ONNX로 엔진을 다시 빌드해 같은 대조를 돌리면 CPU↔iGPU 경계에서의 PoT 유효성이 나온다.

⚠️ 단, TensorRT가 파서/빌더 단계에서 scale을 **재유도하거나 융합할 수 있다**. PoT가 엔진까지 살아남는지
자체가 검증 대상이다 — 빌드 후 엔진의 실효 scale을 확인하지 않은 채 결과를 해석하면 안 된다.

---

## 왜 Qualcomm HTP와 DEEPX DX-M1은 이 실험에서 **구조적으로 제외**되는가

C2(수치 비이식성)와 C3(벤더가 양자화를 소유)의 **교차점**이다. PoT 완화책은 **내가 scale을 정할 수 있을 때만**
쓸 수 있다. 두 벤더 NPU에선 그 전제가 성립하지 않는다 — 실패 양식만 정반대다:

| 타깃 | 외부 QDQ(=내 scale) 지참 시 | PoT 적용 가능성 |
|---|---|---|
| **Qualcomm Hexagon HTP** | **조용히 무시**하고 자체 양자화 → on-device top-1 **0.75 → 0.005 붕괴**(compile·profile은 통과). 정상 경로는 AI Hub `submit_quantize_job`(HTP-native QDQ) → 0.735 | **불가.** 내 scale이 애초에 반영되지 않는다. PoT로 써 넣어도 무시된다. |
| **DEEPX DX-M1** | **시끄럽게 거부**: `dx_com`이 `GraphStructureError: 106 isolated node(s)` → `InternalError`, `.dxnn` 미산출. 정상 경로는 FP32 + `dx_com` 자체 PTQ | **불가.** 외부 QDQ 그래프 자체가 컴파일되지 않는다. |

따라서 **CPU↔벤더NPU 939/1000은 PoT로 닫을 수 있는 종류의 격차가 아니다.** 벤더가 scale을 소유하는 한
크로스디바이스 비트동일성은 **벤더 툴체인이 PoT를 지원해야** 가능하다 — 사용자 쪽 레버가 없다.
이건 이 실험의 한계가 아니라 **결과의 일부**로 보고해야 한다: PoT는 *scale을 내가 정할 수 있는 경계*
(CPU↔CPU, 그리고 아마 CPU↔TensorRT)에서만 처방으로 성립한다.

---

## 캐비앗 (불변)

- **미실행.** 위 표의 PoT 칸은 비어 있고 `results/`도 없다. 스크립트·런북·게이트 정의까지가 산출물이다.
  (개발기에 `onnx`/`onnxruntime` 미설치 — `pot_rewrite.py --selftest`와 `pot_agree.py`의 기준선 재현만
  로컬 검증했다.)
- 절대값(지연·top-1)은 배치1·1,000장 서브셋·경로 상이라 **상대 관계만 유효**(리포지터리 공통 캐비앗).
- 정확도 대가는 **미측정**이다. PoT는 격자를 최대 1비트 거칠게 만든다. 게이트는 *일치*를 보는 것이지
  *정확도 보존*을 보는 것이 아니다 — top-1이 얼마나 떨어지는지는 같은 실행에서 함께 보고해야 한다.
- 1,000장 번들 기준이라 5,000장 실행과 1:1 비교 불가.

## 관련

- 논문: [`../../../publication/paper1_isint8portable.md`](../../../publication/paper1_isint8portable.md) §5(C2 헤드라인) · §6(C3) · §11(Conclusion의 future work 문단)
- 기준선: [`../cpu_proxy/`](../cpu_proxy/) · 리포트 [`../../../logs/stage4_arm_cpu_fallback_report.html`](../../../logs/stage4_arm_cpu_fallback_report.html) · [`../../../logs/stage4_imx8mn_a53_report.html`](../../../logs/stage4_imx8mn_a53_report.html)
- CPU↔iGPU 961/1000: [`../../stage3_tensorrt/jetson_ondevice/accuracy/`](../../stage3_tensorrt/jetson_ondevice/accuracy/) · [`../../../logs/stage3_jetson_orin_accuracy_report.html`](../../../logs/stage3_jetson_orin_accuracy_report.html)
- 벤더 소유 양자화(C3): [`../../../logs/stage4_qualcomm_aihub_report.html`](../../../logs/stage4_qualcomm_aihub_report.html) · [`../../../logs/stage4_deepx_dxm1_accuracy_report.html`](../../../logs/stage4_deepx_dxm1_accuracy_report.html)
