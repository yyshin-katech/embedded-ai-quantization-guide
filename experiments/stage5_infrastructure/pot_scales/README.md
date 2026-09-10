# Power-of-two 스케일 — INT8 크로스디바이스 비트동일성 완화책 (스크립트·런북·**실측: NO-GO**)

> **상태: 실행 완료(2026-09-10). 주 게이트 x86 ↔ Pi 5(A76) INT8 = 958/1000 → 869/1000(ceil)·919/1000(nearest) — NO-GO.**
> 두 라운딩 모드 모두 1000/1000에 못 미친다. 스케일을 2의 거듭제곱으로 강제해도 크로스디바이스 예측이
> 되레 **더 갈린다**(일치 958→869/919). 메커니즘 프로브가 이유를 지목한다 — MLAS는 `M = 2^k`를 **공유 정수
> 시프트로 특수처리하지 않는다.** 재양자화 float32 epilogue는 그대로 돌고, 격자만 거칠어져 x86(no-VNNI)와
> A76(SDOT) 두 커널 경로의 epilogue 발산이 **커진다**(평균 `|Δlogit|` ×4.34 ceil / ×2.35 nearest).
> 정본 수치는 리포트 [`../../../logs/stage5_pot_scales_report.html`](../../../logs/stage5_pot_scales_report.html)와 `results/`.

논문 C2의 헤드라인은 **같은 INT8 QDQ ONNX·같은 scale인데 정수 커널이 다르면 예측이 갈린다**는 것이다
(CPU↔CPU 958~965/1000, CPU↔iGPU 961/1000, CPU↔벤더NPU 939/1000 — FP32는 전부 1000/1000).
Chen 2026이 **단일 GPU·LLM**에서 그 원인을 재양자화 epilogue로 국소화하고 **power-of-two(PoT) 스케일이
커널 간 비트동일성을 복원**함을 보였다. 이 디렉터리가 묻는 것은 하나다:

> **그 완화책이 *물리적 디바이스 경계*를 넘어서도 성립하는가? — 실측 답: 아니다(NO-GO).**

Chen의 단일 GPU·LLM 결과는 x86 MLAS(no-VNNI) ↔ Pi 5 A76 MLAS(SDOT) **물리적 경계로 이식되지 않는다.**

---

## 메커니즘 — 왜 2의 거듭제곱인가 (그리고 왜 그 전제가 실측에서 깨졌는가)

INT8 GEMM의 INT32 누산은 **정확하다**. 두 커널이 갈릴 여지가 없다. 갈리는 곳은 그 뒤의 epilogue다:

```
out_int8 = round(acc_int32 × M) + zp,      M = (s_a · s_w) / s_out
```

`M`은 임의의 실수다. 커널마다 이걸 다르게 근사한다 — 고정소수점 multiplier+shift(ARM MLAS 계열),
FP32 곱(x86 경로), FMA 순서, 반올림 규칙. **`s`를 전부 2의 거듭제곱으로 강제하면**
`M = 2^(k_a + k_w − k_out)`가 되어 **순수 산술 시프트**가 된다 — **가 되어야 한다는 것이 완화책의 전제다.**

> **⚠ 실측이 이 전제를 반증했다.** MLAS는 `M = 2^k`를 감지해 공유 정수 시프트로 접지 **않는다.** float32
> 재양자화 epilogue는 그대로 실행되고, `s'`가 2의 거듭제곱이라 격자만 거칠어진다. 그 결과 두 ISA 경로의
> epilogue 발산은 제거되기는커녕 **확대된다**(평균 `|Δlogit|` 기준선 0.1441 → ceil 0.6254 ×4.34 / nearest
> 0.3383 ×2.35). 아래 [Go / No-Go 게이트](#go--no-go-게이트-실측-no-go) 참조.

`resnet50_int8_qdq.onnx`는 이 실험에 이상적이다 — 대칭 QInt8·per-channel·**zero_point가 전부 0**이고
INT32 bias DQ도 없다(`ActivationSymmetric`/`WeightSymmetric`, `QuantizeBias=False`,
[`../../stage3_tensorrt/t02_latency_3point.py`](../../stage3_tensorrt/t02_latency_3point.py)).
즉 epilogue에서 발산 가능한 항이 **`M` 하나뿐**이다(재작성 리포트 `nonzero_zero_points: []` 확인).
변수를 하나로 좁힌 상태로 그 하나를 제거하려 했고 — 제거되지 않았다.

---

## 기준선 (SSOT)과 실측 게이트

| precision | 대조 쌍 | 기준선 | PoT 후 (게이트) |
|---|---|--:|--:|
| fp32 | 전 6쌍 | **1000/1000** | 미재실행 (대조군, 불변 기대) |
| int8 | imx8mn(A53) ↔ jetson_orin_a78ae(A78AE) | 965/1000 | 미측정¹ |
| int8 | imx8mn(A53) ↔ rpi5(A76) | 965/1000 | 미측정¹ |
| int8 | imx8mn(A53) ↔ x86 | 961/1000 | 미측정¹ |
| int8 | jetson_orin_a78ae(A78AE) ↔ rpi5(A76) | **1000/1000** | 미측정¹ (같은 MLAS SDOT) |
| int8 | jetson_orin_a78ae(A78AE) ↔ x86 | 958/1000 | 미측정¹ |
| int8 | **rpi5(A76) ↔ x86** | **958/1000** | **869 (ceil) / 919 (nearest) — NO-GO** ← **주 게이트** |

¹ PoT 재실행은 **x86·Pi 5(A76) 두 대만** 수행했다. A53·A78AE·Jetson은 재실행하지 않았으므로 그 쌍들의
PoT 게이트는 미측정이다. 주 게이트(A76↔x86)만이 양쪽 다 PoT 덤프를 갖는 유일한 크로스머신 쌍이다.

기준선 출처는 [`../cpu_proxy/raw/*.json`](../cpu_proxy/raw)이고, `pot_agree.py --baseline ../cpu_proxy/raw`가
위 6개 INT8 쌍과 6개 FP32 쌍을 **그대로 재계산**한다(도구 검증 완료 — 논문 §5 표와 1:1 일치).
`results/`의 실측 덤프(주 게이트)에서 기준선 int8↔int8 대조군은 **958/1000로 재현**된다(메커니즘 프로브
`mech_*.json`의 `baseline.agree = 958`과도 일치).

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `pot_rewrite.py` | QDQ 그래프의 **모든** Q/DQ scale을 2의 거듭제곱으로 재작성. weight 초기화자는 새 scale에 맞춰 **재양자화**(scale만 바꾸면 가중치가 조용히 스케일된다 — 기본 `ceil`에선 최대 **2배**, `nearest`에선 최대 √2배). per-channel(`axis`) 처리. **최상위 그래프만** 재작성한다 — 초기화자가 아닌 scale(그래프 입력·`Constant` 노드)과 중첩 서브그래프(If/Loop/Scan 본문) 안의 Q/DQ는 손대지 않으므로 **리포트에 남기고 경고하며 쓰기 후 재검증에서 실패 처리**한다(조용히 빠뜨리느니 쓰기를 거부). `.partial`→check→rename로 "파일 존재=검증 통과" 불변식. `--selftest`로 onnx 없이 산술만 검증 가능. |
| `pot_bench.py` | 한 모델을 1,000장 번들에 돌려 `pred_cls` + **logits md5** 덤프. 전처리는 `../cpu_proxy/rpi_bench.py`와 **바이트 동일**, JSON 키는 그 상위집합이라 기준선과 직접 비교된다. |
| `pot_agree.py` | 덤프들 간 쌍별 top-1 일치 + logits 비트동일 여부 + **go/no-go 게이트** 판정. |
| `pot_mech.py` | 메커니즘 프로브 — x86↔Pi5 logits를 f64로 승격해 요소별 `|Δlogit|`(elem_abs_diff)·x86 margin·`|margin|≤τ` borderline 인구·flip vs noflip margin을 계산. 잔여 위험 #1(발산 확대)/#2(타이 브레이킹)를 실측으로 가른다. `--x86-int8 --pi-int8 --x86-pot --pi-pot --pot-label --out`. |

### 왜 logits md5까지 보는가

논문 C2의 지표는 **top-1 예측 일치**다. 그런데 PoT의 주장은 그보다 강한 **비트동일 출력**이다.
argmax는 1000/1000인데 logits는 다를 수 있다. md5가 그 구분을 4MB 배열 없이 끝낸다.
어긋나면 `--save-logits`로 배열을 남겨 얼마나 벌어졌는지 본다(`pot_mech.py`가 그 배열을 읽는다).

### 라운딩 모드 — `ceil`(기본) vs `nearest`, 그리고 실측 반전

`--rounding`으로 고른다. `resnet50_int8_qdq.onnx` 실모델 재작성 실측(27,616 scale 값·53 weight 텐서 재양자화):

- `ceil` (`k = ceil(log2 s)`, **기본**, `pot_rewrite_ceil.json`): `s' ≥ s`라 격자가 **거칠어지기만** 한다 →
  최대 log2 시프트 **0.99998**(≤1비트), **포화 0건**, 최대 weight 절대오차 0.0311.
- `nearest` (`k = round(log2 s)`, `pot_rewrite_nearest.json`): 격자는 가장 가깝지만(≤0.5비트→√2 이내)
  `s' < s`인 채널에선 int8 범위가 원래 극값에 못 미쳐 **가장 큰 가중치가 포화된다** — 실모델서
  **41,447개 포화**, 최대 weight 절대오차 1.6403.

**실측 반전 — 포화가 아니라 격자 거칠기가 크로스디바이스 발산을 몬다.** 직관은 "포화(nearest 41,447건)가
발산을 키운다"이겠지만 **정반대다**: 포화 0건인 `ceil`이 오히려 **더 갈린다**(주 게이트 869 < 919, 평균
`|Δlogit|` 0.6254 > 0.3383). `ceil`은 격자를 최대 1비트 거칠게 만들고(`nearest`는 0.5비트), 그 **격자 거칠기**가
float32 epilogue의 반올림 경계를 흔들어 두 ISA 경로를 더 자주 가른다. 비트동일성은 어차피 **두 모드 모두
실패**하므로(둘 다 NO-GO), 모드 선택은 이제 정확도·발산 대가의 문제일 뿐이다.

---

## 런북 (실행 완료 — 재현용)

### 0. 입력

- 모델: `_workspace/stage3/resnet50_int8_qdq.onnx`
  ([`../../stage3_tensorrt/t02_latency_3point.py`](../../stage3_tensorrt/t02_latency_3point.py)가 생성. 리포지터리에 커밋되지 않는 빌드 산출물)
- 데이터: `<data>/rpi_sub_u8.npy` (n,224,224,3 uint8) + `rpi_labels.npy` (n,) — `../cpu_proxy`와 **같은 번들**
  (같은 이미지·같은 순서여야 비교가 성립한다. `pot_agree.py`가 길이 불일치를 거부한다.)

### 1. PoT 모델 생성 (개발기 1회 — 두 모드 다)

```bash
mkdir -p results                          # 리포트·덤프가 여기로 간다 (커밋 안 함)

python pot_rewrite.py --selftest          # onnx 없이 산술 검증 (먼저)

# ceil (기본) — results/pot_rewrite_ceil.json
python pot_rewrite.py \
    --in  _workspace/stage3/resnet50_int8_qdq.onnx \
    --out _workspace/stage3/resnet50_int8_pot.onnx \
    --rounding ceil \
    --report results/pot_rewrite_ceil.json

# nearest — results/pot_rewrite_nearest.json, 출력 접미사 _potn
python pot_rewrite.py \
    --in  _workspace/stage3/resnet50_int8_qdq.onnx \
    --out _workspace/stage3/resnet50_int8_potn.onnx \
    --rounding nearest \
    --report results/pot_rewrite_nearest.json
```

리포트에서 확인할 것: `nonzero_zero_points`가 **빈 배열**(아니면 PoT로 못 지우는 발산원이 남는다 — 실측 `[]` 확인),
`scales_not_initializer`·`qdq_in_subgraphs`가 **빈 배열**(하나라도 있으면 그 노드의 `M`은 2의 거듭제곱이
아니고 실험이 무효다 — 실측 둘 다 `[]`), `weight_values_saturated`(ceil **0** / nearest **41,447**),
`scale_values_rewritten`(**27,616**), `--out`을 쓴 경우 `non_pot_after_write`·`scales_not_initializer_after_write`·
`qdq_in_subgraphs_after_write`가 빈 배열(쓰기 후 재검증 — 셋 중 하나라도 비지 않으면 스크립트가 비정상 종료한다).

> **A53만 추가 단계(미수행):** i.MX8M Nano(ORT 1.17.1)는 미사용 opset을 떼야 로드된다.
> `../cpu_proxy/README.md`의 "벽 (b)" 스니펫(노드 0개 변경)을 **두 모델 모두**에 적용해
> `resnet50_int8_qdq_op4.onnx`·`resnet50_int8_pot_op4.onnx`를 만들고, 2단계에서 `SUF=_op4`로 그 쌍을 가리킨다.
>
> ⚠️ **메모리 (미해결):** `pot_bench.py`의 `X = preprocess(u8)`는 `rpi_bench.py`와 똑같이
> `(n,3,224,224)` float32 배열을 통째로 올린다(n=1000이면 **약 602MB**). `../cpu_proxy/README.md` "벽 (a)"가
> 기록한 대로 그 경로는 2GB·무swap A53에서 **SIGKILL(rc=137)**로 죽었고, 그래서
> `rpi_bench_lowmem.py`(uint8 캐시 `mmap_mode="r"` + 이미지 1장씩 lazy 전처리, peak RSS 602→333MB)가 따로 있다.
> **`pot_bench.py`에는 그 lazy 경로가 아직 없다** — A53에서 2단계를 돌리려면 먼저 포팅해야 하고,
> 그 전까진 A53 행을 못 채운다. **주 게이트(x86↔A76)는 A53 무관이라 영향 없음** — 실행 완료.

### 2. 각 보드에서 측정 (x86·Pi 5(A76) 수행 완료)

```bash
# 이 디렉터리(experiments/stage5_infrastructure/pot_scales)에서 실행한다.
SOC=rpi5                                  # 이 보드의 라벨. imx8mn_a53 | jetson_orin_a78ae | x86
SUF=                                      # A53만 SUF=_op4 (위 "A53만 추가 단계"), 나머지는 빈 값
MDIR=_workspace/stage3                    # 1단계가 세 모델(qdq/pot/potn)을 쓴 곳
DATA=<data>                               # rpi_sub_u8.npy + rpi_labels.npy 가 있는 디렉터리
mkdir -p results

# baseline(대조군)·ceil·nearest를 같은 보드·같은 세션에서 연달아 — 스레드 수를 고정한다
for M in int8:resnet50_int8_qdq int8_pot:resnet50_int8_pot int8_potn:resnet50_int8_potn; do
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

**결정성 확인(실측):** 각 보드·정밀도 조합을 5회 반복(`results/repeat/*_r{2..5}.json`) — **2대 × 3정밀도 × 5회 =
30 런이 전부 비트동일**(`logits_md5` 편차 0). 크로스디바이스 불일치는 스레드 노이즈가 아니라 **결정론적 ISA
차이**다 — 게이트 수치는 점추정이 아니라 정확값이다.

### 3. 판정

보드들의 `results/`를 한 디렉터리로 모은 뒤:

```bash
python pot_agree.py --glob 'results/*_int8*.json' --gate
```

> **⚠ 글롭 주의(실행 중 드러난 함정):** 예전 런북은 `--glob 'results/*.json'`이었으나, `results/`에 이제
> 예측 덤프가 아닌 **리포트 JSON**(`pot_rewrite_ceil.json`·`pot_rewrite_nearest.json`·`mech_ceil.json`·
> `mech_nearest.json`)이 함께 있어 그 글롭은 `pot_agree.py`를 깨뜨린다:
> `results/mech_ceil.json has no pred_cls -- not a prediction dump`(exit 1). **`results/*_int8*.json`으로 좁히면**
> 6개 예측 덤프(`{soc}_int8*.json`)만 정확히 잡고 리포트 JSON은 제외된다(비재귀라 `results/repeat/`도 제외).
> 또는 6개 덤프를 명시적으로 나열해도 된다.

글롭을 `*_int8_pot.json`이 아니라 `*_int8*.json`으로 잡는 이유: 2단계가 같은 보드에서 뜬
baseline(`int8`) 덤프도 함께 남겼고, **같은 보드의 `int8` ↔ `int8_pot` 교차 쌍이 재작성의
정확도 대가를 읽는 자리**이기 때문이다(같은 커널·같은 이미지·같은 세션이라 남는 변수가 scale 재작성뿐 —
실측 x86 int8↔int8_pot 928/1000·int8↔int8_potn 858/1000, rpi5 865/872).
게이트는 여전히 `int8_pot`↔`int8_pot`·`int8_potn`↔`int8_potn` 쌍만 채점하므로 이 교차 쌍은 **참고값이지 게이트가 아니다**.
`--glob`은 반복 가능하니 보드별 디렉터리를 따로 넘겨도 된다.

`--baseline ../cpu_proxy/raw`는 **여기에 같이 넣지 말 것.** 그건 SSOT 재현용(위 기준선 표)이고,
2단계가 각 보드의 `int8`을 새로 떴으므로 같은 보드·같은 precision이 두 디렉터리에서 각각 로드돼
태그가 같은 run이 중복 생기고 대조군 표가 자기 자신과의 쌍으로 어지러워진다. 게이트 판정 자체는
영향받지 않지만(중복은 전부 대조군), 읽기 어려워질 뿐이라 두 명령을 따로 돌리는 편이 낫다.

메커니즘까지 보려면(잔여 위험 #1/#2 판정):

```bash
python pot_mech.py --x86-int8 results/x86_int8.json --pi-int8 results/rpi5_int8.json \
    --x86-pot results/x86_int8_pot.json --pi-pot results/rpi5_int8_pot.json \
    --pot-label int8_pot --out results/mech_ceil.json      # nearest는 _potn/_nearest로
```

---

## Go / No-Go 게이트 (실측: NO-GO)

**주 게이트 — x86 ↔ Pi 5(A76), INT8: 958/1000 → 1000/1000 이면 GO.**

**실측 결과 (`pot_agree.py --glob 'results/*_int8*.json' --gate`, exit 1):**

```
GATE: FAIL -- 2 of 2 PoT INT8 pair(s) short of 1000/1000:
   rpi5/int8_pot  <-> x86/int8_pot     869/1000 (131 differ)      # ceil
   rpi5/int8_potn <-> x86/int8_potn    919/1000 ( 81 differ)      # nearest
baseline INT8 pairs (control):
   rpi5/int8      <-> x86/int8         958/1000 ( 42 differ)
```

- **NO-GO 확정.** dotprod 있는 ARM 커널(A76 SDOT)과 없는 x86 커널이 **같은 정수 결과를 내지 않는다.**
  게다가 스케일을 2의 거듭제곱으로 강제하니 일치가 **되레 떨어졌다**(958 → 869/919). 논문 §11의 "untested"는
  이제 **"tested → NO-GO"로 닫힌다** — 안전 관련 이중화 컴퓨트에서 크로스디바이스 비트동일성의 처방은
  스케일 제약이 아니라 **타깃별 재검증**이다.
- logits는 물론 비트동일 아님(md5 전부 상이). argmax조차 갈리므로 "부분 성공"도 아니다.

정확도 대가(참고, 게이트 아님): top-1 x86 int8 0.7530 → ceil 0.7510 / nearest 0.7350; rpi5 int8 0.7500 →
ceil 0.7390 / nearest 0.7440. 어느 모드든 <2%p 손실이나 **게이트(일치)는 둘 다 실패**다.

### 잔여 위험 — 실측 판정

발단은 GO를 자동으로 만들지 못하는 두 위험이었다. 메커니즘 프로브(`mech_*.json`)가 각각을 실측으로 가른다:

1. **float32 epilogue 발산 — 확인·확대(CONFIRMED / AMPLIFIED).** `M`이 2의 거듭제곱이면 `acc × M`이 float
   곱에서 정확해져 발산이 사라져야 한다는 것이 전제였다. **실측은 반대**: MLAS가 `M = 2^k`를 공유 시프트로
   접지 않으므로 float32 epilogue가 그대로 돌고, 격자가 거칠어져 두 ISA 경로의 발산이 **커진다** — 요소별
   평균 `|Δlogit|` 0.1441 → **ceil 0.6254(×4.34)** / **nearest 0.3383(×2.35)**, 최대 `|Δlogit|` 2.184 → 8.0 / 4.75.
   불일치 장수도 42 → 131 / 81로 발산 크기를 **단조 추종**한다.
2. **타이 브레이킹 — 반증(REFUTED).** PoT가 `acc × M`을 정확히 `x.5`로 떨어뜨리는 경우를 늘려 half-even↔
   half-away 커널이 그 지점서 체계적으로 갈릴 것이라는 가설. **반증**: x86 margin이 `τ` 이하인 borderline 인구가
   **평평하다** — 기준선 41 → ceil 36 / nearest 45(모든 `τ` 1e-5~1e-1서 동일). PoT는 타이 인구를 체계적으로
   늘리지 않는다. 불일치는 **발산 크기**(위 #1)를 따라 늘지 타이/margin을 따라 늘지 않는다(flip이 작은 margin에
   몰리는 건 기준선과 같고, 늘어난 건 margin이 작아져서가 아니라 `|Δlogit|`가 커져서다).

즉 헤드라인은 **격자 거칠기가 크로스ISA epilogue 발산을 확대**한다는 것 하나로 수렴한다(#1 확대·#2 반증).
확인 실험이 아니라 **진짜 실험이었고, 결과는 완화책의 반증**이다.

---

## 확장 — CPU↔가속기 경계 (선택, Orin 필요) — CPU↔CPU가 NO-GO라 후순위

CPU↔CPU 주 게이트가 NO-GO로 닫혔으므로, 더 센 CPU↔iGPU 경계는 실익이 떨어졌다(더 약한 경계가
이미 실패). 기록만 남긴다: `rn50_gpu_int8.plan`은 **같은** `resnet50_int8_qdq.onnx`에서 빌드돼 CPU 프록시와
scale이 동일했고, 그 상태로 MLAS INT8 대비 **961/1000**이었다
([`../../stage3_tensorrt/jetson_ondevice/accuracy`](../../stage3_tensorrt/jetson_ondevice/accuracy)).
PoT ONNX로 엔진을 다시 빌드해 같은 대조를 돌리면 CPU↔iGPU 경계에서의 PoT 유효성이 나오지만,
CPU↔CPU조차 실패한 마당에 우선순위는 낮다.

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
이건 이 실험의 한계가 아니라 **결과의 일부**로 보고해야 한다: PoT는 *scale을 내가 정할 수 있는 경계*에서도
(CPU↔CPU) **처방으로 성립하지 않았고**(NO-GO), 벤더가 scale을 소유하는 경계에선 애초에 적용 불가다.

---

## 캐비앗 (불변)

- **실행 완료(2026-09-10, NO-GO).** 주 게이트(x86↔A76)만 측정 — A53·A78AE·Jetson은 PoT 재실행 미수행이라
  그 쌍의 게이트는 미측정이다. `results/`에 6개 예측 덤프 + 리포트/메커니즘 JSON + `repeat/`(30-런 결정성).
- 절대값(지연·top-1)은 배치1·1,000장 서브셋·경로 상이라 **상대 관계만 유효**(리포지터리 공통 캐비앗).
  게이트가 유효한 건 *같은 번들·같은 순서*의 예측을 1:1 비교하기 때문이다(정확도 대가 절대값은 상대만).
- 결론(격자 거칠기가 epilogue 발산 확대)은 **MLAS 계열 커널 쌍(x86 no-VNNI ↔ A76 SDOT)** 실측이다.
  다른 런타임/커널이 `M = 2^k`를 공유 정수 시프트로 특수처리한다면 결과가 다를 수 있으나, 이 경계에선 아니다.
- 입력/출력 측정이지 커널 내부 계측이 아니다 — "MLAS가 시프트로 접지 않는다"는 발산 확대로부터의 추론이다.
- 정확도 대가(top-1)는 측정했으나 **게이트는 *일치*를 보는 것**이지 정확도 보존을 보는 것이 아니다.
- 1,000장 번들 기준이라 5,000장 실행과 1:1 비교 불가.
- Chen 2026의 단일 GPU·LLM PoT 결과 자체를 반증하는 게 아니라, 그것이 **이 물리적 경계로 이식되지 않음**을
  반증한다(한 모델·두 라운딩 모드·한 경계).

## 관련

- 리포트(정본 수치): [`../../../logs/stage5_pot_scales_report.html`](../../../logs/stage5_pot_scales_report.html)
- 논문: [`../../../publication/paper1_isint8portable.md`](../../../publication/paper1_isint8portable.md) §5(완화책 미이식 문단) · §10(음성결과 범위) · §11(Conclusion) · Appendix A
- 기준선: [`../cpu_proxy/`](../cpu_proxy/) · 리포트 [`../../../logs/stage4_arm_cpu_fallback_report.html`](../../../logs/stage4_arm_cpu_fallback_report.html) · [`../../../logs/stage4_imx8mn_a53_report.html`](../../../logs/stage4_imx8mn_a53_report.html)
- CPU↔iGPU 961/1000: [`../../stage3_tensorrt/jetson_ondevice/accuracy/`](../../stage3_tensorrt/jetson_ondevice/accuracy/) · [`../../../logs/stage3_jetson_orin_accuracy_report.html`](../../../logs/stage3_jetson_orin_accuracy_report.html)
- 벤더 소유 양자화(C3): [`../../../logs/stage4_qualcomm_aihub_report.html`](../../../logs/stage4_qualcomm_aihub_report.html) · [`../../../logs/stage4_deepx_dxm1_accuracy_report.html`](../../../logs/stage4_deepx_dxm1_accuracy_report.html)
