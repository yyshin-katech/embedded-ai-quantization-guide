# 1. 양자화 이론 (Quantization Theory)

> **원본 가이드 매핑**: "1단계 — 양자화 이론 (1~2주)"
> **예상 소요**: 1~2주 (이론 3~4일 + 실습 3~4일)
> **선행 조건**: [0.5단계 배포 사다리](02_deployment_ladder.md) 완료 · Ubuntu 22.04 + NVIDIA RTX GPU · Python 3.10+ / PyTorch / CUDA 12.8 동작 확인

---

## 0) 이 단계에서 무엇을·왜 하는가

임베디드 AI 엔지니어 면접에서 **가장 먼저 검증되는 것이 양자화 수식**이다. "INT8로 줄이면 빨라진다"는 누구나 안다. 하지만 `q = round(x/s) + z`를 화이트보드에 유도하고, "왜 weight는 per-channel symmetric이고 activation은 per-tensor asymmetric인가"를 5분 안에 설명하지 못하면 그 자리에서 탈락한다.

이 단계의 목표는 두 가지다.

1. **손으로 유도할 수 있는 수준**의 이론 내재화 — scale/zero-point, symmetric/asymmetric, per-tensor/per-channel, QDQ 그래프, 캘리브레이션 4종, QAT+STE. 단순 암기가 아니라 **"양자화 오차 = rounding error + clipping error"라는 오차 분해**에서 모든 결정(scale 선택, 캘리브레이션 방법, per-channel 여부)이 왜 나오는지를 하나의 논리로 꿴다.
2. **범인 레이어를 특정하는 실전 감각** — ResNet18을 실제로 INT8 PTQ 하고, 캘리브레이터(MinMax vs Entropy vs Percentile)에 따라 top-1이 어떻게 달라지는지, 그리고 **어느 레이어가 정확도를 깎아먹는지**를 SQNR/코사인 유사도로 정량화한다.

이 단계의 산출물 `layer_sensitivity.csv`는 **이후 모든 단계(2단계 Transformer 양자화, 3단계 TensorRT mixed precision)에서 "어디를 FP16으로 남길지" 결정하는 입력**이 된다. 즉, 여기서 만드는 감도 분석 방법론은 재사용된다. 이 CSV는 **이 문서의 1단계 실습(4.5절)이 직접 생성**한다.

> 💡 **팁**: 이 단계는 보드가 필요 없다. x86 PC(RTX GPU)에서 100% 완결된다. ONNX Runtime의 CPU/CUDA만으로 PTQ와 정확도 평가가 끝난다.

### 이 단계를 관통하는 한 문장

> **양자화란 "연속 실수축을 균등한 정수 격자에 스냅(snap)하는" 것이고, 정확도 손실은 오직 두 곳에서만 발생한다 — (1) 격자 안에서 가장 가까운 눈금으로 반올림할 때(rounding), (2) 격자 밖으로 삐져나간 값을 격자 끝으로 잘라낼 때(clipping).** 이 두 오차는 서로 상충(trade-off)하며, scale `s` 하나가 그 균형점을 결정한다.

이 문장을 손에 쥐고 있으면 2.1~2.4절의 모든 내용이 "그래서 이 오차를 어떻게 줄이나"의 변주로 읽힌다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] `q = round(x/s) + z` 와 `x̂ = s·(q − z)` 를 백지에 유도하고, s·z를 min/max로부터 계산할 수 있다.
- [ ] **양자화 오차를 rounding error와 clipping error로 분해**하고, scale을 키우면/줄이면 각 항이 어떻게 움직이는지(trade-off) 수식과 수치 예로 설명할 수 있다.
- [ ] symmetric vs asymmetric, per-tensor vs per-channel을 표로 설명하고 **"weight=per-channel symmetric, activation=per-tensor asymmetric"인 이유**를 채널별 분포 수치 예와 함께 서술할 수 있다.
- [ ] QDQ 그래프가 무엇이며 왜 "모든 툴체인의 공용어"인지 설명하고, ONNX 안의 `QuantizeLinear`/`DequantizeLinear` 노드 쌍을 읽을 수 있다.
- [ ] 캘리브레이션 4종(MinMax / Percentile / Entropy(KL) / MSE)의 원리·장단·언제 쓰는지를 비교표로 설명하고, **특히 Entropy(KL)의 히스토그램 구성 → 후보 threshold별 KL 최소화 절차를 의사코드 수준으로** 설명할 수 있다.
- [ ] QAT의 fake-quant와 STE(Straight-Through Estimator)를 `torch.autograd.Function`으로 구현하고, 미니 학습 루프(옵티마이저·손실)로 돌릴 수 있다. QAT가 실모델(ResNet18)에서 PTQ 손실을 얼마나 되찾는지, 그리고 대조군(동일 학습 FP32) 대비 그 회복이 '공짜'가 아님을 실측으로 설명할 수 있다. LSQ(learnable scale)가 무엇인지 한 문장으로 말할 수 있다.
- [ ] ResNet18을 ONNX로 export하고, ONNX Runtime `quantize_static`으로 INT8 PTQ(MinMax/Entropy/Percentile, per-channel 옵션)를 수행.
- [ ] **타깃 하드웨어에 따라 dtype 선택이 갈리는 이유**를 설명하고(x86 = QUInt8 비대칭 / TensorRT = QInt8 대칭), 잘못 고르면 **에러 없이 FP32보다 느려지는** 무음 폴백이 발생함을 안다(4.3.2).
- [ ] **라이브러리 기본값이 알고리즘을 조용히 무력화할 수 있음**을 안다 — ORT `Entropy`가 MinMax로 퇴화하는 사례를 재현하고, scale 비교로 자가진단할 수 있다(4.3.1).
- [ ] ImageNet-val 1000장으로 FP32 vs INT8(각 캘리브레이션) top-1을 비교하고, **표본 수에 따른 신뢰구간**과 **paired 검정(McNemar)** 으로 "이 차이가 통계적으로 유의한가"를 판단할 수 있다.
- [ ] 레이어별 SQNR/코사인 유사도를 계산해 **`layer_sensitivity.csv`** 를 생성(다음 단계 입력)하고, **어떤 임계로 FP16 승격을 결정하는지** 서술할 수 있다.
- [ ] 논문 3편(Gholami / Nagel / Jacob)을 읽고 각 1문단으로 요약.

---

## 2) 배경 이론 / 개념

### 2.1 균등 아핀 양자화 (Uniform Affine Quantization)

실수(FP32) 텐서 `x`를 정수 `q`로 매핑하는 가장 기본 공식:

```
양자화 (quantize):    q = clamp( round(x / s) + z,  q_min,  q_max )
역양자화 (dequantize): x̂ = s · (q − z)
```

- **s (scale)**: 실수 한 스텝의 크기(격자 눈금 간격). `s = (real_max − real_min) / (q_max − q_min)`
- **z (zero-point)**: 실수 0이 매핑되는 정수 값. `z = q_min − round(real_min / s)`
- **clamp**: `q`를 표현 가능한 정수 범위 `[q_min, q_max]`로 자름 (INT8 signed면 `[-128, 127]`, unsigned면 `[0, 255]`).

#### 왜 이 공식인가 (유도)

우리가 원하는 것은 **실수 구간 `[β, α]`(β=real_min, α=real_max)를 정수 구간 `[q_min, q_max]`에 1차(아핀) 함수로 겹치는 것**이다. 아핀 함수는 `q = a·x + b` 꼴이고, 두 끝점이 대응해야 하므로:

```
q_max = a·α + b       ...(끝 대 끝)
q_min = a·β + b       ...(시작 대 시작)
```

두 식을 빼면 `q_max − q_min = a·(α − β)` → 기울기 `a = (q_max − q_min)/(α − β)`. 그런데 우리는 반대 방향(정수→실수 눈금 간격)을 `s`로 정의하는 게 하드웨어에 편하므로 `s ≡ 1/a = (α − β)/(q_max − q_min)`, 즉 `a = 1/s`. 이제 절편은 아래쪽 끝점 식에서 `b = q_min − a·β = q_min − β/s`. `z ≡ b`로 두면:

```
q = x/s + z,   z = q_min − β/s = q_min − round(real_min / s)
```

`round`와 `clamp`는 "실수축의 아핀 이미지"를 **정수 격자에 스냅**하고 **표현 범위 밖을 잘라내기** 위해 뒤에 씌운 것이다. 역함수는 `q = x/s + z`에서 `x`를 풀면 `x̂ = s·(q − z)`. 이렇게 `s`와 `z`가 min/max에서 유일하게 결정된다.

> 💡 **직관**: `s`는 "자의 눈금 간격", `z`는 "자를 어디서부터 세느냐(원점 위치)"다. symmetric은 자의 원점(정수 0)을 실수 0에 못 박은 것(z를 고정), asymmetric은 자를 좌우로 밀어 실제 데이터에 딱 맞춘 것(z를 자유롭게).

**유도 예시 (asymmetric, uint8):** 활성값 범위가 `real_min = -1.0`, `real_max = 3.0`, uint8이면 `q_min=0, q_max=255`.

```
s = (3.0 - (-1.0)) / (255 - 0) = 4.0 / 255 ≈ 0.01569
z = 0 - round(-1.0 / 0.01569) = 0 - round(-63.75) = 0 - (-64) = 64
```
→ 실수 `x = 0.5` 는 `q = round(0.5/0.01569) + 64 = round(31.9) + 64 = 32 + 64 = 96`.
→ 다시 `x̂ = 0.01569 · (96 - 64) = 0.01569 · 32 = 0.502` (양자화 오차 ≈ 0.002).

> 💡 **팁**: 여기서 발생하는 `x - x̂` 가 **양자화 노이즈**다. 캘리브레이션과 QAT는 전부 "이 노이즈를 어떻게 줄이느냐"의 문제다.

### 2.1.1 양자화 오차의 분해 — rounding error vs clipping error

이 절이 이 문서에서 가장 중요하다. 양자화 오차 `e(x) = x − x̂` 를 **두 성분으로 완전히 분리**하면, scale 선택·캘리브레이션·per-channel의 모든 이유가 한 줄로 설명된다.

임의의 실수 `x`에 대해:

```
                ┌ x가 표현 범위 [β, α] 안에 있으면 → rounding error(반올림 오차)
e(x) = x − x̂ = ┤
                └ x가 범위 밖이면            → clipping error(포화 오차)
```

**(A) Rounding error (반올림 오차).** 범위 **안**의 값은 가장 가까운 격자 눈금(간격 `s`)으로 스냅되므로, 오차는 `[−s/2, +s/2]` 구간에 갇힌다. 값이 격자 안에서 "고르게 퍼져 있다"고 가정하면(uniform 근사), 반올림 오차는 폭 `s`인 균등분포이고 그 분산은 잘 알려진 결과:

```
Var(rounding error) = s² / 12       (균등분포 [−s/2, s/2]의 분산)
```

→ **핵심: scale `s`가 작을수록(격자가 촘촘) 반올림 오차는 작아진다. 오차의 RMS ≈ s/√12.**

**(B) Clipping error (포화 오차).** 범위 **밖**의 값(예: `x > α`)은 전부 `α`(격자 끝)로 잘리므로 오차 `= x − α`가 크게 남는다. 분포의 꼬리를 `T`(=범위 절반, threshold)에서 자른다면, 클리핑 오차의 기댓값은 꼬리에 얼마나 많은 확률질량이 있느냐로 결정된다:

```
E[clipping error²] ≈ ∫_{|x|>T} (|x| − T)² · p(x) dx
```

→ **핵심: 범위 `[β, α]`를 넓게 잡을수록(=T를 크게) 꼬리를 덜 자르니 클리핑 오차는 줄지만, 대신 같은 128개 눈금으로 더 넓은 구간을 덮어야 하므로 `s = (α−β)/(q_max−q_min)`가 커져 반올림 오차가 커진다.**

#### 총오차 = rounding + clipping, 그리고 상충(trade-off)

```
E[e²] ≈ (범위 안 반올림)  +  (범위 밖 클리핑)
      ≈  s²/12 · P(|x|≤T)  +  E[(|x|−T)² · 1{|x|>T}]
                ▲                          ▲
        T ↑ 이면 s ↑ → 증가        T ↑ 이면 꼬리 ↓ → 감소
```

이것이 **캘리브레이션의 본질**이다. `T`를 너무 크게(=MinMax처럼 outlier까지 포함) 잡으면 반올림 오차가 폭증하고, 너무 작게 잡으면 클리핑 오차가 폭증한다. **최적 `T`는 둘의 합이 최소가 되는 지점**이며, 이걸 어떻게 찾느냐가 MinMax/Percentile/Entropy/MSE를 가른다(2.4절).

> 💡 **직관 한 줄**: "자를 촘촘하게(작은 s) 하려면 짧게 잘라야(작은 범위) 하고, 길게 재려면(큰 범위) 눈금이 성겨진다(큰 s). 어느 쪽이든 공짜가 없다."

#### Worked example — scale 선택이 총오차를 어떻게 바꾸는가

한 activation 텐서가 대략 표준정규 `N(0,1)`를 따르되 값 하나가 **outlier로 8.0**에 있다고 하자(예: 10000개 중 1개). signed int8, symmetric(z=0), `q_max=127`. 세 가지 범위 선택 `T`(=절대 최대 표현값)에 대해 총오차를 손으로 비교한다.

| 선택 | T (범위 절반) | s = T/127 | rounding RMS ≈ s/√12 | 클리핑되는 값 | clipping 기여 |
|------|--------------|-----------|----------------------|--------------|---------------|
| **MinMax** | 8.0 (outlier까지 포함) | 0.0630 | **0.0182** | 없음 | 0 |
| **Percentile 99.9%** | ≈ 3.09 (N(0,1)의 99.9% 지점) | 0.0243 | **0.0070** | outlier 1개(8.0→3.09, 오차 4.91) + 극소수 꼬리 | 작음(1/10000 확률) |
| **너무 좁게** | 1.0 | 0.00787 | **0.00227** | 값의 약 32%(|x|>1) | **큼** |

계산 근거:
- rounding RMS: `s/√12 = (T/127)/3.464`. MinMax는 `8/127/3.464 = 0.0182`, Percentile은 `3.09/127/3.464 = 0.0070`, 좁게는 `1/127/3.464 = 0.00227`.
- 클리핑: MinMax는 아무것도 안 자르므로 0. Percentile은 outlier 1개(오차 4.91²≈24.1)를 10000개로 평균 내면 `24.1/10000 ≈ 0.0024`의 제곱기여 → RMS 기여 `√0.0024 ≈ 0.049`이지만 이는 **단 하나의 값에만** 실리고, 나머지 9999개는 반올림 오차 0.0070만 받는다. 좁게(T=1)는 전체의 32%가 크게 잘려 평균 오차가 급증.

**결론(정량적)**:
- **MinMax**: outlier 1개 때문에 s가 8/3.09 ≈ 2.6배 커져, **9999개의 정상 값 전부가 2.6배 큰 반올림 오차**(0.0070 → 0.0182)를 뒤집어쓴다. outlier 1개 살리자고 나머지를 다 희생. → **outlier에 취약**.
- **Percentile 99.9%**: 정상 값 9999개는 최소 반올림 오차(0.0070)를 누리고, outlier 1개만 잘린다. **전체 텐서의 평균 제곱오차(MSE)로 보면 Percentile이 MinMax를 크게 이긴다.**
- **너무 좁게(T=1)**: 반올림은 최소지만 32%를 잘라 클리핑이 지배 → **최악**.

> 🔴 **함정**: "범위를 넉넉히 잡으면 안전하다"는 직관은 틀렸다. 넉넉한 범위(MinMax)는 **모든 정상 값의 해상도를 outlier에 인질로 잡힌다**. 이 예시의 숫자(0.0070 vs 0.0182)를 면접에서 대면 "오차 분해를 이해했다"는 강한 신호다.

> ⚠️ **주의 (이 결론이 성립하는 전제를 반드시 함께 기억할 것)**: 위 계산은 전부 옳지만, **"8.0에 있는 그 값은 버려도 되는 잡음"** 이라는 전제 위에 서 있다. 그 값이 실제로는 **모델이 의존하는 강한 특징 응답**이라면, MSE는 좋아져도 **정확도는 오히려 떨어진다** — MSE는 모든 원소를 동등하게 세지만 신경망의 출력은 그렇지 않기 때문이다. 실제로 이 문서의 ResNet18 실측에서는 **MinMax가 최적이고 Percentile 99.9는 −6.2%p로 무너졌다**(5.1). 즉 2.1.1은 **"오차가 어디서 오는가"를 분해하는 도구**이지 **"어느 캘리브레이터가 이긴다"는 결론이 아니다.** 부등호의 방향은 텐서의 분포가 정한다.

### 2.2 Symmetric vs Asymmetric, Per-tensor vs Per-channel

**Symmetric(대칭)**: `z = 0` 으로 고정. 범위를 `[-|max|, +|max|]`로 잡는다. 정수 0 ↔ 실수 0 이 정확히 일치 → 곱셈 시 zero-point 항이 사라져 **정수 연산이 훨씬 단순**해진다(아래 2.2.1 유도 참고).

**Asymmetric(비대칭)**: `z ≠ 0` 허용. 범위를 실제 `[min, max]`에 딱 맞춤 → **한쪽으로 치우친 분포**(예: ReLU 뒤의 항상 ≥0인 활성값)를 낭비 없이 표현.

| 구분 | 정의 | 장점 | 단점 | 주 사용처 |
|------|------|------|------|-----------|
| **Symmetric** | z=0, `[-a, +a]` | 정수 MAC 단순(zero-point 항 제거), 하드웨어 친화 | 0 기준 비대칭 분포는 절반 낭비 | **weight** |
| **Asymmetric** | z≠0, `[min, max]` | 치우친 분포를 꽉 채워 표현 → 해상도↑ | zero-point 보정 연산 추가 | **activation** |
| **Per-tensor** | 텐서 1개당 (s, z) 1쌍 | 메모리·연산 최소, 모든 HW 지원 | 채널 간 분포 차이 큰 weight에서 손실 | **activation** |
| **Per-channel** | 출력 채널마다 (s, z) | 채널별 범위를 각각 최적화 → 정확도↑ | activation엔 부적합(HW·연산 복잡), weight 전용 | **weight** (Conv/Linear의 output축) |

#### 2.2.1 왜 symmetric이면 정수 MAC이 단순해지는가 (유도)

`y = Σᵢ wᵢ·xᵢ` 를 정수로 계산한다고 하자. asymmetric이면 `wᵢ ≈ s_w(q_wᵢ − z_w)`, `xᵢ ≈ s_x(q_xᵢ − z_x)`이므로:

```
y ≈ Σ s_w(q_wᵢ − z_w) · s_x(q_xᵢ − z_x)
  = s_w·s_x · Σ (q_wᵢ − z_w)(q_xᵢ − z_x)
  = s_w·s_x · [ Σ q_wᵢ·q_xᵢ  −  z_w·Σ q_xᵢ  −  z_x·Σ q_wᵢ  +  N·z_w·z_x ]
                     ▲               ▲              ▲              ▲
                본 계산(INT MAC)   교차항1        교차항2       상수항
```

교차항이 **3개나 추가**된다. 그런데 **weight를 symmetric(z_w = 0)** 으로 두면:

```
y ≈ s_w·s_x · [ Σ q_wᵢ·q_xᵢ  −  z_x·Σ q_wᵢ ]
```

`z_w`가 낀 두 항이 통째로 사라지고, 남은 `z_x·Σ q_wᵢ` 는 `Σ q_wᵢ`가 **weight만의 상수**라 미리(오프라인) 계산해 bias에 흡수할 수 있다. 결과적으로 런타임에는 순수 `Σ q_wᵢ·q_xᵢ`(INT8×INT8→INT32 MAC)만 남는다. 이것이 Jacob 2018 "integer-arithmetic-only inference"의 핵심이고, **weight를 symmetric으로 하는 하드웨어적 이유**다.

> 💡 **직관**: symmetric은 "곱셈의 원점을 0에 맞춰 교차항을 없애는" 트릭이다. activation은 런타임 텐서라 `z_x`를 없앨 수 없지만(치우친 분포를 살려야 해 asymmetric이 이득), weight는 정적이라 `z_w=0`의 대가(약간의 표현 낭비)가 교차항 제거 이득보다 작다.

#### 2.2.2 채널별 분포 차이 — per-channel이 weight에 유리한 이유 (수치 예)

Conv 레이어의 weight는 `[out_channels, in_channels, kh, kw]` 이고, **출력 채널(필터)마다 크기(dynamic range)가 크게 다르다**. 3개 출력 채널의 절대 최댓값이 아래처럼 다르다고 하자:

| 출력 채널 | |max| (그 채널 weight의 절대 최대) |
|-----------|-----------------------------------|
| 채널 A | 0.12 |
| 채널 B | 0.95 |
| 채널 C | 3.40 |

signed int8 symmetric, `q_max = 127`.

**Per-tensor (전 채널 공통 s):** 텐서 전체 `|max| = 3.40` 하나로 `s = 3.40/127 = 0.02677`.
- 채널 A의 값들은 최대 0.12 → 정수로 `0.12/0.02677 ≈ 4.5`, 즉 **±4 정도의 정수 레벨만** 사용. 128단계 중 실질 9단계(−4~+4)만 쓰고 나머지는 낭비. → 채널 A의 유효 비트 ≈ log2(9) ≈ **3.2 bit**.
- 채널 C만 128단계를 꽉 씀. **채널 A는 INT8인데 사실상 3비트 해상도**로 뭉개진다.

**Per-channel (채널별 s):** 채널마다 자기 `|max|`로 s를 따로.
- 채널 A: `s_A = 0.12/127 = 0.000945` → 채널 A 값이 −127~+127 전 범위를 사용. **온전한 8비트.**
- 채널 B: `s_B = 0.95/127`, 채널 C: `s_C = 3.40/127`. 각자 128단계 풀 활용.

**정량 비교 (채널 A의 반올림 RMS):**
- per-tensor: `s/√12 = 0.02677/3.464 = 0.00773`. 채널 A 값(최대 0.12) 대비 상대오차 `0.00773/0.12 ≈ 6.4%`.
- per-channel: `s_A/√12 = 0.000945/3.464 = 0.000273`. 상대오차 `0.000273/0.12 ≈ 0.23%`.
- → **per-channel이 채널 A에서 약 28배(0.00773/0.000273) 정확**하다. dynamic range가 좁은 채널일수록 이득이 폭발한다.

**왜 activation은 per-channel을 안 하나:** weight는 **정적**이라 채널별 s를 학습/PTQ 시 한 번 계산해 저장하면 끝(추가 런타임 비용 0). 반면 activation은 **런타임에 값이 바뀌는 동적 텐서**라, per-channel로 하려면 매 추론마다 채널별 min/max를 실시간 계산해야 하고, 무엇보다 대부분의 정수 MAC 배열이 **채널마다 다른 s를 가진 activation을 곱하는 하드웨어 경로가 없다**(가속기는 텐서 하나에 s 하나를 가정). 그래서 activation은 per-tensor로 캘리브레이션해 고정한다.

> 🔴 **함정**: "그럼 activation도 채널별 min/max를 미리 캘리브레이션해서 고정하면 per-channel 되지 않나?"— activation의 채널 축은 **공간(H,W) 위치마다 통계가 또 다르고**, 다음 레이어 Conv가 입력 채널을 가로질러 합산(`Σ_in`)하므로 채널별 s가 서로 다르면 합산 전에 재정렬(re-quantize)이 필요해 연산이 폭증한다. Transformer의 토큰별 outlier(2단계 SmoothQuant)는 이 문제의 특수 사례다.

#### 왜 weight = per-channel symmetric 인가 (면접 답변, 요약)

1. **Symmetric**: 2.2.1의 교차항 제거로 정수 MAC이 깔끔(Jacob 2018). ONNX Runtime도 `WeightSymmetric` 기본값이 **True**다. weight 분포가 대체로 0 대칭이라 대칭화 손실도 작다.
2. **Per-channel**: 2.2.2처럼 채널별 dynamic range 차이가 크고, weight는 정적이라 채널별 s 저장 비용이 0. 좁은 채널이 INT8 해상도를 온전히 쓴다.

#### 왜 activation = per-tensor asymmetric 인가 (요약)

1. **Asymmetric**: ReLU/GeLU 뒤 활성값은 한쪽으로 치우친다. symmetric `[-a,+a]`는 음수 절반을 버리지만, asymmetric은 `z`를 옮겨 `[0,max]`를 꽉 채워 유효 해상도가 사실상 1비트 늘어난다.
2. **Per-tensor**: activation은 동적 텐서 + 가속기 HW 제약(위) 때문에 텐서당 (s,z) 1쌍을 캘리브레이션으로 고정.

> 🔴 **함정 (타깃에 따라 asymmetric이 아예 금지된다)**: 위 "activation = asymmetric"은 **x86 CPU(VNNI) 계열을 전제한 이야기**다. **TensorRT는 대칭 양자화만 지원**하므로 `zero_point ≠ 0`인 QDQ 그래프를 **파싱조차 하지 못한다** — "TensorRT only supports symmetric uniform quantization, meaning that zeroPt=0"([TensorRT Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)). ONNX Runtime 공식 문서도 "quantization on GPU only supports **S8S8**"이라고 못 박는다([ORT Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). 실측에서 4.3의 기본 설정(비대칭 uint8)으로 만든 QDQ를 TensorRT EP에 물렸더니 **노드를 하나도 못 가져가고 전부 CPU/CUDA로 폴백해 FP32보다 3배 느려졌다**(TRT p50 0.96 → **3.06 ms**, [재실행 로그](../logs/stage1_real_imagenet_log.html)). 즉 위 1번은 **표현 해상도**의 논리이고, 실제 dtype 선택은 **타깃 하드웨어가 먼저 결정한다**. 4.3.2의 **타깃별 dtype 분기표**를 반드시 함께 볼 것.

> 🔴 **함정**: "activation도 per-channel 하면 더 정확하지 않나요?"는 흔한 오해다. Transformer의 activation outlier 문제(2단계 SmoothQuant)를 제외하면, activation per-channel은 HW 미지원·연산 폭증으로 실전에서 거의 안 쓴다. 이 이유를 못 대면 감점이다.

### 2.3 QDQ 그래프 — 모든 툴체인의 공용어

**QDQ**(Quantize-Dequantize)는 원본 그래프의 텐서 앞뒤에 `QuantizeLinear`(Q)와 `DequantizeLinear`(DQ) 노드 쌍을 삽입한 ONNX 표현이다. 값 자체는 여전히 FP지만, **"여기서 이 scale/zero-point로 INT8 양자화가 일어난다"는 정보가 그래프에 박혀** 있다.

```
    (FP32 weight)                (FP32 input)
         │                            │
   QuantizeLinear (s_w, z_w=0)   QuantizeLinear (s_x, z_x)
         │                            │
   DequantizeLinear             DequantizeLinear
         │                            │
         └──────────► Conv ◄──────────┘
                       │
                 QuantizeLinear (s_y, z_y)
                       │
                 DequantizeLinear
                       │
                    (다음 레이어)
```

- **왜 공용어인가**: TensorRT, ONNX Runtime, TIDL, QNN 등 거의 모든 백엔드가 이 QDQ ONNX를 입력으로 받아 **자기 하드웨어에 맞는 진짜 INT8 커널로 fuse**한다. 즉 QDQ ONNX 하나만 잘 만들어 두면 여러 타깃으로 배포할 수 있다.
- **Q/DQ 쌍의 의미**: 연속된 `Q → DQ`는 "이 지점의 텐서를 INT8 정밀도로 반올림했다가 다시 FP로 편다"는 뜻 → **양자화 노이즈를 그래프 상에서 시뮬레이션**한다. 백엔드는 이 쌍을 만나면 실제 INT8 연산으로 대체한다.
- **fuse의 실제**: 백엔드는 `DQ → Conv → Q` 패턴을 만나면 "입력을 INT8로 받아 INT8 커널로 Conv하고 출력을 다시 INT8로 낸다"로 융합한다. Q/DQ가 붙어 있는 텐서 = "여기는 INT8로 흘러도 된다"는 **컴파일러 힌트**인 셈이다.

실제 ONNX QDQ 노드(`onnx`로 로드해 출력한 예시):

```python
# QuantizeLinear 노드 하나
node {
  op_type: "QuantizeLinear"
  input: "input_tensor"      # FP32 텐서
  input: "scale"             # s (float)
  input: "zero_point"        # z (int8/uint8)
  output: "input_quantized"  # int8 텐서
}
# 바로 뒤 DequantizeLinear
node {
  op_type: "DequantizeLinear"
  input: "input_quantized"   # int8
  input: "scale"             # 같은 s
  input: "zero_point"        # 같은 z
  output: "input_dequant"    # FP32 (노이즈 포함)
}
```

> 💡 **팁**: ONNX Runtime `quantize_static`의 기본 포맷이 바로 이 `QuantFormat.QDQ`다(ORT 1.11부터 기본). 대안인 `QOperator` 포맷은 `QLinearConv` 같은 통합 op를 쓰지만, **QDQ가 백엔드 이식성이 좋아 표준**이다.

### 2.4 캘리브레이션 4종

캘리브레이션 = "activation의 (s, z)를 정하기 위해 대표 데이터 몇 백~몇 천 장을 흘려보내 값의 분포(범위)를 관측"하는 과정. **2.1.1의 오차 분해로 보면, 캘리브레이션은 "rounding + clipping 총오차를 최소화하는 threshold `T`를 찾는 문제"** 다. 네 방법은 이 `T`를 찾는 전략이 다를 뿐이다.

| 방법 | 원리(찾는 T) | 장점 | 단점 | 언제 쓰나 |
|------|------|------|------|-----------|
| **MinMax** | 관측된 절대 min/max = T (클리핑 0) | 가장 단순·빠름, **정보를 하나도 안 버림**(클리핑 0) | outlier 하나가 T를 늘려 s 폭증 → 반올림 오차 낭비. **단 그 큰 값이 잡음이 아니라 실제 특징이면 이 "낭비"가 오히려 정답**(5.1 실측) | 분포가 깨끗하거나, **꼬리가 유의미한 신호일 때**, 빠른 baseline |
| **Percentile** | 상·하위 p%(예: 99.9%)를 버린 지점 = T | 꼬리가 **잡음일 때** s 축소 → 해상도↑ | p 값 튜닝 필요. **꼬리가 실제 특징이면 정확도가 급락**(실측: p=99.9에서 −6.2%p, 5.1) | activation outlier가 **잡음**이라고 판단될 때 |
| **Entropy (KL)** | FP 분포와 양자화 분포의 **KL divergence 최소화** 지점 = T | 정보 손실 최소(**탐색 공간이 실제로 열려 있을 때**), CNN에서 보편적 | 계산 무겁고 느림, 히스토그램 bin 설정 필요. **ORT 기본값에서는 탐색이 1개로 줄어 MinMax로 퇴화**(2.4.3·4.3) | 정확도 민감한 CNN(TensorRT 기본과 동계열) |
| **MSE** | quantize→dequantize 재구성 오차(MSE) 최소화 T | 오차(2.1.1 총오차)를 직접 최소화 | 탐색 비용, 레이어별 반복 | outlier 많고 정밀도 중요할 때 |

각 방법을 2.1.1의 언어로 다시 보면:
- **MinMax**는 클리핑을 0으로 강제하고 반올림을 방치 → outlier에 취약(2.1.1 worked example의 MinMax 열).
- **Percentile**은 "꼬리 p%를 버려도 되는 클리핑으로 취급"해 s를 줄이는 휴리스틱.
- **MSE**는 총오차 `E[e²]`를 T의 함수로 직접 그려 최솟값을 찾는 정공법.
- **Entropy(KL)**는 오차를 제곱합이 아니라 **분포 간 정보 손실(KL)**로 재는 것 — 아래에서 자세히.

> ⚠️ **주의 (이 표의 장단점은 "꼬리 = 잡음"을 전제한다)**: 2.1.1 worked example은 outlier가 **분포에서 튀어나온 잡음 1개**인 상황이었다. 그 전제가 깨지면 표의 부등호도 통째로 뒤집힌다 — ResNet18의 post-ReLU activation처럼 **큰 값이 곧 강한 특징 응답**인 경우, 꼬리를 자르는 Percentile/Entropy가 MinMax보다 **나쁘다**(5.1의 실측: MinMax가 최적, Percentile 99.9는 −6.2%p로 붕괴). "어느 캘리브레이터가 좋은가"에는 정답이 없고, **"이 텐서의 꼬리가 잡음인가 특징인가"** 를 실측으로 가려내는 절차만 있다(판단 기준은 5.1 참조). 반대로 Transformer의 activation outlier(2단계 SmoothQuant)는 소수 채널에 몰린 **구조적 잡음**에 가까워, 그쪽에서는 클리핑 계열이 이긴다.

#### 2.4.1 MinMax — 절차

```
1. 캘리브 데이터를 흘려 각 activation 텐서의 running min/max를 갱신.
2. T = max(|min|, |max|)  (symmetric) 또는 [min, max] 그대로 (asymmetric).
3. s, z를 2.1의 공식으로 계산.
```
장점은 O(N) 한 번 훑기로 끝난다는 것. 단점은 2.1.1에서 본 대로 outlier 1개에 s가 끌려간다.

#### 2.4.2 Percentile — 절차

```
1. 각 텐서 값들의 히스토그램(또는 정렬)을 만든다.
2. 하위 (100−p)/2 %, 상위 (100−p)/2 % 를 잘라내고 남은 구간의 끝을 T로.
   (예: p=99.9 → 상·하위 0.05%씩 버림)
3. 버려진 값들은 클리핑(격자 끝으로 포화).
```
`p`가 클수록 MinMax에 수렴, 작을수록 공격적으로 자름. ORT는 기본 `99.999`.

#### 2.4.3 Entropy (KL divergence) — 히스토그램 구성 → 후보 threshold별 KL 최소화 (핵심)

Entropy 캘리브레이션은 NVIDIA Szymon Migacz가 GTC 2017 "8-bit Inference with TensorRT"에서 제시한 알고리즘으로, TensorRT의 전통적 기본 캘리브레이터(`IInt8EntropyCalibrator2`) 계열과 같은 아이디어다. **"FP32 분포(P)와, threshold T에서 잘라 INT8 128레벨로 뭉갠 분포(Q)가 정보량 관점에서 가장 비슷해지는 T"** 를 찾는다.

**왜 KL인가 (직관):** MSE(2.1.1 총오차)는 "값이 얼마나 틀렸나"를 재지만, 신경망 activation에서 정말 중요한 건 **분포의 모양(어디에 확률질량이 있나)이 보존되는가**다. KL divergence `D(P‖Q) = Σ P·log(P/Q)`는 "P를 Q로 근사할 때 잃는 정보량(bits)"이라, 이걸 최소화하면 **양자화 후에도 원분포의 정보를 최대한 보존**한다.

**단계별 절차:**

```
[준비] 캘리브 데이터로 각 activation 텐서의 절댓값 히스토그램을 만든다.
       - bin 개수 = 2048 (TensorRT 관행). 범위는 [0, |max|].
       - 즉 hist[0..2047], 각 bin은 폭 (|max|/2048)의 값 구간의 카운트.

[탐색] threshold 후보 i를 128부터 2048까지 훑는다 (i = 자를 bin 인덱스):

  for i in range(128, 2048):

      # (1) 참조 분포 P: 앞쪽 i개 bin만 취한다.
      P = hist[0 : i]                       # 길이 i
      # (1-a) i 밖으로 잘리는 outlier는 버리지 않고 마지막 bin에 몰아넣는다.
      P[i-1] += sum(hist[i : 2048])         # 꼬리 질량을 경계 bin에 합산
      P = P / sum(P)                        # 확률로 정규화

      # (2) 후보 양자화 분포 Q: 앞쪽 i개 bin을 128 레벨로 뭉갠다.
      Q_128 = quantize_into_128_levels(hist[0 : i])   # i개 → 128개로 병합
      #        (i개 bin을 128 그룹으로 균등 분할, 각 그룹 카운트 합산)

      # (3) Q를 다시 i개 bin으로 "펴서(expand)" P와 길이를 맞춘다.
      #     각 128-레벨의 카운트를 그 레벨이 커버하던 원래 bin들에
      #     (0이 아닌 bin에만) 균등 분배해 되돌린다.
      Q = expand_128_back_to_i_bins(Q_128)  # 길이 i
      Q = Q / sum(Q)                        # 확률로 정규화 (0 방지 epsilon 추가)

      # (4) KL divergence 계산
      divergence[i] = KL(P, Q) = Σ_j P[j] * log( P[j] / Q[j] )

  # (5) 최적 threshold = divergence가 최소인 i
  m = argmin_i divergence[i]
  T = (m + 0.5) * (|max| / 2048)            # bin 인덱스를 실수 threshold로 환산
  s = T / 127                               # symmetric int8 scale
```

**각 단계의 의미:**
- **(1) P는 "정답 분포"**: 앞 i bin의 실제 FP 히스토그램. 꼬리(i 밖)를 마지막 bin에 몰아넣는 것은 "이 값들은 T로 클리핑될 것"임을 P에도 반영해 **공정한 비교**를 만들기 위함(P와 Q 둘 다 클리핑을 겪게).
- **(2) 128 레벨로 병합**: int8 symmetric은 양의 절반에 128 레벨(0~127)만 있으므로, i개 bin을 128개로 뭉개는 것이 "이 threshold로 양자화하면 이렇게 뭉개진다"의 시뮬레이션.
- **(3) 다시 i개로 확장**: P와 길이를 맞춰 bin 대 bin으로 KL을 계산하기 위한 트릭. 정보가 없는(카운트 0) 원 bin에는 분배하지 않아 인위적 확률을 만들지 않는다.
- **(4)(5)**: 모든 후보 T 중 정보 손실(KL)이 최소인 지점 선택.

**작은 worked example (개념 확인용, bin 8개·레벨 2개로 축소):**
FP 히스토그램(카운트) `hist = [40, 30, 15, 8, 4, 2, 1, 0]` (bin 0이 가장 작은 값, 오른쪽이 꼬리). "레벨 2개"로 뭉갠다고 하고 threshold 후보를 `i=4`와 `i=6`으로 비교.

- **i=4**: P의 앞 4 bin `[40,30,15,8]`, 꼬리 `4+2+1+0=7`을 마지막에 → `P=[40,30,15,15]`, 정규화 `[0.40,0.30,0.15,0.15]`. Q(2레벨로): 앞 2 bin→그룹1 `40+30=70`, 뒤 2 bin→그룹2 `15+15=30`; 다시 4 bin으로 균등 확장 → `[35,35,15,15]`, 정규화 `[0.35,0.35,0.15,0.15]`. KL = `0.40·ln(0.40/0.35)+0.30·ln(0.30/0.35)+0.15·ln1+0.15·ln1 = 0.40·0.1335 + 0.30·(−0.1542) = 0.0534 − 0.0463 = 0.0071`.
- **i=6**: P의 앞 6 bin `[40,30,15,8,4,2]`, 꼬리 `1+0=1` → `P=[40,30,15,8,4,3]`, 합 100 → `[0.40,0.30,0.15,0.08,0.04,0.03]`. Q: 앞 3→`85`, 뒤 3→`15`; 6 bin으로 확장 `[28.3,28.3,28.3,5,5,5]`, 정규화 `[0.283,0.283,0.283,0.05,0.05,0.05]`. KL = `0.40·ln(0.40/0.283)+0.30·ln(0.30/0.283)+0.15·ln(0.15/0.283)+0.08·ln(0.08/0.05)+0.04·ln(0.04/0.05)+0.03·ln(0.03/0.05)` = `0.40·0.346+0.30·0.058+0.15·(−0.635)+0.08·0.470+0.04·(−0.223)+0.03·(−0.511)` = `0.1384+0.0174−0.0953+0.0376−0.0089−0.0153 = 0.0739`.

→ **i=4의 KL(0.0071)이 i=6(0.0739)보다 작다** → 이 분포에서는 threshold를 `i=4` 근처로 좁게 잡는 게 정보 손실이 적다(꼬리가 얇아 클리핑해도 손해가 작고, 좁힐수록 앞쪽 해상도가 산다). 실제 알고리즘은 128~2048 전 구간을 이렇게 훑어 최소 KL 지점을 고른다.

> 💡 **팁**: TensorRT는 `divergence`가 최소인 bin `m`에서 `T=(m+0.5)·bin_width`로 threshold를 잡는다(bin 중앙 보정).

> 🔴 **함정 (위 알고리즘은 개념이 맞을 뿐, ORT의 기본 동작이 아니다)**: ORT `CalibrationMethod.Entropy`도 히스토그램+KL 계열 **구현 자체는 같지만**, **기본 파라미터가 위 절차의 탐색 공간을 0으로 만들어 버린다**. `EntropyCalibrater.__init__`의 기본값이 `num_bins=128, num_quantized_bins=128`(ORT 1.23.2 `calibrate.py` 666–667행, **main 브랜치도 동일**)이라, `get_entropy_threshold()`의 탐색 범위가
>
> ```
> zero_bin_index      = num_bins // 2            = 64
> num_half_quantized  = num_quantized_bins // 2  = 64
> kl_divergence       = np.zeros(64 - 64 + 1)    # ← 후보가 정확히 1개
> ```
>
> 즉 **후보 threshold가 "히스토그램 전체 범위" 하나뿐**이라 argmin이 자동으로 전체 범위 = **MinMax와 같은 결과**를 낸다. 위 의사코드의 `for i in range(128, 2048)`이 ORT 기본값에서는 `for i in range(128, 129)`인 셈이다. 실측으로도 activation scale·zero_point가 MinMax와 **32/32 텐서 전부 비트 단위로 동일**했고(산출 ONNX의 md5까지 같다), **50,000장 전량 예측 불일치가 0장**이었다(그런데 캘리브 시간은 9.0s → 25.1s로 **2.8배** 더 씀). 대응은 4.3 참조.

> 💡 **자가진단**: "내가 돌린 Entropy가 진짜 KL 탐색을 했나?"는 **MinMax 산출물과 activation scale을 비교**하면 1분에 끝난다. 두 QDQ ONNX에서 `*_scale` initializer를 뽑아 전부 같으면 퇴화한 것이다.
>
> ```python
> # check_entropy_degenerate.py — Entropy가 MinMax로 퇴화했는지 1분 확인
> import onnx
> from onnx import numpy_helper
>
> def act_scales(path):
>     """QuantizeLinear의 스칼라 scale(=activation per-tensor)만 뽑는다."""
>     m = onnx.load(path)
>     init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
>     out = {}
>     for n in m.graph.node:
>         if n.op_type != "QuantizeLinear":
>             continue
>         s = init.get(n.input[1])
>         if s is None or s.ndim != 0:      # 벡터 scale = weight per-channel → 제외
>             continue
>         out[n.input[0]] = float(s)
>     return out
>
> a, b = act_scales("resnet18_int8_minmax.onnx"), act_scales("resnet18_int8_entropy.onnx")
> same = sum(1 for t in a if t in b and abs(a[t] - b[t]) < 1e-12)
> print(f"activation scale 동일: {same}/{len(a)}  → 전부 같으면 Entropy가 MinMax로 퇴화한 것")
> ```
>
> 이 실습 환경(ORT 1.23.2, ResNet18)에서의 실제 출력은 `activation scale 동일: 32/32`였다([실행 로그](../logs/stage1_quantization_log.html)).

> 🔴 **함정**: KL 계산에서 `Q[j]=0`인데 `P[j]>0`이면 `log(P/Q)=∞`로 발산한다. 그래서 구현은 확장 시 0 bin에 **작은 epsilon**을 더하거나, P가 0인 bin은 KL 합에서 제외한다. 직접 구현할 일은 없지만(ORT/TensorRT가 처리) 원리를 알면 "왜 히스토그램 bin이 비면 안 되나"를 설명할 수 있다.

#### 2.4.4 MSE — 절차

```
1. 텐서 히스토그램(또는 샘플)을 준비.
2. threshold 후보 T들을 훑으며, 각 T로 실제 quantize→dequantize 수행.
3. 재구성 오차 MSE = mean((x − x̂)²) 를 계산 (= 2.1.1의 총오차 E[e²]).
4. MSE가 최소인 T 선택.
```
KL이 "분포 유사도"를 재는 대신, MSE는 2.1.1의 총오차(rounding+clipping)를 **직접** 재 최소화한다. outlier가 많거나 분포가 비대칭일 때 KL보다 robust한 경우가 있다.

> 💡 **팁**: ONNX Runtime의 `CalibrationMethod`는 `MinMax(0) / Entropy(1) / Percentile(2) / Distribution(3)` 4종을 제공한다(2026-07, 정본 ORT 1.23.2의 `calibrate.py`에서 실측 확인). MSE는 ORT에 별도 enum이 없고, **NVIDIA Model Optimizer**(`modelopt`, https://github.com/NVIDIA/Model-Optimizer) 등 다른 툴에서 지원한다. Entropy가 NVIDIA TensorRT의 전통적 기본 캘리브레이터(EntropyCalibrator) 계열과 같은 아이디어다.

### 2.5 QAT + STE (Straight-Through Estimator)

PTQ로 정확도가 부족하면 **QAT(Quantization-Aware Training)**: 학습 그래프에 fake-quant(가짜 양자화) 노드를 넣어 **양자화 노이즈를 학습 중에 겪게** 한다. 문제는 `round()`의 미분이 거의 어디서나 0이라(계단 함수) gradient가 흐르지 못한다는 것. 이를 우회하는 것이 **STE**: forward는 진짜로 round/clamp 하되, **backward에서는 round를 항등함수처럼 취급**해 gradient를 그대로 통과시킨다(clip 범위 밖만 0).

#### 2.5.1 왜 STE가 필요한가 (수식)

fake-quant 함수 `x̂ = s·clamp(round(x/s), q_min, q_max)`의 진짜 미분은:

```
d(x̂)/dx = s · (1/s) · d[round(u)]/du = d[round(u)]/du,   u = x/s
```

그런데 `round(u)`는 정수 지점마다 +1씩 점프하는 계단 함수라, 미분은 **정수 아닌 곳에서 0, 정수 지점에서 ∞(정의 안 됨)**. 이대로면 `∂Loss/∂x = ∂Loss/∂x̂ · 0 = 0` → weight가 절대 안 움직인다(학습 불가).

**STE의 정의:** backward에서 `d[round(u)]/du ≈ 1` (범위 안), `0` (범위 밖)으로 **가짜 미분을 심는다**:

```
∂x̂/∂x ≈ 1{ q_min ≤ round(x/s) ≤ q_max }    # 범위 안이면 1, 밖이면 0
```

즉 forward는 계단(양자화 노이즈를 진짜로 겪음), backward는 기울기 1의 직선(gradient 통과). "forward는 정수의 거친 세계, backward는 매끄러운 FP 기울기"라는 **비대칭**이 STE의 전부다.

#### 2.5.2 `torch.autograd.Function` 전체 구현

```python
import torch

class FakeQuantSTE(torch.autograd.Function):
    """대칭 per-tensor fake-quant + STE.
    forward : 실제 quantize→dequantize (노이즈 발생)
    backward: round의 gradient를 1로 통과(STE), clip 범위 밖은 0."""

    @staticmethod
    def forward(ctx, x, scale, q_min, q_max):
        u = x / scale
        q = torch.round(u)                          # 반올림 (미분 불가 지점)
        q_clamped = torch.clamp(q, q_min, q_max)    # 정수 범위로 clip
        x_hat = q_clamped * scale                   # dequantize
        # clip 범위 안(=gradient 통과 영역) 마스크 저장
        mask = (u >= q_min) & (u <= q_max)
        ctx.save_for_backward(mask)
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        # STE: round는 identity로 간주 → grad 그대로,
        # 단 clip 밖은 0 (그쪽으로는 학습 신호 없음)
        grad_x = grad_output * mask.to(grad_output.dtype)
        return grad_x, None, None, None             # scale/q_min/q_max엔 grad 없음


def fake_quantize(x, num_bits=8, symmetric=True):
    """weight/activation에 삽입하는 편의 함수."""
    q_max = 2 ** (num_bits - 1) - 1                 # 127
    q_min = -q_max if symmetric else 0              # -127 (symmetric)
    scale = x.detach().abs().max() / q_max          # per-tensor symmetric scale
    scale = torch.clamp(scale, min=1e-8)            # 0 나눗셈 방지
    return FakeQuantSTE.apply(x, scale, q_min, q_max)
```

#### 2.5.3 미니 QAT 학습 루프 (옵티마이저·손실 포함, 실측 출력)

STE가 실제로 gradient를 통과시켜 학습이 되는지 **10줄짜리 회귀 문제**로 확인한다. 목표: 랜덤 선형 함수 `y = Wx + b`를 **weight에 fake-quant를 씌운 채** 근사. STE가 없으면 loss가 안 줄고, 있으면 줄어드는 것을 눈으로 본다.

```python
# mini_qat_demo.py — STE가 gradient를 통과시키는지 확인하는 최소 QAT 루프
import torch
# (위 FakeQuantSTE, fake_quantize 정의를 같은 파일에 둔다고 가정)

torch.manual_seed(0)
N, D = 512, 16
X = torch.randn(N, D)
W_true = torch.randn(D, 1)
y = X @ W_true + 0.1 * torch.randn(N, 1)           # 정답 (약간의 노이즈)

W = torch.zeros(D, 1, requires_grad=True)          # 학습 대상 (0에서 시작)
opt = torch.optim.SGD([W], lr=0.1)
loss_fn = torch.nn.MSELoss()

for step in range(200):
    opt.zero_grad()
    W_q = fake_quantize(W, num_bits=8, symmetric=True)   # ← weight를 fake-quant
    pred = X @ W_q                                       # 양자화된 weight로 forward
    loss = loss_fn(pred, y)
    loss.backward()                                      # STE로 grad가 W까지 흐름
    opt.step()
    if step % 40 == 0:
        # W의 grad가 0이 아니어야 학습이 되는 것 (STE 작동 증거)
        gnorm = W.grad.norm().item()
        print(f"step {step:3d}  loss={loss.item():.4f}  |grad|={gnorm:.4f}")

print("final loss:", round(loss.item(), 4))
print("cos(W_q, W_true):",
      round(torch.nn.functional.cosine_similarity(
          fake_quantize(W).flatten(), W_true.flatten(), dim=0).item(), 4))
```

**실측 출력(위 시드 `torch.manual_seed(0)` 고정, torch 2.11.0+cu128 — [실행 로그](../logs/stage1_quantization_log.html)):**

```
step   0  loss=14.7803  |grad|=8.2041
step  40  loss=0.0105   |grad|=0.0397
step  80  loss=0.0108   |grad|=0.0549
step 120  loss=0.0104   |grad|=0.0371
step 160  loss=0.0106   |grad|=0.0474
final loss: 0.0105
cos(W_q, W_true): 1.0
```

> 💡 lr=0.1이라 **40 step 안에 이미 수렴**한다. 이후 loss가 0.0104~0.0108을 오가고 `|grad|`가 0으로 수렴하지 않고 0.03~0.05에서 진동하는 것은, **fake-quant의 반올림 노이즈가 매 step 미세하게 달라지기 때문**이다(양자화 격자 위에서 W가 덜그럭거리는 상태). 이 잔여 진동이 QAT의 정상적인 정상상태다.

> 💡 **초기 loss 14.78은 우연이 아니다**: `W=0`에서 시작하므로 `pred=0`이고, 초기 loss = `mean(y²)` ≈ `E[‖W_true‖²]` = `D = 16`이다(노이즈 0.1²는 무시 가능). 실측 14.78은 `D=16`의 표본 변동 범위 안이다. **"예상 출력이 이론값과 맞는가"를 이렇게 역산해 보는 습관이 스크립트 오류를 조기에 잡는다.**

**해석**: `|grad|`가 0이 아니고 loss가 계속 줄어드는 것이 **STE가 round의 0 gradient를 우회해 학습 신호를 통과시켰다는 직접 증거**다.

**대조군 실측 — STE를 빼면 어떻게 되나.** 아래 `naive_quantize`를 `fake_quantize` 자리에 끼워 같은 루프를 돌린다(`autograd.Function` 없이 `torch.round`를 그래프에 그대로 노출).

```python
def naive_quantize(x, num_bits=8):
    """대조군: STE 없이 torch.round를 그냥 통과 (backward에서 grad가 0이 된다)."""
    q_max = 2 ** (num_bits - 1) - 1
    scale = torch.clamp(x.detach().abs().max() / q_max, min=1e-8)
    return torch.clamp(torch.round(x / scale), -q_max, q_max) * scale
```

```
[STE 없음: torch.round 직접 — 대조군]
step   0  loss=14.7803  |grad|=0.0000
step  40  loss=14.7803  |grad|=0.0000
step  80  loss=14.7803  |grad|=0.0000
step 120  loss=14.7803  |grad|=0.0000
step 160  loss=14.7803  |grad|=0.0000
final loss: 14.7803
cos(W_q, W_true): 0.0
```

**200 step 내내 loss가 초기값 14.7803에 못 박혀 있고 `|grad|`가 정확히 0**이다. `W`가 한 번도 움직이지 않아 `cos(W_q, W_true)=0.0`(W가 0 벡터라 방향이 없음). 두 출력을 나란히 놓으면 "STE는 편의적 근사가 아니라 **QAT가 성립하기 위한 필요조건**"임이 한눈에 보인다.

> 💡 **팁**: STE의 직관 — "forward는 정수의 거친 세계를 겪지만, backward는 매끄러운 FP 기울기를 그대로 받아 weight를 조금씩 옮긴다." 이 한 줄로 설명할 수 있으면 충분하다.

#### 2.5.4 실모델에서 QAT는 PTQ 손실을 얼마나 되찾나 (ResNet18 실측)

2.5.3은 STE가 **gradient를 통과시킨다**는 것을 합성 회귀로 증명했다. 하지만 "gradient가 흐른다"와 "실모델의 정확도를 되찾는다"는 다른 명제다. ResNet18로 **FP32 → PTQ → QAT** 회복 계단을 직접 쟀다([전체 재현: `experiments/qat_recovery/`](../experiments/qat_recovery/README.md) · [렌더 요약: 보고서 §11](../logs/stage1_50k_rerun_reproduction_report.html#s11)).

**먼저 손실을 크게 만들어야 회복률이 읽힌다.** 이 문서의 정본 구성(per-channel 대칭 INT8 weight + per-tensor 비대칭 UINT8 activation, 즉 **W8A8**)은 PTQ 손실이 0.06~0.14%p로 **측정 노이즈 크기**다. 여기서 회복률을 계산하면 `1116.7%` 같은 무의미한 수가 나온다(0에 가까운 값으로 나눔). 그래서 노브 하나만 바꿔(`QAT_WBITS=4` — weight를 4-bit로) PTQ 손실을 **−24.16%p**까지 키운 **손실 변형 W4A8**에서 회복률을 처음으로 노이즈 위에서 측정했다. 설정: `BS_TRAIN=48 · BS_EVAL=128 · EPOCHS=2 · CALIB_N=5000`, ImageNet val을 **40,000 학습 / 10,000 평가(서로소)**로 쪼갬.

**팔 1 — 회복 계단**

| 단계 | top-1 | vs FP32 | vs PTQ |
|---|---|---|---|
| FP32 (학습 없음) | 68.51% | — | — |
| PTQ · weight 4-bit | 44.35% | **−24.16%p** | — |
| QAT ep0 (STE) | 67.59% | −0.92%p | +23.24%p |
| QAT ep1 (STE) | **67.81%** | −0.70%p | **+23.46%p** |

→ PTQ가 무너뜨린 24.16%p 중 QAT가 **97.1%(23.46%p)를 되찾았다**. §2.5 첫 문장("PTQ로 부족하면 QAT")이 손실 큰 구간에서 실증된 것이다.

**팔 2 — 대조군이 없으면 착시가 생긴다.** 팔 1만 보면 "QAT가 정확도를 되찾았다"로 끝내기 쉽다. 그러나 **QAT 팔만 val 40,000장으로 2에폭을 추가 학습**했다. 그 이득이 '양자화 인식(fake-quant를 겪으며 적응)' 때문인지 '그냥 더 학습한 것' 때문인지 구분되지 않는다. **fake-quant만 제거하고 나머지(데이터·에폭·옵티마이저)를 전부 똑같이 맞춘 FP32 파인튜닝**을 대조군으로 돌려 그 몫을 떼어낸다.

| | top-1 | Δ |
|---|---|---|
| FP32 학습 전 | 68.51% | — (팔 1과 일치 ✅) |
| FP32 + 파인튜닝 (대조군) | 69.31% | **+0.80%p** · 순수 추가학습 몫 |
| QAT 4-bit | 67.81% | −0.70%p |
| **QAT − 대조군** | — | **−1.50%p** · 4-bit 환원 불가 대가 |

> 💡 **읽는 법 — 두 가지가 동시에 참이다.** ① **회복은 진짜다**: 추가 학습만으로는 +0.80%p인데, QAT는 손상 지점(44.35%)에서 +23.46%p를 끌어올렸다 → 회복의 본체는 일반 파인튜닝이 아니라 **양자화 인식 적응**이다. ② **공짜는 아니다**: 같은 데이터·같은 에폭을 받은 FP32 대조군보다 QAT는 여전히 **−1.50%p** 낮다 → 이건 4-bit weight(+8-bit activation)의 **환원 불가 용량 비용**이다. "QAT = 손실 0 복구"가 아니라 **"QAT = 손실의 대부분을 되찾되 하한이 있다"**가 정직한 요약이다.

> ⚠️ **이 수치는 상대 관계로만 읽어라.** ImageNet **train split이 없어 val을 쪼개 학습**했다(클래스당 40장 학습/10장 평가, 서로소 assert로 평가 누수는 없음). 그래서 **절대 top-1을 문헌값과 비교하면 안 된다** — FP32 68.51%도 공식 69.758%가 아니다. 유효한 것은 **회복률(97.1%)·대조군 격차(−1.50%p)** 처럼 **같은 실험 안에서의 상대 비교**뿐이다. 회복률을 다른 손실 변형(Entropy 정규화 −9.45%p, Percentile 99.9 −6.83%p)으로도 재보려면 같은 2팔 틀을 재사용하면 된다.

#### 2.5.5 LSQ (Learned Step Size Quantization) — 한 줄 소개

위 코드는 `scale`을 매 step **weight의 max에서 다시 계산**(고정 규칙)한다. **LSQ**(Esser et al., ICLR 2020, arXiv:1902.08153)는 여기서 한 걸음 더 나아가 **scale(step size) `s` 자체를 학습 가능한 파라미터로 두고 gradient descent로 함께 학습**한다. 핵심은 `∂x̂/∂s`(양자화 출력의 scale에 대한 미분)를 STE 방식으로 근사해 흘리고, step size gradient에 `1/√(N·q_max)` 스케일 보정을 걸어 weight 업데이트와 크기 균형을 맞추는 것. **"scale을 손으로 정하지 말고 loss가 정하게 하라"**가 LSQ의 한 문장이며, 저비트(2~4bit) QAT에서 특히 효과가 크다(3bit로 FP 정확도 근접).

> ⚠️ **주의**: 위 코드는 개념 학습용이다. 실전 QAT는 직접 짜지 말고 검증된 라이브러리를 쓴다 — PyTorch 네이티브(`torch.ao.quantization`) 또는 **NVIDIA Model Optimizer**(`modelopt.torch.quantization`, `mtq.quantize()`; https://github.com/NVIDIA/Model-Optimizer). (구 `pytorch-quantization` 패키지는 여전히 `NVIDIA/TensorRT` 저장소에 있으나 사실상 Model Optimizer로 대체되었다 — 2026-07 기준. 자세한 QAT 실습은 이 단계 범위 밖.)

---

## 3) 환경·도구 준비

이론 실습은 x86 PC(RTX GPU)에서 완결. GPU가 없어도 CPU로 실행 가능하나(느림), RTX가 있으면 ONNX Runtime CUDA EP로 평가가 빠르다. 이 스터디의 정본 스택은 **CUDA 12.8 / onnx 1.18.0 / onnxruntime-gpu 1.23.2 / TensorRT 10.16.x LTS** 다(TensorRT는 3단계에서 사용).

```bash
# 1) 가상환경 (프로젝트 루트에서)
python3 -m venv ~/venv/quant && source ~/venv/quant/bin/activate
python -m pip install --upgrade pip

# 2) 핵심 패키지 (2026-07 기준 최신 계열)
#    torch/torchvision: ResNet18 로드 + ONNX export
#    onnx: 그래프 조작/검증,  onnxruntime-gpu: 정적 양자화 + 평가
#    🔴 torch는 반드시 cu128 인덱스에서 받는다. PyPI 기본 휠은 CUDA 13 라인이라
#       CUDA 12.8 정본 스택과 섞이면 ORT CUDA EP가 조용히 CPU로 내려앉는다(0단계 2절).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "onnx==1.18.0" "onnxruntime-gpu<1.27"        # 2026-07 기준 정본 버전 (→ onnx 1.18.0 / ORT 1.23.2)
#    🔴 onnxscript 필수: torch 2.11의 torch.onnx.export는 기본이 dynamo=True이고
#       그 경로가 onnxscript를 요구한다. 없으면 4.1의 export가 "No module named 'onnxscript'"로 죽는다.
#       (onnxscript 0.7.1은 onnx>=1.17만 요구하므로 위 1.18.0 핀을 건드리지 않는다 — 실측 확인)
pip install onnxscript
pip install "numpy<2" pillow tqdm scipy                  # 데이터/평가/통계 유틸 (numpy는 정본과 동일하게 1.x)
```

```bash
# 3) 설치 검증 (버전·EP 확인)
python - <<'PY'
import torch, torchvision, onnx, onnxruntime as ort
print("torch      :", torch.__version__, "cuda?", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("onnx       :", onnx.__version__)
print("onnxruntime:", ort.__version__)
print("providers  :", ort.get_available_providers())  # CUDAExecutionProvider 있어야 GPU 사용
PY
```

> 💡 **팁**: `onnxruntime-gpu`의 정본 버전은 이 스터디 기준 **1.23.2**다. ⚠️ PyPI 기본 wheel의 CUDA 라인이 1.27부터 CUDA 13으로 바뀌었으므로, 이 스터디의 CUDA 12.8 스택에서는 상한 `<1.27`을 걸어 **CUDA 12 대응 wheel**을 받아야 한다(Ubuntu 22.04 기본 Python 3.10에서는 이게 1.23.2로 해석된다). 자세한 CUDA 12/13 전환 캐비앗은 [0단계 2·3절](01_environment_setup.md) 참조. CPU만 쓸 거면 `onnxruntime`(GPU 없는 패키지)을 설치한다. 두 패키지를 동시에 설치하지 말 것(충돌).
>
> 🔴 **`onnx`도 반드시 `1.18.0`으로 고정한다.** ORT 1.23.2는 **ONNX IR 11까지만** 읽는데, `pip install onnx`(무제한)는 최신 1.22.0(**IR 13**)을 깔아 `Unsupported model IR version: 13, max supported IR version: 11`로 **모델 로드가 실패**한다. 그래서 아래 4.x의 export는 **opset ≤ 23**을 전제로 한다([0단계 2절](01_environment_setup.md) 참조).

**ImageNet 검증셋 준비**: 폴더 구조는 클래스별 서브폴더(`val/n01440764/*.JPEG` …) 형태를 권장. 라이선스상 데이터는 직접 받아야 한다. 장수는 **목적에 따라 갈린다**:

| 목적 | 필요 장수 | 근거 |
|------|----------|------|
| 파이프라인이 도는지 확인(export→캘리브→양자화→평가) | **1,000장**(캘리브 100~500 + 평가 1,000) | 코드 경로 검증에는 충분하다 |
| 캘리브레이션 방법·양자화 축의 **우열 판정** | 🔴 **50,000장 전량**(최소 수천 장 층화 표본) | 1,000장의 95% CI 폭이 ±2.9%p라 0.3%p 차이를 못 가른다. 실측으로 **부호 3건·판정 5건이 뒤집혔다**([10단계 함정 0](10_pitfalls.md)) |

**1,000장으로 먼저 돌려 파이프라인을 굳히고, 결론은 전량에서 내라.** 50k 전량의 실측 비용은 **디스크 ≈28 GiB**(tar 6.3G + 펼친 `val_full` 6.4G + 무손실 uint8 전처리 캐시 15G = 224² 전처리 2종 × 7.1G)와 **평가 22분/14회**(≈94초/모델, RTX 3060 + CUDA EP)다. 전처리를 미리 캐시해 두는 것이 핵심이다 — 모델 14개가 **같은 바이트의 입력**을 보므로 paired 검정이 성립한다(4.4).

ImageNet은 계정 등록 후 다운로드해야 한다([image-net.org](https://www.image-net.org/)). 접근이 어려울 때의 **대안과 그 함정**은 아래에 정리한다. 어느 쪽을 쓰든 top-1 절대값이 아니라 **FP32 대비 상대 하락폭**을 보는 것이 이 실습의 목적이다.

#### 3.1 ImageNet val을 못 구할 때 — 대안 데이터셋과 라벨 매핑 함정

| 대안 | 구성 | 쓸 때 반드시 알아야 할 것 |
|------|------|--------------------------|
| **`EliSchwartz/imagenet-sample-images`** ([GitHub](https://github.com/EliSchwartz/imagenet-sample-images)) | 1000 클래스 × **각 1장** = 1000장, 파일명이 `n01440764_tench.JPEG` 형식 | **1000-way 라벨이 그대로 맞아** 매핑 사고가 없다. 단 **큐레이션된 "대표 이미지"라 top-1이 부풀려진다** — 이 실습 실측 FP32 **78.50%** vs ImageNet val 공식 **69.76%**. 절대값 인용 금지, 상대 비교 전용 |
| **Imagenette** (10 클래스 서브셋) | 10 클래스 × 수백 장 | 🔴 **아래 함정**. 폴더가 10개뿐이라 `load_labels()`가 인덱스를 0~9로 매기는데 모델은 1000-way → **top-1 ≈ 0%** |
| ImageNet val 부분 다운로드 | 원본 그대로 | 매핑 안전. 가능하면 이게 최선 |

> 🔴 **함정 (클래스 서브셋 + 1000-way 모델)**: 4.4의 `load_labels()`는 **"val 디렉터리 안의 폴더를 정렬한 순서 = 클래스 인덱스"** 로 가정한다. 이 가정은 **1000개 폴더가 전부 있을 때만** 성립한다. Imagenette처럼 10개 폴더만 있으면 정답 인덱스가 `0~9`로 찍히는데, 모델이 내놓는 인덱스는 1000-way 공간의 `0 · 217 · 482 · 491 · 497 · 566 · 569 · 571 · 574 · 701`이라 **거의 모든 이미지가 오답 처리되어 top-1이 0%에 수렴한다**(6장 "라벨 인덱스 매핑 오류" 증상 그대로). 서브셋을 쓰려면 synset을 **전체 1000개 목록에서의 위치**로 되돌려야 한다.
>
> ```python
> # 서브셋을 쓸 때의 라벨 매핑 — synset을 "1000개 중 몇 번째"로 환산
> import torchvision
>
> # torchvision의 클래스 순서 = ImageNet synset ID의 사전순 정렬 (실측 검증됨, 아래 팁)
> ALL_SYNSETS = sorted(...)   # ImageNet-1k synset 1000개 전체 목록(n01440764 …)을 정렬
> syn2idx = {s: i for i, s in enumerate(ALL_SYNSETS)}
>
> def load_labels_subset(val_dir, syn2idx):
>     """폴더명이 synset ID일 때, 1000-way 인덱스로 매핑."""
>     import os, glob
>     items = []
>     for syn in sorted(os.listdir(val_dir)):
>         idx = syn2idx[syn]                       # ← 0~9가 아니라 진짜 1000-way 인덱스
>         for p in glob.glob(os.path.join(val_dir, syn, "*.JPEG")):
>             items.append((p, idx))
>     return sorted(items)
> ```
>
> `ALL_SYNSETS`를 따로 못 구하면, 위 `EliSchwartz` 저장소의 파일명 1000개(`n........_name.JPEG`)에서 synset만 뽑아 정렬해 쓰면 된다.

> 💡 **팁 (검증된 사실)**: "**ImageNet synset ID의 사전순 정렬 = torchvision 1000-class 인덱스**"는 이 실습에서 1000/1000 일치로 실측 확인했다. 이름이 달라 보이는 2건(`idx 134` crane/crane bird, `idx 639` maillot/maillot tank suit)도 동음이의어를 구분하는 표기 차이일 뿐 매핑은 정확하다([실행 로그](../logs/stage1_quantization_log.html)). 즉 **폴더명이 synset ID이고 1000개가 다 있으면** 4.4의 `load_labels()`를 그대로 써도 된다.

---

## 4) 단계별 실습

### 4.1 ResNet18 로드 → ONNX export

```python
# export_resnet18.py — torchvision ResNet18(사전학습) → ONNX
import torch, torchvision

model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1  # 사전학습 가중치
).eval()

dummy = torch.randn(1, 3, 224, 224)                            # 고정 입력 shape
torch.onnx.export(
    model, dummy, "resnet18_fp32.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # 배치 동적
    opset_version=18,                                          # torch 2.11 dynamo 익스포터의 하한
)
print("saved resnet18_fp32.onnx")
```

```bash
python export_resnet18.py
# ONNX 그래프 유효성 검사
python -c "import onnx; onnx.checker.check_model(onnx.load('resnet18_fp32.onnx')); print('ONNX OK')"
```

> 🔴 **함정 (`opset_version=17`을 요구하면 에러 트레이스백이 쏟아지지만 exit 0이다)**: torch 2.11의 기본 익스포터는 dynamo 경로이고, **이 경로가 구현한 최소 opset이 18**이다. `opset_version=17`을 주면 익스포터는 일단 **18로 만든 뒤 17로 다운컨버트를 시도**하고, 그 변환이 실패하면 **에러를 화면에 찍으면서도 18짜리 모델을 그대로 저장하고 exit 0으로 끝난다**([실행 로그](../logs/stage1_quantization_log.html)):
>
> ```
> W0802 _compat.py:133] Setting ONNX exporter to use operator set version 18 because the
> requested opset_version 17 is a lower version than we have implementations for.
> Failed to convert the model to the target version 17 using the ONNX C API. The model was not modified
> RuntimeError: axes_input_to_attribute.h:65: adapt: Assertion `node->hasAttribute(kaxes)` failed:
> No initializer or constant input to node found
> [torch.onnx] Optimize the ONNX graph... ✅
> saved resnet18_fp32.onnx
> exit code: 0
> ```
>
> **"RuntimeError가 보이는데 exit code가 0"** 이라는 게 이 함정의 핵심이다 — CI에서 반환값만 보면 통과로 집계되고, 실제 산출물의 opset은 요청과 다르다. 정본 스택(ORT 1.23.2)은 opset 18을 문제없이 읽으므로(IR 10, checker 통과, dynamic batch 유지) **실사용에는 지장이 없다**. 애초에 `opset_version=18`을 주면 위 소음이 전부 사라진다. 굳이 17이 필요하면 `dynamo=False`로 레거시 익스포터를 쓴다.

> ⚠️ **주의 (산출물이 `.onnx` + `.onnx.data` 2파일이다)**: dynamo 익스포터는 가중치를 **external data**로 분리한다. 실측 산출물은 `resnet18_fp32.onnx`(91,843 B) + `resnet18_fp32.onnx.data`(46,792,704 B)로, **그래프 파일만 옮기면 로드 시 가중치를 못 찾아 실패**한다. 다른 머신·컨테이너로 복사할 때 `.data`를 빠뜨리지 말 것. (양자화 산출물은 단일 파일 11.32 MB로 합쳐진다 — 4.3.)

### 4.2 캘리브레이션 데이터 리더 구현

`quantize_static`은 `CalibrationDataReader` 인터페이스(추상 메서드 `get_next()`)를 요구한다. `get_next()`는 매 호출마다 `{input_name: np.ndarray}`를 반환하고, 데이터가 끝나면 `None`을 반환한다.

```python
# calib_reader.py — ImageNet 전처리 + CalibrationDataReader
import os, glob, numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path):
    """torchvision 표준: 짧은 변만 256으로(종횡비 유지) → center crop 224.
    torchvision의 Resize(256) + CenterCrop(224)와 동일한 연산이다."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h: nw, nh = 256, round(256 * h / w)      # 짧은 변 = w
    else:     nh, nw = 256, round(256 * w / h)      # 짧은 변 = h
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = np.asarray(img.crop((left, top, left + 224, top + 224)))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))[None, :].astype(np.float32)  # NCHW

class ImageNetCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir, input_name="input", limit=200):
        self.paths = sorted(glob.glob(os.path.join(calib_dir, "**", "*.JPEG"),
                                       recursive=True))[:limit]     # 캘리브 200장
        self.input_name = input_name
        self._it = iter(self.paths)

    def get_next(self):
        path = next(self._it, None)
        if path is None:
            return None                                   # 끝나면 None
        return {self.input_name: preprocess(path)}

    def rewind(self):
        self._it = iter(self.paths)
```

> 🔴 **함정**: 전처리(resize/crop/정규화)가 **원 모델 학습 때와 다르면** 캘리브레이션 범위가 엉뚱해져 INT8 정확도가 폭락한다. torchvision ResNet18은 위 mean/std가 표준이다. 평가 스크립트와 **완전히 동일한 전처리**를 써야 한다.

> 🔴 **함정 (실측 — 전처리 한 줄이 양자화 방법 선택보다 정확도에 크게 영향한다)**: 이 문서는 원래 `.resize((256, 256))`으로 **종횡비를 무시하고 늘린 뒤**(squash) 중앙 224를 잘랐다. ImageNet val **50,000장 전량**으로 두 방식을 재보니([재실행 보고서](../logs/stage1_real_imagenet_report.html)):
>
> | 전처리 | FP32 top-1 | INT8 MinMax | 양자화 손실 | 공개값(69.758%) 대비 |
> |--------|-----------|-------------|------------|---------------------|
> | squash (구 코드) | 68.74% | 68.62% | −0.12%p (p=0.061) | **−1.02%p** |
> | **tv (위 코드)** | **69.81%** | 69.67% | −0.13%p (p=0.034) | **+0.05%p** |
>
> FP32 기준 **+1.07%p**, McNemar **p=1.6e-14**로 압도적으로 유의하다. **양자화 손실(−0.12%p)의 9배**다. 즉 전처리를 틀리면 **어떤 캘리브레이션을 고르든 그 차이보다 큰 손실을 이미 깔고 시작한다.** 위 tv 코드는 공개 재현값과 0.05%p 안에서 만나므로, **FP32가 69.8%±0.1 안에 들어오는지가 전처리·라벨 매핑이 옳다는 가장 빠른 sanity check**다.

### 4.3 INT8 정적 양자화 (MinMax vs Entropy vs Percentile, per-channel 옵션)

```python
# quantize_ptq.py — MinMax / Entropy / Percentile × per-channel 옵션 비교
from onnxruntime.quantization import (
    quantize_static, CalibrationMethod, QuantType, QuantFormat,
)
from calib_reader import ImageNetCalibReader

CALIB_DIR = "imagenet/val"   # 캘리브레이션용 이미지 폴더

def run(method, out_path, per_channel=True, percentile=None):
    reader = ImageNetCalibReader(CALIB_DIR, input_name="input", limit=200)
    extra = {
        "WeightSymmetric": True,             # weight 대칭(기본 True)
        "ActivationSymmetric": False,        # activation 비대칭(기본 False)
    }
    if percentile is not None:
        extra["CalibPercentile"] = percentile   # Percentile일 때만 (버전별 키명 주의)
    quantize_static(
        model_input="resnet18_fp32.onnx",
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,            # QDQ 그래프 생성(기본)
        calibrate_method=method,                 # ← MinMax / Entropy / Percentile
        activation_type=QuantType.QUInt8,        # activation: uint8 (asymmetric)
        weight_type=QuantType.QInt8,             # weight: int8 (symmetric)
        per_channel=per_channel,                 # weight per-channel on/off
        reduce_range=False,
        extra_options=extra,
    )
    print("saved", out_path)

if __name__ == "__main__":
    # 캘리브레이션 3종 비교
    run(CalibrationMethod.MinMax,     "resnet18_int8_minmax.onnx")
    run(CalibrationMethod.Entropy,    "resnet18_int8_entropy.onnx")
    run(CalibrationMethod.Percentile, "resnet18_int8_pct999.onnx", percentile=99.9)
    # per-channel 효과 대조: MinMax를 per-tensor로도 뽑아 비교
    run(CalibrationMethod.MinMax,     "resnet18_int8_minmax_pertensor.onnx",
        per_channel=False)
    # percentile 스윕 — 꼬리를 얼마나 자르느냐가 top-1을 직접 좌우한다(5.1)
    run(CalibrationMethod.Percentile, "resnet18_int8_pct9999.onnx",  percentile=99.99)
    run(CalibrationMethod.Percentile, "resnet18_int8_pct99999.onnx", percentile=99.999)
```

```bash
python quantize_ptq.py
# → resnet18_int8_{minmax,entropy,pct999,pct9999,pct99999,minmax_pertensor}.onnx 생성
```

**실측 소요 시간·산출물 크기**(캘리브 200장, RTX 3060 / i7-6700 — [실행 로그](../logs/stage1_quantization_log.html)):

| 항목 | 값 |
|------|-----|
| 양자화 소요 | MinMax **9.0s** / Entropy(ORT 기본) **25.1s** / Percentile **26.1s** |
| 모델 크기 | FP32 44.71 MB(`.onnx`+`.onnx.data` 2파일) → INT8 **11.32 MB** = **3.95×** 축소 |
| QDQ 노드 수(전 INT8 공통) | `QuantizeLinear` 32 / `DequantizeLinear` 74 / `Conv` 20 |
| per-channel 증거 | 길이>1 벡터 scale initializer **42개**(per-tensor 산출물은 **0개**, 전부 스칼라) |

> 💡 **팁**: 마지막 행이 `per_channel=True`가 실제로 먹었는지 확인하는 가장 빠른 방법이다 — scale initializer 중 `ndim > 0`인 것이 있으면 per-channel, 전부 스칼라면 per-tensor다. 옵션을 줬는데 벡터 scale이 0개면 그 백엔드/설정 조합이 per-channel을 무시한 것이다.

이 코드가 **이 단계 이론의 실체화**다. 정리하면:
- `activation_type=QUInt8` + `ActivationSymmetric=False` → **activation = asymmetric uint8** (2.2절).
- `weight_type=QInt8` + `WeightSymmetric=True` + `per_channel=True` → **weight = per-channel symmetric int8** (2.2절).
- `quant_format=QuantFormat.QDQ` → **QDQ 그래프** 생성(2.3절).
- `calibrate_method` 를 바꿔 **MinMax vs Entropy vs Percentile**(2.4절)를 비교.
- `per_channel=False` 대조군으로 **2.2.2의 per-channel 이득**을 top-1으로 직접 확인.

> 💡 **팁**: x86(VNNI) 하드웨어에서 ONNX Runtime 공식 권장이 `activation=QUInt8, weight=QInt8`이다. 위 설정은 이 권장과 이론(activation 비대칭 / weight 대칭)이 정확히 일치한다. **단 이 권장은 x86 CPU 한정이다 — 타깃이 TensorRT면 이 설정 그대로는 못 쓴다(4.3.2).**

> ✅ **확인됨 (ORT 1.23.2)**: Percentile 옵션의 키 이름은 **`CalibPercentile`이 맞다**. `quantize.py`의 `calib_extra_options_keys`에 `("percentile", "CalibPercentile")` 매핑이 있는 것을 소스에서 확인했다. 실행 시 로그에 `Percentile : (0.0999..., 99.9)`처럼 **실제 적용된 값이 찍히므로** 눈으로 검증할 수 있다. 다른 버전을 쓴다면 아래로 설치된 소스에서 직접 확인하라(이 이름은 함수 **지역 변수**라 속성 접근으로는 못 읽는다):
>
> ```bash
> python -c "import importlib, inspect; \
> print(inspect.getsource(importlib.import_module('onnxruntime.quantization.quantize')))" \
>   | grep -A 7 calib_extra_options_keys
> ```
>
> ⚠️ **다만 `99.9`는 ORT 기본값보다 100배 공격적인 설정이다.** `PercentileCalibrater`의 자체 기본값은 **`percentile=99.999`**(`calibrate.py`)다. 즉 위 예시는 꼬리를 기본값의 **100배 두껍게** 잘라내며, 그 대가가 5.1의 **−6.2%p**다. 처음 돌릴 때는 **99.999부터 시작해 내려오는** 것이 안전하다.

#### 4.3.1 🔴 ORT의 `CalibrationMethod.Entropy`는 기본값에서 MinMax로 조용히 퇴화한다

2.4.3에서 본 KL 탐색 알고리즘은 **ORT 기본값으로는 실행되지 않는다**. `EntropyCalibrater`의 기본 `num_bins=128, num_quantized_bins=128` 때문에 탐색 후보가 정확히 1개(=전체 범위=MinMax)로 줄어든다. 실측에서 Entropy 산출물은 MinMax와 **activation scale이 32/32 전부 동일**했고, **산출된 `.onnx` 파일의 md5까지 같았다**. 50,000장 전량 평가에서도 예측 불일치 **0장**(둘 다 68.62%) — **결과는 바이트 단위로 같은데 시간만 2.8배**(9.0s → 25.1s) 쓴 셈이다. 진단 코드는 2.4.3의 `check_entropy_degenerate.py`.

> 💡 **정확한 표현**: Entropy는 이 조건에서 "더 나쁘다"가 아니라 **"작동하지 않는다"**다. 두 산출물이 같은 파일이므로 정확도를 비교할 대상 자체가 없다.

**그러면 탐색 공간을 열어 주면 되지 않나?** 여기에 두 번째 함정이 있다.

> 🔴 **함정 (`num_bins`는 `quantize_static`으로 전달 자체가 불가능하다)**: `extra_options`에 `{"num_bins": 2048}`을 넣어도 **캘리브레이터까지 도달하지 못한다**. `quantize.py`의 `calib_extra_options_keys` 화이트리스트가 **5개뿐**이기 때문이다:
>
> ```python
> ("symmetric", "CalibTensorRangeSymmetric")
> ("moving_average", "CalibMovingAverage")
> ("averaging_constant", "CalibMovingAverageConstant")
> ("max_intermediate_outputs", "CalibMaxIntermediateOutputs")
> ("percentile", "CalibPercentile")
> ```
>
> `create_calibrator()` 자체는 `num_bins`/`num_quantized_bins`를 **받는데**, `quantize_static()`이 안 넘겨준다.
>
> ⚠️ **정정 (실측)**: "조용히 무시된다"가 아니다. `quantize_static(..., extra_options={"num_bins": 2048})`을 실제로 호출하면 **`TypeError`로 즉시 죽는다** — 화이트리스트에 없는 키가 그대로 내부 호출에 흘러 들어가기 때문이다. **인자 무시보다 낫다**(조용히 틀린 결과를 얻는 대신 바로 실패한다). 다만 결론은 같다: **이 경로로는 전달할 방법이 없다.**

굳이 실험하려면 `create_calibrator`를 감싸 주입해야 한다(동작 확인된 코드):

```python
# quantize_extra.py — quantize_static이 넘기지 않는 히스토그램 파라미터를 주입
import importlib
# ⚠️ `import onnxruntime.quantization.quantize as QZ` 는 안 된다.
#    패키지 __init__ 이 같은 이름의 *함수*를 re-export 해서 모듈이 아니라 함수가 잡히고,
#    QZ.create_calibrator 에서 AttributeError 가 난다. importlib 로 모듈을 직접 가져온다.
QZ = importlib.import_module("onnxruntime.quantization.quantize")

_orig_create_calibrator = QZ.create_calibrator

def patch_hist_bins(num_bins=None, num_quantized_bins=None):
    def _patched(*a, **kw):
        eo = dict(kw.get("extra_options") or {})
        if num_bins is not None:
            eo["num_bins"] = num_bins
        if num_quantized_bins is not None:
            eo["num_quantized_bins"] = num_quantized_bins
        kw["extra_options"] = eo
        return _orig_create_calibrator(*a, **kw)
    QZ.create_calibrator = _patched

def unpatch():
    QZ.create_calibrator = _orig_create_calibrator

# 사용 (4.3 quantize_ptq.py의 run()과 CalibrationMethod를 그대로 재사용):
#   히스토그램 2048 bin, 양자화 128 레벨 → 탐색 후보 1024 − 64 + 1 = 961개
from onnxruntime.quantization import CalibrationMethod
from quantize_ptq import run

patch_hist_bins(num_bins=2048, num_quantized_bins=128)
run(CalibrationMethod.Entropy, "resnet18_int8_entropy_fixed.onnx")
unpatch()
```

> ⚠️ **주의**: 몽키패치는 `quantize_static`이 **내부에서** `create_calibrator`를 부르는 구현 세부에 의존한다. ORT 버전이 바뀌면 조용히 안 먹을 수 있으니, 적용 후 반드시 2.4.3의 `check_entropy_degenerate.py`로 **scale이 실제로 달라졌는지** 확인하라. 프로덕션 코드에 넣을 것은 못 되고, **"기본값이 알고리즘을 무력화했다"는 가설을 검증하는 실험 도구**로만 쓴다.

> 🔴 **함정 (고친다고 좋아지지 않는다 — 오히려 크게 나빠졌다)**: 위 몽키패치로 2.4.3의 알고리즘을 **제대로 돌린** 결과는 ImageNet val 50,000장 전량에서 top-1 **59.29%(−9.45%p, McNemar p<1e-300)**, 시간은 51.0s로 5.7배. 즉 "ORT의 Entropy는 망가져 있으니 고쳐 쓰라"가 아니라, **"이 모델에서는 KL이 고르는 임계가 MinMax보다 해롭다"** 가 실측 결론이다. 이유는 5.1과 같다 — ResNet18의 post-ReLU 꼬리는 잡음이 아니라 신호라서, KL이 "정보 보존"을 명분으로 꼬리를 잘라도(activation scale 중앙값이 MinMax 대비 **0.313배**, 32개 중 18개가 절반 미만) 정확도는 따라오지 않는다. **알고리즘의 우아함과 이 모델에서의 유용함은 별개다.**
>
> 실전 결론: ORT에서 Entropy를 지정하는 것은 **기본값이면 MinMax와 동일한 파일(시간만 낭비)**, **제대로 열면 이 모델에선 −9.45%p 손해**다. 진짜 KL 캘리브레이션이 필요하면 TensorRT의 `IInt8EntropyCalibrator2`(3단계)나 NVIDIA Model Optimizer를 쓰고, ORT에서는 **MinMax를 기본으로 두고 Percentile 99.999부터 스윕**하는 편이 낫다.
>
> ⚠️ **이 결론은 CNN 한정이다.** ResNet18은 post-ReLU activation에 극단적 이상치가 적고 weight를 per-channel로 양자화하므로 MinMax의 약점(이상치 하나가 scale을 늘림)이 드러나지 않는다. **이상치가 심한 Transformer/LLM에서는 결론이 뒤집힌다** — 그래서 SmoothQuant 같은 기법이 필요하다([2단계](04_transformer_quantization.md)). "MinMax가 최적"을 일반 법칙으로 외우면 2단계에서 틀린다.

#### 4.3.2 타깃별 dtype 분기표 — 이 설정은 x86 전용이다

위 4.3 코드의 `activation_type=QUInt8` + `ActivationSymmetric=False`(비대칭)는 **x86 CPU 권장 설정**이고, **TensorRT에서는 그래프가 아예 파싱되지 않는다**. 실측에서 이 QDQ를 TensorRT EP에 물렸더니 노드를 하나도 가져가지 못하고 전부 폴백해 **FP32(0.96 ms)보다 3배 느린 3.06 ms**가 나왔다 — 에러 없이, 로그를 켜지 않으면 보이지도 않는 **무음 열화**다.

TRT ONNX 파서가 뱉는 에러는 **두 종류**다([실행 로그](../logs/stage1_quantization_log.html)):

```
[ERROR] (conv1.weight_bias_DequantizeLinear: input has type Int32 but must have type
         FP8, FP4, Int4, or Int8. In checkType at nodeBase.cpp:455)     ← ① bias INT32 DQ
[ERROR] [6] Assertion failed: shiftIsAllZeros(zeroPoint):
         Non-zero zero point is not supported.                          ← ② 비대칭 zero_point≠0
        ... conv/fc 21개 전부에서 반복
```

① ORT는 bias를 **INT32**로 양자화해 `DequantizeLinear`을 붙이는데 TRT 파서는 INT32 입력 DQ를 받지 않는다. ② TensorRT는 **대칭 양자화만** 지원한다("TensorRT only supports symmetric uniform quantization, meaning that zeroPt=0" — [TensorRT Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)). ORT 공식 문서도 "quantization on GPU only supports **S8S8**"이라고 명시한다.

**그런데 에러가 둘이라고 원인이 둘은 아니다.** 두 변수를 따로 껐다 켜 본 **2×2 절제 실험** 결과, 파싱 성공/실패를 가르는 건 ② 하나뿐이었다:

| # | activation | `QuantizeBias` | act zp 범위 | INT32 bias DQ | TRT EP p50 | CUDA EP p50 | vs FP32 | 판정 |
|---|-----------|----------------|------------|---------------|-----------|-------------|---------|------|
| — | FP32 (기준선) | — | — | — | 0.96 ms | 1.33 ms | 1.00× | — |
| A | `QUInt8` 비대칭 | `True` | `[0, 173]` | 21개 | 3.06 ms | 1.81 ms | 0.31× | 🔴 무음 폴백 |
| B | `QUInt8` 비대칭 | **`False`** | `[0, 173]` | **0개** | 2.97 ms | 1.80 ms | 0.32× | 🔴 **여전히 폴백** |
| C | `QInt8` 대칭 | `True` | **`[0, 0]`** | **21개** | **0.51 ms** | 2.11 ms | **1.86×** | ✅ **TRT 실행** |
| D | `QInt8` 대칭 | `False` | **`[0, 0]`** | 0개 | 0.51 ms | 1.99 ms | 1.86× | ✅ TRT 실행 |

INT32 bias DQ를 **0개로 없애도 실패**하고(B), **21개가 남아 있어도 성공**한다(C). 즉 **하드 블로커는 ② zero-point 하나**이고, **① 은 융합이 깨진 뒤 홀로 남은 bias DQ가 타입 검사에 걸리는 2차 증상**이다. 파서 로그에서 INT32 bias DQ 에러가 44회, zero-point 관련 메시지가 290회 쏟아지지만 **에러 메시지 개수를 원인 개수로 세면 틀린다.** 실무적으로: `activation_type`과 `ActivationSymmetric`을 **같이** 뒤집는 것이 유일한 해법이고, `"QuantizeBias": False`는 그래프를 정리하는 **선택 옵션**(0.51 ↔ 0.51 ms, 차이 없음)이지 해법이 아니다.

> 🔍 **무음 폴백 판별법 (이 표의 CUDA EP 열이 그 방법이다)**: provider 목록에 `TensorrtExecutionProvider`가 **보이는데도** A/B는 FP32(0.96 ms)보다 3배 느리다. 예외도 경고도 없다. 판별법은 **같은 모델을 CUDA EP로도 돌려 p50을 비교**하는 것 — **TRT가 CUDA보다 빠르지 않으면 폴백이다.** 성공한 C/D는 FP32 대비 1.86×이고, 같은 모델의 CUDA EP(2.11 ms)보다 4.1배 빠르다.
>
> 같은 열에서 하나 더: **INT8 QDQ를 CUDA EP로 돌리면 FP32보다 느리다**(INT8 1.80~2.11 ms vs FP32 1.33 ms). CUDA EP는 QDQ를 시뮬레이션(양자화→역양자화)만 하므로 노드가 늘어난 만큼 느려진다. **INT8은 그 자체로 빠른 게 아니라, INT8 커널을 실제로 쓰는 EP에서만 빠르다.**

| 타깃 | `activation_type` | `ActivationSymmetric` | `weight_type` / `WeightSymmetric` | 비고 |
|------|-------------------|-----------------------|-----------------------------------|------|
| **x86 CPU (AVX-VNNI/AVX512-VNNI)** | `QuantType.QUInt8` | `False` (비대칭) | `QInt8` / `True` | ORT 공식 권장(U8S8). 2.2의 이론과 일치 |
| **TensorRT / NVIDIA GPU** | **`QuantType.QInt8`** | **`True` (대칭)** | `QInt8` / `True` | **S8S8만 지원.** 비대칭이면 파싱 실패 → 무음 폴백. 필요 시 `"QuantizeBias": False` 추가 |
| **ARM CPU (모바일)** | `QUInt8` | `False` | `QInt8` / `True` | x86과 동일 계열. 단 백엔드별 재확인 필요 |
| **NPU/DSP (QNN·TIDL 등)** | 벤더 SDK 규약을 따른다 | — | — | 4단계에서 다룸. 대개 대칭 INT8 요구 |

TensorRT용 산출물은 아래처럼 만든다(동작 확인됨):

```python
# quantize_trt.py — TensorRT가 실제로 먹을 수 있는 QDQ
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType, QuantFormat
from calib_reader import ImageNetCalibReader

CALIB_DIR = "imagenet/val"        # 4.3과 동일한 캘리브 이미지 폴더

def run(out, **extra):
    eo = {"WeightSymmetric": True, "ActivationSymmetric": True}   # ← 둘 다 대칭
    eo.update(extra)
    quantize_static(
        "resnet18_fp32.onnx", out,
        ImageNetCalibReader(CALIB_DIR, input_name="input", limit=200),
        quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,       # ← TRT는 int8 대칭만 받는다 (uint8 아님)
        weight_type=QuantType.QInt8,
        per_channel=True, reduce_range=False, extra_options=eo)
    print("saved", out)

run("resnet18_int8_trt_sym.onnx")
run("resnet18_int8_trt_nobias.onnx", QuantizeBias=False)   # bias INT32 QDQ 자체를 제거
```

**실측 효과** (TensorRT EP p50, batch=1 · 100회 — [실행 로그](../logs/stage1_quantization_log.html)):

| 설정 | TRT p50 | vs FP32 | top-1 (50k 전량) | vs MinMax | McNemar p |
|------|---------|---------|------------------|-----------|-----------|
| FP32 | 0.96 ms | 1.00× | 68.74% | — | — |
| INT8 `QUInt8` 비대칭 (위 4.3 기본 설정 = A) | **3.06 ms** | **0.31× (느려짐)** | 68.62% | — (기준) | — |
| INT8 `QUInt8` 비대칭 + `QuantizeBias=False` (B) | 2.97 ms | 0.32× | 68.62% | +0.00%p | 예측 불일치 **1장** |
| INT8 `QInt8` 대칭 (C) | **0.51 ms** | **1.86×** | 68.33% | **−0.29%p** | **9.2e-5 (유의)** |
| INT8 `QInt8` 대칭 + `QuantizeBias=False` (D) | 0.51 ms | 1.86× | 68.33% | −0.29%p | C 대비 예측 불일치 **0장** |

**설정 하나로 6배가 갈린다**(3.06 → 0.51 ms). 그리고 **두 축이 완전히 분리된다**:

- **정확도는 대칭/비대칭 축만 움직인다.** 대칭 전환의 대가는 **−0.29%p이고 통계적으로 유의하다**(p=9.2e-5). zero-point를 0으로 묶으면 post-ReLU처럼 **한쪽만 쓰는 분포에서 표현 구간 절반을 버리기** 때문이다.
- **`QuantizeBias`는 정확도에 영향이 없다.** C와 D의 예측이 **0장 불일치**(완전 동일), B는 A와 **1장**만 다르다. 이 옵션은 **파싱/그래프 정리 관점에서만** 논할 것이다.

2.2에서 "activation은 비대칭이 이득"이라고 배운 그 이득은 실제로 존재하지만(−0.29%p로 측정됨), **TensorRT 타깃에서는 애초에 선택지가 아니다.** 0.29%p를 지불하고 6배를 받는 거래이며, 붙잡고 있으면 정확도 이득도 못 얻고 속도만 잃는다(A/B는 FP32보다 3배 느리다).

> 💡 **3단계로 이어짐**: 이 QDQ ONNX는 [3단계 TensorRT](05_tensorrt.md)의 `trtexec` 입력이 된다. **대칭 설정으로 만들지 않으면 3단계에서 그대로 터진다.** 무음 폴백을 잡아내는 법(EP 로그 레벨 상향, 실제 EP 배정 확인)도 3단계에서 다룬다. 이 사례는 [함정 모음](10_pitfalls.md)의 "fallback 지옥"에 해당하는 실측 케이스다.

### 4.4 top-1 평가 스크립트 (+ 신뢰구간)

```python
# eval_top1.py — FP32/INT8 ONNX를 ImageNet-val N장으로 top-1 평가 + 신뢰구간
import os, glob, math, numpy as np, onnxruntime as ort
from calib_reader import preprocess          # 4.2와 동일 전처리 재사용

def load_labels(val_dir):
    # 클래스 폴더명을 정렬 → 인덱스. (path, label_idx) 리스트 반환
    classes = sorted(os.listdir(val_dir))
    cls2idx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        for p in glob.glob(os.path.join(val_dir, c, "*.JPEG")):
            items.append((p, cls2idx[c]))
    return sorted(items)          # 정렬 고정 = 모든 모델이 "같은 순서"를 봐야 paired 비교 가능

def stratified(items, n, classes=1000):
    """🔴 items[:n] 로 자르면 안 된다 — items는 정렬돼 있어서 앞 n/50 클래스만 뽑힌다.
    클래스마다 같은 수씩 고르는 층화 표본을 만든다. 순서는 고정(paired 비교용)."""
    per, by = max(1, n // classes), {}
    for p, y in items:
        by.setdefault(y, []).append(p)
    out = [(p, y) for y in sorted(by) for p in sorted(by[y])[:per]]
    return out[:n]

def evaluate(onnx_path, subset):
    """이미지별 정오답 bool 배열을 반환(부분집합 재채점·paired 검정에 쓰기 위해)."""
    sess = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    hits = np.zeros(len(subset), dtype=bool)
    for i, (path, label) in enumerate(subset):
        logits = sess.run(None, {iname: preprocess(path)})[0]
        hits[i] = int(np.argmax(logits)) == label
    return hits

def wilson_ci(correct, n, z=1.96):
    """이항 비율 p=correct/n의 95% Wilson 신뢰구간 (정규근사보다 소표본에 안전)."""
    p = correct / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, center - half, center + half

def mcnemar(a, b):
    """paired 정오답 배열 두 개의 McNemar 검정(연속성 보정). returns (b01, b10, chi2, p).
    a, b 는 같은 이미지 순서의 bool 배열. 'a만 맞음'/'b만 맞음' 개수만으로 검정한다."""
    b01 = int(np.sum(a & ~b))          # a만 맞음
    b10 = int(np.sum(~a & b))          # b만 맞음
    n = b01 + b10
    if n == 0:                         # 두 모델의 정오답이 완전히 동일
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n            # 연속성 보정
    p = math.erfc(math.sqrt(chi2 / 2))              # 자유도 1 카이제곱 상측확률
    return b01, b10, chi2, p

if __name__ == "__main__":
    items = load_labels("imagenet/val")
    # 전량(50,000장)이면 items 그대로, 줄일 거면 반드시 층화 표본으로
    subset = items if len(items) <= 50000 else stratified(items, 50000)
    results = {}
    for name, path in [
        ("FP32",            "resnet18_fp32.onnx"),
        ("INT8 MinMax",     "resnet18_int8_minmax.onnx"),
        ("INT8 MinMax(PT)", "resnet18_int8_minmax_pertensor.onnx"),
        ("INT8 Entropy",    "resnet18_int8_entropy.onnx"),
        ("INT8 Pctile99.9", "resnet18_int8_pct999.onnx"),
    ]:
        hits = evaluate(path, subset)
        results[name] = hits
        c, n = int(hits.sum()), len(hits)
        p, lo, hi = wilson_ci(c, n)
        print(f"{name:16s} top-1 = {p*100:5.2f}%  "
              f"95% CI [{lo*100:5.2f}, {hi*100:5.2f}]  (n={n})")

    # paired 비교 — CI 겹침만 보지 말고 McNemar까지 볼 것 (아래 함정)
    print("\n--- paired (McNemar) ---")
    for a, b in [("FP32", "INT8 MinMax"),
                 ("INT8 MinMax", "INT8 MinMax(PT)"),
                 ("INT8 MinMax", "INT8 Entropy"),
                 ("INT8 MinMax", "INT8 Pctile99.9")]:
        b01, b10, chi2, pv = mcnemar(results[a], results[b])
        d = (results[b].sum() - results[a].sum()) / len(results[a]) * 100
        sig = "유의(p<0.05)" if pv < 0.05 else "유의하지 않음"
        print(f"{a:16s} → {b:16s} {d:+5.2f}%p  "
              f"a만={b01:3d} b만={b10:3d} p={pv:.4f}  → {sig}")
```

> 💡 **팁**: `evaluate()`가 정답 개수 대신 **이미지별 정오답 bool 배열**을 돌려주게 한 것이 핵심이다. 이 배열만 `np.savez`로 남겨 두면 재추론 없이 **부분집합 재채점**(예: 캘리브에 쓴 이미지를 뺀 holdout 점수)과 **모든 모델 쌍의 paired 검정**을 나중에 다시 할 수 있다. 실측에서도 캘리브 200장을 제외한 holdout **49,800장** 점수가 전체 50,000장과 일치하는지 이 방식으로 확인했다(FP32 68.740% → 68.755%, MinMax 68.622% → 68.633% — **차이 +0.02%p 이내**라 캘리브·평가 중복이 결론에 영향 없음).

```bash
python eval_top1.py
```

> 🔴 **함정 (표본 수·신뢰구간)**: n=1000장에서 top-1 차이 **±1%p 이내는 통계적으로 유의하지 않을 수 있다**. 이항비율 표준오차는 `√(p(1−p)/n)`이라 p≈0.69, n=1000이면 SE ≈ 1.46%p, 95% 신뢰구간 폭은 대략 **±2.9%p**다. 즉 두 캘리브레이션의 0.7%p 차이는 **1000장으로는 노이즈와 구분이 안 된다**. 결론을 강하게 내려면 (a) 평가 장수를 5000~50000으로 늘리거나, (b) **같은 이미지 집합**에 대해 두 모델의 정오답을 짝지어(paired, McNemar) 비교하라. 위 스크립트의 Wilson CI가 겹치면 "차이 있다"고 단정하지 말 것.
>
> 🔴 **함정 (실측 — 소규모 평가셋으로 "방법 우열"을 결론내면 안 된다)**: 이 문서는 처음에 **클래스당 1장 큐레이션 셋(1000장)** 으로 실습했다. 같은 모델들을 **50,000장 전량**으로 다시 재니 세 가지가 드러났다([재실행 보고서](../logs/stage1_real_imagenet_report.html)):
>
> | 문제 | 실측 |
> |------|------|
> | **① 절대값은 폐기해야 한다** | 큐레이션 셋은 '쉬운 사진'에 치우쳐 top-1이 부풀려진다. 부풀림 평균 **+9.77%p**, 범위 **[+8.41, +10.39]** — 폭이 **1.98%p**나 벌어지므로 "일괄 −9.8%p" 같은 **상수 보정도 불가능**하다. 보정하면 모델 간 순위가 왜곡된다. |
> | **② Δ의 부호가 뒤집힐 수 있다** | 13개 비교 중 **3건이 부호 반전**. MinMax와 Entropy는 큐레이션에서 FP32를 **+0.40%p로 이겼지만**, 50k에서는 **−0.12%p로 진다**. |
> | **③ 유의성 판정이 뒤집힌다** | 13건 중 **5건**이 뒤집혔고 **전부 "유의하지 않음 → 유의"** 한 방향이다. 우연이 아니라 표본이 50배 늘어난 결과다. |
>
> **핵심 교훈**: 작은 셋에서 나온 "유의하지 않음"은 **"차이가 없다"가 아니라 "모른다"**다. n=1000, p≈0.7에서 0.3%p 차이를 80% 검정력으로 잡으려면 수만 장이 필요하다. **1000장으로 파이프라인을 돌려 보는 것은 좋지만, 그 숫자로 캘리브레이션 방법의 우열을 결론내지 말라.** 방법 선택은 50k 전량(또는 최소 수천 장 층화 표본)에서 paired 검정으로 판정한다.
>
> **왜 McNemar가 따로 필요한가 (실측 50k):** Wilson CI는 두 모델을 **독립 표본**처럼 다루지만, 실제로는 **같은 50,000장을 둘 다 본다**. 같은 이미지를 둘 다 맞히거나 둘 다 틀리는 부분은 비교에 정보를 주지 않으므로, **엇갈린 이미지만** 세는 McNemar가 훨씬 민감하다. 실측 예: per-channel(68.62%) vs per-tensor(68.46%)의 Wilson CI는 `[68.21, 69.03]` vs `[68.05, 68.86]`로 **거의 완전히 겹쳐** CI만 보면 아무 말도 못 하지만, 같은 데이터를 paired로 보면 **엇갈린 이미지 1,626장**(per-ch만 맞음 854 / per-tensor만 맞음 772)이고 **p=0.0445로 유의**하다. 반대로 Percentile 99.9는 엇갈림 6,243장에 p<1e-300으로 **압도적으로** 갈린다. **CI는 "말할 수 없음"까지만 알려주고, McNemar가 "어느 쪽이 얼마나"를 알려준다** — 표본을 50배로 늘려도 이 관계는 그대로다(CI는 여전히 겹치고, McNemar는 판정한다).

> 🔴 **함정 (라벨 매핑)**: 위 `load_labels`는 "폴더명 정렬 순서 = torchvision 클래스 인덱스"라고 가정한다. 이 가정은 **폴더가 1000개 다 있을 때만** 성립한다. torchvision ResNet18의 인덱스가 ImageNet **synset ID(n0…)의 사전순 정렬과 일치**한다는 것은 이 실습에서 1000/1000으로 실측 확인했지만(3.1절), **클래스 서브셋(Imagenette 등)에서는 인덱스가 0부터 다시 매겨져 top-1이 0%로 무너진다** — 서브셋 대응 코드는 3.1절 참조. 어느 쪽이든 **먼저 FP32 점수부터 확인하는 sanity check**가 최선의 방어다 — **ImageNet val 50,000장 전량 + torchvision 전처리면 69.8%**(실측 69.81%, 공개값 69.758%와 0.05%p 일치)다. 이 값에서 1%p 이상 벗어나면 양자화를 논하기 전에 **전처리나 라벨 매핑을 의심하라**(전처리 하나로 −1.07%p가 나온다 — 4.2절).

### 4.5 레이어별 SQNR / 코사인 유사도 → `layer_sensitivity.csv`

**아이디어**: FP32 weight와 그것을 INT8로 양자화한 weight를 레이어마다 비교해, 얼마나 달라졌는지를 측정한다. 많이 달라진(=SQNR 낮고, 코사인 유사도 낮은) 레이어가 **양자화에 민감한 "범인 레이어"**다. 이 레이어를 다음 단계에서 FP16으로 남기면(mixed precision) 정확도를 되살릴 수 있다.

#### 4.5.1 SQNR·코사인 유사도 유도

**SQNR (Signal-to-Quantization-Noise Ratio, dB).** "신호 대 양자화잡음 비"를 데시벨로 나타낸 것. 원신호 `x`, 양자화 재구성 `x̂`, 잡음 `n = x − x̂`.

```
SQNR = 10 · log10( 신호 파워 / 잡음 파워 )
     = 10 · log10( E[x²] / E[(x − x̂)²] )
     = 10 · log10( ‖x‖² / ‖x − x̂‖² )      (텐서 전체 합으로 추정)
```

높을수록 잡음이 신호 대비 작다(=안전). **유도로 얻는 통찰(6 dB/bit 규칙):** 2.1.1에서 반올림잡음 파워 ≈ `s²/12`이고 `s = 2A/2ᵇ`(범위 절반 A, b비트)이면 신호 파워를 `σ²`로 둘 때

```
SQNR ≈ 10·log10( σ² / (s²/12) ) = 10·log10( 12σ²/s² )
     = 10·log10(12σ²) − 20·log10(s)
     ∝ 20·log10(2ᵇ) = b · 20·log10(2) ≈ 6.02·b (dB)
```

→ **비트를 1 늘리면 SQNR이 약 6 dB 오른다**(고전 DSP의 "6 dB/bit"). INT8(8bit) weight의 이상적 SQNR은 대략 `6·8 ≈ 48 dB`에 상수항을 더한 값 근처이며, **실측이 이보다 크게 낮으면(예: 20 dB대) 그 레이어는 outlier/넓은 dynamic range로 유효 비트를 못 쓰고 있다**는 신호다. 이것이 감도 지표로 SQNR을 쓰는 이유다.

**코사인 유사도.** 두 벡터의 방향이 얼마나 같은지(크기 무시).

```
cos(x, x̂) = (x · x̂) / (‖x‖ · ‖x̂‖) = Σ xᵢ x̂ᵢ / (√Σxᵢ² · √Σx̂ᵢ²)
```

1에 가까울수록 방향 보존. **SQNR과 상보적**인 이유: SQNR은 크기 오차에 민감하지만, 신경망은 다음 레이어가 내적(방향)을 취하므로 **방향이 틀어지는 것**이 치명적일 수 있다. 예컨대 x̂이 x를 전부 2배 키웠다면 SQNR은 나쁘지만 코사인은 1.0(방향 동일). 두 지표를 함께 봐야 "크기가 나간 건지, 방향이 나간 건지" 진단된다.

> 💡 **팁**: 이전 판에서 "SNR"로 부르던 것과 동일 개념이다. 양자화 문헌에서는 잡음이 양자화에서 오므로 **SQNR**(Signal-to-**Quantization**-Noise Ratio)이 더 정확한 용어다.

여기서는 **각 레이어의 weight를 직접 양자화**했을 때의 재구성 오차로 감도를 측정한다(activation 없이 weight만으로도 범인 레이어 후보를 잘 잡아낸다. ONNX 중간 텐서를 뽑는 방법은 4.5-b 참고).

```python
# layer_sensitivity.py — 레이어별 weight 양자화 SQNR/코사인 → CSV
import csv, numpy as np, torch, torchvision

def quantize_per_channel_symmetric(w, num_bits=8):
    """weight를 output-channel별 symmetric int8로 quantize→dequantize."""
    q_max = 2 ** (num_bits - 1) - 1                       # 127
    w2 = w.reshape(w.shape[0], -1)                        # [out_ch, *]
    scale = w2.abs().max(dim=1).values / q_max            # 채널별 scale
    scale = torch.clamp(scale, min=1e-12).unsqueeze(1)
    q = torch.clamp(torch.round(w2 / scale), -q_max, q_max)
    w_hat = (q * scale).reshape(w.shape)
    return w_hat

def quantize_per_tensor_symmetric(w, num_bits=8):
    """대조군: 텐서 전체에 scale 하나. per-channel 이득을 dB로 재기 위한 것."""
    q_max = 2 ** (num_bits - 1) - 1
    scale = torch.clamp(w.abs().max() / q_max, min=1e-12)
    return torch.clamp(torch.round(w / scale), -q_max, q_max) * scale

def sqnr_db(x, x_hat):
    noise = (x - x_hat).pow(2).sum().item()
    sig   = x.pow(2).sum().item()
    return 10 * np.log10(sig / (noise + 1e-12))

def cosine(x, x_hat):
    a, b = x.flatten(), x_hat.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

if __name__ == "__main__":
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w = module.weight.detach().float()
            w_hat = quantize_per_channel_symmetric(w)
            rows.append({
                "layer": name,
                "type": type(module).__name__,
                "out_channels": w.shape[0],
                "num_params": int(w.numel()),
                "sqnr_db": round(sqnr_db(w, w_hat), 3),
                "cosine": round(cosine(w, w_hat), 6),
                # 대조군: per-tensor로 했을 때의 SQNR (per-channel 이득을 dB로 정량화)
                "sqnr_db_pertensor": round(
                    sqnr_db(w, quantize_per_tensor_symmetric(w)), 3),
            })

    # SQNR 오름차순 = 가장 민감한(위험한) 레이어가 위로
    rows.sort(key=lambda r: r["sqnr_db"])
    with open("layer_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("saved layer_sensitivity.csv  (rows:", len(rows), ")")
    print("\n범인 후보 top-5 (SQNR 낮은 순):")
    for r in rows[:5]:
        print(f'  {r["layer"]:30s} SQNR={r["sqnr_db"]:6.2f} dB  cos={r["cosine"]:.4f}  '
              f'(per-tensor {r["sqnr_db_pertensor"]:6.2f} dB)')
    print("\n안전한 쪽 bottom-3:")
    for r in rows[-3:]:
        print(f'  {r["layer"]:30s} SQNR={r["sqnr_db"]:6.2f} dB  cos={r["cosine"]:.4f}  '
              f'(per-tensor {r["sqnr_db_pertensor"]:6.2f} dB)')
    gains = [r["sqnr_db"] - r["sqnr_db_pertensor"] for r in rows]
    print(f'\nper-channel 이득: 평균 {np.mean(gains):.2f} dB  최대 {np.max(gains):.2f} dB '
          f'({rows[int(np.argmax(gains))]["layer"]})')
    print(f'이론적 INT8 상한(6.02*8) ≈ 48.2 dB / '
          f'실측 중앙값 {np.median([r["sqnr_db"] for r in rows]):.2f} dB')
```

```bash
python layer_sensitivity.py
# → layer_sensitivity.csv 생성 + 범인 후보 top-5 / 안전 bottom-3 / per-channel 이득 출력
```

#### 4.5.2 `layer_sensitivity.csv` 스키마 (컬럼 정의)

이 CSV는 **이 문서(1단계)가 생성하는 정본 산출물**이며, 2·3단계가 그대로 읽는다. 컬럼 정의:

| 컬럼 | 타입 | 의미 | 사용처 |
|------|------|------|--------|
| `layer` | str | 레이어의 `named_modules()` 이름 (예: `layer3.0.conv1`) | FP16 승격 대상 지정 키 |
| `type` | str | `Conv2d` / `Linear` | 레이어 종류별 정책 분기 |
| `out_channels` | int | 출력 채널 수 | 채널 적은 레이어가 민감한지 진단 |
| `num_params` | int | weight 원소 수 (`w.numel()`) | 승격 시 늘어나는 메모리/연산 가늠 |
| `sqnr_db` | float | **per-channel** weight 양자화 SQNR(dB), **낮을수록 위험** | **정렬·임계의 주 지표** |
| `cosine` | float | weight 코사인 유사도, **낮을수록(1에서 멀수록) 위험** | SQNR 보조 진단(방향 붕괴) |
| `sqnr_db_pertensor` | float | 같은 weight를 **per-tensor**로 양자화했을 때의 SQNR(dB) | `sqnr_db − sqnr_db_pertensor` = **그 레이어의 per-channel 이득(dB)**. 2.2.2 이론의 레이어별 검증값 |

행은 **`sqnr_db` 오름차순**(가장 위험한 레이어가 맨 위)으로 저장된다.

> 💡 **팁**: 마지막 두 컬럼의 차이가 2.2.2에서 손으로 계산한 per-channel 이득의 **실측판**이다. 실측에서 ResNet18 21개 레이어의 이득은 평균 **7.38 dB(≈1.2비트)**, 최대 **14.06 dB(≈2.3비트, `layer4.0.downsample.0`)** 였다 — 6 dB/bit 규칙으로 환산하면 "per-channel로 바꾸는 것만으로 어떤 레이어는 **2비트 이상을 되찾는다**"는 뜻이다.

#### 4.5.3 활용법 — 어떤 임계로 FP16 승격을 결정하나

목표: 전체를 INT8로 두되, **가장 민감한 소수 레이어만 FP16으로 되돌려(mixed precision)** 정확도를 회복하고 속도 이득은 지킨다.

> 🔴 **먼저 읽을 것 — weight SQNR은 이 모델에서 예측력이 없다(실측)**: 21개 레이어를 **하나씩만** fake-quant하고 매번 **50,000장 top-1을 실제로** 쟀다. SQNR이 대리지표로 유효하다면 SQNR 낮은 레이어의 낙폭이 커야 한다. 결과는 그렇지 않았다([보고서](../logs/stage1_real_imagenet_report.html)):
>
> | | 레이어 | SQNR(per-ch) | 실제 Δtop-1 |
> |---|---|---|---|
> | SQNR이 "위험"하다고 지목한 3개 | `layer1.0.conv1` / `layer3.0.conv2` / `layer2.1.conv1` | 36.15 / 36.30 / 36.58 dB | **+0.026 / +0.044 / +0.006%p** (전부 양수) |
> | 실제 Δtop-1 최악 3개 | `layer3.0.conv1` / `layer3.1.conv1` / `layer1.1.conv2` | 37.42 / 37.07 / 39.00 dB (중상위권) | −0.020 / 0.000 / 0.000%p |
>
> **교집합 0개. Pearson +0.06 / Spearman −0.04 (n=21)** — 상관이 없다. SQNR은 *"이 레이어의 weight가 얼마나 뭉개졌나"* 는 잘 재지만 *"그게 최종 정확도로 얼마나 옮겨지나"* 는 재지 못한다.
>
> 단 이 모델에서는 **단일 레이어 Δ가 전부 ±0.05%p 이내(측정 노이즈 수준)라 순위 자체가 의미 없다**는 점도 같이 봐야 한다. 21개를 동시에 양자화해도 −0.044%p이고 개별 낙폭의 단순 합은 +0.292%p다 — 둘 다 노이즈라 "오차가 증폭되지 않는다"까지만 말할 수 있다. 즉 **ResNet18 weight-only per-channel INT8은 사실상 무손실**이고, 실제 PTQ 손실(−0.12%p)은 weight가 아니라 **activation 양자화에서 나온다.**
>
> **실무 규칙**: 혼합정밀도 대상은 **실제 과제 지표(top-1/mAP)로 직접 고른다.** 레이어가 많아 전수 평가가 불가능할 때만 대리지표를 쓰되, **그 대리지표의 예측력을 먼저 몇 개 레이어로 검증**하라. 아래 기준들은 그 전제 위에서만 유효하다.

결정 기준 3가지(권장 순):

1. **상대 기준(권장·robust):** CSV를 `sqnr_db` 오름차순으로 보고 **하위 5~10%** 를 FP16 후보로. 절대 임계보다 모델·데이터에 덜 민감하다. 예) ResNet18의 Conv/Linear ~20개면 하위 2~3개. ⚠️ 단 위 실측대로 **이 순위가 실제 낙폭 순위와 일치한다는 보장은 없다** — 후보를 뽑는 도구로 쓰고, 채택은 top-1 재측정으로 판정하라.
2. **절대 기준(보조):** `sqnr_db < 20 dB` **또는** `cosine < 0.99` 인 레이어를 후보로. 단, 임계값은 모델마다 재보정 필요(2.5.1의 6dB/bit로 감을 잡되 실측 분포를 보고 조정). 🔴 **ResNet18 per-channel 실측에서는 이 기준에 걸리는 레이어가 0개**였다(최저 36.1 dB, 최저 cosine 0.99988) — 절대 기준은 "안전한 모델"에서 **아무 후보도 못 내놓기** 때문에 1번이 기본이어야 한다(5.2).
3. **비용가중 기준:** `sqnr_db`가 낮고 `num_params`가 작은 레이어를 우선 승격(적은 비용으로 큰 정확도 회복). ⚠️ **"first conv/downsample이 여기 자주 걸린다"는 통념은 per-channel에서는 틀리다** — per-channel은 채널마다 scale을 따로 주므로 **채널 수가 적을수록 오히려 유리**하고, 실측에서 `conv1`과 downsample 3개는 전부 **가장 안전한 쪽**에 몰렸다(5.2). 이 통념은 **per-tensor를 전제할 때만** 성립하니, 비용가중 기준은 통념이 아니라 **자기 CSV의 실제 값**으로 적용하라.

**실전 절차:** ① 후보 레이어 목록을 CSV에서 뽑는다 → ② 3단계 TensorRT에서 해당 레이어에 `precision=FP16`(또는 Q/DQ 제거)을 지정 → ③ top-1을 4.4로 재측정, 신뢰구간까지 확인 → ④ 목표 정확도(예: FP32 대비 −0.5%p 이내)를 만족할 때까지 후보를 1~2개씩 늘린다. **CSV는 "어디부터 되돌릴지"의 우선순위 큐** 역할을 한다.

> 💡 **팁 (activation SQNR까지 보려면, 4.5-b)**: weight만으로 부족하면 ONNX 중간 텐서를 뽑아 activation SQNR을 잰다. `onnx.utils.extract_model(...)`로 특정 노드까지 잘라낸 서브그래프를 만들거나, 모델의 모든 중간 텐서를 graph output으로 추가한 뒤 FP32/INT8 세션에서 같은 입력으로 실행해 텐서별로 위 `sqnr_db`/`cosine`을 적용하면 된다. QDQ 노드가 텐서 이름을 바꾸므로 이름 매칭에 주의. activation은 데이터 의존적이라 weight-only보다 실제 정확도와 상관이 높지만 계산이 무겁다 — **weight-only로 후보를 좁힌 뒤 상위 후보만 activation SQNR로 재확인**하는 2단계 전략이 실전적이다.

---

## 5) 예시 / 결과 해석

### 5.1 top-1 비교 (실측)

> ✅ **이 표는 ImageNet val 50,000장 전량 실측이다.** 전처리는 4.2의 torchvision 방식, 캘리브는 200장, 모든 모델이 **동일한 50,000장·동일 전처리 캐시**를 본다(paired). 95% CI는 Wilson, McNemar는 FP32와의 paired 비교(4.4). 전체 출력은 [재실행 로그](../logs/stage1_real_imagenet_log.html)·[분석 보고서](../logs/stage1_real_imagenet_report.html).
>
> FP32 **68.74%**(squash 전처리) / **69.81%**(tv 전처리)이고, 후자가 공개 재현값 69.758%와 **0.05%p** 안에서 만난다. 아래 표는 **squash 전처리 기준**으로 통일했다(모델 간 비교가 목적이므로 전처리는 상수로 고정). 전처리를 tv로 바꾸면 전 행이 약 +1.07%p 이동한다.
>
> ⚠️ 캘리브 200장은 평가셋에 포함돼 있다. 캘리브 인덱스를 뺀 holdout 49,800장에서도 같이 계산했고 결론은 바뀌지 않았다(50,000 중 200장이라 영향이 0.01%p 수준).

| 모델 | 캘리브레이션 | per-ch | top-1 | 95% CI | FP32 대비 | McNemar p |
|------|-------------|--------|-------|--------|-----------|-----------|
| **FP32 (기준)** | — | — | **68.74%** | [68.33, 69.14] | — | — |
| INT8 | **MinMax** | ✔ | **68.62%** | [68.21, 69.03] | **−0.12%p** | 0.061 n.s. |
| INT8 | MinMax, 캘리브 1000클래스 | ✔ | 68.65% | [68.25, 69.06] | −0.09%p | 0.189 n.s. |
| INT8 | MinMax | ✘ (per-tensor) | 68.46% | [68.05, 68.86] | −0.28%p | **3.4e-4 유의** |
| INT8 | Entropy (ORT 기본) | ✔ | 68.62% | [68.21, 69.03] | −0.12%p | 0.061 n.s. |
| INT8 | Entropy (탐색 정상화 2048/128) | ✔ | **59.29%** | [58.86, 59.72] | **−9.45%p** | **<1e-300 유의** |
| INT8 | Percentile 99.9 | ✔ | **61.91%** | [61.48, 62.33] | **−6.83%p** | **<1e-300 유의** |
| INT8 | Percentile 99.99 | ✔ | 68.31% | [67.90, 68.71] | −0.43%p | **4.1e-9 유의** |
| INT8 | Percentile 99.999 (ORT 기본) | ✔ | 68.58% | [68.17, 68.99] | −0.16%p | **0.007 유의** |
| INT8 | MinMax, **QInt8 대칭**(TRT용) | ✔ | 68.33% | [67.92, 68.74] | −0.41%p | **5.0e-8 유의** |

**해석**:

- ResNet18 정도의 잘 정규화된 CNN은 INT8 PTQ만으로도 **손실이 사실상 없다**. 정확한 서술은 **"ResNet18 PTQ INT8(MinMax)은 FP32와 통계적으로 구별되지 않는다"** 다 — Δ가 −0.12%p이고 McNemar p=0.061로 유의 경계를 넘지 못한다. **"INT8이 FP32를 상회할 수 있다"고 쓰지 말 것**: 이 문서가 한때 근거로 삼은 +0.40%p는 1000장 큐레이션 셋의 값이었고, 그때조차 p=0.48로 **유의하지 않은 값을 결론으로 승격시킨 오류**였다(4.4의 함정 참조).

- **표본이 50배 늘어나자 판정이 바뀐 것들이 있다.** per-tensor(−0.28%p), Percentile 99.99(−0.43%p), 99.999(−0.16%p)는 1000장에서는 전부 "유의하지 않음"이었지만 50k에서는 **모두 유의**하다. 뒤집힌 5건이 **전부 같은 방향**(유의하지 않음 → 유의)인 것이 표본 부족의 서명이다. **작은 셋의 "차이 없음"은 "모른다"의 다른 표현이다.**

- 🔴 **이 표에서 가장 중요한 것: MinMax가 최적이고, 클리핑하는 방법들이 졌다.** 2.1.1의 worked example은 "MinMax는 outlier에 취약"이라고 가르쳤는데 실측은 그 반대다. **모순이 아니라 전제의 차이**다 — worked example은 outlier가 **분포에서 튀어나온 잡음 1개**인 상황을 가정했다. ResNet18의 post-ReLU activation에서는 **큰 값이 곧 강한 특징 응답**이라 잡으면 잡을수록 손해다. 즉 **"어느 캘리브레이터가 우수한가"에는 모델 무관한 정답이 없다.**

- **클리핑 양과 정확도는 거의 완전한 단조 관계다.** activation 표현범위 상한이 MinMax 대비 몇 배인지(중앙값)와 top-1을 나란히 놓으면:

  | 캘리브레이션 | scale 비율 중앙값(MinMax=1.0) | MinMax와 scale 동일 | 0.5배 미만 | top-1 | Δ vs FP32 |
  |---|---|---|---|---|---|
  | MinMax | 1.000 | 32 / 32 | 0 / 32 | 68.62% | −0.12%p |
  | Entropy (ORT 기본) | 1.000 | **32 / 32** | 0 / 32 | 68.62% | −0.12%p |
  | Percentile 99.999 | 0.668 | 0 / 32 | 2 / 32 | 68.58% | −0.16%p |
  | Percentile 99.99 | 0.486 | 0 / 32 | 19 / 32 | 68.31% | −0.43%p |
  | Entropy (2048 bin) | 0.313 | 1 / 32 | 18 / 32 | **59.29%** | **−9.45%p** |
  | Percentile 99.9 | 0.323 | 0 / 32 | 28 / 32 | **61.91%** | **−6.83%p** |

  **자를수록 나빠진다**, 예외 없이. 그리고 0.5배 근처에서 무너지기 시작한다. 이 표가 이 실습에서 가장 재사용 가치가 큰 결과물이다 — **새 모델에서 캘리브레이터를 고를 때 top-1을 재기 전에 "scale이 MinMax 대비 몇 배로 줄었나"부터 보면 된다.** ("MinMax와 scale 동일" 열이 4.3.1 Entropy 퇴화의 직접 증거다 — 32/32 동일이면 두 산출물은 같은 파일이다.)

- **캘리브 이미지 커버리지는 통념만큼 중요하지 않다.** "캘리브 셋이 전 클래스를 덮어야 한다"는 조언을 200클래스(200장) vs 1000클래스(1000장)로 직접 시험하니 **68.62% vs 68.65%, McNemar p=0.639로 유의한 차이가 없다.** activation 범위는 이미 200장으로 포화한다. ORT 문서의 "수백 장이면 충분" 권고가 실측으로 뒷받침된다 — **커버리지를 늘리는 것보다 캘리브 전처리를 평가 전처리와 정확히 일치시키는 것**(4.2)이 훨씬 중요하다.

> 💡 **판단 기준 — 클리핑이 이득인가 손해인가**: 캘리브레이터 선택은 취향이 아니라 **"이 텐서의 꼬리가 잡음인가 특징인가"** 하나로 갈린다.
>
> | 꼬리의 정체 | 전형적 상황 | 유리한 캘리브레이션 |
> |---|---|---|
> | **실제 특징** (분포가 매끈하게 감쇠, 큰 값 = 강한 응답) | CNN의 post-ReLU/BN activation | **MinMax** (또는 Percentile 99.999처럼 거의 안 자르는 설정) |
> | **구조적 잡음** (소수 채널·토큰에만 몰린 극단값, 본체와 자릿수가 다름) | Transformer/LLM의 activation outlier(2단계) | **Percentile / Entropy / SmoothQuant** 계열 |
>
> 실무 절차: ① **MinMax를 baseline으로 먼저 돌린다**(가장 빠르고, 위 실측처럼 최적인 경우가 많다) → ② Percentile을 **99.999부터** 스윕해 내려오며 top-1과 상한 비율을 함께 기록 → ③ 어느 지점부터 무너지는지 보고 결정. **처음부터 99.9 같은 공격적 값으로 시작하지 말 것**(−6.2%p).

- **per-channel 대조**: MinMax를 per-tensor로 바꾸면 **−0.16%p** 나빠진다(per-ch 68.62% → per-tensor 68.46%, FP32 대비로는 −0.28%p). 방향은 **2.2.2의 이론과 일치**한다. 🔴 **정정** — 1차(큐레이션 1,000장)에서는 이 격차가 −1.20%p로 크게 보이면서 p=0.0592로 "판정 불가"였다. 50k로 재니 **크기는 1/7로 줄었지만 판정은 유의로 바뀌었다**(paired McNemar p=0.0445, 엇갈린 이미지 1,626장: per-ch만 맞음 854 / per-tensor만 맞음 772). 즉 **1차의 −1.20%p는 과대추정, 1차의 "유의하지 않음"은 검정력 부족**이었다 — 작은 셋은 효과크기와 판정을 **양쪽 다** 틀린다. 다만 50k에서도 p=0.0445는 문턱을 겨우 넘은 값이라, per-channel 이득을 **여유 있게** 보고 싶으면 **weight SQNR 쪽이 훨씬 선명하다**(21개 레이어 전부에서 per-ch가 높고, 평균 **+7.38 dB**, 최소 +3.88, 최대 +14.06 dB — 4.5.2). **"top-1로는 간신히, SQNR로는 뚜렷하게"** 가 이 대조의 요점이다: 지표를 하나만 보면 실재하는 효과를 놓치거나 크기를 오판한다.

- ⚠️ **Entropy 두 행의 간극(68.62% vs 59.29% = 9.33%p)이 오타가 아니다.** 같은 `CalibrationMethod.Entropy`인데, ORT 기본값에서는 탐색이 죽어 **MinMax와 완전히 동일한 결과**(scale 32/32 비트 일치, 산출 파일 md5까지 동일)가 나오고, 히스토그램 파라미터를 주입해 **알고리즘을 제대로 돌리면 FP32 대비 −9.45%p로 무너진다**. 원인과 대응은 4.3.1.

### 5.2 `layer_sensitivity.csv` 해석 (실측)

ResNet18 Conv/Linear **21개 전체**의 실측 결과다(torchvision IMAGENET1K_V1, per-channel symmetric INT8 — [실행 로그](../logs/stage1_quantization_log.html)). 양쪽 끝만 보인다:

```
layer                          type    out_channels  num_params  sqnr_db  cosine    sqnr_db_pertensor
--- 가장 민감한 쪽 top-5 (SQNR 낮은 순) ---
layer1.0.conv1                 Conv2d  64            36864       36.149   0.999879  29.931
layer3.0.conv2                 Conv2d  256           589824      36.302   0.999899  30.384
layer2.1.conv1                 Conv2d  128           147456      36.583   0.999891  30.722
layer4.0.conv2                 Conv2d  512           2359296     36.709   0.999982  26.812
layer2.0.conv2                 Conv2d  128           147456      36.874   0.999899  30.882
...
--- 가장 안전한 쪽 bottom-3 ---
layer3.0.downsample.0          Conv2d  256           32768       40.528   0.999956  34.723
layer2.0.downsample.0          Conv2d  128           8192        40.699   0.999957  32.015
conv1                          Conv2d  64            9408        41.149   0.999962  35.571
```

**해석 — 통념이 세 군데서 깨진다**:

- 🔴 **`conv1`이 21개 중 가장 "안전"하다(41.149 dB, 1위).** downsample 3개도 전부 안전한 쪽에 몰렸다. "첫 conv와 downsample은 채널이 적어 민감하다"는 통념과 **정반대**다. 이유는 **per-channel이기 때문**이다 — per-channel은 출력 채널마다 scale을 따로 주므로, 채널 수가 적으면 **scale을 공유하는 채널이 적어 오히려 유리하다**. 통념은 **per-tensor를 전제할 때만** 성립한다. 실제로 같은 표의 `sqnr_db_pertensor` 열을 보면 per-tensor에서는 `layer4.0.downsample.0`이 25.718 dB로 최하위권까지 떨어진다(per-channel 39.773 dB, 이득 **14.06 dB ≈ 2.3비트** — 21개 중 최대). **"어느 레이어가 민감한가"는 granularity를 정하기 전에는 답할 수 없는 질문이다.**

- 🔴 **범인 레이어가 없다.** 전 레이어가 **36.1~41.1 dB의 5 dB 폭 안**에 촘촘히 모여 있고 cosine은 전부 ≥0.9998이다. 4.5.3의 **절대 기준(`sqnr_db < 20 dB` 또는 `cosine < 0.99`)에 걸리는 레이어가 0개** — 절대 기준만 믿으면 **후보를 하나도 못 뽑는다**. 그래서 4.5.3의 **기준 1(상대 기준: 하위 5~10%)** 이 기본이어야 한다. 이 모델에서는 하위 2개인 `layer1.0.conv1`·`layer3.0.conv2`가 후보가 되지만, **최상위와 최하위 차이가 5 dB뿐이라 FP16 승격의 실익도 크지 않다**는 것까지 읽어내는 것이 옳은 해석이다(실제로 5.1에서 INT8 top-1 손실이 없었다).

- **이론 상한 대비**: 6 dB/bit 규칙의 INT8 상한 `6.02 × 8 ≈ 48.2 dB`에 대해 실측 중앙값은 **37.99 dB**로 약 10 dB 낮다. 이 간극은 결함이 아니라 **정상**이다 — 상한은 "값이 표현 범위를 꽉 채우고 균등분포"라는 이상적 가정의 값인데, 실제 weight는 0 근처에 몰린 종형 분포라 격자 상단을 거의 안 쓴다. **절대 dB를 이상치와 비교하지 말고, 같은 모델 안에서 레이어끼리 상대 비교하는 데 쓰라.**

- 이 CSV의 상위 몇 개 레이어를 **다음 단계에서 FP16으로 유지(mixed precision)** 하면, 전체를 INT8로 두는 것보다 정확도를 회복하면서 대부분의 속도 이득은 지킬 수 있다(4.5.3).

> 💡 **팁**: SQNR 절대 임계값(예: "20 dB 미만은 FP16")보다, **CSV를 SQNR 오름차순 정렬해 하위 5~10%를 후보로 보는 상대 기준**이 실전에서 안전하다(4.5.3의 기준 1). 위 실측이 그 이유의 실물 증거다.

> 🔴 **통념이 깨지는 네 번째 지점 — SQNR 순위와 실제 낙폭 순위가 일치하지 않는다**: 21개 레이어를 하나씩 fake-quant하고 매번 50,000장 top-1을 재 보니 **SQNR 최저 3개와 실제 Δtop-1 최악 3개의 교집합이 0개**였고 **Spearman ρ = −0.04**였다. 근거·해석·실무 규칙은 4.5.3 맨 앞의 붉은 박스에 정리해 두었다. 요약하면 **대리지표로 순위를 매기는 것은 검증 없이 신뢰할 수 없다.**

> ⚠️ **주의 (이 결과의 한계)**: 위는 **weight-only** 감도다. ResNet18에서 범인이 안 잡힌 것은 "이 모델의 weight가 INT8에 충분히 안전하다"는 뜻이지, 감도 분석 방법론이 무용하다는 뜻이 아니다. 실제로 정확도가 무너지는 사례(5.1의 Percentile 99.9, −6.2%p)는 **activation 클리핑**에서 왔고 이 CSV에는 전혀 나타나지 않는다. **weight-only에서 후보가 안 나오면 activation 감도(4.5-b)로 넘어가는 것이 정답**이며, 2단계 Transformer는 처음부터 activation 쪽이 주전장이다.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| INT8 top-1이 FP32보다 5%p 이상 폭락 | 전처리(mean/std/crop) 불일치, 캘리브 장수 부족 | 평가·캘리브 전처리를 **동일 함수**로 통일, 캘리브 200장↑ |
| top-1이 0~1%에 가까움 | 라벨 인덱스 매핑 오류 | 먼저 FP32 점수부터 확인(4.4 sanity check). **클래스 서브셋을 쓰면 인덱스가 0부터 다시 매겨져 반드시 터진다** → synset을 1000-way 인덱스로 환산(3.1) |
| `quantize_static`에서 `CalibrationDataReader` 관련 에러 | `get_next()`가 dict/None을 안 돌려줌 | 반환은 `{input_name: ndarray}` 또는 끝에서 `None` |
| `onnxruntime` GPU인데 CPU만 잡힘 | `onnxruntime`(CPU)와 `onnxruntime-gpu` 혼재 | CPU 패키지 제거 후 `pip install "onnxruntime-gpu<1.27"`(→ 1.23.2)만 재설치, `get_available_providers()` 확인 |
| ONNX export가 `RuntimeError` 트레이스백을 찍는데 **exit 0** | torch 2.11 dynamo 익스포터가 opset 18 미만을 못 만들어 다운컨버트 실패 후 **18로 저장** | `opset_version=18` 이상으로 올린다(또는 `dynamo=False`). 산출물은 정상이니 **exit code로 판정하지 말고 `onnx.load(...).opset_import`를 확인**(4.1) |
| 다른 머신에서 `.onnx` 로드 실패(가중치 못 찾음) | dynamo 익스포터가 **`.onnx` + `.onnx.data` 2파일**로 저장 | `.onnx.data`를 같은 디렉터리에 함께 복사(4.1) |
| Entropy 캘리브가 매우 느림/메모리 폭증 | 히스토그램 계산 비용, 캘리브 장수 과다 | 캘리브 장수 축소(100~300장). **애초에 ORT 기본 Entropy는 MinMax와 결과가 같으면서 2.8배 느리기만 하다** → MinMax로 대체(4.3.1) |
| Entropy 결과가 MinMax와 완전히 동일 | 🔴 버그가 아니라 **ORT 기본값(`num_bins=128`)이 KL 탐색 후보를 1개로 만든 것** | 정상 동작이다. 확인은 2.4.3의 `check_entropy_degenerate.py`, 배경은 4.3.1 |
| Percentile 옵션이 안 먹힘 | 키 이름/버전 차이 | ORT 1.23.2는 **`CalibPercentile`이 맞다**(확인됨). 다른 버전은 설치된 `quantize.py`의 `calib_extra_options_keys` 확인(4.3) |
| `extra_options={"num_bins": ...}`가 무시됨 | `quantize_static`의 화이트리스트 5키에 없어 캘리브레이터까지 **도달하지 못함**(에러도 안 남) | `create_calibrator` 몽키패치로 주입(4.3.1). 단 **고쳐도 이 모델에선 정확도가 더 나빠진다** |
| INT8인데 TensorRT EP에서 **FP32보다 느림** | 🔴 비대칭 uint8 QDQ를 TRT 파서가 거부 → 전 노드 폴백(**무음**) | `activation_type=QuantType.QInt8` + `ActivationSymmetric: True`로 재양자화(4.3.2). EP 로그 레벨을 올려 파싱 에러부터 확인 |
| STE 학습이 loss가 안 줄고 grad=0 | `torch.round`를 그냥 forward/backward에 씀 | `FakeQuantSTE.apply`로 backward에서 STE 마스크를 태워야 함(2.5.2). 대조군 실측은 2.5.3 |
| 캘리브레이션 간 top-1 차이가 재현이 안 됨 | 표본 수 부족으로 노이즈에 묻힘 | 평가 장수↑ 또는 paired(McNemar) 비교, Wilson CI 겹침 확인(4.4) |

### 6.1 정확도 급락 시 "범인 레이어" 특정 절차 (확대)

INT8 top-1이 목표보다 크게 떨어졌을 때, **어느 레이어가 원인인지 좁히는 체계적 절차**:

```
[0] Sanity: FP32 ONNX 점수가 그 데이터셋의 기대값인지 먼저 확인
    (ImageNet val ~69.8% / 3.1의 큐레이션 셋 ~78.5%).
    크게 낮으면 양자화 문제가 아니라 라벨·전처리 문제다 → 6장 상단.

[1] 전역 원인 배제 (효과가 큰 순서):
    - 캘리브 전처리 == 평가 전처리 인가? (4.2 함정)   ← 압도적 1순위
    - 캘리브레이터가 꼬리를 과하게 자르고 있지 않은가? (5.1)
        · Percentile을 99.9 같은 공격적 값으로 쓰고 있다면 99.999로 올려 재측정.
        · MinMax를 baseline으로 반드시 한 번 돌려 비교 (실측에선 MinMax가 최적이었다).
    - 캘리브 장수 200장↑ 인가?
    - per_channel=True 인가? (top-1 차이는 작을 수 있으나 weight SQNR 이득은 확실 — 4.5.2)
    실측 경험상 "폭락"의 대부분은 위 두 항목(전처리 / 과도한 클리핑)에서 나온다.

[2] weight-only 감도로 후보 압축 (4.5):
    - layer_sensitivity.csv 를 sqnr_db 오름차순으로 열고 하위 5~10% 를 후보로.

[3] leave-one-in-FP16 (한 레이어씩 FP16 복원):
    - 후보 레이어를 "하나만" FP16으로 되돌린 모델을 만들어 top-1 측정.
    - top-1이 가장 많이 회복되는 레이어 = 진짜 범인.
    - (반대로 leave-one-out: 한 레이어만 INT8로 두고 나머지 FP32 → 그 레이어만의
       손실 기여를 격리. 계산 많지만 가장 확실.)

[4] activation SQNR로 교차검증 (4.5-b):
    - 후보 상위 레이어만 activation 중간 텐서를 뽑아 SQNR/cosine 재측정.
    - weight-only에서 안 잡히던 activation outlier 레이어를 여기서 발견하기도.

[5] 확정된 범인들을 FP16으로 승격(mixed precision) → 목표 정확도까지 1~2개씩 확대(4.5.3).
```

> 💡 **팁**: [3]의 leave-one-in-FP16이 실무에서 가장 신뢰도 높다. weight SQNR(정적)과 실제 정확도(데이터 의존)가 항상 일치하지는 않기 때문 — SQNR은 **후보를 20개→3개로 줄이는 필터**, leave-one-in은 **최종 확인**으로 역할을 나눈다.

> 🔴 **함정**: 범인을 못 찾겠다고 무작정 FP16 레이어를 늘리면 속도 이득이 사라진다(전부 FP16이면 INT8을 한 의미가 없음). CSV 우선순위대로 **최소 개수**만 되돌리는 것이 mixed precision의 요령이다.

---

## 7) 산출물 (Deliverables)

이 단계가 끝나면 아래가 남아야 한다. 특히 **`layer_sensitivity.csv`는 2단계·3단계의 입력**이다(이 문서 4.5절이 생성).

- [ ] `resnet18_fp32.onnx` **+ `resnet18_fp32.onnx.data`** — torchvision ResNet18의 FP32 ONNX (torch 2.11 dynamo 익스포터 기준 **opset 18 / IR 10**, external data 2파일). **`.data`를 빠뜨리면 로드되지 않는다**(4.1).
- [ ] `resnet18_int8_minmax.onnx`, `resnet18_int8_entropy.onnx`, `resnet18_int8_pct999.onnx`, `resnet18_int8_minmax_pertensor.onnx` — INT8 QDQ 그래프(캘리브레이션 3종 + per-tensor 대조). 각 11.32 MB 단일 파일.
- [ ] `resnet18_int8_pct9999.onnx`, `resnet18_int8_pct99999.onnx` — percentile 스윕(꼬리를 얼마나 자르는지와 top-1의 단조 관계를 보이는 근거 — 5.1).
- [ ] **`resnet18_int8_trt_sym.onnx`** — `QInt8` + `ActivationSymmetric=True`로 만든 **TensorRT용** QDQ. **3단계의 입력이며, 비대칭 산출물로는 3단계가 성립하지 않는다**(4.3.2).
- [ ] top-1 비교 결과 (표 또는 로그) — FP32 vs INT8(캘리브레이션 전종) vs per-tensor vs 대칭 설정, **각 Wilson 95% CI + FP32 대비 McNemar p 포함**. 사용한 평가셋과 그 FP32 기준점을 반드시 함께 기록(절대값 비교 금지 — 5.1).
- [ ] **`layer_sensitivity.csv`** — 레이어별 `sqnr_db`/`cosine`/`num_params`/`out_channels`/`type`/`sqnr_db_pertensor` (SQNR 오름차순). 스키마는 4.5.2. **mixed precision 근거로 다음 단계에서 재사용**.
- [ ] QAT/STE 미니 데모 로그 — `mini_qat_demo.py`의 loss 감소·grad≠0 출력, **그리고 STE 없는 대조군의 `|grad|=0.0000` 고정 출력**(둘을 나란히 둔 것이 STE 작동의 증거 — 2.5.3).
- [ ] 논문 3편 요약 메모 — Gholami / Nagel / Jacob 각 1문단(핵심 기여 + 이 단계와의 연결).
- [ ] 수식 유도 노트 — `q=round(x/s)+z` 유도 + **오차 분해(rounding+clipping)** + symmetric/asymmetric·per-tensor/per-channel 표 + "왜 weight=per-channel sym, activation=per-tensor asym" 서술(면접 대비).

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [ONNX Runtime — Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — `quantize_static`, 캘리브레이션, QDQ/QOperator, HW별 권장 dtype (2026-07 확인).
- [ONNX Runtime — quantize.py (소스)](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py) — `quantize_static` 정확한 시그니처/기본값.
- [ONNX Runtime — calibrate.py (소스)](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py) — `CalibrationMethod`(MinMax/Entropy/Percentile/Distribution), `CalibrationDataReader`. **4.3.1의 `EntropyCalibrater` 기본값 `num_bins=128, num_quantized_bins=128`과 `PercentileCalibrater`의 `percentile=99.999`를 여기서 직접 확인할 수 있다**(2026-08 기준 main 브랜치도 동일 — 즉 ORT 1.23.2 한정 이슈가 아니다).
- [TensorRT — Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) — "TensorRT only supports symmetric uniform quantization, meaning that zeroPt=0". 4.3.2의 타깃별 dtype 분기 근거 (2026-08 확인).
- [TensorRT — Explicit Quantization (Q/DQ)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html) — QDQ ONNX를 TensorRT가 어떻게 해석·융합하는지. 3단계 선행 읽기.
- [EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images) — ImageNet 1000클래스 × 1장. val을 못 구할 때의 대안(3.1). **큐레이션 셋이라 top-1이 부풀려지는 점 주의**.
- [PyTorch Quantization 문서](https://pytorch.org/docs/stable/quantization.html) — `torch.ao.quantization`(fake-quant/QAT).
- [NVIDIA Model Optimizer (구 TensorRT Model Optimizer)](https://github.com/NVIDIA/Model-Optimizer) — `modelopt.torch.quantization`, `mtq.quantize()`. PTQ/QAT/MSE 캘리브 실전 툴(2단계·3단계에서 사용).
- [torchvision models](https://pytorch.org/vision/stable/models.html) — ResNet18 사전학습 가중치.

### 논문
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient Neural Network Inference*, arXiv:[2103.13630](https://arxiv.org/abs/2103.13630) — 양자화 전반 분류(PTQ/QAT, uniform/non-uniform, per-tensor/per-channel)의 지도. 용어·개념 정리에 필독.
- Nagel et al. (2021, Qualcomm AI Research), *A White Paper on Neural Network Quantization*, arXiv:[2106.08295](https://arxiv.org/abs/2106.08295) — PTQ 고급 기법(**AdaRound**, **CLE(cross-layer equalization)**, **bias correction**)과 QAT 실무 레시피. 이 단계 PTQ 정확도 회복의 근거.
- Jacob et al. (2018, CVPR; arXiv 2017), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, arXiv:[1712.05877](https://arxiv.org/abs/1712.05877) — 정수 전용 추론 스킴의 원조. symmetric weight(zero-point=0)로 정수 MAC이 단순해지는 이유(2.2.1 유도)의 출처.
- Esser et al. (2020, ICLR), *Learned Step Size Quantization (LSQ)*, arXiv:[1902.08153](https://arxiv.org/abs/1902.08153) — scale(step size)을 학습 파라미터로 두는 QAT. 2.5.5의 출처. 저비트(2~4bit)에서 특히 강력.

> ⚠️ **확인 필요**: Jacob et al.은 arXiv 등록이 2017-12(ID 1712.05877)이고 학회 발표는 **CVPR 2018**이다. 인용 시 맥락에 맞게 연도를 표기.

> 💡 **참고(캘리브레이션 KL의 원전)**: 2.4.3의 KL 알고리즘은 Szymon Migacz, *8-bit Inference with TensorRT* (NVIDIA GTC 2017) 발표에서 유래한다. NVIDIA 슬라이드가 정본이며, 2048 bin·threshold 128~2048 탐색·`T=(m+0.5)·bin_width`가 이 발표에서 제시되었다.

---

## 9) 다음 단계

- **이전**: [0.5단계 — 배포 사다리](02_deployment_ladder.md)
- **다음**: [2단계 — Transformer 양자화](04_transformer_quantization.md) — ViT/Transformer의 activation outlier 문제와 SmoothQuant, FQ-ViT/PTQ4ViT/RepQ-ViT. **이 단계의 `layer_sensitivity.csv` 방법론(4.5)을 Transformer에 확장**해 어떤 레이어를 FP16으로 남길지 결정한다. 특히 2.2.2의 "activation per-channel을 왜 못 하나"가 Transformer의 토큰별 outlier에서 어떻게 문제가 되는지로 이어진다.
