# 10. 실전 함정 5개 (+ 측정의 함정 1개) — 양자화가 조용히 무너지는 지점들

> 원본 가이드 매핑: "함정 5개" · 예상 소요: 반나절(정독) + 실습 시 하루 · 선행 조건: [03](03_quantization_theory.md)~[06](06_multi_soc.md) 개념 숙지 권장

이 문서는 앞선 단계들([01](01_environment_setup.md)~[08](08_capstone.md))과 [12주 로드맵](09_roadmap.md)을 관통하는 **5대 실패 패턴**을 정리한다. 각 함정은 **증상 → 원인(수치·메커니즘) → 예방 → 디버깅 절차 → 재현 코드** 순으로, 재현·검증 가능한 코드와 함께 다룬다. 앞에 놓인 **함정 0**은 원본 가이드의 5개에는 없지만, 1단계를 두 번(큐레이션 1,000장 → 진짜 50,000장) 돌려 보고 **추가한 것**이다 — 나머지 다섯 개를 "진단"하려면 먼저 측정을 믿을 수 있어야 하기 때문이다.

> 💡 이 문서의 함정 0·1-b·2-b·4의 실측 사례는 **이 스터디의 1단계 실습에서 실제로 밟은 것**이다. 측정 환경은 RTX 3060 / ResNet18 / batch=1 p50 / ORT 1.23.2 + TensorRT 10.16.x이며, 로그 원문은 두 개다:
> - [1단계 실습 로그(1차, 큐레이션 1,000장)](../logs/stage1_quantization_log.html) — 메커니즘 규명(Entropy 퇴화·2×2 절제)의 원본
> - [1단계 재실행 로그(2차, ImageNet val 50,000장 전량)](../logs/stage1_real_imagenet_log.html) · [보고서](../logs/stage1_real_imagenet_report.html) — **본문의 모든 top-1 절대값·유의성은 이쪽 값**이다
>
> 1차의 절대 top-1은 클래스당 1장 큐레이션 셋이라 **평균 +9.77%p 부풀려져 있었다**(함정 0). 아래 본문은 그 수치를 전량 50k 값으로 전부 교체한 상태다.

> 💡 팁: 이 다섯 개는 "지식 부족"이 아니라 "무심코"에서 온다. 다 알아도 매번 당한다. 그래서 마지막의 [실무 체크리스트](#실무-체크리스트-양자화-전후-반드시-확인)를 프로젝트마다 복사해 쓰길 권한다.

> ⚠️ 정본 버전 스택(2026-07 기준): **CUDA 12.8 / onnx 1.18.0 (IR 11) / onnxruntime-gpu 1.23.2 / TensorRT 10.16.x LTS / ExecuTorch 1.3.x**. 아래 코드·명령은 이 조합 기준이다. ONNX export는 이 스택의 IR 상한 때문에 **opset ≤ 23**을 지킨다([0단계 2절](01_environment_setup.md)). TensorRT는 10.x부터 plugin이 `IPluginV3`로 통일됐고(함정 5), QNN EP는 dynamic shape·Loop/If를 지원하지 않는다(함정 3·4).

---

## 0) 이 단계에서 무엇을·왜 하는가

양자화가 실패하는 방식은 대개 **요란하지 않다.** 컴파일도 되고, export도 "성공"이라 뜨고, 에러도 없다. 그런데 특정 상황(야간·터널·역광)에서만 정확도가 무너지거나, 가속기에 올렸는데 오히려 느려진다.

이 문서의 목적은 그 "조용한 실패"들을 **미리 이름 붙여** 알아보게 하는 것이다. 이름이 있으면 디버깅이 빨라진다. 각 함정은 로드맵의 특정 주차·산출물과 직접 연결되므로, [09_roadmap.md](09_roadmap.md)를 돌리다 막히면 해당 함정으로 바로 온다.

**왜 "디버깅 절차"까지 적는가.** 함정을 아는 것과 잡는 것은 다르다. "전처리가 문제일 수 있다"는 지식은 흔하지만, **어떤 순서로 무엇을 실행해 그것을 증명하는지**가 실력이다. 그래서 각 함정마다 "증상을 보면 → 이 명령을 이 순서로 → 이 출력이 나오면 확정"이라는 **재현 가능한 절차**를 붙였다. 이 절차 자체가 면접에서 "그 버그 어떻게 잡았어요?"에 대한 답이 된다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] 5대 함정의 **증상**을 보고 원인을 추정할 수 있다.
- [ ] **내 평가셋이 결론을 낼 만큼 큰지**를 신뢰구간·검정력으로 먼저 판정할 수 있다(함정 0).
- [ ] 캘리브레이션 대표성·전처리 일치를 **코드로 검증**할 수 있다.
- [ ] **전처리 불일치의 대가와 양자화의 대가를 같은 단위(%p)로 비교**해 어느 쪽을 먼저 고칠지 정할 수 있다(함정 2-b).
- [ ] "export 성공"과 "칩 동작"을 구분하고, offload 비율을 근거로 fallback 여부를 판단할 수 있다.
- [ ] **"EP가 목록에 있다/결과가 맞다"와 "그 EP가 실제로 그래프를 실행했다"를 구분**하고, 로그와 **FP32 대비 + 하위 EP 대비** latency로 판정할 수 있다(함정 4 실측 사례).
- [ ] **라이브러리 기본값이 내가 고른 알고리즘을 무력화했는지**를 산출물 비교로 자가진단할 수 있다(함정 1-b).
- [ ] `polygraphy inspect capability`로 백엔드 미지원 op를 특정할 수 있다.
- [ ] 양자화 전/후 체크리스트를 프로젝트에 적용할 수 있다.

---

## 함정 0 — 평가셋이 작으면, 나머지 함정을 진단할 수 없다

> 관련 단계: [03_quantization_theory.md](03_quantization_theory.md) 4.4(평가 스크립트), [07_infrastructure.md](07_infrastructure.md)(회귀 게이트), [09_roadmap.md](09_roadmap.md) 1~2주
> 이 함정은 원본 가이드의 5개에 없다. **1단계를 1,000장으로 한 번, 50,000장으로 다시 한 번 돌려 두 결과를 맞춰 보고 추가했다.**

**증상**
- 에러도 경고도 없다. **숫자가 나온다.** 그 숫자로 "A 방법이 B보다 좋다"는 결론까지 낸다.
- 평가셋을 늘리거나 바꾸면 **결론이 바뀐다**(부호까지 뒤집힌다).
- FP32 top-1이 논문/모델 카드 공식값보다 **눈에 띄게 높다**(예: ResNet18 69.76%인데 78.5%가 나온다). "내 파이프라인이 좋은가 보다" 하고 넘어간다.
- 0.1~0.5%p 차이를 근거로 설정을 고정했는데, 다른 셋에서 재보면 재현되지 않는다.

**원인 (수치·메커니즘)**
정확도는 **표본에서 추정한 값**이고 추정에는 오차가 있다. `n`장에서 잰 top-1 `p̂`의 표준오차는 `√(p̂(1−p̂)/n)`이다. `p̂≈0.7`일 때(CI 폭은 Wilson 기준):

| n | 표준오차 | 95% CI 폭 | 이 셋의 **절대값**이 갖는 의미 |
|---|---------|----------|-----------------------------|
| 1,000 | 1.45%p | ±2.84%p | "약 68~74%" 정도. 모델 카드 비교 불가 |
| 10,000 | 0.46%p | ±0.90%p | 1%p 단위 비교 가능 |
| 50,000 | 0.20%p | ±0.40%p | 0.5%p 단위까지 |

양자화 손실은 보통 **0.1~1%p** 대다. 즉 **1,000장 셋의 절대값은 양자화 손실보다 오차가 크다.** 여기에 두 번째 문제가 겹친다 — 작은 셋을 "클래스당 1장" 같은 방식으로 직접 고르면 **표본이 쉬운 사진 쪽으로 치우쳐** 절대값이 통째로 부풀려진다.

**두 모델의 차이는 사정이 다르다.** 같은 이미지로 둘을 재는 **짝지어진 설계**(McNemar)는 공통 정답이 상쇄되므로 위 표보다 훨씬 작은 차이를 잡는다. 실측으로 잰 실질 분해능:

| 평가셋 | 실측 사례 | 판정 | 결론 |
|--------|----------|------|------|
| 1,000장 | Δ=**+0.40%p** (MinMax vs FP32) | p=0.48 | 0.4%p를 볼 검정력이 없다 |
| 50,000장 | Δ=**−0.12%p** (같은 모델) | p=0.061 | 0.12%p는 여전히 못 잡는다 |
| 50,000장 | Δ=**−0.16%p** (Pct 99.999) | **p=0.007** | 여기서부터 잡힌다 |
| 50,000장 | Δ=**−0.29%p** (대칭 강제) | **p=9.2e-5** | 확실히 잡힌다 |

→ **ImageNet val 50,000장 + 짝지어진 검정의 실질 분해능은 0.15%p 근처**다. 그보다 작은 차이는 "없다"가 아니라 **"이 셋으로는 못 본다"**로 보고해야 한다.

**실측 — 같은 실험, 1,000장 vs 50,000장**

> 측정: ResNet18 / ImageNet val / 13개 모델(FP32 + PTQ 변형 12개)을 **동일 전처리 캐시**로 평가. 1차는 클래스당 1장(1,000장), 2차는 val 전량(50,000장). 원문: [재실행 보고서 3~6장](../logs/stage1_real_imagenet_report.html)

**① 절대값 — 부풀림이 상수가 아니라서 보정조차 못 한다**

| 모델 | 큐레이션 1,000장 | 그때의 95% CI | 진짜 50,000장 | 부풀림 |
|------|-----------------|--------------|--------------|--------|
| FP32 | 78.50% | [75.8, 80.9] | 68.74% | +9.76%p |
| INT8 MinMax | 78.90% | [76.3, 81.3] | 68.62% | +10.28%p |
| INT8 weight per-tensor | 77.70% | [75.0, 80.2] | 68.46% | +9.24%p |
| INT8 Entropy(정상화) | 67.70% | [64.7, 70.5] | 59.29% | +8.41%p |

부풀림 평균 **+9.77%p**, 범위 **[+8.41, +10.39]** — 폭이 **2%p**다. 모든 모델이 똑같이 부풀려진다면 "상수를 빼면 된다"고 할 수 있지만, **모델마다 다르게 부풀려지므로 Δ(모델 간 차이)까지 오염된다.** 그리고 위 CI를 보라 — **1,000장에서 계산한 95% CI가 진짜 값 68.74%를 포함하지 못한다.** CI는 "표본 오차"만 담는다. 표본 **선택**이 편향돼 있으면 CI는 그 편향을 알려주지 않는다.

**② Δ의 부호 — 13건 중 2건이 뒤집혔다**

| 모델 | 큐레이션 Δ vs FP32 | 진짜 Δ vs FP32 | 판정 |
|------|-------------------|---------------|------|
| INT8 MinMax | **+0.40%p** ("INT8이 FP32를 이겼다") | **−0.12%p** | 🔴 부호 반전 |
| INT8 Entropy(기본값) | **+0.40%p** | **−0.12%p** | 🔴 부호 반전 |
| INT8 Percentile 99.999 | +0.10%p | −0.16%p | 🔴 부호 반전 |
| INT8 weight per-tensor | −0.80%p | −0.28%p | 방향 유지 |
| INT8 Percentile 99.9 | −6.20%p | −6.83%p | 방향 유지 |

큰 차이(≥1%p)는 방향이 살아남고, **작은 차이는 부호가 랜덤에 가깝다.** "INT8이 FP32보다 좋게 나왔다"는 흔한 관측은 대개 이것이다.

**③ 유의성 — 5/13건이 뒤집혔고, 전부 같은 방향이다**

| 비교 | 큐레이션 p | 진짜 p | 판정 변화 |
|------|-----------|--------|-----------|
| FP32 → MinMax | 0.480 | 0.061 | 동일(유의하지 않음) |
| FP32 → weight per-tensor | 0.230 | **3.4e-4** | 🔴 유의하지 않음 → **유의** |
| FP32 → Percentile 99.99 | 0.540 | **4.1e-9** | 🔴 유의하지 않음 → **유의** |
| FP32 → Entropy(정상화) | 6.5e-17 | <1e-300 | 동일(유의) |

뒤집힌 5건이 **전부 "유의하지 않음 → 유의"** 방향이다. 우연이 아니라 표본이 50배 늘어난 당연한 결과다. 즉 **작은 셋의 "p>0.05"는 "차이가 없다"가 아니라 "이 셋으로는 볼 수 없다"**는 뜻이다. 이걸 "차이 없음"으로 읽으면, 실재하는 0.3%p 손실을 무료라고 판단하고 배포한다.

**예방**
- **표본 수를 먼저 정하라.** 독립표본 기준 상한은 `n ≳ 2·p(1−p)·(1.96/d)²`이다(`p≈0.7`, `d=0.3%p` → **약 18만 장**). 이 숫자에 겁먹을 필요는 없다 — 같은 이미지를 쓰는 짝지어진 설계에서는 위 실측대로 **50,000장이 0.29%p를 p=9.2e-5로 잡는다.** 요점은 **"측정 전에 잡을 차이를 정하고, 그게 가능한 설계인지 확인한다"**는 것이다. 못 잡는 크기라면 "그 차이는 못 잰다"를 결론으로 받아들여라.
- 셋을 줄여야 하면 **층화 표본**(클래스별 비율 유지 + 랜덤)으로 뽑고, `items[:n]` 같은 **정렬 순서 자르기를 절대 쓰지 마라**(클래스 몇 개만 평가된다 — [03 4.4](03_quantization_theory.md)).
- **절대값과 상대값을 분리해 보고하라.** 작은 셋에서도 상대 Δ는 (크면) 쓸 수 있다. 절대값은 큐레이션 셋에서 나온 즉시 폐기한다.
- 결론에 **CI와 p를 항상 같이 적는다.** "68.62%"가 아니라 "68.62% [68.21, 69.03], Δ=−0.12%p, McNemar p=0.061".
- 짝지어진 비교에는 **McNemar**를 쓴다. 두 모델을 같은 이미지로 평가하는 이상, 독립표본 검정은 검정력을 버리는 것이다.

**디버깅 절차**
1. FP32 top-1을 **공식 발표값과 비교**한다. 1%p 이상 높으면 표본 편향 또는 전처리 문제(함정 2)다. **낮으면 전처리, 높으면 표본**을 먼저 의심하라.
2. 잡으려는 차이 `d`를 정하고 위 공식으로 필요한 `n`을 계산한다. 지금 셋이 그보다 작으면 **결론을 내지 말고 셋을 늘린다.**
3. 늘릴 수 없으면 결론의 격을 낮춘다: "A가 B보다 좋다" → "이 셋에서는 구별되지 않는다".

**재현 코드 — 내 평가셋이 결론을 낼 수 있는지 판정**

```python
# eval_power_check.py
# 목적: (1) 지금 top-1의 신뢰구간, (2) 목표 차이를 잡는 데 필요한 n, (3) 짝지어진 두 모델의 McNemar
import math
import numpy as np

def wilson(k, n, z=1.96):
    """Wilson 95% CI — 정규근사보다 작은 n·극단 p에서 안전하다."""
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def n_needed(p, d_pp, z=1.96):
    """d_pp(%p) 차이를 95% 신뢰로 구별하는 데 필요한 대략적 표본 수(보수적·독립표본 기준)."""
    d = d_pp / 100.0
    return math.ceil(2 * p * (1 - p) * (z / d) ** 2)

def mcnemar(correct_a, correct_b):
    """짝지어진 정오 배열(bool) → (b, c, p). b=A만 맞힘, c=B만 맞힘."""
    a, b_arr = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int(np.sum(a & ~b_arr)); c = int(np.sum(~a & b_arr))
    if b + c == 0: return b, c, 1.0
    try:                                       # 정확검정(이항). scipy 있으면 이쪽.
        from scipy.stats import binomtest
        return b, c, binomtest(b, b + c, 0.5).pvalue
    except ImportError:                        # 없으면 연속성 보정 χ²
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        return b, c, math.erfc(math.sqrt(chi2 / 2))

# ── 사용 예 ────────────────────────────────────────────────
n, k = 1000, 785                               # 지금 평가셋: 1000장 중 785장 정답
lo, hi = wilson(k, n)
print(f"top-1 = {k/n:.2%}  95% CI [{lo:.2%}, {hi:.2%}]  (폭 ±{(hi-lo)/2*100:.2f}%p)")
for d in (1.0, 0.5, 0.3, 0.1):
    print(f"  {d:>4.1f}%p 차이를 잡으려면 n ≳ {n_needed(k/n, d):,}장")

# 두 모델의 짝지어진 정오 배열(같은 이미지 순서!)을 넣는다
# b, c, p = mcnemar(fp32_correct, int8_correct)
# print(f"FP32만 맞힘 {b}장 / INT8만 맞힘 {c}장 / McNemar p={p:.4g}")
```

기대 출력(1단계 1차 실행의 실제 수치를 넣은 것):
```
top-1 = 78.50%  95% CI [75.85%, 80.93%]  (폭 ±2.54%p)
   1.0%p 차이를 잡으려면 n ≳ 12,968장
   0.5%p 차이를 잡으려면 n ≳ 51,870장
   0.3%p 차이를 잡으려면 n ≳ 144,082장
   0.1%p 차이를 잡으려면 n ≳ 1,296,733장
```

> ⚠️ 위 `n ≳` 값은 **독립표본 기준의 보수적 상한**이다. 짝지어진 설계는 훨씬 적은 n으로 같은 차이를 잡으므로(실측 50,000장 → 0.29%p, p=9.2e-5) 이 숫자를 "이만큼 없으면 아무것도 못 한다"로 읽지 마라. **`mcnemar()`가 실제 판정 도구이고, `n_needed()`는 "이 차이는 애초에 무리인가"를 가늠하는 눈금**이다. 확실한 건 하나다 — 78.50% ±2.54%p는 **모델 카드의 69.76%와 비교할 수 없는 값**이다.

> 🔴 함정: **"작은 셋에서 재보고 좋으면 큰 셋으로 확인한다"는 순서가 위험하다.** 작은 셋이 A를 고르게 만들면 큰 셋 확인은 A만 하게 되고, 진짜 승자였던 B는 후보에서 이미 탈락해 있다. 실측에서 "MinMax가 FP32를 이겼다(+0.40%p)"를 믿었다면 **"양자화가 무료다"라는 잘못된 전제로 1단계 전체를 통과**했을 것이다. **스크리닝은 작게 해도 되지만, 채택 판정은 반드시 결론 가능한 크기에서 한다.**

> 💡 팁: 평가셋 전체를 매번 돌리는 게 부담이면 **전처리 결과를 무손실 캐시**해 둔다(50,000장 × 3×224×224 float16 ≈ 15 GB, uint8 텐서면 7.5 GB). 1단계 재실행에서 캐시 한 번(약 12분)으로 13개 모델 평가를 반복했다. 매번 JPEG 디코드+resize를 다시 하면 평가가 느려서 "작은 셋으로 때우자"는 유혹에 진다 — **캐시가 곧 함정 0의 예방책**이다.

---

## 함정 1 — 캘리브레이션 데이터가 전부다

> 관련 단계: [03_quantization_theory.md](03_quantization_theory.md)(PTQ·calibration), [09_roadmap.md](09_roadmap.md) 1~2주(`layer_sensitivity.csv`)

**증상**
- 검증셋 전체 정확도는 멀쩡한데, **야간·터널·역광 등 특정 조건**에서만 급락.
- INT8이 FP32보다 특정 클래스/상황에서만 크게 나쁨.
- 캘리브 셋을 바꿔 재양자화하면 정확도가 눈에 띄게 출렁인다(=scale이 데이터에 과민).

**원인 (수치·메커니즘)**
Static PTQ는 **캘리브레이션 데이터로 activation의 min/max(또는 분포)를 추정**해 scale/zero-point를 정한다. INT8 대칭 양자화에서 `scale = max(|x|) / 127`이므로, 캘리브 셋이 관측한 `max(|x|)`가 곧 표현 가능한 상한이다. 캘리브 셋에 없는 밝기·대비·분포는 그 상한을 넘겨 **clipping**되고, 넘어간 값은 전부 `±127`로 뭉개진다.

정량적으로: 실제 입력의 극단값 `x_true`가 캘리브 상한 `T`를 초과하면, 그 activation의 상대 오차는 대략 `(x_true − T) / x_true`까지 벌어진다. 예를 들어 캘리브가 `T=4.0`을 봤는데 야간 입력이 `x_true=6.2`를 만들면, 그 지점은 `(6.2−4.0)/6.2 ≈ 35%`가 통째로 잘려나간다. 이 오차가 레이어를 타고 누적되면 특정 조건에서만 정확도가 붕괴한다. 캘리브 셋이 곧 "양자화가 아는 세상의 전부"다.

> 벤더/문서 공통 권고: 캘리브 데이터는 배포 환경 분포를 대표해야 하며, 같은 도메인에서 뽑는 것이 좋다. 이미지 수가 많을수록(수백 장) INT8 정확도가 안정적이다. ([ONNX Runtime 양자화 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), NVIDIA INT8 가이드)

**예방**
- 캘리브 셋을 **운영 데이터 분포로 층화 샘플링**(주간/야간/터널/역광/우천 등 조건별 최소 표본 확보).
- 너무 적은 표본 금지. 이미지 분류 기준 수백 장, batch size 1은 피한다(정확도 저하·시간 증가 사례 다수).
- 캘리브 셋을 **버전 관리**하고, 어떤 조건이 몇 %인지 매니페스트로 남긴다.
- 캘리브 방법(MinMax / entropy(KL) / percentile)은 **모델의 activation 분포를 보고** 고른다. 극단값이 *잡음*이면 clipping(KL·percentile)이 평균 오차를 줄이지만, 극단값이 *실제 특징*이면 자르는 만큼 그대로 손실이다 — 실측에서 ResNet18(post-ReLU activation)은 MinMax가 최적이었고 percentile 99.9는 **−6.2%p**였다([1단계 실습 로그 5장](../logs/stage1_quantization_log.html#s5)). 기본값으로 clipping 계열을 깔지 말고, 바꿨으면 **아래 함정 1-b로 실제로 바뀌었는지부터** 확인하라.

**디버깅 절차**
1. **조건별로 정확도를 쪼갠다.** "전체 76%"가 아니라 "주간 78% / 야간 61%"를 본다. 조건별로 쪼개지 않으면 이 함정은 절대 안 보인다.
2. 급락 조건이 특정되면, 그 조건 입력의 activation range가 캘리브 range를 넘는지 `calib_coverage.py`로 정량 확인.
3. 넘으면 → 캘리브 셋에 그 조건 표본을 추가하고 재양자화. 안 넘는데도 낮으면 → 함정 2(전처리) 또는 레이어 민감도([09_roadmap.md](09_roadmap.md) 1~2주 스윕)로 이동.

**재현 코드 — 캘리브 셋 대표성 정량 점검**

캘리브 셋과 실제 검증셋의 activation 통계가 실제로 겹치는지 히스토그램/분위수로 비교한다. 겹치지 않으면 그 구간이 양자화 사각지대다. 아래는 **레이어별로** 커버리지를 판정하고, 초과 비율까지 숫자로 뽑도록 확장한 버전이다.

```python
# calib_coverage.py
# 목적: 캘리브 셋 activation 분포가 검증셋 분포를 실제로 커버하는지 레이어별로 정량 점검
# 실행: python3 calib_coverage.py  (Ubuntu 22.04 + RTX, 정본: torch/torchvision, CUDA 12.8)
import torch, numpy as np
from torchvision import models

model = models.resnet50(weights="IMAGENET1K_V2").cuda().eval()

# 관심 레이어 여러 개를 훅으로 수집(첫 conv 뒤 + 중간 블록 + 후반 블록)
acts = {}
def hook(name):
    def fn(_m, _i, out):
        acts.setdefault(name, []).append(out.detach().float().flatten().cpu())
    return fn
model.layer1.register_forward_hook(hook("layer1"))   # 초반
model.layer2.register_forward_hook(hook("layer2"))   # 중반
model.layer4.register_forward_hook(hook("layer4"))   # 후반

@torch.no_grad()
def collect(loader):
    acts.clear()
    for x in loader:                 # loader: 이미 "학습과 동일한" 전처리를 거친 텐서 배치 (함정 2 참고)
        model(x.cuda())
    return {k: torch.cat(v).numpy() for k, v in acts.items()}

def summarize(a, tag):
    qs = np.percentile(a, [0, 0.1, 1, 50, 99, 99.9, 100])
    print(f"[{tag}] min={qs[0]:.3f} p0.1={qs[1]:.3f} p1={qs[2]:.3f} "
          f"med={qs[3]:.3f} p99={qs[4]:.3f} p99.9={qs[5]:.3f} max={qs[6]:.3f}")
    return qs

# calib_loader = 캘리브 셋, val_loader = 실제 검증셋(야간/터널 포함)
calib = collect(calib_loader)
val   = collect(val_loader)

for layer in calib:
    cs = summarize(calib[layer], f"{layer}:calib")
    vs = summarize(val[layer],   f"{layer}:val")
    # 커버리지 판정: 검증셋 값이 캘리브 [min,max]를 벗어나는 비율(clipping 후보)
    lo, hi = cs[0], cs[6]
    over = np.mean((val[layer] < lo) | (val[layer] > hi)) * 100
    verdict = "⚠️ 사각지대" if over > 0.5 else "✅ 커버됨"
    print(f"  → {layer}: 검증셋의 {over:.2f}% 가 캘리브 range 밖  {verdict}\n")
```

기대 출력(문제 있는 경우 예):
```
[layer1:calib] min=-0.512 ... p99.9=3.104 max=3.980
[layer1:val]   min=-0.740 ... p99.9=4.510 max=6.210
  → layer1: 검증셋의 1.83% 가 캘리브 range 밖  ⚠️ 사각지대

[layer4:calib] min=-2.10 ... max=5.44
[layer4:val]   min=-2.31 ... max=5.60
  → layer4: 검증셋의 0.12% 가 캘리브 range 밖  ✅ 커버됨
```
→ `layer1`처럼 초반 레이어에서 초과 비율이 높으면, 그 조건(야간 등) 표본을 캘리브 셋에 넣고 재양자화한다.

> 🔴 함정: "검증셋 정확도 1개 숫자"만 보면 통과한다. **조건별로 쪼갠 정확도**(야간 acc, 역광 acc)를 따로 봐야 이 함정이 드러난다. `over > 0.5%`는 경험칙 임계이니, 정확도 급락과 함께 보라.

### 함정 1-b — 캘리브 데이터는 맞는데, 고른 **캘리브 방법이 실행되지 않은** 경우

> 같은 "scale이 잘못 잡힌다"는 결과지만 **축이 다르다.** 함정 1은 *데이터*가 범위를 못 봐서 생기고, 1-b는 *라이브러리 기본값*이 내가 고른 알고리즘을 무력화해서 생긴다. 데이터를 아무리 손봐도 1-b는 안 고쳐진다.

**증상**
- `CalibrationMethod`를 바꿔 재양자화했는데 정확도가 **소수점까지 똑같다**(예측 불일치 0장).
- 그런데 양자화 **시간은 확실히 늘었다** → "무거운 걸 하고 있구나"라고 착각하게 된다. 이게 이 함정이 안 잡히는 이유다.
- 튜닝 노트에 "entropy 시도 → 개선 없음"이 남고 그 방법이 후보에서 탈락한다. 실제로는 **시도된 적이 없다.**

**원인 — ORT `CalibrationMethod.Entropy`의 탐색 공간 크기가 1**

`EntropyCalibrater`의 기본값은 `num_bins=128`, `num_quantized_bins=128`이다. KL 임계 탐색은 "히스토그램을 어디서 자를까"의 후보들을 훑어 비교하는 알고리즘인데, `get_entropy_threshold`에서 후보 배열이 이렇게 만들어진다.

```python
# onnxruntime/quantization/calibrate.py — get_entropy_threshold() (ORT 1.23.2)
zero_bin_index         = num_bins // 2              # 128 // 2 = 64
num_half_quantized_bin = num_quantized_bins // 2    # 128 // 2 = 64
kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)   # = np.zeros(1)
```

`num_bins == num_quantized_bins`이면 후보가 **정확히 1개**, 그것도 "자르지 않음(= 전체 범위)"뿐이다. 그건 정의상 MinMax다. 즉 **KL 캘리브레이션은 실행되는 시늉만 하고 MinMax와 같은 답을 낸다.**

> 실측(RTX 3060 / ResNet18 / ORT 1.23.2, 캘리브 200장): activation `scale`·`zero_point`가 **32/32 텐서 전부 1e-12 이내로 동일**, 평가 1000장에서 **예측 불일치 0장**. 그런데 양자화 시간은 **9.0s → 25.1s(2.8배)**. ([1단계 실습 로그 4장](../logs/stage1_quantization_log.html#s4))

**게다가 고치는 경로가 막혀 있다.** `num_bins`를 키우면 되지만, `quantize_static`이 캘리브레이터로 넘기는 `extra_options` 키는 아래 5개 화이트리스트뿐이라 **`num_bins`/`num_quantized_bins`는 전달 자체가 불가능**하다(`create_calibrator`는 받는데 `quantize_static`이 안 넘긴다).

```
CalibTensorRangeSymmetric · CalibMovingAverage · CalibMovingAverageConstant
CalibMaxIntermediateOutputs · CalibPercentile
```

> ⚠️ 이 동작은 **2026-08 기준 onnxruntime `main`에서도 그대로**다 — `EntropyCalibrater`의 기본값은 여전히 `num_bins=128, num_quantized_bins=128`이고, `quantize.py`의 `calib_extra_options_keys`도 위 5개뿐이다. 기본값이 128인 것에 대한 이슈([microsoft/onnxruntime#9597](https://github.com/microsoft/onnxruntime/issues/9597))는 **닫혔지만 기본값은 바뀌지 않았다.** 즉 "ORT를 올리면 해결"이 아니다. ([calibrate.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py), [quantize.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py))

**자가진단 — 옵션을 바꾼 두 산출물의 activation scale을 비교한다**

```python
# compare_scales.py
# 목적: 캘리브 옵션을 바꾼 두 QDQ 모델의 activation scale/zero_point가 실제로 달라졌는지 확인
# 실행: python3 compare_scales.py int8_minmax.onnx int8_entropy.onnx
import sys
import numpy as np, onnx
from onnx import numpy_helper

def act_scales(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in init:
            continue                      # 입력이 상수면 weight QDQ — activation만 본다
        s = init.get(n.input[1])
        if s is None:
            continue
        z = init.get(n.input[2]) if len(n.input) > 2 else np.zeros(1, np.uint8)
        out[n.input[0]] = (np.asarray(s).ravel(), np.asarray(z).ravel())   # 양자화 대상 텐서명으로 키
    return out

a, b = act_scales(sys.argv[1]), act_scales(sys.argv[2])
common = sorted(set(a) & set(b))
same = sum(1 for k in common
           if np.allclose(a[k][0], b[k][0], rtol=0, atol=1e-12)
           and np.array_equal(a[k][1], b[k][1]))
print(f"공통 activation 텐서 {len(common)}개 중 scale·zp가 동일한 것: {same}/{len(common)}")
print("🔴 옵션이 안 먹었다(알고리즘 퇴화 또는 인자 미전달)" if same == len(common)
      else "✅ 옵션이 실제로 반영됐다")
```

기대 출력(퇴화한 경우):
```
공통 activation 텐서 32개 중 scale·zp가 동일한 것: 32/32
🔴 옵션이 안 먹었다(알고리즘 퇴화 또는 인자 미전달)
```

**우회 (정말 KL을 돌려야 한다면)** — `quantize_static`이 걸러버리는 인자를, 그 아래의 `create_calibrator`를 몽키패치해 `extra_options`에 직접 꽂는다. 두 가지를 틀리기 쉽다: (1) `num_bins`는 `create_calibrator`의 **직접 kwarg가 아니라 `extra_options`의 소문자 키**다. (2) `import onnxruntime.quantization.quantize as QZ`는 패키지 `__init__`이 re-export한 **동명의 함수**를 잡아버리므로, `importlib`로 **모듈을** 가져와야 한다.

```python
import importlib
QZ = importlib.import_module("onnxruntime.quantization.quantize")   # 모듈을 직접 (함수 아님)
_orig = QZ.create_calibrator

def _patched(*a, **kw):
    eo = dict(kw.get("extra_options") or {})
    eo["num_bins"] = 2048              # 소문자 키 — 화이트리스트를 우회해 직접 주입
    eo["num_quantized_bins"] = 128     # 탐색 후보 = 1024 − 64 + 1 = 961개
    kw["extra_options"] = eo
    return _orig(*a, **kw)

QZ.create_calibrator = _patched
# 이후 quantize_static(...) 호출 → 로그에 "Number of histogram bins : 2048"이 찍히면 성공
# 끝나면 QZ.create_calibrator = _orig 로 되돌린다
```

> ⚠️ 주의: **"고쳤더니 좋아진다"가 아니다.** 실측에서 탐색을 정상화(2048/128)한 Entropy는 오히려 **−10.8%p**로, 퇴화 상태보다 **더 나빴다**(시간도 51.0s). ResNet18의 post-ReLU 극단값은 잡음이 아니라 실제 특징이라, KL이 고르는 임계가 이 모델에선 해롭다. 퇴화를 고치는 것과 정확도가 오르는 것은 별개다.

> 🔴 함정(일반화): **옵션을 바꿨는데 산출물이 비트 단위로 같으면, 그 옵션은 안 먹은 것이다.** 캘리브 방법에 국한된 얘기가 아니다 — `per_channel`, `reduce_range`, opt level, EP provider option 모두 같은 방식으로 조용히 무시될 수 있다. "느려졌으니 뭔가 하고 있다"는 **증거가 아니다**. 옵션을 바꿀 때마다 *바뀌었어야 할 산출물*(scale, initializer 개수, 노드 수, 엔진 크기)을 한 줄이라도 비교하는 습관을 들여라.

---

## 함정 2 — 전처리 불일치 (mean/std가 어긋난다)

> 관련 단계: [03_quantization_theory.md](03_quantization_theory.md), [05_tensorrt.md](05_tensorrt.md)(calibrator 입력), [09_roadmap.md](09_roadmap.md) 12주(전처리 단일 소스화)

**증상**
- 에러 하나 없이 **정확도만 조용히 죽는다**(예: top-1이 10~30%p 통째로 하락).
- FP32는 괜찮은데 양자화 후 유독 나쁘거나, 반대로 학습/캘리브/추론 단계마다 결과가 미묘하게 다름.
- "코드는 안 바꿨는데" 배포 환경으로 옮기니 정확도가 다르다(라이브러리 기본값 차이).

**원인 (수치·메커니즘)**
학습 때 쓴 정규화(mean/std, resize 보간, BGR/RGB, `[0,1]` vs `[0,255]`, NCHW/NHWC)와 **캘리브레이션·추론 때 전처리가 다르면**, 모델이 보는 입력 분포 자체가 어긋난다. calibration은 어긋난 분포로 scale을 잡고, 추론은 또 다른 분포로 들어오니 양자화 오차가 증폭된다. 조용한 이유는 **shape이 맞으면 어디서도 예외가 안 나기 때문**이다.

수치 감각: ImageNet 정규화에서 `mean=0.485, std=0.229`(R채널)를 빼먹고 `x/255`만 하면, 입력 평균이 0 근처가 아니라 0.5 근처로 통째로 shift된다. 첫 conv의 activation 분포가 옆으로 밀리고, 캘리브가 잡은 scale과 추론 분포가 어긋나 **INT8에서 특히** 오차가 커진다. FP32는 표현 범위가 넓어 이런 shift를 어느 정도 버티지만, INT8은 127단계뿐이라 못 버틴다 — 그래서 "FP32는 괜찮은데 INT8만 나쁨" 패턴이 나온다.

**전처리 실수 카탈로그** (shape은 맞아서 에러가 안 나는 것들)

| 실수 | 무엇이 어긋나나 | 흔한 원인 |
|------|----------------|-----------|
| `Resize(256)+CenterCrop(224)` vs `resize((224,224))` | 픽셀 내용 자체가 다름(가장자리 왜곡) | 학습은 torchvision, 배포는 손구현 |
| BGR vs RGB | 채널 순서 반전 | OpenCV(`cv2.imread`는 BGR) ↔ PIL(RGB) |
| `[0,255]` vs `[0,1]` 스케일 | 입력 크기 255배 차이 | `ToTensor()` 유무, `/255.0` 누락 |
| mean/std 누락 또는 다른 상수 | 분포 shift/스케일 | 다른 데이터셋 상수 복붙 |
| 보간법 bilinear vs nearest vs bicubic | 리샘플 픽셀 미세차 | resize 기본값이 라이브러리마다 다름 |
| NCHW vs NHWC 레이아웃 | 채널축 위치 | ONNX/TRT는 NCHW, 일부 SoC는 NHWC 선호 |
| `antialias` on/off | torchvision Resize 결과 달라짐 | 버전별 기본값 변경 |
| uint8 → float 변환 시 반올림 | 미세 오차 | `astype` 순서, `/255` 위치 |

**예방**
- 전처리를 **단일 함수/단일 소스**로 만들어 학습·캘리브·추론이 **똑같은 코드**를 부른다.
- mean/std, 보간법, 채널 순서, 스케일, 레이아웃을 상수로 고정하고 문서화(`design_rules.md`, [09_roadmap.md](09_roadmap.md) 12주).
- 파이프라인 경계(PyTorch↔ONNX↔TRT/SoC)마다 **동일 입력에 대한 출력 일치**를 수치로 검증.
- 가능하면 정규화(mean/std)를 **모델 그래프 안으로** 넣어(입력단 `Sub`/`Div` 노드) 배포 전처리 실수 여지를 줄인다.

**디버깅 절차**
1. **바이트 비교부터.** 같은 원본 이미지를 학습 전처리와 배포 전처리로 각각 통과시켜 `max_abs_diff`를 본다(`preprocess_parity.py`). `> 1e-3`이면 여기서 끝 — 다른 건 볼 필요 없다.
2. 어긋나면 카탈로그 표를 위에서부터 대조(resize 방식 → 채널 순서 → 스케일 → mean/std → 레이아웃).
3. 바이트가 일치하는데도 정확도가 낮으면 → 함정 1(캘리브) 또는 함정 3(백엔드)로 이동.

**재현 코드 — 학습 vs 배포 전처리 바이트 비교**

같은 원본 이미지를 "학습 전처리"와 "배포 전처리"로 각각 통과시켜 **바이트 단위로** 비교한다. 여기서 어긋나면 정확도는 반드시 샌다. 아래는 어느 채널/어느 위치에서 가장 크게 어긋나는지까지 짚도록 확장했다.

```python
# preprocess_parity.py
# 목적: 학습용 전처리와 배포/캘리브용 전처리가 완전히 동일한지 바이트 단위로 검증
import numpy as np
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# (A) 학습 파이프라인의 전처리 (torchvision) — 이것이 "정답"
train_tf = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),                                  # [0,1], RGB, CHW
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# (B) 배포/캘리브 파이프라인이 "직접 구현"한 전처리 (여기서 실수 자주 발생)
def deploy_preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)  # ⚠️ Resize+CenterCrop 조합과 다름!
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return np.transpose(x, (2, 0, 1))              # HWC -> CHW

a = train_tf(Image.open("sample.jpg").convert("RGB")).numpy()
b = deploy_preprocess("sample.jpg")

assert a.shape == b.shape, f"레이아웃/shape 불일치 a={a.shape} b={b.shape}"
diff = np.abs(a - b)
print(f"shape={a.shape}  max_abs_diff={diff.max():.6f}  mean_abs_diff={diff.mean():.6f}")

# 어느 채널이 가장 어긋나는지(채널 순서 실수 탐지에 유용)
per_ch = diff.reshape(a.shape[0], -1).max(axis=1)
print("per-channel max_abs_diff:", np.round(per_ch, 4))

# 채널 순서 뒤바뀜(BGR/RGB) 의심: 채널을 뒤집으면 오차가 줄어드는가?
b_flip = b[::-1].copy()
if np.abs(a - b_flip).max() < diff.max() * 0.5:
    print("⚠️ 채널을 뒤집으니 오차 급감 → BGR/RGB 순서 실수 의심")

if diff.max() > 1e-3:
    print("⚠️ 전처리 불일치! 배포 전처리가 학습과 다르다 → 정확도 조용히 하락")
else:
    print("✅ 전처리 일치")
```

기대 출력(위 예처럼 Resize/Crop이 다르면):
```
shape=(3, 224, 224)  max_abs_diff=0.417293  mean_abs_diff=0.061...
per-channel max_abs_diff: [0.4173 0.3902 0.3661]
⚠️ 전처리 불일치! 배포 전처리가 학습과 다르다 → 정확도 조용히 하락
```

> 💡 팁: 위 예의 함정은 흔하다 — `Resize(256)+CenterCrop(224)` vs `resize((224,224))`는 **다른 이미지**를 만든다. shape은 같아서 아무 에러도 안 난다. per-channel diff가 세 채널 모두 비슷하게 크면 resize/스케일 문제, 특정 채널만 크면 채널별 mean/std 실수를 의심하라.

> ⚠️ 주의: TensorRT INT8 calibrator에 넣는 배치도 **추론과 똑같은 전처리**여야 한다. calibrator만 다른 전처리를 쓰면 scale이 엉뚱하게 잡힌다([05_tensorrt.md](05_tensorrt.md)). calibration cache는 전처리에 종속이므로, 전처리를 고쳤으면 **캐시를 지우고 재생성**하라.

### 함정 2-b — 전처리 한 줄이 양자화 방법 선택보다 정확도에 크게 작용한다

앞의 함정 2는 "학습과 배포의 전처리가 **어긋나면**" 이야기다. 2-b는 더 얌전한 경우다 — **어긋나지 않고, 처음부터 끝까지 일관되게 한 가지 전처리만 썼는데도** 그 전처리가 torchvision 표준이 아니어서 정확도가 통째로 낮게 깔린다. 학습·캘리브·추론이 전부 같은 코드를 부르므로 함정 2의 바이트 비교(`preprocess_parity.py`)로는 **절대 안 잡힌다.**

**실측 — 같은 모델·같은 캘리브, resize 방식만 바꿨다**

> 측정: ResNet18(torchvision 사전학습) / ImageNet val **50,000장 전량** / 캘리브 200장 / 짝지어진 McNemar. 두 전처리 각각에 대해 **전처리 캐시를 따로 만들어** FP32·INT8을 평가했다. 원문: [재실행 보고서 7절](../logs/stage1_real_imagenet_report.html)

| 전처리 | FP32 top-1 | INT8 MinMax | 양자화 손실 | 그 손실의 p | 공개값(69.758%) 대비 |
|--------|-----------|-------------|------------|------------|---------------------|
| `squash` — `resize((256,256))` 후 중앙 224 (종횡비 무시) | 68.74% | 68.62% | −0.12%p | 0.061 n.s. | **−1.02%p** |
| `tv` — 짧은 변만 256(bilinear) 후 `CenterCrop(224)` | **69.81%** | 69.67% | −0.13%p | 0.034 | **+0.05%p** |

읽을 것은 두 가지다.

1. **전처리 교체 = +1.07%p, p=1.6e-14.** 같은 가중치·같은 캘리브 데이터인데 resize 방식만으로 1%p가 오간다. 반면 **양자화 손실은 두 전처리 모두 −0.12~0.13%p**로 사실상 동일하다. 즉 **전처리의 대가가 양자화의 대가보다 약 9배 크다.** 캘리브 방법을 MinMax/Entropy/Percentile로 바꿔가며 며칠을 쓰기 전에, **전처리 한 줄이 표준인지 먼저 확인**하는 게 훨씬 남는 장사다.
2. **`tv` 쪽이 공개값과 0.05%p 안에서 만난다.** 이건 단순한 정확도 개선이 아니라 **파이프라인 전체의 정합성 증명**이다 — tar 무결성·라벨 매핑·전처리·클래스 인덱스 규약 네 가지가 **동시에** 맞아야만 나오는 값이다. 반대로 `squash`의 68.74%만 보고 있었다면 "라벨이 어긋났나, 전처리가 틀렸나, 원래 이 정도인가"를 구분할 방법이 없다.

**왜 종횡비 squash가 손해인가**
`resize((256,256))`은 세로가 긴 사진을 가로로 늘리고 가로가 긴 사진을 세로로 늘린다. 객체의 **가로세로 비율이 학습 때 본 것과 달라진다.** ImageNet val은 종횡비가 4:3~3:4 근처에 몰려 있어 왜곡이 극단적이지 않다 — 그래서 붕괴가 아니라 **−1%p라는 "조용한" 크기**로 나타난다. 덧붙여 PIL `Image.resize()`의 **기본 resample은 BICUBIC**이라, 명시하지 않으면 torchvision의 bilinear와 보간법까지 달라진다.

**예방 / 판정 절차**
1. **사전학습 모델을 쓴다면, 그 모델 카드의 공식 top-1을 재현하는 것이 첫 실험이다.** 양자화는 그 다음이다. 재현이 안 되면 이후 모든 %p 비교는 오염된 기준선 위에서 도는 것이다.
2. 재현 오차 판정: **±0.1%p 안이면 통과**, −0.5%p 이상 낮으면 전처리(2-b), 몇 %p씩 낮으면 라벨 매핑·클래스 인덱스, **높게 나오면 평가셋 표본 편향**(함정 0).
3. 전처리는 직접 구현하지 말고 **모델과 함께 배포되는 것을 쓴다.** torchvision이라면 `weights.transforms()`가 정답을 들고 있다.

```python
# 사전학습 모델의 "정답 전처리"를 직접 구현하지 말고 받아온다
from torchvision.models import resnet18, ResNet18_Weights
w = ResNet18_Weights.IMAGENET1K_V1
print(w.transforms())        # resize_size=[256], crop_size=[224], interpolation=bilinear, mean/std
print(w.meta["_metrics"]["ImageNet-1K"]["acc@1"])   # 69.758 — 재현해야 할 목표값
```

> 🔴 함정: **전처리 문제와 양자화 문제는 증상이 같다(정확도만 조용히 낮다).** 그래서 순서가 중요하다 — **FP32로 공식값을 재현한 뒤에 양자화를 켜라.** 순서를 뒤집으면, 전처리 때문에 깔린 −1%p를 "양자화 손실"로 오인하고 존재하지 않는 원인을 몇 날 며칠 뒤진다. 실제로 1단계 1차 실행이 이 상태였고, **평가셋을 50,000장으로 늘려 공식값과 대조하기 전까지는 알 방법이 없었다**(그래서 함정 0과 한 묶음이다).

---

## 함정 3 — "ONNX export 성공" ≠ "칩에서 동작"

> 관련 단계: [04_transformer_quantization.md](04_transformer_quantization.md)(`onnx_export_failures.md`), [05_tensorrt.md](05_tensorrt.md), [06_multi_soc.md](06_multi_soc.md), [09_roadmap.md](09_roadmap.md) 3~5주

**증상**
- `torch.onnx.export`가 예외 없이 끝나고 `onnx.checker`도 통과. 그런데 TensorRT 엔진 빌드나 TIDL/QNN 컴파일에서 특정 op가 미지원/에러.
- PC(ONNX Runtime CPU)에선 돌지만, 타깃 가속기에선 op가 빠지거나 결과가 다름.
- QNN EP에서 "dynamic shape 미지원" 또는 "Loop/If unsupported"로 세션 생성 자체가 실패.

**원인 (메커니즘)**
ONNX export는 **그래프를 표현**했을 뿐이다. 각 백엔드(TensorRT / TIDL / QNN / DRP-AI)는 **지원 op의 부분집합**만 가진다. 미지원 op, 특정 opset, dynamic shape, control flow(Loop/If)는 백엔드마다 처리 방식이 다르다. 즉 export는 파이프라인의 **시작**이지 끝이 아니다.

구체적으로 어긋나는 지점들:
- **op 부분집합**: TensorRT/TIDL/QNN 각각 지원 op 목록이 다르다. 커스텀 activation, 특이한 reduce, `NonZero`/`Scatter` 같은 데이터 의존 op가 잘 빠진다.
- **opset 불일치**: export한 opset을 백엔드 파서가 아직 지원 안 함(또는 그 op의 특정 opset 버전만 지원).
- **dynamic shape**: QNN EP는 dynamic shape 자체를 지원하지 않는다. TensorRT는 optimization profile을 명시해야 한다.
- **control flow**: QNN EP는 Loop/If를 지원하지 않는다 → 그 서브그래프가 통째로 CPU fallback되거나 세션이 실패.

> QNN EP는 ONNX op의 부분집합만 지원한다(예: Loop/If 미지원, dynamic shape 미지원). ONNX Runtime은 미지원 op를 CPU로 fallback시킨다. ([ONNX Runtime QNN EP 문서](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html))

**예방**
- export 직후 **타깃 백엔드로 곧장 컴파일**해 op 지원을 조기 검증(뒤로 미루지 말 것).
- 백엔드별 **지원 op 목록**을 미리 확인하고, 모델을 그 화이트리스트에 맞춰 설계.
- opset·dynamic axes를 명시하고, 실패 op와 우회법을 `onnx_export_failures.md`에 남긴다.
- dynamic shape가 필요 없으면 **고정 shape로 export**(QNN/TIDL 호환성↑).

**디버깅 절차 — 3-way 정합성 + op 인벤토리**

핵심 전략: 같은 입력을 **(1) PyTorch → (2) ONNX Runtime → (3) 타깃 백엔드** 순으로 흘려, 어느 경계에서 수치가 어긋나는지 좁힌다. "2)는 통과, 3)에서 실패"면 원인은 export가 아니라 **백엔드 op 지원**이다.

```bash
# ── 1) 구조/버전 검증 ─────────────────────────────────────────
python3 -c "import onnx; m=onnx.load('model.onnx'); onnx.checker.check_model(m); \
print('opset:', m.opset_import[0].version, 'ok')"

# ── 2) PyTorch vs ONNXRuntime 수치 비교 (Polygraphy) ─────────
#    여기서 어긋나면 export 자체가 의심(PyTorch↔ONNX 변환 오류).
polygraphy run model.onnx --onnxrt --pytorch --atol 1e-3 --rtol 1e-3

# ── 3) 타깃 백엔드(TensorRT) 빌드 + ONNXRuntime과 3-way 비교 ──
#    빌드 실패 로그에서 미지원 op / plugin 필요 op를 확인.
polygraphy run model.onnx --trt --onnxrt --atol 1e-2 --rtol 1e-2
```

기대: 2)에서 통과, 3)에서 실패하면 원인은 export가 아니라 **백엔드 op 지원**이다. 다음으로 **어떤 op가 미지원인지**를 특정한다 — 두 가지 방법을 병행:

```bash
# (a) Polygraphy로 TensorRT 지원 여부를 그래프 분할까지 해서 리포트
#     미지원 op가 "Operator | Count | Reason | Nodes" 표로 나온다.
polygraphy inspect capability --with-partitioning model.onnx
```
기대 출력(예):
```
Graph is not supported by TensorRT. Partitioning into supported/unsupported subgraphs.
=== Unsupported Operators ===
Operator     | Count | Reason                                             | Nodes
-------------+-------+----------------------------------------------------+------------------
GridSample   |   2   | No importer registered for op: GridSample          | [ [45,46] ]
NonZero      |   1   | Data-dependent shape not supported                 | [ [88,89] ]
```
→ 이 표의 op가 **plugin 대상 또는 우회 대상**이다. (partitioning은 FunctionProto 내부 노드는 못 본다 — 그때는 `--with-partitioning` 없이 정적 리포트를 쓴다. [Polygraphy inspect 예제](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect))

```python
# (b) op_inventory.py — 모델이 쓰는 op를 세어 백엔드 화이트리스트와 대조
import onnx, collections
m = onnx.load("model.onnx")
c = collections.Counter(n.op_type for n in m.graph.node)
for op, cnt in c.most_common():
    print(f"{op:22s} x{cnt}")
# → 이 목록을 TensorRT/TIDL/QNN 지원 op 문서와 대조. 없는 op가 fallback/plugin 대상.
```
기대 출력(예):
```
MatMul                 x48
Add                    x37
LayerNormalization     x25
Softmax                x12
GridSample             x2      ← 백엔드 미지원이면 여기가 범인
NonZero                x1
```

**우회 결정 트리** (미지원 op를 찾은 뒤):
1. 그 op를 **동등 표준 op로 분해** 가능한가? (예: 커스텀 GELU → `Erf` 기반 표준 GELU) → 재export.
2. 안 되면 **모델 앞/뒤로 몰아** 전·후처리로 빼낼 수 있나? → subgraph를 통짜로(함정 4 예방).
3. 둘 다 안 되면 **custom plugin**(TensorRT `IPluginV3`, 함정 5) 또는 그 op만 FP16/CPU 유지.
4. 모든 우회를 `onnx_export_failures.md`에 op·에러 원문·선택한 우회로와 함께 기록.

> 🔴 함정: "checker 통과 = 끝"이라는 착각이 3~5주를 통째로 날린다. export 성공 후 **가장 먼저** 타깃 컴파일(`polygraphy inspect capability`)을 시도하라. 미지원 op를 3주차에 알면 우회할 시간이 있지만, 8주차에 알면 로드맵이 밀린다.

---

## 함정 4 — fallback 지옥 (subgraph가 쪼개지면 FP32보다 느리다)

> 관련 단계: [03_quantization_theory.md](03_quantization_theory.md)(양자화 dtype 선택), [05_tensorrt.md](05_tensorrt.md)(TRT EP/엔진), [06_multi_soc.md](06_multi_soc.md)(TIDL/QNN/DRP-AI), [09_roadmap.md](09_roadmap.md) 9~11주(`four_target_matrix.md`)

**증상**
- 가속기(NPU/DSP)에 올렸는데 **오히려 FP32 CPU보다 느리다.**
- 컴파일 로그에 subgraph가 수십 개로 쪼개짐(예: 20개). 가속기↔CPU를 왔다갔다.
- profile을 보면 연산 시간보다 **메모리 복사/동기화 시간**이 더 크다.
- **EP가 provider 목록에 정상으로 잡히고, 세션도 만들어지고, 결과도 정확한데** INT8이 FP32보다 느리다(아래 실측 사례). 실패를 알리는 예외·리턴코드가 **아무것도 없다.**

**원인 (수치·메커니즘)**
백엔드가 미지원 op를 만나면 그래프를 **여러 subgraph로 분할**하고, 미지원 부분을 CPU/ARM으로 fallback시킨다. 분할이 잦으면 가속기↔호스트 간 **데이터 복사·동기화 오버헤드**가 연산 이득을 잡아먹는다. op 하나가 그래프 중간에서 미지원이면, 앞뒤가 통째로 쪼개진다.

왜 "중간의 op 하나"가 치명적인가: 그래프가 `[가속기 가능]→[미지원 op]→[가속기 가능]`이면, 미지원 op를 CPU에서 처리하려고 (1) 가속기→CPU 텐서 복사, (2) CPU 연산, (3) CPU→가속기 복사가 매 추론마다 일어난다. 이 왕복이 subgraph 경계마다 반복되면, 복사 대역폭이 병목이 되어 **가속기의 연산 이득을 상쇄하고도 남는다.** subgraph가 20개면 최대 그만큼의 왕복이 생긴다.

> - ONNX Runtime은 `GetCapability`로 각 EP가 실행 가능한 subgraph를 질의해 그래프를 분할하고, 나머지는 CPU로 fallback한다. ([ONNX Runtime 아키텍처](https://onnxruntime.ai/docs/reference/high-level-design.html))
> - TIDL은 미지원 레이어를 ARM으로 떨어뜨리며(예: MaxPool을 ARM에 두면 그 지점에서 subgraph가 갈린다), `deny_list`로 특정 op의 오프로드를 강제 제외할 수 있다(ONNX는 op명 예: `'MaxPool, Add'`, TFLite는 레이어 코드 예: `'1, 2'`). ([edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools))
> - QNN EP는 `session.disable_cpu_ep_fallback="1"`로 "전부 HTP에 올라가야만 성공"을 강제할 수 있다(안 올라가면 예외 → 미지원 지점 특정에 유용). ([ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html))

**예방**
- **offload 비율(가속기에 올라간 연산/전체)부터 확인**한다. 이것이 첫 번째 지표다.
- subgraph 개수를 최소화하도록 미지원 op를 **모델 앞/뒤로 몰거나 제거**(전·후처리로 이동).
- fallback op가 그래프 중앙에 있으면, 대체 op로 바꾸거나 custom 구현으로 가속기에 유지.
- 타깃별 지원 op를 보고 **처음부터 화이트리스트 내에서** 모델을 고른다(함정 3의 예방과 동일 뿌리).

**디버깅 절차 — offload/subgraph 정량 확인**

1. **먼저 offload 비율과 subgraph 개수를 센다.** latency는 그 다음이다.
2. QNN: VERBOSE 로그에서 각 EP에 할당된 노드 수를 세거나, `disable_cpu_ep_fallback`로 "전부 HTP인지" 강제 확인.
3. TIDL: 컴파일 로그의 "Runtimes Graphviz"/subgraph 요약에서 C7x vs ARM 분할을 확인.
4. TensorRT EP(ONNX Runtime): `sess.get_providers()`는 **등록된** provider를 돌려줄 뿐 실행 여부가 아니다. `log_severity_level`을 2 이하로 **되돌려**(벤치 스크립트가 올려놨을 가능성이 높다) 파서/파티셔닝 에러를 읽고, **같은 모델의 FP32를 같은 EP로 돌린 latency와 비교**한다(아래 실측 사례·`ep_offload_check.py`).
5. 판정표(아래)로 "정상/경계/지옥"을 분류하고, 지옥이면 미지원 op 위치를 함정 3 절차로 특정해 앞뒤로 몬다.

```python
# qnn_partition_check.py — ONNX Runtime EP가 몇 개 노드를 fallback시키는지 확인
# 세션 로그(VERBOSE)에 "assigned to CPUExecutionProvider" 노드가 fallback 대상.
import onnxruntime as ort
so = ort.SessionOptions()
so.log_severity_level = 0                         # 0=VERBOSE: 파티션/할당 로그 출력

# (선택) "전부 HTP에 올라가야 성공"을 강제 → 안 올라가면 예외로 미지원 지점 노출
# so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

sess = ort.InferenceSession(
    "model.onnx", so,
    providers=["QNNExecutionProvider", "CPUExecutionProvider"],  # QNN 우선, 나머지 CPU
    provider_options=[{"backend_path": "libQnnHtp.so"}, {}],      # HTP 백엔드
)
# 로그에서 세어라:
#   - QNN에 할당된 노드 수 vs CPU에 할당된 노드 수  → offload 비율
#   - 파티션(subgraph) 개수가 많을수록 fallback 지옥 신호
print("providers in use:", sess.get_providers())
```

TIDL의 경우 컴파일 로그에서 subgraph 개수와 각 subgraph의 delegate(C7x vs ARM)를 확인한다.

```bash
# TIDL 컴파일 로그에서 subgraph 분할/오프로드 요약 grep (예시)
grep -Ei "subgraph|deny|delegate|offload|Unsupported|ARM|C7x" tidl_compile.log

# 특정 op를 강제로 ARM에 떨어뜨려(deny) subgraph 분할 영향을 실험(ONNX op명)
#   → 컴파일 옵션의 deny_list에 'MaxPool, Add' 지정 후 offload 비율 변화 관찰
```

판정 기준(경험칙):
| offload 비율 | subgraph 개수 | 해석 | 조치 |
|--------------|---------------|------|------|
| > 90% | 1~3 | 정상. 가속기 이득 기대 | 그대로 진행 |
| 50~90% | 4~10 | 경계. fallback op 위치 점검 | 중앙 미지원 op를 앞/뒤로 이동 |
| < 50% | > 10 | 🔴 fallback 지옥. FP32보다 느릴 수 있음 | 모델 재설계/화이트리스트 재선정 |

**실측 사례 — TensorRT EP가 INT8 QDQ 그래프를 "0%" 가져간다 (offload 비율의 극단값)**

> 측정 환경: RTX 3060 12GB / ResNet18(torchvision) / batch=1, 워밍업 20 + 60회 p50 / **ORT 1.23.2 + TensorRT 10.16.x** / 캘리브 200장. **정확도는 ImageNet val 50,000장 전량**에서 짝지어진 McNemar로 판정했다(전처리 `squash` 기준). 전 과정: [1차 로그 8장](../logs/stage1_quantization_log.html#s8)(메커니즘·2×2 절제 규명) + [재실행 보고서 9~10절](../logs/stage1_real_imagenet_report.html)(아래 수치의 출처)

위 판정표는 NPU/DSP 얘기처럼 보이지만, **데스크톱 GPU + TensorRT EP에서 offload 비율이 0%가 되는 일**이 1단계에서 그대로 재현됐다. [03_quantization_theory.md](03_quantization_theory.md) 4.3의 권장 설정(ORT 기본, `activation_type=QUInt8` 비대칭)으로 만든 INT8 QDQ ONNX를 ONNX Runtime **TensorRT EP**로 돌렸더니:

- `sess.get_providers()`에 `TensorrtExecutionProvider`가 **정상으로 잡히고**, 세션 생성도 성공하고, **출력도 정확하다**(50,000장에서 FP32 대비 −0.12%p, McNemar p=0.061로 유의하지 않음).
- 그런데 p50이 **3.06 ms** — 같은 모델 FP32를 같은 TRT EP로 돌린 **0.96 ms보다 3배 느리다.**

로그를 뒤져 보면 진짜 이유가 나온다. **TRT 파서가 그래프를 하나도 못 가져갔다.**

```
[WARNING] onnxOpImporters.cpp:1695: TensorRT supports QuantizeLinear/DequantizeLinear with
          UINT8 zero_point only on DLA (version >= 3.16). Defaulting to INT8 instead.
[WARNING] onnxOpImporters.cpp:1703: For zero_point with type int32 TensorRT will use INT8 instead.
[ERROR] ITensor::getDimensions: Error Code 4: API Usage Error
        (conv1.weight_bias_DequantizeLinear: input has type Int32 but must have type
         FP8, FP4, Int4, or Int8. In checkType at nodeBase.cpp:455)
[ERROR] ModelImporter.cpp:149: ERROR: In function parseNode:
        [6] Invalid Node - conv1.weight_bias_DequantizeLinear
[ERROR] [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
        ... conv/fc 21개 전부에서 반복
```

맨 위 두 줄이 이 함정의 **가장 알아보기 쉬운 지문**이다. TRT가 "네 uint8/int32 zero_point는 여기서 못 쓴다"고 **말해 주고 있다.**

> ⚠️ 그런데 왜 못 보고 지나가는가. ORT의 기본 severity는 WARNING이라 이 줄들은 **사실 찍히기는 한다** — 다만 추론 한 번에 WARNING/ERROR가 **400줄 넘게** 쏟아지고(실측: 세션 생성+1회 run에 444줄), 그게 다 지나간 뒤 세션은 **정상 리턴**한다. 실패를 알리는 예외도, 종료 코드도, 요약 한 줄도 없다. 게다가 벤치마크 스크립트는 출력을 깔끔하게 하려고 `so.log_severity_level = 3~4`로 **올려두는 게 관행**이라(1단계의 `check_trt.py`가 실제로 `= 4`였다) 그 순간 진짜로 한 줄도 안 남는다. **"로그에 에러가 없었다"가 아니라 "로그를 안 봤다/꺼놨다"인 경우가 대부분이다.**

**에러는 두 종류인데 원인은 하나다.** 위 로그에는 ① INT32 bias DQ 타입 에러와 ② `shiftIsAllZeros`(non-zero zero-point) assertion이 **같이** 뜬다. 둘 다 ORT 기본 설정에서 나오니 "독립적인 원인이 둘"로 읽기 쉽지만, **2×2 절제 실험**(두 변수를 따로 껐다 켜 본 것)으로 갈라 보면 파싱 성공/실패를 가르는 변수는 **하나뿐**이다:

| # | activation | `QuantizeBias` | act zero-point | INT32 bias DQ | TRT p50 | CUDA p50 | vs FP32(TRT) | 판정 |
|---|-----------|----------------|----------------|---------------|---------|----------|--------------|------|
| — | FP32 (기준선) | — | — | — | 0.96 ms | 1.33 ms | 1.00× | — |
| A | `QUInt8` 비대칭 | `True` | `[0, 173]` | 21개 | 3.06 ms | 1.81 ms | **0.31×** | 🔴 폴백 |
| B | `QUInt8` 비대칭 | **`False`** | `[0, 173]` | **0개** | 2.97 ms | 1.80 ms | **0.32×** | 🔴 **여전히 폴백** |
| C | `QInt8` 대칭 | `True` | **`[0, 0]`** | **21개** | **0.51 ms** | 2.11 ms | **1.86×** | ✅ **빌드 성공** |
| D | `QInt8` 대칭 | `False` | **`[0, 0]`** | 0개 | 0.51 ms | 1.99 ms | 1.86× | ✅ 빌드 성공 |

> `act zero-point` 열은 그래프의 모든 activation `QuantizeLinear`의 zero-point 값 범위다(`[최소, 최대]`). **이 열만 보면 성공/실패가 그대로 읽힌다** — 판정을 가르는 유일한 변수라서다. 표는 4개 모델을 한 실행에서 잰 것(워밍업 20 + 60회 p50)이며, 재현 스크립트는 [1차 로그 8장](../logs/stage1_quantization_log.html#s8)·값은 [재실행 보고서 10절](../logs/stage1_real_imagenet_report.html)이다.

B는 INT32 bias DQ를 **0개로 만들어도 실패**하고, C는 **21개가 그대로 남아 있는데도 성공**한다. 즉:

1. **하드 블로커는 ② `zero_point ≠ 0` 하나다.** TensorRT는 Q/DQ에 **대칭(zero_point=0)만** 받는다.
2. **① INT32 bias DQ 에러는 2차 증상이다.** zero-point 때문에 Q/DQ 융합이 깨지고 나면 bias DQ가 홀로 남아 타입 검사에 걸린다. 융합이 정상이면 TRT는 INT32 bias DQ를 그대로 받아들인다(case C).
3. 따라서 `"QuantizeBias": False`는 **해법이 아니라 선택적 정리**다. 이것만 켜고 zero-point를 안 고치면 아무것도 달라지지 않는다.

결과적으로 노드를 하나도 못 가져가고 전부 폴백하며, 파티셔닝 오버헤드까지 얹혀 **FP32보다 느려진다.**

> 💡 **일반 교훈:** 에러 메시지가 N개라고 원인이 N개인 건 아니다. 여러 에러가 같이 뜰 때는 **변수를 하나씩만 바꾼 조합을 만들어** 어느 것이 진짜 블로커인지 갈라야 한다. 여기서는 조합 4개를 뽑는 데 몇 분이면 됐고, 그 결과 처방이 "두 군데 고쳐라"에서 "한 군데 고쳐라(+선택적 정리)"로 정확해졌다.

**해결 — 타깃이 TensorRT면 활성화를 대칭 `QInt8`로 뒤집는다**

```python
# 같은 캘리브 리더·같은 데이터, dtype/대칭성만 바꾼다
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
from calib_reader import ImageNetCalibReader      # 03_quantization_theory.md 4.2의 리더

reader = ImageNetCalibReader("imagenet/val", input_name="input", limit=200)

quantize_static(
    "resnet18_fp32.onnx", "resnet18_int8_trt_sym.onnx", reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,        # ← QUInt8(비대칭) 금지. TRT는 int8 대칭만.
    weight_type=QuantType.QInt8,
    per_channel=True,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options={
        "ActivationSymmetric": True,        # ← 이 둘을 "같이" 바꿔야 zero_point=0이 된다.
                                            #    QUInt8인 채로 이것만 켜면 zp=127이라 여전히 실패.
        # "QuantizeBias": False,            # 선택. bias INT32 DQ를 없애 그래프를 정리(속도·정확도 모두 동일).
    },
)
```

| 설정 | TRT EP p50 | vs FP32(TRT EP) | top-1 (50k) | Δ vs FP32 | McNemar p |
|------|-----------|-----------------|-------------|-----------|-----------|
| FP32 | 0.96 ms | 1.00× | 68.74% | — | — |
| INT8 `QUInt8` 비대칭 (ORT 기본 권장) | **3.06 ms** | **0.31× (느려짐)** | 68.62% | −0.12%p | 0.061 n.s. |
| INT8 `QInt8` 대칭 | **0.51 ms** | **1.86×** | 68.33% | −0.41%p | **5.0e-8 유의** |
| INT8 `QInt8` 대칭 + `QuantizeBias=False` | 0.51 ms | 1.86× | 68.33% | −0.41%p | **5.0e-8 유의** |

**같은 INT8인데 TensorRT에서 6배 차이**(3.06 → 0.51 ms)다. 즉 "activation은 비대칭 uint8"은 **x86 CPU/VNNI 한정 권장**이고, 타깃이 TensorRT면 반드시 뒤집어야 한다([05_tensorrt.md](05_tensorrt.md)).

> ⚠️ **대칭 강제는 무료가 아니다 — 여기서 정정.** 1차 실행(1,000장)에서는 대칭 전환의 대가가 −0.4%p이면서 p=0.52로 "유의하지 않음"이었다. 50,000장에서 다시 재니 **비대칭 68.62% → 대칭 68.33%, −0.29%p가 p=9.2e-5로 유의**하다(FP32 기준으로는 −0.41%p, p=5.0e-8). **양자화 손실이 아니라 "대칭 강제 손실"이 따로 있다**는 뜻이다 — zero-point를 0으로 묶으면 post-ReLU처럼 한쪽만 쓰는 분포에서 표현 구간 절반을 버리기 때문이다. 그래도 **1.86× 속도를 0.29%p로 사는 거래**이므로 대개 남는 장사이지만, **정확도 예산이 0.3%p 이하인 과제라면 계산에 넣어야 한다.** ([함정 0](#함정-0--평가셋이-작으면-나머지-함정을-진단할-수-없다) — 1차에서 이 손실을 "무료"로 판정한 것이 바로 소규모 셋의 검정력 부족 때문이다.)

> 💡 **`QuantizeBias=False`는 정확도에 영향이 0이다.** C와 D의 50,000장 예측이 **한 장도 다르지 않았다**(0장 불일치). bias INT32 DQ는 파싱·그래프 정리 관점에서만 논할 옵션이고, 정확도 트레이드오프는 없다.

> 💡 참고: 같은 벤치를 다른 EP로 돌리면, ORT **CUDA EP**에서는 INT8이 FP32보다 **오히려 느리다**(FP32 1.33 ms → 비대칭 1.81 ms, 대칭 2.11 ms). CUDA EP에는 QDQ INT8 conv 커널이 없어 DQ로 되돌려 FP로 계산하는데, Q/DQ 노드만 늘어난 만큼 손해다. **INT8은 그 자체로 빠른 게 아니라, INT8 커널을 실제로 쓰는 EP에서만 빠르다.** GPU에서 이득을 보려면 TRT EP(또는 `trtexec` 엔진 빌드)가 **필수**다. 반대로 CPU EP에서는 VNNI 없는 i7-6700에서도 13.18 → 7.23 ms(**1.82×**)로 기대대로 빨라졌다.

**판별법 — 상위 EP가 하위 EP보다 빠르지 않으면 폴백이다**

`ep_offload_check.py`(아래)의 판정 기준은 원래 "INT8이 FP32보다 느리면 의심"이었다. 그런데 위 CUDA EP 결과가 보여주듯 **INT8이 FP32보다 느린 건 폴백이 아니어도 일어난다**(커널이 없어서). 그래서 판정을 하나 더 겹친다:

| 비교 | 정상 동작 | 무음 폴백 |
|------|----------|-----------|
| INT8 vs 같은 EP의 FP32 | 빨라짐 (1.86×) | 느려짐 (0.31×) |
| **INT8 on TRT EP vs 같은 INT8 on CUDA EP** | **TRT가 훨씬 빠름** (0.51 vs 2.11 ms = **4.1×**) | **TRT가 더 느림** (3.06 vs 1.81 ms) |

두 번째 줄이 더 강한 판별식이다. **TRT EP가 CUDA EP보다 빠르지 않으면, TRT는 그래프를 안 가져갔다.** 이유는 단순하다 — TRT가 실제로 엔진을 빌드했다면 같은 그래프를 시뮬레이션으로 돌리는 CUDA EP보다 느릴 수가 없다. 폴백 시에는 TRT EP가 파티셔닝만 하고 실행은 아래 체인(CUDA)에 넘기므로, **CUDA 단독보다 오버헤드만큼 느려진다.** 코드로는 `trt_p50 < cuda_p50 * 0.8`을 통과 조건으로 쓰면 된다.

> ⚠️ 이 판정은 **GPU를 단독 점유**해야 유효하다. 다른 평가·학습과 병렬로 p50을 재면 두 값이 함께 출렁여 `trt < cuda×0.8` 판정이 오염된다(실측에서 실제로 겪은 함정이다). 정확도만 재는 스크립트는 병렬로 돌려도 값이 변하지 않지만, **latency 측정은 반드시 단독으로** 돌린다.

**진단 스니펫 — 로그 + FP32 대비 + 하위 EP 대비 latency로 폴백을 잡는다**

```python
# ep_offload_check.py
# 목적: EP가 "목록에 있다"가 아니라 "실제로 그래프를 가져갔다"를 로그+latency로 판정
# 실행: python3 ep_offload_check.py resnet18_fp32.onnx resnet18_int8.onnx
import sys, time
import numpy as np, onnxruntime as ort

ort.set_default_logger_severity(2)     # 2=WARNING. 벤치 스크립트가 3~4로 올려놨으면 되돌린다.
CHAIN = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

def bench(path, provider, iters=100, warmup=20):
    so = ort.SessionOptions()
    so.log_severity_level = 2          # ← 여기서 3~4로 올리는 순간 파서 에러가 사라진다 (0=VERBOSE면 노드 할당까지)
    providers = CHAIN[CHAIN.index(provider):]                     # 그 EP + 아래쪽 폴백 체인
    sess = ort.InferenceSession(path, so, providers=providers)
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]   # dynamic batch → 1
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})                             # TRT 엔진 빌드/워밍업 포함
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, {inp.name: x}); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(ts, 50)), sess.get_providers()

fp32, int8 = sys.argv[1], sys.argv[2]
p50 = {}                                                          # (모델, EP) → p50, 교차 판정용
for provider in CHAIN:
    if provider not in ort.get_available_providers():
        continue
    p_fp32, registered = bench(fp32, provider)
    p_int8, _          = bench(int8,  provider)
    p50[provider] = p_int8
    # 판정 ①: INT8인데 같은 EP의 FP32보다 느리면 그래프를 안 가져갔거나 INT8 커널이 없다
    verdict = f"✅ {p_fp32 / p_int8:.2f}× 가속" if p_int8 < p_fp32 else "🔴 폴백/미가속 의심"
    print(f"{provider:28s} FP32 p50={p_fp32:6.2f}ms  INT8 p50={p_int8:6.2f}ms  {verdict}")
    print(f"    등록된 provider(실행 여부 아님): {registered}")

# 판정 ②(더 강함): 상위 EP가 하위 EP보다 빠르지 않으면 상위 EP는 실행하지 않았다.
#   TRT가 정말 엔진을 빌드했다면, 같은 그래프를 QDQ 시뮬레이션으로 도는 CUDA EP보다 느릴 수 없다.
trt, cuda = p50.get("TensorrtExecutionProvider"), p50.get("CUDAExecutionProvider")
if trt and cuda:
    ok = trt < cuda * 0.8                                         # 0.8 = 측정 노이즈 여유
    print(f"\n[교차 판정] INT8: TRT {trt:.2f}ms vs CUDA {cuda:.2f}ms → "
          f"{'✅ TRT가 실제로 실행 중' if ok else '🔴 TRT 무음 폴백 확정 — 파서 로그를 읽어라'}"
          f"  (TRT/CUDA = {trt / cuda:.2f}×)")
```

기대 출력 — 폴백 중인 모델(`QUInt8` 비대칭):
```
TensorrtExecutionProvider    FP32 p50=  0.96ms  INT8 p50=  3.06ms  🔴 폴백/미가속 의심
    등록된 provider(실행 여부 아님): ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
CUDAExecutionProvider        FP32 p50=  1.33ms  INT8 p50=  1.81ms  🔴 폴백/미가속 의심
CPUExecutionProvider         FP32 p50= 13.18ms  INT8 p50=  7.23ms  ✅ 1.82× 가속

[교차 판정] INT8: TRT 3.06ms vs CUDA 1.81ms → 🔴 TRT 무음 폴백 확정 — 파서 로그를 읽어라  (TRT/CUDA = 1.69×)
```
고친 모델(`QInt8` 대칭)로 같은 스크립트를 돌리면 TRT 줄과 교차 판정이 뒤집힌다:
```
TensorrtExecutionProvider    FP32 p50=  0.96ms  INT8 p50=  0.51ms  ✅ 1.86× 가속
CUDAExecutionProvider        FP32 p50=  1.33ms  INT8 p50=  2.11ms  🔴 폴백/미가속 의심

[교차 판정] INT8: TRT 0.51ms vs CUDA 2.11ms → ✅ TRT가 실제로 실행 중  (TRT/CUDA = 0.24×)
```
**CUDA 줄이 고친 뒤에도 🔴인 것은 정상이다** — CUDA EP에는 INT8 conv 커널이 없어 QDQ를 시뮬레이션만 하므로 FP32보다 느린 게 맞다. 판정 ①의 🔴은 "폴백"과 "커널 없음"을 구분하지 못하고, 그것을 구분해 주는 것이 **판정 ②**다.

→ `TensorrtExecutionProvider`가 목록에 **있는데도** 느리다 = 등록만 됐고 실행은 폴백. 여기서 파서 에러를 읽으면 위의 `Invalid Node` 줄이 나온다.

> ⚠️ 폴백 상태의 **절대값은 믿지 마라.** 폴백이 CUDA EP로 가는지 CPU EP로 가는지(= 어떤 provider를 함께 등록했는지)와 머신 부하에 따라 같은 모델이 3~5 ms대에서 출렁인다(실측: 1차 3.05·2.99·3.07 ms, 2차 3.06·2.97 ms). 판정 근거는 값이 아니라 **부호와 비율**이다. 반대로 정상 동작하는 쪽(0.51~0.55 ms)은 재현성이 높다.

> 🔴 함정: **"EP가 목록에 있다 + 결과가 맞다"는 그 EP를 실제로 썼다는 증거가 아니다.** 판정은 반드시 **로그와 latency**로 한다. 이건 0단계에서 겪은 `libnvinfer.so.10` 미탐지 무음 CPU 폴백(p50 11.83 ms = CPU급 → 픽스 후 0.41 ms)과 **같은 계열의 함정**이다([01_environment_setup.md](01_environment_setup.md) 3-4-a, [0.5단계 실습 로그](../logs/stage0.5_ladder_log.html)). 한쪽은 라이브러리를 못 찾아서, 한쪽은 그래프를 못 파싱해서 폴백하지만, **둘 다 예외 없이 정확한 결과를 내면서 조용히 느려진다.** 새 EP를 붙였으면 **첫 측정은 항상 (a) 같은 EP의 FP32 대비, (b) 한 단계 아래 EP 대비 — 두 가지 상대 속도**로 하라.

> 🔴 함정: latency만 보고 "느리네" 하고 끝내지 마라. **먼저 offload 비율과 subgraph 개수**를 보면 원인이 즉시 보인다. "가속기에 30%만 올라갔고 subgraph가 18개"라는 한 줄이, "느림"이라는 막연함을 즉시 진단으로 바꾼다.

---

## 함정 5 — C++를 피하지 마라

> 관련 단계: [05_tensorrt.md](05_tensorrt.md)(custom plugin), [07_infrastructure.md](07_infrastructure.md)(런타임 통합), [09_roadmap.md](09_roadmap.md) 6~8주

**증상**
- Python으로 다 되는 줄 알았는데, 미지원 op custom plugin·런타임 통합·메모리 최적화 앞에서 막힌다.
- 채용 공고(JD)에 C/C++가 **필수**로 박혀 있는 이유를 실감하게 됨.
- 함정 3에서 "이 op는 plugin이 답"이라는 결론이 났는데, plugin이 전부 C++라 진도가 멈춘다.

**원인 (메커니즘)**
프로덕션 임베디드 추론의 핵심 경로는 C++다. TensorRT custom plugin, 온디바이스 런타임 통합(TIDL-RT, QNN, DRP-AI 런타임 API), 제로카피/메모리 풀 최적화, 전·후처리 커널은 전부 C++/CUDA로 작성한다. Python은 프로토타이핑·오케스트레이션에 좋지만, 칩 위 실행은 C++가 지배한다.

TensorRT 10.x 기준 구체적 벽: plugin API가 `IPluginV3`로 통일됐고(구 `IPluginV2` 계열은 폐기), plugin은 `IPluginV3OneCore` + `IPluginV3OneBuild` + `IPluginV3OneRuntime`을 다중상속하며, creator는 `IPluginCreatorV3One`을 구현해 plugin registry에 등록해야 한다. 연산 본체는 `enqueue`(CUDA 커널 호출)에 들어간다. 겁나 보이지만 **실제로 손대는 건 `enqueue` 한 함수**이고 나머지는 보일러플레이트다. ([TensorRT custom layers 문서](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html))

**예방 / 대응**
- Python 프로토타입과 **병행해** C++ 최소 예제를 일찍 붙여본다(미루면 마지막에 병목).
- 벤더/커뮤니티 plugin/런타임 **템플릿을 복사**하고 `enqueue`/forward만 교체하는 식으로 진입 장벽을 낮춘다.
- 빌드 체계(CMake)·디버깅(gdb/compute-sanitizer)·프로파일링(Nsight)을 [05_tensorrt.md](05_tensorrt.md)/[07_infrastructure.md](07_infrastructure.md) 기준으로 갖춘다.
- IPluginV3의 보일러플레이트는 자동 생성이 아니므로, **동작하는 예제 저장소에서 시작**한다(빈 스켈레톤에서 시작 금지).

**디버깅 절차 — C++ plugin/런타임 통합 진입 체크리스트**

아래를 **위에서부터 순서대로** 통과한다. 한 칸도 건너뛰지 말 것 — 대부분의 좌절은 "내 코드를 먼저 넣어서" 생긴다.

1. **환경/빌드**
   - [ ] TensorRT 헤더/라이브러리 경로가 잡히고, `sample_*`(예: `sampleOnnxMNIST`)가 **그대로 빌드·실행**되는가?
   - [ ] CMake로 최소 실행 파일이 링크되는가(`-ltensorrt`/`-lnvinfer`, CUDA 링크)?
2. **템플릿 검증(내 코드 넣기 전)**
   - [ ] 커뮤니티/벤더 **IPluginV3 예제**([TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example) 등)가 그대로 빌드·등록·실행되는가?
   - [ ] `REGISTER_TENSORRT_PLUGIN`(또는 registry 등록)으로 plugin이 로드되고, 엔진 빌드 시 인식되는가?
3. **내 op 이식**
   - [ ] 예제의 `enqueue`(CUDA 커널 호출부)만 내 연산으로 교체했는가(나머지 보일러플레이트 유지)?
   - [ ] 입출력 dtype/shape 계약(`getOutputDataTypes`/`getOutputShapes`)이 내 op와 맞는가?
4. **정합성/안전성**
   - [ ] plugin 포함 엔진 출력이 reference(ONNX Runtime)와 허용 오차 내인가(`polygraphy run ... --trt --onnxrt`)?
   - [ ] `compute-sanitizer`로 메모리 오류/레이스가 없는가(`compute-sanitizer --tool memcheck ./my_app`)?
5. **재현성**
   - [ ] 빌드가 CI에서 재현되는가([07_infrastructure.md](07_infrastructure.md))?
   - [ ] plugin 버전/네임스페이스가 문서화되어 다른 사람이 빌드할 수 있는가?

빠른 안전성 점검(복붙):
```bash
# CUDA 메모리 오류/레이스 탐지 — plugin이 조용히 틀린 값을 낼 때 1순위 도구
compute-sanitizer --tool memcheck ./my_app     # out-of-bounds/leak
compute-sanitizer --tool racecheck ./my_app    # shared memory race
```

> 💡 팁: C++는 "전부 새로 배우기"가 아니라 "벤더 템플릿 위에서 `enqueue` 한 함수 고치기"부터다. 8주차에 딱 한 개의 op로 성공시키면 벽이 사라진다. IPluginV3의 다중상속 구조에 겁먹지 말고, **동작하는 예제를 먼저 돌리고** 그 위에서 연산만 갈아끼워라.

---

## 2) 함정 요약표

| # | 함정 | 한 줄 증상 | 첫 번째 확인 | 관련 문서 | 실측 사례 |
|---|------|-----------|--------------|-----------|-----------|
| **0** | **평가셋이 작다** | 숫자는 나오는데 **셋을 바꾸면 결론이 바뀐다** | FP32 top-1을 **모델 카드 공식값과 대조** + `eval_power_check.py` | [03](03_quantization_theory.md) 4.4 | [1,000 vs 50,000장](../logs/stage1_real_imagenet_report.html) |
| 1 | 캘리브 데이터가 전부 | 특정 조건(야간/역광)만 급락 | 조건별 분리 정확도 + `calib_coverage.py` | [03](03_quantization_theory.md) | 커버리지 200→1000: p=0.639 (영향 없음) |
| 1-b | 고른 캘리브 방법이 실행 안 됨 | 옵션을 바꿔도 결과가 **똑같은데 시간만 늘어남** | `compare_scales.py` — 두 산출물의 activation scale 비교 | [03](03_quantization_theory.md) | [ORT Entropy 퇴화](../logs/stage1_quantization_log.html#s4) (md5까지 동일) |
| 2 | 전처리 **불일치** | 에러 없이 정확도만 죽음 | `preprocess_parity.py` 바이트 비교 | [03](03_quantization_theory.md), [05](05_tensorrt.md) | — |
| **2-b** | 전처리가 **일관되게 비표준** | 전 구간 일치하는데 정확도가 통째로 낮게 깔림 | 공식값 재현 여부(±0.1%p) + `weights.transforms()` | [03](03_quantization_theory.md) 4.2 | squash→tv **+1.07%p**, p=1.6e-14 |
| 3 | export ≠ 칩 동작 | 컴파일/실행에서 op 미지원 | `polygraphy inspect capability` | [04](04_transformer_quantization.md), [06](06_multi_soc.md) | — |
| 4 | fallback 지옥 | 가속기인데 더 느림 (EP 목록엔 잡히고 결과도 맞음) | offload 비율·subgraph 개수 + **FP32 대비 & 하위 EP 대비 latency**·`ep_offload_check.py` | [05](05_tensorrt.md), [06](06_multi_soc.md) | [TRT EP 무음 폴백](../logs/stage1_quantization_log.html#s8) (TRT 3.06 > CUDA 1.81 ms) |
| 5 | C++ 회피 | plugin/런타임에서 막힘 | 벤더 IPluginV3 template 빌드 여부 | [05](05_tensorrt.md), [07](07_infrastructure.md) | — |

> **0은 다른 함정들의 전제**다. 1~5는 모두 "정확도가/속도가 이상하다"를 출발점으로 삼는데, 그 정확도를 못 믿으면 어느 것도 진단할 수 없다. 그래서 순서상 맨 앞에 둔다.

> 1-b는 별개의 함정이 아니라 **함정 1의 다른 축**이다. 함정 1이 "데이터가 범위를 못 봤다"면 1-b는 "그 데이터로 범위를 정하는 알고리즘이 안 돌았다"이다. 둘 다 최종 증상은 "activation scale이 잘못됐다"로 같아서, scale을 의심하게 되면 두 개를 함께 본다.

> 2-b도 마찬가지로 **함정 2의 다른 축**이다. 2가 "구간마다 전처리가 다르다"(→ 바이트 비교로 잡힘)면, 2-b는 "전 구간이 같은데 그 하나가 표준이 아니다"(→ 바이트 비교로 **안 잡히고** 공식값 대조로만 잡힘)이다.

---

## 실무 체크리스트 (양자화 전/후 반드시 확인)

프로젝트마다 아래를 복사해 채운다. `design_rules.md`([07_infrastructure.md](07_infrastructure.md), [09_roadmap.md](09_roadmap.md) 12주)에 그대로 편입 가능. 각 항목 옆의 **근거**는 "왜 이걸 확인하는가"이며, 대응하는 함정 번호를 붙였다.

### 양자화 전 (Pre-quantization)

| 확인 | 근거(왜) | 함정 |
|------|----------|------|
| - [ ] **평가셋이 잡으려는 차이를 잡을 만큼 큰지** 확인(`eval_power_check.py`, 짝지어진 McNemar 전제) | 못 잡는 셋에서 낸 결론은 부호까지 뒤집힌다. 실측: 1,000장 Δ+0.40%p → 50,000장 Δ−0.12%p. | 0 |
| - [ ] FP32 **baseline 정확도/지연**을 측정·기록, **사전학습 모델이면 공식값과 ±0.1%p 안에서 일치**시킴 | baseline 없으면 "손실"을 정의할 수 없다. 공식값 재현은 tar·라벨·전처리·인덱스 규약의 **동시 검증**이기도 하다. | 0·2-b |
| - [ ] 캘리브 셋이 운영 분포를 대표(조건별 표본, 수백 장, batch≠1) | 캘리브가 안 본 분포는 clipping → 특정 조건 급락. 다만 **장수·클래스 커버리지는 수백 장에서 포화**한다(실측: 200→1000클래스 p=0.639). | 1 |
| - [ ] 학습·캘리브·추론 전처리가 **동일 코드/동일 상수**(mean·std·보간·채널순서·레이아웃) | 분포가 어긋나면 scale이 엉키고 INT8에서 정확도 샘. shape 맞아 에러 없음. | 2 |
| - [ ] 그 "동일한 전처리"가 **모델이 배포한 표준**인지 확인(`weights.transforms()`) | 전 구간 일치해도 비표준이면 통째로 낮게 깔린다. 실측: squash vs tv **1.07%p**(양자화 손실의 9배). | 2-b |
| - [ ] calibration cache가 **현재 전처리로** 생성됨(전처리 바꿨으면 캐시 삭제) | 캐시는 전처리에 종속. 옛 캐시 재사용 시 scale 불일치. | 2 |
| - [ ] 타깃 백엔드의 **지원 op 목록** 확인, 모델 op가 화이트리스트에 듦 | 미지원 op는 빌드 실패 또는 fallback. export 후가 아니라 설계 시 확인. | 3·4 |
| - [ ] 타깃 EP의 **양자화 dtype·대칭성** 요구를 앎(x86 CPU/VNNI → `QUInt8` 비대칭 / GPU·TensorRT → `QInt8` 대칭) | 스킴이 안 맞으면 **에러 없이 전부 폴백**한다. 실측: 같은 INT8이 TRT EP에서 3.06 ms vs 0.51 ms(**6배**). 대칭 강제의 정확도 대가 −0.29%p(p=9.2e-5)도 예산에 넣어라. ([ONNX Runtime 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)) | 3·4 |
| - [ ] dynamic shape 필요 여부 확정(QNN 등은 dynamic shape 미지원 → 고정 shape export) | dynamic shape가 세션 생성 자체를 막는 백엔드가 있음. | 3 |

### 양자화 후 (Post-quantization)

| 확인 | 근거(왜) | 함정 |
|------|----------|------|
| - [ ] INT8 정확도를 **전체 + 조건별(야간/역광 등)로 분리** 측정 | 전체 한 숫자는 조건별 급락을 숨긴다. | 1 |
| - [ ] 정확도 결론을 **Δ + 신뢰구간 + 짝지어진 검정 p값**으로 보고(절대값 단독 금지) | "0.2%p 좋아졌다"는 셋 크기를 모르면 아무 의미가 없다. p>0.05는 "차이 없음"이 아니라 "이 셋으로는 못 봄"이다. | 0 |
| - [ ] 캘리브레이션 **옵션을 바꿨으면 산출물 scale이 실제로 달라졌는지** 확인(`compare_scales.py`) | 라이브러리 기본값이 알고리즘을 무력화할 수 있다. ORT `Entropy`는 기본값에서 MinMax로 퇴화(scale 32/32 동일, 시간만 2.8배). | 1-b |
| - [ ] 파이프라인 경계마다 수치 정합성 검증(PyTorch↔ONNX↔백엔드, Polygraphy) | 어느 경계에서 어긋나는지 좁혀야 원인이 잡힘. | 2·3 |
| - [ ] export 후 **즉시 타깃 컴파일**로 op 지원 확인(`polygraphy inspect capability`, `onnx_export_failures.md`) | 미지원 op를 일찍 알수록 우회 시간이 있다. | 3 |
| - [ ] **offload 비율·subgraph 개수**를 로그로 확인(가속기 이득이 실제로 나는가) | 가속기에 올렸는데 fallback 지옥이면 FP32보다 느림. | 4 |
| - [ ] EP/백엔드가 **실제로 그래프를 가져갔는지** 로그(`log_severity_level` ≤ 2로 되돌리고 정독) + **FP32 대비 latency** + **한 단계 아래 EP 대비 latency**로 확인 | provider 목록에 잡히고 결과가 정확해도 100% 폴백일 수 있다. 실측: TRT EP 3.06 ms(폴백, CUDA 1.81 ms보다도 느림) vs 0.51 ms(정상, CUDA 2.11 ms의 1/4). | 4 |
| - [ ] custom plugin/런타임 통합의 C++ 경로가 빌드·검증됨(compute-sanitizer 통과) | 칩 위 실행은 C++. plugin이 조용히 틀린 값을 낼 수 있음. | 5 |
| - [ ] 위 결과가 **회귀 하네스**로 자동 재측정됨([09_roadmap.md](09_roadmap.md) 12주) | 한 번 잡은 함정이 다음 커밋에서 되살아나는 걸 막는다. | 전부 |

> 💡 팁: 문제가 생기면 이 순서로 의심하라 — **(0) 내가 지정한 옵션/EP가 실제로 적용됐는지**(산출물 diff·로그) → (1) 전처리 일치 → (2) 캘리브 대표성 → (3) op 지원/정합성 → (4) offload 비율. 대부분 상위 세 개에서 잡힌다. 이 순서는 "싸고 흔한 것부터"라서 평균 디버깅 시간이 가장 짧다. (0)이 맨 앞인 이유는, 그게 틀렸으면 아래 셋을 아무리 고쳐도 안 움직이기 때문이다.

---

## 7) 산출물(Deliverables)

- [ ] `pitfall_checklist.md` — 위 체크리스트를 프로젝트에 맞춰 채운 사본.
- [ ] (실습 시) `eval_power_check.py` 출력 — 내 평가셋의 CI와 판정 가능한 차이 크기(함정 0). FP32가 공식값을 ±0.1%p로 재현했는지도 함께 기록.
- [ ] (실습 시) `calib_coverage.py`, `preprocess_parity.py` 실행 결과 로그.
- [ ] (실습 시) `compare_scales.py` 출력 — 캘리브 옵션별 산출물의 activation scale이 실제로 다른지(함정 1-b).
- [ ] (실습 시) `ep_offload_check.py` 출력 — EP별 FP32 vs INT8 p50 비교표(함정 4). 폴백을 발견했으면 로그 원문도 함께.
- [ ] (실습 시) `polygraphy inspect capability --with-partitioning` 리포트(미지원 op 목록).
- [ ] 발견한 실패 사례를 각 산출물 문서(`onnx_export_failures.md`, `four_target_matrix.md`)에 반영.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [ONNX Runtime 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — static/dynamic PTQ, 대표성 있는 캘리브, EP별 대칭성 요구.
- [ONNX Runtime 아키텍처(GetCapability/파티셔닝)](https://onnxruntime.ai/docs/reference/high-level-design.html) — subgraph 분할·CPU fallback 원리.
- [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — 지원 op 부분집합, dynamic shape/Loop·If 미지원, `disable_cpu_ep_fallback`.
- [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) — TRT EP 등록·provider option·서브그래프 위임(함정 4 실측 사례).
- [onnxruntime `calibrate.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py) — `EntropyCalibrater` 기본값 `num_bins=128, num_quantized_bins=128`, `get_entropy_threshold` (함정 1-b의 1차 근거).
- [onnxruntime `quantize.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py) — `calib_extra_options_keys` 화이트리스트 5개(`num_bins` 전달 불가의 근거).
- [microsoft/onnxruntime#9597](https://github.com/microsoft/onnxruntime/issues/9597) — `num_quantized_bins` 기본값 128 이슈. **닫혔으나 기본값은 그대로**(2026-08 확인).
- [NVIDIA TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/) — INT8 calibration, custom plugin. (정본 10.16.x LTS)
- [TensorRT custom layers(IPluginV3)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html) — IPluginV3OneCore/Build/Runtime, IPluginCreatorV3One.
- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) — 백엔드 간 수치 정합성 검증.
- [Polygraphy inspect capability 예제](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect) — ONNX를 TRT 지원/미지원 subgraph로 분할·리포트.
- [TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example) — 자기완결형 IPluginV3 예제(함정 5 진입점).
- [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) — TIDL 오프로드/`deny_list`, subgraph 디버깅.

### 논문
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295
- Jacob et al. (2018), *Quantization and Training of NN for Efficient Integer-Arithmetic-Only Inference*, arXiv:1712.05877
- Xiao et al. (2022), *SmoothQuant*, arXiv:2211.10438

---

## 9) 다음 단계

이 문서는 로드맵의 종점이자, 실무에서 계속 되돌아올 참조 문서다. 로드맵을 돌리다 막히면 이 5개 함정부터 대조하라.

← 이전: [9. 12주 로드맵](09_roadmap.md)

> 전체 인덱스는 [README.md](README.md) 참조(오케스트레이터 작성).
