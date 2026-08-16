# 2. Transformer 양자화 지옥

> 원본 가이드 매핑: **2단계 — Transformer 양자화 지옥 (2~3주·핵심)**
> 예상 소요: **2~3주** (이 가이드의 최난이도·핵심 단계)
> 선행 조건: [1단계 양자화 이론](03_quantization_theory.md) 완료 · PyTorch/ONNX 기본 · GPU 사용 가능한 Ubuntu 22.04

---

## 0) 이 단계에서 무엇을·왜 하는가

CNN 양자화(ResNet/MobileNet INT8)는 사실상 **풀린 문제**다. PTQ 몇 줄이면 top-1이 1% 이내로 떨어진다. 그런데도 임베디드 AI 채용 공고(JD)가 사람을 못 구하는 진짜 이유는 **Transformer**다.

Transformer를 INT8로 내리면 **모델이 그냥 깨진다**. mAP가 반토막 나거나, 애초에 ONNX export 단계에서 100% 실패한다. 이유는 CNN과 근본적으로 다른 4가지 연산 특성 때문이다.

이 단계의 목표는 두 가지다.

1. **왜 Transformer가 INT8에서 깨지는지**를 연산 수준에서 이해하고(LayerNorm/Softmax/GELU/attention matmul), 각 문제의 대응책(SmoothQuant·per-channel·FP16 fallback)을 손에 익힌다. 단순히 "outlier가 문제"라는 구호가 아니라, **outlier 100 하나가 per-tensor scale을 어떻게 망가뜨리는지를 수치로** 계산할 수 있어야 한다.
2. **DETR(또는 BEVFormer-tiny)를 실제로 ONNX export → 실패 → 우회 → INT8 PTQ → 회복**시키는 전 과정을 몸으로 겪고, 그 실패 로그를 `onnx_export_failures.md`로 남긴다. **이 문서가 곧 포트폴리오이자 design rules의 원형**이다. 면접에서 "self-attention을 NPU에 올려봤는가"에 실물로 답하는 자료가 된다.

> 💡 팁: BEV/Occupancy 인식(자율주행)으로 가면 여기에 `grid_sample`, Deformable Attention, `scatter/gather`, dynamic shape라는 **NPU가 극도로 싫어하는 연산**이 추가된다. 이 단계에서 그 지뢰들을 미리 밟아둔다.

> 📌 이 문서의 정본 버전 스택(모든 명령/표는 이 기준): **CUDA 12.8 · onnx 1.18.0 · onnxruntime-gpu 1.23.2 · TensorRT 10.16.x LTS · [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) · ExecuTorch 1.3.x**. 산출물 `onnx_export_failures.md`는 이 문서(2단계)의 실습에서 생성된다.
>
> 🔴 **export 전에 반드시 확인**: 정본 ORT 1.23.2는 **ONNX IR 11 / opset 23까지만** 읽는다(그 상한을 만드는 것이 `onnx==1.18.0` 핀이다 — [0단계 2절](01_environment_setup.md) 참조). 이 문서의 모든 export는 **opset ≤ 23**을 지킨다(실제로는 16~17을 쓴다). `onnx`를 무제한으로 올리면 1.22.0(**IR 13**)이 깔려 export한 모델이 ORT 로드 단계에서 `Unsupported model IR version: 13` 으로 죽는다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] Transformer 4대 문제(LayerNorm·Softmax·GELU·act×act matmul)를 "왜 깨지는가"까지 **수치 예시로** 설명할 수 있다.
- [ ] outlier 100 vs 본체 ±3 상황에서 per-tensor INT8의 **유효 비트 손실을 직접 계산**할 수 있다.
- [ ] SmoothQuant의 스무딩 수식 `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)` 을 **등가 변환에서 유도**하고, α(migration strength) 스윕이 activation/weight 난이도를 어떻게 옮기는지 안다.
- [ ] SmoothQuant를 실제 툴(NVIDIA Model Optimizer 또는 원본 repo)로 적용하고 전후 activation 분포를 비교해봤다.
- [ ] ViT 양자화 3부작(FQ-ViT·PTQ4ViT·RepQ-ViT)의 핵심 메커니즘(PTF·LIS / twin uniform·Hessian / scale reparam)과 arXiv ID를 안다.
- [ ] `facebook/detr-resnet-50`을 torch로 로드하고 `torch.onnx.export`를 **시도해서 실패시켰다**.
- [ ] 실패 원인(`grid_sampler`/`aten::unsupported`/dynamic shape/opset)을 **로그 원문 5~8개**로 채집하고 하나씩 우회했다(opset 상향·op 치환·shape 고정).
- [ ] deformable attention을 **표준 op(grid_sample+bilinear+gather)로 분해**하거나 plugin으로 감싸는 두 경로를 구분해 설명할 수 있다.
- [ ] INT8 PTQ로 mAP 폭락을 재현하고(DETR 실측 42.07→24.02, −42.9%), **op 단위 mixed(`nodes_to_exclude`)로는 회복이 안 된다는 것**과 그 이유(손상이 per-tensor activation 양자화로 망 전체에 분산됨)를 실측으로 설명할 수 있다.
- [ ] **산출물 `onnx_export_failures.md`** 를 (템플릿의 예시 항목까지 채워) 작성했다.

---

## 2) 배경 이론 — 왜 Transformer가 INT8에서 깨지는가

### 2.1 4대 문제 (이유까지)

| 연산 | 무엇이 문제인가 | 근본 원인 | 대응 |
|------|----------------|-----------|------|
| **LayerNorm** | 채널별 activation **outlier가 극심**. 특정 채널만 값이 수십~수백 배 큼 | per-tensor scale이 소수의 outlier 채널에 **끌려가** 나머지 채널의 유효 비트가 소실. LayerNorm은 채널축으로 정규화하므로 채널 간 분산이 그대로 남음 | **SmoothQuant**(outlier를 weight로 이전), **per-channel** quant, LayerNorm 자체는 **FP16 유지** |
| **Softmax** | 출력이 [0,1]에 몰리고 대부분 0 근처. 내부 `exp`의 **dynamic range가 폭발** | attention map은 극단적 비균일 분포(soft one-hot). 선형 INT8 8단계로는 0.001과 0.99를 동시에 표현 불가. `exp(x)`의 입력 범위도 넓음 | 보통 **INT8 금지 → FP16 fallback**. 정수화하려면 Log-Int-Softmax(FQ-ViT) 같은 **비균일(로그) 양자화** |
| **GELU** | 음수 구간이 부드러운 비선형(0으로 완전히 죽지 않음) | ReLU와 달리 음수쪽 정보가 살아있어 **비대칭·비선형**. 대칭 선형 INT8이 음수 근처 곡률을 못 담아 정보 손실 | **LUT(look-up table) 근사**, twin-uniform quant(PTQ4ViT), 또는 재학습 여유가 있으면 **ReLU/HardSwish로 교체** |
| **QKᵀ / (softmax·V) matmul** | **activation × activation** 곱 (weight×activation이 **아님**) | 일반 Linear는 weight가 고정(static)이라 캘리브레이션이 쉬움. 그러나 attention의 두 matmul은 **양쪽 다 런타임에 변하는 dynamic** 텐서 → 캘리브레이션 대상이 2배, outlier도 양쪽에서 발생 | 양쪽 **별도 캘리브레이션**, 동적 범위가 크면 이 matmul만 **FP16**으로 |

> 🔴 함정: "CNN에서 되던 PTQ 스크립트를 Transformer에 그대로 돌린다" → 십중팔구 mAP/accuracy가 폭락한다. 위 4개 연산을 **선택적으로 FP16으로 빼는 mixed precision**이 사실상 필수다.

#### 2.1.1 붕괴 ①·LayerNorm outlier를 **수치로** — per-tensor scale이 왜 끌려가는가

INT8 대칭 양자화의 scale은 텐서의 최댓값 절댓값(absmax)으로 결정된다:

```
scale = max(|x|) / 127        # INT8 대칭, [-127, 127]
```

**한 채널만 outlier가 있는 상황을 가정하자.** LayerNorm 출력 한 토큰의 activation 벡터가 대부분 `±3` 범위인데, **딱 한 채널의 값이 100**이라고 하자(ViT/BERT류에서 실측되는 전형적 패턴).

- per-tensor scale = `100 / 127 ≈ 0.787`
- 본체(±3) 값들이 정수 grid에 매핑되면: `round(3 / 0.787) = round(3.81) = 4`. 즉 **본체 전체가 정수 -4 ~ +4, 실질 9단계**만 사용한다.
- INT8은 원래 256단계인데, outlier 하나 때문에 **본체는 약 9/256 ≈ 3.5%의 표현력만** 쓴다. log2(9) ≈ **3.2비트** — 8비트를 쓰고 있지만 실효는 3비트대다.
- 반대로 본체 기준으로 scale을 잡으면(`scale = 3/127 ≈ 0.0236`) outlier 100은 `round(100/0.0236)=4237` → INT8 상한 127에 **clip**되어 100이 아니라 ~3으로 뭉개진다. outlier가 attention 경로에서 중요한 신호라면 이 clip이 정확도를 죽인다.

**핵심:** outlier를 살리려 하면 본체가 3비트로 뭉개지고, 본체를 살리려 하면 outlier가 clip된다. **per-tensor로는 둘 다 못 산다.** 이래서 (1) per-channel(채널마다 scale 분리) 또는 (2) SmoothQuant(outlier를 weight로 이전)가 필요하다.

> 💡 왜 LayerNorm에서 유독 심한가: LayerNorm은 **채널(feature)축이 아니라 토큰별로 정규화**하므로, 채널 간 스케일 편차(inter-channel variation)를 없애주지 못한다. 특정 채널이 구조적으로 큰 값을 갖는 경향이 그대로 남아 outlier 채널이 고정적으로 나타난다. (FQ-ViT가 지적한 "serious inter-channel variation".)

#### 2.1.2 붕괴 ②·Softmax와 `exp`의 dynamic range 폭발

attention 확률은 `p_i = exp(z_i) / Σ exp(z_j)` 로, **soft one-hot**(하나가 0.9, 나머지가 0.001)에 가까운 극단적 비균일 분포다.

- 표현해야 하는 값의 범위: 예컨대 `0.9`와 `0.0005` — 비율 **1800:1**. INT8 선형 grid(균일 간격)로 [0,1]을 자르면 간격은 `1/255 ≈ 0.0039`. 즉 **0.0005, 0.0009, 0.0013 같은 작은 확률은 전부 0 또는 같은 bin**으로 뭉개진다. 그런데 이 "작은 확률들의 꼬리"가 검출(특히 DETR decoder의 cross-attention)에서 미세한 위치 신호를 나른다.
- 내부 `exp(z)`의 입력 z(pre-softmax logit)도 범위가 넓다. z가 `[-15, +8]`이면 `exp`는 `[3e-7, 3000]` → **10자리 dynamic range**. 정수 8비트로는 애초에 담을 그릇이 안 된다.

→ 그래서 Softmax는 대개 **INT8 금지, FP16 유지**가 정석이다. 굳이 정수화하려면 균일 grid가 아니라 **로그 스케일**로 작은 값에 해상도를 몰아줘야 한다(FQ-ViT의 Log-Int-Softmax, RepQ-ViT의 log√2). 로그 양자화는 "0.9와 0.0005의 비율"을 지수로 표현하므로 작은 확률의 상대오차를 지킨다.

#### 2.1.3 붕괴 ③·GELU 음수 구간의 비대칭 정보 손실

`GELU(x) = x · Φ(x)`. ReLU는 음수를 0으로 죽이지만, GELU는 음수 쪽에 **작지만 0이 아닌 곡선**을 남긴다(최소값 약 `-0.17` 부근, x≈-0.75에서).

- post-GELU 분포는 **비대칭**이다: 음수 쪽은 `[-0.17, 0]`의 좁은 구간에 몰리고, 양수 쪽은 `[0, +∞)`로 넓게 퍼진다.
- 대칭 per-tensor INT8은 scale을 양수 최댓값(예: +8)에 맞추므로 `scale = 8/127 ≈ 0.063`. 그러면 음수 구간 `[-0.17, 0]`은 정수 `round(-0.17/0.063) = -3 ~ 0`, **딱 3~4단계**로만 표현된다. 이 좁은 음수 곡률(비선형의 핵심 정보)이 거의 소실된다.
- 이게 twin-uniform(PTQ4ViT)의 동기다: **양수용 scale과 음수용 scale을 분리**하면, 음수 구간을 `[-0.17, 0]` 자체에 맞춘 촘촘한 grid로 담을 수 있다.

> 📊 세 붕괴의 공통 구조: **"한 텐서 안에 스케일이 다른 두 세계가 공존"**한다(outlier vs 본체 / 큰 확률 vs 꼬리 / 넓은 양수 vs 좁은 음수). 균일·per-tensor·대칭이라는 INT8의 3대 가정이 전부 깨진다. 대응책은 전부 "그 두 세계를 분리"하는 것 — per-channel(공간 분리), 로그(스케일 분리), twin-uniform(부호 분리), SmoothQuant(활성↔가중치 분리).

#### 2.1.4 붕괴 ④·activation×activation matmul — 캘리브레이션 대상이 2배

일반 Linear `Y = X·W`는 W가 학습 후 **고정(static)**이라, W의 양자화 scale은 캘리브레이션 없이 weight만 보고 정한다. 활성 X 하나만 런타임 통계로 잡으면 된다.

attention의 두 matmul `QKᵀ`, `(softmax)·V`는 **양쪽 피연산자가 전부 런타임에 변하는 activation**이다.

- 캘리브레이션해야 할 dynamic 텐서가 Q, K, (softmax P), V로 **2배 이상** 늘고, outlier가 양쪽 모두에서 발생한다.
- 특히 Q·K는 head마다 스케일이 달라, per-tensor 한 scale로 잡으면 특정 head가 뭉개진다. 그래서 이 matmul은 **INT8 이득이 작고 위험은 커서**, 실무에서 가장 먼저 FP16으로 빼는 대상이다.

---

### 2.2 SmoothQuant — activation outlier를 weight로 migrate

핵심 통찰: **activation은 양자화가 어렵고(outlier 많음), weight는 쉽다(분포가 균일).** 그러면 어려움의 일부를 activation에서 weight로 **수학적으로 등가인 변환**으로 옮기면 된다.

#### 2.2.1 등가 변환에서 수식 유도

Linear 연산 `Y = X · W` (X: activation `[T, Cin]`, W: weight `[Cin, Cout]`)를 입력 채널별 스케일 `diag(s)` (`s ∈ R^{Cin}`)로 **항등변환**한다:

```
Y = X · W
  = X · (diag(s) · diag(s)^-1) · W          # diag(s)·diag(s)^-1 = I 삽입 (값 불변)
  = (X · diag(s)^-1) · (diag(s) · W)
  = X̂ · Ŵ
```

- `X̂ = X · diag(s)^-1` : 채널 j의 activation을 `s_j`로 **나눔** → outlier 채널을 눌러 X̂가 양자화 쉬워짐
- `Ŵ = diag(s) · W` : 채널 j의 weight 행을 `s_j`로 **곱함** → weight가 그만큼 어려워짐(원래 여유가 있어 감당 가능)
- `diag(s)^-1`은 대개 **직전 LayerNorm의 γ/β(또는 직전 Linear의 weight)에 흡수(fuse)**되므로 추론 시 추가 연산·추가 레이어가 없다(offline 변환).

**이제 s를 어떻게 고르나?** X̂와 Ŵ의 양자화 난이도를 **동시에** 낮추고 싶다. Xiao et al.(2022)은 채널 j에 대해:

```
s_j = max(|X_j|)^α / max(|W_j|)^(1−α)
```

- `max(|X_j|)`: 채널 j의 activation absmax(캘리브레이션 데이터에서 측정)
- `max(|W_j|)`: 채널 j에 대응하는 weight absmax
- `α` (**migration strength**, ∈[0,1]): 난이도를 activation→weight로 **얼마나 넘길지**
  - `α=0`이면 `s_j = 1/max(|W_j|)` → weight만 정규화(activation 그대로, 어려움이 전부 activation에 남음)
  - `α=1`이면 `s_j = max(|X_j|)` → activation을 완전히 눌러 어려움을 전부 weight로 밀어냄(이번엔 weight가 깨짐)
  - `α=0.5`이면 식이 대칭이 되어 **`s_j = √( max(|X_j|) / max(|W_j|) )`** — 두 세계의 absmax를 기하평균으로 맞춰 균형점이 된다.

> 📐 왜 α=0.5가 "sweet spot"인가: `s_j`로 나눈 뒤 X̂ 채널 absmax는 `max|X_j|/s_j = max|X_j|^{1-α}·max|W_j|^{α}`, Ŵ 채널 absmax는 `max|W_j|·s_j = max|X_j|^{α}·max|W_j|^{1-α}`. α=0.5면 둘 다 `√(max|X_j|·max|W_j|)`로 **정확히 같아진다.** activation과 weight의 양자화 난이도(absmax)를 같게 만드는 지점이라 대부분 모델에서 균형이 좋다.

#### 2.2.2 α 스윕 효과 (해석 틀)

α는 캘리브레이션셋에서 **그리드 서치로 전체 양자화 손실을 최소화**해 고른다. 아래는 "어느 방향으로 움직이면 무엇이 좋아지고 무엇이 나빠지는가"의 감각표다(수치는 예시 경향).

| α | 무슨 일이 일어나나 | activation 양자화 | weight 양자화 | 언제 |
|---|--------------------|-------------------|---------------|------|
| 0.0 | 스무딩 없음(거의) | ❌ 매우 어려움(outlier 그대로) | ✅ 쉬움 | 대조군(baseline) |
| 0.3 | 약하게 이전 | 🔸 여전히 outlier 잔존 | ✅ 쉬움 | activation outlier가 약할 때 |
| **0.5** | **기하평균 균형** | ✅ 좋음 | ✅ 좋음 | **대부분 모델 권장 기본값** |
| 0.75 | 강하게 이전 | ✅ 매우 좋음 | 🔸 weight가 어려워짐 | outlier 극심(예: GLM-130B) |
| 1.0 | 전부 weight로 | ✅ 완벽 | ❌ weight 붕괴 | 사용 금지 영역 |

- 실제 LLM 튜닝값은 대략 **0.6~0.9** 범위(모델·레이어별로 다름; Llama-2 계열 0.85~0.9, Mistral 0.8 등 보고가 있음).
- ViT/DETR의 LayerNorm outlier도 원리가 같아 α≈0.5에서 시작해 스윕한다.

#### 2.2.3 전후 activation 분포 (무엇을 눈으로 확인할 것인가)

SmoothQuant 적용 전/후에 **같은 레이어의 activation absmax를 채널별로** 찍어 비교하는 게 핵심 검증이다.

- **적용 전**: 채널 absmax 히스토그램이 대부분 `~3`에 몰려 있고 **몇 개 채널만 50~100으로 튀는** 뾰족한 spike.
- **적용 후**: `s_j`로 나눈 X̂의 채널 absmax가 **10~20 수준으로 평탄화**(spike가 사라짐). 대신 Ŵ의 채널 absmax는 소폭 상승했지만 여전히 INT8 범위에 안착.
- 정량 지표: 채널 absmax의 **max/median 비율**이 (예) 30배 → 3배로 줄면 성공. per-tensor scale이 더 이상 소수 채널에 끌려가지 않는다.

> 💡 팁: SmoothQuant는 원래 LLM(W8A8)용으로 나왔지만 **원리는 ViT/DETR의 LayerNorm outlier에도 그대로 적용**된다. "activation outlier를 채널 스케일로 눌러 weight로 넘긴다"가 핵심. 적용 코드는 아래 4.4에 있다.

---

### 2.3 ViT 양자화 3부작 (필독) — 메커니즘까지

Transformer를 INT8/INT4로 내리는 계보. arXiv ID로 원문 접근(`https://arxiv.org/abs/<ID>`). 세 논문은 2.1의 붕괴들을 각각 다른 각도로 공략한다.

| 논문 | 연도 | arXiv | 핵심 아이디어 | 대표 결과 |
|------|------|-------|--------------|-----------|
| **FQ-ViT** | 2021 (IJCAI'22) | `2111.13824` | **완전 양자화** ViT 최초. LayerNorm 채널 편차용 **Power-of-Two Factor(PTF)**, Softmax용 **Log-Int-Softmax(LIS)** (로그 양자화 + BitShift 추론) | 완전 양자화에서 정확도 하락 약 1% (ViT-L ImageNet top-1 84.89%) |
| **PTQ4ViT** | 2021 (ECCV'22) | `2111.12293` | post-Softmax·post-GELU의 비가우시안 분포를 **twin uniform quantization**으로 처리. 캘리브레이션 지표를 MSE 대신 **Hessian-guided** 로 | **8-bit에서 0.5% 미만** 하락 (near-lossless) |
| **RepQ-ViT** | 2022 (ICCV'23) | `2212.08254` | 캘리브레이션엔 복잡한 quantizer(LayerNorm=channel-wise, Softmax=log√2), 추론엔 **scale reparameterization**으로 layer-wise·log2로 단순화 | **4-bit PTQ**를 실사용 수준으로 끌어올림, 재학습·복잡한 reconstruction 불필요 |

#### 2.3.1 FQ-ViT — PTF와 LIS

- **Power-of-Two Factor (PTF)** — LayerNorm의 inter-channel variation(2.1.1) 대응. per-channel처럼 채널마다 **완전히 다른 실수 scale**을 주면 정확하지만, 채널마다 scale이 다르면 뒤따르는 matmul에서 정수 누산이 깨져 하드웨어가 싫어한다. PTF는 절충안으로 **레이어 공통 scale `s` 하나 + 채널별 정수 지수 `α_c` (2의 거듭제곱 인자)** 만 둔다: 채널 c의 실효 scale = `s · 2^{α_c}`. `2^{α_c}` 곱은 **BitShift**로 처리되므로 정수 파이프라인이 유지된다. "채널별 미세조정은 하되 shift로만" → 정확도와 하드웨어 친화성을 동시에.
- **Log-Int-Softmax (LIS)** — Softmax의 dynamic range 폭발(2.1.2) 대응. attention 확률을 **균일 grid가 아니라 log2 스케일로 양자화**해 작은 확률(꼬리)에 해상도를 몰아준다. 게다가 곱셈 대신 **BitShift**로 정수만으로 attention·V 곱을 수행 → **4-bit attention**을 실현. "비균일 분포는 비균일(로그) grid로 담는다"의 정석.

#### 2.3.2 PTQ4ViT — twin uniform과 Hessian-guided

- **Twin uniform quantization** — post-Softmax·post-GELU가 가우시안이 아니라는 관찰(2.1.3)에서 출발. 한 텐서를 **두 개의 uniform quantizer(R1, R2)로 분할**한다. 예: post-GELU라면 R1이 **좁은 음수 구간**(`[-0.17, 0]`)을 촘촘히, R2가 **넓은 양수 구간**을 담당. post-Softmax라면 R1이 0 근처의 조밀한 작은 확률, R2가 큰 확률. **MSB(최상위 비트)를 range flag로 써서** 두 영역을 구분하고 나머지 비트로 값을 표현 → 곱셈을 bit-shift로 대체해 하드웨어 효율 유지.
- **Hessian-guided 캘리브레이션** — scale을 고를 때 MSE/코사인 거리는 ViT에서 부정확하다는 관찰. 대신 **손실 함수의 2차 정보(Hessian)로 가중한 양자화 오차**를 최소화하는 scale을 찾는다. "출력 오차"가 아니라 "**최종 loss에 미치는 영향**"을 기준으로 삼아, 민감한 레이어의 scale을 더 정확히 잡는다. 작은 추가 연산으로 캘리브레이션 정확도를 끌어올림.

#### 2.3.3 RepQ-ViT — 캘리브레이션과 추론의 분리(scale reparameterization)

- 문제의식: 정확한 quantizer(LayerNorm=**channel-wise**, Softmax=**log√2**)는 정확하지만 추론이 무겁고, 하드웨어가 좋아하는 quantizer(layer-wise, log2)는 빠르지만 부정확하다.
- 해법: **캘리브레이션엔 복잡·정확한 quantizer를, 추론 직전엔 수학적으로 등가인 단순 quantizer로 "reparameterize"**한다. 구체적으로 channel-wise (s, z)를 **layer-wise scale + 인접 LayerNorm 파라미터 흡수**로 바꾸고, log√2를 log2로 변환한다. SmoothQuant의 "직전 레이어에 흡수" 트릭과 사상이 같다 — **정확도는 복잡한 quantizer로 벌고, 비용은 흡수로 없앤다.**
- 성과: 재학습·복잡한 reconstruction 없이 **4-bit PTQ**를 실사용 수준으로. PTQ만으로 4-bit ViT를 여는 계열의 대표작.

> 읽는 순서 추천: SmoothQuant(`2211.10438`, 필독) → FQ-ViT → PTQ4ViT → RepQ-ViT. 이후 DETR/BEVFormer 계열 양자화 최신 논문으로. **decoder(cross-attention)가 특히 까다롭다** — FQ-PETR(`2502.15488`, 2025, 멀티뷰 3D의 완전 양자화 PE) 같은 최신 후속을 참고.

> ⚠️ 확인 필요: DETR/BEV decoder 전용 양자화 논문은 2026년에도 활발히 갱신 중이다. arXiv에서 "quantization DETR decoder" / "quantization BEVFormer" 로 최신 것을 재검색할 것.

---

## 3) 환경·도구 준비

> 아래는 **보드 없이 데스크톱 GPU만으로** 전부 실습 가능하다. NPU 배포는 [4단계 멀티 SoC](06_multi_soc.md)에서.

```bash
# 1) 가상환경 (conda 예시)
conda create -n tfquant python=3.10 -y
conda activate tfquant

# 2) PyTorch (CUDA 12.8 빌드) — cu128 인덱스의 정본은 torch 2.11.0+cu128 (0단계 실측)
#    실제 CUDA 버전에 맞는 인덱스 URL은 https://pytorch.org/get-started/locally/ 에서 재확인
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) HuggingFace + ONNX 스택 (정본: onnx 1.18.0 / onnxruntime-gpu 1.23.2)
#    🔴 onnx는 반드시 ==1.18.0 (IR 11). 무제한이면 1.22.0(IR 13)이 깔려 ORT 로드가 깨진다.
#    🔴 ORT는 상한 '<1.27' (1.27+는 PyPI 기본이 CUDA 13) → Python 3.10에서 1.23.2로 해석된다.
pip install "transformers>=4.44" "onnx==1.18.0" "onnxruntime-gpu<1.27" onnxsim
#    🔴 timm 필수: `facebook/detr-resnet-50`의 backbone은 `TimmBackbone`이라, timm이 없으면
#       from_pretrained가 곧바로 죽는다(실측, transformers 5.15.0):
#         ImportError: TimmBackbone requires the timm library but it was not found ...
#       config에 backbone_config가 있어 네이티브처럼 보여도 실제 인스턴스는 timm 백본이다.
pip install timm                          # DETR backbone(TimmBackbone) 필수
#    참고: 위 "transformers>=4.44"는 2026-08 실측에서 5.15.0으로 해석된다(정상 동작 확인).
#    🔴 onnxscript 필수: torch 2.11의 torch.onnx.export는 기본이 dynamo=True이고 그 경로가
#       onnxscript를 요구한다. 없으면 아래 4.2의 dynamo=True 시도가 의도한 export 에러가 아니라
#       "No module named 'onnxscript'"로 죽어서 실습이 성립하지 않는다.
#       (onnxscript 0.7.1은 onnx>=1.17만 요구 → 위 1.18.0 핀을 건드리지 않는다. 실측 확인)
pip install onnxscript
pip install "optimum[onnxruntime]"        # HF 모델 ONNX export/최적화 편의 도구
pip install pillow requests               # 이미지 로드용

# 4) ONNX 그래프 수술 도구 (op 치환/shape 고정에 필수)
pip install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com

# 5) SmoothQuant/PTQ 툴 — 아래 4.4 참고 (택1)
pip install nvidia-modelopt            # NVIDIA Model Optimizer (SmoothQuant 내장)
# 또는 pip install neural-compressor    # Intel Neural Compressor (SmoothQuant 지원)
```

```bash
# 설치 확인 — 버전·CUDA 인식 여부
python - <<'PY'
import torch, onnx, onnxruntime as ort
print("torch      :", torch.__version__, "| CUDA:", torch.version.cuda, "| avail:", torch.cuda.is_available())
print("onnx       :", onnx.__version__)
print("onnxruntime:", ort.__version__)                       # 1.23.2 이어야 함
print("onnx IR    :", onnx.IR_VERSION)                       # 11 이어야 함 (ORT 1.23.2의 상한)
print("providers  :", ort.get_available_providers())          # CUDAExecutionProvider 있어야 함
PY
```

예상 출력(환경에 따라 버전 숫자만 다름):

```
torch      : 2.11.0+cu128 | CUDA: 12.8 | avail: True
onnx       : 1.18.0
onnxruntime: 1.23.2
onnx IR    : 11
providers  : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

> 💡 팁: `torch.onnx.export`는 PyTorch 2.5부터 **`dynamo=True`** 경로(torch.export/FX 기반)가 권장이고, 기존 TorchScript 경로는 legacy다. 다만 **커스텀 op/FakeQuantize export에서는 아직 legacy 경로(`dynamo=False`)가 더 안정적인 경우**가 많다. 실습에서는 **둘 다** 돌려보고 에러를 비교 채집한다.

> ⚠️ 주의: `onnxruntime`와 `onnxruntime-gpu`를 **동시에 설치하면 충돌**한다. GPU 실습에서는 `onnxruntime`를 먼저 `pip uninstall`하고 `onnxruntime-gpu`(정본 1.23.2)만 남긴다. `ort.get_available_providers()`에 `CUDAExecutionProvider`가 없으면 CPU 빌드가 잡힌 것이다.

---

## 4) 단계별 실습 — DETR을 깨뜨리고 되살리기

목표 서사: **export 시도 → 실패 → 로그 채집 → 우회 → INT8 PTQ → mAP 폭락 → mixed precision 회복.** 각 실패를 `onnx_export_failures.md`에 그때그때 기록한다.

> 🔴 **2단계 실측 반영 (2026-08-16 · RTX 3080 · COCO val2017 전량 5,000장 · [실측 리포트](../logs/stage2_detr_quantization_report.html))**: 이 절(4)을 실제로 완주했고, **초안과 어긋난 3가지**를 아래 각 소절에 반영했다.
> 1. **export 경로**: torch 2.11의 `torch.onnx.export`는 **기본이 `dynamo=True`** 다. 그래서 이 절의 export 코드처럼 `dynamo=`를 안 주면 legacy가 아니라 **dynamo 경로**로 돌아 요청 opset을 무시하고 **opset 18·IR 10·external data**(main 2.2MB + `.onnx.data` 166MB)를 낸다(→ 4.2·4.3). 그리고 DETR의 legacy 저-opset 실패는 `grid_sampler`가 아니라 **SDPA**(`aten::scaled_dot_product_attention`)다 — **DETR엔 `grid_sample`이 없다**(grid_sampler 지뢰는 4.6 BEVFormer의 것).
> 2. **폭락은 재현됐다**: mAP **42.07 → 24.02**(−18.05, **−42.9%**), 작은 객체는 **−77%**(mAP_s 0.213→0.049). FP32 실측 42.07이 공개값 42.0과 일치해 계측이 신뢰된다.
> 3. **문서의 회복책(문제 op만 `nodes_to_exclude`)은 실패한다**: attention score matmul 36개를 FP로 빼도 **+0.36 mAP뿐**(24.02→24.38)이고, backbone·transformer **어느 절반을 통째로 FP로 둬도** 회복이 안 된다(bb_fp 23.91 / tf_fp 26.53). 손상은 소수 "문제 op"에 있는 게 아니라 **per-tensor activation 양자화가 망 전체에 분산**돼 있다 → 진짜 레버는 op 제외가 아니라 **SmoothQuant(2.2)·activation 캘리브레이션(2.1.1)**이다. (상세: 4.5)

### 4.1 모델 로드 & 정상 추론 확인 (FP32 baseline)

```python
# detr_load.py — DETR을 로드하고 FP32로 한 장 추론해 baseline을 확보
import torch, requests
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

model_id = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_id)
model = DetrForObjectDetection.from_pretrained(model_id).eval()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 고양이 2마리
img = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=img, return_tensors="pt")   # pixel_values: [1,3,H,W]

with torch.no_grad():
    out = model(**inputs)
print("logits :", out.logits.shape)        # [1, 100, 92]
print("boxes  :", out.pred_boxes.shape)     # [1, 100, 4]
```

예상 출력:

```
logits : torch.Size([1, 100, 92])
boxes  : torch.Size([1, 100, 4])
```

> 💡 팁: DETR(resnet-50)은 **backbone이 CNN, head가 Transformer(encoder+decoder)** 인 하이브리드다. 그래서 "CNN은 되는데 Transformer 파트에서 export가 깨지는" 현상을 **한 모델 안에서** 관찰하기 좋다. `grid_sample`을 직접 쓰진 않지만, 아래 4.6에서 BEVFormer-tiny로 넘어가면 `grid_sample`/deformable attention 지뢰를 만난다.

### 4.2 ONNX export 시도 → 실패 재현

```python
# export_try.py — export 경로별로 에러/산출물을 채집한다 (torch 2.11 실측)
import torch
from export_common import model, inputs   # 4.1에서 만든 것을 재사용한다고 가정

pixel_values = inputs["pixel_values"]

# 🔴 torch 2.11은 torch.onnx.export의 기본이 dynamo=True다. 그래서 dynamo=를 생략하면
#    '낮은 opset으로 legacy 실패를 보려던' 시도조차 dynamo 경로로 돌아 조용히 '성공'해 버린다.
#    legacy 실패를 재현하려면 아래 (A)처럼 dynamo=False를 반드시 명시해야 한다.

# (A) legacy(TorchScript) + 낮은 opset — DETR은 여기서 SDPA 때문에 실패한다
try:
    torch.onnx.export(
        model, (pixel_values,), "detr_legacy_op11.onnx",
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
        opset_version=11, do_constant_folding=True, dynamo=False,
    )
    print("[OK legacy op11]")
except Exception as e:
    print("[FAIL legacy op11]", repr(e))
# 실측 → FAIL:
#   UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention'
#   to ONNX opset version 11 is not supported. Support for this operator was added in version 14 ...
#   (grid_sampler가 아니다 — DETR엔 grid_sample이 없다. 첫 블로커는 SDPA다.)

# (B) dynamo 경로(기본값) — DETR에선 '성공'하지만 요청 opset을 무시한다
try:
    torch.onnx.export(
        model, (pixel_values,), "detr_dynamo_op17.onnx",
        opset_version=17, dynamo=True,   # dynamo= 생략해도 결과 동일(기본 True)
    )
    print("[OK dynamo op17]")
except Exception as e:
    print("[FAIL dynamo]", repr(e))
# 실측 → OK지만 opset 17이 아니라 opset 18로 나온다:
#   W ... Setting ONNX exporter to use operator set version 18 because the requested
#         opset_version 17 is a lower version than we have implementations for
#   (opset 17 다운컨버트도 시도하나 RuntimeError: No Adapter To Version $17 for Resize 로 실패 → 18 유지)
#   산출물: main 2.2MB + detr_dynamo_op17.onnx.data 166.5MB (external data), IR 10
```

**4가지 경로 실측 결과 (2026-08-16, torch 2.11.0+cu128 · [리포트](../logs/stage2_detr_quantization_report.html)):**

| 시도 | kwargs | 결과 | IR | opset | 산출물 |
|------|--------|------|----|-------|--------|
| default op11 | `opset_version=11` (dynamo 생략→기본 True) | ✅ OK | 10 | **18**(요청 무시) | main 2.2MB + `.onnx.data` 166.5MB |
| dynamo op17 | `opset_version=17, dynamo=True` | ✅ OK | 10 | **18**(요청 무시) | main 2.2MB + `.onnx.data` 166.5MB |
| **legacy op11** | `opset_version=11, dynamo=False` | 🔴 **FAIL** | — | — | `aten::scaled_dot_product_attention ... not supported`(v14부터) |
| **legacy op17** | `opset_version=17, dynamo=False` | ✅ OK | 8 | 17 | **단일 파일 170.4MB** |

> 🔴 **핵심 교훈 3가지 (실측)**: ① **dynamo가 기본**이라 `dynamo=`를 안 주면 legacy 실패를 볼 수 없다(조용히 성공). ② **dynamo 경로는 요청 opset을 무시**하고 opset 18로 고정하며(다운컨버트는 `Resize` 어댑터 부재로 실패), 가중치를 **external data로 분리**해 main 파일이 2.2MB로 쪼그라든다 — 배포·검증 파이프라인이 단일 파일을 가정한다면 여기서 어긋난다. ③ **깨끗한 단일 파일(IR 8·opset 17)을 원하면 `dynamo=False` + opset ≥ 17** 이 유일한 경로다. 이 문서의 이후 실습(4.3~4.5)은 전부 이 legacy 단일 파일을 쓴다.

#### 4.2.1 export 실패 카탈로그 (실제 로그 문구 → 원인 → 우회)

**아래 로그 문구는 `onnx_export_failures.md`에 통째로 붙여넣을 것.** 모델·버전에 따라 문구가 조금씩 다르니 **본인 환경에서 실제로 나온 것**을 기록한다(여기 표는 채집 가이드).

| # | 증상(로그 핵심 문구) | 발생 경로 | 원인 | 우회 |
|---|---|---|---|---|
| **0** | `UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not supported. Support for this operator was added in version 14` | **legacy(dynamo=False), opset ≤ 13** | SDPA(attention) symbolic은 **opset 14**부터. **DETR의 실제 첫 블로커**(grid_sampler 아님) | **opset ≥ 17**(dynamo=False). dynamo 경로는 SDPA를 분해해 통과하지만 opset 18로 나감 |
| 1 | `Exporting the operator 'aten::grid_sampler' to ONNX opset version 11 is not supported.` | legacy, opset≤15 | ONNX `GridSample`는 **opset 16**부터 표준 | **opset ≥ 16** (`opset_version=16`/17). **DETR엔 grid_sample이 없다 — 이 행은 4.6 BEVFormer의 것** |
| 2 | `Unsupported: ONNX export of operator GridSample with 5D volumetric input.` | legacy, opset 16/17 | opset 16 `GridSample`는 **4D(2D 샘플링)만** | 5D→4D reshape 후 샘플링, 또는 커스텀 op/plugin. ONNX 표준은 **opset 20**에서 5D 추가 |
| 3 | `torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::<op>' to ONNX ... is not supported.` | legacy | 특정 aten op 미지원(모델별 상이; `aten::unique`, `aten::nonzero` 등) | opset 상향 / 서브그래프를 지원 op로 치환 / **custom symbolic** 등록 |
| 4 | `torch._dynamo.exc.Unsupported: ... could not be traced` / `Failed to export the model with torch.export` | dynamo=True | dynamo가 데이터 의존 제어흐름(control flow)·커스텀 op를 트레이스 못 함 | `dynamo=False`(legacy)로 재시도, 또는 문제 블록을 `torch.cond`/wrapper로 감싸기 |
| 5 | `RuntimeError: Exporting the operator ... with FakeQuantize is not supported` / QDQ 노드 export 실패 | dynamo=True, 양자화 그래프 | dynamo 경로의 QDQ/FakeQuantize export가 아직 불안정 | 양자화 그래프 export는 **legacy(`dynamo=False`)** 로 |
| 6 | `Type Error: Type 'tensor(int64)' of input parameter ... is invalid` (ORT 로드 시) | 로드/추론 | export는 됐으나 dtype 불일치(예: index가 int64인데 backend가 int32 기대) | `onnxsim`/GraphSurgeon로 cast 삽입, opset 상향 |
| 7 | `[ShapeInferenceError] ... Dynamic shape ... inference failed` / downstream shape 경고 | 로드/컴파일 | DETR은 입력 H,W 가변 → dynamic axes가 ORT/TRT/NPU에서 문제 | 아래 4.3처럼 **입력 shape 고정** 또는 명시적 `dynamic_axes` |
| 8 | `In node ... GridSample ... attribute 'align_corners' ...` / opset mismatch | 로드 | export opset과 ORT가 기대하는 opset 속성 불일치 | export opset과 `onnxruntime` 버전(정본 1.23.2 → **opset ≤ 23**)이 지원하는 opset을 맞추기 |

> 🔴 함정: "일단 export만 되면 끝"이 아니다. export가 성공해도 **NPU 컴파일 단계에서 다시 깨진다**(4.6, 6절). 그래서 실패 로그를 export/컴파일 **양쪽 모두** 남겨야 design rules가 된다.

> 💡 legacy vs dynamo 채집 전략: **같은 모델을 `dynamo=False`와 `dynamo=True`로 각각 돌려** 에러를 나란히 기록하라. 둘은 **다른 이유로 다른 지점에서** 깨진다(legacy는 aten symbolic 부재, dynamo는 트레이스/제어흐름). 어느 경로가 이 모델에 맞는지는 이 비교로만 알 수 있다. 양자화 QDQ export는 현재 대체로 **legacy가 안정적**이다.

### 4.3 우회 1 — opset 상향 + 입력 shape 고정

```python
# export_fixed.py — opset 17 + 고정 shape로 안정적인 export
import torch
from export_common import model

dummy = torch.randn(1, 3, 800, 1066)   # DETR 표준 입력 크기 근처로 고정

torch.onnx.export(
    model, (dummy,), "detr_fixed.onnx",
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,                       # 🔴 필수: 생략하면 dynamo(기본)로 돌아 opset 18 +
                                        #    external data가 나온다. dynamo=False라야 opset 17·
                                        #    IR 8·단일 파일(≈170MB)이 나온다(4.2 실측표).
    dynamic_axes=None,                  # 우선 완전 고정으로 성공시킨 뒤,
    # dynamic_axes={"pixel_values": {0: "batch"}},  # 필요 시 batch만 열기
)
print("export OK -> detr_fixed.onnx")
```

```bash
# 그래프 단순화 + 검증
onnxsim detr_fixed.onnx detr_sim.onnx        # 불필요 노드 제거
python -c "import onnx; onnx.checker.check_model('detr_sim.onnx'); print('onnx check OK')"

# Polygraphy로 ONNX Runtime 추론이 실제로 도는지 확인
polygraphy run detr_sim.onnx --onnxrt
```

예상 출력(Polygraphy 성공 시 말미):

```
[I] PASSED | Runtime: ... | Command: polygraphy run detr_sim.onnx --onnxrt
```

> 💡 팁: DETR은 backbone에서 위치 인코딩이 입력 크기에 의존하므로 **shape를 고정**하면 export/컴파일이 훨씬 순탄하다. 임베디드 배포에서 **dynamic shape는 거의 항상 적**이다 — 고정하는 습관을 들인다.

> 🔴 실측 주의 (고정 vs 동적의 상충): **COCO val 전량(5,000장) mAP를 재려면** DETR 전처리가 이미지마다 최단변 800·최장변 ≤1333으로 리사이즈해 **입력 H,W가 장마다 다르다**. 완전 고정 모델로는 이 셋을 못 돌린다. 그래서 4.5의 실측은 같은 legacy(`dynamo=False`)·opset 17로 **H,W·batch를 `dynamic_axes`로 연** 모델을 썼다(정확도 판정용). 배포용(고정)과 정확도 측정용(동적)을 **분리**하는 셈이다. 단, 이 **동적 shape가 4.5의 Percentile/Entropy 캘리브레이션을 깨뜨린다**(장마다 activation 텐서 shape가 달라 히스토그램 수집이 실패) — 상세는 4.5의 "캘리브레이션은 레버가 되는가" 참조.

### 4.4 우회 2 — SmoothQuant 적용 (activation outlier 완화)

INT8 PTQ 전에 SmoothQuant로 LayerNorm outlier를 눌러두면 정확도 하락이 줄어든다(원리는 2.2). **원본 repo** 또는 **프로덕션 툴** 중 택1.

**옵션 A) NVIDIA Model Optimizer (modelopt)** — SmoothQuant를 quant config로 내장 지원(2026 기준 유지·활성. repo: `https://github.com/NVIDIA/Model-Optimizer`).

```python
# smoothquant_modelopt.py — modelopt로 SmoothQuant 캘리브레이션 (실행 골격)
import torch
import modelopt.torch.quantization as mtq
from export_common import model, calib_loader   # calib_loader: 대표 이미지 몇 배치

# INT8 SmoothQuant 프리셋. alpha 기본 0.5(균형점). config는 dict이므로 override 가능
config = mtq.INT8_SMOOTHQUANT_CFG
# alpha 조정이 필요하면(2.2.2 스윕) 아래처럼 override 지점을 실제 키로 확인 후 수정:
# config["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

def forward_loop(m):
    for batch in calib_loader:      # 캘리브레이션 데이터로 activation 통계 수집
        m(batch)                    # batch는 pixel_values 텐서

model = mtq.quantize(model, config, forward_loop)   # 스무딩+QDQ 노드 삽입
mtq.print_quant_summary(model)                       # 레이어별 양자화 요약 출력
print("SmoothQuant calibration done")

# 이후 legacy 경로로 QDQ 그래프를 내보내 TensorRT/ORT로 넘김 (dynamo=False 권장)
torch.onnx.export(model, (torch.randn(1,3,800,1066),), "detr_sq_int8.onnx",
                  opset_version=17, do_constant_folding=True)
```

**전후 activation 분포를 직접 비교**(2.2.3의 검증)하는 최소 스니펫:

```python
# smooth_check.py — 특정 Linear 입력의 채널별 absmax를 전/후로 찍어 spike가 사라졌는지 본다
import torch

def channel_absmax(x):                 # x: [.., Cin]
    return x.abs().amax(dim=tuple(range(x.ndim - 1)))   # [Cin]

acts = {}
def hook(name):
    def _h(mod, inp, out): acts[name] = channel_absmax(inp[0].detach())
    return _h

# 스무딩 적용 '전' 모델과 '후' 모델 각각에서 같은 레이어에 hook을 걸어 1배치 통과 후 비교
# ratio = absmax.max() / absmax.median()  # 30배(전) -> 3배(후) 면 성공
```

예상 관찰:

```
[before] layer=encoder.layers.0.self_attn.q_proj  channel-absmax  max=98.4  median=3.1  ratio=31.7x
[after ] layer=encoder.layers.0.self_attn.q_proj  channel-absmax  max=14.2  median=4.8  ratio=2.96x
```

> ⚠️ 확인 필요: `modelopt`의 프리셋 상수명·API는 버전에 따라 바뀐다(예: `INT8_SMOOTHQUANT_CFG`, `mtq.quantize` 시그니처, alpha 지정 위치). 설치 후 아래로 실제 노출 config를 확인하고, 공식 예제(`NVIDIA/Model-Optimizer` repo의 `examples/`)를 기준으로 맞출 것.
> ```bash
> python -c "import modelopt.torch.quantization as mtq; print([c for c in dir(mtq) if 'CFG' in c])"
> ```
> 2026-07 기준 modelopt는 **SmoothQuant/AWQ/SVDQuant/FP8/INT4** 를 지원한다(INT8 SmoothQuant는 MIT HAN Lab·NVIDIA 공동).

**옵션 B) 원본 SmoothQuant repo (`mit-han-lab/smoothquant`)** — 개념 이해용. `generate_act_scales.py`로 activation 스케일을 뽑고, 그 스케일로 스무딩을 적용한 뒤 W8A8 추론.

```bash
# 원본 repo 흐름 (LLM 예제 기준; 원리 학습용)
git clone https://github.com/mit-han-lab/smoothquant
# 1) 대표 데이터로 채널별 activation max 통계 수집 -> act_scales
python smoothquant/examples/generate_act_scales.py --model-name <hf-model> ...
# 2) 위 스케일로 s_j = max|X|^a / max|W|^(1-a) 를 적용해 스무딩 -> W8A8
```

> 💡 팁: 원본 repo는 **LLM(OPT/LLaMA)** 예제 중심이라 DETR/ViT에 그대로 안 붙는다. **원리(2.2절 수식)만 원본에서 익히고**, 실제 비전 모델 적용은 modelopt/Neural Compressor 같은 툴로 하는 것이 실전적이다.

### 4.5 INT8 PTQ → mAP 폭락 확인 → mixed precision 회복

```python
# ptq_onnxruntime.py — ORT static PTQ (QDQ). 먼저 '전부 INT8'로 깨뜨려 본다
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class Reader(CalibrationDataReader):     # 대표 이미지 N장을 흘려 캘리브레이션
    def __init__(self, samples): self.it = iter(samples)
    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"pixel_values": b}

quantize_static(
    "detr_sim.onnx", "detr_int8.onnx",
    calibration_data_reader=Reader(calib_samples),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,                     # LayerNorm/Conv weight는 per-channel이 유리
)
```

```python
# eval_map.py — COCO val2017 mAP를 pycocotools(COCOeval 'bbox')로 잰다
# 후처리는 HF API 대신 표준 DETR 수식으로 직접(버전 독립):
#   prob = softmax(logits, -1)[..., :91]     # 마지막 92번째 = no-object 드롭
#   score, label = prob.max(-1)              # label = COCO category_id (index==id)
#   box: cxcywh(정규화) → xyxy(절대) × (W,H); 임계값·NMS 없이 100 쿼리 전부 제출
# 🔴 판정용은 val 전량(5,000장)으로. 소표본은 mAP가 흔들려 폭락 크기를 못 가린다.
# 실측 하네스: experiments/stage2_detr/s2_07_coco_eval.py (FP32/INT8/mixed 한 패스 동시 평가)
```

**회복 전략 — mixed precision:** 문제 연산만 FP로 되돌린다. **이것이 초안의 핵심 처방이었는데, DETR 실측에선 통하지 않았다**(아래).

```python
# nodes_to_exclude로 '문제 op'를 INT8에서 제외 (초안 예시)
quantize_static(
    "detr_dyn.onnx", "detr_mixed.onnx",
    calibration_data_reader=Reader(calib_samples),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
    nodes_to_exclude=[  # 🔴 아래 4개 중 3개는 DETR에서 '무효'다(측정으로 확인):
        # "*/Gelu",          # → 매칭 0개. DETR은 GELU가 없다(FFN이 ReLU 63개, Gelu/Erf 0개)
        # "*/Softmax",       # → Softmax 18개 존재하나 ORT QDQ는 Softmax를 애초에 양자화 안 함(무효)
        # "*LayerNorm*",     # → LayerNormalization 31개 존재하나 역시 양자화 대상 아님(무효)
        # 실제로 그래프를 바꾸는 건 이것뿐 ↓ (attention score의 act×act matmul 36개)
        # 정규식: r"/(self_attn|encoder_attn)/MatMul(_1)?$"
    ],
)
```

**측정 결과 (COCO val2017 전량 5,000장 · CUDA EP · MinMax 캘리브 100장 · per-channel QDQ · [리포트](../logs/stage2_detr_quantization_report.html)):**

| 구성 | 무엇을 INT8로 | mAP | mAP50 | mAP_s | mAP_l | vs FP32 |
|------|--------------|-----|-------|-------|-------|---------|
| FP32 baseline | 없음 | **0.4207** | 0.6231 | 0.213 | 0.610 | — (공개값 42.0 재현✓) |
| **전부 INT8** | 190개(Conv 54+MatMul 136) 전부 | **0.2402** | 0.4708 | 0.049 | 0.450 | **−0.1805 (−42.9%)** |
| mixed (attn score 36개 FP) | 위에서 act×act matmul 36개만 제외 | 0.2438 | 0.4736 | 0.048 | 0.457 | −0.1769 (**+0.36 vs INT8**) |

> 🔴 **판정 1 — 폭락은 재현됐다**: 42.07 → 24.02, 절반 가까이(**−42.9%**) 무너진다. 가장 심한 곳은 **작은 객체**로 mAP_s가 0.213→0.049(**−77%**), 큰 객체는 0.610→0.450(−26%)로 상대적으로 버틴다. FP32가 공개값 42.0과 일치하므로 이 폭락은 계측 오차가 아니다.

> 🔴 **판정 2 — 초안의 mixed 처방(문제 op만 FP)은 실패한다**: attention score matmul 36개를 FP로 빼도 **+0.36 mAP**(24.02→24.38)뿐이다. 게다가 초안 예시의 `nodes_to_exclude` 4개 중 3개는 애초에 **무효**다 — DETR엔 **GELU가 없고**(ReLU만), Softmax·LayerNorm은 **ORT QDQ가 Conv/MatMul/Gemm만 양자화**해서 제외할 것도 없다. "4대 문제 연산만 FP16으로 빼면 baseline 근처로 회복"은 **DETR에서 성립하지 않는다.**

**결정적 절제 — 범인은 backbone인가 transformer인가:** 어느 한쪽만 INT8로 내려 본다([`s2_09_quantize_ablation.py`](../experiments/stage2_detr/s2_09_quantize_ablation.py)).

| 구성 | INT8 대상 | mAP | vs FP32 | 해석 |
|------|-----------|-----|---------|------|
| bb_fp | transformer(137)만 INT8, backbone(53 Conv) FP | 0.2391 | −0.1816 | transformer만 내려도 **거의 풀 폭락** |
| tf_fp | backbone(53 Conv)만 INT8, transformer(137) FP | 0.2653 | −0.1554 | backbone만 내려도 **−15.5 mAP** |

> 🔴 **판정 3 — 손상은 분산돼 있다(범인이 없다)**: transformer만 INT8로 내리면 23.91(≈ 전부 INT8 24.02), backbone만 내려도 26.53. **두 절반이 각자 독립적으로 폭락의 대부분을 만든다**(합산 아님, sub-additive). 즉 소수의 "문제 op"를 FP로 빼는 것으론 회복이 불가능하다 — 문제는 **per-tensor activation 양자화가 망 전체에 퍼져** 있다는 것이다. 이래서 판정 2의 op-제외가 실패한다.

**캘리브레이션은 레버가 되는가 (2.1.1 outlier 명제 직접 검증):** per-tensor scale이 outlier에 끌려가는 게 원인이라면, outlier를 clip하는 **Percentile/Entropy 캘리브레이션**으로 회복돼야 한다. 그러나 정본 ORT 스택에서 **이 경로는 DETR에 대해 3가지로 연달아 죽는다**([`s2_10_quantize_percentile.py`](../experiments/stage2_detr/s2_10_quantize_percentile.py)):

1. **동적 shape 충돌**: 4.3에서 본 대로 mAP 측정 모델은 H,W가 열려 있어 **장마다 activation 텐서 shape가 다르다** → ORT 히스토그램 수집기가 per-image 배열을 하나로 stack하려다 `ValueError: setting an array element with a sequence ... inhomogeneous shape`.
2. **메모리 폭발**: 캘리브 입력을 고정 shape로 맞춰 (1)을 우회하면 이번엔 Percentile이 이미지당 activation 분포를 통째로 쥐고 있어 **약 3.6GB/장** — N=8에서 peak 30.9GB로 **OOM(SIGKILL)**(장비 31GB).
3. **비유한값**: N=4로 줄여 메모리를 맞추면(peak 19GB) `numpy.histogram`이 DETR의 attention mask 채움값 등 **`inf`를 만나** `ValueError: autodetected range of [inf, inf] is not finite`.

> 🔴 **판정 4 — 스톡 ORT에서 histogram 캘리브레이션은 DETR에 사실상 못 쓴다**: 살아남는 건 **MinMax(스칼라 min/max)** 뿐이고, 위 모든 mAP가 MinMax 결과다. 즉 "Percentile로 바꿔 outlier를 clip해 회복" 같은 손쉬운 노브는 **DETR에선 존재하지 않는다**. 진짜 레버는 캘리브레이터 교체가 아니라 **activation outlier 자체를 구조적으로 없애는** 쪽 — **SmoothQuant(2.2)** 로 outlier를 weight로 이전하거나(activation을 per-tensor로도 담을 수 있게 만듦), per-token/per-group activation 양자화(스톡 ORT QDQ엔 없음)로 가야 한다. **본 실습에선 SmoothQuant(modelopt) 적용을 다음 검증 과제로 남긴다**(4.4는 API 확인 필요 상태). 다만 절제 결과(판정 3)로 볼 때, op 단위가 아니라 **activation 스케일 문제 자체(2.1.1)** 를 공략해야 한다는 방향은 분명하다.

> 💡 왜 1단계 ResNet18(−0.1%p)보다 이렇게 심한가: **검출(위치 회귀)은 분류보다 양자화에 훨씬 민감**하다. 박스 좌표는 정밀도가 곧 IoU이고, DETR은 100개 쿼리를 **임계값·NMS 없이** 순위로 제출하므로 미세한 점수 교란이 순위를 흔든다. 특히 작은 객체는 미세한 공간 특징에 의존하는데, per-tensor activation 양자화가 outlier에 끌려가 그 작은 값들을 뭉갠다(2.1.1) → mAP_s −77%.

> 💡 팁: 노드명을 모를 땐 `polygraphy inspect model detr_dyn.onnx --show layers` 또는 Netron으로 그래프를 열어 노드 이름을 확보한다. DETR에서 실제로 제외 효과가 있는 건 `/model/*/self_attn/MatMul(_1)`·`/model/*/encoder_attn/MatMul(_1)`(총 36개)뿐이다.

### 4.6 (심화) BEVFormer-tiny — grid_sample & Deformable Attention 지뢰

DETR로 흐름을 익혔다면, BEV/Occupancy 실전 난이도를 맛본다. BEVFormer-tiny를 export하면 **DETR보다 훨씬 험한** 실패를 만난다.

```bash
# BEVFormer는 mmdet3d/mmcv 계열 의존이 무겁다. 공식 repo 기준으로 환경을 맞춘다.
# 참고 구현(플러그인 포함): https://github.com/DerryHub/BEVFormer_tensorrt
#   지원 커스텀 op: Grid Sampler / Multi-scale Deformable Attention /
#                   Modulated Deformable Conv2d / Rotate / Inverse / BEV Pool V2 / Flash MHA
git clone https://github.com/DerryHub/BEVFormer_tensorrt
```

#### 4.6.1 `grid_sample` — opset·백엔드별 지원 현황

`grid_sample`은 BEV 뷰 변환(카메라 feature를 BEV 격자로 샘플링)의 핵심 op다. **opset과 백엔드 조합**을 정확히 알아야 어디서 깨질지 예측한다.

| 경로 | 4D(2D 샘플링) | 5D(3D 볼류메트릭) | 비고 |
|------|--------------|-------------------|------|
| ONNX 표준 opset 16/17 | ✅ `GridSample` | ❌ | 최초 표준화(4D만). DETR/2D BEV엔 충분 |
| ONNX 표준 opset 20 | ✅ | ✅(스펙에 5D 추가) | 표준상 지원. 런타임 지원은 별개 |
| **onnxruntime 1.23.2** | ✅ CUDA EP에서 실행 | ⚠️ **CPU로 조용히 fallback** | **정본**. 2026-07-31 실측: 5D는 `CUDA kernel not found in registries for Op type: GridSample` 로그를 남기고 노드가 CPU에 배치된다(에러 없이 느려짐) |
| onnxruntime 1.26 | ✅(CPU/CUDA/WebGPU) | ❌ | WebGPU GridSample 추가 |
| onnxruntime 1.27 | ✅ | ✅(CUDA, 볼류메트릭 3D) | CUDA EP에 3D GridSample 추가 |
| onnxruntime 1.28.0 | ✅ | ✅(1.27 계승) | 좌표 NaN/Inf/범위초과의 int64 cast **hardening** 추가. **단 CUDA 13 라인**이라 이 스터디 스택에선 안 씀 |
| **TensorRT 10.16.x LTS** | ✅ `IGridSampleLayer`(native) | ❌ **rank-4만** | 5D 볼류메트릭은 native 미지원(issue #3890). 5D는 plugin/분해 필요 |
| NPU(TIDL/QNN/DRP-AI) | 대체로 ❌ 또는 제한적 | ❌ | 미지원 다수. op 치환/분해가 [4단계](06_multi_soc.md) 과제 |

> 🔴 함정: **TensorRT native GridSample은 4D 전용**이다. BEVFormer의 3D 볼류메트릭 샘플링(spatial cross-attention의 pillar 샘플링)을 그대로 5D로 내보내면 TRT 파싱이 실패한다. 실무 우회는 (1) **5D를 4D로 reshape**해 여러 번 샘플링하거나, (2) 커스텀 **GridSample plugin**을 쓰는 것(`DerryHub/BEVFormer_tensorrt`가 예). 또 경계 밖 좌표(`padding_mode`)에서 백엔드마다 값이 미묘히 달라지는 버그 사례가 있으니 export 옵션(`align_corners`, `padding_mode`)을 **고정하고 수치 검증**하라.

#### 4.6.2 Deformable Attention을 **표준 op로 분해**하기

BEVFormer의 심장인 **Multi-Scale Deformable Attention(MSDeformAttn)** 은 대부분 툴체인에서 그대로 export되지 않는다. 두 갈래 해법이 있고, 면접에서 이 둘을 구분해 말할 수 있어야 한다.

**(a) 표준 op로 분해** — MSDeformAttn의 수학은 사실 **몇 개 표준 op의 조합**이다:

1. **sampling offset·attention weight 예측**: query에 `Linear` → `[num_heads, num_levels, num_points, 2]` 오프셋과 `[..., num_points]` 가중치. (표준 MatMul/Add)
2. **sampling location 계산**: reference point + offset → 각 레벨 feature map의 정규화 좌표 `[-1,1]`. (Add/Mul, 좌표 스케일링)
3. **bilinear 샘플링**: 각 레벨에서 `grid_sample`(bilinear, `align_corners=False`)로 fractional 좌표의 값을 추출. **이 단계가 곧 4D `GridSample`** 이다(레벨마다 4D 텐서를 개별 샘플링).
4. **가중 합산(aggregation)**: softmax된 attention weight로 샘플 값을 가중합 → `Mul` 후 `ReduceSum`(또는 `Einsum`).

즉 **MSDeformAttn = (Linear) + (좌표연산) + (레벨별 grid_sample) + (weighted sum)**. 커스텀 CUDA 커널(`MSDeformAttnFunction`)을 이 **표준 op 시퀀스로 다시 쓰면** opset 16+에서 export가 된다. 실제로 mmcv의 `multi_scale_deformable_attn_pytorch`(순수 PyTorch 폴백)가 정확히 이 분해다 — export 전에 커스텀 함수를 **이 폴백으로 교체**하면 표준 그래프가 나온다.

> ⚠️ 대가: 분해 버전은 레벨·포인트 루프가 펼쳐져 노드 수가 폭증하고 grid_sample이 여러 번 호출돼 **느리다**. 그리고 **여전히 grid_sample이 남으므로**, grid_sample을 못 받는 NPU에서는 이것마저 안 통한다(→ 그때는 (b)).

**(b) 커스텀 op/plugin으로 감싸기** — 성능·NPU 대응을 위해 MSDeformAttn 전체를 **하나의 커스텀 op**로 export하고, 런타임(TensorRT)에서 **C++/CUDA plugin**으로 실행한다. `DerryHub/BEVFormer_tensorrt`의 `MultiscaleDeformableAttnPlugin`이 대표 예이며, 이 plugin은 **불규칙 메모리 접근(grid sampling)을 커널 내부에서 한 번에** 처리해 (a)보다 훨씬 빠르다. **이 plugin을 직접 빌드·연결해본 경험 자체가 이력서의 차별점**이다. 자세한 plugin 등록·빌드는 [3단계 TensorRT](05_tensorrt.md)에서 다룬다.

> 📚 왜 하드웨어가 이걸 싫어하나(배경): MSDeformAttn의 **random-access grid sampling**은 규칙적 conv/matmul과 달리 **메모리 접근이 불규칙**해 NPU/가속기의 PE 활용률을 떨어뜨린다. 이게 DEFA(arXiv:2403.10913)·"Towards Efficient MSDA on NPU"(arXiv:2505.14022) 같은 최신 연구의 출발점이다 — 이들은 sampling point pruning·연산 융합·multi-scale 병렬로 grid sampling 병목을 완화한다. "왜 deformable attention이 NPU에서 지뢰인가"의 근거로 인용할 수 있다.

#### 4.6.3 scatter/gather/dynamic shape — voxelization·pooling의 지뢰

- **`scatter`/`gather`**: BEV pooling(BEV Pool), voxelization에서 점군을 격자에 뿌릴 때 필수. 인덱스가 런타임에 정해지는 **동적 인덱싱**이라 NPU가 극도로 싫어한다(정적 그래프 가정 위배). 가능하면 **고정 격자·고정 point 수**로 바꿔 `scatter`를 정적 인덱스로 만들거나, plugin(예: `BEV Pool V2`)으로 감싼다.
- **dynamic shape**: point 수·detection 수가 프레임마다 달라 텐서 shape가 변한다. TRT는 optimization profile로 어느 정도 흡수하지만, **NPU 대부분은 완전 정적 shape만** 받는다. → padding으로 **최대 크기 고정**하고 mask로 유효 영역을 구분하는 패턴이 정석.

> 🔴 함정: Deformable Attention을 "ONNX 표준 op로만" 내보내려 하면(위 (a)) 대개 **노드 폭증 + 여전한 grid_sample** 때문에 배포에서 다시 막힌다. 실전 결론: **GPU(TensorRT) 타깃이면 plugin (b)** 가 정답, **NPU 타깃이면** grid_sample 자체를 못 쓰는 경우가 많아 **모델 구조를 deformable-free로 바꾸거나 벤더 전용 op**로 가야 한다(→ [4단계](06_multi_soc.md)).

> ⚠️ 확인 필요: `DerryHub/BEVFormer_tensorrt`의 README는 **TensorRT 8.5.x / CUDA 11.6 / PyTorch 1.12.1** 기준이다(구버전). 정본 스택(**TensorRT 10.16.x LTS · CUDA 12.8**)에서는 plugin을 **재빌드**해야 하며 plugin API(`IPluginV2` → `IPluginV3` 계열)가 바뀌었을 수 있다. plugin은 [3단계 TensorRT](05_tensorrt.md)의 커스텀 플러그인 절과 함께 다룬다.

---

## 5) 예시 / 결과 해석

### 5.1 opset·런타임별 grid_sample 지원 (요약)

| opset | `grid_sample`(4D) export | 5D 볼류메트릭 | 비고 |
|-------|--------------------------|----------------|------|
| 9 / 11 / 12 | ❌ `not supported` | ❌ | `GridSample` 미표준. 가장 흔한 첫 실패 |
| **16, 17** | ✅ | ❌ (`5D volumetric` 에러) | 4D만. DETR/2D BEV엔 충분 |
| 20+ | ✅ | ✅(표준상) | ONNX 표준에 5D 추가. 런타임 지원은 별개 |

> 런타임 쪽(4.6.1 상세 표 참고): **TensorRT 10.16.x LTS** 는 `GridSample`을 native로 파싱하되 **4D(rank-4)만**(5D 미지원, issue #3890). **정본 onnxruntime 1.23.2** 는 4D는 CUDA EP에서 돌리지만 **5D는 CUDA 커널이 없어 CPU로 조용히 fallback**한다(실측). CUDA EP의 볼류메트릭(3D) GridSample은 **1.27**에서 추가됐고 1.28.0이 이를 계승했지만, 그 라인은 CUDA 13이라 이 스택에서는 쓰지 않는다 — 즉 **5D는 정본 스택에서 분해/plugin이 사실상 필수**다. **NPU(TIDL 등)는 여전히 미지원 다수** → [4단계](06_multi_soc.md)에서 op 치환/분해 필요.

### 5.2 precision별 정확도/특성 (해석 틀)

| 구성 | 정확도 | 속도 | 언제 쓰나 |
|------|--------|------|-----------|
| FP32 | 기준 | 느림 | baseline·정확도 상한 |
| 전부 INT8 | 폭락(DETR 실측 −42.9%) | 빠름 | Transformer엔 **사실상 불가** |
| mixed **(스톡 ORT 노드 제외)** | 🔴 **회복 실패**(DETR 실측 +0.36 mAP뿐) | 빠름(대부분 INT8) | per-tensor·op 단위 제외로는 부족 |
| mixed **(SmoothQuant+twin-uniform 등)** | 문헌상 near-lossless(8-bit <0.5%) | 빠름 | 제대로 된 기법 조합이 전제(본 실습 미검증) |
| FP16 전체 | ≈FP32 | 중간 | INT8이 도저히 안 될 때의 안전판 |

> 🔴 **실측 정정 (2026-08-16, DETR)**: 초안은 "mixed = baseline 근처 회복"을 권장 처방으로 뒀지만, **스톡 ONNX Runtime의 per-tensor INT8 + `nodes_to_exclude`로는 DETR에서 회복되지 않는다**(42.07→24.02, mixed 24.38, 4.5). 문헌의 "near-lossless"(PTQ4ViT 등)는 **per-channel/per-token activation 양자화 + twin-uniform + Hessian 캘리브** 같은 기법 조합의 결과이지, "문제 op만 FP로 빼기"의 결과가 아니다. 즉 **회복은 도구가 결정한다** — 스톡 ORT가 주는 노브(per-tensor + 노드 제외)와 논문이 쓰는 노브는 다르다.

**핵심 해석 (실측 반영):** Transformer 양자화의 승부처는 "무엇을 FP로 지키느냐"라는 **op 선택**이 **아니라**, DETR 실측으로 보면 **activation을 어떤 입도(granularity)로 양자화하느냐**다. per-tensor로는 outlier 하나가 텐서 전체 scale을 끌고 가(2.1.1) 손상이 망 전체에 분산되고(4.5 판정 3), op를 몇 개 빼는 것으론 못 되돌린다. 되돌리려면 **activation 스케일 문제 자체를 공략**해야 한다 — **SmoothQuant**(activation outlier를 weight로 이전, 2.2) + **per-channel/per-token**(공간 분리) + 로그/twin-uniform(비균일 분포). op 단위 mixed는 이들이 다 실패했을 때의 마지막 수단이지 1차 처방이 아니다.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| `Exporting the operator 'aten::grid_sampler' to ONNX opset version 11 is not supported` | `GridSample`는 opset 16부터 표준 | `opset_version=16`(이상)으로 export |
| `Unsupported: ONNX export of operator GridSample with 5D volumetric input` | opset 16 GridSample은 4D만 | 5D→4D reshape 후 샘플링, 또는 커스텀 op/plugin. 런타임(ORT≥1.27 / 표준 opset≥20)도 확인 |
| `torch.onnx ... aten::xxx is not supported` / `UnsupportedOperatorError` | 특정 aten op 미지원 | opset 상향 / 서브그래프를 지원 op로 치환 / custom symbolic 등록 |
| `torch._dynamo.exc.Unsupported ... could not be traced` | dynamo가 데이터 의존 제어흐름/커스텀 op 트레이스 실패 | `dynamo=False`(legacy)로 재시도, 문제 블록 wrapper/`torch.cond` |
| export 성공했으나 downstream에서 shape/dynamic 에러 | dynamic axes를 backend가 못 받음 | 입력 **shape 고정**, 또는 필요한 축만 `dynamic_axes` 지정 |
| `torch.onnx.export(..., dynamo=True)`가 FakeQuantize/QDQ에서 깨짐 | dynamo 경로의 QDQ export가 아직 불안정 | 양자화 그래프 export는 **legacy 경로(`dynamo=False`)** 로 |
| ORT 로드 시 `Type 'tensor(int64)' ... is invalid` | index dtype 불일치(int64 vs int32) | GraphSurgeon/`onnxsim`로 cast 삽입, opset 상향 |
| MSDeformAttn을 표준 op로 분해했더니 **너무 느림/노드 폭증** | 레벨·포인트 루프가 펼쳐져 grid_sample 다회 호출 | GPU면 **plugin (4.6.2-b)**, NPU면 구조 변경/벤더 op |
| INT8 후 mAP/accuracy 폭락 | Softmax/GELU/attention/LayerNorm이 INT8을 못 견딤 | mixed precision — 해당 노드 `nodes_to_exclude`로 FP16 유지, LayerNorm 전에 SmoothQuant |
| **TIDL: QDQ + self-attention 모델 컴파일 에러** | TIDL이 self-attention 블록의 QDQ 그래프를 못 컴파일 | 릴리스 노트에 보고된 사례. attention을 FP16으로 두거나, 해당 서브그래프를 non-QDQ로 우회 |

> 🔴 함정(릴리스 노트 실사례): TI `edgeai-tidl-tools`에서 **self-attention 블록이 포함된 QDQ 모델**은 컴파일 중 다음과 같이 죽는 사례가 보고되어 있다:
> `RUNTIME_EXCEPTION : Non-zero status code returned while running TIDL_0 node. ... Status Message: CHECK failed: (index) < (current_size_)`
> → NPU에서 self-attention QDQ는 **아직 지뢰**. [4단계](06_multi_soc.md)에서 우회 전략을 다룬다.

---

## 7) 산출물 (Deliverables)

이 단계의 **진짜 결과물은 코드가 아니라 실패의 기록**이다. 아래를 남긴다.

- [ ] `detr_fixed.onnx` (opset 17·shape 고정 export 성공본)
- [ ] `detr_int8.onnx` (전부 INT8·폭락 재현본) + `detr_mixed.onnx` (mixed·회복본)
- [ ] `detr_sq_int8.onnx` (SmoothQuant 적용본, 옵션)
- [ ] mAP 비교표 (FP32 vs INT8 vs mixed)
- [ ] SmoothQuant 전후 채널 absmax 비교(ratio 30x→3x) 로그/그림
- [ ] **`onnx_export_failures.md`** ← 포트폴리오 핵심. 아래 템플릿을 **예시 항목까지 채워** 사용.

### `onnx_export_failures.md` 템플릿 (예시 항목 채움)

````markdown
# ONNX Export & Quantization Failure Log — DETR / BEVFormer-tiny
> 환경(예시, 본 실습 실측): Ubuntu 22.04 · RTX 3080 · driver 595.84 · CUDA 12.8 · torch 2.11.0+cu128
>       transformers 5.15.0 · timm 1.0.28 · onnx 1.18.0 (IR 11) · onnxruntime-gpu 1.23.2 · TensorRT 10.16.x LTS
> 목적: "무엇이 왜 깨졌고 어떻게 우회했는가" = 재사용 가능한 design rules

## 요약 표
| # | 단계(export/compile/PTQ) | 증상(로그 핵심) | 원인 가설 | 우회/해결 | 상태 |
|---|--------------------------|-----------------|-----------|-----------|------|
| 0 | export(opset11, **legacy** dynamo=False) | `aten::scaled_dot_product_attention ... opset 11 is not supported` (v14+) | SDPA symbolic opset14+ | `opset_version=17, dynamo=False` | ✅ (DETR 실측) |
| 0b | export(opset11/17, **기본** dynamo=True) | (실패 아님) opset이 18로 고정·external data 분리 | torch 2.11 기본 dynamo | 단일파일 원하면 `dynamo=False` | ✅ (DETR 실측) |
| 1 | export(opset11, legacy) | `aten::grid_sampler ... opset 11 is not supported` | GridSample은 opset16+ | `opset_version=17` | ⏳ (BEVFormer, DETR엔 없음) |
| 2 | export(opset17, 5D)      | `GridSample with 5D volumetric input` unsupported | opset16/17은 4D만 | 5D→4D reshape / plugin | ⏳ (BEVFormer) |
| 3 | export(dynamo=True)      | `torch._dynamo.exc.Unsupported ... could not be traced` | 데이터 의존 제어흐름 | `dynamo=False`로 전환 | ⏳ (DETR은 dynamo 성공) |
| 4 | export(QDQ, dynamo)      | FakeQuantize export 실패 | dynamo QDQ 불안정 | legacy 경로로 QDQ export | ⏳ |
| 5 | load(ORT 1.28)           | `Type 'tensor(int64)' ... is invalid` | index dtype 불일치 | GraphSurgeon cast 삽입 | ⏳ |
| 6 | PTQ(all-int8)            | **mAP 42.07 → 24.02 폭락(−42.9%)** | per-tensor act 양자화가 망 전체 분산 | op 제외 mixed 실패(+0.36); SmoothQuant 필요 | ✅ (DETR 실측) |
| 7 | export(MSDeformAttn)     | 표준 op 분해 후 노드 폭증·느림 | grid_sample 다회 호출 | TensorRT plugin | ⏳ |
| 8 | TRT build(GridSample 5D) | rank-4만 지원(issue #3890) | TRT native 4D 한정 | 5D→4D 분해 / plugin | ⏳ |
| 9 | TIDL compile             | `CHECK failed (index)<(current_size_)` | self-attn QDQ 미지원 | attn FP16 유지 | ⏳ |

## 상세 로그 (케이스별)
### Case 0 — SDPA opset 미지원 (DETR 실측 첫 블로커)
- **시도한 명령/코드**:
  `torch.onnx.export(model, (pv,), "detr_legacy_op11.onnx", opset_version=11, dynamo=False)`
- **전체 에러 로그**(잘라내지 말 것):
  `torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 11 is not supported. Support for this operator was added in version 14, try exporting with this version`
- **원인 분석**: op 미지원(순수 opset 문제). DETR self/cross-attention이 SDPA로 트레이스되고 SDPA symbolic은 opset 14부터. **grid_sampler가 아니다 — DETR엔 grid_sample이 없다.**
- **우회 방법**: `opset_version=17, dynamo=False`(단일 파일 IR8) 또는 dynamo=True(성공하되 opset 18·external data).
- **재현성**: opset 17 legacy로 export 성공(170.4MB 단일 파일). 정확도 영향 없음.

### Case 1 — grid_sampler opset 미지원 (BEVFormer, 4.6)
- **전체 에러 로그**: `UnsupportedOperatorError: Exporting the operator 'aten::grid_sampler' to ONNX opset version 11 is not supported. Support for this operator was added in version 16 ...`
- **원인 분석**: ONNX GridSample 표준화가 opset 16. **DETR엔 해당 없음** — BEVFormer/deformable 계열에서 발생(4.6에서 다룸).
- **우회 방법**: `opset_version=16`(이상)으로 상향(4D). 5D는 분해/plugin.

### Case 6 — 전부 INT8에서 mAP 폭락 (DETR 실측, COCO val 전량 5,000장)
- **시도한 명령/코드**: `quantize_static(..., activation_type=QInt8, weight_type=QInt8, per_channel=True)` (Conv 54+MatMul 136 = 190 노드 전부)
- **관측**: COCO val2017 mAP **FP32 0.4207 → INT8 0.2402**(−0.1805, **−42.9%**). 작은 객체 mAP_s 0.213→0.049(**−77%**). FP32가 공개값 42.0과 일치 → 계측 신뢰.
- **원인 분석**: 소수 "문제 op"가 아니라 **per-tensor activation 양자화가 망 전체에 분산**. 절제 실측: transformer만 INT8=0.2391, backbone만 INT8=0.2653 — **두 절반이 각자 폭락 대부분을 만든다**(sub-additive). Softmax/LayerNorm은 ORT QDQ가 애초에 양자화 안 하고, DETR엔 GELU도 없음.
- **우회 시도 & 결과**: ① `nodes_to_exclude`로 attention score matmul 36개 FP → **+0.36 mAP뿐(실패)**. ② Percentile 캘리브 회복 시도 → 동적 shape·OOM·`inf`로 **3중 실패**, MinMax만 생존. → **op 제외로는 회복 불가**, SmoothQuant(2.2)/per-token 양자화가 진짜 레버(다음 과제).
- **재현성**: 위 수치는 [`experiments/stage2_detr/`](../experiments/stage2_detr/) 스크립트로 재현. [실측 리포트](../logs/stage2_detr_quantization_report.html) 참조.

## Design Rules (이 로그에서 도출한 규칙)
- [ ] BEV 모델은 처음부터 **opset ≥ 16**, 입력 **shape 고정**으로 export한다.
- [ ] `grid_sample` 5D는 표준 opset 20 / ORT 1.27+; **TensorRT는 4D만** → 5D는 분해/plugin 전제로 설계한다.
- [ ] Softmax/GELU/attention/LayerNorm은 **기본 FP16**, Conv/Linear만 INT8부터 시도한다. 🔴 **단, "특정 op만 FP로 빼면 회복된다"는 기대는 DETR 실측에서 반증됐다** — attention score matmul 36개를 FP로 남겨도 +0.36 mAP뿐(4.5). 손상이 망 전체에 분산돼 있어 **op 선택(granularity가 op 단위)이 아니라 activation 양자화 자체(per-tensor→per-token/SmoothQuant)를 바꿔야** 한다. 이 규칙은 "탐색 시작점"이지 "회복 보장"이 아니다.
- [ ] LayerNorm outlier가 크면 INT8 전에 **SmoothQuant(α≈0.5)** 를 걸고, 채널 absmax ratio가 줄었는지 확인한다.
- [ ] 양자화 QDQ export는 **legacy(`dynamo=False`)** 를 기본으로 한다.
- [ ] Deformable Attention은 **GPU=plugin / NPU=구조변경** 을 사전 결정한다.
- [ ] NPU 타깃이면 grid_sample/deformable/scatter/gather/dynamic shape를 **사전 점검**한다.
````

> 💡 팁: **에러 로그는 절대 요약하지 말고 통째로** 붙여넣어라. 6개월 뒤 다른 모델에서 같은 에러를 만났을 때, 그 원문이 검색 앵커가 된다. 이 파일이 쌓이면 그대로 팀의 design rules 문서가 된다.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [PyTorch ONNX Export (torch.export 기반, dynamo=True)](https://docs.pytorch.org/docs/stable/onnx.html) — 최신 exporter. legacy는 TorchScript 경로.
- [ONNX GridSample operator](https://onnx.ai/onnx/operators/onnx__GridSample.html) — opset 16(4D)/20(5D 추가) 표준 스펙.
- [ONNX Runtime — 모델 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — `quantize_static`, QDQ, per-channel.
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) — GridSample: 1.26 WebGPU, 1.27 볼류메트릭(3D) CUDA, **1.28.0** 좌표 cast hardening.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) — SmoothQuant/AWQ/SVDQuant/FP8/INT4 PTQ 내장(2026 기준 활성). `mtq.INT8_SMOOTHQUANT_CFG`, `mtq.quantize`.
- [NVIDIA TensorRT 10.16 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html) — 정본 LTS. GridSample native(4D), plugin API.
- [TensorRT GridSample 5D 지원 이슈 #3890](https://github.com/NVIDIA/TensorRT/issues/3890) — GridSample은 **4D(rank-4)만**, 5D 미지원.
- [Intel Neural Compressor](https://github.com/intel/neural-compressor) — SmoothQuant 지원 대안 툴.
- [mit-han-lab / smoothquant](https://github.com/mit-han-lab/smoothquant) — SmoothQuant 원본 구현(원리 학습용, LLM 예제).
- [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon) · [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) — op 치환/shape 고정/그래프 검증.
- [DerryHub / BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) — grid_sample·Deformable Attention 등 커스텀 TensorRT 플러그인(FP16/INT8) 참고 구현.
- [mmcv MultiScaleDeformableAttention](https://github.com/open-mmlab/mmcv) — `multi_scale_deformable_attn_pytorch` 순수 PyTorch 폴백(표준 op 분해의 레퍼런스).
- [TI edgeai-tidl-tools Releases](https://github.com/TexasInstruments/edgeai-tidl-tools/releases) — self-attention QDQ 컴파일 에러 등 op 지원 현황(릴리스 노트).
- [HuggingFace DETR (facebook/detr-resnet-50)](https://huggingface.co/facebook/detr-resnet-50) — 실습 모델.

### 논문
- Xiao et al. (2022), *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, arXiv:2211.10438 — **필독**. `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)`, 기본 α=0.5(대칭 시 `√(max|X|/max|W|)`).
- Lin et al. (2021), *FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer*, arXiv:2111.13824 (IJCAI'22) — Power-of-Two Factor(LayerNorm inter-channel), Log-Int-Softmax(로그+BitShift, 4-bit attention).
- Yuan et al. (2021), *PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization*, arXiv:2111.12293 (ECCV'22) — twin uniform(Softmax/GELU 비가우시안, MSB=range flag), Hessian-guided, 8-bit <0.5% 하락.
- Li et al. (2022), *RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers*, arXiv:2212.08254 (ICCV'23) — 캘리브레이션(channel-wise/log√2) ↔ 추론(layer-wise/log2) 분리, 4-bit PTQ 실사용화.
- Li et al. (2022), *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*, arXiv:2203.17270 — deformable attention 기반 BEV(spatial/temporal cross-attention).
- Xu et al. (2024), *DEFA: Efficient Deformable Attention Acceleration via Pruning-Assisted Grid-Sampling and Multi-Scale Parallel Processing*, arXiv:2403.10913 — MSDeformAttn의 grid-sampling 병목·불규칙 메모리 접근 분석과 가속.
- Huang et al. (2025), *Towards Efficient Multi-Scale Deformable Attention on NPU*, arXiv:2505.14022 — NPU(Ascend)에서 MSDA grid sampling 최적화(분해·융합·병렬).
- (참고, 최신) *FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection*, arXiv:2502.15488 (2025) — BEV/멀티뷰 3D 완전 양자화 후속(PE 양자화).

> arXiv 원문: `https://arxiv.org/abs/<ID>`. DETR/BEV decoder 전용 양자화는 2026년에도 갱신 중이므로 최신 후속을 재검색할 것.

---

## 9) 다음 단계

여기서 만든 ONNX(그리고 실패 로그)를 들고 **실제 GPU 엔진으로 컴파일**하러 간다. TensorRT에서 mixed precision·QDQ·커스텀 플러그인(deformable attention)을 다룬다.

- 이전: [1단계 — 양자화 이론](03_quantization_theory.md)
- **다음: [3단계 — TensorRT](05_tensorrt.md)**
