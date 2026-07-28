# 10. 실전 함정 5개 — 양자화가 조용히 무너지는 지점들

> 원본 가이드 매핑: "함정 5개" · 예상 소요: 반나절(정독) + 실습 시 하루 · 선행 조건: [03](03_quantization_theory.md)~[06](06_multi_soc.md) 개념 숙지 권장

이 문서는 앞선 단계들([01](01_environment_setup.md)~[08](08_capstone.md))과 [12주 로드맵](09_roadmap.md)을 관통하는 **5대 실패 패턴**을 정리한다. 각 함정은 **증상 → 원인(수치·메커니즘) → 예방 → 디버깅 절차 → 재현 코드** 순으로, 재현·검증 가능한 코드와 함께 다룬다.

> 💡 팁: 이 다섯 개는 "지식 부족"이 아니라 "무심코"에서 온다. 다 알아도 매번 당한다. 그래서 마지막의 [실무 체크리스트](#실무-체크리스트-양자화-전후-반드시-확인)를 프로젝트마다 복사해 쓰길 권한다.

> ⚠️ 정본 버전 스택(2026-07 기준): **CUDA 12.8 / onnxruntime-gpu 1.28.0 / TensorRT 10.16.x LTS / ExecuTorch 1.3.x**. 아래 코드·명령은 이 조합 기준이다. TensorRT는 10.x부터 plugin이 `IPluginV3`로 통일됐고(함정 5), QNN EP는 dynamic shape·Loop/If를 지원하지 않는다(함정 3·4).

---

## 0) 이 단계에서 무엇을·왜 하는가

양자화가 실패하는 방식은 대개 **요란하지 않다.** 컴파일도 되고, export도 "성공"이라 뜨고, 에러도 없다. 그런데 특정 상황(야간·터널·역광)에서만 정확도가 무너지거나, 가속기에 올렸는데 오히려 느려진다.

이 문서의 목적은 그 "조용한 실패"들을 **미리 이름 붙여** 알아보게 하는 것이다. 이름이 있으면 디버깅이 빨라진다. 각 함정은 로드맵의 특정 주차·산출물과 직접 연결되므로, [09_roadmap.md](09_roadmap.md)를 돌리다 막히면 해당 함정으로 바로 온다.

**왜 "디버깅 절차"까지 적는가.** 함정을 아는 것과 잡는 것은 다르다. "전처리가 문제일 수 있다"는 지식은 흔하지만, **어떤 순서로 무엇을 실행해 그것을 증명하는지**가 실력이다. 그래서 각 함정마다 "증상을 보면 → 이 명령을 이 순서로 → 이 출력이 나오면 확정"이라는 **재현 가능한 절차**를 붙였다. 이 절차 자체가 면접에서 "그 버그 어떻게 잡았어요?"에 대한 답이 된다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] 5대 함정의 **증상**을 보고 원인을 추정할 수 있다.
- [ ] 캘리브레이션 대표성·전처리 일치를 **코드로 검증**할 수 있다.
- [ ] "export 성공"과 "칩 동작"을 구분하고, offload 비율을 근거로 fallback 여부를 판단할 수 있다.
- [ ] `polygraphy inspect capability`로 백엔드 미지원 op를 특정할 수 있다.
- [ ] 양자화 전/후 체크리스트를 프로젝트에 적용할 수 있다.

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
- entropy(KL) calibrator를 기본으로. 극단 outlier가 소수면 percentile clip(99.9%)로 상한을 살짝 낮춰 오히려 평균 오차를 줄인다.

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

> 관련 단계: [06_multi_soc.md](06_multi_soc.md)(TIDL/QNN/DRP-AI), [09_roadmap.md](09_roadmap.md) 9~11주(`four_target_matrix.md`)

**증상**
- 가속기(NPU/DSP)에 올렸는데 **오히려 FP32 CPU보다 느리다.**
- 컴파일 로그에 subgraph가 수십 개로 쪼개짐(예: 20개). 가속기↔CPU를 왔다갔다.
- profile을 보면 연산 시간보다 **메모리 복사/동기화 시간**이 더 크다.

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
4. 판정표(아래)로 "정상/경계/지옥"을 분류하고, 지옥이면 미지원 op 위치를 함정 3 절차로 특정해 앞뒤로 몬다.

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

| # | 함정 | 한 줄 증상 | 첫 번째 확인 | 관련 문서 |
|---|------|-----------|--------------|-----------|
| 1 | 캘리브 데이터가 전부 | 특정 조건(야간/역광)만 급락 | 조건별 분리 정확도 + `calib_coverage.py` | [03](03_quantization_theory.md) |
| 2 | 전처리 불일치 | 에러 없이 정확도만 죽음 | `preprocess_parity.py` 바이트 비교 | [03](03_quantization_theory.md), [05](05_tensorrt.md) |
| 3 | export ≠ 칩 동작 | 컴파일/실행에서 op 미지원 | `polygraphy inspect capability` | [04](04_transformer_quantization.md), [06](06_multi_soc.md) |
| 4 | fallback 지옥 | 가속기인데 더 느림 | offload 비율·subgraph 개수 | [06](06_multi_soc.md) |
| 5 | C++ 회피 | plugin/런타임에서 막힘 | 벤더 IPluginV3 template 빌드 여부 | [05](05_tensorrt.md), [07](07_infrastructure.md) |

---

## 실무 체크리스트 (양자화 전/후 반드시 확인)

프로젝트마다 아래를 복사해 채운다. `design_rules.md`([07_infrastructure.md](07_infrastructure.md), [09_roadmap.md](09_roadmap.md) 12주)에 그대로 편입 가능. 각 항목 옆의 **근거**는 "왜 이걸 확인하는가"이며, 대응하는 함정 번호를 붙였다.

### 양자화 전 (Pre-quantization)

| 확인 | 근거(왜) | 함정 |
|------|----------|------|
| - [ ] FP32 **baseline 정확도/지연**을 측정·기록 | baseline 없으면 "손실"을 정의할 수 없다. 모든 판정의 기준선. | 전부 |
| - [ ] 캘리브 셋이 운영 분포를 대표(조건별 표본, 수백 장, batch≠1) | 캘리브가 안 본 분포는 clipping → 특정 조건 급락. | 1 |
| - [ ] 학습·캘리브·추론 전처리가 **동일 코드/동일 상수**(mean·std·보간·채널순서·레이아웃) | 분포가 어긋나면 scale이 엉키고 INT8에서 정확도 샘. shape 맞아 에러 없음. | 2 |
| - [ ] calibration cache가 **현재 전처리로** 생성됨(전처리 바꿨으면 캐시 삭제) | 캐시는 전처리에 종속. 옛 캐시 재사용 시 scale 불일치. | 2 |
| - [ ] 타깃 백엔드의 **지원 op 목록** 확인, 모델 op가 화이트리스트에 듦 | 미지원 op는 빌드 실패 또는 fallback. export 후가 아니라 설계 시 확인. | 3·4 |
| - [ ] 타깃 EP의 대칭성 요구를 앎(예: GPU/TRT는 symmetric activation+weight) | EP가 요구하는 양자화 스킴과 안 맞으면 재작업. ([ONNX Runtime 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)) | 3 |
| - [ ] dynamic shape 필요 여부 확정(QNN 등은 dynamic shape 미지원 → 고정 shape export) | dynamic shape가 세션 생성 자체를 막는 백엔드가 있음. | 3 |

### 양자화 후 (Post-quantization)

| 확인 | 근거(왜) | 함정 |
|------|----------|------|
| - [ ] INT8 정확도를 **전체 + 조건별(야간/역광 등)로 분리** 측정 | 전체 한 숫자는 조건별 급락을 숨긴다. | 1 |
| - [ ] 파이프라인 경계마다 수치 정합성 검증(PyTorch↔ONNX↔백엔드, Polygraphy) | 어느 경계에서 어긋나는지 좁혀야 원인이 잡힘. | 2·3 |
| - [ ] export 후 **즉시 타깃 컴파일**로 op 지원 확인(`polygraphy inspect capability`, `onnx_export_failures.md`) | 미지원 op를 일찍 알수록 우회 시간이 있다. | 3 |
| - [ ] **offload 비율·subgraph 개수**를 로그로 확인(가속기 이득이 실제로 나는가) | 가속기에 올렸는데 fallback 지옥이면 FP32보다 느림. | 4 |
| - [ ] custom plugin/런타임 통합의 C++ 경로가 빌드·검증됨(compute-sanitizer 통과) | 칩 위 실행은 C++. plugin이 조용히 틀린 값을 낼 수 있음. | 5 |
| - [ ] 위 결과가 **회귀 하네스**로 자동 재측정됨([09_roadmap.md](09_roadmap.md) 12주) | 한 번 잡은 함정이 다음 커밋에서 되살아나는 걸 막는다. | 전부 |

> 💡 팁: 문제가 생기면 이 순서로 의심하라 — (1) 전처리 일치 → (2) 캘리브 대표성 → (3) op 지원/정합성 → (4) offload 비율. 대부분 상위 두 개에서 잡힌다. 이 순서는 "싸고 흔한 것부터"라서 평균 디버깅 시간이 가장 짧다.

---

## 7) 산출물(Deliverables)

- [ ] `pitfall_checklist.md` — 위 체크리스트를 프로젝트에 맞춰 채운 사본.
- [ ] (실습 시) `calib_coverage.py`, `preprocess_parity.py` 실행 결과 로그.
- [ ] (실습 시) `polygraphy inspect capability --with-partitioning` 리포트(미지원 op 목록).
- [ ] 발견한 실패 사례를 각 산출물 문서(`onnx_export_failures.md`, `four_target_matrix.md`)에 반영.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [ONNX Runtime 양자화](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — static/dynamic PTQ, 대표성 있는 캘리브, EP별 대칭성 요구.
- [ONNX Runtime 아키텍처(GetCapability/파티셔닝)](https://onnxruntime.ai/docs/reference/high-level-design.html) — subgraph 분할·CPU fallback 원리.
- [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — 지원 op 부분집합, dynamic shape/Loop·If 미지원, `disable_cpu_ep_fallback`.
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
