# 5. 인프라화 (시니어와 갈리는 지점)

> 원본 가이드 매핑: "5단계 — 인프라화" · 예상 소요: 2~3주 (반복 개선 포함) · 선행 조건: [3단계 TensorRT](05_tensorrt.md), [4단계 멀티 SoC](06_multi_soc.md) 완료 (각 백엔드로 1회 이상 빌드·측정 경험)

> 정본 버전 스택(이 문서 전체 고정): **CUDA 12.8 · TensorRT 10.16.x LTS(`build_serialized_network`) · onnxruntime-gpu 1.28.0**. 앞 단계 산출물 이름 고정: `layer_sensitivity.csv`(1단계), `onnx_export_failures.md`(2단계).

---

## 0) 이 단계에서 무엇을·왜 하는가

앞의 1~4단계는 **"내가 손으로 한 번 돌려봤다"** 의 세계다. 이 단계는 **"조직에 시스템을 남긴다"** 의 세계다. 원본 가이드의 표현을 빌리면, 여기가 **혼자 튜닝하는 사람과 시니어가 갈리는 지점**이다.

무엇이 다른가? 주니어는 "BEVFormer를 Orin에서 INT8로 돌렸더니 12ms 나왔어요"라고 **한 번** 말한다. 시니어는 다음을 **매일 자동으로** 만든다.

- **성능 매트릭스**: 모델 × SoC × precision → latency / peak memory / mAP 를 한 장의 표로. 새 모델·새 SoC·새 양자화 옵션이 추가될 때마다 자동 갱신.
- **회귀 방지**: 누군가 커밋으로 정확도를 1% 떨어뜨리면 **CI가 빨간불**을 켠다. 사람이 매번 재검증하지 않는다.
- **설계 규칙(`design_rules.md`)**: "이 SoC에서 이 op는 쓰지 마라"를 문서화. 채용 공고(JD)가 명시적으로 요구하는 산출물이다.
- **의사결정 로그(decision log)**: "왜 이 레이어만 FP16으로 남겼는가"를 기록. 6개월 뒤의 나와 후임자를 위한 기록.

> 💡 팁: 이 단계의 결과물은 면접에서 가장 강력한 무기다. GitHub에 `bench/` 레포와 `design_rules.md`를 올려두면, "양자화 해봤어요"가 아니라 "양자화 파이프라인을 운영해봤어요"가 된다. 후자가 시니어 시그널이다.

### 왜 "손으로 한 번"이 위험한가 — 재현성의 붕괴

주니어의 "12ms"는 재현이 안 된다. 그 수치가 나온 **엔진의 빌드 플래그·TensorRT 버전·드라이버·워크스페이스·클럭 상태·입력 배치**가 어디에도 기록돼 있지 않기 때문이다. 3개월 뒤 같은 명령을 쳐도 다른 숫자가 나오고, 그때는 "왜 느려졌지?"를 처음부터 다시 파야 한다. 인프라화의 본질은 **"수치가 나온 조건 전체를 자동으로 봉인(freeze)해서, 언제 누가 실행해도 같은 숫자가 나오게 만드는 것"** 이다. 이것이 되면 성능은 "느낌"이 아니라 **버전 관리되는 사실(fact)** 이 된다.

이 관점에서 이 단계의 세 산출물을 다시 읽으면 각각의 존재 이유가 분명해진다.

| 산출물 | 봉인하는 것 | 봉인이 없으면 생기는 사고 |
|--------|-------------|---------------------------|
| 성능 매트릭스 (`report/`) | "어떤 조합이 몇 ms/몇 mAP였는가" | 회의 때마다 숫자가 달라 논쟁만 반복 |
| 회귀 게이트 (`ci.yml`+`tests/`) | "정확도의 하한선(baseline)" | 누군가의 커밋이 조용히 mAP를 깎아도 출시 직전에야 발견 |
| `design_rules.md` / `decision_log.md` | "왜 이 구조·이 precision인가" | 후임이 "이거 왜 FP16이지?" 하고 되돌려 사고 재발 |

### 앞 단계 산출물이 여기로 모인다

이 단계는 새 작업을 시작하는 게 아니라, **앞 단계의 산출물을 한 시스템으로 묶는** 작업이다.

| 출처 단계 | 산출물 | `bench/`에서의 역할 |
|-----------|--------|--------------------|
| [1단계](03_quantization_theory.md) | `layer_sensitivity.csv` (레이어별 양자화 민감도) | `design_rules.md`의 "FP16 유지 레이어" 근거 + decision log 링크 |
| [2단계](04_transformer_quantization.md) | `onnx_export_failures.md` (export 실패 op 목록) | `design_rules.md`의 ❌ 금지 op 목록의 근거 |
| [4단계](06_multi_soc.md) | 4-target 결과 (TRT/TIDL/QNN/DRP-AI latency·mem) | `report/` 성능 매트릭스의 행(row)이 됨 |
| [3~4단계](06_multi_soc.md) | FP32 체크포인트 + 캘리브레이션셋 | `bench/models/`, `bench/calib/`에 고정(freeze) |

> 💡 팁: 위 표의 화살표는 **단방향이 아니다**. `report/` 매트릭스에서 "int8 mAP -3%p"가 관측되면, 그 원인 레이어를 `layer_sensitivity.csv`로 되짚어 `design_rules.md`에 규칙으로 승격시키고, 그 판단을 `decision_log.md`에 남긴다. 이 되먹임 루프(측정 → 규칙 → 재측정)가 돌기 시작하면 그때부터 "파이프라인을 운영한다"고 말할 수 있다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] `bench/` 디렉토리 구조를 만들고 각 하위 폴더의 역할을 설명할 수 있다
- [ ] `backends/`의 **공통 인터페이스**(`build → run → measure → collect`)를 추상 클래스로 정의하고, `trt.py`로 1개 백엔드를 구현했다
- [ ] `tidl.py`/`qnn.py`/`drpai.py`를 **동일 시그니처 stub**으로 채워, 보드 없이도 매트릭스에 '보드필요'로 나타나게 했다
- [ ] `config.yaml`(모델 × 타깃 × precision 매트릭스) 스키마를 정의하고 `run_bench.py` 러너로 조합을 순회했다
- [ ] `report/generate.py`로 여러 백엔드 결과(JSON)를 수집해 **마크다운 + CSV + HTML 매트릭스**를 자동 생성했다(정렬·하이라이트 포함)
- [ ] `ci.yml`로 **커밋마다 벤치가 돌고, mAP가 baseline 대비 1% 이상 하락하면 CI가 실패**하도록 만들었다
- [ ] self-hosted GPU runner를 등록(`config.sh`/`svc.sh`)하고 워크플로가 그 runner에서 도는 것을 확인했다
- [ ] `pytest-regressions` 골든 파일 비교로 회귀 테스트를 이중화했다
- [ ] `design_rules.md`를 ✅권장/⚠️주의/❌금지/📏제약 4분류로 작성하고, 각 규칙에 **앞 단계 산출물 근거**를 연결했다
- [ ] `decision_log.md`에 "왜 이 결정을 했는가" 항목을 최소 3개 기록했다
- [ ] (선택) MLflow(무료 셀프호스팅) 또는 W&B로 벤치 결과를 실험 추적에 남길지 결정하고 근거를 적었다

---

## 2) 배경 이론 / 개념 — "벤치 하네스"가 왜 추상화여야 하는가

4개 백엔드(TensorRT / TIDL / QNN / DRP-AI)는 API가 전부 다르다. 각각을 그때그때 스크립트로 돌리면, SoC가 5개·모델이 10개가 되는 순간 조합 폭발(5×10×3 precision = 150 케이스)로 관리 불능이 된다.

핵심 아이디어는 **"백엔드마다 다른 것"과 "모든 백엔드가 공통으로 하는 것"을 분리**하는 것이다.

```
공통 (하네스가 강제):  build(onnx, precision, calib) → run(inputs) → measure() → collect() → {latency, peak_mem, accuracy}
다른 것 (백엔드가 구현): TRT는 build_serialized_network, TIDL은 compile 옵션, QNN은 context binary ...
```

이렇게 하면 `report/`는 백엔드가 뭐든 **동일한 dict**만 받으면 되고, 새 SoC 추가 = `backends/new_soc.py` 파일 하나 추가로 끝난다. 이것이 원본 가이드가 말하는 "조직에 남기는 시스템"의 구조적 실체다.

### 왜 하필 "추상 베이스 클래스(ABC)"인가 — 세 가지 대안과의 비교

"공통/개별 분리"는 여러 방식으로 구현할 수 있다. 왜 이 가이드는 ABC를 고르는가?

| 방식 | 장점 | 단점 | 이 프로젝트 적합성 |
|------|------|------|-------------------|
| if/elif 분기 (`if backend=="trt": ...`) | 파일 1개로 시작 빠름 | 백엔드 늘면 함수마다 분기 폭발. `collect`가 백엔드를 알아야 함 | ❌ 조합 폭발과 정면충돌 |
| 덕 타이핑(규약만 문서로) | 상속 없이 유연 | 계약 위반이 **런타임**에야 터짐(측정 몇 시간 뒤 크래시) | △ 팀 커지면 위험 |
| `typing.Protocol` (구조적) | 상속 불필요, 정적 검사 가능 | "공통 유틸 재사용(`_timeit`)"을 강제로 물려주기 어려움 | △ 공통 코드 공유가 약함 |
| **`abc.ABC` (명목적)** | 미구현 메서드는 **인스턴스화 시점**에 즉시 `TypeError`. 공통 유틸을 부모에 담아 상속 | 상속 트리가 생김(경량) | ✅ "계약을 어기면 벤치 시작 전에 죽는다" + 공통 코드 1곳 |

핵심은 **실패를 앞당기는 것(fail fast)** 이다. `abc.abstractmethod`를 안 채운 백엔드는 `TIDLBackend()`를 호출하는 순간 `TypeError: Can't instantiate abstract class`로 죽는다 — GPU를 3시간 돌린 뒤 리포트 단계에서 `AttributeError`가 나는 것보다 압도적으로 낫다. 임베디드 벤치는 한 케이스가 수십 분~수시간이라, "일찍 죽는 것"의 가치가 웹 서비스보다 훨씬 크다.

### 측정 방법론 — 왜 median/p95이고, 왜 warmup인가

latency 한 숫자를 뽑는 데도 함정이 많다. 아래는 임베디드 성능 리포팅의 최소 상식이다.

- **첫 실행은 버린다(warmup)**: 최초 추론은 CUDA 커널 JIT/캐시 워밍업, cuDNN 알고리즘 탐색, 클럭 램프업이 섞여 2~10배 느리다. warmup을 20회 이상 준 뒤 측정한다.
- **평균이 아니라 median/p95**: 평균은 스로틀링·OS 스케줄러 튐 하나에 크게 흔들린다. **median(p50)** 은 대표값, **p95** 는 "최악에 가까운 상황"을 본다. 안전 규격(자동차)에서는 p95/p99를 요구하기도 한다.
- **GPU는 반드시 동기화**: `execute_async_v3`는 비동기다. 스트림 동기화(`stream.synchronize()`/`Context.synchronize()`) 없이 시간을 재면 커널 큐잉 시간만 재서 "0.1ms" 같은 거짓 수치가 나온다.
- **클럭 고정(선택, 재현성용)**: dGPU는 `sudo nvidia-smi -lgc <clock>`로, Orin은 `jetson_clocks`로 클럭을 고정하면 실행 간 분산이 크게 준다. CI 재현성을 높이려면 고려한다.

```
평균(mean)  →  워밍업/스로틀 1방에 오염     ✗ 리포트 금지
p50(median) →  "보통 이 정도"               ✓ 기본값
p95         →  "나쁠 때도 이 안"            ✓ 안전 마진 판단용
```

---

## 3) 환경·도구 준비

기본 환경(Ubuntu 22.04 + RTX GPU + CUDA/TensorRT)은 [1단계](01_environment_setup.md)·[3단계](05_tensorrt.md)에서 이미 갖췄다고 가정한다. 이 단계에서 추가로 필요한 것만 설치한다.

```bash
# 벤치 하네스 + 리포트 + 테스트 도구 (호스트 또는 venv)
python3 -m pip install --upgrade \
  pandas \
  tabulate \
  jinja2 \
  pytest \
  pytest-regressions \
  pyyaml
# pandas             : 성능 매트릭스 집계·피벗
# tabulate           : DataFrame → 마크다운 표 (df.to_markdown)
# jinja2             : DataFrame.style → HTML (하이라이트 매트릭스 렌더)
# pytest             : 회귀 테스트 러너
# pytest-regressions : 골든 파일(baseline) 비교 (v3.0+ 권장, 2026-07 기준)
# pyyaml             : bench 설정(config.yaml) 로드
```

> ⚠️ 주의: TensorRT / TIDL / QNN / DRP-AI SDK 자체는 앞 단계에서 설치한 것을 그대로 쓴다. 이 단계는 그 위에 얇은 오케스트레이션 레이어를 얹는 것이다. 새로 SDK를 깔지 않는다.

TensorRT는 **10.16.x LTS 계열**(2026-07 기준, [TensorRT Python API 문서 10.16](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html))을 전제로 한다. 10.x에서는 `builder.build_serialized_network(network, config)`가 표준 빌드 경로다(8.x의 `build_engine`+`serialize`는 불필요).

### onnxruntime-gpu 1.28.0을 CUDA 12.8에 맞춰 설치하기 (함정 주의)

일부 백엔드 스텁(예: QNN EP)은 onnxruntime로 도는데, **버전-CUDA 조합에서 사고가 잦다.** PyPI의 `onnxruntime-gpu`는 **1.27부터 기본이 CUDA 13.0** 빌드다. 즉 `pip install onnxruntime-gpu==1.28.0`을 그냥 하면 CUDA 13 런타임을 기대하는 휠이 깔려, CUDA 12.8 환경에서 `LoadLibrary`/`libcublas` 로드 실패가 난다. 정본 스택(CUDA 12.8)을 지키려면 **CUDA 12용 전용 인덱스**에서 받아야 한다.

```bash
# onnxruntime-gpu 1.28.0을 CUDA 12.x 빌드로 설치 (정본 스택 = CUDA 12.8)
python3 -m pip install onnxruntime-gpu==1.28.0 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# aiinfra 피드 = Microsoft가 CUDA 12용으로 유지하는 Azure Artifacts 인덱스
# (PyPI 기본 휠은 1.27+부터 CUDA 13.0 → 12.8 환경에서 로드 실패)
```

```python
# 설치 검증: 실제로 CUDA EP가 잡히는지 (예상 출력 포함)
import onnxruntime as ort
print("ort:", ort.__version__)               # 1.28.0
print("providers:", ort.get_available_providers())
# 기대: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# CUDAExecutionProvider가 없으면 CUDA/드라이버 버전 불일치 → 위 인덱스로 재설치
```

> 🔴 함정 — "onnxruntime-gpu를 설치했는데 CPU로 돈다": `get_available_providers()`에 `CUDAExecutionProvider`가 있어도, 세션 생성 시 provider를 명시하지 않으면 CPU로 폴백한다. 반드시 `ort.InferenceSession(onnx, providers=["CUDAExecutionProvider"])`처럼 명시하고, 세션 생성 직후 `sess.get_providers()`로 실제 활성 provider를 확인하라. "설치됨 ≠ 사용됨"이다. ([ONNX Runtime CUDA EP 문서](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html))

---

## 4) 단계별 실습

### 4-1. `bench/` 디렉토리 생성

원본 가이드의 구조를 그대로 만든다.

```bash
# 프로젝트 루트에서 실행
mkdir -p bench/{models,calib,backends,report,results,tests}
cd bench

# 뼈대 파일 생성
touch backends/__init__.py backends/base.py backends/trt.py \
      backends/tidl.py backends/qnn.py backends/drpai.py
touch report/generate.py
touch run_bench.py config.yaml
touch design_rules.md decision_log.md
touch tests/__init__.py tests/test_regression.py
```

완성된 구조:

```
bench/
├── models/              # 모델 정의 + FP32 체크포인트 (앞 단계에서 freeze)
│   ├── bevformer.onnx
│   └── bevformer_fp32.pth
├── calib/               # 캘리브레이션 데이터셋 — 대표성 확보가 생명 (아래 함정 참고)
│   └── nuscenes_calib_200.npy
├── backends/
│   ├── __init__.py
│   ├── base.py          # 공통 인터페이스 (추상 클래스) + BenchResult 스키마
│   ├── trt.py           # TensorRT 구현 (동작)
│   ├── tidl.py          # TI TIDL stub (공통 시그니처)
│   ├── qnn.py           # Qualcomm QNN stub (공통 시그니처)
│   └── drpai.py         # Renesas DRP-AI stub (공통 시그니처)
├── report/
│   ├── generate.py      # 결과 수집 → 마크다운/CSV/HTML 매트릭스
│   ├── matrix.md        # (자동 생성)
│   ├── matrix.csv       # (자동 생성) — 회귀 baseline의 소스
│   └── matrix.html      # (자동 생성) — 하이라이트 대시보드
├── results/             # 각 백엔드가 뱉는 raw JSON (모델×백엔드×precision별)
├── tests/
│   ├── __init__.py
│   ├── test_regression.py   # mAP/latency 회귀 테스트
│   └── baseline_matrix.csv  # 골든 (git에 커밋)
├── run_bench.py         # CLI 러너 (config.yaml 순회 → results/*.json)
├── config.yaml          # 무엇을 × 무엇으로 벤치할지 정의 (모델×타깃×precision)
├── design_rules.md      # SoC별 설계 규칙 (JD가 요구하는 산출물)
├── decision_log.md      # "왜 이렇게 결정했는가" 기록
└── ci.yml               # → .github/workflows/ci.yml 로 심볼릭/복사
```

> 🔴 함정 — 캘리브레이션 데이터셋의 대표성: `calib/`에 아무 이미지나 200장 넣으면 안 된다. **야간·역광·터널·눈길 등 실제 배포 분포를 대표**해야 PTQ가 제대로 스케일을 잡는다. 낮 시내 주행만 넣으면 야간에서 mAP가 무너진다. 이건 코드 버그가 아니라 데이터 버그라 CI로도 안 잡힌다 — 사람이 큐레이션해야 한다.

> 💡 팁 — 대용량 자산은 Git LFS로: `models/*.pth`, `calib/*.npy`는 수백 MB~GB다. 일반 git에 넣으면 clone이 지옥이 된다. `git lfs track "*.pth" "*.npy" "*.onnx"`로 LFS에 넣고, CI checkout에서 `lfs: true`를 준다(4-8 참고). 단, self-hosted runner에도 `git-lfs`가 설치돼 있어야 한다.

### 4-2. 공통 인터페이스 `backends/base.py`

모든 백엔드가 반드시 구현해야 하는 계약(contract)을 추상 클래스로 못 박는다. 계약은 4개다: **build / run / measure / collect**.

```python
# bench/backends/base.py
from __future__ import annotations
import abc
import json
import pathlib
import time
from dataclasses import dataclass, asdict, field


@dataclass
class BenchResult:
    """모든 백엔드가 반환해야 하는 공통 결과 스키마.

    report/ 와 tests/ 는 이 필드 이름에만 의존한다.
    필드를 바꾸면 report·회귀테스트·baseline CSV가 전부 영향을 받으므로,
    스키마 변경은 decision_log에 남긴다.
    """
    model: str            # 예: "bevformer"
    soc: str              # 예: "rtx", "orin", "tda4vm", "qcs8550", "rzv2h"
    precision: str        # "fp32" | "fp16" | "int8"
    latency_ms: float     # 대표값(중앙값 권장)
    peak_mem_mb: float    # 추론 중 peak memory
    accuracy: float       # mAP 등 (0~1)
    engine_build_s: float # 빌드 소요(참고용)
    latency_p95_ms: float = float("nan")  # 안전 마진 판단용(선택)
    trt_version: str = ""                 # 재현성: 어떤 TRT로 뽑았나
    notes: str = ""                       # 실패/특이사항

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(abc.ABC):
    """모든 SoC 백엔드의 공통 인터페이스.

    새 SoC 추가 = 이 클래스를 상속한 파일 하나 추가.
    report/ 와 CI는 이 계약에만 의존한다.

    계약(4):
      build(onnx, precision, calib) -> None   # ONNX를 엔진으로
      run(inputs) -> np.ndarray                # 1회 추론
      measure(...) -> BenchResult              # latency/mem/acc 측정
      collect(result) -> pathlib.Path          # 결과를 results/*.json으로 봉인
    """
    soc_name: str = "unknown"

    def __init__(self):
        self._build_s: float = 0.0   # build()가 채운다

    @abc.abstractmethod
    def build(self, onnx_path: str, precision: str, calib_path: str | None) -> None:
        """ONNX → 백엔드 엔진으로 컴파일/빌드. 빌드 소요를 self._build_s에 저장."""
        ...

    @abc.abstractmethod
    def run(self, inputs) -> "np.ndarray":
        """1회 추론. 출력 텐서 반환."""
        ...

    @abc.abstractmethod
    def measure(self, model: str, precision: str,
                loader, evaluator, warmup: int = 20, iters: int = 200) -> BenchResult:
        """latency / peak_mem / accuracy 를 측정해 BenchResult로 반환."""
        ...

    # --- collect: 결과를 디스크에 봉인 (모든 백엔드 공통, 오버라이드 불필요) ---
    def collect(self, result: BenchResult, out_dir: str = "results") -> pathlib.Path:
        """BenchResult를 results/{model}__{soc}__{precision}.json 으로 저장.

        파일명 규약이 report/·tests/의 파싱 기준이므로 여기서 단일화한다.
        """
        out = pathlib.Path(out_dir)
        out.mkdir(exist_ok=True)
        fn = out / f"{result.model}__{result.soc}__{result.precision}.json"
        fn.write_text(json.dumps(result.to_dict(), indent=2))
        return fn

    # --- 공통 유틸 (모든 백엔드가 재사용) ---
    @staticmethod
    def _timeit(fn, warmup: int, iters: int) -> tuple[float, float]:
        """(median_ms, p95_ms) 반환. GPU면 각 백엔드가 fn 안에서 동기화를 넣는다."""
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        p95 = samples[min(int(len(samples) * 0.95), len(samples) - 1)]
        return median, p95
```

> 💡 팁: latency는 **평균이 아니라 중앙값(median) 또는 p95**로 보고하는 게 임베디드 관행이다(2절 참고). `_timeit`이 median과 p95를 함께 돌려주도록 만들어 두면, 안전 규격 요구가 생겨도 코드 변경 없이 대응된다.

> 💡 팁 — `collect`를 왜 베이스에 두는가: 파일명 규약(`model__soc__precision.json`)을 **단 한 곳**에 두기 위해서다. 백엔드마다 각자 파일을 쓰면 규약이 미묘하게 어긋나고(`_` vs `-`, 순서 뒤바뀜), `report/`·`tests/`의 glob 파싱이 조용히 깨진다. 계약의 마지막 단계(collect)를 부모가 소유하면 이 사고가 원천 차단된다.

### 4-3. 구체 구현 `backends/trt.py` (TensorRT)

가장 익숙한 TensorRT부터 구현한다. 나머지(tidl/qnn/drpai)는 이 골격을 복사해 각 SDK 호출부만 바꾸면 된다.

```python
# bench/backends/trt.py
import numpy as np
import tensorrt as trt        # TensorRT 10.16.x LTS (2026-07 기준)
import pycuda.driver as cuda
import pycuda.autoinit        # noqa: F401  (컨텍스트 자동 초기화)
import time

from .base import Backend, BenchResult

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTBackend(Backend):
    soc_name = "rtx"  # 데스크톱 dGPU. Orin이면 "orin"으로.

    def __init__(self):
        super().__init__()
        self.engine = None
        self.context = None

    def build(self, onnx_path: str, precision: str, calib_path: str | None) -> None:
        t0 = time.perf_counter()
        builder = trt.Builder(TRT_LOGGER)
        # TensorRT 10.x: EXPLICIT_BATCH는 기본이지만 명시해도 무방
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                msgs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed: {msgs}")

        config = builder.create_builder_config()
        # 워크스페이스: 4GB (RTX 기준, Orin이면 줄일 것)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # 실제로는 여기서 IInt8Calibrator를 붙인다(calib_path의 npy 사용).
            # 3단계에서 만든 캘리브레이터를 재사용. 지면상 생략.
            # config.int8_calibrator = MyCalibrator(calib_path)

        # TensorRT 10.x 표준 빌드 경로
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:                       # 10.x는 None 반환으로 실패 표현
            raise RuntimeError("build_serialized_network returned None "
                               "(미지원 op이거나 config 오류 — parser 로그 확인)")

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self._build_s = time.perf_counter() - t0

    def run(self, inputs: np.ndarray) -> np.ndarray:
        # 최소 예시: 단일 입력/출력 가정. (실제 BEV 모델은 다중 입력이므로 확장 필요)
        in_name = self.engine.get_tensor_name(0)
        out_name = self.engine.get_tensor_name(1)
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_in = cuda.mem_alloc(inputs.nbytes)
        d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(d_in, np.ascontiguousarray(inputs))
        # TensorRT 10.x: name 기반 I/O 주소 설정
        self.context.set_tensor_address(in_name, int(d_in))
        self.context.set_tensor_address(out_name, int(d_out))
        stream = cuda.Stream()
        self.context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh(out, d_out)
        return out

    def measure(self, model, precision, loader, evaluator,
                warmup=20, iters=200) -> BenchResult:
        sample = loader.one_batch()  # 대표 입력 1개(latency용)

        def _once():
            self.run(sample)
            cuda.Context.synchronize()  # GPU 동기화 필수

        median, p95 = self._timeit(_once, warmup, iters)

        # 정확도: 검증셋 전체를 돌려 evaluator가 mAP 계산
        preds = [self.run(x) for x in loader.eval_set()]
        acc = evaluator.compute_map(preds, loader.gts())

        # peak mem: TRT 엔진 디바이스 메모리 크기(근사). Orin에선 tegrastats 병행 권장.
        peak_mem = self.engine.device_memory_size / (1024 ** 2)

        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=round(median, 3), latency_p95_ms=round(p95, 3),
            peak_mem_mb=round(peak_mem, 1), accuracy=round(acc, 4),
            engine_build_s=round(self._build_s, 1),
            trt_version=trt.__version__,   # 재현성: 어떤 TRT로 뽑았는지 봉인
        )
```

> ✅ 확인됨 (2026-07): `execute_async_v3` / `set_tensor_address` / `get_tensor_name`은 TensorRT 10.x의 name 기반 I/O API가 맞다. 8.x의 bindings 배열 방식은 10.x에서 제거되었다([TensorRT 8→10 Python API 마이그레이션](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/tensorrt-8x-to-10x-python-api.html)). BEV 모델은 입력이 여러 개(멀티뷰 이미지 + intrinsics)이므로 위 단일 I/O 예시를 실제 텐서 수에 맞게 확장해야 한다 — `engine.num_io_tensors`를 순회하며 `engine.get_tensor_mode(name)`으로 입력/출력을 구분해 주소를 세팅한다.

### 4-4. 나머지 백엔드 stub — 공통 시그니처만 맞춘다

핵심 원칙: **보드가 없어도 CI는 돌아야 한다.** 보드가 필요한 백엔드는 `measure`에서 죽이지 말고 `accuracy=NaN` + `notes="board required"`를 남긴다. 그러면 매트릭스에 '보드필요'로 회색 표기되고, RTX 결과만으로 회귀 게이트가 성립한다. 세 stub 모두 **base.py의 계약(build/run/measure)을 동일 시그니처로** 구현한다 — 그래야 `run_bench.py`가 백엔드를 구분하지 않고 순회할 수 있다.

```python
# bench/backends/tidl.py  (TI TDA4VM — edgeai-tidl-tools)
from .base import Backend, BenchResult


class TIDLBackend(Backend):
    soc_name = "tda4vm"

    def build(self, onnx_path, precision, calib_path):
        # edgeai-tidl-tools의 onnxrt/tvm 컴파일 훅. INT8 우선.
        # 컴파일 아티팩트(artifacts_folder)를 만들고 self._build_s 기록.
        raise NotImplementedError("TIDL compile hook — TDA4VM/PC-emulation 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        # 보드/에뮬레이션 없으면 사유 남기고 회색 처리
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="board/emulation required (TDA4VM, edgeai-tidl-tools)",
        )
```

```python
# bench/backends/qnn.py  (Qualcomm QCS8550 — ONNX Runtime QNN EP)
from .base import Backend, BenchResult


class QNNBackend(Backend):
    soc_name = "qcs8550"

    def build(self, onnx_path, precision, calib_path):
        # ort.InferenceSession(onnx, providers=["QNNExecutionProvider"], ...)로
        # context binary(.bin) 생성/캐시. HTP(NPU) 백엔드 옵션 지정.
        raise NotImplementedError("QNN context binary 생성 — QNN SDK/디바이스 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="device required (QCS8550, ONNX Runtime QNN EP)",
        )
```

```python
# bench/backends/drpai.py  (Renesas RZ/V2H — rzv_drp-ai_tvm : feed-forward only)
from .base import Backend, BenchResult


class DRPAIBackend(Backend):
    soc_name = "rzv2h"

    def build(self, onnx_path, precision, calib_path):
        # renesas rzv_drp-ai_tvm 컴파일러 호출. INT8 고정.
        # feed-forward만 지원 → 그래프에 Loop/If 있으면 여기서 실패시켜 CI가 잡게.
        raise NotImplementedError("DRP-AI TVM compile hook — 보드 필요")

    def run(self, inputs):
        raise NotImplementedError

    def measure(self, model, precision, loader, evaluator, warmup=20, iters=200):
        return BenchResult(
            model=model, soc=self.soc_name, precision=precision,
            latency_ms=float("nan"), peak_mem_mb=float("nan"),
            accuracy=float("nan"), engine_build_s=0.0,
            notes="board required (RZ/V2H, feed-forward only)",
        )
```

> 💡 팁: 보드가 없는 SoC는 **에러로 죽이지 말고** `notes`에 사유를 남기고 매트릭스에서 회색 처리하라. 데스크톱(RTX) 결과만으로도 CI 회귀 테스트는 성립한다. 보드가 도착하면 `build`/`run`의 `NotImplementedError`만 실제 SDK 호출로 채우면 되고, `measure`/`collect`/리포트/CI는 **한 줄도 바꿀 필요가 없다.** 이것이 4-2에서 계약을 못 박은 배당금이다.

### 4-5. 벤치 설정 `config.yaml` + 실행 러너 `run_bench.py`

무엇을 무엇으로 돌릴지 **코드가 아니라 설정**으로 관리한다. `config.yaml`은 **모델 × 타깃(SoC 백엔드) × precision** 3차원 매트릭스를 선언한다.

```yaml
# bench/config.yaml
# 스키마:
#   models[]:  name, onnx, calib      — 무엇을(모델)
#   backends[]:                        — 무엇으로(타깃 SoC 백엔드 키)
#   precisions[]:                      — 어떤 정밀도로
#   exclude[]: (선택) 특정 조합 제외    — 예: DRP-AI는 fp16 없음
defaults:
  warmup: 20
  iters: 200

models:
  - name: bevformer
    onnx: models/bevformer.onnx
    calib: calib/nuscenes_calib_200.npy
  # - name: yolox            # 모델 추가 = 여기 한 블록 추가
  #   onnx: models/yolox.onnx
  #   calib: calib/coco_calib_200.npy

backends:                    # 이 실행 환경(runner)에서 가능한 것만 켠다
  - trt                      # 데스크톱/CI: TRT. 보드 백엔드는 label 붙은 runner에서.
  # - tidl
  # - qnn
  # - drpai

precisions: [fp32, fp16, int8]

exclude:                     # 물리적으로 불가능한 조합은 제외 (매트릭스에서 빠짐)
  - {backend: drpai, precision: fp16}   # DRP-AI는 INT8 고정
  - {backend: drpai, precision: fp32}
  - {backend: tidl,  precision: fp32}   # TIDL은 실효 없음(예시)
```

```python
# bench/run_bench.py  (CLI 엔트리포인트: config.yaml 순회 → results/*.json)
import argparse
import importlib
import itertools
import pathlib
import sys
import yaml

BACKENDS = {
    "trt":   ("backends.trt",   "TRTBackend"),
    "tidl":  ("backends.tidl",  "TIDLBackend"),
    "qnn":   ("backends.qnn",   "QNNBackend"),
    "drpai": ("backends.drpai", "DRPAIBackend"),
}


def load_backend(key):
    mod, cls = BACKENDS[key]
    return getattr(importlib.import_module(mod), cls)()


def is_excluded(backend, precision, exclude):
    for rule in exclude or []:
        if rule.get("backend") == backend and rule.get("precision") == precision:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only-backend", default=None,
                    help="config를 무시하고 이 백엔드만 (예: CI에서 trt만)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg.get("defaults", {})
    warmup, iters = d.get("warmup", 20), d.get("iters", 200)
    exclude = cfg.get("exclude", [])
    backends = [args.only_backend] if args.only_backend else cfg["backends"]

    # loader/evaluator 는 프로젝트 데이터셋에 맞게 구현(3단계 자산 재사용)
    from data import Loader, Evaluator  # noqa

    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    n_ok = n_skip = 0
    # 3중 순회 = 매트릭스의 셀 하나하나
    for m, prec, bk in itertools.product(cfg["models"], cfg["precisions"], backends):
        if is_excluded(bk, prec, exclude):
            print(f"skip (excluded): {m['name']} × {bk} × {prec}")
            n_skip += 1
            continue
        be = load_backend(bk)
        try:
            be.build(m["onnx"], prec, m.get("calib"))
            res = be.measure(m["name"], prec, Loader(m), Evaluator(),
                             warmup=warmup, iters=iters)
        except NotImplementedError as e:
            # stub 백엔드(보드 없음): 회색 결과로 봉인하고 계속
            from backends.base import BenchResult
            res = BenchResult(m["name"], be.soc_name, prec,
                              float("nan"), float("nan"), float("nan"), 0.0,
                              notes=f"not implemented: {e}")
        fn = be.collect(res)            # base.py의 공통 collect
        print("wrote", fn)
        n_ok += 1
    print(f"done: {n_ok} results, {n_skip} skipped")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

```bash
# 실행 (bench/ 에서)
python3 run_bench.py
# → results/bevformer__rtx__fp32.json, ...__fp16.json, ...__int8.json 생성

# CI에서 GPU runner는 TRT만: (config 편집 없이)
python3 run_bench.py --only-backend trt
```

예상 출력:

```
wrote results/bevformer__rtx__fp32.json
wrote results/bevformer__rtx__fp16.json
wrote results/bevformer__rtx__int8.json
done: 3 results, 0 skipped
```

> 💡 팁 — `itertools.product`가 핵심: 모델·precision·백엔드 3중 순회가 한 줄이다. 매트릭스가 커져도 코드는 그대로고 `config.yaml`만 자란다. "조합을 코드가 아니라 데이터로 관리"하는 것이 조합 폭발을 견디는 유일한 방법이다.

### 4-6. `report/generate.py` — md / csv / HTML 매트릭스 자동 생성

`results/`의 JSON들을 모아 pandas로 피벗해 **마크다운 + CSV + HTML** 3종을 뱉는다. CSV는 회귀 baseline의 소스, 마크다운은 README/PR용, HTML은 하이라이트가 들어간 대시보드다.

```python
# bench/report/generate.py
import glob
import json
import pathlib
import pandas as pd

RESULTS_DIR = "results"
OUT_DIR = pathlib.Path("report")


def collect(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """results/*.json → 하나의 DataFrame (긴 형식)."""
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/*.json"))]
    return pd.DataFrame(rows)


def _highlight_html(df_long: pd.DataFrame) -> str:
    """HTML 대시보드: precision별 최저 latency를 초록, mAP 하락을 빨강으로.

    pandas Styler는 jinja2를 요구한다(3절에서 설치).
    """
    lat = df_long.pivot_table(index=["model", "soc"], columns="precision",
                              values="latency_ms")
    styler = (
        lat.style
        .format("{:.2f}")
        # 각 행(모델·soc)에서 가장 빠른 precision 셀을 초록 하이라이트
        .highlight_min(axis=1, color="#c6efce")
        .set_caption("Latency (ms, median) — 행별 최저값 강조")
        .set_table_styles([
            {"selector": "th", "props": [("background", "#f2f2f2"),
                                         ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "right"),
                                         ("padding", "4px 10px")]},
        ])
    )
    return styler.to_html()


def main():
    df = collect()
    if df.empty:
        raise SystemExit("no results/*.json — run_bench.py 먼저 실행")
    OUT_DIR.mkdir(exist_ok=True)

    # (1) CSV — 정렬해 저장 (회귀 테스트의 baseline 소스). 정렬은 diff 안정성을 위해.
    key = ["model", "soc", "precision"]
    df_sorted = df.sort_values(key).reset_index(drop=True)
    df_sorted.to_csv(OUT_DIR / "matrix.csv", index=False)

    # (2) 사람이 읽는 피벗: (model,soc) × precision
    lat = df.pivot_table(index=["model", "soc"], columns="precision",
                         values="latency_ms")
    acc = df.pivot_table(index=["model", "soc"], columns="precision",
                         values="accuracy")

    # (3) Markdown (README/PR 코멘트용)
    md = [
        "# 성능 매트릭스 (자동 생성)\n",
        f"> 생성: {pd.Timestamp.now():%Y-%m-%d %H:%M} · 총 {len(df)} 케이스 "
        f"· TRT {df['trt_version'].dropna().unique().tolist()}\n",
        "## Latency (ms, median)\n", lat.round(2).to_markdown(), "\n",
        "## Accuracy (mAP)\n", acc.round(4).to_markdown(), "\n",
    ]
    (OUT_DIR / "matrix.md").write_text("\n".join(md))

    # (4) HTML 대시보드 (하이라이트)
    (OUT_DIR / "matrix.html").write_text(_highlight_html(df))

    print("wrote report/matrix.csv, matrix.md, matrix.html")


if __name__ == "__main__":
    main()
```

```bash
python3 report/generate.py   # bench/ 에서
# → report/matrix.csv, report/matrix.md, report/matrix.html
```

예상 출력(`report/matrix.md`):

```markdown
## Latency (ms, median)
| ('model','soc')      |   fp32 |   fp16 |   int8 |
|:---------------------|-------:|-------:|-------:|
| ('bevformer','rtx')  |  41.20 |  18.70 |  12.30 |

## Accuracy (mAP)
| ('model','soc')      |   fp32 |   fp16 |   int8 |
|:---------------------|-------:|-------:|-------:|
| ('bevformer','rtx')  | 0.4120 | 0.4100 | 0.3820 |
```

> 💡 팁: `to_markdown()`과 Styler의 `to_html()`은 각각 `tabulate`·`jinja2`가 있어야 동작한다(3절에서 설치). **CSV는 회귀 테스트의 baseline**, 마크다운은 PR/README, **HTML은 팀 대시보드**(초록=행별 최속, 셀 색으로 한눈에)로 쓴다. `sort_values`로 CSV를 정렬해 저장하면 결과 순서가 실행마다 안 흔들려 git diff가 깨끗해진다.

### 4-7. 회귀 테스트 `tests/test_regression.py` — mAP 1% 하락 시 실패 + 골든 파일 이중화

핵심 규칙: **커밋 후 측정한 mAP가 baseline보다 1%p(절대) 이상 낮으면 테스트 실패.** 두 층으로 지킨다 — (A) 임계값 기반 assertion, (B) `pytest-regressions` 골든 파일 비교.

먼저 baseline을 한 번 만들어 커밋한다.

```bash
# 골든(기준) 매트릭스 확정 후 저장 — 이 파일을 git에 커밋
cp report/matrix.csv tests/baseline_matrix.csv
git add tests/baseline_matrix.csv
```

```python
# bench/tests/test_regression.py
import glob
import json
import pathlib
import pandas as pd
import pytest

MAP_DROP_TOLERANCE = 0.01      # 1%p 절대 하락 허용치
LAT_REGRESS_FACTOR = 1.10      # latency 10% 이상 악화도 경고성 실패(선택)

BASELINE = pathlib.Path(__file__).parent / "baseline_matrix.csv"


def _load_current() -> pd.DataFrame:
    rows = [json.load(open(p)) for p in glob.glob("results/*.json")]
    assert rows, "results/*.json 없음 — 벤치를 먼저 실행하라"
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def merged():
    base = pd.read_csv(BASELINE)
    cur = _load_current()
    key = ["model", "soc", "precision"]
    return base.merge(cur, on=key, suffixes=("_base", "_cur"))


# --- (A) 임계값 기반 회귀 ---
def test_no_map_regression(merged):
    """mAP가 baseline 대비 1%p 이상 하락한 케이스가 있으면 실패."""
    m = merged.dropna(subset=["accuracy_base", "accuracy_cur"]).copy()
    m["drop"] = m["accuracy_base"] - m["accuracy_cur"]
    bad = m[m["drop"] > MAP_DROP_TOLERANCE]
    assert bad.empty, (
        "mAP 회귀 감지:\n"
        + bad[["model", "soc", "precision",
               "accuracy_base", "accuracy_cur", "drop"]].to_string(index=False)
    )


def test_no_latency_regression(merged):
    """latency가 baseline 대비 10% 이상 악화되면 실패(선택)."""
    m = merged.dropna(subset=["latency_ms_base", "latency_ms_cur"])
    bad = m[m["latency_ms_cur"] > m["latency_ms_base"] * LAT_REGRESS_FACTOR]
    assert bad.empty, (
        "Latency 회귀 감지:\n"
        + bad[["model", "soc", "precision",
               "latency_ms_base", "latency_ms_cur"]].to_string(index=False)
    )


# --- (B) 골든 파일 비교 (pytest-regressions) ---
def test_matrix_matches_golden(dataframe_regression):
    """현재 매트릭스를 골든과 통째로 비교. 의도적 변경은 --force-regen으로만 갱신.

    dataframe_regression fixture는 pytest-regressions(v3.0+)가 제공.
    최초 실행(또는 --force-regen) 시 tests/test_regression/ 아래에
    골든 파일을 생성하고, 이후엔 그와 비교한다.
    """
    cur = _load_current().sort_values(["model", "soc", "precision"]).reset_index(drop=True)
    # 부동소수 흔들림 허용치(default_tolerance)로 flaky 방지
    dataframe_regression.check(
        cur[["model", "soc", "precision", "latency_ms", "accuracy"]],
        default_tolerance=dict(atol=1e-3, rtol=1e-3),
    )
```

```bash
pytest tests/ -v   # bench/ 에서
# 골든을 의도적으로 갱신할 때만:
pytest tests/ --force-regen
```

예상 실패 출력:

```
FAILED tests/test_regression.py::test_no_map_regression
AssertionError: mAP 회귀 감지:
    model soc precision  accuracy_base  accuracy_cur   drop
bevformer rtx      int8         0.3820        0.3660  0.016
```

> 💡 팁 — 골든 파일 갱신 규율: baseline을 **의도적으로** 바꿀 때(모델 개선 등)만 갱신하고, 커밋 메시지에 사유를 남긴다. 임계값 방식은 `baseline_matrix.csv`를, 골든 방식은 `--force-regen`을 쓴다([pytest-regressions](https://pypi.org/project/pytest-regressions/), v3.0+, 2026-07 기준). 무심코 갱신하는 것이 회귀 방지 시스템을 무력화하는 1번 원인이다. **두 층(A/B)을 같이 두는 이유**: (A)는 "얼마나 나빠지면 실패"라는 정책을, (B)는 "숫자가 조용히 바뀌면 눈에 띄게"라는 감시를 담당한다 — 역할이 다르다.

### 4-8. self-hosted GPU runner 등록

GPU 벤치는 GitHub 클라우드 러너(GPU 없음)에서 못 돈다. **RTX가 달린 우리 데스크톱을 self-hosted runner로 등록**한다.

먼저 GPU가 보이는지 확인한다.

```bash
nvidia-smi   # GPU가 보여야 함. 안 보이면 드라이버부터 (1단계 참고)
```

레포의 **Settings → Actions → Runners → New self-hosted runner** 에서 나오는 토큰으로 등록한다(아래 버전/토큰은 예시).

```bash
# 러너 다운로드 (버전은 릴리스 페이지에서 최신 확인, 2026-07 기준 v2.336.0)
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.336.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.336.0/actions-runner-linux-x64-2.336.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.336.0.tar.gz

# 등록: --labels 에 gpu 를 꼭 넣어 워크플로가 지목할 수 있게 함
./config.sh --url https://github.com/<org>/<repo> \
  --token <RUNNER_TOKEN> \
  --name "rtx-bench-01" \
  --labels "self-hosted,linux,x64,gpu" \
  --work "_work"

# 상시 서비스로 등록(재부팅에도 살아있게)
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

> ⚠️ 주의 — 최소 버전 강제(2026): GitHub는 self-hosted runner의 **최소 버전 강제**를 재개했다. 릴리스 30일 이상 지난 오래된 runner는 job 실행이 거부될 수 있다([GitHub Changelog, 2026-06-12](https://github.blog/changelog/2026-06-12-github-actions-minimum-version-enforcement-timeline-for-self-hosted-runners/)). runner를 `svc.sh`로 상시 서비스로 돌린 채 **방치**하면 어느 날 CI가 "runner too old"로 멈춘다. 정기적으로 최신 버전으로 재설치(또는 자동 업데이트 유지)하라. 이건 실제로 팀들이 자주 당하는 함정이다.

> ⚠️ 주의 — 보안: self-hosted runner는 **private 레포에서만** 쓰는 것이 GitHub 권장이다([GitHub Docs](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners)). public 레포에서 fork PR이 우리 GPU 머신에서 임의 코드를 실행할 수 있기 때문이다. private 레포이거나, 최소한 fork PR에는 runner를 붙이지 않도록 트리거를 제한하라.

> 💡 팁 — Docker 격리: runner 호스트를 더럽히기 싫으면 워크플로 job을 `container:`로 감싸고 `--gpus all`을 준다. 이때 호스트에 `nvidia-container-toolkit`이 설치돼 있어야 한다(`sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker`). ([NVIDIA Container Toolkit 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### 4-9. `ci.yml` — 커밋마다 벤치 + 회귀 게이트 (캐시 포함)

```yaml
# .github/workflows/ci.yml
name: quant-bench-regression

on:
  push:
    branches: [main, "feat/**"]
  pull_request:
    branches: [main]

# 같은 브랜치에 새 푸시가 오면 이전 실행 취소 (GPU runner 점유 낭비 방지)
concurrency:
  group: bench-${{ github.ref }}
  cancel-in-progress: true

jobs:
  bench:
    # 위에서 등록한 GPU runner를 지목 (모든 라벨이 매칭돼야 함)
    runs-on: [self-hosted, linux, x64, gpu]
    defaults:
      run:
        working-directory: bench
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true   # 체크포인트/calib을 Git LFS로 관리하는 경우

      - name: Sanity - GPU visible
        run: nvidia-smi

      # pip 캐시 (self-hosted에서도 유효 — 의존성 재설치 시간 절약)
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ hashFiles('bench/requirements.txt') }}
          restore-keys: pip-${{ runner.os }}-

      # TRT 엔진 캐시 (빌드가 수 분 걸리므로 계산 그래프 해시로 캐시)
      - name: Cache TRT engines
        uses: actions/cache@v4
        with:
          path: bench/.trt_cache
          key: trt-${{ hashFiles('bench/models/*.onnx') }}-${{ hashFiles('bench/config.yaml') }}

      - name: Install deps
        run: |
          python3 -m pip install --upgrade \
            pandas tabulate jinja2 pytest pytest-regressions pyyaml

      - name: Run benchmark (build + measure) — GPU runner는 TRT만
        run: python3 run_bench.py --only-backend trt

      - name: Generate performance matrix (md/csv/html)
        run: python3 report/generate.py

      - name: Regression gate (mAP drop >1%p fails)
        run: pytest tests/ -v --maxfail=1

      - name: Upload matrix as artifact
        if: always()   # 실패해도 매트릭스는 올려서 원인 확인
        uses: actions/upload-artifact@v4
        with:
          name: performance-matrix
          path: |
            bench/report/matrix.md
            bench/report/matrix.csv
            bench/report/matrix.html
```

```bash
# 로컬에서 CI가 하는 일을 그대로 재현(푸시 전 검증)
cd bench && python3 run_bench.py --only-backend trt \
  && python3 report/generate.py && pytest tests/ -v
```

> 💡 팁: `runs-on` 배열은 **AND 조건**이다 — 러너가 `gpu` 라벨을 안 달고 있으면 job이 영원히 대기(pending)한다. 등록 시 `--labels`와 워크플로의 `runs-on`을 반드시 일치시켜라([self-hosted runner 설정, oneuptime 2026-01](https://oneuptime.com/blog/post/2026-01-25-github-actions-self-hosted-runners/view)).

> 💡 팁 — `concurrency` + `cancel-in-progress`: GPU runner는 대개 1대다. 이 블록이 없으면 빠르게 여러 번 푸시할 때 벤치가 큐에 쌓여 몇 시간씩 밀린다. 브랜치별로 "최신 커밋만 벤치"하도록 이전 실행을 취소하면 GPU 점유가 깔끔해진다.

### 4-10. (선택) 실험 추적 도입 여부 — MLflow(무료 셀프호스팅) vs W&B

`results/*.json` + git이면 사실 **최소 추적은 이미 된다**(커밋 = 실험 스냅샷). 그 위에 실험 추적 도구를 얹을지는 팀 상황에 따른다.

| 항목 | MLflow | Weights & Biases (W&B) |
|------|--------|------------------------|
| 라이선스/비용 | Apache-2.0, **무료·셀프호스팅** | SaaS, 유료(대략 $50/seat/월 수준, 2026 기준) |
| 설치 | `pip install mlflow` → `mlflow ui` (localhost:5000) | 클라우드 계정 + `pip install wandb` |
| 저장 | 백엔드 스토어(SQLite/Postgres) + 아티팩트 스토어(로컬/S3) 분리 | W&B 클라우드(또는 self-hosted 엔터프라이즈) |
| 강점 | 로컬 통제·온프렘·규제 환경 | 협업 대시보드·시각 비교·sweep·리포트 |
| 임베디드 팀 적합 | 사내망/보안 환경에서 무난 | 클라우드 반출 가능할 때 편함 |

> 출처: [MLflow vs W&B 2026 비교](https://deploybase.ai/articles/mlflow-vs-wandb), [MLflow Tracking Server 문서](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

임베디드/보안 환경(자동차 SoC 프로젝트가 흔히 그렇다)에서는 **MLflow 셀프호스팅**이 무난하다. 아래처럼 벤치 결과를 파라미터·메트릭·아티팩트로 남긴다.

```python
# bench/report/track_mlflow.py  (선택) — results/*.json을 MLflow에 로깅
import glob
import json
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")   # 사내 서버 IP:5000
mlflow.set_experiment("quant-bench")               # 없으면 자동 생성

for p in glob.glob("results/*.json"):
    r = json.load(open(p))
    if r.get("accuracy") != r.get("accuracy"):     # NaN(보드필요) 케이스는 스킵
        continue
    run_name = f"{r['model']}-{r['soc']}-{r['precision']}"
    with mlflow.start_run(run_name=run_name):
        # 파라미터: "무엇을 어떻게 돌렸나" (조건)
        mlflow.log_params({
            "model": r["model"], "soc": r["soc"],
            "precision": r["precision"], "trt_version": r.get("trt_version", ""),
        })
        # 메트릭: "결과 수치"
        mlflow.log_metrics({
            "latency_ms": r["latency_ms"],
            "latency_p95_ms": r.get("latency_p95_ms", float("nan")),
            "peak_mem_mb": r["peak_mem_mb"],
            "mAP": r["accuracy"],
        })
        # 아티팩트: 결과 산출물 파일 자체
        mlflow.log_artifact("report/matrix.csv")
        mlflow.log_artifact("report/matrix.html")
        mlflow.log_artifact(p)                      # 이 케이스의 raw json
print("logged to MLflow at http://127.0.0.1:5000")
```

```bash
# 셀프호스팅 서버 (사내 서버에서 상시 구동)
# 백엔드 스토어는 SQLite, 아티팩트는 로컬 디렉토리에 (팀 규모 커지면 Postgres+S3로)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000
# 대시보드: http://<서버IP>:5000  → run 비교, 메트릭 정렬, 아티팩트 다운로드

# 로깅 실행 (bench/ 에서)
python3 report/track_mlflow.py
```

> ⚠️ 주의 — LAN 노출 보안(MLflow 3.5.0+): `--host 0.0.0.0`으로 사내망에 열면, 최신 MLflow는 DNS rebinding/CORS 보호 미들웨어가 기본 활성이라 브라우저 접근이 막힐 수 있다. 사내 IP/호스트를 신뢰 목록에 넣어야 한다([MLflow Tracking Server 문서](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)). 인증 없이 넓은 네트워크에 열지 마라.

> 💡 팁: 도입 자체가 목적이 되면 안 된다. **CI 회귀 게이트가 1순위**, 실험 추적은 실험 수가 많아지고 여러 사람이 결과를 비교해야 할 때 붙여라. `log_params`(조건) / `log_metrics`(수치) / `log_artifact`(파일)의 3분할을 지키면, 나중에 "fp16인데 왜 이 커밋만 mAP가 낮지?"를 대시보드에서 정렬 한 번으로 찾는다. 결정을 내렸으면 그 근거를 `decision_log.md`에 남겨라(그게 이 단계의 정신이다).

---

## 5) 예시 / 결과 해석 — 산출물 3종

### 5-1. 성능 매트릭스 (자동 생성)

`report/matrix.md`의 실제 형태(예시 값):

| model / soc | precision | latency(ms) | peak mem(MB) | mAP |
|-------------|-----------|-------------|--------------|-----|
| bevformer / rtx | fp32 | 41.2 | 3120 | 0.412 |
| bevformer / rtx | fp16 | 18.7 | 1980 | 0.410 |
| bevformer / rtx | int8 | 12.3 | 1240 | 0.382 |
| bevformer / orin | int8 | 33.5 | 980 | 0.379 |
| bevformer / tda4vm | int8 | 71.0 | 512 | 0.361 |
| bevformer / rzv2h | int8 | — | — | 보드필요 |

해석 포인트:

- **fp16은 mAP 손실 거의 없이 2배↑ 빠름** → 안전한 기본값. "일단 fp16"은 대부분 옳다.
- **int8은 mAP 3%p 하락** → 이게 허용 범위인가가 팀의 의사결정 사항이고, 그 판단을 decision log에 남긴다.
- **SoC를 내려갈수록**(rtx → orin → tda4vm) latency는 늘고 mem은 준다 → 하드웨어 예산과 정확도의 트레이드오프. 매트릭스는 이 트레이드오프를 **한 화면**에서 논쟁 가능하게 만든다.
- **회색(보드필요) 행**도 매트릭스에 남긴다 → "아직 측정 못 함"과 "측정했더니 나쁨"은 다르다. 빈칸이 아니라 명시적 '보드필요'가 정직한 보고다.

### 5-2. `design_rules.md` — 작성 방법론 + 실제 도출

**이 파일이 JD가 명시적으로 요구하는 산출물**이다. 하지만 핵심은 4분류 표가 아니라, **각 규칙이 앞 단계 산출물에서 어떻게 도출됐는지**가 추적 가능해야 한다는 것이다.

#### 작성 방법론 (규칙은 "의견"이 아니라 "근거"에서 나온다)

| 규칙의 근거 소스 | 무엇을 읽고 | 어떤 규칙으로 승격되나 |
|------------------|-------------|------------------------|
| `layer_sensitivity.csv` (1단계) | INT8 시 mAP 하락이 큰 레이어 | → ⚠️ "이 레이어는 FP16 유지" |
| `onnx_export_failures.md` (2단계) | export/compile에서 죽은 op | → ❌ "이 op는 그래프에서 제거" |
| 4-target 매트릭스 (`report/`) | SoC별로 특정 op가 실패/느림 | → 📏 "이 SoC엔 이 제약" / SoC×규칙 표 |

즉 `design_rules.md`를 쓰는 절차는 이렇다:

1. `layer_sensitivity.csv`를 mAP-drop 내림차순으로 정렬 → 상위 레이어를 ⚠️(FP16 유지)로.
2. `onnx_export_failures.md`의 실패 op를 ❌로, 각 항목에 실패 번호(#3 등)를 인용.
3. `report/matrix.md`에서 특정 SoC만 mAP가 유독 낮거나 NaN이면 → 그 SoC의 📏 제약 또는 SoC×규칙 표의 ❌/⚠️로.
4. 각 규칙 옆에 **근거 파일명·라인/번호**를 병기 → "왜?"에 즉답 가능하게.

#### 실제 도출 예시

- ⚠️ "Softmax/Attention은 FP16 유지" ← `layer_sensitivity.csv`에서 cross-attention INT8 시 mAP **-2.8%p**(전체 손실의 70%). → decision_log 2026-07-27 항목으로 링크.
- ❌ "grid_sample 금지" ← `onnx_export_failures.md #3`: TIDL 컴파일 시 grid_sample 크래시. → SoC×규칙 표에서 TIDL/DRP-AI 열이 ❌.
- 📏 "DRP-AI feed-forward only" ← 4-target 매트릭스에서 RZ/V2H가 Loop 포함 그래프로 컴파일 실패(보드 로그). 

#### `design_rules.md` (전체 템플릿)

```markdown
# design_rules.md — 양자화·배포 설계 규칙

> 근거: layer_sensitivity.csv(1단계), onnx_export_failures.md(2단계), 4-target 성능 매트릭스.
> 갱신 규율: 규칙 추가/변경 시 decision_log에 근거 링크를 남긴다.
> 각 규칙 끝의 (근거: ...) 는 추적용 — 리뷰어가 즉시 검증할 수 있어야 한다.

## ✅ 권장 (fuse 가능 · per-channel 양자화 친화)
- Conv + BN + ReLU 패턴  → fuse 가능, per-channel INT8 안전  (근거: layer_sensitivity.csv, drop<0.2%p)
- Depthwise-separable Conv → 대부분 백엔드에서 잘 지원
- ReLU6 / clamp 계열 활성화 → 양자화 범위 안정적

## ⚠️ 주의 (되지만 조건부)
- LayerNorm       → activation outlier로 INT8 손실 큼. **SmoothQuant 필수**(arXiv:2211.10438)
- GELU            → 하드웨어에 따라 **LUT 근사**로 대체 필요(정확도 검증할 것)
- Softmax/Attention → FP16 유지 권장 (근거: layer_sensitivity.csv, cross-attn INT8 -2.8%p)

## ❌ 금지 (백엔드 미지원/불안정 — export 단계에서 제거)
- grid_sample     → TIDL에서 불안정 (근거: onnx_export_failures.md #3)
- dynamic shape   → 다수 임베디드 백엔드 미지원. 고정 shape로 export
- Loop / If (제어흐름) → QNN·DRP-AI 미지원 (근거: 4-target 매트릭스, RZ/V2H 컴파일 실패)

## 📏 제약 (SoC별 하드 리밋)
- BEV grid 200×200 초과 → SRAM 초과. **128×128 권장** (SoC별 SRAM 확인)
- DRP-AI(RZ/V2H)  → **feed-forward only**. 모든 노드 입력에 initializer 필요
- TDA4VM          → workspace/heap 상한 확인, 큰 conv는 분할 필요
- Orin            → TRT workspace를 dGPU 대비 축소(공유 메모리)

## SoC × 규칙 적용표
| 규칙 | TensorRT(RTX/Orin) | TIDL(TDA4) | QNN(QCS) | DRP-AI(RZ/V2H) |
|------|:--:|:--:|:--:|:--:|
| grid_sample | ⚠️ | ❌ | ⚠️ | ❌ |
| dynamic shape | ⚠️ | ❌ | ❌ | ❌ |
| Loop/If | ⚠️ | ❌ | ❌ | ❌ |
| LayerNorm(SmoothQuant) | ✅ | ⚠️ | ⚠️ | ⚠️ |
```

### 5-3. `decision_log.md` — 작성 방법론 + 템플릿

"왜 이 레이어를 FP16으로 남겼는가"를 기록. 미래의 나와 후임이 같은 실수를 반복하지 않게 한다.

#### 작성 방법론 (좋은 로그 = 미래에 "되돌릴 조건"이 적혀 있다)

한 항목은 5칸을 채운다: **날짜 / 결정 / 근거(데이터) / 대안과 기각 사유 / 재검토 조건.** 이 중 가장 자주 빠지고 가장 중요한 게 **재검토 조건**이다 — "언제 이 결정을 뒤집을 것인가"가 없으면, 그 결정은 6개월 뒤 "왜 이렇게 돼 있지?"라는 미스터리로 굳는다. 근거 칸에는 반드시 **수치와 출처 파일명**(layer_sensitivity.csv, onnx_export_failures.md #N, 매트릭스 셀)을 적어 `design_rules.md`와 상호 링크되게 한다.

```markdown
# decision_log.md — 양자화 의사결정 기록

각 항목: 날짜 / 결정 / 근거(데이터) / 대안과 기각 사유 / 재검토 조건

---
## 2026-07-27 — Attention 블록을 INT8이 아닌 FP16으로 유지
- **결정**: BEVFormer의 cross-attention을 FP16으로 남기고 backbone만 INT8.
- **근거**: layer_sensitivity.csv에서 해당 레이어 INT8 시 mAP -2.8%p (전체 손실의 70% 차지).
- **대안·기각**: 전체 INT8(mAP 0.382, 목표 0.40 미달) / SmoothQuant 후 INT8(시도했으나 attention은 여전히 -1.9%p) → FP16 혼합이 최선.
- **재검토 조건**: 타깃 SoC가 INT8 attention을 하드웨어 지원하면 재측정.

---
## 2026-07-27 — grid_sample 제거, bilinear 수동 구현으로 대체
- **결정**: BEV projection의 grid_sample을 export 전에 커스텀 op로 치환.
- **근거**: onnx_export_failures.md #3 — TIDL 컴파일 시 grid_sample에서 크래시.
- **대안·기각**: 플러그인 작성(유지보수 부담↑, SoC마다 재작성) → 그래프 치환이 이식성 우위.
- **재검토 조건**: TIDL 릴리스 노트에 grid_sample 정식 지원 명시되면 원복 검토.

---
## 2026-07-27 — 실험 추적 도구: MLflow 셀프호스팅 채택
- **결정**: W&B 대신 사내 MLflow(:5000) 사용.
- **근거**: 데이터 반출 제약(자동차 고객 데이터). 무료·온프렘.
- **대안·기각**: W&B(대시보드 우수하나 SaaS·비용·데이터 반출) → 규제 환경에서 부적합.
- **재검토 조건**: 팀 확장·외부 협업 시 재평가.
```

> 💡 팁 — 로그와 규칙의 연결: `decision_log.md`의 결정이 일반화되면 `design_rules.md`의 규칙으로 **승격**시킨다(예: "이 모델의 attention"이 "attention 계열 전반"으로). 반대로 규칙에 예외가 생기면 그 예외를 로그에 남긴다. 두 문서가 서로를 인용하며 자라는 것이 건강한 상태다.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| CI job이 영원히 `Queued/pending` | `runs-on` 라벨과 runner `--labels` 불일치 | 두 곳의 라벨을 정확히 일치. `gpu` 라벨 누락이 흔함 |
| CI가 `runner version too old`로 거부 | 2026 최소 버전 강제 + 오래된 self-hosted runner | runner를 최신(v2.336.0대)으로 재설치. 자동 업데이트 유지 |
| runner에서 `nvidia-smi: command not found` | 드라이버 미설치 / PATH 문제 | 호스트에 NVIDIA 드라이버 설치, 재부팅 ([1단계](01_environment_setup.md)) |
| container job에서 GPU 안 보임 | `--gpus all` 없음 / toolkit 미설치 | 워크플로에 `options: --gpus all`, 호스트에 `nvidia-container-toolkit` |
| `df.to_markdown()`에서 ImportError | `tabulate` 미설치 | `pip install tabulate` |
| `Styler.to_html()`에서 ImportError | `jinja2` 미설치 | `pip install jinja2` |
| `onnxruntime-gpu` 설치했는데 CPU로만 돔 | CUDA 13 기본 휠을 CUDA 12.8에 설치 / provider 미지정 | CUDA 12 인덱스로 재설치(3절), 세션에 `providers=[...]` 명시 |
| 회귀 테스트가 baseline 없다고 실패 | `tests/baseline_matrix.csv` 미커밋 | 첫 골든 매트릭스를 만들어 커밋 |
| `dataframe_regression` fixture 없음 | `pytest-regressions` 미설치 | `pip install pytest-regressions` (v3.0+) |
| mAP가 매 실행마다 미세하게 흔들려 CI 불안정(flaky) | 시드 미고정 / 비결정적 커널 | eval 시드 고정, latency는 median, tolerance를 소폭(예: atol=1e-3) 여유 |
| latency가 실행마다 크게 튐 | warmup 부족 / GPU 클럭 변동 | warmup↑, `nvidia-smi -lgc`로 클럭 고정, median/p95 사용 |
| `build_serialized_network`가 `None` 반환 | ONNX에 미지원 op / config 오류 | parser 에러 로그 확인, `design_rules.md`의 ❌ op가 그래프에 있는지 점검 |
| GPU runner 1대에 벤치가 계속 밀림 | 동시 실행 미제어 | `ci.yml`에 `concurrency` + `cancel-in-progress` 추가(4-9) |
| fork PR이 self-hosted runner에서 도는 게 불안 | public 레포 + self-hosted 조합 | private 레포로 옮기거나 `pull_request` 트리거를 신뢰 브랜치로 제한 |

> 🔴 함정 — "CI는 초록불인데 실차에서 성능이 다르다": CI는 **캘리브레이션셋의 대표성**과 **평가셋의 대표성**만큼만 믿을 수 있다. 매트릭스가 예쁘게 나와도 calib/eval 분포가 실배포와 다르면 소용없다. 데이터 큐레이션은 자동화의 사각지대다(4-1 함정 참고).

> 🔴 함정 — "골든 파일이 조용히 갱신돼 회귀 게이트가 무력화됨": `--force-regen`이나 `baseline_matrix.csv` 덮어쓰기를 **아무 커밋에서나** 하면, 정확도가 떨어져도 골든이 함께 내려가 테스트가 영원히 통과한다. 골든 갱신은 별도 PR로 분리하고, 리뷰어가 "왜 갱신하는가"를 decision_log에서 확인하게 하라.

---

## 7) 산출물(Deliverables)

이 단계를 마치면 **레포에 다음이 남아야** 한다(면접·인수인계용 핵심 자산).

- [ ] `bench/` 디렉토리 (models/calib/backends/report/tests + config.yaml + run_bench.py)
- [ ] `backends/base.py` (공통 인터페이스 build/run/measure/collect + BenchResult) + `backends/trt.py` (동작하는 구현 1개 이상)
- [ ] `backends/{tidl,qnn,drpai}.py` (동일 시그니처 stub — 보드 없이도 매트릭스에 '보드필요'로 표기)
- [ ] `config.yaml` (모델×타깃×precision 매트릭스 스키마) + `run_bench.py` (조합 순회 러너)
- [ ] `report/generate.py` → **자동 생성된 `report/matrix.md` + `matrix.csv` + `matrix.html`** (성능 매트릭스 ①)
- [ ] `.github/workflows/ci.yml` (커밋마다 벤치 + mAP 회귀 게이트 + 캐시) + 등록된 self-hosted GPU runner
- [ ] `tests/test_regression.py` (임계값 + 골든 파일 이중화) + `tests/baseline_matrix.csv` (골든)
- [ ] **`design_rules.md`** (JD가 요구하는 산출물 ②) — ✅/⚠️/❌/📏 4분류 + **각 규칙에 앞 단계 산출물 근거 연결**
- [ ] **`decision_log.md`** (산출물 ③) — 근거·재검토 조건 있는 결정 최소 3건
- [ ] (선택) 실험 추적 도입 여부 결정 + `report/track_mlflow.py` + `decision_log.md`에 근거 기록

> 💡 원본 가이드 핵심 재확인: 산출물 3종 = **① 성능 매트릭스 · ② design_rules.md · ③ decision log**. 이 3개가 "혼자 튜닝한 사람"과 "조직에 시스템을 남긴 사람"을 가른다. 그리고 이 3개는 서로를 인용한다 — 매트릭스가 규칙의 근거가 되고, 규칙 변경이 로그에 남고, 로그가 매트릭스의 특정 셀을 가리킨다.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [GitHub Actions — Self-hosted runners 관리](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners) — 등록·보안·라벨 공식 가이드
- [GitHub Actions runner 릴리스](https://github.com/actions/runner/releases) — `config.sh`/`svc.sh` 최신 버전(2026-07 기준 v2.336.0)
- [GitHub Changelog — self-hosted runner 최소 버전 강제 타임라인 (2026-06)](https://github.blog/changelog/2026-06-12-github-actions-minimum-version-enforcement-timeline-for-self-hosted-runners/) — 오래된 runner 거부 정책
- [Self-hosted runner 설정 실전 가이드 (oneuptime, 2026-01)](https://oneuptime.com/blog/post/2026-01-25-github-actions-self-hosted-runners/view) — download→config→svc 흐름
- [NVIDIA Container Toolkit 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — container job GPU 접근
- [TensorRT Python API 문서 (10.16)](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html) — `build_serialized_network`/`execute_async_v3` (2026-07 기준)
- [TensorRT 8→10 Python API 마이그레이션](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/tensorrt-8x-to-10x-python-api.html) — name 기반 I/O API
- [ONNX Runtime — CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) — CUDA 버전 호환·provider 명시
- [ONNX Runtime — QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — QNN 백엔드(context binary)
- [pandas — DataFrame.to_markdown](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html) — 매트릭스 마크다운화(tabulate 필요)
- [pandas — Styler.to_html](https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_html.html) — HTML 하이라이트 매트릭스(jinja2 필요)
- [pytest-regressions](https://pypi.org/project/pytest-regressions/) — 골든 파일 회귀(`dataframe_regression`, `--force-regen`, v3.0+)
- [MLflow Tracking Server (셀프호스팅)](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) — `log_param`/`log_metric`/`log_artifact`/`mlflow server`
- [MLflow vs W&B 2026 비교](https://deploybase.ai/articles/mlflow-vs-wandb) — 실험 추적 도구 선택 근거
- 백엔드 SDK: [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/) · [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) · [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) · [rzv_drp-ai_tvm](https://github.com/renesas-rz/rzv_drp-ai_tvm)

### 논문
- Xiao et al. (2022), *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*, arXiv:2211.10438 — `design_rules.md`의 LayerNorm 규칙 근거
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630 — 혼합정밀도·민감도 개념 배경
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295 — per-channel/캘리브레이션 배경

---

## 9) 다음 단계

인프라가 갖춰졌으니, 이제 이 시스템 위에서 **하나의 모델을 처음부터 끝까지** 관통시키는 [캡스톤 프로젝트](08_capstone.md)로 넘어간다. 앞선 단계: [4단계 — 멀티 SoC](06_multi_soc.md).
