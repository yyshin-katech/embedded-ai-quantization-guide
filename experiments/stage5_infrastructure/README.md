# experiments/stage5_infrastructure — 5단계 벤치 하네스 실기 검증

`study_guide/07_infrastructure.md`(5단계: 벤치마크 인프라·CI·회귀 게이트) **초안을 실제 RTX 3080에서 관통시켜** 정정한 산출물.
전체 정정·로그·설계규칙은 **[`harness_constraints.md`](harness_constraints.md)**, 사람이 읽는 리포트는 **[`logs/stage5_infrastructure_report.html`](../../logs/stage5_infrastructure_report.html)**.

## 무엇을 검증했나

문서의 하네스 골격(ABC 백엔드 인터페이스 → TRT 백엔드 → config 순회 러너 → pandas 성능 매트릭스 → pytest 회귀 게이트)을 **실행 가능한 형태로 완성**하고, 실측으로 **8건**을 정정했다. 검증 인스턴스는 **ResNet50 / ImageNet val**(3단계 자산 재사용) — 문서 예시의 BEVFormer는 2단계에서 INT8 유효 export 경로가 없어(포크 필요) 범위 밖이라, RTX에서 실제 빌드·측정 가능한 분류 모델로 관통시켰다. `BenchResult` 스키마는 mAP·top-1 공용이라 불변.

### 정정 요약(8건)

| # | 정정 | 종류 |
|---|------|------|
| 1 | `pycuda` 부재 → polygraphy `TrtRunner` (3단계 trtexec 부재와 같은 결) | 🔴 실행 불가 |
| 2 | INT8 캘리브레이터 주석("지면상 생략") → 실제 `IInt8EntropyCalibrator2` 배선 | 🔴 무의미 엔진 |
| 3 | `device_memory_size` deprecated → `_v2`(값=실행 scratch, 엔진파일 아님) | 🟡 deprecated |
| 4 | `from data import Loader, Evaluator`인데 `data.py` 미제공 → 제공 | 🔴 실행 불가 |
| 5 | polygraphy zero-copy → eval `.copy()` 없으면 **acc 0.0014=우연**(무음 오답) | 🔴 무음 오답 |
| 6 | `pivot_table` dropna 기본=True가 §5-1 "회색행 보존" 원칙 자기위반 → `dropna=False` | 🟡 무음 누락 |
| 7 | `EXPLICIT_BATCH` 명시 = 10.16서 DeprecationWarning(무인자가 이미 explicit) | 🟢 정밀화 |
| 8 | pytest-regressions "v3.0+"는 사실오류(최신 **2.11.0**, v3.x 부재) | 🟢 사실정정 |

## 실측 매트릭스 (정본)

ResNet50 / RTX 3080 / TensorRT 10.16.1.11 / polygraphy 0.50.3 / ImageNet val 5,000장(batch-1) / calib 200장.

| soc | precision | latency (ms, median) | top-1 | scratch mem_v2 (MB) | vs fp32 |
|-----|-----------|---------------------:|------:|--------------------:|--------:|
| rtx | fp32 | 1.837 | 0.7688 | 8.4 | ×1.00 |
| rtx | fp16 | 1.0231 | 0.7686 | 3.9 | ×1.80 |
| rtx | int8 | 0.8628 | 0.768 | 1.7 | ×2.13 |
| tda4vm/qcs8550/rzv2h | int8 | 보드필요(NaN) | — | — | stub(회색) |

INT8 top-1 0.768 = 3단계 t04(implicit) 일치. FP32 0.7688 = 공개값 일치. 절대 지연은 wall-clock(`_timeit`)이라 3단계 event-timed보다 높고 배율도 압축(방법론 의존) — **상대 관계만 유효**.

## 디렉토리

```
bench/
  backends/
    base.py        # ABC 인터페이스 + BenchResult + collect + _timeit (순수 파이썬, 정정 0)
    trt.py         # TRT 백엔드 — 정정 1·2·3·5 반영(polygraphy/캘리브/mem_v2/.copy())
    tidl.py qnn.py drpai.py   # SoC stub(NotImplementedError → 회색 NaN 셀)
  data.py          # 정정 4 — ResNet50/ImageNet 데이터층(문서가 비운 자리)
  config.yaml      # 모델×백엔드×precision 매트릭스 + exclude
  run_bench.py     # itertools.product 순회 러너(+ 정정 4 calib 주입 배선)
  report/
    generate.py    # 정정 6 — pivot dropna=False(회색행 보존)
    matrix.csv/md/html   # 자동 생성 매트릭스(6셀: 3실측 + 3회색)
  tests/
    test_regression.py       # (A) 임계값 + (B) pytest-regressions 골든 이중화
    baseline_matrix.csv      # 회귀 baseline(정본 매트릭스 스냅샷)
    test_regression/test_matrix_matches_golden.csv   # pytest-regressions 골든
  results/*.json   # 6개 셀 원자료(회귀·리포트 소스)
```

## 재현

```bash
source ~/emb-ai/bin/activate
cd experiments/stage5_infrastructure/bench
nvidia-smi -L                       # 필수 라이브니스
python run_bench.py                 # results/*.json 6개 (fp32~12s·fp16~30s·int8~50-65s 빌드 + 각 5k eval)
python report/generate.py           # report/matrix.{csv,md,html}
cp report/matrix.csv tests/baseline_matrix.csv   # 최초 baseline(1회)
python -m pytest tests/ -q          # 최초엔 골든 생성으로 1건 created-fail, 재실행 시 3 passed
```

- 전제 데이터: `~/stage1-work/data/cache/{tv.npy,labels.npy,calib200_idx.npy}`(3·1단계 자산, 저장소 밖).
- ONNX: `_workspace/stage3/resnet50_fp32.onnx`(3단계 export).
- **주의**: 전체 순회를 태스크 러너의 wall-clock 한계 안에서 돌리려면 `nohup`으로 분리 실행 권장(fp32+fp16+int8 빌드 누적 ~2분).

## 캐비앗

절대 지연·top-1·mem은 CUDA·polygraphy wall-clock·batch-1·implicit 캘리브·5k 서브셋 기준. 상대 관계(배율·회귀 델타·회색행 유무)만 이식 가능. BEVFormer·SoC 실물·MLflow는 범위 밖(각각 2단계 결론·4단계 과제·선택).
