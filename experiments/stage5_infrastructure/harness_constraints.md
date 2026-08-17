# Stage 5 — 벤치 하네스 실측 제약·정정 기록 (`harness_constraints.md`)

> 검증 환경: AI-LAP / RTX 3080, Ubuntu 22.04, 정본 venv `~/emb-ai`
> TensorRT **10.16.1.11** (pip 휠 `tensorrt-cu12`), polygraphy **0.50.3**, pandas/pytest/pytest-regressions **2.11.0**
> 모델: torchvision **ResNet50**(3단계 자산 `_workspace/stage3/resnet50_fp32.onnx`), ImageNet val 5,000장(batch-1), calib 200장
> 일자: 2026-08-17. 이 문서는 `study_guide/07_infrastructure.md`(5단계) 초안을 **실제로 실행해** 드러난 정정의 로그 원문·설계규칙 원천이다.

---

## 0. 한 줄 요약

문서의 하네스 골격(ABC 인터페이스 → TRT 백엔드 → config 순회 → pandas 매트릭스 → pytest 회귀 게이트 → self-hosted GPU CI)은 **구조적으로 건전**하고 대부분 그대로 돌아간다. 단, **실제로 관통시켜 보면** 정본 pip 스택·TRT 10.x·polygraphy·데이터층에서 **8건**이 어긋난다. 그중 2건(정정 5·6)은 **exit 0·에러 0으로 통과하면서 결과만 조용히 틀리는 무음 오답**이라 코드 리뷰로는 안 잡히고 실행해야만 드러난다.

| # | 위치(문서) | 정정 | 심각도 |
|---|-----------|------|--------|
| 1 | §4-3 L329–330,382–399 | `pycuda` 부재 → polygraphy `TrtRunner`로 대체 | 🔴 ImportError(실행 불가) |
| 2 | §4-3 L367–369 | INT8 캘리브레이터가 주석("지면상 생략") → 실제 `IInt8EntropyCalibrator2` 배선 | 🔴 스케일 없는 무의미 엔진 |
| 3 | §4-3 L416 | `engine.device_memory_size` deprecated → `device_memory_size_v2` | 🟡 deprecated(값 의미도 정정) |
| 4 | §4-5 L591 | `from data import Loader, Evaluator`인데 `data.py` 미제공 → 제공 | 🔴 ImportError(실행 불가) |
| 5 | §4-3 L412 | polygraphy zero-copy 러너 → eval에 `.copy()` 없으면 **acc 0.0014=우연** | 🔴 **무음 오답**(exit 0) |
| 6 | §4-6 L699–702 | `pivot_table` dropna 기본=True가 §5-1 "회색행 보존" 원칙을 자기위반 → `dropna=False` | 🟡 **무음 누락**(정직성 훼손) |
| 7 | §4-3 L349–351 | `EXPLICIT_BATCH` 명시 = 10.16서 DeprecationWarning(무인자가 이미 explicit) | 🟢 정밀화 |
| 8 | §3 L125·§4-7 L813·840·§6 L1168·§8 L1214 | pytest-regressions "v3.0+"는 **사실오류**(최신 **2.11.0**, v3.x 부재) | 🟢 사실정정 |

정정 1·4는 "정본 pip 휠 스택엔 NVIDIA/CUDA C++ 계열 도구가 빠져 있다"는 **3단계와 같은 결**(3단계: `trtexec` 바이너리 부재)이다. 하네스도 같은 벽을 만나고, 같은 해법(polygraphy Python API)으로 넘는다.

---

## 1. 정정 1 — `pycuda` 부재 → polygraphy `TrtRunner`

문서 §4-3의 `backends/trt.py`는 raw TensorRT Python API(`trt.Builder`/`build_serialized_network`/`execute_async_v3`/`set_tensor_address`) + **pycuda**(`cuda.mem_alloc`/`memcpy_htod`/`Stream`)로 디바이스 메모리·H2D/D2H·동기화를 직접 처리한다. 정본 venv엔 pycuda가 없다:

```
$ python -c "import pycuda"
ModuleNotFoundError: No module named 'pycuda'
```

그래서 문서의 `run()`(L382–399)·`_once`(L405–407)가 첫 줄 import에서 죽는다. polygraphy(설치돼 있음)의 `TrtRunner.infer`가 그 블록 전체를 대체한다 — 내부에서 디바이스 버퍼 할당·H2D·`execute_async_v3`·stream sync·D2H를 끝내고 반환한다. 3단계 지연측정과 **동일 경로**가 되어 비교가 공정하다. (pycuda를 직접 설치하면 원안도 동작하지만, 정본 스택엔 없으므로 하네스는 polygraphy를 표준 경로로 삼는다.)

- 코드: `bench/backends/trt.py` `run()` = `self._runner.infer({in_name: ascontiguousarray(x)})` → `list(out.values())[0]`
- `base.py._timeit`의 동기화 주석도 정정: 원안은 `fn` 안에서 `cuda.Context.synchronize()`가 필요했으나, `TrtRunner.infer`가 내부 sync를 끝내므로 `fn=self.run`만으로 충분(`base.py` L91–94).

## 2. 정정 2 — INT8 캘리브레이터 배선(주석 → 실제)

문서 §4-3 L365–369:

```python
elif precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    # 실제로는 여기서 IInt8Calibrator를 붙인다(calib_path의 npy 사용).
    # 3단계에서 만든 캘리브레이터를 재사용. 지면상 생략.
    # config.int8_calibrator = MyCalibrator(calib_path)
```

이대로면 INT8 플래그만 켜지고 **스케일(캘리브 통계)이 없어** 엔진이 무의미하다. 3단계 t04와 동일하게 polygraphy `Calibrator`로 `IInt8EntropyCalibrator2`를 실제 배선했다(`trt.py` L52–58):

```python
Calibrator(data_loader=self._calib_feed(), cache=self._calib_cache,
           BaseClass=trt.IInt8EntropyCalibrator2)   # 3단계 실습3의 그 API — 10.16서 생존
```

실측 로그(최초 빌드, 200장 캘리브):

```
[I] Building engine with configuration:
    Flags | [FP16, INT8]
    Calibrator | Calibrator(<generator ...>, cache='.trt_cache/resnet50__trt__int8.calib',
                 BaseClass=<class 'tensorrt_bindings.tensorrt.IInt8EntropyCalibrator2'>)
[I] Saving calibration cache to .trt_cache/resnet50__trt__int8.calib
[I] Finished engine building in 65.075 seconds
```

- 결과: INT8 top-1 **0.768** = 3단계 t04(implicit 캘리브레이터) top-1 0.768과 **정확히 일치**. 캘리브 캐시 재사용 시 빌드 65s→~50s.
- `deprecated`인 `IInt8EntropyCalibrator2`가 TRT 10.16에서 **빌드 성공**(3단계 t04에서 확인한 것과 동일). 신규 코드엔 explicit QDQ 권장이지만, 하네스의 "FP32 ONNX 하나로 3 precision 관통" 편의엔 implicit이 맞다.

## 3. 정정 3 — `device_memory_size` deprecated → `_v2`(값 의미도 정정)

문서 §4-3 L416 `peak_mem = self.engine.device_memory_size / (1024**2)`. TRT 10.x에서 `device_memory_size`는 deprecated. `device_memory_size_v2`를 우선 사용하고 어느 쪽이 먹혔는지 notes에 봉인한다(`trt.py._peak_mem_mb`):

```python
for attr in ("device_memory_size_v2", "device_memory_size"):
    v = getattr(self.engine, attr, None)
    if v: self._mem_attr = attr; return v/(1024**2)
```

실측: 전 셀 `mem_api=device_memory_size_v2`(v2가 먹힘). **값의 의미 주의** — 이 수치는 실행컨텍스트 스크래치(activation) 디바이스 메모리이지 **가중치/엔진파일 크기가 아니다**. 실측 8.4→3.9→**1.7 MB**(fp32→fp16→int8)로 단조 감소(저정밀일수록 activation 스크래치 축소). 3단계가 따로 본 **엔진 파일 크기**(122→49→25 MiB)와는 다른 축의 지표다. 리포트/문서에서 둘을 혼동하지 않게 병기했다.

## 4. 정정 4 — 데이터층 `data.py` 부재 → 제공

문서 §4-5 `run_bench.py` L591 `from data import Loader, Evaluator`인데 `data.py`가 저장소에 없다(데이터층은 프로젝트마다 다르므로 문서가 **의도적으로 비운 자리**). 검증 인스턴스로 ResNet50/ImageNet-val 분류를 채웠다(`bench/data.py`):

- `Loader`: `~/stage1-work/data/cache/tv.npy`(50k NHWC uint8, torchvision 공식 전처리 캐시) + `labels.npy`(=`val_synset_map.txt` col2, 두 규약 100% 일치 확인). `one_batch()`/`eval_set()`(1장씩 (1,3,224,224))/`gts()`/`calib_feed(n)`.
- `Evaluator.compute_acc` = top-1, `compute_map`=별칭(문서 원안이 검출 mAP 계약이라 이름만 맞춤).
- **검증**: FP32 top-1 **0.7688** = 공개값·3단계와 1:1 일치 → 데이터층·전처리·라벨정렬 전부 올바름의 증거.

> **모델 스코프 결정**: 문서 예시는 BEVFormer/mAP지만, BEVFormer INT8은 2단계에서 "유효 export 경로 없음(포크 필요)"으로 **범위 밖**. 그래서 RTX에서 실제 빌드·측정 가능한 ResNet50(분류/top-1)으로 하네스를 관통시켰다. `BenchResult.accuracy`는 0~1 실수(mAP·top-1 공용)라 스키마는 불변.

## 5. 정정 5 — polygraphy zero-copy 러너의 **무음 오답**(핵심 발견)

문서 §4-3 measure()의 정확도 루프:

```python
preds = [self.run(x) for x in loader.eval_set()]   # ← 원안 그대로
acc = evaluator.compute_map(preds, loader.gts())
```

**문서의 pycuda 원안에선 안전하다** — `run()`이 매 호출 `out = np.empty(out_shape)`를 새로 할당하고 `memcpy_dtoh(out, d_out)`로 복사해 반환하므로, 5,000개가 **서로 다른 배열**이다. 그런데 정정 1로 `run()`을 polygraphy `TrtRunner.infer`로 바꾸면 — **`TrtRunner`는 호스트 출력버퍼를 재사용(zero-copy)**한다. `run()`의 반환은 그 단일 버퍼의 view라, 리스트에 모으면 **5,000개 전부가 마지막 추론 결과를 가리킨다**. argmax가 전부 동일 → 대부분 오답.

실측(64장 재현):

```
refs 전부 동일?(에일리어싱 증거) True  argA[:8] [46,46,46,46,46,46,46,46]
copy 다양?                       61 distinct  argB[:8] [65,795,230,809,520,58,334,852]
gts[:8]                          [65,970,230,809,516,57,334,415]
acc(refs)=0.0156   acc(copy)=0.7188
```

전량(5,000장)에선 acc = **0.0014 ≈ 1/1000**(우연 수준). exit 0, 예외 0, 로그 정상 — **오직 숫자만 조용히 틀린다.** 정정:

```python
preds = [self.run(x).copy() for x in loader.eval_set()]   # 추론 직후 스냅샷
```

→ FP32 acc 0.0014 → **0.7688**(공개값 일치). 이 함정은 **pycuda→polygraphy 치환의 부작용**이라 3단계엔 없던 새 항목이다. 교훈: "추상 measure() 계약(preds를 모은 뒤 평가)"이 zero-copy 러너와 만나면 무음 오답이 된다. **버퍼 소유권**이 바뀌는 치환에선 반드시 즉시 복사하거나 루프 안에서 소비하라(3단계 t3_common의 `evaluate_top1`은 루프 안에서 즉시 argmax해 이 함정을 원천 회피).

## 6. 정정 6 — `pivot_table` dropna 기본값이 §5-1 원칙을 자기위반

문서 §4-6 `generate.py` L699–702는 `df.pivot_table(index=[...], values="latency_ms")`. pandas `pivot_table`은 `dropna=True`(기본)이라 **값이 전부 NaN인 행을 조용히 버린다.** stub 백엔드(tidl/qnn/drpai)는 `NotImplementedError`→NaN BenchResult로 봉인되는데, 이 NaN 행이 pivot에서 통째로 사라진다:

```
dropna=True (문서 원안):          dropna=False (정정):
precision  fp16   fp32   int8     precision       fp16   fp32   int8
resnet50 rtx 1.02  1.84  0.86     resnet50 qcs8550 NaN   NaN   NaN
                                          rtx     1.02  1.84  0.86
(← tda4vm/qcs8550/rzv2h 소멸)              rzv2h   NaN   NaN   NaN
                                          tda4vm  NaN   NaN   NaN
```

그런데 문서 **§5-1**(L1051)은 정반대를 명령한다: *"회색(보드필요) 행도 매트릭스에 남긴다 → '아직 측정 못 함'과 '측정했더니 나쁨'은 다르다. 빈칸이 아니라 명시적 '보드필요'가 정직한 보고다."* 즉 **§4-6 코드가 §5-1 원칙을 스스로 위반**한다 — 그리고 이는 실행해 matrix.md를 눈으로 봐야만 드러난다(코드만 읽으면 안 보임). 정정: 모든 pivot에 `dropna=False`, HTML은 `na_rep="보드필요"`로 회색 셀에 글자를 박는다. **CSV(long-form)는 원래도 6행 전부 보존**하므로 회귀 baseline엔 영향 없다(그래서 회귀 테스트는 정상 통과했고, 사람이 읽는 표에서만 정직성이 훼손됐다).

## 7. 정정 7 — `EXPLICIT_BATCH` 명시 = 10.16서 deprecated(정밀화)

문서 §4-3 L349–351 주석은 "TensorRT 10.x: EXPLICIT_BATCH는 기본이지만 명시해도 무방". 10.16 실측:

```
>>> f = trt.NetworkDefinitionCreationFlag
>>> hasattr(f,"EXPLICIT_BATCH"), int(f.EXPLICIT_BATCH)
(True, 0)
>>> b.create_network(1 << int(f.EXPLICIT_BATCH))
DeprecationWarning: Use Implicit batch dimensions support has been removed instead.
>>> b.create_network().has_implicit_batch_dimension
False   # 무인자가 이미 explicit
```

즉 "무방"이 아니라 **명시하면 DeprecationWarning**이고, 무인자 `create_network()`가 이미 explicit-batch다. 헤드라인 반전이 아닌 정밀화. polygraphy 경로(`network_from_onnx_path`)는 네트워크 생성을 내부에서 처리하므로 이 지점을 아예 우회한다.

## 8. 정정 8 — pytest-regressions "v3.0+"는 사실오류

문서 5곳(§3 L125, §4-7 L813·L840, §6 L1168, §8 L1214)이 pytest-regressions **"v3.0+ 권장"**이라 적었다. PyPI 실측:

```
$ pip install "pytest-regressions==99.99"
ERROR: Could not find a version ... (from versions: 0.1.0, ... , 2.11.0)
$ pip show pytest-regressions | grep Version
Version: 2.11.0
```

**v3.x는 존재하지 않는다**(0.1.0 ~ **2.11.0**이 최신). 다만 코드가 쓰는 `dataframe_regression` fixture는 2.11.0에 정상 존재하므로 **코드는 그대로 동작**한다 — "v3.0+"라는 버전 표기만 틀렸다. 문서를 "2.11.0(2026-08 기준 최신)"으로 정정.

---

## 9. 실측 매트릭스 (검증 인스턴스, 정본 SSOT)

`bench/results/*.json` → `bench/report/matrix.csv`. ResNet50 / RTX 3080 / TRT 10.16.1.11 / polygraphy 0.50.3 / ImageNet val 5,000장(batch-1) / calib 200장.

| soc | precision | latency_ms (median, wall) | p95 | scratch mem_v2 (MB) | top-1 | build (s) | vs fp32 |
|-----|-----------|--------------------------:|-----:|--------------------:|------:|----------:|--------:|
| rtx | fp32 | **1.837** | 1.8832 | 8.4 | **0.7688** | 12.8 | ×1.00 |
| rtx | fp16 | **1.0231** | 1.0603 | 3.9 | 0.7686 | 29.8 | **×1.80** |
| rtx | int8 | **0.8628** | 0.8927 | 1.7 | **0.768** | 49.8 | **×2.13** |
| tda4vm | int8 | 보드필요(NaN) | — | — | — | — | TIDL 미구현(stub) |
| qcs8550 | int8 | 보드필요(NaN) | — | — | — | — | QNN 미구현(stub) |
| rzv2h | int8 | 보드필요(NaN) | — | — | — | — | DRP-AI 미구현(stub) |

**3단계 교차검증**: INT8 top-1 0.768 = 3단계 t04(implicit) 0.768 **일치**; FP32 top-1 0.7688 = 3단계·공개값 일치. 지연 **절대값**은 3단계(event-timed `last_inference_time`, fp32 1.6615ms)보다 높다 — 하네스 `_timeit`은 **wall-clock**(`time.perf_counter`)이라 H2D/D2H·Python 오버헤드를 포함한다. 그래서 **비율도 압축**(×1.80/×2.13 vs 3단계 ×1.96/×2.12): 정밀도와 무관한 고정 오버헤드가 분모에 상수로 얹혀 배율을 줄인다. → **절대·비율 모두 방법론(측정계) 의존, 상대 관계만 유효**(1단계 캐비앗과 동종).

## 10. 회귀 게이트 실증 (통과 + 의도적 실패)

`bench/tests/test_regression.py` — (A) 임계값(map 1%p·latency ×1.10) + (B) pytest-regressions 골든 이중화.

- **baseline 생성**: `cp report/matrix.csv tests/baseline_matrix.csv`(문서 절차).
- **정상**: `pytest -q` → 최초 실행은 골든 최초 생성으로 `test_matrix_matches_golden`만 "created" fail(설계된 동작), 재실행 시 **3 passed**.
- **의도적 회귀 주입**(int8 acc 0.768→0.740=−2.8%p, latency 0.8628→1.10=×1.27): **3 tests 모두 FAIL** —
  - `test_no_map_regression`: −2.8%p > 1%p → 잡음 ✓
  - `test_no_latency_regression`: ×1.27 > 1.10 → 잡음 ✓
  - `test_matrix_matches_golden`: diff(latency 0.2372, acc 0.028) → 잡음 ✓
- **복원 후 3 passed.** 두 층(임계값+골든)이 실제 회귀를 잡음을 실증.

## 11. 범위 밖(정직한 폴백)

- **SoC 백엔드 3종(tidl/qnn/drpai)**: 실물 보드·SDK 없음 → `NotImplementedError` stub. 하네스는 죽지 않고 회색 NaN 셀로 봉인(설계 의도대로 검증됨). 실제 구현은 4단계(멀티 SoC) 과제.
- **BEVFormer INT8**: 2단계 결론(유효 export 경로 없음, 포크 필요)으로 범위 밖. 하네스 스키마(BenchResult)는 모델 무관이라, export만 되면 config에 한 블록 추가로 편입 가능.
- **MLflow(§4-10)**: 선택 사항, 이번 검증에선 미도입(로컬 파일 매트릭스로 충분).

## 12. 캐비앗(불변)

절대 지연·top-1·scratch mem은 CUDA EP·polygraphy wall-clock·batch-1·implicit 캘리브·5,000장 서브셋 기준이다. 5,000장 서브셋 top-1은 공개 50k 대비 부풀 수 있다(1단계 함정 0). **상대 관계(배율·회귀 델타·회색행 유무)만 이식 가능**하며, 절대값의 문헌 비교는 하지 않는다.
