# experiments/stage3_tensorrt — 3단계 TensorRT 실측 검증

3단계 문서(`study_guide/05_tensorrt.md`)를 실제 RTX 3080에서 완주한 재현 스크립트·로그·JSON.
문서가 **전부 `trtexec` 명령으로** 쓰여 있으나 정본 pip 스택에 trtexec가 없다는 것부터,
§2.2.1(1단계 ResNet18 ORT-EP 관측)이 직접 파서에서 어떻게 갈리는지까지를 실측으로 종결한다.

- **머신/환경**: RTX 3080 · Ubuntu 22.04 · venv `~/emb-ai` · **TensorRT 10.16.1.11**(pip 휠) · polygraphy 0.50.3 · onnxruntime 1.23.2 (`t01_env.json`)
- **모델**: torchvision **ResNet50**(IMAGENET1K_V1, 공개 top-1 76.13%) — 1단계 ResNet 계열 연속 + export 리스크 최소. 지연은 배치1(정본 지표), 정확도는 ImageNet val 5,000장 서브셋
- **리포트**: [`logs/stage3_tensorrt_report.html`](../../logs/stage3_tensorrt_report.html)
- **파서/빌더 제약 로그 원문 + 설계 규칙**: [`parser_constraints.md`](parser_constraints.md)

## 핵심 결과 (실측)

### 1) 헤드라인 정정 — trtexec 부재 → polygraphy Python API (`t01`)

`trtexec`는 정본 pip 휠(`tensorrt-cu12`)에 **실행파일이 없다**(PATH null, 파일시스템 0건). 문서의 모든
`trtexec --onnx=... --int8` 명령은 그대로는 실행 불가 → **polygraphy 0.50.3**(`network_from_onnx_path` +
`engine_from_network` + `CreateConfig`)로 동일 결과를 낸다. modelopt.onnx는 `[onnx]` 선택 의존성(onnxslim)
누락으로 import 불가(`importable=false`) — modelopt.torch만 가용(2단계 §4.4에서 사용).

### 2) 실습1 FP32/FP16/INT8 3점 (ResNet50, 배치1, eval 5,000장, `t02`)

| 구성 | p50 (ms) | vs FP32 | top-1 | 엔진 크기 | INT8 커널줄 |
|---|---|---|---|---|---|
| FP32 | 1.6615 | ×1.00 | 76.88% | 122.3 MiB | 0 |
| FP16 | 0.8459 | **×1.96** | 76.88%(동일) | 49.2 MiB | 0 |
| INT8(+FP16 폴백) | 0.7843 | **×2.12** | 76.36% (−0.52%p) | 25.5 MiB | **74** |

> 공개 FP32 76.13% 대비 서브셋 76.88%는 5,000장 서브셋 부풀림(1단계 함정 0). 상대 델타만 유효.

### 3) §2.2.1 정밀화 — 직접 파서엔 **두 개**의 독립 하드 블로커 (`t03`, 5케이스 절제)

| 케이스 | 구성 | parse | build | 블로커 |
|---|---|:---:|:---:|---|
| A | 대칭 QInt8·bias off·stem 제외 | ✅ | ✅ | — (정본 처방) |
| B | 대칭 QInt8·**bias INT32**·stem 제외 | ❌ | — | INT32 bias DQ (대칭인데도) |
| C | **비대칭 QUInt8**·stem 제외 | ❌ | — | `shiftIsAllZeros`(zp≠0) |
| D | 대칭 QInt8·bias off·**stem 포함** | ✅ | ❌ | stem INT8 커널 부재(빌더) |
| E | DETR `detr_int8.onnx` 실제 | ❌ | — | zp≠0 + INT32 bias 동시 |

> §2.2.1(ORT-EP)은 "zp≠0 하나뿐"으로 봤지만, 직접 파서는 **INT32 bias DQ도 독립 블로커**(B: 대칭 zp=0인데도 죽음).
> 차이는 경로 — ORT-EP가 파서 전 bias DQ를 흡수. 반전 아닌 **경로 병기 정밀화**([`parser_constraints.md`](parser_constraints.md) §5).

### 4) 실습3 implicit 캘리브레이터 — TRT 10.16에서 여전히 살아있음 (`t04`)

`IInt8EntropyCalibrator2`(implicit, QDQ 없이 FP32 ONNX + 캘리브 200장)로 INT8 엔진 **빌드 성공**(63.8s,
캐시 5,776B). deprecation 경고는 **Python 레벨 134건**(strong-typing/10.12 표시 126 + "Superseded by explicit quantization"=10.1 표시 **8건** → 문서 §2.2 "10.1 deprecated" 확증), TRT 로그엔 0건.

| implicit vs explicit(t02 INT8) | p50 (ms) | top-1 | INT8 커널줄 |
|---|---|---|---|
| explicit QDQ | 0.7843 | 76.36% | 74 |
| implicit calib | **0.7074** | **76.80%** | 57 |

> implicit이 더 빠르고 정확한 건 TRT가 **지연 최소화로 층별 정밀도를 자동 선택**(INT8을 57층만)했기 때문 — explicit QDQ는 감싼 층을 전부 INT8로 강제. 그래도 implicit은 deprecated(제어성↓·제거 예정)라 신규는 explicit 권장. 이 모델에서 우연히 유리했을 뿐.

## 범위 밖(정직한 폴백)

- **실습5 (DLA)**: `num_DLA_cores=0`(RTX 3080은 dGPU) → DLA 오프로드 하드웨어 부재. Jetson Orin 전용. 2단계 BEVFormer INT8과 같은 하드웨어 범위 밖 처리.
- **실습6 (IPluginV3)**: 커스텀 플러그인 컴파일 툴체인 필요 — 헤비, 미실행.

## 스크립트

| 파일 | 역할 | 산출 |
|---|---|---|
| `t3_common.py` | 공통 모듈(경로·tv 캐시·전처리·엔진빌드·지연벤치·top1) | — |
| `t01_env.py` | 환경 실측(trtexec 부재·TRT introspection·DLA·polygraphy·modelopt) | `t01_env.json` |
| `t02_latency_3point.py` | 실습1 FP32/FP16/INT8 3점(polygraphy 빌드) | `t02.json` |
| `t03_parser_constraints.py` | §2.2.1 정밀화 5케이스 절제(파서 vs 빌더) | `t03.json` |
| `t04_implicit_calibrator.py` | 실습3 implicit `IInt8EntropyCalibrator2` 생존 확인 | `t04.json` |

```bash
# 재현 (venv ~/emb-ai, TensorRT 10.16 + polygraphy 필요)
cd <repo>
~/emb-ai/bin/python experiments/stage3_tensorrt/t01_env.py
~/emb-ai/bin/python experiments/stage3_tensorrt/t02_latency_3point.py
~/emb-ai/bin/python experiments/stage3_tensorrt/t03_parser_constraints.py   # 선행: t02(INT8 onnx) + 2단계 detr_int8.onnx
~/emb-ai/bin/python experiments/stage3_tensorrt/t04_implicit_calibrator.py
```

> `t3_common.py`가 `sys.path`에 자기 디렉토리를 넣으므로 `PYTHONPATH` 불필요. 산출 엔진/onnx는 `_workspace/stage3/`(gitignore). 재실행 시 존재 산출물은 건너뜀.

## 캐비앗

- 절대 지연·top-1은 **CUDA/RTX 3080 · polygraphy 빌드 · 배치1** 기준. 다른 GPU·배치·TRT 버전과 1:1 비교 불가 — 유효 결론은 **같은 경로 안의 상대 관계**(FP16/INT8 배수, implicit↔explicit 격차, 파서/빌더 절제 판정).
- top-1 서브셋(5,000장)은 공개값보다 부풀려짐(1단계 함정 0). 여기 논점은 정확도 순위(FP32≥implicit≥explicit INT8)와 배수이지 절대 정확도가 아니다.
- §2.2.1 정밀화는 **경로 병기**(ORT-EP vs 직접 파서) — 1단계 ResNet18 결과를 반전하지 않는다. 상세 근거는 `parser_constraints.md` §5.
