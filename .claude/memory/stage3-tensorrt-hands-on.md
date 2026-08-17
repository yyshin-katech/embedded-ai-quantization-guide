---
name: stage3-tensorrt-hands-on
description: "3단계 TensorRT 완료(2026-08-17, ResNet50, 커밋 70c82e9 푸시완료): trtexec가 pip 휠에 부재→polygraphy API. §2.2.1은 경로 병기(직접 파서 하드 블로커 둘: INT32 bias DQ+zp≠0). implicit 캘리브레이터 10.16서 생존"
metadata:
  node_type: memory
  type: project
---

`study_guide/05_tensorrt.md`(1135줄 초안, 전부 `trtexec` 명령 기반)를 AI-LAP/RTX3080
([[machine-ai-lap-rtx3080]])에서 완주(2026-08-17). venv `~/emb-ai` · **TensorRT 10.16.1.11**(pip 휠
`tensorrt-cu12`) · polygraphy 0.50.3 · onnxruntime 1.23.2. 모델 torchvision **ResNet50**(공개 76.13%),
지연 배치1 / 정확도 ImageNet val 5,000장. **커밋 70c82e9**(main, 푸시 완료 — [[repo-is-public-scan-before-commit]]).

**헤드라인 반전 — trtexec가 없다:** 정본 pip 휠에 `trtexec` **실행파일이 없다**(PATH·파일시스템 0건).
문서의 모든 `trtexec --onnx=… --int8` 명령이 그대로는 실행 불가 → **polygraphy 0.50.3 Python API**
(`network_from_onnx_path`+`engine_from_network`+`CreateConfig`)로 동일 결과. modelopt.onnx는 `[onnx]`
extra(onnxslim) 누락으로 **import 불가**(modelopt.torch만 가용, 2단계 §4.4에서 사용).

**실습1 3점(ResNet50 배치1, eval 5,000):** FP32 1.6615ms(122.3MiB) → FP16 0.8459ms **×1.96**(49.2MiB,
top-1 동일) → INT8 0.7843ms **×2.12**(25.5MiB, 76.36% **−0.52%p**, INT8 커널 74줄).

**§2.2.1 경로 병기 정밀화(반전 아님):** 1단계 ResNet18 ORT-EP는 "TRT 폴백 원인=activation zp≠0
하나뿐"([[stage1-quantization-hands-on]])이었으나, polygraphy/trtexec **직접 파서**엔 하드 블로커가 **둘** —
① **INT32 bias DQ**(case B: 대칭·`act_zp_nonzero_frac=0.0`인데도 parse❌, `DequantizeLayer can only run in
kINT8/kFP8/kFP4/kINT4`) ② **zp≠0**(case C: `shiftIsAllZeros`, zp≠0 0.213). 차이는 **경로** — ORT-EP가
파서 전 QDQ 그래프 수술로 bias DQ를 흡수해 1단계 결론이 그 경로선 유효. 5케이스 절제(A~E)로 파서 vs
**빌더** 축 분리(D: stem conv1 3ch7×7은 parse✅ **build❌** Error10 "no implementation" →
`nodes_to_exclude=["/conv1/Conv"]`). E: DETR `detr_int8.onnx` 실제가 두 블로커 동시(zp≠0 0.831).

**implicit 캘리브레이터 생존:** deprecated `IInt8EntropyCalibrator2`(QDQ 없는 FP32 ONNX + 캘리브 200장)가
TRT 10.16서 **빌드 성공**. deprecation 경고 **134건**(Python 바인딩 레벨 = strong-typing/10.12 표시 126 +
`Superseded by explicit quantization`=10.1 표시 8), **TRT 로그엔 0건**. 이 모델선 explicit보다 빠르고
(0.7074ms **×2.35**) 정확(76.80%) — TRT가 층별 정밀도 자동선택(INT8 57층 vs explicit 강제 74층). 그래도
제어성·제거예정 탓에 신규는 explicit 권장(이 모델서 우연히 유리했을 뿐).

**범위 밖(정직 폴백):** DLA(`num_DLA_cores=0`, RTX 3080 dGPU) · IPluginV3(컴파일 툴체인 헤비) — 2단계
BEVFormer INT8과 같은 하드웨어 범위 밖 처리.

**산출물:** `logs/stage3_tensorrt_report.html`(§1~8·SVG), `experiments/stage3_tensorrt/`
(t01~t04 스크립트/JSON + `t3_common.py` + `parser_constraints.md` 로그원문·설계규칙 + README). 검증:
tech-reviewer 팬인이 **🔴 1건 수정**(§2.2.1 삽입부 L168 "반증"→"정밀화 지점": "반전 아님" 프레이밍과 자기모순,
기술내용 보존) + 자체 독립 재검증(t04 배열 134=126+8·SVG factor 288.9·t03 5케이스) 일치. 남은 🟡:
§2.2.1 내부 case(A~D)↔삽입 case(A~E) 라벨 충돌(같은 "C"가 상반 의미, §2.2.1 수식 있어 저위험·미수정).

**캐비앗:** 절대 지연·top-1은 **CUDA/RTX3080·polygraphy·배치1** 기준 → **상대 관계만 유효**(FP16/INT8 배수,
implicit↔explicit 격차, 파서/빌더 절제 판정). top-1 5,000장 서브셋은 공개값보다 부풀려짐(1단계 함정 0).
**다음:** 4단계(멀티 SoC)~7단계+캡스톤은 아직 웹 검증만. 2단계는 [[stage2-detr-hands-on]] 참고.
