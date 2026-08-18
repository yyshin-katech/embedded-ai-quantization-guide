---
name: stage4-qualcomm-aihub-hands-on
description: "4단계 Qualcomm 벤더-NPU 실측(2026-08-18, AI Hub 클라우드·보드 없이, ResNet50): HTP 두 종(QCS8550·SA8775P ADP) 100% NPU offload·INT8 fp16 대비 ×1.77/×2.03. 외부 ORT-QDQ 지참=on-device 무음붕괴 0.005 vs AI Hub 자체 quantize 0.735 회복(748µs로 더 빠름). Qualcomm 축만(TI/Renesas 보드 대기). tech-reviewer 팬인 PASS(🔴0)"
metadata:
  node_type: memory
  type: project
---

4단계(멀티 SoC) 벤더 NPU를 **보드 없이** 실측하는 경로 = **Qualcomm AI Hub** 클라우드 실기기(qai_hub 0.54.0,
`workbench.aihub.qualcomm.com`). CPU 폴백 프록시([[stage4-arm-cpu-fallback-proxy]], offload 0% 바닥값) 위에 **벤더
NPU가 실제로 얼마나 버는가 + 얼마나 offload되는가**를 채움. 3·5단계 자산 **ResNet50**([[stage3-tensorrt-hands-on]]·
[[stage5-infrastructure-hands-on]]) FP32/INT8 QDQ ONNX를 `--target_runtime qnn_context_binary`(ONNX→Hexagon HTP
context binary)로 컴파일. QAIRT SDK 2.45.0/HTP v73. 격리 venv `~/qaihub-venv`. 인증 토큰은 `~/.qai_hub/client.ini`
(repo 밖, 어떤 스크립트에도 하드코딩 안 함 — [[repo-is-public-scan-before-commit]]). 산출물:
`logs/stage4_qualcomm_aihub_report.html` · `experiments/stage4_qualcomm_aihub/`(scripts 9·results 7·raw 5·
`aihub_constraints.md`·README) · `study_guide/06_multi_soc.md` §4-B 🔬 콜아웃.

**헤드라인 — 지연·offload(신뢰, SSOT=results/*.json + raw 사이클 합산):** compile_job→profile_job은 그래프 구조
충실 → 정확도 함정과 무관하게 유효.
- **QCS8550 (Proxy)**: FP32(→HTP fp16) 1864µs/4,677,822cyc → INT8 QDQ 1052µs/3,754,903cyc = **×1.77**, 둘 다
  **100% NPU offload**(125/125·128/128).
- **SA8775P ADP**(자동차 Snapdragon Ride 물리 보드): FP32 3056µs/6,192,577cyc → INT8 1505µs/4,462,570cyc =
  **×2.03**, 100% offload. 프록시보다 ~1.6× 느리지만 INT8/FP32 관계 유지.
- **두 디바이스 100% NPU offload** — 깨끗한 CNN이라 폴백 0 → §06 목표("Offloaded≈Total, subgraph 최소")를 벤더
  실기기에서 정량 달성. INT8 배속은 execution_cycles로 교차확증(FP32 사이클 > INT8 사이클).

**🔴 무음 오답 — 외부 QDQ = on-device 정확도 붕괴(silent-wrong):** compile/profile 통과(100% offload)해도 수치가
맞는단 뜻 아님. QCS8550 200장 on-device 대조(scratchpad `acc/*.npy` → aihubq_summary.json):
- FP32(→HTP fp16): top-1 **0.745**·ORT 일치 0.96·distinct 183 → **충실**(입력 NCHW·전처리·업로드·파싱·하네스 정상 증명).
- INT8 · 외부 ORT-QDQ: top-1 **0.005**·일치 0.005·distinct 35(409/862/818/506/723 집중) → **붕괴**. exit 0·정상
  shape라 조용. 20장 대조군도 FP32 0.75/INT8 0.0(스케일 무관 구조적 붕괴).
- **범인:** ORT 양자화기 QDQ scale을 HTP 임포트가 존중 안 함. **동일 ONNX가 4단계 CPU 프록시 x86 CPUEP에선
  0.753**(`experiments/stage5_infrastructure/cpu_proxy/results/resnet50__x86_cpu__int8.json`, 1,000장)인데 HTP에선
  0.005 → 자산 아니라 임포트가 범인.
- **올바른 경로 = AI Hub 자체 `submit_quantize_job`**(HTP-native QDQ 생성): top-1 **0.735**·일치 0.94·distinct 184 →
  FP32 0.745 근접 회복. 게다가 **748µs·1,985,339cyc·127층**으로 외부-QDQ INT8(1052µs·3,754,903cyc·128층)보다
  **빠르고 leaner**(native 양자화가 더 최적 그래프). 잡: quantize jp1jz48kp→compile jgnnv69vg→profile jpyx314r5→
  inference jglxmv0eg. 시그니처: `submit_quantize_job(model, calibration_data={"input":[xs...]}, weights_dtype=
  QuantizeDtype.INT8, activations_dtype=QuantizeDtype.INT8)` → QuantizeJob → `get_target_model()` = HTP-native QDQ ONNX.

**툴체인 발견 3종:**
- **#1 AI Hub 프론트엔드는 ORT/TRT보다 엄격:** ORT가 shape-inference로 넣은 `logits`가 value_info+graph-IO 양쪽
  (ONNX 스펙 위반)이면 컴파일 거부(`Tensors {'logits'} occur in value_info but also in model IO`). ORT/TRT는 통과.
  → `clean_valueinfo_for_aihub.py`로 IO 충돌 value_info 제거(122→121, checker PASS, 계산 불변). FP32 ONNX는
  value_info=0이라 무영향 — 문제는 양자화기 산출 QDQ에 국한.
- **#2 HTP는 fp16-native(native fp32 없음):** FP32 ONNX 올리면 그래프 첫머리에
  `QNN_DATATYPE_FLOAT_32_converted_input_QNN_DATATYPE_FLOAT_16` 노드(execution_cycles 35758) 자동 삽입→fp16 실행.
  "FP32 대비 배속"=실은 "fp16 대비". FP32 top-1도 0.745(−0.005 vs ORT 진짜 fp32).
- **#3 엄격 NCHW(NHWC 가설 기각):** `job.target_shapes={'input':((1,3,224,224),'float32')}`. NHWC (1,224,224,3) 피드는
  FAILED(`Expected [1,3,224,224], got [1,224,224,3]`, 잡 jgj7nve1g). 붕괴 원인은 레이아웃 아님. 추론 입력 키는 모델
  실제 입력명(ResNet50 = `"input"`)과 일치해야.

**캐비앗(불변):** ① 절대 지연=on-device `estimated_inference_time`(HTP 스케줄러 추정)·배치1 → 다른 단계
wall-clock/event-timed와 1:1 비교 불가. ② top-1=200장 서브셋(1단계 함정 0). native 0.735도 **ORT 0.750보다 낮아
"공짜 아님"** — 상대 관계(FP32 충실 vs 외부 QDQ 붕괴 vs native 회복)만 유효. ③ AI Hub는 **Qualcomm 전용** → 세 벤더
중 Qualcomm 축만. TI TDA4VM(TIDL)·Renesas RZ/V2H(DRP-AI TVM)는 보드/툴체인 대기. "QCS8550 (Proxy)"=프록시 디바이스,
"SA8775P ADP"=실제 자동차 보드.

**tech-reviewer 팬인 PASS(🔴0·🟡1·🟢8):** 8-JSON+raw+scratchpad배열 SSOT 독립 재계산 전건 일치 — 산술(×1.77=
1864/1052·×2.03=3056/1505)·정확도 4경로 배열 재현(0.750/0.745/0.005/0.735·agreement·distinct)·raw 사이클 5건 합산
(native 1,985,339/127·외부 3,754,903/128)·SVG 8막대 비례(지연 µs×0.18·정확도 top1×600)·§오염 0(06_multi_soc 21
insert/0 delete)·잡ID 12개·크로스링크·캐비앗. 🟡1 = 외부QDQ ORT-CPU `0.753`이 200장 격자 밖(0.753×200=150.6 정수
불가)·미소싱 → **실제 cpu_proxy x86 CPUEP 1,000장 값으로 출처 명시**해 해소(리뷰어 지적 후 수정).

**미커밋** — 규약대로 사용자 요청 시 커밋([[repo-is-public-scan-before-commit]]). 스캔: 스크립트/문서 전건 시크릿
청결(AI Hub 토큰은 `~/.qai_hub/client.ini` repo 밖·모든 문서에서 `<AI_HUB_TOKEN>` 마스킹, 계정 비번은 어떤 qai_hub
CLI/API도 안 써 미저장). 격리 venv라 정본 ORT 1.23.2 오염 없음([[machine-ai-lap-rtx3080]]).

**남은 과제:** TI TDA4VM(TIDL)·Renesas RZ/V2H(DRP-AI TVM) 벤더 NPU 실측은 보드/툴체인 대기(4-A·4-C). AI Hub
디바이스 팜의 다른 자동차 SoC(SA8295P 등)는 `scripts/qaihub_device.py "<device>" <slug>`로 확장 가능.
