## 하네스: AI 양자화 → 임베디드 배포 학습 가이드 제작

**목표:** `guide (1).html`의 단계 구조를 기반으로, Ubuntu 22.04 + NVIDIA RTX 환경에서 실행 가능한 단계별 학습 가이드 MD 세트(`study_guide/`)를 생성/갱신한다.

**트리거:** 양자화·임베디드 가이드 문서의 작성/보완/업데이트/부분수정/재검토 요청 시 `embedded-guide-orchestrator` 스킬을 사용하라. 단순 질문(개념 설명 등)은 직접 응답 가능.

**구성 요소:** 에이전트 `guide-author`(리서치+작성), `tech-reviewer`(검증) / 스킬 `embedded-ai-research`, `stage-guide-writing`, `tech-guide-review`, `md-to-html`(HTML 렌더러) / 오케스트레이터 `embedded-guide-orchestrator`. (상세는 `.claude/agents/`, `.claude/skills/` 참조)

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-07-27 | 초기 구성 (Fan-out/Fan-in + Producer-Reviewer) | 전체 | 단계별 가이드 MD 제작 하네스 신규 구축 |
| 2026-07-28 | MD 심화 재작성 + `md-to-html` 스킬(HTML 렌더러) 추가 | skills/md-to-html, study_guide/* | 각 단계 상세화 및 HTML 버전 제공 요청 |
| 2026-07-31 | 실측 반영 정정 (부분 재실행: author-1 → reviewer-1) | study_guide/01·02·03·04·05·06·07·09·10, README | 0단계를 실제 머신에 설치해보니 정본 핀 `onnxruntime-gpu==1.28.0`이 미존재, `onnx` IR 상한·cuDNN 경로·`numpy<2`·`onnxscript` 누락이 드러남 → 실측 스택(ORT 1.23.2 / onnx 1.18.0)으로 전 문서 통일 |
| 2026-07-31 | 0.5단계 실습 반영: TensorRT EP LD_LIBRARY_PATH 픽스 (3-4-a 보강, 부분수정) | study_guide/01_environment_setup.md·html, README | 0.5단계(배포 사다리) Lv.3 실습 중 `TensorrtExecutionProvider`가 provider 목록엔 나오지만 `libnvinfer.so.10`(venv의 `tensorrt_libs` 패키지, `nvidia/*/lib` 글롭 밖)을 못 찾아 조용히 CPU로 폴백하는 걸 실측 발견(p50 11.83ms=CPU급 → 픽스 후 0.41ms) → cuDNN과 동일한 LD_LIBRARY_PATH 패턴으로 해결, `verify_trt_ep.py` 검증 스크립트·트러블슈팅 행 추가 |
| 2026-08-02 | 1단계 실습 반영: 정정 10건 (부분수정, author-3·5·9 팬아웃 → reviewer 팬인) | study_guide/03·05·10 (+각 html), README, study_guide/README, logs/stage1_quantization_log.html | 1단계(양자화 이론)를 실제로 완주하니 문서의 경험적 단정 여러 개가 실측과 어긋남 → ① ORT `CalibrationMethod.Entropy`가 기본값(`num_bins=128`)에서 후보 1개로 퇴화해 **MinMax와 비트 단위 동일**(scale 32/32 일치), `num_bins`는 `quantize_static` 화이트리스트 5키 밖이라 **전달 자체가 불가** ② "Entropy/Percentile > MinMax" 통념이 ResNet18에선 반대(MinMax 78.90% 최적, pct99.9 −6.20%p, Entropy 정상화 시 −10.80%p) — 클리핑 비율↔top-1 단조 관계로 대체 ③ ORT 권장 `QUInt8` 비대칭 QDQ를 TensorRT가 파싱 못 해 **무음 폴백**(3.05ms, FP32 0.95ms보다 3배 느림) → 대칭 `QInt8`로 0.55ms(5.5배), **2×2 절제 실험으로 하드 블로커가 zero-point≠0 하나뿐**이고 INT32 bias DQ는 2차 증상임을 확정 ④ layer sensitivity 실측이 "conv1이 취약" 통념을 뒤집음(21개 중 가장 안전, per-channel 전제에선 채널 수가 적을수록 유리) ⑤ opset 17 다운컨버트 실패가 **exit 0**으로 통과, external data 2파일 산출 ⑥ QAT/STE 기대 출력값 정정 + STE 미적용 대조군 추가 ⑦ 1000장 큐레이션 셋의 top-1 부풀림 단서 + paired 검정(McNemar) 도입 |
