---
name: study-guide-project
description: "0.5단계(배포 난이도 사다리) 실습을 이 머신에서 실행 완료 — 새 venv 2개, 실측 발견/수정 사항, 산출물 위치"
metadata: 
  node_type: memory
  type: project
---

2026-07-31에 `study_guide/02_deployment_ladder.md`(0.5단계, Lv.1~4 + Qualcomm AI Hub)를 이 머신에서
실제로 손으로 실행하고, 커맨드+터미널 출력을 `logs/stage0.5_ladder_log.html`에 정리했다.
[[stage0-env-installed]]의 `~/emb-ai`는 Lv.3에 재사용했고, Lv.2/Lv.4용으로 venv 2개를 새로 만들었다.

**새 환경:**
- `~/ladder-litert` (Lv.2 LiteRT): `ai-edge-litert==2.1.6` + `tensorflow==2.21.0`. GPU 라이브러리
  LD_LIBRARY_PATH 픽스 없음(변환 작업엔 불필요, CPU로 정상 동작).
- `~/ladder-et` (Lv.4 ExecuTorch + Qualcomm AI Hub): 최종 검증된 조합은
  **`torch==2.12.1` + `torchvision==0.27.1` + `executorch==1.3.1`** (+`qai-hub==0.53.0`).

**핵심 발견 — executorch/torch/torchvision 3자 버전 핀 충돌:**
`pip install executorch`만 실행하면 `torch==2.13.0`이 자동으로 깔리는데(executorch 1.3.1의 제약은
`torch>=2.12.0a0`로 하한만 있음), executorch의 컴파일된 pybindings 확장이 실제로는 그보다 좁은 ABI에
맞춰 빌드돼 있어 `torch 2.13`에서 `ImportError: undefined symbol:
_ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`로 깨진다. `torch==2.12.1`로 내리면
고쳐지지만, 이번엔 `torchvision==0.28.0`(torch==2.13.0 전용 컴파일)이 `RuntimeError: operator
torchvision::nms does not exist`로 깨진다 → `torchvision==0.27.1 --no-deps`로 맞춰야 셋 다 동작.
**Why:** pip의 버전 제약이 느슨해도(`>=`) 컴파일된 C++ 확장은 정확한 ABI 매칭을 요구할 수 있음 —
[[stage0-env-installed]]의 TensorRT `libnvinfer.so.10` LD_LIBRARY_PATH 문제와 같은 계열("설치는
성공했지만 조용히/불명확하게 깨지는 실행부") 함정.
**How to apply:** 이후 이 venv를 재현하거나 executorch 버전을 올릴 때는 반드시 `torch`/`torchvision`
페어링을 먼저 고정하고(torch 2.X ↔ torchvision 0.(15+X) 패턴, 예: 2.12↔0.27, 2.13↔0.28) executorch를
그 위에 설치할 것 — `pip install executorch`의 기본 의존성 해석 결과를 그대로 믿지 말 것.

**기타 실측 발견 (상세는 `logs/stage0.5_ladder_log.html` 참고):**
- Edge Impulse CLI(`edge-impulse-cli@1.39.2`): 문서의 `edge-impulse-runner`는 실제로
  `edge-impulse-run-impulse`로 리네임되어 있음. `daemon`/`run-impulse` 서브커맨드는 `--help`를 무시하고
  로그인 프롬프트/시리얼 탐색으로 바로 진입 — 자동화 시 `timeout N ... </dev/null`로 감싸야 함.
- `ai-edge-litert==2.1.6`에는 문서가 "확인 필요"로 플래그한 `CompiledModel` API가 실제로 없음
  (`interpreter` 모듈엔 `Interpreter`/`InterpreterWithCustomOps`/`SignatureRunner`뿐) — 레거시
  `Interpreter` 경로만 유효.
- Qualcomm AI Hub(`qai-hub==0.53.0`): 계정 없이는 `hub.get_devices()`가 즉시
  `UserError: ~/.qai_hub/client.ini not found`로 깨짐 (정상 동작, 에러 메시지가 가입 경로를 안내).

**반영 완료(2026-07-31):** `study_guide/01_environment_setup.md` 3-4-a절에 TensorRT `tensorrt_libs`
LD_LIBRARY_PATH 픽스를 사용자 요청으로 반영함 — 자세한 내용은 [[stage0-env-installed]] 참고.

**산출물:** `logs/stage0.5_ladder_log.html`(실행 로그, 이 세션의 stage0_setup_log.html과 같은 스타일),
`~/ladder-work/ladder_notes.md`(Lv.2/Lv.3 비교표 + 5단계 골격 매핑), `~/ladder-work/{lv2,lv3,lv4}/`의
실제 산출 모델 파일들.

**Lv.2 PTQ 심화 문서 추가(2026-07-31):** `logs/lv2_ptq_deep_dive.html` — `lv2_convert.py`의 PTQ 4종
(FP32/dynamic-range/float16/full-INT8)을 이론(논문 4편 + 공식 LiteRT 문서 3페이지 인용)과 실측으로
대조한 문서. 작성 시 `ai_edge_litert.interpreter.Interpreter`로 실제 `.tflite` 파일을 열어 양자화
파라미터를 직접 뽑아 썼다. 그때 확인된, 문서화해 둘 만한 사실 2가지:
- `model_dynamic.tflite`와 `model_int8.tflite`의 **per-channel 가중치 scale이 완전히 동일** — 가중치
  양자화는 `representative_dataset`과 무관하게 가중치 텐서 자체의 min/max로만 계산되므로, 두 변형의
  실질적 차이는 "활성화를 언제 양자화하느냐"뿐임을 실측으로 확인.
- full-INT8이 4종 중 **가장 느리게** 측정됨(0.020ms). 이 데모는 MAC 수가 수천 단위라 overhead-bound
  이고 파이썬 루프 측정 노이즈와 같은 자릿수여서 그렇다 — 일반 결론이 아니므로 인용할 때 주의.

**공개 상태:** `logs/` 3개 파일 + README 구조 갱신을 2026-07-31에 커밋·푸시 완료(`40a725a`). 저장소가
public이라 커밋 전 비밀정보 스캔이 필요하다 — [[repo-is-public-scan-before-commit]], 푸시 인증 상태는
[[repo-git-push-auth]] 참고.
