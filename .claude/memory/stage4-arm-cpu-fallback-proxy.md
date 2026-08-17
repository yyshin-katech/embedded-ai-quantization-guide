---
name: stage4-arm-cpu-fallback-proxy
description: "4단계 CPU 폴백 프록시 실측(2026-08-18, Raspberry Pi 5 Cortex-A76): 같은 ResNet50 INT8 QDQ·같은 ORT CPUEP인데 CPU ISA가 양자화 이득 부호 결정 — Pi dotprod INT8 ×1.83 빠름 vs x86 no-VNNI 1.76× 느림. FP32 크로스플랫폼 예측 100% 일치·INT8 95.8%. Pi는 NPU 아닌 폴백 바닥값(offload 0%)"
metadata:
  node_type: memory
  type: project
---

4단계(멀티 SoC) 벤더 NPU(TIDL/QNN/DRP-AI)는 보드 대기지만, 세 SoC가 **공통으로 가진 ARM Cortex-A 폴백
경로**(미지원 op가 떨어지는 곳)는 오늘 측정 가능 → 사용자 요청 실기 **Raspberry Pi 5**(Cortex-A76, `home-rpi`,
Tailscale)를 프록시로 완주(2026-08-18). 3·5단계 자산 **ResNet50 INT8 QDQ ONNX**([[stage3-tensorrt-hands-on]]·
[[stage5-infrastructure-hands-on]])를 순수 `CPUExecutionProvider`로 관통. AI-LAP/RTX3080([[machine-ai-lap-rtx3080]])이
x86 대조. 산출물: `logs/stage4_arm_cpu_fallback_report.html` · `experiments/stage5_infrastructure/cpu_proxy/`
(results 4·raw 4·이식형 `rpi_bench.py`·README) · `study_guide/06_multi_soc.md` §2-2 🔬 콜아웃.

**헤드라인 — 양자화 이득의 부호는 CPU ISA가 결정한다(SSOT=cpu_proxy/*.json):**
같은 INT8 QDQ 그래프·같은 ORT CPUEP인데 CPU만 바꾸면 부호가 반대.
- **Pi 5 Cortex-A76**(`asimddp`/SDOT INT8 dot-product **있음**, `i8mm` 없음): FP32 **144.9519ms → INT8 79.0827ms
  = ×1.83 빠름**. top-1 0.7620→0.7500. ORT 1.28.0(cp313 aarch64 휠). 런 직후 78.5°C 스로틀 없음.
- **x86 i9-10900K**(Comet Lake, **VNNI 없음**·AVX2까지): FP32 **9.2765ms → INT8 16.3376ms = 1.76× 느림**(×0.57).
  top-1 0.7620→0.7530. ORT 1.23.2.
- **왜:** ORT CPU 커널(MLAS)은 정수 dot-product 명령 있으면 INT8 가속(ARM `SDOT`=`asimddp` / x86 `VPDPBUSD`=VNNI),
  없으면 못 함. A76엔 있고 Comet Lake엔 없다. x86은 가속 부재 + `QuantizeLinear`/`DequantizeLinear` 노드 비용까지
  얹혀 FP32보다 느려짐. → "양자화 이득은 하드웨어 의존"을 CPU 축에서 직접 실증. **함의:** NPU는 INT8 전용 MAC이라
  항상 이김; 부호 반전은 **폴백된 부분**에서 벌어진다 → 폴백 많을수록 A-코어 dotprod 유무가 성능 좌우(dotprod 없는
  구형 A53/A72면 폴백 INT8이 독).

**크로스플랫폼 예측 동일성은 FP32에서만:** raw `pred_cls` 1000개 x86↔ARM 1:1 → **FP32 1000/1000(100%, differ 0)**,
**INT8 958/1000(95.8%, differ 42)**. 정수 누산·재양자화 반올림이 ISA 민감(SDOT vs AVX2 정수경로). FP32가 100%
일치하므로 갈림은 ORT 버전(1.28.0/1.23.2) 아니라 ISA의 INT8 경로. → **INT8 정확도를 다른 타겟에서 비트 단위로
기대 금지.**

**부수 발견 — golden 회귀는 TRT 전용:** `bench/tests/test_regression.py::test_matrix_matches_golden`은
`dataframe_regression.check(atol=1e-3, rtol=1e-3)` — 결정론적 TRT 지연엔 맞지만 **비결정 CPU wall-clock**(Pi 145ms를
±0.145ms로 재현 불가)엔 부적합. CPU 4행을 golden-glob되는 `bench/results/`에 넣으면 `Obtained (10,) vs Expected
(6,)` 실패(실측 확인) → 별도 **`cpu_proxy/results/`로 격리**, 커밋 하네스 **6행·pytest 3 passed 유지**. (임계값 2
테스트는 baseline과 inner-merge라 새 SoC 애초에 제외 — 정책상 정상.) `cpu_proxy/results/*.json`은 매트릭스 스키마
동일(peak_mem_mb=NaN·engine_build_s=0.0·trt_version="").

**캐비앗(불변):** ① Pi는 **ARM Cortex-A 폴백 프록시일 뿐 자동차 NPU(QCS8550/RZ-V2H/TDA4VM) 아님** — 가속 수치
전이 불가, 측정=offload 0% 바닥값. ② 절대 지연·top-1은 CPUEP·wall-clock 단일입력·배치1·1,000장 서브셋·ORT 버전
상이 기준 → **상대 관계(부호·배율·예측 일치율)만 유효**. ③ Pi↔x86 절대속도는 다른 급 기계라 논점 아님. ④ 1,000장
서브셋 top-1 부풀림 가능(1단계 함정 0).

**tech-reviewer 팬인 PASS(🔴 0·material 🟡 0, 수정 0건):** 8-JSON SSOT 1:1·크로스플랫폼 순수파이썬 재계산
(1000/1000·958/1000)·산술(×1.83=144.9519/79.0827·1.76×=16.3376/9.2765·×0.57)·SVG rect 4개
(260.0/141.85/260.0/457.91, 기준선 x=400)·크로스링크 실재·§오염 0(git diff 18 insertions·0 deletions)·회귀 3
passed·캐비앗 3종 병기 확인.

**커밋:** 콘텐츠 커밋 **2ae48be**(main, 푸시완료 — 규약 [[repo-is-public-scan-before-commit]], 사용자 요청).
스캔 청결(전체 Tailscale 호스트 IP·PAT·평문 암호 0건). 이 메모리 갱신은 뒤따르는 동기화 커밋.

**남은 과제:** 4단계 벤더 NPU 본실측(TIDL/QNN/DRP-AI 컴파일·offload% 실측)은 보드·SDK 확보 시(4-A~C). 그때 이
CPU 바닥값이 "가속기가 이겨야 할 최소선" 대조축.
