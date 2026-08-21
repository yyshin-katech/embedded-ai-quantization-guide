---
name: stage4-arm-cpu-fallback-proxy
description: "4단계 CPU 폴백 프록시 실측(2026-08-18~20): 같은 ResNet50 INT8 QDQ·같은 ORT CPUEP인데 dot-product 명령(SDOT/VNNI) 유무가 양자화 이득 부호 결정. Pi5 A76(dotprod) ×1.83·Jetson AGX Orin A78AE(dotprod) ×2.11 빨라짐 vs x86 no-VNNI 1.76×·imx8mn A53(no-dotprod ARM) 1.65× 느려짐(부호는 ISA 계열 아닌 dotprod 유무, ARM 내부 A76/A78AE vs A53에서 확정). FP32 크로스 전쌍 100%; INT8은 같은 정수커널 경로면 100%(Jetson↔Pi5) 경로 다르면 ~96%. 커밋 2ae48be(Pi5/x86)·581e951(A53)·49e30ff(Jetson 데이터/리포트) 푸시완료; 문서 06 §2-2 Jetson 행+신설 §2-3 반영 완료(1a8f073). 경로의존 CPU↔가속기 확장(2026-08-21): 같은 Orin scale서 TRT INT8 vs A78AE MLAS 961/1000=4단계 x86 대역 → 정수커널 규칙이 가속기까지, 온디바이스 정확도 상세 [[stage3-jetson-ondevice-hands-on]] 후속3"
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
스캔 청결(전체 Tailscale 호스트 IP·PAT·평문 암호 0건).

---

**확장 (2026-08-19) — i.MX8M Nano Cortex-A53: 빈 사분면 "dotprod 없는 ARM" 채움 → sign-flip 3-사분면 완성:**
프록시(1)의 두 "느려짐"이 **둘 다 x86**이라 "ARM 계열 vs dotprod 유무" 가설이 미구분이었음 → 사용자 실기
**NXP i.MX8M Nano LPDDR4 EVK**(Cortex-A53·**ARMv8.0**·`asimddp` **없음**·**NPU 없음**[Plus에만], 2GB LPDDR4·swap 0,
ORT **1.17.1**)로 결정적 한 점 측정. SSOT=`cpu_proxy/{results,raw}/*imx8mn_a53*.json`.
- **A53 INT8 느려짐:** FP32 **680.2026ms → INT8 1123.0230ms = 1.65× 느림**(×0.61). top-1 0.7620→0.7560. A76·A53는
  같은 ARMv8 계열인데 부호 반대 → **부호를 가르는 건 ISA 계열이 아니라 dot-product 명령(SDOT/VPDPBUSD) 유무**임을
  ARM 내부에서 확정. A53(dotprod ×)·x86(VNNI ×)이 같은 부호. 통념 "ARM이면 INT8 유리"를 ARM 안에서 반박.
- **크로스플랫폼 3-플랫폼:** FP32 세 쌍 모두 **1000/1000(100%)**(ORT 1.17.1/1.23.2/1.28.0 무관). INT8 imx↔pi5
  **965/1000**·imx↔x86 **961/1000**·x86↔pi5 958/1000(≈96%). → INT8 비트단위 기대 금지가 3점으로 확립.
- **실물 저사양 벽 2건(결과 불변 증명 후 해소):** (a) 2GB no-swap FP32 **OOM rc=137**(602MB float32 배열) →
  `rpi_bench_lowmem.py`(uint8 캐시 mmap + 이미지 1장씩 lazy 전처리, 전처리 elementwise라 예측 비트 동일, peak
  ≈602→333MB). (b) INT8 **opset skew** — 모델 `opset_import`가 미사용 `ai.onnx.ml v5`(+training/com.microsoft/
  org.pytorch.aten) 선언, ORT 1.17.1 상한 opset 4라 로드 거부 → 미사용 opset 항목 strip해 `('',17)`만 남김(실제
  415 노드 전부 기본 도메인 → 연산 그래프 불변, `resnet50_int8_qdq_op4.onnx`).
- **커밋:** 콘텐츠 커밋 **581e951**(main, 푸시완료 784b31e..581e951 — 규약 [[repo-is-public-scan-before-commit]],
  사용자 요청). 스캔 청결(보드 IP 192.168.x·WiFi SSID·sudo·SSH키·홈경로 0건). 신규 6파일: report HTML +
  results 2 + raw 2 + `rpi_bench_lowmem.py`. **주의: sudo 필요 명령은 사용자가 `!` 프리픽스로 직접 실행**(저장 암호
  미사용). 보드 접근 세부(SSH 키경로·네트워크 인터페이스·호스트라우트)는 비공개 로컬 메모리 — 공개 레포 사본에서 제외.
- **가이드 문서 반영 완료(2026-08-19, `embedded-guide-orchestrator` 부분 재실행 — author-6 직접 + `tech-reviewer`
  서브에이전트 팬인):** ① `06_multi_soc.md` §2-2 sign-flip 표 **2→3-사분면**(A53 행 680.20→1123.02ms 1.65× 느림
  추가)·"부호는 dotprod 유무가 가른다" 핵심 불릿·크로스플랫폼 3-플랫폼·벽 2건 불릿 + HTML 재렌더(승번 오염 0:
  git diff 11 insert/8 delete 전부 §2-2 내부 2→3사분면 교체) ② `cpu_proxy/README.md` 3-사분면 헤드라인/크로스
  플랫폼 표 + **벽 2건 섹션 신설**(OOM→`rpi_bench_lowmem.py`·opset strip 코드 스니펫)·A53 재현법 ③ 최상위
  README·study_guide/README 4단계 콜아웃 3-사분면 확장 + A53 리포트 링크.
- **tech-reviewer 팬인 PASS(🔴 0·material 🟡 0·🟢 2 해소):** 6-raw-JSON SSOT 독립 재계산 전건 일치 —
  크로스플랫폼 6쌍(FP32 1000·1000·1000 / INT8 965·961·958), 1.65×=1123.0230/680.2026, SVG 6폭
  (260.0×3/141.85/429.26/457.91·기준선 x=400), §번호 오염 0, 크로스링크·벽 2건 서술·캐비앗 병기 확인. 🟢 2건은
  리뷰 후 해소 — ① A53 리포트 §6 "확장 예정→반영 완료" 시제 정정(2곳), ② sign-flip SVG 3-플랫폼 라벨이
  viewBox 800 우변 초과 → **viewBox 800→860**(width="100%"라 rect 폭·그리드선·기준선 절대값 전부 불변, 라벨
  공간만 확보).
- **문서 반영 커밋 47e83a9**(main, 푸시완료 581e951..47e83a9 — 규약 [[repo-is-public-scan-before-commit]], 사용자
  요청). 문서 6파일(06 md/html·cpu_proxy README·최상위 README·study_guide README·리포트 §6/viewBox 수정),
  133 insert/65 delete. 스캔 청결(IP·암호·SSH키·토큰·WiFi·홈경로 0건; 매치 2건은 산문 단어 "sudo"뿐).

---

**확장 (2026-08-20) — Jetson AGX Orin Cortex-A78AE: 2번째 dotprod-ARM 확증 + 새 발견 "같은 정수커널 경로 = INT8 100% 동일":**
사용자 요청 실기 **NVIDIA Jetson AGX Orin Dev Kit**(Cortex-**A78AE**·**ARMv8.2**·`asimddp`/SDOT **있음**, 12코어·61GB·MAXN,
접속 [[jetson-agx-orin-board-access]])로 4번째 CPU 점 측정. 같은 3·5단계 ResNet50 자산·같은 `rpi_bench.py`·**ORT 1.23.2**
(x86와 동일 버전 → 버전 교란 0). RAM 충분해 lowmem 아닌 풀 `rpi_bench.py`(Pi5/x86 경로). SSOT=`cpu_proxy/{results,raw}/*jetson_orin_a78ae*.json`.
- **A78AE INT8 빨라짐:** FP32 **38.4720ms → INT8 18.2178ms = ×2.11 빠름**(int8/fp32=0.474). top-1 0.7620→0.7500. dotprod
  있는 ARM이라 Pi5 A76(×1.83)와 **같은 부호**, 배율은 더 큼. → **사분면 신설 아님**(빈 칸은 x86+VNNI인데 하드웨어 없음) —
  "ARM+dotprod→INT8 빨라짐" 셀의 **2번째 기기 확증**(dotprod 유무 가설이 A76 단일점 우연 아님을 재현).
- **🆕 새 발견 — INT8 예측 정체성은 "정수커널 경로"가 가른다:** 크로스플랫폼 pairwise pred_cls(젯슨 기준) — FP32 **세 쌍
  1000/1000(100%)**; INT8 **Jetson↔Pi5 = 1000/1000(100%!)** vs Jetson↔x86 958·Jetson↔imx8 965(≈96%). 두 dotprod-ARM
  (A76·A78AE)은 **같은 MLAS SDOT 정수커널** → INT8 예측 **비트 동일**. 교차확증: Jetson↔x86 958 = 기존 x86↔pi5 958(둘 다
  "SDOT-ARM vs x86-scalar"), Jetson↔imx 965 = 기존 imx↔pi5 965(둘 다 "SDOT-ARM vs A53-nondotprod") — **완전 일치**.
  → 기존 "INT8 ≈96% ISA 민감"을 **"같은 정수커널 경로면 100%, 경로 다르면 ~96%"로 정밀화**(부호도 정체성도 dotprod/커널경로가 지배).
- **환경 벽:** 젯슨에 pip/venv 부재(python3-venv 없음, ensurepip 실패) → sudo 없이 `get-pip.py --user`+`pip install --user
  onnxruntime onnx`로 해소(~/.local 격리, 시스템 numpy 1.21.5 불변). CPU 휠이라 providers=[Azure,CPU] — CPUEP만 씀(GPU
  discovery 경고 무해). 자산 scp 전 서브셋 npy를 `tv.npy[:1000]`로 재생성(원본 유실) → **x86 재현 pred_cls 1000/1000 일치로
  비트정확 검증** 후 전송(file md5 3c0e151…/08b054c… 양단 일치).
- **산출물(커밋 49e30ff 푸시완료):** `cpu_proxy/{raw,results}/jetson_orin_a78ae_{fp32,int8}.json` 4개 + `logs/stage4_jetson_agx_orin_a78ae_report.html`(§1~5·4-플랫폼 sign-flip SVG·INT8 6쌍 매트릭스·결정적 대조). golden 회귀 무영향(cpu_proxy 격리 유지, 6행·3 passed 불변). 리포트 스캔서 접속세부 2곳(호스트명·ssh 키경로/무암호) 걷어냄(비밀값 아님, public 레포 불필요 인프라 정보). **문서 06 §2-2 Jetson 행 + 신설 §2-3 반영 완료**(2026-08-20, `embedded-guide-orchestrator` 부분 재실행 — author-5·6 직접 + `tech-reviewer` 팬인 PASS, 문서 커밋 **1a8f073** 푸시완료 00fd97d..1a8f073). 이때 같은 Jetson 실기로 **온디바이스 가속기(iGPU 정밀도 사다리 + DLA 오프로드/성능·와트)** 도 실측 → [[stage3-jetson-ondevice-hands-on]](데이터 커밋 00fd97d): CPU 폴백 바닥값(이 메모리)의 짝 = 가속기 천장.

**확장 (2026-08-21) — 정수커널 경로의존이 CPU↔가속기 경계를 넘음(온디바이스 정확도 축):** 위 "같은 정수커널
경로면 100%, 다르면 ~96%"를 CPU↔CPU 너머로 확장. 같은 Orin 실리콘·같은 `resnet50_int8_qdq.onnx` scale에서
**TRT INT8 커널(GPU) vs A78AE MLAS SDOT INT8(CPU) = 961/1000**(39장) — 4단계 Jetson↔x86 CPU 교차(958)와 **같은
대역**, TRT는 별개 정수커널이므로 규칙이 **CPU↔가속기**에도 성립. FP32는 iGPU vs CPU **1000/1000**(가속기까지
100% 유지). 겸사 온디바이스 accuracy-valid iGPU INT8(explicit QDQ) top-1 0.762 = FP32 > 이 메모리의 CPU MLAS
INT8 0.750. 상세·정확도 사다리 [[stage3-jetson-ondevice-hands-on]] 후속3(SSOT `jetson_ondevice/accuracy/`).

**남은 과제:** 4단계 벤더 NPU 본실측(TIDL/DRP-AI 컴파일·offload% 실측)은 TI TDA4VM·Renesas RZ/V2H 보드 확보 시
(Qualcomm HTP는 [[stage4-qualcomm-aihub-hands-on]]로 완료). i.MX8M Nano CPU 바닥값(offload 0%)과 HTP 상한선이
"가속기가 이겨야 할 최소선 ↔ 최대 이득" 양 끝 대조축. (S32G 보드 테스트도 사용자 관심사로 대기.)
