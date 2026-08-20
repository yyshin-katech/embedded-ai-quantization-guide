# 작업 메모리 사본

이 가이드를 실제 머신에서 따라 하며 쌓인 **실측 기록**입니다. Claude Code의 로컬 메모리
(`~/.claude/projects/<project>/memory/`)를 저장소로 복사한 것이라, 다른 PC에서 작업을
이어받을 때 맥락이 끊기지 않게 하려는 목적입니다.

정본은 로컬 메모리 쪽이고 이건 사본입니다. 작업 상태 요약은 [`../../HANDOFF.md`](../../HANDOFF.md)를 먼저 보세요.

> ⚠️ 메모리는 **작성 시점의 관찰**입니다. 파일·함수·플래그를 지목한 서술은 지금도 유효한지
> 코드로 확인한 뒤 사실로 인용하세요.

현재 작업 머신은 **AI-LAP (RTX 3080)** 입니다 — 이전 Nuvo-6108GC(RTX 3060)가 GPU 고장으로 교체됐습니다.

| 파일 | 내용 |
|---|---|
| [`machine-ai-lap-rtx3080.md`](machine-ai-lap-rtx3080.md) | 현재 작업 머신(AI-LAP 노트북, RTX 3080) — 죽은 Nuvo/RTX3060 대체, venv·데이터 이관, GPU가 QAT를 300W로 완주 |
| [`stage0-env-installed.md`](stage0-env-installed.md) | 0단계 환경 — 확정 스택 버전, `LD_LIBRARY_PATH` 픽스 2개(cuDNN·TensorRT), `dynamo=True`가 요구하는 `onnxscript`, opset 다운컨버트 무음 폴백 |
| [`study-guide-project.md`](study-guide-project.md) | 0.5단계 배포 사다리 — `executorch`/`torch`/`torchvision` 3자 ABI 핀 충돌과 해법, LiteRT `CompiledModel` API 부재, Lv.2 PTQ 4종 실측 |
| [`stage1-quantization-hands-on.md`](stage1-quantization-hands-on.md) | 1단계 양자화 이론 2회 실행 — ORT Entropy가 MinMax로 퇴화(산출 md5 동일), TensorRT 폴백 원인은 activation zero-point≠0 하나뿐, 50k 재실행 정정 12건 |
| [`imagenet-val-50k-local.md`](imagenet-val-50k-local.md) | ImageNet val 50,000장 확보·검증 경위, 라벨 규약, 전처리 2종(`crop_tv` vs `crop_squash`)의 −1.07%p 차이 |
| [`qat-recovery-experiment.md`](qat-recovery-experiment.md) | **완료** — QAT 회복(W4A8 손실변형): FP32→PTQ 4-bit −24.16%p→QAT 97.1% 회복, QAT−대조군 −1.50%p("공짜 아님") |
| [`stage2-detr-hands-on.md`](stage2-detr-hands-on.md) | 2단계 DETR INT8(커밋 41dc49e) — 초안 단정 3건 반전(export 블로커=SDPA·op선택 mixed 실패·손상 분산), §4.4·§4.6 후속 완료 |
| [`stage2-bevformer-hands-on.md`](stage2-bevformer-hands-on.md) | 2단계 BEVFormer-tiny §4.6 — FP32 nuScenes-mini mAP 0.2647(스모크), op 단정 반전 0·실전 함정 +2(mmcv op CPU-only export·전체 export는 point_sampling에서 사망), 전체 INT8은 포크 필요(범위 밖), 무컴파일 레거시 env 레시피 |
| [`stage2-smoothquant-hands-on.md`](stage2-smoothquant-hands-on.md) | 2단계 §4.4 SmoothQuant(nvidia-modelopt 0.45.0) — per-tensor INT8 폭락(0.4209→0.3301)의 59.9%를 SmoothQuant(α=1.0)가 회복(→0.3845, +0.0544 mAP), op-선택 mixed의 ~15배 → "activation 입도가 레버" 확증. 프리셋 기본 α=1.0(논문 0.5 아님)·absmax 3.69×→1.96×. torch fake-quant 경로라 상대 관계만 |
| [`stage3-tensorrt-hands-on.md`](stage3-tensorrt-hands-on.md) | 3단계 TensorRT(ResNet50) — 정본 pip 휠에 **trtexec 부재**→polygraphy Python API(FP16 ×1.96·INT8 ×2.12·−0.52%p), 1단계 §2.2.1 "zp≠0 하나뿐"이 **직접 파서에선 INT32 bias DQ까지 둘**(경로 병기·반전 아님), deprecated implicit 캘리브레이터가 TRT 10.16서 여전히 빌드(경고 134건=126+8). DLA는 범위 밖(num_DLA_cores=0) |
| [`stage3-jetson-ondevice-hands-on.md`](stage3-jetson-ondevice-hands-on.md) | **3·4단계 Jetson AGX Orin 온디바이스**(데이터 커밋 00fd97d·문서 1a8f073 푸시완료, ResNet50) — 실 `trtexec` 실존으로 3·5단계 "trtexec 부재→polygraphy" 반전·3단계 "DLA 범위 밖" 해소. **DLA INT8 = 성능/와트 챔피언 51.29 inf/s/W**(iGPU INT8의 **×1.547**·전력 절반, 지연만 1.262×)·**DLA는 INT8 전용기**(레이어 배치 동일한데 FP16 **13.87× 느림** = 순수 NVDLA v2 데이터패스)·오프로드 GR3D 95%→3~16%로 증명(DLA-후보 2/2, GPU폴백 2층뿐)·작은 Ampere iGPU **INT8≈FP16 0.984**(RTX3080 0.927). 05 §2.3 + 06 §2-3 반영, tech-reviewer 팬인 PASS. ⚠️ DLA INT8은 `--int8` 암묵 캘리브라 지연·전력만 유효, 정확도 미주장 |
| [`stage5-infrastructure-hands-on.md`](stage5-infrastructure-hands-on.md) | 5단계 인프라화(커밋 ff523de 푸시완료, ResNet50) — 벤치 하네스 polygraphy `TrtRunner`(FP16 ×1.80·INT8 ×2.13, 하네스 wall-clock이라 3단계 event-timed보다 factor 압축). **무음 오답 2건**: zero-copy 버퍼 에일리어싱(`.copy()` 누락→top-1 0.0014 붕괴, pycuda→polygraphy 치환이 도입)·pivot `dropna` 회색행 드롭(CSV/회귀 baseline은 무영향). `device_memory_size_v2`=scratch mem≠엔진파일. pytest-regressions 최신 2.11.0(문서 "v3.0+" 허구). tech-reviewer 팬인 PASS(🔴0). 4단계는 보드 대기 |
| [`stage4-arm-cpu-fallback-proxy.md`](stage4-arm-cpu-fallback-proxy.md) | **4단계 CPU 폴백 프록시**(부분·프록시 — 벤더 NPU 대신 세 SoC 공통 **ARM Cortex-A 폴백 경로** 실측; 커밋 2ae48be Pi5/x86·581e951 A53·49e30ff Jetson A78AE·문서 1a8f073 푸시완료) — 같은 ResNet50 INT8 QDQ·같은 ORT `CPUExecutionProvider`인데 **dot-product 명령(ARM SDOT/x86 VNNI) 유무가 양자화 이득의 부호를 결정**(ISA 계열 아님). **sign-flip 3-사분면**: Pi5 A76 **×1.83**·Jetson A78AE **×2.11** 빠름(둘 다 dotprod) vs x86 no-VNNI **1.76×**·imx8mn A53(no-dotprod ARM) **1.65×** 느림. **새 발견**: INT8 크로스플랫폼 예측 동일성은 **정수커널 경로 의존** — 같은 MLAS SDOT면 **100%**(Jetson↔Pi5 1000/1000), 경로 다르면 ~96%; FP32는 전쌍 100%. golden 회귀는 결정론적 TRT 전용→CPU 4행 `cpu_proxy/`로 격리(하네스 6행·3 passed). Pi/Nano/Jetson은 NPU 아닌 CPU 폴백 바닥값(offload 0%). 온디바이스 가속기 실측은 [`stage3-jetson-ondevice-hands-on.md`](stage3-jetson-ondevice-hands-on.md), Qualcomm NPU는 [`stage4-qualcomm-aihub-hands-on.md`](stage4-qualcomm-aihub-hands-on.md); TI/Renesas 보드 대기 |
| [`stage4-qualcomm-aihub-hands-on.md`](stage4-qualcomm-aihub-hands-on.md) | **4단계 Qualcomm 벤더-NPU**(부분·클라우드, 보드 없이 — Qualcomm AI Hub, ResNet50) — CPU 프록시(offload 0% 바닥값) 위에 벤더 NPU 실측을 얹음. Hexagon HTP 두 종(**QCS8550**·**SA8775P ADP** 자동차 보드) `qnn_context_binary`로 컴파일 → **둘 다 100% NPU offload**·INT8 fp16 대비 **×1.77/×2.03**(execution_cycles 교차확증). **🔴 무음 오답**: 외부 ORT-QDQ 지참 시 compile/profile은 통과해도 on-device top-1 **0.75→0.005 붕괴**(같은 경로 FP32(→HTP fp16)는 0.745 충실; 동일 ONNX는 x86 CPUEP서 0.753 정상 → 범인=HTP 임포트가 외부 QDQ scale 무시) → **올바른 경로=AI Hub 자체 `submit_quantize_job`**로 **0.735 회복**(748µs로 외부-QDQ INT8 1052µs보다 빠르고 leaner). 툴체인 발견 3종(value_info↔IO 엄격거부→`clean_valueinfo_for_aihub.py`·HTP fp16-native·엄격 NCHW). AI Hub는 Qualcomm 전용→TI/Renesas는 보드/툴체인 대기. tech-reviewer 팬인 PASS(🔴0·🟡1해소) |
| [`stage8-capstone-hands-on.md`](stage8-capstone-hands-on.md) | **8단계 캡스톤 BEVDet end-to-end**(커밋 64e4c84 + 후속 INT8 463df43 푸시완료) — 커스텀 CUDA op `bev_pool_v2`(python fallback 없음)를 sudo·Docker 없이 **user-space cu117 툴체인(제3의 길)**으로 컴파일(문서 §3 2안 다 불가)→nuScenes-mini **FP32 walking skeleton 관통**. 정식 가중치 **Baidu-locked**→init 가중치라 mAP 0.0000(예상값), latency **p50 34.06ms**(공식 33.3ms 교차확증). **후속 세션 INT8/TRT-plugin 6벽(W1~W6) 관통**: 커스텀 `TRTBEVPoolV2` 플러그인 직접 빌드(58,896 B, 풀 mmdeploy CMake 우회)+export shim(W5 `_Map_base::at`→`new_zeros` monkeypatch)+`build_serialized_network`(W6 API)로 **FP32→FP16→INT8 엔진 사다리** 실측(지연 14.68→4.91→2.63ms **×5.58**·엔진 245/90/47MB·출력편차 INT8 corr 0.985~1.000). init 가중치라 지연·크기·출력편차만 유효. 정본 emb-ai 오염 0(legacy env 전용). tech-reviewer 팬인 PASS(🔴0·🟡1해소) |
| [`gpu-xid79-fallen-off-bus.md`](gpu-xid79-fallen-off-bus.md) | (구 머신 이력·해소) RTX 3060 Xid 79 3회 재발 진단 — SW Power Cap 상시 점등, 배치 축소 무효. 3080 이관으로 해결 |
| [`repo-is-public-scan-before-commit.md`](repo-is-public-scan-before-commit.md) | 커밋 규약 — 시크릿 스캔, main 직접 커밋, 푸시는 요청 시만 |

## 공개본에서 손댄 것

저장소가 public이라 복사 시 다음을 처리했습니다.

- **sudo 암호 마스킹/제외** — 이전 작업 머신의 실제 sudo 암호가 평문으로 있던 것을 마스킹했고,
  갱신본(2026-08-16)부터는 아예 기록하지 않습니다. 비밀값은 문서의 교육적 가치에 기여하지 않아 손실이 없습니다.
- **git 자격증명 세부 제외** — `repo-git-push-auth.md`(자격증명 저장 방식)와 PAT 저장 위치 언급은
  공개 가치가 없고 노출 위험만 있어 공개본에서 뺐습니다. 새 PC에서는 각자 `git` 인증을 새로 설정하면 됩니다.
- **로컬 메타 제거** — `originSessionId`, `modified` 필드는 로컬 세션 식별자라 지웠습니다.
- **보드 접근 인프라 세부 제외** — SSH 키 경로·네트워크 인터페이스·호스트 라우트 등 개발보드 접속 세부는
  공개 가치가 없고 인프라 노출만 되어 사본에서 뺐습니다(비밀값은 아님). 접속은 각자 환경에서 새로 설정하면 됩니다.
- **`jetson-agx-orin-board-access` 미포함** — Jetson 보드 접속 메모리(SSID·내부 IP 포함)는 애초에 사본으로
  복사하지 않습니다. 온디바이스 실측 결과 자체는 [`stage3-jetson-ondevice-hands-on.md`](stage3-jetson-ondevice-hands-on.md)에 있습니다.
