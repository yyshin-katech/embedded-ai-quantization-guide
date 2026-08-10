---
name: gpu-xid79-fallen-off-bus
description: "RTX 3060이 Xid 79(fallen off the bus)로 반복 이탈 — 2026-08-04·08-10 총 3회. 3차 텔레메트리로 SW Power Cap(0x4) 상시 걸림 확인 → 배치 축소는 무효한 레버, 전력 상한(nvidia-smi -pl)과 팬 하한이 다음 레버. 복구는 재부팅뿐"
metadata:
  node_type: memory
  type: project
---

`yuyeong-Nuvo-6108GC`(Neousys 산업용 임베디드 박스)의 RTX 3060이 PCIe 버스에서 떨어지는
사고가 **3회** 재발했다(같은 날 2회 포함). 드라이버 595.84([[stage0-env-installed]]).

| 회차 | 시각 | 워크로드 | 사망 직전 온도/팬/전력 | 부하 후 생존 |
|---|---|---|---|---|
| 1차 | 2026-08-04 10:43:44 | `qat_recovery.py` BS=96 | 미측정 | — |
| 2차 | 2026-08-10 11:14:36 | `qat_recovery.py` 종료 직후 새 잡 | 74°C / 38% / 138W | ~40초 |
| 3차 | 2026-08-10 11:38:15 | `qat_recovery.py` **BS=48** ep1 step 500/833 | **78°C / 43% / 129.7W** | **~190초** |

**공통 시그니처:** `journalctl -k`에 **Xid 79 "GPU has fallen off the bus"** → 즉시
**Xid 154 "Node Reboot Required"**. `lspci`가 `(rev ff)`, `nvidia-smi`는 "Unable to determine
the device handle". 파이썬은 `torch.AcceleratorError: CUDA error: unspecified launch failure`.
Xid 79 줄의 `pid=... name=nvidia-smi`는 텔레메트리 프로세스가 우연히 GPU를 만진 것일 뿐 원인이 아니다.

**배치 축소는 무효한 레버였다(3차에서 실측 반증).** BS 96→48로 반감해도 최대 전력은
138W → 129.7W(**약 8W**)만 줄었다. 배치가 작아지면 GPU가 커널을 더 자주 띄워 빈자리를
메우기 때문이다. 생존 시간은 40초 → 190초로 약 5배 늘었지만 **여전히 죽었다**.

**진짜 시그널은 SW Power Cap이다.** 3차 2초 텔레메트리(98행) 분포:
`0x4`(SW Power Cap) **73행** / `0x0` 23행 / `0x1`(Idle) 2행. 부하 시작 **26초** 뒤
(11:35:31, 123W/53°C) 전력 상한에 걸려 죽을 때까지 **164초 내내 ~129W에 붙어 있었다**.
2차가 `0x0`뿐이었던 건 램프 40초에 죽어 상한 상태에 도달할 시간이 없었기 때문이다.
RTX 3060 정격 TGP는 170W인데 상한이 ~130W에서 걸리는 건 임베디드 섀시 전력 예산에
맞춘 OEM 설정 가능성 → 재부팅 후 `nvidia-smi -q -d POWER`로 enforced/min/max 확인할 것.

**다음 레버 2개(둘 다 sudo — 사용자 결정 사항):**
1. `sudo nvidia-smi -pl <100 등>` — 상한 자체를 내린다. 배치와 달리 draw를 직접 구속한다.
   ~130W에 붙은 채로 죽으므로, 낮춰서 생존하면 전원 계통 가설이 사실상 확정된다.
2. 팬 하한 상향 — 온도 하락 → 누설전류 감소 → 같은 클럭에서 전력 감소.

**열은 "플래그로는" 배제되지만 깨끗하진 않다(2차 결론 수정).** thermal 계열 플래그
(`0x20`/`0x40`/`0x80`)는 3회 모두 0이다. 그러나 3차는 **70°C 이상 54행에서 평균 76°C인데
팬이 평균 39%**였다 — 온도에 팬 커브가 반응하지 않는다. 밀폐 산업용 박스에서 정상 상태로
읽을 값이 아니다. 남은 후보: 섀시 12V 레일/전원 예산, PCIe 슬롯·라이저 접점(진동), VRM 열화.

**팬 100% 고착은 이탈의 증상이다**(드라이버가 팬 제어권을 잃고 안전 기본값으로 감). 단
위처럼 **부하 중 팬이 낮게 붙어 있는 것은 별개의 이상 신호**다 — 둘을 혼동하지 말 것.

**복구는 재부팅뿐이다.** 모듈 리로드는 안 된다(config space 사망). 재부팅은 사용자 결정.
**재부팅 전에 `sudo nvidia-bug-report.sh`로 크래시 덤프를 받을 것** — 모듈 언로드 시 사라지고,
3회 반복이라 벤더 문의의 근거가 된다.

**GPU 작업 전에 `nvidia-smi -L` 생존 확인이 먼저다.** 죽어 있으면 CUDA/TensorRT 벤치는
전부 무의미하고 ORT는 조용히 CPU로 폴백해 잘못된 수치를 뱉는다
(무음 폴백 판별법은 [[stage1-quantization-hands-on]]).

참고: `dmesg`는 이 머신에서 권한 거부(`kernel.dmesg_restrict=1`)라 `journalctl -k`를 쓸 것.
3차 텔레메트리 원본: `~/stage1-work/telemetry_bs48.csv`, 로그 `qat_recovery_bs48.log`.
