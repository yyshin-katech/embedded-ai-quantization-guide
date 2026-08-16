---
name: gpu-xid79-fallen-off-bus
description: "(구 머신 이력·해소됨) Nuvo-6108GC의 RTX 3060이 Xid 79로 3회 이탈 — SW Power Cap(0x4) 상시 점등이 진짜 시그널, 배치 축소는 무효한 레버. AI-LAP/RTX3080으로 옮겨 해소(3080은 300W로 완주)"
metadata: 
  node_type: memory
  type: project
---

> ✅ **해소됨(2026-08-16):** 작업을 [[machine-ai-lap-rtx3080]](RTX 3080)으로 옮겨 QAT·2단계를 완주했다.
> 3080은 300W/95%/70°C로 **SW Power Cap 없이** 돌았다 — 아래는 구 머신 진단 이력(3060 박스를
> 되살릴 경우를 위해 보존). "옮기는 게 해결책"(HANDOFF §6) 가설이 실측으로 확인됨.

`yuyeong-Nuvo-6108GC`(Neousys 산업용 임베디드 박스)의 RTX 3060이 PCIe 버스에서 떨어지는 사고가
**3회** 재발했다(같은 날 2회 포함). 드라이버 595.84.

| 회차 | 시각 | 워크로드 | 사망 직전 온도/팬/전력 | 부하 후 생존 |
|---|---|---|---|---|
| 1차 | 2026-08-04 10:43 | `qat_recovery.py` BS=96 | 미측정 | — |
| 2차 | 2026-08-10 11:14 | `qat_recovery.py` 종료 직후 새 잡 | 74°C / 38% / 138W | ~40초 |
| 3차 | 2026-08-10 11:38 | `qat_recovery.py` **BS=48** ep1 step 500/833 | **78°C / 43% / 129.7W** | **~190초** |

**공통 시그니처:** `journalctl -k`에 **Xid 79 "GPU has fallen off the bus"** → 즉시 Xid 154
"Node Reboot Required". `lspci` `(rev ff)`, `nvidia-smi` "Unable to determine the device handle",
파이썬 `CUDA error: unspecified launch failure`. Xid 79 줄의 `name=nvidia-smi`는 텔레메트리가
우연히 GPU를 만진 것일 뿐 원인 아님.

**배치 축소는 무효한 레버였다(3차 실측 반증).** BS 96→48로 반감해도 최대 전력은 138W→129.7W
(**약 8W**)만 줄었다 — 배치가 작으면 커널을 더 자주 띄워 빈자리를 메우기 때문. 생존은 40→190초로
늘었지만 여전히 죽었다.

**진짜 시그널은 SW Power Cap이다.** 3차 텔레메트리 분포: `0x4`(SW Power Cap) 73행 / `0x0` 23행 /
`0x1` 2행. 부하 26초 뒤 상한에 걸려 죽을 때까지 164초 내내 ~129W에 붙어 있었다. RTX 3060 정격
TGP 170W인데 ~130W에서 걸리는 건 임베디드 섀시 전력 예산에 맞춘 OEM 설정 가능성. thermal 플래그
(`0x20/0x40/0x80`)는 3회 모두 0이나, 3차는 평균 76°C에 팬 39%로 **팬 커브가 온도에 반응 안 함** —
밀폐 박스의 별개 이상 신호(전원 레일/PCIe 접점/VRM 열화 후보).

**미시도 레버(3060 박스 되살릴 때):** `sudo nvidia-smi -pl <100 등>`(상한 자체를 내림, 배치와 달리
draw 직접 구속), 팬 하한 상향. 복구는 **재부팅뿐**(모듈 리로드 불가). 재부팅 전
`sudo nvidia-bug-report.sh`로 덤프 확보(벤더 문의 근거). `dmesg`는 권한 거부라 `journalctl -k` 사용.
원본: `~/stage1-work/telemetry_bs48.csv`, `qat_recovery_bs48.log`.

**교훈(현 머신에도 유효):** GPU 작업 전 `nvidia-smi -L` 생존 확인이 먼저다 — 죽어 있으면 ORT가
조용히 CPU로 폴백해 잘못된 수치를 뱉는다(판별법 [[stage1-quantization-hands-on]]).
