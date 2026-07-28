# 0. 환경 준비 — 보드 없어도 80%는 가능합니다

> 원본 가이드 매핑: "0. 환경 준비 — 보드 없어도 80%는 가능합니다"
> 예상 소요: 반나절 ~ 하루 (드라이버 설치 + 재부팅 + 데이터셋 다운로드 대기 포함)
> 선행 조건: Ubuntu 22.04 LTS 데스크톱, NVIDIA RTX 계열 dGPU, 인터넷, sudo 권한

---

## 0) 이 단계에서 무엇을·왜 하는가

임베디드 AI 배포(양자화 → 컴파일 → 정확도 검증 → latency 측정)에서 **실제로 타깃 보드가 꼭 필요한 작업은 의외로 적습니다.** 컴파일, INT8 캘리브레이션, 정확도 벤치마크 같은 "무거운 두뇌 노동"의 약 80%는 x86 PC(RTX GPU)에서 그대로 할 수 있습니다. 보드는 **실제 하드웨어 latency·전력·DLA/NPU 오프로드**를 측정하는 마지막 단계에서만 필수입니다.

| 타깃 | 보드 없이 x86에서 가능한 것 | 보드가 필요한 것 |
|------|----------------------------|-----------------|
| **NVIDIA Orin** | TensorRT 엔진 빌드, INT8 캘리브레이션 전 과정 (x86 dGPU에서 동일 워크플로) | DLA 오프로드, 실제 on-device latency/전력 |
| **TI TIDL** | x86에서 모델 컴파일 + host emulation 추론, 정확도 벤치마크 | 실제 C7x/MMA 성능·latency |
| **Qualcomm QNN** | x86_64에서 QDQ 양자화 (`get_qnn_qdq_config`) | HTP(Hexagon) 실행 → **Qualcomm AI Hub** 클라우드 디바이스로 대체 가능 |
| **Renesas DRP-AI** | DRP-AI TVM 컴파일, Interpreter mode로 양자화 ONNX 정확도 확인 | 실제 DRP-AI on-device 실행 |

> 💡 팁: 그래서 이 스터디는 **RTX GPU가 달린 Linux PC 1대 + Docker**만 있으면 대부분 완주할 수 있습니다. Jetson Orin 보드가 있으면 실측까지 한 번에 완결됩니다. 없어도 3단계(TensorRT)까지는 x86에서 그대로 따라올 수 있습니다.

이 단계의 목표는 딱 하나입니다: **"GPU가 컨테이너 안에서 보이고, PyTorch/ONNX Runtime/TensorRT가 돌고, nuScenes mini가 로드되는" 상태를 만드는 것.** 이후 모든 단계가 이 기반 위에서 돌아갑니다.

**이 문서를 끝냈을 때 손에 쥐게 되는 것** (7절 산출물과 연결):
- GPU를 인식하는 호스트 + 컨테이너
- 버전이 못 박힌 재현 가능한 Python 환경(`ENV.md`)
- 로드가 검증된 nuScenes mini

> 💡 팁: 이 단계는 **뒤 단계 전체의 재현성 기반**입니다. 여기서 "대충 되는 것 같다"로 넘어가면 3단계(TensorRT INT8)에서 `CUDAExecutionProvider`가 조용히 CPU로 fallback해 벤치마크 숫자가 통째로 틀어집니다. 각 검증 스니펫의 **예상 출력과 실제 출력을 눈으로 대조**하고 넘어가세요.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] NVIDIA 드라이버 설치 후 `nvidia-smi`가 GPU와 드라이버 버전 / 지원 CUDA 상한을 출력한다
- [ ] `nvidia-smi` 출력의 각 필드(Driver Version, CUDA Version, Persistence-M, Memory-Usage, GPU-Util, Processes)를 해석할 수 있다
- [ ] CUDA Toolkit(또는 컨테이너 내 CUDA)이 준비되어 `nvcc --version`이 동작하고, **드라이버가 보여주는 CUDA 상한과 `nvcc`의 Toolkit 버전 차이**를 설명할 수 있다
- [ ] Docker Engine + NVIDIA Container Toolkit 설치 후 `docker run --gpus all ... nvidia-smi`가 컨테이너 안에서 GPU를 보여준다
- [ ] Python 가상환경에 PyTorch(CUDA 12.8), ONNX, ONNX Runtime(GPU/CUDA 12), TensorRT 10.x LTS, polygraphy가 설치되고 import된다
- [ ] `torch.cuda.is_available()` → `True`, ONNX Runtime의 provider 목록에 `CUDAExecutionProvider`가 보이고, **실제로 CUDA로 세션이 만들어지는지**까지 확인했다
- [ ] nuScenes **mini(v1.0-mini)** 다운로드 + `nuscenes-devkit` 설치, `NuScenes(version='v1.0-mini', ...)` 로드 성공 (`scenes: 10 / samples: 404`)
- [ ] 각 벤더 SDK(TIDL/QNN/DRP-AI)는 **지금 설치하지 않고** [4단계](06_multi_soc.md)에서 다룬다는 것을 이해한다

---

## 2) 배경 이론 / 개념 — 버전 스택을 왜 신경 써야 하나

임베디드 AI에서 가장 흔한 삽질은 코드 버그가 아니라 **CUDA / cuDNN / 프레임워크 버전 불일치**입니다. 핵심 규칙 3가지만 기억하세요.

1. **드라이버 ≥ CUDA 런타임.** NVIDIA 드라이버는 자신보다 같거나 낮은 CUDA 런타임을 실행할 수 있습니다(minor-version forward compatibility). 그래서 드라이버는 넉넉히 최신으로, CUDA 런타임은 프레임워크가 요구하는 버전에 맞춥니다.
2. **`nvidia-smi`의 "CUDA Version"은 드라이버가 지원하는 상한선**이지, 실제 설치된 CUDA Toolkit 버전이 아닙니다. 실제 Toolkit 버전은 `nvcc --version`으로 확인합니다. (이 둘의 차이는 3-2절에서 실물로 대조합니다.)
3. **pip로 설치하는 PyTorch/ONNX Runtime/TensorRT wheel은 각자 자기 CUDA/cuDNN을 번들**합니다. 문제는 이들이 **서로 다른 CUDA major에 묶여 있으면** 충돌한다는 점입니다.

### 왜 굳이 "CUDA 12 라인"으로 통일하나 (직관)

CUDA는 **major 버전(12 ↔ 13)이 다르면 런타임 `.so`가 호환되지 않습니다.** 한 프로세스 안에 `libcudart.so.12`와 `libcudart.so.13`을 동시에 로드하면, 먼저 로드된 쪽이 이기고 나머지는 심볼 충돌·초기화 실패를 냅니다. 그런데 ONNX Runtime의 `CUDAExecutionProvider`는 초기화에 실패하면 **예외를 던지지 않고 조용히 CPU로 내려앉습니다.** 그래서 "돌긴 도는데 GPU를 안 쓰는" 최악의 디버깅 지옥이 열립니다. 이걸 원천 차단하려고, 이 가이드는 세 도구를 **전부 CUDA 12 위에서만** 고르게 맞춥니다.

> 🔴 함정 (2026-07 기준 매우 중요): 2026년 7월 현재 세 도구의 **PyPI 기본** CUDA 라인이 갈라져 있습니다.
> - PyTorch 최신 stable wheel: `cu128` = **CUDA 12.8** (이건 그대로 씀 — 우리 기준선)
> - `onnxruntime-gpu` PyPI **기본**: **1.27.0부터 CUDA 13.0으로 변경됨** → 그냥 `pip install onnxruntime-gpu` 하면 CUDA 13 wheel이 딸려와 CUDA 12 스택과 충돌
> - TensorRT PyPI **최신** `11.1.x`: **CUDA 13.x 기본** / LTS `10.16.x`: **CUDA 12.x** (우리는 LTS 10.x를 주경로로)
>
> 이걸 아무 생각 없이 pip로 한 venv에 섞으면 CUDA 12/13이 충돌해 `CUDAExecutionProvider`가 조용히 CPU로 fallback하거나 `import tensorrt`가 `.so` 로드 실패로 깨집니다.
>
> **그래서 이 가이드는 두 가지 안전한 경로를 제시합니다:**
> - **(A) CUDA 12 라인으로 버전을 명시 고정한 pip 설치** — onnxruntime-gpu는 CUDA 12 전용 인덱스, TensorRT는 `tensorrt-cu12==10.16.x`로 못 박음.
> - **(B) NGC 컨테이너(권장)** — 한 이미지 안에 CUDA/cuDNN/TensorRT/ORT가 이미 정합하게 맞춰져 있음.

### 정본 버전 스택 (이 스터디 전체 고정 — 변경 금지)

| 구성요소 | 고정 버전 | CUDA 라인 | 메모 |
|----------|-----------|-----------|------|
| CUDA Toolkit | **12.8** | 12 | PyTorch `cu128` 기준선에 맞춤 |
| PyTorch | `torch ... +cu128` | 12.8 | `pytorch.org/get-started`에서 최신 명령 확인 |
| onnxruntime-gpu | **1.28.0 (CUDA 12 wheel)** | 12 | PyPI 기본(13)이 아닌 **CUDA 12 전용 인덱스**로 설치 |
| TensorRT | **10.16.x LTS** | 12 | 10.x 주경로. 11.x는 참고만 |
| ExecuTorch | **1.3.x** | 12(호스트 빌드 시) | 이 단계에선 설치 안 함 — 온디바이스 단계에서 다룸 |
| polygraphy | 최신 | - | TensorRT 검증/디버깅 CLI |

> 💡 팁: **ExecuTorch 1.3.x**는 PyTorch edge 런타임으로, 이 스터디의 온디바이스/모바일 경로에서 등장합니다. 0단계(환경 준비)에서는 **설치하지 않습니다.** CUDA 스택과 무관한 별도 툴체인(보통 별도 venv)이라, 지금 섞으면 혼란만 커집니다. 여기서는 "정본 스택에 1.3.x로 고정되어 있다"는 사실만 기억하세요.

---

## 3) 환경·도구 준비

아래는 **베어메탈(호스트)에 직접 설치**하는 경로입니다. 컨테이너만 쓸 계획이면 3-1(드라이버) → 3-3(Docker+Toolkit)까지만 하고 3-2·3-4는 건너뛰어 4절의 "경로 B(NGC 컨테이너)"로 가도 됩니다.

> 🖥️ WSL2 사용자 먼저 읽기: Windows의 WSL2 Ubuntu에서 진행한다면 **3-1(드라이버 설치)과 3-2(호스트 CUDA 드라이버)는 건너뜁니다.** 드라이버는 Windows 쪽에만 설치하고, WSL 안에는 CUDA "Toolkit"만 깔면 됩니다. 자세한 주의점은 [3-5](#3-5-wsl2-사용-시-주의점-해당-시)를 먼저 보세요.

### 3-1. NVIDIA 드라이버 설치 (3가지 경로 비교)

드라이버 설치는 크게 세 갈래입니다. **대부분은 경로 ①로 충분**하고, 특정 버전이 필요하면 ②(APT 버전 고정) 또는 ③(PPA), 배포판 패키지가 막힌 특수 상황에서만 ④(.run)를 씁니다.

| 경로 | 명령 요지 | 장점 | 단점/주의 |
|------|-----------|------|-----------|
| **① `ubuntu-drivers`** (권장) | `sudo ubuntu-drivers autoinstall` 또는 `sudo ubuntu-drivers install` | 커널 헤더/DKMS/Secure Boot 서명까지 APT가 관리, 재부팅 후 커널 업데이트에도 자동 재빌드 | 배포판 저장소가 제공하는 브랜치까지만 |
| **② APT 버전 고정** | `sudo apt-get install -y nvidia-driver-<번호>` | ①의 장점을 유지하며 **특정 브랜치 고정** | 존재하는 브랜치 번호를 `ubuntu-drivers devices`로 먼저 확인해야 |
| **③ graphics-drivers PPA** | `sudo add-apt-repository ppa:graphics-drivers/ppa` 후 APT 설치 | 배포판보다 **최신/베타** 브랜치 | 안정성 낮을 수 있음, 시스템 불안정 리스크 |
| **④ `.run` 러너파일** | NVIDIA 사이트 `.run` 직접 실행 | 최신·특정 빌드 강제 | **APT가 관리하지 않음** → 커널 업데이트마다 수동 재설치, nouveau 수동 blacklist 필요, 유지보수 최악 |

> 💡 팁: `autoinstall`과 `install`은 22.04에서 사실상 동일하게 "recommended 브랜치 자동 설치"입니다. 다만 **Ubuntu 26.04부터 `autoinstall`이 폐기(deprecated)** 예정이라 앞으로를 생각하면 `install`을 습관화하는 편이 좋습니다. 22.04에서는 둘 다 동작합니다.

**경로 ① (권장) — `ubuntu-drivers`**

```bash
# 1) 시스템 최신화 + 빌드 도구 (DKMS 커널 모듈 빌드에 필요)
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential dkms

# 2) 이 PC에 권장되는 드라이버 확인 (recommended 라벨 확인)
ubuntu-drivers devices
#   출력 예: nvidia-driver-570 - distro non-free recommended

# 3) 권장 드라이버 자동 설치
sudo ubuntu-drivers install
#   (26.04부터는 install만 표준. 22.04는 autoinstall도 가능)

# 4) 재부팅 (커널 모듈 로드에 필수)
sudo reboot
```

**경로 ② — 특정 버전 APT 고정** (예: 문제 브랜치를 피하거나 CUDA 12.8 검증 조합에 맞출 때)

```bash
# ubuntu-drivers devices 로 존재하는 번호를 확인한 뒤:
sudo apt-get install -y nvidia-driver-570   # 번호는 devices 출력에 맞출 것
sudo reboot
```

**경로 ③ — graphics-drivers PPA** (배포판 저장소에 없는 최신 브랜치가 필요할 때만)

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
ubuntu-drivers devices          # PPA가 추가한 최신 브랜치가 목록에 나타남
sudo apt-get install -y nvidia-driver-<번호>
sudo reboot
```

> ⚠️ 주의: `-open` 접미사가 붙은 패키지(open kernel module)와 proprietary 버전이 함께 보일 수 있습니다. RTX 데스크톱에서 문제가 생기면 proprietary(비 `-open`) 브랜치로 되돌리세요. 두 변형을 섞어 깔지 말 것.

재부팅 후 검증합니다.

```bash
nvidia-smi
```

예상 출력(값은 환경마다 다름 — 필드 해석은 바로 아래 3-1-a):

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx.xx    Driver Version: 5xx.xx.xx    CUDA Version: 12.8                  |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off  | 00000000:01:00.0  On   |                  N/A |
|  0%   38C    P8              22W / 450W  |    412MiB / 24564MiB   |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|                                                                                         |
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                             GPU Memory  |
|=========================================================================================|
|    0   N/A  N/A      1523      G   /usr/lib/xorg/Xorg                            180MiB  |
+-----------------------------------------------------------------------------------------+
```

#### 3-1-a. `nvidia-smi` 출력 필드별 해석

이 표를 눈으로 읽을 수 있어야 이후 모든 트러블슈팅이 쉬워집니다.

| 필드 | 의미 | 자주 헷갈리는 점 |
|------|------|------------------|
| **NVIDIA-SMI** / **Driver Version** | 설치된 드라이버 버전 | 이게 실제로 깔린 커널 드라이버 버전 |
| **CUDA Version** (헤더 우측) | **이 드라이버가 지원하는 CUDA 런타임 상한** | ⚠️ 설치된 Toolkit 버전이 아님! `nvcc --version`과 다를 수 있고 다른 게 정상. 드라이버가 12.8까지 지원해도 실제 Toolkit은 12.8이 아닐 수 있음 |
| **Persistence-M** | Persistence Mode (Off/On) | Off면 유휴 시 드라이버 언로드로 첫 커널 실행이 수백 ms 느려질 수 있음. 벤치마크 전 `sudo nvidia-smi -pm 1`로 On 권장 |
| **Bus-Id** | PCI 버스 주소 (`0000:01:00.0`) | 멀티 GPU에서 `CUDA_VISIBLE_DEVICES`로 특정 GPU 지정 시 대조 |
| **Disp.A** | 이 GPU가 디스플레이 출력 중인지(On/Off) | On이면 그 GPU 메모리 일부를 데스크톱이 씀 |
| **Fan / Temp / Perf / Pwr:Usage/Cap** | 팬%/온도℃/P-State/전력(현재/최대) | P8=저전력 유휴, P0=최대 성능. 벤치마크 중 P0인지 확인 |
| **Memory-Usage** | 사용/총 VRAM (`412MiB / 24564MiB`) | 총량이 모델+배치가 들어갈지 판단 기준. RTX 4090=24GB |
| **GPU-Util** | 최근 샘플 구간의 GPU 코어 점유율% | 0%인데 학습이 도는 것 같으면 CPU fallback 의심 |
| **Compute M.** | Compute Mode (Default/Exclusive 등) | 보통 Default |
| **Processes** | GPU를 점유 중인 PID/이름/메모리 | 여기 내 파이썬 PID가 보이면 진짜 GPU를 쓰는 것. `Xorg`만 있으면 유휴 |

> 🔴 함정: "`nvidia-smi`가 CUDA 12.8이라는데 왜 PyTorch가 12.8을 요구해?"는 **잘못된 질문**입니다. 헤더의 CUDA Version은 드라이버 상한일 뿐, 프레임워크는 자기 번들 CUDA를 씁니다. 드라이버 상한 ≥ 프레임워크 CUDA면 됩니다(예: 드라이버 상한 12.8 ≥ torch cu128의 12.8, OK).

#### 3-1-b. 드라이버 설치 실패 복구 (Secure Boot / nouveau / 재부팅 후 실패)

재부팅했는데 `nvidia-smi`가 안 뜨는 3대 원인과 복구법입니다.

**증상 A: `nvidia-smi`가 "command not found" 또는 "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"**

```bash
# 1) 커널 모듈이 빌드/로드됐는지
sudo dkms status                 # nvidia/5xx: installed 가 보여야 함
lsmod | grep nvidia              # nvidia, nvidia_uvm 등이 로드돼야 함

# 2) 로드가 안 됐으면 로그 확인
sudo dmesg | grep -i -E "nvidia|nouveau"

# 3) 부팅 시 nouveau(오픈소스 기본 드라이버)가 nvidia를 밀어냈는지
lsmod | grep nouveau             # 출력이 있으면 nouveau가 살아있는 것 = 문제
```

**증상 B: Secure Boot가 서명 안 된 nvidia 모듈을 차단** — `dmesg`에 `nvidia: module verification failed` 또는 부팅 후 커널 모듈이 로드 안 됨.

```bash
# Secure Boot 상태 확인
mokutil --sb-state             # "SecureBoot enabled" 이면 서명 필요

# 해결 1 (권장): DKMS/APT가 만든 MOK 키를 등록
#   ubuntu-drivers/apt 설치 중 "Configuring Secure Boot" 화면에서 비밀번호를 설정했다면,
#   재부팅 시 파란 MOK Management 화면 → Enroll MOK → 그 비밀번호 입력으로 등록.
#   수동으로 키를 등록하려면:
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
#   → 비밀번호 설정 후 재부팅 → 파란 화면에서 Enroll MOK → 방금 비밀번호 입력

# 해결 2 (간단하지만 보안 낮춤): BIOS/UEFI에서 Secure Boot를 임시 비활성화 후 재부팅
```

**증상 C: nouveau가 아직 살아있어 nvidia가 로드 실패** — `.run` 설치나 일부 수동 설치에서 발생. APT/`ubuntu-drivers` 경로는 보통 자동 처리되지만, 수동으로 해야 할 때:

```bash
# 1) nouveau blacklist 파일 작성
sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF

# 2) initramfs 재생성 후 재부팅
sudo update-initramfs -u -k all
sudo reboot

# 3) 재부팅 후 nouveau가 죽었는지 확인 (출력 없으면 성공)
lsmod | grep nouveau
```

> ⚠️ 주의: `.run` 러너파일(경로 ④)로 깔았다면 **커널을 업데이트할 때마다 nvidia 모듈이 깨져** `nvidia-smi`가 다시 실패합니다. 이때는 `.run`을 다시 실행하거나(`sudo sh NVIDIA-Linux-*.run`), 아예 `.run` 드라이버를 제거(`sudo nvidia-uninstall`)하고 경로 ①(APT)로 재설치하는 게 장기적으로 편합니다.

### 3-2. CUDA Toolkit 설치 + 확인 (deb vs runfile)

호스트에서 직접 컴파일(`nvcc`)까지 하거나 일부 소스 빌드가 필요하면 CUDA Toolkit이 필요합니다. **컨테이너(경로 B)만 쓸 거면 이 절 전체를 건너뛰어도 됩니다** — NGC 이미지 안에 Toolkit이 들어 있습니다.

설치 방식은 두 갈래이며, 이 스터디는 **deb(network)** 를 권장합니다.

| 방식 | 특징 | 언제 |
|------|------|------|
| **deb (network)** (권장) | APT 저장소 등록 후 `apt install`. 의존성/업데이트를 APT가 관리 | 대부분의 경우. 버전 고정이 쉬움 |
| **deb (local)** | 큰 `.deb` 로컬 리포를 받아 설치. 인터넷 제한 환경에 적합 | 오프라인/에어갭 |
| **runfile** | 단일 `.run` 실행. **드라이버까지 같이 깔려는 경향** → 이미 있는 드라이버와 충돌 주의 | 특수 상황. 설치 시 드라이버 항목 체크 해제 권장 |

> ⚠️ 주의: `.run` runfile로 Toolkit을 깔 때 화면에서 **Driver 항목을 반드시 체크 해제**하세요. 3-1에서 APT로 깐 드라이버를 runfile이 덮어써 버전이 어긋나면 위의 "증상 A"가 재발합니다.

**deb(network) 설치 — CUDA 12.8 라인 고정** (정본 스택)

```bash
# NVIDIA CUDA APT 저장소 키링 설치 (Ubuntu 22.04 / x86_64)
#   최신 keyring 파일명/URL은 https://developer.nvidia.com/cuda-downloads 에서
#   Linux > x86_64 > Ubuntu > 22.04 > deb(network) 를 골라 재확인.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# CUDA 12.8 Toolkit만 고정 설치 (드라이버는 3-1에서 이미 설치했으므로 toolkit만)
sudo apt-get install -y cuda-toolkit-12-8
#   🔴 'sudo apt-get install -y cuda' 는 항상 최신(13.x)을 끌어와
#      PyTorch cu128(12.8)과 CUDA major가 어긋납니다. 반드시 'cuda-toolkit-12-8'로 고정.
#   💡 'cuda-toolkit-12-8'은 드라이버 메타패키지를 끌어오지 않아 3-1의 드라이버를 건드리지 않음.
```

PATH / LD_LIBRARY_PATH 등록 후 확인합니다. CUDA 12.8을 깔면 실제 경로는 `/usr/local/cuda-12.8`이고, `/usr/local/cuda`는 보통 그쪽을 가리키는 심볼릭 링크입니다.

```bash
# ~/.bashrc 에 추가 (심볼릭 링크 /usr/local/cuda 를 쓰면 버전 올려도 그대로 유효)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 심볼릭 링크가 12.8을 가리키는지 확인
ls -l /usr/local/cuda            # -> /usr/local/cuda-12.8 이어야 함

nvcc --version
```

`nvcc --version` 예상 출력:

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.8, V12.8.xx
Build cuda_12.8.r12.8/compiler.xxxxxxxx_0
```

> 💡 팁 (`nvcc` 버전 vs 런타임 버전의 차이 — 실물 대조):
> - `nvidia-smi` 헤더 CUDA Version = **드라이버가 지원하는 상한** (예: 12.8까지 OK)
> - `nvcc --version` release = **호스트에 설치된 CUDA Toolkit(컴파일러) 버전** (예: 12.8)
> - `python -c "import torch; print(torch.version.cuda)"` = **PyTorch wheel이 번들한 런타임 CUDA** (예: 12.8)
>
> 이 셋은 **서로 달라도 됩니다.** 규칙은 단 하나: *드라이버 상한 ≥ (nvcc 및 각 프레임워크의 CUDA)*. 예컨대 드라이버 상한 12.8, nvcc 12.8, torch cu128(12.8)이면 완벽 정합입니다. 만약 드라이버 상한이 12.6인데 torch가 cu128(12.8)이라면 12.8 런타임 심볼을 못 찾아 깨질 수 있으니 드라이버를 올리세요.

> ⚠️ 확인 필요: 2026-07 기준 CUDA APT 저장소의 **최신** 라인은 13.x(예: 13.3)로 확인됩니다. 이 스터디는 정본 스택대로 **12.8 라인(`cuda-toolkit-12-8`)** 을 고정합니다. CUDA 13.x는 신규 빌드에서 구형 아키텍처(Maxwell/Pascal/Volta) 타깃을 제외하므로, 구형 GPU 사용자는 [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)에서 GPU 호환성을 반드시 확인하세요.

### 3-3. Docker Engine + NVIDIA Container Toolkit

먼저 Docker Engine을 공식 저장소로 설치합니다.

```bash
# 1) Docker 공식 GPG 키 + 저장소 등록
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 2) Docker Engine 설치
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 3) sudo 없이 docker 쓰기 (재로그인 필요)
sudo usermod -aG docker $USER
newgrp docker   # 또는 로그아웃/로그인

# 4) 설치 확인
docker run --rm hello-world
```

이제 NVIDIA Container Toolkit을 설치해 컨테이너가 GPU를 볼 수 있게 합니다. (아래는 [공식 install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 기준)

```bash
# 1) NVIDIA Container Toolkit 저장소 등록 (stable/deb)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2) 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3) Docker 런타임에 nvidia 등록 + 재시작
#    이 명령이 /etc/docker/daemon.json 에 "nvidia" 런타임을 추가한다.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`nvidia-ctk runtime configure`가 실제로 무엇을 하는지 확인해 두면 디버깅이 쉽습니다.

```bash
# daemon.json 에 nvidia 런타임이 등록됐는지 확인
cat /etc/docker/daemon.json
#   예상: {"runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}}}
```

**컨테이너에서 GPU가 보이는지 검증**합니다 — 이 단계 전체에서 가장 중요한 확인입니다.

```bash
# CUDA 12.8 base 이미지로 컨테이너 안에서 nvidia-smi 실행
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

호스트에서 봤던 것과 동일한 GPU 표가 컨테이너 안에서 뜨면 성공입니다. (`--gpus all` 대신 특정 GPU만 노출하려면 `--gpus '"device=0"'`.)

> 🔴 함정: `--gpus all`에서 `could not select device driver "" with capabilities: [[gpu]]` 오류가 나면 3번(런타임 등록)이 안 된 것입니다. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`를 다시 실행하세요.

#### rootless Docker 주의점

`sudo` 없이 돌리는 **rootless Docker**(Docker를 사용자 권한으로 실행)에서는 런타임 설정 위치와 cgroup 처리가 다릅니다.

```bash
# 1) rootless는 사용자 daemon.json 을 대상으로 configure
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json

# 2) rootless에서는 cgroup 제어를 끄지 않으면 초기화 실패가 잦다
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place

# 3) 사용자 데몬 재시작
systemctl --user restart docker
```

> ⚠️ 주의: rootless에서 `nvidia-smi`가 컨테이너 안에서 실패하면 대개 위 2번(`no-cgroups`) 누락입니다. Podman을 쓴다면 런타임 대신 **CDI**(`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) 방식이 표준입니다.
> ⚠️ 확인 필요: 공식 install-guide 예시 기준 NVIDIA Container Toolkit 최신 버전 표기는 문서마다 갱신됩니다. 버전 고정이 필요하면 install-guide의 `NVIDIA_CONTAINER_TOOLKIT_VERSION` 방식을 참고하고, `nvidia/cuda:<tag>` 태그는 [Docker Hub nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)에 실재하는 태그로 맞추세요.

### 3-4. Python 가상환경 + 프레임워크 (경로 A: CUDA 12 라인 통일)

> 🔴 함정: 아래는 **CUDA 12 라인으로 버전을 명시 고정한** pip 경로입니다. 핵심은 onnxruntime-gpu를 **PyPI 기본(CUDA 13)이 아니라 CUDA 12 전용 인덱스**에서 받고, TensorRT를 **`tensorrt-cu12==10.16.x`(CUDA 12 명시 패키지)** 로 못 박는 것입니다. 최신 CUDA 13 라인을 섞고 싶으면 4절 "경로 B(NGC 컨테이너)"가 훨씬 안전합니다.

#### venv vs conda — 어느 걸 쓰나

| 기준 | `venv` (권장 기본) | `conda`/`mamba` |
|------|--------------------|-----------------|
| 설치 | OS python + pip만 | Miniforge/Miniconda 별도 |
| CUDA 런타임 | **wheel이 번들** (torch/ort/trt가 각자 가져옴) | conda가 `cudatoolkit`을 따로 제공할 수도 |
| 재현성 | `requirements.txt` + 정확한 버전 핀 | `environment.yml` |
| 이 스터디 권장 | ✅ 가볍고 wheel 스택과 궁합 좋음 | 비 CUDA 네이티브 의존성(예: 특정 geo/pcl)이 많을 때 |

> 💡 팁: 이 스터디의 스택은 **wheel이 CUDA를 번들**하므로 conda의 `cudatoolkit`이 필요 없습니다. 오히려 conda `cudatoolkit`과 wheel 번들 CUDA가 겹쳐 혼란을 줄 수 있어, 특별한 이유가 없으면 `venv`를 권장합니다.

```bash
# 1) venv 생성 (Ubuntu 22.04 기본 python은 3.10)
sudo apt-get install -y python3.10-venv
python3 -m venv ~/emb-ai
source ~/emb-ai/bin/activate
python -m pip install --upgrade pip

# 2) PyTorch (CUDA 12.8 wheel). 최신 명령은 https://pytorch.org/get-started/locally/ 에서 확인
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3) ONNX
pip install onnx

# 4) ONNX Runtime GPU — CUDA 12 전용 인덱스에서 1.28.0 고정
#    ⚠️ 그냥 'pip install onnxruntime-gpu' 하면 1.27+부터 PyPI 기본이 CUDA 13 wheel!
#    아래 CUDA-12 전용 Azure DevOps 피드에서 받아야 CUDA 12 스택과 정합.
pip install onnxruntime-gpu==1.28.0 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 5) TensorRT — CUDA 12 명시 패키지로 10.16.x LTS 고정
#    'tensorrt-cu12'는 CUDA 12용 메타패키지(런타임 라이브러리 포함).
pip install "tensorrt-cu12==10.16.1.11"

# 6) polygraphy (TensorRT 디버깅/검증 CLI)
pip install polygraphy
```

> 💡 팁: `onnxruntime-gpu`를 위 CUDA-12 인덱스에서 못 받는 상황(사내 프록시 등)이면, 임시로 **컨테이너(경로 B)** 로 전환하는 게 가장 빠릅니다. `--index-url`(우선 인덱스 전환)과 `--extra-index-url`(보조 인덱스 추가)은 동작이 다릅니다 — 위처럼 **`--index-url`로 CUDA-12 피드를 우선**시켜야 PyPI의 CUDA-13 기본 wheel이 끼어들지 않습니다.

> 🔴 함정: `tensorrt`(접미사 없는 메타패키지)는 시점에 따라 CUDA 13 라인(11.x)을 끌어올 수 있습니다. **CUDA 12 스택에서는 반드시 `tensorrt-cu12`** 를 쓰세요. 그래도 `.so` 로드가 안 되면 [TensorRT 다운로드](https://developer.nvidia.com/tensorrt)에서 CUDA 12용 tar를 받아 `LD_LIBRARY_PATH`에 수동 등록하는 편이 확실합니다.

**설치 검증** — 아래 스크립트가 전부 통과해야 합니다.

```python
# verify_env.py — 스택 정합성 한 번에 점검
import torch, onnx
import onnxruntime as ort

print("== PyTorch ==")
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("torch CUDA build:", torch.version.cuda)           # 12.8 이어야 함
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("== ONNX / ONNX Runtime ==")
print("onnx:", onnx.__version__)
print("onnxruntime:", ort.__version__)                   # 1.28.0
print("ORT providers:", ort.get_available_providers())   # CUDAExecutionProvider 있어야 함

print("== TensorRT ==")
import tensorrt as trt
print("TensorRT:", trt.__version__)                      # 10.16.1.11

# 세 도구의 CUDA 라인이 12로 정합인지 눈으로 확인
assert torch.version.cuda.startswith("12"), "torch가 CUDA 12 빌드가 아님"
assert "CUDAExecutionProvider" in ort.get_available_providers(), "ORT에 CUDA EP 없음"
print("\nOK: CUDA 12 스택 정합")
```

```bash
python verify_env.py
```

예상 출력(값은 GPU/마이너 버전에 따라 다름):

```text
== PyTorch ==
torch: 2.x.x+cu128 | CUDA available: True
torch CUDA build: 12.8
GPU: NVIDIA GeForce RTX 4090
== ONNX / ONNX Runtime ==
onnx: 1.x.x
onnxruntime: 1.28.0
ORT providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
== TensorRT ==
TensorRT: 10.16.1.11

OK: CUDA 12 스택 정합
```

`get_available_providers()`에 `CUDAExecutionProvider`가 **있다는 것만으로는 부족**합니다 — 실제로 그 provider로 세션이 만들어지는지까지 확인해야 조용한 CPU fallback을 잡습니다.

```python
# verify_cuda_ep.py — CUDAExecutionProvider가 '실제로' 잡히는지 확인
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# 1) 아주 작은 ONNX 모델(항등 Add) 하나를 메모리에 생성
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
node = helper.make_node("Identity", ["X"], ["Y"])
graph = helper.make_graph([node], "id", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
onnx.save(model, "tiny.onnx")

# 2) CUDA를 '요구'해서 세션 생성 (CPU로 조용히 안 내려가게 provider를 명시)
sess = ort.InferenceSession(
    "tiny.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("실제 활성 providers:", sess.get_providers())     # 첫 항목이 CUDA여야 함

out = sess.run(None, {"X": np.ones((1, 4), dtype=np.float32)})
print("추론 결과:", out[0])
assert sess.get_providers()[0] == "CUDAExecutionProvider", \
    "CUDA로 세션이 안 잡힘 → CPU fallback (CUDA/cuDNN major 불일치 의심)"
print("OK: CUDAExecutionProvider 실제 활성")
```

```bash
python verify_cuda_ep.py
```

예상 출력:

```text
실제 활성 providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
추론 결과: [[1. 1. 1. 1.]]
OK: CUDAExecutionProvider 실제 활성
```

> ⚠️ 주의: `sess.get_providers()`의 **첫 항목이 CPU**로 나오면, provider 목록엔 CUDA가 있어도 런타임 초기화에 실패해 CPU로 내려간 것입니다. 최근 ONNX Runtime은 이때 경고 로그를 남기기도 합니다. 원인은 대개 (1) ONNX Runtime의 CUDA/cuDNN major가 시스템과 불일치, (2) `LD_LIBRARY_PATH`에 CUDA 12 lib가 안 잡힘. 경로 A에서 안 풀리면 경로 B(NGC)로 전환이 가장 빠릅니다.

> ⚠️ 확인 필요: `tensorrt-cu12==10.16.1.11`은 2026-07 기준 PyPI에 존재하는 10.x LTS(CUDA 12) 릴리스로 확인했습니다. pip에서 못 찾으면 [TensorRT PyPI](https://pypi.org/project/tensorrt/) 또는 [TensorRT 다운로드](https://developer.nvidia.com/tensorrt)에서 CUDA 12 대응 tar/deb를 직접 받으세요.

### 3-5. WSL2 사용 시 주의점 (해당 시)

Windows + WSL2 Ubuntu 22.04에서 이 스터디를 진행할 수도 있습니다. 핵심 규칙 하나만 지키면 됩니다.

> 🔴 함정 (WSL2에서 가장 흔한 파괴적 실수): **WSL 안에 리눅스용 NVIDIA GPU 드라이버를 절대 설치하지 마세요.** WSL2에서 GPU는 Windows 호스트 드라이버가 담당하고, WSL 안에는 `/usr/lib/wsl/lib/`에 `libcuda.so.1`·`libnvidia-ml.so.1`·`nvidia-smi` 같은 **얇은 shim**이 자동 마운트되어 Windows 드라이버로 호출을 넘깁니다. 여기에 리눅스 드라이버를 깔면 이 shim을 덮어써 GPU 패스스루가 통째로 깨집니다.

- **설치할 것**: Windows에 최신 NVIDIA 드라이버(게임/스튜디오 드라이버 = WSL 지원 포함) + (WSL 안에는) **CUDA Toolkit만**. 이때도 `cuda-toolkit-12-8`처럼 **드라이버를 안 끌어오는 패키지**를 쓰세요(`cuda` 메타패키지 금지).
- **`nvidia-smi`**: WSL 안에서 실행되는 것은 `/usr/lib/wsl/lib/nvidia-smi`이며, 표시되는 Driver Version은 **Windows 드라이버 버전**입니다.
- **3-1(리눅스 드라이버)·3-1-b(nouveau/Secure Boot)** 는 WSL2에선 **건너뜁니다.**
- **Docker**: Windows의 Docker Desktop(WSL2 백엔드) 또는 WSL 내부 Docker Engine + Container Toolkit 모두 GPU 패스스루가 됩니다. 검증은 동일하게 `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.

> 💡 팁: WSL 안에서 `nvidia-smi`가 "Unable to determine the device handle" 등으로 실패하면, 대개 Windows 드라이버가 오래됐거나(WSL 지원 이전 버전) `wsl --update`로 WSL 커널을 갱신하지 않은 경우입니다. 자세한 규칙은 [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 참고.

---

## 4) 단계별 실습

### 경로 B (권장): NGC 컨테이너로 정합 스택 한 번에

버전 충돌을 원천 차단하는 가장 확실한 방법입니다. NVIDIA NGC의 PyTorch/TensorRT 컨테이너에는 CUDA·cuDNN·TensorRT·ONNX Runtime이 **정합하게 맞춰진 상태**로 들어 있습니다. 호스트에는 드라이버 + Docker + Container Toolkit(3-1, 3-3)만 있으면 됩니다.

```bash
# 1) NGC PyTorch 컨테이너 pull (TensorRT 포함). 태그는 YY.MM 형식.
#    최신/호환 태그는 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch 에서 확인
docker pull nvcr.io/nvidia/pytorch:26.06-py3

# 2) 작업 디렉토리와 데이터셋을 마운트해 컨테이너 진입
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $HOME/emb-ai-work:/workspace \
  -v /data/sets/nuscenes:/data/sets/nuscenes \
  nvcr.io/nvidia/pytorch:26.06-py3
```

컨테이너 안에서 검증합니다.

```bash
# (컨테이너 내부)
nvidia-smi
python -c "import torch, tensorrt; print(torch.__version__, torch.cuda.is_available(), tensorrt.__version__)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
pip install polygraphy nuscenes-devkit   # 컨테이너에 없으면 추가
```

> 💡 팁: NGC 태그(예: `26.06-py3`)마다 번들된 CUDA/TensorRT 조합이 정해져 있습니다. 어떤 TensorRT/CUDA가 들어있는지는 해당 태그의 [PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)에서 확인하세요. 이 방식이면 2절의 CUDA 12/13 충돌 문제를 신경 쓸 필요가 없습니다. 다만 **정본 스택(TensorRT 10.16.x LTS)과 태그의 번들 TensorRT가 다를 수 있으니**, 3단계에서 특정 TensorRT 버전이 필요하면 그 버전을 담은 태그를 고르거나 경로 A로 버전을 못 박으세요.

> ⚠️ 확인 필요: NGC PyTorch 태그 `26.06-py3`은 형식 예시입니다. 실제 존재하는 최신 태그는 위 NGC 카탈로그에서 확인 후 사용하세요.

### 4-1. nuScenes mini 데이터셋 다운로드

이 스터디의 예제 모델(BEV 인식 등)은 nuScenes를 씁니다. 전체(수백 GB) 대신 **mini(v1.0-mini, 10 scenes / 404 samples)** 만 받으면 정확도 검증 파이프라인을 돌리기에 충분합니다.

```bash
# 1) 데이터 디렉토리 생성 (devkit 관례: /data/sets/nuscenes)
sudo mkdir -p /data/sets/nuscenes
sudo chown -R $USER:$USER /data/sets/nuscenes
cd /data/sets/nuscenes

# 2) mini 아카이브 다운로드 후 해제
#    ※ 다운로드에는 nuScenes 계정 로그인/약관 동의가 필요할 수 있음(아래 주의 참고)
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf v1.0-mini.tgz     # samples/  sweeps/  maps/  v1.0-mini/  생성
```

**압축 해제 후 디렉토리 구조** — 아래처럼 4개 폴더가 나와야 정상입니다.

```text
/data/sets/nuscenes/
├── samples/        # 키프레임 센서 데이터 (CAM_FRONT, LIDAR_TOP, RADAR_* ...)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/         # 키프레임 사이 중간 프레임 (annotation 없음)
│   └── ...
├── maps/           # 맵: 래스터 .png + 벡터 .json
└── v1.0-mini/      # 메타데이터 JSON 테이블들 (아래)
    ├── scene.json          # 10개 scene
    ├── sample.json         # 404개 sample(키프레임)
    ├── sample_data.json    # 센서별 파일 경로
    ├── sample_annotation.json
    ├── calibrated_sensor.json
    ├── ego_pose.json
    ├── category.json / attribute.json / instance.json / sensor.json ...
    └── ...
```

- **samples vs sweeps**: `samples`는 어노테이션이 붙은 **키프레임**, `sweeps`는 그 사이 **중간 프레임**(어노테이션 없음). devkit이 동작하려면 최소한 메타데이터(`v1.0-mini/`)와 `samples/`가 있어야 하고, `sweeps`는 선택입니다.
- **maps**: 맵 폴더(래스터 `.png` + 벡터 `.json`). BEV 시각화에 사용됩니다.

> ⚠️ 주의: nuScenes 다운로드는 [nuscenes.org](https://www.nuscenes.org/nuscenes#download)에서 **계정 생성 + Terms of Use 동의** 후에만 가능한 경우가 많습니다. `wget`가 HTML 로그인 페이지를 받아오면(파일 크기가 몇 KB로 작으면 이 경우), 브라우저로 로그인해 `v1.0-mini.tgz`를 받은 뒤 이 디렉토리에 옮기세요.
> ⚠️ 확인 필요: `v1.0-mini.tgz`의 정확한 파일 크기는 공식 페이지에서 재확인하세요(대략 수 GB 규모). 다운로드 링크 형식 `https://www.nuscenes.org/data/v1.0-mini.tgz`는 devkit 문서/튜토리얼에서 참조되는 형태입니다.

### 4-2. nuscenes-devkit 설치 + 로드 확인

```bash
# venv/컨테이너 안에서
pip install nuscenes-devkit
```

```python
# load_nuscenes.py — mini 로드가 되는지만 확인
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini',
                dataroot='/data/sets/nuscenes',
                verbose=True)

print("scenes:", len(nusc.scene))      # 10 이어야 함
print("samples:", len(nusc.sample))    # 404 이어야 함
first = nusc.sample[0]
print("첫 sample 토큰:", first['token'])
print("센서 채널 예:", list(first['data'].keys())[:5])  # CAM_FRONT, LIDAR_TOP ...
```

```bash
python load_nuscenes.py
```

예상 출력:

```text
======
Loading NuScenes tables for version v1.0-mini...
Loading nuScenes-lidarseg... skipped
23 category, 8 attribute, ...
Done loading in X.XXX seconds.
======
scenes: 10
samples: 404
첫 sample 토큰: e93e98b63d3b40209056d129dc53ceee
센서 채널 예: ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'CAM_FRONT', 'LIDAR_TOP', ...]
```

여기까지 통과하면 **0단계 완료**입니다. `scenes: 10 / samples: 404`가 뜨면 데이터셋 배치가 올바른 것입니다.

**흔한 로드 오류 3가지 (증상 → 원인 → 해결)** — 자세한 표는 6절에도 있습니다:
- `AssertionError: Database version not found: /data/sets/nuscenes/v1.0-mini` → `dataroot`가 틀렸거나 `v1.0-mini/`가 해제 안 됨. `dataroot`는 **`v1.0-mini/`의 상위 폴더**여야 합니다(즉 `/data/sets/nuscenes`, 끝에 `/v1.0-mini` 붙이지 말 것).
- `FileNotFoundError: .../samples/CAM_FRONT/....jpg` → 메타데이터는 있는데 `samples/`(또는 이미지)가 없음. mini 아카이브가 완전히 풀렸는지 확인.
- `json.decoder.JSONDecodeError` → `.tgz`가 손상됐거나 절반만 받음(HTML 로그인 페이지를 받은 경우 포함). 다시 받아 `tar -xf`.

### 4-3. 벤더 SDK는 지금 설치하지 않습니다 (4단계 예고)

TI TIDL(`edgeai-tidl-tools`), Qualcomm QNN(ONNX Runtime QNN EP / AI Engine Direct SDK), Renesas DRP-AI TVM은 **각 SoC 전용 설치·환경변수·버전 매칭**이 필요해 지금 깔면 CUDA 스택과 섞여 혼란만 줍니다.

> 💡 팁: 이들은 [4단계 — 멀티 SoC](06_multi_soc.md)에서 **타깃별 격리된 Docker 이미지**로 각각 설치합니다. 지금은 "존재한다"는 것만 알고 넘어가세요. Qualcomm의 경우 실기 없이도 [Qualcomm AI Hub](https://aihub.qualcomm.com/)의 클라우드 디바이스로 HTP 실행을 대체할 수 있다는 점만 기억하면 됩니다.

---

## 5) 예시 / 결과 해석

**설치 경로 선택 요약** — 자신의 상황에 맞는 경로를 고르세요.

| 상황 | 권장 경로 | 이유 |
|------|-----------|------|
| 버전 충돌로 시간 낭비하기 싫다 | **경로 B (NGC 컨테이너)** | CUDA/cuDNN/TensorRT/ORT가 이미 정합 |
| 호스트에서 자유롭게 실험/디버깅하고 싶다 | 경로 A (pip, CUDA 12 통일) | IDE·프로파일러 붙이기 쉬움 |
| 정본 스택(TensorRT 10.16.x LTS)을 정확히 고정해야 한다 | 경로 A | 버전을 pip로 못 박음 |
| 최신 TensorRT 11.x / ORT CUDA 13 기능이 필요 | 경로 B (CUDA 13 태그) | pip로 CUDA 13 혼합은 깨지기 쉬움 |

**검증 신호 해석**

| 확인 명령 | 정상 신호 | 비정상 → 의심 지점 |
|-----------|-----------|-------------------|
| `nvidia-smi` (호스트) | GPU 표 + 드라이버/CUDA 상한 | 드라이버 미로딩 → 재부팅/Secure Boot/nouveau |
| `nvcc --version` | `release 12.8` | 다른 버전 → PATH가 다른 CUDA를 가리킴 |
| `docker run --gpus all ... nvidia-smi` | 컨테이너 안에서 동일 표 | Container Toolkit 미설정 → `nvidia-ctk runtime configure` |
| `torch.cuda.is_available()` | `True` | `False` → torch가 CPU wheel / 드라이버 상한 부족 |
| `ort.get_available_providers()` | `CUDAExecutionProvider` 포함 | 없으면 ORT가 CUDA 13 wheel일 가능성(인덱스 확인) |
| `sess.get_providers()[0]` | `CUDAExecutionProvider` | CPU면 조용한 fallback → CUDA/cuDNN 불일치 |
| `load_nuscenes.py` | `scenes: 10 / samples: 404` | 경로/압축해제 문제 |

**2026-07 기준 확인한 버전 스냅샷** (참고용 — 실제 설치 시점에 재확인):

| 구성요소 | 정본 스택(이 스터디 고정) | 2026-07 PyPI/APT 최신 | 비고 |
|----------|--------------------------|----------------------|------|
| CUDA Toolkit | **12.8** (고정) | 13.x (APT 최신) | 신규 빌드는 12.8 라인 유지 권장 |
| PyTorch wheel index | `cu128` = CUDA 12.8 | `cu128` | `pytorch.org/get-started` 확인 |
| onnxruntime-gpu | **1.28.0 (CUDA 12 wheel)** | 1.28.0 (PyPI 기본은 CUDA 13) | 1.27부터 PyPI 기본 CUDA=13 → CUDA-12 인덱스로 설치 |
| TensorRT | **10.16.x LTS (`tensorrt-cu12`)** | 11.1.x (CUDA 13 기본) | 10.x 주경로, 11.x는 참고 |
| ExecuTorch | **1.3.x** | - | 0단계에선 미설치 |
| NVIDIA Container Toolkit | 최신 stable | - | install-guide 기준 |

---

## 6) 흔한 오류와 해결 (Troubleshooting)

증상 → 원인 → 해결 순서로 정리했습니다. 실제로 마주치는 **에러 문구 그대로**를 기준으로 찾으세요.

| 증상(실제 문구) | 원인 | 해결 |
|-----------------|------|------|
| 재부팅 후 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | 드라이버 모듈 미로딩 (Secure Boot 차단 / nouveau 잔존 / DKMS 빌드 실패) | `mokutil --sb-state`로 Secure Boot 확인 → MOK 등록 또는 비활성화(3-1-b). `lsmod \| grep nouveau`로 nouveau 잔존 확인 후 blacklist. `sudo dkms status`로 빌드 확인 |
| `nvidia-smi: command not found` | 드라이버 미설치 또는 PATH 문제 | `ubuntu-drivers install` 재실행 후 재부팅 |
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit 런타임 미등록 | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| rootless docker에서 컨테이너 `nvidia-smi` 실패 | cgroup 제어 미비활성 | `sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place` 후 `systemctl --user restart docker` |
| `torch.cuda.is_available()` = `False`, `torch.version.cuda` = `None` | CPU 전용 torch가 깔림 | `--index-url https://download.pytorch.org/whl/cu128`로 재설치 |
| `torch.cuda.is_available()` = `False`, `torch.version.cuda` = `12.8` | 드라이버 상한 < 12.8, 또는 드라이버 미로딩 | 드라이버 업그레이드(3-1), `nvidia-smi` 먼저 통과시키기 |
| `ort.get_available_providers()`에 CUDA EP 없음 | onnxruntime-gpu가 **CUDA 13 wheel**(PyPI 기본)로 깔림 | CUDA-12 인덱스로 재설치: `pip install onnxruntime-gpu==1.28.0 --index-url .../onnxruntime-cuda-12/pypi/simple/` |
| CUDA EP는 목록에 있는데 `sess.get_providers()[0]`이 CPU | 런타임 초기화 실패(cuDNN major 불일치, lib 경로 문제) | `LD_LIBRARY_PATH`에 CUDA 12 lib 확인, 안 되면 경로 B(NGC)로 전환 |
| `import tensorrt` 시 `ImportError: libnvinfer.so.10: cannot open shared object file` | TensorRT wheel의 CUDA와 로컬 CUDA 불일치, 또는 lib 미탑재 | `tensorrt-cu12==10.16.1.11`(CUDA 12 명시)로 재설치, 또는 [TensorRT 다운로드](https://developer.nvidia.com/tensorrt) tar를 `LD_LIBRARY_PATH`에 등록 |
| NGC 컨테이너에서 DataLoader가 `Bus error`/`DataLoader worker ... killed` | 공유 메모리 부족 | `docker run`에 `--shm-size=8g` 추가 |
| `apt` 설치 중 `NO_PUBKEY`(NVIDIA 저장소) | GPG 키 미등록/만료 | 위 keyring 등록 단계 재실행(구 `apt-key` 대신 keyring 방식) |
| `AssertionError: Database version not found: .../v1.0-mini` | `dataroot`가 틀리거나 `v1.0-mini/` 미해제 | `dataroot`는 `v1.0-mini/`의 **상위** 폴더(`/data/sets/nuscenes`). 끝에 `/v1.0-mini` 붙이지 말 것 |
| `NuScenes(...)`에서 `FileNotFoundError: .../samples/...` | 메타데이터는 있으나 `samples/` 미해제 | mini `.tgz`가 완전히 풀렸는지, `samples/ sweeps/ maps/ v1.0-mini/` 4개가 있는지 확인 |
| `wget v1.0-mini.tgz`가 몇 KB HTML을 받음 | 로그인/약관 동의 필요 | 브라우저로 로그인 후 다운로드해 수동 배치 |
| WSL2에서 `nvidia-smi`가 `Unable to determine the device handle for GPU` | Windows 드라이버 구버전 또는 WSL 미갱신, 혹은 WSL에 리눅스 드라이버를 깔아 shim 손상 | Windows 드라이버 최신화 + `wsl --update`. WSL 내부 리눅스 드라이버는 제거(3-5) |

---

## 7) 산출물 (Deliverables)

이 단계가 끝나면 아래가 남아야 합니다.

- [ ] `nvidia-smi` / 컨테이너 `nvidia-smi` 출력 캡처 (GPU 인식 증빙)
- [ ] `verify_env.py` 실행 로그 — torch/ORT/TensorRT 버전 + `CUDAExecutionProvider` 확인
- [ ] `verify_cuda_ep.py` 실행 로그 — `sess.get_providers()[0] == CUDAExecutionProvider` 확인(조용한 CPU fallback 없음 증빙)
- [ ] `/data/sets/nuscenes/` 에 배치된 nuScenes **mini** (폴더: `samples/ sweeps/ maps/ v1.0-mini/`)
- [ ] `load_nuscenes.py` 실행 로그 — `scenes: 10 / samples: 404`
- [ ] 선택한 경로(A: pip CUDA 12 / B: NGC 컨테이너)와 사용한 **정확한 버전**을 적은 `ENV.md` 메모 (다음 단계 재현성용). 최소 항목: 드라이버 버전, `nvcc` 버전, torch/onnxruntime-gpu/tensorrt 버전, (컨테이너면) NGC 태그.

> 💡 팁: `ENV.md`는 뒤 단계에서 "내 숫자가 왜 가이드와 다르지?"를 추적할 때 결정적입니다. 아래 명령으로 자동 수집해 붙여두면 편합니다.
>
> ```bash
> {
>   echo '## ENV snapshot'; date
>   nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
>   nvcc --version | grep release
>   python -c "import torch,onnxruntime,tensorrt as t; print('torch',torch.__version__);\
> print('ort',onnxruntime.__version__);print('trt',t.__version__)"
> } > ENV.md
> cat ENV.md
> ```

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) — OS/아키텍처별 CUDA Toolkit 설치 명령 생성기(deb/runfile)
- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) — 공식 Linux 설치 가이드
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) — WSL2에서 리눅스 드라이버 금지 등 필수 규칙
- [NVIDIA Container Toolkit — Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — apt 저장소/`nvidia-ctk`/rootless 설정
- [Docker Engine 설치 (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/) — Docker 공식 저장소 설치
- [PyTorch — Get Started Locally](https://pytorch.org/get-started/locally/) — CUDA 버전별 pip 설치 명령(`cu128`)
- [ONNX Runtime — Install](https://onnxruntime.ai/docs/install/) / [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) — CUDA 12/13 빌드 구분과 CUDA-12 전용 인덱스
- [TensorRT PyPI](https://pypi.org/project/tensorrt/) / [TensorRT 다운로드](https://developer.nvidia.com/tensorrt) — 버전·릴리스 노트(10.x LTS / `tensorrt-cu12`)
- [Polygraphy (TensorRT tools)](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) — ONNX/TensorRT 검증·디버깅 CLI
- [NGC PyTorch 컨테이너](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) / [PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/) — 정합 스택 컨테이너(권장 경로 B)와 번들 버전 확인
- [nuScenes 다운로드](https://www.nuscenes.org/nuscenes#download) / [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) — 데이터셋과 devkit
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 실기 없이 클라우드 디바이스로 HTP 실행(4단계 예고)

### 논문
- Caesar et al. (2020), *nuScenes: A Multimodal Dataset for Autonomous Driving*, arXiv:1903.11027 — 이 스터디 데이터셋의 원 논문

---

## 9) 다음 단계

환경이 준비됐으면, 어떤 타깃/툴체인부터 어떤 순서로 오를지 "난이도 사다리"를 먼저 잡습니다.

➡️ [0.5단계 — 배포 난이도 사다리](02_deployment_ladder.md)
