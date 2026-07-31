# 0. 환경 준비 — 보드 없어도 80%는 가능합니다

> 원본 가이드 매핑: "0. 환경 준비 — 보드 없어도 80%는 가능합니다"
> 예상 소요: 반나절 ~ 하루 (드라이버 설치 + 재부팅 + 데이터셋 다운로드 대기 포함)
> 선행 조건: Ubuntu 22.04 LTS 데스크톱, NVIDIA RTX 계열 dGPU, 인터넷, sudo 권한

> ✅ **이 스택은 2026-07-31 실제 머신에서 설치·검증되었습니다.**
> 검증 환경: Ubuntu 22.04.5 LTS / **GeForce RTX 3060 (12GB)** / 드라이버 **595.84** / `nvcc` **12.8.93** / Docker 29.6.2 + Container Toolkit 1.19.1 / Python 3.10.12 venv.
> 아래 4개 검증 스크립트(`nvidia-smi` → `verify_env.py` → `verify_cuda_ep.py` → `load_nuscenes.py`)가 전부 통과한 스택을 그대로 적었습니다. 예상 출력 블록의 값도 이 머신의 **실측 출력**입니다(GPU명·버전 마이너는 환경마다 다를 수 있음).

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
- [ ] Python 가상환경에 PyTorch(CUDA 12.8), `onnx==1.18.0`, ONNX Runtime(GPU/CUDA 12), TensorRT 10.x LTS, polygraphy, `onnxscript`가 설치되고 import된다
- [ ] **`onnxscript`가 왜 0단계에 필요한지** 설명할 수 있다 (torch 2.11의 `torch.onnx.export`는 기본 `dynamo=True` → 없으면 3·4·5단계 export가 시작조차 안 됨). `torch.onnx.export`가 실제로 파일을 뽑는 것까지 확인했다
- [ ] **`onnx`를 왜 1.18.0으로 고정하는지** 설명할 수 있다 (ONNX **IR 버전 상한** — 2절 참고). 이걸 어기면 3·4단계 ONNX export가 로드 단계에서 터진다
- [ ] **cuDNN 9가 CUDA Toolkit에 포함되지 않는다**는 것을 알고, venv 번들 `nvidia/*/lib`와 TensorRT `tensorrt_libs`를 `LD_LIBRARY_PATH`에 노출했다 (3-4-a)
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
> - TensorRT PyPI **최신** `11.2.x`: **CUDA 13.x 기본** / LTS `10.16.x`: **CUDA 12.x** (우리는 LTS 10.x를 주경로로)
>
> 이걸 아무 생각 없이 pip로 한 venv에 섞으면 CUDA 12/13이 충돌해 `CUDAExecutionProvider`가 조용히 CPU로 fallback하거나 `import tensorrt`가 `.so` 로드 실패로 깨집니다.
>
> **그래서 이 가이드는 두 가지 안전한 경로를 제시합니다:**
> - **(A) CUDA 12 라인으로 버전을 명시 고정한 pip 설치** — onnxruntime-gpu는 **버전 상한 `<1.27`** 로, TensorRT는 `tensorrt-cu12==10.16.x`로 못 박음.
> - **(B) NGC 컨테이너** — 한 이미지 안에 CUDA/cuDNN/TensorRT/ORT가 이미 정합하게 맞춰져 있음.

`onnxruntime-gpu`의 CUDA 라인은 PyPI 메타데이터로 직접 확인할 수 있습니다(2026-07 실측):

| onnxruntime-gpu | nvidia 의존성 | CUDA 라인 | Python 3.10 wheel |
|-----------------|---------------|-----------|-------------------|
| ~ 1.23.2 | `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12` | **12** | ✅ **있음 (마지막)** |
| 1.24.1 ~ 1.26.0 | `...-cu12` | 12 | ❌ 없음 (cp311+ 만) |
| **1.27.0 이상** | `nvidia-cuda-runtime-cu13`, `nvidia-cudnn-cu13` | **13** | ❌ 없음 |

> 💡 팁: 그래서 **Ubuntu 22.04 기본 Python 3.10**에서는 `pip install "onnxruntime-gpu<1.27"`이 **1.23.2**로 해석됩니다. 상한 `<1.27`이 CUDA 13을 막고, Python 3.10이 다시 1.23.2로 묶기 때문입니다(1.24.1부터 cp310 wheel이 사라짐). Python 3.11+ venv라면 같은 명령이 1.26.0을 받습니다 — 그래도 CUDA 12라 정합은 유지됩니다.

### 또 하나의 상한: ONNX **IR 버전** (3·4단계에서 바로 터지는 함정)

CUDA 라인만 맞추면 끝이 아닙니다. **ONNX Runtime은 자기가 빌드될 때 링크한 `onnx` 버전의 IR(Intermediate Representation) 버전까지만 모델을 읽을 수 있습니다.** 그보다 높은 IR로 저장된 모델은 **예외를 던지며 로드 자체가 실패**합니다(이건 조용한 fallback이 아니라 명시적 에러).

ONNX Runtime 소스에서 이 검사는 이렇게 생겼습니다 (`onnxruntime/core/graph/model.cc`, rel-1.23.2):

```cpp
if (const auto ir_version = model_proto.ir_version();
    ir_version > ONNX_NAMESPACE::Version::IR_VERSION) {
  ORT_THROW("Unsupported model IR version: ", ir_version,
            ", max supported IR version: ", ONNX_NAMESPACE::Version::IR_VERSION);
}
```

즉 상한은 **ORT가 빌드에 사용한 onnx 버전**이 결정합니다. ORT `rel-1.23.2`의 `cmake/deps.txt`는 `onnx v1.18.0`을 고정하고 있고, **onnx 1.18.0의 IR_VERSION은 11**입니다. 그래서 ORT 1.23.2의 상한은 **IR 11**입니다.

문제는 pip로 `onnx`를 무제한 설치하면 훨씬 높은 IR을 쓰는 버전이 깔린다는 점입니다:

| onnx 버전 | IR_VERSION | 최대 opset | ORT 1.23.2에서 |
|-----------|-----------|-----------|----------------|
| **1.18.0** | **11** | 23 | ✅ **정합 (이 스터디 정본)** |
| 1.19.x | 12 | 24 | ❌ 로드 실패 |
| 1.20 ~ 1.21 | 13 | 25~26 | ❌ 로드 실패 |
| 1.22.0 (2026-07 PyPI 최신) | 13 | 27 | ❌ 로드 실패 |

> 🔴 함정: `pip install onnx`(버전 무제한)를 하면 최신 **1.22.0(IR 13)** 이 깔립니다. 그러면 3단계에서 `torch.onnx.export`로 뽑은 모델을 ORT로 열 때 이런 에러가 납니다:
> ```text
> onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
> Load model from model.onnx failed:/onnxruntime_src/onnxruntime/core/graph/model.cc:181
> onnxruntime::Model::Model(...) Unsupported model IR version: 13, max supported IR version: 11
> ```
> (위 문구는 실측 재현한 것입니다. 예외 클래스는 `InvalidGraph`가 아니라 **`Fail`** 이고, 핵심 단서는 마지막 줄의 `Unsupported model IR version:` 입니다 — 이 문자열로 검색하세요.)
> **ORT가 상한이므로 `onnx` 쪽을 내려서 맞춥니다 → `onnx==1.18.0`으로 고정.** (반대로 onnx를 최신으로 두고 ORT를 올리려면 CUDA 13 라인으로 넘어가야 해서 스택 전체가 흔들립니다.)
>
> **[3단계](05_tensorrt.md)·[4단계](06_multi_soc.md) 예고:** 이 상한 때문에 ONNX export 시 **opset ≤ 23**을 쓰고, 저장된 모델의 `ir_version`이 11을 넘지 않는지 확인해야 합니다. export 직후 아래 한 줄로 점검하는 습관을 들이세요.
> ```bash
> python -c "import onnx,sys; m=onnx.load(sys.argv[1]); print('ir_version:', m.ir_version, '| opset:', [(i.domain or 'ai.onnx', i.version) for i in m.opset_import])" model.onnx
> ```

### cuDNN은 CUDA Toolkit에 들어있지 않습니다

마지막 함정이자 이 단계에서 **가장 값비쌌던** 지점입니다. `CUDAExecutionProvider`는 `libcudnn.so.9`를 필요로 하는데:

- **cuDNN 9는 `cuda-toolkit-12-8` deb에 포함되지 않습니다.** 3-2절을 다 해도 `/usr/local/cuda/lib64/libcudnn.so.9`는 존재하지 않습니다(실측 확인).
- 대신 이 스택에서는 **PyTorch가 끌어온 `nvidia-cudnn-cu12` 패키지**가 venv 안(`.../site-packages/nvidia/cudnn/lib/libcudnn.so.9`)에 들고 있습니다.
- 그런데 그 경로는 동적 로더(`ld.so`) 검색 경로에 **없습니다.** 그래서 ORT가 cuDNN을 못 찾고 → CUDA EP 초기화 실패 → **조용히 CPU로 fallback**합니다.

해결은 NGC로 도망갈 필요 없이 **venv 번들 라이브러리 경로를 `LD_LIBRARY_PATH`에 노출**하면 끝입니다. 구체적 방법은 [3-4-a](#3-4-a-cudnn-등-venv-번들-nvidia-라이브러리를-ld_library_path에-노출-필수)에서 다룹니다. (동일한 함정이 TensorRT의 `libnvinfer.so.10`에도 있습니다 — `tensorrt_libs` 패키지 디렉터리가 `nvidia/*/lib` 글롭 밖에 있어서 똑같이 못 찾힙니다. 3-4-a에서 함께 해결합니다.)

### 정본 버전 스택 (이 스터디 전체 고정 — 변경 금지)

아래는 2026-07-31 실제 머신(Ubuntu 22.04.5 / RTX 3060 / Python 3.10.12)에서 **설치 후 4개 검증 스크립트를 전부 통과한** 조합입니다.

| 구성요소 | 고정 버전 | CUDA 라인 | 메모 |
|----------|-----------|-----------|------|
| CUDA Toolkit | **12.8** (`nvcc` 12.8.93) | 12 | PyTorch `cu128` 기준선에 맞춤 |
| PyTorch | **`torch 2.11.0+cu128`** (`torchvision 0.26.0`, `torchaudio 2.11.0`) | 12.8 | `pytorch.org/get-started`에서 최신 명령 확인 |
| **onnx** | **`1.18.0` (IR 11)** | - | 🔴 **반드시 고정.** ORT 1.23.2의 IR 상한이 11 (위 "ONNX IR 버전" 절) |
| onnxruntime-gpu | **`1.23.2` (CUDA 12 wheel)** | 12 | `pip install "onnxruntime-gpu<1.27"` — 1.27+는 PyPI 기본이 CUDA 13. Python 3.10의 마지막 wheel |
| TensorRT | **`tensorrt-cu12==10.16.1.11`** (10.x LTS) | 12 | 10.x 주경로. 11.x는 CUDA 13이라 참고만 |
| **numpy** | **`1.26.4` (`numpy<2`)** | - | `nuscenes-devkit 1.2.0`이 `numpy<2.0.0`을 요구 (3-4절) |
| polygraphy | **`0.50.3`** | - | TensorRT 검증/디버깅 CLI |
| **onnxscript** | **`0.7.1`** (+ `onnx-ir 0.2.1`) | - | 🔴 **ONNX export 필수.** torch 2.11의 `torch.onnx.export`는 기본이 `dynamo=True`이고 그 경로가 이걸 요구 → 없으면 3·4·5단계 export가 첫 줄에서 죽음 |
| nuscenes-devkit | **`1.2.0`** | - | mini 로드용. numpy 상한의 원인 |
| ExecuTorch | **1.3.x** | 12(호스트 빌드 시) | 이 단계에선 설치 안 함 — 온디바이스 단계에서 다룸 |

> 💡 팁: **ExecuTorch 1.3.x**는 PyTorch edge 런타임으로, 이 스터디의 온디바이스/모바일 경로에서 등장합니다. 0단계(환경 준비)에서는 **설치하지 않습니다.** CUDA 스택과 무관한 별도 툴체인(보통 별도 venv)이라, 지금 섞으면 혼란만 커집니다. 여기서는 "정본 스택에 1.3.x로 고정되어 있다"는 사실만 기억하세요.

---

## 3) 환경·도구 준비

아래는 **베어메탈(호스트)에 직접 설치**하는 경로입니다. 컨테이너만 쓸 계획이면 3-1(드라이버) → 3-3(Docker+Toolkit)까지만 하고 3-2·3-4는 건너뛰어 4절의 "경로 B(NGC 컨테이너)"로 가도 됩니다.

> 🖥️ WSL2 사용자 먼저 읽기: Windows의 WSL2 Ubuntu에서 진행한다면 **3-1(드라이버 설치)과 3-2(호스트 CUDA 드라이버)는 건너뜁니다.** 드라이버는 Windows 쪽에만 설치하고, WSL 안에는 CUDA "Toolkit"만 깔면 됩니다. 자세한 주의점은 [3-5](#3-5-wsl2-사용-시-주의점-해당-시)를 먼저 보세요.

### 3-1. NVIDIA 드라이버 설치 (3가지 경로 비교)

> ⏭️ **먼저 확인: 이 절을 건너뛸 수 있습니다.** 드라이버가 이미 깔려 있다면 새로 설치할 이유가 없습니다. 아래를 실행해 보세요.
>
> ```bash
> # 드라이버 버전과 '드라이버가 지원하는 CUDA 상한'을 한 줄로 확인
> nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
> nvidia-smi | grep -o "CUDA Version: [0-9.]*"
> ```
>
> 이 명령이 정상 동작하고 **CUDA 상한이 12.8 이상**이면 (예: `CUDA Version: 13.2`) → **3-1과 재부팅을 모두 건너뛰고 [3-2](#3-2-cuda-toolkit-설치--확인-deb-vs-runfile)로 가세요.** 2절의 규칙 그대로 *드라이버 상한 ≥ 프레임워크 CUDA*만 만족하면 되므로, 드라이버를 굳이 최신으로 다시 깔 필요가 없습니다. 오히려 잘 돌던 드라이버를 건드리면 3-1-b의 복구 작업을 자초합니다.
> *실측 예: 이 가이드를 검증한 머신은 드라이버 595.84 / CUDA 상한 13.2가 이미 설치돼 있어 3-1과 재부팅을 생략했습니다.*
>
> `nvidia-smi`가 실패하거나 CUDA 상한이 12.8 미만이면 아래를 진행하세요.

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

예상 출력 — 아래는 2026-07-31 실측 출력입니다(값은 환경마다 다름 — 필드 해석은 바로 아래 3-1-a):

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.84                 Driver Version: 595.84         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:01:00.0  On |                  N/A |
| 30%   35C    P8             15W /  170W |     116MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|                                                                              Usage      |
|=========================================================================================|
|    0   N/A  N/A            1052      G   /usr/lib/xorg/Xorg                       66MiB |
|    0   N/A  N/A            1278      G   /usr/bin/gnome-shell                     22MiB |
+-----------------------------------------------------------------------------------------+
```

> 💡 팁: 이 실측 출력이 2절의 규칙을 그대로 보여줍니다 — 헤더의 **`CUDA Version: 13.2`는 드라이버가 지원하는 상한**이고, 같은 머신의 `nvcc`는 **12.8.93**입니다. 둘이 다른 게 정상이며, *상한(13.2) ≥ 우리 스택(12.8)* 이므로 정합입니다. 이 머신은 드라이버가 이미 충분해서 3-1을 건너뛴 케이스입니다.

#### 3-1-a. `nvidia-smi` 출력 필드별 해석

이 표를 눈으로 읽을 수 있어야 이후 모든 트러블슈팅이 쉬워집니다.

| 필드 | 의미 | 자주 헷갈리는 점 |
|------|------|------------------|
| **NVIDIA-SMI** / **Driver Version** | 설치된 드라이버 버전 | 이게 실제로 깔린 커널 드라이버 버전 |
| **CUDA Version** (헤더 우측) | **이 드라이버가 지원하는 CUDA 런타임 상한** | ⚠️ 설치된 Toolkit 버전이 아님! `nvcc --version`과 다를 수 있고 **다른 게 정상**. 위 실측이 그 예 — 헤더는 `13.2`(상한)인데 `nvcc`는 `12.8.93`(실제 Toolkit) |
| **Persistence-M** | Persistence Mode (Off/On) | Off면 유휴 시 드라이버 언로드로 첫 커널 실행이 수백 ms 느려질 수 있음. 벤치마크 전 `sudo nvidia-smi -pm 1`로 On 권장 |
| **Bus-Id** | PCI 버스 주소 (`0000:01:00.0`) | 멀티 GPU에서 `CUDA_VISIBLE_DEVICES`로 특정 GPU 지정 시 대조 |
| **Disp.A** | 이 GPU가 디스플레이 출력 중인지(On/Off) | On이면 그 GPU 메모리 일부를 데스크톱이 씀 |
| **Fan / Temp / Perf / Pwr:Usage/Cap** | 팬%/온도℃/P-State/전력(현재/최대) | P8=저전력 유휴, P0=최대 성능. 벤치마크 중 P0인지 확인 |
| **Memory-Usage** | 사용/총 VRAM (실측 `116MiB / 12288MiB`) | 총량이 모델+배치가 들어갈지 판단 기준. RTX 3060=12GB, RTX 4090=24GB |
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
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

(위는 2026-07-31 실측 출력입니다. 중요한 것은 `release 12.8` — 패치 번호 `V12.8.xx`는 설치 시점에 따라 다를 수 있습니다.)

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

> 🔴 함정: 아래는 **CUDA 12 라인으로 버전을 명시 고정한** pip 경로입니다. 네 가지를 동시에 못 박아야 합니다 — ① onnxruntime-gpu는 **`<1.27`**(1.27+는 PyPI 기본이 CUDA 13), ② TensorRT는 **`tensorrt-cu12==10.16.x`**(CUDA 12 명시 패키지), ③ **`onnx==1.18.0`**(ORT의 IR 11 상한), ④ **`numpy<2`**(nuscenes-devkit 요구). 최신 CUDA 13 라인을 섞고 싶으면 4절 "경로 B(NGC 컨테이너)"가 훨씬 안전합니다.

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

# 2) numpy를 '먼저' 2.x 미만으로 고정
#    이유: nuscenes-devkit 1.2.0이 'numpy<2.0.0'을 요구한다(4-2절).
#    나중에 깔면 numpy가 2.x → 1.26.x로 강제 다운그레이드되면서
#    이미 설치된 패키지들이 재컴파일/ABI 경고를 뿜는다. 처음부터 못 박는 게 깔끔하다.
pip install "numpy<2"

# 3) PyTorch (CUDA 12.8 wheel). 최신 명령은 https://pytorch.org/get-started/locally/ 에서 확인
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4) ONNX — 반드시 1.18.0으로 고정 (IR 11)
#    🔴 'pip install onnx'(무제한)는 1.22.0(IR 13)을 깔고, ORT 1.23.2는 IR 11까지만 읽는다.
#       → 3·4단계 ONNX export가 "Unsupported model IR version: 13" 으로 로드 실패. (2절 참고)
pip install "onnx==1.18.0"

# 5) ONNX Runtime GPU — CUDA 12 라인 유지를 위해 버전 상한만 걸어 PyPI에서 설치
#    ⚠️ 1.27부터 PyPI 기본이 CUDA 13이므로 상한 '<1.27'로 CUDA 12 라인에 묶는다.
#    Python 3.10(Ubuntu 22.04 기본)에서는 이게 1.23.2로 해석된다 — cp310 wheel의 마지막 버전.
#    (참고: cp310에는 1.24+ wheel이 아예 없어서 상한을 빼도 지금은 1.23.2가 깔린다.
#     그래도 상한을 명시하는 이유는 Python 3.11+ venv나 향후 wheel 추가 시 조용히
#     CUDA 13으로 넘어가는 걸 막기 위한 것이다. 재현성 핀은 남겨두는 편이 안전하다.)
pip install "onnxruntime-gpu<1.27"

# 6) TensorRT — CUDA 12 명시 패키지로 10.16.x LTS 고정
#    'tensorrt-cu12'는 CUDA 12용 메타패키지(런타임 라이브러리 포함).
pip install "tensorrt-cu12==10.16.1.11"

# 7) polygraphy (TensorRT 디버깅/검증 CLI)
pip install polygraphy

# 8) onnxscript — ONNX export의 필수 의존성 (3·4·5단계에서 반드시 필요)
#    🔴 torch 2.11의 torch.onnx.export는 기본이 dynamo=True이고, 그 경로가 onnxscript를 요구한다.
#       없으면 3단계 첫 export가 "ModuleNotFoundError: No module named 'onnxscript'"로 죽는다.
#       0단계에서 미리 깔아두는 이유: 3·4·5단계가 전부 이걸 쓴다.
#    onnxscript 0.7.1은 'onnx>=1.17'만 요구하므로 위 onnx==1.18.0 핀은 그대로 유지된다(실측 확인).
pip install onnxscript

# 9) 실제로 뭐가 깔렸는지 확인 (아래 값과 일치해야 함)
pip list | grep -E "^(torch|torchvision|onnx|onnx-ir|onnxscript|onnxruntime-gpu|tensorrt|numpy|polygraphy)"
```

`pip list` 예상 출력 (2026-07-31 실측):

```text
numpy                                1.26.4
onnx                                 1.18.0
onnx-ir                              0.2.1
onnxruntime-gpu                      1.23.2
onnxscript                           0.7.1
polygraphy                           0.50.3
tensorrt_cu12                        10.16.1.11
tensorrt_cu12_bindings               10.16.1.11
tensorrt_cu12_libs                   10.16.1.11
torch                                2.11.0+cu128
torchaudio                           2.11.0+cu128
torchvision                          0.26.0+cu128
```

> 💡 팁: `tensorrt-cu12`를 하나 설치하면 `tensorrt_cu12_bindings`(Python 바인딩)와 `tensorrt_cu12_libs`(실제 `.so`)가 **함께** 깔립니다. 세 개의 버전이 모두 동일해야 정상이며, `pip list`에서 하이픈이 밑줄로 표시되는 것은 정규화된 이름 표기일 뿐 문제가 아닙니다. 셋 중 하나만 버전이 다르면 `import tensorrt`가 `.so` 로드 실패로 깨집니다.

> 💡 팁: `onnxruntime-gpu<1.27`이 실제로 어떤 버전을 골랐는지는 `pip index versions onnxruntime-gpu`로 후보 목록을 먼저 볼 수도 있습니다. Python 3.11+ venv에서는 같은 명령이 **1.26.0**을 받는데, 그것도 CUDA 12이므로 스택 정합은 유지됩니다(그 경우 `onnx` 상한은 ORT 1.26이 링크한 onnx 버전에 맞춰 재확인하세요).

> ⚠️ 확인 필요 — **구 가이드의 Azure DevOps CUDA-12 피드는 이제 권하지 않습니다.** 예전에는 PyPI 기본이 CUDA 11이라 `https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/` 를 써야 CUDA 12 wheel을 받을 수 있었습니다. 2026-07 실측 결과 이 피드는 **응답은 하지만(HTTP 200) stable 최신이 `1.19.2`에서 멈춰 있고**, 그 위는 2024~2025년 `.dev` 스냅샷뿐입니다(사실상 유지보수 중단). 1.23~1.26이 PyPI에서 그대로 CUDA 12 wheel로 제공되므로 **PyPI + `<1.27` 상한이 정답**입니다. 사내 프록시로 PyPI가 막힌 경우에만 이 피드(≤1.19.2)나 경로 B(NGC)를 고려하세요.

> 💡 팁: 대체 인덱스를 쓸 일이 있다면 `--index-url`(우선 인덱스를 **교체**)과 `--extra-index-url`(보조 인덱스를 **추가**)의 차이를 기억하세요. `--extra-index-url`로 붙이면 pip이 두 인덱스에서 더 높은 버전을 고르므로, PyPI의 CUDA-13 wheel이 끼어들 수 있습니다. 위 경로처럼 **PyPI 하나만 쓰고 버전 상한으로 제어**하는 편이 예측 가능합니다.

> 🔴 함정: `tensorrt`(접미사 없는 메타패키지)는 시점에 따라 CUDA 13 라인(11.x)을 끌어올 수 있습니다. **CUDA 12 스택에서는 반드시 `tensorrt-cu12`** 를 쓰세요. 그래도 `.so` 로드가 안 되면 [TensorRT 다운로드](https://developer.nvidia.com/tensorrt)에서 CUDA 12용 tar를 받아 `LD_LIBRARY_PATH`에 수동 등록하는 편이 확실합니다.

#### 3-4-a. cuDNN 등 venv 번들 NVIDIA 라이브러리를 `LD_LIBRARY_PATH`에 노출 (필수)

> 🔴 함정 (0단계에서 가장 값비싼 함정 — 건너뛰지 마세요): 위 설치를 끝내고 바로 검증하면 `ort.get_available_providers()`에는 `CUDAExecutionProvider`가 **보이는데도** 실제 세션은 CPU로 잡히는 일이 흔합니다. 원인은 CUDA 버전 불일치가 아니라 **`libcudnn.so.9`를 못 찾는 것**입니다. **`TensorrtExecutionProvider`도 같은 함정에 걸립니다** — 이쪽은 `libnvinfer.so.10`을 못 찾는데, 이 라이브러리는 `nvidia/*/lib`가 아니라 별도의 `tensorrt_libs` 패키지 디렉터리에 있어서 cuDNN 픽스만 적용하면 놓치기 쉽습니다. 두 라이브러리를 한 번에 잡는 방법을 아래에서 함께 다룹니다.

먼저 원인을 눈으로 확인합니다.

```bash
source ~/emb-ai/bin/activate

# 1) CUDA Toolkit에는 cuDNN이 없다 (출력 없음이 정상 = 이게 문제의 출발점)
ls /usr/local/cuda/lib64/libcudnn* 2>/dev/null || echo "→ CUDA toolkit deb에 cuDNN 없음 (정상)"

# 2) 그런데 venv 안에는 있다 (torch가 nvidia-cudnn-cu12로 끌어옴)
find "$VIRTUAL_ENV" -name "libcudnn.so.9" 2>/dev/null
#   예: ~/emb-ai/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9

# 3) 하지만 그 경로는 동적 로더 검색 경로에 없다
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 4) TensorRT도 마찬가지: libnvinfer.so.10은 nvidia/*/lib이 아니라 tensorrt_libs 안에 있다
find "$VIRTUAL_ENV" -name "libnvinfer.so.10" 2>/dev/null
#   예: ~/emb-ai/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.10
```

즉 라이브러리는 이미 디스크에 **있는데** `ld.so`가 못 찾는 상황입니다. `cuda-toolkit-12-8`을 다시 깔거나 cuDNN을 별도로 다운로드할 필요가 없습니다 — **경로만 알려주면 됩니다.**

아래 블록을 `~/emb-ai/bin/activate` **맨 끝에 한 번만** 추가하면 venv를 활성화할 때마다 자동 적용되어 영구 해결됩니다.

```bash
# venv activate 스크립트 끝에 venv 번들 NVIDIA 라이브러리 경로 주입 (한 번만 실행)
cat >> ~/emb-ai/bin/activate <<'EOF'

# --- emb-ai: NVIDIA libs (venv 번들 cuDNN9/CUDA12/TensorRT 라이브러리를 동적 로더에 노출) ---
# ORT CUDAExecutionProvider가 libcudnn.so.9 등을 찾도록. activate 시 동적 계산(패키지 업데이트를 견딤).
_EMBAI_NVLIBS=$(python -c "import os,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(os.path.join(b,d,'lib') for d in os.listdir(b) if os.path.isdir(os.path.join(b,d,'lib')))))" 2>/dev/null)
# ORT TensorrtExecutionProvider가 libnvinfer.so.10 등을 찾도록. tensorrt_libs는 nvidia/*/lib 글롭 밖에 있어 별도 계산이 필요.
_EMBAI_TRTLIBS=$(python -c "import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null)
if [ -n "$_EMBAI_NVLIBS" -o -n "$_EMBAI_TRTLIBS" ]; then
  export LD_LIBRARY_PATH="$_EMBAI_NVLIBS:$_EMBAI_TRTLIBS:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
unset _EMBAI_NVLIBS _EMBAI_TRTLIBS
EOF

# 새 설정을 적용하려면 venv를 다시 활성화한다
deactivate 2>/dev/null; source ~/emb-ai/bin/activate
```

적용됐는지 확인합니다.

```bash
# nvidia/*/lib 경로들, tensorrt_libs, /usr/local/cuda/lib64 가 앞쪽에 붙어야 함
echo $LD_LIBRARY_PATH | tr ':' '\n' | head -20

# 동적 로더가 실제로 cuDNN·TensorRT를 해결하는지 (경로가 찍히면 성공)
python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('libcudnn.so.9 로드 OK')"
python -c "import ctypes; ctypes.CDLL('libnvinfer.so.10'); print('libnvinfer.so.10 로드 OK')"
```

`LD_LIBRARY_PATH` 예상 출력 (실측):

```text
/home/<user>/emb-ai/lib/python3.10/site-packages/nvidia/cublas/lib
/home/<user>/emb-ai/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
/home/<user>/emb-ai/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
/home/<user>/emb-ai/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
/home/<user>/emb-ai/lib/python3.10/site-packages/nvidia/cudnn/lib
...
/home/<user>/emb-ai/lib/python3.10/site-packages/tensorrt_libs
/usr/local/cuda/lib64
```

> 💡 팁: 위 블록은 경로를 **하드코딩하지 않고 activate 시점에 `nvidia`/`tensorrt_libs` 패키지 위치로부터 계산**합니다. 그래서 나중에 `pip install -U torch`나 `pip install -U tensorrt-cu12`로 라이브러리 구성이 바뀌어도 그대로 동작하고, venv 경로를 옮겨도 깨지지 않습니다. `export LD_LIBRARY_PATH=...`를 `~/.bashrc`에 직접 박는 방식은 **권하지 않습니다** — venv를 안 쓰는 다른 작업에도 CUDA 12 라이브러리가 전역으로 새어나가 엉뚱한 충돌을 만듭니다.

> ⚠️ 주의: 이 문제는 **경로 A(호스트 pip)에서만** 발생합니다. 경로 B(NGC 컨테이너)는 cuDNN과 TensorRT가 시스템 경로(`/usr/lib/x86_64-linux-gnu`)에 정상 설치되어 있어 이 작업이 필요 없습니다.

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
print("onnx:", onnx.__version__)                         # 1.18.0 (IR 11)
print("onnxruntime:", ort.__version__)                   # 1.23.2
print("ORT providers:", ort.get_available_providers())   # CUDAExecutionProvider 있어야 함

print("== TensorRT ==")
import tensorrt as trt
print("TensorRT:", trt.__version__)                      # 10.16.1.11

# 세 도구의 CUDA 라인이 12로 정합인지 눈으로 확인
assert torch.version.cuda.startswith("12"), "torch가 CUDA 12 빌드가 아님"
assert "CUDAExecutionProvider" in ort.get_available_providers(), "ORT에 CUDA EP 없음"

# ONNX IR 상한 정합 확인 — 3·4단계 export가 로드 단계에서 터지지 않게 미리 못 박는다 (2절 참고)
# onnx.IR_VERSION = 이 onnx가 모델을 저장할 때 쓰는 IR 버전. ORT 1.23.2의 상한은 11.
assert onnx.IR_VERSION <= 11, (
    f"onnx {onnx.__version__}는 IR {onnx.IR_VERSION}을 생성 → ORT의 상한(11) 초과. "
    "'pip install onnx==1.18.0'으로 고정하세요"
)
print("\nOK: CUDA 12 스택 정합")
```

```bash
python verify_env.py
```

예상 출력 — 아래는 2026-07-31 실측 출력입니다 (GPU명은 자신의 GPU로, 마이너 버전은 설치 시점에 따라 다를 수 있음):

```text
== PyTorch ==
torch: 2.11.0+cu128 | CUDA available: True
torch CUDA build: 12.8
GPU: NVIDIA GeForce RTX 3060
== ONNX / ONNX Runtime ==
onnx: 1.18.0
onnxruntime: 1.23.2
ORT providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
== TensorRT ==
TensorRT: 10.16.1.11

OK: CUDA 12 스택 정합
```

> 💡 팁: `ORT providers`에 `TensorrtExecutionProvider`가 **먼저** 오는 것이 정상입니다. `onnxruntime-gpu` wheel은 TensorRT EP도 함께 담고 있어서, 목록은 우선순위 순으로 나옵니다. 이건 "TensorRT가 실제로 쓰인다"는 뜻이 아니라 "쓸 수 있다"는 뜻입니다 — 실제 사용은 [3단계](05_tensorrt.md)에서 다룹니다.

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

> 🔴 함정: `sess.get_providers()`의 **첫 항목이 CPU**로 나오면, provider 목록엔 CUDA가 있어도 런타임 초기화에 실패해 CPU로 내려간 것입니다. **실측 기준 1순위 원인은 `libcudnn.so.9` 미발견**이며, 해결은 [3-4-a](#3-4-a-cudnn-등-venv-번들-nvidia-라이브러리를-ld_library_path에-노출-필수)의 `LD_LIBRARY_PATH` 블록입니다. 3-4-a를 건너뛰었다면 지금 적용하고 다시 실행하세요.
>
> 어떤 `.so`가 실제로 빠졌는지 직접 확인하려면 ORT 로그를 최대로 올려 보세요 — 누락된 라이브러리 이름이 그대로 찍힙니다.
> ```bash
> # ORT 상세 로그로 CUDA EP 초기화 실패 사유 확인
> python - <<'PY'
> import onnxruntime as ort
> so = ort.SessionOptions()
> so.log_severity_level = 0          # 0=VERBOSE
> sess = ort.InferenceSession("tiny.onnx", so,
>                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
> print(sess.get_providers())
> PY
> #   'libcudnn.so.9: cannot open shared object file' 같은 줄이 보이면 3-4-a로 해결
> ```
> 그래도 안 풀리면 (드물게) ORT wheel의 CUDA major가 시스템과 어긋난 경우이므로 `pip list | grep onnxruntime`로 CUDA 13 wheel(1.27+)이 깔리지 않았는지 확인하고, 최후 수단으로 경로 B(NGC)로 전환하세요.

`get_available_providers()`에 `TensorrtExecutionProvider`가 있다는 것도 마찬가지로 **불충분**합니다 — 실측에서 이 EP는 라이브러리를 못 찾아도 예외 없이 조용히 CUDA(또는 CPU)로 내려갑니다.

```python
# verify_trt_ep.py — TensorrtExecutionProvider가 '실제로' 잡히는지 확인 (tiny.onnx는 위 verify_cuda_ep.py에서 생성됨)
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "tiny.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("실제 활성 providers:", sess.get_providers())     # 첫 항목이 Tensorrt여야 함

out = sess.run(None, {"X": np.ones((1, 4), dtype=np.float32)})
print("추론 결과:", out[0])
assert sess.get_providers()[0] == "TensorrtExecutionProvider", \
    "TensorRT로 세션이 안 잡힘 → CUDA/CPU fallback (libnvinfer.so.10 미발견 의심)"
print("OK: TensorrtExecutionProvider 실제 활성")
```

```bash
python verify_trt_ep.py
```

예상 출력:

```text
실제 활성 providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
추론 결과: [[1. 1. 1. 1.]]
OK: TensorrtExecutionProvider 실제 활성
```

> 🔴 함정 (실측 사례): 0.5단계 실습에서 실제로 겪은 경로입니다 — TensorRT가 예외 없이 세션을 만들고 벤치마크도 정상적으로 돌았지만, p50 지연시간이 CPU와 같은 자릿수(11ms대)였습니다. 원인은 `sess.get_providers()[0]`이 조용히 CPU로 내려가 있던 것이었고, 위 `assert`처럼 **활성 provider를 직접 확인하지 않으면 잡히지 않습니다.** 해결은 3-4-a의 `LD_LIBRARY_PATH` 블록에 `tensorrt_libs` 경로가 포함되어 있는지 확인하는 것입니다(위 최신 3-4-a는 기본 포함). 어떤 `.so`가 빠졌는지는 CUDA EP와 동일하게 `log_severity_level = 0`으로 확인할 수 있습니다 — `'libnvinfer.so.10: cannot open shared object file'`이 보이면 3-4-a를 (다시) 적용하세요.

> ✅ 검증 완료: `tensorrt-cu12==10.16.1.11`은 2026-07 기준 PyPI에 실재하며(10.x 계열의 최신 패치), 실측 머신에서 `import tensorrt` → `10.16.1.11`로 정상 동작했습니다. 참고로 `tensorrt-cu12`의 전체 최신은 **11.2.1.2**(CUDA 13 라인)이므로, 버전을 명시하지 않으면 CUDA 13 쪽이 깔립니다. pip에서 못 찾으면 [TensorRT PyPI](https://pypi.org/project/tensorrt/) 또는 [TensorRT 다운로드](https://developer.nvidia.com/tensorrt)에서 CUDA 12 대응 tar/deb를 직접 받으세요.

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

### 경로 B (유효한 대안): NGC 컨테이너로 정합 스택 한 번에

> 💡 이 가이드가 실측 검증한 것은 **경로 A**(3-4 + 3-4-a)입니다. 경로 B는 삭제되지 않은 **여전히 유효한 대안**이며, 특히 *버전 충돌 디버깅에 시간을 쓰고 싶지 않을 때* 또는 *CUDA 13 라인이 필요할 때* 더 낫습니다. 다만 경로 A의 유일한 실질적 함정(cuDNN 경로)은 이제 3-4-a로 해결되므로, "조용한 CPU fallback을 만났다"는 이유만으로 경로 B로 도망갈 필요는 없습니다.

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

# 설치 후 numpy가 2.x로 올라가지 않았는지 확인 (1.26.x 여야 함)
python -c "import numpy; print('numpy:', numpy.__version__)"
```

> 🔴 함정 (numpy 상한): `nuscenes-devkit 1.2.0`의 의존성은 **`numpy<2.0.0,>=1.22.0`** 입니다(2026-07 PyPI 실측). 그래서 venv에 numpy 2.x가 있으면 이 설치가 **numpy를 1.26.x로 강제 다운그레이드**합니다. 문제는 그 시점에 이미 numpy 2.x ABI로 빌드된 패키지들이 깔려 있으면 `numpy.dtype size changed` 류의 경고·오류가 줄줄이 나온다는 것입니다.
>
> 그래서 3-4절에서 **맨 처음에 `pip install "numpy<2"`** 를 실행했습니다. 순서를 지켰다면 여기서 다운그레이드가 일어나지 않고 `numpy: 1.26.4`가 그대로 유지됩니다. 만약 2.x로 올라가 있었다면 아래로 되돌리세요.
>
> ```bash
> pip install "numpy<2"
> python -c "import torch, onnxruntime, numpy; print('numpy', numpy.__version__, '| torch ok')"
> ```
>
> torch 2.11.0 / ORT 1.23.2 / TensorRT 10.16.1.11 모두 **numpy 1.26.4에서 정상 동작**을 실측 확인했습니다.

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

예상 출력 (2026-07-31 실측):

```text
======
Loading NuScenes tables for version v1.0-mini...
Loading nuScenes-lidarseg... skipped
23 category, 8 attribute, ...
Done loading in X.XXX seconds.
======
scenes: 10
samples: 404
첫 sample 토큰: ca9a282c9e77460f8360f564131a8af5
센서 채널 예: ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
```

> 💡 팁: `첫 sample 토큰`이 `ca9a282c...`로 정확히 일치하면 v1.0-mini를 온전히 받은 것입니다(mini의 sample 순서는 고정). `scenes: 10 / samples: 404`와 함께 이 토큰까지 맞으면 데이터셋 검증은 끝입니다.

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
| 호스트에서 자유롭게 실험/디버깅하고 싶다 | **경로 A (pip, CUDA 12 통일)** | IDE·프로파일러 붙이기 쉬움. **이 가이드가 실측 검증한 경로** |
| 정본 스택(TensorRT 10.16.x LTS)을 정확히 고정해야 한다 | 경로 A | 버전을 pip로 못 박음 |
| 최신 TensorRT 11.x / ORT CUDA 13 기능이 필요 | 경로 B (CUDA 13 태그) | pip로 CUDA 13 혼합은 깨지기 쉬움 |

> 💡 팁: 2026-07-31 실측은 **경로 A**로 진행했고, 3-4-a의 `LD_LIBRARY_PATH` 픽스만 적용하면 경로 A도 충분히 안정적이었습니다. 경로 A의 유일한 실질적 함정이 cuDNN 경로였고, 그건 이제 문서화되어 있습니다.

**검증 신호 해석**

| 확인 명령 | 정상 신호 | 비정상 → 의심 지점 |
|-----------|-----------|-------------------|
| `nvidia-smi` (호스트) | GPU 표 + 드라이버/CUDA 상한 | 드라이버 미로딩 → 재부팅/Secure Boot/nouveau |
| `nvcc --version` | `release 12.8` | 다른 버전 → PATH가 다른 CUDA를 가리킴 |
| `docker run --gpus all ... nvidia-smi` | 컨테이너 안에서 동일 표 | Container Toolkit 미설정 → `nvidia-ctk runtime configure` |
| `torch.cuda.is_available()` | `True` | `False` → torch가 CPU wheel / 드라이버 상한 부족 |
| `ort.get_available_providers()` | `CUDAExecutionProvider` 포함 | 없으면 ORT가 CUDA 13 wheel(1.27+)일 가능성 → `pip list \| grep onnxruntime` |
| `sess.get_providers()[0]` | `CUDAExecutionProvider` | CPU면 조용한 fallback → **1순위: `libcudnn.so.9` 미발견(3-4-a)** |
| `python -c "import onnx; print(onnx.IR_VERSION)"` | `11` | 12·13이면 onnx가 너무 높음 → `pip install onnx==1.18.0` |
| `python -c "import numpy; print(numpy.__version__)"` | `1.26.x` | 2.x면 nuscenes-devkit이 다운그레이드시킴 → `pip install "numpy<2"` |
| `load_nuscenes.py` | `scenes: 10 / samples: 404` | 경로/압축해제 문제 |

**2026-07 기준 확인한 버전 스냅샷** — 정본 스택은 2026-07-31 실측 설치값이고, "PyPI/APT 최신"은 같은 날 조회한 값입니다. 둘이 다른 이유(= 왜 최신을 안 쓰는지)를 비고에 적었습니다.

| 구성요소 | 정본 스택(이 스터디 고정, 실측) | 2026-07 PyPI/APT 최신 | 왜 최신을 안 쓰나 |
|----------|--------------------------------|----------------------|------------------|
| CUDA Toolkit | **12.8** (`nvcc` 12.8.93) | 13.x (APT 최신) | PyTorch `cu128` 기준선. 13은 구형 아키텍처 타깃 제외 |
| PyTorch wheel index | **`torch 2.11.0+cu128`** | `cu128` | 동일 — 최신이 곧 정본 |
| **onnx** | **`1.18.0` (IR 11)** | **1.22.0 (IR 13)** | ORT 1.23.2의 **IR 상한이 11**. 1.19부터 IR 12+ → 로드 실패 |
| onnxruntime-gpu | **`1.23.2` (CUDA 12)** | 1.28.0 (CUDA 13) | 1.27+는 CUDA 13. **Python 3.10 wheel도 1.23.2가 마지막** |
| TensorRT | **`tensorrt-cu12==10.16.1.11`** | 11.2.1.2 (CUDA 13 기본) | 11.x는 CUDA 13 라인 → 10.x LTS가 CUDA 12 주경로 |
| **numpy** | **`1.26.4`** | 2.x | `nuscenes-devkit 1.2.0`이 `numpy<2.0.0` 요구 |
| polygraphy | **`0.50.3`** | 0.50.3 | 동일 — 최신이 곧 정본 |
| nuscenes-devkit | **`1.2.0`** | 1.2.0 | 동일 — 최신이 곧 정본 |
| ExecuTorch | **1.3.x** | - | 0단계에선 미설치 |
| NVIDIA Container Toolkit | **1.19.1** (실측) | 최신 stable | install-guide 기준. 버전 민감도 낮음 |

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
| `ort.get_available_providers()`에 CUDA EP 없음 | onnxruntime-gpu가 **CUDA 13 wheel**(1.27+, PyPI 기본)로 깔림 | 상한을 걸어 재설치: `pip install --force-reinstall "onnxruntime-gpu<1.27"` (Python 3.10 → 1.23.2) |
| `ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu==1.28.0` (Azure CUDA-12 피드) | 그 피드는 **stable이 1.19.2에서 멈춤**(유지보수 중단). 1.28.0은 PyPI에만 있고 **CUDA 13**임 | 피드를 쓰지 말고 PyPI에서: `pip install "onnxruntime-gpu<1.27"` (3-4절) |
| **CUDA EP는 목록에 있는데 `sess.get_providers()[0]`이 CPU** (조용한 fallback) | **`libcudnn.so.9` 미발견.** cuDNN 9는 `cuda-toolkit-12-8` deb에 없고 venv의 `nvidia-cudnn-cu12`에만 있는데, 그 경로가 `ld.so` 검색 경로 밖 | **[3-4-a](#3-4-a-cudnn-등-venv-번들-nvidia-라이브러리를-ld_library_path에-노출-필수)의 `LD_LIBRARY_PATH` 블록을 `~/emb-ai/bin/activate`에 추가** 후 venv 재활성화. 확인: `python -c "import ctypes; ctypes.CDLL('libcudnn.so.9')"` |
| **TensorRT EP는 목록에 있고 세션도 예외 없이 만들어지는데 실측 지연시간이 CPU급** (조용한 fallback) | **`libnvinfer.so.10` 미발견.** `tensorrt-cu12`는 이 라이브러리를 `nvidia/*/lib`가 아니라 별도의 `tensorrt_libs` 패키지 디렉터리에 두는데, cuDNN용 글롭만으론 못 잡음 | **[3-4-a](#3-4-a-cudnn-등-venv-번들-nvidia-라이브러리를-ld_library_path에-노출-필수)의 `LD_LIBRARY_PATH` 블록에 `tensorrt_libs` 경로가 포함되도록 갱신** 후 venv 재활성화. 확인: `python -c "import ctypes; ctypes.CDLL('libnvinfer.so.10')"` 또는 `verify_trt_ep.py`로 `sess.get_providers()[0]` 직접 확인(단순 `get_available_providers()` 목록 확인으론 못 잡음) |
| `Fail: ... Unsupported model IR version: 13, max supported IR version: 11` (3·4단계 ONNX 로드 시) | `pip install onnx`로 최신 1.22.0(IR 13)이 깔림. ORT 1.23.2는 IR 11까지만 | `pip install "onnx==1.18.0"`. 확인: `python -c "import onnx; print(onnx.IR_VERSION)"` → `11`. export 시 opset ≤ 23 (2절) |
| `torch.onnx.export`에서 `ModuleNotFoundError: No module named 'onnxscript'` (3·4단계 export 시) | **torch 2.11의 `torch.onnx.export`는 기본이 `dynamo=True`** 이고 그 경로가 `onnxscript`를 요구한다. 0단계 스택에는 없다 | `pip install onnxscript` (0.7.1은 `onnx>=1.17`만 요구 → 1.18.0 핀 안 깨짐, 실측 확인). 또는 legacy 경로로 `torch.onnx.export(..., dynamo=False)` |
| `numpy.dtype size changed, may indicate binary incompatibility` / `pip`가 numpy를 2.x↔1.26으로 오르내림 | `nuscenes-devkit 1.2.0`이 `numpy<2.0.0`을 요구해 뒤늦게 다운그레이드시킴 | 설치 **순서**를 지켜 처음부터 `pip install "numpy<2"` (3-4절 2번). 이미 어긋났으면 `pip install --force-reinstall "numpy<2"` |
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
- [ ] `verify_trt_ep.py` 실행 로그 — `sess.get_providers()[0] == TensorrtExecutionProvider` 확인(조용한 CPU fallback 없음 증빙)
- [ ] `/data/sets/nuscenes/` 에 배치된 nuScenes **mini** (폴더: `samples/ sweeps/ maps/ v1.0-mini/`)
- [ ] `load_nuscenes.py` 실행 로그 — `scenes: 10 / samples: 404` + 첫 sample 토큰 `ca9a282c...`
- [ ] (경로 A) `~/emb-ai/bin/activate`에 추가된 `LD_LIBRARY_PATH` 블록 — cuDNN·TensorRT 픽스 적용 증빙(3-4-a)
- [ ] 선택한 경로(A: pip CUDA 12 / B: NGC 컨테이너)와 사용한 **정확한 버전**을 적은 `ENV.md` 메모 (다음 단계 재현성용). 최소 항목: 드라이버 버전, `nvcc` 버전, torch/**onnx**/onnxruntime-gpu/tensorrt/**numpy** 버전, (컨테이너면) NGC 태그.

> 💡 팁: `ENV.md`는 뒤 단계에서 "내 숫자가 왜 가이드와 다르지?"를 추적할 때 결정적입니다. 아래 명령으로 자동 수집해 붙여두면 편합니다. **`onnx`의 IR 버전과 `numpy`까지 기록**해 두는 게 중요합니다 — 3·4단계에서 가장 먼저 의심할 값들입니다.
>
> ```bash
> {
>   echo '## ENV snapshot'; date
>   nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
>   nvcc --version | grep release
>   python -c "import torch,onnx,onnxruntime,numpy,tensorrt as t;\
> print('torch',torch.__version__);print('onnx',onnx.__version__,'(IR',onnx.IR_VERSION,')');\
> print('ort',onnxruntime.__version__);print('trt',t.__version__);print('numpy',numpy.__version__)"
> } > ENV.md
> cat ENV.md
> ```

**이 문서의 기준 스택은 실측입니다.** 위 산출물 6종은 2026-07-31에 Ubuntu 22.04.5 + RTX 3060(드라이버 595.84, `nvcc` 12.8.93) 머신에서 **경로 A로 전부 통과**했습니다. 그 결과가 2절 "정본 버전 스택"과 5절 스냅샷 표이며, 두 표는 서로 일치합니다. 자신의 `ENV.md`를 그 표와 대조해 **다른 줄이 있으면 그 줄을 기록해 두세요** — 뒤 단계에서 숫자가 갈릴 때 그 줄이 원인일 확률이 높습니다.

---

## 8) 참고 사이트 & 참고문헌

### 공식 문서 / 도구
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) — OS/아키텍처별 CUDA Toolkit 설치 명령 생성기(deb/runfile)
- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) — 공식 Linux 설치 가이드
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) — WSL2에서 리눅스 드라이버 금지 등 필수 규칙
- [NVIDIA Container Toolkit — Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — apt 저장소/`nvidia-ctk`/rootless 설정
- [Docker Engine 설치 (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/) — Docker 공식 저장소 설치
- [PyTorch — Get Started Locally](https://pytorch.org/get-started/locally/) — CUDA 버전별 pip 설치 명령(`cu128`)
- [ONNX Runtime — Install](https://onnxruntime.ai/docs/install/) / [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) — CUDA 12/13 빌드 구분, CUDA/cuDNN 요구 버전
- [onnxruntime-gpu PyPI](https://pypi.org/project/onnxruntime-gpu/) — 버전별 `nvidia-*-cu12` / `cu13` 의존성으로 CUDA 라인을 직접 확인 가능(1.27부터 cu13)
- [ONNX — Versioning.md (IR/opset 대응표)](https://github.com/onnx/onnx/blob/main/docs/Versioning.md) — onnx 릴리스 ↔ `IR_VERSION` ↔ 최대 opset 표. `1.18.0 → IR 11 / opset 23` 근거
- [ONNX Runtime — 버전 호환성](https://onnxruntime.ai/docs/reference/compatibility.html) — ORT ↔ opset/IR 대응. ⚠️ 2026-07 현재 이 표는 **1.20까지만** 갱신돼 있어, 1.23의 상한은 소스(`cmake/deps.txt`의 onnx 핀)로 확인하는 편이 정확합니다
- [ONNX Runtime `cmake/deps.txt` (rel-1.23.2)](https://github.com/microsoft/onnxruntime/blob/rel-1.23.2/cmake/deps.txt) — 이 릴리스가 `onnx v1.18.0`을 링크한다는 1차 근거 → 그래서 max IR = 11
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
