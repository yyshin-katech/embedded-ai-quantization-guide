# 8. 캡스톤 프로젝트 — 이거 하나면 지원 가능

> 원본 가이드 매핑: "캡스톤 프로젝트 — nuScenes mini 멀티카메라 BEV detector 4-target 배포"
> 예상 소요: 3~5주 (앞 1~5단계 완료 후, 하루 3~4시간 기준)
> 선행 조건: [1단계 양자화 이론](03_quantization_theory.md) · [2단계 Transformer 양자화](04_transformer_quantization.md) · [3단계 TensorRT](05_tensorrt.md) · [4단계 멀티 SoC](06_multi_soc.md) · [5단계 인프라화](07_infrastructure.md)를 모두 끝냈다고 가정한다.
> 정본 버전 스택(호스트/배포 계층): **CUDA 12.8 / TensorRT 10.16.x LTS** (2026-07 기준). BEV 모델 학습·export 스택은 아래 §2-1 함정에서 설명하듯 의도적으로 구버전(cu116)에 묶어 둔다 — 이 버전 간극 자체가 실습의 일부다.

---

## 0) 이 단계에서 무엇을·왜 하는가

앞 1~5단계는 각각 "이론", "TensorRT", "멀티 SoC", "인프라"를 **따로** 연습했다. 채용 담당자는 그걸 **하나로 꿰맨 결과물**을 본다. 이 캡스톤의 목표는 딱 한 문장이다.

> **nuScenes mini 기반 멀티카메라 BEV detector 하나를 골라, PyTorch → ONNX → INT8 PTQ → mixed precision 튜닝을 거쳐 4개 타깃(TensorRT/TIDL/QNN/DRP-AI)에 배포하고, 성능·정확도를 한 표로 비교한다.**

**왜 BEV detector인가?** 자율주행 인식(perception) JD가 요구하는 ①멀티카메라 3D 검출 ②Transformer(deformable attention) 배포 ③INT8 양자화 ④멀티 SoC 이식 을 **한 모델**로 전부 건드릴 수 있는 거의 유일한 과제다. BEV(Bird's-Eye-View)는 6개 카메라 이미지를 조감도 격자로 합쳐 3D 박스를 뽑는데, 이 과정에 CNN 백본 + Transformer 어텐션 + 커스텀 op(`grid_sample`, deformable attn, BEV pooling)가 전부 들어간다. 즉 **양자화·배포에서 터질 만한 함정이 다 모여 있다**. 그래서 "완주하면 지원 가능"이라는 말이 성립한다.

**왜 "따로 배운 걸 꿰매는" 게 그렇게 중요한가?** 실무에서 모델 최적화 엔지니어가 실제로 하는 일이 바로 이 "꿰매기"다. 이론(1단계)만 아는 사람, TensorRT(3단계)만 돌려 본 사람은 흔하다. 하지만 "이 모델의 이 레이어가 왜 INT8에서 터지는지 → 민감도로 찾아 → mixed precision으로 회복 → 그걸 4개 SoC로 옮길 때 어떤 op가 각각 어디서 막히는지"를 **한 흐름으로 설명할 수 있는 사람**은 드물다. JD의 4개 항목은 서로 독립이 아니라 **하나의 파이프라인의 단면들**이고, 이 캡스톤은 그 파이프라인을 통째로 관통시키는 훈련이다.

> 💡 팁: 이 조합(BEV + INT8 + 4-target + 실패 로그 공개)을 GitHub에 정리해 둔 사람은 국내에 거의 없다. 완성도가 낮아도 "끝까지 밀어붙인 흔적"만으로 차별화된다. 리크루터·면접관이 보는 것은 최종 mAP 숫자가 아니라 **"이 사람이 실제로 배포 파이프라인을 손으로 만져 봤는가"**이다.

> 🔴 함정: 이 문서의 목표는 **SOTA mAP 재현이 아니다**. "가벼운 모델 하나를 실제로 4개 타깃에 올려 보고, 어디서 왜 터지는지를 기록"하는 것이 핵심이다. mAP 0.01을 올리려고 몇 주를 쓰지 마라. mini 데이터셋으로는 애초에 논문 수치가 안 나온다(§2-1, §5). "정확도를 얼마나 지키면서 얼마나 줄였나"의 **트레이드오프 곡선**이 주인공이지, 절대 mAP가 아니다.

---

## 1) 학습 목표 & 완료 체크리스트

- [ ] 프로젝트 목표·범위·평가지표(mAP / NDS / latency / peak mem)를 `README.md`에 한 페이지로 정의했다.
- [ ] BEVDet / BEVFormer-tiny / PETR 중 하나를 근거를 갖고 선택했다(§2-1 비교표 + §2-3 결정 트리).
- [ ] nuScenes **v1.0-mini**를 받아 디렉토리 구조를 확인하고 devkit로 샘플 한 장을 시각화했다(§4-1).
- [ ] mini를 mmdetection3d 형식으로 전처리했다(`.pkl` 생성).
- [ ] 공식 체크포인트로 PyTorch FP32 baseline mAP/NDS를 재현하고 `reports/baseline_fp32.json`에 저장했다.
- [ ] ONNX export에 성공했다(deformable attn / `grid_sample` / BEV pool 처리 포함, [2단계](04_transformer_quantization.md) 기법 사용). PyTorch parity(출력 오차)를 검증했다.
- [ ] INT8 PTQ(Entropy calibration, 200 프레임)로 엔진/모델을 만들었다([1단계](03_quantization_theory.md) 이론 적용).
- [ ] 레이어 민감도 분석 후 `layer_sensitivity.csv`(1단계 산출물명 유지)를 만들고, mixed precision(INT8 + 일부 FP16)으로 정확도를 회복했다.
- [ ] **최소 2개** 타깃(TensorRT 필수 + 나머지 중 1개 이상)에 배포하고 수치를 얻었다. 4개 전부면 만점.
- [ ] 성능 대시보드(표 + Pareto 그래프), `design_rules.md`, `failures.md`(실패 로그)를 작성했다.
- [ ] GitHub 공개 + 기술 블로그 5편 초안을 썼다(§8, 각 편 상세 목차 포함).

> ⚠️ 주의: "4개 타깃 전부 성공"은 **완주 조건이 아니다**. §9의 완주 판정 기준을 먼저 읽어라. TIDL/QNN/DRP-AI에서 deformable attn이 안 올라가는 건 흔한 일이고, **그 실패를 로그로 남기는 것 자체가 산출물**이다.

> 💡 팁 — 이 체크리스트를 GitHub 이슈로 만들어라: 위 항목들을 리포의 Issue 하나씩(또는 Projects 칸반 카드)으로 만들면, 커밋 히스토리와 이슈 클로즈 타임라인이 그대로 "3~5주간 꾸준히 밀어붙인 증거"가 된다. 면접에서 "언제부터 언제까지 어떻게 진행했나"를 리포가 대신 증언한다.

---

## 2) 배경 이론 / 모델 선택 가이드

### 2-1. BEV detector 3종 비교 (가벼운 것부터)

세 모델 모두 nuScenes 6-카메라 입력 → 3D 박스 출력이지만, **배포 난이도**가 크게 다르다. 핵심은 "어떤 커스텀 op가 들어가느냐"다.

| 항목 | **BEVDet** (권장 시작점) | **PETR** | **BEVFormer-tiny** |
|------|------|------|------|
| arXiv | 2112.11790 | 2203.05625 | 2203.17270 |
| 공식 repo | `HuangJunJie2017/BEVDet` (dev3.0) | `megvii-research/PETR` | `fundamentalvision/BEVFormer` |
| 뷰 변환 방식 | LSS(Lift-Splat-Shoot) + **BEV Pooling** | 3D position embedding + sparse query | **deformable attention** (spatial/temporal) |
| 핵심 커스텀 op | `bev_pool_v2` | 표준 attention(대체로 op 표준) | `MultiScaleDeformableAttn` + `grid_sample` |
| 배포 난이도 | ★★☆ (LSS pooling만 처리하면 됨) | ★★★ (query 기반, NMS-free) | ★★★★ (deformable attn이 mmdeploy 미지원) |
| INT8 우호도 | 높음(CNN 비중 큼) | 중간 | 낮음(attn 민감) |
| baseline (mini는 낮음) | R50: mAP 0.283 / NDS 0.350* | val: mAP 0.441 / NDS 0.504* | tiny: mAP 0.252 / NDS 0.354* |
| 추천 대상 | **처음/시간 부족** | 중급 | 완주 자신 있을 때 |

\* 수치는 **full nuScenes val** 기준 공식값(mmdetection3d/논문/DerryHub repo). `v1.0-mini`(10 scene)로 학습·평가하면 이보다 크게 낮게 나오는 게 정상이다 — mini는 재현·파이프라인 검증용이지 SOTA 재현용이 아니다(§5-0 참고).

> 💡 팁: **BEVDet부터 시작하라.** LSS pooling(`bev_pool_v2`) 커스텀 op 하나만 해결하면 TensorRT까지 공식 스크립트(`convert_bevdet_to_TRT.py`)가 INT8까지 지원한다. BEVFormer-tiny는 "deformable attention을 어떻게 배포하나"라는 가장 값진 경험을 주지만, mmdeploy가 안 받아줘서 커스텀 플러그인이 필요하다(§4-4 경로 B). 시간이 있으면 BEVDet 완주 후 BEVFormer-tiny를 "도전 과제"로 얹어라.

> 🔴 함정 (2026 기준 재현 난점 — **의도된 함정**): BEVDet/BEVFormer/PETR는 전부 **mmcv 1.x + mmdetection3d 1.0** 계열에 묶여 있다. 이 스택은 **PyTorch < 2.0, CUDA < 12.0** 만 공식 지원한다. 정본 호스트 스택(CUDA 12.8 / RTX Blackwell 등)에서 그냥 `pip install mmcv-full` 하면 **컴파일이 깨진다** — mmcv 1.x의 CUDA 커널이 CUDA 12.x 헤더/아키텍처(`sm_90`, `sm_120`)와 맞지 않기 때문이다. 이건 버그가 아니라 **이 캡스톤이 일부러 남겨 둔 함정**이다. 실무에서 "논문 코드가 최신 드라이버에서 안 돌아간다"는 상황을 그대로 재현한 것이고, 해결하는 과정 자체가 포트폴리오 소재다. 해결책은 두 가지 —
> - **(a) Docker로 구버전 스택을 통째로 고정** (가장 안전, §3). 학습·export 컨테이너는 cu116로 격리하고, 최종 배포/벤치만 정본 CUDA 12.8 / TRT 10.16 호스트에서 돌린다.
> - **(b) 커뮤니티 패치**: [`nabe2030/bevformer-blackwell`](https://github.com/nabe2030/bevformer-blackwell)가 mmcv 1.x를 PyTorch 2.x/CUDA 12.x(Blackwell 포함)에 올리는 패치를 공개한다. RTX 50 시리즈 등 구 CUDA를 아예 못 까는 GPU에서 유용.
>
> 이 문서는 (a)를 기본, (b)를 대안으로 둔다. **어느 쪽을 택했고 왜인지를 `docs/design_rules.md`에 적는 것**이 산출물의 일부다.

### 2-2. 평가지표 정의

**mAP (nuScenes 방식)** — 일반 IoU가 아니라 **BEV 중심점 거리** 임계값 {0.5, 1, 2, 4} m 로 매칭해 계산한 평균 정밀도. 즉 "박스 겹침"이 아니라 "예측 중심점이 GT 중심점에서 얼마나 가까운가"로 TP를 정하고, 4개 거리 임계값의 AP를 클래스·임계값 평균한다. 3D 박스 IoU를 안 쓰는 이유는 원거리 소형 객체에서 IoU가 지나치게 불안정하기 때문이다.

**NDS (nuScenes Detection Score)** — mAP에 5개 True-Positive 오차(위치 ATE, 크기 ASE, 방향 AOE, 속도 AVE, 속성 AAE)를 합친 종합 점수:

```
NDS = (1/10) · [ 5·mAP + Σ_{mTP ∈ {ATE,ASE,AOE,AVE,AAE}} (1 − min(1, mTP)) ]
```

직관: mAP가 절반의 가중치(5/10)를 갖고, 나머지 절반은 "박스를 맞췄다면 위치·크기·방향·속도·속성을 얼마나 정밀하게 맞췄나"가 채운다. 각 오차 mTP는 0(완벽)~1(형편없음)로 정규화돼 `1 − min(1, mTP)`로 점수화된다. **양자화가 NDS를 깎는 방식이 mAP와 다를 수 있다** — INT8이 검출 개수(mAP)는 유지해도 회귀 정밀도(ATE/AOE)를 흔들어 NDS만 떨어지는 패턴이 흔하다. 그래서 대시보드에는 둘 다 넣는다.

**latency** — 배포 타깃별 1프레임(6-카메라) 추론 시간(ms). 전처리 제외/포함을 명시. p50/p99를 함께 기록(p99는 실시간성 판단의 핵심 — 자율주행은 평균이 아니라 최악을 본다).

**peak memory** — 엔진 로드 + 추론 시 최대 사용 메모리. TensorRT는 `trtexec --dumpProfile`/Nsight Systems, host emu는 프로세스 RSS(`/usr/bin/time -v`의 Maximum resident set size)로 측정.

> 💡 팁: 캡스톤의 스토리는 "**정확도(mAP/NDS)를 얼마나 지키면서 latency/mem을 얼마나 줄였나**"의 4차원 트레이드오프다. INT8이 FP32 대비 mAP를 몇 점 떨어뜨리고 latency/mem을 몇 % 줄였는지가 대시보드의 주인공이다. 표 한 칸에 "mAP −0.4%p / latency −64% / mem −71%"처럼 **세 축을 함께** 적으면 한눈에 트레이드오프가 읽힌다.

### 2-3. 모델 선택 결정 트리 (2분 안에 정하기)

고민으로 며칠을 태우지 말고 아래로 즉시 결정하라.

```
Q1. 처음 시도이거나 남은 기간이 3주 이하인가?
   └ 예  → BEVDet-R50 (공식 TRT 스크립트 INT8까지 지원, 함정 최소)  ← 90%는 여기
   └ 아니오 ↓
Q2. "deformable attention 배포"를 포트폴리오의 핵심 서사로 삼고 싶은가?
   └ 예  → BEVFormer-tiny (+ DerryHub 커스텀 플러그인 경로, §4-4 B). 단, mmdeploy 미지원 우회를 감수.
   └ 아니오 ↓
Q3. NMS-free query 기반(표준 attention)으로 op 커버리지를 넓게 확보하고 싶은가?
   └ 예  → PETR (attention이 대체로 표준 op라 SoC 이식이 상대적으로 수월)
   └ 아니오 → BEVDet로 돌아가라.
```

> 💡 팁: **가장 강한 포트폴리오 = BEVDet로 4-target을 관통 + BEVFormer-tiny로 "deformable attn은 여기서 막혔다"를 추가**. 즉 "쉬운 것으로 파이프라인을 완주"하고, "어려운 것으로 한계를 기록"하는 2단 구성이 이상적이다. 둘 다 못 하겠으면 BEVDet 하나만으로도 완주 판정(§9)을 충족한다.

---

## 3) 환경·도구 준비

[1단계 환경 준비](01_environment_setup.md)에서 만든 Docker + NVIDIA Container Toolkit 환경을 재사용한다. **핵심 아이디어: 두 개의 환경을 분리한다.**

1. **학습·export 컨테이너 (cu116, 격리)** — mmcv 1.x가 필요한 BEV 스택. 여기서 PyTorch baseline·ONNX export까지 한다.
2. **배포·벤치 호스트 (정본 CUDA 12.8 / TensorRT 10.16.x LTS)** — ONNX를 받아 최종 INT8 엔진 빌드·`trtexec` 벤치·4-target 실험. 1단계에서 이미 깔아 둔 정본 스택을 그대로 쓴다.

이렇게 나누면 "논문 코드의 구버전 의존성"과 "최신 배포 툴체인"이 서로를 오염시키지 않는다. **ONNX가 두 세계를 잇는 다리**다.

```bash
# 0. 프로젝트 루트 생성 (5단계 인프라 규약 재사용)
mkdir -p ~/capstone-bev/{configs,work_dirs,onnx,engines,calib,reports,scripts,docs}
cd ~/capstone-bev

# 1. BEVDet 공식 repo (dev3.0 브랜치 = 배포/CUDA 가속 지원)
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet && git checkout dev3.0 && cd ..
```

```bash
# 2. 학습·export용 구버전 스택을 Docker로 고정 (2026 재현 난점 회피)
#    BEVDet dev3.0 도커파일을 기반으로 빌드하는 것이 가장 안전하다.
docker build -t bev-capstone:cu116 -f BEVDet/docker/Dockerfile BEVDet/
# 핵심 버전(2026-07 기준, BEVDet dev3.0 문서 검증):
#   CUDA 11.6 / cuDNN 8.6 / PyTorch 1.12.1+cu116 / mmcv-full 1.5.x / mmdet 2.25.1
```

```bash
# 2-b. 컨테이너 실행 (GPU + 데이터 볼륨 마운트). 학습/export는 이 안에서.
docker run --gpus all -it --rm \
  -v ~/capstone-bev:/workspace \
  -v ~/capstone-bev/BEVDet/data:/workspace/BEVDet/data \
  --shm-size=16g bev-capstone:cu116 bash
```

> ⚠️ 확인 필요: BEVDet dev3.0의 `docker/Dockerfile`이 그대로 빌드되는지는 base 이미지 태그 만료 여부에 따라 달라질 수 있다. 실패하면 mmcv/mmdet 버전을 위 표대로 고정해 수동 설치한다. 출처: [BEVDet dev3.0 Dockerfile](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docker/Dockerfile), [BEVDet getting_started](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/docs/en/getting_started.md).

```bash
# 3. deformable attn INT8 배포까지 갈 거면 이 repo도 클론(§4-4 경로 B)
git clone https://github.com/DerryHub/BEVFormer_tensorrt.git
# 이쪽 요구 스택(README 검증, 2026-07):
#   CUDA 11.6 / TensorRT 8.5.1.7 / cuDNN 8.6.0 / PyTorch 1.12.1+cu116 /
#   mmcv-full 1.5.0 / mmdet 2.25.1 / mmdeploy 0.10.0
#   ※ base 모델은 8.5.1.7에서 OOM 알려짐 → TensorRT 8.4.3.1로 빌드 권장(§6)
```

```bash
# 4. nuScenes devkit (평가·시각화용) — 배포 호스트/컨테이너 어디서든 OK
pip install nuscenes-devkit   # Python 3.9~3.12 지원

# 5. 정본 배포 호스트 툴 확인 (1단계에서 깐 것)
trtexec --version    # TensorRT 10.16.x (CUDA 12.8 빌드) 확인
nvcc --version       # CUDA 12.8 확인
```

> 🔴 함정 (버전 3중고): 이 프로젝트는 **세 개의 TensorRT 세계**를 동시에 다룬다 — ① 학습/export의 DerryHub 경로(TRT 8.5.1.7, 구현식 INT8 calibrator), ② 정본 호스트 배포(**TRT 10.16.x LTS, CUDA 12.8**, 명시식 QDQ 양자화), ③ Orin의 JetPack 동봉 TensorRT. **엔진(`.engine`)은 빌드한 환경에서만 로드된다** — 8.5로 만든 엔진은 10.16에서 안 열리고, RTX에서 만든 엔진은 Orin에서 안 열린다. 각 타깃에서 **ONNX부터 다시 빌드**해야 한다([3단계](05_tensorrt.md) 참고). 이 3중고를 명확히 정리해 두는 것 자체가 실력의 증거다.

> ⚠️ 주의 (TRT 10.x의 INT8 방식 변화 — 중요): **TensorRT 10.1부터 구현식(implicit) INT8 양자화와 `IInt8EntropyCalibrator2` 등 캘리브레이터 API가 deprecated**됐고, **명시식(explicit) 양자화 = QDQ 노드 삽입**으로 대체됐다(2026-07 기준, TRT 10.3 릴리스노트). 즉 정본 10.16 호스트에서 "Entropy calibration 200프레임"을 하려면 옛날처럼 빌더에 calibrator를 꽂는 게 아니라, **NVIDIA TensorRT Model Optimizer(ModelOpt)로 Entropy 캘리브레이션을 돌려 QDQ가 박힌 ONNX를 만든 뒤** 그걸 `trtexec`로 빌드한다(§4-4 경로 A2). DerryHub 경로(TRT 8.5)는 구현식 calibrator를 그대로 쓰므로 "옛 방식의 참고 구현"으로 남는다. 출처: [TensorRT 10.3 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.3.0.html), [Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html).

---

## 4) 단계별 실습

전체 파이프라인. 각 단계가 **앞 문서의 어떤 산출물을 쓰는지** 화살표로 표시했다.

```
[nuScenes mini]                          ← 이 문서 §4-1
   │  create_data → .pkl
   ▼
[PyTorch FP32 baseline] ← 공식 checkpoint  ← §4-2 (mAP/NDS 재현)
   │  torch → onnx (deformable attn/grid_sample/bev_pool 처리)
   ▼
[ONNX]                  ← 2단계 기법        ← §4-3  (04_transformer_quantization.md)
   │  Entropy PTQ, 200 calib frames
   ▼
[INT8 engine/model]     ← 1·3단계 이론      ← §4-4  (03_quantization_theory.md)
   │  layer sensitivity → mixed precision
   ▼
[INT8+FP16 tuned]                          ← §4-5  → layer_sensitivity.csv
   │  4-target build
   ▼
[TensorRT | TIDL | QNN | DRP-AI]           ← 3·4단계 도구  ← §4-6 (05/06)
   │
   ▼
[dashboard + design_rules.md + failures.md] ← 5단계 인프라   ← §4-7 (07_infrastructure.md)
```

### 4-1. 데이터 준비 (nuScenes v1.0-mini)

**디렉토리 구조부터 이해한다.** v1.0-mini는 4GB, 10개 scene(장면)으로, 압축을 풀면 아래 4개 폴더가 나온다:

```
data/nuscenes/
├── maps/        # 지도 파일 (rasterized .png + vectorized .json)
├── samples/     # keyframe(=annotated) 센서 데이터: CAM_FRONT/, CAM_BACK/, LIDAR_TOP/ ...
├── sweeps/      # keyframe 사이 중간 프레임(annotation 없음) — temporal 모델용
└── v1.0-mini/   # 모든 메타데이터·어노테이션 JSON 테이블(sample.json, sample_data.json ...)
```

- **`samples/`** = 어노테이션이 붙은 키프레임. mAP 평가는 여기서 이뤄진다.
- **`sweeps/`** = 어노테이션 없는 중간 프레임. BEVFormer 같은 **temporal 모델**이 과거 프레임을 참조할 때 쓴다. devkit만 돌릴 거면 optional이지만, temporal BEV 모델을 제대로 재현하려면 필요.
- **`v1.0-mini/`** = 6개 카메라 + LiDAR + radar의 관계, ego pose, calibration을 담은 JSON 테이블 묶음. devkit이 이 테이블을 읽어 프레임을 조립한다.

```bash
# 1. mini 셋 다운로드(약 4GB) — 10개 scene. (nuscenes.org 로그인 후 받은 URL 사용)
cd ~/capstone-bev/BEVDet
mkdir -p data/nuscenes
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf v1.0-mini.tgz -C data/nuscenes   # data/nuscenes/{samples,sweeps,maps,v1.0-mini}
```

```python
# 2. devkit로 "진짜 로드되는지 + 한 장 시각화" 스모크 테스트 (scripts/peek_nuscenes.py)
#    보드 없이 데스크톱에서 실행. 이걸로 데이터 무결성을 먼저 확인한다.
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini',
                dataroot='data/nuscenes', verbose=True)
print("scenes:", len(nusc.scene))          # 기대: 10
print("samples:", len(nusc.sample))        # 기대: 약 400 (키프레임)

sample = nusc.sample[0]                     # 첫 키프레임
# 6개 카메라 + LiDAR가 한 프레임에 묶여 있는지 확인
for ch, tok in sample['data'].items():
    print(ch, tok[:8])                      # CAM_FRONT, CAM_BACK_LEFT, LIDAR_TOP ...

# 한 장 렌더링(주석 박스 포함) → PNG 저장. README에 넣을 대표 이미지.
nusc.render_sample_data(sample['data']['CAM_FRONT'],
                        out_path='reports/sample_cam_front.png')
```

```bash
# 3. mmdetection3d/BEVDet 전처리 → .pkl 어노테이션 생성 (cu116 컨테이너 안에서 실행)
python tools/create_data_bevdet.py     # BEVDet 전용 스크립트
# 산출물: data/nuscenes/bevdetv2-nuscenes_infos_{train,val}.pkl
```

> 💡 팁: mini는 train/val이 각각 몇 백 프레임 수준이라 학습이 몇 분~수십 분이면 끝난다. **학습은 옵션**이다 — 공식 full 체크포인트를 그냥 로드해 mini val로 평가만 해도 파이프라인 검증엔 충분하다. 학습을 굳이 한다면 "mini로 few-epoch fine-tune 후 mAP가 어떻게 변하나"를 블로그 소재로 삼을 수 있지만, 완주 조건은 아니다.

> ⚠️ 주의: `render_sample_data`가 `matplotlib` backend 문제로 헤드리스 환경에서 멈추면, 스크립트 맨 위에 `import matplotlib; matplotlib.use('Agg')`를 넣어라. 컨테이너/SSH 환경에서 흔한 함정이다.

**예상 산출물**: `data/nuscenes/*.pkl`, `reports/sample_cam_front.png`(대표 시각화), scene 10개 전처리 완료 로그.

### 4-2. PyTorch FP32 Baseline 재현

```bash
# 공식 체크포인트로 mini val 평가 (baseline 수치 확보). cu116 컨테이너 내부.
python tools/test.py \
  configs/bevdet/bevdet-r50.py \
  work_dirs/bevdet-r50.pth \
  --eval mAP           # nuScenes 지표(mAP/NDS/mATE...) 출력
```

**공식 체크포인트 획득 방법**: BEVDet dev3.0의 `README.md`/`docs/en/getting_started.md`에 각 config별 체크포인트(대개 Google Drive/Baidu 링크)가 표로 정리돼 있다. `bevdet-r50` 계열 `.pth`를 받아 `work_dirs/`에 둔다. (링크가 만료됐으면 mmdetection3d model zoo의 동등 config 체크포인트로 대체.)

**예상 출력(발췌, full 체크포인트 → mini val이라 값은 낮게 나올 수 있음)**:
```
mAP: 0.28xx   NDS: 0.35xx
mATE: 0.6x   mASE: 0.27   mAOE: 0.5x   mAVE: 0.9x   mAAE: 0.2x
```

```python
# reports/baseline_fp32.json 로 저장 (이후 모든 비교의 기준선)
import json
baseline = {"model": "bevdet-r50", "precision": "fp32",
            "split": "v1.0-mini/val", "mAP": 0.283, "NDS": 0.350,
            "mATE": 0.63, "mASE": 0.27, "mAOE": 0.54, "mAVE": 0.91, "mAAE": 0.21,
            "latency_p50_ms": 33.3}   # 자신의 실측으로 교체
json.dump(baseline, open("reports/baseline_fp32.json", "w"), indent=2)
```

> 이 수치가 **모든 비교의 기준선(baseline)**이다. "논문값"이 아니라 **"내 파이프라인의 FP32 값"**을 기준으로 삼는다(mini라서 논문보다 낮은 게 정상 — §5-0).

### 4-3. ONNX Export — 커스텀 op 처리 (2단계 기법)

여기가 첫 번째 관문이다. [2단계 Transformer 양자화](04_transformer_quantization.md)에서 배운 **op 대체·커스텀 심볼릭·QDQ 삽입** 기법을 그대로 쓴다. 목표는 `onnx_export_failures.md`(2단계 산출물명 유지)에 **무엇이 왜 막혔고 어떻게 우회했는지**를 남기는 것.

- **BEVDet 경로**: LSS의 `bev_pool_v2`가 표준 ONNX op가 아니다. 공식 배포 스크립트가 이 op를 TensorRT 플러그인으로 매핑해 준다.
- **BEVFormer 경로**: `MultiScaleDeformableAttention`, `grid_sample`이 문제. **mmdeploy의 mmdet3d 코드베이스는 deformable attention을 지원하지 않는다** → 표준 export가 막힌다(§4-4 경로 B에서 커스텀 플러그인으로 우회).

```bash
# BEVDet: 공식 스크립트가 torch→onnx→TRT를 한 번에 (FP16/INT8 옵션)
python tools/convert_bevdet_to_TRT.py \
  configs/bevdet/bevdet-r50.py \
  work_dirs/bevdet-r50.pth \
  work_dirs/ \
  --fuse-conv-bn --fp16          # 우선 FP16으로 export가 되는지부터 확인
# 산출물: work_dirs/bevdet-r50.onnx, work_dirs/bevdet-r50_fp16.engine
```

```python
# ONNX ↔ PyTorch parity 검증 (scripts/check_parity.py) — 반드시 통과시킬 것
# "export는 됐는데 결과가 틀린" 조용한 실패를 여기서 잡는다.
import numpy as np, onnxruntime as ort, torch

dummy = {...}                                  # 실제 6-cam 입력 텐서 dict
torch_out = model(**dummy).detach().cpu().numpy()

sess = ort.InferenceSession("onnx/bevdet-r50.onnx",
                            providers=["CUDAExecutionProvider"])
onnx_out = sess.run(None, {k: v.cpu().numpy() for k, v in dummy.items()})[0]

max_abs = np.abs(torch_out - onnx_out).max()
print("max |Δ| =", max_abs)                    # 기대: < 1e-3 (FP32 기준)
assert max_abs < 1e-2, "parity 실패 → 심볼릭/op 매핑 확인"
```

> 🔴 함정 (`grid_sample`): `F.grid_sample`은 여러 백엔드에서 **op 커버리지 1순위 실패 지점**이다. TensorRT는 자체 plugin/native로 처리 가능하지만, TFLite/TIDL/QNN 경로에선 자주 막힌다(§4-6, §6). ONNX opset은 **16 이상**을 써야 `GridSample` 표준 op가 나온다. `torch.onnx.export(..., opset_version=16)`을 반드시 명시하라 — 기본값이 낮으면 `grid_sample`이 아예 export되지 않는다.

> 💡 팁: parity가 깨지면 대부분 (a) opset이 낮아 op가 근사 대체됐거나, (b) `align_corners`/`padding_mode` 같은 `grid_sample` 인자가 export 시 무시됐거나, (c) 전처리(정규화·리사이즈)가 PyTorch와 ONNX 쪽에서 미묘하게 다른 경우다. 셋 다 `onnx_export_failures.md`에 적을 값진 사례다.

**예상 산출물**: `onnx/bevdet-r50.onnx` + `reports/onnx_parity.txt`(FP32 ONNX vs PyTorch 출력 오차) + `onnx_export_failures.md`(막힌 op·우회법 로그).

### 4-4. INT8 PTQ — Entropy Calibration, 200 프레임 (1·3단계 이론)

[1단계 양자화 이론](03_quantization_theory.md)의 **Entropy(KL) calibration**과 per-tensor/per-channel 개념을 여기서 실제로 돌린다. calibration 프레임은 mini train에서 **대표성 있게 200장**을 뽑는다(다양한 scene·시간대·주야). 200이라는 수는 "통계가 수렴하기엔 충분하고 시간은 아끼는" 경험적 균형점이다 — 너무 적으면(<50) 활성값 히스토그램이 불안정하고, 너무 많으면(>1000) 캘리브레이션만 오래 걸리고 mAP 개선은 미미하다.

**calibration 프레임 샘플링(대표성 확보)**:
```python
# scripts/select_calib_frames.py — 10 scene에 고르게 200장 배분
import json, random
random.seed(0)
infos = json.load(open("data/nuscenes/bevdetv2-nuscenes_infos_train.pkl.json"))  # 개념 예시
by_scene = {}                                   # scene별로 프레임 그룹핑
for f in infos["infos"]:
    by_scene.setdefault(f["scene_token"], []).append(f)
picked = []
per = max(1, 200 // len(by_scene))
for toks in by_scene.values():                  # 각 scene에서 균등 추출
    picked += random.sample(toks, min(per, len(toks)))
picked = picked[:200]
json.dump([f["token"] for f in picked], open("calib/calib_200.json", "w"))
```

#### 경로 A1 — BEVDet (공식 스크립트, TRT 8.x 구현식, 가장 쉬움)
```bash
python tools/convert_bevdet_to_TRT.py \
  configs/bevdet/bevdet-r50.py work_dirs/bevdet-r50.pth work_dirs/ \
  --fuse-conv-bn --int8          # PTQ 캘리브레이션 자동 수행(구현식 calibrator)
# 산출물: work_dirs/bevdet-r50_int8.engine
```

#### 경로 A2 — 정본 호스트(TRT 10.16, **명시식 QDQ**) 재빌드 (권장 최종 경로)
정본 CUDA 12.8 / TRT 10.16 호스트에서는 구현식 calibrator가 deprecated이므로(§3 주의), **NVIDIA TensorRT Model Optimizer(ModelOpt)로 Entropy 캘리브레이션을 돌려 QDQ ONNX를 만든 뒤** `trtexec`로 빌드한다.

```bash
# (1) ModelOpt 설치 (정본 호스트)
pip install nvidia-modelopt

# (3) QDQ가 박힌 ONNX를 trtexec로 INT8 빌드 (--int8; QDQ가 있으면 명시식으로 동작)
trtexec --onnx=onnx/bevdet-r50_qdq.onnx --int8 \
        --saveEngine=engines/bevdet_int8.engine --useCudaGraph
```
```python
# (2) ModelOpt로 Entropy PTQ → QDQ ONNX (scripts/ptq_int8.py 핵심부)
import modelopt.onnx.quantization as moq
moq.quantize(
    onnx_path="onnx/bevdet-r50.onnx",
    calibration_data="calib/calib_200.npz",   # 200프레임 전처리 입력
    calibration_method="entropy",             # 1단계에서 배운 KL/Entropy
    output_path="onnx/bevdet-r50_qdq.onnx",
)   # per-channel(weight)/per-tensor(activation) 정책은 ModelOpt 기본을 따름
```

#### 경로 B — BEVFormer (DerryHub 커스텀 플러그인, deformable attn INT8)
`DerryHub/BEVFormer_tensorrt`는 **Grid Sampler / Multi-scale Deformable Attention / Modulated Deformable Conv2d / BEV Pool V2 / Flash MHA** 커스텀 TensorRT 플러그인을 float/half/half2/**int8**로 제공한다. calibration은 **entropy + per-tensor**(구현식, TRT 8.5.1.7)를 쓴다. 실행은 `samples/<model>/<variant>/`의 셸 스크립트로 한다.

```bash
cd BEVFormer_tensorrt
# (1) PyTorch → ONNX
sh samples/bevformer/tiny/pth2onnx.sh -d 0
# (2) ONNX → TensorRT FP32 / FP16 엔진
sh samples/bevformer/tiny/onnx2trt.sh      -d 0     # FP32
sh samples/bevformer/tiny/onnx2trt_fp16.sh -d 0     # FP16
# (3) ONNX → INT8 엔진 (entropy PTQ, per-tensor) 및 INT8+FP16 하이브리드
sh samples/bevformer/tiny/onnx2trt_int8.sh      -d 0
sh samples/bevformer/tiny/onnx2trt_int8_fp16.sh -d 0
# ※ base 모델이면 tiny→base로 바꾸고, OOM 시 TensorRT 8.4.3.1 사용(§6)
```

> 💡 팁 (검증된 참고 수치, **full nuScenes** 기준 — DerryHub README):
> | 모델 | 구성 | NDS | mAP | FPS(배속) | 엔진 크기 | GPU mem |
> |------|------|-----|-----|-----------|-----------|---------|
> | BEVFormer-base | FP32(기준) | 0.517 | 0.416 | 1.5 | 265 MB | 5333 MB |
> | BEVFormer-base | FP16+INT8 plugin | 0.514 | 0.413 | **8.0 (5.33×)** | **131 MB (↓92%)** | **2429 MB (↓79%)** |
> | BEVFormer-tiny | FP32(기준) | 0.354 | 0.252 | ~38 | — | — |
> | BEVFormer-tiny | FP16+INT8 plugin | 0.351 | 0.249 | **107.4** | 52 MB | 1095 MB |
>
> "정확도 거의 유지(NDS −0.003) + 4~5배 가속 + 엔진 90%↓"가 이 프로젝트가 재현하려는 스토리다. **주의: 이건 full nuScenes 수치다.** mini로 돌리면 절대값은 낮아지지만 "FP32→INT8의 상대적 하락폭·가속비" 패턴은 재현돼야 한다 — 대시보드에는 절대값이 아니라 이 **패턴**을 강조하라. 출처: [BEVFormer_tensorrt README](https://github.com/DerryHub/BEVFormer_tensorrt/blob/main/README.md).

**예상 산출물**: `engines/*_int8.engine`, INT8 mAP/NDS(`reports/int8.json`), FP32 대비 하락폭, (경로 A2면) `onnx/*_qdq.onnx`.

### 4-5. 레이어 민감도 분석 → Mixed Precision 튜닝

INT8로 mAP가 많이 떨어지면(특히 BEVFormer), **어떤 레이어가 정확도를 깎아먹는지**를 찾아 그 레이어만 FP16으로 되돌린다. [1단계 이론](03_quantization_theory.md)의 민감도 분석과 [3단계 TensorRT](05_tensorrt.md)의 precision 제어를 결합한다. 산출물 파일명은 **`layer_sensitivity.csv`**(1단계와 동일)로 통일한다.

방법(택1 또는 병행):
1. **레이어별 INT8↔FP16 스윕**: 한 레이어씩 FP16으로 고정하고 mAP 변화를 측정 → 민감 레이어 랭킹. 정확하지만 레이어 수만큼 평가를 돌려야 해 느리다.
2. **NVIDIA TensorRT Model Optimizer / `polygraphy`** 로 레이어 출력의 양자화 오차(SQNR/MSE)를 계산해 상위 민감 레이어 자동 선별. 빠르고 스윕 없이 후보를 좁힌다. `polygraphy run --trt --onnxrt --atol/--rtol`로 레이어별 출력 발산을 비교하는 방식.

```python
# 개념 스니펫: 민감도 → precision 정책 결정 → layer_sensitivity.csv (핵심만)
import csv
sensitivity = {}                       # layer_name -> mAP drop when INT8
for layer in quantizable_layers:
    mAP_int8_here = eval_with_int8_only(layer)      # 이 레이어만 INT8, 나머지 FP16
    sensitivity[layer] = baseline_mAP - mAP_int8_here

# 1단계와 같은 파일명으로 저장 (재사용성)
with open("reports/layer_sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "mAP_drop", "rank"])
    for r, (l, d) in enumerate(sorted(sensitivity.items(),
                                      key=lambda x: -x[1]), 1):
        w.writerow([l, f"{d:.4f}", r])

# 상위 k개(가장 민감)만 FP16 유지, 나머지는 INT8
keep_fp16 = [l for l, d in sorted(sensitivity.items(),
             key=lambda x: -x[1])[:k]]
```

> 💡 팁: BEV 모델에서 민감한 곳은 대체로 **어텐션의 softmax/LayerNorm, view-transform(LSS/deformable) 직전후, 검출 헤드**다. 여기만 FP16으로 지켜도 mAP 대부분이 회복되는 경우가 많다. "INT8 100% → mixed"로 mAP를 N점 회복하며 latency는 소폭만 증가하는 곡선을 대시보드에 넣어라 — 이게 "전부 INT8이 답이 아니다"라는 블로그 4편의 핵심 그림이다.

> ⚠️ 주의(정본 10.16 경로): 명시식 QDQ 워크플로에서는 "특정 레이어만 FP16 유지"를 **그 레이어에 QDQ를 넣지 않는 것**으로 구현한다(ModelOpt의 `op_types_to_exclude`/노드 이름 필터). 구현식 시절의 `layer.precision = fp16` API와 개념이 다르니, ModelOpt 문서의 부분 양자화 예시를 따르라.

**예상 산출물**: `reports/layer_sensitivity.csv`(레이어별 mAP 기여, 1단계 산출물명 유지), mixed-precision 엔진, 회복된 mAP/NDS.

### 4-6. 4-Target 배포 (3·4단계 도구)

동일한 ONNX(또는 QDQ ONNX)를 4개 백엔드로 내보낸다. **TensorRT 하나는 반드시**, 나머지는 [4단계](06_multi_soc.md)에서 준비한 host emulator로 "돌려는 본다". 아래 표가 이 캡스톤의 **4-target 매트릭스**(4·5단계 산출물명 유지)의 뼈대다.

| 타깃 | 도구 | 실행 환경 | 이 프로젝트에서의 현실 | 참고 문서 |
|------|------|-----------|------------------------|-----------|
| **TensorRT (dGPU)** | `trtexec`/API (TRT 10.16) | RTX 호스트 | INT8/mixed 다 됨. 기준 배포 | [3단계](05_tensorrt.md) |
| **TensorRT (Orin GPU+DLA)** | JetPack TensorRT | Orin 보드 | GPU는 OK. **DLA는 어텐션 미지원→GPU fallback** | [3단계](05_tensorrt.md) |
| **TIDL (TI)** | `edgeai-tidl-tools` host emu | x86 host | 미지원 op는 **Cortex-A로 offload**(부분 가속) → 로그 | [4단계](06_multi_soc.md) |
| **QNN (Qualcomm)** | ORT **QNN EP** + **AI Hub** (QDQ) | x86 / AI Hub 디바이스 | QDQ ONNX로 op 커버리지 확인, attn 일부 fallback | [4단계](06_multi_soc.md) |
| **DRP-AI (Renesas)** | DRP-AI TVM (RUHMI) | RZ/V2H 또는 TVM host | 지원 op 범위 좁음 → 백본만/실패 로그 | [4단계](06_multi_soc.md) |

아래는 **각 타깃의 구체 명령/코드**다. 4단계에서 세팅한 툴을 여기서 캡스톤 모델에 적용한다.

**(1) TensorRT dGPU (정본 10.16) — 기준 배포 + 벤치**
```bash
# latency p50/p99 + 메모리 프로파일. --dumpProfile로 레이어별 시간까지.
trtexec --loadEngine=engines/bevdet_int8.engine \
        --iterations=200 --avgRuns=200 \
        --dumpProfile --exportTimes=reports/trt_dgpu_times.json
# 결과에서 GPU compute p50/p99(ms), throughput(qps)를 reports/trt_dgpu.json으로 정리
```

**(2) TensorRT Orin GPU + DLA — 명시적 분할**
```bash
# 백본은 DLA, 미지원(어텐션/헤드)은 GPU로 fallback 허용. 빌드 로그에서 분할 결과 확인.
trtexec --onnx=onnx/bevdet-r50_qdq.onnx --int8 \
        --useDLACore=0 --allowGPUFallback \
        --saveEngine=engines/bevdet_orin_dla.engine
# 빌드 로그의 "Layer(...) running on DLA / GPU" 를 grep 해 분할표를 reports/orin_dla_split.txt 로
```

**(3) TIDL (TI) — x86 host emulation, 미지원 op는 Cortex-A로 offload**
```python
# TI 포크 ONNX Runtime의 TIDLCompilationProvider로 컴파일(호스트 emu).
# 지원 op는 C7x/MMA로, 미지원 op는 자동으로 Cortex-A(ARM)로 떨어진다 → "부분 가속" 로그.
import onnxruntime as ort
so = ort.SessionOptions()
ep = [("TIDLCompilationProvider", {
        "tidl_tools_path": "/opt/edgeai-tidl-tools/tidl_tools",
        "artifacts_folder": "reports/tidl_artifacts",
        "tensor_bits": 8,                       # INT8
        "accuracy_level": 1,
        "debug_level": 1})]                     # 어떤 op가 offload됐는지 상세 로그
sess = ort.InferenceSession("onnx/bevdet-r50.onnx", so, providers=ep)
# 컴파일 로그에서 "Layer ... is not supported by TIDL, will run on ARM" 목록을 수집 → failures.md
```

**(4) QNN (Qualcomm) — ① 로컬 ORT QNN EP 로 op 커버리지 확인, ② AI Hub 로 실제 디바이스 프로파일**
```bash
# ① 로컬: QDQ ONNX를 QNN EP로 로드해 미지원 op 경고 수집(x86, HTP 시뮬)
python scripts/run_qnn_ep.py --onnx onnx/bevdet_qdq.onnx   # 콘솔 fallback 경고 수집
```
```python
# ② Qualcomm AI Hub: 클라우드로 compile→quantize→(qnn context)→profile
#    실제 Snapdragon 디바이스에서 latency/mem을 측정할 수 있는 유일한 무보드 경로.
import qai_hub as hub
dev = hub.Device("Samsung Galaxy S24 (Family)")
# (a) 소스→ONNX (이미 ONNX면 스킵 가능)
onnx_model = compile_onnx_job.get_target_model()
# (b) INT8 양자화 (calibration_data = 전처리된 200프레임 dict)
qjob = hub.submit_quantize_job(model=onnx_model, calibration_data=calib_dict,
                               weights_dtype=hub.QuantizeDtype.INT8,
                               activations_dtype=hub.QuantizeDtype.INT8)
qmodel = qjob.get_target_model()
# (c) QNN context binary 로 컴파일
cjob = hub.submit_compile_job(model=qmodel, device=dev,
        options="--target_runtime qnn_context_binary --quantize_io")
# (d) 실제 디바이스에서 프로파일(latency/mem)
pjob = hub.submit_profile_job(model=cjob.get_target_model(), device=dev)
```

**(5) DRP-AI (Renesas) — DRP-AI TVM(RUHMI)로 컴파일, 지원 op 좁음**
```bash
# renesas-rz/rzv_drp-ai_tvm. RZ/V2H 타깃. 지원 op 범위가 좁아 백본만 되는 경우가 흔하다.
cd $TVM_ROOT/tutorials/
python3 compile_onnx_model_quant.py \
    ../../onnx/bevdet-r50.onnx \
    -o bevdet_drpai \
    -t $SDK -d $TRANSLATOR -c $QUANTIZER \
    --images calib/calib_images/ \
    -s 1,3,256,704                     # BEV 백본 입력 shape (모델에 맞게)
# 산출물: deploy.json / deploy.params / deploy.so (+ preprocess/) → RZ/V2H로 복사해 실행
# 미지원 레이어는 컴파일이 거부/경고 → 어디까지 컴파일되는지를 failures.md에
```

> 🔴 함정 (DLA): Jetson Orin의 **DLA는 transformer 어텐션·dynamic shape·커스텀 CUDA 플러그인을 지원하지 않아 GPU로 fallback**된다(JetPack 6.2 기준). BEV 모델을 DLA에 "통째로" 올리려는 시도는 거의 실패한다 — **백본 CNN만 DLA, 어텐션/헤드는 GPU**로 나누는 게 현실적. 이 분할 실험 자체가 훌륭한 블로그 소재다. 빌드 로그의 DLA/GPU 분할표를 그대로 캡처하라. 출처: [Supported ONNX Operators on Orin DLA](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/operators/README.md), [TensorRT DLA GPU Fallback](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-runtime-configuration.html).

> 🔴 함정 (TIDL/QNN/DRP-AI): 이 세 백엔드에서 `grid_sample`·deformable attention이 안 올라가거나 ARM으로 떨어지는 것은 **정상이고 예상된 결과**다. 목표는 "성공"이 아니라 "**어디까지 되고 어디서 왜 막히는지**"를 `failures.md`에 기록하는 것. TIDL은 미지원 op를 **Cortex-A로 자동 offload**하므로 "전체가 죽는" 대신 "부분 가속 + ARM 폴백"이 되고(그 폴백 목록이 로그의 핵심), DRP-AI는 지원 op가 좁아 **백본만 컴파일**되는 경우가 많다. 안 되면 (a) 해당 op만 CPU/GPU로 빼는 hybrid, (b) 백본만 SoC에 올리기 로 대체하고 그 판단 근거를 적는다. 참고: TIDL의 `GridSample`은 지원 목록에 있으나 elem-type/구현 제약(`TIDL_ERROR_GRIDSAMPLE_*`)이 있어 실패할 수 있다 — 이 에러 코드를 로그에 남기면 훌륭한 근거가 된다.

**예상 산출물**: 타깃별 `reports/<target>.json`(latency/mem/mAP 또는 "미지원 op 목록"), `failures.md`(타깃별 실패·offload 로그).

### 4-7. 인프라·리포트 (5단계)

[5단계 인프라화](07_infrastructure.md)에서 만든 **자동 벤치 스크립트·리포트 템플릿·CI**를 그대로 붙인다. 모든 수치는 스크립트가 `reports/*.json`을 모아 대시보드 표/그래프로 렌더링하게 한다(수기 표 금지 — 재현성·신뢰성의 핵심).

```python
# scripts/bench_all.py 골격: 타깃별 json을 모아 하나의 DataFrame으로
import json, glob, pandas as pd
rows = [json.load(open(p)) for p in glob.glob("reports/*.json")]
df = pd.DataFrame(rows)                 # columns: model,target,precision,mAP,NDS,latency_p50,peak_mem
df["dmAP_vs_fp32"] = df["mAP"] - df.loc[df.precision=="fp32","mAP"].iloc[0]
df.to_csv("reports/summary.csv", index=False)
print(df.to_markdown(index=False))      # README에 붙일 표
```

---

## 5) 예시 / 결과 해석

### 5-0. mini 데이터의 지표 특성 (먼저 이걸 이해하라)

`v1.0-mini`는 **10개 scene**뿐이다. full nuScenes는 1000 scene이다. 그래서:

- **절대 mAP/NDS가 논문보다 크게 낮게 나온다** — 클래스 불균형이 심하고(어떤 클래스는 mini에 몇 개 없음), val 프레임이 적어 통계가 튄다. full 체크포인트를 mini val로 평가해도 논문값이 안 나온다. **이건 버그가 아니라 mini의 정상 특성**이다.
- **일부 클래스 AP가 0 또는 NaN**으로 뜰 수 있다(해당 클래스 인스턴스가 mini에 거의/전혀 없음). 평균을 낼 때 이 클래스들이 mAP를 끌어내린다.
- 그래서 baseline을 **"논문값"이 아니라 "내 파이프라인의 FP32 mini 값"**으로 정의하고, 모든 비교(FP16/INT8/mixed)를 **그 값 대비 상대 변화**로 말한다.

> ✅ 면접에서 이렇게 말하라: "mini는 파이프라인 검증용이라 절대 mAP는 낮지만, **FP32 대비 INT8이 mAP를 X%p 떨어뜨리고 latency를 Y% 줄이는 트레이드오프 패턴은 full과 동일하게 재현**됐습니다." 이게 mini를 쓴 것에 대한 완벽한 방어다.

### 5-1. 성능 대시보드 예시 (표)

> 아래는 **형식 예시**다(수치는 자신의 실측으로 채운다). full 체크포인트→mini val이면 mAP는 예시보다 낮게 나올 수 있다(§5-0).

| 구성 | 타깃 | precision | mAP | NDS | latency p50 (ms) | peak mem (MB) | vs FP32 mAP | 비고 |
|------|------|-----------|-----|-----|------|------|------|------|
| BEVDet-R50 | PyTorch | FP32 | 0.283 | 0.350 | 33.3 | — | baseline | 기준 |
| BEVDet-R50 | TRT dGPU | FP16 | 0.282 | 0.349 | ~11 | ↓ | −0.001 | 사실상 무손실 |
| BEVDet-R50 | TRT dGPU | INT8 | 0.27x | 0.34x | ~7 | ↓↓ | −0.0x | QDQ(ModelOpt) |
| BEVDet-R50 | TRT dGPU | INT8+FP16(mixed) | 0.28x | 0.35x | ~8 | ↓↓ | −0.00x | 민감층 FP16 회복 |
| BEVDet-R50 | TRT Orin GPU | INT8 | 0.27x | — | 측정 | 측정 | — | DLA는 fallback |
| BEVDet-R50 | QNN(AI Hub) | INT8/QDQ | — | — | 측정 | 측정 | — | S24 프로파일 |
| BEVDet-R50 | TIDL host | INT8 | — | — | — | — | — | 일부 op ARM offload(로그) |
| BEVDet-R50 | DRP-AI TVM | INT8 | — | — | — | — | — | 백본만/실패 로그 |

> 해석 포인트(블로그·면접에서 말할 스토리):
> 1. **FP16은 거의 무손실**인데 latency 3배 ↓ → "먼저 FP16부터"가 왜 정석인지.
> 2. **INT8은 mAP를 얼마 떨어뜨리고** 대신 latency/mem을 얼마나 더 줄이나 → 트레이드오프.
> 3. **mixed precision이 민감 레이어 몇 개만 FP16으로 되돌려** mAP를 회복 → "전부 INT8"이 답이 아님.
> 4. **타깃마다 op 지원이 달라** 같은 모델도 배포 난이도가 다름 → 하드웨어 인지 설계의 필요성.

### 5-2. 그래프로 넣을 것 + 생성 코드

- **Pareto 곡선**: x=latency, y=mAP. 각 (타깃×precision) 점을 찍어 최적 전선(frontier)을 그린다.
- **민감도 막대그래프**: 레이어별 INT8 mAP 하락폭 top-15 (`layer_sensitivity.csv`에서).
- **precision별 엔진 크기/메모리 막대**.

```python
# notebooks/dashboard.ipynb — 표 + Pareto 곡선 (matplotlib, 의존성 최소)
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("reports/summary.csv")

fig, ax = plt.subplots(figsize=(6,4))
for tgt, g in df.groupby("target"):
    ax.scatter(g["latency_p50"], g["mAP"], label=tgt, s=60)
    for _, r in g.iterrows():                       # 각 점에 precision 라벨
        ax.annotate(r["precision"], (r["latency_p50"], r["mAP"]),
                    fontsize=8, xytext=(3,3), textcoords="offset points")

# Pareto frontier(좌상단이 최적: 낮은 latency + 높은 mAP) 계산
pts = df.sort_values("latency_p50")[["latency_p50","mAP"]].values
front, best = [], -1
for x, y in pts:
    if y > best: front.append((x, y)); best = y
fx, fy = zip(*front)
ax.plot(fx, fy, "k--", alpha=.6, label="Pareto frontier")

ax.set_xlabel("latency p50 (ms)"); ax.set_ylabel("mAP")
ax.set_title("Accuracy vs Latency (target × precision)")
ax.legend(); fig.tight_layout()
fig.savefig("reports/pareto.png", dpi=150)          # README 최상단에 박을 그림
```

```python
# 민감도 top-15 막대그래프
s = pd.read_csv("reports/layer_sensitivity.csv").head(15)
fig, ax = plt.subplots(figsize=(6,5))
ax.barh(s["layer"][::-1], s["mAP_drop"][::-1])
ax.set_xlabel("mAP drop when this layer is INT8")
ax.set_title("Top-15 quantization-sensitive layers")
fig.tight_layout(); fig.savefig("reports/sensitivity_top15.png", dpi=150)
```

### 5-3. GitHub 리포 구조 (권장)

```
capstone-bev-quant/
├─ README.md                 # 목표·지표·결과 요약표·Pareto 곡선·재현법(한 페이지)
├─ docs/
│  ├─ design_rules.md        # "이 하드웨어엔 이 precision/이 op는 피하라" 규칙집
│  ├─ failures.md            # 타깃별 실패 로그(op·에러·우회)  ← 이게 차별점
│  └─ onnx_export_failures.md# 2단계 산출물명 유지: export에서 막힌 op·우회
├─ configs/                  # 모델/캘리브레이션 설정
├─ scripts/
│  ├─ peek_nuscenes.py       # devkit 로드/시각화 스모크
│  ├─ select_calib_frames.py # 200프레임 대표 샘플링
│  ├─ export_onnx.py
│  ├─ check_parity.py        # ONNX↔PyTorch 출력 오차
│  ├─ ptq_int8.py            # ModelOpt Entropy → QDQ ONNX (정본 경로)
│  ├─ sensitivity.py         # 레이어 민감도 → layer_sensitivity.csv
│  ├─ run_qnn_ep.py          # QNN EP op 커버리지
│  └─ bench_all.py           # 4-target 벤치 → reports/*.json → summary.csv
├─ reports/                  # baseline/int8/layer_sensitivity.csv/<target>.json + png
├─ notebooks/dashboard.ipynb # 표 + Pareto 곡선 + 민감도 막대
├─ docker/Dockerfile         # cu116 학습/export 스택 고정
└─ .github/workflows/ci.yml  # export→PTQ→bench 스모크 테스트(5단계 CI)
```

> 💡 팁: `README.md` 최상단에 **결과 요약표 1개 + Pareto 곡선 1장(`reports/pareto.png`)**을 박아라. 리크루터는 스크롤을 안 한다. `failures.md`는 "이 사람은 실제로 보드/emu에서 삽질해 봤다"는 유일한 증거다 — 숨기지 말고 자랑하라.

---

## 6) 흔한 오류와 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| mmcv/mmdet 설치가 CUDA 12.x/PyTorch 2.x에서 컴파일 실패 | BEV 스택이 mmcv 1.x(=PyTorch<2.0, CUDA<12.0)에 묶임 | **Docker로 cu116 스택 고정**(§3). 또는 `nabe2030/bevformer-blackwell` 패치 참고 |
| `convert_..._to_TRT.py`가 `bev_pool_v2` op에서 에러 | 커스텀 op 컴파일 안 됨 | BEVDet의 op 확장(`ops/`) 빌드 확인, `python setup.py develop` 재실행 |
| ONNX export 시 `MultiScaleDeformableAttention` 미지원 | mmdeploy가 mmdet3d의 deformable attn 미지원 | 표준 export 포기 → **DerryHub 커스텀 플러그인 경로**(§4-4 경로 B). `onnx_export_failures.md`에 기록 |
| `grid_sample` export 실패/결과 이상 | opset<16 또는 백엔드 미지원 | `torch.onnx.export(opset_version=16)` 명시. TIDL/QNN이면 미지원/제약일 수 있음 → `failures.md` 후 hybrid |
| ONNX parity 깨짐(export는 됐는데 결과 틀림) | `align_corners`/`padding_mode` 무시, 전처리 불일치 | §4-3 parity 스크립트로 재현, 전처리를 PyTorch와 1:1 일치시킴 |
| TensorRT-8.5.1.7에서 base 모델 OOM | 알려진 메모리 이슈(DerryHub 문서) | base는 **TensorRT 8.4.3.1** 사용. 또는 tiny/small로 축소 |
| 정본 10.16에서 `IInt8EntropyCalibrator2` API가 안 먹힘/deprecated 경고 | TRT 10.1+에서 구현식 INT8 deprecated | **ModelOpt로 QDQ ONNX 생성**(§4-4 경로 A2) 후 `trtexec --int8` |
| RTX(10.16)에서 만든 `.engine`이 Orin/8.5에서 로드 실패 | 엔진은 빌드 환경(TRT 버전·아키텍처) 종속 | 각 타깃에서 **ONNX부터 재빌드**(§3 함정) |
| DLA 지정했는데 GPU만 바쁨 | 어텐션/dynamic shape가 DLA 미지원→fallback | `--allowGPUFallback` + 백본만 DLA로 **명시적 분할**. fallback 로그를 결과로 |
| TIDL 컴파일이 통째로 안 되는 게 아니라 "느림" | 미지원 op가 Cortex-A로 offload돼 병목 | `debug_level=1`로 ARM offload 목록 확인 → 그 op만 빼거나 백본만 가속 |
| DRP-AI 컴파일이 특정 레이어에서 거부 | DRP-AI 지원 op 범위 좁음 | 백본만 컴파일해 부분 배포, 나머지는 실패 로그. `-s` 입력 shape 재확인 |
| INT8 mAP가 크게 폭락 | 어텐션/헤드가 양자화에 민감 | §4-5 민감도 분석 → 해당 레이어 FP16 mixed(QDQ 제외) |
| mini val mAP가 논문값과 딴판 | mini는 10 scene, SOTA 재현용 아님 | 정상(§5-0). baseline은 "내 파이프라인의 FP32 값"으로 정의 |

---

## 7) 산출물 (Deliverables)

- `README.md` — 목표·지표·결과 요약표·Pareto 곡선(`reports/pareto.png`)·재현 명령.
- `reports/baseline_fp32.json`, `int8.json`, `layer_sensitivity.csv`, `<target>.json`, `summary.csv` — 원시 측정치 + 집계.
- `notebooks/dashboard.ipynb` — 표 + Pareto + 민감도/메모리 그래프(생성 코드 §5-2).
- `docs/design_rules.md` — "타깃 X에선 op Y 피하라 / precision Z 권장" 규칙집(+ cu116 격리 vs blackwell 패치 선택 근거).
- `docs/failures.md` — **타깃별 실패 로그**(어떤 op, 어떤 에러 코드, 어떤 우회/offload). ← 핵심 차별점.
- `docs/onnx_export_failures.md` — 2단계 산출물명 유지: export 단계에서 막힌 op와 우회.
- `docker/Dockerfile` — cu116 학습/export 재현 환경.
- `.github/workflows/ci.yml` — export→PTQ→bench 스모크 테스트.
- 기술 블로그 5편 초안(§8, 각 편 상세 목차 포함).

---

## 8) 기술 블로그 5편 — 상세 목차

JD의 ①멀티카메라 3D검출 ②Transformer 배포 ③INT8 양자화 ④멀티 SoC 를 각 편이 하나씩 커버하도록 설계했다. 아래는 **각 편의 절 목차**다 — 그대로 소제목으로 쓰면 된다.

**1편 — "BEV detector를 처음부터: nuScenes mini로 BEVDet 재현기" (JD ①)**
1. 왜 BEV인가: 6-카메라 → 조감도 격자 → 3D 박스 (그림 1장)
2. nuScenes v1.0-mini 해부: samples/sweeps/maps/v1.0-mini 폴더의 의미
3. devkit로 첫 프레임 시각화(`render_sample_data`) — 대표 이미지
4. `create_data_bevdet.py`로 `.pkl` 만들기와 그때 터진 것들
5. 공식 체크포인트로 FP32 baseline: 논문값이 왜 안 나오나(mini 특성 §5-0)
6. 결론: "내 파이프라인의 FP32"를 기준선으로 정의하기

**2편 — "deformable attention을 ONNX로 내보내기: mmdeploy가 안 받아줄 때" (JD ②)**
1. `MultiScaleDeformableAttention`·`grid_sample`이 왜 표준 op가 아닌가
2. mmdeploy가 mmdet3d deformable attn을 지원하지 않는다는 벽
3. 우회 1: opset 16 + 커스텀 심볼릭으로 `grid_sample` 살리기
4. 우회 2: DerryHub 커스텀 TensorRT 플러그인(GridSampler/MS-DeformAttn) 경로
5. ONNX↔PyTorch parity: "조용한 실패"를 잡는 법(`check_parity.py`)
6. `onnx_export_failures.md`에 남긴 것들

**3편 — "BEV 모델 INT8 PTQ: Entropy calibration 200프레임과 mAP 손실" (JD ③)**
1. 왜 200프레임인가: 히스토그램 수렴 vs 시간 (경험적 균형)
2. 대표 프레임 샘플링: scene 균등 추출(`select_calib_frames.py`)
3. Entropy(KL) calibration 직관: 활성값 분포를 얼마나 자르나
4. **TRT 10.x의 변화**: 구현식 calibrator deprecated → ModelOpt QDQ 워크플로
5. INT8 mAP/NDS 손실 측정과 해석(NDS가 mAP보다 더/덜 떨어지는 이유)
6. DerryHub 참고 수치(full)와 내 mini 수치의 패턴 비교

**4편 — "민감도로 고르는 mixed precision: 전부 INT8이 답이 아니다" (JD ③)**
1. INT8 100%의 함정: 어떤 레이어가 mAP를 다 깎아먹나
2. 민감도 측정 두 방법: 레이어 스윕 vs polygraphy/ModelOpt 오차
3. `layer_sensitivity.csv` 읽는 법 + top-15 막대그래프
4. BEV에서 민감한 곳: softmax/LayerNorm, view-transform 전후, 검출 헤드
5. QDQ 부분 양자화로 민감층만 FP16 되돌리기(`op_types_to_exclude`)
6. 회복 곡선: mAP N점 회복 vs latency 소폭 증가 (Pareto 위에서)

**5편 — "같은 모델, 4개 SoC: TensorRT/TIDL/QNN/DRP-AI 배포 실패 로그" (JD ④)**
1. 4-target 매트릭스 한 장으로 보기
2. TensorRT dGPU(10.16) 기준선 + Orin DLA/GPU 명시적 분할 실험
3. TIDL: 미지원 op의 Cortex-A offload와 `TIDL_ERROR_GRIDSAMPLE_*`
4. QNN: ORT QNN EP op 커버리지 + AI Hub로 실제 S24 프로파일
5. DRP-AI TVM(RUHMI): 좁은 op 범위, 백본만 컴파일되는 현실
6. `design_rules.md`로 정리: 하드웨어 인지 설계 5원칙

> 💡 팁: 5편을 **하나의 GitHub 리포로 연결**하고, 각 글 말미에 다음 글로 링크를 걸면 "시리즈"가 된다. 시리즈는 단발 글보다 훨씬 강한 시그널이다. 각 편 맨 위에 "이 글의 코드는 리포의 `scripts/xxx.py`" 링크를 달아 글↔코드를 왕복하게 하라.

---

## 9) 마일스톤 체크리스트 & "완주 판정" 기준

### 주차별 마일스톤 (예시, 3~5주)

- [ ] **W1**: 환경 이원화(cu116 학습 컨테이너 + 정본 10.16 배포 호스트) + nuScenes mini 다운로드·디렉토리 확인·devkit 시각화 + `.pkl` 전처리 + FP32 baseline 재현(`baseline_fp32.json`) (§3, §4-1~2).
- [ ] **W2**: ONNX export 성공 + opset16 `grid_sample`/`bev_pool` 처리 + PyTorch parity 통과 + FP16 엔진 + `onnx_export_failures.md` (§4-3).
- [ ] **W3**: 200프레임 캘리브 샘플링 + INT8 PTQ(구현식 A1 또는 ModelOpt QDQ A2) + 민감도 분석 → `layer_sensitivity.csv` + mixed precision 회복(§4-4~5).
- [ ] **W4**: TensorRT(dGPU 10.16) 완성 + 나머지 타깃 최소 1개 시도(Orin DLA / TIDL / QNN AI Hub / DRP-AI 중) + `failures.md` (§4-6).
- [ ] **W5**: 대시보드(표+Pareto, `pareto.png`)·`design_rules.md`·CI + GitHub 공개 + 블로그 초안 5편(§4-7, §5-2, §8).

### 완주 판정 기준 (이 4개면 "완주"로 인정) — JD ①②③④ 매핑

- [ ] **① 파이프라인 관통 (↔ JD ① 멀티카메라 3D검출 + 전체 흐름)**: 선택 모델이 PyTorch→ONNX→INT8까지 **끊김 없이** 흐르고, 각 단계 산출물이 `reports/`에 있다. nuScenes 6-카메라 입력으로 3D 박스가 나오는 baseline이 재현됐다.
- [ ] **② 정확도-성능 4차원 표 (↔ JD ③ INT8 양자화)**: 최소 (FP32, FP16, INT8, mixed) × (TensorRT dGPU) 가 mAP/NDS/latency/mem으로 채워졌고, INT8→mixed 회복 곡선이 Pareto에 그려졌다.
- [ ] **③ 멀티 타깃 증거 (↔ JD ④ 멀티 SoC + JD ② Transformer 배포)**: TensorRT 외 **최소 1개** 타깃에 대해 "성공 수치" 또는 "미지원 op/offload를 명시한 실패 로그"가 있다. deformable attn/`grid_sample`이 각 백엔드에서 어떻게 처리/거부되는지가 기록됐다.
- [ ] **④ 공개 + 서사 (↔ JD 전체 커뮤니케이션)**: GitHub 리포(README 요약표 + Pareto + `failures.md` + `design_rules.md`)와 블로그 최소 1편이 공개됐다.

> ✅ ①②③④를 채우면 JD의 ①②③④를 그대로 커버한다. 매핑을 명시하면(리포 README에 "이 항목이 JD의 이 요구를 증명한다" 표를 넣으면) 리크루터가 매칭을 대신 안 해도 된다. **4개 타깃 전부 성공은 보너스**지, 완주 조건이 아니다. "TensorRT는 됐고, TIDL/QNN/DRP-AI는 여기까지 되고 여기서 이 op 때문에 막혔다"를 **논리적으로 설명할 수 있으면** 이미 지원 가능한 상태다.

> 🔴 함정: 완벽주의로 W1에서 몇 주를 태우지 마라. **먼저 BEVDet FP16을 dGPU에서 끝까지 관통**시켜 파이프라인 뼈대(walking skeleton)를 만든 뒤, INT8·민감도·멀티타깃을 붙여 나가는 순서가 안전하다. "동작하는 얇은 관통"이 "완벽한 한 단계"보다 낫다. W1~W2에서 막히면 BEVFormer 도전을 접고 BEVDet 하나로 완주부터 확정하라.

---

## 10) 참고 사이트 & 참고문헌

### 공식 문서 / 도구 / 코드

- [BEVDet 공식 repo (HuangJunJie2017, dev3.0)](https://github.com/HuangJunJie2017/BEVDet) — LSS+BEVPoolv2, `convert_bevdet_to_TRT.py`로 FP16/INT8 TensorRT 변환 지원. 체크포인트·getting_started 문서 포함.
- [PETR 공식 repo (megvii-research)](https://github.com/megvii-research/PETR) — detr3d 기반, PETR/PETRv2. 표준 attention 비중이 커 상대적으로 배포 우호적.
- [BEVFormer 공식 repo (fundamentalvision)](https://github.com/fundamentalvision/BEVFormer) — deformable attention 기반. tiny는 경량 spatial-only 변형.
- [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) — BEVFormer/BEVDet TensorRT 배포 + **INT8(entropy PTQ, per-tensor)** + 커스텀 플러그인(GridSampler/MS-DeformAttn/ModulatedDeformConv2d/BEVPoolV2/FlashMHA, float/half/half2/int8). `samples/<model>/<variant>/*.sh`로 실행. 배포·INT8의 사실상 레퍼런스. (요구 스택: CUDA 11.6 / TRT 8.5.1.7 / mmcv 1.5.0 / mmdet 2.25.1 / mmdeploy 0.10.0)
- [mmdeploy — mmdet3d 배포 문서](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmdet3d.html) — 지원/미지원 범위(deformable attn 미지원 등) 확인용.
- [nuScenes 데이터/미니셋](https://www.nuscenes.org/) · [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) — v1.0-mini 다운로드·평가·시각화(`NuScenes`, `render_sample_data`).
- [NVIDIA TensorRT 10.16 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.16.0.html) · [TensorRT 10.3 Release Notes(구현식 INT8 deprecation 고지)](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.3.0.html) · [Working with Quantized Types(명시식 QDQ)](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) — 정본 CUDA 12.8 / TRT 10.16 스택과 QDQ 워크플로 근거.
- [NVIDIA TensorRT Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) — Entropy PTQ로 QDQ ONNX 생성(정본 INT8 경로 A2).
- [Supported ONNX Operators & Functions on Orin DLA](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/operators/README.md) · [TensorRT DLA GPU Fallback 문서](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-runtime-configuration.html) — DLA 미지원 op / fallback 근거(어텐션·dynamic shape·커스텀 플러그인 미지원).
- [TexasInstruments/edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) · [supported_ops_rts_versions.md](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/supported_ops_rts_versions.md) — TIDL 지원 op·`TIDLCompilationProvider`/`TIDLExecutionProvider`·미지원 op의 Cortex-A offload.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) · [AI Hub Quantization 문서](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html) · [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) · [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) — QDQ ONNX → AIMET 양자화 → `qnn_context_binary` 컴파일 → 실제 디바이스 프로파일.
- [renesas-rz/rzv_drp-ai_tvm](https://github.com/renesas-rz/rzv_drp-ai_tvm) · [RZ/V2H 튜토리얼](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/tutorials/tutorial_RZV2H.md) · [RUHMI 컴파일러 포털](https://renesas-rz.github.io/rzv_drp-ai_tvm/) — `compile_onnx_model_quant.py`로 ONNX→DRP-AI 컴파일(`deploy.json/params/so`).
- [nabe2030/bevformer-blackwell](https://github.com/nabe2030/bevformer-blackwell) — mmcv 1.x를 PyTorch 2.x/CUDA 12.x(Blackwell 포함)에 올리는 커뮤니티 패치(2026 재현 난점 우회 대안 (b)).

### 논문 (arXiv)

- Huang et al. (2021), *BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View*, arXiv:2112.11790.
- Liu et al. (2022), *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*, arXiv:2203.05625.
- Li et al. (2022), *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*, arXiv:2203.17270.
- Zhu et al. (2020), *Deformable DETR: Deformable Transformers for End-to-End Object Detection*, arXiv:2010.04159. — deformable attention의 원류(BEVFormer가 차용).
- Philion & Fidler (2020), *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D*, arXiv:2008.05711. — BEVDet의 LSS 뷰 변환 원류.
- Gholami et al. (2021), *A Survey of Quantization Methods for Efficient NN Inference*, arXiv:2103.13630.
- Nagel et al. (2021), *A White Paper on Neural Network Quantization*, arXiv:2106.08295.
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Integer-Arithmetic-Only Inference*, arXiv:1712.05877.

> ⚠️ 확인 필요: 위 mAP/NDS·FPS·메모리 수치는 **full nuScenes** 기준 공식/repo값이다. 실제 캡스톤은 `v1.0-mini`로 돌리므로 값이 다르게 나오는 것이 정상이며(§5-0), 자신의 실측으로 대시보드를 채워야 한다. DerryHub/BEVDet의 정확한 TensorRT·mmcv 버전, TIDL/QNN/DRP-AI의 지원 op 목록은 repo·SDK가 갱신될 수 있으니 클론/설치 시 각 README·supported-ops 문서를 재확인할 것. TensorRT 10.16은 CUDA 12.9 빌드가 CUDA 12.x(=12.8 포함) 전체와 호환된다 — 정본 12.8 호스트에서 동작하나, 세부 패치는 릴리스노트로 확인.

---

## 11) 다음 단계

전체 여정을 시간표로 묶는다.

- 다음: [12주 로드맵](09_roadmap.md) — 1~5단계 + 캡스톤을 12주에 배치하는 학습 계획.
- 이전: [5단계 인프라화](07_infrastructure.md) — 이 캡스톤이 쓰는 벤치 자동화·리포트·CI의 출처.
- 함정 모음: [함정 5개](10_pitfalls.md) — 여기서 만난 op 미지원·DLA fallback·구현식 INT8 deprecation 등을 재정리.
