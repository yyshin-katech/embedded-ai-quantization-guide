# 인수인계 — 다른 PC에서 이어서 하기

이 문서 하나만 읽으면 다른 머신에서 작업을 이어받을 수 있게 쓴 것입니다. 가이드 본문의 학습
내용이 아니라 **작업 상태**를 다룹니다. 학습을 시작하려는 것이면 [`study_guide/README.md`](study_guide/README.md)로 가세요.

- 기준 커밋: `f7ca2a1` (2026-08-17) · 마지막 갱신: 2026-08-17 (2단계 DETR + BEVFormer §4.6 실측 완주)
- 이전 작업 머신: `Ubuntu 22.04.5` / `RTX 3060 12GB` / 드라이버 `595.84` (Neousys Nuvo-6108GC) — GPU 고장(§6)
- 현재 작업 머신: `Ubuntu 22.04` / `RTX 3080 10GB` / 드라이버 `595.84` — QAT 회복 실험을 여기서 완주(GPU 이상 없음)

---

## 1. 30초 요약

문서는 **0·0.5·1·2단계까지 실제 머신에서 완주·검증하고 커밋 완료**입니다. 2단계(Transformer
양자화)는 DETR INT8로 완주했고(커밋 `41dc49e`), **§4.6 BEVFormer-tiny도 실측 완주**했습니다 —
무컴파일 레거시 venv(torch 1.13+mmcv 1.7 프리빌트 휠)로 `fundamentalvision/BEVFormer`를 돌려
FP32 nuScenes-mini **mAP 0.2647**(81샘플 스모크·문헌비교 불가), op 단정은 **반전 0건**(초안이
맞음)·실전 함정 **2건**(mmcv 커스텀 op은 CPU에서만 유효 export·전체 export는 `point_sampling`에서
사망)을 확인했습니다. 전체 INT8은 유효 export가 안 나와 **범위 밖**(포크 필요).

1단계에서 파생된 **QAT 회복 2팔 실험**은 이제 **완주**했습니다(2026-08-15, RTX 3080, 커밋
`10461f2`). 이전 머신 GPU 고장으로 막혀 있던 대조군을 정상 카드에서 돌렸고, 무손실급 W8A8
대신 **weight 4-bit 손실변형(W4A8)**으로 회복률을 처음 측정했습니다: FP32 68.51% → PTQ
44.35%(**−24.16%p**) → QAT 67.81% = **97.1% 회복**, 단 동일 학습 대조군보다 **−1.50%p**
낮음(4-bit 잔여 대가). → §5

**이 교착은 다른 PC(RTX 3080)로 옮겨 해소됐습니다.** 고장은 특정 카드/섀시 증상이었고(§6),
3080은 QAT 워크로드를 300W·95%·70°C로 완주했습니다 — SW Power Cap 점등 없음.

**남은 것:** ① §4.4 SmoothQuant(modelopt) 적용 ② BEVFormer 전체 INT8(포크 플러그인 재빌드 후) ③ 3~7단계·캡스톤. → §7

---

## 2. 저장소에 있는 것 / 없는 것

저장소는 46MB입니다. 실습 산출물 대부분은 **저장소 밖**에 있어 같이 옮겨야 합니다.

| 대상 | 크기 | git에 있나 | 새 PC에서 확보 방법 |
|---|---|---|---|
| 가이드 MD·HTML, 실행 로그, 하네스 | 46MB | ✅ | `git clone` |
| 작업 메모리 사본 | — | ✅ | [`.claude/memory/`](.claude/memory/) — 이 문서보다 상세한 실측 기록 |
| QAT 실험 (완결) | 26KB | ✅ | [`experiments/qat_recovery/`](experiments/qat_recovery/) — 스크립트 2 + 실행 로그 2 + 결과 JSON + 완결 README |
| `~/stage1-work/*.py` (1단계 작업 스크립트 28개) | 0.7MB | ❌ | 아래 tar로 전송 (§3) |
| ImageNet val 50,000장 (`data/`) | **27GB** | ❌ | 전송 또는 재다운로드 (§4) |
| 전처리 캐시 (`data/cache/*.npy`) | 15GB | ❌ | 전송하거나 `prep_cache.py`로 재생성(수십 분) |
| venv `~/emb-ai` | 14GB | ❌ | **전송하지 말고 재설치** (§3) |
| 논문 PDF | 44MB | ❌ | `python3 paper/fetch_papers.py` (저작권상 재배포 안 함) |

---

## 3. 새 PC 세팅

### 3-1. 저장소

```bash
git clone https://github.com/yyshin-katech/embedded-ai-quantization-guide.git
cd embedded-ai-quantization-guide
```

### 3-2. 환경 (0단계)

정본은 [`study_guide/01_environment_setup.md`](study_guide/01_environment_setup.md)입니다. **venv는 복사하지 말고 새로 만드세요** —
14GB인데다 CUDA 라이브러리 경로가 절대경로로 박혀 있어 옮기면 조용히 깨집니다.

이전 머신에서 실측으로 확정된 스택(가이드 원안과 다른 부분이 있으니 그대로 쓸 것):

| 도구 | 버전 | 이유 |
|---|---|---|
| onnx | **1.18.0** | ORT 1.23.2의 IR 상한이 11. 최신 onnx(IR 13)는 **로드 실패** |
| onnxruntime-gpu | **1.23.2** | `pip install "onnxruntime-gpu<1.27"`. 가이드 원안의 `==1.28.0`은 **존재하지 않는 버전** |
| numpy | **1.26.4** | `nuscenes-devkit`이 `numpy<2` 요구 |
| onnxscript | 0.7.1 | torch 2.11의 `torch.onnx.export`는 기본 `dynamo=True`이고 이걸 요구. 빠지면 3·4·5단계 export가 첫 줄에서 죽음 |
| tensorrt-cu12 | 10.16.1.11 | 11.x는 `--int8/--fp16` 제거(strongly-typed) |

**반드시 같이 하는 픽스 2개** — 안 하면 ONNX Runtime이 **조용히 CPU로 폴백해 틀린 성능
수치를 뱉습니다**(에러가 안 납니다). `~/emb-ai/bin/activate`에 `LD_LIBRARY_PATH` 블록을
넣어 해결합니다. 절차는 `01_environment_setup.md` **3-4-a절**:

1. `libcudnn.so.9` → venv의 `nvidia/*/lib`
2. `libnvinfer.so.10` → venv의 `tensorrt_libs/` (`nvidia/*/lib` 글롭 **밖**이라 별도 처리)

검증: `python3 verify_trt_ep.py`가 `TensorrtExecutionProvider`를 **실제 활성 EP**로 잡아야
합니다. provider 목록에 이름만 뜨는 것으로는 부족합니다.

### 3-3. 1단계 작업 스크립트 옮기기

이전 머신에서:

```bash
tar czf stage1-scripts.tgz -C ~/stage1-work \
    --exclude=data --exclude=models_v2 --exclude=__pycache__ \
    --exclude='*.onnx' --exclude='*.onnx.data' .
# 0.7MB. eval_hits_v2.npz(예측 hit 배열), layer_sensitivity*.csv, *.log, telemetry_bs48.csv 포함
```

새 머신에서 `~/stage1-work/`에 풀면 됩니다. 스크립트는 `~/stage1-work`를 `os.path.expanduser`로
잡으므로 홈 아래 같은 이름이면 경로 수정이 필요 없습니다.

`models_v2/`(148MB, 양자화된 ONNX 모델들)는 `quantize_v2.py`로 재생성할 수 있으니 굳이
옮기지 않아도 됩니다.

---

## 4. 데이터 — ImageNet val 50,000장

**정확도 주장을 하려면 이 데이터가 필요합니다.** 1단계에서 배운 가장 비싼 교훈이 여기서
나왔습니다: 클래스당 1장 큐레이션 셋(1,000장)으로 재면 절대 top-1이 **평균 +9.77%p
부풀려지고, Δ의 부호가 3건·유의성 판정이 5건 뒤집힙니다**. 자세한 건
[`10_pitfalls.md` 함정 0](study_guide/10_pitfalls.md).

| 항목 | 값 |
|---|---|
| `ILSVRC2012_img_val.tar` | 6,744,924,160 B · MD5 `29b22e2961454d5413ddabcf34fc5622` (공식값 일치) |
| 출처 | 이전 머신에는 사용자 개인 공유 링크로 받아둠. 새로 받으려면 [image-net.org](https://image-net.org/download.php) 가입 후 ILSVRC2012 validation |

받은 뒤 순서:

```bash
python3 restructure_val.py   # tar → data/val_full/<wnid>/*.JPEG (1000×50)
python3 verify_labels.py     # devkit 유도 라벨 vs Caffe 배포 라벨 50,000건 교차검증
python3 prep_cache.py        # 전처리 캐시 2종 생성 (각 7.1GB, NHWC uint8 memmap)
```

**전처리 2종을 반드시 구분하세요** — 라벨 규약은 **정렬된 WNID 0-based**(torchvision
`ImageFolder` 규약)입니다. 다른 규약을 섞으면 top-1이 0.1% 근처로 무너집니다.

| 캐시 | 방식 | ResNet18 FP32 top-1 |
|---|---|---|
| `crop_tv` | 짧은 변 256 bilinear + center crop 224 (**torchvision 공식**) | **69.81%** — 공개값 69.758%와 0.05%p 일치 |
| `crop_squash` | 종횡비 무시 256×256 + `[16:240]` (가이드 본문 `preprocess()`) | −1.07%p (p=1.6e-14) |

`crop_squash`의 −1.07%p는 **양자화 손실(−0.12%p)의 약 9배**입니다. 정확도 주장에는 `crop_tv`를
쓰고, FP32로 공식값을 재현한 뒤에 양자화를 켜세요.

---

## 5. QAT 회복 실험 — 완주 (2026-08-15, RTX 3080)

> **상태:** 미완이던 2팔 실험을 완주했습니다(커밋 `10461f2`). 렌더된 요약은
> [`logs/stage1_50k_rerun_reproduction_report.html` §11](logs/stage1_50k_rerun_reproduction_report.html),
> 실험 상세·재실행법은 [`experiments/qat_recovery/README.md`](experiments/qat_recovery/README.md).

### 무엇을 왜 하는가

`03_quantization_theory.md` §2.5는 QAT/STE를 설명만 하고, 실습은 합성 텐서로 gradient가
통과하는지만 봅니다. val 50,000장이 있으니 **"QAT는 PTQ가 잃은 정확도를 되찾는다"는 주장
자체를 측정**할 수 있습니다.

ImageNet train split이 없어 **val을 쪼개서** 학습합니다 — 클래스당 40장 학습 / 10장 평가,
`rng(0)`, 서로소 assert. 평가셋 누수는 없지만 val 분포로 학습한 모델이라 **절대 top-1을
문헌값과 비교하면 안 됩니다.** 상대 관계만 유효합니다.

### 2팔 설계가 핵심입니다

| 팔 | 스크립트 | 내용 |
|---|---|---|
| 1 | `qat_recovery.py` | FP32 → PTQ(fake-quant, 학습 없음) → QAT(STE 2에폭) |
| 2 | `qat_control_finetune.py` | **fake-quant만 제거**하고 분할·optimizer·LR·schedule·배치·에폭·전처리 전부 동일한 FP32 파인튜닝 |

팔 1만 보면 QAT가 FP32를 이깁니다(BS=96에서 69.13% vs 68.52%, +0.61%p). 하지만 **QAT 팔만
val 40,000장으로 2에폭 추가 학습을 받았습니다.** 그 이득이 '양자화 인식'인지 '그냥 더 학습한
것'인지 구분되지 않습니다 — 50k 재실행에서 정정한 "INT8이 FP32보다 +0.40%p 낫다"(실제로는
p=0.48 노이즈)와 **구조가 같은 오류**입니다.

> 🔴 **설정에 따라 읽는 값이 다릅니다.** 무손실 W8A8에선 PTQ 손실이 0.06~0.14%p(노이즈)라
> 회복률이 `1116.7%` 같은 무의미한 수가 되어 **`QAT − 대조군` 격차만** 읽습니다. 손실변형
> **W4A8(`QAT_WBITS=4`)**에선 PTQ 손실이 −24.16%p라 **회복률(97.1%)이 처음 노이즈 위에서
> 측정**됩니다 — 그래도 최종 판정은 여전히 `QAT − 대조군`(−1.50%p)입니다.

### 실측된 값 (평가 10,000장)

| 설정 | FP32 | PTQ | QAT ep0 | QAT ep1 | 대조군 | 회복률 |
|---|---|---|---|---|---|---|
| **W4A8 · BS=48 / calib 5,000** ✅ | 68.51% | **44.35%** | 67.59% | **67.81%** | **69.31%** | **97.1%** |
| W8A8 · BS=96 / calib 5,000 (부분) | 68.52% | 68.46% | 68.89% | 69.13% | ❌ 미실행 | 무의미 |
| W8A8 · BS=48 / calib 2,560 (부분) | 68.51% | 68.37% | 69.47% | ❌ GPU 사망 | ❌ 미실행 | 무의미 |

첫 행(**W4A8**)이 이번에 완주한 정본입니다 — `QAT − 대조군 = 67.81 − 69.31 = −1.50%p`. 아래
두 W8A8 부분 실행은 이전 머신 것으로, 무손실급이라 회복률이 노이즈입니다(서로 비교 불가 — step
수 416→833로 2배, 캘리브 장수도 달라 아래 함정 3). BS=48에서 FP32가 68.51%로 일관 재현된 것은
분할·전처리·eval 모드가 같다는 확인입니다.

### 재실행

```bash
cd ~/stage1-work
cp <repo>/experiments/qat_recovery/*.py .    # 저장소 사본을 쓸 경우
nvidia-smi -L                                 # 🔴 GPU 생존 확인 먼저

# 완주한 정본 설정 — W4A8 (회복률이 읽힘)
QAT_WBITS=4 python3 qat_recovery.py         2>&1 | tee qat_recovery_bs48_w4.log   # 팔 1 (~2분, 3080)
ls qat_recovery_result_bs48_w4.json                                              # JSON 생성 확인
QAT_WBITS=4 python3 qat_control_finetune.py 2>&1 | tee qat_control_bs48_w4.log    # 팔 2 (~2분)
```

배치·에폭·weight 비트폭은 환경변수로 덮어씁니다: `QAT_WBITS=4 QAT_BS_TRAIN=48 QAT_EPOCHS=2`.
**두 팔을 반드시 같은 설정으로** 돌리세요 — 팔 2는 JSON의 `(bs_train, epochs, wbits)`가 자기
설정과 다르면 실행을 거부합니다(결과 파일명도 `..._bs{N}_w{W}.json`으로 분리됩니다).

### 이미 고쳐둔 함정 3가지

1. **`.eval()` 누락** — 대조군에서 빼먹으면 BatchNorm이 running stats 대신 배치 통계를 써
   학습 전 기준선이 68.52% → 68.04%로 어긋납니다. 지금은 `model.eval()` + ±0.05%p 가드가 있습니다.
2. **하드코딩 상수 → JSON 핸드오프** — 대조군이 `QAT_EP1 = 69.13`을 상수로 물고 있었습니다.
   배치를 바꾸면 BS=96 값과 BS=48 대조군을 조용히 비교해 오답을 냅니다. 3차 실행에서 이 가드가
   실제로 작동해 오비교를 막았습니다.
3. **캘리브 장수가 배치에 묶여 있었음** — `range(0, 20*BS_EVAL, BS_EVAL)`이라 `BS_EVAL`을
   250→128로 줄이자 관측 이미지가 5,000→2,560장으로 반토막 나 PTQ가 −0.09%p 움직였습니다.
   배치는 결과를 바꿔선 안 되는 노브입니다. `CALIB_N=5000`으로 분리 완료.

### 남은 설계 개선

기본 구성(per-channel 대칭 INT8 weight + per-tensor 비대칭 UINT8 act)은 PTQ 손실이 무손실급이라
**"회복할 것이 없다"**가 정직한 결론입니다. 회복률을 재려면 PTQ 손실이 큰 변형이 필요 — 후보
셋(Entropy 정규화 −9.45%p, Percentile 99.9 −6.83%p, **weight 4-bit**) 중 **weight 4-bit는
완주**했습니다(`QAT_WBITS=4`, 위 표). 나머지 둘은 같은 2팔 틀에서 `ActFakeQuant`만 바꿔
재사용할 수 있습니다(미착수).

---

## 6. 하드웨어 이슈 — 이전 머신의 GPU 이탈

**새 PC라면 해당 없을 수 있습니다.** 특정 카드/섀시 증상으로 보이며, 옮기는 것 자체가
해결책일 수 있습니다. 다만 같은 증상을 만나면 진단에 시간을 쓰지 않도록 정리해 둡니다.

> ✅ **확인됨(2026-08-15, RTX 3080):** 같은 QAT 워크로드(W4A8 · 2팔 각 2에폭)를 3080에서
> 돌렸을 때 **이탈 없이 완주**했습니다 — 300W/320W·95% util·70°C·boost 1950MHz 지속, SW Power
> Cap 점등 없음, 종료 후 정상 idle(55°C/23W). 아래 3060 증상은 **이전 머신 한정**입니다.

RTX 3060이 QAT 워크로드에서 PCIe 버스를 이탈했습니다 — **3회**(2026-08-04, 08-10 ×2).

| 회차 | 워크로드 | 사망 직전 온도/팬/전력 | 부하 후 생존 |
|---|---|---|---|
| 1 | `qat_recovery.py` BS=96 | 미측정 | — |
| 2 | 종료 직후 새 잡 | 74°C / 38% / 138W | ~40초 |
| 3 | **BS=48** ep1 step 500/833 | **78°C / 43% / 129.7W** | ~190초 |

**시그니처:** `journalctl -k`에 `Xid 79 "GPU has fallen off the bus"` → 즉시
`Xid 154 "Node Reboot Required"`. `lspci`가 `(rev ff)`(config space 무응답), `nvidia-smi`는
"Unable to determine the device handle". 파이썬은 `CUDA error: unspecified launch failure`.
**복구는 재부팅뿐**입니다(모듈 리로드 불가 — config space가 죽어 커널이 장치를 다시 못 엶).

이 머신은 `kernel.dmesg_restrict=1`이라 `dmesg` 대신 **`journalctl -k`**를 씁니다.

**시도했고 실패한 것 — 배치 축소.** BS 96→48로 반감해도 최대 전력은 138W → 129.7W(**약 8W**)
만 줄었습니다. 배치가 작아지면 GPU가 커널을 더 자주 띄워 빈자리를 메웁니다. 생존 시간만
40초 → 190초로 늘고 결국 죽었습니다. **배치는 전력 레버가 아닙니다.**

**3차에서 얻은 진짜 단서 — SW Power Cap 상시 점등.** 2초 텔레메트리 98행 중 **73행이
`0x4`(SW Power Cap)**. 부하 26초 뒤(123W/53°C) 전력 상한에 걸려 죽을 때까지 **164초 내내
~130W에 붙어** 있었습니다. 2차가 `0x0`뿐이었던 건 40초에 죽어 상한 상태에 도달할 시간이
없었기 때문입니다. RTX 3060 정격 TGP는 170W인데 ~130W에서 걸리는 건 임베디드 섀시 전력
예산에 맞춘 OEM 설정 가능성이 있습니다.

**thermal 플래그(`0x20`/`0x40`/`0x80`)는 3회 모두 0**이라 드라이버는 열로 보지 않습니다.
다만 3차는 **70°C 이상 54행에서 평균 76°C인데 팬이 평균 39%**였습니다 — 온도에 팬 커브가
반응하지 않습니다. 밀폐 산업용 박스에서 정상으로 읽을 값은 아닙니다.

**다음에 시도할 것 (둘 다 sudo):**

```bash
sudo nvidia-bug-report.sh          # 🔴 재부팅 전에! 모듈 언로드되면 덤프가 사라짐
nvidia-smi -q -d POWER             # enforced / min / max 상한 확인
sudo nvidia-smi -pl 100            # 상한을 직접 내림 — 배치와 달리 draw를 구속함
```

~130W에 붙은 채로 죽으므로, 상한을 낮춰 생존하면 전원 계통 가설이 사실상 확정됩니다.
팬 하한 상향도 같이 하면 온도↓ → 누설전류↓ → 같은 클럭에서 전력↓입니다.

**GPU 작업 전에는 항상 `nvidia-smi -L`로 생존 확인**을 먼저 하세요. 죽어 있으면 CUDA/TensorRT
벤치는 전부 무의미하고 ORT는 조용히 CPU로 폴백해 잘못된 수치를 뱉습니다.

**팬 100% 고착과 부하 중 팬 저속은 다른 신호입니다.** 100% 고착은 드라이버가 팬 제어권을
잃은 **이탈의 증상**이고, 위처럼 부하 중 팬이 낮게 붙어 있는 것은 **별개의 이상**입니다.

---

## 7. 우선순위 다음 할 일

1. ~~**GPU 교착 해소**~~ ✅ — RTX 3080으로 옮겨 해소(§6). 새 PC에서도 GPU 작업 전 `nvidia-smi -L`만.
2. ~~**QAT 2팔 재실행**~~ ✅ — W4A8로 완주(§5). 회복 97.1% · 대조군 격차 −1.50%p.
3. ~~**결과를 §2.5에 반영**~~ ✅ — `03_quantization_theory.md`에 §2.5.4(실모델 QAT 회복) 신설(커밋 `eb45b33`).
4. ~~**2단계 DETR 착수**~~ ✅ — `04_transformer_quantization.md` §4.1~4.5를 `facebook/detr-resnet-50`으로 완주(커밋 `41dc49e`). COCO val 5,000장 mAP FP32 0.4207→INT8 0.2402, 초안 단정 3건 반전. 산출물 `experiments/stage2_detr/`·`logs/stage2_detr_quantization_report.html`.
5. ~~**§4.6 BEVFormer-tiny 실기 검증**~~ ✅ — 2-tier로 완주. Tier A(정본 venv torch 2.11)에서 grid_sample/MSDeformAttn op 지뢰 5종 실증(`experiments/stage2_bevformer/` b01~b05), Tier B(무컴파일 레거시 venv torch 1.13+mmcv-full 1.7.0 프리빌트 휠+mmdet3d 1.0.0rc6)로 실 mmcv op(b06)+실모델 FP32 mAP(b08). **op 단정 반전 0**(초안 맞음)·함정 +2(mmcv op CPU-only 유효 export·전체 export는 `point_sampling` `lidar2img`에서 사망). FP32 nuScenes-mini mAP 0.2647(81샘플 스모크). 전체 INT8은 유효 export 없어 **범위 밖**. 산출물 `logs/stage2_bevformer_quantization_report.html`·`experiments/stage2_bevformer/onnx_export_failures.md`. 데이터는 `~/bevformer_work/`(git 밖, §2).
6. **§4.4 SmoothQuant** — DETR·BEVFormer 공통으로 "진짜 레버"로 지목된 modelopt SmoothQuant(per-token activation) 적용(미착수). §4.4는 API 확인 필요 상태.
7. **BEVFormer 전체 INT8 / 3~7단계·캡스톤** — 전자는 포크(`DerryHub/BEVFormer_tensorrt`) 커스텀 op 플러그인 정본 재빌드 후. 후자는 아직 웹 검증만.

---

## 8. 작업 규칙

- **저장소는 public입니다.** `logs/`나 터미널 출력 기반 문서를 커밋하기 전에
  `grep -rniE "password|비밀번호|ghp_|github_pat|secret|api[_-]?key|token|PRIVATE KEY|Bearer "`
  + 이메일/IP 패턴으로 스캔하세요. 발견 시 처리 방식은 **마스킹 후 커밋**(값을 `<암호>` 같은
  플레이스홀더로) — 파일을 빼거나 `.gitignore`로 돌리는 것보다 이 방식을 씁니다. 단 발견
  사실은 푸시 전에 알립니다. 실제로 2회 걸렀습니다(첫 로그 커밋 전, 메모리 git 복사 전).
- **MD가 정본, HTML은 파생물입니다.** 내용을 고친 뒤 재렌더:
  `python3 .claude/skills/md-to-html/scripts/render.py study_guide`
  렌더러의 `slugify`는 GitHub(`github-slugger`) 규칙과 일치시켜 놨습니다 — 공백을 접거나 끝을
  strip하면 MD 크로스링크 앵커가 HTML에서만 깨집니다.
- **수치는 출처를 남기세요.** 지금 문서의 모든 50k 수치는 `~/stage1-work/eval_hits_v2.npz`
  (17키 × 50,000 per-image bool hit 배열)에서 재계산 가능합니다. paired 비교는 McNemar
  (`scipy.stats.binomtest(n10, n10+n01, 0.5)`)를 쓰고, 절대 top-1 차이만으로 판정하지 마세요.
- 제작 하네스(에이전트·스킬) 구성은 [`CLAUDE.md`](CLAUDE.md), 상세 실측 기록은
  [`.claude/memory/`](.claude/memory/)에 있습니다.
