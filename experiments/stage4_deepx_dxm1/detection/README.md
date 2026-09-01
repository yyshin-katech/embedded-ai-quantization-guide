# stage4 — DX-M1 검출 축 (YOLO26n INT8 정확도 + NMS-fold 레짐)

정확도 축([`../accuracy/`](../accuracy/))과 크로스오버 축([`../crossover/`](../crossover/))을 **하나의 검출 모델**로 잇는 후속 축. 정확도 축은 분류(ResNet50 top-1)였고 크로스오버는 host-bound↔NPU-bound 레짐이었으나 **정확도 미측정**(yolo26n 라벨셋)이었다. 이 축은 **같은 `yolo26n` 640으로 COCO val2017 mAP(정확도)와 온보드 스테이지 프로파일(레짐)을 동시에** 측정해 두 결론을 검출 태스크에서 잇는다.

핵심 장치(정확도 축과 동일): **비트 동일 letterbox npy + 공유 `[1,300,6]` 디코드**로 FP32(x86 ORT)와 INT8(Pi DX-M1)이 같은 입력을 읽고 같은 코드로 디코드 → mAP 차이는 오직 **양자화**에서 온다.

리포트: [`../../../logs/stage4_deepx_dxm1_detection_report.html`](../../../logs/stage4_deepx_dxm1_detection_report.html)

---

## 헤드라인 (SSOT: `results/detection_summary.json`)

### (1) 검출 INT8은 DX-M1에서 **무손실급** — 분류 축의 검출판 확증

| 변형 | 무엇 | mAP 50-95 | mAP_s |
|------|------|-----------|-------|
| **FP32 ref** | stock `yolo26n.onnx` · ORT float · 500장 | **0.448** | 0.2283 |
| **INT8 · COCO-캘리브 (B)** | 내 dx_com 컴파일(**동일 weight**) | **0.439** | 0.2144 |
| INT8 · 벤더 prebuilt (A) | 쇼케이스 `.dxnn`(블랙박스 캘리브) | 0.4357 | 0.2149 |

| 분해 | 계산 | 값 | 해석 |
|------|------|----|------|
| **순수 양자화** | FP32 − B | **−0.009 (−2.0%)** | 동일 weight·in-domain 캘리브 → 무손실급 |
| 캘리브 도메인 | B − A | +0.0033 | in-domain이 근소 우위 — 서브셋 노이즈 내 |
| 소객체 손실 | mAP_s: FP32 − B | −0.0139 (−6.1%) | 손실 집중되나 경미 |

→ 순수 양자화 대가는 mAP 50-95 **−0.009(−2.0%)** 뿐. 이는 [정확도 축](../accuracy/)의 분류 결론(native 0.766 ≈ FP32 0.762, 무손실급)을 **검출로 확장**한다 — DX-M1 native PTQ는 **분류·검출 둘 다** 무손실급. 소객체(mAP_s −6.1%)에 손실이 몰리나, stage2 DETR의 mAP_s −77~85% 폭락과는 **차원이 다른 경미함**.

### (2) NMS-folding은 레짐을 **못 바꾼다** — dx_com이 raw 헤드에서 자름

**가설(반증됨)**: `[1,300,6]`로 NMS가 접힌 "end2end" export는 출력을 2.82MB→7.2KB(392×)로 줄여 **D2H-bound→compute-bound로 뒤집을 것**.

**결정적 관찰** — dxbenchmark로 본 "end2end" `.dxnn`(prebuilt A·내 컴파일 B **둘 다**)의 NPU 출력 텐서는 여전히 **6개 raw conv 헤드**(`one2one_cv2/cv3` @80/40/20, 합 **2,822,400 B**). `dx_engine.run()`이 돌려주는 `[1,300,6]`은 **호스트(CPU)에서** decode+NMS를 돌린 결과 — `.dxnn` 패키지에 번들된 호스트 후처리다. **dx_com은 NMS를 NPU에 못 접는다** → 매 프레임 2.82MB raw 헤드가 PCIe를 건넌다.

| 스테이지 p50 | Option B (COCO) | Option A (prebuilt) | 크로스오버 raw-head yolo26n |
|------|------|------|------|
| H2D | 3.47 ms | 3.45 ms | — |
| Inference | 7.303 ms | 8.995 ms | 9.00 ms |
| **D2H (2.82MB)** | **21.805 ms** | 21.801 ms | 21.81 ms |
| D2H / 연산 | 2.99× | 2.42× | 2.42× |
| async 3코어 fps | 90.81 | 90.85 | 91.2 |
| **코어 잡 분포** | **473 / 27 / 2** | 494 / 6 / 2 | 472 / 28 / 2 |
| 코어 스케일 3c/1c | 1.00× | 1.00× | 1.00× |

→ raw-head export든 end2end export든 **NPU→호스트 D2H는 동일한 2.82MB** → **둘 다 D2H-bound**(코어 잡 분포 473/27/2 vs 472/28/2로 동일 시그니처). 이는 [크로스오버 축](../crossover/)의 결론("yolo26n은 출력 크기 탓에 D2H-bound")을 **반증이 아니라 심화**한다 — D2H 병목은 export 선택의 우연이 아니라 **구조적**(이 툴체인에선 NMS를 접어 출력을 줄이는 길 자체가 없음). "그냥 NMS 접어 출력 줄이면 되지"라는 순진한 가정을 닫는다.

---

## 양자화 격리 방법 (정확도 축 top-1 방법의 검출판)

- **평가셋**: COCO val2017 **정렬 첫 500장**(결정론적) → 640 letterbox(pad114 중앙) → **하나의 uint8 npy**(md5 `984e0457…`)로 스택. FP32(x86 ORT)·NPU(Pi dx_engine)가 **같은 배열**을 읽음 → letterbox 기하 **비트 동일**(전처리 confound 0).
- **디코드**: 공유 모듈 `yolo_det_common.decode()` — 두 백엔드가 같은 `[1,300,6]`(xyxy@640, class 0-79)을 같은 코드로 언-letterbox → COCO xywh + COCO80→91 매핑. mAP 차 = 양자화 only.
- **채점**: 같은 pycocotools 스코어러 + 같은 500장 imgIds(세 변형 공통).
- **Option A**: 벤더 쇼케이스 prebuilt `.dxnn`(DEEPX 파이프라인 캘리브, 블랙박스 weight).
- **Option B**: 내 `dxcom -c yolo_coco_cfg.json` 컴파일 — **동일 stock onnx** + 100장 **in-domain COCO 캘리브**(val2017[500:600], eval[0:500]과 **disjoint**). → FP32−B = **동일 weight** 순수 양자화(confound 0).

**함정 — yolo 캘리브는 ÷255만 (ImageNet norm 금지)**: 정확도 축 ResNet50 config는 `div 255` 다음에 ImageNet `normalize`를 접었으나, yolo는 **÷255만** 쓰므로 config에서 `normalize` 블록을 **반드시 제거**. 접힘=`Div·Transpose`, 스킵=`convertColor·expandDim` → 런타임 입력 **uint8 NHWC RGB**. img0 `cat_ids`가 거실 클래스(person/chair/tv…)로 나와 RGB 정합 확증.

**FP32 ref 노트**: CUDA EP는 `libcudnn.so.9` 미해결로 실패 → CPU EP 폴백(mAP 동일, ref 용도라 무관).

---

## 지연 (Pi batch1, host-timed)

| 변형 | p50 | 비고 |
|------|-----|------|
| INT8 COCO (B) | 36.746 ms | 호스트 decode/NMS 포함, 연산 7.3ms |
| INT8 prebuilt (A) | 38.4 ms | 연산 9.0ms |

**지연≠처리량**: host-timed e2e 지연(≈37ms/img, 순차)은 decode/NMS·PCIe 왕복 포함, async 처리량(≈91 fps)은 D2H로 게이트. FP32 ref는 x86 CPU라 **mAP 기준일 뿐 Pi 지연 비교 대상 아님**.

---

## 파일 구성

```
scripts/
  preprocess_eval_set.py     # val2017 첫 500장 → 640 letterbox → eval_u8.npy + eval_meta.json
  fp32_infer.py              # ORT FP32 ref (x86) — (u8/255).transpose RGB NCHW
  make_calib_png_det.py      # (B) 캘리브 PNG 100장, val2017[500:600] (eval과 disjoint) — x86
  yolo_coco_cfg.json         # (B) dx_com config: ÷255-only(normalize 없음), fold Div/Transpose
  npu_infer_det.py           # DX-M1 dx_engine 러너 (Pi) — uint8 NHWC 입력, [1,300,6] 디코드
  yolo_det_common.py         # 공유 letterbox + decode + COCO80→91 (두 백엔드 공통)
  eval_map.py                # pycocotools COCOeval, 서브셋 imgIds 제한 → map_<tag>.json
  build_detection_summary.py # 호스트 조립 → SSOT detection_summary.json (crossover analyze_profiler 재사용)
results/
  eval_meta.json                                   # 500장 letterbox 메타(r/left/top/w0/h0/image_id)
  predictions_{fp32,npu_ppe,npu_coco}.json         # per-변형 COCO 검출
  predictions_npu_{ppe,coco}_lat.json              # dx_engine host-timed 지연
  map_{fp32,npu_ppe,npu_coco}.json                 # pycocotools 12 metric
  detection_summary.json                           # ← SSOT (정확도 분해·레짐·지연·캐비앗)
raw/
  coco_prof/     profiler.json + stdout.txt + DXBENCHMARK*.csv   # (B) 온보드 스테이지 분해
  e2e_ppe_prof/  profiler.json + stdout.txt                      # (A) 온보드 스테이지 분해
```

## 재현

```bash
# --- x86 (AI-LAP) ---
# 1) 평가셋(첫 500장) 전처리 → eval_u8.npy + eval_meta.json
emb-ai/bin/python scripts/preprocess_eval_set.py --coco <coco> --n 500 --out eval_u8.npy
# 2) FP32 ref (ORT)
emb-ai/bin/python scripts/fp32_infer.py --onnx yolo26n.onnx --npy eval_u8.npy --meta eval_meta.json --out results/predictions_fp32.json
# 3) (B) in-domain COCO 캘리브 PNG(eval과 disjoint) + config-path 컴파일 (dxcom-venv)
emb-ai/bin/python scripts/make_calib_png_det.py --start 500 --n 100 --out_dir calib_png
#    yolo_coco_cfg.json 의 <CALIB_PNG_DIR> 를 calib_png 절대경로로 치환 후:
dxcom -m yolo26n.onnx -c yolo_coco_cfg.json -o out_coco --opt_level 1   # → out_coco/yolo26n.dxnn

# --- Pi 5 온디바이스 (venv-dx-runtime) ---
# eval_u8.npy + .dxnn scp 후, INT8 추론 (Option A prebuilt / Option B COCO)
python npu_infer_det.py --model yolo26n_coco.dxnn --npy eval_u8.npy --meta eval_meta.json --out results/predictions_npu_coco.json --color rgb
# 온보드 프로파일: dxbenchmark --dir <dxnn_dir>  → raw/*/profiler.json

# --- 호스트 채점 + SSOT (AI-LAP) ---
emb-ai/bin/python scripts/eval_map.py --gt <instances_val2017.json> --pred results/predictions_npu_coco.json --meta results/eval_meta.json --tag npu_coco
emb-ai/bin/python scripts/build_detection_summary.py   # → results/detection_summary.json
```

`.dxnn`·`eval_u8.npy`·calib PNG는 **미커밋**(데이터/대용량 정책). config는 크기-결정적이나 바이트는 run마다 상이(컴파일러 비결정성) → 정확도는 검증된 빌드 기준.

---

## 캐비앗

1. **500장 val2017 서브셋** → 상대 Δ만 유효, 절대 mAP는 문헌 비교 대상 아님(paired FP32−INT8 Δ는 서브셋 노이즈에 강건).
2. **prebuilt/PTQ·batch1·Pi5 PCIe Gen2×1** → D2H-bound은 **Pi 5 링크 성질**(네이티브 Gen3×4면 D2H 벽 위치가 달라짐, 미측정) — 크로스오버 축과 동일 근본.
3. 지연은 `dx_engine` host-timed(호스트 decode/NMS 포함) — 순수 NPU 연산(profiler 7.3ms)과 구분.
4. Option B 컴파일 경고(cosine 0.927 @output0)는 surgeon 근사 — mAP엔 −0.009만 반영(과대 아님).
5. Pi 5는 DEEPX 호스트일 뿐 **자동차 3벤더(TI/Qualcomm/Renesas) 아님**. 정확도 축(외부 QDQ 하드 거부)과 동일하게 **벤더가 양자화 소유**.
