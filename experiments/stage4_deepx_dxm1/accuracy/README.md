# stage4 — DX-M1 정확도 축 (native-QDQ vs 외부-QDQ · CPU↔NPU 예측 일치)

DX-M1 벤치([`../`](../))가 캐비앗 #4로 남긴 **"정확도 미측정(prebuilt·라벨셋 부재)"** 을 닫는 후속 축. 벤치는 `yolo26n`(라벨셋 없음)이었으므로, **정확도가 필요한 이 축은 3·5단계와 공유하는 ResNet50 + ImageNet 1000장 번들**로 스코프를 옮긴다(실제 top-1을 라벨과 대조 가능).

핵심 장치(벤치와 동일): **같은 Cortex-A76이 CPU 폴백 프록시([`../../stage5_infrastructure/cpu_proxy/`](../../stage5_infrastructure/cpu_proxy/))이자 DX-M1 호스트** → 같은 보드·**같은 1000장 번들**(md5 `3c0e151c…`, = 커밋된 Orin/CPU-proxy 번들)로 CPU ↔ NPU를 이미지 단위로 뺄셈. 번들 동일성은 md5 + CPU arm이 커밋된 stage4 Pi5 값(0.7620/0.7500)을 **비트 동일 재현**(§ 아래)으로 이중 확인.

리포트: [`../../../logs/stage4_deepx_dxm1_accuracy_report.html`](../../../logs/stage4_deepx_dxm1_accuracy_report.html)

---

## 헤드라인 (SSOT: `results/accuracy_summary.json`)

### (a) 외부 QDQ는 컴파일 단계에서 **하드 거부** (Qualcomm과 정반대 실패양식)

| 경로 | 결과 |
|------|------|
| **native** (FP32 ONNX + dx_com 자체 PTQ) | 컴파일 OK → `.dxnn` 산출, on-device top-1 **0.7660** (accuracy-valid) |
| **외부 QDQ** (`resnet50_int8_qdq.onnx` 지참) | 컴파일 **실패** — `GraphStructureError: 106 isolated nodes` (weight/bias DequantizeLinear 분기), 일반 `InternalError: contact DEEPX`로 래핑. **`.dxnn` 미산출** |

→ **Qualcomm HTP는 외부 QDQ를 조용히 무시**(compile/profile 통과 → 런타임 top-1 0.75**→0.005 붕괴**)했으나, **DEEPX dx_com은 시끄럽게 거부**(컴파일 자체가 죽어 산출물 없음). 두 벤더 공통 근본원인 = **벤더 NPU가 양자화를 소유**(BYO-QDQ 미지원)이나, **실패양식이 정반대** — DEEPX 쪽이 "조용히 깨진 모델을 배포할 수 없다"는 점에서 안전. (a)의 거부가 곧 (b)에서 **동일-scale 교차 커널 다리를 만들 수 없는 이유**다.

### (b) CPU↔NPU 예측 일치 — "정수커널 경로의존" 실 vendor-NPU로 확장

| top-1 (n=1000) | 값 | 비고 |
|----|----|------|
| cpu_fp32 (A76 ORT) | **0.7620** | 커밋된 Pi5 프록시와 비트 동일 |
| cpu_int8 (A76 MLAS SDOT) | **0.7500** | 커밋된 Pi5 프록시와 비트 동일 |
| **npu_native (DX-M1 INT8)** | **0.7660** | native PTQ, **FP32 기준 대비 무손실급**(+0.4%p, 1000장 노이즈 내) — **붕괴 없음** |

| 일치 다리 | 무엇이 다른가 | agree/1000 |
|-----------|--------------|-----------|
| (stage4) Jetson↔Pi5 | 없음 (같은 MLAS SDOT) | 1000 |
| (stage3) CPU↔iGPU (TRT vs MLAS) | 커널만 (**같은** QDQ scale) | 961 |
| **(여기) CPU↔NPU (MLAS vs DX-M1)** | **커널 + scale + 전처리 위치** | **939** |
| cpu_int8 vs cpu_fp32 (동일 A76) | 양자화만 (같은 커널족) | 950 |
| npu_int8 vs cpu_fp32 | 양자화+커널+전처리 | 969 |

→ **일치율은 단조 감소** — 다름의 원천이 하나씩 더해질수록 떨어진다: 같은 커널+scale=1000 → 같은 scale·다른 커널=961 → **다른 커널·다른 scale·다른 전처리 위치=939**. 939가 stage3의 동일-scale 961보다 낮은 것은 **(a) 때문에 scale을 고정할 수 없어서**다(NPU는 자체 native scale 강제). 헤드라인 61장 불일치 분해: NPU 정답 26 · MLAS 정답 10 · 둘다 오답 25 → **net +16 = top-1 0.7660−0.7500 = +16/1000** (산술 정합).

---

## .dxnn 입력 계약 (컴파일러가 자체 문서화 — `raw/native_compile.log`)

dx_com은 전처리를 **항상 그래프에 접는다**(fold): 런타임 `.dxnn` 입력 = **uint8 NHWC [1,224,224,3]**(get_input_size=150528), 출력 `logits` [1,1000] f32. 컴파일 로그가 명시:
- **접힘**: `Div(x=255) → Normalize(mean,std)` + `Transpose(HWC→CHW)`(입력 포맷으로 처리)
- **스킵**(그래프에 못 넣음): `convertColor`, `expandDim`
- 지침: *"For NPU inference, provide uint8 HWC input directly."*

→ **함정**: python DataLoader 경로로 컴파일하면 **잘못된 기본 normalize가 접혀 on-device top1=0**. accuracy-valid native 빌드는 **CLI config 경로**(`native_cfg.json`: raw uint8 PNG 캘리브 + 명시적 `div/255`+ImageNet `normalize`)로 산출. convertColor가 런타임에 스킵되므로 RGB uint8을 직접 먹이면 mean/std(RGB 순서)와 정합 → top-1 0.766 확인.

---

## 파일 구성

```
scripts/
  probe_extqdq.py      # (a) 외부 QDQ 거부 재현: op-inventory A/B + chained GraphStructureError
  compile_dxm1.py      # dx_com 컴파일 드라이버(참조; DataLoader 경로 함정 주석) — x86 dxcom-venv
  native_cfg.json      # (a) native 빌드 config(=authoritative): raw uint8 → fold div/255+ImageNet norm
  make_calib_png.py    # 캘리브 PNG 100장 재생성(eval 번들과 disjoint) — x86
  npu_infer.py         # NPU on-device 러너(dx_engine, 입력 계약 자동감지) — Pi
  cpu_infer.py         # A76 CPU baseline(ORT CPUEP, fp32/int8) — Pi
  analyze_dxm1_acc.py  # 호스트 교차대조 → SSOT accuracy_summary.json
results/
  npu_native.json  cpu_fp32.json  cpu_int8.json    # per-arm pred_cls(1000) + top-1
  accuracy_summary.json                            # ← SSOT (top-1·일치·헤드라인 분해·cross-run)
  rpi_labels.npy                                   # 정답 라벨(= 커밋 Orin/CPU-proxy와 동일, md5 08b054c2…)
raw/
  native_compile.log        # native 빌드 로그(전처리 fold 문서·exit 0·.dxnn 51,746,482 B)
  compile_extqdq_reject.log # (a) 거부 원문(op-inventory + 106 isolated nodes)
  inference_runs.log        # n=200 검증 + n=1000 3-arm 실행 요약 + 버전
```

## 재현

```bash
# --- x86 (AI-LAP, dxcom-venv; dx-compiler는 x86 전용) ---
python scripts/make_calib_png.py --calib-npy calib_u8.npy --out calib_png --n 100
# native_cfg.json 의 <CALIB_PNG_DIR> 를 calib_png 절대경로로 치환 후:
dxcom -m resnet50_fp32.onnx -c native_cfg.json -o native --opt_level 1   # → native/resnet50_fp32.dxnn
python scripts/probe_extqdq.py --fp32 resnet50_fp32.onnx --qdq resnet50_int8_qdq.onnx --calib-png calib_png  # (a)

# --- Pi 5 온디바이스 ---
# NPU (venv-dx-runtime): resnet50_fp32.dxnn 을 scp 후
python npu_infer.py --model resnet50_native.dxnn --data . --out results/npu_native.json --tag npu_native --n 1000
# CPU (cpu-venv, ORT CPUEP)
python cpu_infer.py --model resnet50_fp32.onnx     --out results/cpu_fp32.json --tag cpu_fp32 --n 1000 --data .
python cpu_infer.py --model resnet50_int8_qdq.onnx --out results/cpu_int8.json --tag cpu_int8 --n 1000 --data .

# --- 호스트 교차대조 (SSOT) ---
python scripts/analyze_dxm1_acc.py    # → results/accuracy_summary.json
```

`.dxnn`(51,746,482 B)·번들 npy(150MB)·calib은 **미커밋**(데이터/대용량 정책). config는 **크기-결정적**(51,746,482 B)이나 바이트는 run마다 상이(컴파일러 비결정성) → 정확도는 검증된 빌드 기준.

## cross-run 재현성 (번들 동일성 이중 확인)

이 세션 CPU arm(ORT **1.29.0**) vs 커밋된 stage4 Pi5 프록시(ORT **1.28.0**), 같은 A76·같은 번들:
`cpu_fp32 1000/1000` · `cpu_int8 1000/1000` **비트 동일**. → 번들이 이미지 단위로 동일함을 md5 외에 예측으로도 확인 + **MLAS SDOT 정수커널이 ORT 1.28→1.29에서 불변**(같은 커널=100%, 경로의존 스레드 정합).

---

## 캐비앗

1. **모델 스코프 이동** — 벤치의 yolo26n(라벨셋 없음) → 정확도 대조 가능한 **ResNet50 + ImageNet 1000장**. 지연/처리량 벤치와 1:1 아님(다른 모델).
2. **(b) 헤드라인 다리(939)는 3중 confound** — 정수커널 + 양자화 scale + 전처리 위치(NPU는 div/normalize를 그래프서 양자화, CPU는 host-side float). (a) 때문에 scale 고정 불가 → **동일-scale 순수 커널 다리는 원리상 불가능**. 939는 커널-only 상한이 아님.
3. **npu_native 0.7660 > cpu_fp32 0.7620은 "이김" 아님** — +0.4%p는 1000장 서브셋 노이즈 내(1단계 함정 0). 결론은 "**붕괴 없음·무손실급**"이지 "FP32 초과"가 아님.
4. 절대 top-1은 **1000장 서브셋** → 3·5단계 5000장 RTX와 비교 불가, **같은 번들 상대 관계만** 유효.
5. Pi 5는 DEEPX 호스트일 뿐 **자동차 3벤더(TI/Qualcomm/Renesas) 아님**.
