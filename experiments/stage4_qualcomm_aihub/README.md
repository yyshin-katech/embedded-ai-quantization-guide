# experiments/stage4_qualcomm_aihub

4단계 멀티 SoC의 **Qualcomm 벤더-NPU 축**을 보드 없이 **Qualcomm AI Hub** 클라우드 실기기에서 실측한 산출물.
CPU 폴백 프록시(`experiments/stage5_infrastructure/cpu_proxy/`, offload 0% 바닥값) 위에, **벤더 NPU가 실제로 얼마나 빠른가 + 얼마나 offload되는가**를 채운다.

- 모델: torchvision **ResNet50** (3·5단계 자산 FP32/INT8 QDQ ONNX 재사용)
- 디바이스: **QCS8550 (Proxy)** · **SA8775P ADP**(자동차 Snapdragon Ride 물리 보드)
- 런타임: `--target_runtime qnn_context_binary` (ONNX → Hexagon HTP context binary)
- 스택: qai_hub 0.54.0 (격리 venv) / QAIRT SDK 2.45.0 / HTP v73

전체 서술·SVG·판정은 **`logs/stage4_qualcomm_aihub_report.html`**, 로그 원문·설계규칙은 **`aihub_constraints.md`**.

## 헤드라인 (지연 · offload)

| 디바이스 · 정밀도 | on-device 지연 | INT8 배속 | NPU offload | cycles |
|---|---:|---:|---:|---:|
| QCS8550 · FP32(→fp16) | 1864 µs | — | 100% (125/125) | 4,677,822 |
| QCS8550 · INT8 QDQ | 1052 µs | ×1.77 | 100% (128/128) | 3,754,903 |
| SA8775P ADP · FP32(→fp16) | 3056 µs | — | 100% (125/125) | 6,192,577 |
| SA8775P ADP · INT8 QDQ | 1505 µs | ×2.03 | 100% (128/128) | 4,462,570 |

## on-device 정확도 (QCS8550, 200장)

| 경로 | top-1 | ORT 일치 | 비고 |
|---|---:|---:|---|
| ORT-CPU (기준) | 0.750 | — | |
| FP32(→fp16) | 0.745 | 0.96 | 충실 |
| INT8 · 외부 ORT-QDQ | 0.005 | 0.005 | **조용한 붕괴** (발견 #2) |
| INT8 · AI Hub 자체 quantize | 0.735 | 0.94 | 올바른 경로 (근접 회복) |

## 발견 요약

1. **AI Hub 프론트엔드 엄격성** — `logits`가 value_info+IO 양쪽인 ORT-QDQ ONNX를 거부(컴파일 실패). ORT/TRT는 통과. → `clean_valueinfo_for_aihub.py`로 정리.
2. **외부 QDQ = on-device 정확도 함정 (silent-wrong)** — compile/profile은 통과해도 INT8 top-1이 0.005로 붕괴. FP32(fp16)는 충실. 올바른 경로는 AI Hub 자체 quantize → **0.735 회복**(게다가 748µs로 더 빠름).
3. **HTP는 fp16-native** — FP32 ONNX도 자동 변환돼 fp16으로 실행(native fp32 없음).
4. **엄격 NCHW 인터페이스** — NHWC 피드 거부(레이아웃 가설 기각).

## 파일

```
scripts/
  qaihub_run.py              # 최초 QCS8550 FP32+INT8(원본) compile+profile
  qaihub_int8.py             # 정리본 INT8 QCS8550 compile+profile
  qaihub_device.py           # 파라미터화 러너: <device> <slug> → fp32+int8 compile+profile
  qaihub_acc.py              # 외부-QDQ INT8 200장 on-device 정확도 (붕괴 재현)
  qaihub_fp32_acc200.py      # FP32(fp16) 200장 on-device 정확도 (충실 확인)
  qaihub_fp32_acc.py         # FP32 vs INT8 20장 감별 (붕괴 원인 격리)
  qaihub_native_quant.py     # AI Hub 자체 quantize → compile → profile → inference (올바른 경로)
  qaihub_layout_test.py      # NCHW vs NHWC 레이아웃 반증
  clean_valueinfo_for_aihub.py  # 발견 #1 fix
results/
  qcs8550_summary.json, qcs8550_int8clean_summary.json, sa8775p_summary.json
  acc_fp32_htp_200.json, acc_int8_foreign_qdq_200.json, acc_fp32_vs_int8_20.json
  aihubq_summary.json        # AI Hub-native quantize 경로 결과
raw/
  profile_*_raw.json         # AI Hub 프로파일 원본 (per-layer compute_unit·cycles)
```

## 재현 / 캐비앗

재현 절차와 캐비앗(절대값 비교 금지, 상대 관계만)은 `aihub_constraints.md` 참조.
인증 토큰은 `~/.qai_hub/client.ini`(repo 밖)에서 읽으며 어떤 스크립트에도 하드코딩하지 않는다.
