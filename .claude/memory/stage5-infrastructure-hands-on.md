---
name: stage5-infrastructure-hands-on
description: "5단계 인프라화 완료(2026-08-17, 커밋 ff523de 푸시완료): 벤치 하네스 polygraphy TrtRunner. 무음 오답 2건(zero-copy 버퍼 에일리어싱 top-1 0.0014·pivot dropna 회색행 드롭). device_memory_size_v2=scratch mem≠엔진파일. pytest-regressions 최신 2.11.0"
metadata:
  node_type: memory
  type: project
---

`study_guide/07_infrastructure.md`(1229줄 초안, hands-on 미검증)를 AI-LAP/RTX3080
([[machine-ai-lap-rtx3080]])에서 완주(2026-08-17). venv `~/emb-ai` · **TensorRT 10.16.1.11** ·
polygraphy 0.50.3 · pandas · pytest-regressions 2.11.0. 모델 torchvision **ResNet50**(3단계 자산 재사용 —
[[stage3-tensorrt-hands-on]]), 지연 배치1 / 정확도 ImageNet val 5,000장. **커밋 ff523de**(main, 푸시 완료 —
[[repo-is-public-scan-before-commit]]). 산출물: `logs/stage5_infrastructure_report.html` ·
`experiments/stage5_infrastructure/`(bench/*·harness_constraints.md·README).

**모델 스코핑:** 문서는 BEVFormer/mAP 예시지만 2단계서 BEVFormer INT8 **유효 export 경로 없음** 확정
([[stage2-bevformer-hands-on]]) → 실제 빌드 가능한 **ResNet50/top-1**로 대체. `BenchResult.accuracy`는
mAP·top-1 겸용 0~1 제네릭.

**헤드라인 매트릭스(RTX 3080, 하네스 wall-clock):** FP32 1.837ms → FP16 1.0231 **×1.80** →
INT8 0.8628 **×2.13**; top-1 0.7688 / 0.7686 / 0.768; scratch-mem 8.4 / 3.9 / 1.7 MB; 엔진빌드 12.8/29.8/49.8s.

**정정 8건:**
- **정정 1 trtexec 부재**(3단계와 동일): 정본 pip 휠에 `trtexec` 실행파일 없음 → 문서의 모든 `trtexec`
  명령을 **polygraphy `TrtRunner`**로 대체.
- **🔴 정정 5 무음 오답(zero-copy 버퍼 에일리어싱):** `TrtRunner.infer`가 출력 host 버퍼 **하나를 재사용** →
  `.copy()` 없이 예측 리스트를 모으면 전부 마지막 추론을 가리켜 top-1이 **0.0014(=1/1000)로 붕괴**.
  문서의 pycuda 원본은 호출마다 `np.empty` 새로 할당하므로 안전 → 이 버그는 **pycuda→polygraphy 치환이 도입**한 것
  (exit 0 silent-wrong). 픽스: `[self.run(x).copy() for x in loader.eval_set()]` → 0.7688.
- **🔴 정정 6 무음 오답(pivot dropna 자기위반):** pandas `pivot_table` 기본 `dropna=True`가 all-NaN
  스텁행(보드필요 SoC)을 조용히 드롭 → §5-1 "회색행 남긴다" 원칙 위반. 픽스: `dropna=False` +
  HTML `na_rep="보드필요"`. **CSV long-form·회귀 baseline은 6행 보존이라 무영향**(사람이 읽는 pivot만 손실).
- **정정 2** deprecated `IInt8EntropyCalibrator2`가 TRT 10.16서 여전히 빌드(INT8 top-1 0.768).
- **정정 3** 작동 attr은 `device_memory_size_v2`(무접미는 deprecated), 값은 실행 컨텍스트
  **scratch/activation 메모리**(8.4→3.9→1.7MB)로 **가중치·엔진파일 크기 아님**(3단계 엔진파일은 122→49→25MiB).
- **정정 7** `EXPLICIT_BATCH` flag는 10.16서 값=0·DeprecationWarning; 맨 `create_network()`가 이미 explicit.
- **정정 4** `data.py` eval-set 정정.
- **정정 8** 문서 "pytest-regressions v3.0+"가 허구 → 실제 최신 **2.11.0**(0.1.0…2.11.0, v3.x 없음).
  `dataframe_regression` fixture는 2.11.0에 존재하므로 코드는 정상 — 버전 주장만 틀림.

**tech-reviewer 팬인 PASS(🔴 0·material 🟡 0, 수정 0건):** 6-JSON SSOT 1:1 전건·산술
(×1.80=1.837/1.0231·×2.13=1.837/0.8628)·SVG 비례(지연 530.7/295.6/249.3=×288.9px/ms,
`.copy()` 461.3 vs 0.84=×600)·승번 오염 0(삽입만, §0~9·§4-1~4-10·§5-1~5-3 보존)·크로스링크 6개 실재·캐비앗 병기 확인.

**캐비앗(불변):** 하네스 `_timeit`은 wall-clock(H2D/D2H+파이썬 오버헤드)이라 3단계 event-timed 대비
**절대 지연↑·factor 압축**(×1.80/×2.13 vs 3단계 ×1.96/×2.12 — 고정 오버헤드가 분모 상수). 5,000장 서브셋
top-1 부풀림 가능(1단계 함정 0). **절대값 아닌 상대 관계만 유효.**

**남은 과제:** 4단계(멀티 SoC) 실기는 벤더 보드·SDK·토큰 확보 시까지 보류(하네스에 qcs8550/rzv2h/tda4vm
스텁행 NaN="보드필요"로 예약).
