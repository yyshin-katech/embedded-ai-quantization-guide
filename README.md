# embedded-ai-quantization-guide

**AI 모델 양자화 → 임베디드 배포 실전 학습 가이드.** 멀티카메라 Transformer 인식 모델을 INT8 양자화하여 멀티 SoC(NVIDIA Orin/Thor · TI Jacinto · Qualcomm · Renesas RZ/V)에 올려 구동하는 전 과정을, **Ubuntu 22.04 + NVIDIA RTX GPU**에서 "읽고 따라 하면 실제로 실행되는" 단계별 문서로 정리했습니다.

> 모든 버전·링크는 2026-07 기준으로 웹 검증했습니다. 실제 설치 시점엔 각 공식 페이지에서 재확인하세요.
>
> ✅ **0단계 환경은 2026-07-31 실제 머신에서 설치·검증 완료**했습니다(Ubuntu 22.04.5 / RTX 3060 / 드라이버 595.84 / `nvcc` 12.8.93). 아래 버전 스택과 [`01_environment_setup.md`](study_guide/01_environment_setup.md)의 예상 출력은 그 **실측값**입니다.

---

## 🚀 바로 시작

- **가이드 인덱스**: [`study_guide/README.md`](study_guide/README.md) — 여기서 시작하세요.
- **HTML로 편하게 보기**(다크 테마 · 진행률 체크박스 · 목차): 저장소를 클론한 뒤 `study_guide/README.html`을 브라우저로 엽니다.
  ```bash
  git clone https://github.com/yyshin-katech/embedded-ai-quantization-guide.git
  # study_guide/README.html 더블클릭 (또는 브라우저로 열기)
  ```
  > HTML을 웹에서 바로 보고 싶으면 GitHub Pages(Settings → Pages → `main` / root)를 켜면 됩니다.

---

## 📚 학습 로드맵

| # | 문서 | 단계 | 핵심 산출물 |
|---|------|------|------------|
| 01 | [환경 준비](study_guide/01_environment_setup.md) | 0 | 검증된 개발 환경 |
| 02 | [배포 난이도 사다리](study_guide/02_deployment_ladder.md) | 0.5 | 첫 온디바이스 배포 경험 |
| 03 | [양자화 이론](study_guide/03_quantization_theory.md) | 1 | `layer_sensitivity.csv` |
| 04 | [Transformer 양자화 지옥](study_guide/04_transformer_quantization.md) | 2 ★ | `onnx_export_failures.md` |
| 05 | [TensorRT로 첫 완주](study_guide/05_tensorrt.md) | 3 | Orin 성능 리포트 |
| 06 | [멀티 SoC 확장](study_guide/06_multi_soc.md) | 4 | 4-target 성능 매트릭스 |
| 07 | [인프라화](study_guide/07_infrastructure.md) | 5 | `design_rules.md`, 회귀 하네스 |
| 08 | [캡스톤 프로젝트](study_guide/08_capstone.md) | 캡스톤 | 공개 리포 + 블로그 |
| 09 | [12주 로드맵](study_guide/09_roadmap.md) | 로드맵 | 학습 스케줄 |
| 10 | [함정 5개](study_guide/10_pitfalls.md) | 함정 | 실무 체크리스트 |

각 문서 구조: `왜 → 체크리스트 → 이론 → 환경 → 실습 → 결과해석 → 트러블슈팅 → 산출물 → 참고문헌 → 다음`. 총 ~8,000줄.

---

## 📌 정본 버전 스택 (2026-07)

| 도구 | 버전 | 비고 |
|---|---|---|
| CUDA | 12.8 라인 고정 | 12/13 분열 회피 |
| PyTorch | `torch 2.11.0+cu128` | 기준선 |
| **onnx** | **`1.18.0` (IR 11)** | 🔴 **반드시 고정.** ORT 1.23.2의 IR 상한이 11 → 최신 onnx(IR 13)는 로드 실패. export 시 opset ≤ 23 |
| onnxruntime-gpu | **`1.23.2` (CUDA 12)** | `pip install "onnxruntime-gpu<1.27"`. 1.27+는 PyPI 기본이 CUDA 13 |
| TensorRT | `tensorrt-cu12==10.16.1.11` (10.x LTS) | 11.x는 `--int8/--fp16` 제거(strongly-typed) + CUDA 13 → 실습 호환 위해 10.x |
| numpy | **`1.26.4` (`numpy<2`)** | `nuscenes-devkit 1.2.0`이 `numpy<2.0.0` 요구 |
| ExecuTorch | 1.3.x | v1.0 GA(2025-10) 이후. 0단계에선 설치 안 함 |

> ⚠️ 경로 A(호스트 pip)에서는 **`libcudnn.so.9`(cuDNN)와 `libnvinfer.so.10`(TensorRT)를 못 찾아 ONNX Runtime이 조용히 CPU로 fallback**하는 함정이 있습니다(둘 다 CUDA Toolkit deb에는 없고, 각각 venv의 `nvidia-cudnn-cu12`/`tensorrt_libs` 패키지 디렉터리에만 있음). 해결법은 `01_environment_setup.md`의 **3-4-a절**에 있습니다 — 건너뛰지 마세요.

정확한 스택은 [`study_guide/01_environment_setup.md`](study_guide/01_environment_setup.md)를 정본으로 따르세요.

---

## 🗂️ 저장소 구조

```
.
├── study_guide/            # 학습 가이드 (MD + HTML)
│   ├── README.md/.html     # 인덱스
│   └── 01_*.md … 10_*.md   # 단계별 문서 (+ 각 .html)
├── guide (1).html          # 원본 기획 문서 (출발점)
├── CLAUDE.md               # 제작에 쓰인 하네스 포인터
└── .claude/                # 에이전트 팀 + 스킬 (제작 하네스)
    ├── agents/             # guide-author, tech-reviewer
    └── skills/             # research / writing / review / md-to-html / orchestrator
```

---

## 🛠️ 어떻게 만들었나 — 하네스(에이전트 팀)

이 가이드는 [Claude Code](https://claude.com/claude-code) 위에서 **다중 에이전트 하네스**로 제작했습니다: `guide-author` 여러 명이 단계별 문서를 병렬 리서치·작성(Fan-out)하고, `tech-reviewer`가 버전 정합성·명령어·링크를 교차 검증(Fan-in)한 뒤, `md-to-html` 스킬(의존성 없는 자체 파이썬 렌더러 [`.claude/skills/md-to-html/scripts/render.py`](.claude/skills/md-to-html/scripts/render.py))로 HTML을 생성합니다. `.claude/`에 그 구성이 그대로 들어 있습니다.

MD가 정본이고 HTML은 파생물입니다. 내용을 고칠 땐 MD를 수정한 뒤 재렌더하세요:
```bash
python3 .claude/skills/md-to-html/scripts/render.py study_guide
```

---

## 📎 원본

이 저장소는 [`guide (1).html`](<guide (1).html>)의 단계 구조를 기반으로, 실행 가능한 명령어·코드·예시·참고문헌으로 확장한 것입니다.

## 📄 라이선스

[MIT License](LICENSE) — 문서·코드 모두 자유롭게 사용·복사·수정·배포·재라이선스할 수 있으며, 저작권 고지와 라이선스 문구만 포함하면 됩니다.
