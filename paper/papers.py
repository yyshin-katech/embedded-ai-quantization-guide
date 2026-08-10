"""가이드 문서에서 추출한 논문 목록 — 단일 출처(SoT). 다운로더와 HTML 생성기가 같이 읽는다."""

# (arxiv_id, 저자/연도, 제목, 학회, 그룹, 파일명 stem, 한 줄 설명, 인용 문서들)
PAPERS = [
    # ── A. 양자화 코어 (기초 → 심화) ─────────────────────────────────────────
    ("1712.05877", "Jacob et al. (2018)",
     "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
     "CVPR 2018 (arXiv 2017-12)", "core", "jacob2018-integer-only-inference",
     "정수 전용 추론 스킴의 <b>원조</b>. affine 양자화 수식 <code>r = S(q − Z)</code>와 정수 MAC 유도가 여기서 나온다. "
     "대칭 weight(zero-point=0)가 왜 정수 곱을 단순하게 만드는지가 이 논문의 §2.",
     ["02", "03", "05", "06", "08", "10", "deep-dive"]),

    ("1806.08342", "Krishnamoorthi (2018)",
     "Quantizing deep convolutional networks for efficient inference: A whitepaper",
     "arXiv 2018-06", "core", "krishnamoorthi2018-quantizing-cnn-whitepaper",
     "Jacob 논문의 <b>실무 확장판</b>. per-channel weight 양자화가 per-tensor보다 왜 이기는지, PTQ만으로 2% 이내 손실이 "
     "가능한 조건을 벤치마크로 보여준다. 1단계 layer sensitivity 실습의 배경.",
     ["deep-dive"]),

    ("2103.13630", "Gholami et al. (2021)",
     "A Survey of Quantization Methods for Efficient Neural Network Inference",
     "arXiv 2021-03 (Berkeley)", "core", "gholami2021-quantization-survey",
     "<b>가장 먼저 읽을 지도</b>. PTQ/QAT, uniform/non-uniform, per-tensor/per-channel, 대칭/비대칭 — 이 가이드가 쓰는 "
     "용어 체계가 전부 여기서 정의된다. 수식보다 분류에 강하다.",
     ["02", "03", "05", "06", "07", "08", "09", "10", "deep-dive"]),

    ("2106.08295", "Nagel et al. (2021)",
     "A White Paper on Neural Network Quantization",
     "arXiv 2021-06 (Qualcomm AI Research)", "core", "nagel2021-quantization-white-paper",
     "<b>실무 레시피의 정본</b>. 캘리브레이션(MinMax/MSE/entropy) 비교, AdaRound, CLE(cross-layer equalization), "
     "bias correction, QAT 루프까지 '무엇을 어떤 순서로 시도할지'를 플로차트로 준다.",
     ["03", "05", "06", "07", "08", "09", "10", "deep-dive"]),

    ("1902.08153", "Esser et al. (2020)",
     "Learned Step Size Quantization",
     "ICLR 2020", "core", "esser2020-learned-step-size-quantization",
     "scale(step size) 자체를 학습 파라미터로 두는 QAT. 1단계 §2.5.4의 출처. 8bit에선 이득이 작지만 "
     "<b>2~4bit에서 격차가 벌어진다</b>.",
     ["03"]),

    # ── B. Transformer / ViT / LLM 양자화 ────────────────────────────────────
    ("2211.10438", "Xiao et al. (2022)",
     "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models",
     "ICML 2023 (arXiv 2022-11, MIT HAN Lab)", "transformer", "xiao2022-smoothquant",
     "2단계 <b>필독</b>. activation outlier의 어려움을 weight로 옮기는 수학적 트릭 "
     "<code>s_j = max(|X_j|)^α / max(|W_j|)^(1−α)</code> (기본 α=0.5). LayerNorm 뒤 INT8이 왜 무너지는지의 답.",
     ["04", "07", "09", "10"]),

    ("2111.13824", "Lin et al. (2021)",
     "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer",
     "IJCAI 2022", "transformer", "lin2021-fq-vit",
     "ViT를 <b>완전 정수화</b>한 초기 성과. Power-of-Two Factor로 LayerNorm의 inter-channel 분산을, "
     "Log-Int-Softmax로 attention을 4bit까지 내린다.",
     ["04"]),

    ("2111.12293", "Yuan et al. (2021)",
     "PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization",
     "ECCV 2022", "transformer", "yuan2021-ptq4vit",
     "Softmax·GELU 출력이 <b>비가우시안</b>이라 단일 uniform 격자로 못 덮는다는 진단 + twin uniform 해법. "
     "Hessian-guided scale 탐색으로 8bit에서 0.5% 미만 하락.",
     ["04"]),

    ("2212.08254", "Li et al. (2022)",
     "RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers",
     "ICCV 2023", "transformer", "li2022-repq-vit",
     "캘리브레이션은 정확한 격자(channel-wise/log√2)로, 추론은 하드웨어 친화 격자(layer-wise/log2)로 — "
     "<b>둘을 분리</b>해 4bit PTQ를 실사용 가능하게 만든다.",
     ["04"]),

    # ── C. BEV / 3D 검출 (캡스톤 대상 모델) ──────────────────────────────────
    ("2010.04159", "Zhu et al. (2020)",
     "Deformable DETR: Deformable Transformers for End-to-End Object Detection",
     "ICLR 2021", "bev", "zhu2020-deformable-detr",
     "deformable attention의 <b>원류</b>. BEVFormer가 그대로 차용하며, 이 op이 ONNX 표준에 없어서 "
     "2단계·캡스톤의 커스텀 플러그인 문제가 시작된다.",
     ["08"]),

    ("2008.05711", "Philion & Fidler (2020)",
     "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D",
     "ECCV 2020", "bev", "philion2020-lift-splat-shoot",
     "카메라 이미지를 BEV로 올리는 LSS 뷰 변환의 원류. BEVDet의 전제.",
     ["08"]),

    ("2112.11790", "Huang et al. (2021)",
     "BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View",
     "arXiv 2021-12", "bev", "huang2021-bevdet",
     "LSS + BEVPoolv2 기반. attention이 없어 <b>배포 난도가 셋 중 가장 낮다</b> — 캡스톤에서 TensorRT INT8까지 "
     "가장 먼저 내려볼 후보.",
     ["08"]),

    ("2203.05625", "Liu et al. (2022)",
     "PETR: Position Embedding Transformation for Multi-View 3D Object Detection",
     "ECCV 2022", "bev", "liu2022-petr",
     "표준 attention 비중이 커서 deformable 계열보다 <b>배포 우호적</b>. 3D 위치를 PE로 주입한다.",
     ["08"]),

    ("2203.17270", "Li et al. (2022)",
     "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers",
     "ECCV 2022", "bev", "li2022-bevformer",
     "spatial + temporal cross-attention. 정확도는 높지만 MSDeformAttn 때문에 <b>양자화·배포 난도가 최상</b>. "
     "2단계 전체가 이 모델의 배포 실패를 다룬다.",
     ["04", "08"]),

    ("2403.10913", "Xu et al. (2024)",
     "DEFA: Efficient Deformable Attention Acceleration via Pruning-Assisted Grid-Sampling and Multi-Scale Parallel Processing",
     "arXiv 2024-03", "bev", "xu2024-defa",
     "MSDeformAttn의 <b>random-access grid sampling</b>이 왜 NPU에서 PE 활용률을 떨어뜨리는지를 하드웨어 관점에서 "
     "분해한다. 2단계 §552의 근거.",
     ["04"]),

    ("2505.14022", "Huang et al. (2025)",
     "Towards Efficient Multi-Scale Deformable Attention on NPU",
     "arXiv 2025-05", "bev", "huang2025-msda-on-npu",
     "Ascend NPU에서 MSDA grid sampling을 분해·융합·병렬화. '가속기가 이 op을 싫어한다'를 정면으로 다룬 최신 작업.",
     ["04"]),

    ("2502.15488", "FQ-PETR (2025)",
     "FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection",
     "arXiv 2025-02", "bev", "fq-petr-2025",
     "멀티뷰 3D 검출의 <b>완전 양자화</b> 후속. PE 양자화를 다뤄 2단계 decoder 문제의 최신 대안을 보여준다.",
     ["04"]),

    # ── D. 데이터셋 / 런타임 백서 ────────────────────────────────────────────
    ("1903.11027", "Caesar et al. (2020)",
     "nuScenes: A Multimodal Dataset for Autonomous Driving",
     "CVPR 2020", "infra", "caesar2020-nuscenes",
     "0단계에서 내려받는 데이터셋의 원 논문. mAP/NDS 지표 정의가 여기 있어 캡스톤 평가 해석에 필요하다.",
     ["01", "08"]),

    ("2605.08195", "Nachin et al. (2026)",
     "ExecuTorch — A Unified PyTorch Solution to Run AI Models On-Device",
     "arXiv 2026-05", "infra", "nachin2026-executorch",
     "0.5단계 Lv.4에서 쓰는 ExecuTorch의 설계 백서. PT2E 양자화와 <code>.pte</code> 파이프라인의 1차 근거.",
     ["02"]),
]
