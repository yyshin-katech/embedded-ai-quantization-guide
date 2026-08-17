"""
Stage 3 (TensorRT) 실기 검증 — 공통 모듈.

핵심 실측 전제(2026-08-17, AI-LAP / RTX 3080):
- TensorRT 10.16.1.11 (pip 휠, ~/emb-ai). trtexec 바이너리는 휠에 없음 → 엔진 빌드는
  polygraphy(0.50.3) Python API로 수행한다. 이 모듈이 그 대체 경로를 캡슐화한다.
- ImageNet val 50k: ~/stage1-work/data (저장소 밖). crop_tv 캐시가 torchvision 공식
  전처리(ResNet FP32 공개값 재현)이며 정확도 판정에 쓴다. squash와 −1.07%p 차이가 있으니
  절대 섞지 않는다 (1단계 함정 2-b).
"""
import os
import subprocess
import numpy as np

REPO = "/home/yuyeong/embedded-ai-quantization-guide"
WS = os.path.join(REPO, "_workspace", "stage3")
DATA = "/home/yuyeong/stage1-work/data"
VAL_FULL = os.path.join(DATA, "val_full")
LABELS = os.path.join(DATA, "labels", "val_synset_map.txt")
CACHE = os.path.join(DATA, "cache")
os.makedirs(WS, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def liveness():
    """필수 규약: 모든 GPU 작업 전 라이브니스 확인."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    assert "GPU 0" in out, "GPU not live:\n" + out
    return out.strip()


# ---------- ImageNet (crop_tv = torchvision 공식 전처리) ----------
def load_tv_cache():
    """crop_tv 캐시(NHWC uint8 memmap, 50000x224x224x3)와 정수 라벨을 반환.

    1단계 검증: 이 캐시로 ResNet FP32 top-1이 공개값과 0.05%p 일치.
    """
    tv = np.load(os.path.join(CACHE, "tv.npy"), mmap_mode="r")  # (50000,224,224,3) uint8
    labels = []
    with open(LABELS) as f:
        for line in f:
            parts = line.split()
            labels.append(int(parts[2]))
    return tv, np.array(labels, dtype=np.int64)


def preprocess_nchw(tv_uint8_nhwc):
    """(N,224,224,3) uint8 → (N,3,224,224) float32, ImageNet 정규화."""
    x = tv_uint8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x, dtype=np.float32)


def calib_indices(n=200):
    """클래스별 첫 장 캘리브 분할(calib200 ⊂ calib1000)."""
    p = os.path.join(CACHE, f"calib{n}_idx.npy")
    return np.load(p)


# ---------- TensorRT 엔진 빌드 (trtexec 대체 = polygraphy Python API) ----------
def build_engine(onnx_path, precision, calibrator=None, workspace_gb=6, save_path=None):
    """polygraphy로 ONNX→TRT 엔진 빌드. precision ∈ {fp32, fp16, int8, int8_fp16}.

    - fp32/fp16: 순수 정밀도 플래그.
    - int8(+QDQ 그래프): Q/DQ 노드가 스케일을 운반(explicit). int8 플래그만.
    - int8(+calibrator): QDQ 없는 FP 그래프 + implicit 캘리브레이션(IInt8*Calibrator2).
    """
    import tensorrt as trt
    from polygraphy.backend.trt import network_from_onnx_path, engine_from_network, CreateConfig, save_engine

    flags = {}
    if precision in ("fp16", "int8_fp16"):
        flags["fp16"] = True
    if precision in ("int8", "int8_fp16"):
        flags["int8"] = True
        if calibrator is not None:
            flags["calibrator"] = calibrator
    config = CreateConfig(
        memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_gb << 30},
        **flags,
    )
    engine = engine_from_network(network_from_onnx_path(onnx_path), config=config)
    if save_path:
        save_engine(engine, save_path)
    return engine


def bench_latency(engine, feed, iters=300, warmup=80):
    """polygraphy TrtRunner로 steady-state 지연 측정. feed={input_name: np.ndarray}.

    반환: p50/p90/mean(ms), throughput(infer/s). last_inference_time()은 execute+sync 구간.
    입력 shape/dtype이 정밀도 간 동일하므로 상대 비교는 공정(절대값엔 H2D/D2H 포함).
    """
    from polygraphy.backend.trt import TrtRunner
    times = []
    with TrtRunner(engine) as runner:
        for _ in range(warmup):
            runner.infer(feed)
        for _ in range(iters):
            runner.infer(feed)
            times.append(runner.last_inference_time())
    t = np.array(times) * 1000.0  # ms
    return dict(
        p50=float(np.percentile(t, 50)),
        p90=float(np.percentile(t, 90)),
        mean=float(t.mean()),
        std=float(t.std()),
        throughput=float(1000.0 / np.median(t)),
        iters=int(iters),
    )


def engine_layer_info(engine):
    """엔진 레이어/전술 정보(ONELINE) — QDQ fusion(int8/imma 커널) 증거 확보용."""
    import tensorrt as trt
    insp = engine.create_engine_inspector()
    return insp.get_engine_information(trt.LayerInformationFormat.ONELINE)


def evaluate_top1(engine, tv, labels, indices, input_name, batch=64):
    """TRT 엔진으로 top-1 정확도 측정(subset). tv=NHWC uint8 memmap."""
    from polygraphy.backend.trt import TrtRunner
    correct = 0
    total = 0
    with TrtRunner(engine) as runner:
        for i in range(0, len(indices), batch):
            idx = indices[i:i + batch]
            imgs = np.stack([tv[j] for j in idx])  # (b,224,224,3) uint8
            x = preprocess_nchw(imgs)
            # 엔진이 고정 배치일 수 있으므로 마지막 배치가 작으면 패딩 후 잘라냄
            out = runner.infer({input_name: x})
            logits = list(out.values())[0]
            pred = logits.argmax(axis=1)
            correct += int((pred == labels[idx]).sum())
            total += len(idx)
    return correct / total, total
