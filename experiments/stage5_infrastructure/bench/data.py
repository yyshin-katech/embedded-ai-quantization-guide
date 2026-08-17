# bench/data.py
# 학습가이드 §4-5 run_bench.py는 `from data import Loader, Evaluator`를 import하지만
# 정작 data.py를 제공하지 않는다(데이터층은 프로젝트마다 다르므로 문서가 비워둔 자리).
# 이 검증 인스턴스는 그 자리를 ResNet50/ImageNet-val 분류로 채운다(2026-08-17).
#
# 데이터: ~/stage1-work/data/cache/tv.npy  — torchvision 공식 전처리(crop_tv)로 캐싱된
#   (50000,224,224,3) uint8 memmap. 1단계 검증에서 ResNet FP32 top-1이 공개값과 일치한 그 캐시.
#   squash 캐시와 −1.07%p 차이가 있으니 섞지 않는다(1단계 함정 2-b).
# accuracy 필드는 여기서 top-1(0~1). BenchResult 스키마는 mAP와 동일하게 쓴다.
import os
import numpy as np

DATA = os.path.expanduser("~/stage1-work/data")
CACHE = os.path.join(DATA, "cache")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _preprocess_nchw(tv_uint8_nhwc):
    """(N,224,224,3) uint8 → (N,3,224,224) float32, ImageNet 정규화."""
    x = tv_uint8_nhwc.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x, dtype=np.float32)


class Loader:
    """엔진이 고정 배치1이라 latency 샘플·eval 모두 (1,3,224,224)로 공급한다.

    - one_batch(): 대표 입력 1개(latency 측정용)
    - eval_set():  eval_n장을 1장씩 yield (정확도용)
    - gts():       eval_n개 정수 라벨
    """
    def __init__(self, model_cfg=None, eval_n=5000):
        self.eval_n = eval_n
        self._tv = np.load(os.path.join(CACHE, "tv.npy"), mmap_mode="r")  # (50000,224,224,3) uint8
        self._labels = np.load(os.path.join(CACHE, "labels.npy")).astype(np.int64)
        self._eval_idx = np.arange(0, eval_n)
        self._sample = _preprocess_nchw(self._tv[0][None])  # (1,3,224,224)

    def one_batch(self):
        return self._sample

    def eval_set(self):
        for j in self._eval_idx:
            yield _preprocess_nchw(self._tv[j][None])

    def gts(self):
        return self._labels[self._eval_idx]

    # INT8 캘리브레이션 피드(백엔드에 주입) — calib200 분할을 1장씩.
    def calib_feed(self, n=200, in_name="input"):
        idx = np.load(os.path.join(CACHE, f"calib{n}_idx.npy"))

        def _gen():
            for j in idx:
                yield {in_name: _preprocess_nchw(self._tv[int(j)][None])}
        return _gen


class Evaluator:
    """분류 top-1. preds=[(1,1000) logits ...], gts=[label ...]."""
    def compute_acc(self, preds, gts):
        gts = np.asarray(gts)
        pred_cls = np.array([np.asarray(p).reshape(-1).argmax() for p in preds])
        return float((pred_cls == gts).mean())

    # 문서 원안은 compute_map(검출). 분류 검증이라 top-1으로 계약을 채운다(이름만 다름).
    compute_map = compute_acc
