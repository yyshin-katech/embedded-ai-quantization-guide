#!/usr/bin/env python
"""벤치용 실 입력 샘플 덤프: BEVDetTRT 모델로 nuScenes-mini 첫 샘플의
img + 5개 ranks(metas)를 계산해 npz로 저장. 벤치 스크립트가 무거운 모델
빌드 없이 엔진만 돌릴 수 있게 분리.

엔진 입력 순서(convert export와 동일):
  img, ranks_depth(metas[1]), ranks_feat(metas[2]), ranks_bev(metas[0]),
  interval_starts(metas[3]), interval_lengths(metas[4])
"""
import os, sys, numpy as np, torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

CFG = os.path.expanduser("~/capstone-bev/BEVDet/configs/bevdet/bevdet-r50.py")
CKPT = os.path.expanduser("~/capstone-bev/BEVDet/work_dirs/capstone/init_r50.pth")
OUT = os.path.expanduser("~/capstone-bev/BEVDet/work_dirs/capstone/bench_sample.npz")

cfg = Config.fromfile(CFG)
cfg.model.pretrained = None
cfg.model.type = cfg.model.type + 'TRT'
cfg.gpu_ids = [0]
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                          dist=False, shuffle=False)
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu')
model.cuda().eval()

for data in loader:
    inputs = [t.cuda() for t in data['img_inputs'][0]]
    metas = model.get_bev_pool_input(inputs)
    img = inputs[0].squeeze(0).float().contiguous()
    arrs = dict(
        img=img.detach().cpu().numpy().astype(np.float32),
        ranks_depth=metas[1].int().detach().cpu().numpy().astype(np.int32),
        ranks_feat=metas[2].int().detach().cpu().numpy().astype(np.int32),
        ranks_bev=metas[0].int().detach().cpu().numpy().astype(np.int32),
        interval_starts=metas[3].int().detach().cpu().numpy().astype(np.int32),
        interval_lengths=metas[4].int().detach().cpu().numpy().astype(np.int32),
    )
    break

np.savez(OUT, **arrs)
for k, v in arrs.items():
    print(f"  {k}: {v.shape} {v.dtype}")
print("SAVED:", OUT)
os._exit(0)
