#!/usr/bin/env python
"""BEVDet-R50 FP32 forward latency (batch 1, CUDA event-timed).

bev_pool_v2 커스텀 CUDA op이 GPU에서 실제 실행됨을 정량 증명한다.
init 가중치(torchvision backbone + random head)라도 forward 계산 그래프는 정식 모델과 동일 →
latency는 가중치 값과 무관하게 구조 유효(정확도는 별건, fp32_baseline_eval.json 참조).

실행(격리 legacy env + user-space cu117 런타임):
  SP=~/bevf-legacy/lib/python3.10/site-packages
  export CUDA_HOME=~/capstone-bev/cuda-home
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
  cd ~/capstone-bev/BEVDet && export PYTHONPATH=$PWD
  ~/bevf-legacy/bin/python bench_fp32_latency.py
"""
import json
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader

CFG = 'configs/bevdet/bevdet-r50.py'
CKPT = 'work_dirs/capstone/init_r50.pth'
WARMUP, N = 5, 30

cfg = Config.fromfile(CFG)
cfg.model.pretrained = None
cfg.data.test.test_mode = True

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu')
model = MMDataParallel(model.cuda().eval(), device_ids=[0])

# 실제 val 샘플 1개 (LSS view-transform이 요구하는 sensor calib 메타 포함) — dummy tensor로는 forward 불가
ds = build_dataset(cfg.data.test)
dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
data = next(iter(dl))

torch.cuda.reset_peak_memory_stats()
ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

with torch.no_grad():
    for _ in range(WARMUP):
        model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize()
    for i in range(N):
        ev_s[i].record()
        model(return_loss=False, rescale=True, **data)
        ev_e[i].record()
    torch.cuda.synchronize()

lat = np.array([s.elapsed_time(e) for s, e in zip(ev_s, ev_e)])  # ms
peak = torch.cuda.max_memory_allocated() / 1024**2
n_params = sum(p.numel() for p in model.parameters()) / 1e6

out = {
    'model': 'BEVDet-r50', 'device': torch.cuda.get_device_name(0), 'batch': 1,
    'precision': 'FP32',
    'latency_ms_p50': round(float(np.percentile(lat, 50)), 4),
    'latency_ms_mean': round(float(lat.mean()), 4),
    'latency_ms_min': round(float(lat.min()), 4),
    'latency_ms_max': round(float(lat.max()), 4),
    'n_iters': N, 'peak_mem_MiB': round(peak, 1), 'params_M': round(n_params, 2),
    'note': 'init-only weights; latency structure valid, accuracy not',
}
print(json.dumps(out, indent=2))
