#!/usr/bin/env bash
# BEVDet bev_pool_v2 커스텀 CUDA op을 sudo·Docker 없이 user-space에서 컴파일한다.
# 근본 벽: bundled mmdet3d(1.0.0rc4)는 torch<=1.13+CUDA<=11.7 커널을 요구하나
#          이 머신의 유일 toolkit은 CUDA 12.8 → torch._check_cuda_version가 MAJOR 불일치로 hard-error.
# 문서 §3은 Docker cu116 / blackwell 패치 2안만 제시 → 둘 다 이 머신엔 불가.
# 해법: cu117 툴체인을 user-space에 조립(제3의 길). 정본 emb-ai venv와 완전 격리.
set -euo pipefail

# ── 0) 격리 legacy venv (torch 1.13.1+cu117). 정본 emb-ai(torch 2.11+cu128) 절대 오염 금지 ──
SP=~/bevf-legacy/lib/python3.10/site-packages
PY=~/bevf-legacy/bin/python

# ── 1) nvcc: micromamba cuda-nvcc=11.7 (완전한 프론트엔드; pip nvidia-cuda-nvcc-cu11은 ptxas만 있어 불충분) ──
#   micromamba create -p ~/capstone-bev/cu117 -c nvidia cuda-nvcc=11.7 cuda-cudart-dev=11.7
#   → ~/capstone-bev/cu117/bin/nvcc = release 11.7 V11.7.99

# ── 2) libcudart: pip nvidia-cuda-runtime-cu11 (실제 libcudart.so.11.0; torch 번들 libcudart-<hash>.so는 -lcudart 링크 불가) ──
#   ~/bevf-legacy/bin/pip install nvidia-cuda-runtime-cu11==11.7.99
#   → $SP/nvidia/cuda_runtime/lib/libcudart.so.11.0

# ── 3) python 헤더: micromamba python=3.10 env (python3.10-dev 미설치·sudo 불요 우회) ──
#   micromamba create -p ~/capstone-bev/pyhdr python=3.10
#   → ~/capstone-bev/pyhdr/include/python3.10/Python.h

# ── 4) CUDA_HOME 조립: bin/include/nvvm는 cu117 심링크, lib64는 실디렉토리 ──
#   mkdir -p ~/capstone-bev/cuda-home/lib64
#   ln -s ~/capstone-bev/cu117/{bin,include,nvvm} ~/capstone-bev/cuda-home/
#   ln -s $SP/nvidia/cuda_runtime/lib/libcudart.so.11.0 ~/capstone-bev/cuda-home/lib64/libcudart.so
#   ln -s $SP/nvidia/cuda_runtime/lib/libcudart.so.11.0 ~/capstone-bev/cuda-home/lib64/libcudart.so.11.0
#   (핵심: `-lcudart`가 libcudart.so → 실제 11.0 런타임으로 해결되게)

# ── 5) 빌드 ──
export CUDA_HOME=~/capstone-bev/cuda-home
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6                       # RTX 3080 sm_86
export CPATH=~/capstone-bev/pyhdr/include/python3.10  # Python.h
cd ~/capstone-bev/BEVDet

nvcc --version   # 반드시 release 11.7 V11.7.99 확인
$PY setup.py build_ext --inplace

# 산출: mmdet3d/ops/bev_pool_v2/bev_pool_v2_ext.cpython-310-x86_64-linux-gnu.so (9,131,040 bytes)
# exports: bev_pool_v2_forward / bev_pool_v2_backward

# ── 6) 두 번째 벽: spconv (bundled mmdet3d detectors/__init__.py가 DAL[LiDAR-fusion] eager import) ──
#   카메라 전용 BEVDet도 이 eager import 때문에 spconv 필요.
#   6개 __init__ 다중패치보다 격리 env에 설치가 깨끗:
#   ~/bevf-legacy/bin/pip install spconv-cu117   # → spconv-cu117 2.3.6 + cumm-cu117 0.4.11

# ── 7) 런타임 로드 검증 (LD_LIBRARY_PATH에 실제 cudart 필요) ──
export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
$PY - <<'EOF'
import torch
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2  # 정확한 경로(패키지 __init__ 재export 없음)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("bev_pool_v2 loaded OK:", bev_pool_v2)
EOF
