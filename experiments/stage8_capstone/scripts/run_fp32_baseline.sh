#!/usr/bin/env bash
# BEVDet-R50 FP32 baseline — nuScenes-mini val 81장 end-to-end eval (stock tools/test.py).
# walking skeleton 완주 확인: export→bev_pool_v2 CUDA op(GPU)→nuScenes eval 하네스 관통.
# 산출 → results/fp32_baseline_eval.json
set -euo pipefail

SP=~/bevf-legacy/lib/python3.10/site-packages
export CUDA_HOME=~/capstone-bev/cuda-home
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}   # user-space cu117 libcudart
cd ~/capstone-bev/BEVDet
export PYTHONPATH=$PWD

nvidia-smi -L   # GPU liveness

# init_r50.pth = torchvision://resnet50 backbone + init_weights() head (Baidu 정식 가중치 대체).
#   ── 정직한 폴백 ──
#   BEVDet-R50 detection 정식 가중치는 Baidu Pan 전용(헤드리스 접근 불가, WebSearch 2회로 미러 못 찾음).
#   → backbone만 ImageNet 실사전학습, LSS/BEV-encoder/head는 랜덤 → mAP 0은 정상·예상값(파이프라인 검증용).
~/bevf-legacy/bin/python tools/test.py \
    configs/bevdet/bevdet-r50.py \
    work_dirs/capstone/init_r50.pth \
    --eval bbox

# 실측(81 val): mAP 0.0000 · NDS 0.0260 · mATE 1.0442 · mASE 0.9705 · mAOE 1.0837 · mAVE 0.8942 · mAAE 0.8750
# 해석: walking skeleton(문서 §9 완주 기준) 충족. 절대 정확도는 Baidu 가중치 확보 시 재실행.
