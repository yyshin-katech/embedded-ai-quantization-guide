#!/usr/bin/env bash
# W3: bevpoolv2 TensorRT 플러그인(libmmdeploy_tensorrt_ops.so) 직접 빌드
# 풀 mmdeploy CMake 트리를 우회 — bevpoolv2는 2개 TU(kernel.cu + trt_bev_pool.cpp)로 self-contained
set -euo pipefail

FORK=~/capstone-bev/mmdeploy-bevdet
BP=$FORK/csrc/mmdeploy/backend_ops/tensorrt
SP=~/bevf-legacy/lib/python3.10/site-packages
NVCC=~/capstone-bev/cu117/bin/nvcc
OUT=$FORK/build/lib
OBJ=$(mktemp -d /tmp/bevpool_obj.XXXXXX)   # 중간 .o 임시 디렉토리
mkdir -p "$OUT" "$OBJ"

INC="-I$BP/bevpoolv2 -I$BP/common \
-I$HOME/capstone-bev/trt85/include \
-I$HOME/capstone-bev/cu117/include \
-I$SP/nvidia/cublas/include \
-I$SP/nvidia/cudnn/include"

echo "===== [1/3] nvcc: trt_bev_pool_kernel.cu -> kernel.o (sm_86, c++14, fPIC) ====="
$NVCC -c "$BP/bevpoolv2/trt_bev_pool_kernel.cu" -o "$OBJ/kernel.o" \
  -arch=sm_86 -std=c++14 -Xcompiler -fPIC \
  -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1 \
  $INC
echo "  kernel.o: $(stat -c%s "$OBJ/kernel.o") bytes"

echo "===== [2/3] g++: trt_bev_pool.cpp -> plugin.o (c++17, fPIC) ====="
g++ -c "$BP/bevpoolv2/trt_bev_pool.cpp" -o "$OBJ/plugin.o" \
  -fPIC -std=c++17 -O2 \
  $INC
echo "  plugin.o: $(stat -c%s "$OBJ/plugin.o") bytes"

echo "===== [3/3] link -> libmmdeploy_tensorrt_ops.so (MODULE) ====="
g++ -shared "$OBJ/kernel.o" "$OBJ/plugin.o" -o "$OUT/libmmdeploy_tensorrt_ops.so" \
  -L$HOME/capstone-bev/trt85/lib -lnvinfer -lnvinfer_plugin \
  -L$SP/nvidia/cuda_runtime/lib -l:libcudart.so.11.0 \
  -L$SP/nvidia/cublas/lib -l:libcublas.so.11 \
  -L$SP/nvidia/cudnn/lib -l:libcudnn.so.8
echo ""
echo "===== 결과 ====="
ls -la "$OUT/libmmdeploy_tensorrt_ops.so"
echo "--- 미해결 심볼(U) 중 bev_pool/plugin 관련(정상: nvinfer/cudart 런타임 로드분만 남아야) ---"
nm -D -u "$OUT/libmmdeploy_tensorrt_ops.so" | grep -iE "bev_pool|Plugin|Creator" | head
echo "--- 정의된 심볼 중 등록자/creator(T=정의됨) ---"
nm -D "$OUT/libmmdeploy_tensorrt_ops.so" | grep -iE " T .*(Creator|bev_pool)" | head