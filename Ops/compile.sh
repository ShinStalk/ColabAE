#!/bin/bash

# Define base directories
BASE_DIR="/content/ColabAE/Ops"
CUDA_DIR="/usr/local/cuda"

# Get paths using Python
TF_INCLUDE=$(python -c "import tensorflow as tf; print(tf.sysconfig.get_include())") # /usr/local/lib/python3.10/dist-packages/tensorflow/include
TF_LIB=$(python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())") # /usr/local/lib/python3.10/dist-packages/tensorflow
LINK_FLAGS=($(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))")) # ['-L/usr/local/lib/python3.10/dist-packages/tensorflow', '-l:libtensorflow_framework.so.2']
COMPILE_FLAGS=($(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))")) # ['-I/usr/local/lib/python3.10/dist-packages/tensorflow/include', '-D_GLIBCXX_USE_CXX11_ABI=1', '--std=c++17', '-DEIGEN_MAX_ALIGN_BYTES=64']

# Associative array of operation/library names
declare -A OP_NAMES
OP_NAMES=( ["approxmatch"]=1 ["grouping"]=1 ["interpolate"]=0 ["nndistance"]=1)  # 1: CUDA, 0: NO CUDA

for OP_NAME in "${!OP_NAMES[@]}"; do
    OP_DIR="$BASE_DIR/$OP_NAME"

    if [ "${OP_NAMES[$OP_NAME]}" -eq 1 ]; then
        # Lib building with $OP_DIR/${OP_NAME}_g.cu.o included
        # Cuda compilation
        $CUDA_DIR/bin/nvcc $OP_DIR/${OP_NAME}_g.cu -o $OP_DIR/${OP_NAME}_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
        # shellcheck disable=SC2068
        g++ ${COMPILE_FLAGS[@]} $OP_DIR/$OP_NAME.cpp $OP_DIR/${OP_NAME}_g.cu.o -o $OP_DIR/${OP_NAME}.so -shared -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIB ${LINK_FLAGS[@]} -I$CUDA_DIR/include -lcudart -L$CUDA_DIR/lib64/ -O2
    else
        # Lib building without $OP_DIR/${OP_NAME}_g.cu.o
        # shellcheck disable=SC2068
        g++ ${COMPILE_FLAGS[@]} $OP_DIR/$OP_NAME.cpp -o $OP_DIR/${OP_NAME}.so -shared -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIB ${LINK_FLAGS[@]} -I$CUDA_DIR/include -lcudart -L$CUDA_DIR/lib64/ -O2
    fi
done