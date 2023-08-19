#!/bin/bash

# Define base directories
BASE_DIR="/content/ColabAE/Ops"
CUDA_DIR="/usr/local/cuda"

# Get paths using Python
TF_INCLUDE=$(python -c "import tensorflow as tf; print(tf.sysconfig.get_include())")
TF_LIB=$(python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())")
LINK_FLAGS=($(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"))
COMPILE_FLAGS=($(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"))

# List of operation/library names
OP_NAMES=("approxmatch" "anotherop" "yetanotherop")  # Add more names as required

for OP_NAME in "${OP_NAMES[@]}"; do
    OP_DIR="$BASE_DIR/$OP_NAME"

    # Cuda compilation
    $CUDA_DIR/bin/nvcc $OP_DIR/${OP_NAME}_g.cu -o $OP_DIR/${OP_NAME}_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

    # Lib building
    g++ ${COMPILE_FLAGS[@]} $OP_DIR/$OP_NAME.cpp $OP_DIR/${OP_NAME}_g.cu.o -o $OP_DIR/${OP_NAME}_so.so -shared -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIB ${LINK_FLAGS[@]} -I$CUDA_DIR/include -lcudart -L$CUDA_DIR/lib64/ -O2
done

# FLAGS and Folders dynamic paths
# import tensorflow as tf;
# print(tf.sysconfig.get_include()) # /usr/local/lib/python3.10/dist-packages/tensorflow/include
# print(tf.sysconfig.get_lib()) # /usr/local/lib/python3.10/dist-packages/tensorflow
# print(tf.sysconfig.get_link_flags()) # ['-L/usr/local/lib/python3.10/dist-packages/tensorflow', '-l:libtensorflow_framework.so.2']
# print(tf.sysconfig.get_compile_flags()) # ['-I/usr/local/lib/python3.10/dist-packages/tensorflow/include', '-D_GLIBCXX_USE_CXX11_ABI=1', '--std=c++17', '-DEIGEN_MAX_ALIGN_BYTES=64']
