/usr/local/cuda/bin/nvcc /content/ColabAE/Ops/approxmatch/tf_approxmatch_g.cu -o /content/ColabAE/Ops/approxmatch/tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF>=1.4.0
g++ --std=c++17 /content/ColabAE/Ops/approxmatch/tf_approxmatch.cpp /content/ColabAE/Ops/approxmatch/tf_approxmatch_g.cu.o -o /content/ColabAE/Ops/approxmatch/tf_approxmatch_so.so -shared -fPIC -I/usr/local/lib/python3.10/dist-packages/tensorflow/include/ -I/usr/local/lib/python3.10/dist-packages/tensorflow/include/external/nsync/public -L/usr/local/lib/python3.10/dist-packages/tensorflow -l:libtensorflow_framework.so.2 -I/usr/local/cuda/include -lcudart -L/usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64
