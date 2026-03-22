# mnist-cuda

## System 

## Setup Docker
`docker run -it --name=mnist-cuda --gpus all -v $(pwd):/root/MNIST-CUDA/ nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 `

``` bash
apt update
export DEBIAN_FRONTEND=noninteractive
apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    python3 \
    python3-pip \
    python3-venv \
    protobuf-compiler \
    libhdf5-dev \
    jq \
    gdb
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

```
## Run
### Download Data
In this project, I use HDF5 data format
...
### Training with CPU
Use openBLAS for speed up computation time

### Training with GPU
1. PyTorch CUDA
2. C++ and cuBLAS
3. C++ and cuBLASLt
### Benmark

## Material
1. https://github.com/karpathy/llm.c
2. https://github.com/Infatoshi/cuda-course?tab=readme-ov-file