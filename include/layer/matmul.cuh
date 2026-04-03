#ifndef LAYER_MATMUL_H
#define LAYER_MATMUL_H
#include "../common.h"

// SHAPE:
// inp is (N,H,W), weight is (O,W), bias is (O)
// out is (N,H,O)

enum class MATMUL_TYPE{
    CPU,
    GPU_NAIVE,
    GPU_CUBLAS,
    GPU_CUBLASLT
};

void matmul_forward(float *out,
                    const float *input, const float *weight, const float *bias,
                    int N, int H, int W, int O,
                    const int sqrt_block_size = 0, ACTIVATION_FN act_fn = ACTIVATION_FN::NONE, MATMUL_TYPE type = MATMUL_TYPE::CPU);

void matmul_backward();

#endif