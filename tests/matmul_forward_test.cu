#include "../include/utils/float_utils.h"
#include "../include/layer/matmul.cuh"

int main(int argc, char **argv){
    srand(2101);
    
    int N = 32;
    int H = 512;
    int W = 768;
    int O = 768 * 4;

    int deviceIdx = 0;
    CHECK_CUDA(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);


    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasLtCreate(&cublaslt_handle));

    int enable_tf32 = deviceProp.major >= 8 ? 1:0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    CHECK_CUDA(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    float *out = (float*)malloc(N * H * O * sizeof(float));
    float *input = create_rand_float(N * H * W);
    float *weight = create_rand_float(O * W);
    float *bias = create_rand_float(O);

    float *d_out;
    float *d_inp;
    float *d_weight;
    float *d_bias;
    CHECK_CUDA(cudaMalloc(&d_out, N*H*O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inp, N*H*W*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, W*O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, O*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, input, N*H*W*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight, W*O*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, O*sizeof(float), cudaMemcpyHostToDevice));

    int n_kernel = 4;
    
    int sqrt_block_sizes[] = {4,8,16,32};
    printf("CPU start\n");
    matmul_forward(out, input, weight, bias, N, H, W, O);
    printf("CPU end\n");

    for (int i = 0; i < sizeof(sqrt_block_sizes)/sizeof(int); i++ ){
        int sqrt_block_size = sqrt_block_sizes[i] ;
        for (int j = 1; j < n_kernel; j++){
            printf("Block size: %d \n Kernel: %d \n", sqrt_block_size, j);
            CHECK_CUDA(cudaMemset(d_out, 0, N * H * O * sizeof(float)));
            matmul_forward(d_out, d_inp, d_weight, d_bias, N, H, W, O, sqrt_block_size, ACTIVATION_FN::NONE, static_cast<MATMUL_TYPE>(j));
            validate_result(d_out, out, "Matmul output", N*H*O, 1e-1f);
        }
    }
    free(out);
    free(input);
    free(weight);
    free(bias);
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_inp));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(cublaslt_workspace));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUBLAS(cublasLtDestroy(cublaslt_handle));
    return 0;
}