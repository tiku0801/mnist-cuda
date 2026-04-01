#include "layer/matmul.cuh"


void matmul_cpu(float *out,
                const float *inp, const float *weight, const float *bias,
                int N, int H, int W, int O){
    for (int n = 0; n < N; n++){
        for (int h = 0; h < H; h++){
            float *out_NH = out + n*H*O + h*O;
            const float *inp_NH = inp + n*H*W + h*W;
            for (int o = 0; o < O; o++){
                float val = (bias != nullptr) ? bias[o] : 0.0f;
                const float *wrow = weight + o*W;
                for (int w = 0; w < W ; w++){
                    val += inp_NH[w]*wrow[w];
                }
                out_NH[o] = val;
            }  
        }
    }
}


__global__ void matmul_naive_kernel_1(float *out,
                const float *inp, const float *weight, const float *bias,
                int N, int H, int W, int O){
    int nh = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (nh < N*H && o < O){
        float val = (bias != nullptr) ? bias[o] : 0.0f;
        const float *wrow = weight + o*W;
        const float *inp_NH = inp + nh*W;
        for (int w = 0; w < W; w++){
            val += wrow[w]*inp_NH[w];
        }
        out[nh*O + o] = val;
    }
}

void matmul_naive(float *out,
            const float *inp, const float *weight, const float *bias,
            int N, int H, int W, int O, const int sqrt_block_size){
    dim3 gridDim(ceil_div(N*H, sqrt_block_size), ceil_div(O, sqrt_block_size));
    dim3 blockDim(sqrt_block_size,sqrt_block_size);
    matmul_naive_kernel_1<<<gridDim, blockDim>>>(out, inp, weight, bias, N, H, W, O);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void add_bias(float *out, const float *bias, int N, int H, int O){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N * H * O; i+=stride){
        int col = i % O;
        out[i] += bias[col];
    }
}


void matmul_cublas(float *out,
            const float *inp, const float *weight, const float *bias,
            int N, int H, int W, int O, const int sqrt_block_size){
/*
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const float           *alpha,
                            const float           *A, int lda,
                            const float           *B, int ldb,
                            const float           *beta,
                            float           *C, int ldc)

    C = alpha*op(A) @ op(B) + beta*C;
    inp is (N,H,W), weight is (O,W)
    out is (N,H,O)
    => out = inp @ weight.T
    => out.T = weight @ inp.T 
    but cublas use col-major => all tensor/matrix is transposed
    => out.T = weight.T @ inp
*/
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                O, N*H, W,
                &alpha, 
                weight, W,
                inp, W,
                &beta,
                out, O));

    if (bias != nullptr){
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(N*H*O, block_size);
        add_bias<<<grid_size,block_size>>>(out, bias, N, H, O);
        CHECK_CUDA(cudaGetLastError());
    }
}

void matmul_cublasLt(float *out, const float *inp, const float *weight, const float *bias,
                    int N, int H, int W, int O, ACTIVATION_FN act_fn){
/*
     C = alpha*op(A) @ op(B) + beta*C;
    inp is (N,H,W), weight is (O,W)
    out is (N,H,O)
    => out = inp @ weight.T
    => out.T = weight @ inp.T 
    but cublas use col-major => all tensor/matrix is transposed
    => out.T = weight.T @ inp
*/
   bool has_bias = (bias != nullptr);

    int returnResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout, inputLayout, biasLayout, outputLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNormal = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtEpilogue_t epilouge = CUBLASLT_EPILOGUE_DEFAULT;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));

    if (has_bias){
        switch (act_fn)
        {
        case ACTIVATION_FN::RELU:
            epilouge = CUBLASLT_EPILOGUE_RELU_BIAS;         
            break;
        case ACTIVATION_FN::GELU:
            epilouge = CUBLASLT_EPILOGUE_GELU_BIAS;
        default:
            epilouge = CUBLASLT_EPILOGUE_BIAS;
            break;
        }
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));

    }
    else {
        switch (act_fn)
        {
        case ACTIVATION_FN::RELU:
            epilouge = CUBLASLT_EPILOGUE_RELU;         
            break;
        case ACTIVATION_FN::GELU:
            epilouge = CUBLASLT_EPILOGUE_GELU;
        default:
            break;
        } 
    }

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNormal, sizeof(opNormal)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilouge, sizeof(epilouge)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, W, O, W));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, W, N*H, W));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, O, N*H, O));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, O, 1, O));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, weightLayout, inputLayout, biasLayout, outputLayout, preference, 1, &heuristic, &returnResults));

    if (returnResults == 0){
        printf("No suitable CublasLt Matmul algorithm ");
        exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(cublaslt_handle, operationDesc,
                            &alpha, weight, weightLayout, inp, inputLayout,
                            &beta, out, outputLayout, out, outputLayout, &heuristic.algo,
                            cublaslt_workspace, cublaslt_workspace_size, 0));
    
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(weightLayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(inputLayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(outputLayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(biasLayout));
}

void matmul_forward(float *out,
                    const float *input, const float *weight, const float *bias,
                    int N, int H, int W, int O,
                    const int sqrt_block_size = 0, ACTIVATION_FN act_fn = ACTIVATION_FN::NONE, MATMUL_TYPE type = MATMUL_TYPE::CPU){
    switch (type)
    {
    case MATMUL_TYPE::GPU_NAIVE:
        matmul_naive(out, input, weight, bias, N, H, W, O, sqrt_block_size);
        break;
    case MATMUL_TYPE::GPU_CUBLAS:
        matmul_cublas(out, input, weight, bias, N, H, W, O, sqrt_block_size);
        break;
    case MATMUL_TYPE::GPU_CUBLASLT:
        matmul_cublasLt(out, input, weight, bias, N, H, W, O, act_fn);
        break;
    default:
        matmul_cpu(out, input, weight, bias, N, H, W, O);
        break;
    }
}

// int main(int argc, char **argv){
//     srand(2101);
    
//     int N = 32;
//     int H = 1024;
//     int W = 768;
//     int O = 768 * 4;

//     int deviceIdx = 0;
//     CHECK_CUDA(cudaSetDevice(deviceIdx));
//     cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, deviceIdx);
//     printf("Device %d: %s\n", deviceIdx, deviceProp.name);


//     CHECK_CUBLAS(cublasCreate(&cublas_handle));
//     CHECK_CUBLAS(cublasLtCreate(&cublaslt_handle));

//     int enable_tf32 = deviceProp.major >= 8 ? 1:0;
//     printf("enable_tf32: %d\n", enable_tf32);
//     cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
//     cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
//     CHECK_CUBLAS(cublasSetMathMode(cublas_handle, cublas_math_mode));
//     // setup the (global) cuBLASLt workspace
//     CHECK_CUDA(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

//     float *out = (float*)malloc(N * H * O);
//     float *input = create_rand_float(N * H * W);
//     float *weight = create_rand_float(O * W);
//     float *bias = create_rand_float(O);

//     float *d_out;
//     float *d_inp;
//     float *d_weight;
//     float *d_bias;
//     CHECK_CUDA(cudaMalloc(&d_out, N*H*O*sizeof(float)));
//     CHECK_CUDA(cudaMalloc(&d_inp, N*H*W*sizeof(float)));
//     CHECK_CUDA(cudaMalloc(&d_weight, W*O*sizeof(float)));
//     CHECK_CUDA(cudaMalloc(&d_bias, O*sizeof(float)));

//     CHECK_CUDA(cudaMemcpy(d_inp, input, N*H*W*sizeof(float), cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_weight, weight, W*O*sizeof(float), cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_bias, bias, O*sizeof(float), cudaMemcpyHostToDevice));

//     int act_fn = 0; 
//     int n_kernel = 4;
    
//     int sqrt_block_sizes[] = {4,8,16,32};
//     matmul_forward(out, input, weight, bias, N, H, W, O);

//     for (int i = 0; i < sizeof(sqrt_block_sizes)/sizeof(int); i++ ){
//         int sqrt_block_size = sqrt_block_sizes[i] ;
//         for (int j = 1; j < n_kernel; j++){
//             printf("Block size: %d \n Kernel: %d \n", sqrt_block_size, j);
//             CHECK_CUDA(cudaMemset(d_out, 0, N * H * O * sizeof(float)));
//             matmul_forward(d_out, d_inp, d_weight, d_bias, N, H, W, O, sqrt_block_size, ACTIVATION_FN::NONE, j);
//             validate_result(d_out, out, "Matmul output", N*H*O, 1e-1f);
//         }
//     }
//     free(out);
//     free(input);
//     free(weight);
//     free(bias);
//     CHECK_CUDA(cudaFree(d_out));
//     CHECK_CUDA(cudaFree(d_inp));
//     CHECK_CUDA(cudaFree(d_weight));
//     CHECK_CUDA(cudaFree(d_bias));
//     CHECK_CUDA(cudaFree(cublaslt_workspace));
//     CHECK_CUBLAS(cublasDestroy(cublas_handle));
//     CHECK_CUBLAS(cublasLtDestroy(cublaslt_handle));
//     return 0;
// }