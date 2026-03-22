#include "common.h"
// 
void matmul_cpu(float *out,
                const float *inp, const float *weight, const float *bias,
                int B, int T, int C, int OC){
    
    for (int b = 0 ; b < B; b++){
        for (int t = 0 ; t < T; t++){
            float *out_bt = out + b*T*OC + t*OC;
            const float *inp_bt = inp + b*T*C + t*C;
            for (int oc = 0 ; oc < OC; oc++) {
                float val = (bias != NULL) ? bias[oc] : 0.0f;
                const float *wrow = weight + oc*C;
                for (int c ; c < C ; c++){
                    val += wrow[c] * inp_bt[c];
                }
                out_bt[oc] = val;
            }
        }
    }
}

// 
__global__ void matmul_forward_kernel_naive(float *out,
                const float *inp, const float *weight, const float *bias,
                int BT, int C, int OC){
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + bt * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// 
__global__ void matmul_forward_kernal_cublas(float *out,
                const float *inp, const float *weight, const float *bias,
                int B, int T, int C, int OC){

}

__global__ void add_bias(float *out, float *bias, int C, int T, int OC){

}

// 
__global__ void matmul_forward_kernel_cublasLt(float *out,
                const float *inp, const float *weight, const float *bias,
                int B, int T, int C, int OC){

}

