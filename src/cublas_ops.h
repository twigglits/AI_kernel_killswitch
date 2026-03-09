#ifndef CUBLAS_OPS_H
#define CUBLAS_OPS_H

#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// cuBLAS GEMM wrappers (row-major convention)
// ---------------------------------------------------------------------------

// C = alpha * A * op(B) + beta * C
// A: [M, K], C: [M, N]
// If transpose_B=false: B is [K, N], computes C = A * B
// If transpose_B=true:  B is [N, K], computes C = A * B^T
void cublas_matmul(cublasHandle_t handle,
                   float *C, const float *A, const float *B,
                   int M, int N, int K,
                   float alpha, float beta,
                   bool transpose_B);

// Batched strided matmul for attention
// C[b] = alpha * A[b] * op(B[b]) + beta * C[b]
// If transpose_B=false: A[b]:[M,K], B[b]:[K,N], C[b]:[M,N]
// If transpose_B=true:  A[b]:[M,K], B[b]:[N,K], C[b]:[M,N]
void cublas_batched_matmul(cublasHandle_t handle,
                           float *C, const float *A, const float *B,
                           int batch, int M, int N, int K,
                           int strideA, int strideB, int strideC,
                           float alpha, float beta,
                           bool transpose_B);

#endif // CUBLAS_OPS_H
