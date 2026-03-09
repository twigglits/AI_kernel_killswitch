#include "cublas_ops.h"
#include "utils.h"

// ---------------------------------------------------------------------------
// Row-major GEMM via cuBLAS (column-major).
//
// Key insight: if A is [M,K] row-major, cuBLAS sees it as [K,M] col-major.
// We exploit C^T = op(B)^T * A^T to get row-major results.
//
// For C = A * B (no transpose):
//   A:[M,K], B:[K,N], C:[M,N]
//   C^T = B^T * A^T
//   cuBLAS: Ccm = Bcm * Acm  (both OP_N)
//   where Acm=[K,M] ld=K, Bcm=[N,K] ld=N, Ccm=[N,M] ld=N
//
// For C = A * B^T:
//   A:[M,K], B:[N,K], C:[M,N]
//   C^T = B * A^T
//   cuBLAS: Ccm = op(Bcm) * Acm where Bcm=[K,N] ld=K, need transpose
// ---------------------------------------------------------------------------
void cublas_matmul(cublasHandle_t handle,
                   float *C, const float *A, const float *B,
                   int M, int N, int K,
                   float alpha, float beta,
                   bool transpose_B) {
    if (!transpose_B) {
        // C = A * B, where A:[M,K], B:[K,N], C:[M,N]
        // C^T = B^T * A^T → Ccm = Bcm * Acm
        // Bcm: B row-major [K,N] → col-major [N,K], ld=N
        // Acm: A row-major [M,K] → col-major [K,M], ld=K
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 B, N,    // Bcm: [N,K] col-major, ld=N
                                 A, K,    // Acm: [K,M] col-major, ld=K
                                 &beta,
                                 C, N));  // Ccm: [N,M] col-major, ld=N
    } else {
        // C = A * B^T, where A:[M,K], B:[N,K], C:[M,N]
        // C^T = B * A^T → Ccm = Bcm_T * Acm
        // Bcm: B row-major [N,K] → col-major [K,N], ld=K. Need transpose → [N,K]
        // Acm: A row-major [M,K] → col-major [K,M], ld=K
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 B, K,    // Bcm: [K,N] col-major, ld=K, transposed to [N,K]
                                 A, K,    // Acm: [K,M] col-major, ld=K
                                 &beta,
                                 C, N));  // Ccm: [N,M] col-major, ld=N
    }
}

// ---------------------------------------------------------------------------
// Batched strided GEMM
// ---------------------------------------------------------------------------
void cublas_batched_matmul(cublasHandle_t handle,
                           float *C, const float *A, const float *B,
                           int batch, int M, int N, int K,
                           int strideA, int strideB, int strideC,
                           float alpha, float beta,
                           bool transpose_B) {
    if (!transpose_B) {
        // C[b] = A[b] * B[b], A:[M,K], B:[K,N], C:[M,N]
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, N, strideB,
            A, K, strideA,
            &beta,
            C, N, strideC,
            batch));
    } else {
        // C[b] = A[b] * B[b]^T, A:[M,K], B:[N,K], C:[M,N]
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, K, strideB,
            A, K, strideA,
            &beta,
            C, N, strideC,
            batch));
    }
}
