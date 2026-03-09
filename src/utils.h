#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t stat = (call);                                          \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",            \
                    __FILE__, __LINE__, (int)stat);                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Simple wall-clock timer
// ---------------------------------------------------------------------------
typedef struct {
    struct timeval start;
    struct timeval end;
} Timer;

static inline void timer_start(Timer *t) {
    gettimeofday(&t->start, NULL);
}

static inline double timer_stop(Timer *t) {
    gettimeofday(&t->end, NULL);
    double elapsed = (t->end.tv_sec - t->start.tv_sec) * 1000.0 +
                     (t->end.tv_usec - t->start.tv_usec) / 1000.0;
    return elapsed; // milliseconds
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#endif // UTILS_H
