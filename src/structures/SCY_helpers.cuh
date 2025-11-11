#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// Compute (k + 1)! as float: Γ(k + 2)
static __device__ __forceinline__ float gamma_int_plus2(int k) {
    int m = k + 1;
    float g = 1.0f;
    #pragma unroll
    for (int i = 2; i <= m; ++i) g *= (float)i;
    return g;
}

// π^{k/2} without powf: π^m * sqrt(π) if k is odd
static __device__ __forceinline__ float pi_pow_k_over_2(int k) {
    float r = 1.0f;
    int m = k >> 1;  // floor(k/2)
    #pragma unroll
    for (int i = 0; i < m; ++i) r *= CUDART_PI_F;
    if (k & 1) r *= sqrtf(CUDART_PI_F);
    return r;
}

// c(k) = π^{k/2} / Γ(k+2)
static __device__ __forceinline__ float c_prune_gpu(int k) {
    return pi_pow_k_over_2(k) / gamma_int_plus2(k);
}

// alpha(k) = 2 n eps^k c(k) / [ v^k (k + 2) ]
static __device__ __forceinline__ float alpha_prune_gpu(int k, float eps, int n, float v) {
    float epsk = 1.0f, vk = 1.0f;
    #pragma unroll
    for (int i = 0; i < k; ++i) { epsk *= eps; vk *= v; }
    float r = 2.0f * n * epsk * c_prune_gpu(k);
    return r / (vk * (k + 2.0f));
}

// expDen(k) = n c(k) eps^k / v^k
static __device__ __forceinline__ float expDen_prune_gpu(int k, float eps, int n, float v) {
    float epsk = 1.0f, vk = 1.0f;
    #pragma unroll
    for (int i = 0; i < k; ++i) { epsk *= eps; vk *= v; }
    return (n * c_prune_gpu(k) * epsk) / vk;
}

static __device__ __forceinline__ float omega_prune_gpu(int k) {
    return 2.0f / (k + 2.0f);
}

// Small device helpers used by kernels
static __device__ __forceinline__ int get_lvl_size_gpu(int *d_dim_start, int dim_i, int number_of_dims, int number_of_nodes) {
    return (dim_i == number_of_dims - 1 ? number_of_nodes : d_dim_start[dim_i + 1]) - d_dim_start[dim_i];
}

static __device__ __forceinline__ float dist_prune_gpu(int p_id, int q_id, float *X, int d, int *subspace, int subsapce_size) {
    float *p = &X[p_id * d];
    float *q = &X[q_id * d];
    float distance = 0.0f;
    for (int i = 0; i < subsapce_size; i++) {
        int d_i = subspace[i];
        float diff = p[d_i] - q[d_i];
        distance += diff * diff;
    }
    return sqrtf(distance);
}
