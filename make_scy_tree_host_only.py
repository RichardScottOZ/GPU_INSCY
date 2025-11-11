import re, sys, pathlib, shutil

HELPERS_CUH = r'''
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// Γ(k + 2) == (k + 1)! as float
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
    int m = k >> 1;
    #pragma unroll
    for (int i = 0; i < m; ++i) r *= CUDART_PI_F;
    if (k & 1) r *= sqrtf(CUDART_PI_F);
    return r;
}

static __device__ __forceinline__ float c_prune_gpu(int k) {
    return pi_pow_k_over_2(k) / gamma_int_plus2(k);
}

static __device__ __forceinline__ float alpha_prune_gpu(int k, float eps, int n, float v) {
    float epsk = 1.0f, vk = 1.0f;
    #pragma unroll
    for (int i = 0; i < k; ++i) { epsk *= eps; vk *= v; }
    float r = 2.0f * n * epsk * c_prune_gpu(k);
    return r / (vk * (k + 2.0f));
}

static __device__ __forceinline__ float expDen_prune_gpu(int k, float eps, int n, float v) {
    float epsk = 1.0f, vk = 1.0f;
    #pragma unroll
    for (int i = 0; i < k; ++i) { epsk *= eps; vk *= v; }
    return (n * c_prune_gpu(k) * epsk) / vk;
}

static __device__ __forceinline__ float omega_prune_gpu(int k) {
    return 2.0f / (k + 2.0f);
}

// Lightweight device helpers
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
'''.lstrip()

KERNELS_CUH_HEADER = r'''#pragma once
extern "C" {
'''
KERNELS_CUH_FOOTER = r'''
} // extern "C"
'''

KERNELS_CU_PREAMBLE = r'''
#include <cuda_runtime.h>
#include <math_constants.h>
#include "SCY_helpers.cuh"

#ifndef BLOCK_WIDTH
#define BLOCK_WIDTH 64
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
'''

def find_kernels(src: str):
    pat = re.compile(r'__global__\s*(?:\n\s*)*void\s+([A-Za-z_]\w*)\s*\(', re.M)
    out = []
    for m in pat.finditer(src):
        nm = m.group(1)
        sig_start = m.start()
        open_brace = src.find('{', m.end())
        if open_brace < 0: continue
        depth, i = 0, open_brace
        while i < len(src):
            if src[i] == '{': depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    out.append((nm, sig_start, i+1))
                    break
            i += 1
    return out

def extract_proto(block: str):
    head_end = block.find('{')
    head = ' '.join(block[:head_end].split())
    close = head.rfind(')')
    return head[:close+1] + ';'

def remove_device_helpers(src: str):
    names = [
        r'gamma_int_plus2', r'pi_pow_k_over_2',
        r'c_prune_gpu', r'alpha_prune_gpu', r'expDen_prune_gpu', r'omega_prune_gpu',
        r'get_lvl_size_gpu', r'dist_prune_gpu'
    ]
    for nm in names:
        rx = re.compile(rf'__device__[\s\S]{{0,160}}{nm}\s*\([^)]*\)\s*\{{', re.M)
        while True:
            m = rx.search(src)
            if not m: break
            # delete balanced braces block
            i, depth = m.end()-1, 0
            while i < len(src):
                if src[i] == '{': depth += 1
                elif src[i] == '}':
                    depth -= 1
                    if depth == 0:
                        src = src[:m.start()] + src[i+1:]
                        break
                i += 1
    return src

def strip_cuda_device_includes(src: str):
    # Host TU should not pull in device headers; keep runtime API if needed
    src = re.sub(r'#include\s*<math_constants\.h>\s*', '', src)
    # Keep cuda_runtime_api.h if you call cudaMemcpy/cudaMalloc; else leave as-is.
    # If file includes <cuda_runtime.h>, change to runtime_api:
    src = re.sub(r'#include\s*<cuda_runtime\.h>', '#include <cuda_runtime_api.h>', src)
    return src

def ensure_host_includes(src: str):
    lines = src.splitlines()
    ins = 0
    for i, ln in enumerate(lines[:120]):
        if ln.strip().startswith('#include'):
            ins = i+1
    add = []
    if 'GPU_SCY_tree_kernels.cuh' not in src:
        add.append('#include "GPU_SCY_tree_kernels.cuh"')
    if add:
        lines[ins:ins] = add
    return "\n".join(lines) + ("\n" if not src.endswith("\n") else "")

def force_floats_in_kernels_text(txt: str):
    # Change max(...) to fmaxf(...) where applicable (left in kernels)
    txt = re.sub(r'>=\s*max\s*\(\s*F\s*\*\s*a\s*,\s*num_obj\s*\*\s*w\s*\)',
                 r'>= fmaxf(F * a, (float)num_obj * w)', txt)
    txt = re.sub(r'>=\s*max\s*\(\s*F\s*\*\s*a\s*,\s*\(float\)\s*num_obj\s*\)',
                 r'>= fmaxf(F * a, (float)num_obj)', txt)
    return txt

def main(path):
    p = pathlib.Path(path)
    root = p.parent
    src0 = p.read_text(encoding='utf-8')

    # 1) find and extract kernels
    kernels = find_kernels(src0)
    if not kernels:
        print("No __global__ kernels found; aborting.")
        return

    # carve kernels out
    host = src0
    parts = []
    for nm, a, b in sorted(kernels, key=lambda x: x[1], reverse=True):
        parts.append((nm, host[a:b]))
        host = host[:a] + host[b:]
    parts.reverse()

    # 2) generate kernels.cuh / kernels.cu
    protos = [extract_proto(blk) for _, blk in parts]
    bodies = [force_floats_in_kernels_text(blk.strip()) for _, blk in parts]
    kernels_cuh = KERNELS_CUH_HEADER + "\n".join(protos) + KERNELS_CUH_FOOTER + "\n"
    kernels_cu  = KERNELS_CU_PREAMBLE + "\n\n".join(bodies) + "\n"

    # 3) write SCY_helpers.cuh
    (root / "SCY_helpers.cuh").write_text(HELPERS_CUH, encoding='utf-8')

    # 4) clean host TU from device helpers and device includes; include prototypes
    host = remove_device_helpers(host)
    host = strip_cuda_device_includes(host)
    host = ensure_host_includes(host)

    # 5) rename host TU to .cpp (host-only)
    host_cpp = root / "GPU_SCY_tree_host.cpp"
    bak = p.with_suffix(p.suffix + ".bak")
    bak.write_text(src0, encoding='utf-8')
    host_cpp.write_text(host, encoding='utf-8')

    # 6) write kernels files
    (root / "GPU_SCY_tree_kernels.cuh").write_text(kernels_cuh, encoding='utf-8')
    (root / "GPU_SCY_tree_kernels.cu").write_text(kernels_cu, encoding='utf-8')

    print(f"Backed up original to: {bak}")
    print(f"Wrote host-only file : {host_cpp}")
    print(f"Wrote helpers header : {root / 'SCY_helpers.cuh'}")
    print(f"Wrote kernels header : {root / 'GPU_SCY_tree_kernels.cuh'}")
    print(f"Wrote kernels TU     : {root / 'GPU_SCY_tree_kernels.cu'}")
    print("\nNext: update your build to use GPU_SCY_tree_host.cpp instead of GPU_SCY_tree.cu.")
    print("Add GPU_SCY_tree_kernels.cu to the sources if not already.")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_scy_tree_host_only.py X:\\src\\structures\\GPU_SCY_tree.cu")
        sys.exit(1)
    main(sys.argv[1])