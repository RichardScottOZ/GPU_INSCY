#!/usr/bin/env python3
"""
Make GPU_INSCY Windows/C++17 friendly:

1) src/utils/util.cuh:
   - Wrap thrust includes + device_vector overload with #ifdef __CUDACC__.

2) src/utils/util.cu:
   - Replace #include "nvToolsExt.h" with a __has_include guarded macro wrapper:
       NVTX_PUSH_RANGE / NVTX_POP_RANGE
     and rewrite call sites.

3) src/algorithms/GPU_Clustering.cu:
   - Replace variable-length arrays (VLAs) with vectors + pointer alias so
     existing code still compiles unchanged.

Backups: *.bak are created next to each modified file.
"""

from __future__ import annotations
import re
from pathlib import Path
import difflib
import sys

ROOT = Path(".")
UTIL_CUH = ROOT / "src" / "utils" / "util.cuh"
UTIL_CU  = ROOT / "src" / "utils" / "util.cu"
GPU_CLUSTERING_CU = ROOT / "src" / "algorithms" / "GPU_Clustering.cu"

def patch_file(path: Path, transform) -> bool:
    if not path.exists():
        print(f"[WARN] {path} not found; skipping")
        return False
    old = path.read_text(encoding="utf-8", errors="ignore")
    new = transform(old)
    if new == old:
        print(f"[INFO] {path}: no changes")
        return False
    # show diff
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=str(path) + " (old)",
        tofile=str(path) + " (new)",
    )
    sys.stdout.writelines(diff)
    # backup + write
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(old, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    print(f"[OK] updated {path} (backup {bak})")
    return True

# -------- util.cuh: guard Thrust for non-CUDA TUs --------

def transform_util_cuh(text: str) -> str:
    out = text

    # Wrap the thrust include with __CUDACC__ guards
    out = re.sub(
        r'^\s*#\s*include\s*<\s*thrust/device_vector\.h\s*>\s*$',
        '#ifdef __CUDACC__\n#include <thrust/device_vector.h>\n#endif',
        out,
        flags=re.MULTILINE,
    )

    # Guard the thrust-based overload declaration:
    # void print_array(thrust::device_vector<int> x, int n);
    out = re.sub(
        r'^\s*void\s+print_array\s*\(\s*thrust::device_vector<\s*int\s*>\s*\w+\s*,\s*int\s+\w+\s*\)\s*;\s*$',
        '#ifdef __CUDACC__\n\\g<0>\n#endif',
        out,
        flags=re.MULTILINE,
    )

    # Ensure <vector> include exists (for other parts of the header)
    if re.search(r'^\s*#\s*include\s*<\s*vector\s*>\s*$', out, flags=re.MULTILINE) is None:
        # Insert after first include or at top
        m = re.search(r'^\s*#\s*include\s*<[^>]+>\s*$', out, flags=re.MULTILINE)
        if m:
            pos = m.end()
            out = out[:pos] + "\n#include <vector>" + out[pos:]
        else:
            out = "#include <vector>\n" + out

    return out

# -------- util.cu: NVTX safe macros + callsite rewrite --------

NVTX_BLOCK = """\
#if defined(__has_include)
#  if __has_include("nvToolsExt.h")
#    include "nvToolsExt.h"
#    define NVTX_PUSH_RANGE(name) nvtxRangePushA(name)
#    define NVTX_POP_RANGE()      nvtxRangePop()
#  else
#    define NVTX_PUSH_RANGE(name) do {} while(0)
#    define NVTX_POP_RANGE()      do {} while(0)
#  endif
#else
#  define NVTX_PUSH_RANGE(name) do {} while(0)
#  define NVTX_POP_RANGE()      do {} while(0)
#endif
"""

def transform_util_cu(text: str) -> str:
    out = text

    # Replace bare include with guarded block (idempotent)
    out = re.sub(
        r'^\s*#\s*include\s*"nvToolsExt\.h"\s*$',
        NVTX_BLOCK.strip(),
        out,
        flags=re.MULTILINE,
    )

    # If include not present at all, add the block after the last system include
    if "nvToolsExt.h" not in out and "NVTX_PUSH_RANGE" not in out:
        # insert after the last #include <...> / "..."
        m = list(re.finditer(r'^\s*#\s*include\s*[<"].*[>"]\s*$', out, flags=re.MULTILINE))
        if m:
            pos = m[-1].end()
            out = out[:pos] + "\n" + NVTX_BLOCK + "\n" + out[pos:]
        else:
            out = NVTX_BLOCK + "\n" + out

    # Rewrite call sites to use macros
    out = out.replace("nvtxRangePushA(", "NVTX_PUSH_RANGE(")
    out = out.replace("nvtxRangePop(",  "NVTX_POP_RANGE(")
    # Also handle possible "nvtxRangePop()" without space
    out = out.replace("nvtxRangePop()", "NVTX_POP_RANGE()")

    return out

# -------- GPU_Clustering.cu: replace VLAs with vectors + pointer alias --------

REPLACEMENTS = [
    # group 1
    (r'^\s*int\s*\*\s*h_restricted_dims_list\s*\[\s*size\s*\]\s*;\s*$', 
     'std::vector<int*> h_restricted_dims_list_vec(size);\nint** h_restricted_dims_list = h_restricted_dims_list_vec.data();'),
    (r'^\s*int\s+h_number_of_restricted_dims\s*\[\s*size\s*\]\s*;\s*$',
     'std::vector<int> h_number_of_restricted_dims_vec(size);\nint* h_number_of_restricted_dims = h_number_of_restricted_dims_vec.data();'),
    (r'^\s*int\s*\*\s*h_points_list\s*\[\s*size\s*\]\s*;\s*$',
     'std::vector<int*> h_points_list_vec(size);\nint** h_points_list = h_points_list_vec.data();'),
    (r'^\s*int\s+h_number_of_points\s*\[\s*size\s*\]\s*;\s*$',
     'std::vector<int> h_number_of_points_vec(size);\nint* h_number_of_points = h_number_of_points_vec.data();'),
    (r'^\s*int\s*\*\s*h_new_neighborhood_sizes_list\s*\[\s*size\s*\]\s*;\s*$',
     'std::vector<int*> h_new_neighborhood_sizes_list_vec(size);\nint** h_new_neighborhood_sizes_list = h_new_neighborhood_sizes_list_vec.data();'),
    # later block with float
    (r'^\s*float\s+h_v\s*\[\s*size\s*\]\s*;\s*$',
     'std::vector<float> h_v_vec(size);\nfloat* h_v = h_v_vec.data();'),
]

def ensure_include_vector(text: str) -> str:
    if re.search(r'^\s*#\s*include\s*<\s*vector\s*>\s*$', text, flags=re.MULTILINE):
        return text
    # Insert after last include
    m = list(re.finditer(r'^\s*#\s*include\s*[<"].*[>"]\s*$', text, flags=re.MULTILINE))
    if m:
        pos = m[-1].end()
        return text[:pos] + "\n#include <vector>" + text[pos:]
    return "#include <vector>\n" + text

def transform_gpu_clustering_cu(text: str) -> str:
    out = ensure_include_vector(text)
    for pat, repl in REPLACEMENTS:
        out = re.sub(pat, repl, out, flags=re.MULTILINE)
    return out

def main():
    changed = False
    changed |= patch_file(UTIL_CUH, transform_util_cuh)
    changed |= patch_file(UTIL_CU,  transform_util_cu)
    changed |= patch_file(GPU_CLUSTERING_CU, transform_gpu_clustering_cu)
    if not changed:
        print("\n[INFO] No modifications were necessary.")
    else:
        print("\n[NOTE] Rebuild your extension now (same command you used before).")

if __name__ == "__main__":
    main()