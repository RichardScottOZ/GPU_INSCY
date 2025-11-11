#!/usr/bin/env python3
"""
Make GPU_INSCY Windows/MSVC friendly:

- Guard CUDA-only constructs in src/utils/util.cuh ( __global__, cudaStream_t, Thrust ).
- Replace VLA in src/structures/SCY_tree.cpp with std::vector<bool>.

Backups: creates util.cuh.bak and SCY_tree.cpp.bak next to the originals.
"""

from __future__ import annotations
import re
from pathlib import Path
import difflib
import sys

ROOT = Path(".")
UTIL_CUH = ROOT / "src" / "utils" / "util.cuh"
SCY_TREE_CPP = ROOT / "src" / "structures" / "SCY_tree.cpp"

def show_and_write(path: Path, old: str, new: str) -> bool:
    if new == old:
        print(f"[INFO] {path}: no changes.")
        return False
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=str(path) + " (old)",
        tofile=str(path) + " (new)",
    )
    sys.stdout.writelines(diff)
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(old, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    print(f"[OK] Updated {path}  (backup: {bak})")
    return True

# -------------------- util.cuh transforms --------------------

def ensure_vector_include(text: str) -> str:
    if re.search(r'^\s*#\s*include\s*<\s*vector\s*>\s*$', text, flags=re.MULTILINE):
        return text
    # Insert after first include, else top
    m = re.search(r'^\s*#\s*include\s*<[^>]+>\s*$', text, flags=re.MULTILINE)
    if m:
        return text[:m.end()] + "\n#include <vector>" + text[m.end():]
    return "#include <vector>\n" + text

def guard_thrust(text: str) -> str:
    # Wrap the thrust include
    text = re.sub(
        r'^\s*#\s*include\s*<\s*thrust/device_vector\.h\s*>\s*$',
        '#ifdef __CUDACC__\n#include <thrust/device_vector.h>\n#endif',
        text, flags=re.MULTILINE
    )
    # Wrap the thrust overload declaration
    text = re.sub(
        r'^\s*void\s+print_array\s*\(\s*thrust::device_vector\s*<\s*int\s*>\s*\w+\s*,\s*int\s+\w+\s*\)\s*;\s*$',
        '#ifdef __CUDACC__\n\\g<0>\n#endif',
        text, flags=re.MULTILINE
    )
    return text

def guard_cuda_stream(text: str) -> str:
    # Any single-line prototype that mentions cudaStream_t
    def wrap(m):
        return "#ifdef __CUDACC__\n" + m.group(0) + "\n#endif"
    return re.sub(
        r'^\s*[^;\n]*\bcudaStream_t\b[^;\n]*;\s*$',
        wrap,
        text, flags=re.MULTILINE
    )

def guard_global_prototypes(text: str) -> str:
    """
    Wrap single-line __global__ function declarations (up to a semicolon).
    Examples:
      __global__ void foo(int*, int);
      __global__ void bar(float* x, int n);
    """
    def wrap(m):
        return "#ifdef __CUDACC__\n" + m.group(0) + "\n#endif"
    return re.sub(
        r'^\s*__global__\s+[^;{]+\)\s*;\s*$',
        wrap,
        text, flags=re.MULTILINE
    )

def fix_util_cuh(path: Path) -> bool:
    if not path.exists():
        print(f"[WARN] {path} not found; skipping.")
        return False
    old = path.read_text(encoding="utf-8", errors="ignore")
    new = old
    new = ensure_vector_include(new)
    new = guard_thrust(new)
    new = guard_cuda_stream(new)
    new = guard_global_prototypes(new)
    return show_and_write(path, old, new)

# -------------------- SCY_tree.cpp transforms --------------------

def replace_vla_in_scy_tree(text: str) -> str:
    """
    Replace:   bool is_weak_dense[leaf->points.size()];
    With:      std::vector<bool> is_weak_dense(leaf->points.size());
    """
    # Ensure <vector> include present (good hygiene; file uses std::vector heavily)
    if not re.search(r'^\s*#\s*include\s*<\s*vector\s*>\s*$', text, flags=re.MULTILINE):
        # Insert after last include
        incs = list(re.finditer(r'^\s*#\s*include\s*[<"].*[>"]\s*$', text, flags=re.MULTILINE))
        if incs:
            pos = incs[-1].end()
            text = text[:pos] + "\n#include <vector>" + text[pos:]
        else:
            text = "#include <vector>\n" + text

    # Replace the VLA declaration (only the bool[] one used for density flags)
    text = re.sub(
        r'^\s*bool\s+is_weak_dense\s*\[\s*[^;\]]+\s*\]\s*;\s*$',
        r'std::vector<bool> is_weak_dense(\g<0>)',  # temporary trick — then fix
        text, flags=re.MULTILINE
    )
    # The previous replacement carried the whole bracket expression; fix to parens-form.
    text = re.sub(
        r'std::vector<bool>\s+is_weak_dense\(\s*bool\s+is_weak_dense\s*\[\s*([^\]]+)\s*\]\s*;\s*\)',
        r'std::vector<bool> is_weak_dense(\1);',
        text
    )
    # If the cute trick above didn’t match (different whitespace), do a direct pattern:
    text = re.sub(
        r'^\s*bool\s+is_weak_dense\s*\[\s*leaf->points\.size\(\)\s*\]\s*;\s*$',
        r'std::vector<bool> is_weak_dense(leaf->points.size());',
        text, flags=re.MULTILINE
    )
    return text

def fix_scy_tree_cpp(path: Path) -> bool:
    if not path.exists():
        print(f"[WARN] {path} not found; skipping.")
        return False
    old = path.read_text(encoding="utf-8", errors="ignore")
    new = replace_vla_in_scy_tree(old)
    return show_and_write(path, old, new)

# -------------------- main --------------------

if __name__ == "__main__":
    any_change = False
    any_change |= fix_util_cuh(UTIL_CUH)
    any_change |= fix_scy_tree_cpp(SCY_TREE_CPP)

    if not any_change:
        print("\n[INFO] No modifications were necessary.")
    else:
        print("\n[NOTE] Done. Now clean Torch’s build cache and rebuild:")
        print("  set TORCH_CUDA_ARCH_LIST=7.5")
        print(r"  rmdir /S /Q %USERPROFILE%\.cache\torch_extensions  2>nul")
        print("  python run_example.py")