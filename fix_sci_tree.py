#!/usr/bin/env python3
"""
Rename the CUDA kernel 'memset' in src/structures/GPU_SCY_tree.cu to avoid
conflict with the CRT's memset and NVCC/MSVC front-end crashes (cudafe++ 0xC0000409).

- __global__ void memset(...)   -> __global__ void set_int_at(...)
- memset<<<...>>>(...)          -> set_int_at<<<...>>>(...)

Creates a .bak next to the original file.
"""

from __future__ import annotations
import re
from pathlib import Path
import difflib
import sys

PATH = Path("src/structures/GPU_SCY_tree.cu")

def main():
    if not PATH.exists():
        print(f"[ERROR] {PATH} not found. Run this from the repo root.", file=sys.stderr)
        sys.exit(1)

    orig = PATH.read_text(encoding="utf-8", errors="ignore")
    new = orig

    # 1) Rename the kernel declaration/definition exactly:
    #    __global__
    #    void memset(int *a, int i, int val) { ... }
    new = re.sub(
        r'(__global__\s*\n\s*void\s+)memset(\s*\(\s*int\s*\*\s*a\s*,\s*int\s*i\s*,\s*int\s*val\s*\)\s*\{)',
        r'\1set_int_at\2',
        new
    )

    # 2) Rename ONLY triple-chevron launches of that kernel.
    #    Do not touch cudaMemset calls.
    new = re.sub(
        r'(?<!cuda)memset\s*<<<',
        'set_int_at<<<',
        new
    )

    if new == orig:
        print("[INFO] No changes made; patterns not found (already patched?).")
        return

    # Show diff for safety
    diff = difflib.unified_diff(
        orig.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=str(PATH) + " (old)",
        tofile=str(PATH) + " (new)",
    )
    sys.stdout.writelines(diff)

    # Backup + write
    bak = PATH.with_suffix(PATH.suffix + ".bak")
    bak.write_text(orig, encoding="utf-8")
    PATH.write_text(new, encoding="utf-8")
    print(f"\n[OK] Updated {PATH}")
    print(f"[OK] Backup at {bak}")

if __name__ == "__main__":
    main()