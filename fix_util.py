#!/usr/bin/env python3
"""
Modernize src/utils/util.cuh for C++17 by removing inheritance from std::binary_function
and keeping only a declaration for vec_cmp::operator(), since the body exists in util.cu.

- Backs up util.cuh to util.cuh.bak
- Adds <vector> include if missing
- Rewrites 'struct vec_cmp : public std::binary_function<...> { ... };' to
  'struct vec_cmp { bool operator()(const std::vector<int>&, const std::vector<int>&) const; };'
- Does NOT inline any comparator body (util.cu has it already)

Usage:
  python fix_util_cuh_vec_cmp.py
"""

from __future__ import annotations
import difflib
from pathlib import Path
import re
import sys

UTIL_CUH = Path("src/utils/util.cuh")

# Detect a vec_cmp with the obsolete base + a declaration-only operator()
PATTERNS = [
    re.compile(
        r"""
        struct\s+vec_cmp\s*
        :\s*public\s*std::binary_function\s*<\s*
           (?:std::)?vector\s*<\s*int\s*>\s*,\s*
           (?:std::)?vector\s*<\s*int\s*>\s*,\s*
           bool\s*>\s*
        \{\s*
        bool\s+operator\(\)\s*\(\s*
            (?:const\s+)?(?:std::)?vector\s*<\s*int\s*>\s*&\s*a\s*,\s*
            (?:const\s+)?(?:std::)?vector\s*<\s*int\s*>\s*&\s*b\s*
        \)\s*const\s*;\s*
        \}\s*;
        """,
        re.VERBOSE | re.DOTALL,
    ),
    # Fallback: any vec_cmp with std::binary_function base
    re.compile(
        r"struct\s+vec_cmp\s*:\s*public\s*std::binary_function\s*<[^>]+>\s*\{[^}]*\};",
        re.DOTALL,
    ),
]

INCLUDE_VECTOR_RE = re.compile(r'^\s*#\s*include\s*<vector>\s*$', re.MULTILINE)

DECL_REPLACEMENT = (
    "struct vec_cmp {\n"
    "  bool operator()(const std::vector<int>& a, const std::vector<int>& b) const;\n"
    "};"
)

def ensure_vector_include(text: str) -> tuple[str, bool]:
    if INCLUDE_VECTOR_RE.search(text):
        return text, False
    # Insert after the first include, or after the guard, or at top
    m_inc = re.search(r'^\s*#\s*include\s*<[^>]+>\s*$', text, re.MULTILINE)
    if m_inc:
        insert_at = m_inc.end()
        return text[:insert_at] + "\n#include <vector>" + text[insert_at:], True
    m_guard = re.search(r'#ifndef\s+\w+\s+#define\s+\w+\s*', text, re.DOTALL)
    if m_guard:
        insert_at = m_guard.end()
        return text[:insert_at] + "\n#include <vector>\n" + text[insert_at:], True
    return "#include <vector>\n" + text, True

def rewrite_vec_cmp(text: str) -> tuple[str, bool]:
    for pat in PATTERNS:
        if pat.search(text):
            return pat.sub(DECL_REPLACEMENT, text), True
    # If already modern but parameters are unqualified 'vector<int>'
    unqual = re.sub(
        r"bool\s+operator\(\)\s*\(\s*const\s+vector\s*<\s*int\s*>\s*&\s*a\s*,\s*const\s+vector\s*<\s*int\s*>\s*&\s*b\s*\)\s*const\s*;",
        "bool operator()(const std::vector<int>& a, const std::vector<int>& b) const;",
        text,
    )
    return unqual, (unqual != text)

def main() -> None:
    if not UTIL_CUH.exists():
        print(f"[ERROR] {UTIL_CUH} not found. Run this from the repo root.", file=sys.stderr)
        sys.exit(1)

    original = UTIL_CUH.read_text(encoding="utf-8", errors="ignore")
    updated = original

    # 1) Ensure <vector> include
    updated, added_vector = ensure_vector_include(updated)

    # 2) Rewrite vec_cmp declaration
    updated, changed_decl = rewrite_vec_cmp(updated)

    if updated == original:
        print("[INFO] No changes needed; util.cuh already C++17-safe.")
        return

    # Show diff
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(UTIL_CUH) + " (old)",
        tofile=str(UTIL_CUH) + " (new)",
    )
    sys.stdout.writelines(diff)

    # Backup and write
    backup = UTIL_CUH.with_suffix(UTIL_CUH.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    UTIL_CUH.write_text(updated, encoding="utf-8")

    actions = []
    if added_vector: actions.append("added <vector> include")
    if changed_decl: actions.append("rewrote vec_cmp declaration")
    print("\n[OK] Wrote changes to", UTIL_CUH)
    print("[OK] Backup at", backup)
    if actions:
        print("[CHANGES]: " + ", ".join(actions))

if __name__ == "__main__":
    main()