#!/usr/bin/env python3
"""Tier 1 semantic differential for structured ufunc kernels (G10).

Byte identity is the signal: each of UfuncCPU_add.cpp, UfuncCPUKernel_add.cpp
and UfuncCUDA_add.cu must be byte-identical between frozen torchgen and
SonicCross.
"""

import difflib
import sys
from pathlib import Path

FILES = [
    "UfuncCPU_add.cpp",
    "UfuncCPUKernel_add.cpp",
    "UfuncCUDA_add.cu",
]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-ufunc.py ORACLE_DIR SCHEME_DIR")
    oracle_dir = Path(sys.argv[1])
    scheme_dir = Path(sys.argv[2])

    failures = 0
    for name in FILES:
        oracle = oracle_dir / name
        scheme = scheme_dir / name
        if not oracle.is_file():
            print("missing oracle file: %s" % name)
            failures += 1
            continue
        if not scheme.is_file():
            print("missing scheme file: %s" % name)
            failures += 1
            continue
        ob = oracle.read_bytes()
        sb = scheme.read_bytes()
        if ob == sb:
            print("%s: OK (byte-identical)" % name)
            continue
        print("%s: MISMATCH" % name)
        diff = list(
            difflib.unified_diff(
                oracle.read_text(encoding="utf-8").split("\n"),
                scheme.read_text(encoding="utf-8").split("\n"),
                lineterm="",
                fromfile="oracle/" + name,
                tofile="scheme/" + name,
            )
        )
        print("\n".join(diff[:80]))
        failures += 1

    if failures:
        print("ufunc differential: %d file(s) differ" % failures)
        return 1
    print("ufunc differential: OK (all %d files byte-identical)" % len(FILES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
