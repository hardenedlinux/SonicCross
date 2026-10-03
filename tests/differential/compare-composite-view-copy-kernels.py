#!/usr/bin/env python3
"""Tier 1 semantic differential for CompositeViewCopyKernels.cpp (G11+G12).

Byte identity is the primary signal.  On mismatch, a per-kernel fallback
compares the generated kernel definitions individually so a single wrong
emitter is localised without obscuring the byte-level diff.
"""

import difflib
import re
import sys
from pathlib import Path

NAME = "CompositeViewCopyKernels.cpp"

# A kernel definition begins at a line matching a C++ function signature
# (return-type name(args)) at column 0 and ends at a lone "}".
_KERNEL_START = re.compile(
    r"^(::?(?:std::vector<at::Tensor>|at::Tensor(?:\s*&)?|void|bool|int64_t|"
    r"double|::std::tuple<.*>|[A-Za-z_][\w:<>,\s]*))\s+"
    r"([A-Za-z_]\w*)\s*\(.*\)\s*\{\s*$"
)


def split_kernels(text):
    """Return the list of top-level kernel definitions (signature .. })."""
    kernels = []
    buf = []
    depth = 0
    for line in text.split("\n"):
        if not buf and _KERNEL_START.match(line):
            buf.append(line)
            depth = 1
            continue
        if buf:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                kernels.append("\n".join(buf))
                buf = []
    return kernels


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-composite-view-copy-kernels.py ORACLE_DIR SCHEME_DIR"
        )
    oracle_dir = Path(sys.argv[1])
    scheme_dir = Path(sys.argv[2])

    oracle = oracle_dir / NAME
    scheme = scheme_dir / NAME
    for p, label in ((oracle, "oracle"), (scheme, "scheme")):
        if not p.is_file():
            print("missing %s file: %s" % (label, p))
            return 1

    ob = oracle.read_bytes()
    sb = scheme.read_bytes()
    if ob == sb:
        print("CompositeViewCopyKernels differential: OK (byte-identical)")
        return 0

    ok = split_kernels(oracle.read_text(encoding="utf-8"))
    sk = split_kernels(scheme.read_text(encoding="utf-8"))
    if ok == sk and ok:
        print(
            "CompositeViewCopyKernels differential: kernels OK "
            "(%d kernels, prologue differs)" % len(ok)
        )
        return 0

    print("CompositeViewCopyKernels differential: MISMATCH")
    print("  kernel count: oracle=%d scheme=%d" % (len(ok), len(sk)))
    n = min(len(ok), len(sk))
    for i in range(n):
        if ok[i] != sk[i]:
            print("  first diff at kernel %d:" % i)
            diff = list(
                difflib.unified_diff(
                    ok[i].split("\n"), sk[i].split("\n"), lineterm=""
                )
            )
            print("\n".join(diff[:80]))
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
