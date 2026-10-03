#!/usr/bin/env python3
"""Tier 1 semantic differential for NativeMetaFunctions.h (G9).

Byte identity is the primary signal; if the whole-file bytes differ, a
per-declaration fallback compares the structured-meta struct bodies.
"""

import difflib
import sys
from pathlib import Path

NAME = "NativeMetaFunctions.h"


def split_declarations(text):
    """Return the struct bodies as a list of (struct) strings.

    Everything between the `namespace meta {` / `} // namespace meta` markers is
    the declaration block; each `struct TORCH_API structured_...` .. `};` is one
    declaration.
    """
    decls = []
    lines = text.split("\n")
    inside = False
    buf = []
    for line in lines:
        s = line.strip()
        if not inside:
            if s == "namespace meta {":
                inside = True
            continue
        if s == "} // namespace meta":
            inside = False
            continue
        buf.append(line)
        if s == "};" and buf:
            decls.append("\n".join(buf))
            buf = []
    return decls


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-native-meta-functions.py ORACLE_DIR SCHEME_DIR")
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
        print("NativeMetaFunctions differential: OK (byte-identical)")
        return 0

    od = split_declarations(oracle.read_text(encoding="utf-8"))
    sd = split_declarations(scheme.read_text(encoding="utf-8"))
    if od == sd:
        print(
            "NativeMetaFunctions differential: declaration body OK "
            "(%d structs, prologue differs)" % len(od)
        )
        return 0

    print("NativeMetaFunctions differential: MISMATCH")
    if len(od) != len(sd):
        print(
            "  declaration count differs: oracle=%d scheme=%d" % (len(od), len(sd))
        )
    n = min(len(od), len(sd))
    for i in range(n):
        if od[i] != sd[i]:
            print("  first diff at struct %d:" % i)
            diff = list(
                difflib.unified_diff(
                    od[i].split("\n"), sd[i].split("\n"), lineterm=""
                )
            )
            print("\n".join(diff[:60]))
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
