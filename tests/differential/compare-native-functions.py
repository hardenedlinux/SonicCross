#!/usr/bin/env python3
"""Tier 1 semantic differential for NativeFunctions.h (G7).

Extracts the ordered sequence of kernel declarations (every line between the
``namespace native {`` open and its ``} // namespace native`` close) and
compares it between the oracle and SonicCross.  Byte identity is reported
separately as the primary signal, since this emitter is deterministic.
"""

import sys


def extract(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    decls = []
    inside = False
    for line in lines:
        if line == "namespace native {":
            inside = True
            continue
        if line == "} // namespace native":
            inside = False
            break
        if inside:
            decls.append(line)
    return decls


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-native-functions.py ORACLE.h SCHEME.h")
    oracle = extract(sys.argv[1])
    scheme = extract(sys.argv[2])
    byte_identical = (
        open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
    )

    if byte_identical:
        print(
            "NativeFunctions.h semantic differential: OK "
            "(%d declarations, byte-identical)" % len(oracle)
        )
        return 0

    if oracle == scheme:
        print(
            "NativeFunctions.h semantic differential: OK "
            "(%d declarations, header prologue differs)" % len(oracle)
        )
        return 0

    print("NativeFunctions.h semantic differential: MISMATCH")
    n = min(len(oracle), len(scheme))
    for i in range(n):
        if oracle[i] != scheme[i]:
            print("first diff at declaration %d:" % i)
            print("  oracle: %r" % (oracle[i],))
            print("  scheme: %r" % (scheme[i],))
            break
    if len(oracle) != len(scheme):
        print(
            "declaration count differs: oracle=%d scheme=%d"
            % (len(oracle), len(scheme))
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
