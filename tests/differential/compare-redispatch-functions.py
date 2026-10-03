#!/usr/bin/env python3
"""Tier 1 semantic differential for RedispatchFunctions.h (G6).

Extracts the ordered sequence of redispatch ``inline`` declarations (each line
that, ignoring the four-space ``namespace redispatch`` indent, begins with
``inline ``) and compares them between the oracle and SonicCross.  Each such
line carries the return type, the unambiguous operator name, the leading
``c10::DispatchKeySet dispatchKeySet`` argument and the remaining argument
types, i.e. the whole redispatch signature.  Byte identity is reported
separately as a secondary signal.
"""

import sys


def extract(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    records = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("inline "):
            # The next non-blank line is the dispatcher redispatch call.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            call = lines[j].lstrip() if j < len(lines) else ""
            records.append((stripped, call))
            i = j + 1
        else:
            i += 1
    return records


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-redispatch-functions.py ORACLE.h SCHEME.h")
    oracle = extract(sys.argv[1])
    scheme = extract(sys.argv[2])
    byte_identical = (
        open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
    )

    if oracle == scheme:
        print(
            "RedispatchFunctions.h semantic differential: OK "
            "(%d declarations%s)"
            % (len(oracle), ", byte-identical" if byte_identical else "")
        )
        return 0

    print("RedispatchFunctions.h semantic differential: MISMATCH")
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
