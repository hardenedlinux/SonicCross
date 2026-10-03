#!/usr/bin/env python3
"""Byte differential for the G16-G20 core-source / operators artifacts.

Every artifact emitted by both the frozen torchgen oracle and SonicCross must
be byte-identical.  A mismatch reports the first differing offset per file.
"""

import sys

FILES = [
    "Operators.h",
    "OperatorsEverything.cpp",
    "Operators_0.cpp",
    "Operators_1.cpp",
    "Operators_2.cpp",
    "Operators_3.cpp",
    "Operators_4.cpp",
    "TensorBody.h",
    "aten_interned_strings.h",
    "enum_tag.h",
    "TensorMethods.cpp",
    "ATenOpList.cpp",
    "RegisterBackendSelect.cpp",
]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-core-sources.py ORACLE_DIR SCHEME_DIR")
    oracle_dir, scheme_dir = sys.argv[1], sys.argv[2]

    failures = []
    identical = 0
    for name in FILES:
        oracle = open(oracle_dir + "/" + name, "rb").read()
        scheme = open(scheme_dir + "/" + name, "rb").read()
        if oracle == scheme:
            identical += 1
            continue
        failures.append(name)
        print("%s: MISMATCH" % name)
        n = min(len(oracle), len(scheme))
        first = None
        for i in range(n):
            if oracle[i] != scheme[i]:
                first = i
                break
        if first is not None:
            print("  first byte diff at offset %d" % first)
            lo = max(0, first - 40)
            hi = min(n, first + 40)
            print("  oracle: %r" % oracle[lo:hi])
            print("  scheme: %r" % scheme[lo:hi])
        if len(oracle) != len(scheme):
            print(
                "  length differs: oracle=%d scheme=%d"
                % (len(oracle), len(scheme))
            )

    if not failures:
        print("core-sources differential: OK (%d files byte-identical)" % identical)
        return 0
    print(
        "core-sources differential: MISMATCH (%d/%d identical)"
        % (identical, len(FILES))
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
