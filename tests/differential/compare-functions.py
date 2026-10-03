#!/usr/bin/env python3
"""Tier 1 semantic differential for Functions.h / Functions.cpp (G5).

Functions.cpp is a verbatim template write, so it is compared byte-for-byte.
Functions.h carries the ComputeFunction declarations: the ordered sequence of
``inline`` function declarations and the ``namespace symint`` template blocks.
The extractor collects those generated declaration lines, ignoring the fixed
header/namespace framing and the hand-written C++-only overloads that are part
of the template itself.  Byte identity is reported separately as a secondary
signal.
"""

import sys


def extract_functions_h(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("inline "):
            records.append(("decl", line))
        elif line.startswith(
            "  template <typename T, typename = std::enable_if_t"
        ):
            records.append(("template", line))
            if i + 1 < len(lines):
                records.append(("symint-decl", lines[i + 1]))
                i += 1
        i += 1
    return records


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-functions.py ORACLE_DIR SCHEME_DIR")
    oracle_dir, scheme_dir = sys.argv[1], sys.argv[2]

    oracle_cpp = open(oracle_dir + "/Functions.cpp", "rb").read()
    scheme_cpp = open(scheme_dir + "/Functions.cpp", "rb").read()
    oracle_h = open(oracle_dir + "/Functions.h", "rb").read()
    scheme_h = open(scheme_dir + "/Functions.h", "rb").read()

    cpp_byte_identical = oracle_cpp == scheme_cpp
    h_byte_identical = oracle_h == scheme_h

    oracle_records = extract_functions_h(oracle_dir + "/Functions.h")
    scheme_records = extract_functions_h(scheme_dir + "/Functions.h")

    if cpp_byte_identical and oracle_records == scheme_records:
        print(
            "Functions.h/Functions.cpp semantic differential: OK "
            "(%d declarations%s)"
            % (
                len(oracle_records),
                ", Functions.h byte-identical" if h_byte_identical else "",
            )
        )
        return 0

    if not cpp_byte_identical:
        print("Functions.cpp differential: MISMATCH (byte)")
        n = min(len(oracle_cpp), len(scheme_cpp))
        for i in range(n):
            if oracle_cpp[i] != scheme_cpp[i]:
                print("  first byte diff at offset %d" % i)
                break
        if len(oracle_cpp) != len(scheme_cpp):
            print(
                "  length differs: oracle=%d scheme=%d"
                % (len(oracle_cpp), len(scheme_cpp))
            )

    if oracle_records != scheme_records:
        print("Functions.h semantic differential: MISMATCH")
        n = min(len(oracle_records), len(scheme_records))
        for i in range(n):
            if oracle_records[i] != scheme_records[i]:
                print("first diff at declaration %d:" % i)
                print("  oracle: %r" % (oracle_records[i],))
                print("  scheme: %r" % (scheme_records[i],))
                break
        if len(oracle_records) != len(scheme_records):
            print(
                "declaration count differs: oracle=%d scheme=%d"
                % (len(oracle_records), len(scheme_records))
            )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
