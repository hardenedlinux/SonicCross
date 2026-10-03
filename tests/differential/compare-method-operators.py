#!/usr/bin/env python3
"""Tier 1 semantic differential for MethodOperators.h (G4).

Extracts the ordered sequence of ``struct TORCH_API {name}`` declarations and
compares the semantic record of each operator between the oracle and
SonicCross:

    (name, schema_type, aten_name, overload_name, schema_str, call, redispatch)

``name`` is the unambiguous (overload-disambiguated) operator name, ``schema_type``
is the dispatcher C++ function type, ``aten_name``/``overload_name`` are the
compile-time name fields, ``schema_str`` the escaped canonical schema string,
and ``call``/``redispatch`` the dispatcher entry points.  Only the struct body is
parsed; the file header, includes and namespace framing are not part of the
semantic record.  Byte identity is reported separately as a secondary signal.
"""

import re
import sys


def extract(path):
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"(?m)^struct TORCH_API ", text)
    records = []
    for block in blocks[1:]:
        m = re.match(r"(\w+) \{\n(.*)\n\};", block, re.S)
        if not m:
            raise SystemExit("malformed struct block: %r" % block[:60])
        name = m.group(1)
        body = m.group(2)
        fields = {}
        for line in body.split("\n"):
            s = line.strip()
            if s.startswith("using schema = "):
                fields["schema_type"] = s[len("using schema = ") : -1]
            elif s.startswith("static constexpr const char* name = "):
                fields["name"] = s[len("static constexpr const char* name = ") : -1]
            elif s.startswith("static constexpr const char* overload_name = "):
                fields["overload_name"] = s[
                    len("static constexpr const char* overload_name = ") : -1
                ]
            elif s.startswith("static constexpr const char* schema_str = "):
                fields["schema_str"] = s[
                    len("static constexpr const char* schema_str = ") : -1
                ]
            elif " call(" in s:
                fields["call"] = s
            elif " redispatch(" in s:
                fields["redispatch"] = s
        missing = {
            k
            for k in ("schema_type", "name", "overload_name", "schema_str", "call", "redispatch")
        } - set(fields)
        if missing:
            raise SystemExit("struct %s missing fields: %s" % (name, sorted(missing)))
        records.append(
            (
                name,
                fields["schema_type"],
                fields["name"],
                fields["overload_name"],
                fields["schema_str"],
                fields["call"],
                fields["redispatch"],
            )
        )
    return records


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-method-operators.py ORACLE.h SCHEME.h")
    oracle = extract(sys.argv[1])
    scheme = extract(sys.argv[2])
    byte_identical = (
        open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
    )

    if oracle == scheme:
        print(
            "MethodOperators.h semantic differential: OK "
            "(%d operators%s)"
            % (len(oracle), ", byte-identical" if byte_identical else "")
        )
        return 0

    print("MethodOperators.h semantic differential: MISMATCH")
    n = min(len(oracle), len(scheme))
    labels = ("name", "schema_type", "name_field", "overload_name", "schema_str", "call", "redispatch")
    for i in range(n):
        if oracle[i] != scheme[i]:
            print("first diff at operator %d (%s):" % (i, oracle[i][0]))
            for label, ov, sv in zip(labels, oracle[i], scheme[i]):
                if ov != sv:
                    print("  %s:\n    oracle: %r\n    scheme: %r" % (label, ov, sv))
            break
    if len(oracle) != len(scheme):
        print("operator count differs: oracle=%d scheme=%d" % (len(oracle), len(scheme)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
