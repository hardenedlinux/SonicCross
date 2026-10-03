#!/usr/bin/env python3
"""Tier 1 semantic differential for RegistrationDeclarations.h.

Extracts the ordered sequence of registration declarations and compares the
semantic record — (returns_type, name, args, schema, dispatch, default) — of
the oracle against SonicCross.  The schema is decoded from the JSON comment, so
the comparison is against the canonical schema string, not its JSON encoding.
Byte identity is reported separately as a secondary signal.
"""

import json
import re
import sys

DECL_RE = re.compile(r"^(.*)\s+(\w+)\((.*)\); // (\{.*\})\s*$")


def extract(path):
    text = open(path, encoding="utf-8").read()
    records = []
    for line in text.split("\n"):
        m = DECL_RE.match(line)
        if not m:
            continue
        returns_type = m.group(1).strip()
        name = m.group(2)
        args = m.group(3)
        comment = json.loads(m.group(4))
        records.append(
            (returns_type, name, args, comment["schema"], comment["dispatch"], comment["default"])
        )
    return records


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-registration-declarations.py ORACLE.h SCHEME.h"
        )
    oracle = extract(sys.argv[1])
    scheme = extract(sys.argv[2])
    byte_identical = open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()

    if oracle == scheme:
        print(
            "RegistrationDeclarations.h semantic differential: OK "
            "(%d declarations%s)"
            % (len(oracle), ", byte-identical" if byte_identical else "")
        )
        return 0

    print("RegistrationDeclarations.h semantic differential: MISMATCH")
    n = min(len(oracle), len(scheme))
    for i in range(n):
        if oracle[i] != scheme[i]:
            o, s = oracle[i], scheme[i]
            print("first diff at declaration %d:" % i)
            for label, ov, sv in zip(
                ("returns_type", "name", "args", "schema", "dispatch", "default"),
                o,
                s,
            ):
                if ov != sv:
                    print("  %s:\n    oracle: %r\n    scheme: %r" % (label, ov, sv))
            break
    if len(oracle) != len(scheme):
        print(
            "declaration count differs: oracle=%d scheme=%d"
            % (len(oracle), len(scheme))
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
