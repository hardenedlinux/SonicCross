#!/usr/bin/env python3
"""Tier 1 semantic differential for RegisterSchema.cpp.

Extracts the canonical semantic record from a rendered RegisterSchema.cpp — the
ordered sequence of ``(schema_string, sorted_tag_set)`` registrations — and
compares the oracle's and SonicCross's records.  Byte identity is reported
separately as a secondary/spot-check signal, since it is not the ground of
comparison.
"""

import re
import sys


def unescape_cpp(s):
    """Reverse torchgen gen.cpp_string: \\\\ \\\" \\a \\b \\f \\n \\v \\t."""
    mapping = {
        "\\": "\\",
        '"': '"',
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "v": "\v",
        "t": "\t",
    }
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in mapping:
            out.append(mapping[s[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract(path):
    text = open(path, encoding="utf-8").read()
    tag_vectors = {}
    registrations = []
    for line in text.split("\n"):
        stripped = line.strip()
        m = re.match(r"^const std::vector<at::Tag> tags_(\d+) = \{(.*)\};\s*$", stripped)
        if m:
            idx = int(m.group(1))
            body = m.group(2).strip()
            tags = tuple(sorted(t.strip() for t in body.split(",") if t.strip()))
            tag_vectors[idx] = tags
            continue
        m = re.match(r'^m\.def\("((?:[^"\\]|\\.)*)",\s*(.*?)\);\s*$', stripped)
        if m:
            schema = unescape_cpp(m.group(1))
            arg = m.group(2).strip()
            if arg == "{}":
                tags = ()
            elif arg.startswith("tags_"):
                tags = tag_vectors[int(arg[len("tags_") :])]
            else:
                tags = ("<unresolved:%s>" % arg,)
            registrations.append((schema, tags))
    return registrations


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-register-schema.py ORACLE.cpp SCHEME.cpp")
    oracle = extract(sys.argv[1])
    scheme = extract(sys.argv[2])
    byte_identical = open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()

    if oracle == scheme:
        print(
            "RegisterSchema.cpp semantic differential: OK "
            "(%d registrations, %d distinct tag vectors%s)"
            % (len(oracle), len({t for _, t in oracle}), ", byte-identical" if byte_identical else "")
        )
        return 0

    print("RegisterSchema.cpp semantic differential: MISMATCH")
    n = min(len(oracle), len(scheme))
    for i in range(n):
        if oracle[i] != scheme[i]:
            print("first diff at registration %d:" % i)
            print("  oracle schema: %r tags %r" % (oracle[i][0], oracle[i][1]))
            print("  scheme schema: %r tags %r" % (scheme[i][0], scheme[i][1]))
            break
    if len(oracle) != len(scheme):
        print("registration count differs: oracle=%d scheme=%d" % (len(oracle), len(scheme)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
