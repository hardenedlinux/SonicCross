#!/usr/bin/env python3
"""Tier 1 semantic differential for RegisterFunctionalization*.cpp (G13).

Byte identity across the five shards is the primary signal.  On mismatch the
first differing shard is diffed so a single wrong emitter is localised.
"""

import difflib
import sys
from pathlib import Path

FILES = [
    "RegisterFunctionalizationEverything.cpp",
    "RegisterFunctionalization_0.cpp",
    "RegisterFunctionalization_1.cpp",
    "RegisterFunctionalization_2.cpp",
    "RegisterFunctionalization_3.cpp",
]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-register-functionalization.py ORACLE_DIR SCHEME_DIR"
        )
    oracle_dir = Path(sys.argv[1])
    scheme_dir = Path(sys.argv[2])

    for name in FILES:
        op = oracle_dir / name
        sp = scheme_dir / name
        for p, label in ((op, "oracle"), (sp, "scheme")):
            if not p.is_file():
                print("missing %s file: %s" % (label, name))
                return 1
        ob = op.read_bytes()
        sb = sp.read_bytes()
        if ob != sb:
            print("RegisterFunctionalization differential: MISMATCH in %s" % name)
            print("  oracle=%d bytes scheme=%d bytes" % (len(ob), len(sb)))
            o_text = op.read_text(encoding="utf-8").split("\n")
            s_text = sp.read_text(encoding="utf-8").split("\n")
            diff = list(difflib.unified_diff(o_text, s_text, lineterm=""))
            print("\n".join(diff[:120]))
            return 1

    print(
        "RegisterFunctionalization differential: OK (byte-identical, %d shards)"
        % len(FILES)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
