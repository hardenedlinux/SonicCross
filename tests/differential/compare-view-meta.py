#!/usr/bin/env python3
"""Tier 1 semantic differential for FunctionalInverses.h + ViewMetaClasses.h/.cpp.

Byte identity across the three artifacts is the primary signal.  On mismatch the
first differing file is diffed so a single wrong emitter is localised.
"""

import difflib
import sys
from pathlib import Path

FILES = [
    "FunctionalInverses.h",
    "ViewMetaClasses.h",
    "ViewMetaClasses.cpp",
]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-view-meta.py ORACLE_DIR SCHEME_DIR")
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
            print("view-meta differential: MISMATCH in %s" % name)
            print("  oracle=%d bytes scheme=%d bytes" % (len(ob), len(sb)))
            o_text = op.read_text(encoding="utf-8").split("\n")
            s_text = sp.read_text(encoding="utf-8").split("\n")
            diff = list(difflib.unified_diff(o_text, s_text, lineterm=""))
            print("\n".join(diff[:120]))
            return 1

    print("view-meta differential: OK (byte-identical, %d files)" % len(FILES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
