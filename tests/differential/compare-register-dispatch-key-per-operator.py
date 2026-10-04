#!/usr/bin/env python3
"""Per-operator Register{DispatchKey}.cpp byte differential.

Compares every Register{DispatchKey}{_0,_1,...}.cpp file between the oracle
and SonicCross output directories for whole-file byte identity.  The dispatch
header includes (`#include <ATen/ops/...>`) are part of the comparison, so a
mismatch in the per-operator include set is caught here.
"""

import sys
from pathlib import Path


def register_files(directory):
    return sorted(
        p.name
        for p in Path(directory).iterdir()
        if p.name.startswith("Register") and p.name.endswith(".cpp")
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-register-dispatch-key-per-operator.py ORACLE_DIR SCHEME_DIR"
        )
    oracle_dir, scheme_dir = sys.argv[1], sys.argv[2]
    oracle_files = register_files(oracle_dir)
    scheme_files = register_files(scheme_dir)

    if oracle_files != scheme_files:
        print("per-operator Register{DispatchKey}.cpp differential: MISMATCH (file set)")
        only_oracle = set(oracle_files) - set(scheme_files)
        only_scheme = set(scheme_files) - set(oracle_files)
        if only_oracle:
            print("  only in oracle: %s" % ", ".join(sorted(only_oracle)))
        if only_scheme:
            print("  only in scheme: %s" % ", ".join(sorted(only_scheme)))
        return 1

    mismatches = 0
    for name in oracle_files:
        ob = (Path(oracle_dir) / name).read_bytes()
        sb = (Path(scheme_dir) / name).read_bytes()
        if ob != sb:
            mismatches += 1
            print("MISMATCH: %s (oracle=%dB scheme=%dB)" % (name, len(ob), len(sb)))
            ol = ob.split(b"\n")
            sl = sb.split(b"\n")
            for i in range(max(len(ol), len(sl))):
                o = ol[i] if i < len(ol) else b"<eof>"
                s = sl[i] if i < len(sl) else b"<eof>"
                if o != s:
                    print("  first diff at line %d:" % (i + 1))
                    print("    oracle: %r" % (o,))
                    print("    scheme: %r" % (s,))
                    break

    if mismatches:
        print(
            "per-operator Register{DispatchKey}.cpp differential: %d/%d files mismatch"
            % (mismatches, len(oracle_files))
        )
        return 1

    print(
        "per-operator Register{DispatchKey}.cpp differential: OK (%d files, byte-identical)"
        % len(oracle_files)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
