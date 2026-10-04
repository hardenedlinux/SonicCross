#!/usr/bin/env python3
"""Per-operator header differential: byte-compare the whole header closure.

Compares every captured header (ops/{name}[_.*].h, the five aggregate shims,
and the fourteen {DispatchKey}Functions*.h) between the oracle and SonicCross
output directories.  The ground of comparison is whole-file byte identity;
the file set must match exactly (no missing/extra files on either side).
"""

import sys
from pathlib import Path


def files_in(directory):
    return sorted(p.name for p in Path(directory).iterdir() if p.is_file())


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-per-operator-headers.py ORACLE_DIR SCHEME_DIR"
        )
    oracle_dir = Path(sys.argv[1])
    scheme_dir = Path(sys.argv[2])

    oracle_files = files_in(oracle_dir)
    scheme_files = files_in(scheme_dir)

    if oracle_files != scheme_files:
        print("per-operator-headers differential: MISMATCH (file set)")
        only_oracle = set(oracle_files) - set(scheme_files)
        only_scheme = set(scheme_files) - set(oracle_files)
        if only_oracle:
            print("  only in oracle (%d): %s"
                  % (len(only_oracle), ", ".join(sorted(only_oracle)[:10])))
        if only_scheme:
            print("  only in scheme (%d): %s"
                  % (len(only_scheme), ", ".join(sorted(only_scheme)[:10])))
        return 1

    mismatches = 0
    for name in oracle_files:
        ob = (oracle_dir / name).read_bytes()
        sb = (scheme_dir / name).read_bytes()
        if ob != sb:
            mismatches += 1
            print("MISMATCH: %s (oracle=%dB scheme=%dB)"
                  % (name, len(ob), len(sb)))
            # First differing line for diagnosis.
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
            "per-operator-headers differential: %d/%d files mismatch"
            % (mismatches, len(oracle_files))
        )
        return 1

    print(
        "per-operator-headers differential: OK (%d files, byte-identical)"
        % len(oracle_files)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
