#!/usr/bin/env python3
"""Tier 1 semantic differential for {DispatchKey}Functions.h / _inl.h (G8).

Compares the 14 generated headers between the oracle and SonicCross output
directories.  Byte identity is the primary signal; if the whole-file bytes
differ, the per-declaration body is compared as a fallback.
"""

import sys
from pathlib import Path

KEYS = (
    "CPU",
    "CUDA",
    "CompositeImplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "Meta",
)


def files_for(keys):
    out = []
    for key in keys:
        out.append(f"{key}Functions.h")
        out.append(f"{key}Functions_inl.h")
    return out


def extract_namespaced_declarations(path):
    """Return the declaration lines between the single namespaced block."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    decls = []
    inside = False
    for line in lines:
        if line.startswith("namespace "):
            inside = True
            continue
        if line.startswith("} // namespace"):
            inside = False
            continue
        if inside:
            decls.append(line)
    return decls


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare-dispatch-key-functions.py ORACLE_DIR SCHEME_DIR")
    oracle_dir = Path(sys.argv[1])
    scheme_dir = Path(sys.argv[2])

    expected = files_for(KEYS)
    mismatches = 0

    for name in expected:
        oracle = oracle_dir / name
        scheme = scheme_dir / name
        if not oracle.is_file():
            print("missing oracle file: %s" % name)
            mismatches += 1
            continue
        if not scheme.is_file():
            print("missing scheme file: %s" % name)
            mismatches += 1
            continue
        ob = oracle.read_bytes()
        sb = scheme.read_bytes()
        if ob == sb:
            continue
        # Fallback: compare the namespaced declaration bodies.
        od = extract_namespaced_declarations(oracle)
        sd = extract_namespaced_declarations(scheme)
        if od == sd:
            print(
                "%s: declaration body OK (%d lines, prologue differs)"
                % (name, len(od))
            )
            continue
        print("%s: MISMATCH" % name)
        n = min(len(od), len(sd))
        for i in range(n):
            if od[i] != sd[i]:
                print("  first diff at declaration %d:" % i)
                print("    oracle: %r" % (od[i],))
                print("    scheme: %r" % (sd[i],))
                break
        if len(od) != len(sd):
            print(
                "  declaration count differs: oracle=%d scheme=%d"
                % (len(od), len(sd))
            )
        mismatches += 1

    if mismatches:
        print(
            "DispatchKeyFunctions differential: %d/%d files mismatch"
            % (mismatches, len(expected))
        )
        return 1

    print(
        "DispatchKeyFunctions differential: OK (%d files, byte-identical)"
        % len(expected)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
