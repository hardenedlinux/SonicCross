#!/usr/bin/env python3
"""Tier 1 semantic differential for Register{DispatchKey}.cpp (G2).

Extracts the semantic record from each generated Register{DispatchKey}.cpp:
the ordered sequence of ``(dispatch_key, schema_name, kernel_name)``
``m.impl`` registrations.  The comparator normalises whitespace/newlines (the
unstructured and structured ``m.impl`` layouts differ only in where the
``TORCH_FN(...)`` payload lands), comments, and other presentation details, so
the ground of comparison is the registration mapping itself — not its
formatting.

Byte identity is retained as a secondary/spot-check signal: it is computed for
every file and reported, but it is not the ground of comparison.
"""

import re
import sys
from pathlib import Path


def register_files(directory):
    return sorted(
        p.name
        for p in Path(directory).iterdir()
        if p.name.startswith("Register") and p.name.endswith(".cpp")
    )


# The registration kind is uniform in G2: every operator is registered via
# ``m.impl("schema", TORCH_FN(kernel));``.  There is no m.def / m.fallback /
# structured-macro registration in these files.  The ``m.impl`` call may span
# lines in the unstructured form, hence the ``\s*`` between tokens.
IMPL_RE = re.compile(
    r'm\.impl\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*TORCH_FN\(\s*(\w+)\s*\)\s*\);'
)

# Dispatch key is taken from the enclosing TORCH_LIBRARY_IMPL(aten, <key>, m)
# block.  Every block in a given file shares the same key, so the first match
# names the whole file's key.
KEY_RE = re.compile(r"TORCH_LIBRARY_IMPL\(\s*aten\s*,\s*(\w+)\s*,\s*m\s*\)")


def extract(directory):
    """Return {filename: [(dispatch_key, schema, kernel), ...]}."""
    result = {}
    for name in register_files(directory):
        text = (Path(directory) / name).read_text(encoding="utf-8")
        key_match = KEY_RE.search(text)
        dispatch_key = key_match.group(1) if key_match else None
        result[name] = [
            (dispatch_key, m.group(1), m.group(2))
            for m in IMPL_RE.finditer(text)
        ]
    return result


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare-register-dispatch-key.py ORACLE_DIR SCHEME_DIR"
        )
    oracle_dir, scheme_dir = sys.argv[1], sys.argv[2]
    oracle_files = register_files(oracle_dir)
    scheme_files = register_files(scheme_dir)

    if oracle_files != scheme_files:
        print("Register{DispatchKey}.cpp semantic differential: MISMATCH (file set)")
        only_oracle = set(oracle_files) - set(scheme_files)
        only_scheme = set(scheme_files) - set(oracle_files)
        if only_oracle:
            print("  only in oracle: %s" % ", ".join(sorted(only_oracle)))
        if only_scheme:
            print("  only in scheme: %s" % ", ".join(sorted(only_scheme)))
        return 1

    oracle = extract(oracle_dir)
    scheme = extract(scheme_dir)
    byte_identical = True

    for name in oracle_files:
        oracle_bytes = (Path(oracle_dir) / name).read_bytes()
        scheme_bytes = (Path(scheme_dir) / name).read_bytes()
        if oracle_bytes != scheme_bytes:
            byte_identical = False

        o_records = oracle[name]
        s_records = scheme[name]
        if o_records != s_records:
            print("Register{DispatchKey}.cpp semantic differential: MISMATCH")
            print("  first diff: %s" % name)
            n = min(len(o_records), len(s_records))
            for i in range(n):
                if o_records[i] != s_records[i]:
                    print("  at registration %d:" % i)
                    print("    oracle: key=%r schema=%r kernel=%r"
                          % o_records[i])
                    print("    scheme: key=%r schema=%r kernel=%r"
                          % s_records[i])
                    break
            if len(o_records) != len(s_records):
                print("  registration count differs: oracle=%d scheme=%d"
                      % (len(o_records), len(s_records)))
            return 1

    total = sum(len(records) for records in oracle.values())
    print(
        "Register{DispatchKey}.cpp semantic differential: OK "
        "(%d files, %d registrations%s)"
        % (len(oracle_files), total, ", byte-identical" if byte_identical else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
