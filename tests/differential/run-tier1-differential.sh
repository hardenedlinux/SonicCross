#!/bin/sh
set -eu
# Tier 1 semantic differential harness.
#
# Proves that the semantic information observable in the generated C++ is
# equivalent between frozen torchgen and SonicCross.  Each of the three P1c
# artifacts is compared by an artifact-specific extractor against the generated
# C++ itself (never SonicCross's internal IR), normalising only whitespace,
# formatting and comments.  Byte-identity is retained per artifact as a
# secondary signal, not as the ground of comparison.
#
#   G1  RegisterSchema.cpp           -> (schema_string, sorted_tag_set)*
#   G2  Register{DispatchKey}.cpp    -> (dispatch_key, schema, kernel)* per file
#   G3  RegistrationDeclarations.h   -> (returns_type, name, args, schema,
#                                        dispatch, default)*
#
# Developer-only: requires the frozen PyTorch checkout.

root=${SONICCROSS_PYTORCH_ROOT:-}
if test -z "$root"; then
    echo "set SONICCROSS_PYTORCH_ROOT to frozen PyTorch commit 41ffbc4a994e058af9fe00ed5caba73fc1033359" >&2
    exit 2
fi

if test ! -f "$root/torchgen/gen.py"; then
    echo "invalid PyTorch checkout: $root" >&2
    exit 2
fi

dir="$(dirname "$0")"
status=0

for runner in \
    run-register-schema-differential.sh \
    run-register-dispatch-key-differential.sh \
    run-registration-declarations-differential.sh; do
    echo "=== Tier 1: $runner ==="
    if SONICCROSS_PYTORCH_ROOT="$root" sh "$dir/$runner"; then
        echo ""
    else
        echo "=== FAILED: $runner ===" >&2
        status=1
    fi
done

if test "$status" -eq 0; then
    echo "Tier 1 semantic differential: ALL PASS"
else
    echo "Tier 1 semantic differential: FAILED" >&2
fi
exit "$status"
