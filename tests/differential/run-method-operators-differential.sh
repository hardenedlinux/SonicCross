#!/bin/sh
set -eu
# G4 differential: MethodOperators.h must be semantically identical (and, as a
# secondary signal, byte-identical) between frozen torchgen and SonicCross.
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

native_yaml="$root/aten/src/ATen/native/native_functions.yaml"
if test ! -f "$native_yaml"; then
    echo "checkout does not contain native_functions.yaml: $root" >&2
    exit 2
fi

dir="$(dirname "$0")"
out=${SONICCROSS_METHOD_OPERATORS_OUTPUT:-$(mktemp -d)}
trap 'rm -rf "$out"' EXIT

SONICCROSS_PYTORCH_ROOT="$root" python3 "$dir/method-operators-oracle.py" > "$out/oracle.h"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile --no-auto-compile "$dir/scheme-method-operators.scm" "$native_yaml" > "$out/scheme.h"

python3 "$dir/compare-method-operators.py" "$out/oracle.h" "$out/scheme.h"
