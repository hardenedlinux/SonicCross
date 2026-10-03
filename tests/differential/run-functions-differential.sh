#!/bin/sh
set -eu
# G5 differential: Functions.h and Functions.cpp must be semantically identical
# (and, as a secondary signal, byte-identical) between frozen torchgen and
# SonicCross.
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
out=${SONICCROSS_FUNCTIONS_OUTPUT:-$(mktemp -d)}
oracle_dir="$out/oracle"
scheme_dir="$out/scheme"
mkdir -p "$oracle_dir" "$scheme_dir"
trap 'rm -rf "$out"' EXIT

SONICCROSS_PYTORCH_ROOT="$root" python3 "$dir/functions-oracle.py" "$oracle_dir"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile --no-auto-compile "$dir/scheme-functions.scm" "$native_yaml" "$scheme_dir"

python3 "$dir/compare-functions.py" "$oracle_dir" "$scheme_dir"
