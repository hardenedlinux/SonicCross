#!/bin/sh
set -eu
# G16-G20 differential: Operators.h/.cpp, TensorBody.h, aten_interned_strings.h,
# enum_tag.h, TensorMethods.cpp, ATenOpList.cpp, and RegisterBackendSelect.cpp
# must be byte-identical between frozen torchgen and SonicCross.
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
out=${SONICCROSS_CORE_SOURCES_OUTPUT:-$(mktemp -d)}
mkdir -p "$out/oracle" "$out/scheme"
trap 'rm -rf "$out"' EXIT

SONICCROSS_PYTORCH_ROOT="$root" python3 "$dir/core-sources-oracle.py" "$out/oracle"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile --no-auto-compile "$dir/scheme-core-sources.scm" "$native_yaml" "$out/scheme"

python3 "$dir/compare-core-sources.py" "$out/oracle" "$out/scheme"
