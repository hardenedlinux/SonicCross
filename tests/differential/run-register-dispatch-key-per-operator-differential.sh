#!/bin/sh
set -eu
# Per-operator Register{DispatchKey}.cpp differential: the per-operator
# dispatch/ops header includes baked into Register{DispatchKey}.cpp must be
# byte-identical between frozen torchgen and SonicCross.
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
out=${SONICCROSS_G2P_OUTPUT:-$(mktemp -d)}
trap 'rm -rf "$out"' EXIT
mkdir -p "$out/oracle" "$out/scheme"

SONICCROSS_PYTORCH_ROOT="$root" SONICCROSS_G2P_OUTPUT="$out/oracle" \
  python3 "$dir/register-dispatch-key-per-operator-oracle.py" > "$out/oracle.manifest"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile --no-auto-compile "$dir/scheme-register-dispatch-key-per-operator.scm" \
  "$native_yaml" "$out/scheme" > "$out/scheme.manifest" 2>/dev/null

python3 "$dir/compare-register-dispatch-key-per-operator.py" "$out/oracle" "$out/scheme"
