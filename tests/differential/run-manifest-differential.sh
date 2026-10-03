#!/bin/sh
set -eu
# Tier 0 artifact-manifest differential: the frozen torchgen orchestration's
# artifact manifest (file set, sharding, per-shard operator membership,
# deterministic ordering) must be byte-identical to SonicCross's.
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

out=${SONICCROSS_MANIFEST_OUTPUT:-$(mktemp)}
trap 'rm -f "$out" "$out.python" "$out.scheme"' EXIT

python3 "$(dirname "$0")/manifest-oracle.py" > "$out.python"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile "$(dirname "$0")/scheme-manifest.scm" "$native_yaml" > "$out.scheme"

if cmp -s "$out.python" "$out.scheme"; then
    echo "artifact manifest differential: OK ($(wc -l < "$out.python") artifacts)"
else
    echo "artifact manifest differential: MISMATCH" >&2
    diff -u "$out.python" "$out.scheme" >&2 || true
    exit 1
fi
