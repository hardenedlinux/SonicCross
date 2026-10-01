#!/bin/sh
set -eu

root=${SONICCROSS_PYTORCH_ROOT:-}
if test -z "$root"; then
    echo "set SONICCROSS_PYTORCH_ROOT to frozen PyTorch commit 41ffbc4a994e058af9fe00ed5caba73fc1033359" >&2
    exit 2
fi

if test ! -f "$root/torchgen/model.py"; then
    echo "invalid PyTorch checkout: $root" >&2
    exit 2
fi

fixture=${SONICCROSS_DIFFERENTIAL_FIXTURE:-$(dirname "$0")/fixtures/core-v0.yaml}
out=${SONICCROSS_DIFFERENTIAL_OUTPUT:-$(mktemp)}
trap 'rm -f "$out" "$out.python" "$out.scheme"' EXIT

python3 "$(dirname "$0")/python-oracle.py" "$fixture" > "$out.python"
GUILE_LOAD_PATH="$(pwd)/modules:$(pwd)/scripts:${GUILE_LOAD_PATH:-}" \
  guile "$(dirname "$0")/scheme-dump.scm" "$fixture" > "$out.scheme"

python3 "$(dirname "$0")/compare.py" "$fixture" "$out.python" "$out.scheme"
