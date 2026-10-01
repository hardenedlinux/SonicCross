# SonicCross Development Guide

SonicCross is a pure Guile 3.0 / Scheme semantic compiler.

## Frozen baseline

Semantic compatibility targets PyTorch:

- repository: `pytorch/pytorch`
- branch: `release/2.14`
- commit: `41ffbc4a994e058af9fe00ed5caba73fc1033359`
- version: `2.14.1a0`

Do not use a newer PyTorch version to fill semantic gaps. Do not perform new
torchgen archaeology unless the task explicitly authorizes a narrowly defined
source audit.

## Project rules

- Preserve frozen semantic contracts and canonical representations.
- Prefer small, deterministic, idiomatic Guile 3.0 data structures.
- Do not redesign Core IR, Schema Parser, NativeFunction, or generation APIs
  to accommodate convenience.
- Do not implement out-of-scope code generation or dispatcher behavior.
- Keep `SONICCROSS-SPEC.md` and frozen semantic documents unchanged.
- Keep normal `make check` self-contained; external PyTorch checkouts belong to
  developer-only differential validation.

## Build and test

```sh
./bootstrap
./configure
make clean
make
make check
```

For differential validation, use only the pinned checkout:

```sh
SONICCROSS_PYTORCH_ROOT=/path/to/pytorch \
  tests/differential/run-differential.sh
```

The oracle must reject any checkout whose HEAD is not the frozen commit.

## Scope boundaries

Do not add BackendIndex semantics, NativeFunctionsViewGroup, dispatcher,
registration, boxing/unboxing, C++ code generation, or generated kernel bodies
unless a later task explicitly enables them.
