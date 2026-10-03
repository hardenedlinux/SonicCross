# SonicCross Development Guide

SonicCross is a pure Guile 3.0 / Scheme reimplementation of PyTorch `torchgen`
— a semantic-equivalent compiler for the C++ operator registration, dispatch,
and declaration sources.

**Status: v0 frozen** (commit `13aa2e0`, tag `v0.0.1`). The generation closure (manifest +
G1–G20) is complete and byte-identical against the frozen baseline. No further
SonicCross work is planned unless a concrete bug is found; the next phase is
the **SonicBoom** runtime architecture (C++ → runtime → C ABI).

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
- The generation closure (G1–G20) is frozen: do not add emitters, artifacts,
  or dispatcher behavior to it.
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

The oracle must reject any checkout whose HEAD is not the frozen commit. The
full per-group suite is `tests/differential/run-*-differential.sh` (manifest +
G1–G20 plus the fixture semantic dump), each byte-comparing oracle vs
SonicCross.

## Scope boundaries

SonicCross v0 is frozen. The emitter closure (G1–G20) and the differential
harness are complete. Do not add new emitters, artifacts, or IR; do not
redesign Core IR, Schema Parser, NativeFunction, or the generation APIs.

Out of scope (never part of SonicCross): G21 `Declarations.yaml` (YAML
serialization), AOTI C shim, `VmapGeneratedPlumbing.h`, per-operator headers,
Python bindings, ExecuTorch / lazy / selective build / static dispatch, and the
runtime dispatcher / boxing / unboxing (that is SonicBoom's domain).
