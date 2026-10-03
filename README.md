# SonicCross

SonicCross is a pure Guile 3.0 / Scheme reimplementation of PyTorch's
`torchgen` — the code generator that turns `native_functions.yaml` and
`tags.yaml` into the C++ operator registration, dispatch, and declaration
sources that ship in libtorch. It is a *semantic-equivalent* compiler: given
the same frozen PyTorch inputs it produces C++ artifacts that are byte-identical
to upstream torchgen's output.

## Status

**v0 is frozen** (commit `8c6838b`). The full generation closure — the 75-entry
artifact manifest plus emitter groups G1–G20 — is complete and byte-identical
against the frozen baseline. No further SonicCross work is planned unless a
concrete bug is found; development now moves to the **SonicBoom** runtime
architecture (C++ → runtime → C ABI).

## Frozen baseline

SonicCross targets one pinned PyTorch revision and does not track upstream:

- repository: `pytorch/pytorch`
- branch: `release/2.14`
- commit: `41ffbc4a994e058af9fe00ed5caba73fc1033359`
- version: `2.14.1a0`

## Architecture

```
native_functions.yaml ─┐
tags.yaml ─────────────┼─► Semantic IR (SRFI-9 records) ─► emitters ─► C++ artifacts
                       ┘
```

1. **YAML ingestion** — `yaml.scm` / `yaml/ffi.scm` (libyaml-backed) parse the
   frozen inputs.
2. **Semantic IR** — `core-ir.scm`, `native-function.scm`, `schema-parser.scm`,
   `generated-functions.scm` reimplement torchgen's Python object model
   (`OperatorName`, `FunctionSchema`, `Type`, `NativeFunction`, …) as
   compiler-internal records. The IR is not a persistence or serialization
   contract.
3. **Generation closure** — one module per emitter group, e.g. `operators.scm`,
   `tensor-body.scm`, `core-sources.scm`, `backend-select.scm`, `ufunc.scm`,
   `functionalization.scm`. Each renderer is byte-compared against frozen
   torchgen.

The exact generation scope (dispatch keys, sharding, file managers, and the
G1–G21 group map) is catalogued in
`design/SonicCross_Generation_Configuration_v0.md`.

## Build & test

```sh
./bootstrap
./configure
make
make check     # 7 self-contained unit tests — no PyTorch checkout required
```

The emitters are Guile modules invoked directly (see the differential harness
below); the `guild torchgen` CLI entry point in `scripts/` is still a
placeholder and is not yet wired to the generation closure.

## Differential validation (developer-only)

Byte-compares SonicCross output against a frozen torchgen oracle:

```sh
SONICCROSS_PYTORCH_ROOT=/path/to/pytorch \
  tests/differential/run-differential.sh
```

The complete suite is the set of `tests/differential/run-*-differential.sh`
runners (manifest + G1–G20 + the fixture semantic dump); every oracle rejects a
checkout whose HEAD is not the frozen commit.

## Repository layout

- `modules/sonic-cross/` — compiler modules: YAML, IR, parser, and the G1–G20 emitters.
- `tests/` — self-contained unit tests (`make check`) and the differential harness.
- `design/` — frozen semantic documents and the generation configuration.
- `scripts/` — the `(scripts torchgen)` CLI module (bootstrap stub).

## Documentation

- `design/SONICCROSS-SPEC.md` — the frozen specification.
- `design/SonicCross_Core_IR_v0_Frozen_Semantics.md` — frozen Core IR semantics.
- `design/SonicCross_Generation_Configuration_v0.md` — generation scope and the G1–G21 group map.
