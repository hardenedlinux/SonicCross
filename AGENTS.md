# SonicCross Agent Instructions

These instructions apply to work in this repository.

**Status: SonicCross v0 is frozen** (commit `13aa2e0`, tag `v0.0.1`). The generation closure
(manifest + G1–G20) is complete and byte-identical against frozen torchgen.
Per-operator header generation (`ATen/ops/*.h` + per-operator
`Register{key}.cpp`) is implemented as a flag-gated mode; the frozen G1–G20
aggregate closure is unchanged. The next phase is the SonicBoom runtime
architecture.

## Before editing

1. Read the relevant frozen semantic document and current implementation.
2. Inspect existing APIs and tests before adding interfaces.
3. Confirm the requested change is within the stated scope.
4. Do not read unrelated upstream PyTorch code or re-derive already frozen
   semantics.

## Implementation

- Use Guile 3.0 conventions and existing SonicCross module structure.
- Preserve deterministic ordering and structural equality.
- Make the smallest compatible change.
- Do not silently normalize, broaden, or reinterpret frozen semantics.
- Do not replace irregex, the YAML backend, or existing parser layers.
- Keep generated build files synchronized when Autotools inputs change.

## Validation

At minimum, run the tests relevant to the change. For semantic changes, run:

```sh
make clean
make
make check
```

Report exact failures and the first semantic mismatch. Never hide a
differential mismatch by changing the comparator or fixture.

## Differential validation

The only accepted Python oracle is the checkout at:

```text
41ffbc4a994e058af9fe00ed5caba73fc1033359
```

Use:

```sh
SONICCROSS_PYTORCH_ROOT=/path/to/pytorch \
  tests/differential/run-differential.sh
```

If the checkout is missing or has another revision, report the blocker. Do
not fall back to another revision and do not fabricate expected output.

## Out of scope (v0 frozen)

SonicCross v0 is frozen. Do not add features. Still out of scope:

- new torchgen archaeology beyond the frozen `41ffbc4a…` baseline;
- Core IR / Schema Parser / NativeFunction / generation-API redesign;
- BackendIndex semantics and NativeFunctionsViewGroup (never needed — the
  closure uses the nop selector);
- G21 `Declarations.yaml` (YAML serialization);
- AOTI C shim, `VmapGeneratedPlumbing.h`;
- Python bindings, ExecuTorch / lazy / selective build / static dispatch;
- runtime dispatcher, boxing/unboxing, and C++ runtime codegen (SonicBoom).
