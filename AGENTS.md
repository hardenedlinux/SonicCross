# SonicCross Agent Instructions

These instructions apply to work in this repository.

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

## Out of scope unless explicitly requested

- new torchgen archaeology;
- Core IR redesign;
- BackendIndex behavior;
- NativeFunctionsViewGroup;
- dispatcher or registration;
- autograd or functionalization codegen;
- unboxing or C++ code generation.
