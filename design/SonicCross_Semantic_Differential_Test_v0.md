# SonicCross Semantic Differential Test v0

This validation compares the frozen Python torchgen implementation at
commit `41ffbc4a994e058af9fe00ed5caba73fc1033359` with SonicCross.

The canonical dump is line-oriented and UTF-8 encoded. Records are emitted
in semantic order: original input order, followed by generated records in
the order returned by the single-pass generator. Sets are sorted with the
ordinary lexical string order.

```text
record-begin
index: 000000
namespace: aten
operator: add.Tensor
base: add
overload: Tensor
schema: add.Tensor(Tensor self) -> Tensor
schema-kind: functional
generated: false
tags: core,pt2_compliant_tag
dispatch-begin
dispatch: CPU|add_cpu|false|at::native|false
dispatch-end
record-end
```

The fields are:

* `operator`, `base`, and `overload` identify the operator;
* `schema` is the canonical FunctionSchema rendering;
* `schema-kind` is one of `functional`, `inplace`, `mutable`, or `out`;
* `generated` is supplied by the transformation boundary, not inferred from
  tags;
* `tags` is a lexically sorted set, or `-` when empty;
* each dispatch line is
  `DispatchKey|kernel|structured|cpp_namespace|supports_symint`.

Empty optional strings are represented by `-`; booleans are `true` or
`false`. Dispatch entries are sorted by dispatch key and then kernel. No
object repr, hash, address, or source-order-dependent map representation is
included.

## Running the differential corpus

The Python oracle requires a local checkout of the frozen baseline:

```sh
SONICCROSS_PYTORCH_ROOT=/path/to/pytorch-41ffbc4a994e \
  tests/differential/run-differential.sh
```

An alternate developer fixture can be selected with
`SONICCROSS_DIFFERENTIAL_FIXTURE=/path/to/fixture.yaml`. The checked-in
corpus remains the default and is the only corpus intended for a
self-contained regression run.

The script verifies that the checkout contains the expected commit before
running the actual `torchgen.model` and
`torchgen.native_function_generation.add_generated_native_functions` code.
The ordinary Automake test suite does not require this external checkout.
