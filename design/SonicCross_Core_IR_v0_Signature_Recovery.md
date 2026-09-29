# SonicCross Core IR v0 — Signature Recovery

This file records the current implementation-level recovery supplied for
Core IR v0. It supplements the earlier frozen semantic document; for this
slice it resolves the previously marked unknowns.

## OperatorName

Parse on `.` into a base/operator part and an optional overload. Ordinary
names ending in `_` are inplace names, with the trailing `_` excluded from
the semantic base. Dunder parsing strips an optional `namespace::` prefix and
uses exactly `^__([^_]+)__$`. The augmented assignment names are:

```text
add sub mul div mod pow lshift rshift and xor or
```

`__iNAME__` is represented as inplace with base `NAME`; an unrecognized
`i...` dunder is an error. Other dunders are non-inplace with base equal to
the captured name. Parsed names have `functional_overload = false`; generated
functional variants may set it explicitly.

## SchemaKind

The only values are `functional`, `inplace`, `mutable`, and `out`. Classify
in this order: explicit out arguments, mutable/write self, mutable/write
non-self argument, otherwise functional. `is_out_fn` means explicit out
arguments, independent of the operator spelling.

## ViewSchemaKind

The only values are `aliasing`, `aliasing_inplace`, and `non_aliasing`.

## Type

Use a structured recursive representation for Tensor, Scalar, Any, Number,
Int, Float, Bool, String, SymInt, MemoryFormat, Dimname, Optional(Type), and
List(Type). Int and SymInt remain distinct. `has_symint` recurses through
Optional and List. Tensor is the tensor-like type in this slice.

## Argument, Return, Arguments

Argument preserves `name`, `type`, `default`, `annotation`, and `is_write`.
Return preserves `name`, `type`, and `annotation`. Arguments preserve the
semantic groups `pre_self_positional`, `self_arg`, `post_self_positional`,
`pre_tensor_options_kwarg_only`, `tensor_options`,
`post_tensor_options_kwarg_only`, and `out`.

## FunctionSchema.signature

The signature is a normalized FunctionSchema. It keeps the original base and
dunder bit while setting inplace false, overload empty, and
functional_overload false. It strips all argument/return annotations, drops
tensor_options and out arguments, merges post tensor-options keyword-only
arguments into the pre group, and empties the post group.

Original return order is preserved. Mutable/write arguments not already
represented by a return produce synthetic returns; with
`keep_return_names=true` their names are `<argument-name>_out`. With
`keep_return_names=false`, every return name is `None`; with true, original
return names are preserved. Signature equality is structural equality of the
normalized name, arguments, and returns.
