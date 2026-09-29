# SonicCross Core IR v0 — Frozen Semantic Recovery

> Status: implementation-level specification recovered from the previously frozen SonicCross archaeology/audit materials.
>
> Scope: **Core IR v0 only**. No new archaeology, no redesign, no inferred semantics.
>
> PyTorch baseline:
> - commit: `41ffbc4a994e058af9fe00ed5caba73fc1033359`
> - branch: `release/2.14`
> - tag: `v2.14.1-rc1`

---

## 1. OperatorName / BaseOperatorName

### 1.1 Structure

```text
OperatorName =
    BaseOperatorName
    +
    overload_name: string
```

```text
BaseOperatorName =
    base: string
    inplace: bool
    dunder_method: bool
    functional_overload: bool
```

These fields are semantic IR fields.

### 1.2 Parsing

`OperatorName.parse(string)` splits on `.`:

```text
"add.Tensor"
    base/operator part = "add"
    overload_name      = "Tensor"
```

`overload_name` may be empty.

### 1.3 Inplace

The `inplace` flag is inferred from the base name ending in `_`.

```text
add
    base = "add"
    inplace = false

add_
    base = "add"
    inplace = true
```

The `_` is therefore part of the semantic interpretation of the base name, not the overload name.

Examples:

```text
add.Tensor
    base = "add"
    inplace = false
    overload_name = "Tensor"

add_.Tensor
    base = "add"
    inplace = true
    overload_name = "Tensor"
```

### 1.4 functional_overload

`functional_overload` is **not** a normal string-parsing result.

It is explicitly set when torchgen creates a generated functional variant.

In particular, when a mutable operator is converted to a functional variant:

```text
mutable source
    ->
functional generated variant
```

the generated `BaseOperatorName` has:

```text
inplace = false
functional_overload = true
```

This bit exists to disambiguate generated functional variants.

### 1.5 Examples

```text
add
    base = "add"
    inplace = false
    overload_name = ""

add.Tensor
    base = "add"
    inplace = false
    overload_name = "Tensor"

add.out
    base = "add"
    inplace = false
    overload_name = "out"

add.Tensor_out
    base = "add"
    inplace = false
    overload_name = "Tensor_out"

add_
    base = "add"
    inplace = true
    overload_name = ""
```

### 1.6 dunder_method

`dunder_method` is a frozen field of `BaseOperatorName`.

**The exact parsing predicate was not recovered from the frozen materials.**

Therefore:

```text
dunder_method predicate = UNKNOWN
```

Do not invent a predicate during implementation.

---

# 2. SchemaKind

The four schema kinds are:

```text
functional
inplace
mutable
out
```

## 2.1 Semantics

| SchemaKind | Confirmed condition |
|---|---|
| `out` | explicit out arguments exist |
| `inplace` | mutable `self` / `self!` exists |
| `mutable` | mutable non-self argument exists |
| `functional` | no mutable arguments and no out arguments |

The key distinction is:

```text
self mutation
    -> inplace

non-self mutation
    -> mutable
```

## 2.2 `is_out_fn()`

Confirmed semantic definition:

```text
is_out_fn()
    <=> operator has explicit out arguments
```

It is **not** determined by whether the operator name contains `.out`.

## 2.3 Decision procedure

The recovered semantic decision table is:

```text
if has_out_arguments:
    OUT

else if self_is_mutable:
    INPLACE

else if has_other_mutable_arguments:
    MUTABLE

else:
    FUNCTIONAL
```

### 2.4 Important unresolved detail

The frozen materials do not contain a complete exhaustive `kind()` decision table covering every possible combination of:

- mutable self
- mutable non-self arguments
- out arguments
- return alias annotations

Therefore the exact behavior of every unusual combination is:

```text
UNKNOWN / NOT FROZEN
```

In particular, do not invent a rule that return alias annotations independently determine `SchemaKind`.

---

# 3. ViewSchemaKind

Three values are frozen:

```text
aliasing
aliasing_inplace
non_aliasing
```

## 3.1 Mapping

| Operator category | ViewSchemaKind |
|---|---|
| `view` | `aliasing` |
| `inplace_view` | `aliasing_inplace` |
| `view_copy` | `non_aliasing` |

### `view`

Returned value aliases the input.

```text
view -> aliasing
```

### `inplace_view`

Combines in-place mutation with aliasing.

```text
inplace_view -> aliasing_inplace
```

### `view_copy`

Produces a non-aliasing copy.

```text
view_copy -> non_aliasing
```

`view_copy` is a normal NativeFunction declaration; it is not itself a generated operator variant.

---

# 4. Type

## 4.1 Required semantic model

Core IR must preserve structured Type information rather than reducing types to arbitrary strings.

The recovered hierarchy includes:

```text
TensorType
OptionalType
ListType
ScalarType
AnyType
NumberType
IntType
FloatType
BoolType
StringType
SymIntType
MemoryFormatType
DimnameType
Const
BaseType
```

## 4.2 Tensor-like

The IR must support the semantic predicate:

```text
type.is_tensor_like()
```

This predicate is used by generated out-variant construction.

## 4.3 SymInt

`SymInt` is a distinct semantic type.

It must not be collapsed into ordinary `int`.

The generator also has a `has_symint()` semantic check because generated kernel names may receive a `_symint` suffix.

## 4.4 Optional / List

Nested type structure such as:

```text
Optional<T>
List<T>
```

is part of the Type model.

## 4.5 What can be deferred?

The frozen materials do **not** define a safe reduced Type subset for Core IR v0.

Therefore:

```text
"these Type nodes may safely be omitted"
    = UNKNOWN / NOT FROZEN
```

---

# 5. Argument

## 5.1 Confirmed semantic fields

```text
Argument:
    name
    type
    default
    annotation
    is_write
```

### `name`

Argument identifier, e.g.:

```text
self
other
alpha
out
```

### `type`

Structured Type.

### `default`

Default value is part of the Argument semantic data.

Example:

```text
Scalar alpha=1
```

must preserve:

```text
name = alpha
type = Scalar
default = 1
```

### `annotation`

Alias/mutation annotation.

This is semantic and is used by schema classification and schema transformations.

### `is_write`

Confirmed field used for write/out semantics.

## 5.2 Keyword-only

`Arguments` structurally distinguishes positional and keyword arguments, but the recovered materials do not freeze a separate `Argument.keyword_only` field or its exact canonical representation.

Therefore:

```text
explicit Argument.keyword_only field = UNKNOWN
```

Do not invent one merely for convenience.

## 5.3 Default and signature

Confirmed:

```text
Argument.default must be preserved in the IR.
```

Not confirmed:

```text
whether default participates in FunctionSchema.signature()
```

Therefore:

```text
signature treatment of defaults = UNKNOWN
```

---

# 6. Return

## 6.1 Confirmed fields

```text
Return:
    name
    type
    annotation
```

### `type`

Structured Type.

### `name`

Return name.

### `annotation`

Alias annotation.

Return alias information is semantically relevant to generated out variants.

## 6.2 Return mutability

The recovered frozen materials do not establish a separate:

```text
Return.is_write
Return.mutable
```

field.

Therefore:

```text
independent Return mutability field = UNKNOWN
```

Do not add one without a later frozen decision.

---

# 7. FunctionSchema

Confirmed structure:

```text
FunctionSchema:
    name: OperatorName
    arguments: Arguments
    returns: sequence[Return]
```

It also exposes:

```text
kind()
is_out_fn()
signature()
```

as semantic operations.

---

# 8. FunctionSchema.signature()

This is the most important Core IR semantic function.

## 8.1 Confirmed role

`FunctionSchema.signature()` is the canonical grouping key used by:

```text
pre_group_native_functions()
```

Conceptually:

```text
NativeFunction
    |
    +-- func.signature()
             |
             v
       canonical group key
```

Operators with different schema kinds can therefore belong to the same conceptual group when their normalized signatures match.

The grouping key is **not simply `BaseOperatorName`**.

## 8.2 Confirmed normalization

The recovered materials explicitly establish that `signature()`:

1. strips alias annotations;
2. normalizes mutable arguments toward the corresponding functional signature;
3. erases schema-kind differences sufficiently for functional/out/inplace/mutable variants to group together.

In particular:

```text
functional
inplace
mutable
out
```

are not supposed to remain distinguishable merely because their mutation/out representation differs.

## 8.3 `keep_return_names=True`

The API must support:

```text
signature()
signature(keep_return_names=True)
```

Generated functional variants explicitly use:

```text
func.signature(keep_return_names=True)
```

Therefore `keep_return_names=True` is part of the required Core IR v0 API.

Confirmed meaning:

```text
keep_return_names=True
    -> preserve return names during signature construction
```

## 8.4 What is NOT frozen

The recovered materials do **not** provide a complete field-by-field canonical representation equivalent to:

```text
signature =
    (
        operator_name,
        (
            ...
        ),
        (
            ...
        )
    )
```

The following details therefore remain unresolved:

```text
- whether operator name is included in the canonical signature object
- exact treatment of overload_name
- exact treatment of argument names
- exact treatment of keyword-only markers
- whether defaults participate
- exact canonical encoding of Type
- exact canonical encoding of aliases after stripping
- exact default behavior of return names
- exact return tuple representation
```

These must **not** be guessed.

---

# 9. Equality / Hashing

## Confirmed

`FunctionSchema.signature()` is used as the grouping key.

Therefore the resulting canonical representation must be:

```text
deterministic
comparable
hashable
```

for use as a map key.

## Not frozen

The recovered materials do not explicitly freeze whether SonicCross should implement this as:

```text
record equality + hash
```

or:

```text
canonical string
```

or another immutable structural key.

The semantic requirement is only that equal canonical signatures compare equal and can be used as grouping keys.

---

# 10. FROZEN / CONFIRMED

The following are confirmed and may be implemented directly.

### OperatorName

```text
OperatorName =
    BaseOperatorName
    + overload_name

BaseOperatorName =
    base
    inplace
    dunder_method
    functional_overload
```

### Parsing

```text
"." separates base/operator name and overload_name.
```

### Inplace

```text
base ending in "_"
    -> inplace = true
```

### Functional overload

```text
generated mutable -> functional variant
    -> functional_overload = true
```

### SchemaKind

```text
functional
inplace
mutable
out
```

with the core distinction:

```text
out arguments        -> out
mutable self         -> inplace
mutable non-self     -> mutable
none of the above    -> functional
```

### `is_out_fn()`

```text
true iff explicit out arguments exist
```

### ViewSchemaKind

```text
view         -> aliasing
inplace_view -> aliasing_inplace
view_copy    -> non_aliasing
```

### Type

Structured Type IR is required.

At minimum the recovered semantic hierarchy includes:

```text
Tensor
Optional
List
Scalar
Any
Number
Int
Float
Bool
String
SymInt
MemoryFormat
Dimname
Const
BaseType
```

### Argument

```text
name
type
default
annotation
is_write
```

### Return

```text
name
type
annotation
```

### FunctionSchema

```text
name
arguments
returns
```

### `signature()`

Confirmed:

```text
canonical grouping key
strips alias annotations
normalizes schema-kind differences
normalizes mutable representation toward functional form
```

### `signature(keep_return_names=True)`

Confirmed API and semantics:

```text
preserve return names
```

---

# 11. UNKNOWN / NOT FROZEN

The following must remain explicitly unresolved until a later targeted audit.

| Item | Status |
|---|---|
| Exact `dunder_method` parsing predicate | UNKNOWN |
| Complete display/canonical rules for `functional_overload` | UNKNOWN |
| Exhaustive `SchemaKind.kind()` precedence for every unusual combination | UNKNOWN |
| Whether return alias annotations independently affect `kind()` | UNKNOWN |
| Whether `Argument.default` participates in `signature()` | UNKNOWN |
| Whether argument names participate in `signature()` | UNKNOWN |
| Exact keyword-only encoding in `signature()` | UNKNOWN |
| Whether Return needs an independent mutability field | UNKNOWN |
| Default `keep_return_names=False` return-name representation | UNKNOWN |
| Exact field-by-field representation of `FunctionSchema.signature()` | UNKNOWN |
| Exact operator-name/overload encoding inside `signature()` | UNKNOWN |
| Exact Type subset that may safely be deferred | UNKNOWN |
| Concrete implementation form of the canonical signature key | UNKNOWN |

---

# 12. Implementation Guardrail

Until the remaining UNKNOWN items are separately frozen:

**Codex must not invent semantics.**

In particular, do not:

```text
- infer dunder rules from general Python conventions;
- decide whether defaults belong in signatures;
- decide how argument names are encoded;
- invent a string representation for signatures;
- add return mutability fields;
- reduce Type to a smaller ad-hoc subset;
- replace structural canonicalization with a hand-written heuristic.
```

The implementation should preserve the confirmed semantic surface and leave unresolved points isolated behind explicit TODO/UNFROZEN markers rather than silently choosing behavior.

---

## Status

```text
Core IR v0 data model:
    MOSTLY FROZEN

OperatorName:
    FROZEN except exact dunder predicate

SchemaKind:
    CORE SEMANTICS FROZEN
    exhaustive edge-case matrix not frozen

ViewSchemaKind:
    FROZEN

Type:
    semantic hierarchy confirmed
    safe reduced v0 subset not frozen

Argument:
    core fields frozen
    keyword-only field/canonicalization not frozen

Return:
    core fields frozen
    independent mutability not frozen

FunctionSchema:
    structure frozen

FunctionSchema.signature():
    PURPOSE AND HIGH-LEVEL NORMALIZATION FROZEN
    EXACT CANONICAL REPRESENTATION NOT YET FROZEN
```

**Important:** `FunctionSchema.signature()` is the remaining critical specification gap because it is the identity function used by NativeFunction grouping.
