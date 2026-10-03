# SonicCross Generation Configuration (frozen v0)

Recovered verbatim from `torchgen/gen.py` `main()` at the frozen commit
`41ffbc4a994e058af9fe00ed5caba73fc1033359` (PyTorch 2.14.1a0), **default
invocation** (no CLI flags beyond `--generate` defaults).

This is the *generation scope*, not the *semantic IR*. It is derived only from
torchgen's own default configuration path; no backend is hand-removed.

## 1. Default parameters

| Parameter | Default | Notes |
|---|---|---|
| `per_operator_headers` | `False` | aggregated headers path |
| `skip_dispatcher_op_registration` | `False` | |
| `force_schema_registration` | `False` | |
| `rocm` | `False` | CUDA path, not HIP |
| `mps` / `xpu` / `mtia` | `False` | device keys filtered out (below) |
| `backend_whitelist` | `None` | no `is_generic_dispatch_key` filtering |
| `static_dispatch_backend` | `None` | `static_dispatch_idx = []` |
| `op_registration_whitelist` / `op_selection_yaml_path` | `None` | `selector = SelectiveBuilder.get_nop_selector()` (select all) |
| `generate` | `["headers", "sources", "declarations_yaml"]` | |
| `update_aoti_c_shim` / `extend_aoti_c_shim` | `False` | AOTI shim out of scope |

## 2. Dispatch keys

`torchgen/model.py` `dispatch_keys` (full, 30) is mutated in `main()` by
removing, in place, the keys gated by `--mps` / `--xpu` / `--mtia` (all
default-off):

- `MPS_KEYS = {MPS, SparseMPS, SparseCsrMPS}`
- `XPU_KEYS = {XPU, SparseXPU, SparseCsrXPU, NestedTensorXPU}`
- `MTIA`

**Default dispatch_keys (22, in order):**

```
CPU, SparseCPU, SparseCsrCPU, MkldnnCPU, CUDA, SparseCUDA, SparseCsrCUDA,
QuantizedCPU, QuantizedCUDA, CompositeImplicitAutograd,
CompositeImplicitAutogradNestedTensor, CompositeExplicitAutograd,
CompositeExplicitAutogradNonFunctional, NestedTensorCPU, NestedTensorCUDA,
NestedTensorHPU, Meta, SparseMeta, SparseCsrMeta, QuantizedMeta,
NestedTensorMeta, ZeroTensor
```

Note `NestedTensorHPU` is **not** removed by any flag (it is not gated), and so
remains in the default scope.

## 3. functions_keys

`functions_keys = {CPU, CUDA, CompositeImplicitAutograd,
CompositeImplicitAutogradNestedTensor, CompositeExplicitAutograd,
CompositeExplicitAutogradNonFunctional, Meta, MTIA}`.

`{Key}Functions.h` / `{Key}Functions_inl.h` are emitted for
`dispatch_keys ∩ functions_keys` (MTIA drops out because it is not in
`dispatch_keys`):

```
CPU, CUDA, CompositeImplicitAutograd,
CompositeImplicitAutogradNestedTensor, CompositeExplicitAutograd,
CompositeExplicitAutogradNonFunctional, Meta
```

## 4. File managers & install dirs

| manager | install dir (default) |
|---|---|
| `cpu_fm`, `cpu_vec_fm`, `cuda_fm` | `options.install_dir` (`build/aten/src/ATen`) |
| `core_fm` | `{install_dir}/core` |
| `ops_fm` | `{install_dir}/ops` |
| `headeronly_fm` | `{install_dir}/core` (default) |
| `device_fms` | `{"cuda": cuda_fm}` (no xpu) |

Dispatch-key → manager: `is_cuda_dispatch_key` (`{CUDA, QuantizedCUDA,
SparseCUDA, SparseCsrCUDA, NestedTensorCUDA, AutogradCUDA}`) → `cuda_fm`;
otherwise `cpu_fm`.

## 5. Sharding (`FileManager.write_sharded*`)

`num_shards`: `Register{Key}.cpp` = 4 if `CPU` else 1; `Operators.cpp` = 5;
`RegisterFunctionalization.cpp` = 4. Shard assignment is
`string_stable_hash(root_name) % num_shards`. The `Everything` shard is written
to disk but discarded from the `files` set (the compile manifest).

## 6. Artifact manifest (default, aggregated; AOTI + vmap excluded)

**Headers** (`gen_headers` → `gen_aggregated_headers`):

- `core_fm`: `TensorBody.h`, `aten_interned_strings.h`
- `headeronly_fm`: `enum_tag.h`
- `cpu_fm`: `NativeMetaFunctions.h`, `MethodOperators.h`, `Operators.h`,
  `Functions.h`, `NativeFunctions.h`, `RedispatchFunctions.h`,
  `RegistrationDeclarations.h`
- per key ∈ (functions_keys ∩ dispatch_keys): `{Key}Functions.h`,
  `{Key}Functions_inl.h`

**Sources** (`gen_source_files`):

- per dispatch key (22): `Register{Key}.cpp` (sharded)
- per structured ufunc (`out.ufunc_inner_loop` && key ∈ `{CPU, CUDA}`):
  `UfuncCPU_{name}.cpp`, `UfuncCPUKernel_{name}.cpp` (CPU);
  `UfuncCUDA_{name}.cu` (CUDA)
- `cpu_fm`: `RegisterBackendSelect.cpp`, `RegisterSchema.cpp`,
  `Operators.cpp` (sharded×5), `Functions.cpp` (static),
  `RegisterFunctionalization.cpp` (sharded×4), `FunctionalInverses.h`,
  `ViewMetaClasses.h`, `ViewMetaClasses.cpp`, `CompositeViewCopyKernels.cpp`
- `core_fm`: `TensorMethods.cpp` (static), `ATenOpList.cpp`

**Declarations** (`gen_declarations_yaml`):

- `cpu_fm`: `Declarations.yaml`

**Excluded (out of scope):** AOTI C shim (`aoti_fm`), `VmapGeneratedPlumbing.h`
(vmap), per-operator headers (`--per-operator-headers`), Python bindings,
`gen_backend_stubs.py`.

## 7. Generation closure → differential record groups

The 21 groups (G1–G21) from the plan map 1:1 onto the manifest above. The
differential gate is the semantic record set per file (see the plan's §3), not
raw byte equality.
