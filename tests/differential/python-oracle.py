#!/usr/bin/env python3
"""Run the actual frozen torchgen implementation and emit canonical records."""

import inspect
import os
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"


def fail(message):
    raise SystemExit("python oracle: " + message)


root = os.environ.get("SONICCROSS_PYTORCH_ROOT")
if not root:
    fail("set SONICCROSS_PYTORCH_ROOT to the frozen checkout")
root_path = Path(root)
if not (root_path / "torchgen" / "model.py").is_file():
    fail("checkout does not contain torchgen/model.py: " + root)
import subprocess
try:
    commit = subprocess.check_output(
        ["git", "-C", str(root_path), "rev-parse", "HEAD"],
        text=True).strip()
except Exception as exc:
    fail("cannot verify frozen checkout commit: %s" % exc)
if commit != EXPECTED:
    fail("checkout HEAD is %s, expected %s" % (commit, EXPECTED))

sys.path.insert(0, str(root_path))
try:
    from torchgen.model import DispatchKey, Location, NativeFunction
    from torchgen.gen import parse_tags_yaml
    from torchgen.native_function_generation import add_generated_native_functions
except Exception as exc:
    fail("cannot import frozen torchgen: %s" % exc)

tags_path = root_path / "aten" / "src" / "ATen" / "native" / "tags.yaml"
if not tags_path.is_file():
    fail("checkout does not contain frozen tags.yaml: " + str(tags_path))
try:
    valid_tags = parse_tags_yaml(str(tags_path))
except Exception as exc:
    fail("cannot parse frozen tags.yaml: %s" % exc)


def invoke_from_yaml(item, location):
    method = NativeFunction.from_yaml
    try:
        value = method(item, Location(location, 1), valid_tags)
    except Exception as exc:
        func_text = item.get("func", "<unknown>")
        fail("NativeFunction.from_yaml adapter failed at %s (%s): %s" %
             (location, func_text, exc))
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, {}


def invoke_generation(functions):
    signature = inspect.signature(add_generated_native_functions)
    if len(signature.parameters) == 1:
        indices = {}
        result = add_generated_native_functions(functions)
    else:
        # The frozen implementation's second argument is its backend index
        # collection.  An empty collection is sufficient for this fixture;
        # generated metadata is returned/updated by the actual function.
        indices = {DispatchKey.CompositeExplicitAutograd: {}}
        result = add_generated_native_functions(functions, indices)
    if result is None:
        return functions, indices
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, indices


def text(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def kind_name(value):
    name = getattr(value, "name", None)
    if name is None:
        name = getattr(value, "value", str(value))
    return str(name).lower()


def metadata_lines(dispatch, operator_name=None):
    rows = []
    if hasattr(dispatch, "items"):
        for key, value in dispatch.items():
            key_text = getattr(key, "name", str(key))
            if hasattr(value, "items"):
                values = value.items()
            else:
                values = [(None, value)]
            for operator, metadata in values:
                if operator_name is not None and str(operator) != operator_name:
                    continue
                kernel = getattr(metadata, "kernel", None)
                structured = getattr(metadata, "structured", False)
                namespace = getattr(metadata, "cpp_namespace", None)
                supports = "_symint" in str(kernel)
                rows.append((key_text, text(kernel), structured, text(namespace), supports))
    return sorted(rows)


def emit(index, function, dispatch, generated):
    schema = function.func
    name = schema.name
    base = name.name
    tags = sorted(str(tag) for tag in function.tags)
    print("record-begin")
    print("index: %06d" % index)
    print("namespace: %s" % text(function.namespace))
    print("operator: %s" % text(name))
    print("base: %s" % text(base.base))
    print("overload: %s" % text(name.overload_name))
    print("schema: %s" % text(schema))
    print("schema-kind: %s" % kind_name(schema.kind()))
    print("generated: %s" % ("true" if generated else "false"))
    print("tags: %s" % (",".join(tags) if tags else "-"))
    print("dispatch-begin")
    for key, kernel, structured, namespace, supports in metadata_lines(
            dispatch, str(name)):
        print("dispatch: %s|%s|%s|%s|%s" %
              (key, kernel, text(structured), namespace,
               "true" if supports else "false"))
    print("dispatch-end")
    print("record-end")


if len(sys.argv) != 2:
    fail("usage: python-oracle.py FIXTURE")

try:
    import yaml
except Exception as exc:
    fail("PyYAML is required by the oracle: %s" % exc)

fixture = Path(sys.argv[1])
raw = yaml.safe_load(fixture.read_text())
if not isinstance(raw, list):
    fail("fixture root must be a sequence")

original = []
dispatch_by_id = {}
for index, item in enumerate(raw):
    function, dispatch = invoke_from_yaml(item, "%s:%d" % (fixture, index + 1))
    original.append(function)
    dispatch_by_id[id(function)] = dispatch

original_count = len(original)
generated_result, generated_indices = invoke_generation(original)
generated_ids = {id(function) for function in generated_result[original_count:]}
all_functions = list(generated_result)
for index, function in enumerate(all_functions):
    dispatch = dispatch_by_id.get(id(function), generated_indices)
    emit(index, function, dispatch, id(function) in generated_ids)
