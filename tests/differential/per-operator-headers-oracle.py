#!/usr/bin/env python3
"""Render the per-operator header closure from frozen torchgen (--per-operator-headers).

Mirrors the frozen `gen.py main()` with `--per-operator-headers` set, but only
materialises the header artifacts that SonicCross's per-operator mode emits:

  * every `ATen/ops/{name}[_.*].h` per-operator header (the ops file manager),
  * the five include-only aggregate shims
    (Functions.h / Operators.h / NativeMetaFunctions.h / NativeFunctions.h /
    MethodOperators.h), and
  * the fourteen `{DispatchKey}Functions.h` / `_inl.h` headers.

Every other FileManager write is recorded-but-not-rendered.  Each captured file
is written into the output directory named by argv[1] under its basename (which
is unique across the three capture sets), exactly as it would land on disk.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"

FUNCTIONS_KEYS = (
    "CPU",
    "CUDA",
    "CompositeImplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "Meta",
)

SHIM_NAMES = {
    "Functions.h",
    "Operators.h",
    "NativeMetaFunctions.h",
    "NativeFunctions.h",
    "MethodOperators.h",
}

DISPATCH_HEADER_NAMES = {f"{key}Functions.h" for key in FUNCTIONS_KEYS} | {
    f"{key}Functions_inl.h" for key in FUNCTIONS_KEYS
}


def fail(message):
    raise SystemExit("per-operator-headers oracle: " + message)


if len(sys.argv) != 2:
    fail("usage: per-operator-headers-oracle.py OUTDIR")

outdir = Path(sys.argv[1])

root = os.environ.get("SONICCROSS_PYTORCH_ROOT")
if not root:
    fail("set SONICCROSS_PYTORCH_ROOT to the frozen checkout")
root_path = Path(root)
if not (root_path / "torchgen" / "gen.py").is_file():
    fail("checkout does not contain torchgen/gen.py: " + root)
try:
    commit = subprocess.check_output(
        ["git", "-C", str(root_path), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception as exc:
    fail("cannot verify frozen checkout commit: %s" % exc)
if commit != EXPECTED:
    fail("checkout HEAD is %s, expected %s" % (commit, EXPECTED))

sys.path.insert(0, str(root_path))
os.chdir(root_path)

import __main__  # noqa: E402

__main__.__file__ = str(root_path / "torchgen" / "gen.py")

import torchgen.utils as utils  # noqa: E402

_STATE = {"files": {}, "ops_count": 0}

_OriginalFileManager = utils.FileManager


class RecordingFileManager(_OriginalFileManager):
    def write_with_template(self, filename, template_fn, env_callable):
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        file = self.install_dir / filename
        if file in self.files:
            raise AssertionError(f"duplicate file write {file}")
        self.files.add(file)
        name = filename.name
        if (
            self.install_dir.name == "ops"
            or name in SHIM_NAMES
            or name in DISPATCH_HEADER_NAMES
        ):
            if self.install_dir.name == "ops":
                _STATE["ops_count"] += 1
            _STATE["files"][name] = self.substitute_with_template(
                template_fn=template_fn, env_callable=env_callable
            )

    def write_sharded_with_template(
        self,
        filename,
        template_fn,
        items,
        *,
        key_fn,
        env_callable,
        num_shards,
        base_env=None,
        sharded_keys=None,
    ):
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")


utils.FileManager = RecordingFileManager

import torchgen.gen as gen  # noqa: E402

source_path = str(root_path / "aten" / "src" / "ATen")
install_dir = "/tmp/sc-per-operator-headers-install"
aoti_install_dir = "/tmp/sc-per-operator-headers-aoti-install"

sys.argv = [
    "gen.py",
    "-s",
    source_path,
    "-d",
    install_dir,
    "--aoti-install-dir",
    aoti_install_dir,
    "--per-operator-headers",
]

_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    gen.main()
finally:
    sys.stdout = _stdout

if _STATE["ops_count"] == 0:
    fail("no ops/*.h files were captured (per-operator mode not engaged?)")

outdir.mkdir(parents=True, exist_ok=True)
for name in sorted(_STATE["files"]):
    (outdir / name).write_text(_STATE["files"][name], encoding="utf-8")

print(
    "per-operator-headers oracle: %d files (%d ops, %d shim+dispatch)"
    % (
        len(_STATE["files"]),
        _STATE["ops_count"],
        len(_STATE["files"]) - _STATE["ops_count"],
    )
)
