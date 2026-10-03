#!/usr/bin/env python3
"""Render the G16-G20 core-source / operators artifacts from the frozen torchgen
orchestration, for byte-comparison against SonicCross.

Materialised artifacts (each written to OUTDIR under its own filename):

  G16  Operators.h, OperatorsEverything.cpp, Operators_0.._4.cpp
  G17  TensorBody.h
  G18  aten_interned_strings.h, enum_tag.h
  G19  TensorMethods.cpp, ATenOpList.cpp
  G20  RegisterBackendSelect.cpp

Only these files are intercepted; every other (often very large) generated
artifact is skipped by stubbing out non-Operators sharded writes.  Operators.cpp
is the sole sharded file we need, so its shards are materialised through the
parent FileManager sharding logic.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"

TARGETS = {
    "Operators.h",
    "OperatorsEverything.cpp",
    "Operators_0.cpp",
    "Operators_1.cpp",
    "Operators_2.cpp",
    "Operators_3.cpp",
    "Operators_4.cpp",
    "TensorBody.h",
    "aten_interned_strings.h",
    "enum_tag.h",
    "TensorMethods.cpp",
    "ATenOpList.cpp",
    "RegisterBackendSelect.cpp",
}


def fail(message):
    raise SystemExit("core-sources oracle: " + message)


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

if len(sys.argv) != 2:
    fail("usage: core-sources-oracle.py OUTDIR")
outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(root_path))
os.chdir(root_path)

import __main__  # noqa: E402

__main__.__file__ = str(root_path / "torchgen" / "gen.py")

import torchgen.utils as utils  # noqa: E402

_STATE = {"content": {}}

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
        if filename.as_posix() in TARGETS:
            _STATE["content"][filename.as_posix()] = self.substitute_with_template(
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
        if filename.as_posix() == "Operators.cpp":
            super().write_sharded_with_template(
                filename,
                template_fn,
                items,
                key_fn=key_fn,
                env_callable=env_callable,
                num_shards=num_shards,
                base_env=base_env,
                sharded_keys=sharded_keys,
            )


utils.FileManager = RecordingFileManager

import torchgen.gen as gen  # noqa: E402

source_path = str(root_path / "aten" / "src" / "ATen")
install_dir = "/tmp/sc-core-sources-install"
aoti_install_dir = "/tmp/sc-core-sources-aoti-install"

sys.argv = [
    "gen.py",
    "-s",
    source_path,
    "-d",
    install_dir,
    "--aoti-install-dir",
    aoti_install_dir,
]

_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    gen.main()
finally:
    sys.stdout = _stdout

for name in sorted(TARGETS):
    if name not in _STATE["content"]:
        fail(name + " was not generated")
    (outdir / name).write_text(_STATE["content"][name], encoding="utf-8")
