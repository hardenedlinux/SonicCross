#!/usr/bin/env python3
"""Render MethodOperators.h (G4) from the frozen torchgen orchestration.

Mirrors the frozen `gen.py main()` with the default flag set, but only
materialises the MethodOperators.h artifact.  The rendered text is written to
stdout, exactly as it would land on disk.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"


def fail(message):
    raise SystemExit("method-operators oracle: " + message)


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

LABELS = ["core", "cpu", "cpu_vec", "cuda", "ops", "aoti", "headeronly"]
_STATE = {"label_idx": 0, "content": None}

_OriginalFileManager = utils.FileManager


class RecordingFileManager(_OriginalFileManager):
    def __init__(self, install_dir, template_dir, dry_run):
        super().__init__(install_dir, template_dir, dry_run)
        self.sc_label = LABELS[_STATE["label_idx"]]
        _STATE["label_idx"] += 1

    def write_with_template(self, filename, template_fn, env_callable):
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        file = self.install_dir / filename
        if file in self.files:
            raise AssertionError(f"duplicate file write {file}")
        self.files.add(file)
        if filename.as_posix() == "MethodOperators.h":
            _STATE["content"] = self.substitute_with_template(
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
install_dir = "/tmp/sc-method-operators-install"
aoti_install_dir = "/tmp/sc-method-operators-aoti-install"

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

if _STATE["content"] is None:
    fail("MethodOperators.h was not generated")

sys.stdout.write(_STATE["content"])
