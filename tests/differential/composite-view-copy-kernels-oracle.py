#!/usr/bin/env python3
"""Render CompositeViewCopyKernels.cpp (G11 structured composite kernels +
G12 view_copy kernels) from frozen torchgen.

Mirrors the frozen `gen.py main()` with the default flag set, but only
materialises the single CompositeViewCopyKernels.cpp artifact.  The file is
written into the output directory given as argv[1], exactly as it would land
on disk.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"

EXPECTED_FILES = {"CompositeViewCopyKernels.cpp"}


def fail(message):
    raise SystemExit("composite-view-copy-kernels oracle: " + message)


if len(sys.argv) != 2:
    fail("usage: composite-view-copy-kernels-oracle.py OUTDIR")

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

_STATE = {"files": {}}

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
        if filename.as_posix() in EXPECTED_FILES:
            _STATE["files"][filename.as_posix()] = self.substitute_with_template(
                template_fn=template_fn, env_callable=env_callable
            )


utils.FileManager = RecordingFileManager

import torchgen.gen as gen  # noqa: E402

source_path = str(root_path / "aten" / "src" / "ATen")
install_dir = "/tmp/sc-composite-view-copy-kernels-install"
aoti_install_dir = "/tmp/sc-composite-view-copy-kernels-aoti-install"

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

missing = EXPECTED_FILES - set(_STATE["files"])
if missing:
    fail("missing generated files: %s" % sorted(missing))

outdir.mkdir(parents=True, exist_ok=True)
for name in sorted(_STATE["files"]):
    (outdir / name).write_text(_STATE["files"][name], encoding="utf-8")
