#!/usr/bin/env python3
"""Render RegisterFunctionalization*.cpp from the frozen torchgen orchestration.

Mirrors the frozen `gen.py main()` with the default flag set, but only
materialises the RegisterFunctionalization.cpp artifacts (G13).  Every other
FileManager write is recorded-but-not-rendered.  Rendered files are written
under the directory named by SONICCROSS_G13_OUTPUT (default: a temp dir), and a
manifest (filename<TAB>byte-count) is printed to stdout.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"


def fail(message):
    raise SystemExit("register-functionalization oracle: " + message)


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

out_dir = os.environ.get("SONICCROSS_G13_OUTPUT")
if not out_dir:
    out_dir = "/tmp/sc-g13-output"
os.makedirs(out_dir, exist_ok=True)

sys.path.insert(0, str(root_path))
os.chdir(root_path)

import __main__  # noqa: E402

__main__.__file__ = str(root_path / "torchgen" / "gen.py")

import torchgen.utils as utils  # noqa: E402

_OriginalFileManager = utils.FileManager

_TARGET_TEMPLATE = "RegisterFunctionalization.cpp"


class RecordingFileManager(_OriginalFileManager):
    def write_with_template(self, filename, template_fn, env_callable):
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        file = self.install_dir / filename
        if file in self.files:
            raise AssertionError(f"duplicate file write {file}")
        self.files.add(file)
        if template_fn == _TARGET_TEMPLATE:
            content = self.substitute_with_template(
                template_fn=template_fn, env_callable=env_callable
            )
            out_path = Path(out_dir) / file.name
            out_path.write_text(content, encoding="utf-8")


utils.FileManager = RecordingFileManager

import torchgen.gen as gen  # noqa: E402

source_path = str(root_path / "aten" / "src" / "ATen")
install_dir = "/tmp/sc-register-functionalization-install"
aoti_install_dir = "/tmp/sc-register-functionalization-aoti-install"

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

for name in sorted(os.listdir(out_dir)):
    if name.startswith("RegisterFunctionalization") and name.endswith(".cpp"):
        p = Path(out_dir) / name
        print("%s\t%d" % (name, p.stat().st_size))
