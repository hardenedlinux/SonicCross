#!/usr/bin/env python3
"""Run the frozen torchgen orchestration and emit the Tier 0 artifact manifest.

This mirrors the frozen `gen.py main()` with the default flag set (no
--mps/--xpu/--mtia/--per-operator-headers/--rocm), intercepting every
FileManager write call so that no template content is rendered.  The output is
the canonical artifact manifest:

    <label>\t<filename>\t<ops>

where <label> is one of core/cpu/cpu_vec/cuda/ops/aoti/headeronly, <filename> is
the relative output filename (sharded files carry their _N suffix), and <ops> is
either "-" (unsharded artifact) or a space-separated, sorted multiset of
root-names assigned to that shard.
"""

import os
import subprocess
import sys
from pathlib import Path

EXPECTED = "41ffbc4a994e058af9fe00ed5caba73fc1033359"


def fail(message):
    raise SystemExit("manifest oracle: " + message)


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

import torchgen.utils as utils  # noqa: E402
from torchgen.utils import string_stable_hash  # noqa: E402

# The label each FileManager gets is determined by creation order inside main():
# core, cpu, cpu_vec, cuda, ops, aoti, headeronly.
LABELS = ["core", "cpu", "cpu_vec", "cuda", "ops", "aoti", "headeronly"]
_STATE = {"label_idx": 0, "records": []}

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
        _STATE["records"].append((self.sc_label, filename.as_posix(), None))

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
        keys = [key_fn(item) for item in items]
        _STATE["records"].append(
            (self.sc_label, filename.as_posix(), (num_shards, keys))
        )


utils.FileManager = RecordingFileManager

import torchgen.gen as gen  # noqa: E402

source_path = str(root_path / "aten" / "src" / "ATen")
install_dir = "/tmp/sc-manifest-install"
aoti_install_dir = "/tmp/sc-aoti-install"

sys.argv = [
    "gen.py",
    "-s",
    source_path,
    "-d",
    install_dir,
    "--aoti-install-dir",
    aoti_install_dir,
]

# gen.main() prints a couple of "not found" diagnostics to stdout when the
# aoti shim headers are absent; capture them so the manifest stays clean.
import io

_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    gen.main()
finally:
    sys.stdout = _stdout


def split_ext(filename):
    p = Path(filename)
    return p.stem, p.suffix


artifacts = []
for label, filename, shard in _STATE["records"]:
    if shard is None:
        artifacts.append((label, filename, "-"))
    else:
        num_shards, keys = shard
        membership = [[] for _ in range(num_shards)]
        for key in keys:
            sid = string_stable_hash(key) % num_shards
            membership[sid].append(key)
        stem, ext = split_ext(filename)
        for i in range(num_shards):
            ops = " ".join(sorted(membership[i]))
            artifacts.append((label, f"{stem}_{i}{ext}", ops))

# A sharded write produces exactly num_shards distinct files; an unsharded
# write produces one.  Guard against any accidental collision.
seen = set()
for label, filename, ops in artifacts:
    key = (label, filename)
    if key in seen:
        fail("duplicate artifact: %s %s" % key)
    seen.add(key)

for label, filename, ops in sorted(artifacts):
    sys.stdout.write("%s\t%s\t%s\n" % (label, filename, ops))
