"""Read-only historical FH12 source reconstruction and isolated parity probes.

No historical metadata or source identity is rewritten. Only a temporary Git
archive is materialized; the original and current numerical readers run in
separate Python interpreters, avoiding module-cache contamination.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import torch

from fh12.data import canonical_sha, sha256_file

ATOL, RTOL = 3e-6, 2e-5


def validate_source_identity(source):
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("git_release", "")):
        raise ValueError("Historical source must name an exact Git commit")
    if not source.get("files") or canonical_sha(source["files"]) != source.get("content_sha256"):
        raise ValueError("Historical source content hash does not match its file manifest")
    for name, digest in source["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Unsafe historical source path/hash")


@contextmanager
def historical_snapshot(root, source):
    validate_source_identity(source)
    root = Path(root)
    revision = source["git_release"]
    # Archive these code packages from the pinned commit, not working-tree code.
    archive = subprocess.check_output(["git", "archive", "--format=tar", revision,
                                       "fh12", "model", "pa", "tools"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="fh20r1-origin-reader-") as name:
        target = Path(name)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                    raise ValueError("Unsafe path or link in historical Git archive")
                if member.isdir():
                    (target / path).mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest = target / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
        for path, expected in source["files"].items():
            if not (target / path).is_file() or sha256_file(target / path) != expected:
                raise ValueError(f"Pinned Git archive differs from historical source: {path}")
        yield target


def compare_probe_arrays(old, new):
    if set(old) != set(new):
        raise ValueError("Historical/current parity output keys differ")
    report = {}
    for key in sorted(old):
        a, b = np.asarray(old[key]), np.asarray(new[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid parity arrays: {key}")
        if not np.allclose(a, b, atol=ATOL, rtol=RTOL):
            raise ValueError(f"FH20R1 historical parity failed ({key}); tolerance is never widened")
        report[key] = {"shape": list(a.shape), "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.}
    return report


def run_parity(root, origin, resolved, data, device="cuda"):
    root = Path(root).resolve()
    request = {"config": resolved["teacher_config"], "checkpoint": resolved["teacher_checkpoint"],
               "splits": data["splits"], "device": str(device),
               "tau_R": origin["tau_R"], "q_ref": origin["q_ref"],
               "q_cache": resolved["q_cache_path"], "source_identity": origin["source_identity"],
               "probe_runtime_flags": {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
               "dlpan": os.environ.get("PANCRAFTER_DLPAN", str(root.parent / "DLPan-Toolbox"))}
    helper = Path(__file__).with_name("probe_reader.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # Never inherit caller PYTHONPATH into the historical interpreter.
    env.pop("PYTHONPATH", None)
    with historical_snapshot(root, origin["source_identity"]) as historical, \
         tempfile.TemporaryDirectory(prefix="fh20r1-parity-") as temporary:
        tmp = Path(temporary)
        req = tmp / "request.json"
        req.write_text(json.dumps(request))
        outputs, meta = {}, {}
        for label, source in (("historical", historical), ("current", root)):
            dest = tmp / f"{label}.npz"
            subprocess.run([sys.executable, "-I", str(helper), str(source), str(req), str(dest)],
                           cwd=temporary, env=env, check=True)
            with np.load(dest, allow_pickle=False) as values:
                outputs[label] = {key: values[key].copy() for key in values.files}
            meta[label] = json.loads(dest.with_suffix(".json").read_text())
        differences = compare_probe_arrays(outputs["historical"], outputs["current"])
        if meta["historical"]["step0_state_hash"] != meta["current"]["step0_state_hash"]:
            raise ValueError("FH20R1 current named fresh initialization differs from original reader")
        return {"status": "PASS", "atol": ATOL, "rtol": RTOL, "comparisons": differences,
                "sample_counts": {"train64": 8, "rr256": 2, "fr512": 2},
                "train_ids": list(range(8)), "rr_ids": [0, 1], "fr_ids": [0, 1],
                "readers": meta, "origin_git_release": origin["source_identity"]["git_release"],
                "origin_source_identity": origin["source_identity"],
                "runtime_policy": "Both readers use the actual current consumer FP32/TF32 settings; historical metadata is preserved separately",
                "loss_and_q_probe": "Teacher even/odd losses, Student LU/LA and output gradients, train8 x rot4 AXIS16",
                "original_metadata_modified": False, "source_override": False}
