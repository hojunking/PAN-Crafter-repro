"""Measured selected-step times from existing logs, never ETA extrapolation.

The training trace is an append-only source. A separate integrity-checked cache
stores cumulative observed seconds and offsets; it never changes the trace.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from directshift48 import publication as direct
from sheet_publish_v2.selection import CAMPAIGN_ID as OLD_CAMPAIGN


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("TIMING_SECONDS_NEGATIVE_NONFINITE_OR_INVALID")
    return value


def _directory(config, root, history_roots):
    root = Path(root).resolve(strict=True)
    if history_roots is None:
        old = direct.ROOT / "work_dir/edgeres48" / OLD_CAMPAIGN / config["server"] / config["dataset"]
        history_roots = [old] if old.is_dir() else []
    if isinstance(history_roots, (str, Path)):
        history_roots = [history_roots]
    matches = []
    for source in [root, *[Path(p).resolve(strict=True) for p in history_roots]]:
        path = (source / "runs" / config["run_id"] / ("attempt_%02d" % config["attempt"])).resolve()
        if source not in path.parents:
            raise ValueError("TIMING_PATH_ESCAPE")
        if (path / "config.json").is_file():
            if json.loads((path / "config.json").read_bytes()) != config:
                raise ValueError("TIMING_CONFIG_FILE_MISMATCH")
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("TIMING_RUN_SOURCE_MISSING_OR_AMBIGUOUS")
    return matches[0]


def _atomic(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".timing-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _train_seconds(path, step, cache_dir, config_sha):
    cache_path = cache_dir / (_sha([str(path), config_sha]) + ".json")
    with _lock(cache_path.with_suffix(".lock")), path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        fingerprint = {"device": stat.st_dev, "inode": stat.st_ino,
                       "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        first = stream.readline(stat.st_size)
        head = hashlib.sha256(first).hexdigest()
        cached = None
        try:
            envelope = json.loads(cache_path.read_bytes())
            candidate = envelope["payload"]
            old = candidate["fingerprint"]
            same_identity = all(old[k] == fingerprint[k] for k in ("device", "inode"))
            append_or_unchanged = (stat.st_size > old["size"] or
                (stat.st_size == old["size"] and all(old[k] == fingerprint[k] for k in ("mtime_ns", "ctime_ns"))))
            if (envelope["payload_sha256"] == _sha(candidate) and same_identity and append_or_unchanged
                    and candidate["config_sha256"] == config_sha and candidate["head_sha256"] == head
                    and candidate["offset"] <= stat.st_size
                    and len(candidate["prefix_hashes"]) == len(candidate["prefix_seconds"])):
                stream.seek(candidate["last_line_offset"])
                tail = stream.read(candidate["offset"] - candidate["last_line_offset"])
                if hashlib.sha256(tail).hexdigest() == candidate["last_line_sha256"]:
                    cached = candidate
        except (OSError, ValueError, TypeError, KeyError):
            pass
        payload = cached or {"config_sha256": config_sha, "prefix_seconds": [0.], "offset": 0,
            "last_line_offset": 0, "last_line_sha256": hashlib.sha256(b"").hexdigest(),
            "head_sha256": head, "prefix_hashes": [hashlib.sha256(b"").hexdigest()]}
        prefix = payload["prefix_seconds"]
        stream.seek(payload["offset"])
        error = None
        while len(prefix) <= step:
            offset = stream.tell()
            if offset >= stat.st_size:
                error = "TRAIN_TRACE_NOT_YET_AT_SELECTED_STEP"; break
            line = stream.readline(stat.st_size - offset)
            if not line.endswith(b"\n"):
                error = "TRAIN_TRACE_PARTIAL_TAIL_BEFORE_SELECTED_STEP"; break
            try:
                row = json.loads(line)
                expected = len(prefix)
                if type(row.get("step")) is not int or row["step"] != expected or row.get("update", expected) != expected:
                    raise ValueError("TRAIN_TRACE_NONCONTIGUOUS_STEPS")
                seconds = _seconds(row["seconds_including_IO"])
                prefix.append(math.fsum((prefix[-1], seconds)))
                payload["prefix_hashes"].append(hashlib.sha256((payload["prefix_hashes"][-1]
                    + hashlib.sha256(line).hexdigest()).encode()).hexdigest())
                payload.update(offset=stream.tell(), last_line_offset=offset,
                    last_line_sha256=hashlib.sha256(line).hexdigest())
            except (ValueError, TypeError, KeyError) as exc:
                error = "TRAIN_TRACE_INVALID:" + str(exc); break
        payload["fingerprint"] = fingerprint
        _atomic(cache_path, {"payload": payload, "payload_sha256": _sha(payload)})
        if error:
            raise ValueError(error)
        return prefix[step], {"path": str(path), "selected_step": step, "contiguous_updates": step,
            "selected_prefix_row_hash_chain": payload["prefix_hashes"][step],
            "hash_chain_rule": "h0=SHA256(empty); hi=SHA256(hex(h[i-1])+hex(SHA256(raw JSONL row i)))",
            "cache_assumption": "append-only trace; inode/size/timestamps and first/last cached row checked; same-size rewrites invalidate",
            "scope": "sum synchronized update seconds including batch IO; excludes logging, evaluation, probes, initialization"}


def timing_for_config(config, selected_step, root, history_roots=None, cost_root=None):
    """Only return timing columns backed by complete observed source intervals.

    Wall(h) is intentionally omitted: update/evaluation durations exclude real
    overhead and cannot be added to invent wall time. No GPU-hour estimate.
    """
    if type(selected_step) is not int or selected_step <= 0:
        raise ValueError("POSITIVE_INTEGER_SELECTED_STEP_REQUIRED")
    directory = _directory(config, root, history_roots)
    cost_root = Path(cost_root) if cost_root is not None else direct.ROOT / "work_dir/experiment_cost" / config["server"]
    fields, proof, omitted = {}, {}, {"Wall(h)": "No authenticated per-run start/end interval joined; Train+Eval is not total wall time."}
    try:
        seconds, source = _train_seconds(directory / "train.jsonl", selected_step, cost_root / "timing_cache", _sha(config))
        fields["Train(h)"] = seconds / 3600
        proof["Train(h)"] = source
    except (OSError, ValueError, TypeError, KeyError) as exc:
        omitted["Train(h)"] = str(exc)
    try:
        # Include every formal evaluation actually present up to the chosen step,
        # and require every planned cadence step in that interval.
        planned = {s for s in config.get("evaluation_steps", []) if s <= selected_step}
        planned.add(selected_step)
        files, evaluated, seconds = [], set(), []
        for path in sorted(directory.glob("evaluation_*.json")):
            number = int(path.stem.split("_")[-1])
            if number > selected_step:
                continue
            raw = path.read_bytes(); value = json.loads(raw)
            ident = value["state_before"]
            if (ident["run_id"] != config["run_id"] or ident["attempt"] != config["attempt"] or
                    ident["completed_update"] != number or value.get("record_type") != "EVAL_SUMMARY"
                    or value.get("full_method_eligible") is not True
                    or value.get("measurement_domain") != "OFFICIAL_FULL20_NATIVE"
                    or any(value.get(k) != ident for k in ("rr_state", "fr_state", "state_after"))):
                raise ValueError("EVALUATION_TIMING_IDENTITY_OR_DOMAIN_MISMATCH")
            direct._check_complete(value["qualification"])
            for split in ("rr", "fr"):
                ids = [r["scene_id"] for r in value[split]["per_scene"]]
                if len(ids) != 20 or set(ids) != {f"{config['dataset']}:{split}:{i}" for i in range(20)}:
                    raise ValueError("EVALUATION_TIMING_SCENES_INCOMPLETE")
            seconds.append(_seconds(value["elapsed_seconds"]))
            evaluated.add(number)
            files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "step": number})
        if not planned <= evaluated:
            raise ValueError("EVALUATION_TIMING_PLANNED_STEPS_MISSING:" + str(sorted(planned - evaluated)))
        fields["Eval(h)"] = math.fsum(seconds) / 3600
        proof["Eval(h)"] = {"files": files, "completed_evaluations": len(files),
            "scope": "sum measured full20 RR+full20 FR evaluator elapsed_seconds through selected step; excludes probes/artifact export"}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        omitted["Eval(h)"] = str(exc)
    return {"sheet_fields": fields, "scope": "MEASURED_CUMULATIVE_THROUGH_SELECTED_STEP_NOT_FULL_RUN_ETA",
        "proof": proof, "omitted": omitted}
