#!/usr/bin/env python
"""Materialize a local MIX20H checkout without changing experiment definitions.

This is a one-command launch helper, not a queue activation command.  A config
from another checkout may differ only in its checkout-root-based work_dir and
four dataroot values.  Everything else must match the explicit generator.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

import yaml

from kdv import mix20h_plan as plan
from tools import gen_mix20h_configs as gen


DATA_FIELDS = ("train_feeder_args", "val_feeder_args",
               "test_reduced_feeder_args", "test_full_feeder_args")


@contextmanager
def _migration_lock(root):
    # Share prepare()'s lock so a config rewrite cannot race manifest publication.
    directory = root / plan.STATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".migration.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mix20-config-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_root_relocation(actual, expected, run_id, root):
    """Only accept the five generator-owned paths, all using ONE old root."""
    if not isinstance(actual, dict):
        return False
    work = Path(str(actual.get("work_dir", "")))
    if not work.is_absolute() or work.name != run_id or work.parent.name != "work_dir":
        return False
    old_root = work.parent.parent
    if old_root == root:
        return False
    relocated = copy.deepcopy(actual)
    relocated["work_dir"] = expected["work_dir"]
    for field in DATA_FIELDS:
        if not isinstance(actual.get(field), dict):
            return False
        suffix = Path(expected[field]["dataroot"]).relative_to(root / "data")
        if actual[field].get("dataroot") != str(old_root / "data" / suffix):
            return False
        relocated[field]["dataroot"] = expected[field]["dataroot"]
    return relocated == expected


def _inspect(root, server):
    cases = plan.cases_for(server)
    # Any explicit local case is enough to forbid rewriting this campaign's
    # input files, including a run belonging to a different server identity.
    frozen = ((root / plan.PLAN_MANIFEST).exists() or
              any((root / "work_dir" / case.run_id).exists() for case in plan.CASES))
    summary = dict(server=server, frozen=frozen, generated=[], rebased=[],
                   unchanged=[], backups=[], queue={})
    writes = []
    for case in cases:
        relative = Path("config") / (case.run_id + ".yaml")
        path = root / relative
        expected = gen.build_config(case.run_id, root=root)
        old = path.read_bytes() if path.exists() else None
        if old is None:
            if frozen:
                raise ValueError(f"MIX20H config is frozen; missing original config: {relative}")
            summary["generated"].append(str(relative))
            writes.append((path, gen.render_config(case.run_id, root=root).encode(), None, None))
            continue
        try:
            actual = yaml.safe_load(old)
        except yaml.YAMLError as exc:
            raise ValueError(f"MIX20H config cannot be parsed, preserved unchanged: {relative}") from exc
        if actual == expected:
            summary["unchanged"].append(str(relative))
            continue
        if not _is_root_relocation(actual, expected, case.run_id, root):
            raise ValueError(f"MIX20H config differs beyond checkout paths, preserved unchanged: {relative}")
        if frozen:
            raise ValueError(f"MIX20H config is frozen; cannot relocate existing campaign: {relative}")
        digest = hashlib.sha256(old).hexdigest()
        backup_relative = Path(plan.STATE_DIR) / "config_before" / (path.name + "." + digest + ".bak")
        backup = root / backup_relative
        if backup.exists() and backup.read_bytes() != old:
            raise ValueError(f"MIX20H backup collision: {backup_relative}")
        summary["rebased"].append(str(relative))
        summary["backups"].append(str(backup_relative))
        writes.append((path, gen.render_config(case.run_id, root=root).encode(), backup, old))
    relative = Path("config/queues") / f"qrc24_mix20h_{server}.txt"
    queue = root / relative
    if queue.exists():
        ids = [line.strip() for line in queue.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
        if ids != [case.run_id for case in cases]:
            raise ValueError(f"MIX20H static queue differs, preserved unchanged: {relative}")
        summary["queue"] = dict(path=str(relative), action="unchanged")
    else:
        summary["queue"] = dict(path=str(relative), action="generated")
        writes.append((queue, gen.queue_text(server).encode(), None, None))
    return summary, writes


def ensure_configs(root, server, dry_run=False):
    """Validate/materialize this server's configs, never activate or train.

    All files are checked before the first write.  A dry run creates no files,
    directories, or locks.  Existing semantically identical configs retain
    their exact bytes, so saved config SHA identities remain stable on restart.
    """
    root = Path(root).resolve()
    plan.cases_for(server)  # Validate identity before creating even a lock file.
    if dry_run:
        summary, _ = _inspect(root, server)
        return dict(summary, dry_run=True)
    with _migration_lock(root):
        summary, writes = _inspect(root, server)
        for path, data, backup, old in writes:
            if backup is not None and not backup.exists():
                _atomic_bytes(backup, old)
            _atomic_bytes(path, data)
    return dict(summary, dry_run=False)
