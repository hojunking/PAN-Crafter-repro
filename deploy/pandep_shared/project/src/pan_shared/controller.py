"""Single-writer indefinite fair-block controller; no legacy queues or leases."""
from __future__ import annotations

import fcntl
import gc
import os
from pathlib import Path
import signal

from .common import atomic_json, read_json, timestamp


class Controller:
    def __init__(self, work_root, registry, runner_factory, admission, *, stage_report=None):
        self.root = Path(work_root).resolve()
        self.registry = registry
        self.runs = {r["run_id"]: r for r in registry["runs"]}
        self.runner_factory = runner_factory
        self.admission = admission
        self.stage_report = stage_report
        self.path = self.root / "controller_state.json"
        self.state = read_json(self.path) if self.path.exists() else {
            "schema": "PANDEP_CONTROLLER_STATE_v1", "campaign_id": registry["campaign_id"],
            "registry_sha256": registry["registry_sha256"], "status": "READY", "extension_cursor": 0,
            "progress": {name: 0 for name in self.runs}, "reported_stages": [], "current_task": None,
        }
        if self.state["registry_sha256"] != registry["registry_sha256"]:
            raise ValueError("BLOCKED_IDENTITY: controller registry changed")
        self.interrupted = False

    def _persist(self):
        self.state["updated_at"] = timestamp()
        atomic_json(self.path, self.state)

    def control(self):
        if self.interrupted:
            return "STOP"
        path = self.root / "control.json"
        if not path.exists():
            raise ValueError("control.json is required; no implicit RUN")
        value = read_json(path)
        if value.get("campaign_id") != self.registry["campaign_id"]:
            raise ValueError("control campaign identity mismatch")
        action = value.get("action")
        if action not in ("RUN", "PAUSE", "STOP_AFTER_BLOCK", "STOP"):
            raise ValueError("unknown control action")
        return action

    def _ordered_runs(self):
        return [next(r for r in self.registry["runs"] if r["repeat"] == repeat and r["case_id"] == case)
                for repeat in self.registry["repeat_order"] for case in self.registry["extension_case_order"]]

    def completed_stage(self):
        return min(self.state["progress"][r["run_id"]] // r["exposure_stage_updates"]
                   for r in self.registry["runs"])

    def next_task(self):
        # A block is not complete merely because its optimizer endpoint was
        # committed. Failed/pending validation, benchmark or publication must
        # finish before queue advancement, even when progress == target_step.
        if self.state.get("current_task") is not None:
            return dict(self.state["current_task"])
        progress = self.state["progress"]
        for task in self.registry["initial_queue"]:
            if progress[task["run_id"]] < task["target_step"]:
                return dict(task)
        stage = self.completed_stage() + 1
        order = self._ordered_runs()
        for offset in range(len(order)):
            cursor = (self.state["extension_cursor"] + offset) % len(order)
            run = order[cursor]; current = progress[run["run_id"]]
            if current < stage * run["exposure_stage_updates"]:
                next_boundary = (current // 50000 + 1) * 50000
                return {"run_id": run["run_id"], "target_step": next_boundary,
                        "exposure_stage": stage, "extension_cursor_after": (cursor + 1) % len(order)}
        raise RuntimeError("queue invariant violated")

    def run(self, *, until_operator_stop=False, max_blocks=None):
        if not until_operator_stop:
            raise ValueError("explicit --until-operator-stop is required")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / "controller.lock"
        if not lock_path.resolve().is_relative_to(self.root):
            raise ValueError("BLOCKED_PATH: controller lock escapes campaign")
        lock = lock_path.open("a+")
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("WRITER_CONFLICT: controller already active") from error
            lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + "\n"); lock.flush()
            if self.state["status"] in ("STOPPED_BY_OPERATOR", "PAUSED_SAFE", "PAUSED_DISK", "DIVERGED", "PAUSED_OOM", "BLOCKED_EXECUTION"):
                # A fresh RUN text alone is insufficient; safety.resume explicitly
                # authorizes and changes the persisted state before restarting.
                raise RuntimeError("explicit resume authorization required for " + self.state["status"])
            handlers = {}
            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, lambda *_: setattr(self, "interrupted", True))
                blocks = 0
                while max_blocks is None or blocks < max_blocks:
                    action = self.control()
                    if action in ("PAUSE", "STOP", "STOP_AFTER_BLOCK"):
                        self.state["status"] = "PAUSED_SAFE" if action == "PAUSE" else "STOPPED_BY_OPERATOR"
                        self._persist(); return self.state
                    task = self.next_task()
                    try:
                        admitted = self.admission(task)
                    except Exception as error:
                        self.state["status"] = str(error).split(":", 1)[0] or "BLOCKED_EXECUTION"
                        self.state["admission_error"] = str(error); self._persist(); raise
                    if isinstance(admitted, dict) and admitted.get("status") not in (None, "READY", "PASS", "ADMITTED", "RESOURCE_AVAILABLE"):
                        self.state["status"] = admitted["status"]
                        self.state["admission"] = admitted; self._persist(); return self.state
                    self.state["status"] = "RUNNING"; self.state["current_task"] = task; self._persist()
                    try:
                        runner = self.runner_factory(self.runs[task["run_id"]])
                    except Exception as error:
                        self.state["status"] = "BLOCKED_EXECUTION"
                        self.state["runner_error"] = str(error); self._persist(); raise
                    try:
                        result = runner.run_until(task["target_step"], self.control)
                        self.state["progress"][task["run_id"]] = runner.step
                        self.state["status"] = result
                        if result != "BLOCK_COMPLETE":
                            self._persist(); return self.state
                        if runner.step != task["target_step"]:
                            raise RuntimeError("block completed at wrong optimizer step")
                    except Exception:
                        self.state["progress"][task["run_id"]] = runner.step
                        self.state["status"] = runner.state.get("status", "BLOCKED_EXECUTION")
                        self._persist(); raise
                    finally:
                        del runner
                        gc.collect()
                    if "extension_cursor_after" in task:
                        self.state["extension_cursor"] = task["extension_cursor_after"]
                    self.state["current_task"] = None
                    stage = self.completed_stage()
                    if stage and stage not in self.state["reported_stages"] and self.stage_report:
                        self.stage_report(stage, self.state)
                        self.state["reported_stages"].append(stage)
                    blocks += 1; self._persist()
                return self.state
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        finally:
            lock.close()
