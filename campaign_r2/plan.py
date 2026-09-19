"""R2 priorities for existing FH20R1 cases; never a new training recipe.

The companion priority JSON/CSV mentioned by the supplied MD were not supplied.
This implementation derives an explicit overlay from MD section 4 and Appendix
B, and verifies that Appendix B exactly matches the unchanged local registry.
Runtime planning is read-only. Artifact verification remains the controller's
responsibility: an unreported run is not automatically a fresh run.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from fh12.common import ROOT, object_sha, sha256
from fh20r1 import plan as base

PRIORITY_REVISION = "FH20R1_PRIORITY_METHOD_PRESERVE_R2_20260919"
SOURCE_PLAN = "results_log/PAN_FH20R1_R2_MethodPreserving_Adjustment_2026-09-19.md"
EXPECTED_PLAN_SHA256 = "3604fdf53bddeacdb602666d60f64cfdc036dfa22830ea25bbe6a4016a62dc10"
EXPECTED_REGISTRY_SHA256 = "fbbb49a2040e2953a362b80552afa00eb668454d8e5d0d90157637ffcb133bad"
CAMPAIGN_ID = "WV3_FH20R1_20260919_v1"
METHOD_REVISION = "FH12_SYNC_FREQ_NATIVE_TEACHER_v1"
REGISTRY_REVISION = "FH20R1_REGISTRY_20260919_v1"
EXPECTED_BRANCHES = {
    "s1": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE",
    "s2": "S2_DONOR_OR_REFERENCE_UNAVAILABLE",
    "s3": "STANDARD", "s4": "STANDARD", "s5": "STANDARD",
}
CORE_NUMBERS = {
    "s1": (1, 2, 3, 4, 5, 6, 7),
    "s2": (1, 2, 3, 4, 5),
    "s3": (1, 2, 6, 3, 4, 5, 7, 8, 9, 10),
    "s4": (1, 2, 6, 3, 7, 4, 5),
    "s5": (1, 2, 6, 3, 4, 5),
}
RESERVE_NUMBERS = {
    "s1": (8, 9, 10, 11), "s2": (6, 7, 8, 9),
    "s3": (11, 12, 13, 14), "s4": (8, 9), "s5": (7, 8),
}
DONE = frozenset({"DONE", "DONE_REUSED"})
UNADMITTED = frozenset({"", "PENDING", "QUEUED", "REGISTERED", "NOT_STARTED"})


def _server(server):
    if server not in EXPECTED_BRANCHES:
        raise ValueError(f"Unknown R2 server: {server}")
    return server


def priority_blocks(server):
    """All original Block objects in R2 preference order, CORE then reserve."""
    _server(server)
    lookup = {b.block_id: b for b in base.blocks_for(server)}
    ids = [f"FH20R1_{server.upper()}_B{i:02d}"
           for i in CORE_NUMBERS[server] + RESERVE_NUMBERS[server]]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError(f"R2 block membership/duplication mismatch: {server}")
    return tuple(lookup[key] for key in ids)


def require_branch_record(server, record):
    """Validate an already recorded branch, never infer/create/reselect it.

    Both historical ``branch`` and current ``condition`` keys are accepted, but
    disagreeing aliases, another server/campaign, and a changed branch fail.
    """
    _server(server)
    if not isinstance(record, Mapping) or not record:
        raise ValueError(f"Existing local branch_record.json required: {server}")
    values = [record[key] for key in ("condition", "branch") if key in record]
    if not values or any(value != values[0] for value in values):
        raise ValueError("Missing or conflicting branch/condition aliases")
    if values[0] != EXPECTED_BRANCHES[server]:
        raise ValueError(f"R2 cannot change/reselect recorded branch: {server}: {values[0]}")
    if record.get("server_id", server) != server or record.get("campaign_id", CAMPAIGN_ID) != CAMPAIGN_ID:
        raise ValueError("Foreign campaign/server branch record")
    return values[0]


def _appendix_blocks(text):
    """Read only Appendix B's explicit block headers and fenced run lists."""
    try:
        appendix = text.split("## 부록 B.", 1)[1]
    except IndexError as error:
        raise ValueError("R2 source MD lacks Appendix B") from error
    pattern = re.compile(
        r"^#### \d+\. (FH20R1_S[1-5]_B\d{2}) · (CORE|RESERVE)[^\n]*\n"
        r".*?^```text\n(.*?)^```", re.MULTILINE | re.DOTALL)
    rows = []
    for match in pattern.finditer(appendix):
        bid, tier, body = match.groups()
        runs = [line.strip() for line in body.splitlines() if line.strip()]
        if any(not run.startswith("FH20R1_") for run in runs):
            raise ValueError(f"Invalid Appendix B run list: {bid}")
        rows.append(dict(block_id=bid, tier=tier, run_ids=runs))
    return rows


def canonical_overlay(root=ROOT):
    """A machine-readable specification derived from the supplied MD only."""
    root = Path(root)
    path = root / SOURCE_PLAN
    if not path.is_file() or sha256(path) != EXPECTED_PLAN_SHA256:
        raise ValueError("R2 source MD missing or changed; review before application")
    servers = {}
    for server, condition in EXPECTED_BRANCHES.items():
        servers[server] = {
            "required_existing_branch": condition,
            "blocks": [dict(block_id=b.block_id, tier=b.tier,
                            run_ids=list(b.order_for(condition)))
                       for b in priority_blocks(server)],
        }
    actual = _appendix_blocks(path.read_text())
    expected = [row for server in servers.values() for row in server["blocks"]]
    if actual != expected:
        raise ValueError("MD Appendix B differs from registry membership/block-internal order")
    return dict(
        priority_revision=PRIORITY_REVISION, campaign_id=CAMPAIGN_ID,
        method_revision=METHOD_REVISION, registry_revision=REGISTRY_REVISION,
        registry_sha256=EXPECTED_REGISTRY_SHA256,
        provenance=dict(source_plan=SOURCE_PLAN, source_plan_sha256=EXPECTED_PLAN_SHA256,
                        derivation="MD section 4 and Appendix B; verified against existing fh20r1.plan",
                        companion_priority_json_csv_supplied=False),
        policy=dict(admitted_blocks_first=True, preserve_recorded_run_order=True,
                    preserve_local_advanced_state=True, reset_campaign_clock=False,
                    create_new_training_ids=False, change_numeric_config=False,
                    completed_results="verify and reuse, never retrain for upload"),
        servers=servers,
    )


def validate_definition(root=ROOT, *, require_configs=True):
    """Fail closed on any change to the original registry or numeric configs."""
    if (base.CAMPAIGN_ID, base.METHOD_REVISION, base.REGISTRY_REVISION) != (
            CAMPAIGN_ID, METHOD_REVISION, REGISTRY_REVISION):
        raise ValueError("Original campaign/method/registry revision changed")
    if base.registry_sha256() != EXPECTED_REGISTRY_SHA256:
        raise ValueError("Original registry definition changed")
    overlay = canonical_overlay(root)
    rows = [row for server in overlay["servers"].values() for row in server["blocks"]]
    ids = [run for row in rows for run in row["run_ids"]]
    active = [case.run_id for server, condition in EXPECTED_BRANCHES.items()
              for case in base.active_cases(server, condition)]
    core = sum(len(row["run_ids"]) for row in rows if row["tier"] == "CORE")
    reserve = sum(len(row["run_ids"]) for row in rows if row["tier"] == "RESERVE")
    if (len(base.CASES), core, reserve) != (145, 84, 40):
        raise ValueError("R2 must preserve 145 definitions / CORE 84 / RESERVE 40")
    if len(ids) != len(set(ids)) or set(ids) != set(active):
        raise ValueError("R2 run membership changed or contains duplicates")
    if any(base.case_for(run).role != "S" for run in ids):
        raise ValueError("Current R2 branch must contain existing Students only")
    config_hashes = {}
    if require_configs:
        import yaml
        for case in base.CASES:
            path = Path(root) / "config" / f"{case.run_id}.yaml"
            if not path.is_file():
                raise ValueError(f"Missing registered config: {path}")
            cfg = yaml.safe_load(path.read_text())
            if cfg != base.build_config(case):
                raise ValueError(f"Registered numeric config changed: {path}")
            config_hashes[case.run_id] = dict(file_sha256=sha256(path), config_sha256=object_sha(cfg))
    return dict(priority_revision=PRIORITY_REVISION, campaign_id=CAMPAIGN_ID,
                method_revision=METHOD_REVISION, registry_sha256=EXPECTED_REGISTRY_SHA256,
                source_plan_sha256=EXPECTED_PLAN_SHA256, definitions=145,
                core_students=core, reserve_students=reserve, new_training_ids=0,
                exact_membership=True, block_internal_orders_preserved=True,
                numeric_configs_checked=len(config_hashes), config_hashes=config_hashes,
                overlay_sha256=object_sha(overlay))


def _runtime(server, state, branch_record):
    """Validate state structure without treating missing Sheet rows as fresh."""
    condition = require_branch_record(server, branch_record)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Existing local campaign state required")
    if state.get("campaign_id", CAMPAIGN_ID) != CAMPAIGN_ID or state.get("server_id", server) != server:
        raise ValueError("Foreign runtime campaign/server")
    blocks, runs = state.get("blocks", {}), state.get("runs", {})
    if not isinstance(blocks, Mapping) or not isinstance(runs, Mapping):
        raise ValueError("Runtime blocks/runs must be keyed records")
    lookup = {b.block_id: b for b in priority_blocks(server)}
    if set(blocks) - set(lookup):
        raise ValueError(f"Unknown/inactive runtime blocks: {sorted(set(blocks) - set(lookup))}")
    active = {case.run_id for case in base.active_cases(server, condition)}
    if set(runs) - active:
        raise ValueError(f"Unknown/inactive runtime runs: {sorted(set(runs) - active)}")
    for run, row in runs.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid runtime run record: {run}")
    for bid, row in blocks.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid runtime block record: {bid}")
        expected = list(lookup[bid].order_for(condition))
        recorded = row.get("run_ids")
        if recorded is not None and (not isinstance(recorded, list) or recorded != expected):
            raise ValueError(f"Recorded block order/membership differs from immutable branch: {bid}")
        if row.get("tier", lookup[bid].tier) != lookup[bid].tier:
            raise ValueError(f"Recorded block tier changed: {bid}")
        if row.get("status") in DONE and any(
                runs.get(run, {}).get("status", "") not in DONE | UNADMITTED for run in expected):
            raise ValueError(f"Completed block contains unfinished active run: {bid}")
    return condition, blocks, runs, lookup


def planned_queue(server, state, branch_record):
    """Pending atomic blocks: all admitted work first, then new R2 priority.

    A block is admitted when its record says so, or any member has local state
    beyond an explicit pending marker. Such a block requires its original stored
    run order, and remains ahead even if the MD snapshot did not know it existed.
    Within the admitted group, an actually training block precedes other admitted
    blocks; otherwise the original block order is retained. The controller must
    corroborate ``DONE`` and local artifacts before calling its execution path.
    """
    condition, block_rows, run_rows, lookup = _runtime(server, state, branch_record)
    admitted, unstarted = [], []
    original_rank = {b.block_id: i for i, b in enumerate(base.blocks_for(server))}
    for block in priority_blocks(server):
        saved = block_rows.get(block.block_id, {})
        if saved.get("status") in DONE:
            continue
        expected = list(block.order_for(condition))
        observed = any(run in run_rows and (
            run_rows[run].get("status", "") not in UNADMITTED
            or run_rows[run].get("training_started")
            or run_rows[run].get("training_complete")) for run in expected)
        entered = (saved.get("status", "") not in UNADMITTED
                   or saved.get("admitted") is True or observed)
        if entered and "run_ids" not in saved:
            raise ValueError(f"Admitted block lacks recorded run_ids; reconcile first: {block.block_id}")
        order = list(saved["run_ids"] if entered else expected)
        done = [run for run in order if run_rows.get(run, {}).get("status") in DONE]
        pending = [run for run in order if run not in done]
        actions = {run: ("reuse_verified_complete" if run in done else
                         "resume_or_postrun_existing" if run in run_rows and (
                             run_rows[run].get("status", "") not in UNADMITTED
                             or run_rows[run].get("training_started")) else
                         "verify_local_artifacts_before_fresh") for run in order}
        row = dict(block_id=block.block_id, tier=block.tier, run_ids=order,
                   admitted=entered, completed_run_ids=done, pending_run_ids=pending,
                   run_actions=actions, priority_revision=PRIORITY_REVISION)
        if entered:
            # Parent status is not authoritative: original runner leaves it
            # REGISTERED during normal training. Child rows identify live work.
            active = any(run_rows.get(run, {}).get("status") in {"TRAINING", "POSTRUN", "EVALUATING"}
                         for run in order)
            admitted.append((not active, original_rank[block.block_id], row))
        else:
            unstarted.append(row)
    admitted.sort(key=lambda item: item[:2])
    return [row for _, _, row in admitted] + unstarted


def pending_blocks(server, state, branch_record):
    """Original Block objects corresponding exactly to :func:`planned_queue`."""
    lookup = {b.block_id: b for b in base.blocks_for(_server(server))}
    return tuple(lookup[row["block_id"]] for row in planned_queue(server, state, branch_record))
