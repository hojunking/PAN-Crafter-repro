"""All joint passes plus a minimum-ERGAS or highest-HQNR representative.

The local publication module validates immutable schedule-campaign evidence.
Only this campaign's rows are owned; earlier campaigns are never migrated.
Stage1 RRE selection and all raw evaluations/checkpoints remain unchanged.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from panda_schedule30 import publication as legacy
from panda_schedule30 import reporting as R

INDEX = R.INDEX
POLICY = "PANDA_SCHEDULE30_RRE_STAGE2_H9585_PASS_ALL_OR_BEST_FALLBACK_V1"
HQNR_MIN, ERGAS_MAX = .9585, 2.040
SELECTOR = "HQNR_QUALIFIED_MIN_ERGAS"
FALLBACK_SELECTOR = "BEST_HQNR_FALLBACK"
PASS_SELECTOR = "PASS_ALL_STEPS"
REPRESENTATIVE_SLOT = "REPRESENTATIVE"
RANKING = "HQNR_GE_0.9585_THEN_ERGAS_ASC_HQNR_DESC_STEP_ASC"
FALLBACK_RANKING = "HQNR_DESC_ERGAS_ASC_STEP_ASC"


def project(root, server, *, require_cost=False):
    # Reuse complete same-state RR/FR, source, deployment and artifact checks.
    value = legacy.project_campaign(root, server)
    value["legacy_policy"] = value["policy"]
    value["policy"] = POLICY
    value["publication_thresholds"] = {"hqnr_min": HQNR_MIN, "ergas_max": ERGAS_MAX,
        "inclusive": True, "checkpoint_selection": RANKING,
        "publish_all_joint_passes": True, "representative_requires_ergas_max": False,
        "deduplicate_representative_and_pass": True,
        "below_hqnr_fallback": FALLBACK_RANKING, "scope": "STAGE2_PER_RUN_SEED_ATTEMPT"}
    if value["conflicts"]:
        return value
    stage1 = [(r, n, s) for r, n, s in zip(
        value["rows"], value["notes"], value["selected"])
        if r[INDEX["Selector"]] == "BEST_VAL_RRE"]
    value["rows"] = [r for r, _, _ in stage1]
    value["notes"] = [n for _, n, _ in stage1]
    value["selected"] = [s for _, _, s in stage1]
    groups = defaultdict(list)
    for source in value["source_evidence"].values():
        note = json.loads(source["row"][INDEX["Original description / notes"]])
        if note["extra_parameters"]["configuration"]["phase"] == "stage2":
            groups[tuple(source["candidate"]["key"])].append(source)
    runs = {tuple(r["run_key"]): r for r in value["runs"]}
    for key, sources in sorted(groups.items()):
        # Publication thresholds override historical config annotations only.
        # Always use original full-precision, same-checkpoint RR/FR metrics.
        qualified = [s for s in sources if s["candidate"]["hqnr"] >= HQNR_MIN]
        passed = [s for s in qualified if s["candidate"]["ergas"] <= ERGAS_MAX]
        run = runs[key]
        run.update(pass_count=len(passed), hqnr_qualified_count=len(qualified),
                   kept_count=0, selection="NO_QUALIFYING_CHECKPOINT",
                   hqnr_min=HQNR_MIN, ergas_max=ERGAS_MAX)
        representative = min(qualified, key=lambda s: (
            s["candidate"]["ergas"], -s["candidate"]["hqnr"],
            s["candidate"]["step"], s["candidate"]["source_result_id"])) if qualified else min(
                sources, key=lambda s: (-s["candidate"]["hqnr"], s["candidate"]["ergas"],
                    s["candidate"]["step"], s["candidate"]["source_result_id"]))
        representative_selector = SELECTOR if qualified else FALLBACK_SELECTOR
        representative_ranking = RANKING if qualified else FALLBACK_RANKING
        # If any joint pass exists, the minimum-ERGAS representative is one of
        # them. Publish that checkpoint once, with its representative role.
        kept = [representative] + sorted(
            (s for s in passed if s is not representative),
            key=lambda s: (s["candidate"]["step"], s["candidate"]["source_result_id"]))
        run.update(kept_count=len(kept), selection="PASS_ALL_WITH_REPRESENTATIVE" if passed else representative_selector,
                   representative_source_result_id=representative["candidate"]["source_result_id"])
        for source in kept:
            candidate = source["candidate"]
            is_representative = source is representative
            joint_pass = candidate["hqnr"] >= HQNR_MIN and candidate["ergas"] <= ERGAS_MAX
            selector = representative_selector if is_representative else PASS_SELECTOR
            row = list(source["row"])
            note = json.loads(row[INDEX["Original description / notes"]])
            extra = note["extra_parameters"]
            cfg = extra["configuration"]
            try:
                artifact = legacy._artifact({
                    **candidate, "record": {"source": {"extra": extra}}}, root)
            except (ValueError, KeyError, OSError) as exc:
                value["conflicts"].append({
                    "source_result_id": candidate["source_result_id"], "reason": str(exc)})
                continue
            publication = {
                "policy": POLICY, "selection": selector,
                "checkpoint_selection": representative_ranking if is_representative else PASS_SELECTOR,
                "candidate_count": len(sources), "passed_count": run["pass_count"],
                "hqnr_qualified_count": len(qualified),
                "hqnr_min": HQNR_MIN, "ergas_max": ERGAS_MAX,
                "publication_joint_pass": joint_pass, "is_case_representative": is_representative,
                "representative_requires_ergas_max": False,
                "publish_all_joint_passes": True,
                "target_thresholds_are_annotations_only": False,
                "below_hqnr_fallback": FALLBACK_RANKING,
                "fallback_used": selector == FALLBACK_SELECTOR,
                "final_horizon": candidate["step"] == cfg["planned_horizon"],
                "planned_horizon": cfg["planned_horizon"],
                "inherited_updates": cfg.get("inherited_updates", 0),
                "locally_completed_updates": candidate["step"] - cfg.get("inherited_updates", 0),
                "separate_final_row": False, "fixed_step": None,
                "source_result_id": candidate["source_result_id"],
                "evaluation_step": candidate["step"],
                "evaluation_evidence": source["evaluation_evidence"], "artifact": artifact,
                "last_logged_status": run["last_logged_status"],
                "last_observed_at": run["last_observed_at"], "live_process_verified": False,
            }
            row[INDEX["Result_ID"]] = legacy.display_id(
                POLICY, key, candidate["protocol"], REPRESENTATIVE_SLOT if is_representative else candidate["step"])
            row[INDEX["Selector"]] = selector
            label = "대표 · HQNR 최고 · HQNR 미달" if selector == FALLBACK_SELECTOR else (
                "대표 · ERGAS 최소" if is_representative else "조건 충족")
            if is_representative and selector != FALLBACK_SELECTOR:
                label += " · 조건 충족" if joint_pass else " · ERGAS 미달"
            row[INDEX["Selection@Step"]] = f"{label} · {candidate['step']:,}"
            row[INDEX["Checkpoint_SHA"]] = artifact.get("file_sha256") or ""
            row[INDEX["Readback"]] = "VERIFIED"
            row[INDEX["Status"]] = run["last_logged_status"]
            note["publication"] = publication
            note["publication_policy"] = POLICY
            note["checkpoint_selection"] = publication["checkpoint_selection"]
            row[INDEX["Original description / notes"]] = R.canonical_json(note)
            short = R.canonical_json({"configuration": cfg, "publication": publication})
            if max(len(row[INDEX["Original description / notes"]]), len(short)) >= 45000:
                raise ValueError("Qualified publication note exceeds safe cell size")
            value["rows"].append(row)
            value["notes"].append(short)
            value["selected"].append(candidate)
    from panda_schedule30.cost import attach_cost
    attach_cost(value, Path(root), require_cost=require_cost)
    value["view_hash"] = R.digest({"rows": value["rows"], "notes": value["notes"]})
    return value


def verify_owned(row, value):
    """Authenticate current own rows; never adopt another campaign's policies."""
    publication = json.loads(row[INDEX["Original description / notes"]])["publication"]
    policy = publication.get("policy")
    if policy != POLICY:
        return legacy._verify_owned(row, value)
    selector = publication.get("selection")
    allowed = (SELECTOR, FALLBACK_SELECTOR, PASS_SELECTOR)
    source = value["source_evidence"].get(publication.get("source_result_id"))
    if source is None or selector not in allowed or row[INDEX["Selector"]] != selector:
        raise ValueError("Unknown representative row/source; never erase it")
    candidate = source["candidate"]
    slot = candidate["step"] if selector == PASS_SELECTOR else REPRESENTATIVE_SLOT
    if row[INDEX["Result_ID"]] != legacy.display_id(
            policy, candidate["key"], candidate["protocol"], slot):
        raise ValueError("Representative display identity mismatch")
    fallback = selector == FALLBACK_SELECTOR
    invalid_hqnr = candidate["hqnr"] >= HQNR_MIN if fallback else candidate["hqnr"] < HQNR_MIN
    if invalid_hqnr or (selector == PASS_SELECTOR and candidate["ergas"] > ERGAS_MAX):
        raise ValueError("Qualified row does not meet publication thresholds")
    raw = json.loads(source["row"][INDEX["Original description / notes"]])
    if raw["extra_parameters"]["configuration"]["phase"] != "stage2":
        raise ValueError("Stage2 representative row required")
    # Normalize only a temporary migration envelope to reuse the strict legacy
    # identity/metric/evaluation-proof validator. No source or cloud data changes.
    checked = deepcopy(row)
    note = json.loads(checked[INDEX["Original description / notes"]])
    note["publication"].update(policy=legacy.POLICY, selection="BEST_HQNR_FALLBACK")
    checked[INDEX["Original description / notes"]] = R.canonical_json(note)
    checked[INDEX["Result_ID"]] = legacy.display_id(
        legacy.POLICY, candidate["key"], candidate["protocol"], "FALLBACK")
    legacy._verify_owned(checked, value)


def row_requests(snapshot, value, server):
    """Migrate own rows in place; backup/clear surplus, never shift other rows."""
    from sheet_publish_v2.transport import _typed, _matrix_equal
    if value["conflicts"]:
        raise ValueError("Local evidence conflicts; cloud untouched")
    own, keys = {}, {tuple(k) for k in value["owned_run_keys"]}
    for number, row in enumerate(snapshot["rows"], 5):
        if row[INDEX["Campaign_ID"]] != R.CAMPAIGN_ID or row[INDEX["Server"]] != server:
            continue
        if (row[INDEX["Dataset"]] != "WV3"
                or not str(row[INDEX["Run_ID"]]).startswith("PSC30_")
                or legacy._key(row) not in keys):
            raise ValueError("Unknown own run; never erase it")
        verify_owned(row, value)
        rid = row[INDEX["Result_ID"]]
        if rid in own:
            raise ValueError("Duplicate owned display ID; no overwrite")
        own[rid] = (number, row, snapshot["notes"][number - 5])
    expected = {r[INDEX["Result_ID"]]: (r, n)
                for r, n in zip(value["rows"], value["notes"])}
    if len(expected) != len(value["rows"]) or len(value["rows"]) != len(value["notes"]):
        raise ValueError("Duplicate desired IDs or mismatched notes")
    stale = {rid: item for rid, item in own.items() if rid not in expected}

    def cells(row, note):
        return {"values": [{"userEnteredValue": _typed(v), "note": note if j == 0 else ""}
                           for j, v in enumerate(row)]}

    def update(number, payload):
        return {"updateCells": {"range": {
            "sheetId": value["destination"]["sheet_id"], "startRowIndex": number,
            "endRowIndex": number + 1, "startColumnIndex": 0, "endColumnIndex": 64},
            "rows": [payload], "fields": "userEnteredValue,note"}}

    requests, additions = [], []
    for rid, (row, note) in expected.items():
        existing = own.get(rid)
        if existing is None:
            reusable = next((key for key, item in stale.items()
                             if legacy._key(item[1]) == legacy._key(row)), None)
            if reusable is None:
                additions.append(cells(row, note))
                continue
            existing = stale.pop(reusable)
        number, old, old_note = existing
        if not _matrix_equal([old], [row]) or old_note != note:
            requests.append(update(number, cells(row, note)))
    for number, _, _ in stale.values():
        requests.append(update(number, {"values": [{} for _ in range(64)]}))
    if additions:
        requests.append({"appendCells": {
            "sheetId": value["destination"]["sheet_id"],
            "rows": additions, "fields": "userEnteredValue,note"}})
    return requests

