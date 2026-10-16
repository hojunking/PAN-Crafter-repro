"""Registered, paired 24h Stage 1 search. Import has no training/cloud effects."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from donor30.plan import (canonical_sha256, stream_seeds, selection_key,
    best_hqnr_key, select_donor, _record_valid, admission_decision,
    RETENTION_POLICY, LOSS_MASK_POLICY, SEED_PROTOCOL)
from directshift48.plan import alignment_protocol_digest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "PANDA_WV3_DONOR_S13_24H_20261016_v1"
METHOD_REVISION = "DirectShift-Scharr-MASK2-DonorSearch24-v1"
LEGACY_CAMPAIGNS = {}
HORIZON = 50000
LANES = {s: {"dataset": "WV3", "hqnr_min": .959} for s in ("s1", "s3")}
SEEDS = {"R1": 261016101, "R2": 261016102}
# Common A0/A1 isolate server effects. Other rows explore complementary axes.
RECIPES = {
    "A0": (3., .003, 1e-5, "MASK2 DF07 shared anchor"),
    "A1": (2., .003, 1e-5, "weaker absolute edge; shared bridge"),
    "A2": (4., .003, 1e-5, "stronger absolute edge local bracket"),
    "A3": (3., .0015, 1e-5, "half epsilon pressure"),
    "A4": (3., .006, 1e-5, "double epsilon pressure"),
    "A5": (2., .0015, 1e-5, "weaker absolute x epsilon interaction"),
    "B1": (1., .0003, 1e-5, "masked DF01 RR-friendly reference"),
    "B2": (.3, .001, 1e-5, "masked DF04 low-absolute reference"),
    "B3": (1.5, .001, 1e-5, "intermediate absolute/epsilon compromise"),
    "B4": (3., .003, 5e-6, "half aligner LR; same DF07 objective"),
}
QUEUES = {"s1": ["A0", "A1", "A3", "A2", "A4", "A5"],
          "s3": ["A0", "A1", "B1", "B2", "B3", "B4"]}
BUDGET = {"total_seconds": 86400, "closeout_seconds": 3600, "safety_factor": 1.2}
BASELINE_PATH = ROOT / "config/directshift48/s1.baseline.json"
# Independent pin: the old s3 GF2 baseline is deliberately NOT read or changed.
BASELINE_SHA = "58cff0453e97748ad8016ea960c9783726a8a261debff91cb713306b277e43d4"


def evaluation_steps():
    # Densify only the donor-relevant 4K..20K window; retain original H50 LR.
    return tuple(sorted(set([1000, 25000, *range(2000, 50001, 2000),
                             *range(4000, 20001, 1000)])))


def baseline(server):
    import hashlib
    if server not in LANES:
        raise ValueError("Only s1/WV3 and s3/WV3 are registered")
    raw = BASELINE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError("Registered source baseline changed")
    b = json.loads(raw)
    b.update(server=server, dataset="WV3")
    return b


def default_config(server="s1"):
    return {"campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
        "server": server, "dataset": "WV3", "baseline_resolved": baseline(server),
        "loss_mask_policy": LOSS_MASK_POLICY, "budget": dict(BUDGET),
        "search_order": list(QUEUES[server]), "confirmation": "LOCAL_WINNER_R2",
        "donor": {"hqnr_min": .959, "allow_exploratory_fallback": True},
        "stage2": "DISABLED_SHARED_DONOR_NOT_YET_SELECTED"}


def validate_config(config):
    if config != default_config(config.get("server")):
        raise ValueError("Unregistered campaign settings; create a new plan version")


def load_config(server):
    if server not in LANES:
        raise ValueError("Only s1/WV3 and s3/WV3 are registered")
    cfg = json.loads((ROOT/'config/donor24'/f'{server}.json').read_text())
    validate_config(cfg)
    return cfg


def short_run(case, replica="R1"):
    a, e, lr, _ = RECIPES[case]
    return f"{case} a{a:g} eps{e:g} lrA{lr:g} R3 M2 {replica}"


def resolve_case(config, case, replica="R1", attempt=1):
    validate_config(config)
    if case not in config["search_order"] or replica not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError("Unknown recipe/replica/attempt")
    a, eps, lr, question = RECIPES[case]
    seed, b = SEEDS[replica], config["baseline_resolved"]
    run = f"D24_WV3_{config['server'].upper()}_{case}_a{a:g}_eps{eps:g}_lrA{lr:g}_R3_M2_H50_{replica}_S{seed}"
    return {"campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
        "server": config["server"], "dataset": "WV3", "phase": "donor",
        "case": case, "loss_case": case, "replica": replica, "attempt": attempt,
        "run_id": run, "run_display": short_run(case, replica), "question": question,
        "seed": seed, "stream_seeds": stream_seeds(seed), "stream_seed_protocol": SEED_PROTOCOL,
        "baseline_resolved": deepcopy(b), "loss_mask_policy": LOSS_MASK_POLICY,
        "stage2": False, "initialization": "FRESH", "donor_checkpoint": None,
        "schedule": "H50", "planned_horizon": HORIZON, "evaluation_steps": list(evaluation_steps()),
        "lambda_rec": 1., "lambda_abs": a, "lambda_epsilon": eps, "lambda_eps_RR": eps,
        "epsilon_radius_rr": 3., "epsilon_distribution": "UNIFORM_AREA_DISK",
        "epsilon_every_update": True, "epsilon_reduction": "RR_SAMPLE_MEAN",
        "lr_U": 1e-4, "lr_A": lr, "weight_decay": .01,
        "retention_policy": RETENTION_POLICY, "hqnr_min": .959, "ergas_max": 2.040,
        "alignment_protocol_digest": alignment_protocol_digest(b),
        "requires_fr_train": False, "requires_source_scene_map": False,
        "pruning": "DISABLED_COMPLETE_ADMITTED_H50", "config_digest": canonical_sha256(config)}


def validate_case(cfg, config=None):
    config = config or default_config(cfg.get("server"))
    expected = resolve_case(config, cfg.get("case"), cfg.get("replica"), cfg.get("attempt"))
    actual = {k: v for k, v in cfg.items() if k != "disposable_preflight"}
    if actual != expected or ("disposable_preflight" in cfg and cfg["disposable_preflight"] is not True):
        raise ValueError("Resolved run differs from registered recipe")


def target_gap(h, e, *, server):
    # Reuse identical WV3 official target math, not the legacy s3 GF2 threshold.
    from directshift48.plan import target_gap as legacy
    if server not in LANES:
        raise ValueError("Unknown WV3 lane")
    return legacy(h, e, server="s1")


def choose_completed(observations, runs, *, replica=None):
    completed = {(r['run_id'], r['attempt']) for r in runs if r['status'] == 'COMPLETED'}
    valid = [r for r in observations if (r['run_id'], r['attempt']) in completed
             and (replica is None or r.get('replica') == replica)]
    return select_donor(valid, .959, allow_fallback=True)


def catalog_csv():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['server','dataset','order','case','replica','seed','lambda_abs',
                     'lambda_epsilon','lr_A','lr_T','radius','horizon','mask','run_display','question'])
    for server, cases in QUEUES.items():
        for i, case in enumerate(cases, 1):
            a,e,lr,q = RECIPES[case]
            writer.writerow([server,'WV3',i,case,'R1',SEEDS['R1'],a,e,lr,1e-4,3,HORIZON,
                             'MASK2',short_run(case),q])
        writer.writerow([server,'WV3',7,'LOCAL_WINNER','R2',SEEDS['R2'],
                         'inherit','inherit','inherit',1e-4,3,HORIZON,'MASK2',
                         'selected recipe R2','reserve confirmation before admitting search'])
    return out.getvalue()
