"""EdgeRes48H-v2 pure policy: RR-only per-input alignment, no prior assets.

Explicit operational proposals are pinned before official metrics. This module
does not launch training, inspect real images, persist weights or contact Sheets.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from itertools import combinations
import hashlib
import math
import numbers
import re
from typing import Any, Mapping, Sequence

CAMPAIGN_ID = "PANDA_S1_EDGERES_TEACHER48H_S123_20261013_v2"
PARENT_CAMPAIGN_ID = "PANDA_S1_M12_TEACHER48H_S123_20261013_v1"
METHOD_REVISION = "EdgeGuidedResidual-S1-v1"
METHOD_FILE_NAME = "PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md"
METHOD_SHA256 = "ac9e2cc7ef58defb03bf94b615faa525c911ead64471e9e7f6348a91d63d30d7"
PLAN_FILE_NAME = "PANDA_Stage1_EdgeGuided_TargetSearch_S123_48H_2026-10-13_v2.md"
PLAN_SHA256 = "4a606ace09746d957dbf850453b0c2b296ddb4532bcd3472939092dccdd4f1e3"
METHOD_CONTRACT = {"method_revision": METHOD_REVISION, "method_sha256": METHOD_SHA256,
                   "experiment_plan_sha256": PLAN_SHA256}
RETENTION_POLICY = "STRICT_TARGET_ONLY"
PROBE_SEED = 261013900
SEED_DERIVATION_PROTOCOL = "edgeres48.sha256.seed-label.v2"
STREAM_SEED_NAMESPACE = "edgeres48"
LANES = {
 "s1": {"dataset":"WV3","hqnr_min":.958,"ergas_max":2.040,"seeds":(261013101,261013102,261013103)},
 "s2": {"dataset":"QB","hqnr_min":.923,"ergas_max":3.500,"seeds":(261013201,261013202,261013203)},
 "s3": {"dataset":"GF2","hqnr_min":.965,"ergas_max":.522,"seeds":(261013301,261013302,261013303)},
}
LOSS_CASES = {
 "EF00":(.001,1.),"EF01":(.0003,1.),"EF02":(.0001,1.),"EF03":(.003,1.),
 "EF04":(.001,.3),"EF05":(.001,.1),"EF06":(.001,3.),"EF07":(.0003,.3),
 "EF08":(.0001,.3),"EF09":(.003,.3),"EF10":(.0003,.1),"EF11":(.0001,.1),
 "EF12":(.003,.1),"EF13":(.0003,3.),"EF14":(.0001,3.),"EF15":(.003,3.),
 "EF16":(.00001,.1),"EF17":(.00001,1.),"EF18":(.0001,.03),"EF19":(.001,.03),
}
CONDITIONAL_CASES = {"EF16":"EDGE_DOMINANT_EPS_NATIVE_CONFLICT",
 "EF17":"EDGE_DOMINANT_EPS_NATIVE_CONFLICT","EF18":"ABS_GRADIENT_QUALITY_TRADEOFF",
 "EF19":"ABS_GRADIENT_QUALITY_TRADEOFF"}
OPTIMIZER_CASES = {
 "ET00":(.1,1.,1.),"ET01":(.3,1.,1.),"ET02":(3.,1.,1.),
 "ET03":(1.,.5,1.),"ET04":(1.,1.5,1.),"ET05":(1.,1.,.1),
 "ET06":(1.,1.,0.),"ET07":(.3,1.5,1.),"ET08":(3.,.5,1.),
}
LOSS_ORDER = {
 "s1":("EF00","EF01","EF04","EF07","EF02","EF06","EF08","EF05","EF10","EF11","EF13","EF14","EF03","EF09","EF12","EF15"),
 "s2":("EF00","EF02","EF08","EF11","EF01","EF07","EF04","EF10","EF05","EF06","EF14","EF13","EF03","EF09","EF12","EF15"),
 "s3":("EF00","EF06","EF13","EF01","EF14","EF04","EF07","EF02","EF08","EF05","EF10","EF11","EF03","EF15","EF09","EF12"),
}
OPTIMIZER_ORDER = {"s1":("ET01","ET03"),"s2":("ET01","ET02"),"s3":("ET04","ET05")}
HORIZONS = {"H50":50000,"H100":100000,"H150":150000}
FR_METRICS = ("HQNR","D_s","D_lambda","JQM")
RR_METRICS = ("ERGAS","SCC","SAM","PSNR","SSIM","RMSE","CC")
IDENTITY_FIELDS = ("run_id","attempt","seed","completed_update","model_state_sha256",
 "inference_policy_digest","evaluator_revision","edge_protocol_digest", *METHOD_CONTRACT)
BASELINE_REQUIRED = (
 *METHOD_CONTRACT, "teacher_architecture","aligner_architecture","teacher_input_layout","teacher_base",
 "optimizer","optimizer_betas","optimizer_eps","lr_U0","lr_A0","weight_decay0","warmup_updates0",
 "min_lr","scheduler_family","scheduler_indexing","effective_batch_RR","microbatch_RR","accumulation_RR",
 "RR_train_HW","precision","amp","tf32","augmentation","edge_match","absolute_roi_margin",
 "boundary_policy","gradient_log_interval","pruning","protocol_decisions",
)
def _finite(v):
 return isinstance(v,numbers.Real) and not isinstance(v,bool) and math.isfinite(v)
def _integer(v,minimum=0):
 return isinstance(v,numbers.Integral) and not isinstance(v,bool) and v>=minimum
def _unresolved(v):
 if v is None: return True
 if isinstance(v,str): return not v.strip() or any(x in v.upper() for x in ('MUST_PIN','TODO','UNRESOLVED'))
 if isinstance(v,Mapping): return not v or any(_unresolved(x) for x in v.values())
 if isinstance(v,(tuple,list)): return not v or any(_unresolved(x) for x in v)
 return False
def canonical_sha256(v):
 import json
 return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def edge_protocol_digest(baseline):
 """Pin the executable descriptor + matcher manifest, not merely its options."""
 from panda_method.edge_match import EdgeMatchSpec, EdgeMatcher
 return canonical_sha256(EdgeMatcher(EdgeMatchSpec(**baseline['edge_match'])).manifest())

def validate_baseline(b):
 from panda_method.edge_match import EdgeMatchSpec
 errors = [k+" missing" for k in BASELINE_REQUIRED if k not in b or b[k] is None]
 for k,v in METHOD_CONTRACT.items():
  if b.get(k)!=v: errors.append(k+" differs from EdgeRes-v2")
 if b.get("requires_fr_train",False) is not False or b.get("requires_source_scene_map",False) is not False:
  errors.append("FR training/source-scene mapping forbidden")
 if any(k in b for k in ("effective_batch_FR","microbatch_FR","accumulation_FR","FR_train_HW")):
  errors.append("Legacy FR training fields forbidden")
 for k in ("lr_A0","lr_U0","optimizer_eps"):
  if not _finite(b.get(k)) or b[k]<=0: errors.append(k+" must be positive")
 for k in ("weight_decay0","min_lr"):
  if not _finite(b.get(k)) or b[k]<0: errors.append(k+" must be nonnegative")
 betas=b.get("optimizer_betas")
 if not isinstance(betas,(list,tuple)) or len(betas)!=2 or any(not _finite(v) or not 0<=v<1 for v in betas):
  errors.append("Explicit optimizer betas required")
 if b.get("optimizer")!="AdamW" or b.get("scheduler_family")!="warmup_cosine":
  errors.append("Pinned AdamW/warmup_cosine recipe required")
 if b.get("scheduler_indexing")!="ONE_BASED_WARMUP_LINEAR_PEAK_THEN_COSINE_EXACT_ENDPOINT_MIN":
  errors.append("Explicit scheduler indexing required")
 if not _integer(b.get("warmup_updates0")) or b.get("warmup_updates0",50000)>=50000:
  errors.append("Warmup must be in [0,50000)")
 keys=("effective_batch_RR","microbatch_RR","accumulation_RR")
 if any(not _integer(b.get(k),1) for k in keys) or b.get(keys[0])!=b.get(keys[1],0)*b.get(keys[2],0):
  errors.append("RR effective batch must equal microbatch*accumulation")
 if b.get("RR_train_HW")!=[64,64] or b.get("precision")!="fp32" or b.get("amp") is not False or b.get("tf32") is not False:
  errors.append("PAN64 FP32 AMP/TF32-off recipe required")
 if b.get("teacher_input_layout")!="P0" or b.get("teacher_base")!="BICUBIC_MS":
  errors.append("Unchanged P0/BICUBIC_MS Teacher required")
 arch=b.get("teacher_architecture",{})
 if (set(arch)!={"bands","width","depth"} or arch.get("bands") not in (4,8)
     or not _integer(arch.get("width"),1) or len(arch.get("depth",()))!=3
     or any(not _integer(d,1) for d in arch.get("depth",()))):
  errors.append("Explicit current Teacher architecture required")
 a=b.get("aligner_architecture",{})
 if a.get("name")!="ResidualPANAligner" or not _integer(a.get("view_margin")) or a.get("view_margin",64)>30:
  errors.append("Explicit Edge residual Aligner and nonempty view required")
 try:
  spec=EdgeMatchSpec(**b.get("edge_match",{}))
  if (spec.radius_rr!=spec.radius_fr or spec.radius_rr not in (4,6,8)
      or spec.support_policy!="fixed_radius_plus5_v2"
      or spec.tie_policy!="min_norm_then_l1_lexicographic"
      or spec.low_texture_policy!="report" or spec.boundary_policy!="report"):
   errors.append("v2 fixed-support deterministic EdgeMatch policy required")
  if spec.radius_rr!=4 and not b.get("edge_radius_train_only_evidence"):
   errors.append("Radius6/8 requires pre-official train-only decision evidence")
  if arch.get("bands")!=(8 if spec.sensor=="WV3" else 4): errors.append("Sensor/bands disagree")
 except (TypeError,ValueError) as exc: errors.append("edge_match: "+str(exc))
 if not _integer(b.get("absolute_roi_margin")) or not 1<=b.get("absolute_roi_margin",0)<32:
  errors.append("Fixed nonempty absolute interior required")
 if b.get("boundary_policy")!="require_spatial_support":
  errors.append("v2 spatial support loss guard required")
 if not _integer(b.get("gradient_log_interval"),1): errors.append("Explicit gradient log cadence required")
 aug=b.get("augmentation",{})
 if (set(aug)!={"hflip_probability","vflip_probability","rot90"}
     or any(not _finite(aug.get(k)) or not 0<=aug[k]<=1 for k in ("hflip_probability","vflip_probability"))
     or type(aug.get("rot90")) is not bool):
  errors.append("Explicit Hflip/Vflip/rot90 augmentation required")
 p=b.get("pruning",{})
 if not _integer(p.get("stagnation_window"),3) or not _finite(p.get("tolerance")) or p["tolerance"]<0:
  errors.append("Explicit pruning recent-window>=3 and nonnegative tolerance required")
 if not b.get("protocol_decisions"): errors.append("Operational proposals need recorded rationale")
 if errors: raise ValueError("Invalid EdgeRes baseline: "+"; ".join(errors))

def validate_method_config(c):
 if any(c.get(k)!=v for k,v in METHOD_CONTRACT.items()) or c.get("campaign_id")!=CAMPAIGN_ID:
  raise ValueError("EdgeRes-v2 campaign/method identity required")
 if not str(c.get("run_id","")).startswith("EGR2_"): raise ValueError("Separate EGR2 run namespace required")
 if c.get("initialization")!="FRESH" or c.get("donor_checkpoint") is not None:
  raise ValueError("Fresh training only; no inherited checkpoints")
 if c.get("requires_fr_train") is not False or c.get("requires_source_scene_map") is not False:
  raise ValueError("RR-only input-specific contract required")
 if c.get("stage2") is not False: raise ValueError("Stage2 is outside this campaign")
 if any(k in c for k in ("lambda_eps_FR","epsilon_radius_fr","scene_prior","b_global")):
  raise ValueError("Legacy FR/global-prior method fields forbidden")
 validate_baseline(c.get("baseline_resolved",{}))
 lane=LANES.get(c.get("server"))
 if not lane or c.get("dataset")!=lane["dataset"] or c.get("seed") not in lane["seeds"]:
  raise ValueError("Server/dataset/seed identity mismatch")
 if c["baseline_resolved"]["edge_match"]["sensor"]!=lane["dataset"]: raise ValueError("Matcher lane mismatch")
 for key in ("lambda_abs","lambda_epsilon"):
  if not _finite(c.get(key)) or c[key]<=0: raise ValueError("Positive full-method coefficients required")
 ef,et=c.get('loss_case'),c.get('optimizer_case')
 if ef not in LOSS_CASES or (et is not None and et not in OPTIMIZER_CASES): raise ValueError('Unknown registered EF/ET')
 ep,ab=LOSS_CASES[ef]
 if (c.get('lambda_abs')!=ab or c.get('lambda_epsilon')!=ep or c.get('lambda_eps_RR')!=ep
     or c.get('lambda_rec')!=1. or c.get('epsilon_radius_rr')!=2.
     or c.get('epsilon_every_update') is not True or c.get('epsilon_reduction')!='RR_SAMPLE_MEAN'):
  raise ValueError('Resolved coefficients/epsilon differ from registered case')
 replica=c.get('replica')
 if replica not in ('R1','R2','R3') or c.get('seed')!=lane['seeds'][int(replica[1])-1]:
  raise ValueError('Seed must match the registered replica')
 schedule=c.get('schedule')
 if (schedule not in HORIZONS or c.get('planned_horizon')!=HORIZONS[schedule]
     or c.get('evaluation_steps')!=list(evaluation_steps(schedule))):
  raise ValueError('Horizon/evaluation protocol differs from registered schedule')
 b=c['baseline_resolved'];ma,mu,mw=OPTIMIZER_CASES.get(et,(1.,1.,1.))
 if (c.get('lr_A')!=b['lr_A0']*ma or c.get('lr_U')!=b['lr_U0']*mu
     or c.get('weight_decay')!=b['weight_decay0']*mw):
  raise ValueError('Optimizer multipliers must apply to baseline, not previous ET')
 if (c.get('stream_seed_protocol')!=SEED_DERIVATION_PROTOCOL
     or c.get('stream_seeds')!=stream_seeds(c['seed'])):
  raise ValueError('Independent registered RNG streams required')
 expected_id=f"EGR2_{lane['dataset']}_{c['server'].upper()}_{ef}_{et or 'BASE'}_{schedule}_{replica}_S{c['seed']}"
 if c['run_id']!=expected_id or not _integer(c.get('attempt'),1):
  raise ValueError('Run namespace/case/attempt mismatch')
 if c.get('edge_protocol_digest')!=edge_protocol_digest(b):
  raise ValueError('Executable EdgeMatch protocol digest differs from baseline')
 if c.get('epsilon_distribution')!='UNIFORM_AREA_DISK':
  raise ValueError('Registered area-uniform disk epsilon required')
 if c.get('runnable') and not conditional_activation(ef,c.get('conditional_activation',{}).get('decision'))['triggered']:
  raise ValueError('Conditional case requires explicit recorded evidence')

def evaluation_steps(schedule):
 horizon=HORIZONS.get(schedule,schedule)
 if not _integer(horizon) or horizon not in HORIZONS.values(): raise ValueError("Unsupported schedule")
 return tuple(sorted({1000,25000,horizon,*range(2000,horizon+1,2000)}))

def stream_seeds(seed):
 if not _integer(seed): raise ValueError("Nonnegative seed required")
 labels=("A_init","T_init","rr_data","rr_augmentation","rr_epsilon")
 out={label:1+int.from_bytes(hashlib.sha256(f"{STREAM_SEED_NAMESPACE}:{seed}:{label}".encode()).digest()[:8],"big")%(2**63-1) for label in labels}
 out["diagnostic_probe"]=PROBE_SEED
 return out

def conditional_activation(case,decision=None):
 if case not in CONDITIONAL_CASES: return {"triggered":True,"reason":"BASIC_REGISTERED_CASE"}
 valid=(isinstance(decision,Mapping) and decision.get("condition")==CONDITIONAL_CASES[case]
        and isinstance(decision.get("reason"),str) and bool(decision["reason"].strip())
        and isinstance(decision.get("evidence_record_ids"),list) and bool(decision["evidence_record_ids"])
        and all(isinstance(x,str) and x for x in decision["evidence_record_ids"]))
 return {"triggered":bool(valid),"reason":"RECORDED_CONDITIONAL_DECISION" if valid else "NOT_TRIGGERED",
         "condition":CONDITIONAL_CASES[case],"decision":deepcopy(decision)}

def resolve_case(server,*,baseline,loss_case="EF00",optimizer_case=None,schedule="H50",replica="R1",attempt=1,decision=None):
 validate_baseline(baseline)
 if server not in LANES or loss_case not in LOSS_CASES or schedule not in HORIZONS: raise ValueError("Unknown case/lane/horizon")
 if optimizer_case is not None and optimizer_case not in OPTIMIZER_CASES: raise ValueError("Unknown optimizer profile")
 if replica not in ("R1","R2","R3") or not _integer(attempt,1): raise ValueError("Explicit replica/attempt required")
 lane=LANES[server]
 if baseline["edge_match"]["sensor"]!=lane["dataset"]: raise ValueError("Baseline belongs to another lane")
 seed=lane["seeds"][int(replica[1])-1]
 ep,ab=LOSS_CASES[loss_case];ma,mu,mw=OPTIMIZER_CASES.get(optimizer_case,(1.,1.,1.))
 lr_a,lr_u=baseline["lr_A0"]*ma,baseline["lr_U0"]*mu
 if baseline["min_lr"]>min(lr_a,lr_u): raise ValueError("Minimum LR exceeds resolved peak LR")
 activation=conditional_activation(loss_case,decision)
 duplicate=optimizer_case in ("ET05","ET06") and baseline["weight_decay0"]==0
 status="SKIP_DUPLICATE_CONFIG" if duplicate else "PLANNED" if activation["triggered"] else "NOT_TRIGGERED"
 result={"campaign_id":CAMPAIGN_ID,"parent_campaign_id":PARENT_CAMPAIGN_ID,**METHOD_CONTRACT,
  "server":server,"dataset":lane["dataset"],"loss_case":loss_case,"optimizer_case":optimizer_case,
  "case":loss_case+("_"+optimizer_case if optimizer_case else ""),"parent_recipe":loss_case if optimizer_case else None,
  "run_id":f"EGR2_{lane['dataset']}_{server.upper()}_{loss_case}_{optimizer_case or 'BASE'}_{schedule}_{replica}_S{seed}",
  "attempt":attempt,"replica":replica,"seed":seed,"initialization":"FRESH","donor_checkpoint":None,
  "schedule":schedule,"planned_horizon":HORIZONS[schedule],"evaluation_steps":list(evaluation_steps(schedule)),
  "confirmation":replica!="R1","baseline_resolved":deepcopy(dict(baseline)),
  "lambda_rec":1.,"lambda_abs":ab,"lambda_epsilon":ep,"lambda_eps_RR":ep,
  "epsilon_radius_rr":2.,"epsilon_every_update":True,"epsilon_distribution":"UNIFORM_AREA_DISK",
  "epsilon_reduction":"RR_SAMPLE_MEAN","lr_A":lr_a,"lr_U":lr_u,"weight_decay":baseline["weight_decay0"]*mw,
  "lr_A_multiplier":ma,"lr_U_multiplier":mu,"weight_decay_multiplier":mw,"stream_seeds":stream_seeds(seed),
  "stream_seed_protocol":SEED_DERIVATION_PROTOCOL,"hqnr_min":lane["hqnr_min"],"ergas_max":lane["ergas_max"],
  "retention_policy":RETENTION_POLICY,"stage2":False,"requires_fr_train":False,"requires_source_scene_map":False,
  "edge_protocol_digest":edge_protocol_digest(baseline),
  "status":status,"runnable":status=="PLANNED","conditional_activation":activation}
 validate_method_config(result)
 return result

def optimizer_priority(server,observation=None):
 choices={"PRIOR_HARMFUL_RESIDUAL_NOT_RECOVERING":("ET02","ET08"),
 "RESIDUAL_DEGRADES_PRIOR":("ET01","ET00"),"RR_CONVERGENCE_BOTTLENECK":("ET04","ET05"),
 "FINITE_OSCILLATION":("ET03",)}
 return choices.get(observation,OPTIMIZER_ORDER[server])

def target_gap(hqnr: float, ergas: float, *, server: str) -> dict[str, Any]:
    """Full-precision, inclusive AND gate.  Invalid numbers never pass."""
    lane = LANES[server]
    valid = _finite(hqnr) and _finite(ergas) and 0 <= hqnr <= 1 and ergas >= 0
    h_pass = bool(valid and hqnr >= lane["hqnr_min"])
    e_pass = bool(valid and ergas <= lane["ergas_max"])
    vh = max(0., (lane["hqnr_min"] - hqnr) / lane["hqnr_min"]) if valid else None
    ve = max(0., (ergas - lane["ergas_max"]) / lane["ergas_max"]) if valid else None
    return {"H_target": lane["hqnr_min"], "E_target": lane["ergas_max"],
            "h_pass": h_pass, "e_pass": e_pass, "target_pair_pass": h_pass and e_pass,
            "v_H": vh, "v_E": ve, "D_inf": max(vh, ve) if valid else None,
            "D1": vh + ve if valid else None}


def qualify_evaluation(record: Mapping[str, Any], *, server: str,
                       expected_rr_ids: Sequence[str], expected_fr_ids: Sequence[str]) -> dict[str, Any]:
    """Return an auditable gate result without writing artifacts.

    Schema: rr/fr={metrics:{...}, per_scene:[{scene_id, metrics:{...}}]}.
    Identity fields are required in state_before, rr_state, fr_state,
    state_after. Their complete per-input inference/edge rules must match;
    input-dependent RR/FR correction vectors may differ legitimately.  No round-off tolerance is used anywhere in the gate.
    """
    lane = LANES[server]
    reasons: list[str] = []
    complete, finite = True, True
    counts: dict[str, Any] = {}
    for branch, expected in (("rr", expected_rr_ids), ("fr", expected_fr_ids)):
        expected = list(expected)
        expected_valid = len(expected) == 20 and all(isinstance(x, str) and x for x in expected)
        expected_valid = expected_valid and len(expected) == len(set(expected))
        if not expected_valid:
            complete = False
            reasons.append(f"{branch.upper()}_INVALID_EXPECTED_IDS")
        section = record.get(branch, {})
        section = section if isinstance(section, Mapping) else {}
        rows = section.get("per_scene", [])
        if not isinstance(rows, (tuple, list)):
            rows = []
        ids = [row.get("scene_id") if isinstance(row, Mapping) else None for row in rows]
        valid_ids = [x for x in ids if isinstance(x, str) and x]
        duplicates = sorted(x for x, n in Counter(valid_ids).items() if n > 1)
        missing = sorted(set(x for x in expected if isinstance(x, str)) - set(valid_ids))
        unexpected = sorted(set(valid_ids) - set(x for x in expected if isinstance(x, str)))
        if missing or unexpected or duplicates or len(valid_ids) != len(ids) or len(rows) != len(expected):
            complete = False
            reasons.append(f"{branch.upper()}_SCENE_SET_INCOMPLETE")
        counts[branch] = {"expected": len(expected), "completed": len(rows),
                          "missing": missing, "unexpected": unexpected, "duplicates": duplicates}
        required = FR_METRICS if branch == "fr" else RR_METRICS + (("Q8" if lane["dataset"] == "WV3" else "Q4"),)
        for label, metrics in [("aggregate", section.get("metrics"))] + [
                (f"scene:{i}", row.get("metrics") if isinstance(row, Mapping) else None)
                for i, row in enumerate(rows)]:
            if not isinstance(metrics, Mapping) or any(key not in metrics for key in required):
                complete = False
                reasons.append(f"{branch.upper()}_MISSING_METRIC:{label}")
            if not isinstance(metrics, Mapping) or any(not _finite(metrics.get(key)) for key in required):
                finite = False
                reasons.append(f"{branch.upper()}_NONFINITE_METRIC:{label}")
        aggregate = section.get("metrics")
        if rows and isinstance(aggregate, Mapping):
            for key in required:
                values = [row.get("metrics", {}).get(key) if isinstance(row, Mapping)
                          and isinstance(row.get("metrics"), Mapping) else None for row in rows]
                if _finite(aggregate.get(key)) and all(_finite(value) for value in values):
                    # Means must come from these exact scene records, not a
                    # different checkpoint's best or mean(D_lambda)*mean(D_s).
                    if not math.isclose(aggregate[key], math.fsum(values) / len(values),
                                        rel_tol=0., abs_tol=1e-12):
                        complete = False
                        reasons.append(f"{branch.upper()}_AGGREGATE_MISMATCH:{key}")
    identities = [record.get(key) for key in ("state_before", "rr_state", "fr_state", "state_after")]
    identity_valid = all(isinstance(identity, Mapping)
                         and all(key in identity and not _unresolved(identity[key]) for key in IDENTITY_FIELDS)
                         for identity in identities)
    if identity_valid:
        for identity in identities:
            identity_valid &= all(identity.get(key) == expected for key, expected in METHOD_CONTRACT.items())
            identity_valid &= bool(_integer(identity["attempt"], 1)
                                   and _integer(identity["seed"])
                                   and _integer(identity["completed_update"], 1)
                                   and all(isinstance(identity[key], str) and identity[key].strip()
                                           for key in ("run_id", "inference_policy_digest",
                                                       "evaluator_revision", "edge_protocol_digest"))
                                   and isinstance(identity["model_state_sha256"], str)
                                   and re.fullmatch(r"[a-f0-9]{64}", identity["model_state_sha256"]))
        identity_valid &= all(all(identity[key] == identities[0][key] for key in IDENTITY_FIELDS)
                              for identity in identities[1:])
    if not identity_valid:
        reasons.append("SAME_STATE_IDENTITY_NOT_VERIFIED")
    method_eligible = record.get("full_method_eligible") is True
    if not method_eligible:
        reasons.append("FULL_METHOD_NOT_ELIGIBLE")
    if record.get("dataset", lane["dataset"]) != lane["dataset"] or record.get("server", server) != server:
        method_eligible = False
        reasons.append("LANE_MISMATCH")
    # Cross-check outer provenance if provided: do not accept identity from a
    # different run attached to otherwise valid aggregates.
    if identity_valid and any(key in record and record[key] != identities[0][key]
                              for key in IDENTITY_FIELDS):
        identity_valid = False
        reasons.append("OUTER_IDENTITY_MISMATCH")
    rr = record.get("rr", {})
    fr = record.get("fr", {})
    rr = rr if isinstance(rr, Mapping) else {}
    fr = fr if isinstance(fr, Mapping) else {}
    rr_metrics = rr.get("metrics", {})
    fr_metrics = fr.get("metrics", {})
    ergas = rr_metrics.get("ERGAS") if isinstance(rr_metrics, Mapping) else None
    hqnr = fr_metrics.get("HQNR") if isinstance(fr_metrics, Mapping) else None
    gap = target_gap(hqnr, ergas, server=server)
    if gap["D_inf"] is None:
        reasons.append("INVALID_TARGET_METRIC_RANGE")
        finite = False
    joint_pass = bool(method_eligible and complete and finite and identity_valid and gap["target_pair_pass"])
    return {**gap, "HQNR": hqnr, "ERGAS": ergas, "joint_pass": joint_pass,
            "full_method_eligible": method_eligible, "metrics_complete": complete,
            "metrics_finite": finite, "same_state_verified": bool(identity_valid),
            "scene_completeness": counts, "reasons": reasons,
            "selection": "TARGET_QUALIFIED" if joint_pass else "BEST_OBSERVED_NOT_QUALIFIED"}


def _pair(record: Mapping[str, Any]) -> tuple[float, float]:
    h, e = record.get("HQNR"), record.get("ERGAS")
    if not _finite(h) or not _finite(e) or not 0 <= h <= 1 or e < 0:
        raise ValueError("Ranking requires finite, valid same-state HQNR/ERGAS")
    return h, e


def representative_rank(record: Mapping[str, Any]) -> tuple:
    h, e = _pair(record)
    return (-h, e, record["completed_update"], record["run_id"])


def gap_rank(record: Mapping[str, Any], *, server: str) -> tuple:
    h, e = _pair(record)
    gap = target_gap(h, e, server=server)
    return (gap["D_inf"], gap["D1"], -h, e, record["completed_update"], record["run_id"])


def representative(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Only records qualified by the full retention gate are representatives."""
    passed = [r for r in records if r.get("joint_pass") is True]
    return min(passed, key=representative_rank) if passed else None


def rr_reserve(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passed = [r for r in records if r.get("joint_pass") is True]
    return min(passed, key=lambda r: (_pair(r)[1], -_pair(r)[0], r["completed_update"], r["run_id"])) if passed else None


def pareto_front(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Numeric-only frontier; Pareto membership does NOT authorize saving."""
    for record in records:
        _pair(record)
    return sorted([r for r in records if not any(
        _pair(other)[0] >= _pair(r)[0] and _pair(other)[1] <= _pair(r)[1]
        and _pair(other) != _pair(r) for other in records)], key=representative_rank)

def should_prune(current, previous, *, stagnation_window, tolerance):
 """Two preceding recipes, three common points, explicit recent stagnation.

 Tolerance is in dimensionless relative target-gap units, not HQNR units.
 A lexicographic (D_inf,D1) reduction beyond tolerance inside the pinned recent
 window prevents pruning. Confirmations and EF00/BASE/R1 are always protected.
 """
 if not _integer(stagnation_window,3) or not _finite(tolerance) or tolerance<0:
  raise ValueError("Pin recent stagnation window>=3 and finite nonnegative tolerance")
 no=lambda reason:{"prune":False,"reason":reason,"comparators":[],"common_steps":[],
                   "stagnation_window":stagnation_window,"tolerance":tolerance}
 if current.get("replica")!="R1" or current.get("confirmation"): return no("CONFIRMATION_PROTECTED")
 if current.get("loss_case")=="EF00" and current.get("optimizer_case") is None: return no("EF00_PROTECTED")
 if not _integer(current.get("completed_update"),25000): return no("BEFORE_25K")
 server=current.get("server")
 if server not in LANES or current.get("dataset")!=LANES[server]["dataset"]: return no("LANE_MISMATCH")
 if current.get("seed")!=LANES[server]["seeds"][0]: return no("NOT_REGISTERED_R1_SEED")
 def history(run):
  out={}
  for row in run.get("evaluations",()):
   step=row.get("completed_update")
   if not _integer(step,1): return None
   if step>current["completed_update"]: continue
   if step in out or any(row.get(k) is not True for k in ("metrics_complete","metrics_finite","same_state_verified")): return None
   try: _pair(row)
   except ValueError: return None
   out[step]=row
  return out
 own=history(current)
 if not own or len(own)<stagnation_window: return no("INSUFFICIENT_RECENT_WINDOW")
 recent=sorted(own)[-stagnation_window:]
 gaps=[gap_rank(own[s],server=server)[:2] for s in recent]
 def improvement(a,b):
  return b[0]<a[0]-tolerance or (abs(b[0]-a[0])<=tolerance and b[1]<a[1]-tolerance)
 if any(improvement(a,b) for a,b in zip(gaps,gaps[1:])): return no("TARGET_GAP_IMPROVING")
 comps=[];recipes=set()
 for run in previous:
  recipe=(run.get("loss_case"),run.get("optimizer_case"))
  if recipe in recipes or recipe==(current.get("loss_case"),current.get("optimizer_case")): continue
  if run.get("run_id")==current.get("run_id"): continue
  if any(run.get(k)!=current.get(k) for k in ("dataset","schedule","seed","server","replica")): continue
  hist=history(run)
  if hist: comps.append((run,hist));recipes.add(recipe)
 for (left,lh),(right,rh) in combinations(comps,2):
  common=sorted(set(own)&set(lh)&set(rh))[-3:]
  if len(common)<3: continue
  if all(_pair(hist[s])[0]>=_pair(own[s])[0] and _pair(hist[s])[1]<=_pair(own[s])[1] for hist in (lh,rh) for s in common):
   return {"prune":True,"reason":"PRUNED_DOMINATED","comparators":[left["run_id"],right["run_id"]],
           "common_steps":common,"stagnation_steps":recent,"stagnation_window":stagnation_window,"tolerance":tolerance}
 return no("INSUFFICIENT_CONSISTENT_DOMINATION")


def estimate_remaining_seconds(*,remaining_updates,update_p90_seconds,remaining_evaluations,
                               rr_fr_eval_seconds,artifact_seconds,probe_seconds=0.):
 """Raw measured wall time; admission applies the 1.2 safety factor exactly once."""
 if not _integer(remaining_updates) or not _integer(remaining_evaluations): raise ValueError("Nonnegative counts required")
 if any(not _finite(x) or x<0 for x in (update_p90_seconds,rr_fr_eval_seconds,artifact_seconds,probe_seconds)):
  raise ValueError("Observed finite nonnegative timing required")
 if remaining_updates and update_p90_seconds<=0: raise ValueError("Positive measured update timing required")
 return remaining_updates*update_p90_seconds+remaining_evaluations*rr_fr_eval_seconds+artifact_seconds+probe_seconds


def admit_new_run(*,elapsed_seconds,estimated_to_25k_seconds,estimated_full_seconds,
                  closing_reserve_seconds=7200.,selected_recipe_seconds=None,
                  remaining_confirmations=2,phase_end_hours=None):
 """Reserve closeout >=2h and remaining repeats before admitting any new run.

 Timing estimates are RAW (without 1.2); selected_recipe_seconds is likewise raw.
 For R2 itself remaining_confirmations=1; for R3=0. Exploration defaults to2.
 If no selected recipe exists, its same-horizon estimate is used for both repeats.
 Phase boundaries govern starts, not automatic termination of a live run.
 """
 vals=(elapsed_seconds,estimated_to_25k_seconds,estimated_full_seconds,closing_reserve_seconds)
 if any(not _finite(x) or x<0 for x in vals) or estimated_full_seconds<estimated_to_25k_seconds or estimated_to_25k_seconds<=0:
  raise ValueError("Explicit measured nonnegative ETA, positive judgment ETA required")
 if closing_reserve_seconds<7200: raise ValueError("v2 requires at least two hours closeout")
 if not _integer(remaining_confirmations) or remaining_confirmations>2: raise ValueError("Remaining confirmations must be0/1/2")
 selected=estimated_full_seconds if selected_recipe_seconds is None else selected_recipe_seconds
 if not _finite(selected) or selected<=0: raise ValueError("Positive selected-recipe estimate required")
 if phase_end_hours is not None and (not _finite(phase_end_hours) or not 0<phase_end_hours<=46):
  raise ValueError("Phase admission boundary must be in(0,46]hours")
 eta=1.2*estimated_full_seconds
 repeat_reserve=1.2*selected*remaining_confirmations
 finish=elapsed_seconds+eta+repeat_reserve+closing_reserve_seconds
 if elapsed_seconds>=46*3600: reason="NOT_STARTED_NO_NEW_AFTER_46H"
 elif phase_end_hours is not None and elapsed_seconds>=phase_end_hours*3600: reason="NOT_STARTED_PHASE_BOUNDARY"
 elif elapsed_seconds+1.2*estimated_to_25k_seconds+repeat_reserve+closing_reserve_seconds>48*3600: reason="NOT_STARTED_BUDGET_25K"
 elif finish>48*3600: reason="NOT_STARTED_BUDGET_WITH_CONFIRMATIONS"
 else: reason="ADMITTED"
 return {"admitted":reason=="ADMITTED","reason":reason,"safety_multiplier":1.2,
  "estimated_run_seconds":eta,"confirmation_reserve_seconds":repeat_reserve,
  "remaining_confirmations":remaining_confirmations,"closing_reserve_seconds":closing_reserve_seconds,
  "estimated_finish_with_reserves_seconds":finish,"deadline_seconds":48*3600,
  "switch_to_confirmation":reason.startswith("NOT_STARTED_BUDGET") and remaining_confirmations>0}


def catalog(server):
 lane=LANES[server]
 return {"campaign_id":CAMPAIGN_ID,**METHOD_CONTRACT,"server":server,**lane,
  "loss_cases":[{"case":f,"lambda_epsilon":LOSS_CASES[f][0],"lambda_abs":LOSS_CASES[f][1],
                 "schedule":"H50","replica":"R1","fresh":True,"priority":i+1}
                for i,f in enumerate(LOSS_ORDER[server])],
  "conditional_cases":[{"case":f,"lambda_epsilon":LOSS_CASES[f][0],"lambda_abs":LOSS_CASES[f][1],
                        "condition":condition,"status":"NOT_TRIGGERED","requires_decision_evidence":True}
                       for f,condition in CONDITIONAL_CASES.items()],
  "optimizer_menu":[{"case":case,"lr_A_multiplier":v[0],"lr_U_multiplier":v[1],"weight_decay_multiplier":v[2]}
                    for case,v in OPTIMIZER_CASES.items()],
  "optimizer_default_priority":list(OPTIMIZER_ORDER[server]),
  "schedules":{k:{"horizon":v,"evaluation_steps":list(evaluation_steps(k)),"fresh":True} for k,v in HORIZONS.items()},
  "confirmation_priority":"chosen_recipe_R2_R3_reserved_on_every_admission",
  "requires_fr_train":False,"requires_source_scene_map":False,"stage2":False,
  "spreadsheet_tab":lane["dataset"]+"-ablations","local_first":True,"retention_policy":RETENTION_POLICY,
  "selection":"HQNR_DESC_THEN_ERGAS_ASC_THEN_STEP_ASC_THEN_RUN_ID",
  "selection_override":"User confirmed v2 HQNR then ERGAS; SCC remains measured, not tie breaker",
  "sheet_override":"Latest user instruction: new fixed64 workbook and dataset ablations, not v2 obsolete analysis152"}
