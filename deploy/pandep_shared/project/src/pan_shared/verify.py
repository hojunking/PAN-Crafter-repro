"""Evidence-bearing G0–G5 gates; no synthetic success or formal-run reuse."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .common import CAMPAIGN, SENSORS, atomic_json, canonical_sha, file_sha, read_json, seed_for, timestamp
from . import preflight
from .registry import build_registry
from .safety import SafetyStop, admit, inventory, control

PROJECT = Path(__file__).resolve().parents[2]
IDENTITY_FIELDS = ("source_sha256", "environment_sha256", "dataset_sha256", "subset_sha256", "registry_sha256")


def tests_manifest(project=PROJECT):
    files = sorted((Path(project) / "tests").glob("test_*.py"))
    if not files:
        raise SafetyStop("BLOCKED_GATES: no tests discovered")
    return {str(p.relative_to(project)): file_sha(p) for p in files}


def hardware_identity(inv):
    rows = []
    for line in inv.get("gpu", {}).get("stdout", "").splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 5:
            raise SafetyStop("BLOCKED_GATES: GPU UUID/driver inventory could not be parsed")
        rows.append(dict(uuid=values[0], name=values[1], total_memory_mb=values[2], driver=values[4]))
    if not rows:
        raise SafetyStop("BLOCKED_GATES: actual GPU identity required")
    return {"hostname": inv["hostname"], "server": inv["server"], "gpus": rows}


def run_unit_tests(project=PROJECT):
    """Run a fresh process so test fixtures cannot perturb production RNG/mode."""
    script = r'''
import json,sys,unittest
suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_*.py')
def ids(node):
 for item in node:
  if isinstance(item,unittest.TestSuite): yield from ids(item)
  else: yield item.id()
names=list(ids(suite))
result=unittest.TextTestRunner(verbosity=2).run(suite)
payload=dict(test_count=result.testsRun,success=result.wasSuccessful(),test_ids=names,
             skipped=[(test.id(),reason) for test,reason in result.skipped],
             failures=[(test.id(),detail) for test,detail in result.failures],
             errors=[(test.id(),detail) for test,detail in result.errors])
print('PANDEP_TEST_RESULT='+json.dumps(payload,sort_keys=True))
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 else 1)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(Path(project) / "src"), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", script, str(Path(project) / "tests")],
                            cwd=project, env=env, text=True, capture_output=True)
    markers = [line.split("=", 1)[1] for line in result.stdout.splitlines()
               if line.startswith("PANDEP_TEST_RESULT=")]
    if len(markers) != 1:
        raise SafetyStop("BLOCKED_GATES: unit runner produced no unique result: " + result.stderr[-4000:])
    receipt = json.loads(markers[0])
    receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    required = {"test_registry_safety", "test_model_data", "test_training", "test_eval_sheet", "test_verify"}
    observed = {name.split(".")[0] for name in receipt["test_ids"]}
    if result.returncode or not receipt["success"] or receipt["test_count"] <= 0 or receipt["skipped"] or not required <= observed:
        raise SafetyStop("BLOCKED_GATES: tests failed, skipped, empty, or missing required categories: " + result.stderr[-5000:])
    receipt["status"] = "PASS"
    return receipt


def validate_dataset_evidence(work_root):
    """Rehash actual immutable files, bind full-scan/native-LP receipts to bytes."""
    from .data import COUNTS, SPLITS, frozen_catalog_path
    from .frontend import SENSOR_PROFILES, LP_RECIPE, AUGMENTATION
    work = Path(work_root).resolve()
    dataset = read_json(work / "dataset_manifest.json")
    subsets = read_json(work / "subset_manifest.json")
    frozen = read_json(frozen_catalog_path())
    if set(dataset.get("sensors", {})) != set(SENSORS) or dataset.get("recipe") != LP_RECIPE or dataset.get("augmentation") != AUGMENTATION:
        raise SafetyStop("BLOCKED_GATES: dataset sensor/LP/augmentation contract")
    if dataset.get("original_data_read_only") is not True:
        raise SafetyStop("BLOCKED_GATES: missing read-only source provenance")
    if subsets != preflight.subset_and_probes(dataset):
        raise SafetyStop("BLOCKED_GATES: subset/probe IDs do not match fixed hash ranking")
    receipts = []
    for sensor in SENSORS:
        profile = dataset["sensors"][sensor]
        for key in ("bands", "max_dn", "band_order"):
            if profile.get(key) != SENSOR_PROFILES[sensor][key]:
                raise SafetyStop("BLOCKED_GATES: sensor profile mismatch")
        if set(profile.get("splits", {})) != set(SPLITS):
            raise SafetyStop("BLOCKED_GATES: all four benchmark splits required")
        for split in SPLITS:
            value = profile["splits"][split]
            if value.get("sha256") != frozen["sensors"][sensor]["splits"][split]["sha256"]:
                raise SafetyStop("BLOCKED_DATA_IDENTITY: fixed twelve-source byte SHA mismatch")
            expected_count = 20 if split in ("rr", "fr") else COUNTS[sensor][split]
            if value.get("count") != expected_count or value.get("full_scan") is not True or value.get("full_lp_verified") is not True:
                raise SafetyStop("BLOCKED_GATES: count/full-scan/full-native-LP proof missing")
            required_arrays = {"pan", "ms", "lms"} | ({"gt"} if split != "fr" else set())
            stats = value.get("statistics", {})
            if not required_arrays <= set(stats) or any(v.get("nonfinite_count") != 0 for v in stats.values()):
                raise SafetyStop("BLOCKED_GATES: full array statistics missing or nonfinite")
            source, lp = Path(value["dataroot"]).resolve(strict=True), Path(value["lpan_path"]).resolve(strict=True)
            if not lp.is_relative_to(work) or source.is_relative_to(work) or not preflight._read_only(source):
                raise SafetyStop("BLOCKED_PATH: original source/cache mount boundary changed")
            if file_sha(source) != value["sha256"] or file_sha(lp) != value["lpan_sha256"]:
                raise SafetyStop("BLOCKED_DATA_IDENTITY: source/LP bytes changed after preflight")
            if sensor == "QB" and split in ("train", "val"):
                proof = value.get("msfix_audit", {})
                if proof.get("status") != "PASS" or proof.get("gt_unchanged") is not True or proof.get("pan_unchanged") is not True or proof.get("samples_verified") != expected_count:
                    raise SafetyStop("BLOCKED_GATES: QB msfix full GT/PAN preservation proof missing")
            receipts.append({"sensor": sensor, "split": split, "count": expected_count,
                             "source_sha256": value["sha256"], "lpan_sha256": value["lpan_sha256"]})
    return {"status": "PASS", "actual_byte_rechecks": len(receipts), "splits": receipts,
            "subset_sha256": subsets["subset_sha256"], "all_native_lp_preflight_receipts_bound_to_actual_bytes": True}


def q00_smoke(work_root, manifest, *, device="cuda:0"):
    import torch
    from .data import SensorDataset
    from .model import SharedPLHUNet
    from .train import Trainer
    from .checkpoints import tensor_state_hash
    work = Path(work_root).resolve()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise SafetyStop("BLOCKED_GATES: actual S2 CUDA Q00 is mandatory; CPU is unit evidence only")
    inv = inventory(work, manifest["server_identity_file"])
    admit(inv, expected_hostname=manifest["hostname"], own_pid=os.getpid())
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise SafetyStop("BLOCKED_ENVIRONMENT: deterministic CUBLAS_WORKSPACE_CONFIG required before CUDA initialization")
    scratch = work / "smoke_q00" / uuid.uuid4().hex
    if not scratch.resolve().is_relative_to(work) or scratch.exists():
        raise SafetyStop("BLOCKED_PATH: Q00 scratch must be new and campaign-owned")
    run = dict(next(r for r in build_registry(manifest["microbatch"])["runs"] if r["case_id"] == "C00" and r["repeat"] == 1))
    run.update(case_id="Q00", run_id="Q00_" + scratch.name,
               seed=seed_for(271001, "q00") % (2**32 - 1), formal_run=False)
    run["config_sha256"] = canonical_sha({k: v for k, v in run.items() if k != "config_sha256"})
    identity = {key: manifest[key] for key in IDENTITY_FIELDS}
    identity.update(run_id=run["run_id"], config_sha256=run["config_sha256"], campaign_id=CAMPAIGN,
                    case_id="Q00", formal_run=False, purpose="six-step actual-data smoke, never a performance row")
    dataset = read_json(work / "dataset_manifest.json")
    datasets = {s: SensorDataset(dataset, s, "train", work) for s in SENSORS}
    model = SharedPLHUNet(width=104, depth=(1, 2, 2), seed=run["seed"])
    architecture = model.architecture_manifest()
    trainer = Trainer(model, datasets, run, identity, scratch, device=device)
    def smoke_control():
        return control(work)["action"] if (work / "control.json").exists() else "RUN"
    result = trainer.run_until(3, smoke_control)
    if result != "BLOCK_COMPLETE":
        raise SafetyStop(result + ": Q00 interrupted by operator; smoke fullstate preserved")
    before_resume = tensor_state_hash(trainer.model.state_dict())
    next_batch = trainer.sampler.peek()
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    restored = Trainer(SharedPLHUNet(width=104, depth=(1, 2, 2), seed=run["seed"]),
                       datasets, run, identity, scratch, device=device)
    if restored.step != 3 or tensor_state_hash(restored.model.state_dict()) != before_resume or restored.sampler.peek() != next_batch:
        raise SafetyStop("BLOCKED_GATES: actual Q00 state/next batch failed resume")
    result = restored.run_until(6, smoke_control)
    if result != "BLOCK_COMPLETE":
        raise SafetyStop(result + ": Q00 interrupted by operator; smoke fullstate preserved")
    exposures = restored.sampler.exposures()
    if any(value["optimizer_visits"] != 2 or value["committed_samples"] != 96 for value in exposures.values()):
        raise SafetyStop("BLOCKED_GATES: Q00 is not exactly two visits per sensor")
    if not all(torch.isfinite(parameter).all() for parameter in restored.model.parameters()):
        raise SafetyStop("BLOCKED_GATES: nonfinite Q00 parameters after updates")
    receipt = {"status": "PASS", "completed_updates": 6, "width": 104, "depth": [1, 2, 2],
               "effective_batch_size": 48, "microbatch": manifest["microbatch"],
               "device": str(device), "device_name": torch.cuda.get_device_name(torch.device(device)),
               "hardware_identity": hardware_identity(inv), "all_parameters_finite": True,
               "exposures": exposures, "architecture": architecture, "scratch_run_dir": str(scratch),
               "resume_boundary": 3, "formal_weights_reused": False, "sheet_rows_written": 0,
               "gpu_peak_bytes": torch.cuda.max_memory_allocated(torch.device(device)),
               "final_tensor_sha256": tensor_state_hash(restored.model.state_dict())}
    atomic_json(scratch / "q00_receipt.json", receipt)
    del restored
    gc.collect(); torch.cuda.empty_cache()
    return receipt


def verify(work_root, *, all_gates=True, credentials, device="cuda:0"):
    from .evaluate import evaluator_sha
    from .sheets import GoogleSheetAdapter, inspect_target
    work = Path(work_root).resolve()
    if not all_gates:
        raise SafetyStop("BLOCKED_GATES: production verification requires --all-gates")
    receipt = {"schema": "PANDEP_GATES_v1", "campaign_id": CAMPAIGN, "status": "RUNNING",
               "started_at": timestamp(), "gates": {}, "sheet_rows_written": 0, "formal_training_started": False}
    history = work / "preflight" / ("verify_" + uuid.uuid4().hex + ".json")
    latest = work / "preflight" / "gates.json"
    def persist():
        receipt["receipt_sha256"] = canonical_sha({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        atomic_json(history, receipt); atomic_json(latest, receipt)
    persist()  # Invalidate an older PASS before doing any new work.
    try:
        manifest = preflight.check_identity(work, require_registered=False)
        receipt["identity"] = {key: manifest[key] for key in IDENTITY_FIELDS}
        receipt["microbatch"] = manifest["microbatch"]
        receipt["identity"].update(evaluator_sha256=evaluator_sha(), tests_sha256=canonical_sha(tests_manifest()))
        receipt["gates"]["G0"] = {"status": "PASS", "paths_and_environment_bound": True,
                                   "hostname": manifest["hostname"], "server": manifest["server"]}
        persist()
        units = run_unit_tests()
        receipt["unit_tests"] = units
        receipt["gates"]["G1"] = {"status": "PASS", "evidence": "fixed-source synthetic architecture tests and active/inactive branch checks"}
        receipt["gates"]["G2"] = {"status": "PASS", "evidence": "isolated pinned-source nontrivial synthetic-state forward parity; not trained-weight parity"}
        receipt["gates"]["G3"] = validate_dataset_evidence(work)
        receipt["gates"]["G4"] = {"status": "PASS", "evidence": "deterministic CPU split-resume plus atomic corruption/interruption/lock unit tests"}
        persist()
        sheet = inspect_target(GoogleSheetAdapter(credentials))
        receipt["sheet_inspection"] = sheet
        if sheet.get("status") != "PASS":
            raise SafetyStop("BLOCKED_GATES: actual target Sheet read-only inspection failed")
        receipt["gates"]["G5"] = {"status": "PASS", "metric_evidence": "fixed-source synthetic RR/FR parity fixtures",
            "sheet_evidence": "mock conflict/readback/idempotence tests plus actual target/header read-only inspection",
            "actual_sheet_write_test": False, "sheet_rows_written": 0}
        receipt["q00"] = q00_smoke(work, manifest, device=device)
        if receipt["q00"].get("status") != "PASS":
            raise SafetyStop("BLOCKED_GATES: actual Q00 did not pass")
        # Reject changes during verification; source/test/evaluator hashes cannot
        # silently change between evidence generation and formal admission.
        final = preflight.check_identity(work, require_registered=False)
        if any(final[key] != receipt["identity"][key] for key in IDENTITY_FIELDS) or canonical_sha(tests_manifest()) != receipt["identity"]["tests_sha256"] or evaluator_sha() != receipt["identity"]["evaluator_sha256"]:
            raise SafetyStop("BLOCKED_IDENTITY: source or test evidence changed during verification")
        receipt.update(status="PASS", completed_at=timestamp())
    except Exception as error:
        receipt.update(status="FAILED", completed_at=timestamp(), error_type=type(error).__name__, error=str(error))
        persist()
        raise
    persist()
    return receipt


def check_gates(work_root, *, require_registered=True):
    from .evaluate import evaluator_sha
    work = Path(work_root).resolve()
    manifest = preflight.check_identity(work, require_registered=require_registered)
    path = work / "preflight" / "gates.json"
    if not path.is_file():
        raise SafetyStop("BLOCKED_GATES: run verify --all-gates on S2 first")
    receipt = read_json(path)
    if receipt.get("receipt_sha256") != canonical_sha({k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise SafetyStop("BLOCKED_GATES: receipt seal mismatch")
    if receipt.get("campaign_id") != CAMPAIGN or receipt.get("status") != "PASS" or receipt.get("q00", {}).get("status") != "PASS":
        raise SafetyStop("BLOCKED_GATES: actual S2 verification is missing or failed")
    if set(receipt.get("gates", {})) != {"G0", "G1", "G2", "G3", "G4", "G5"} or any(v.get("status") != "PASS" for v in receipt["gates"].values()):
        raise SafetyStop("BLOCKED_GATES: incomplete G0–G5 evidence")
    units, smoke = receipt.get("unit_tests", {}), receipt["q00"]
    if units.get("test_count", 0) <= 0 or units.get("status") != "PASS" or units.get("skipped"):
        raise SafetyStop("BLOCKED_GATES: successful nonempty unit evidence required")
    if (smoke.get("completed_updates") != 6 or smoke.get("width") != 104 or smoke.get("effective_batch_size") != 48 or
            smoke.get("formal_weights_reused") is not False or smoke.get("all_parameters_finite") is not True or
            set(smoke.get("exposures", {})) != set(SENSORS) or
            any(v.get("optimizer_visits") != 2 or v.get("committed_samples") != 96 for v in smoke["exposures"].values())):
        raise SafetyStop("BLOCKED_GATES: actual six-update three-sensor Q00 evidence incomplete")
    identities = {key: manifest[key] for key in IDENTITY_FIELDS}
    identities.update(evaluator_sha256=evaluator_sha(), tests_sha256=canonical_sha(tests_manifest()))
    if receipt.get("identity") != identities:
        raise SafetyStop("BLOCKED_GATES: evidence no longer matches source/environment/data/subset/registry/tests")
    if receipt.get("microbatch") != manifest["microbatch"]:
        raise SafetyStop("BLOCKED_GATES: microbatch policy differs")
    current_hardware = hardware_identity(inventory(work, manifest["server_identity_file"]))
    if receipt["q00"].get("hardware_identity") != current_hardware:
        raise SafetyStop("BLOCKED_GATES: actual GPU UUID/driver/hostname changed")
    return receipt
