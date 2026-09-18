#!/usr/bin/env python
"""M20-only postrun: strict grid identity -> explicit v2 -> exact50K -> opt-in upload.

Never starts training. Exit 0 also covers no_eligible and upload-pending; exit 2
means local evaluation is incomplete and must be retried, not retrained.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qrecon24_select as selector

GRID = tuple(range(1010, 50000, 1010)) + (50000,)
SELECTOR = selector.SELECTOR_V2
TARGET_FILE = f"qrecon24_target_selection_{SELECTOR}.json"
STATUS_FILE = "mix20h_postrun_status.json"
CAMPAIGN = "QRC24_MIX20H_20260918_v1"


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def before_deadline(deadline):
    if not deadline:
        return True
    stamp = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("deadline must include timezone")
    return dt.datetime.now(dt.timezone.utc) < stamp


def bound_deadline(manifest, requested=None):
    """The CLI may restate the shared deadline, never replace or extend it."""
    declared = manifest.get("deadline_at_utc")
    if not declared:
        raise ValueError("M20 runtime manifest deadline missing")

    def stamp(value):
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("deadline must include timezone")
        return parsed

    actual = stamp(declared)
    if requested is not None and stamp(requested) != actual:
        raise ValueError("CLI deadline differs from immutable M20 manifest deadline")
    return declared


def finite_metrics(record, keys):
    return all(isinstance(record.get(key), (int, float)) and math.isfinite(record[key]) for key in keys)


def validate_grid(wd, *, sha=selector.sha256_file, evaluator_hash=None):
    """Require all 50 bound checkpoints. A proxy/v1 artifact is never completion."""
    import yaml
    from pa.evalviews import evaluator_hash as current_evaluator, PROTOCOL_ID
    wd = Path(wd)
    cfgp = wd / "meta/config.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    manifest = read(wd / "meta/mix20h_run_manifest.json")
    if manifest.get("campaign_id") != CAMPAIGN:
        raise ValueError("not an M20 runtime manifest")
    if manifest.get("original_run_id", manifest.get("run_id")) != wd.name:
        raise ValueError("run identity mismatch")
    if cfg.get("num_iter") != 50000 or int(read(wd / "last_meta.json").get("step", -1)) != 50000:
        raise ValueError("training is not exact50K complete")
    if not (wd / "meta/finished_at.txt").is_file():
        raise ValueError("training finished marker missing")
    if not (wd / "checkpoint_metrics.csv").is_file():
        raise FileNotFoundError("checkpoint_metrics.csv missing")
    rows = selector.load_rows(str(wd))
    steps = [r["step"] for r in rows]
    if len(steps) != 50 or set(steps) != set(GRID):
        raise ValueError("GRID1010_50K_v1 missing/duplicate/extra candidate rows")
    if not all(math.isfinite(r["hqnr"]) for r in rows):
        raise ValueError("raw-original grid has nonfinite HQNR")
    evid = evaluator_hash if evaluator_hash is not None else current_evaluator()
    fr = Path(cfg["test_full_feeder_args"]["dataroot"])
    rr = Path(cfg["test_reduced_feeder_args"]["dataroot"])
    if not fr.is_absolute():
        fr = ROOT / fr
    if not rr.is_absolute():
        rr = ROOT / rr
    # _rr uses the official GT path, so the inference input must be byte-identical.
    from tools.eval_dlpan import GT_H5
    rr_gt = ROOT / GT_H5["wv3"]
    hashes = dict(fr=sha(fr), rr=sha(rr))
    if hashes["rr"] != sha(rr_gt):
        raise ValueError("RR config dataset differs from official evaluator GT dataset")
    if "mat20" not in str(fr):
        raise ValueError("FR must be the paper mat20 dataset, not deployment H5")
    if hashes["fr"] != sha(ROOT / "data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5"):
        raise ValueError("FR config dataset differs from the official paper mat20 dataset")
    cfgsha = sha(cfgp)
    state = read(wd / "selector_state_raw.json")
    if (state.get("protocol_id") != PROTOCOL_ID or state.get("fr_h5_sha256") != hashes["fr"]
            or state.get("evaluator_hash") != evid or state.get("n_scenes") != 20):
        raise ValueError("raw grid protocol/data/evaluator identity mismatch")
    bindings = []
    for row in sorted(rows, key=lambda item: item["step"]):
        candidate = wd / "candidates" / f"step-{row['step']}"
        binding = read(candidate / "mix20h_identity.json")
        expected = dict(run_id=wd.name, step=row["step"], checkpoint_sha256=sha(candidate / "model.safetensors"),
                        fr_h5_sha256=hashes["fr"], evaluator_hash=evid, eval_mode="A_ON", raw_hqnr=row["hqnr"])
        for key, value in expected.items():
            if binding.get(key) != value:
                raise ValueError(f"candidate {row['step']} identity mismatch: {key}")
        if binding.get("evaluation_config_sha256", binding.get("config_sha256")) != cfgsha:
            raise ValueError(f"candidate {row['step']} config identity mismatch")
        if binding.get("precision") not in ("no", "fp32"):
            raise ValueError(f"candidate {row['step']} unsupported training precision")
        bindings.append(expected)
    return dict(run_id=wd.name, selector=SELECTOR, grid_id="GRID1010_50K_v1", grid_identity=digest(bindings),
                dataset_hashes=hashes, evaluator=selector.evaluator_identity(strict=True), raw_evaluator_hash=evid,
                config_sha256=cfgsha, eval_mode="A_ON", precision="fp32"), rows, cfg, manifest


def target_complete(target, identity):
    if target.get("completion_identity") != identity or target.get("selector") != SELECTOR:
        return False
    status = target.get("target_status")
    if target.get("threshold", selector.THRESHOLD) != selector.THRESHOLD:
        return False
    if status == "no_eligible":
        return target.get("n_candidates") == 50 and target.get("n_eligible") == 0 and target.get("target") is None
    chosen = target.get("target") or {}
    return (status == "official" and target.get("official") is True and target.get("n_unevaluated") == 0
            and target.get("n_eligible", 0) == target.get("n_official_evaluated")
            and finite_metrics(chosen, selector.RR_KEYS + ("hqnr",))
            and chosen["hqnr"] >= selector.THRESHOLD)


def exact_complete(result, identity):
    return (result.get("completion_identity") == identity and result.get("step") == 50000
            and result.get("eval_mode") == "A_ON" and result.get("official_complete") is True
            and finite_metrics(result, selector.RR_KEYS + ("hqnr",)))


def exact_fr(wd, cfg, identity, device):
    """FR A_ON inference cache is hash-bound; an arbitrary saved .mat is not trusted."""
    import h5py
    import numpy as np
    import torch
    from scipy.io import loadmat, savemat
    from tools import eval_fr_paperset as efp
    from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
    wd = Path(wd)
    ck = "candidates/step-50000"
    ident = dict(identity, checkpoint_sha256=selector.sha256_file(wd / ck / "model.safetensors"), step=50000)
    mat = wd / "results/full_exact50k_AON_mat20.mat"
    side = mat.with_suffix(".json")
    old = read(side) if side.exists() else {}
    cached = old.get("identity") == ident and mat.exists() and old.get("mat_sha256") == selector.sha256_file(mat)
    if cached:
        sr = loadmat(mat)["sr"]
    else:
        model, fwd, how = efp.build(cfg, str(wd), ck)
        model = model.to(device).eval()
        if getattr(model, "aligner", None) is None:
            raise ValueError("A_ON exact50K evaluation has no aligner")
        feeder = efp.import_class(cfg["feeder"])(**cfg["test_full_feeder_args"])
        outputs = []
        with torch.no_grad():
            for index in range(len(feeder)):
                lms, ms, lpan, pan = (value.unsqueeze(0).to(device) for value in feeder[index])
                out = fwd(pan, lpan, ms, lms)
                outputs.append(((out.clip(-1, 1).float().cpu().numpy() + 1) / 2 * float(feeder.max_pixel))[0])
        sr = np.stack(outputs)
        savemat(mat, dict(sr=sr))
        write(side, dict(identity=ident, mat_sha256=selector.sha256_file(mat), forward=how, evaluated_at=utcnow()))
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    with h5py.File(cfg["test_full_feeder_args"]["dataroot"]) as dataset:
        lms = np.asarray(dataset["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(dataset["pan"], dtype=np.float64)[:, 0]
    if len(sr) != 20 or len(lms) != 20:
        raise ValueError("FR paper mat20 must contain exactly 20 scenes")
    scene = []
    for index in range(20):
        fused = sr[index].astype(np.float64).transpose(1, 2, 0)
        dl = d_lambda_k(fused, lms[index], "wv3", 4, 32, wald)
        ds = d_s(fused, lms[index], pan[index], 4, 32, wald)
        scene.append(dict(hqnr=float((1-dl)*(1-ds)), d_lambda=float(dl), d_s=float(ds)))
    return dict(hqnr=float(np.mean([s["hqnr"] for s in scene])), per_scene=scene, n=20,
                checkpoint_sha256=ident["checkpoint_sha256"], mat=str(mat.relative_to(wd)),
                mat_sha256=selector.sha256_file(mat), cached=cached)


def legacy_consistency(wd, rows, device, deadline):
    """Check the legacy selected checkpoint really binds to the same grid and FR."""
    from tools import eval_fr_paperset as efp
    step = int(read(wd / "best_hqnr_meta.json")["step"])
    by_step = {r["step"]: r for r in rows}
    if step not in by_step:
        raise ValueError("legacy checkpoint is outside the 50-point grid")
    selected = selector.sha256_file(wd / "best_hqnr/model.safetensors")
    if selected != selector.sha256_file(wd / f"candidates/step-{step}/model.safetensors"):
        raise ValueError("legacy best/grid checkpoint bytes differ")
    if not before_deadline(deadline):
        raise TimeoutError("deadline: legacy FR consistency pending")
    wald = efp.load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    result, how = efp.run_one(wd.name, None, wald, device, False)
    if result is None or not math.isfinite(result.get("hqnr", float("nan"))):
        raise ValueError(f"legacy official FR unavailable: {how}")
    delta = abs(result["hqnr"] - by_step[step]["hqnr"])
    if delta > 1e-7:
        raise ValueError(f"legacy official FR/grid H mismatch: {delta}")
    return dict(step=step, checkpoint_sha256=selected, csv_hqnr=by_step[step]["hqnr"],
                official_fr_hqnr=result["hqnr"], abs_diff=delta, tolerance=1e-7)


def process(run, device="cuda", upload=False, deadline=None):
    wd = ROOT / "work_dir" / run
    if Path(run).name != run or not wd.is_dir():
        raise ValueError("run must be an existing canonical work_dir basename")
    started = time.monotonic()
    status = dict(run_id=run, selector=SELECTOR, training_complete=False, raw_grid_complete=False,
                  official_target_complete=False, exact50k_complete=False, sheet_uploaded=False,
                  status="EVAL_PENDING", started_at_utc=utcnow())
    path = wd / "results" / STATUS_FILE
    previous = read(path) if path.exists() else {}
    try:
        # Expose training completion independently even if the grid is damaged.
        status["training_complete"] = (read(wd / "last_meta.json").get("step") == 50000
                                       and (wd / "meta/finished_at.txt").is_file())
        identity, rows, cfg, manifest = validate_grid(wd)
        deadline = bound_deadline(manifest, deadline)
        status.update(raw_grid_complete=True, completion_identity=identity)
        target_path = wd / "results" / TARGET_FILE
        target = read(target_path) if target_path.exists() else {}
        if (target_complete(target, identity) and target.get("h_consistency")
                and target["h_consistency"].get("abs_diff", 1) <= 1e-7):
            # Fully hash-bound cached consistency allows network-only retry after deadline.
            status["h_consistency"] = target["h_consistency"]
        else:
            status["h_consistency"] = legacy_consistency(wd, rows, device, deadline)
        if not target_complete(target, identity):
            cmd = [sys.executable, str(ROOT / "tools/qrecon24_select.py"), run, "--official",
                   "--threshold", "0.9585", "--selector", SELECTOR, "--strict-identity", "--device", device]
            if deadline:
                cmd += ["--deadline-utc", deadline]
            proc = subprocess.run(cmd, cwd=ROOT, check=False)
            if proc.returncode:
                raise RuntimeError(f"v2 selector exit {proc.returncode}")
            target = read(target_path)
            target["completion_identity"] = identity
            target["h_consistency"] = status["h_consistency"]
            # Legacy selector JSON permits NaN in proxy-only fields; leave its format unchanged.
            with open(target_path, "w") as stream:
                json.dump(target, stream, indent=2, ensure_ascii=False)
        status.update(official_target_complete=target_complete(target, identity), target_status=target.get("target_status"))
        exact_path = wd / "results/exact50k_official_AON.json"
        exact = read(exact_path) if exact_path.exists() else {}
        if not exact_complete(exact, identity):
            if not before_deadline(deadline):
                raise TimeoutError("deadline: exact50K pending")
            rr, why = selector.official_rr(run, str(wd), 50000, device, identity["evaluator"])
            if rr is None or not finite_metrics(rr, selector.RR_KEYS):
                raise RuntimeError(f"exact50K official RR incomplete: {why}")
            if not before_deadline(deadline):
                raise TimeoutError("deadline: exact50K FR pending")
            fr = exact_fr(wd, cfg, identity, device)
            csv_h = next(r["hqnr"] for r in rows if r["step"] == 50000)
            if abs(fr["hqnr"] - csv_h) > 1e-7:
                raise ValueError("exact50K official FR/grid HQNR mismatch")
            if rr["identity"]["checkpoint_sha256"] != fr["checkpoint_sha256"]:
                raise ValueError("exact50K RR/FR checkpoint mismatch")
            exact = dict(run_id=run, step=50000, eval_mode="A_ON", precision="fp32", official_complete=True,
                         completion_identity=identity, checkpoint_sha256=fr["checkpoint_sha256"], hqnr=fr["hqnr"],
                         **{key: rr[key] for key in selector.RR_KEYS}, rr=rr, fr=fr, evaluated_at_utc=utcnow())
            write(exact_path, exact)
        status["exact50k_complete"] = exact_complete(exact, identity)
        if not status["official_target_complete"]:
            status["status"] = "INCOMPLETE_OFFICIAL_RR"
        else:
            status["status"] = "EVAL_COMPLETE_UPLOAD_PENDING"
            # Persist completion first. Upload failures must never invalidate evaluation.
            status["evaluation_seconds"] = float(previous.get("evaluation_seconds", 0)) + time.monotonic()-started
            write(path, status)
            if upload:
                try:
                    spec = importlib.util.spec_from_file_location("_mix20_upload", ROOT / "gspread/mix20h_upload.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    receipt = module.upload_run(run)
                    status.update(sheet_uploaded=True, status="COMPLETE", upload_receipt=receipt)
                except Exception as exc:
                    status["upload_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status.update(status="EVAL_INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    elapsed = time.monotonic()-started
    status.update(evaluation_seconds=float(previous.get("evaluation_seconds", 0)) + elapsed,
                  evaluation_attempt_seconds=elapsed, updated_at_utc=utcnow())
    write(path, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["official_target_complete"] and status["exact50k_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    lock_dir = ROOT / "work_dir/_qrc24_mix20h"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / ".postrun.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("M20 evaluation writer busy; retry postrun only")
            return 2
        return process(args.run, args.device, args.upload, args.deadline_utc)


if __name__ == "__main__":
    sys.exit(main())
