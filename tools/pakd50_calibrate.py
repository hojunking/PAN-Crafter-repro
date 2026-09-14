#!/usr/bin/env python
"""PAKD50 공통 calibration package (계획 §5): τR = T0 Teacher 오차의 train-subset 중앙값(max 1e-6), λE = J0 seed1234 exact50K pilot 에서 plain GT L1 과 signed edge 의 출력 gradient RMS 비 × 0.05.
trainer 와 같은 함수(kdv.calibration.calibrate_rec / calibrate_lambda, calibration_batches: train 고정 3072 patch, seed 1234, batch 48, 증강 off) 를 쓴다. 산출 work_dir/_pakd50/calibration_resolved.json (세 서버가 같은 숫자를 config 로 받는다).

    python tools/pakd50_calibrate.py --tau                # T0 만 필요 (stage 1 전)
    python tools/pakd50_calibrate.py --lambda-e           # J0 S1234 exact50K(last) 가 이 서버에 있어야 한다 (s1)"""
import argparse, hashlib, json, os, sys, time, types
import torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools import gen_pakd50_configs as G
from kdv.calibration import calibration_batches, calibrate_rec, calibrate_lambda
from kdv.teacher_assets import load_run_model, freeze, sha256_file
from main import import_class

OUT = os.path.join(ROOT, G.CAL_PATH)


def args_from_config(cfg):
    a = types.SimpleNamespace(**{k: v for k, v in cfg.items() if k not in ("kdv",)}); return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", action="store_true"); ap.add_argument("--lambda-e", action="store_true"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--server", default=None)
    a = ap.parse_args(); dev = torch.device(a.device); srv = a.server or open(os.path.join(ROOT, "gspread", "server.txt")).read().strip()
    cal = json.load(open(OUT)) if os.path.exists(OUT) else dict(campaign_id=G.CAMPAIGN_ID, package="T0", max_train_patches=3072, seed=1234, batch=48, geometric_augmentation=False)
    tpl = yaml.safe_load(open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml"))); args = args_from_config(tpl); args.batch_size = 48
    batches, man = calibration_batches(args, 3072, 1234, 48); cal["patch_manifest"] = man; cal["patch_manifest_sha256"] = hashlib.sha256(json.dumps(man, sort_keys=True, default=str).encode()).hexdigest()
    Model = import_class(tpl["model"])
    if a.tau:
        t0, tman = load_run_model(G.t0_dir(srv), G.T0_TAG, Model); freeze(t0); t0.to(dev)
        r = calibrate_rec(t0, batches, dev, 1e-6)
        cal.update(tau_R=float(r["tau_R"]), tau_source=dict(teacher_run=tman["run"], tag=G.T0_TAG, file_sha256=tman["file_sha256"], tensors_sha256_16=tman["tensors_sha256_16"], selected_update=(tman.get("tag_meta") or {}).get("step"), e_T_quantiles=r["e_T"], rule="median e_T over calibration patches, floor 1e-6 (§5.1)"),
                   tau_computed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), tau_server=srv)
        print(f"[cal] τR = {cal['tau_R']:.6f} (T0 {tman['file_sha256'][:16]} step {cal['tau_source']['selected_update']}; e_T p50 {r['e_T'].get('p50')})")
    if a.lambda_e:
        prun = G.pilot_run(); pdir = os.path.join(ROOT, "work_dir", prun)
        lm = os.path.join(pdir, "last_meta.json")
        if not (os.path.exists(lm) and json.load(open(lm)).get("step") == 50000):
            sys.exit(f"!! pilot {prun} 의 exact-50K last 가 없다 (계획 §5.2: J0 seed1234 exact50K)")
        pm, pman = load_run_model(pdir, "last", Model); freeze(pm); pm.to(dev)
        r = calibrate_lambda(pm, batches, "edge", 5, 0.05, dev)
        if r.get("degenerate"):
            sys.exit(f"!! λE calibration degenerate: {r}")
        cal.update(lambda_E=float(r["lambda_V"]), lambda_E_source=dict(pilot_run=prun, tag="last", file_sha256=pman["file_sha256"], tensors_sha256_16=pman["tensors_sha256_16"], step=(pman.get("tag_meta") or {}).get("step"), r_grad=0.05,
                                                                          rule="0.05 × median_b RMS(∂L0/∂Z)/RMS(∂LE/∂Z) on the calibration patches, Z detached (§5.2)", raw=r), lambda_computed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), lambda_server=srv)
        print(f"[cal] λE = {cal['lambda_E']:.6g} (pilot {prun}/last sha {pman['file_sha256'][:16]})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True); cal["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S"); json.dump(cal, open(OUT, "w"), indent=1, ensure_ascii=False, default=str)
    print("  ->", os.path.relpath(OUT, ROOT), {k: cal.get(k) for k in ("tau_R", "lambda_E")})


if __name__ == "__main__":
    main()
