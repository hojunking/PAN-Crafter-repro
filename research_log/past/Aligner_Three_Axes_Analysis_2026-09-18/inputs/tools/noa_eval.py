#!/usr/bin/env python
"""전 서버 Student 평가 — 같은 checkpoint의 A_ON / NOA(A_BYPASS_RAW) 쌍 RR·FR 평가
(계획 research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §3–§6, protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918).

    python tools/noa_eval.py --capture-cohort                 # 이 서버의 완료 Student 목록을 고정 (cohort.json)
    python tools/noa_eval.py --status                         # 진행 상황 · N_* 보고표
    python tools/noa_eval.py --run <run> [--modes A_ON,A_BYPASS_RAW] [--device cuda]
    python tools/noa_eval.py --all [--limit N]                # cohort 전체 (완료된 것은 건너뛴다)
    python tools/noa_eval.py --sanity [--run <run>]           # §4.4 사전 검사만 (호출 0회·state 불변·ZERO 대조)

학습은 하지 않는다. 기존 산출물(reduced_best_hqnr.mat, fr_mat20.json, best_* 메타)을 덮어쓰지 않고 아래에만 쓴다.

    work_dir/<run>/results/aligner_eval_v2/<checkpoint_sha16>/<mode>/{rr_metrics.json, fr_metrics.json, per_scene_rr.csv, per_scene_fr.csv, identity.json, sr_rr.mat, sr_fr.mat}
    work_dir/_eval_phase/{cohort.json, records/<run>.json}

지표는 기존 공식 경로 그대로다: RR 은 gspread_upload._rr(= tools/eval_dlpan 경로, crop 20:-21, peak 2047, SCC=SCC.m, SSIM=Gaussian 11×11, RMSE/CC 포함),
FR 은 tools/metrics/eval_fr 의 d_lambda_k·d_s 와 pa/evalviews.raw_views(V64·fSCC) — 논문 세트 .mat20. 모드별로 crop·reference·clamp 를 바꾸지 않는다(§5.1).
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kdv.eval_modes import MODES, MODE_LABEL, CallCounter, forward_mode, state_hash, expected_calls  # noqa: E402
from tools.qrecon24_version_audit import audit_run  # noqa: E402

PROTOCOL_ID = "PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918"
EVAL_DIR = "aligner_eval_v2"
PHASE_DIR = os.path.join("work_dir", "_eval_phase")
DEFAULT_MODES = ("A_ON", "A_BYPASS_RAW")
FR_H5 = os.path.join("data", "PanCollection", "WV3", "full_examples_mat20", "test_wv3_OrigScale_mat20.h5")
LEDGER = os.path.join("work_dir", "_qrecon24_budget", "ledger.json")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path)); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def _json(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def sha256_file(p, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def server_id():
    from tools.gen_pakd50_configs import server_id as sid
    return sid(open(os.path.join(ROOT, "gspread", "server.txt")).read())


def sheet_checkpoint(run):
    """현재 Sheet 본 행이 가리키는 checkpoint (§3.4): kdv/pa 계열의 best_hqnr(= best_raw alias), 없으면 best_val/best_reduced."""
    wd = os.path.join(ROOT, "work_dir", run)
    for ck in ("best_hqnr", "best_val", "best_reduced"):
        f = os.path.join(wd, ck, "model.safetensors")
        if os.path.exists(f):
            meta = _json(os.path.join(wd, f"{ck}_meta.json")) or _json(os.path.join(wd, "best_raw_meta.json")) or {}
            return ck, f, meta
    return None, None, {}


def completed_runs(prefix="PAKD50_QRC24_"):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "work_dir", prefix + "*"))):
        r = os.path.basename(d)
        if os.path.exists(os.path.join(d, "results", "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(d, "results", "full_best_hqnr.mat")):
            out.append(r)
    return out


def running_runs():
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    out = []
    for l in ps.splitlines():
        if "main.py" in l and "--config" in l and "PAKD50_" in l:
            out.append(os.path.basename(l.split("--config")[1].strip().split()[0]).replace(".yaml", ""))
    return sorted(set(out))


def capture_cohort(force=False):
    """§3.2 유한한 로컬 cohort 고정: 이 시점의 완료 run + 진행 중 run 1건(pending_current_run_id). 이후 새 학습은 이 phase 대상이 아니다."""
    p = os.path.join(ROOT, PHASE_DIR, "cohort.json")
    if os.path.exists(p) and not force:
        c = _json(p); print(f"[noa] cohort 이미 고정: {c['id']} · {len(c['completed_run_keys'])} run (--capture-cohort --force 로 다시 잡는다)"); return c
    srv = server_id(); runs = completed_runs(); run_now = running_runs()
    recs = []
    for r in runs:
        ck, ckf, meta = sheet_checkpoint(r); va = audit_run(r, write=False)
        recs.append(dict(original_run_id=r, source_train_server=srv, source_selector=ck, checkpoint_step=meta.get("step"),
                         checkpoint_sha256=(sha256_file(ckf) if ckf else None), definition=va["verdict"],
                         status=("pending" if va["verdict"] == "single_definition_factor1" else "definition_unverified"),
                         variant_kind=("control" if any(t in r for t in ("_UNIF", "_SHUF", "_FREEZE", "ALPHA0", "BETA0")) else "main")))
    c = dict(protocol_id=PROTOCOL_ID, id=f"{srv}_{time.strftime('%Y%m%d-%H%M%S')}", server=srv, captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
             completed_run_keys=[r["original_run_id"] for r in recs], pending_current_run_id=(run_now[0] if run_now else None),
             include_future_main_runs=False, records=recs)
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(c, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"[noa] cohort {c['id']}: 완료 {len(recs)} run · 진행 중 {c['pending_current_run_id'] or '없음'} → {os.path.relpath(p, ROOT)}")
    for r in recs:
        if r["status"] != "pending":
            print(f"   !! {r['original_run_id']}: {r['definition']} — 별도 cohort/제외 대상 (§3.1)")
    return c


def eval_dir(run, sha, mode):
    return os.path.join(ROOT, "work_dir", run, "results", EVAL_DIR, (sha or "nosha")[:16], mode)


def identity_of(run, ck, ckf, mode, cfgp, device):
    ev = dict(eval_rr=sha256_file(os.path.join(ROOT, "tools", "metrics", "eval_rr.py")), eval_fr=sha256_file(os.path.join(ROOT, "tools", "metrics", "eval_fr.py")),
              eval_dlpan=sha256_file(os.path.join(ROOT, "tools", "eval_dlpan.py")), gspread_upload=sha256_file(os.path.join(ROOT, "gspread", "gspread_upload.py")),
              eval_modes=sha256_file(os.path.join(ROOT, "kdv", "eval_modes.py")), pa_model=sha256_file(os.path.join(ROOT, "pa", "model.py")))
    fr5 = os.path.join(ROOT, FR_H5)
    return dict(protocol_id=PROTOCOL_ID, run=run, source_selector=ck, checkpoint_sha256=sha256_file(ckf), config_sha256=sha256_file(cfgp),
                fr_h5_sha256=(sha256_file(fr5) if os.path.exists(fr5) else None), eval_mode=mode, mode_label=MODE_LABEL[mode],
                evaluator=ev, device=str(device), full_frame_unet=True, strict_load=True, trained=False)


def _infer(run, ck, mode, device, cfg, split):
    """split 'rr'|'fr' 로 전체 테스트셋 추론 → (sr [N,C,H,W] DN, how, calls, state_ok, extras). 학습·grad 없음."""
    import torch
    from main import import_class
    efp = _load(os.path.join("tools", "eval_fr_paperset.py"), "_efp_noa")
    wd = os.path.join(ROOT, "work_dir", run)
    m, _fwd, how = efp.build(cfg, wd, ck)
    dev = torch.device(device); m = m.to(dev).eval()
    h0 = state_hash(m)
    fa = dict(cfg["test_reduced_feeder_args"] if split == "rr" else cfg["test_full_feeder_args"])
    if split == "fr":
        fa["dataroot"] = os.path.join(ROOT, FR_H5)
    ds = import_class(cfg["feeder"])(**fa); mp = float(ds.max_pixel); srs = []; extras = []
    with CallCounter(m) as cc, torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            lms, ms, lpan, pan = item[1:] if split == "rr" else item        # RR feeder: (gt, lms, ms, lpan, pan) · FR feeder: (lms, ms, lpan, pan)
            t = lambda x: x.unsqueeze(0).to(dev)
            y, ex = forward_mode(m, t(pan), t(ms), t(lpan), mode, return_extra=True)
            srs.append(((y.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * mp)[0]); extras.append(ex)
    ok = (state_hash(m) == h0) and all(p.grad is None for p in m.parameters())
    return np.stack(srs), how, cc.as_dict(), ok, extras


def rr_metrics(mat, cfg):
    gu = _load(os.path.join("gspread", "gspread_upload.py"), "_gu_noa")
    droot = cfg["test_reduced_feeder_args"]["dataroot"]; dsn = next((s for s in ("wv3", "qb", "gf2", "wv2") if f"test_{s}_" in droot), "wv3")
    r = gu._rr(mat, dsn)
    return dict(ergas=r.get("ergas"), sam=r.get("sam"), psnr=r.get("psnr"), ssim=r.get("ssim"), scc=r.get("scc"), q8=r.get("q2n", r.get("q8")),
                rmse=r.get("rmse"), cc=r.get("cc"), ergas_sd=r.get("ergas_sd"), sam_sd=r.get("sam_sd"), q8_sd=r.get("q2n_sd"), dataset=dsn, n=20)


def fr_metrics(sr):
    """논문 .mat20 FR: d_lambda_k·d_s·HQNR(장면별 곱의 평균) + V64·fSCC (기존 eval_fr_paperset 와 같은 부품)."""
    import h5py
    from tools.metrics.eval_fr import d_lambda_k, d_s, load_dlpan
    from pa.evalviews import raw_views
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    with h5py.File(os.path.join(ROOT, FR_H5)) as f:
        lms = np.asarray(f["lms"], np.float64).transpose(0, 2, 3, 1); pan = np.asarray(f["pan"], np.float64)[:, 0]
    dl, ds_ = [], []
    for i in range(len(sr)):
        s = sr[i].astype(np.float64).transpose(1, 2, 0)
        dl.append(d_lambda_k(s, lms[i], "wv3", 4, 32, wald)); ds_.append(d_s(s, lms[i], pan[i], 4, 32, wald))
    dl, ds_ = np.array(dl), np.array(ds_); h = (1 - dl) * (1 - ds_)
    out = dict(hqnr_raw=float(h.mean()), hqnr_sd=float(h.std(ddof=1)), d_lambda=float(dl.mean()), d_lambda_sd=float(dl.std(ddof=1)),
               d_s=float(ds_.mean()), d_s_sd=float(ds_.std(ddof=1)), n=int(len(sr)),
               per_scene_hqnr=[float(x) for x in h], per_scene_d_lambda=[float(x) for x in dl], per_scene_d_s=[float(x) for x in ds_])
    try:
        srl = [sr[i].astype(np.float64).transpose(1, 2, 0) for i in range(len(sr))]
        v = raw_views(srl, lms, pan, "wv3", wald, 2047.0) if raw_views.__code__.co_argcount >= 6 else None
        if isinstance(v, dict):
            out.update({k: v[k] for k in v if k.startswith(("hqnr_v64", "fscc"))})
    except Exception as e:                                                             # noqa
        out["aux_note"] = f"V64/fSCC 생략: {e!r}"
    return out


def evaluate_run(run, modes=DEFAULT_MODES, device="cuda", force=False, splits=("rr", "fr")):
    import yaml
    from scipy.io import savemat
    wd = os.path.join(ROOT, "work_dir", run); cfgp = os.path.join(wd, "meta", "config.yaml")
    ck, ckf, meta = sheet_checkpoint(run)
    if ck is None:
        return dict(run=run, status="missing_checkpoint")
    cfg = yaml.safe_load(open(cfgp)); sha = sha256_file(ckf); rec = dict(run=run, protocol_id=PROTOCOL_ID, source_selector=ck, checkpoint_step=meta.get("step"),
                                                                        checkpoint_sha256=sha, modes={}, evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for mode in modes:
        od = eval_dir(run, sha, mode); os.makedirs(od, exist_ok=True)
        ident = identity_of(run, ck, ckf, mode, cfgp, device); ip = os.path.join(od, "identity.json")
        prev = _json(ip)
        done = (not force) and prev is not None and all(prev.get(k) == ident[k] for k in ("checkpoint_sha256", "config_sha256", "fr_h5_sha256", "eval_mode")) and prev.get("evaluator") == ident["evaluator"] \
            and all(os.path.exists(os.path.join(od, f)) for f in ("rr_metrics.json", "fr_metrics.json"))
        if done:
            rec["modes"][mode] = dict(rr=_json(os.path.join(od, "rr_metrics.json")), fr=_json(os.path.join(od, "fr_metrics.json")), cached=True, calls=prev.get("calls"))
            print(f"   {MODE_LABEL[mode]:<9} 캐시 재사용 (identity 일치)"); continue
        t0 = time.time(); mrec = dict(cached=False)
        if "rr" in splits:
            sr, how, calls, ok, _ = _infer(run, ck, mode, device, cfg, "rr")
            exp = expected_calls(mode, n_forward=len(sr))
            for k, v in exp.items():
                if v is not None and calls[k] != v:
                    raise SystemExit(f"!! {run} {mode}: {k} {calls[k]} ≠ 기대 {v} (계획 V02/V03)")
            if not ok:
                raise SystemExit(f"!! {run} {mode}: 평가 중 모델 state 가 변했다 (계획 V06)")
            mat = os.path.join(od, "sr_rr.mat"); savemat(mat, dict(sr=sr))
            m_rr = rr_metrics(mat, cfg); json.dump(m_rr, open(os.path.join(od, "rr_metrics.json"), "w"), indent=1)
            mrec["rr"] = m_rr; mrec["forward"] = how; ident["calls_rr"] = calls
        if "fr" in splits:
            sr, how, calls, ok, extras = _infer(run, ck, mode, device, cfg, "fr")
            exp = expected_calls(mode, n_forward=len(sr))
            for k, v in exp.items():
                if v is not None and calls[k] != v:
                    raise SystemExit(f"!! {run} {mode}: FR {k} {calls[k]} ≠ 기대 {v}")
            if not ok:
                raise SystemExit(f"!! {run} {mode}: FR 평가 중 state 변경 (계획 V06)")
            savemat(os.path.join(od, "sr_fr.mat"), dict(sr=sr))
            m_fr = fr_metrics(sr); json.dump(m_fr, open(os.path.join(od, "fr_metrics.json"), "w"), indent=1)
            with open(os.path.join(od, "per_scene_fr.csv"), "w") as f:
                f.write("scene,hqnr,d_lambda,d_s,delta_dy,delta_dx\n")
                for i in range(m_fr["n"]):
                    d = (extras[i].get("delta") or [float("nan")] * 2)
                    f.write(f"{i},{m_fr['per_scene_hqnr'][i]:.6f},{m_fr['per_scene_d_lambda'][i]:.6f},{m_fr['per_scene_d_s'][i]:.6f},{d[0]:.6f},{d[1]:.6f}\n")
            mrec["fr"] = m_fr; mrec["forward"] = how; ident["calls_fr"] = calls
        ident["seconds"] = round(time.time() - t0, 1); ident["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); ident["calls"] = ident.get("calls_rr") or ident.get("calls_fr")
        json.dump(ident, open(ip, "w"), indent=1, ensure_ascii=False)
        mrec["seconds"] = ident["seconds"]; rec["modes"][mode] = mrec
        print(f"   {MODE_LABEL[mode]:<9} RR ERGAS {mrec.get('rr', {}).get('ergas', float('nan')):.4f} · FR HQNR {mrec.get('fr', {}).get('hqnr_raw', float('nan')):.4f} · {ident['seconds']:.0f}s · calls {ident['calls']}")
        ledger_add(f"noaeval_{run}_{mode}", ident["seconds"], f"{PROTOCOL_ID} {mode} ({ck})")
    rec.update(deltas_vs_on(rec)); rec["legacy_on_check"] = legacy_on_check(run, rec)
    rp = os.path.join(ROOT, PHASE_DIR, "records"); os.makedirs(rp, exist_ok=True)
    json.dump(rec, open(os.path.join(rp, run + ".json"), "w"), indent=1, ensure_ascii=False)
    return rec


def legacy_on_check(run, rec):
    """§4.5: 이번 paired A_ON 이 기존 Sheet 본 행(공식 저장값) 과 같은가. 다르면 legacy_on_discrepancy 로 남기고 '기존 열과 직접 비교 가능' 이라고 쓰지 않는다."""
    on = rec["modes"].get("A_ON") or {}
    wd = os.path.join(ROOT, "work_dir", run)
    fr_old = (_json(os.path.join(wd, "results", "fr_mat20.json")) or {}).get("hqnr")
    sel = _json(os.path.join(wd, "results", "qrecon24_target_selection.json")) or {}
    rr_old = ((sel.get("target") or sel.get("proxy_target") or {}) or {}).get("ergas")
    out = dict(stored_fr_hqnr=fr_old, stored_rr_ergas=rr_old, paired_fr_hqnr=(on.get("fr") or {}).get("hqnr_raw"), paired_rr_ergas=(on.get("rr") or {}).get("ergas"))
    d = []
    if fr_old is not None and out["paired_fr_hqnr"] is not None:
        out["fr_abs_diff"] = abs(fr_old - out["paired_fr_hqnr"]); d.append(out["fr_abs_diff"] > 1e-6)
    if rr_old is not None and out["paired_rr_ergas"] is not None:
        out["rr_abs_diff"] = abs(rr_old - out["paired_rr_ergas"]); d.append(out["rr_abs_diff"] > 1e-6)
    out["status"] = "legacy_on_discrepancy" if any(d) else ("match" if d else "no_stored_reference")
    return out


def deltas_vs_on(rec):
    """§5.2 부호 규약: ΔE = E_NOA − E_ON, ΔH = H_NOA − H_ON. 반올림 표시값이 아니라 원본에서 계산."""
    on, noa = rec["modes"].get("A_ON"), rec["modes"].get("A_BYPASS_RAW")
    if not (on and noa and on.get("rr") and noa.get("rr")):
        return {}
    d = dict(delta_rr_ergas=noa["rr"]["ergas"] - on["rr"]["ergas"])
    if on.get("fr") and noa.get("fr"):
        d["delta_fr_hqnr_raw"] = noa["fr"]["hqnr_raw"] - on["fr"]["hqnr_raw"]
        d["noa_joint_pass"] = bool(noa["fr"]["hqnr_raw"] >= 0.9585 and noa["rr"]["ergas"] < 2.040)
        d["on_joint_pass"] = bool(on["fr"]["hqnr_raw"] >= 0.9585 and on["rr"]["ergas"] < 2.040)
    return d


def ledger_add(key, sec, note):
    import fcntl
    lp = os.path.join(ROOT, LEDGER)
    if not os.path.exists(lp):
        return
    with open(lp + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(lp)); e = d.setdefault("entries", {}); prev = e.get(key, {})
        e[key] = dict(kind="diag", hours=float(prev.get("hours") or 0.0) + sec / 3600.0, runs=int(prev.get("runs", 0)) + 1, note=note, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = lp + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, lp)


def sanity(run, device="cuda"):
    """§4.4 사전 검사: 한 장면으로 네(또는 세) 모드의 호출 횟수·state 불변·ZERO−NOA 차이를 기록한다."""
    import torch, yaml
    from main import import_class
    efp = _load(os.path.join("tools", "eval_fr_paperset.py"), "_efp_sanity")
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    ck, ckf, meta = sheet_checkpoint(run)
    m, _f, how = efp.build(cfg, wd, ck); dev = torch.device(device); m = m.to(dev).eval(); h0 = state_hash(m)
    ds = import_class(cfg["feeder"])(**dict(cfg["test_reduced_feeder_args"]))
    item = ds[0]; lms, ms, lpan, pan = item[1:]
    t = lambda x: x.unsqueeze(0).to(dev)
    out = dict(run=run, checkpoint=ck, step=meta.get("step"), forward=how, device=str(dev), modes={})
    ys = {}
    with torch.no_grad():
        for mode in MODES:
            with CallCounter(m) as cc:
                y, ex = forward_mode(m, t(pan), t(ms), t(lpan), mode, return_extra=True)
            ys[mode] = y.float().cpu(); exp = expected_calls(mode, 1)
            ok = all(v is None or cc.as_dict()[k] == v for k, v in exp.items())
            out["modes"][mode] = dict(calls=cc.as_dict(), expected=exp, calls_ok=bool(ok), delta=ex.get("delta"), extra={k: v for k, v in ex.items() if k != "delta"},
                                      sampler_restored=bool(m.sampler))
    out["state_unchanged"] = bool(state_hash(m) == h0); out["no_grads"] = all(p.grad is None for p in m.parameters())
    d = (ys["A_ZERO_WARP"] - ys["A_BYPASS_RAW"]).abs()
    out["zero_minus_noa"] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()), note="0 이 아닐 수 있다 — bicubic grid_sample 재샘플 차이. bitwise 동일을 가정하지 않는다 (§4.4)")
    d2 = (ys["A_ON"] - ys["A_BYPASS_RAW"]).abs(); out["on_minus_noa"] = dict(max_abs=float(d2.max()), mean_abs=float(d2.mean()))
    p = os.path.join(ROOT, PHASE_DIR, "sanity_%s.json" % run); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("state_unchanged", "no_grads", "zero_minus_noa", "on_minus_noa")}, ensure_ascii=False, indent=1))
    for mode, v in out["modes"].items():
        print(f"   {mode:<14} calls {v['calls']} 기대 {v['expected']} {'OK' if v['calls_ok'] else '!! 불일치'} · sampler 복원 {v['sampler_restored']}")
    print(f"[noa] → {os.path.relpath(p, ROOT)}")
    return out


def status():
    c = _json(os.path.join(ROOT, PHASE_DIR, "cohort.json"))
    if not c:
        print("[noa] cohort 없음 — 먼저 --capture-cohort"); return 1
    recs = {os.path.basename(p)[:-5]: _json(p) for p in glob.glob(os.path.join(ROOT, PHASE_DIR, "records", "*.json"))}
    req = [r for r in c["records"] if r["status"] == "pending"]
    n = lambda f: sum(1 for r in req if f(recs.get(r["original_run_id"]) or {}))
    has = lambda rec, mode, split: bool((rec.get("modes", {}).get(mode) or {}).get(split))
    print(f"[noa] cohort {c['id']} (server {c['server']}, {c['captured_at']})")
    print(f"   N_identified {len(c['records'])} · N_required {len(req)} · 정의 미확정 {len(c['records']) - len(req)}")
    print(f"   N_on_rr {n(lambda r: has(r, 'A_ON', 'rr'))} · N_on_fr {n(lambda r: has(r, 'A_ON', 'fr'))} · N_noa_rr {n(lambda r: has(r, 'A_BYPASS_RAW', 'rr'))} · N_noa_fr {n(lambda r: has(r, 'A_BYPASS_RAW', 'fr'))}")
    print(f"   N_pair_verified {n(lambda r: 'delta_rr_ergas' in r)} · N_uploaded {n(lambda r: r.get('uploaded'))}")
    for r in req:
        rec = recs.get(r["original_run_id"]) or {}
        mark = "완료" if "delta_rr_ergas" in rec else ("부분" if rec else "대기")
        extra = ""
        if "delta_rr_ergas" in rec:
            extra = f"ΔERGAS {rec['delta_rr_ergas']:+.4f} · ΔHQNR {rec.get('delta_fr_hqnr_raw', float('nan')):+.5f}"
        print(f"   {r['original_run_id'].replace('PAKD50_QRC24_', ''):<40} step {str(r['checkpoint_step']):<6} {mark:<4} {extra}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-cohort", action="store_true"); ap.add_argument("--force-cohort", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--run", default=None); ap.add_argument("--all", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES)); ap.add_argument("--device", default="cuda"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits", default="rr,fr")
    a = ap.parse_args()
    if a.capture_cohort or a.force_cohort:
        capture_cohort(force=a.force_cohort); return 0
    if a.status:
        return status()
    modes = tuple(x.strip() for x in a.modes.split(",") if x.strip()); splits = tuple(x.strip() for x in a.splits.split(",") if x.strip())
    if a.sanity:
        run = a.run or (capture_cohort()["completed_run_keys"] or [None])[0]
        if not run:
            print("[noa] 평가할 run 이 없다"); return 1
        sanity(run, a.device); return 0
    targets = [a.run] if a.run else None
    if a.all:
        c = _json(os.path.join(ROOT, PHASE_DIR, "cohort.json")) or capture_cohort()
        targets = [r["original_run_id"] for r in c["records"] if r["status"] == "pending"]
    if not targets:
        print("[noa] --run 또는 --all 이 필요하다"); return 1
    if a.limit:
        targets = targets[:a.limit]
    for i, run in enumerate(targets, 1):
        print(f"[noa] ({i}/{len(targets)}) {run}")
        rec = evaluate_run(run, modes=modes, device=a.device, force=a.force, splits=splits)
        if rec.get("status") == "missing_checkpoint":
            print("   !! checkpoint 없음 — missing_checkpoint 로 남긴다"); continue
        if "delta_rr_ergas" in rec:
            print(f"   Δ(NOA−ON) ERGAS {rec['delta_rr_ergas']:+.4f} · HQNR {rec.get('delta_fr_hqnr_raw', float('nan')):+.5f} · NOA joint_pass {rec.get('noa_joint_pass')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
