#!/usr/bin/env python
"""PALS24 실행 전 metric 검증 gate (계획 §8 G-M0–G-M8) + 기존 NF16 P0/P2/P3 재사용 registry (§11.3) — 새 학습보다 먼저.

    python tools/pals24_metric_gate.py               # GPU 로 G-M1 재평가까지 (NF16 P0/P1/P2/P3 best_hqnr 저장 checkpoint → 같은 evaluator)
    python tools/pals24_metric_gate.py --skip-repro  # CSV/hash 검사만

산출 work_dir/_pals24_campaign/{metric_contract.json, reuse_registry.json, metric_gate_report.json} + work_dir/<run>/results/pals24_repro_best_hqnr.json (기존 fr_mat20.json 은 건드리지 않는다).
판정 기준은 raw_original HQNR → fSCC(같은 checkpoint, 원 PAN 참조) 다. 여기의 tolerance 는 **수치 재현 검사용**이고 방법 판정선(0.0031) 과 다르다 (§7.7)."""
import argparse, csv, hashlib, json, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools.gen_pals24_configs import NF16, REUSED, BACKGROUND, DIAG_REFERENCE, CAMPAIGN_ID, PROTOCOL_ID, DONOR_RUN, LAMBDA, PLAN
from pa.evalviews import evaluator_hash, PROTOCOL_ID as EVAL_PROTOCOL, MARGIN, SUPPORT, MAX_ELIGIBLE_SHIFT
from pa.selector import BestSelector
from kdv.teacher_assets import sha256_file, tensors_sha

CAMP = os.path.join(ROOT, "work_dir", "_pals24_campaign")           # --campaign palsv18 이면 main() 이 _palsv18_campaign 으로 바꾼다
ATOL_REPRO = 1e-6; SHEET_ROUND = 5e-5; TIE_HQNR = 1e-4; METHOD_MARGIN = 0.0031; ATOL_IDENTITY = 1e-9
VIEWS = ("raw_original", "raw_valid", "aligned_valid", "aligned_fixed_v64")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rows_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def sha16(p):
    return sha256_file(p)[:16] if os.path.exists(p) else None


def init_sha16(path):
    """trainer 가 기록한 unet_init_sha256_16 과 같은 방식(state_dict tensors sha) — 파일 sha 도 함께 돌려준다."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        t = tensors_sha(sd)
        t = t[:16] if isinstance(t, str) else t
    except Exception:
        t = None
    return dict(tensors_sha16=t, file_sha16=sha16(path))


def contract():
    """G-M0: 무엇을 어떤 함수로 재는지 고정한다 — 모호한 SCC/fSCC 이름을 남기지 않는다."""
    from tools.eval_fr_paperset import EVAL_VERSION
    import tools.eval_dlpan as ed, tools.metrics.eval_rr as er, pa.evalviews as ev
    cache = os.path.join(ROOT, "work_dir", "_kdv_init_w112_d123", "sha256_cache.json"); mat20 = None
    if os.path.exists(cache):
        for k, v in json.load(open(cache)).items():
            if k.split("|")[0].endswith("full_examples_mat20/test_wv3_OrigScale_mat20.h5"):
                mat20 = v
    sel = BestSelector("raw")
    return dict(campaign_id=CAMPAIGN_ID, plan_protocol_id=PROTOCOL_ID, plan=PLAN, evaluator_protocol_id=EVAL_PROTOCOL, evaluator_hash=evaluator_hash(), fr_eval_version=EVAL_VERSION,
                fr_set=dict(manifest_id="fr_mat20", n_scenes=20, h5="data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5", sha256=mat20),
                hqnr=dict(formula="mean_i (1 - D_lambda_i)(1 - D_s_i) over 20 scenes (mean of scene products, NOT product of means)",
                          d_lambda="tools/metrics/eval_fr.d_lambda_k (genMTF.m-faithful kernel, q2n S=32)", d_s="tools/metrics/eval_fr.d_s (block-UQI S=32, interp23tap(imresize symmetric(PAN,1/4)))",
                          dn="SR clip(-1,1) -> (x+1)/2 * max_pixel, no rounding; lms/pan float64 from h5", python_reimplementation_of_matlab=True, bit_identical_claim=False),
                views=dict(raw_original="original PAN, full frame (primary; best_raw selection)", raw_valid=f"original PAN, fixed V{MARGIN} interior ([{MARGIN}:H-{MARGIN}, {MARGIN}:W-{MARGIN}], block-aligned)",
                           aligned_valid="PAN = W(P, c_hat_self) (the PAN the network consumed), same V64 — diagnostic only", aligned_fixed_v64="PAN = W(P, c_donor) from the frozen N2 donor on the native pair, same V64 — diagnostic only",
                           filters="all filters (MTF, imresize/interp23tap, Sobel) on the full frame first, then crop (phase preserved)", lowres_pan="regenerated from the chosen high-res PAN reference (never mixed)",
                           support=SUPPORT, max_eligible_abs_shift_hr=MAX_ELIGIBLE_SHIFT),
                selector=dict(primary_checkpoint="best_raw (= best_hqnr directory alias)", primary_metric="hqnr raw_original (mat20, 20 scenes)", secondary_metric_key="fscc",
                              secondary_definition="pa/evalviews.fscc_from_maps: Sobel gradient-magnitude maps of SR intensity vs ORIGINAL PAN on the full frame, SCC (utils.SCC_full_numpy definition), then crop; split = FR mat20; reference = original PAN",
                              tie_band_hqnr=sel.tol_h, tie_band_fscc=sel.tol_f, tie_anchor="running maximum", second_tie_preference="later update (pa/selector.BestSelector)",
                              not_used_for_selection=["aligned views", "RR metrics", "closure/response", "ERGAS/SAM (reference only)"]),
                rr_scc_reference_only=dict(sheet_and_reports="tools/eval_dlpan.scc_dlpan — DLPan Quality_Indices/SCC.m port (Sobel magnitude, zero padding)", training_log="utils.SCC_numpy / tools/metrics/eval_rr.scc — reflect padding / Laplacian legacy (relative use only)",
                                           note="RR GT-SCC is a reference metric here; the selector tie-break is FR fSCC (above). Do not rename fSCC as SCC."),
                tolerances=dict(reproduction_atol=ATOL_REPRO, sheet_rounding=SHEET_ROUND, checkpoint_tie_hqnr=TIE_HQNR, identity_atol=ATOL_IDENTITY, method_margin_raw_hqnr=METHOD_MARGIN,
                                method_margin_note="S1 §6 empirical seed-2σ line (0.0031); raw HQNR only; not an equivalence proof; not copied to D_lambda/D_s/PSNR/closure"),
                aggregation_note="scene-level HQNR products are averaged; the covariance between D_lambda and D_s across scenes makes (1-mean Dl)(1-mean Ds) differ — not an evaluator error",
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def reproduce(tag, ckpt, dev, wald):
    """G-M1: 저장 checkpoint 를 같은 evaluator 에 다시 넣는다 (fr_mat20.json 을 덮지 않는다) → results/pals24_repro_<ckpt>.json."""
    import torch, yaml, h5py
    from tools.eval_fr_paperset import build, h5_for, sensor_of
    from tools.metrics.eval_fr import d_lambda_k, d_s
    from main import import_class
    wd = os.path.join(ROOT, "work_dir", tag); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml"))); sensor = sensor_of(cfg); h5 = h5_for(sensor)
    m, fwd, how = build(cfg, wd, ckpt); m = m.to(dev).eval()
    Feeder = import_class(cfg["feeder"]); fargs = dict(cfg["test_full_feeder_args"]); fargs["dataroot"] = h5; ds = Feeder(**fargs); mp = float(ds.max_pixel)
    srs = []
    with torch.no_grad():
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); y = fwd(pan, lpan, ms, lms)
            srs.append(((y.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * mp)[0])
    with h5py.File(h5) as f:
        lms_all = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1); pan_all = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    dl, dsv = [], []
    for i, s in enumerate(srs):
        s = s.astype(np.float64).transpose(1, 2, 0); dl.append(d_lambda_k(s, lms_all[i], sensor, 4, 32, wald)); dsv.append(d_s(s, lms_all[i], pan_all[i], 4, 32, wald))
    dl, dsv = np.array(dl), np.array(dsv); h = (1 - dl) * (1 - dsv)
    out = dict(run=tag, ckpt=ckpt, ckpt_sha256=sha256_file(os.path.join(wd, ckpt, "model.safetensors")), n=int(len(h)), hqnr=float(h.mean()), d_lambda=float(dl.mean()), d_s=float(dsv.mean()),
               per_scene_hqnr=[float(x) for x in h], per_scene_d_lambda=[float(x) for x in dl], per_scene_d_s=[float(x) for x in dsv], forward=how, evaluator_hash=evaluator_hash(), h5=os.path.relpath(h5, ROOT),
               evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), note="reproduction for PALS24 G-M1 — same code path as tools/eval_fr_paperset.run_one (d_lambda_k/d_s), stored fr_mat20.json untouched")
    os.makedirs(os.path.join(wd, "results"), exist_ok=True); json.dump(out, open(os.path.join(wd, "results", f"pals24_repro_{ckpt}.json"), "w"), indent=1)
    return out


def check_run(tag, role, case, seed, repro=None):
    """한 run 의 G-M2–G-M7 + 자산 hash. 반환 entry dict (approval 포함)."""
    import yaml
    wd = os.path.join(ROOT, "work_dir", tag); e = dict(run=tag, role=role, case=case, seed=seed, lambda_off=LAMBDA.get(case), checks={}, notes=[])
    if not os.path.exists(os.path.join(wd, "best_hqnr_meta.json")):
        e["approval"] = "MISSING"; return e
    bm = json.load(open(os.path.join(wd, "best_hqnr_meta.json"))); lm = json.load(open(os.path.join(wd, "last_meta.json"))) if os.path.exists(os.path.join(wd, "last_meta.json")) else {}
    cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml"))); cm = rows_csv(os.path.join(wd, "checkpoint_metrics.csv")); sm = rows_csv(os.path.join(wd, "scene_metrics.csv"))
    hashes = json.load(open(os.path.join(wd, "init_and_teacher_hashes.json"))) if os.path.exists(os.path.join(wd, "init_and_teacher_hashes.json")) else {}
    fr = json.load(open(os.path.join(wd, "results", "fr_mat20.json"))) if os.path.exists(os.path.join(wd, "results", "fr_mat20.json")) else {}
    bstep = int(bm["step"]); ok = e["checks"]
    e["best"] = dict(tag="best_hqnr", step=bstep, epoch=bm.get("epoch"), hqnr=float(bm["hqnr"]), fscc=fl(bm.get("fscc")), d_lambda=fl(bm.get("d_lambda")), d_s=fl(bm.get("d_s")),
                     ckpt_sha256=sha256_file(os.path.join(wd, "best_hqnr", "model.safetensors")) if os.path.exists(os.path.join(wd, "best_hqnr", "model.safetensors")) else None,
                     fr_mat20_hqnr=fr.get("hqnr"), fr_mat20_eval_version=fr.get("eval_version"), eval_epoch=int(cfg.get("eval_epoch", 0)))
    e["last"] = dict(step=lm.get("step"), ckpt_sha256=sha256_file(os.path.join(wd, "last", "model.safetensors")) if os.path.exists(os.path.join(wd, "last", "model.safetensors")) else None)
    # G-M2 장면 집계 (best step, raw_original): 20 scene, 중복 없음, 곱의 평균 == 기록
    sc = [r for r in sm if int(r["step"]) == bstep and r["view"] == "raw_original"]; ids = [r["scene"] for r in sc]
    prod = [(1 - fl(r["d_lambda"])) * (1 - fl(r["d_s"])) for r in sc]; hs = [fl(r["hqnr"]) for r in sc]
    mean_prod = float(np.mean(prod)) if prod else float("nan"); prod_mean = float((1 - np.mean([fl(r["d_lambda"]) for r in sc])) * (1 - np.mean([fl(r["d_s"]) for r in sc]))) if sc else float("nan")
    ok["G-M2_scene_aggregation"] = dict(n_scenes=len(sc), unique=len(set(ids)) == len(ids) == 20, mean_of_products=mean_prod, product_of_means=prod_mean, covariance_term=mean_prod - prod_mean,
                                        abs_diff_vs_meta=abs(mean_prod - float(bm["hqnr"])), abs_diff_scene_hqnr_vs_product=float(np.max(np.abs(np.array(hs) - np.array(prod)))) if sc else float("nan"),
                                        pass_=bool(len(sc) == 20 and len(set(ids)) == 20 and abs(mean_prod - float(bm["hqnr"])) <= ATOL_IDENTITY))
    # G-M3/G-M4/G-M5 view identity (전 epoch)
    def maxdiff(a, b):
        d = [abs(fl(r[a]) - fl(r[b])) for r in cm if r.get(a) and r.get(b)]; return (float(max(d)) if d else float("nan"), len(d))
    if case == "CTRLP0":
        d, n = maxdiff("aligned_valid.hqnr", "raw_valid.hqnr"); ds_, _ = maxdiff("aligned_valid.d_s", "raw_valid.d_s")
        ok["G-M3_p0_aligned_self_eq_raw_v64"] = dict(max_abs_diff_hqnr=d, max_abs_diff_d_s=ds_, n_epochs=n, pass_=bool(n > 0 and d <= ATOL_IDENTITY and ds_ <= ATOL_IDENTITY))
    if case == "P1":
        d, n = maxdiff("aligned_valid.hqnr", "aligned_fixed_v64.hqnr"); ds_, _ = maxdiff("aligned_valid.d_s", "aligned_fixed_v64.d_s")
        ok["G-M4_p1_self_eq_fixedN2"] = dict(max_abs_diff_hqnr=d, max_abs_diff_d_s=ds_, n_epochs=n, pass_=bool(n > 0 and d <= ATOL_IDENTITY and ds_ <= ATOL_IDENTITY))
    d1, n1 = maxdiff("raw_valid.d_lambda", "aligned_valid.d_lambda"); d2, n2 = maxdiff("raw_valid.d_lambda", "aligned_fixed_v64.d_lambda")
    ok["G-M5_spectral_invariance"] = dict(max_abs_diff_raw_vs_self=d1, max_abs_diff_raw_vs_fixedN2=d2, n_epochs=n1, pass_=bool(n1 > 0 and d1 <= ATOL_IDENTITY and (math.isnan(d2) or d2 <= ATOL_IDENTITY)))
    # G-M6 checkpoint 대응: best 의 실제 update·hash 보존, last 는 정확한 50K
    g10 = None
    if e["best"]["eval_epoch"] != 10:
        try:
            from tools.best_on_grid import pick
            pk = pick(tag, 10); g10 = dict(hqnr=pk["grid_hqnr"], epoch=pk["grid_ep"], recorded_hqnr=pk["rec_hqnr"], recorded_epoch=pk["rec_ep"], note="eval_epoch 5 run re-selected on the 10-epoch grid (tools/best_on_grid.py) for a fair candidate grid vs eval_epoch-10 runs") if pk else None
        except Exception as ex:
            g10 = dict(error=str(ex))
        e["notes"].append(f"eval_epoch {e['best']['eval_epoch']} (candidate grid differs from the new runs' 10) — grid-10 re-selection recorded")
    e["best_on_grid10"] = g10
    ok["G-M6_checkpoint_correspondence"] = dict(best_step=bstep, best_ckpt_sha256_16=(e["best"]["ckpt_sha256"] or "")[:16], last_step=lm.get("step"), last_sha256_16=(e["last"]["ckpt_sha256"] or "")[:16],
                                                last_is_exact_50k=bool(lm.get("step") == 50000), best_ne_last=bool(e["best"]["ckpt_sha256"] != e["last"]["ckpt_sha256"]), pass_=bool(lm.get("step") == 50000 and e["best"]["ckpt_sha256"]))
    # G-M7 support: best step 의 scene 적격, invalid 없음
    elig = [r.get("selection_eligible") == "True" for r in sc]; inval = [r.get("invalid_reason") for r in sc if r.get("invalid_reason")]
    cmb = next((r for r in cm if int(r["step"]) == bstep), None); cml = cm[-1] if cm else None
    ok["G-M7_support"] = dict(n_eligible_at_best=int(sum(elig)), invalid_reasons=inval[:5], aligned_eligible_at_best=(cmb or {}).get("aligned_eligible"), n_invalid_at_best=(cmb or {}).get("n_invalid_scenes"),
                              n_invalid_at_last=(cml or {}).get("n_invalid_scenes"), pass_=bool(sc and all(elig) and not inval))
    # 자산: U-Net 초기값(seed 별 저장 tensor) · donor · config · evaluator
    ih = hashes.get("init_hashes") or {}; init_file = ih.get("unet_init_file") or os.path.join(ROOT, "work_dir", "_kdv_init_w112_d123", f"init_unet_seed{seed}.pt")
    cur = init_sha16(init_file) if os.path.exists(init_file) else {}
    e["init"] = dict(recorded_sha16=ih.get("unet_init_sha256_16"), file=os.path.relpath(init_file, ROOT) if init_file else None, current=cur,
                     match=(bool(ih["unet_init_sha256_16"] in (cur.get("tensors_sha16"), cur.get("file_sha16"))) if ih.get("unet_init_sha256_16") else None))   # None: 기록 없음 (po trainer 의 donor run 등) — 검사 불가, 실패 아님
    if e["init"]["match"] is None:
        e["notes"].append("init tensor hash not recorded by this trainer (not a kdv run) — init check N/A")
    dn = hashes.get("donor") or {}; dfile = os.path.join(ROOT, "work_dir", DONOR_RUN, "last", "model.safetensors")
    e["donor"] = (dict(recorded_file_sha256=dn.get("file_sha256"), recorded_aligner_sha16=dn.get("aligner_tensors_sha256_16"), donor_step=dn.get("donor_step"), current_file_sha256=sha256_file(dfile) if os.path.exists(dfile) else None,
                       match=bool(dn.get("file_sha256") and os.path.exists(dfile) and dn["file_sha256"] == sha256_file(dfile)), aligner_only=True) if dn else dict(none=True, note="no aligner (A-ID)"))
    e["config_sha256"] = sha256_file(os.path.join(wd, "meta", "config.yaml")); e["evaluator"] = dict(current_hash=evaluator_hash(), fr_mat20_eval_version=fr.get("eval_version"), roi_hash_at_best=(sc[0].get("roi_hash") if sc else None))
    ok["assets"] = dict(init_match=e["init"]["match"], donor_match=e["donor"].get("match", True), fr_eval_version_current=bool(fr) and fr.get("eval_version") == __import__("tools.eval_fr_paperset", fromlist=["EVAL_VERSION"]).EVAL_VERSION,
                        pass_=bool(e["init"]["match"] is not False and e["donor"].get("match", True)))
    # G-M1 재현 (있으면)
    if repro:
        d_meta = abs(repro["hqnr"] - float(bm["hqnr"])); d_fr = abs(repro["hqnr"] - float(fr["hqnr"])) if fr.get("hqnr") is not None else float("nan")
        ps = np.array(fr.get("per_scene_hqnr") or []); d_ps = float(np.max(np.abs(ps - np.array(repro["per_scene_hqnr"])))) if len(ps) == len(repro["per_scene_hqnr"]) else float("nan")
        d_sc = float(np.max(np.abs(np.array(hs) - np.array(repro["per_scene_hqnr"])))) if len(hs) == len(repro["per_scene_hqnr"]) else float("nan")
        ok["G-M1_reproduction"] = dict(repro_hqnr=repro["hqnr"], meta_hqnr=float(bm["hqnr"]), fr_mat20_hqnr=fr.get("hqnr"), abs_diff_vs_meta=d_meta, abs_diff_vs_fr_mat20=d_fr, max_abs_diff_per_scene_vs_fr_mat20_rounded6=d_ps,
                                       max_abs_diff_per_scene_vs_training_log=d_sc, repro_ckpt_sha256_16=repro["ckpt_sha256"][:16], sheet_value_4dp=round(float(bm["hqnr"]), 4), repro_4dp=round(repro["hqnr"], 4),
                                       pass_=bool(d_meta <= ATOL_REPRO and d_sc <= ATOL_REPRO))
    hard = [k for k, v in ok.items() if isinstance(v, dict) and v.get("pass_") is False]
    e["approval"] = ("REJECTED" if hard else ("APPROVED_WITH_NOTE" if e["notes"] else "APPROVED")); e["failed_checks"] = hard
    e["metric_contract"] = "work_dir/_pals24_campaign/metric_contract.json"; e["reevaluation_cost_gpu_min"] = None
    return e


def provenance_g_m8():
    """G-M8: S1 §1 의 PO10 D_λ 0.0112–0.0116 과 시트/S4 의 0.0171–0.0175 는 **view 가 다르다** — raw_valid(V64) vs raw_original(전체 프레임), 같은 last checkpoint."""
    out = {}
    for run in ("PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT", NF16[0], NF16[2]):
        cm = rows_csv(os.path.join(ROOT, "work_dir", run, "checkpoint_metrics.csv"))
        if cm:
            l = cm[-1]; out[run] = dict(step=int(l["step"]), raw_original_d_lambda=fl(l["raw_original.d_lambda"]), raw_valid_d_lambda=fl(l["raw_valid.d_lambda"]), aligned_valid_d_lambda=fl(l["aligned_valid.d_lambda"]))
    return dict(explanation="S1 §1 D_lambda 0.0112-0.0116 = raw_valid/aligned_valid (fixed V64 ROI) at last for the PO10 R200 runs; sheet/S4 0.0171-0.0175 = raw_original (full frame) at last of the same runs. "
                            "The 'no-align 0.0177' matches NF16 P0/P2 raw_valid at last (0.0158/0.0177) only as an order of magnitude — the sweep uses the reproduced P0/P2/P3 raw_original protocol (this gate) as the fixed reference.",
                values=out, resolved=bool(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-repro", action="store_true"); ap.add_argument("--device", default=None)
    ap.add_argument("--campaign", default="pals24", choices=("pals24", "palsv18"), help="palsv18: 대조군 9벌(P0/L000/L1E4 × 3 seed) + N2 last·L1E2 참조, 산출 work_dir/_palsv18_campaign (+ selection_grid.json, initial_tensor_registry.json)")
    a = ap.parse_args()
    import torch
    global CAMP, REUSED, DIAG_REFERENCE, BACKGROUND, CAMPAIGN_ID
    if a.campaign == "palsv18":
        from tools import gen_palsv18_configs as V
        CAMP = os.path.join(ROOT, "work_dir", "_palsv18_campaign"); REUSED, DIAG_REFERENCE, BACKGROUND, CAMPAIGN_ID = V.REUSED, V.DIAG_REFERENCE, {}, V.CAMP["campaign_id"]
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu")); os.makedirs(CAMP, exist_ok=True); t0 = time.time()
    con = contract(); con["campaign_id"] = CAMPAIGN_ID; json.dump(con, open(os.path.join(CAMP, "metric_contract.json"), "w"), indent=1, ensure_ascii=False)
    if a.campaign == "palsv18":                                       # G-S selection grid (계획 §6.3): PALS24 L1E4 seed1234 의 실제 평가 update 목록을 고정, 신규 run 은 같은 eval_epoch 10
        from tools.gen_pals24_configs import run_name as _rn
        import csv as _csv, yaml as _yaml
        ref = _rn("L1E4", 1234); cm = list(_csv.DictReader(open(os.path.join(ROOT, "work_dir", ref, "checkpoint_metrics.csv"))))
        steps = [int(r["step"]) for r in cm]; epochs = [int(r["epoch"]) for r in cm]; cfg = _yaml.safe_load(open(os.path.join(ROOT, "work_dir", ref, "meta", "config.yaml")))
        grid = dict(reference_run=ref, eval_epoch=int(cfg["eval_epoch"]), n_candidates=len(steps), optimizer_updates=steps, epochs=epochs, exact_last=50000, last_included=(50000 in steps),
                    note="eval_epoch 10 = every 10 epochs (202 updates/epoch), not every 10 updates; NF16 P0 (eval_epoch 5) has 2x candidates -> matched-grid re-selection via tools/best_on_grid.py",
                    sha256=hashlib.sha256(json.dumps(steps).encode()).hexdigest())
        json.dump(grid, open(os.path.join(CAMP, "selection_grid.json"), "w"), indent=1)
        # G-C initial tensor registry (계획 §2.2): seed 별 저장 U-Net 초기값 hash 가 기존 PALS24/NF16 run 의 기록과 같아야 통제 비교가 성립한다
        reg_i = {}
        for sd_ in (1234, 7777, 2025):
            f = os.path.join(ROOT, "work_dir", "_kdv_init_w112_d123", f"init_unet_seed{sd_}.pt"); cur = init_sha16(f) if os.path.exists(f) else {}
            recs = {}
            for (c, s_), run in REUSED.items():
                if s_ == sd_:
                    hp = os.path.join(ROOT, "work_dir", run, "init_and_teacher_hashes.json")
                    if os.path.exists(hp):
                        recs[run] = (json.load(open(hp)).get("init_hashes") or {}).get("unet_init_sha256_16")
            reg_i[sd_] = dict(file=os.path.relpath(f, ROOT), current=cur, recorded_in_controls=recs, all_match=bool(recs) and all(v in (cur.get("tensors_sha16"), cur.get("file_sha16")) for v in recs.values()))
        json.dump(dict(campaign_id=CAMPAIGN_ID, seeds=reg_i, donor=dict(run="PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT", kind="last", copy="aligner_only"), created=time.strftime("%Y-%m-%dT%H:%M:%S")),
                  open(os.path.join(CAMP, "initial_tensor_registry.json"), "w"), indent=1)
        print(f"[G-S] selection grid: {len(steps)} candidates (eval_epoch {grid['eval_epoch']}, last {grid['last_included']}) · [G-C] init tensors match: { {k: v['all_match'] for k, v in reg_i.items()} }")
    print(f"[G-M0] evaluator {con['evaluator_hash']} · FR eval {con['fr_eval_version']} · selector best_raw: HQNR(raw_original, mat20) tie {con['selector']['tie_band_hqnr']} → fSCC(원 PAN, full-frame Sobel SCC) tie {con['selector']['tie_band_fscc']} · method margin {METHOD_MARGIN}")
    wald = None
    if not a.skip_repro:
        from tools.eval_fr_paperset import load_dlpan
        wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    cells = [(c, s, r, "reuse") for (c, s), r in REUSED.items()] + [(c, s, r, "diagnostic_reference") for (c, s), r in DIAG_REFERENCE.items()] + [(c, s, r, "background") for (c, s), r in BACKGROUND.items()]
    entries = []
    for case, seed, run, role in cells:
        rp = None
        if wald is not None and role != "background" and os.path.exists(os.path.join(ROOT, "work_dir", run, "best_hqnr", "model.safetensors")):
            t1 = time.time(); rp = reproduce(run, "best_hqnr", dev, wald); rp["gpu_seconds"] = time.time() - t1
        e = check_run(run, role, case, seed, rp)
        if rp:
            e["reevaluation_cost_gpu_min"] = round(rp["gpu_seconds"] / 60, 2)
        entries.append(e)
        g1 = e["checks"].get("G-M1_reproduction"); g2 = e["checks"].get("G-M2_scene_aggregation", {})
        print(f"  {run} [{role} {case} s{seed}] → {e['approval']} " + (f"| G-M1 repro {g1['repro_hqnr']:.6f} vs meta {g1['meta_hqnr']:.6f} (Δ {g1['abs_diff_vs_meta']:.1e}, scene max Δ {g1['max_abs_diff_per_scene_vs_training_log']:.1e}) " if g1 else "")
              + (f"| G-M2 mean-of-products {g2.get('mean_of_products', float('nan')):.6f} (prod-of-means {g2.get('product_of_means', float('nan')):.6f}) " if g2 else "") + (f"| grid10 {e['best_on_grid10']['hqnr']:.5f}@ep{e['best_on_grid10']['epoch']} " if e.get("best_on_grid10") and e["best_on_grid10"].get("hqnr") else "")
              + (f"| notes {e['notes']}" if e["notes"] else "") + (f"| FAILED {e['failed_checks']}" if e["failed_checks"] else ""))
    reg = dict(campaign_id=CAMPAIGN_ID, created=time.strftime("%Y-%m-%dT%H:%M:%S"), rule="reuse only if all hard checks pass and the init tensor / donor hash / evaluator match; scores are never copied into a new run folder (§11.3)",
               entries=entries, approved={e["case"]: e["run"] for e in entries if e["role"] == "reuse" and e["approval"].startswith("APPROVED")},
               rejected={e["case"]: e["run"] for e in entries if e["role"] == "reuse" and not e["approval"].startswith("APPROVED")}, provenance_g_m8=provenance_g_m8())
    json.dump(reg, open(os.path.join(CAMP, "reuse_registry.json"), "w"), indent=1, ensure_ascii=False)
    rep = dict(created=reg["created"], gpu_minutes=round((time.time() - t0) / 60, 2), device=str(dev), repro_done=wald is not None, entries={e["run"]: dict(approval=e["approval"], failed=e["failed_checks"], checks=e["checks"]) for e in entries},
               all_reuse_approved=bool(reg["approved"]) and not reg["rejected"], g_m8=reg["provenance_g_m8"]["explanation"])
    json.dump(rep, open(os.path.join(CAMP, "metric_gate_report.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[gate] reuse approved: {reg['approved']} · rejected: {reg['rejected']} · G-M8 {'resolved' if reg['provenance_g_m8']['resolved'] else 'UNRESOLVED'} · {rep['gpu_minutes']} min")
    return 0 if rep["all_reuse_approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
