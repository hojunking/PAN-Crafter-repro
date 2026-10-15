"""D10 — 같은-checkpoint sample atlas: native e, offset consistency q (bank A/B), 네 집단, H1 (계획 §7)."""
import os, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS


@torch.no_grad()
def atlas_patches(L, man, gt, ms, lpan, pan, record_probes):
    y, d = C.native_forward(L, pan, ms, lpan)
    e_full = (y - gt).abs().mean(dim=(1, 2, 3)); e_roi = C.l1_roi(y, gt, "native64"); eb = C.l1_roi(y, gt, "native64", per_band=True); ee = C.edge_l1_roi(y, gt, "native64")
    st = C.input_stats(pan, ms); rows = []; recs = []
    pr = C.probe_responses(L, pan, ms) if L.has_aligner else None
    for i in range(len(man)):
        r = dict(model_key=L.key, family=L.fam, seed=L.seed, checkpoint_kind=L.tag, actual_update=L.step, sample_id=int(man.sample_id.iloc[i]), source_group_id=man.source_group_id.iloc[i], split_role=man.split_role.iloc[i], scale="native64",
                 e_native_full=float(e_full[i]), e_native_roi=float(e_roi[i]), edge_l1_roi=float(ee[i]), **{f"e_band{b}_roi": float(eb[i, b]) for b in range(8)},
                 c0_dy=float(d[i, 0]), c0_dx=float(d[i, 1]), c0_norm=float(d[i].norm()), **{k: float(v[i]) for k, v in st.items()})
        if pr is not None:
            for b in ("A", "B"):
                r[f"q_{b}"] = float(pr[b]["q"][i]); r[f"epe_{b}"] = float(pr[b]["epe"][i])
                for rad in C.RADII:
                    r[f"q_{b}_r{rad}"] = float(pr[b]["q_by_radius"][str(rad)][i])
                r.update({f"{b}_{k}": v for k, v in C.fit_response(pr[b]["resp"][i].numpy(), pr[b]["probes"]).items()} if b == "A" else {})
                if record_probes:
                    for k, p in enumerate(pr[b]["probes"]):
                        recs.append(dict(model_key=L.key, sample_id=int(man.sample_id.iloc[i]), scale="native64", probe_bank=b, probe_id=p["probe_id"], r=p["r"], epsilon_dy_hr=p["ey"], epsilon_dx_hr=p["ex"], kernel="bicubic", padding="border",
                                         c0_dy=float(d[i, 0]), c0_dx=float(d[i, 1]), ce_dy=float(d[i, 0] + pr[b]["resp"][i, k, 0]), ce_dx=float(d[i, 1] + pr[b]["resp"][i, k, 1]), residual_dy=float(pr[b]["resid"][i, k, 0]), residual_dx=float(pr[b]["resid"][i, k, 1]),
                                         scalar_scope="probe"))
            r["invalid_support"] = bool(not C.two_stage_ok(64, 64, (2.0, 2.0), d[i], 16))
        rows.append(r)
    return rows, recs


@torch.no_grad()
def atlas_scenes(L, fd):
    """RR20 (e·reduced metrics·band/edge·q) 와 FR20 (raw views·q) — 서로 다른 strata."""
    from pa.evalviews import scene_views, VIEWS
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    from utils import reduced_metrics
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rows = []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); o = L.m(pan, ms, lpan); y = o["y"]; d = o["delta"][0].float().cpu()
        rm = reduced_metrics(x_true=gt, x_pred=y, max_pixel=mp); r = dict(model_key=L.key, family=L.fam, seed=L.seed, checkpoint_kind=L.tag, actual_update=L.step, sample_id=i, source_group_id=f"rr{i:02d}", split_role="RR", scale="rr256",
                                                                          e_native_full=float((y - gt).abs().mean()), e_native_roi=float(C.l1_roi(y, gt, "rr256")[0]), edge_l1_roi=float(C.edge_l1_roi(y, gt, "rr256")[0]),
                                                                          **{f"e_band{b}_roi": float(v) for b, v in enumerate(C.l1_roi(y, gt, "rr256", per_band=True)[0])}, **{f"rr_{k}": float(v) for k, v in rm.items()},
                                                                          c0_dy=float(d[0]), c0_dx=float(d[1]), c0_norm=float(d.norm()))
        if L.has_aligner:
            pr = C.probe_responses(L, pan.cpu(), ms.cpu())
            for b in ("A", "B"):
                r[f"q_{b}"] = float(pr[b]["q"][0]); r[f"epe_{b}"] = float(pr[b]["epe"][0]); r.update({f"q_{b}_r{rad}": float(pr[b]["q_by_radius"][str(rad)][0]) for rad in C.RADII})
                if b == "A":
                    r.update({f"A_{k}": v for k, v in C.fit_response(pr["A"]["resp"][0].numpy(), pr["A"]["probes"]).items()})
        rows.append(r)
    for i in range(len(fd["fr"])):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["fr"][i]); o = L.m(pan, ms, lpan); d = o["delta"][0].double().cpu().numpy()
        sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]
        pa_eval = C.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d)[None])[0, 0].numpy(); v, ok, why = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp)
        r = dict(model_key=L.key, family=L.fam, seed=L.seed, checkpoint_kind=L.tag, actual_update=L.step, sample_id=i, source_group_id=f"fr{i:02d}", split_role="FR", scale="fr512", c0_dy=float(d[0]), c0_dx=float(d[1]), c0_norm=float(np.hypot(*d)),
                 eligible_valid_views=bool(ok), **{f"{k}_{q}": float(v[k][q]) for k in VIEWS for q in ("hqnr", "d_lambda", "d_s", "fscc")})
        if L.has_aligner:
            pr = C.probe_responses(L, pan.cpu(), ms.cpu())
            for b in ("A", "B"):
                r[f"q_{b}"] = float(pr[b]["q"][0]); r[f"epe_{b}"] = float(pr[b]["epe"][0]); r.update({f"q_{b}_r{rad}": float(pr[b]["q_by_radius"][str(rad)][0]) for rad in C.RADII})
                if b == "A":
                    r.update({f"A_{k}": v for k, v in C.fit_response(pr["A"]["resp"][0].numpy(), pr["A"]["probes"]).items()})
        rows.append(r)
    return rows


def thresholds_and_quadrants(df):
    """A calibration 의 model·scale 별 median (e_native_full, q_A, q_B) → 전 sample 에 적용 (tie → low). RR/FR 은 descriptive median 만."""
    thr = {}; out = []
    for mk, g in df.groupby("model_key"):
        a = g[(g.split_role == "A") & (g.scale == "native64")]
        t = dict(model_key=mk, n_A=int(len(a)), e_median=float(a.e_native_full.median()), q_A_median=(float(a.q_A.median()) if "q_A" in a and a.q_A.notna().any() else None),
                 q_B_median=(float(a.q_B.median()) if "q_B" in a and a.q_B.notna().any() else None), threshold_hash=None)
        for sc in ("rr256", "fr512"):
            s = g[g.scale == sc]
            if len(s):
                t[f"{sc}_descriptive"] = dict(e_median=(float(s.e_native_full.median()) if "e_native_full" in s and s.e_native_full.notna().any() else None), q_A_median=(float(s.q_A.median()) if "q_A" in s and s.q_A.notna().any() else None),
                                             hqnr_median=(float(s.raw_original_hqnr.median()) if "raw_original_hqnr" in s and s.raw_original_hqnr.notna().any() else None), note="not used for gates")
        t["threshold_hash"] = C.hashlib.sha256(json.dumps([t["e_median"], t["q_A_median"], t["q_B_median"]]).encode()).hexdigest()[:16]; thr[mk] = t
        n64 = g[g.scale == "native64"]
        for _, r in n64.iterrows():
            el = bool(r.e_native_full <= t["e_median"]); cl = (bool(r.q_A <= t["q_A_median"]) if t["q_A_median"] is not None and pd.notna(r.get("q_A", np.nan)) else None)
            clB = (bool(r.q_B <= t["q_B_median"]) if t["q_B_median"] is not None and pd.notna(r.get("q_B", np.nan)) else None)
            quad = (("Ed" if el else "Eu") + ("Cd" if cl else "Cu")) if cl is not None else None
            out.append(dict(model_key=mk, sample_id=int(r.sample_id), split_role=r.split_role, source_group_id=r.source_group_id, e_low=el, c_low=cl, c_low_by_qB=clB, quadrant_id=quad,
                            quadrant_id_by_qB=((("Ed" if el else "Eu") + ("Cd" if clB else "Cu")) if clB is not None else None), threshold_hash=t["threshold_hash"]))
    return thr, pd.DataFrame(out)


def occupancy(qa):
    rows = []
    for mk, g in qa.groupby("model_key"):
        for role in ("A", "B", "C", "D", "ALL"):
            s = g if role == "ALL" else g[g.split_role == role]
            for q in QUADS:
                x = s[s.quadrant_id == q]; rows.append(dict(model_key=mk, split_role=role, quadrant_id=q, n=int(len(x)), n_source_groups=int(x.source_group_id.nunique()),
                                                            insufficient_support=bool(len(x) < 32 or x.source_group_id.nunique() < 5)))
    return rows


def h1_stats(df, qa, thr):
    out = {}
    for mk, g in df.groupby("model_key"):
        if "q_A" not in g or not g.q_A.notna().any():
            continue
        n = g[(g.scale == "native64")]; res = dict(model_key=mk)
        for role, sel in (("BCD", n[n.split_role.isin(["B", "C", "D"])]), ("D", n[n.split_role == "D"]), ("A", n[n.split_role == "A"])):
            res[f"spearman_qA_e_{role}"] = C.spearman_boot(sel.q_A, sel.e_native_full, sel.source_group_id, seed=1); res[f"spearman_qA_qB_{role}"] = C.spearman(sel.q_A, sel.q_B)
            res[f"spearman_qB_e_{role}"] = C.spearman(sel.q_B, sel.e_native_full); res[f"spearman_c0norm_e_{role}"] = C.spearman(sel.c0_norm, sel.e_native_full)
        bcd = n[n.split_role.isin(["B", "C", "D"])].copy(); bcd["tex"] = C.texture_tertile(bcd.pan_scharr_energy.values); bcd["c0t"] = C.texture_tertile(bcd.c0_norm.values)
        res["spearman_qA_e_by_texture"] = {t: C.spearman(s.q_A, s.e_native_full) for t, s in bcd.groupby("tex")}; res["spearman_qA_e_by_c0norm_tertile_auxiliary"] = {t: C.spearman(s.q_A, s.e_native_full) for t, s in bcd.groupby("c0t")}
        bcd["qdec"] = pd.qcut(bcd.q_A.rank(method="first"), 10, labels=False)
        res["qA_decile_table"] = [dict(decile=int(k), n=int(len(s)), qA_mean=float(s.q_A.mean()), e_mean=float(s.e_native_full.mean()), e_median=float(s.e_native_full.median()), e_iqr=float(s.e_native_full.quantile(.75) - s.e_native_full.quantile(.25)),
                                       qB_mean=float(s.q_B.mean()), edge_l1_mean=float(s.edge_l1_roi.mean())) for k, s in bcd.groupby("qdec")]
        lab = qa[(qa.model_key == mk) & qa.quadrant_id.notna() & qa.quadrant_id_by_qB.notna()]; res["quadrant_label_agreement_qA_vs_qB"] = (float((lab.quadrant_id == lab.quadrant_id_by_qB).mean()) if len(lab) else None)
        lo, hi = bcd.q_A.quantile(.3), bcd.q_A.quantile(.7); res["extreme_30pct"] = dict(low_q=dict(n=int((bcd.q_A <= lo).sum()), e_mean=float(bcd[bcd.q_A <= lo].e_native_full.mean())), high_q=dict(n=int((bcd.q_A >= hi).sum()), e_mean=float(bcd[bcd.q_A >= hi].e_native_full.mean())),
                                    middle_40_e_mean=float(bcd[(bcd.q_A > lo) & (bcd.q_A < hi)].e_native_full.mean()))
        fr = g[g.scale == "fr512"]; rr = g[g.scale == "rr256"]
        if len(fr):
            res["FR_spearman_qA_hqnr"] = C.spearman(fr.q_A, fr.raw_original_hqnr); res["FR_spearman_qA_dlambda"] = C.spearman(fr.q_A, fr.raw_original_d_lambda); res["FR_spearman_qA_ds"] = C.spearman(fr.q_A, fr.raw_original_d_s); res["FR_spearman_qA_fscc"] = C.spearman(fr.q_A, fr.raw_original_fscc)
        if len(rr):
            res["RR_spearman_qA_e"] = C.spearman(rr.q_A, rr.e_native_full); res["RR_spearman_qA_ergas"] = C.spearman(rr.q_A, rr.rr_ergas) if "rr_ergas" in rr else None
        res["note"] = "within-checkpoint only; E↓ has low e by definition — the informative results are the intervention/stress/utility differences per quadrant (D30/D40/K10)"
        out[mk] = res
    return out


def pick_subsets(df, qa):
    """primary model 의 D 집단에서 상세(64)/gradient(32)/landscape(16) subset 을 source×texture 층화 추출 + 대표 4개 (§3.4·§7). 다른 model 은 같은 sample 을 쓴다."""
    pk = C.mkey(*C.PRIMARY); n = df[(df.model_key == pk) & (df.scale == "native64")].copy(); n["texture"] = C.texture_tertile(n.pan_scharr_energy.values)
    q = qa[qa.model_key == pk][["sample_id", "quadrant_id"]]; n = n.merge(q, on="sample_id"); D = n[n.split_role == "D"]; out = dict(primary=pk, source_split="D", quadrants={})
    for qd in QUADS:
        s = D[D.quadrant_id == qd]; det = C.stratified_pick(s, C.DETAIL_PER_QUAD, seed=11); ids = [int(x) for x in det.sample_id] if len(det) else []
        out["quadrants"][qd] = dict(n_available=int(len(s)), detail=ids, grad=ids[:C.GRAD_PER_QUAD], landscape=ids[:C.LANDSCAPE_PER_QUAD], representative=ids[:4], insufficient_support=bool(len(s) < 32 or s.source_group_id.nunique() < 5))
    C.dump_json(os.path.join(CAMP, "detail_subsets.json"), out); return out


def figures(df, qa, thr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [C.mkey("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")]; keys = [k for k in keys if k in thr]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9)); axs = axs.ravel()
    for ax, mk in zip(axs, keys):
        g = df[(df.model_key == mk) & (df.scale == "native64")]; t = thr[mk]
        ax.scatter(np.log(g.q_A + 1e-6), np.log(g.e_native_full + 1e-6), s=3, alpha=.4); ax.axvline(np.log(t["q_A_median"] + 1e-6), c="r", lw=.8); ax.axhline(np.log(t["e_median"] + 1e-6), c="r", lw=.8)
        ax.set_title(f"{mk} (upd {int(g.actual_update.iloc[0])})", fontsize=8); ax.set_xlabel("log(q_A+eps)"); ax.set_ylabel("log(e_T+eps)")
    fig.suptitle("D10: e_T vs q_A (native64, all splits; red = A-calibration medians)"); fig.tight_layout(); fig.savefig(os.path.join(C.FIG, "fig1_eq_quadrant_scatter.png"), dpi=110); plt.close(fig)
    pk = C.mkey(*C.PRIMARY); g = df[(df.model_key == pk) & (df.scale == "native64")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5)); ax[0].scatter(g.q_A, g.q_B, s=3, alpha=.4); ax[0].set_xlabel("q_A"); ax[0].set_ylabel("q_B"); ax[0].set_title(f"{pk}: q_A vs q_B (repeatability)")
    fr = df[(df.model_key == pk) & (df.scale == "fr512")]
    if len(fr):
        ax[1].scatter(fr.q_A, fr.raw_original_hqnr); ax[1].set_xlabel("q_A (fr512)"); ax[1].set_ylabel("raw-original HQNR"); ax[1].set_title("FR20: q_A vs HQNR (separate stratum)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG, "fig1b_qA_qB_and_fr.png"), dpi=110); plt.close(fig)


def main(profile=False):
    C.ensure_dirs(); man = C.make_manifest(); fd = C.feeders()
    with C.Stage("D10" + ("-profile" if profile else ""), "sample atlas: e/q/quadrants/H1"):
        if profile:
            man = man[man.split_role == "A"].iloc[:128]
        gt, ms, lpan, pan = C.load_patches(man.sample_id.tolist()); rows_all, recs_all, scene_all = [], [], []
        models = C.CORE + C.EXTRA
        for fam, seed, tag in models:
            if C.ckpt_dir(fam, seed, tag) is None:
                print(f"  skip missing {C.mkey(fam, seed, tag)}"); continue
            L = C.load_model(fam, seed, tag); t0 = C.time.time()
            rows, recs = atlas_patches(L, man, gt, ms, lpan, pan, record_probes=((fam, seed, tag) in C.CORE) and not profile); rows_all += rows; recs_all += recs
            if not profile:
                scene_all += atlas_scenes(L, fd)
            print(f"  {L.key}: {len(rows)} patches, {C.time.time() - t0:.0f}s", flush=True); del L; torch.cuda.empty_cache()
        df = pd.DataFrame(rows_all + scene_all); df.to_csv(os.path.join(CAMP, "native_sample_metrics.csv" if not profile else "native_sample_metrics_profile.csv"), index=False)
        if profile:
            return df
        if recs_all:
            pd.DataFrame(recs_all).to_csv(os.path.join(CAMP, "offset_probe_records.csv.gz"), index=False, compression="gzip")
        thr, qa = thresholds_and_quadrants(df); C.dump_json(os.path.join(CAMP, "calibration_thresholds.json"), thr); qa.to_csv(os.path.join(CAMP, "quadrant_assignments.csv"), index=False)
        occ = occupancy(qa); C.write_csv(os.path.join(CAMP, "quadrant_occupancy.csv"), occ); h1 = h1_stats(df, qa, thr); C.dump_json(os.path.join(CAMP, "h1_stats.json"), h1)
        pick_subsets(df, qa); figures(df, qa, thr)
        pk = C.mkey(*C.PRIMARY); print(json.dumps(dict(primary=pk, thresholds=thr.get(pk), h1_BCD=h1.get(pk, {}).get("spearman_qA_e_BCD"), h1_FR=h1.get(pk, {}).get("FR_spearman_qA_hqnr"), occupancy=[o for o in occ if o["model_key"] == pk and o["split_role"] == "ALL"]), default=str, ensure_ascii=False))
