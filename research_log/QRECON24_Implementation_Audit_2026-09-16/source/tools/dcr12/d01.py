"""D01 — 네 사분면을 실제 자료에서 정의한다 (계획 §5). sample_metrics.csv · quadrant_thresholds.json · quadrant_summary.csv · d01_stats.json · scene_metrics.csv · panel_ids.json(grad_ids).
다시 부르면 아직 없는 pipeline(새 checkpoint) 분만 덧붙인다. Teacher threshold(tC, tR) 는 T0 의 CAL median 으로 한 번 고정; Student bin 은 같은 seed B0 best_hqnr(고정 reference) 의 median (B1 에도 적용)."""
import json, os
import numpy as np, pandas as pd, torch
from tools.dcr12 import common as C
PHASE = os.environ.get("DCR12_PHASE", "")
SCENE_TAGS = ("best_hqnr",)


def sample_rows(P, panel, ids):
    gt, ms, lpan, pan = C.load_patches(ids); y, delta = C.reconstruct(P, pan, ms, lpan); rm = C.r_metrics(y, gt)
    cm = C.consistency(P, pan, ms, C.probe_set("main")); cc = C.consistency(P, pan, ms, C.probe_set("confirm")); st = C.E.input_stats(pan, ms); sat = C.saturation(pan, ms)
    sup = (cm["c0"].abs().max(1).values + max(C.PROBE_R) + 4 <= P.mg + 4 + 1e-9)
    rows = []
    for i, sid in enumerate(ids):
        rows.append(dict(host=C.SERVER, pipeline=P.key, role=P.role, case=P.case, seed=(P.seed if P.seed is not None else 2025), tag=P.tag, step=P.step, panel=panel, sample_id=int(sid), group=C.group_of(sid),
                         C=float(cm["C"][i]), C_r05=float(cm["C_by_r"]["0.5"][i]), C_r10=float(cm["C_by_r"]["1.0"][i]), C_confirm=float(cc["C"][i]), C_axis_y=float(cm["C_axis"][i, 0]), C_axis_x=float(cm["C_axis"][i, 1]), identity_floor=float(cm["identity"][i]),
                         R_plain=float(rm["R_plain"][i]), R_interior=float(rm["R_interior"][i]), edge_plain=float(rm["edge_plain"][i]), c_y=float(delta[i, 0]), c_x=float(delta[i, 1]), c_norm=float(delta[i].norm()),
                         texture=float(st["pan_scharr_energy"][i]), pan_std=float(st["pan_std"][i]), pan_msmean_corr=float(st["pan_msmean_corr"][i]), pan_sat_frac=float(sat["pan_sat_frac"][i]), ms_sat_frac=float(sat["ms_sat_frac"][i]), support_ok=bool(sup[i])))
    return rows


def scene_rows(P, fd):
    """RR20 (R with GT, reduced metrics) · FR20 (raw views; R_GT 없음) — 장면 단위. thresholds 는 native64 정의라 label 은 참고(threshold_domain 표시)."""
    from pa.evalviews import scene_views, VIEWS
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    from utils import reduced_metrics
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rows = []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0) for t in fd["rr"][i]); y, d = C.reconstruct(P, pan, ms, lpan); cm = C.consistency(P, pan, ms); rmx = reduced_metrics(x_true=gt.to(C.DEV), x_pred=y.to(C.DEV), max_pixel=mp)
        rows.append(dict(host=C.SERVER, pipeline=P.key, role=P.role, case=P.case, seed=P.seed, tag=P.tag, step=P.step, split="RR", scale="rr256", sample_id=i, group=f"rr{i:02d}", C=float(cm["C"][0]), C_r05=float(cm["C_by_r"]["0.5"][0]), C_r10=float(cm["C_by_r"]["1.0"][0]),
                         R_plain=float((y - gt).abs().mean()), R_roi=float(C.E.l1_roi(y, gt, "rr256")[0]), edge_roi=float(C.E.edge_l1_roi(y, gt, "rr256")[0]), c_y=float(d[0, 0]), c_x=float(d[0, 1]), **{f"rr_{k}": float(v) for k, v in rmx.items()}))
    for i in range(len(fd["fr"])):
        lms, ms, lpan, pan = (t.unsqueeze(0) for t in fd["fr"][i]); y, d = C.reconstruct(P, pan, ms, lpan); cm = C.consistency(P, pan, ms); dd = d[0].double().numpy()
        sr = ((y[0].clip(-1, 1).float().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]
        pa_eval = C.E.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(dd)[None])[0, 0].numpy(); v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_eval, sensor, wald, dd, 4, mp)
        rows.append(dict(host=C.SERVER, pipeline=P.key, role=P.role, case=P.case, seed=P.seed, tag=P.tag, step=P.step, split="FR", scale="fr512", sample_id=i, group=f"fr{i:02d}", C=float(cm["C"][0]), C_r05=float(cm["C_by_r"]["0.5"][0]), C_r10=float(cm["C_by_r"]["1.0"][0]),
                         c_y=float(dd[0]), c_x=float(dd[1]), eligible_valid_views=bool(ok), **{f"{k}_{q}": float(v[k][q]) for k in VIEWS for q in ("hqnr", "d_lambda", "d_s", "fscc")}))
    return rows


def label(C_, R_, tC, tR):
    return C.LABELS[("low" if C_ <= tC else "high", "low" if R_ <= tR else "high")]


def thresholds(sm, floor):
    """T0 CAL median → tC/tR (고정). Student: seed 별 B0 best_hqnr(reference) 의 CAL median. DEGENERATE_METRIC: C 의 IQR 이 반복 수치오차의 10 배 미만."""
    out = {}; t = sm[(sm.pipeline == "T0") & (sm.panel == "CAL")]
    out["T0"] = dict(tC=float(t.C.median()), tR=float(t.R_plain.median()), n=int(len(t)), C_iqr=float(t.C.quantile(.75) - t.C.quantile(.25)), R_iqr=float(t.R_plain.quantile(.75) - t.R_plain.quantile(.25)), numerical_floor_C=floor,
                     degenerate_metric=bool((t.C.quantile(.75) - t.C.quantile(.25)) < 10 * max(floor, 1e-9)), C_cv=float((t.C.quantile(.75) - t.C.quantile(.25)) / max(t.C.median(), 1e-12)), source="T0 CAL median (진단상 상대 구간; 물리적 참값 아님)")
    for seed in C.SEEDS:
        ref = C.reference_student(seed)
        if ref:
            key = C.pipe_key("S", *ref); r = sm[(sm.pipeline == key) & (sm.panel == "CAL")]
            if len(r):
                out[f"S{seed}"] = dict(reference=key, tC=float(r.C.median()), tR=float(r.R_plain.median()), n=int(len(r)), source="같은 seed B0 best_hqnr 의 CAL median — B1 에도 같은 bin (§5.3 선택 편향 회피)")
    return out


def assign_cells(sm, thr):
    sm = sm.copy(); sm["cell_ref"] = None; sm["cell_own"] = None
    for key, g in sm.groupby("pipeline"):
        idx = g.index
        if key == "T0":
            th = thr["T0"]
        else:
            seed = int(g.seed.iloc[0]); th = thr.get(f"S{seed}")
        if th:
            sm.loc[idx, "cell_ref"] = [label(c, r, th["tC"], th["tR"]) for c, r in zip(g.C, g.R_plain)]
        cal = g[g.panel == "CAL"]
        if len(cal):
            tc, tr_ = float(cal.C.median()), float(cal.R_plain.median()); sm.loc[idx, "cell_own"] = [label(c, r, tc, tr_) for c, r in zip(g.C, g.R_plain)]
    return sm


def summary(sm):
    rows = []
    for (pipe, panel, cell), g in sm.dropna(subset=["cell_ref"]).groupby(["pipeline", "panel", "cell_ref"]):
        q = lambda s: (float(s.median()), float(s.quantile(.25)), float(s.quantile(.75)))
        rows.append(dict(pipeline=pipe, panel=panel, cell=cell, cell_desc=C.CELL_DESC[cell], n=int(len(g)), n_groups=int(g.group.nunique()), small_cell=bool(len(g) < C.SMALL_CELL_N or g.group.nunique() < C.SMALL_CELL_GROUPS),
                         R_med=q(g.R_plain)[0], R_q25=q(g.R_plain)[1], R_q75=q(g.R_plain)[2], edge_med=q(g.edge_plain)[0], C_med=q(g.C)[0], C_q25=q(g.C)[1], C_q75=q(g.C)[2], c_norm_med=q(g.c_norm)[0], texture_med=q(g.texture)[0], pan_sat_med=q(g.pan_sat_frac)[0]))
    return rows


def stats(sm, thr):
    out = {}
    for (pipe, panel), g in sm.groupby(["pipeline", "panel"]):
        d = dict(n=int(len(g)), n_groups=int(g.group.nunique()), spearman_C_R=C.spearman_boot(g.C, g.R_plain, g.group), pearson_C_R=C.pearson(g.C, g.R_plain), spearman_C_edge=C.spearman(g.C, g.edge_plain),
                 spearman_C_r05_R=C.spearman(g.C_r05, g.R_plain), spearman_C_r10_R=C.spearman(g.C_r10, g.R_plain), spearman_Cconfirm_R=C.spearman(g.C_confirm, g.R_plain), spearman_C_Cconfirm=C.spearman(g.C, g.C_confirm),
                 spearman_C_texture=C.spearman(g.C, g.texture), spearman_R_texture=C.spearman(g.R_plain, g.texture), C_median=float(g.C.median()), C_iqr=float(g.C.quantile(.75) - g.C.quantile(.25)), R_median=float(g.R_plain.median()),
                 identity_floor_max=float(g.identity_floor.max()), support_ok_frac=float(g.support_ok.mean()), cells=g.cell_ref.value_counts().to_dict() if "cell_ref" in g else None)
        out[f"{pipe}|{panel}"] = d
    out["thresholds"] = thr; return out


def pick_grad(sm, panels):
    """GRAD64: DISC 에서 T0 cell 균형(≤16/cell, 부족하면 큰 cell 에서 채움), seed 314159."""
    t = sm[(sm.pipeline == "T0") & (sm.panel == "DISC")].dropna(subset=["cell_ref"]); rng = np.random.RandomState(C.SPLIT_SEED + 3); picked = []
    per = {c: t[t.cell_ref == c].sample_id.tolist() for c in "ABCD"}
    for c in "ABCD":
        rng.shuffle(per[c]); picked += [(int(s), c) for s in per[c][:16]]
    rest = [(int(s), c) for c in "ABCD" for s in per[c][16:]]; rng.shuffle(rest); picked += rest[:C.GRAD_N - len(picked)]
    panels["grad_ids"] = [s for s, _ in picked]; panels["grad_cells"] = {str(s): c for s, c in picked}; panels["grad_note"] = "DISC 에서 T0 cell_ref 균형 ≤16/cell (seed 314159+3); 부족분은 나머지 cell 에서"
    C.dump_json(os.path.join(C.CAMP, "panel_ids.json"), panels); return panels


def main(profile=False):
    with C.Stage("D01" + (f"-{PHASE}" if PHASE else ""), "quadrant panel / thresholds / labels / RR-FR scenes"):
        panels = C.make_panels(); T = C.load_pipe("T"); smp = os.path.join(C.CAMP, "sample_metrics.csv"); scp = os.path.join(C.CAMP, "scene_metrics.csv")
        have = set(pd.read_csv(smp).pipeline.unique()) if os.path.exists(smp) else set(); have_sc = set(pd.read_csv(scp).pipeline.unique()) if os.path.exists(scp) else set()
        pipes = [T] + [C.load_pipe("S", ck, seed, tag) for ck, seed, tag in C.available_students(("best_hqnr", "last"))]
        rows = []
        for P in pipes:
            if P.key in have:
                continue
            for panel in ("CAL", "DISC", "CONF"):
                ids = panels["ids"][panel][:64] if profile else panels["ids"][panel]; rows += sample_rows(P, panel, ids); print(f"[d01] {P.key} {panel} {len(ids)}", flush=True)
        sm = C.append_rows(smp, rows, ["pipeline", "panel", "sample_id"]) if rows else pd.read_csv(smp)
        floor = float((C.load_json(os.path.join(C.CAMP, "d00_smoke.json"), {}).get("numerical_floor") or {}).get("C_repeat_max_abs") or 0.0)
        thr = thresholds(sm, floor); C.dump_json(os.path.join(C.CAMP, "quadrant_thresholds.json"), thr); sm = assign_cells(sm, thr); sm.to_csv(smp, index=False)
        C.write_csv(os.path.join(C.CAMP, "quadrant_summary.csv"), summary(sm)); C.dump_json(os.path.join(C.CAMP, "d01_stats.json"), stats(sm, thr))
        if not panels.get("grad_ids"):
            pick_grad(sm, panels)
        fd = C.E.feeders(); srows = []
        for P in pipes:
            if P.key in have_sc or (P.role == "S" and P.tag not in SCENE_TAGS):
                continue
            srows += scene_rows(P, fd); print(f"[d01] scenes {P.key}", flush=True)
        if srows:
            C.append_rows(scp, srows, ["pipeline", "split", "sample_id"])
        sc = pd.read_csv(scp); fr = sc[(sc.split == "FR")]; st = C.load_json(os.path.join(C.CAMP, "d01_stats.json"))
        for pipe, g in fr.groupby("pipeline"):
            st[f"{pipe}|FR"] = dict(n=int(len(g)), spearman_C_hqnr=C.spearman(g.C, g.raw_original_hqnr), spearman_C_dlambda=C.spearman(g.C, g.raw_original_d_lambda), spearman_C_ds=C.spearman(g.C, g.raw_original_d_s), hqnr_mean=float(g.raw_original_hqnr.mean()), note="장면 단위; FR 에는 R_GT 가 없어 사분면을 만들지 않는다")
        for pipe, g in sc[sc.split == "RR"].groupby("pipeline"):
            st[f"{pipe}|RR"] = dict(n=int(len(g)), spearman_C_R=C.spearman(g.C, g.R_plain), R_mean=float(g.R_plain.mean()), rr_ergas_mean=float(g.rr_ergas.mean()) if "rr_ergas" in g else None, threshold_domain="native64 (RR 은 기술통계만)")
        C.dump_json(os.path.join(C.CAMP, "d01_stats.json"), st)
        t = thr["T0"]; print(f"[d01] T0 thresholds tC {t['tC']:.4f} tR {t['tR']:.5f} · degenerate {t['degenerate_metric']} · cells(T0 DISC) {sm[(sm.pipeline == 'T0') & (sm.panel == 'DISC')].cell_ref.value_counts().to_dict()}", flush=True)
