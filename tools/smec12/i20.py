"""I20 — 같은 잔여 PAN 이동에 대한 출력 민감도 (§9). raw/fixed_residual_response.csv · analysis/i20_stats.json.
I20-A: Ŷ(δ) = Φ(P, M; c0+δ), δ ∈ bank S (원본 P 에서 매번 한 번 sampling, A 재호출 없음) → s_out(r) = E|Ŷ(δ)−Ŷ(0)|, d_GT(δ) signed(음수 유지). I20-B: 위치 Jacobian h=.125/.25 (Jᵀ J/(CHW) 고유값·방향). I20-C: bank B 실제 residual r_i 와 같은 norm 의 회전 residual · 선형 예측 오차.
대상: primary L1E4(+REP 있으면) 의 상세 subset(네 집단당 ≤32 × DISC/CONF; 결과를 보기 전에 seed 로 고정)."""
import os
import numpy as np, pandas as pd, torch
from tools.smec12 import common as C
PHASE = os.environ.get("SMEC12_PHASE", "")


def detail_ids(sm, sensor, model, seed=271828 + 7):
    """네 집단당 ≤32 × DISC/CONF (source 균형: block 라운드로빈), 결과와 무관하게 고정."""
    out = {}
    for part in C.DETAIL_PARTS:
        g = sm[(sm.sensor == C.SENSORS[sensor]["S"]) & (sm.model == model) & (sm.part == part) & sm.quadrant.notna()]; rng = np.random.RandomState(seed); pick = []
        for qd in C.QUADS:
            h = g[g.quadrant == qd]; blocks = [b for _, b in h.groupby("source_group")]; rng.shuffle(blocks); i = 0
            while len([x for x in pick if x[1] == qd]) < C.DETAIL_PER_QUAD and any(len(b) > i for b in blocks):
                for b in blocks:
                    if len(b) > i and len([x for x in pick if x[1] == qd]) < C.DETAIL_PER_QUAD:
                        pick.append((int(b.sample_id.iloc[i]), qd))
                i += 1
        out[part] = pick
    return out


def response(L, sensor, ids, quads, part):
    gt, ms, lpan, pan = C.load_patches(sensor, ids); c0 = C.predict_shift(L, pan, ms); y0 = C.reconstruct_at(L, pan, ms, lpan, c0); e0 = C.e_metrics(y0, gt); rows = []; ys = {}
    for pr in C.probe_bank("S"):
        d = torch.tensor([[pr["ey"], pr["ex"]]]).expand(len(ids), 2); yd = C.reconstruct_at(L, pan, ms, lpan, c0 + d); ed = C.e_metrics(yd, gt); ys[pr["probe_id"]] = yd
        so = (yd - y0).abs(); so_roi = C.roi(so, "native64").mean(dim=(1, 2, 3)); so_full = so.mean(dim=(1, 2, 3))
        for i, sid in enumerate(ids):
            rows.append(dict(sensor=C.SENSORS[sensor]["S"], model=L.key, part=part, sample_id=int(sid), source_group=C.group_of(sensor, sid, part), quadrant=quads[i], probe_id=pr["probe_id"], r=pr["r"], theta=pr["theta_deg"], dy=pr["ey"], dx=pr["ex"], kind="S",
                             s_out_full=float(so_full[i]), s_out_roi=float(so_roi[i]), d_gt_full=float(ed["e_full"][i] - e0["e_full"][i]), d_gt_roi=float(ed["e_roi"][i] - e0["e_roi"][i]), d_edge=float(ed["edge_l1"][i] - e0["edge_l1"][i]), e0_full=float(e0["e_full"][i]), c0_y=float(c0[i, 0]), c0_x=float(c0[i, 1])))
    # I20-B Jacobian (central difference; h = .125/.25) → Jᵀ J/(CHW) 고유값
    jac = []
    for h in (0.125, 0.25):
        Jy = (ys[f"S_r{h}_t90"] - ys[f"S_r{h}_t270"]) / (2 * h); Jx = (ys[f"S_r{h}_t0"] - ys[f"S_r{h}_t180"]) / (2 * h); n = Jy[0].numel()
        for i, sid in enumerate(ids):
            a = float((Jy[i] ** 2).sum() / n); b = float((Jy[i] * Jx[i]).sum() / n); c = float((Jx[i] ** 2).sum() / n); tr = a + c; det = a * c - b * b; disc = max(tr * tr / 4 - det, 0.0) ** 0.5; l1, l2 = tr / 2 + disc, tr / 2 - disc
            ang = 0.5 * np.degrees(np.arctan2(2 * b, a - c)) if (a - c) != 0 or b != 0 else 0.0
            jac.append(dict(sensor=C.SENSORS[sensor]["S"], model=L.key, part=part, sample_id=int(sid), source_group=C.group_of(sensor, sid, part), quadrant=quads[i], h=h, J_l1=l1, J_l2=l2, J_anisotropy=(l2 / (l1 + 1e-12)), J_dir_deg=float(ang), J_y_l1=float(Jy[i].abs().mean()), J_x_l1=float(Jx[i].abs().mean()), note="현재 U-Net+sampler 의 국소 민감도(물리 noise covariance 아님)"))
    # I20-C: bank B 실제 residual 과 같은 norm 의 회전(90°) residual · 선형 예측 ΔY ≈ J r 의 오차 (h=.125 Jacobian)
    rb = C.measure_offset_bank(L, pan, ms, "B"); rmean = rb["resid"].mean(1); rot = torch.stack([-rmean[:, 1], rmean[:, 0]], 1); rc = []
    Jy, Jx = (ys["S_r0.125_t90"] - ys["S_r0.125_t270"]) / 0.25, (ys["S_r0.125_t0"] - ys["S_r0.125_t180"]) / 0.25
    for name, vec in (("resid_B_mean", rmean), ("resid_B_rot90", rot)):
        yv = C.reconstruct_at(L, pan, ms, lpan, c0 + vec); ev = C.e_metrics(yv, gt); pred = Jy * vec[:, 0].view(-1, 1, 1, 1) + Jx * vec[:, 1].view(-1, 1, 1, 1)
        for i, sid in enumerate(ids):
            rc.append(dict(sensor=C.SENSORS[sensor]["S"], model=L.key, part=part, sample_id=int(sid), source_group=C.group_of(sensor, sid, part), quadrant=quads[i], vector=name, dy=float(vec[i, 0]), dx=float(vec[i, 1]), norm=float(vec[i].norm()), s_out_full=float((yv[i] - y0[i]).abs().mean()), d_gt_full=float(ev["e_full"][i] - e0["e_full"][i]),
                           linear_pred_err=float((yv[i] - y0[i] - pred[i]).abs().mean()), linear_pred_rel=float((yv[i] - y0[i] - pred[i]).abs().mean() / ((yv[i] - y0[i]).abs().mean() + 1e-12)), q_B=float(rb["q"][i])))
    return rows, jac, rc


def stats(fr, jac, rc):
    out = {}
    for (sensor, model, part), g in fr[fr.kind == "S"].groupby(["sensor", "model", "part"]):
        d = dict(n=int(g.sample_id.nunique())); byq = {}
        for qd, h in g.groupby("quadrant"):
            byq[qd] = {f"s_out_r{r}": float(h[h.r == r].s_out_full.mean()) for r in C.RADII_S}; byq[qd].update({f"d_gt_r{r}": float(h[h.r == r].d_gt_full.mean()) for r in C.RADII_S}); byq[qd]["n"] = int(h.sample_id.nunique()); byq[qd]["d_gt_r0.5_neg_frac"] = float((h[h.r == 0.5].d_gt_full < 0).mean())
        d["by_quadrant"] = byq
        s25 = g[g.r == 0.25].groupby("sample_id").agg(s=("s_out_full", "mean"), qd=("quadrant", "first"), grp=("source_group", "first"), e0=("e0_full", "first"))
        for a, b in (("EuCd", "EuCu"), ("EuCd", "EdCd")):
            x, y = s25[s25.qd == a], s25[s25.qd == b]
            if len(x) >= 4 and len(y) >= 4:
                vals = np.r_[x.s.values, -y.s.values]; d[f"s_out_r0.25_{a}_minus_{b}"] = dict(diff=float(x.s.mean() - y.s.mean()), ci95=C.block_bootstrap(np.r_[x.s.values, y.s.values] * np.r_[np.ones(len(x)), -np.ones(len(y))] * 2, np.r_[x.grp.values, y.grp.values], fn=np.mean)["ci95"], n=[int(len(x)), int(len(y))], note="raw diff; matched(e_base/contrast) 대조는 x40 에서")
        if len(s25) >= 8:
            d["spearman_sout025_e0"] = C.spearman(s25.s, s25.e0)["rho"]
        j = jac[(jac.sensor == sensor) & (jac.model == model) & (jac.part == part) & (jac.h == 0.125)]
        if len(j):
            d["jacobian_h0125_by_quadrant"] = {qd: dict(l1=float(h.J_l1.median()), l2=float(h.J_l2.median()), anisotropy=float(h.J_anisotropy.median())) for qd, h in j.groupby("quadrant")}
            j2 = jac[(jac.sensor == sensor) & (jac.model == model) & (jac.part == part) & (jac.h == 0.25)]; d["jacobian_h_stability_corr"] = C.spearman(j.sort_values("sample_id").J_l1, j2.sort_values("sample_id").J_l1)["rho"] if len(j2) == len(j) else None
        r = rc[(rc.sensor == sensor) & (rc.model == model) & (rc.part == part)]
        if len(r):
            d["residual_B"] = {v: dict(s_out=float(h.s_out_full.mean()), d_gt=float(h.d_gt_full.mean()), linear_pred_rel_err=float(h.linear_pred_rel.median()), norm=float(h["norm"].median())) for v, h in r.groupby("vector")}
        out[f"{sensor}|{model}|{part}"] = d
    return out


def main(profile=False):
    with C.Stage("I20" + (f"-{PHASE}" if PHASE else ""), "fixed residual injection / Jacobian"):
        sm = pd.read_csv(C.p_("raw", "sample_metrics.csv")); fp = C.p_("raw", "fixed_residual_response.csv"); have = set(pd.read_csv(fp).model.unique()) if os.path.exists(fp) else set(); rows, jac, rc = [], [], []
        for s in ("wv3", "qb", "gf2"):
            for (sensor, role, seed, tag) in C.available(s, roles=("L1E4", "L1E4REP"), tags=("best_hqnr",)):
                L = C.load_asset((sensor, role, seed), tag)
                if L.key in have or L.key not in set(sm.model):
                    continue
                det = detail_ids(sm, s, L.key); C.dump_json(C.p_("manifests", f"detail_ids_{L.key.replace('|', '_')}.json"), {k: v for k, v in det.items()})
                for part, pick in det.items():
                    pick = pick[:16] if profile else pick
                    if not pick:
                        continue
                    r1, j1, c1 = response(L, s, [p[0] for p in pick], [p[1] for p in pick], part); rows += r1; jac += j1; rc += c1; print(f"[i20] {L.key} {part} n {len(pick)}", flush=True)
        if rows:
            C.append_rows(fp, rows, ["model", "part", "sample_id", "probe_id"]); C.append_rows(C.p_("raw", "position_jacobian.csv"), jac, ["model", "part", "sample_id", "h"]); C.append_rows(C.p_("raw", "residual_direction.csv"), rc, ["model", "part", "sample_id", "vector"])
        fr = pd.read_csv(fp); jd = pd.read_csv(C.p_("raw", "position_jacobian.csv")); rd = pd.read_csv(C.p_("raw", "residual_direction.csv")); st = stats(fr, jd, rd); C.dump_json(C.p_("analysis", "i20_stats.json"), st)
        for k, v in st.items():
            if "s_out_r0.25_EuCd_minus_EuCu" in v:
                print(f"[i20] {k}: s_out(.25) EuCd−EuCu {v['s_out_r0.25_EuCd_minus_EuCu']['diff']:+.5f} CI {v['s_out_r0.25_EuCd_minus_EuCu']['ci95']}", flush=True)
