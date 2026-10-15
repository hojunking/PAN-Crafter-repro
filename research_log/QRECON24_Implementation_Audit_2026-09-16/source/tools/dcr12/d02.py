"""D02 — 같은 backbone 에서 correction 만 바꾸는 대조 (계획 §6). correction_interventions.csv (DISC128·RR20·FR20/FR8) · closure_invariance.csv (§6.3) · synthetic_composition.csv (§6.4) · d02_stats.json (B_R, B_Q, ±bias 평균/최선/최악).
F ∈ {T0, 각 Student best_hqnr}. 모든 개입은 원래 PAN 에서 같은 sampler 를 정확히 한 번 호출한다(delta_override). 다시 부르면 새 pipeline 분만 덧붙인다."""
import os
import numpy as np, pandas as pd, torch
from tools.dcr12 import common as C
PHASE = os.environ.get("DCR12_PHASE", "")
B = C.BIAS_HR


def modes_of(c0):
    z = torch.zeros_like(c0); return {"I0": c0, "IZ": z, "IN": -c0, "IB+y": c0 + torch.tensor([B, 0.0]), "IB-y": c0 - torch.tensor([B, 0.0]), "IB+x": c0 + torch.tensor([0.0, B]), "IB-x": c0 - torch.tensor([0.0, B])}


def patch_interventions(P, ids, cells):
    gt, ms, lpan, pan = C.load_patches(ids); _, c0 = C.reconstruct(P, pan, ms, lpan); rows = []
    for mode, cv in modes_of(c0).items():
        y, _ = C.reconstruct(P, pan, ms, lpan, delta_override=cv); rm = C.r_metrics(y, gt)
        for i, sid in enumerate(ids):
            rows.append(dict(pipeline=P.key, role=P.role, seed=P.seed, tag=P.tag, step=P.step, scale="native64", sample_id=int(sid), group=C.group_of(sid), cell_T0=cells.get(int(sid)), mode=mode, applied_dy=float(cv[i, 0]), applied_dx=float(cv[i, 1]), c0_dy=float(c0[i, 0]), c0_dx=float(c0[i, 1]),
                             sampler_calls=1, domain="full64 + interior32", R_plain=float(rm["R_plain"][i]), R_interior=float(rm["R_interior"][i]), edge_plain=float(rm["edge_plain"][i]), support_ok=bool(abs(float(cv[i, 0])) + abs(float(cv[i, 1])) + 4 <= P.mg + 4 + B + 1e-9)))
    return rows


def scene_interventions(P, fd, fr8):
    from pa.evalviews import scene_views, VIEWS
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    from utils import reduced_metrics
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rows = []
    for scale, ds in (("rr256", fd["rr"]), ("fr512", fd["fr"])):
        for i in range(len(ds)):
            t = [x.unsqueeze(0) for x in ds[i]]; pan, ms, lpan = t[-1], t[-3], t[-2]; _, c0 = C.reconstruct(P, pan, ms, lpan)
            for mode, cv in modes_of(c0).items():
                if scale == "fr512" and mode.startswith("IB") and i not in fr8:
                    continue                                                                    # ±bias 는 FR8 고정 subset 만 (partial) — 공식 HQNR 은 전체 20 장의 I0/IZ/IN
                y, _ = C.reconstruct(P, pan, ms, lpan, delta_override=cv); r = dict(pipeline=P.key, role=P.role, seed=P.seed, tag=P.tag, step=P.step, scale=scale, sample_id=i, group=f"{scale[:2]}{i:02d}", cell_T0=None, mode=mode,
                                                                                     applied_dy=float(cv[0, 0]), applied_dx=float(cv[0, 1]), c0_dy=float(c0[0, 0]), c0_dx=float(c0[0, 1]), sampler_calls=1, partial=bool(scale == "fr512" and mode.startswith("IB")))
                if scale == "rr256":
                    gt = t[0]; r.update(domain="full256 + roi192", R_plain=float((y - gt).abs().mean()), R_interior=float(C.E.l1_roi(y, gt, "rr256")[0]), edge_plain=float(C.E.edge_l1_roi(y, gt, "rr256")[0]), **{f"rr_{k}": float(v) for k, v in reduced_metrics(x_true=gt.to(C.DEV), x_pred=y.to(C.DEV), max_pixel=mp).items()})
                else:
                    sr = ((y[0].clip(-1, 1).float().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]; d = cv[0].double().numpy()
                    pa_eval = C.E.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d)[None])[0, 0].numpy(); v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp)
                    r.update(domain="raw_original full frame (+valid views)", eligible_valid_views=bool(ok), **{f"{k}_{q}": float(v[k][q]) for k in VIEWS for q in ("hqnr", "d_lambda", "d_s", "fscc")})
                rows.append(r)
    return rows


def closure_rows(P, ids):
    """§6.3: 예측된 두 correction 에 같은 b 를 더하면 C 는 수치오차 안에서 같지만 R 은 변할 수 있다."""
    gt, ms, lpan, pan = C.load_patches(ids); base = C.consistency(P, pan, ms); rows = []; _, c0 = C.reconstruct(P, pan, ms, lpan); y0, _ = C.reconstruct(P, pan, ms, lpan, delta_override=c0); R0 = C.r_metrics(y0, gt)["R_plain"]
    for name, b in (("b+y", (B, 0.0)), ("b+x", (0.0, B))):
        cb = C.consistency(P, pan, ms, c_bias=b); yb, _ = C.reconstruct(P, pan, ms, lpan, delta_override=c0 + torch.tensor(b)); Rb = C.r_metrics(yb, gt)["R_plain"]
        for i, sid in enumerate(ids):
            rows.append(dict(pipeline=P.key, sample_id=int(sid), group=C.group_of(sid), bias=name, C_base=float(base["C"][i]), C_biased=float(cb["C"][i]), C_abs_diff=float(abs(cb["C"][i] - base["C"][i])), R_base=float(R0[i]), R_biased=float(Rb[i]), R_diff=float(Rb[i] - R0[i])))
    return rows


def composition_rows(P, ids):
    """§6.4: Z_seq = F(W(W(P,ε), c_ε)) vs Z_comp = F(W(P, ε + c_ε)); 추정용 P_ε 는 같다. ε = main probe r=1.0 8 방향."""
    gt, ms, lpan, pan = C.load_patches(ids); rows = []; mb = torch.nn.functional.interpolate(ms.to(C.DEV), scale_factor=4, mode="bicubic")
    for pr in [p for p in C.probe_set("main") if p["r"] == 1.0]:
        e = torch.tensor([[pr["ey"], pr["ex"]]]).expand(len(ids), 2)
        with torch.no_grad():
            pe = C.E.warp_pan(pan.to(C.DEV), e.to(C.DEV)); ce = C.E.predict_c(P.m.aligner, pe, mb, P.mg).cpu()
        yseq, _ = C.reconstruct(P, pe.cpu(), ms, lpan, delta_override=ce); ycomp, _ = C.reconstruct(P, pan, ms, lpan, delta_override=e + ce)
        rs, rc = C.r_metrics(yseq, gt)["R_plain"], C.r_metrics(ycomp, gt)["R_plain"]; dz = (yseq - ycomp).abs().mean(dim=(1, 2, 3))
        for i, sid in enumerate(ids):
            rows.append(dict(pipeline=P.key, sample_id=int(sid), group=C.group_of(sid), probe_id=pr["probe_id"], eps_dy=pr["ey"], eps_dx=pr["ex"], c_eps_dy=float(ce[i, 0]), c_eps_dx=float(ce[i, 1]), R_seq=float(rs[i]), R_comp=float(rc[i]), R_seq_minus_comp=float(rs[i] - rc[i]), Z_abs_diff=float(dz[i])))
    return rows


def stats(ci):
    out = {}
    for (pipe, scale), g in ci.groupby(["pipeline", "scale"]):
        key = "raw_original_hqnr" if scale == "fr512" else "R_plain"; piv = g.pivot_table(index="sample_id", columns="mode", values=key); grp = g.groupby("sample_id").group.first().reindex(piv.index).values
        d = dict(n=int(len(piv)), metric=key)
        if scale == "fr512":
            d["B_Q_learned_minus_zero"] = C.block_bootstrap((piv["I0"] - piv["IZ"]).values, grp); d["learned_minus_wrong"] = C.block_bootstrap((piv["I0"] - piv["IN"]).values, grp)
            d["hqnr_I0"] = float(piv["I0"].mean()); d["hqnr_IZ"] = float(piv["IZ"].mean()); d["hqnr_IN"] = float(piv["IN"].mean())
            pb = piv.dropna(subset=[m for m in ("IB+y", "IB-y", "IB+x", "IB-x") if m in piv]) if any(m in piv for m in ("IB+y",)) else piv.iloc[0:0]
            if len(pb):
                bias = pb[["IB+y", "IB-y", "IB+x", "IB-x"]].sub(pb["I0"], axis=0); d["bias_hqnr_delta"] = dict(n=int(len(pb)), partial=True, mean=float(bias.values.mean()), best=float(bias.values.max()), worst=float(bias.values.min()), per_mode_mean=bias.mean().to_dict(), oracle_note="GT/HQNR 로 고른 best bias 는 oracle diagnostic 이지 method 가 아니다")
        else:
            d["B_R_zero_minus_learned"] = C.block_bootstrap((piv["IZ"] - piv["I0"]).values, grp); d["wrong_minus_learned"] = C.block_bootstrap((piv["IN"] - piv["I0"]).values, grp)
            bias = piv[["IB+y", "IB-y", "IB+x", "IB-x"]].sub(piv["I0"], axis=0); d["bias_R_delta"] = dict(mean=float(bias.values.mean()), best=float(bias.values.min()), worst=float(bias.values.max()), per_mode_mean=bias.mean().to_dict(), frac_any_bias_better=float((bias.min(axis=1) < 0).mean()))
            d["R_I0"] = float(piv["I0"].mean()); d["R_IZ"] = float(piv["IZ"].mean()); d["R_IN"] = float(piv["IN"].mean()); d["gain_pos_frac"] = float(((piv["IZ"] - piv["I0"]) > 0).mean())
            if "cell_T0" in g and g.cell_T0.notna().any():
                cells = g.groupby("sample_id").cell_T0.first().reindex(piv.index)
                d["B_R_by_cell_T0"] = {c: dict(n=int((cells == c).sum()), mean=float((piv["IZ"] - piv["I0"])[cells == c].mean())) for c in "ABCD" if (cells == c).any()}
        out[f"{pipe}|{scale}"] = d
    return out


def main(profile=False):
    with C.Stage("D02" + (f"-{PHASE}" if PHASE else ""), "correction interventions / closure invariance / composition"):
        panels = C.make_panels(); ids = panels["ids"]["DISC128"][:32] if profile else panels["ids"]["DISC128"]; fr8 = set(panels["ids"]["FR8"])
        sm = pd.read_csv(os.path.join(C.CAMP, "sample_metrics.csv")); t0 = sm[(sm.pipeline == "T0") & (sm.panel == "DISC")]; cells = dict(zip(t0.sample_id.astype(int), t0.cell_ref))
        cip = os.path.join(C.CAMP, "correction_interventions.csv"); have = set(pd.read_csv(cip).pipeline.unique()) if os.path.exists(cip) else set()
        pipes = [C.load_pipe("T")] + [C.load_pipe("S", ck, seed, tag) for ck, seed, tag in C.available_students(("best_hqnr",))]; fd = C.E.feeders(); rows, crows, srows = [], [], []
        for P in pipes:
            if P.key in have:
                continue
            rows += patch_interventions(P, ids, cells); rows += ([] if profile else scene_interventions(P, fd, fr8)); crows += closure_rows(P, ids); srows += composition_rows(P, ids); print(f"[d02] {P.key} done", flush=True)
        if rows:
            C.append_rows(cip, rows, ["pipeline", "scale", "sample_id", "mode"]); C.append_rows(os.path.join(C.CAMP, "closure_invariance.csv"), crows, ["pipeline", "sample_id", "bias"]); C.append_rows(os.path.join(C.CAMP, "synthetic_composition.csv"), srows, ["pipeline", "sample_id", "probe_id"])
        ci = pd.read_csv(cip); st = stats(ci); cl = pd.read_csv(os.path.join(C.CAMP, "closure_invariance.csv")); co = pd.read_csv(os.path.join(C.CAMP, "synthetic_composition.csv"))
        st["closure_invariance"] = {p: dict(C_abs_diff_max=float(g.C_abs_diff.max()), R_diff_mean=float(g.R_diff.mean()), R_diff_abs_mean=float(g.R_diff.abs().mean()), n=int(len(g)), counterexample=bool(g.C_abs_diff.max() < 1e-5 and g.R_diff.abs().mean() > 1e-5)) for p, g in cl.groupby("pipeline")}
        st["synthetic_composition"] = {p: dict(R_seq_mean=float(g.R_seq.mean()), R_comp_mean=float(g.R_comp.mean()), seq_minus_comp_mean=float(g.R_seq_minus_comp.mean()), Z_abs_diff_mean=float(g.Z_abs_diff.mean()), n=int(len(g))) for p, g in co.groupby("pipeline")}
        C.dump_json(os.path.join(C.CAMP, "d02_stats.json"), st)
        for k, v in st.items():
            if "|native64" in k:
                print(f"[d02] {k}: B_R(zero−learned) {v['B_R_zero_minus_learned']['point']:+.5f} CI {v['B_R_zero_minus_learned']['ci95']} · pos {v['gain_pos_frac']:.3f}", flush=True)
            if "|fr512" in k:
                print(f"[d02] {k}: B_Q(learned−zero HQNR) {v['B_Q_learned_minus_zero']['point']:+.5f} CI {v['B_Q_learned_minus_zero']['ci95']}", flush=True)
