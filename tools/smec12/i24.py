"""I24-A — 두 번 sampling 의 정확한 출력 분해 (§13): Y_seq = M + F([W(W(P,ε), c_ε), M]) · Y_single = Φ(P, M; ε + c_ε) · Y_0 = Φ(P, M; c_0); Y_seq − Y_0 = (Y_seq − Y_single) + (Y_single − Y_0). known-inverse(c0−ε) 도 seq vs single.
raw/interpolation_controls.csv · analysis/i24_stats.json. I24-B(matched filtering control)·I24-C(amplitude/phase) 는 pending_compute(control_not_identified 로 기록)."""
import os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.smec12 import common as C
from tools.smec12.i20 import detail_ids
PHASE = os.environ.get("SMEC12_PHASE", "")


def decomposition(L, sensor, ids, quads, part):
    gt, ms, lpan, pan = C.load_patches(sensor, ids); c0 = C.predict_shift(L, pan, ms); y0 = C.reconstruct_at(L, pan, ms, lpan, c0); mb = F.interpolate(ms.to(C.DEV), scale_factor=4, mode="bicubic"); rows = []
    for pr in [p for p in C.probe_bank("B") if p["r"] in (0.5, 1.0, 2.0)] + [dict(p, probe_id=p["probe_id"] + "_axis") for p in C.probe_bank("A") if p["r"] == 1.0]:
        e = torch.tensor([[pr["ey"], pr["ex"]]]).expand(len(ids), 2)
        with torch.no_grad():
            pe = C.E.warp_pan(pan.to(C.DEV), e.to(C.DEV)); ce = C.E.predict_c(L.m.aligner, pe, mb, L.mg).cpu()
        y_seq = C.reconstruct_at(L, pe.cpu(), ms, lpan, ce); y_single = C.reconstruct_at(L, pan, ms, lpan, e + ce); y_kinv_seq = C.reconstruct_at(L, pe.cpu(), ms, lpan, c0 - e); y_kinv_single = C.reconstruct_at(L, pan, ms, lpan, c0)
        for i, sid in enumerate(ids):
            rows.append(dict(sensor=C.SENSORS[sensor]["S"], model=L.key, part=part, sample_id=int(sid), source_group=C.group_of(sensor, sid, part), quadrant=quads[i], probe_id=pr["probe_id"], r=pr["r"], theta=pr["theta_deg"], dy=pr["ey"], dx=pr["ex"], c_eps_dy=float(ce[i, 0]), c_eps_dx=float(ce[i, 1]), resid_norm=float((ce[i] + e[i] - c0[i]).norm()),
                             seq_minus_zero_l1=float((y_seq[i] - y0[i]).abs().mean()), seq_minus_single_l1=float((y_seq[i] - y_single[i]).abs().mean()), single_minus_zero_l1=float((y_single[i] - y0[i]).abs().mean()),
                             d_gt_seq=float((y_seq[i] - gt[i]).abs().mean() - (y0[i] - gt[i]).abs().mean()), d_gt_single=float((y_single[i] - gt[i]).abs().mean() - (y0[i] - gt[i]).abs().mean()),
                             known_inverse_seq_minus_native=float((y_kinv_seq[i] - y0[i]).abs().mean()), known_inverse_gt_delta=float((y_kinv_seq[i] - gt[i]).abs().mean() - (y0[i] - gt[i]).abs().mean()), note="L1 norm 은 가법이 아니다(signed 차이만 망원합)"))
    return rows


def stats(df):
    out = {}
    for (sensor, model, part), g in df.groupby(["sensor", "model", "part"]):
        d = dict(n=int(g.sample_id.nunique()))
        for r in (0.5, 1.0, 2.0):
            h = g[(g.r == r) & (~g.probe_id.str.endswith("_axis"))]
            if len(h):
                d[f"r{r}"] = dict(seq_minus_single=float(h.seq_minus_single_l1.mean()), single_minus_zero=float(h.single_minus_zero_l1.mean()), seq_minus_zero=float(h.seq_minus_zero_l1.mean()), interp_share=float(h.seq_minus_single_l1.mean() / (h.seq_minus_zero_l1.mean() + 1e-12)), d_gt_seq=float(h.d_gt_seq.mean()), d_gt_single=float(h.d_gt_single.mean()), known_inverse_seq_vs_native=float(h.known_inverse_seq_minus_native.mean()))
        ax = g[g.probe_id.str.endswith("_axis")]
        if len(ax):
            d["axis_r1_known_inverse_seq_vs_native"] = float(ax.known_inverse_seq_minus_native.mean()); d["diag_r1_known_inverse_seq_vs_native"] = float(g[(g.r == 1.0) & (~g.probe_id.str.endswith("_axis"))].known_inverse_seq_minus_native.mean())
        d["by_quadrant_r0.5"] = {q: dict(seq_minus_single=float(h.seq_minus_single_l1.mean()), single_minus_zero=float(h.single_minus_zero_l1.mean())) for q, h in g[(g.r == 0.5) & (~g.probe_id.str.endswith("_axis"))].groupby("quadrant")}
        out[f"{sensor}|{model}|{part}"] = d
    out["pending"] = dict(I24_B="matched filtering control — control_not_identified (pending_compute)", I24_C="amplitude/phase decomposition — pending_compute (helper 검증 전)"); return out


def main(profile=False):
    with C.Stage("I24" + (f"-{PHASE}" if PHASE else ""), "sequential vs single warp decomposition (I24-A)"):
        sm = pd.read_csv(C.p_("raw", "sample_metrics.csv")); ip = C.p_("raw", "interpolation_controls.csv"); have = set(pd.read_csv(ip).model.unique()) if os.path.exists(ip) else set(); rows = []
        for s in ("wv3", "qb", "gf2"):
            for (sensor, role, seed, tag) in C.available(s, roles=("L1E4",), tags=("best_hqnr",)):
                L = C.load_asset((sensor, role, seed), tag)
                if L.key in have or L.key not in set(sm.model):
                    continue
                det = detail_ids(sm, s, L.key)
                for part, pick in det.items():
                    pick = pick[:16] if profile else pick
                    if len(pick) >= 4:
                        rows += decomposition(L, s, [p[0] for p in pick], [p[1] for p in pick], part); print(f"[i24] {L.key} {part} n {len(pick)}", flush=True)
        if rows:
            C.append_rows(ip, rows, ["model", "part", "sample_id", "probe_id"])
        st = stats(pd.read_csv(ip)); C.dump_json(C.p_("analysis", "i24_stats.json"), st)
        for k, v in st.items():
            if isinstance(v, dict) and "r1.0" in v:
                print(f"[i24] {k}: r1 seq−single {v['r1.0']['seq_minus_single']:.5f} · single−0 {v['r1.0']['single_minus_zero']:.5f} · interp share {v['r1.0']['interp_share']:.2f} · known-inverse seq vs native {v['r1.0']['known_inverse_seq_vs_native']:.2e}", flush=True)
