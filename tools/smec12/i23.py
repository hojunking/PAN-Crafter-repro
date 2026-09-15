"""I23-B — 같은 U-Net 에서 correction 치환 (§12): I-L(learned) · I-Z(0) · I-W(−c0) · I-C(CAL median) · I-S(source-matched shuffle 10 derangement). g_corr = e(I-Z) − e(I-L) (coupling: 같은 GT).
raw/correction_interventions.csv · analysis/i23_stats.json. I23-A(독립 estimator)·I23-C(landscape)·I23-D 는 이번 release 에 없음(pending_compute)."""
import os
import numpy as np, pandas as pd, torch
from tools.smec12 import common as C
from tools.smec12.i20 import detail_ids
PHASE = os.environ.get("SMEC12_PHASE", "")


def derangements(n, k=10, seed=271828 + 11):
    rng = np.random.RandomState(seed); out = []
    while len(out) < k:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            out.append(p)
    return out


def interventions(L, sensor, ids, quads, part, c_cal):
    gt, ms, lpan, pan = C.load_patches(sensor, ids); c0 = C.predict_shift(L, pan, ms); n = len(ids); modes = {"I-L": c0, "I-Z": torch.zeros_like(c0), "I-W": -c0, "I-C": c_cal.expand(n, 2).clone()}
    grp = np.array([C.group_of(sensor, s, part) for s in ids])
    for k, p in enumerate(derangements(n)):
        modes[f"I-S{k + 1}"] = c0[p]
    rows = []
    for name, cv in modes.items():
        y = C.reconstruct_at(L, pan, ms, lpan, cv); em = C.e_metrics(y, gt)
        for i, sid in enumerate(ids):
            rows.append(dict(sensor=C.SENSORS[sensor]["S"], model=L.key, part=part, sample_id=int(sid), source_group=grp[i], quadrant=quads[i], mode=name, applied_dy=float(cv[i, 0]), applied_dx=float(cv[i, 1]), c0_dy=float(c0[i, 0]), c0_dx=float(c0[i, 1]), applied_minus_c0=float((cv[i] - c0[i]).norm()),
                             sampler_calls=1, e_full=float(em["e_full"][i]), e_roi=float(em["e_roi"][i]), edge_l1=float(em["edge_l1"][i]), e_band_max=float(em["e_band"][i].max()), invalid_support=bool(cv[i].abs().max() + 4 > L.mg + 4 + 2.0)))
    return rows


def stats(ci):
    out = {}
    for (sensor, model, part), g in ci.groupby(["sensor", "model", "part"]):
        piv = g.pivot_table(index="sample_id", columns="mode", values="e_full"); grp = g.groupby("sample_id").source_group.first().reindex(piv.index).values; qd = g.groupby("sample_id").quadrant.first().reindex(piv.index)
        d = dict(n=int(len(piv)), g_corr_zero_minus_learned=C.block_bootstrap((piv["I-Z"] - piv["I-L"]).values, grp), wrong_minus_learned=C.block_bootstrap((piv["I-W"] - piv["I-L"]).values, grp), cal_minus_learned=C.block_bootstrap((piv["I-C"] - piv["I-L"]).values, grp),
                 shuffle_minus_learned=float(np.nanmean([(piv[c] - piv["I-L"]).mean() for c in piv.columns if c.startswith("I-S")])), gain_pos_frac=float(((piv["I-Z"] - piv["I-L"]) > 0).mean()), coupling_note="gain = e_zero − e_learned 는 e_learned 와 항을 공유 (§4.5 6)")
        d["by_quadrant"] = {q: dict(n=int((qd == q).sum()), g_corr=float((piv["I-Z"] - piv["I-L"])[qd == q].mean()), e_learned=float(piv["I-L"][qd == q].mean()), shuffle_minus_learned=float(np.nanmean([(piv[c] - piv["I-L"])[qd == q].mean() for c in piv.columns if c.startswith("I-S")]))) for q in C.QUADS if (qd == q).any()}
        out[f"{sensor}|{model}|{part}"] = d
    return out


def main(profile=False):
    with C.Stage("I23" + (f"-{PHASE}" if PHASE else ""), "correction substitution (I23-B)"):
        sm = pd.read_csv(C.p_("raw", "sample_metrics.csv")); cp = C.p_("raw", "correction_interventions.csv"); have = set(pd.read_csv(cp).model.unique()) if os.path.exists(cp) else set(); rows = []
        for s in ("wv3", "qb", "gf2"):
            for (sensor, role, seed, tag) in C.available(s, roles=("L1E4", "L1E4REP"), tags=("best_hqnr",)):
                L = C.load_asset((sensor, role, seed), tag)
                if L.key in have or L.key not in set(sm.model):
                    continue
                cal = sm[(sm.model == L.key) & (sm.part == "CAL")]; c_cal = torch.tensor([[float(cal.c0_y.median()), float(cal.c0_x.median())]]); det = detail_ids(sm, s, L.key)
                for part, pick in det.items():
                    pick = pick[:16] if profile else pick
                    if len(pick) >= 4:
                        rows += interventions(L, s, [p[0] for p in pick], [p[1] for p in pick], part, c_cal); print(f"[i23] {L.key} {part} n {len(pick)} · CAL median c {c_cal.tolist()}", flush=True)
        if rows:
            C.append_rows(cp, rows, ["model", "part", "sample_id", "mode"])
        ci = pd.read_csv(cp); st = stats(ci); st["pending"] = dict(I23_A="independent native geometry proxy (secondary estimator) — pending_compute", I23_C="correction landscape (grid c_rec vs c_geo) — pending_compute", I23_D="P0/L000 landscape 반복 — pending_compute"); C.dump_json(C.p_("analysis", "i23_stats.json"), st)
        for k, v in st.items():
            if isinstance(v, dict) and "g_corr_zero_minus_learned" in v:
                print(f"[i23] {k}: g_corr {v['g_corr_zero_minus_learned']['point']:+.5f} CI {v['g_corr_zero_minus_learned']['ci95']} · pos {v['gain_pos_frac']:.2f} · by quadrant {({q: round(x['g_corr'], 5) for q, x in v['by_quadrant'].items()})}", flush=True)
