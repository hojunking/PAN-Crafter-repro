"""I25-A — Loss gradient 측정 (§14): frozen source 에서 sample 별 g_r = ∇φ L_rec, g_o = ∇φ L_off^SG (ε = bank A r=1 4 축 고정 — 결정적; 학습의 무작위 ε 와 다름을 기록), A stem/joint/head 별 norm, weighted ratio 1e-4‖g_o‖/(‖g_r‖+floor), cosine(near-zero 는 NA).
raw/gradient_interventions.csv · analysis/i25_stats.json. I25-B(되돌릴 수 있는 parameter 개입) 는 pending_compute."""
import copy, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.smec12 import common as C
from tools.smec12.i20 import detail_ids
PHASE = os.environ.get("SMEC12_PHASE", "")
OFF_EPS = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]


def part_slices(A_named):
    """A 파라미터를 stem(pan/ms stem) / joint(중간 conv·residual) / head(fc) 로 나눈 flat index 범위."""
    parts, off = {}, 0
    for name, p in A_named:
        grp = "head" if name.startswith("fc") else ("stem" if ("stem" in name or name.startswith("pan") or name.startswith("ms")) else "joint"); parts.setdefault(grp, []).append((off, off + p.numel())); off += p.numel()
    return parts


def grads(m, mg, pan, ms, lpan, gt):
    A_named = [(n, p) for n, p in m.aligner.named_parameters()]; A = [p for _, p in A_named]; mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
    o = m(pan, ms, lpan); lr = (o["y"] - gt).abs().mean(); gr = C.E.flat if hasattr(C.E, "flat") else None
    g_r = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(torch.autograd.grad(lr, A, retain_graph=True, allow_unused=True), A)]); c0 = o["delta"].detach(); lo = 0
    for ey, ex in OFF_EPS:
        e = torch.tensor([[ey, ex]], device=pan.device).expand(pan.shape[0], 2)
        with torch.no_grad():
            pe = C.E.warp_pan(pan, e)
        ce = C.E.predict_c(m.aligner, pe, mb, mg); lo = lo + C.E.offset_loss(ce, c0, e, stop_reference=True) / len(OFF_EPS)
    g_o = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(torch.autograd.grad(lo, A, allow_unused=True), A)])
    return g_r, g_o, float(lr), float(lo), part_slices(A_named)


def main(profile=False):
    with C.Stage("I25" + (f"-{PHASE}" if PHASE else ""), "aligner gradient measurement (I25-A)"):
        sm = pd.read_csv(C.p_("raw", "sample_metrics.csv")); gp = C.p_("raw", "gradient_interventions.csv"); have = set(pd.read_csv(gp).model.unique()) if os.path.exists(gp) else set(); rows = []
        for s in ("wv3", "qb", "gf2"):
            for (sensor, role, seed, tag) in C.available(s, roles=("L1E4", "L1E4REP"), tags=("best_hqnr",)):
                L = C.load_asset((sensor, role, seed), tag)
                if L.key in have or L.key not in set(sm.model):
                    continue
                m = copy.deepcopy(L.m).eval(); m.requires_grad_(False)
                for p in m.aligner.parameters():
                    p.requires_grad_(True)
                det = detail_ids(sm, s, L.key)
                for part, pick in det.items():
                    pick = pick[:16] if profile else pick
                    if len(pick) < 4:
                        continue
                    gt, ms, lpan, pan = (t.to(C.DEV) for t in C.load_patches(s, [p[0] for p in pick])); floor = None
                    for i, (sid, qd) in enumerate(pick):
                        g_r, g_o, lr, lo, parts = grads(m, L.mg, pan[i:i + 1], ms[i:i + 1], lpan[i:i + 1], gt[i:i + 1])
                        if floor is None:
                            g_r2, _, _, _, _ = grads(m, L.mg, pan[i:i + 1], ms[i:i + 1], lpan[i:i + 1], gt[i:i + 1]); floor = float((g_r - g_r2).norm())
                        nr, no = float(g_r.norm()), float(g_o.norm()); near = (nr < 10 * max(floor, 1e-12)) or (no < 10 * max(floor, 1e-12))
                        r = dict(sensor=C.SENSORS[s]["S"], model=L.key, part=part, sample_id=int(sid), source_group=C.group_of(s, sid, part), quadrant=qd, L_rec=lr, L_off=lo, g_r_norm=nr, g_o_norm=no, weighted_ratio=1e-4 * no / (nr + max(floor, 1e-12)), cos_gr_go=(None if near else float(g_r @ g_o / (nr * no))),
                                 na_zero_grad=bool(near), grad_floor=floor, eps="bank A r=1 4 축 고정(SG); 학습 ε(b=2 원판) 와 다름")
                        for grp, sl in parts.items():
                            gr_p = torch.cat([g_r[a:b] for a, b in sl]); go_p = torch.cat([g_o[a:b] for a, b in sl]); r[f"g_r_{grp}"] = float(gr_p.norm()); r[f"g_o_{grp}"] = float(go_p.norm()); r[f"cos_{grp}"] = (None if (gr_p.norm() < 10 * floor or go_p.norm() < 10 * floor) else float(gr_p @ go_p / (gr_p.norm() * go_p.norm())))
                        rows.append(r)
                    print(f"[i25] {L.key} {part} n {len(pick)} floor {floor:.2e}", flush=True)
        if rows:
            C.append_rows(gp, rows, ["model", "part", "sample_id"])
        g = pd.read_csv(gp); st = {}
        for (sensor, model, part), h in g.groupby(["sensor", "model", "part"]):
            st[f"{sensor}|{model}|{part}"] = dict(n=int(len(h)), cos_mean=float(h.cos_gr_go.mean()), cos_neg_frac=float((h.cos_gr_go < 0).mean()), na_zero=int(h.na_zero_grad.sum()), weighted_ratio_median=float(h.weighted_ratio.median()), by_quadrant={q: dict(n=int(len(x)), cos_mean=float(x.cos_gr_go.mean()), neg_frac=float((x.cos_gr_go < 0).mean()), ratio=float(x.weighted_ratio.median())) for q, x in h.groupby("quadrant")},
                                                   by_part={grp: dict(cos=float(h[f"cos_{grp}"].mean()), g_r=float(h[f"g_r_{grp}"].median()), g_o=float(h[f"g_o_{grp}"].median())) for grp in ("stem", "joint", "head") if f"cos_{grp}" in h}, spearman_q_e_vs_cos="q–e 상관을 gradient 충돌로 번역하지 않는다 (§14)")
        st["pending"] = dict(I25_B="되돌릴 수 있는 parameter 개입(−g_r/−g_o/합; CAL step ≈.02 HR px) — pending_compute"); C.dump_json(C.p_("analysis", "i25_stats.json"), st)
        for k, v in st.items():
            if isinstance(v, dict) and "cos_mean" in v:
                print(f"[i25] {k}: cos(g_r,g_o) {v['cos_mean']:+.3f} (neg {v['cos_neg_frac']:.2f}) · 1e-4‖g_o‖/‖g_r‖ {v['weighted_ratio_median']:.3g} · by quadrant {({q: round(x['cos_mean'], 3) for q, x in v['by_quadrant'].items()})}", flush=True)
