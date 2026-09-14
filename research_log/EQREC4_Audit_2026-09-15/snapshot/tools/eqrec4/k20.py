"""K20 — 조건부 cue gate 검증 pilot (계획 §13): EA / EAQ utility 표 → ρ, 다섯 arm × 5K update, D locked holdout 이 주 endpoint."""
import copy, json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV
from tools.eqrec4.k10 import k_step, cells_for_pair, eval_l1, GAMMA, LR_U, LR_A

UPDATES = 5000; BATCH = 48; SHRINK = 4.0; ARMS = ("R", "U", "EA", "EAQ", "EAQ_SHUF"); SAVE_AT = (0, 2500, 5000)


def utility_tables(ut, pair, dfB):
    """EA(4 cell) / EAQ(8 cell) 의 상대 U_soft-hard 평균, 부모로의 shrinkage(4), n_g ≤ 독립 fit-pool 수 (source 중복이면 보수적)."""
    g = ut[(ut.pair == pair) & (ut.repeat == 0)].copy(); g["ea"] = g.cell.map(ea_key); u_all = float(g.U_soft_hard_rel.mean()) if len(g) else 0.0
    ea = {}
    for k, s in g.groupby("ea"):
        n = int(min(len(s), s.n_source_groups.sum())); ea[k] = dict(n=n, raw=float(s.U_soft_hard_rel.mean()), value=(n * float(s.U_soft_hard_rel.mean()) + SHRINK * u_all) / (n + SHRINK))
    eaq = {}
    for k, s in g.groupby("cell"):
        parent = ea_key(k); n = int(min(len(s), s.n_source_groups.sum())); pv = ea.get(parent, dict(value=u_all))["value"]; eaq[k] = dict(n=n, raw=float(s.U_soft_hard_rel.mean()), value=(n * float(s.U_soft_hard_rel.mean()) + SHRINK * pv) / (n + SHRINK), parent=parent)
    for key in ("EdApos", "EdAneg", "EuApos", "EuAneg"):
        ea.setdefault(key, dict(n=0, raw=None, value=u_all))
    for e_ in ("Ed", "Eu"):
        for c_ in ("Cd", "Cu"):
            for a_ in ("Apos", "Aneg"):
                eaq.setdefault(f"{e_}{c_}_{a_}", dict(n=0, raw=None, value=ea[e_ + a_]["value"], parent=e_ + a_))
    noise = (C.load_json(os.path.join(CAMP, f"k10_noise_{pair}.json")) or {}).get("noise_scale_rel") or 0.0; vals = g.U_soft_hard_rel.values
    s_u = max(1.4826 * float(np.median(np.abs(vals - np.median(vals)))) if len(vals) else 0.0, noise, 1e-12)
    resolved = bool(len(vals) and np.abs(vals).max() > 1e-9)
    return dict(u_all=u_all, EA=ea, EAQ=eaq, s_u=s_u, noise_scale=noise, no_resolved_utility_signal=(not resolved))


def rho_for(dfB, table, kind, tab):
    """ρ_g = σ(clip(ū_g/s_u, −4, 4) + b_mass), b_mass: B 평균 ρ = 0.5 (bisection)."""
    if tab["no_resolved_utility_signal"]:
        return np.full(len(dfB), 0.5), dict(b_mass=0.0, mean_rho=0.5, note="no_resolved_utility_signal")
    key = dfB.cell.map(ea_key) if kind == "EA" else dfB.cell; u = np.array([table[k]["value"] for k in key]); z = np.clip(u / tab["s_u"], -4, 4)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        b = (lo + hi) / 2; m = float((1 / (1 + np.exp(-(z + b)))).mean())
        if m > 0.5:
            hi = b
        else:
            lo = b
    b = (lo + hi) / 2; rho = 1 / (1 + np.exp(-(z + b))); return rho, dict(b_mass=float(b), mean_rho=float(rho.mean()), rho_std=float(rho.std()))


def conditional_shuffle(dfB, seed):
    """EA cell 안에서 (rank e, rank a) 가 가까운 블록(16, 최소 8) 을 만들고 블록 안에서 q-label(c_low) 만 derangement."""
    rng = np.random.RandomState(seed); cl = dfB.c_low.values.copy(); status = np.array(["ok"] * len(dfB), dtype=object); ea = dfB.cell.map(ea_key)
    for k, s in dfB.groupby(ea):
        order = s.assign(re=s.e_T.rank(), ra=s.a_T.rank()).sort_values(["re", "ra"]).index.values; blocks = [order[i:i + 16] for i in range(0, len(order), 16)]
        if len(blocks) > 1 and len(blocks[-1]) < 8:
            blocks[-2] = np.concatenate([blocks[-2], blocks[-1]]); blocks = blocks[:-1]
        for blk in blocks:
            if len(blk) < 2:
                status[dfB.index.get_indexer(blk)] = "shuffle_unmatched"; continue
            pos = dfB.index.get_indexer(blk); vals = cl[pos].copy(); cl[pos] = np.roll(vals, 1)
    out = dfB.copy(); out["c_low"] = cl; out["cell"] = out.apply(lambda r: ("Ed" if r.e_low else "Eu") + ("Cd" if r.c_low else "Cu") + ("_Apos" if r.a_pos else "_Aneg"), axis=1); return out, status


@torch.no_grad()
def eval_scenes(m, fd):
    from pa.evalviews import scene_views, VIEWS
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    from utils import reduced_metrics
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; rr, fr = [], []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); y = m(pan, ms, lpan)["y"]; rr.append(dict(l1=float((y - gt).abs().mean()), **{k: float(v) for k, v in reduced_metrics(x_true=gt, x_pred=y, max_pixel=mp).items()}))
    for i in range(len(fd["fr"])):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["fr"][i]); o = m(pan, ms, lpan); d = o["delta"][0].double().cpu().numpy(); sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]
        v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, C.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d)[None])[0, 0].numpy(), sensor, wald, d, 4, mp); fr.append({f"{k}_{q}": float(v[k][q]) for k in VIEWS for q in ("hqnr", "d_lambda", "d_s", "fscc")})
    return dict(**{f"rr_{k}": float(np.mean([r[k] for r in rr])) for k in rr[0]}, **{f"fr_{k}": float(np.mean([r[k] for r in fr])) for k in fr[0]}, fr_per_scene_hqnr=[r["raw_original_hqnr"] for r in fr])


def train_arm(S, src_state, cfg, arm, Bt, tpB, rho, order, fd, Dt, tag, updates=UPDATES, save_at=SAVE_AT):
    from diffusers.optimization import get_scheduler
    from safetensors.torch import save_file
    gt, ms, lpan, pan = Bt; m = copy.deepcopy(S.m).train(); m.load_state_dict(src_state); m.requires_grad_(True); opt = torch.optim.AdamW([dict(params=list(m.backbone.parameters()), lr=LR_U), dict(params=list(m.aligner.parameters()), lr=LR_A)], lr=LR_U, weight_decay=float(cfg.get("weight_decay", 0.01)))
    sch = get_scheduler("cosine", optimizer=opt, num_warmup_steps=100, num_training_steps=updates); gen = torch.Generator().manual_seed(2026); logs, ends = [], []; out_dir = os.path.join(CAMP, "k20_states", tag, arm); os.makedirs(out_dir, exist_ok=True)
    def endpoint(u):
        m.eval(); e = eval_l1(m, *Dt); r = dict(arm=arm, update=u, D_l1=e["l1"], D_l1_roi=e["l1_roi"], D_edge=e["edge"]); np.save(os.path.join(out_dir, f"D_per_sample_{u}.npy"), e["per_sample"])
        if u in (0, updates):
            r.update(eval_scenes(m, fd)); save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, os.path.join(out_dir, f"step-{u}.safetensors"))
        m.train(); return r
    ends.append(endpoint(0)); rho_t = torch.tensor(rho, dtype=torch.float32)
    for u in range(updates):
        sel = order[u]; b = tuple(t[sel].to(DEV) for t in (gt, ms, lpan, pan)); r = k_step(m, S.mg, b, u, gen, tpB[sel].to(DEV), ("K20" if arm != "R" else "R0"), gamma=(GAMMA if arm != "R" else 0.0), rho=(rho_t[sel].to(DEV) if arm != "R" else None), opt=opt); sch.step()
        if u % 500 == 0 or u == updates - 1:
            logs.append(dict(arm=arm, update=u, lr_U=opt.param_groups[0]["lr"], **{k: v for k, v in r.items()}, extra_over_base_U=(r["g_extra_U"] / (r["g_base_U"] + 1e-12)), mean_rho_batch=(float(rho_t[sel].mean()) if arm != "R" else None)))
        if (u + 1) in save_at:
            ends.append(endpoint(u + 1))
    return ends, logs


def ea_key(cell):
    """'EdCd_Apos' → 'EdApos' (e 집단 + a 부호; q 집단 제거)."""
    return cell[:2] + cell[5:]


def main(profile=False, pair="A"):
    man = C.make_manifest(); fd = C.feeders(); ut = C.read_csv(os.path.join(CAMP, "k10_utility.csv"))
    with C.Stage("K20" + ("-profile" if profile else ""), f"conditional gate pilot (pair {pair})"):
        T = C.load_model(*C.PAIRS[pair]["T"]); S = C.load_model(*C.PAIRS[pair]["S"]); cells, thr = cells_for_pair(pair, T, S, man); dfB = cells["B"]["df"]; tab = utility_tables(ut, pair, dfB)
        rho_ea, iea = rho_for(dfB, tab["EA"], "EA", tab); rho_eaq, ieaq = rho_for(dfB, tab["EAQ"], "EAQ", tab); dfS, sstat = conditional_shuffle(dfB, C.SPLIT_SEED + 5); rho_sh, ish = rho_for(dfS, tab["EAQ"], "EAQ", tab)
        rhos = dict(R=np.zeros(len(dfB)), U=np.full(len(dfB), 0.5), EA=rho_ea, EAQ=rho_eaq, EAQ_SHUF=rho_sh)
        C.dump_json(os.path.join(CAMP, "k20_policy_tables.json"), dict(pair=pair, teacher=T.key, student=S.key, tables=tab, info=dict(EA=iea, EAQ=ieaq, EAQ_SHUF=ish), shuffle_status={k: int(v) for k, v in zip(*np.unique(sstat, return_counts=True))},
                                                                     fraction_cell_changed_by_shuffle=float((dfS.cell.values != dfB.cell.values).mean()), source_student_advantage_frozen=True, gate_table_frozen=True, thresholds=thr))
        wm = pd.DataFrame(dict(sample_id=dfB.sample_id, cell=dfB.cell, cell_shuffled=dfS.cell, e_T=dfB.e_T, a_T=dfB.a_T, q_A=dfB.q_A, **{f"rho_{a}": rhos[a] for a in ARMS})); wm.to_csv(os.path.join(CAMP, "k20_weight_mass.csv"), index=False)
        print(json.dumps(dict(mean_rho={a: float(rhos[a].mean()) for a in ARMS}, hard_mass={a: float((1 - rhos[a]).mean()) for a in ARMS}, s_u=tab["s_u"], no_signal=tab["no_resolved_utility_signal"]), default=str))
        gt, ms, lpan, pan = cells["B"]["tensors"]; tpB = cells["B"]["teacher_pred"]; rng = np.random.RandomState(C.SPLIT_SEED + 1); order = []
        while len(order) < UPDATES:
            perm = rng.permutation(len(dfB))
            for s in range(0, len(perm) - BATCH + 1, BATCH):
                order.append(perm[s:s + BATCH])
        order = order[:UPDATES]; Did = man[man.split_role == "D"].sample_id.tolist(); Dt = C.load_patches(Did); src_state = copy.deepcopy(S.m.state_dict()); ends, logs = [], []
        n_upd, save_at = (UPDATES, SAVE_AT) if not profile else (40, (0, 40))
        for arm in (ARMS if not profile else ("EAQ",)):
            t0 = C.time.time(); e, l = train_arm(S, src_state, S.cfg, arm, (gt, ms, lpan, pan), tpB, rhos[arm], order, fd, Dt, f"pair{pair}", updates=n_upd, save_at=save_at); ends += e; logs += l
            print(f"  {arm}: D L1 {e[0]['D_l1']:.5f} → {e[-1]['D_l1']:.5f}; FR HQNR {e[0].get('fr_raw_original_hqnr'):.5f} → {e[-1].get('fr_raw_original_hqnr'):.5f} ({(C.time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(ends).drop(columns=["fr_per_scene_hqnr"], errors="ignore").to_csv(os.path.join(CAMP, "k20_endpoint_metrics.csv"), index=False); C.dump_json(os.path.join(CAMP, "k20_endpoint_scenes.json"), [dict(arm=e["arm"], update=e["update"], fr_per_scene_hqnr=e.get("fr_per_scene_hqnr")) for e in ends if "fr_per_scene_hqnr" in e])
        pd.DataFrame(logs).to_csv(os.path.join(CAMP, "k20_training_log.csv"), index=False)
        groups = man.set_index("sample_id").source_group_id.reindex(Did).values; per = {a: np.load(os.path.join(CAMP, "k20_states", f"pair{pair}", a, f"D_per_sample_{UPDATES}.npy")) for a in ARMS if os.path.exists(os.path.join(CAMP, "k20_states", f"pair{pair}", a, f"D_per_sample_{UPDATES}.npy"))}
        comp = {}
        for a, b in (("EAQ", "EA"), ("EAQ", "EAQ_SHUF"), ("EA", "U"), ("U", "R"), ("EAQ", "U"), ("EAQ", "R")):
            if a in per and b in per:
                comp[f"{a}_minus_{b}"] = C.block_bootstrap(per[a] - per[b], groups, seed=9)
        C.dump_json(os.path.join(CAMP, "k20_D_paired.json"), dict(pair=pair, endpoint="exact_5000 D L1 (lower better; negative diff = first arm better)", comparisons=comp)); print(json.dumps(comp, default=str))
