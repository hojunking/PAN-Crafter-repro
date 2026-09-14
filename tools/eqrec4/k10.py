"""K10 — 네 집단 × Teacher 우위 부호에서 hard/soft 추가 감독의 국소 유용성 (계획 §12·§14). 손실별 수신자 분리: base(Rec+Off) → A+U, extra(hard/soft) → U only."""
import copy, json, math, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, DEV, QUADS

GAMMA = 0.1; UPDATES = 64; LR_U = 1e-5; LR_A = 1e-6; POOL = 48; TRIALS = 4; NVAL = 128; ARMS = ("R0", "RH", "RS")


def make_student(L):
    m = copy.deepcopy(L.m).train(); m.requires_grad_(True); return m


def fresh_opt(m, cfg):
    wd = float(cfg.get("weight_decay", 0.01)); return torch.optim.AdamW([dict(params=list(m.backbone.parameters()), lr=LR_U), dict(params=list(m.aligner.parameters()), lr=LR_A)], lr=LR_U, weight_decay=wd)


def k_step(m, mg, batch, upd, gen, teacher_pred, arm, gamma=GAMMA, rho=None, opt=None, check=False):
    """한 update. batch = (gt, ms, lpan, pan) on DEV. arm: R0 | RH | RS | K20 (rho [B] 사용). 반환 dict(norms, losses). check=True 면 수신자 검사 값을 함께."""
    gt, ms, lpan, pan = batch; A = list(m.aligner.parameters()); Fp = list(m.backbone.parameters())
    o = m(pan, ms, lpan); y = o["y"]; c0 = o["delta"]; h = (y - gt).abs().flatten(1).mean(1); base = h.mean(); off = None
    if upd % 2 == 1:
        eps = C.sample_offsets(pan.shape[0], 2.0, gen).to(pan.device)
        with torch.no_grad():
            pe = C.warp_pan(pan, eps)
        ce = C.predict_c(m.aligner, pe, o["ms_base"], mg); off = C.offset_loss(ce, c0, eps, stop_reference=True); base = base + 1e-4 * off
    extra = None; d = None
    if arm in ("RS", "K20"):
        d = (y - teacher_pred).abs().flatten(1).mean(1)
    if arm == "RH":
        extra = gamma * h.mean()
    elif arm == "RS":
        extra = gamma * d.mean()
    elif arm == "K20" and rho is not None and gamma > 0:
        extra = gamma * ((1.0 - rho) * h + rho * d).mean()
    gb = torch.autograd.grad(base, A + Fp, retain_graph=(extra is not None) or check, allow_unused=True)
    if any(g is None for g in gb):
        raise RuntimeError("base gradient missing on some A/U parameter (contract: Rec→A/U, Off→A)")
    ge = torch.autograd.grad(extra, Fp, retain_graph=check, allow_unused=True) if extra is not None else None
    if ge is not None and any(g is None for g in ge):
        raise RuntimeError("extra gradient missing on some U parameter")
    out = dict(loss_base=float(base), loss_h=float(h.mean()), loss_off=(float(off) if off is not None else None), loss_extra=(float(extra) if extra is not None else None), loss_d=(float(d.mean()) if d is not None else None),
               g_base_A=float(torch.sqrt(sum((g ** 2).sum() for g in gb[:len(A)]))), g_base_U=float(torch.sqrt(sum((g ** 2).sum() for g in gb[len(A):]))), g_extra_U=(float(torch.sqrt(sum((g ** 2).sum() for g in ge))) if ge is not None else 0.0))
    if check:
        out["off_to_U_absent"] = bool(off is None or all(g is None or float(g.abs().sum()) == 0 for g in torch.autograd.grad(off, Fp, retain_graph=True, allow_unused=True)))
        out["extra_to_A_would_exist"] = bool(extra is not None and any(g is not None and float(g.abs().sum()) > 0 for g in torch.autograd.grad(extra, A, retain_graph=True, allow_unused=True)))
        out["rec_to_A_exists"] = bool(any(float(g.abs().sum()) > 0 for g in torch.autograd.grad(h.mean(), A, retain_graph=True, allow_unused=True) if g is not None))
    with torch.no_grad():
        for p, g in zip(A, gb[:len(A)]):
            p.grad = g.detach().clone()
        for k, (p, g) in enumerate(zip(Fp, gb[len(A):])):
            p.grad = (g + ge[k]).detach().clone() if ge is not None else g.detach().clone()
    if opt is not None:
        opt.step(); opt.zero_grad(set_to_none=True)
    return out


def receiver_check(pair="A"):
    """§6-7 / §14.2: 실제 wrapper 에서 Rec→A/U, Off→A only, extra→U only; extra 를 켜도 A 의 gradient 는 base 만 쓴 경우와 같다; Teacher 불변."""
    T = C.load_model(*C.PAIRS[pair]["T"]); S = C.load_model(*C.PAIRS[pair]["S"]); man = C.make_manifest(); ids = man[man.split_role == "B"].sample_id.iloc[:4].tolist(); gt, ms, lpan, pan = (t.to(DEV) for t in C.load_patches(ids))
    with torch.no_grad():
        tp = T.m(pan, ms, lpan)["y"]
    th0 = C.state_hash(T.m); m = make_student(S); res = {}; gA = {}
    for arm in ("R0", "RH", "RS"):
        gen = torch.Generator().manual_seed(99); r = k_step(m, S.mg, (gt, ms, lpan, pan), 1, gen, tp, arm, check=True); gA[arm] = torch.cat([p.grad.flatten() for p in m.aligner.parameters()]).detach().clone()
        res[arm] = r; [setattr(p, "grad", None) for p in m.parameters()]
    ref = gA["R0"].abs().max(); res["A_grad_max_rel_diff"] = {a: float((gA[a] - gA["R0"]).abs().max() / (ref + 1e-12)) for a in ("RH", "RS")}          # 같은 forward 의 base grad — cuDNN 비결정성만 허용 (1e-4 상대)
    ok = (res["R0"]["off_to_U_absent"] and res["R0"]["rec_to_A_exists"] and res["RH"]["extra_to_A_would_exist"] and res["RS"]["extra_to_A_would_exist"] and all(v < 1e-4 for v in res["A_grad_max_rel_diff"].values())
          and res["RH"]["g_extra_U"] > 0 and res["RS"]["g_extra_U"] > 0 and C.state_hash(T.m) == th0)
    return dict(ok=bool(ok), pair=pair, student=S.key, teacher=T.key, arms=res, note="A gradient identical across R0/RH/RS on the same forward = extra is routed to U only; extra would reach A through the aligned PAN if not routed")


@torch.no_grad()
def eval_l1(m, gt, ms, lpan, pan, chunk=64):
    ys = []
    for s in range(0, pan.shape[0], chunk):
        ys.append(m(pan[s:s + chunk].to(DEV), ms[s:s + chunk].to(DEV), lpan[s:s + chunk].to(DEV))["y"].float().cpu())
    y = torch.cat(ys); return dict(l1=float((y - gt).abs().mean()), l1_roi=float(C.l1_roi(y, gt, "native64").mean()), edge=float(C.edge_l1_roi(y, gt, "native64").mean()), per_sample=(y - gt).abs().mean(dim=(1, 2, 3)).numpy())


def cells_for_pair(pair, T, S, man):
    """B/C 의 Teacher e/q(bank A)·Student e → (e_low, c_low, a_pos) cell. Threshold = D10 calibration_thresholds 의 Teacher model 값."""
    thr = C.load_json(os.path.join(CAMP, "calibration_thresholds.json"))[T.key]; out = {}
    for role in ("B", "C"):
        ids = man[man.split_role == role].sample_id.tolist(); gt, ms, lpan, pan = C.load_patches(ids)
        yT, _ = C.native_forward(T, pan, ms, lpan); yS, _ = C.native_forward(S, pan, ms, lpan); eT = (yT - gt).abs().mean(dim=(1, 2, 3)); eS = (yS - gt).abs().mean(dim=(1, 2, 3)); pr = C.probe_responses(T, pan, ms, banks=("A",))
        df = pd.DataFrame(dict(sample_id=ids, source_group_id=man.set_index("sample_id").source_group_id.reindex(ids).values, e_T=eT.numpy(), e_S=eS.numpy(), a_T=(eS - eT).numpy(), q_A=pr["A"]["q"].numpy()))
        df["e_low"] = df.e_T <= thr["e_median"]; df["c_low"] = df.q_A <= thr["q_A_median"]; df["a_pos"] = df.a_T > 0; df["cell"] = df.apply(lambda r: ("Ed" if r.e_low else "Eu") + ("Cd" if r.c_low else "Cu") + ("_Apos" if r.a_pos else "_Aneg"), axis=1)
        out[role] = dict(df=df, teacher_pred=yT, tensors=(gt, ms, lpan, pan), ids=ids)
    return out, thr


def fit_pools(dfB, seed):
    rng = np.random.RandomState(seed); pools = []
    for cell, g in dfB.groupby("cell"):
        idx = g.index.values.copy(); rng.shuffle(idx); k = 0
        while k < TRIALS:
            chunk = idx[k * POOL:(k + 1) * POOL]
            if len(chunk) >= POOL:
                pools.append(dict(cell=cell, trial=k, rows=chunk, n_unique=int(len(chunk)), cycled=False))
            elif len(chunk) >= 32:
                pools.append(dict(cell=cell, trial=k, rows=chunk, n_unique=int(len(chunk)), cycled=True)); break
            else:
                break
            k += 1
    return pools


def run_trial(S, m_src_state, cfg, pool, Bt, tp_B, valC, tp_dummy, seed, arm):
    gt, ms, lpan, pan = Bt; rows = pool["rows"]; sel = np.array(rows)
    if pool["cycled"]:
        sel = np.resize(sel, POOL)
    b = tuple(t[sel].to(DEV) for t in (gt, ms, lpan, pan)); tp = tp_B[sel].to(DEV); m = copy.deepcopy(S.m).train(); m.load_state_dict(m_src_state); m.requires_grad_(True); h0 = C.state_hash(m)
    opt = fresh_opt(m, cfg); gen = torch.Generator().manual_seed(seed); logs = []
    fit0 = eval_l1(m, *(t[sel] for t in (gt, ms, lpan, pan))); c0 = eval_l1(m, *valC)
    for u in range(UPDATES):
        r = k_step(m, S.mg, b, u, gen, tp, arm, opt=opt)
        if u in (0, UPDATES - 1):
            logs.append(dict(update=u, **r))
    m.eval(); fit1 = eval_l1(m, *(t[sel] for t in (gt, ms, lpan, pan))); c1 = eval_l1(m, *valC)
    return dict(restore_hash=h0, fit_l1_before=fit0["l1"], fit_l1_after=fit1["l1"], C_l1_before=c0["l1"], C_l1_after=c1["l1"], C_edge_before=c0["edge"], C_edge_after=c1["edge"], C_l1_roi_after=c1["l1_roi"],
                g_base_A_first=logs[0]["g_base_A"], g_base_U_first=logs[0]["g_base_U"], g_extra_U_first=logs[0]["g_extra_U"], g_base_U_last=logs[-1]["g_base_U"], g_extra_U_last=logs[-1]["g_extra_U"], loss_h_first=logs[0]["loss_h"], loss_h_last=logs[-1]["loss_h"], loss_d_first=logs[0]["loss_d"],
                C_per_sample_after=c1["per_sample"])


def h4_diag(ut):
    """§15.3: pool 단위 utility 를 B0(e_T·a_T·source) vs B1(+q_A, 상호작용) ridge 로 예측 — pool 단위 leave-one-out (pool 은 서로 겹치지 않는다)."""
    def ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = (Xtr - mu) / sd; ym = ytr.mean(); w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym)); return ((Xte - mu) / sd) @ w + ym
    out = {}; fitted = {}
    for pair, g in ut.groupby("pair"):
        g = g[g.repeat == 0]
        if len(g) < 6:
            out[pair] = dict(status="insufficient_pools", n=int(len(g))); continue
        y = g.U_soft_hard_rel.values; X0 = np.c_[g.pool_eT_mean, g.pool_eT_std, g.pool_aT_mean, g.pool_apos_frac]; X1 = np.c_[X0, g.pool_qA_mean, g.pool_qA_std, g.pool_eT_mean * g.pool_qA_mean, g.pool_aT_mean * g.pool_qA_mean]
        res = {}
        for name, X in (("B0", X0), ("B1", X1)):
            pred = np.zeros_like(y)
            for i in range(len(y)):
                tr = np.arange(len(y)) != i; pred[i] = ridge_fit_predict(X[tr], y[tr], X[i][None])[0]
            res[name] = dict(loo_mae=float(np.abs(pred - y).mean()), direction_acc=float(((pred > 0) == (y > 0)).mean()), n_pools=int(len(y)))
        res["q_added_mae_improvement"] = res["B0"]["loo_mae"] - res["B1"]["loo_mae"]; out[pair] = res; fitted[pair] = (X0, X1, y)
    if "A" in fitted and "B" in fitted:                                   # §15.3: Pair A 로 맞춘 관계를 Pair B 에 적용 (B 의 utility 로 다시 맞추지 않는다)
        (XA0, XA1, yA), (XB0, XB1, yB) = fitted["A"], fitted["B"]; out["cross_pair_A_to_B"] = {n: dict(mae=float(np.abs(ridge_fit_predict(Xa, yA, Xb) - yB).mean()), direction_acc=float(((ridge_fit_predict(Xa, yA, Xb) > 0) == (yB > 0)).mean())) for n, Xa, Xb in (("B0", XA0, XB0), ("B1", XA1, XB1))}
    out["note"] = "pool-level utilities; ridge(alpha=1) leave-one-pool-out; pools are disjoint by construction; EA/EAQ medians do not perfectly control e/a (§15.3)"
    return out


def main(profile=False, pairs=("A", "B")):
    man = C.make_manifest()
    with C.Stage("K10" + ("-profile" if profile else ""), "paired hard/soft microtrials"):
        trials, manifest = [], []
        for pair in pairs:
            T = C.load_model(*C.PAIRS[pair]["T"]); S = C.load_model(*C.PAIRS[pair]["S"]); cells, thr = cells_for_pair(pair, T, S, man); dfB, dfC = cells["B"]["df"], cells["C"]["df"]
            dfB.assign(pair=pair, role="B").to_csv(os.path.join(CAMP, f"k10_cells_{pair}_B.csv"), index=False); dfC.assign(pair=pair, role="C").to_csv(os.path.join(CAMP, f"k10_cells_{pair}_C.csv"), index=False)
            pools = fit_pools(dfB, C.SPLIT_SEED + (0 if pair == "A" else 1)); valC_df = C.stratified_pick(dfC.assign(texture="na"), NVAL, seed=7, strata_cols=("source_group_id",)); vsel = valC_df.index.values
            gtC, msC, lpC, pnC = cells["C"]["tensors"]; valC = tuple(t[vsel] for t in (gtC, msC, lpC, pnC)); Bt = cells["B"]["tensors"]; tpB = cells["B"]["teacher_pred"]; src_state = copy.deepcopy(S.m.state_dict())
            if profile:
                pools = pools[:1]
            reps = {}
            for pi, pool in enumerate(pools):
                nrep = 3 if (pi == 0 and not profile) else 1
                for rep in range(nrep):
                    res = {}
                    for arm in ARMS:
                        res[arm] = run_trial(S, src_state, S.cfg, pool, Bt, tpB, valC, None, seed=1000 + pool["trial"] + 17 * pi, arm=arm)
                    g = dfB.loc[pool["rows"]]; row = dict(pair=pair, teacher=T.key, student=S.key, student_update=S.step, cell=pool["cell"], trial=pool["trial"], repeat=rep, n_unique_fit=pool["n_unique"], cycled=pool["cycled"], n_source_groups=int(g.source_group_id.nunique()),
                                                          pool_eT_mean=float(g.e_T.mean()), pool_eT_std=float(g.e_T.std()), pool_aT_mean=float(g.a_T.mean()), pool_apos_frac=float(g.a_pos.mean()), pool_qA_mean=float(g.q_A.mean()), pool_qA_std=float(g.q_A.std()),
                                                          restore_ok=bool(len({res[a]["restore_hash"] for a in ARMS}) == 1), C_l1_source=res["R0"]["C_l1_before"])
                    for a in ARMS:
                        row.update({f"{a}_fit_l1_before": res[a]["fit_l1_before"], f"{a}_fit_l1_after": res[a]["fit_l1_after"], f"{a}_C_l1_after": res[a]["C_l1_after"], f"{a}_C_edge_after": res[a]["C_edge_after"], f"{a}_g_base_A": res[a]["g_base_A_first"], f"{a}_g_base_U": res[a]["g_base_U_first"], f"{a}_g_extra_U": res[a]["g_extra_U_first"], f"{a}_loss_d_first": res[a]["loss_d_first"]})
                    L0 = res["R0"]["C_l1_before"]; row.update(U_soft_hard=res["RH"]["C_l1_after"] - res["RS"]["C_l1_after"], U_soft_base=res["R0"]["C_l1_after"] - res["RS"]["C_l1_after"], U_hard_base=res["R0"]["C_l1_after"] - res["RH"]["C_l1_after"])
                    row.update(U_soft_hard_rel=row["U_soft_hard"] / L0, U_soft_base_rel=row["U_soft_base"] / L0, U_hard_base_rel=row["U_hard_base"] / L0, fit_gain_RH=res["RH"]["fit_l1_before"] - res["RH"]["fit_l1_after"], fit_gain_RS=res["RS"]["fit_l1_before"] - res["RS"]["fit_l1_after"])
                    trials.append(row); reps.setdefault(pi, []).append(row["U_soft_hard_rel"])
                    print(f"  {pair} {pool['cell']} t{pool['trial']} r{rep}: U_soft-hard {row['U_soft_hard']:+.2e} (rel {row['U_soft_hard_rel']:+.3f})", flush=True)
                manifest.append(dict(pair=pair, cell=pool["cell"], trial=pool["trial"], sample_ids=[int(dfB.sample_id.iloc[k]) for k in pool["rows"]], n_unique=pool["n_unique"], cycled=pool["cycled"]))
            noise = float(np.std(reps.get(0, [0.0]))) if len(reps.get(0, [])) > 1 else None
            C.dump_json(os.path.join(CAMP, f"k10_noise_{pair}.json"), dict(pair=pair, repeat_utilities_rel=reps.get(0), noise_scale_rel=noise, note="same source/pool/RNG re-run (GPU nondeterminism); scale for K20 s_u"))
            del T, S; torch.cuda.empty_cache()
        ut = pd.DataFrame(trials); ut.to_csv(os.path.join(CAMP, "k10_utility.csv"), index=False); C.dump_json(os.path.join(CAMP, "k10_trial_manifest.json"), manifest)
        try:
            C.dump_json(os.path.join(CAMP, "k10_h4_diag.json"), h4_diag(ut))
        except Exception as ex:
            C.dump_json(os.path.join(CAMP, "k10_h4_diag.json"), dict(status=f"failed: {ex!r}"))
        summ = ut[ut.repeat == 0].groupby(["pair", "cell"]).agg(n=("trial", "count"), U_soft_hard_rel_mean=("U_soft_hard_rel", "mean"), U_soft_hard_rel_median=("U_soft_hard_rel", "median"), n_positive=("U_soft_hard_rel", lambda x: int((x > 0).sum())), U_soft_base_rel_mean=("U_soft_base_rel", "mean"), U_hard_base_rel_mean=("U_hard_base_rel", "mean")).reset_index()
        summ.to_csv(os.path.join(CAMP, "k10_cell_summary.csv"), index=False); print(summ.to_string())
