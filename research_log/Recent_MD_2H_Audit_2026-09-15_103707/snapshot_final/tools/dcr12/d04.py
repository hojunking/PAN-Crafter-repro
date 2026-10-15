"""D04 — 사분면이 실제 KD utility 를 구분하는지 (계획 §9; 조건부·선택 확장). B0 의 같은 checkpoint U 를 H-copy(L_H + λE L_E) / K-copy(L_H + L_K + λE L_E) 로 짧게 갱신해
query 의 plain GT L1 차이 U = R_H − R_K 를 cell 별로 잰다. micro_update_utility.csv · routing_gate.json (§9.2 조건; 통과 시에만 routing_policy.json — B2/B3 는 자동 실행하지 않는다).
Optimizer state 는 저장된 candidate 에 optimizer.bin 이 있어도 **fresh AdamW**(그 step 의 cosine LR) 로 쓴다(양쪽 동일; 기존 Adam trajectory 의 continuation 이 아니다 — 기록)."""
import copy, os
import numpy as np, pandas as pd, torch
from tools.dcr12 import common as C
SUPPORT_MB, MB, UPDATES, QUERY_N, EPISODES = 4, 16, 8, 16, 8


def episode_split(cell_ids, rT, rng):
    """support 64(4×16) 와 query 16 을 같은 cell 에서 분리 — R_T 순으로 정렬해 번갈아 뽑아 두 집합의 R_T 구간을 맞춘다 (§9.1)."""
    order = np.argsort(rT); ids = np.array(cell_ids)[order]; n = len(ids)
    if n < SUPPORT_MB * MB + QUERY_N:
        return None
    pick = rng.permutation(n)[:SUPPORT_MB * MB + QUERY_N]; pick.sort(); chosen = ids[pick]; q_pos = np.arange(2, len(chosen), 5)[:QUERY_N]          # 2,7,12,… 번째를 query — support 의 R_T 범위 안쪽 전 구간
    qmask = np.zeros(len(chosen), bool); qmask[q_pos] = True; sup = chosen[~qmask][:SUPPORT_MB * MB]; qry = chosen[qmask]; return [int(x) for x in sup], [int(x) for x in qry]


def micro_update(P, T, cal, sup_ids, qry_ids, lr, seed):
    gt, ms, lpan, pan = (t.to(C.DEV) for t in C.load_patches(sup_ids)); gq, mq, lq, pq = (t.to(C.DEV) for t in C.load_patches(qry_ids))
    with torch.no_grad():
        y_t = T.m(pan, ms, lpan)["y"]
    out = {}
    for arm in ("H", "K"):
        m = copy.deepcopy(P.m).train(); m.requires_grad_(False)
        for p in m.backbone.parameters():
            p.requires_grad_(True)                                                       # A 는 B0 그대로 frozen; T0 고정
        opt = torch.optim.AdamW(list(m.backbone.parameters()), lr=lr, weight_decay=C.WD); g = torch.Generator().manual_seed(seed); order = torch.randperm(SUPPORT_MB, generator=g).tolist() * 2
        with torch.no_grad():
            r_before = float((m(pq, mq, lq)["y"] - gq).abs().mean())
        for u in range(UPDATES):
            j = order[u]; sl = slice(j * MB, (j + 1) * MB); o = m(pan[sl], ms[sl], lpan[sl]); q = C.q12_terms(o["y"], y_t[sl], gt[sl], cal)
            loss = q["LH"] + cal["lambda_E"] * q["LE"] + (q["LK"] if arm == "K" else 0.0); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            r_after = float((m(pq, mq, lq)["y"] - gq).abs().mean())
        out[arm] = dict(before=r_before, after=r_after, groups=sorted({C.group_of(s) for s in sup_ids}))
    return out


def gate(df, thr, d01):
    """§9.2: (1) Teacher C 변동·cell support (2) R 수준이 같은 cell 쌍(A vs C, B vs D) 에서 U 차이 (3) soft 가 해로운 cell(U<0) 이 DISC/CONF·두 checkpoint 에서 같은 방향 (4) CONF group-bootstrap CI (5) 다중 cell 선택 기록."""
    t0 = thr["T0"]; cond = {}
    cond["1_teacher_C_varies_and_support"] = dict(ok=(not t0["degenerate_metric"]) and t0["C_cv"] > 0.05, C_cv=t0["C_cv"], degenerate=t0["degenerate_metric"])
    pairs = {}
    for pn in df.panel.unique():
        for a, b in (("A", "C"), ("B", "D")):
            ua, ub = df[(df.panel == pn) & (df.cell == a)].U.values, df[(df.panel == pn) & (df.cell == b)].U.values
            if len(ua) >= 3 and len(ub) >= 3:
                d = ua.mean() - ub.mean(); rng = np.random.RandomState(0); bs = [rng.choice(ua, len(ua)).mean() - rng.choice(ub, len(ub)).mean() for _ in range(1000)]
                pairs[f"{pn}:{a}-{b}"] = dict(diff=float(d), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=[int(len(ua)), int(len(ub))], excludes_zero=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0))
    cond["2_U_differs_by_C_at_same_R"] = dict(ok=any(v["excludes_zero"] for v in pairs.values()), pairs=pairs)
    harm = {}
    for c in "ABCD":
        g = df[df.cell == c]; per = {}
        for (pn, st), h in g.groupby(["panel", "step"]):
            per[f"{pn}@{st}"] = dict(n=int(len(h)), U_median=float(h.U.median()), neg_frac=float((h.U < 0).mean()))
        harm[c] = dict(per=per, harmful_everywhere=bool(per and all(v["U_median"] < 0 for v in per.values())), n_contexts=len(per))
    cand = [c for c, v in harm.items() if v["harmful_everywhere"] and v["n_contexts"] >= 2]
    cond["3_harmful_cell_reproduced"] = dict(ok=bool(cand), candidates=cand, detail=harm)
    ci = {}
    for c in cand:
        g = df[(df.cell == c) & (df.panel == "CONF")]
        if len(g) >= 3:
            b = C.block_bootstrap(g.U.values, g.group_key.values); ci[c] = b; ci[c]["consistent_negative"] = bool(b["ci95"] and b["ci95"][1] < 0)
    cond["4_conf_bootstrap_negative"] = dict(ok=any(v.get("consistent_negative") for v in ci.values()), by_cell=ci)
    zstar = [c for c in cand if ci.get(c, {}).get("consistent_negative")]
    cond["5_selection_record"] = dict(cells_considered=list("ABCD"), candidates=cand, z_star=(min(zstar, key=lambda c: harm[c]["per"][list(harm[c]["per"])[0]]["U_median"]) if zstar else None), note="Test HQNR 로 고르지 않았다 (DISC 후보 → CONF 확인)")
    passed = all(cond[k]["ok"] for k in ("1_teacher_C_varies_and_support", "2_U_differs_by_C_at_same_R", "3_harmful_cell_reproduced", "4_conf_bootstrap_negative"))
    return dict(passed=bool(passed), status=("ROUTING_CANDIDATE_OPEN" if passed else "NOT_OPENED_NO_INCREMENTAL_EVIDENCE"), conditions=cond, z_star=cond["5_selection_record"]["z_star"], attenuation=0.5, note="통과해도 B2/B3 는 자동 실행하지 않는다 (§9.4: 별도 연장 예산·사람 결정)")


def main(profile=False):
    with C.Stage("D04", "micro-update utility H vs K (conditional)"):
        cal = C.calibration(); T = C.load_pipe("T"); panels = C.make_panels(); sm = pd.read_csv(os.path.join(C.CAMP, "sample_metrics.csv")); thr = C.load_json(os.path.join(C.CAMP, "quadrant_thresholds.json"))
        t0 = sm[sm.pipeline == "T0"].set_index("sample_id"); up = os.path.join(C.CAMP, "micro_update_utility.csv"); have = set(pd.read_csv(up).pipeline.unique()) if os.path.exists(up) else set(); rows = []
        for seed in C.SEEDS:
            for st in C.D04_STEPS:
                tag = f"cand:{st}"
                if not (C.run_complete("B0", seed) and C.ckpt_path("B0", seed, tag)):
                    continue
                P = C.load_pipe("S", "B0", seed, tag)
                if P.key in have:
                    continue
                lr = C.cosine_lr(st, C.LR_U)
                for panel in ("DISC", "CONF"):
                    ids = panels["ids"][panel]; tt = t0.loc[[i for i in ids if i in t0.index]]
                    for cell in "ABCD":
                        cid = tt[tt.cell_ref == cell].index.tolist(); rT = tt.loc[cid].R_plain.values
                        n_ep = 1 if profile else EPISODES
                        for ep in range(n_ep):
                            rng = np.random.RandomState(C.SPLIT_SEED + 100 * ep + 7 * "ABCD".index(cell) + (0 if panel == "DISC" else 1000)); sp = episode_split(cid, rT, rng)
                            if sp is None:
                                rows.append(dict(pipeline=P.key, seed=seed, step=st, panel=panel, cell=cell, episode=ep, status="insufficient_cell", n_cell=len(cid))); break
                            sup, qry = sp; r = micro_update(P, T, cal, sup, qry, lr, seed=C.SPLIT_SEED + ep)
                            rows.append(dict(pipeline=P.key, seed=seed, step=st, panel=panel, cell=cell, episode=ep, status="ok", n_cell=len(cid), n_support=len(sup), n_query=len(qry), n_support_groups=len(r["H"]["groups"]), group_key=r["H"]["groups"][0], lr=lr,
                                             R_query_before=r["H"]["before"], R_H=r["H"]["after"], R_K=r["K"]["after"], U=r["H"]["after"] - r["K"]["after"], optimizer="fresh AdamW (not a continuation)", updates=UPDATES))
                print(f"[d04] {P.key} done", flush=True)
        if rows:
            C.append_rows(up, rows, ["pipeline", "panel", "cell", "episode"])
        df = pd.read_csv(up); ok = df[df.status == "ok"]
        summ = {f"{p}|{pn}|{c}": dict(n=int(len(g)), U_mean=float(g.U.mean()), U_median=float(g.U.median()), pos_frac=float((g.U > 0).mean())) for (p, pn, c), g in ok.groupby(["pipeline", "panel", "cell"])}
        g8 = gate(ok, thr, None) if len(ok) else dict(passed=False, status="NOT_RUN_NO_EPISODES", conditions={}); g8["summary"] = summ; g8["insufficient_cells"] = df[df.status != "ok"][["pipeline", "panel", "cell", "n_cell"]].drop_duplicates().to_dict("records")
        C.dump_json(os.path.join(C.CAMP, "routing_gate.json"), g8)
        if g8["passed"]:
            C.dump_json(os.path.join(C.CAMP, "routing_policy.json"), dict(z_star=g8["z_star"], m={c: (0.5 if c == g8["z_star"] else 1.0) for c in "ABCD"}, applies_to="soft term only (L_K^q); hard/edge unchanged", status="CANDIDATE — B2/B3 not executed"))
        print(f"[d04] gate {g8['status']} · cells {list(summ)[:8]}", flush=True)
