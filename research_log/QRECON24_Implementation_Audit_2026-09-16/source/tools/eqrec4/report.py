"""Report — 표 A–F · 그림 · H1–H4/Hspec 판정 (계획 §15–§16·§20; 감사 Q03/Q04/Q12 반영).
판정 규칙(사전 지정, protocol_changes.md 에 기록): 증거 단위 = **seed** (같은 seed 의 best/last 는 독립이 아니다). seed 판정 = best 와 last 가 같은 방향이고 각각 CI(source-block bootstrap; FR 은 scene p<0.05) 가 0 을 제외.
Supported = 세 seed 모두 예측 방향 · Opposed = 세 seed 모두 반대 방향(같은 엄격성) · 그 밖은 Insufficient/unresolved. 투표 비율·점추정치만으로 결론내지 않는다."""
import json, math, os
import numpy as np, pandas as pd
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, FIG, QUADS

SEEDS = (1234, 7777, 2025)


def _md(df, cols=None, fmt="{:.5f}"):
    if df is None or len(df) == 0:
        return "_(없음)_\n"
    df = df if cols is None else df[[c for c in cols if c in df]]; out = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join((fmt.format(v) if isinstance(v, (float, np.floating)) and not (isinstance(v, float) and math.isnan(v)) else str(v)) for v in r.values) + " |\n"
    return out


def _ci_txt(b):
    if not b or b.get("point") is None:
        return "NA"
    return f"{b['point']:+.4g}" + (f" [{b['ci95'][0]:+.4g}, {b['ci95'][1]:+.4g}] (n_grp {b['n_groups']})" if b.get("ci95") else " (CI 불가)")


def _dir(b, want, use_p=False):
    """+1 / −1 / 0: CI(또는 p) 로 확정된 방향. want 는 예측 방향(기록용)."""
    if not b or b.get("rho", b.get("point")) is None:
        return 0
    v = b.get("rho", b.get("point"))
    if use_p:
        return 0 if (b.get("p") is None or b["p"] >= 0.05) else int(np.sign(v))
    if not b.get("ci95"):
        return 0
    lo, hi = b["ci95"]; return 1 if lo > 0 else (-1 if hi < 0 else 0)


def seed_verdict(items, want):
    """items: {(seed, tag): dir} → seed 별 합의 방향 (best·last 둘 다 같은 확정 방향일 때만) → 전체 상태."""
    per = {}
    for s in SEEDS:
        ds = [d for (sd, _), d in items.items() if sd == s]
        per[s] = (ds[0] if ds and all(d == ds[0] for d in ds) and ds[0] != 0 else 0)
    n_for = sum(1 for d in per.values() if d == want); n_against = sum(1 for d in per.values() if d == -want)
    state = "Supported in tested setting" if n_for == 3 else "Opposed in tested setting" if n_against == 3 else "Insufficient / unresolved"
    return state, dict(per_seed={str(s): int(d) for s, d in per.items()}, n_seeds_for=n_for, n_seeds_against=n_against, rule="seed = unit; best&last must agree with CI excluding 0; all 3 seeds required")


def _key(mk):
    fam, s, tag = mk.split("_", 2); return fam, int(s[1:]), tag


def verdicts(J):
    v = {}; ev_rows = []
    h1 = J.get("h1", {}); items = {}
    for mk, r in h1.items():
        if mk.startswith("L1E4"):
            b = r.get("spearman_qA_e_BCD", {}); _, s, t = _key(mk); items[(s, t)] = _dir(b, +1); ev_rows.append(dict(hypothesis="H1_native64", model=mk, stat="spearman(q_A,e) BCD", value=b.get("rho"), ci=b.get("ci95"), n=b.get("n"), n_groups=b.get("n_groups"), direction=items[(s, t)]))
    st, det = seed_verdict(items, +1); v["H1_native64"] = dict(state=st, **det, claim="같은 checkpoint 안에서 q_A 가 낮은 sample 이 native L1 도 낮은가 (예측 ρ>0)", scope="native64 B∪C∪D, within-checkpoint, source-block(proxy) bootstrap")
    items = {}
    for mk, r in h1.items():
        if mk.startswith("L1E4"):
            b = r.get("FR_spearman_qA_hqnr", {}); _, s, t = _key(mk); items[(s, t)] = _dir(b, -1, use_p=True); ev_rows.append(dict(hypothesis="H1_FR", model=mk, stat="spearman(q_A,raw HQNR) FR20", value=b.get("rho"), p=b.get("p"), n=b.get("n"), direction=items[(s, t)]))
    st, det = seed_verdict(items, -1); v["H1_FR"] = dict(state=st, **det, claim="FR20 에서 q_A 낮음 ↔ raw HQNR 높음 (예측 ρ<0)", scope="20 scene, scene-level p<0.05 (다른 stratum; RR 과 합치지 않음)")
    h2 = J.get("h2", {}); items = {}
    for mk, r in h2.items():
        if mk.startswith("L1E4") and "native64" in r:
            bb = (r["native64"].get("by_bank", {}).get("B", {}).get("response", {})); dirs = []
            for rad in ("0.5", "1.0", "2.0"):
                b = bb.get(rad, {}).get("spearman_qA_d", {}); d = _dir(b, +1); dirs.append(d); ev_rows.append(dict(hypothesis="H2", model=mk, stat=f"spearman(q_A,d_e) bank B r={rad} response", value=b.get("rho"), ci=b.get("ci95"), n=b.get("n"), direction=d))
            _, s, t = _key(mk); items[(s, t)] = (1 if dirs.count(1) >= 2 and -1 not in dirs else -1 if dirs.count(-1) >= 2 and 1 not in dirs else 0)
    st, det = seed_verdict(items, +1); v["H2"] = dict(state=st, **det, claim="q_A 낮은 sample 의 독립 방향(bank B) 추가 이동 저하 d_e 가 작은가 (예측 ρ>0; 반경 3개 중 2개 이상 일치, 반대 없음)", scope="native64 D 상세 subset, response path, paired valid set; bank A r=1 은 재현 표")
    h3 = J.get("h3_core") or {}; src = "D30B core (native64 4,096, own quadrant)" if h3 else "D30 subset"; h3 = h3 or J.get("h3", {}); items = {}
    for mk, r in h3.items():
        if mk.startswith("L1E4"):
            b = (r.get("ALL") or r.get("native64") or {}).get("gain_learned_vs_zero"); _, s, t = _key(mk); items[(s, t)] = _dir(b, +1); ev_rows.append(dict(hypothesis="H3a", model=mk, stat="paired gain learned−zero (ROI L1)", value=(b or {}).get("point"), ci=(b or {}).get("ci95"), n=(b or {}).get("n"), direction=items[(s, t)]))
    st, det = seed_verdict(items, +1); v["H3a"] = dict(state=st, **det, claim="같은 가중치에서 learned correction 이 zero 보다 유리 (paired, 예측 gain>0)", scope=f"{src}; U-Net 은 learned 로 학습된 것 — 별도 P0 와의 causal 비교가 아니다")
    items = {}
    for mk, r in h3.items():
        if mk.startswith("L1E4"):
            b = (r.get("ALL") or r.get("native64") or {}).get("H3b_spearman_qA_gain"); _, s, t = _key(mk); items[(s, t)] = _dir(b, -1) if (b and b.get("ci95")) else _dir(b, -1, use_p=True); ev_rows.append(dict(hypothesis="H3b", model=mk, stat="spearman(q_A, gain)", value=(b or {}).get("rho"), ci=(b or {}).get("ci95"), p=(b or {}).get("p"), direction=items[(s, t)]))
    st, det = seed_verdict(items, -1); v["H3b"] = dict(state=st, **det, claim="q_A 가 learned−zero 이득의 크기를 예측 (예측 ρ<0)", scope="change-score coupling 주의 (e 와 gain 이 항을 공유)")
    k = J.get("k10_h4", {}); kA = k.get("A", {}) if isinstance(k.get("A"), dict) else {}; kB = k.get("cross_pair_A_to_B", {}); nz = J.get("k10_noise", {})
    imp = kA.get("q_added_mae_improvement"); dacc = (kA.get("B1", {}).get("direction_acc", 0) - kA.get("B0", {}).get("direction_acc", 0)) if kA else None
    cross_ok = bool(kB and kB.get("B1", {}).get("mae") is not None and kB["B1"]["mae"] <= kB["B0"]["mae"] and kB["B1"]["direction_acc"] >= 0.5)
    conds = dict(source_disjoint=bool(k.get("source_disjoint")), pairA_mae_improved=bool(imp is not None and imp > 0), pairA_direction_improved=bool(dacc is not None and dacc > 0), crossA_to_B_consistent=cross_ok, noise_scale_available=bool(nz.get("noise_scale_rel") is not None))
    v["H4_diag"] = dict(state=("Supported in tested setting" if all(conds.values()) else "Insufficient / unresolved"), conditions=conds, evidence=k, claim="q 를 더하면 K10 utility 예측이 좋아지는가 (source 분리 pool LOO + Pair A→B 교차, 재실행 noise 기록)")
    gate = J.get("k20_gate", {}); kp = J.get("k20_paired", {}).get("comparisons", {}); a = kp.get("EAQ_minus_EA"); b = kp.get("EAQ_minus_EAQ_SHUF")
    if not gate or not gate.get("passed"):
        v["H4_exec"] = dict(state=("H4 final_training_pilot_not_run" if not gate else "pilot_not_identifiable"), gate=gate, claim="K20 pilot (진입 조건 미충족이면 실행하지 않는다 — 계획 §13.1·§17.3)")
    else:
        da, db = _dir(a, -1), _dir(b, -1); v["H4_exec"] = dict(state=("Supported in tested setting" if da == -1 and db == -1 else "Opposed in tested setting" if (da == 1 or db == 1) else "Insufficient / unresolved"), gate=gate,
                                                                evidence=dict(EAQ_minus_EA=_ci_txt(a), EAQ_minus_SHUF=_ci_txt(b), others={k_: _ci_txt(v_) for k_, v_ in kp.items()}), claim="K20: EAQ < EA 그리고 EAQ < EAQ-SHUF (D locked L1, exact 5000; CI 로)", scope="Pair A, 5K 단기 적응; 최종 50K KD 성공이 아니다")
    v["Hspec"] = dict(state="descriptive (no automatic verdict)", claim="spectral 재조합만으로 q/추정 c 가 달라지는가 (D20-B/D40-D) — 조건부 해석만; 표 C 참조")
    return v, pd.DataFrame(ev_rows)


def quadrant_explanations(pk, nm, qa, st, ci, core, ps, gs, k10s):
    rows = []                                                              # (2026-09-15 03:00: 쓰지 않는 k10_cells CSV 를 load_json 으로 읽던 죽은 줄 제거 — dry-run 때는 파일이 없어 None 이라 통과했고 실전에서 JSONDecodeError)
    for q in QUADS:
        n64 = nm[(nm.model_key == pk) & (nm.scale == "native64")].merge(qa[qa.model_key == pk][["sample_id", "quadrant_id", "split_role"]].rename(columns={"split_role": "role"}), on="sample_id"); s = n64[n64.quadrant_id == q]
        r = dict(quadrant=q, n_atlas=int(len(s)), n_groups_atlas=int(s.source_group_id.nunique()), e_mean=float(s.e_native_full.mean()), qA_mean=float(s.q_A.mean()), c0_norm=float(s.c0_norm.mean()), texture=float(s.pan_scharr_energy.mean()))
        obs, unv = [], []
        if st is not None:
            ss = st[(st.model_key == pk) & (st.scale == "native64") & (st.probe_bank == "B") & (st.quadrant_primary == q) & st.valid_support]
            if len(ss):
                g = ss.groupby("path").d_e.mean(); r.update(n_stress=int(ss.sample_id.nunique()), stress_response=float(g.get("response", np.nan)), stress_no_response=float(g.get("no_response", np.nan)), stress_known_inverse=float(g.get("known_inverse", np.nan))); obs.append("stress(D subset, bank B)")
            else:
                unv.append("stress")
        if ps is not None:
            p = ps[(ps.model_key == pk) & (ps.quadrant_primary == q)]
            if len(p):
                g = p.groupby("variant").d_e_c0_fixed.mean(); r.update(pan_blur_dE=float(g.get("blur_sigma1", np.nan)), pan_other_dE=float(g.get("other_scene", np.nan)), pan_mean_dE=float(g.get("mean_constant", np.nan))); obs.append("PAN sensitivity (c0 fixed)")
            else:
                unv.append("PAN sensitivity")
        if gs and pk in gs:
            bq = gs[pk].get("by_quadrant", {}).get(q); os_ = gs[pk].get("one_step", {}).get("JC@1e-05", {}).get(q)
            if bq:
                r.update(n_grad=bq["n"], cos_gr_gc=bq["cos_mean"], R_g_median=bq["R_g_median"]); obs.append("gradient (D50)")
            if os_:
                r.update(JC_frac_e_down=os_["frac_e_down"], JC_frac_qA_down=os_["frac_qA_down"], JC_frac_qB_down=os_["frac_qB_down"]); obs.append("one-step JC")
        else:
            unv.append("gradient/one-step")
        if core is not None:
            c = core[(core.model_key == pk) & (core.quadrant_own == q)]
            if len(c):
                r.update(n_core=int(len(c)), gain_learned_zero_core=float(c.gain_learned_vs_zero_roi.mean()), gain_pos_frac_core=float((c.gain_learned_vs_zero_roi > 0).mean())); obs.append("learned−zero (core 4,096, own quadrant)")
        elif ci is not None:
            c = ci[(ci.model_key == pk) & (ci.scale == "native64") & (ci.quadrant_primary == q)]; L_, Z = c[c["mode"] == "I-L"].set_index("sample_id").l1_roi, c[c["mode"] == "I-Z"].set_index("sample_id").l1_roi
            if len(L_):
                r.update(n_core=int(len(L_)), gain_learned_zero_core=float((Z - L_).mean())); obs.append("learned−zero (D subset)")
        if k10s is not None:
            kk = k10s[(k10s.pair == "A") & k10s.cell.str.startswith(q)]
            if len(kk):
                r.update(n_k10_pools=int(kk.n.sum()), k10_U_soft_hard_rel=float(kk.U_soft_hard_rel_mean.mean()), k10_cells=";".join(kk.cell)); obs.append("K10 utility (B pools)")
            else:
                unv.append("K10 (no supported pool)")
        r["observed"] = "; ".join(obs); r["unverified"] = "; ".join(unv + ["edge profile per quadrant (RR scene-level only)", "parent-context (fixed 64² patches)"]); r["status"] = "mixed/unresolved (automatic report assigns no cause; see results_log interpretation)"; rows.append(r)
    return pd.DataFrame(rows)


def main(profile=False):
    with C.Stage("REPORT" + ("-profile" if profile else ""), "tables A–F, figures, verdicts"):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rd = lambda n: C.read_csv(os.path.join(CAMP, n)) if os.path.exists(os.path.join(CAMP, n)) else None
        am, nm, qa, occ, ci, core, st, rm, rmB, gc, k10s, k20e, ps, ld, bp = (rd(n) for n in ("assets_manifest.csv", "native_sample_metrics.csv", "quadrant_assignments.csv", "quadrant_occupancy.csv", "correction_interventions.csv", "correction_interventions_core.csv", "output_stress.csv", "response_matrices.csv",
                                                                                                "response_matrices_bankB.csv", "geometry_controls_summary.csv", "k10_cell_summary.csv", "k20_endpoint_metrics.csv", "pan_sensitivity.csv", "correction_landscape_summary.csv", "band_edge_profiles.csv"))
        J = dict(h1=C.load_json(os.path.join(CAMP, "h1_stats.json"), {}), h2=C.load_json(os.path.join(CAMP, "h2_stats.json"), {}), h3=C.load_json(os.path.join(CAMP, "h3_stats.json"), {}), h3_core=C.load_json(os.path.join(CAMP, "h3_core_stats.json"), None),
                 k10_h4=C.load_json(os.path.join(CAMP, "k10_h4_diag.json"), {}), k10_noise=C.load_json(os.path.join(CAMP, "k10_noise_A.json"), {}), k20_paired=C.load_json(os.path.join(CAMP, "k20_D_paired.json"), {}), k20_gate=C.load_json(os.path.join(CAMP, "k20_gate.json"), {}),
                 g00=C.load_json(os.path.join(CAMP, "g00_reproduction.json"), {}), gs=C.load_json(os.path.join(CAMP, "gradient_summary.json"), {}), blurA=C.load_json(os.path.join(CAMP, "blur_control_status_A.json"), {}))
        pk = C.mkey(*C.PRIMARY); V, ev = verdicts(J); C.dump_json(os.path.join(CAMP, "hypothesis_verdicts.json"), V); ev.to_csv(os.path.join(CAMP, "verdict_evidence.csv"), index=False)
        with open(os.path.join(CAMP, "protocol_changes.md"), "a") as f:
            f.write(f"- REPORT ({C.time.strftime('%Y-%m-%d %H:%M')}): 판정 규칙 = seed 단위(best/last 합의, CI 가 0 제외), 세 seed 모두 일치해야 Supported/Opposed; H2 는 bank B 만; H4_exec 는 K20 gate 통과 시에만. 투표 비율·점추정치 규칙 없음.\n")
        md = [f"# {C.CAMPAIGN_ID} report ({C.time.strftime('%Y-%m-%d %H:%M')})\n\n계획 `{C.PLAN}` · 출력 root `{os.path.relpath(CAMP, C.ROOT)}/` · primary Teacher `{pk}` · 서버 {C.SERVER}\n\n"
              "관측 단위 주의: atlas n = 4,096 patch(전수) · stress/개입 상세 = D 의 집단별 ≤64 patch(primary 4분면 기준 추출) · core 개입 = 4,096(자기 checkpoint 4분면) · K10 = B 의 fit pool(48) 단위 · RR/FR = 20 scene. source group 은 index-block proxy 다.\n"]
        md.append("\n## Table A — 자산·재현\n" + _md(am, ["model_key", "actual_update", "model_hash", "aligner_hash", "unet_hash", "exists"], fmt="{}") + f"\nPrimary best_raw raw-original HQNR: 기록 {J['g00'].get('recorded_best_raw_hqnr')} / 재현 {J['g00'].get('reproduced_raw_original_hqnr')} (Δ {J['g00'].get('abs_diff')}); RR L1(정규화) {J['g00'].get('rr_l1_normalized_mean')}\n")
        qe = quadrant_explanations(pk, nm, qa, st, ci, core, ps, J["gs"], k10s) if nm is not None and qa is not None else None
        md.append("\n## Table B — 네 집단 (primary 4분면; 열마다 지원 n 이 다르다)\n" + _md(qe, [c for c in (qe.columns if qe is not None else []) if c not in ("observed", "unverified", "status")]) + "\n### 집단별 관측 / 미검증\n" + "".join(f"- **{r.quadrant}**: 관측 = {r.observed} · 미검증 = {r.unverified} · 상태 = {r.status}\n" for _, r in (qe.iterrows() if qe is not None else [])) +
                  "\n집단 정의상 E↓ 의 e 가 낮은 것은 결과가 아니다. 원인 해석은 자동 생성하지 않는다(감사 Q12) — results_log 문서에서 개입·반증·coverage 를 근거로 쓴다.\n")
        md.append("\n## Table C — 정합의 세 수준\n### relative response (PAN-only / MS-only / common; bank A · bank B)\n" + _md(pd.concat([x for x in (rm, rmB) if x is not None]) if rm is not None else None, ["model_key", "scale", "mode", "probe_bank", "n", "B_yy_mean", "B_xx_mean", "B_yx_mean", "B_xy_mean", "epe_mean", "fit_rmse_mean"], fmt="{:.4f}"))
        md.append("### known absolute synthetic error (동일 격자 합성; calibration/OOD 진단)\n" + _md(gc, fmt="{:.4f}"))
        pf = rd("native_geometry_proxy_full.csv")
        if pf is not None:
            g = pf.groupby(["model_key", "scale"]).agg(n=("scene", "count"), n_both_accepted=("both_accepted", "sum"), n_high_conf=("high_confidence", "sum"), before_mag_median=("before_mag", "median"), after_mag_median=("after_mag", "median"), n_improved=("improved", "sum"), sign_ok=("known_shift_sign_ok", "all")).reset_index()
            md.append("### native before/after proxy (estimator primary+secondary; 센서 GT 아님)\n" + _md(g, fmt="{:.4f}"))
        if ci is not None:
            td = ci[ci.status.fillna("ok") == "ok"].copy() if "status" in ci else ci.copy(); td["metric"] = np.where(td.scale == "fr512", td.get("raw_original_hqnr"), td.get("l1_roi")); td["mode_group"] = td["mode"].str.replace(r"I-S\d+", "I-S(mean10)", regex=True)
            md.append("\n## Table D — correction 개입 (mode 평균; native64(D subset)/RR = ROI L1 ↓, FR = raw-original HQNR ↑)\n" + _md(td.groupby(["model_key", "scale", "mode_group"]).metric.mean().reset_index().pivot(index=["model_key", "scale"], columns="mode_group", values="metric").reset_index(), fmt="{:.5f}"))
        if core is not None:
            md.append("### core (native64 4,096, own quadrant) learned−zero gain\n" + _md(core.groupby(["model_key", "quadrant_own"]).agg(n=("sample_id", "count"), gain_mean=("gain_learned_vs_zero_roi", "mean"), pos_frac=("gain_learned_vs_zero_roi", lambda x: float((x > 0).mean()))).reset_index()))
        md.append("\n## Table E — cue 유용성\n### K10 cell summary (Pair 별; U_soft-hard_rel > 0 = 추가 soft 가 추가 hard 보다 C 에서 유리)\n" + _md(k10s, fmt="{:.4f}") + f"\nK10 noise scale(rel, 같은 pool 3회): {J['k10_noise'].get('noise_scale_rel')}\n### K20 gate\n" + json.dumps(J["k20_gate"], ensure_ascii=False, default=str)[:1500] + "\n### K20 endpoints\n" + _md(k20e, fmt="{:.5f}") + "\n### K20 D paired (exact 5000)\n" + "\n".join(f"- {k}: {_ci_txt(v)}" for k, v in J["k20_paired"].get("comparisons", {}).items()) + "\n")
        md.append("\n## Table F — 판정 (증거 표 verdict_evidence.csv)\n| 가설 | 상태 | seed 별 방향 | 주장 | 범위 |\n|---|---|---|---|---|\n" + "".join(f"| {k} | **{v['state']}** | {v.get('per_seed', '')} | {v.get('claim', '')} | {v.get('scope', '')} |\n" for k, v in V.items()))
        md.append(f"\n## 한 문장 결론 (§20 형식; 자동 판정 상태를 그대로 넣는다)\n> 현재 L1E4 의 offset consistency 는 추가 이동 반응(H2: {V['H2']['state']}) · native 위치 개선(proxy 표 C, 판정 없음) · native 복원(H1 native64: {V['H1_native64']['state']}; H1 FR: {V['H1_FR']['state']}) · Student 감독 유용성(H4 diag: {V['H4_diag']['state']}; H4 exec: {V['H4_exec']['state']}) 로 검증됐고, "
                  f"reconstruction 과 엇갈리는 sample 의 원인은 Table B 의 관측/미검증 목록대로 남긴다(자동 원인 배정 없음). e_T·a_T 에 q_T 를 추가한 soft/hard 선택은 [{V['H4_exec']['state']}] 이었고, 그 판단은 동일 checkpoint·대응 개입·source(proxy) 분할·실제 KD endpoint(D locked, exact 5000) 에 근거한다.\n"
                  "\nL1E4 의 채택과 q_T 의 Teacher-quality gate 채택은 별개의 결정으로 유지한다. FR20 은 개발에 반복 사용된 benchmark 다. source group 은 index-block proxy 라 독립성 가정이 약하다(CI 는 보수적으로 읽는다).\n")
        open(os.path.join(CAMP, "report_EQREC4.md"), "w").write("".join(md))
        try:
            if rm is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); r = rm[(rm.model_key == pk) & (rm.scale == "native64")]; x = np.arange(len(r)); ax.bar(x - 0.2, r.B_yy_mean, 0.4, label="B_yy"); ax.bar(x + 0.2, r.B_xx_mean, 0.4, label="B_xx"); ax.set_xticks(x); ax.set_xticklabels(r["mode"]); ax.axhline(-1, c="k", ls=":"); ax.axhline(1, c="k", ls=":"); ax.legend(); ax.set_title(f"Fig2 signed response diag ({pk}, bank A)"); fig.savefig(os.path.join(FIG, "fig2_modality_response.png"), dpi=110); plt.close(fig)
            if st is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); s = st[(st.model_key == pk) & (st.scale == "native64") & (st.probe_bank == "B") & st.valid_support]; g = s.groupby(["quadrant_primary", "path"]).d_e.mean().unstack(); g.plot.bar(ax=ax); ax.set_ylabel("d_e (ROI L1 stress − native)"); ax.set_title("Fig3 native→stress by quadrant (bank B r∈{.5,1,2}, primary labels)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_stress_by_quadrant.png"), dpi=110); plt.close(fig)
            if ci is not None:
                fig, axs = plt.subplots(1, 2, figsize=(11, 4))
                for ax, sc, key in zip(axs, ("rr256", "fr512"), ("l1_roi", "raw_original_hqnr")):
                    c = ci[(ci.model_key == pk) & (ci.scale == sc)]
                    if len(c) and key in c:
                        piv = c[c["mode"].isin(["I-L", "I-Z", "I-W", "I-C"])].pivot(index="sample_id", columns="mode", values=key)
                        for mode in ("I-Z", "I-W", "I-C"):
                            if mode in piv:
                                ax.scatter(piv["I-L"], piv[mode], s=14, label=mode)
                        lim = [piv.min().min(), piv.max().max()]; ax.plot(lim, lim, "k:"); ax.set_xlabel(f"learned {key}"); ax.set_ylabel("intervened"); ax.legend(); ax.set_title(f"Fig4 paired ({sc})")
                fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_paired_interventions.png"), dpi=110); plt.close(fig)
            ldf = rd("correction_landscape.csv")
            if ldf is not None:
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                for ax, q in zip(axs, QUADS):
                    s = ldf[(ldf.model_key == pk) & (ldf.quadrant_primary == q) & (ldf.kind == "grid")]
                    if len(s):
                        sid = s.sample_id.iloc[0]; g = s[s.sample_id == sid].pivot(index="c_dy", columns="c_dx", values="l1_roi"); im = ax.imshow(g.values, origin="lower", extent=[-2.25, 2.25, -2.25, 2.25]); ax.set_title(f"Fig6 L1 landscape {q} #{sid}"); c0 = ldf[(ldf.sample_id == sid) & (ldf.kind == "c0") & (ldf.model_key == pk)]
                        if len(c0):
                            ax.plot(c0.c_dx, c0.c_dy, "r*", ms=10)
                        plt.colorbar(im, ax=ax)
                fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_landscape_examples.png"), dpi=110); plt.close(fig)
            if k10s is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); k = k10s[k10s.pair == "A"]; ax.bar(k.cell, k.U_soft_hard_rel_mean); ax.axhline(0, c="k"); ax.set_ylabel("U_soft-hard (rel, C)"); ax.set_title("Fig7 K10 utility by cell (pair A)"); plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_k10_utility.png"), dpi=110); plt.close(fig)
            if bp is not None:
                fig, ax = plt.subplots(figsize=(11, 4)); g = bp.groupby(["model_key", "correction"] if "correction" in bp else ["model_key"])[["crossing_offset_mean_abs", "width_ratio", "edge_mae_dn"]].mean(); g.plot.bar(ax=ax); ax.set_title("Fig5 edge profile summary (RR, GT edge mask; learned vs zero)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_edge_profiles.png"), dpi=110); plt.close(fig)
        except Exception as ex:
            open(os.path.join(CAMP, "report_figures_error.txt"), "a").write(repr(ex) + "\n")
        print(json.dumps({k: v["state"] for k, v in V.items()}, ensure_ascii=False))
