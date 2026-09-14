"""Report — 표 A–F · 그림 2–7 · H1–H4/Hspec 판정 (계획 §15–§16·§20). 판정 규칙은 사전 지정: 방향 + source-block bootstrap 95% CI + 대응 대조 일관성."""
import json, math, os
import numpy as np, pandas as pd
from tools.eqrec4 import common as C
from tools.eqrec4.common import CAMP, FIG, QUADS


def _md(df, cols=None, fmt="{:.5f}"):
    if df is None or len(df) == 0:
        return "_(없음)_\n"
    df = df if cols is None else df[cols]; out = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join((fmt.format(v) if isinstance(v, (float, np.floating)) and not (isinstance(v, float) and math.isnan(v)) else str(v)) for v in r.values) + " |\n"
    return out


def _ci_txt(b):
    if not b or b.get("point") is None:
        return "NA"
    return f"{b['point']:+.4g}" + (f" [{b['ci95'][0]:+.4g}, {b['ci95'][1]:+.4g}] (n_grp {b['n_groups']})" if b.get("ci95") else " (CI 불가)")


def _sig(b, want):
    """want=+1: CI 전체 > 0 · −1: < 0. CI 없으면 None."""
    if not b or not b.get("ci95"):
        return None
    lo, hi = b["ci95"]; return (lo > 0) if want > 0 else (hi < 0)


def verdicts(J):
    v = {}
    # H1 RR: L1E4 checkpoints 의 within-checkpoint Spearman(q_A, e) (BCD) — 방향 일관 + CI
    h1 = J.get("h1", {}); rows = []
    for mk, r in h1.items():
        if mk.startswith("L1E4"):
            b = r.get("spearman_qA_e_BCD", {}); rows.append((mk, b.get("rho"), _sig(b, +1), _sig(b, -1)))
    pos, neg = sum(1 for _, _, p, _ in rows if p), sum(1 for _, _, _, n in rows if n)
    v["H1_RR"] = dict(state=("Supported in tested setting" if pos >= 4 and neg == 0 else "Opposed in tested setting" if neg >= 4 and pos == 0 else "Insufficient / unresolved"), evidence=[dict(model=m, rho=rh, ci_pos=p, ci_neg=n) for m, rh, p, n in rows],
                      claim="같은 checkpoint 안에서 q_A 가 낮은 sample 이 native L1 도 낮은가 (양의 ρ = 지지)", scope="within-checkpoint, native64, B∪C∪D, source-block bootstrap")
    fr = [(mk, r.get("FR_spearman_qA_hqnr", {})) for mk, r in h1.items() if mk.startswith("L1E4")]; negs = sum(1 for _, s in fr if s.get("rho") is not None and s["rho"] < 0 and s.get("p", 1) < 0.05)
    v["H1_FR"] = dict(state=("Supported in tested setting" if negs >= 4 else "Insufficient / unresolved"), evidence=[dict(model=m, rho=s.get("rho"), p=s.get("p")) for m, s in fr], claim="FR20 에서 q_A 낮음 ↔ raw HQNR 높음 (음의 ρ)", scope="20 scene, scene-level; 다른 stratum")
    # H2: response path, r=1, Spearman(q_A, d_e) > 0 (q 낮을수록 저하 작음)
    h2 = J.get("h2", {}); rows = []
    for mk, r in h2.items():
        if mk.startswith("L1E4") and "native64" in r:
            for rad in ("0.5", "1.0", "2.0"):
                b = r["native64"]["by_path"].get("response", {}).get(rad, {}).get("spearman_qA_d", {}); rows.append((mk, rad, b.get("rho"), _sig(b, +1)))
    pos = sum(1 for *_, p in rows if p); n = len(rows)
    v["H2"] = dict(state=("Supported in tested setting" if n and pos >= 0.6 * n else "Opposed in tested setting" if n and sum(1 for _, _, rh, _ in rows if rh is not None and rh < 0) >= 0.6 * n else "Insufficient / unresolved"), evidence=[dict(model=m, r=rad, rho=rh, ci_pos=p) for m, rad, rh, p in rows],
                   claim="q_A 낮은 sample 의 독립 방향(bank B) 추가 이동 저하 d_e 가 작은가 (반경별)", scope="native64 detail subsets (D), response path, common ROI")
    # H3a: learned − zero paired gain (native64 / RR / FR) for L1E4
    h3 = J.get("h3", {}); ev = []
    for mk, r in h3.items():
        if mk.startswith("L1E4"):
            for sc, s in r.items():
                b = s.get("gain_learned_vs_zero"); ev.append(dict(model=mk, scale=sc, gain=_ci_txt(b), ci_pos=_sig(b, +1), pos_frac=s.get("positive_fraction_vs_zero")))
    pos = sum(1 for e in ev if e["ci_pos"]); v["H3a"] = dict(state=("Supported in tested setting" if pos >= max(1, 0.6 * len(ev)) else "Insufficient / unresolved"), evidence=ev, claim="같은 가중치에서 learned correction 이 zero 보다 유리 (paired)", scope="I-L vs I-Z; U-Net 은 learned 로 학습된 것 — 별도 P0 와의 causal 비교가 아니다")
    ev = [dict(model=mk, scale=sc, spearman=s.get("H3b_spearman_qA_gain")) for mk, r in h3.items() if mk.startswith("L1E4") for sc, s in r.items()]
    sig = [e for e in ev if e["spearman"] and e["spearman"].get("rho") is not None and e["spearman"].get("p", 1) < 0.05]
    v["H3b"] = dict(state=("Insufficient / unresolved" if len(sig) < 0.5 * max(1, len(ev)) else ("Supported in tested setting" if all(e["spearman"]["rho"] < 0 for e in sig) else "Opposed in tested setting" if all(e["spearman"]["rho"] > 0 for e in sig) else "Insufficient / unresolved")),
                    evidence=ev, claim="q_A 가 learned−zero 이득의 크기를 예측 (음의 ρ = q 낮을수록 이득 큼)", scope="change-score coupling 주의 (e 와 gain 이 항을 공유)")
    # H4 diag/exec
    k = J.get("k10_h4", {}); kA = k.get("A", {}); kd = kA.get("q_added_mae_improvement") if isinstance(kA, dict) else None
    v["H4_diag"] = dict(state=("Supported in tested setting" if kd is not None and kd > 0 and kA.get("B1", {}).get("direction_acc", 0) > kA.get("B0", {}).get("direction_acc", 0) else "Insufficient / unresolved"), evidence=k, claim="q 를 더하면 K10 utility 예측이 좋아지는가 (pool 단위 LOO)")
    kp = J.get("k20_paired", {}).get("comparisons", {}); a = kp.get("EAQ_minus_EA"); b = kp.get("EAQ_minus_EAQ_SHUF")
    v["H4_exec"] = dict(state=("Supported in tested setting" if _sig(a, -1) and _sig(b, -1) else "Opposed in tested setting" if (_sig(a, +1) or _sig(b, +1)) else "Insufficient / unresolved") if kp else "H4 final_training_pilot_not_run",
                        evidence=dict(EAQ_minus_EA=_ci_txt(a), EAQ_minus_SHUF=_ci_txt(b), others={k_: _ci_txt(v_) for k_, v_ in kp.items()}), claim="K20: EAQ < EA 그리고 EAQ < EAQ-SHUF (D locked L1, exact 5000)", scope="Pair A, 5K 단기 적응; 최종 50K KD 성공이 아니다")
    v["Hspec"] = dict(state="descriptive", claim="spectral 재조합만으로 q/추정 c 가 달라지는가 (D20-B/D40-D) — 조건부 해석만", evidence=J.get("spectral", {}))
    return v


def main(profile=False):
    with C.Stage("REPORT" + ("-profile" if profile else ""), "tables A–F, figures, verdicts"):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rd = lambda n: C.read_csv(os.path.join(CAMP, n)) if os.path.exists(os.path.join(CAMP, n)) else None
        am, nm, qa, occ, ci, st, rm, gc, k10s, k20e = (rd(n) for n in ("assets_manifest.csv", "native_sample_metrics.csv", "quadrant_assignments.csv", "quadrant_occupancy.csv", "correction_interventions.csv", "output_stress.csv", "response_matrices.csv", "geometry_controls_summary.csv", "k10_cell_summary.csv", "k20_endpoint_metrics.csv"))
        J = dict(h1=C.load_json(os.path.join(CAMP, "h1_stats.json"), {}), h2=C.load_json(os.path.join(CAMP, "h2_stats.json"), {}), h3=C.load_json(os.path.join(CAMP, "h3_stats.json"), {}), k10_h4=C.load_json(os.path.join(CAMP, "k10_h4_diag.json"), {}), k20_paired=C.load_json(os.path.join(CAMP, "k20_D_paired.json"), {}),
                 g00=C.load_json(os.path.join(CAMP, "g00_reproduction.json"), {}), spectral=(gc.to_dict("records") if gc is not None else {}))
        pk = C.mkey(*C.PRIMARY); md = [f"# EQREC4-S1-v1 report ({C.time.strftime('%Y-%m-%d %H:%M')})\n\n계획 `{C.PLAN}` · 출력 root `work_dir/_eqrec4_s1_campaign/` · primary Teacher `{pk}`\n"]
        md.append("\n## Table A — 자산·재현\n" + _md(am[["model_key", "actual_update", "model_hash", "aligner_hash", "unet_hash", "exists"]] if am is not None else None, fmt="{}") + f"\nPrimary best_raw raw-original HQNR: 기록 {J['g00'].get('recorded_best_raw_hqnr')} / 재현 {J['g00'].get('reproduced_raw_original_hqnr')} (Δ {J['g00'].get('abs_diff')}); RR L1(정규화) {J['g00'].get('rr_l1_normalized_mean')}\n")
        # Table B — 네 집단 (primary)
        tb = []
        if nm is not None and qa is not None:
            n64 = nm[(nm.model_key == pk) & (nm.scale == "native64")].merge(qa[qa.model_key == pk][["sample_id", "quadrant_id"]], on="sample_id")
            for q in QUADS:
                s = n64[n64.quadrant_id == q]; row = dict(quadrant=q, n=int(len(s)), n_groups=int(s.source_group_id.nunique()), e_mean=float(s.e_native_full.mean()), qA_mean=float(s.q_A.mean()), qB_mean=float(s.q_B.mean()), c0_norm=float(s.c0_norm.mean()), edge_l1=float(s.edge_l1_roi.mean()), texture=float(s.pan_scharr_energy.mean()))
                if st is not None:
                    ss = st[(st.model_key == pk) & (st.scale == "native64") & (st.path == "response") & (st.quadrant_primary == q) & st.valid_support]; row["stress_d_e_r1"] = float(ss[ss.r == 1.0].d_e.mean()) if len(ss) else None
                if ci is not None:
                    c = ci[(ci.model_key == pk) & (ci.scale == "native64") & (ci.quadrant_primary == q)]; L_, Z = c[c["mode"] == "I-L"].set_index("sample_id").l1_roi, c[c["mode"] == "I-Z"].set_index("sample_id").l1_roi; row["gain_learned_vs_zero"] = float((Z - L_).mean()) if len(L_) else None
                if k10s is not None:
                    kk = k10s[(k10s.pair == "A") & k10s.cell.str.startswith(q)]; row["k10_U_soft_hard_rel"] = float(kk.U_soft_hard_rel_mean.mean()) if len(kk) else None
                tb.append(row)
        md.append("\n## Table B — 네 집단 (primary, native64; 같은 checkpoint)\n" + _md(pd.DataFrame(tb)) + "\n집단 정의상 E↓ 의 e 가 낮은 것은 결과가 아니다. 의미 있는 열은 stress·개입 이득·K10 utility 의 집단 차이다.\n")
        md.append("\n## Table C — 정합의 세 수준\n### relative response (PAN-only / MS-only / common)\n" + _md(rm[rm.model_key.str.startswith("L1E4")][["model_key", "scale", "mode", "n", "B_yy_mean", "B_xx_mean", "B_yx_mean", "B_xy_mean", "epe_mean" if "epe_mean" in rm else "epe_A_mean", "fit_rmse_mean"]] if rm is not None else None, fmt="{:.4f}"))
        md.append("### known absolute synthetic error (동일 격자 합성; calibration/OOD 진단)\n" + _md(gc, fmt="{:.4f}"))
        pxs = [C.load_json(os.path.join(CAMP, f)) for f in sorted(os.listdir(CAMP)) if f.startswith("native_proxy_summary_")]
        md.append("### native before/after proxy (FR512 PAN↔up(MS); 센서 GT 아님)\n" + _md(pd.DataFrame([dict(file=f, **{k: p["fr512"].get(k) for k in ("n", "n_both_accepted", "before_mag_median", "after_mag_median", "n_improved", "n_improved_accepted")}) for f, p in zip(sorted(x for x in os.listdir(CAMP) if x.startswith("native_proxy_summary_")), pxs)]), fmt="{:.4f}"))
        if ci is not None:
            td = ci[ci.status.fillna("ok") == "ok"].copy() if "status" in ci else ci.copy(); td["metric"] = np.where(td.scale == "fr512", td.get("raw_original_hqnr"), td.get("l1_roi")); td["mode_group"] = td["mode"].str.replace(r"I-S\d+", "I-S(mean10)", regex=True)
            md.append("\n## Table D — correction 개입 (mode 평균; native64/RR = ROI L1 ↓, FR = raw-original HQNR ↑)\n" + _md(td.groupby(["model_key", "scale", "mode_group"]).metric.mean().reset_index().pivot(index=["model_key", "scale"], columns="mode_group", values="metric").reset_index(), fmt="{:.5f}"))
        md.append("\n## Table E — cue 유용성\n### K10 cell summary (U_soft-hard_rel > 0 = 추가 soft 가 추가 hard 보다 C 에서 유리)\n" + _md(k10s, fmt="{:.4f}") + "\n### K20 endpoints\n" + _md(k20e, fmt="{:.5f}") + "\n### K20 D paired (exact 5000)\n" + "\n".join(f"- {k}: {_ci_txt(v)}" for k, v in J["k20_paired"].get("comparisons", {}).items()) + "\n")
        V = verdicts(J); C.dump_json(os.path.join(CAMP, "hypothesis_verdicts.json"), V)
        md.append("\n## Table F — 판정\n| 가설 | 상태 | 주장 | 범위 |\n|---|---|---|---|\n" + "".join(f"| {k} | **{v['state']}** | {v.get('claim', '')} | {v.get('scope', '')} |\n" for k, v in V.items()))
        md.append(f"\n## 한 문장 결론 (§20 형식)\n> 현재 L1E4 의 offset consistency 는 [추가 이동 반응: {V['H2']['state']} / native 위치 개선: proxy 표 참조 / native 복원: H1 {V['H1_RR']['state']} / Student 감독 유용성: H4 {V['H4_exec']['state']}] 까지를 설명했으며, "
                  f"reconstruction 과 엇갈리는 sample 에서는 [D30/D40/D50 표의 검증된 원인 / 남은 후보: mixed/unresolved 로 표시] 가 관측됐다. e_T·a_T 에 q_T 를 추가한 soft/hard 선택은 [{V['H4_exec']['state']}] 이었고, 그 판단은 [동일 checkpoint·대응 개입·source 분할·실제 KD endpoint(D locked, exact 5000)] 에 근거한다.\n"
                  "\nL1E4 의 채택과 q_T 의 Teacher-quality gate 채택은 별개의 결정으로 유지한다. FR20 은 개발에 반복 사용된 benchmark 다. source group 은 index-block proxy 라 독립성 가정이 약하다(CI 는 보수적으로 읽는다).\n")
        open(os.path.join(CAMP, "report_EQREC4.md"), "w").write("".join(md))
        # figures 2–7 (best effort)
        try:
            if rm is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); r = rm[(rm.model_key == pk) & (rm.scale == "native64")]; x = np.arange(len(r)); ax.bar(x - 0.2, r.B_yy_mean, 0.4, label="B_yy"); ax.bar(x + 0.2, r.B_xx_mean, 0.4, label="B_xx"); ax.set_xticks(x); ax.set_xticklabels(r["mode"]); ax.axhline(-1, c="k", ls=":"); ax.axhline(1, c="k", ls=":"); ax.legend(); ax.set_title(f"Fig2 signed response diag ({pk})"); fig.savefig(os.path.join(FIG, "fig2_modality_response.png"), dpi=110); plt.close(fig)
            if st is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); s = st[(st.model_key == pk) & (st.scale == "native64") & st.valid_support]; g = s.groupby(["quadrant_primary", "path"]).d_e.mean().unstack(); g.plot.bar(ax=ax); ax.set_ylabel("d_e (ROI L1 stress − native)"); ax.set_title("Fig3 native→stress by quadrant (bank B r∈{.5,1,2} mean)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_stress_by_quadrant.png"), dpi=110); plt.close(fig)
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
            ld = rd("correction_landscape.csv")
            if ld is not None:
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                for ax, q in zip(axs, QUADS):
                    s = ld[(ld.model_key == pk) & (ld.quadrant_primary == q) & (ld.kind == "grid")]
                    if len(s):
                        sid = s.sample_id.iloc[0]; g = s[s.sample_id == sid].pivot(index="c_dy", columns="c_dx", values="l1_roi"); im = ax.imshow(g.values, origin="lower", extent=[-2.25, 2.25, -2.25, 2.25]); ax.set_title(f"Fig6 L1 landscape {q} #{sid}"); c0 = ld[(ld.sample_id == sid) & (ld.kind == "c0") & (ld.model_key == pk)]
                        if len(c0):
                            ax.plot(c0.c_dx, c0.c_dy, "r*", ms=10)
                        plt.colorbar(im, ax=ax)
                fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_landscape_examples.png"), dpi=110); plt.close(fig)
            if k10s is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); k = k10s[k10s.pair == "A"]; ax.bar(k.cell, k.U_soft_hard_rel_mean); ax.axhline(0, c="k"); ax.set_ylabel("U_soft-hard (rel, C)"); ax.set_title("Fig7 K10 utility by cell (pair A)"); plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_k10_utility.png"), dpi=110); plt.close(fig)
            bp = rd("band_edge_profiles.csv")
            if bp is not None:
                fig, ax = plt.subplots(figsize=(10, 4)); g = bp.groupby("model_key")[["crossing_offset_mean_abs", "width_ratio", "edge_mae_dn"]].mean(); g.plot.bar(ax=ax, subplots=False); ax.set_title("Fig5 edge profile summary (RR, GT edge mask)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_edge_profiles.png"), dpi=110); plt.close(fig)
        except Exception as ex:
            open(os.path.join(CAMP, "report_figures_error.txt"), "a").write(repr(ex) + "\n")
        print(json.dumps({k: v["state"] for k, v in V.items()}, ensure_ascii=False))
