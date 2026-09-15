"""X40 — 여러 데이터셋에서 무엇이 반복되는가 (§16). analysis/cross_sensor_results.csv: 관계 × 센서(in-domain WV3/QB/GF2 · zero-shot WV2) 행렬 — measured / pending_dependency / zero_shot / not_implemented.
matched 대조(§8.4·§9): EuCd↔EuCu 는 e_base·contrast 를 1:1 nearest neighbour(caliper .2 CAL SD) 로 맞춘 뒤 s_out(.25) 차이; EuCd↔EdCd 는 q 를 맞춘 뒤 contrast/e_base 차이."""
import os
import numpy as np, pandas as pd
from tools.smec12 import common as C


def match_pairs(a, b, cols, caliper_sd):
    """1:1 nearest neighbour without replacement (표준화 거리, caliper = .2 SD per col)."""
    if not len(a) or not len(b):
        return []
    A = a[cols].values.astype(float); Bv = b[cols].values.astype(float); sd = np.array([max(s, 1e-12) for s in caliper_sd]); used = set(); pairs = []
    for i in range(len(A)):
        d = np.abs((Bv - A[i]) / sd); ok = np.where((d <= 0.2).all(1))[0]; ok = [j for j in ok if j not in used]
        if ok:
            j = min(ok, key=lambda jj: float(d[jj].sum())); used.add(j); pairs.append((a.index[i], b.index[j]))
    return pairs


def main(profile=False):
    with C.Stage("X40", "cross-sensor matrix"):
        sm = pd.read_csv(C.p_("raw", "sample_metrics.csv")); d10 = C.load_json(C.p_("analysis", "d10_stats.json"), {}); i20 = C.load_json(C.p_("analysis", "i20_stats.json"), {}); i23 = C.load_json(C.p_("analysis", "i23_stats.json"), {})
        frp = C.p_("raw", "fixed_residual_response.csv"); fr = pd.read_csv(frp) if os.path.exists(frp) else pd.DataFrame(); rows = []; matched = []
        reg = C.asset_registry()
        for s in ("wv3", "qb", "gf2", "wv2"):
            S = C.SENSORS[s]["S"]; role = C.SENSORS[s]["role"]
            prim = [k for k in d10 if k.startswith(f"{S}|") and ("|L1E4|" in k) and k.endswith(("|native64|DISC", "|rr256tile|RRTILE"))] if s != "wv2" else [k for k in d10 if k.startswith("WV2|") and k.endswith("|rr256tile|RRTILE") and "@wv2" in k and "|L1E4|" in k]
            dep = ("pending_dependency: " + ", ".join(k for k, v in reg.items() if k.startswith(S + "|") and v["status"] != "available")) if (s != "wv2" and not prim) else None
            def row(rel, status, value=None, note=""):
                rows.append(dict(relation=rel, sensor=S, role=role, status=status, value=value, note=note))
            if not prim:
                for rel in ("raw q–e", "contrast/e_base 통제 q–e", "EuCd descriptor 차이", "동일 residual 출력 민감도", "q–correction 이득", "spectral-only absolute bias", "c_geo 대 c_rec", "q 추가 outcome 예측"):
                    row(rel, "pending_dependency" if s != "wv2" else "pending_compute", note=dep or "WV3 primary atlas 필요")
                continue
            k = prim[0]; d = d10[k]; sq = d.get("spearman_q_e") or {}; sqc = d.get("spearman_q_e_contrastnorm") or {}
            row("raw q–e", "measured" if s != "wv2" else "zero_shot", sq.get("rho"), f"CI {sq.get('ci95')} n {d['n']} sources {d['n_sources']} · LOSO {d.get('loso_q_e')} · by r {d.get('spearman_q_e_by_r')}")
            row("contrast/e_base 통제 q–e", "measured" if s != "wv2" else "zero_shot", sqc.get("rho"), f"contrast-normalized e: CI {sqc.get('ci95')} · ρ(q,e_base) {d.get('spearman_q_ebase')} · ρ(e,e_base) {d.get('spearman_e_ebase')} · matched raw 는 아래 matched 표")
            bq = d.get("by_quadrant") or {}
            if bq and "EuCd" in bq and "EuCu" in bq:
                row("EuCd descriptor 차이", "measured" if s != "wv2" else "zero_shot", None, "median: " + "; ".join(f"{q}: e {v['e']:.4f} q {v['q']:.3f} e_base {v['e_base']} texture {v['texture']:.4f} contrast {v['contrast']}" for q, v in bq.items()) + " (descriptor bank D12 전체·blind gallery 는 pending_compute)")
                # matched EuCd↔EuCu on e_base & contrast (CAL SD caliper .2), then s_out(.25) difference
                g = sm[(sm.sensor == S) & (sm.part == ("DISC" if s != "wv2" else "RRTILE")) & sm.quadrant.notna()]
                prim_model = next((m for m in sorted(g.model.unique()) if "|L1E4|S2025|best_hqnr" in m), None); g = g[g.model == prim_model] if prim_model else g
                cal = sm[(sm.sensor == S) & (sm.part == ("CAL" if s != "wv2" else "RRTILE")) & (sm.model == prim_model)] if prim_model else sm[(sm.sensor == S) & (sm.part == ("CAL" if s != "wv2" else "RRTILE"))]
                sd = [float(cal.e_base.std()) if "e_base" in cal and cal.e_base.notna().any() else 1.0, float(cal.contrast_gt.std()) if "contrast_gt" in cal and cal.contrast_gt.notna().any() else 1.0]
                a, b = g[g.quadrant == "EuCd"].set_index("sample_id"), g[g.quadrant == "EuCu"].set_index("sample_id")
                if "e_base" in g and len(a) and len(b):
                    pairs = match_pairs(a, b, ["e_base", "contrast_gt"], sd); matched.append(dict(sensor=S, pair="EuCd~EuCu", matched_on="e_base,contrast_gt (caliper .2 CAL SD)", n_a=int(len(a)), n_b=int(len(b)), n_pairs=len(pairs), unmatched_frac=1 - len(pairs) / max(1, len(a))))
                    if pairs and len(fr):
                        f25 = fr[(fr.sensor == S) & (fr.model == prim_model) & (fr.r == 0.25) & (fr.part == "DISC")].groupby("sample_id").s_out_full.mean(); dif = [f25.get(i) - f25.get(j) for i, j in pairs if i in f25 and j in f25]     # primary model 만 (REP 과 섞지 않는다)
                        if len(dif) >= 4:
                            bs = C.block_bootstrap(np.array(dif), np.arange(len(dif))); row("동일 residual 출력 민감도", "measured", float(np.mean(dif)), f"matched(e_base·contrast) EuCd−EuCu s_out(.25), primary {prim_model} [CI {bs['ci95']}] n_pairs {len(dif)} (raw 집단 차이는 i20_stats); patch-level CI (source proxy) → descriptive")
                        else:
                            row("동일 residual 출력 민감도", "measured_unmatched", None, f"matched pair 에 I20 상세 subset 이 부족 (pairs {len(pairs)}, with I20 {len(dif)}) — raw 집단 차이는 i20_stats")
                    else:
                        row("동일 residual 출력 민감도", "pending_compute" if not len(fr) else "comparison_not_supported", None, "I20 미실행 또는 matched pair 없음")
                # EuCd↔EdCd matched on q → contrast/e_base difference
                a2, b2 = g[g.quadrant == "EuCd"].set_index("sample_id"), g[g.quadrant == "EdCd"].set_index("sample_id"); qsd = [float(pd.to_numeric(cal.q_A, errors="coerce").std())]
                if len(a2) and len(b2):
                    a2 = a2.assign(q_num=pd.to_numeric(a2.q_A, errors="coerce")); b2 = b2.assign(q_num=pd.to_numeric(b2.q_A, errors="coerce")); pr2 = match_pairs(a2, b2, ["q_num"], qsd)
                    if pr2:
                        dc = [a2.loc[i].contrast_gt - b2.loc[j].contrast_gt for i, j in pr2]; de = [a2.loc[i].e_base - b2.loc[j].e_base for i, j in pr2]
                        matched.append(dict(sensor=S, pair="EuCd~EdCd", matched_on="q_A (caliper .2 CAL SD)", n_a=int(len(a2)), n_b=int(len(b2)), n_pairs=len(pr2), unmatched_frac=1 - len(pr2) / max(1, len(a2)), d_contrast=float(np.mean(dc)), d_e_base=float(np.mean(de))))
            gi = [kk for kk in i23 if kk.startswith(f"{S}|") and kk.endswith("|DISC")]
            if gi:
                v = i23[gi[0]]; row("q–correction 이득", "measured", v["g_corr_zero_minus_learned"]["point"], f"CI {v['g_corr_zero_minus_learned']['ci95']} · by quadrant {({q: round(x['g_corr'], 5) for q, x in v['by_quadrant'].items()})} · coupling: gain 은 e_learned 와 항 공유")
            else:
                row("q–correction 이득", "pending_compute" if s != "wv2" else "not_implemented", None, "I23-B 미실행" if s != "wv2" else "WV2 tile 개입은 이번 release 없음")
            row("spectral-only absolute bias", "not_implemented", None, "I22 (합성 2×N 대조) — pending_compute"); row("c_geo 대 c_rec", "not_implemented", None, "I23-A/C (독립 estimator·landscape) — pending_compute"); row("q 추가 outcome 예측", "not_implemented", None, "C50 — pending_compute")
        C.write_csv(C.p_("analysis", "cross_sensor_results.csv"), rows); C.write_csv(C.p_("analysis", "matched_sample_pairs.csv"), matched)
        print("[x40] rows", len(rows), "matched", [(m["sensor"], m["pair"], m["n_pairs"]) for m in matched], flush=True)
