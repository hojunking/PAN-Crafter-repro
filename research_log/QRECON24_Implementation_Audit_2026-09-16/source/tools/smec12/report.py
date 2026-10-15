"""REPORT — report_SMEC12.md · analysis/hypothesis_verdicts.csv (M1–M7; 기본 insufficient) · stage 상태(complete / pending_dependency / pending_compute / not_implemented) · coverage 갱신."""
import json, os, time
import numpy as np, pandas as pd
from tools.smec12 import common as C
STAGE_PLAN = [("A00", "a00", "자산·데이터·source audit"), ("B01–B04", None, "누락 모델 준비 학습 (config/queues/smec12_<server>.txt)"), ("D10", "d10", "전 센서 atlas · q–e · label"), ("D11", "d10", "대비 정규화·e_base·band/edge (D10 안에서 A/B/C 일부)"), ("D12", None, "descriptor bank · blind gallery"),
              ("I20", "i20", "동일 잔여 이동 민감도 · Jacobian"), ("I21", None, "aligner view 개입"), ("I22", None, "spectral / known-geometry 2×N"), ("I23", "i23", "correction 치환 (B); A/C/D 미구현"), ("I24", "i24", "seq/single 분해 (A); B/C 미구현"), ("I25", "i25", "gradient 측정 (A); B 미구현"),
              ("S30", None, "context·checkpoint·seed membership"), ("X40", "x40", "센서별/교차센서 행렬"), ("C50", None, "q 추가 정보 검증(descriptor vs +q)"), ("R60", None, "조건부 맞춤 대응 pilot")]


def stage_status(led):
    done = {e["stage"].split("-")[0] for e in led if e["status"] == "done"}; out = []
    for name, cli, desc in STAGE_PLAN:
        if cli is None:
            st = "pending_compute (not_implemented in this release)" if name not in ("B01–B04",) else "training dependency (bootstrap queue)"
        else:
            st = "complete (available assets)" if cli.upper() in done else "pending_compute"
        out.append(dict(stage=name, status=st, description=desc))
    return out


def verdicts(d10, i20, i23, i25, xs):
    rows = []; avail = {r["sensor"]: r for _, r in xs[xs.relation == "raw q–e"].iterrows()} if len(xs) else {}
    def v(h, state, evidence, scope):
        rows.append(dict(hypothesis=h, verdict=state, evidence=evidence, scope=scope, generated=time.strftime("%Y-%m-%dT%H:%M:%S")))
    wv3 = [k for k in d10 if k.startswith("WV3|") and "|L1E4|S2025|best_hqnr|native64|DISC" in k]
    if wv3:
        d = d10[wv3[0]]; sq = d.get("spearman_q_e", {}); sqc = d.get("spearman_q_e_contrastnorm", {})
        same_sign = (sq.get("rho") is not None and sqc.get("rho") is not None and np.sign(sq["rho"]) == np.sign(sqc["rho"]) and sqc.get("ci95") and (sqc["ci95"][0] > 0 or sqc["ci95"][1] < 0))
        v("M1 대비/척도", "opposed_within_condition (정규화 뒤에도 관계 유지)" if same_sign else ("supported_within_condition" if sq.get("rho") is not None and sqc.get("rho") is not None else "insufficient"), f"raw ρ {sq.get('rho')} CI {sq.get('ci95')} · contrast-normalized ρ {sqc.get('rho')} CI {sqc.get('ci95')} (WV3 L1E4 S2025 DISC)", "정규화만으로 원인 확정 아님; matched raw 는 x40 matched 표")
    else:
        v("M1 대비/척도", "insufficient", "WV3 atlas 없음", "")
    v("M2 구조 관측 가능성", "insufficient", "descriptor bank(D12) 미구현 — texture/contrast 중앙값만 (d10 by_quadrant)", "pending_compute")
    ks = [k for k in i20 if k.startswith("WV3|") and "|L1E4|S2025|" in k and k.endswith("|DISC") and "s_out_r0.25_EuCd_minus_EuCu" in i20[k]]          # primary 만 (REP 은 별도 행)
    if ks:
        e = i20[ks[0]]["s_out_r0.25_EuCd_minus_EuCu"]; conf = [k for k in i20 if k.startswith("WV3|") and "|L1E4|S2025|" in k and k.endswith("|CONF") and "s_out_r0.25_EuCd_minus_EuCu" in i20[k]]; ec = i20[conf[0]]["s_out_r0.25_EuCd_minus_EuCu"] if conf else None
        reps = {k: i20[k]["s_out_r0.25_EuCd_minus_EuCu"]["diff"] for k in i20 if "L1E4REP" in k and "s_out_r0.25_EuCd_minus_EuCu" in i20[k]}
        st = "supported_within_condition" if (e["ci95"] and (e["ci95"][0] > 0) and ec and ec["ci95"] and ec["ci95"][0] > 0) else ("opposed" if (e["ci95"] and e["ci95"][1] < 0) else "insufficient")
        v("M3 출력 민감도", st, f"s_out(.25 px) EuCd−EuCu (primary L1E4 S2025): DISC {e['diff']:+.5f} CI {e['ci95']}" + (f" · CONF {ec['diff']:+.5f} CI {ec['ci95']}" if ec else "") + f" · REP {({k.split('|')[3]: round(x, 5) for k, x in reps.items()})}", "raw 집단 차이(matched 는 x40); source proxy 라 CI 는 descriptive; DISC 와 CONF 가 모두 0 을 제외해야 supported")
    else:
        v("M3 출력 민감도", "insufficient", "I20 미실행", "")
    v("M4 spectral 불일치", "insufficient", "I22 미구현", "pending_compute"); v("M5 context/모델", "insufficient", "S30 미구현 (seed 별 atlas 는 d10 에 있음 — membership 전이표는 pending)", "pending_compute")
    ins = [s for s in ("QB", "GF2") if avail.get(s, {}).get("status") == "measured"]
    v("M6 데이터 재현", ("replicated_across_sensors" if len(ins) >= 1 and all(np.sign(float(avail[s]["value"])) == np.sign(float(avail.get("WV3", {}).get("value") or 0)) for s in ins if avail[s].get("value") is not None) else "insufficient"), f"raw q–e 부호: " + "; ".join(f"{s}: {r['value']} ({r['status']})" for s, r in avail.items()), "QB/GF2 는 준비 학습 뒤 (pending_dependency)" if len(ins) == 0 else "1 seed in-domain")
    v("M7 대응 가능성", "insufficient", "C50/R60 미구현", "pending_compute"); return rows


def main(profile=False):
    with C.Stage("REPORT", "report_SMEC12.md / verdicts"):
        led = [json.loads(l) for l in open(C.p_("ledger.jsonl"))] if os.path.exists(C.p_("ledger.jsonl")) else []
        d10 = C.load_json(C.p_("analysis", "d10_stats.json"), {}); i20 = C.load_json(C.p_("analysis", "i20_stats.json"), {}); i23 = C.load_json(C.p_("analysis", "i23_stats.json"), {}); i24 = C.load_json(C.p_("analysis", "i24_stats.json"), {}); i25 = C.load_json(C.p_("analysis", "i25_stats.json"), {})
        xs = pd.read_csv(C.p_("analysis", "cross_sensor_results.csv")) if os.path.exists(C.p_("analysis", "cross_sensor_results.csv")) else pd.DataFrame(); mp = C.p_("analysis", "matched_sample_pairs.csv"); matched = pd.read_csv(mp) if os.path.exists(mp) and os.path.getsize(mp) > 5 else pd.DataFrame()
        reg = C.asset_registry(); st = stage_status(led); vd = verdicts(d10, i20, i23, i25, xs); C.write_csv(C.p_("analysis", "hypothesis_verdicts.csv"), vd)
        cov = []
        for s, X in C.SENSORS.items():
            S = X["S"]; cov.append(dict(sensor=S, role=X["role"], data_ready=bool(C.data_registry()[S]["data_ready"]), models_ready=any(k.startswith(S + "|") and v["status"] == "available" for k, v in reg.items()) if s != "wv2" else "WV3 models", atlas_done=any(k.startswith(S + "|") for k in d10),
                            intervention_done=any(k.startswith(S + "|") for k in i20), confirm_done=any(k.startswith(S + "|") and k.endswith("|CONF") for k in i20), training_dependency=";".join(k for k, v in reg.items() if k.startswith(S + "|") and v["status"] != "available")))
        C.write_csv(C.p_("manifests", "coverage.csv"), cov)
        L = [f"# SMEC12 report — {C.CAMPAIGN_ID} ({C.SERVER}, {time.strftime('%Y-%m-%dT%H:%M:%S')})", "", f"계획 `{C.PLAN}` · 노트 `{C.NOTE}` · root `{os.path.relpath(C.CAMP, C.ROOT)}/`. 이 보고서는 **있는 자산에 대한 부분 결과**이며 미실행 stage 를 complete 로 적지 않는다 (§19.3).", "",
             "관측 단위: in-domain = train 64² patch(CAL 512 / DISC 1536 / CONF 1024; 32-index block = source proxy, seen_in_pretraining) · WV2 zero-shot = RR/FR 20 scene × 16 tile · 상세 개입 = 네 집단당 ≤32 × DISC/CONF. e = native full64 L1(정규화 단위), q = bank A(16 probe) offset-consistency 잔차 평균(HR px).", "",
             "## 0. Stage 상태", "", "| stage | 상태 | 내용 |", "|---|---|---|"] + [f"| {r['stage']} | {r['status']} | {r['description']} |" for r in st]
        L += ["", "## 1. Coverage", "", "| sensor | role | data | models | atlas | intervention | confirm | dependency |", "|---|---|---|---|---|---|---|---|"] + [f"| {c['sensor']} | {c['role']} | {c['data_ready']} | {c['models_ready']} | {c['atlas_done']} | {c['intervention_done']} | {c['confirm_done']} | {c['training_dependency'] or '—'} |" for c in cov]
        L += ["", "## 2. Atlas · q–e (D10/D11)", "", "| sensor · model · part | n | sources | ρ(q,e) [CI] | ρ contrast-norm | ρ by r (.25/.5/1/2) | LOSO | ρ(q,e_base) | ρ(q,texture) | quadrants | boundary_near |", "|---|---:|---:|---|---|---|---|---|---|---|---|"]
        for k, d in d10.items():
            if "|" in k and isinstance(d, dict) and d.get("spearman_q_e"):
                sq = d["spearman_q_e"]; L.append(f"| {k} | {d['n']} | {d['n_sources']} | {sq.get('rho') and round(sq['rho'], 3)} {sq.get('ci95')} | {(d.get('spearman_q_e_contrastnorm') or {}).get('rho') and round(d['spearman_q_e_contrastnorm']['rho'], 3)} | {[(v and round(v, 3)) for v in (d.get('spearman_q_e_by_r') or {}).values()]} | {d.get('loso_q_e') and [round(d['loso_q_e']['min'], 3), round(d['loso_q_e']['max'], 3)]} | {d.get('spearman_q_ebase') and round(d['spearman_q_ebase'], 3)} | {d.get('spearman_q_texture') and round(d['spearman_q_texture'], 3)} | {d.get('quadrants')} | {d.get('boundary_near_frac') and round(d['boundary_near_frac'], 3)} |")
        L += ["", "네 집단 중앙값 (primary WV3 L1E4 S2025 DISC): " + "; ".join(f"{q}: e {v['e']:.4f} · q {v['q']:.3f} · e_base {v['e_base']:.4f} · g_restore {v['g_restore']:.4f} · texture {v['texture']:.4f} · contrast {v['contrast']:.3f} · edge {v['edge']:.4f}" for q, v in (d10.get("WV3|WV3|L1E4|S2025|best_hqnr|native64|DISC", {}).get("by_quadrant") or {}).items()) if any("WV3|L1E4|S2025|best_hqnr|native64|DISC" in k for k in d10) else "", ""]
        L += ["## 3. 같은 잔여 이동에 대한 출력 민감도 (I20)", ""]
        for k, v in i20.items():
            if isinstance(v, dict) and "by_quadrant" in v:
                L.append(f"- {k}: s_out(r=.125/.25/.5/1) by quadrant: " + "; ".join(f"{q}: {[round(x[f's_out_r{r}'], 5) for r in C.RADII_S]} (d_gt .5: {x['d_gt_r0.5']:+.5f}, neg {x['d_gt_r0.5_neg_frac']:.2f}, n {x['n']})" for q, x in v["by_quadrant"].items()) + (f" · EuCd−EuCu(.25) {v['s_out_r0.25_EuCd_minus_EuCu']['diff']:+.5f} CI {v['s_out_r0.25_EuCd_minus_EuCu']['ci95']}" if "s_out_r0.25_EuCd_minus_EuCu" in v else "") + (f" · EuCd−EdCd(.25) {v['s_out_r0.25_EuCd_minus_EdCd']['diff']:+.5f}" if "s_out_r0.25_EuCd_minus_EdCd" in v else "") + f" · Jacobian(h .125) {v.get('jacobian_h0125_by_quadrant')} · h 안정성 ρ {v.get('jacobian_h_stability_corr')} · residual_B {v.get('residual_B')}")
        L += ["", "## 4. Correction 치환 (I23-B) · seq/single 분해 (I24-A) · gradient (I25-A)", ""]
        for k, v in i23.items():
            if isinstance(v, dict) and "g_corr_zero_minus_learned" in v:
                L.append(f"- I23 {k}: g_corr = e(0)−e(c0) {v['g_corr_zero_minus_learned']['point']:+.5f} CI {v['g_corr_zero_minus_learned']['ci95']} · wrong−learned {v['wrong_minus_learned']['point']:+.5f} · CAL-median−learned {v['cal_minus_learned']['point']:+.5f} · shuffle−learned {v['shuffle_minus_learned']:+.5f} · 양 {v['gain_pos_frac']:.2f} · by quadrant {({q: round(x['g_corr'], 5) for q, x in v['by_quadrant'].items()})} (coupling 주의)")
        for k, v in i24.items():
            if isinstance(v, dict) and "r1.0" in v:
                L.append(f"- I24 {k}: r=1 seq−single {v['r1.0']['seq_minus_single']:.5f} · single−0 {v['r1.0']['single_minus_zero']:.5f} · interp share {v['r1.0']['interp_share']:.2f} · known-inverse seq vs native {v['r1.0']['known_inverse_seq_vs_native']:.2e} (axis r1 {v.get('axis_r1_known_inverse_seq_vs_native')}) · r=.5 by quadrant {v.get('by_quadrant_r0.5')}")
        for k, v in i25.items():
            if isinstance(v, dict) and "cos_mean" in v:
                L.append(f"- I25 {k}: cos(g_r,g_o) {v['cos_mean']:+.3f} (neg {v['cos_neg_frac']:.2f}, NA {v['na_zero']}) · 1e-4‖g_o‖/‖g_r‖ {v['weighted_ratio_median']:.3g} · by quadrant {({q: (round(x['cos_mean'], 3), round(x['ratio'], 4)) for q, x in v['by_quadrant'].items()})} · by part {({p: round(x['cos'], 3) for p, x in v['by_part'].items()})}")
        L += ["", "## 5. 센서 행렬 (X40)", "", "| relation | sensor | role | status | value | note |", "|---|---|---|---|---|---|"] + [f"| {r.relation} | {r.sensor} | {r.role} | {r.status} | {'' if pd.isna(r.value) else round(float(r.value), 5)} | {r.note} |" for _, r in xs.iterrows()]
        if len(matched):
            L += ["", "matched 표: " + "; ".join(f"{m.sensor} {m.pair} on {m.matched_on}: pairs {m.n_pairs}/{m.n_a} (unmatched {m.unmatched_frac:.2f})" + (f", Δcontrast {m.d_contrast:+.4f}, Δe_base {m.d_e_base:+.4f}" if "d_contrast" in matched and not pd.isna(m.get("d_contrast")) else "") for _, m in matched.iterrows())]
        L += ["", "## 6. 판정표 (M1–M7; 기본 insufficient)", "", "| 가설 | 판정 | 근거 | 범위 |", "|---|---|---|---|"] + [f"| {r['hypothesis']} | {r['verdict']} | {r['evidence']} | {r['scope']} |" for r in vd]
        L += ["", "## 7. 미실행·미구현 (complete 로 적지 않음)", "", "- D12 descriptor bank·blind gallery, I21 aligner view 개입, I22 spectral/known-geometry, I23-A/C/D(독립 estimator·landscape·baseline 반복), I24-B/C(matched filtering·amplitude/phase), I25-B(parameter 개입), S30(context/seed membership), C50(q 추가 정보), R60(맞춤 대응): pending_compute — 이번 release 의 backbone(A00/D10/I20/I23-B/I24-A/I25-A/X40) 위에 추가한다.",
              "- QB/GF2: 준비 학습(B01–B04) 완료 뒤 같은 stage 를 재실행하면 덧붙는다 (pending_dependency).", "- source group 은 index-block proxy → 모든 CI 는 descriptive (§4.5 3). FIT70 source-holdout cohort 없음."]
        open(C.p_("report_SMEC12.md"), "w").write("\n".join(L) + "\n"); print("[report] verdicts:", [(r["hypothesis"][:3], r["verdict"][:28]) for r in vd], flush=True)
