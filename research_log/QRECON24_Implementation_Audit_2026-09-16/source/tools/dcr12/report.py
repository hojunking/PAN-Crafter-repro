"""REPORT — 계획 §13.2 의 7 질문 + §14.1 판정 라벨. report.md · run_status.json. 자동 판정은 §14 규칙만 적용하고 원인 해석은 하지 않는다."""
import json, os, time
import numpy as np, pandas as pd
from tools.dcr12 import common as C


def _j(p, d=None):
    return C.load_json(os.path.join(C.CAMP, p), d)


def _csv(p):
    q = os.path.join(C.CAMP, p); return pd.read_csv(q) if os.path.exists(q) else pd.DataFrame()


def fmt_ci(b):
    return f"{b['point']:+.5f} [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] (n_grp {b['n_groups']})" if b and b.get("ci95") else (f"{b['point']:+.5f} (CI 없음)" if b and b.get("point") is not None else "—")


def verdicts(d01, d02, d03, pr, gate):
    v = {}
    t0 = d01.get("thresholds", {}).get("T0", {}); v["cells_and_C_variability"] = dict(degenerate=t0.get("degenerate_metric"), C_cv=t0.get("C_cv"), tC=t0.get("tC"), tR=t0.get("tR"))
    sc = d01.get("T0|DISC", {}).get("spearman_C_R", {}); fr = d01.get("T0|FR", {}).get("spearman_C_hqnr", {})
    v["C_vs_native_R"] = dict(rho=sc.get("rho"), ci95=sc.get("ci95"), scene_C_vs_hqnr_rho=fr.get("rho"), scene_p=fr.get("p"))
    bR = d02.get("T0|native64", {}).get("B_R_zero_minus_learned"); bQ = d02.get("T0|fr512", {}).get("B_Q_learned_minus_zero"); v["learned_vs_zero_T0"] = dict(B_R=bR, B_Q=bQ)
    cos = d03.get("T0", {}); v["gradient_conflict_T0"] = dict(cos_gT_gC_mean=cos.get("cos_gT_gC_mean"), neg_frac=cos.get("cos_gT_gC_neg_frac"), by_cell=cos.get("by_cell"))
    paired = [p for p in pr.to_dict("records") if p.get("status") == "paired"]; d = [p["d_selected_hqnr"] for p in paired if p.get("d_selected_hqnr") is not None]
    v["B1_minus_B0"] = dict(n_seeds=len(d), d_selected_hqnr=d, all_positive=(bool(d) and all(x > 0 for x in d)), all_negative=(bool(d) and all(x < 0 for x in d)))
    # §14.1 라벨
    labels = []
    if len(d) >= 2 and all(x > 0 for x in d):
        labels.append("TASK_ADAPTATION_CANDIDATE (B1−B0 HQNR 두 seed 양; 제3 seed 확인 필요)")
    elif len(d) >= 2 and (any(x > 0 for x in d) and any(x < 0 for x in d)):
        labels.append("INCONCLUSIVE (B1−B0 seed 별 부호 반전)")
    elif len(d) >= 2:
        labels.append("NO_SUPPORT_AT_TESTED_SETTING (B1−B0 HQNR 두 seed 모두 비양)")
    else:
        labels.append("INCONCLUSIVE (paired seed < 2)")
    c_pred = (sc.get("rho") is not None and sc.get("ci95") and (sc["ci95"][0] > 0 or sc["ci95"][1] < 0))
    a_gain = (bR and bR.get("ci95") and bR["ci95"][0] > 0) or (len(d) >= 2 and all(x > 0 for x in d))
    if a_gain and not c_pred:
        labels.append("ALIGNER_USEFUL_METRIC_UNPROVEN (A 의 이득은 보이나 C 기반 선별은 지지되지 않음)")
    if gate and gate.get("passed"):
        labels.append("QUADRANT_ROUTING_CANDIDATE (D04 gate 통과; B2/B3·CMASS 확인 필요)")
    elif c_pred and not (gate and gate.get("passed")):
        labels.append("ROBUSTNESS_ONLY? — C–R 상관은 있으나 KD utility 의 추가정보 없음 (D04) — 사람이 표 C/D02 와 함께 판단")
    v["labels"] = labels; return v


def main(profile=False):
    with C.Stage("REPORT", "report.md / run_status.json"):
        man = _j("manifest.json", {}); d00 = _j("d00_smoke.json", {}); d01 = _j("d01_stats.json", {}); d02 = _j("d02_stats.json", {}); d03 = _j("d03_summary.json", {}); gate = _j("routing_gate.json", None); rm = _j("run_metrics.json", []); pr = _csv("paired_results.csv"); qs = _csv("quadrant_summary.csv"); mu = _csv("micro_update_utility.csv")
        v = verdicts(d01, d02, d03, pr, gate); hp = C.host_pairing()
        if hp["seed_host_coupled"]:
            v["labels"].append("SEED_HOST_COUPLED (§8.2 host A/B 배치: seed 간 학습 호스트가 다르다 — 두 seed 의 방향 일치는 호스트 차이와 결합; 최종 재현 주장은 한 호스트 두 seed 또는 pair 복제로 보강)")
        C.dump_json(os.path.join(C.CAMP, "verdicts.json"), v)
        led = [json.loads(l) for l in open(os.path.join(C.CAMP, "time_ledger.jsonl"))] if os.path.exists(os.path.join(C.CAMP, "time_ledger.jsonl")) else []
        stages = {}
        for e in led:
            if e["status"] == "done":
                stages[e["stage"]] = e["wall_hours"]
        t0c = open(os.path.join(C.CAMP, "T0.txt")).read().strip() if os.path.exists(os.path.join(C.CAMP, "T0.txt")) else None
        status = dict(campaign=C.CAMPAIGN_ID, generated=time.strftime("%Y-%m-%dT%H:%M:%S"), host=C.SERVER, host_pairing=hp,
                      runs={r["run"]: dict(case=r["case"], seed=r["seed"], status=r["status"], trained_on=r.get("trained_on"), reused=bool(r["case"] == "B0" and r.get("finished_at") and t0c and r["finished_at"][:19] < t0c[:19])) for r in rm},
                      stages_done_hours=stages, D04=("run" if len(mu) else "not_run"), B2_B3=("NOT_OPENED_NO_INCREMENTAL_EVIDENCE" if not (gate and gate.get("passed")) else "CANDIDATE (not executed)"), CMASS="not_run", B1_NOOFF="not_run", labels=v["labels"])
        C.dump_json(os.path.join(C.CAMP, "run_status.json"), status)
        L = [f"# DCR12 report — {C.CAMPAIGN_ID} ({C.SERVER}, {status['generated']})", "", f"계획 `{C.PLAN}` · 출력 root `{os.path.relpath(C.CAMP, C.ROOT)}/` · Teacher {man.get('teacher', {}).get('logical')} (file sha {str(man.get('teacher', {}).get('file_sha256'))[:16]}…) · τR {man.get('calibration', {}).get('tau_R')} · λE {man.get('calibration', {}).get('lambda_E')}", "",
             "- 학습 호스트: " + " · ".join(f"seed {k[1:]}: B0 {v['B0'] or '없음'} / B1 {v['B1'] or '없음'}" for k, v in hp["by_seed"].items()) + (" — pair 안 같은 호스트" if all(x for x in hp["pair_same_host"].values() if x is not None) else " — !! pair 안 호스트 불일치") + (" · seed 간 호스트 다름 (§8.2 host A/B 배치: seed×host 결합)" if hp["seed_host_coupled"] else " · 단일 호스트"), "",
             "관측 단위: CAL/DISC/CONF = train 64² patch (모든 model 이 학습에 본 자료; 분석 규칙의 개발/확인 분리) · RR20/FR20 = 논문 세트 scene · source group = 32 연속 index proxy(independence=patch_only). C = 16 probe(r 0.5/1.0) 의 offset-consistency 잔차 평균(HR px), R = plain GT L1(정규화 단위).", "",
             "## 1. 네 cell 의 개수·특성과 C 의 변동성", ""]
        t0 = d01.get("thresholds", {}).get("T0", {}); L.append(f"- T0 threshold (CAL median): tC {t0.get('tC', float('nan')):.4f} px · tR {t0.get('tR', float('nan')):.5f} · C IQR {t0.get('C_iqr', float('nan')):.4f} (수치 floor {t0.get('numerical_floor_C')}) · DEGENERATE_METRIC = {t0.get('degenerate_metric')} · C 상대 IQR {t0.get('C_cv', float('nan')):.3f}")
        if len(qs):
            L.append(""); L.append("| pipeline | panel | cell | n | groups | small | R med | edge med | C med | |c| med | texture med |"); L.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
            for _, r in qs[qs.panel.isin(["DISC", "CAL"])].iterrows():
                L.append(f"| {r.pipeline} | {r.panel} | {r.cell} ({r.cell_desc}) | {r.n} | {r.n_groups} | {'SMALL' if r.small_cell else ''} | {r.R_med:.5f} | {r.edge_med:.5f} | {r.C_med:.4f} | {r.c_norm_med:.3f} | {r.texture_med:.4f} |")
        L += ["", "## 2. C 는 native R/HQNR 과 관련됐는가, probe 강건성에만 관련됐는가", ""]
        for k, s in d01.items():
            if "|" in k and isinstance(s, dict) and "spearman_C_R" in s and isinstance(s["spearman_C_R"], dict):
                b = s["spearman_C_R"]; L.append(f"- {k}: Spearman(C, R) ρ {b.get('rho') if b.get('rho') is None else round(b['rho'], 3)} CI {b.get('ci95')} (n {s['n']}, groups {s['n_groups']}) · Pearson {s.get('pearson_C_R') and round(s['pearson_C_R'], 3)} · ρ(C, texture) {s['spearman_C_texture'].get('rho') and round(s['spearman_C_texture']['rho'], 3)} · ρ(C_r0.5,R) {s['spearman_C_r05_R'].get('rho') and round(s['spearman_C_r05_R']['rho'], 3)} / ρ(C_r1,R) {s['spearman_C_r10_R'].get('rho') and round(s['spearman_C_r10_R']['rho'], 3)} · confirm set ρ(C,C') {s['spearman_C_Cconfirm'].get('rho') and round(s['spearman_C_Cconfirm']['rho'], 3)}")
            if k.endswith("|FR") and isinstance(s, dict):
                L.append(f"- {k} (scene): ρ(C, raw HQNR) {s['spearman_C_hqnr'].get('rho') and round(s['spearman_C_hqnr']['rho'], 3)} p {s['spearman_C_hqnr'].get('p') and round(s['spearman_C_hqnr']['p'], 4)} · ρ(C, Dλ) {s['spearman_C_dlambda'].get('rho') and round(s['spearman_C_dlambda']['rho'], 3)} · ρ(C, Ds) {s['spearman_C_ds'].get('rho') and round(s['spearman_C_ds']['rho'], 3)} · HQNR 평균 {s['hqnr_mean']:.5f}")
        L += ["", "## 3. 같은 F 의 learned correction 이 zero 보다 유리했는가 · C 가 그 이득을 설명했는가", ""]
        for k, s in d02.items():
            if "|native64" in k:
                L.append(f"- {k}: B_R = R(0) − R(c0) = {fmt_ci(s['B_R_zero_minus_learned'])} · 양인 sample {s['gain_pos_frac']:.3f} · wrong(−c0)−learned {fmt_ci(s['wrong_minus_learned'])} · ±0.25 px bias ΔR 평균 {s['bias_R_delta']['mean']:+.5f} (최선 {s['bias_R_delta']['best']:+.5f} / 최악 {s['bias_R_delta']['worst']:+.5f}; 어느 bias 라도 나아지는 sample {s['bias_R_delta']['frac_any_bias_better']:.2f})" + (f" · cell 별 B_R {({c: round(x['mean'], 5) for c, x in s['B_R_by_cell_T0'].items()})}" if s.get("B_R_by_cell_T0") else ""))
            if "|fr512" in k:
                L.append(f"- {k}: B_Q = HQNR(c0) − HQNR(0) = {fmt_ci(s['B_Q_learned_minus_zero'])} · HQNR I0 {s['hqnr_I0']:.5f} / IZ {s['hqnr_IZ']:.5f} / IN {s['hqnr_IN']:.5f}" + (f" · FR8 ±bias ΔHQNR 평균 {s['bias_hqnr_delta']['mean']:+.5f} (partial)" if s.get("bias_hqnr_delta") else ""))
            if "|rr256" in k:
                L.append(f"- {k}: B_R(RR full) {fmt_ci(s['B_R_zero_minus_learned'])}")
        for k, s in (d02.get("closure_invariance") or {}).items():
            L.append(f"- closure 불변 대조 {k}: 같은 b 를 두 correction 에 더하면 C 변화 최대 {s['C_abs_diff_max']:.2e} (수치오차), R 변화 평균 |ΔR| {s['R_diff_abs_mean']:.5f} → {'반례 성립(C 는 native 품질의 충분한 인증값이 아니다)' if s['counterexample'] else '반례 불성립'}")
        for k, s in (d02.get("synthetic_composition") or {}).items():
            L.append(f"- 합성 이동 {k}: R_seq {s['R_seq_mean']:.5f} vs R_comp {s['R_comp_mean']:.5f} (seq−comp {s['seq_minus_comp_mean']:+.5f}; |Z_seq−Z_comp| {s['Z_abs_diff_mean']:.5f}) — 두 번 보간 vs 한 번 보간의 차이")
        L += ["", "## 4. Task/offset gradient 충돌은 언제·어떤 cell 에서 나타났는가 (GRAD64, per-sample, φ = Student A)", ""]
        for k, s in d03.items():
            if isinstance(s, dict) and "cos_gT_gC_mean" in s:
                L.append(f"- {k} (step {s.get('step')}): cos(gT, gC) 평균 {s['cos_gT_gC_mean']:+.3f}, 음 비율 {s['cos_gT_gC_neg_frac']:.2f} · cos(g0, gC) {s['cos_g0_gC_mean']:+.3f} · cos(gT, gK) {s['cos_gT_gK_mean']:+.3f} · ‖1e-4·gC‖/‖gT‖ 중앙값 {s['ratio_wC_over_T_median']:.3g} · NA_ZERO {s['na_zero_count']} · 1차 예측 vs 실제 ΔL_task 상관 {s.get('dL_task_actual_vs_pred_corr') and round(s['dL_task_actual_vs_pred_corr'], 3)} · cell 별 {({c: round(x['cos_gT_gC_mean'], 3) for c, x in (s.get('by_cell') or {}).items()})}")
        for r in d03.get("routing_checks", []):
            L.append(f"- routing 검사 {r['pipeline']}: |Δ| {r['max_abs_err']:.2e} / ref {r['ref_max_abs']:.2e} → {'OK' if r['ok'] else 'MISMATCH'} ({r['policy']})")
        L += ["", "## 5. B1 − B0 의 HQNR · plain GT L1 · C 변화 (seed 별, 같은 checkpoint 규약)", "", "| seed | 상태 | Δ selected HQNR | Δ plateau | Δ last | Δ Dλ | Δ Ds | scene Δ mean [CI] (+n) | Δ R_plain DISC | Δ R_plain CONF | C median B0 → B1 (CAL) |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
        for p in pr.to_dict("records"):
            if p.get("status") == "paired":
                L.append(f"| {p['seed']} | paired | {p['d_selected_hqnr']:+.5f} | {p['d_plateau_mean_hqnr']:+.5f} | {p['d_last_hqnr']:+.5f} | {p['d_fr20_d_lambda']:+.5f} | {p['d_fr20_d_s']:+.5f} | {p.get('d_hqnr_scene_mean', float('nan')):+.5f} {p.get('d_hqnr_scene_ci95')} (+{p.get('d_hqnr_scene_pos')}/{p.get('n_scenes')}) | {p.get('d_R_plain_DISC', float('nan')):+.5f} | {p.get('d_R_plain_CONF', float('nan')):+.5f} | {p.get('C_median_B0_CAL', float('nan')):.4f} → {p.get('C_median_B1_CAL', float('nan')):.4f} |")
            else:
                L.append(f"| {p['seed']} | {p['status']} | | | | | | | | | |")
        L.append(""); L.append("| run | case | status | selected step | selected HQNR | numerical max (step) | plateau mean | last | Dλ | Ds | train h |"); L.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for r in rm:
            if r["status"] == "complete":
                L.append(f"| {r['run']} | {r['case']}/{r['alias']} | {r['status']} | {r['selected_step']} | {r['selected_hqnr']:.6f} | {r['numerical_max_hqnr']:.6f} ({r['numerical_max_step']}) | {r['plateau_mean_hqnr']:.6f} | {r['last_hqnr']:.6f} | {r['fr20_d_lambda']:.5f} | {r['fr20_d_s']:.5f} | {r['train_hours']} |")
            else:
                L.append(f"| {r['run']} | {r['case']}/{r['alias']} | {r['status']} | | | | | | | | |")
        L += ["", "## 6. D04 / B2 / B3 를 열었는가", ""]
        if gate:
            L.append(f"- D04 실행: gate {gate['status']} (passed {gate['passed']}); 조건: " + "; ".join(f"{k}: {v.get('ok')}" for k, v in gate.get("conditions", {}).items() if isinstance(v, dict) and "ok" in v))
            if gate.get("summary"):
                L.append("- cell 별 U = R_H − R_K (양이면 soft 유리): " + ", ".join(f"{k} {v['U_mean']:+.5f} (pos {v['pos_frac']:.2f}, n {v['n']})" for k, v in gate["summary"].items()))
            if gate.get("insufficient_cells"):
                L.append(f"- episode 를 만들 수 없던 cell: {gate['insufficient_cells']}")
        else:
            L.append("- D04 미실행 (B0 candidate checkpoint 25250/45450 없음 또는 stage 미도달)")
        L.append(f"- B2/B3: {status['B2_B3']} · CMASS: not_run · B1-NOOFF: not_run")
        L += ["", "## 7. case 표 (실행·재사용·실패·미실행)", "", "| case | run | 상태 |", "|---|---|---|"]
        for k, r in status["runs"].items():
            L.append(f"| {r['case']} ({C.CASES[r['case']]}) S{r['seed']} | {k} | {r['status']}{' (재사용: s1 PAKD50 완료본)' if r['reused'] else ''} |")
        L += ["", "## 판정 (§14.1 규칙의 자동 적용; 원인 해석 없음)", ""] + [f"- {x}" for x in v["labels"]] + ["", f"- C–R (T0 DISC) ρ {v['C_vs_native_R']['rho']} CI {v['C_vs_native_R']['ci95']}; scene ρ(C, HQNR) {v['C_vs_native_R']['scene_C_vs_hqnr_rho']} (p {v['C_vs_native_R']['scene_p']})", f"- B1−B0 selected HQNR: {v['B1_minus_B0']['d_selected_hqnr']}",
              "", "FR20 은 선택·보고에 쓰인 세트라 미사용 test 가 아니다. HQNR 상승을 물리적 registration GT 회복과 동일시하지 않는다. scene bootstrap CI 만으로 학습 seed 일반성을 주장하지 않는다."]
        open(os.path.join(C.CAMP, "report.md"), "w").write("\n".join(L) + "\n"); print("[report] labels:", v["labels"], flush=True)
