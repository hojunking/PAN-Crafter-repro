#!/usr/bin/env python
"""PALSV18 집계 (계획 §12·§14) — 네 표(A 공식 성능 / B 같은 checkpoint 의 정합 능력 / C 위치·선명도·보간 / D 실행·한계) + §14.1 csv 목록 + final_report.md.

    python tools/palsv18_report.py        # work_dir/_palsv18_campaign/{official_best_raw_results.csv, matched_grid_best_raw_results.csv, paired_seed_differences.csv, response_signed_2x2.csv, offset_component_mae_epe.csv,
                                          #   native_shift_proxy_before_after.csv, shortcut_controls.csv, correction_interventions.csv, synthetic_stress_native_reference.csv, edge_and_frequency_metrics.csv, checkpoint_manifest.csv,
                                          #   campaign_budget_ledger.json, alignment_evidence_summary.md, final_report.md}
읽기 전용·부분 결과 가능. 판정 규약: best_raw raw_original HQNR → fSCC, 판정선 0.0031(HQNR 에만). performance_leader(3-seed 평균 수치 선두) 와 working_reference(L1E4) 와 alignment_evidence 를 분리한다 (§12.1)."""
import csv, glob, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools import gen_pals24_configs as G, gen_palsv18_configs as V
from tools.pals24_report import rows_csv, jload, fl, write_csv, md

CAMP = os.path.join(ROOT, "work_dir", "_palsv18_campaign"); MARGIN = 0.0031; SEEDS = V.SEEDS; CASES = ["CTRLP0", "L000", "L3E5", "L1E4", "L3E4"]


def runs():
    R = {}
    for (c, s), r in V.REUSED.items():
        R[(c, s)] = dict(run=r, source="reuse")
    for c, s in V.CELLS:
        r = V.run_name(c, s)
        if os.path.isdir(os.path.join(ROOT, "work_dir", r)):
            R[(c, s)] = dict(run=r, source="new")
    for (c, s), r in V.DIAG_REFERENCE.items():
        R[(c, s)] = dict(run=r, source="diagnostic_reference")
    return R


def matched_grid_best(run, grid_steps, tol_h=1e-4, tol_f=1e-4):
    """공통 격자(selection_grid.json 의 실제 optimizer update 목록, exact 50K 포함) 위에서 **실제 selector 규칙**(pa/selector.BestSelector: running-max HQNR 1e-4 band → fSCC 1e-4 band → 늦은 update) 을 재생한다.
    checkpoint_metrics.csv 의 raw_original 만 쓴다(raw selector 는 항상 eligible). 반환 dict(step, hqnr, fscc, n_candidates, n_grid) 또는 None."""
    from pa.selector import BestSelector
    cm = rows_csv(os.path.join(ROOT, "work_dir", run, "checkpoint_metrics.csv")); g = set(int(x) for x in grid_steps)
    rows = [r for r in cm if int(r["step"]) in g and r.get("raw_original.hqnr")]
    if not rows:
        return None
    sel = BestSelector("raw", tol_hqnr=tol_h, tol_fscc=tol_f)
    for r in rows:
        sel.update(int(r["step"]), int(r["epoch"]), float(r["raw_original.hqnr"]), float(r["raw_original.fscc"]), True, f"step-{r['step']}")
    b = sel.best; return dict(step=b["step"], hqnr=b["hqnr"], fscc=b["fscc"], n_candidates=len(rows), n_grid=len(g), complete=(len(rows) == len(g)))


def resumed_flag(wd):
    j = jload(os.path.join(wd, "kdv_config_resolved.json"), {}); ev = os.path.join(wd, "resume_events.jsonl")
    n = sum(1 for _ in open(ev)) if os.path.exists(ev) else 0
    return dict(resumed=bool(j.get("resumed")) or n > 0, resumed_from=j.get("resumed_from"), n_resume_events=n, exact=False if (j.get("resumed") or n) else True)


def ckpt_sha(wd, ck):
    try:
        from kdv.teacher_assets import sha256_file
        return sha256_file(os.path.join(wd, ck, "model.safetensors"))[:16]
    except Exception:
        return None


def table_a(R):
    grid = jload(os.path.join(CAMP, "selection_grid.json"), {}); gsteps = grid.get("optimizer_updates") or []; rows = []
    for (c, s), info in sorted(R.items(), key=lambda kv: (kv[0][1], G.LAMBDA.get(kv[0][0], -1) if kv[0][0] in G.LAMBDA else -2)):
        if info["source"] == "diagnostic_reference":
            continue
        wd = os.path.join(ROOT, "work_dir", info["run"]); bm = jload(os.path.join(wd, "best_hqnr_meta.json")); cm = rows_csv(os.path.join(wd, "checkpoint_metrics.csv")); lm = jload(os.path.join(wd, "last_meta.json"), {})
        if not bm:
            rows.append(dict(case=c, seed=s, lambda_=G.LAMBDA.get(c), run=info["run"], source=info["source"], status="NOT FINISHED")); continue
        b = next((r for r in cm if int(r["step"]) == int(bm["step"])), {}); mg_ = matched_grid_best(info["run"], gsteps) if gsteps else None; rf = resumed_flag(wd)
        rows.append(dict(case=c, seed=s, lambda_=G.LAMBDA.get(c), run=info["run"], source=info["source"], status=("DONE" if lm.get("step") == 50000 else "RUNNING(partial best — not a result)"), selected_update=int(bm["step"]),
                         matched_grid_update=(mg_ or {}).get("step"), matched_grid_HQNR=(mg_ or {}).get("hqnr"), matched_grid_fSCC=(mg_ or {}).get("fscc"), matched_grid_n=(mg_ or {}).get("n_candidates"), matched_grid_complete=(mg_ or {}).get("complete"),
                         resumed_nonexact=rf["resumed"], n_resume_events=rf["n_resume_events"], checkpoint_sha16=ckpt_sha(wd, "best_hqnr"), last_update=lm.get("step"),
                         HQNR_raw_original=float(bm["hqnr"]), fSCC_raw_original=fl(bm.get("fscc")), D_lambda=fl(b.get("raw_original.d_lambda")), D_s=fl(b.get("raw_original.d_s")), HQNR_raw_v64=fl(b.get("raw_valid.hqnr")), fSCC_raw_v64=fl(b.get("raw_valid.fscc")),
                         RR_ERGAS_py=fl(b.get("rr_ergas")), RR_SAM_py=fl(b.get("rr_sam")), RR_SCC_py=fl(b.get("rr_scc"))))
    by = {(r["case"], r["seed"]): r for r in rows if r.get("status") == "DONE"}
    for r in rows:
        if r.get("status") != "DONE":
            continue
        for ref in ("CTRLP0", "L000", "L1E4"):
            q = by.get((ref, r["seed"])); r[f"delta_H_vs_{ref}_original"] = (r["HQNR_raw_original"] - q["HQNR_raw_original"]) if q else None
            r[f"delta_H_vs_{ref}"] = (r["matched_grid_HQNR"] - q["matched_grid_HQNR"]) if q and r.get("matched_grid_HQNR") is not None and q.get("matched_grid_HQNR") is not None else None   # 주 대응 차이 = matched grid (§6.3)
    return rows


def paired(A):
    by = {(r["case"], r["seed"]): r for r in A if r.get("status") == "DONE" and r.get("matched_grid_HQNR") is not None}; rows = []; summ = {}
    K = "matched_grid_HQNR"                                                   # 대응 차이·leader 는 공통 격자 값 (리뷰 P1-2); original 값은 표 A 에 병기
    for c in ("L3E5", "L1E4", "L3E4"):
        for s in SEEDS:
            a = by.get((c, s))
            rows.append(dict(case=c, seed=s, lambda_=G.LAMBDA[c], H=a and a[K], H_original=a and a["HQNR_raw_original"], H_L1E4=(by.get(("L1E4", s)) or {}).get(K), H_L000=(by.get(("L000", s)) or {}).get(K), H_P0=(by.get(("CTRLP0", s)) or {}).get(K), resumed_nonexact=a and a.get("resumed_nonexact"),
                             d_vs_L1E4=a and a.get("delta_H_vs_L1E4"), d_vs_L000=a and a.get("delta_H_vs_L000"), d_vs_P0=a and a.get("delta_H_vs_CTRLP0"), complete=bool(a and by.get(("L000", s)) and by.get(("CTRLP0", s)))))
        v = [r["H"] for r in rows if r["case"] == c and r["H"] is not None]
        summ[c] = dict(n=len(v), mean=(float(np.mean(v)) if v else None), sd=(float(np.std(v, ddof=1)) if len(v) > 1 else None),
                       **{f"d_vs_{k}": dict(values=[r[f"d_vs_{k}"] for r in rows if r["case"] == c and r.get(f"d_vs_{k}") is not None],
                                              mean=(float(np.mean([r[f"d_vs_{k}"] for r in rows if r["case"] == c and r.get(f"d_vs_{k}") is not None])) if any(r.get(f"d_vs_{k}") is not None for r in rows if r["case"] == c) else None),
                                              sign_consistent=(len({np.sign(r[f"d_vs_{k}"]) for r in rows if r["case"] == c and r.get(f"d_vs_{k}") is not None}) == 1 if any(r.get(f"d_vs_{k}") is not None for r in rows if r["case"] == c) else None),
                                              n_beyond_margin=int(sum(abs(r[f"d_vs_{k}"]) > MARGIN for r in rows if r["case"] == c and r.get(f"d_vs_{k}") is not None))) for k in ("L1E4", "L000", "P0")})
    done = {c: v for c, v in summ.items() if v["n"] == 3}
    leader = max(done, key=lambda c: done[c]["mean"]) if done else None
    summ["_decision"] = dict(performance_leader=leader, working_reference="L1E4", n_seeds_required=3, margin=MARGIN, basis="matched-grid best_raw HQNR (selection_grid.json, BestSelector replay)", resumed_runs=[r["run"] for r in A if r.get("resumed_nonexact")],
                             leader_vs_L1E4_mean=(done[leader]["mean"] - done["L1E4"]["mean"]) if leader and "L1E4" in done else None,
                             note="performance_leader = numerically highest 3-seed mean of best_raw raw HQNR on the same selection grid; within 0.0031 it does not replace the working reference (§12.2); alignment evidence is a separate claim (§12.3)")
    return rows, summ


def diag_rows(R):
    """B/C 표 원자료: run·ckpt 별 po10_diag_<ck>_palsv18.json(V1) + results/palsv18_<ck>.json(V2–V4)."""
    B, C, resp2x2, epe, prox, short, interv, stress, edge, manifest = [], [], [], [], [], [], [], [], [], []
    for (c, s), info in sorted(R.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        wd = os.path.join(ROOT, "work_dir", info["run"])
        for ck in ("best_hqnr", "last"):
            meta = jload(os.path.join(wd, f"{ck}_meta.json"), {}); v1 = jload(os.path.join(wd, "results", f"po10_diag_{ck}_palsv18.json")) or {}; vv = jload(os.path.join(wd, "results", f"palsv18_{ck}.json")) or {}
            if not meta and not os.path.exists(os.path.join(wd, ck, "model.safetensors")):
                continue
            manifest.append(dict(run=info["run"], case=c, seed=s, role=info["source"], checkpoint_kind=("best_raw" if ck == "best_hqnr" else "exact50k_last"), optimizer_update=meta.get("step"), checkpoint_sha16=ckpt_sha(wd, ck),
                                 v1_done=bool(v1), v234_done=bool(vv), v234_parts=",".join(vv.get("parts", []))))
            fr = (v1.get("response") or {}).get("fr512") or {}; sl = fr.get("scene_level") or {}; cm = rows_csv(os.path.join(wd, "checkpoint_metrics.csv")); row = next((r for r in cm if meta.get("step") is not None and int(r["step"]) == int(meta["step"])), {})
            v2 = (vv.get("v2") or {}).get("fr512") or {}
            B.append(dict(case=c, seed=s, ckpt=ck, update=meta.get("step"), sha16=manifest[-1]["checkpoint_sha16"], raw_original=fl(row.get("raw_original.hqnr")), aligned_self_v64=fl(row.get("aligned_valid.hqnr")), aligned_fixedN2_v64=fl(row.get("aligned_fixed_v64.hqnr")),
                          native_dy_median=fl(row.get("dy_median")), native_dx_median=fl(row.get("dx_median")), native_norm_median=fl(row.get("delta_norm_median")),
                          B_yy=(sl.get("B_mean") or [[None, None]] * 2)[0][0], B_yx=(sl.get("B_mean") or [[None, None]] * 2)[0][1], B_xy=(sl.get("B_mean") or [[None, None]] * 2)[1][0], B_xx=(sl.get("B_mean") or [[None, None]] * 2)[1][1],
                          sv1=(sl.get("sv_mean") or [None, None])[0], sv2=(sl.get("sv_mean") or [None, None])[1], epe_scene_mean=sl.get("epe_mean"), epe_p90=sl.get("epe_p90"), mae_component=sl.get("mae_component_mean"), no_response_ref_epe=sl.get("no_response_reference_epe"),
                          legacy_closure_mean=fr.get("closure_mean"), drift_vs_donor_median=(fr.get("drift_vs_reference") or {}).get("drift_norm_median"),
                          v2_ms_only_B_diag=(v2.get("ms_only") or {}).get("B_diag"), v2_ms_only_epe=(v2.get("ms_only") or {}).get("epe"), v2_common_B_diag=(v2.get("common") or {}).get("B_diag"), v2_common_epe=(v2.get("common") or {}).get("epe"), v2_pan_only_epe=(v2.get("pan_only") or {}).get("epe")))
            for sc in ("native64", "rr256", "fr512"):
                r_ = (v1.get("response") or {}).get(sc) or {}; s2 = r_.get("scene_level") or {}
                if s2:
                    resp2x2.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, scale=sc, mode="pan_only", B_yy=s2["B_mean"][0][0], B_yx=s2["B_mean"][0][1], B_xy=s2["B_mean"][1][0], B_xx=s2["B_mean"][1][1], sv1=s2["sv_mean"][0], sv2=s2["sv_mean"][1], fit_rmse=s2["fit_rmse_mean"], n_scenes=s2["n_scenes"]))
                    epe.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, scale=sc, mode="pan_only", epe_mean=s2["epe_mean"], epe_p50=s2["epe_p50"], epe_p90=s2["epe_p90"], mae_component=s2["mae_component_mean"], no_response_ref=s2["no_response_reference_epe"], **{f"epe_r{k}": v for k, v in s2["epe_by_radius"].items()}))
                for mode in ("ms_only", "common"):
                    m2 = ((vv.get("v2") or {}).get(sc) or {}).get(mode)
                    if m2:
                        resp2x2.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, scale=sc, mode=mode, B_yy=None, B_yx=None, B_xy=None, B_xx=None, sv1=None, sv2=None, fit_rmse=None, n_scenes=None, B_diag=m2["B_diag"], B_cross=m2["B_cross"]))
                        epe.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, scale=sc, mode=mode, epe_mean=m2["epe"], epe_p90=m2["epe_p90"]))
            np_ = vv.get("v3_native_proxy") or {}; en = vv.get("v3_energy") or {}; ed = vv.get("v3_edge") or {}; iv = vv.get("v4_interventions") or {}; st = vv.get("v4_stress") or {}
            C.append(dict(case=c, seed=s, ckpt=ck, update=meta.get("step"), proxy_before_median=np_.get("before_mag_median"), proxy_after_median=np_.get("after_mag_median"), proxy_n_accepted=np_.get("n_both_accepted"), proxy_n_improved=np_.get("n_improved"),
                          edge_crossing_offset=ed.get("crossing_offset_mean"), edge_crossing_offset_abs=ed.get("crossing_offset_mean_abs"), edge_width_gt=ed.get("width_gt_mean"), edge_width_out=ed.get("width_out_mean"), edge_measured=ed.get("n_measured"), edge_missing=ed.get("n_missing"), edge_multiple=ed.get("n_multiple"),
                          edge_mae_dn=ed.get("edge_mae_dn"), flat_mae_dn=ed.get("flat_mae_dn"), edge_grad_mae_dn=ed.get("edge_grad_mae_dn"), raw_fscc=fl(row.get("raw_original.fscc")), aligned_self_fscc=fl(row.get("aligned_valid.fscc")),
                          energy_ratio_fr=en.get("fr_energy_ratio_mean"), blur_sigma=en.get("sigma_star"), blur_status=en.get("blur_status"),
                          **{f"iv_{k}_raw_hqnr": v["raw_hqnr"] for k, v in iv.items()}, **{f"iv_{k}_raw_fscc": v["raw_fscc"] for k, v in iv.items()},
                          stress_eps0_raw=(st.get("(+0.0,+0.0)") or {}).get("raw"), stress_mean_raw=(float(np.nanmean([x["raw"] for k, x in st.items() if k != "(+0.0,+0.0)" and x.get("raw") is not None])) if st else None)))
            if np_:
                prox.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, **{k: v for k, v in np_.items() if not isinstance(v, (dict, list))}))
            ic = (jload(os.path.join(wd, "results", "po10_diag_last_pals24.json")) or jload(os.path.join(wd, "results", "po10_diag_last.json")) or {}).get("interpolation_controls") or {}
            if ic and ck == "last":
                short.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, ms_swap_B_diag=(ic.get("ms_swap") or {}).get("B_diag"), ms_swap_closure=(ic.get("ms_swap") or {}).get("closure_mean"), ms_const_B_diag=(ic.get("ms_const") or {}).get("B_diag"), ms_const_closure=(ic.get("ms_const") or {}).get("closure_mean"),
                                  kernel_bilinear_B_diag=(ic.get("kernel_bilinear") or {}).get("B_diag"), padding_border_vs_reflection_max_abs=ic.get("padding_border_vs_reflection_max_abs_diff")))
            for k, v in iv.items():
                interv.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, intervention=k, raw_hqnr=v["raw_hqnr"], raw_fscc=v["raw_fscc"], raw_v64_hqnr=v["raw_v64_hqnr"]))
            for k, v in st.items():
                stress.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, eps=k, raw_native_hqnr=v.get("raw"), aligned_fixedN2_hqnr=v.get("fixed"), eligible_all=v.get("eligible")))
            if ed or en:
                edge.append(dict(run=info["run"], case=c, seed=s, ckpt=ck, **{f"edge_{k}": v for k, v in ed.items()}, **{f"energy_{k}": v for k, v in en.items() if not isinstance(v, list)}))
    return B, C, resp2x2, epe, prox, short, interv, stress, edge, manifest


def budget():
    led = jload(os.path.join(ROOT, V.CAMP["ledger"]), {}); ent = led.get("entries", {}); by = {}
    for k, e in ent.items():
        by[e.get("kind", "?")] = by.get(e.get("kind", "?"), 0.0) + float(e.get("hours_total") or e.get("hours") or 0.0)
    used = sum(by.values())
    return dict(campaign_id=V.CAMP["campaign_id"], cap_gpu_hours=18.0, include_previous_spent=False, reserved_plan=led.get("reserved_plan"), used_hours=used, used_by_kind=by, unused_hours=18.0 - used,
                deferred=[k for k, e in ent.items() if e.get("status") == "DEFERRED_BUDGET"], running=[k for k, e in ent.items() if e.get("status") == "RUNNING"], finished=[k for k, e in ent.items() if str(e.get("status", "")).startswith("FINISHED")],
                entries={k: dict(kind=e.get("kind"), hours=round(float(e.get("hours_total") or e.get("hours") or 0.0), 3), status=e.get("status")) for k, e in ent.items()}, generated=time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    os.makedirs(CAMP, exist_ok=True); R = runs(); A = table_a(R); P, S = paired(A); B, C, r2, ep, px, sh, iv, st, ed, mf = diag_rows(R); bud = budget(); gate = jload(os.path.join(CAMP, "metric_gate_report.json"), {})
    cols_a = ["case", "seed", "lambda", "run", "source", "status", "resumed_nonexact", "selected_update", "checkpoint_sha16", "last_update", "HQNR_raw_original", "fSCC_raw_original", "D_lambda", "D_s", "HQNR_raw_v64", "fSCC_raw_v64", "RR_ERGAS_py", "RR_SAM_py", "RR_SCC_py", "delta_H_vs_CTRLP0_original", "delta_H_vs_L000_original", "delta_H_vs_L1E4_original"]
    write_csv(os.path.join(CAMP, "official_best_raw_results.csv"), cols_a, A)
    write_csv(os.path.join(CAMP, "matched_grid_best_raw_results.csv"), ["case", "seed", "lambda", "run", "matched_grid_update", "matched_grid_HQNR", "matched_grid_fSCC", "matched_grid_n", "matched_grid_complete", "selected_update", "HQNR_raw_original", "delta_H_vs_CTRLP0", "delta_H_vs_L000", "delta_H_vs_L1E4"], A)
    write_csv(os.path.join(CAMP, "paired_seed_differences.csv"), ["case", "seed", "lambda", "H", "H_original", "H_L1E4", "H_L000", "H_P0", "d_vs_L1E4", "d_vs_L000", "d_vs_P0", "complete", "resumed_nonexact"], P)
    for name, cols, rows in (("response_signed_2x2.csv", None, r2), ("offset_component_mae_epe.csv", None, ep), ("native_shift_proxy_before_after.csv", None, px), ("shortcut_controls.csv", None, sh), ("correction_interventions.csv", None, iv),
                             ("synthetic_stress_native_reference.csv", None, st), ("edge_and_frequency_metrics.csv", None, ed), ("checkpoint_manifest.csv", None, mf), ("table_B_alignment.csv", None, B), ("table_C_position_sharpness.csv", None, C)):
        keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["empty"]; write_csv(os.path.join(CAMP, name), keys, rows)
    json.dump(dict(budget=bud, paired_summary=S, gate=dict(all_reuse_approved=gate.get("all_reuse_approved"), entries={k: v["approval"] for k, v in (gate.get("entries") or {}).items()})), open(os.path.join(CAMP, "campaign_budget_ledger.json"), "w"), indent=1, ensure_ascii=False)
    ca = ["case", "seed", "lambda", "source", "resumed_nonexact", "selected_update", "HQNR_raw_original", "fSCC_raw_original", "D_lambda", "D_s", "matched_grid_update", "matched_grid_HQNR", "delta_H_vs_CTRLP0", "delta_H_vs_L000", "delta_H_vs_L1E4"]
    cb = ["case", "seed", "ckpt", "update", "native_norm_median", "B_yy", "B_xx", "B_yx", "B_xy", "epe_scene_mean", "mae_component", "v2_ms_only_B_diag", "v2_common_epe", "drift_vs_donor_median"]
    cc = ["case", "seed", "ckpt", "proxy_before_median", "proxy_after_median", "proxy_n_improved", "edge_crossing_offset", "edge_width_gt", "edge_width_out", "edge_mae_dn", "flat_mae_dn", "raw_fscc", "energy_ratio_fr", "blur_sigma", "blur_status", "iv_learned_raw_hqnr", "iv_zero_raw_hqnr", "iv_wrong_sign_raw_hqnr", "iv_scene_shuffle_raw_hqnr", "iv_constant_calibration_raw_hqnr", "iv_blur_energy_match_raw_hqnr", "stress_mean_raw"]
    dec = S["_decision"]
    txt = (f"# PALSV18 집계 ({time.strftime('%Y-%m-%d %H:%M')}) — 판정 best_raw raw_original HQNR → fSCC, 판정선 {MARGIN} (HQNR 에만)\n\n약명→세팅: CTRLP0 = aligner 없음(NF16 P0 정의) · L000 = λ 0(NF16 P2 정의) · L3E5/L1E4/L3E4 = λ 3e-5/1e-4/3e-4 · L1E2 = λ 0.01. 대조군 P0/L000/L1E4 는 NF16/PALS24 재사용.\n\n"
           f"performance_leader: **{dec['performance_leader']}** (3-seed 평균 선두, matched-grid 값; L1E4 대비 {dec['leader_vs_L1E4_mean']}) · working_reference: L1E4 · 판정선 안이면 교체하지 않는다. 재개(non-exact) run: {dec['resumed_runs']}\n\n## 표 A — 공식 성능 (best_raw, raw_original, mat20; 대응 차이 열은 matched-grid 기준, original 값은 csv 병기)\n\n{md(ca, A)}\n\n"
           f"### seed 대응 차이 (λ − L1E4 / L000 / P0)\n\n{md(['case', 'seed', 'H', 'H_L1E4', 'H_L000', 'H_P0', 'd_vs_L1E4', 'd_vs_L000', 'd_vs_P0', 'complete'], P)}\n\n요약: {json.dumps({k: {q: (w if not isinstance(w, dict) else {a: b for a, b in w.items() if a != 'values'}) for q, w in v.items()} for k, v in S.items() if k != '_decision'}, ensure_ascii=False)}\n\n"
           f"## 표 B — 같은 checkpoint 의 정합 능력 (V1/V2; 진단)\n\n{md(cb, B, '.4f')}\n\n## 표 C — 위치·선명도·보간·개입 (V3/V4; 진단)\n\n{md(cc, C, '.4f')}\n\n"
           f"## 표 D — 실행·한계\n\n사용 {bud['used_hours']:.2f} / 18 GPU-h (kind 별 {json.dumps({k: round(v, 2) for k, v in bud['used_by_kind'].items()})}) · 완료 {len(bud['finished'])} · 보류 {bud['deferred']} · 실행 중 {bud['running']} · 미사용 {bud['unused_hours']:.2f}\n"
           f"재사용 gate: {gate.get('all_reuse_approved')} · 고정 donor(N2 last, seed 2025)·기존 FR 20장 사용 · 3-seed 는 downstream 초기화 반복(독립 test seed 아님) · native proxy 는 센서 GT 아님 · secondary 독립 추정기 not_available(census 는 같은 추정기의 gate)\n")
    open(os.path.join(CAMP, "final_report.md"), "w").write(txt); open(os.path.join(CAMP, "alignment_evidence_summary.md"), "w").write(txt[txt.index("## 표 B"):] if "## 표 B" in txt else txt); print(txt)


if __name__ == "__main__":
    main()
