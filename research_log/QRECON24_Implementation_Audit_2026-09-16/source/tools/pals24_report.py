#!/usr/bin/env python
"""PALS24 결과 집계 (계획 §11.7·§13) — 표 A(공식 품질) / 표 B(같은 50K last 의 학습 동작) / 표 C(효과 귀속·구현 검증) + paired_seed_results.csv + campaign_budget_ledger.json.

    python tools/pals24_report.py            # work_dir/_pals24_campaign/{table_A_quality.csv, table_B_last50k.csv, table_C_attribution.csv, paired_seed_results.csv, campaign_budget_ledger.json, report.md}

읽기 전용. 부분 결과에서도 돈다(없는 값은 빈 칸). 판정은 raw_original best HQNR → fSCC, 판정선 0.0031 (HQNR 에만). 학습 로그의 RR(py) 과 export MAT 평가의 RR 을 한 열에 섞지 않는다 — 여기 RR 열은 전부 학습 로그(checkpoint_metrics.csv, `_py`)."""
import csv, glob, json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools.gen_pals24_configs import LAMBDA, REUSED, BACKGROUND, CAMPAIGN_ID, TOTAL_HOURS, RESERVE_HOURS, RUN_RESERVED_HOURS, LEDGER, NEW_LAMBDAS, SCREEN_SEED, CONFIRM_SEEDS

CAMP = os.path.join(ROOT, "work_dir", "_pals24_campaign"); METHOD_MARGIN = 0.0031
TABLE_A = ["case", "seed", "lambda", "run", "stage", "selected_update", "checkpoint_sha16", "HQNR_raw_original", "fSCC_raw_original", "D_lambda", "D_s", "HQNR_raw_v64", "RR_ERGAS_py", "RR_SAM_py", "RR_SCC_py", "delta_H_vs_P2", "delta_H_vs_P0", "source"]
TABLE_B = ["case", "seed", "lambda", "exact_update", "raw_original", "raw_v64", "aligned_self_v64", "aligned_fixedN2_v64", "RR_ERGAS_py", "RR_SAM_py", "RR_SCC_py", "native_delta_dy_median", "native_delta_dx_median", "native_delta_norm_median",
           "B_resp_yy", "B_resp_yx", "B_resp_xy", "B_resp_xx", "B_intercept_y", "B_intercept_x", "offset_mae_component", "offset_epe", "legacy_closure_mean", "drift_vs_donor_median", "response_probe_set", "response_scale"]
TABLE_C = ["case", "seed", "lambda", "diag_step", "grad_rec_A_norm", "grad_off_A_norm", "weighted_grad_off_A_norm", "rho_g", "cos_psi", "raw_Loff", "weighted_Loff", "off_unet_grad_absent", "support_n_eligible_at_best", "init_sha16", "donor_aligner_sha16", "config_sha16",
           "metric_reproduction_abs_err", "ms_swap_B_diag", "ms_const_B_diag", "kernel_bilinear_B_diag"]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rows_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def parse_name(tag):
    m = re.match(r"PALS24_([A-Z0-9]+)_W112_D123_WV3_S(\d+)_", tag)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def runs():
    """(case, seed) → dict(run, stage, source). 재사용(registry 승인) + PALS24_* 폴더."""
    out = {}
    reg = jload(os.path.join(CAMP, "reuse_registry.json"), {})
    for e in reg.get("entries", []):
        if e.get("role") == "reuse":
            out[(e["case"], e["seed"])] = dict(run=e["run"], stage="screen(reused NF16)", source=f"reuse:{e['approval']}", registry=e)
    for (c, s), r in BACKGROUND.items():
        if os.path.exists(os.path.join(ROOT, "work_dir", r, "best_hqnr_meta.json")):
            out[(c, s)] = dict(run=r, stage="background(NF16 S7777, not in the PALS24 queue)", source="background", registry=None)
    for d in sorted(glob.glob(os.path.join(ROOT, "work_dir", "PALS24_*_v1"))):                     # dry/기타 버전 폴더는 집계하지 않는다
        tag = os.path.basename(d); c, s = parse_name(tag)
        if c and os.path.isdir(d):
            out[(c, s)] = dict(run=tag, stage=("screen" if s == SCREEN_SEED else "confirmation"), source="PALS24 run", registry=None)
    return out


def best_row(wd, step, cm):
    return next((r for r in cm if int(r["step"]) == step), None)


def table_a(R):
    rows = []
    for (c, s), info in sorted(R.items(), key=lambda kv: (kv[0][1], LAMBDA.get(kv[0][0], -1))):
        wd = os.path.join(ROOT, "work_dir", info["run"]); bm = jload(os.path.join(wd, "best_hqnr_meta.json")); cm = rows_csv(os.path.join(wd, "checkpoint_metrics.csv"))
        if not bm:
            rows.append(dict(case=c, seed=s, lambda_=LAMBDA.get(c), run=info["run"], stage=info["stage"], source="NOT FINISHED")); continue
        b = best_row(wd, int(bm["step"]), cm) or {}; mx = rows_csv(os.path.join(wd, "metrics.csv"))
        sha = None
        try:
            from kdv.teacher_assets import sha256_file
            sha = sha256_file(os.path.join(wd, "best_hqnr", "model.safetensors"))[:16]
        except Exception:
            pass
        rows.append(dict(case=c, seed=s, lambda_=LAMBDA.get(c), run=info["run"], stage=info["stage"], selected_update=int(bm["step"]), checkpoint_sha16=sha, HQNR_raw_original=float(bm["hqnr"]), fSCC_raw_original=fl(bm.get("fscc")),
                         D_lambda=fl(b.get("raw_original.d_lambda")), D_s=fl(b.get("raw_original.d_s")), HQNR_raw_v64=fl(b.get("raw_valid.hqnr")), RR_ERGAS_py=fl(b.get("rr_ergas")), RR_SAM_py=fl(b.get("rr_sam")), RR_SCC_py=fl(b.get("rr_scc")), source=info["source"]))
    by = {(r["case"], r["seed"]): r for r in rows}
    for r in rows:
        p2 = by.get(("L000", r["seed"])); p0 = by.get(("CTRLP0", r["seed"]))
        r["delta_H_vs_P2"] = (r["HQNR_raw_original"] - p2["HQNR_raw_original"]) if p2 and r.get("HQNR_raw_original") is not None and p2.get("HQNR_raw_original") is not None else None
        r["delta_H_vs_P0"] = (r["HQNR_raw_original"] - p0["HQNR_raw_original"]) if p0 and r.get("HQNR_raw_original") is not None and p0.get("HQNR_raw_original") is not None else None
        if r["case"] == "CTRLP0" and r["seed"] == SCREEN_SEED and by.get(("CTRLP0", SCREEN_SEED), {}).get("source", "").startswith("reuse"):
            g = ((R[("CTRLP0", SCREEN_SEED)].get("registry") or {}).get("best_on_grid10") or {})
            if g.get("hqnr"):
                r["source"] += f"; grid-10 re-selection {g['hqnr']:.5f}@ep{g['epoch']} (eval_epoch 5 run)"
    return rows


def table_b(R):
    rows = []
    for (c, s), info in sorted(R.items(), key=lambda kv: (kv[0][1], LAMBDA.get(kv[0][0], -1))):
        wd = os.path.join(ROOT, "work_dir", info["run"]); cm = rows_csv(os.path.join(wd, "checkpoint_metrics.csv")); lm = jload(os.path.join(wd, "last_meta.json"), {})
        if not cm or not lm:
            continue
        l = cm[-1]; d = jload(os.path.join(wd, "results", "po10_diag_last_pals24.json")) or jload(os.path.join(wd, "results", "po10_diag_last.json")) or {}
        rs = (d.get("response") or {}).get("fr512") or {}; B = rs.get("B") or [[None, None], [None, None]]; b = rs.get("b") or [None, None]
        rows.append(dict(case=c, seed=s, lambda_=LAMBDA.get(c), exact_update=lm.get("step"), raw_original=fl(l.get("raw_original.hqnr")), raw_v64=fl(l.get("raw_valid.hqnr")), aligned_self_v64=fl(l.get("aligned_valid.hqnr")), aligned_fixedN2_v64=fl(l.get("aligned_fixed_v64.hqnr")),
                         RR_ERGAS_py=fl(l.get("rr_ergas")), RR_SAM_py=fl(l.get("rr_sam")), RR_SCC_py=fl(l.get("rr_scc")), native_delta_dy_median=fl(l.get("dy_median")), native_delta_dx_median=fl(l.get("dx_median")), native_delta_norm_median=fl(l.get("delta_norm_median")),
                         B_resp_yy=B[0][0], B_resp_yx=B[0][1], B_resp_xy=B[1][0], B_resp_xx=B[1][1], B_intercept_y=b[0], B_intercept_x=b[1], offset_mae_component=rs.get("offset_mae_component"), offset_epe=rs.get("offset_epe"),
                         legacy_closure_mean=rs.get("closure_mean"), drift_vs_donor_median=(rs.get("drift_vs_reference") or {}).get("drift_norm_median"), response_probe_set=d.get("probe_set", "po10(legacy)" if d else ""), response_scale="fr512"))
    return rows


def table_c(R, reg):
    rows = []; regmap = {e["run"]: e for e in (reg or {}).get("entries", [])}
    for (c, s), info in sorted(R.items(), key=lambda kv: (kv[0][1], LAMBDA.get(kv[0][0], -1))):
        wd = os.path.join(ROOT, "work_dir", info["run"]); gd = [json.loads(x) for x in open(os.path.join(wd, "gradient_diagnostics.jsonl"))] if os.path.exists(os.path.join(wd, "gradient_diagnostics.jsonl")) else []
        h = jload(os.path.join(wd, "init_and_teacher_hashes.json"), {}); d = jload(os.path.join(wd, "results", "po10_diag_last_pals24.json")) or {}; d_legacy = jload(os.path.join(wd, "results", "po10_diag_last.json")) or {}
        ic = d.get("interpolation_controls") or d_legacy.get("interpolation_controls") or {}          # 재사용 NF16 run 은 legacy(po10 probe) 파일에만 보간/MS 대조가 있다
        e = regmap.get(info["run"], {}); g1 = (e.get("checks") or {}).get("G-M1_reproduction") or {}
        sc = rows_csv(os.path.join(wd, "scene_metrics.csv")); bm = jload(os.path.join(wd, "best_hqnr_meta.json"), {})
        n_el = sum(1 for r in sc if bm and int(r["step"]) == int(bm["step"]) and r["view"] == "raw_original" and r.get("selection_eligible") == "True")
        cfg_sha = None
        try:
            from kdv.teacher_assets import sha256_file
            cfg_sha = sha256_file(os.path.join(wd, "meta", "config.yaml"))[:16]
        except Exception:
            pass
        lam = LAMBDA.get(c) or 0.0; targets = (10001, 25001, 49001)
        odd = [x for x in gd if x.get("grad_off_A") is not None]
        for t in targets:
            x = min(odd, key=lambda z: abs(z["step"] - t)) if odd else None
            if x is None or abs(x["step"] - t) > 1000:
                x = dict(step=t)
            rows.append(dict(case=c, seed=s, lambda_=LAMBDA.get(c), diag_step=x.get("step"), grad_rec_A_norm=x.get("grad_rec_A"), grad_off_A_norm=x.get("grad_off_A"),
                             weighted_grad_off_A_norm=x.get("grad_off_A_weighted", (lam * x["grad_off_A"]) if x.get("grad_off_A") is not None else None), rho_g=x.get("rho_g"), cos_psi=x.get("cos_psi", "n/a(pre-PALS24 diag)" if x.get("grad_off_A") is not None and "cos_psi" not in x else None),
                             raw_Loff=x.get("loss_off_raw"), weighted_Loff=x.get("loss_off_weighted"), off_unet_grad_absent=x.get("off_unet_grad_absent"), support_n_eligible_at_best=n_el,
                             init_sha16=(h.get("init_hashes") or {}).get("unet_init_sha256_16"), donor_aligner_sha16=(h.get("donor") or {}).get("aligner_tensors_sha256_16"), config_sha16=cfg_sha,
                             metric_reproduction_abs_err=g1.get("abs_diff_vs_meta"), ms_swap_B_diag=(ic.get("ms_swap") or {}).get("B_diag"), ms_const_B_diag=(ic.get("ms_const") or {}).get("B_diag"), kernel_bilinear_B_diag=(ic.get("kernel_bilinear") or {}).get("B_diag")))
    return rows


def paired(A, sel):
    ls = (sel or {}).get("selected_case"); rows = []
    if not ls:
        return rows, {}
    by = {(r["case"], r["seed"]): r for r in A if r.get("HQNR_raw_original") is not None}
    for s in [SCREEN_SEED] + CONFIRM_SEEDS:
        a, p2, p0 = by.get((ls, s)), by.get(("L000", s)), by.get(("CTRLP0", s))
        rows.append(dict(seed=s, role=("screen/selection" if s == SCREEN_SEED else "confirmation"), lambda_star=ls, H_lambda_star=a and a["HQNR_raw_original"], H_L000=p2 and p2["HQNR_raw_original"], H_P0=p0 and p0["HQNR_raw_original"],
                         d_vs_P2=(a["HQNR_raw_original"] - p2["HQNR_raw_original"]) if a and p2 else None, d_vs_P0=(a["HQNR_raw_original"] - p0["HQNR_raw_original"]) if a and p0 else None, complete=bool(a and p2 and p0)))
    def summ(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return dict(n=len(v), mean=(float(np.mean(v)) if v else None), sd=(float(np.std(v, ddof=1)) if len(v) > 1 else None), signs=[("+" if x > 0 else "-" if x < 0 else "0") for x in v], consistent=(len(set(np.sign(v))) == 1 if v else None), beyond_margin=[abs(x) > METHOD_MARGIN for x in v])
    conf = [r for r in rows if r["role"] == "confirmation"]
    S = dict(confirmation_only=dict(d_vs_P2=summ(conf, "d_vs_P2"), d_vs_P0=summ(conf, "d_vs_P0")), all_seeds_including_screen=dict(d_vs_P2=summ(rows, "d_vs_P2"), d_vs_P0=summ(rows, "d_vs_P0")),
             note="screen seed 1234 was used for candidate selection; confirmation seeds ran the fixed lambda*; 0.0031 is the empirical line (S1 §6), not a significance level")
    return rows, S


def budget(R):
    led = jload(os.path.join(ROOT, LEDGER), {}); ent = led.get("entries", {})
    used_by = {}
    for k, e in ent.items():
        used_by[e.get("kind", "?")] = used_by.get(e.get("kind", "?"), 0.0) + float(e.get("hours_total") or e.get("hours") or 0.0)
    used = sum(used_by.values()); deferred = [k for k, e in ent.items() if e.get("status") == "DEFERRED_BUDGET"]; running = [k for k, e in ent.items() if e.get("status") == "RUNNING"]
    plan = dict(gate=2.0, screen_seed1234=3 * RUN_RESERVED_HOURS, confirm_seed7777=3 * RUN_RESERVED_HOURS, confirm_seed2025=3 * RUN_RESERVED_HOURS, final_evaluation=3.0, buffer=1.0, total=TOTAL_HOURS)
    return dict(campaign_id=CAMPAIGN_ID, ledger=LEDGER, total_gpu_hours=led.get("total_gpu_hours", TOTAL_HOURS), reserve_hours_config=RESERVE_HOURS, reserved_plan=plan, used_hours=used, used_by_kind=used_by, unused_hours=(led.get("total_gpu_hours", TOTAL_HOURS) - used),
                deferred_runs=deferred, running=running, n_entries=len(ent), gpu_count=1, server="s1", donor_fixed=True, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                note="GPU hours only (training + evaluation/diagnostics on GPU); CPU prep and human wait time are wall-clock, not in this ledger")


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (r.get("lambda_") if k == "lambda" else r.get(k)) for k in cols})


def md(cols, rows, fmt=".5f"):
    def f(v):
        return "" if v is None else (format(v, fmt) if isinstance(v, float) else str(v))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r.get("lambda_") if c == "lambda" else r.get(c)) for c in cols) + " |" for r in rows)


def main():
    os.makedirs(CAMP, exist_ok=True); R = runs(); reg = jload(os.path.join(CAMP, "reuse_registry.json"), {}); sel = jload(os.path.join(CAMP, "selected_lambda.json"), {})
    A = table_a(R); B = table_b(R); C = table_c(R, reg); P, S = paired(A, sel); bud = budget(R)
    write_csv(os.path.join(CAMP, "table_A_quality.csv"), TABLE_A, A); write_csv(os.path.join(CAMP, "table_B_last50k.csv"), TABLE_B, B); write_csv(os.path.join(CAMP, "table_C_attribution.csv"), TABLE_C, C)
    write_csv(os.path.join(CAMP, "paired_seed_results.csv"), ["seed", "role", "lambda_star", "H_lambda_star", "H_L000", "H_P0", "d_vs_P2", "d_vs_P0", "complete"], P)
    json.dump(dict(budget=bud, paired_summary=S), open(os.path.join(CAMP, "campaign_budget_ledger.json"), "w"), indent=1, ensure_ascii=False)
    ca = ["case", "seed", "lambda", "stage", "selected_update", "HQNR_raw_original", "fSCC_raw_original", "D_lambda", "D_s", "RR_ERGAS_py", "delta_H_vs_P2", "delta_H_vs_P0"]
    cb = ["case", "seed", "exact_update", "raw_original", "raw_v64", "aligned_self_v64", "aligned_fixedN2_v64", "native_delta_norm_median", "B_resp_yy", "B_resp_xx", "offset_mae_component", "offset_epe", "legacy_closure_mean", "response_probe_set"]
    cc = ["case", "seed", "diag_step", "grad_rec_A_norm", "grad_off_A_norm", "weighted_grad_off_A_norm", "rho_g", "cos_psi", "raw_Loff", "off_unet_grad_absent", "metric_reproduction_abs_err"]
    txt = (f"# PALS24 집계 ({time.strftime('%Y-%m-%d %H:%M')}) — 판정 raw best HQNR → fSCC, 판정선 {METHOD_MARGIN} (HQNR 에만)\n\n약명→세팅: CTRLP0 = aligner 없음(NF16 P0 정의) · L000 = λ 0(NF16 P2 정의) · L1E4/L1E3/L3E3 = λ 1e-4/1e-3/3e-3 · L1E2 = λ 0.01(NF16 P3 정의)\n\n"
           f"λ*: {sel.get('selected_case')} (λ {sel.get('selected_lambda')}) · confirmation {sel.get('confirmation')}\n\n## 표 A — 공식 품질 (best_raw, raw_original, mat20)\n\n{md(ca, A)}\n\n## 표 B — 같은 50K last 의 학습 동작 (진단; 주 판정 아님)\n\n{md(cb, B, '.4f')}\n\n"
           f"## 표 C — 효과 귀속·구현 검증 (홀수 진단 step)\n\n{md(cc, C, '.4g')}\n\n## seed 반복 (λ* − L000, λ* − P0)\n\n{md(['seed', 'role', 'H_lambda_star', 'H_L000', 'H_P0', 'd_vs_P2', 'd_vs_P0', 'complete'], P)}\n\n"
           f"확인 seed 만: {json.dumps(S.get('confirmation_only'), ensure_ascii=False)}\n\n## 예산\n\n사용 {bud['used_hours']:.2f} / {bud['total_gpu_hours']} GPU-h (kind 별 {json.dumps({k: round(v, 2) for k, v in bud['used_by_kind'].items()})}) · 보류 {bud['deferred_runs']} · 미사용 {bud['unused_hours']:.2f}\n")
    open(os.path.join(CAMP, "report.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
