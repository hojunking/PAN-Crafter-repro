#!/usr/bin/env python
"""PALS24 λ* 선택 — seed 1234 탐색(A1–A3) 이 끝난 뒤 **한 번** 고정한다 (계획 §10.3·§11.5).

    python tools/pals24_select_lambda.py            # work_dir/_pals24_campaign/selected_lambda.json 을 쓰고 stage 2 config 6벌을 만든다 (이미 있으면 다시 고르지 않는다)
    python tools/pals24_select_lambda.py --policy continue   # 세 λ 가 모두 P2 보다 0.0031 초과 낮아도 stage 2 를 연다 (사용자 결정)

규칙: raw best HQNR(best_hqnr_meta.json = best_raw 의 raw_original, 논문 세트 20장) 최대 → 동률(1e-4 band, checkpoint selector 와 같은 값) 은 fSCC(같은 checkpoint,
      원 PAN 참조, 1e-4 band) → 그래도 동률이면 더 작은 양의 λ. aligned 점수·closure·확인 seed 결과로 바꾸지 않는다.
종료 조건(§10.3/§11.5): 세 신규 λ 가 모두 seed 1234 의 P2(재사용 NF16 P2 best_raw) 보다 raw HQNR 에서 0.0031 초과 낮으면 확인 반복(B/C) 을 열지 않고 원인 분석으로 끝낸다 —
      selected_lambda.json 에 TERMINATE 로 남긴다. 그래도 돌리려면 --policy continue (선택 λ* 는 같은 규칙으로 이미 정해져 있다)."""
import argparse, json, math, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools.gen_pals24_configs import run_name, NEW_LAMBDAS, LAMBDA, REUSED, stage2, CAMPAIGN_ID

CAMP = os.path.join(ROOT, "work_dir", "_pals24_campaign"); SEL = os.path.join(CAMP, "selected_lambda.json"); REG = os.path.join(CAMP, "reuse_registry.json")
HQNR_METHOD_MARGIN = 0.0031          # S1 §6 경험적 판정선 (seed 2σ) — raw HQNR 에만 적용, 동등성 증명 아님 (§1.7, §7.7)
TOL_HQNR = 1e-4; TOL_FSCC = 1e-4     # checkpoint selector (pa/selector.py BestSelector 기본값) 와 같은 band


def select_lambda(rows, tol_hqnr=TOL_HQNR, tol_fscc=TOL_FSCC):
    """rows: [dict(case, lam, hqnr, fscc)] (hqnr 없는 행은 제외). 반환 (best row, why dict)."""
    ok = [r for r in rows if r.get("hqnr") is not None and math.isfinite(r["hqnr"])]
    if not ok:
        return None, dict(reason="no finished candidate")
    hmax = max(r["hqnr"] for r in ok); band = [r for r in ok if r["hqnr"] >= hmax - tol_hqnr]
    if len(band) == 1:
        return band[0], dict(rule="raw best HQNR", hqnr_band=[r["case"] for r in band])
    fmax = max((r["fscc"] if r.get("fscc") is not None else -math.inf) for r in band); tie = [r for r in band if r.get("fscc") is not None and r["fscc"] >= fmax - tol_fscc] or band
    if len(tie) == 1:
        return tie[0], dict(rule="HQNR tie (1e-4) -> fSCC", hqnr_band=[r["case"] for r in band], fscc_tie=[r["case"] for r in tie])
    best = min(tie, key=lambda r: r["lam"])
    return best, dict(rule="HQNR tie -> fSCC tie -> smaller positive lambda", hqnr_band=[r["case"] for r in band], fscc_tie=[r["case"] for r in tie])


def read_run(tag):
    wd = os.path.join(ROOT, "work_dir", tag); out = dict(run=tag)
    bm = os.path.join(wd, "best_hqnr_meta.json")
    if not os.path.exists(bm):
        return dict(out, status="MISSING")
    b = json.load(open(bm)); out.update(status="DONE", best_step=int(b["step"]), hqnr=float(b["hqnr"]), fscc=float(b.get("fscc", float("nan"))), d_lambda=b.get("d_lambda"), d_s=b.get("d_s"))
    fr = os.path.join(wd, "results", "fr_mat20.json")
    if os.path.exists(fr):
        j = json.load(open(fr)); out["fr_mat20_hqnr"] = j.get("hqnr"); out["fr_mat20_checkpoint"] = j.get("checkpoint"); out["fr_mat20_vs_meta_absdiff"] = (abs(float(j["hqnr"]) - out["hqnr"]) if j.get("hqnr") is not None else None)
    ck = os.path.join(wd, "best_hqnr", "model.safetensors")
    if os.path.exists(ck):
        from kdv.teacher_assets import sha256_file
        out["best_ckpt_sha256"] = sha256_file(ck)
    lm = os.path.join(wd, "last_meta.json")
    if os.path.exists(lm):
        out["last_step"] = json.load(open(lm)).get("step")
    return out


def references():
    """P2(λ=0)·P0 의 seed 1234 참조값 — reuse_registry.json(재사용 gate) 이 있으면 그것, 없으면 NF16 run 폴더에서 직접."""
    ref = {}
    if os.path.exists(REG):
        reg = json.load(open(REG))
        for e in reg.get("entries", []):
            if e.get("role") == "reuse":
                ref[e["case"]] = dict(run=e["run"], hqnr=e["best"]["hqnr"], fscc=e["best"].get("fscc"), hqnr_grid10=(e.get("best_on_grid10") or {}).get("hqnr"), approval=e.get("approval"), source="reuse_registry.json")
    for case, run in REUSED.items():
        c = case[0]
        if c not in ref:
            r = read_run(run)
            if r["status"] == "DONE":
                ref[c] = dict(run=run, hqnr=r["hqnr"], fscc=r["fscc"], hqnr_grid10=None, approval="UNVERIFIED(no reuse_registry)", source="run folder")
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="plan", choices=("plan", "continue"), help="plan: 세 λ 모두 P2−0.0031 미만이면 TERMINATE · continue: 그래도 stage 2 를 연다")
    ap.add_argument("--force", action="store_true", help="이미 고정된 selected_lambda.json 을 다시 쓴다 (기록은 selected_lambda.prev.json 으로 남긴다)")
    ap.add_argument("--no-configs", action="store_true"); a = ap.parse_args()
    if os.path.exists(SEL) and not a.force:
        j = json.load(open(SEL)); print(f"[pals24] λ* 는 이미 고정됨: {j['selected_case']} (λ {j['selected_lambda']}) · {j['confirmation']} · {j['selected_at']}"); return 0
    rows = []
    for c in NEW_LAMBDAS:
        r = read_run(run_name(c, 1234)); r.update(case=c, lam=LAMBDA[c]); rows.append(r)
    done = [r for r in rows if r["status"] == "DONE"]
    if not done:
        print("[pals24] 탐색 run 이 하나도 끝나지 않았다 — 선택 불가"); return 2
    best, why = select_lambda(done)
    ref = references(); p2 = ref.get("L000"); p0 = ref.get("CTRLP0")
    for r in rows:
        if r["status"] == "DONE":
            r["delta_H_vs_P2"] = (r["hqnr"] - p2["hqnr"]) if p2 else None; r["delta_H_vs_P0"] = (r["hqnr"] - p0["hqnr"]) if p0 else None
            r["delta_H_vs_P0_grid10"] = (r["hqnr"] - p0["hqnr_grid10"]) if p0 and p0.get("hqnr_grid10") else None
    all_below = bool(p2) and all(r["hqnr"] < p2["hqnr"] - HQNR_METHOD_MARGIN for r in done)
    incomplete = [r["case"] for r in rows if r["status"] != "DONE"]
    if all_below and a.policy == "plan":
        conf, reason = "TERMINATE", f"세 신규 λ 가 모두 seed 1234 P2({p2['hqnr']:.5f}) 보다 raw HQNR 0.0031 초과 낮다 — §10.3/§11.5 에 따라 확인 반복(B/C) 을 열지 않고 원인 분석으로 종료. 열려면 --policy continue"
    else:
        conf, reason = "PROCEED", ("사용자 정책 continue (모두 P2−0.0031 미만이지만 진행)" if all_below else "신규 λ 중 P2−0.0031 이상인 후보가 있다 — 확인 seed 7777·2025 에서 P0·L000·λ* 대응 비교")
    s2 = [run_name(c, s) for c, s in stage2(best["case"])]
    out = dict(campaign_id=CAMPAIGN_ID, selected_case=best["case"], selected_lambda=LAMBDA[best["case"]], selected_run=best["run"], selected_best_step=best["best_step"], selected_best_ckpt_sha256=best.get("best_ckpt_sha256"),
               selected_hqnr=best["hqnr"], selected_fscc=best["fscc"], rule="raw best HQNR (best_raw, raw_original, mat20) -> fSCC (1e-4 band) -> smaller positive lambda; fixed once, never reselected after confirmation seeds",
               why=why, tol_hqnr=TOL_HQNR, tol_fscc=TOL_FSCC, method_margin=HQNR_METHOD_MARGIN, reference_P2=p2, reference_P0=p0, screen=rows, incomplete_screen=incomplete,
               all_new_lambdas_below_P2_margin=all_below, confirmation=conf, confirmation_reason=reason, policy=a.policy, stage2_runs=s2,
               selected_at=time.strftime("%Y-%m-%dT%H:%M:%S"), selection_source="best_hqnr_meta.json (training-time selector value; fr_mat20.json cross-check recorded per run)")
    os.makedirs(CAMP, exist_ok=True)
    if os.path.exists(SEL):
        os.replace(SEL, os.path.join(CAMP, "selected_lambda.prev.json"))
    json.dump(out, open(SEL, "w"), indent=1, ensure_ascii=False)
    print(f"[pals24] λ* = {best['case']} (λ {LAMBDA[best['case']]}) · HQNR {best['hqnr']:.5f} fSCC {best['fscc']:.4f} @step {best['best_step']} · {why['rule']}")
    for r in rows:
        print(f"   {r['case']:5s} λ {r['lam']:<7g} " + (f"HQNR {r['hqnr']:.5f} fSCC {r['fscc']:.4f} step {r['best_step']} ΔP2 {r['delta_H_vs_P2']:+.5f}" + (f" ΔP0 {r['delta_H_vs_P0']:+.5f}" if r.get('delta_H_vs_P0') is not None else "") if r["status"] == "DONE" else r["status"]))
    print(f"[pals24] confirmation: {conf} — {reason}")
    if conf == "PROCEED" and not a.no_configs:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_pals24_configs.py"), "--stage2", "--lambda-star", best["case"]], cwd=ROOT, capture_output=True, text=True)
        print(r.stdout.strip()); 
        if r.returncode != 0:
            print(r.stderr); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
