#!/usr/bin/env python
"""NA104 20H 우선순위 (research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md) — 판정 함수·예산·회신 보고. 읽기 전용 (큐·프로세스를 바꾸지 않는다).

    python tools/na104_20h.py report [--server s2|s3]      # §회신: 현재 run · 차감한 완료분 · 큐 hash · core 결과(common-grid best/plateau/last, Dλ/Ds) · CF01/X02 진입 여부 · 폐기 목록 · hash · 다음 tuning 축
    python tools/na104_20h.py decide [--server …]          # 조건부 분기 판정만 (campaign_gate 의 na104_20h gate 가 같은 함수를 쓴다)

판정 축: 원본 FR 논문 세트 20장 HQNR(raw_original, best_raw), **common grid** = 비교 두 run 의 실제 평가 update 교집합 위에서 pa/selector.BestSelector 규칙(HQNR 1e-4 band → fSCC 1e-4 → 늦은 update) 을 재생.
3-seed 기준(§6): 같은 서버의 N0(Q00) 대비 seed 1234·777·2026 의 Δ 가 평균 > 0 이고 최소 2/3 양수. plateau/last 는 안정성 설명, ERGAS 는 보조. 원래 best 는 보존한다(§2)."""
import argparse, csv, glob, hashlib, json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.selector import BestSelector

CAMP = os.path.join(ROOT, "work_dir", "_na104_20h"); VIEW = "raw_original"; SEEDS = [1234, 777, 2026]; BUDGET_H = 20.0
ARCH = "W104_D122_WV3"
RUN = {("Q00", 1234): "NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1", ("Q00", 777): "NA104_Q00_W104_D122_WV3_N0_OFF_S777_v2", ("Q00", 2026): "NA104_Q00_W104_D122_WV3_N0_OFF_S2026_v2",
       ("Q36", 1234): "NA104_Q36_W104_D122_WV3_N0_GCH_S1234_v1", ("Q36", 777): "NA104_Q36_W104_D122_WV3_N0_GCH_S777_v2", ("Q36", 2026): "NA104_Q36_W104_D122_WV3_N0_GCH_S2026_v2",
       ("Q12", 1234): "NA104_Q12_W104_D122_WV3_R3_EDGEH_S1234_v1", ("Q12", 777): "NA104_Q12_W104_D122_WV3_R3_EDGEH_S777_v2", ("Q12", 2026): "NA104_Q12_W104_D122_WV3_R3_EDGEH_S2026_v2",
       ("CF01", 777): "NA104_CF01_W104_D122_WV3_N0_GCFIX_S777_v2", ("CF01", 1234): "NA104_CF01_W104_D122_WV3_N0_GCFIX_S1234_v2", ("CF01", 2026): "NA104_CF01_W104_D122_WV3_N0_GCFIX_S2026_v2",
       ("X02", 1234): "NA104_X02_W104_D122_WV3_R1_EDGEH_S1234_v2", ("X02", 777): "NA104_X02_W104_D122_WV3_R1_EDGEH_S777_v2", ("X02", 2026): "NA104_X02_W104_D122_WV3_R1_EDGEH_S2026_v2"}
P1 = [("Q36", 777), ("Q00", 2026), ("Q36", 2026), ("Q12", 2026)]                                   # §3 필수 4개 (두 서버 같음)
TEACHER = "NA104_T00_W104_D122_WV3_N0_OFF_S2025_v1"; PILOT = "NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1"
RETIRED = ["독립 R1 스윕", "GV-FIX", "TRI-A/B/C", "IV/SC 통계", "광범위 gate(CTL*)", "장기학습(LONG/CONTN0)", "CX02", "TCOPY", "PX/LX 계열", "X01/X03–X12 (이번 20h 개발 목록에서 제외; 완료분은 보존)"]
NEXT_AXES = {"Q36": "λ_C × {0.5, 1, 2} (한 축씩)", "Q12": "λ_E × {0.5, 1, 2} (한 축씩)", "CF01": "Q36 을 반복해서 넘을 때만 β_C 국소 조정"}
ALIAS = {"Q00": "N0 — KD 없음·통계 없음 (독립 Student baseline)", "Q36": "N0 + GC-H — GT covariance 통계(hard, window 5), Teacher 는 평가 전용", "Q12": "R3 + EDGE-H — GT-anchored adaptive KD(R3) + signed GT edge(hard)",
         "CF01": "N0 + GC-FIX — GT_L1 + λ_C·(mean|C_S−C_GT| + 0.1·mean|C_S−C_T|), window 5, λ_C = Q36(S1234 v1) 재사용, Student 통계만 gradient", "X02": "R1 + EDGE-H — Q12 의 output soft 제거 대조"}


def server():
    p = os.path.join(ROOT, "gspread", "server.txt"); return open(p).read().strip() if os.path.exists(p) else "?"


def evals(run):
    p = os.path.join(ROOT, "work_dir", run, "checkpoint_metrics.csv"); out = []
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        try:
            out.append(dict(step=int(r["step"]), epoch=int(r["epoch"]), hqnr=float(r[f"{VIEW}.hqnr"]), fscc=float(r.get(f"{VIEW}.fscc") or "nan"), d_lambda=float(r.get(f"{VIEW}.d_lambda") or "nan"), d_s=float(r.get(f"{VIEW}.d_s") or "nan"), rr_ergas=float(r.get("rr_ergas") or "nan")))
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda x: x["step"])


def finished(run, why=False):
    """완료 = last_meta.step 50000 · last/model.safetensors 존재 · 평가 마지막 행이 50000 이고 HQNR 유한 · best_hqnr_meta · reduced/full_best_hqnr.mat · finished_at (리뷰 P1/P2-3)."""
    wd = os.path.join(ROOT, "work_dir", run); reasons = []
    lm = os.path.join(wd, "last_meta.json")
    if not os.path.exists(lm):
        reasons.append("last_meta 없음")
    else:
        try:
            if int(json.load(open(lm)).get("step", -1)) != 50000:
                reasons.append("last step ≠ 50000")
        except Exception:
            reasons.append("last_meta 손상")
    if not os.path.exists(os.path.join(wd, "last", "model.safetensors")):
        reasons.append("last 가중치 없음")
    ev = evals(run)
    if not ev or ev[-1]["step"] != 50000 or not np.isfinite(ev[-1]["hqnr"]):
        reasons.append("exact-50K FR 평가 없음/비유한")
    if not os.path.exists(os.path.join(wd, "best_hqnr_meta.json")):
        reasons.append("best_hqnr_meta 없음")
    for f in ("reduced_best_hqnr.mat", "full_best_hqnr.mat"):
        if not os.path.exists(os.path.join(wd, "results", f)):
            reasons.append(f"{f} 없음")
    if not os.path.exists(os.path.join(wd, "meta", "finished_at.txt")):
        reasons.append("finished_at 없음")
    return (not reasons, reasons) if why else not reasons


def _hashes(run):
    wd = os.path.join(ROOT, "work_dir", run)
    h = json.load(open(os.path.join(wd, "init_and_teacher_hashes.json"))) if os.path.exists(os.path.join(wd, "init_and_teacher_hashes.json")) else {}
    ds = json.load(open(os.path.join(wd, "dataset_hashes.json"))) if os.path.exists(os.path.join(wd, "dataset_hashes.json")) else {}
    am = json.load(open(os.path.join(wd, "architecture_manifest.json"))) if os.path.exists(os.path.join(wd, "architecture_manifest.json")) else {}
    ss = json.load(open(os.path.join(wd, "selector_state_raw.json"))) if os.path.exists(os.path.join(wd, "selector_state_raw.json")) else {}
    sm = os.path.join(wd, "scene_metrics.csv"); roi = None
    if os.path.exists(sm):
        with open(sm) as fh:
            r = next(csv.DictReader(fh), None); roi = (r or {}).get("roi_hash")
    return dict(teacher=(h.get("teacher") or {}).get("tensors_sha256_16") if isinstance(h.get("teacher"), dict) else h.get("teacher"), init=(h.get("init_hashes") or {}).get("unet_init_sha256_16"), seed=(h.get("init_hashes") or {}).get("seed"),
                data=json.dumps({k: v for k, v in ds.items() if k != "computed_at"}, sort_keys=True) if ds else None, arch=json.dumps({k: am.get(k) for k in ("width", "depth", "params_m", "in_channels", "class_path") if k in am}, sort_keys=True) if am else None,
                evaluator=(ss.get("evaluator_hash") or (ss.get("extra") or {}).get("evaluator_hash")), roi=roi)


def comparable(run_a, run_b, same_seed=True):
    """같은 비교 조건인가 (리뷰 P2-3): Teacher sha · 데이터 hash · 골격 · evaluator/ROI hash 가 같고, same_seed 면 U-Net 초기 tensor sha 도 같아야 한다. 반환 (ok, mismatches)."""
    ha, hb = _hashes(run_a), _hashes(run_b); bad = []
    for k in ("data", "arch", "evaluator", "roi") + (("init", "seed") if same_seed else ()):
        if ha.get(k) is None or hb.get(k) is None:
            bad.append(f"{k}: 기록 없음")
        elif ha[k] != hb[k]:
            bad.append(f"{k}: 불일치")
    if (ha.get("teacher") or hb.get("teacher")) and ha.get("teacher") != hb.get("teacher"):
        bad.append("teacher: 불일치")          # Teacher 가 없는 run(Q00 은 eval_only 라도 싣는다) 끼리는 None==None
    return (not bad, bad)


def running(run):
    wd = os.path.join(ROOT, "work_dir", run, "meta"); return os.path.exists(os.path.join(wd, "started_at.txt")) and not os.path.exists(os.path.join(wd, "finished_at.txt"))


def replay_best(ev, steps, tol_h=1e-4, tol_f=1e-4):
    """주어진 update 집합 위에서 실제 selector 규칙 재생."""
    sel = BestSelector("raw", tol_hqnr=tol_h, tol_fscc=tol_f); g = set(steps)
    for e in ev:
        if e["step"] in g and np.isfinite(e["hqnr"]) and np.isfinite(e["fscc"]):
            sel.update(e["step"], e["epoch"], e["hqnr"], e["fscc"], True, f"step-{e['step']}")
    return sel.best


def common_grid_pair(run_a, run_b, require_comparable=True):
    """두 run 이 **둘 다 유효한**(HQNR·fSCC 유한) 평가 update 의 교집합 위에서 각각 재선택 → Δ = a − b (리뷰 P2-3: 한쪽만 유효한 시점은 선택 기회에서 뺀다). 원래 best 도 병기.
    require_comparable: Teacher/데이터/골격/evaluator/초기 tensor 가 같지 않으면 None (사유는 comparable() 로)."""
    if require_comparable and not comparable(run_a, run_b)[0]:
        return None
    ea, eb = evals(run_a), evals(run_b)
    if not ea or not eb:
        return None
    valid = lambda ev: {e["step"] for e in ev if np.isfinite(e["hqnr"]) and np.isfinite(e["fscc"])}
    common = sorted(valid(ea) & valid(eb))
    if not common:
        return None
    ba, bb = replay_best(ea, common), replay_best(eb, common)
    if ba is None or bb is None:
        return None
    oa = json.load(open(os.path.join(ROOT, "work_dir", run_a, "best_hqnr_meta.json"))) if os.path.exists(os.path.join(ROOT, "work_dir", run_a, "best_hqnr_meta.json")) else {}
    ob = json.load(open(os.path.join(ROOT, "work_dir", run_b, "best_hqnr_meta.json"))) if os.path.exists(os.path.join(ROOT, "work_dir", run_b, "best_hqnr_meta.json")) else {}
    la, lb = ea[-1], eb[-1]; pa_ = [e["hqnr"] for e in ea if 40000 <= e["step"] <= 50000]; pb = [e["hqnr"] for e in eb if 40000 <= e["step"] <= 50000]
    return dict(a=run_a, b=run_b, n_common=len(common), grid_note=("same grid" if len(common) == len(ea) == len(eb) else f"intersection(valid both) {len(common)} of {len(ea)}/{len(eb)}"),
                a_best=dict(step=ba["step"], hqnr=ba["hqnr"], fscc=ba["fscc"]), b_best=dict(step=bb["step"], hqnr=bb["hqnr"], fscc=bb["fscc"]), delta_common=ba["hqnr"] - bb["hqnr"],
                a_original=oa.get("hqnr"), b_original=ob.get("hqnr"), delta_original=((oa.get("hqnr") - ob.get("hqnr")) if oa.get("hqnr") is not None and ob.get("hqnr") is not None else None),
                a_last=la["hqnr"], b_last=lb["hqnr"], delta_last=la["hqnr"] - lb["hqnr"], a_plateau=(float(np.mean(pa_)) if pa_ else None), b_plateau=(float(np.mean(pb)) if pb else None),
                a_best_d_lambda=next((e["d_lambda"] for e in ea if e["step"] == ba["step"]), None), a_best_d_s=next((e["d_s"] for e in ea if e["step"] == ba["step"]), None))


def three_seed(case, ref="Q00", seeds=SEEDS):
    """§6: seed 별 Δ(common grid) 평균 > 0 이고 최소 2/3 양수. 미완 seed 가 있으면 pass=None(판정 보류)."""
    rows = []
    for s in seeds:
        ra, rb = RUN.get((case, s)), RUN.get((ref, s))
        if not ra or not rb or not finished(ra) or not finished(rb):
            rows.append(dict(seed=s, status="MISSING", a=ra, b=rb, why=(finished(ra, True)[1] if ra and os.path.isdir(os.path.join(ROOT, "work_dir", ra)) else ["run 없음"]))); continue
        ok, bad = comparable(ra, rb)
        if not ok:
            rows.append(dict(seed=s, status="NOT_COMPARABLE", a=ra, b=rb, why=bad)); continue
        cg = common_grid_pair(ra, rb); rows.append(dict(seed=s, status="OK", **cg) if cg else dict(seed=s, status="NO_COMMON_GRID", a=ra, b=rb))
    ok = [r for r in rows if r["status"] == "OK"]; deltas = [r["delta_common"] for r in ok]
    complete = len(ok) == len(seeds)
    return dict(case=case, ref=ref, rows=rows, n_ok=len(ok), mean_delta=(float(np.mean(deltas)) if deltas else None), n_positive=int(sum(d > 0 for d in deltas)),
                passed=((float(np.mean(deltas)) > 0 and sum(d > 0 for d in deltas) >= 2) if complete else None), complete=complete, rule="mean Δ(common grid best_raw HQNR vs same-seed N0) > 0 and ≥ 2/3 seeds positive")


def ledger_path():
    return os.path.join(CAMP, "ledger.json")


def _ts(s_):
    return time.mktime(time.strptime(s_.strip()[:19], "%Y-%m-%dT%H:%M:%S"))


def budget():
    """20h 예산 (리뷰 P1-2): 시계는 switch 시각. 계상 = ① 전환 전 run 의 switch 이후 잔여 시간 ② switch 이후 시작한 모든 run(whitelist 밖·실패·중단 포함) 의 실측(진행 중이면 경과)
    ③ 준비·smoke(ledger overhead_hours) ④ 완료 run 당 export/업로드 overhead(ledger post_run_overhead_hours, 기본 0.1). 예약(잔여 run × 실측 평균×1.1) 은 decide 가 더한다."""
    lp = ledger_path(); d = json.load(open(lp)) if os.path.exists(lp) else {}
    t0s = d.get("start"); t0 = _ts(t0s) if t0s else None; items = {}
    def span(run):
        wd = os.path.join(ROOT, "work_dir", run, "meta")
        try:
            a = _ts(open(os.path.join(wd, "started_at.txt")).read()); b = _ts(open(os.path.join(wd, "finished_at.txt")).read()) if os.path.exists(os.path.join(wd, "finished_at.txt")) else time.time(); return a, b
        except Exception:
            return None
    if t0 is not None:
        for wdp in sorted(glob.glob(os.path.join(ROOT, "work_dir", "NA104_*"))):
            run = os.path.basename(wdp); sp_ = span(run)
            if not sp_:
                continue
            a, b = sp_
            if run == d.get("pre_switch_run") and a < t0 < b + 1:
                items[run] = dict(hours=round((b - t0) / 3600.0, 3), kind="pre_switch_remainder")        # ① 전환 전 run 의 잔여 (계획 §2)
            elif a >= t0:
                items[run] = dict(hours=round((b - a) / 3600.0, 3), kind=("run" if finished(run) else "running" if running(run) else "failed_or_partial"))   # ② 실패·중단 시도도 계상
        post = float(d.get("post_run_overhead_hours", 0.1)) * sum(1 for r, v in items.items() if v["kind"] == "run")
        overhead = float(d.get("overhead_hours", 0.0))
    else:
        post = overhead = 0.0
    used = sum(v["hours"] for v in items.values()) + post + overhead
    done = [v["hours"] for r, v in items.items() if v["kind"] == "run" and r in RUN.values()]
    avg = float(np.mean(done)) if done else float(d.get("est_run_hours", 1.5))
    return dict(start=t0s, cap_hours=BUDGET_H, used_hours=round(used, 3), items=items, post_run_overhead_hours=round(post, 3), prep_overhead_hours=overhead, est_run_hours=round(avg, 3), n_measured=len(done),
                remaining_hours=round(BUDGET_H - used, 3), elapsed_wallclock_hours=(round((time.time() - t0) / 3600.0, 3) if t0 else None),
                note="switch 시계 기준. 전환 전 run 잔여·실패/중단 시도·준비·export overhead 포함 (계획 §2·§6). 완료 시간 보장이 아니라 실측 갱신")


def decide(srv):
    """조건부 분기 (§4·§5·§6). 반환 dict(open=[run...], reasons=[...], q36, q12, cf01_pilots, cf01_final, budget, planned_hours).
    예산은 **승인할 때마다 누적**한다(리뷰 P1-2): 같은 잔여시간을 CF01 과 X02 에 중복 승인하지 않는다."""
    out = dict(server=srv, open=[], reasons=[]); q36 = three_seed("Q36"); q12 = three_seed("Q12"); out["q36"] = q36; out["q12"] = q12
    bud = budget(); out["budget"] = bud; planned = [0.0]
    def affordable(n):
        need = bud["used_hours"] + planned[0] + n * bud["est_run_hours"] * 1.1; ok = need <= BUDGET_H
        out["reasons"].append(f"예산: used {bud['used_hours']:.2f} + 이미 승인 {planned[0]:.2f} + {n}×{bud['est_run_hours']:.2f}×1.1 = {need:.2f} ≤ 20 → {'OK' if ok else 'STOP'}")
        if ok:
            planned[0] += n * bud["est_run_hours"] * 1.1
        return ok
    # CF01 최종 판정 (§6): pilot(S777·S1234 vs Q36) 과 별개로, 3 seed 완료 뒤 N0 대비·Q36 대비 둘 다 3-seed 기준
    cf_n0, cf_q36 = three_seed("CF01", "Q00"), three_seed("CF01", "Q36")
    out["cf01_final"] = dict(vs_N0=dict(passed=cf_n0["passed"], mean_delta=cf_n0["mean_delta"], n_positive=cf_n0["n_positive"], n_ok=cf_n0["n_ok"]), vs_Q36=dict(passed=cf_q36["passed"], mean_delta=cf_q36["mean_delta"], n_positive=cf_q36["n_positive"], n_ok=cf_q36["n_ok"]),
                            passed=((cf_n0["passed"] and cf_q36["passed"]) if (cf_n0["complete"] and cf_q36["complete"]) else None), rule="both: mean Δ > 0 and ≥ 2/3 seeds positive (vs N0 and vs Q36), common grid")
    # P2 CF01
    if q36["passed"]:
        if srv == "s3":
            pil = [RUN[("CF01", 777)], RUN[("CF01", 1234)]]
            if not all(finished(r) for r in pil):
                if affordable(sum(1 for r in pil if not finished(r))):
                    out["open"] += [r for r in pil if not finished(r)]; out["reasons"].append("s3 §4: Q36 3-seed 통과 → CF01 pilot S777·S1234 (50K)")
            else:
                pos = [common_grid_pair(RUN[("CF01", s)], RUN[("Q36", s)]) for s in (777, 1234)]; out["cf01_pilots"] = pos
                if all(p and p["delta_common"] > 0 for p in pos):
                    os.makedirs(CAMP, exist_ok=True); json.dump(dict(server=srv, pilots=pos, at=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(CAMP, "cf01_pilots_positive.json"), "w"), indent=1)
                    if not finished(RUN[("CF01", 2026)]) and affordable(1):
                        out["open"].append(RUN[("CF01", 2026)]); out["reasons"].append("s3 §4: 두 pilot 모두 Q36 대비 Δ(common grid) > 0 → CF01 S2026; s2 에 교차 확인 요청(work_dir/_na104_20h/cf01_pilots_positive.json 을 전달)")
                else:
                    out["reasons"].append("s3 §4: pilot 중 비양성 → CF01 분기 종료 (seed 교체·β 변경 없음)")
        else:   # s2: s3 의 두 pilot 확인 token + 자기 Q36 통과
            tok = os.path.join(CAMP, "cf01_approved_by_s3.txt")
            if os.path.exists(tok):
                todo = [RUN[("CF01", s)] for s in (777, 1234, 2026) if not finished(RUN[("CF01", s)])]
                if todo and affordable(len(todo)):
                    out["open"] += todo; out["reasons"].append(f"s2 §4: s3 pilot 확인 token + 자기 Q36 통과 → CF01 {len(todo)}벌 (S777 → S1234 → S2026)")
            else:
                out["reasons"].append("s2 §4: s3 의 CF01 pilot 확인 token(work_dir/_na104_20h/cf01_approved_by_s3.txt) 없음 — CF01 닫힘")
    else:
        out["reasons"].append(f"Q36 3-seed 기준 {'미완' if q36['passed'] is None else '불통과'} (n_ok {q36['n_ok']}, mean Δ {q36['mean_delta']}, 양수 {q36['n_positive']}) → CF01 닫힘")
    # P3 X02
    if q12["passed"]:
        todo = [RUN[("X02", s)] for s in (777, 2026) if not finished(RUN[("X02", s)])]
        if todo and affordable(len(todo)):
            out["open"] += todo; out["reasons"].append("§5: Q12 3-seed 통과 → X02 S777·S2026 (Q12 와 같은 Teacher/λ_E/seed/update/eval)")
    else:
        out["reasons"].append(f"Q12 3-seed 기준 {'미완' if q12['passed'] is None else '불통과'} → X02 닫힘")
    out["planned_hours"] = planned[0]; out["remaining_after_plan"] = BUDGET_H - bud["used_hours"] - planned[0]
    return out


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None


def report(srv):
    q = os.path.join(ROOT, "config", "queues", f"na104_20h_{srv}.txt"); qs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")] if os.path.exists(q) else []
    cur = [r for r in RUN.values() if running(r)]; done = [r for r in RUN.values() if finished(r)]
    core = {}
    for c in ("Q00", "Q36", "Q12", "CF01", "X02"):
        core[c] = {}
        for s in SEEDS:
            r = RUN.get((c, s))
            if r and finished(r):
                cg = common_grid_pair(r, RUN[("Q00", s)]) if c != "Q00" else None; ev = evals(r); b = json.load(open(os.path.join(ROOT, "work_dir", r, "best_hqnr_meta.json")))
                core[c][s] = dict(run=r, best_original=dict(step=b["step"], hqnr=b["hqnr"], fscc=b.get("fscc")), best_d_lambda=next((e["d_lambda"] for e in ev if e["step"] == b["step"]), None), best_d_s=next((e["d_s"] for e in ev if e["step"] == b["step"]), None),
                                  last=ev[-1]["hqnr"] if ev else None, plateau=(float(np.mean([e["hqnr"] for e in ev if 40000 <= e["step"] <= 50000])) if ev else None), delta_vs_N0_common=(cg or {}).get("delta_common"), common_grid=(cg or {}).get("grid_note"), n_evals=len(ev))
            else:
                core[c][s] = dict(run=r, status=("RUNNING" if r and running(r) else "PENDING" if r in qs else "MISSING"))
    hashes = dict(teacher=dict(run=TEACHER, best_hqnr_sha16=sha16(os.path.join(ROOT, "work_dir", TEACHER, "best_hqnr", "model.safetensors"))), pilot=dict(run=PILOT, last_sha16=sha16(os.path.join(ROOT, "work_dir", PILOT, "last", "model.safetensors"))),
                  q36_lambda_source=dict(run=RUN[("Q36", 1234)], calibration_resolved_sha16=sha16(os.path.join(ROOT, "work_dir", RUN[("Q36", 1234)], "calibration_resolved.json"))),
                  configs={r: sha16(os.path.join(ROOT, "config", r + ".yaml")) for r in qs + [RUN[("CF01", s)] for s in (777, 1234, 2026)] + [RUN[("X02", s)] for s in (777, 2026)]},
                  queue_sha16=sha16(q), evaluator=__import__("pa.evalviews", fromlist=["evaluator_hash"]).evaluator_hash(), git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    dec = decide(srv)
    out = dict(server=srv, generated=time.strftime("%Y-%m-%dT%H:%M:%S"), current_runs=cur, whitelist_p1=[RUN[c] for c in P1], p1_done=[RUN[c] for c in P1 if finished(RUN[c])], p1_pending=[RUN[c] for c in P1 if not finished(RUN[c])],
               deducted_completed=[r for r in done if r in [RUN[c] for c in P1]], queue=q, queue_runs=qs, core=core, decision=dict(open=dec["open"], reasons=dec["reasons"], q36=dict(passed=dec["q36"]["passed"], mean_delta=dec["q36"]["mean_delta"], n_positive=dec["q36"]["n_positive"], n_ok=dec["q36"]["n_ok"]),
               q12=dict(passed=dec["q12"]["passed"], mean_delta=dec["q12"]["mean_delta"], n_positive=dec["q12"]["n_positive"], n_ok=dec["q12"]["n_ok"]), cf01_pilots=dec.get("cf01_pilots"), cf01_final=dec["cf01_final"], planned_hours=dec["planned_hours"]),
               budget=dec["budget"], retired=RETIRED, hashes=hashes, next_tuning_axes=NEXT_AXES, alias=ALIAS, comparability={f"{c}-S{s}": comparable(RUN[(c, s)], RUN[("Q00", s)]) for c in ("Q36", "Q12", "CF01", "X02") for s in SEEDS if (c, s) in RUN and finished(RUN[(c, s)]) and finished(RUN[("Q00", s)])},
               selector_note="선택 기준은 best_raw HQNR → fSCC (best_rr_val 은 보조 selector 일 뿐 판정 축이 아니다; ERGAS 로 바꾸지 않는다)")
    os.makedirs(CAMP, exist_ok=True); json.dump(out, open(os.path.join(CAMP, f"report_{srv}.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[20h {srv}] 현재 run {cur or '없음'} · P1 완료 {len(out['p1_done'])}/4 · 예산 used {dec['budget']['used_hours']} h / 20 (run 평균 {dec['budget']['est_run_hours']} h)")
    for c in ("Q00", "Q36", "Q12", "CF01", "X02"):
        print(f"  {c}: " + " | ".join(f"S{s} " + (f"best {v['best_original']['hqnr']:.5f}@{v['best_original']['step']} last {v['last']:.5f} plateau {v['plateau']:.5f}" + (f" ΔN0(cg) {v['delta_vs_N0_common']:+.5f}" if v.get('delta_vs_N0_common') is not None else "") if "best_original" in v else v["status"]) for s, v in core[c].items()))
    print(f"  판정: Q36 {out['decision']['q36']} · Q12 {out['decision']['q12']} · CF01 최종 {out['decision']['cf01_final']['passed']} (vs N0 {out['decision']['cf01_final']['vs_N0']['passed']}, vs Q36 {out['decision']['cf01_final']['vs_Q36']['passed']})\n  열림: {dec['open'] or '없음'} (승인 예약 {dec['planned_hours']:.2f} h)\n  사유: " + " / ".join(dec["reasons"]))
    print(f"  hash: teacher {hashes['teacher']['best_hqnr_sha16']} pilot {hashes['pilot']['last_sha16']} queue {hashes['queue_sha16']} git {hashes['git']} evaluator {hashes['evaluator']}\n  -> work_dir/_na104_20h/report_{srv}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=("report", "decide")); ap.add_argument("--server", default=None); a = ap.parse_args()
    srv = a.server or server()
    if a.cmd == "report":
        report(srv)
    else:
        d = decide(srv); print(json.dumps(dict(open=d["open"], reasons=d["reasons"], q36=d["q36"]["passed"], q12=d["q12"]["passed"], budget=d["budget"]["used_hours"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
