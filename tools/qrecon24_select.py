#!/usr/bin/env python
"""QRECON24 target selector HQNR9585_RR_v1 (계획 research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §7.1–§7.3; 감사 F06/F07/F10 반영).

    python tools/qrecon24_select.py <run> [--official] [--threshold 0.9585] [--device cuda] [--out <json>]

같은 checkpoint 안에서 고른다: 후보 격자(GRID1010_50K_v1; checkpoint_metrics.csv 의 50 후보) 중 raw-original HQNR ≥ 0.9585(1e-4 tie band 없음) 를 통과한 후보를
RR SCC 큰 순 → ERGAS 작은 순 → PSNR 큰 순 → SAM 작은 순 → Q8 큰 순 → SSIM 큰 순 → H 큰 순으로 정렬한다. 여섯 RR 를 전부 넘는 후보가 있으면 그 집합의 최상도 따로 둔다.
기존 best_hqnr(legacy) · raw-max · exact50K · 후반 6 점(45450…50000) 은 그대로 보존·병기한다 (§7.3). 적격 후보가 없으면 target_feasible=false 와 legacy best 를 같이 기록한다.

--official: 적격 후보의 checkpoint(work_dir/<run>/candidates/step-N) 로 reduced 테스트셋을 다시 추론해 **공식 RR 6 지표**(tools/metrics/eval_rr; gspread 의 _rr 와 같은 경로·crop) 로 정렬한다.
  · 서로 다른 evaluator 값을 섞어 정렬하지 않는다(감사 F06): 공식 RR 가 있는 후보만 순위에 들어가고, 하나라도 없으면 target 을 확정하지 않는다(target_status incomplete_official_rr).
  · 캐시 `results/reduced_candidate_step-N.mat` 은 sidecar(.json) 의 checkpoint sha256 · config sha256 · reduced h5 sha256 · evaluator sha 가 현재와 같을 때만 재사용한다(감사 F07). 선택 결과에도 같은 hash 를 적는다.
--official 없이(proxy): CSV 의 학습 중 rr_scc/rr_ergas/rr_sam 만으로 정렬하고 official=false · target_status proxy 를 남긴다 (PSNR/Q8/SSIM 없음; 업로더에 '지원완료' 로 가장하지 않는다 §9.3).
H 의 적격 판정은 CSV 의 raw_original.hqnr (학습 중 공식 evaluator 와 같은 mat20 protocol) 이며, legacy best 의 fr_mat20.json 값과의 일치를 h_consistency 로 기록한다.
display_tie: 논문 표시 정밀도(SCC 3 · ERGAS 3 · PSNR 3 · SAM 3 · Q8 3 · SSIM 3 자리) 로 반올림해 같으면 tie (numeric_pass 와 구분, §7.1).
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
SELECTOR = "HQNR9585_RR_v1"; THRESHOLD = 0.9585
RR_TARGETS = dict(scc=(">", 0.988), ergas=("<", 2.040), psnr=(">", 37.956), sam=("<", 2.787), q8=(">", 0.922), ssim=(">", 0.976))     # PAN-Crafter Table 1 WV3 표시값 (§7.1)
RR_DISPLAY_DECIMALS = dict(scc=3, ergas=3, psnr=3, sam=3, q8=3, ssim=3)
ORDER = (("scc", -1), ("ergas", 1), ("psnr", -1), ("sam", 1), ("q8", -1), ("ssim", -1), ("hqnr", -1))
LATE6 = (45450, 46460, 47470, 48480, 49490, 50000)
RR_KEYS = ("scc", "ergas", "psnr", "sam", "q8", "ssim")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def sha256_file(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(wd):
    p = os.path.join(wd, "checkpoint_metrics.csv")
    if not os.path.exists(p):
        sys.exit(f"!! {p} 없음")
    rows = []
    for r in csv.DictReader(open(p)):
        rows.append(dict(step=int(float(r["step"])), hqnr=fl(r.get("raw_original.hqnr")), fscc=fl(r.get("raw_original.fscc")), scc=fl(r.get("rr_scc")), ergas=fl(r.get("rr_ergas")), sam=fl(r.get("rr_sam")), psnr=float("nan"), q8=float("nan"), ssim=float("nan")))
    return rows


def evaluator_identity():
    """공식 RR 경로의 identity: eval_rr / eval_dlpan / gspread _rr 코드 sha (+ eval_fr_paperset EVAL_VERSION)."""
    ids = {}
    for k, rel in (("eval_rr", "tools/metrics/eval_rr.py"), ("eval_dlpan", "tools/eval_dlpan.py"), ("gspread_upload", "gspread/gspread_upload.py"), ("q2n", "tools/metrics/q2n.py")):
        p = os.path.join(ROOT, rel); ids[k] = sha256_file(p) if os.path.exists(p) else None
    try:
        spec2 = importlib.util.spec_from_file_location("_efp", os.path.join(ROOT, "tools", "eval_fr_paperset.py")); efp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(efp); ids["eval_fr_paperset_version"] = getattr(efp, "EVAL_VERSION", None)
    except Exception:                                                                  # noqa
        ids["eval_fr_paperset_version"] = None
    return ids


def official_rr(tag, wd, step, device, evid):
    """후보 checkpoint 로 reduced 테스트셋 추론 → results/reduced_candidate_step-<N>.mat (+ sidecar .json: checkpoint/config/h5/evaluator hash) → gspread._rr 의 6 지표. sidecar 가 현재 identity 와 같을 때만 캐시 재사용."""
    import torch, yaml
    from scipy.io import savemat
    spec = importlib.util.spec_from_file_location("_gu", os.path.join(ROOT, "gspread", "gspread_upload.py")); gu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gu)
    spec2 = importlib.util.spec_from_file_location("_efp", os.path.join(ROOT, "tools", "eval_fr_paperset.py")); efp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(efp)
    from main import import_class
    cfgp = os.path.join(wd, "meta", "config.yaml"); ck = f"candidates/step-{step}"; ckf = os.path.join(wd, ck, "model.safetensors")
    if not os.path.exists(ckf):
        return None, f"{ck} 없음"
    if not os.path.exists(cfgp):
        return None, "meta/config.yaml 없음"
    cfg = yaml.safe_load(open(cfgp))
    h5 = cfg["test_reduced_feeder_args"]["dataroot"]; droot = h5; dsn = next((s for s in ("wv3", "qb", "gf2", "wv2") if f"test_{s}_" in droot), "wv3")
    ident = dict(checkpoint_sha256=sha256_file(ckf), config_sha256=sha256_file(cfgp), reduced_h5_sha256=(sha256_file(h5) if os.path.exists(h5) else None), evaluator=evid, step=int(step), device=None)
    mat = os.path.join(wd, "results", f"reduced_candidate_step-{step}.mat"); side = mat[:-4] + ".json"
    cached = False
    if os.path.exists(mat) and os.path.exists(side):
        try:
            sj = json.load(open(side)); cached = all(sj.get(k) == ident[k] for k in ("checkpoint_sha256", "config_sha256", "reduced_h5_sha256", "evaluator", "step"))
        except Exception:                                                              # noqa
            cached = False
    t0 = time.time()
    if not cached:
        m, fwd, how = efp.build(cfg, wd, ck); dev = torch.device(device); m = m.to(dev).eval()
        Feeder = import_class(cfg["feeder"]); fa = dict(cfg["test_reduced_feeder_args"]); ds = Feeder(**fa); mp = float(ds.max_pixel); srs = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]; gt, (lms, ms, lpan, pan) = item[0], item[1:]
                y = fwd(pan.unsqueeze(0).to(dev), lpan.unsqueeze(0).to(dev), ms.unsqueeze(0).to(dev), lms.unsqueeze(0).to(dev))
                srs.append(((y.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * mp)[0])
        os.makedirs(os.path.dirname(mat), exist_ok=True); savemat(mat, dict(sr=np.stack(srs))); ident["device"] = str(dev); ident["forward"] = how; ident["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(ident, open(side, "w"), indent=1)
    r = gu._rr(mat, dsn); sec = time.time() - t0
    return dict(scc=r.get("scc"), ergas=r.get("ergas"), psnr=r.get("psnr"), sam=r.get("sam"), q8=r.get("q2n", r.get("q8")), ssim=r.get("ssim"), mat=os.path.relpath(mat, ROOT), cached=cached, rr_eval_seconds=sec, identity=(json.load(open(side)) if os.path.exists(side) else ident)), "official"


def rank(cands):
    def key(c):
        return tuple((s * c[k]) if (c.get(k) is not None and np.isfinite(c[k])) else float("inf") for k, s in ORDER)
    return sorted(cands, key=key)


def rr_pass(c):
    out = {}
    for k, (op, thr) in RR_TARGETS.items():
        v = c.get(k)
        out[k] = (None if v is None or not np.isfinite(v) else (v > thr if op == ">" else v < thr))
    return out


def display_tie(c):
    """논문 표시 정밀도로 반올림해 같은 값이면 tie (표시값을 넘는 numeric_pass 와 구분)."""
    out = {}
    for k, (op, thr) in RR_TARGETS.items():
        v = c.get(k); d = RR_DISPLAY_DECIMALS[k]
        out[k] = (None if v is None or not np.isfinite(v) else (round(float(v), d) == round(thr, d)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--official", action="store_true"); ap.add_argument("--threshold", type=float, default=THRESHOLD); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); wd = os.path.join(ROOT, "work_dir", a.run); rows = load_rows(wd)
    bm = os.path.join(wd, "best_hqnr_meta.json"); legacy = json.load(open(bm)) if os.path.exists(bm) else {}
    fr = os.path.join(wd, "results", "fr_mat20.json"); frj = json.load(open(fr)) if os.path.exists(fr) else {}
    by_step = {r["step"]: r for r in rows}; leg_step = legacy.get("step"); h_cons = None
    if leg_step in by_step and frj.get("hqnr") is not None:
        h_cons = dict(step=leg_step, csv_hqnr=by_step[leg_step]["hqnr"], fr_mat20_hqnr=frj["hqnr"], abs_diff=abs(by_step[leg_step]["hqnr"] - frj["hqnr"]))
    elig = [dict(r) for r in rows if np.isfinite(r["hqnr"]) and r["hqnr"] >= a.threshold]
    notes = []; evid = evaluator_identity() if a.official else None; t_eval = 0.0
    if a.official:
        for c in elig:
            m, how = official_rr(a.run, wd, c["step"], a.device, evid)
            if m is None:
                c["official_rr"] = False; notes.append(f"step {c['step']}: {how}"); continue
            c.update({k: m[k] for k in RR_KEYS}); c.update(rr_mat=m["mat"], official_rr=True, rr_cached=m["cached"], rr_identity=m["identity"]); t_eval += float(m["rr_eval_seconds"])
        evaluated = [c for c in elig if c.get("official_rr")]; unevaluated = [c for c in elig if not c.get("official_rr")]
        official = bool(elig) and not unevaluated
        ranked = rank(evaluated)                                                       # 감사 F06: proxy 값이 든 후보는 공식 순위에 넣지 않는다
        if elig and unevaluated:
            target_status = "incomplete_official_rr"; target = None                     # 일부 후보의 공식 RR 가 없으면 target 을 확정하지 않는다
        else:
            target_status = ("official" if ranked else "no_eligible"); target = ranked[0] if ranked else None
    else:
        evaluated, unevaluated = [], list(elig); official = False; ranked = rank(elig); target = ranked[0] if ranked else None; target_status = ("proxy" if ranked else "no_eligible")
        notes.append("proxy 정렬: CSV 의 학습 중 rr_scc/rr_ergas/rr_sam 만 (PSNR/Q8/SSIM 없음; SCC 는 학습 로그 정의) — --official 로 공식 RR 6 지표를 채운다")
    six = [c for c in ranked if all(v is True for v in rr_pass(c).values())]
    raw_max = max(rows, key=lambda r: (r["hqnr"] if np.isfinite(r["hqnr"]) else -1)) if rows else None
    ex50 = by_step.get(50000); late = [by_step[s]["hqnr"] for s in LATE6 if s in by_step and np.isfinite(by_step[s]["hqnr"])]
    def _t(c):
        return None if c is None else dict(c, rr_pass=rr_pass(c), numeric_pass_all6=bool(all(v is True for v in rr_pass(c).values())), display_tie=display_tie(c))
    out = dict(run=a.run, selector=SELECTOR, threshold=a.threshold, official=bool(official), target_status=target_status, n_candidates=len(rows), n_eligible=len(elig), n_official_evaluated=len(evaluated), n_unevaluated=len(unevaluated),
               target_feasible=(bool(target is not None) if target_status != "incomplete_official_rr" else None), target=_t(target),
               proxy_target=(_t(ranked[0]) if (not a.official and ranked) else None), six_pass_best=_t(six[0]) if six else None,
               legacy_best=dict(step=leg_step, hqnr=(by_step.get(leg_step) or {}).get("hqnr"), fr_mat20_hqnr=frj.get("hqnr")), h_consistency=h_cons,
               raw_max=(dict(step=raw_max["step"], hqnr=raw_max["hqnr"]) if raw_max else None), exact50K=(dict(step=50000, hqnr=ex50["hqnr"], scc=ex50["scc"], ergas=ex50["ergas"]) if ex50 else None),
               late6=(dict(steps=list(LATE6), n=len(late), hqnr_mean=float(np.mean(late))) if late else None),
               eligible_ranked=[{k: v for k, v in c.items()} for c in ranked], eligible_unevaluated=[dict(step=c["step"], hqnr=c["hqnr"]) for c in unevaluated],
               evaluator=evid, rr_eval_seconds_total=t_eval, notes=notes, rr_targets={k: f"{op} {thr}" for k, (op, thr) in RR_TARGETS.items()}, display_decimals=RR_DISPLAY_DECIMALS,
               note="같은 checkpoint 의 raw H(mat20 전체 frame) 하한 통과 뒤 RR 정렬; legacy/raw-max/exact50K/late6 는 보존·병기 (§7.2–§7.3). FR fSCC 와 RR SCC 는 다른 값이다. 공식 RR 는 candidate checkpoint 재추론(hash 로 캐시 검증).")
    p = a.out or os.path.join(wd, "results", "qrecon24_target_selection.json"); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"[qrecon24-select] {a.run}: 후보 {len(rows)} · 적격(H ≥ {a.threshold}) {len(elig)} · status {target_status} · target {'step %d (H %.6f, SCC %s, ERGAS %s)' % (target['step'], target['hqnr'], target.get('scc'), target.get('ergas')) if target else '없음'}"
          f" · legacy best step {leg_step} · raw-max step {raw_max['step'] if raw_max else '?'} · official {official}{' · 미평가 ' + str([c['step'] for c in unevaluated]) if (a.official and unevaluated) else ''} → {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
