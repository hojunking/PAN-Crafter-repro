#!/usr/bin/env python
"""QRECON24 target selector HQNR9585_RR_v1 (계획 research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §7.1–§7.3).

    python tools/qrecon24_select.py <run> [--official] [--threshold 0.9585] [--device cuda]

같은 checkpoint 안에서 고른다: 후보 격자(GRID1010_50K_v1; checkpoint_metrics.csv 의 50 후보) 중 raw-original HQNR ≥ 0.9585(1e-4 tie band 없음) 를 통과한 후보를
RR SCC 큰 순 → ERGAS 작은 순 → PSNR 큰 순 → SAM 작은 순 → Q8 큰 순 → SSIM 큰 순 → H 큰 순으로 정렬한다. 여섯 RR 를 전부 넘는 후보가 있으면 그 집합의 최상도 따로 둔다.
기존 best_hqnr(legacy) · raw-max · exact50K · 후반 6 점(45450…50000) 은 그대로 보존·병기한다 (§7.3). 적격 후보가 없으면 target_feasible=false 와 legacy best 를 같이 기록한다.

--official: 적격 후보의 checkpoint(work_dir/<run>/candidates/step-N) 로 reduced 테스트셋을 다시 추론해 **공식 RR 6 지표**(tools/metrics/eval_rr; gspread 의 _rr 와 같은 경로·crop) 로 정렬한다.
없으면(proxy) checkpoint_metrics.csv 의 학습 중 rr_scc/rr_ergas/rr_sam 만으로 정렬하고 official=false 를 남긴다 (PSNR/Q8/SSIM 없음; 업로더에 '지원완료' 로 가장하지 않는다 §9.3).
H 의 적격 판정은 CSV 의 raw_original.hqnr (학습 중 공식 evaluator 와 같은 mat20 protocol) 이며, legacy best 의 fr_mat20.json 값과의 일치를 h_consistency 로 기록한다.
"""
import argparse, csv, importlib.util, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
SELECTOR = "HQNR9585_RR_v1"; THRESHOLD = 0.9585
RR_TARGETS = dict(scc=(">", 0.988), ergas=("<", 2.040), psnr=(">", 37.956), sam=("<", 2.787), q8=(">", 0.922), ssim=(">", 0.976))     # PAN-Crafter Table 1 WV3 표시값 (§7.1)
ORDER = (("scc", -1), ("ergas", 1), ("psnr", -1), ("sam", 1), ("q8", -1), ("ssim", -1), ("hqnr", -1))
LATE6 = (45450, 46460, 47470, 48480, 49490, 50000)


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(wd):
    p = os.path.join(wd, "checkpoint_metrics.csv")
    if not os.path.exists(p):
        sys.exit(f"!! {p} 없음")
    rows = []
    for r in csv.DictReader(open(p)):
        rows.append(dict(step=int(float(r["step"])), hqnr=fl(r.get("raw_original.hqnr")), fscc=fl(r.get("raw_original.fscc")), scc=fl(r.get("rr_scc")), ergas=fl(r.get("rr_ergas")), sam=fl(r.get("rr_sam")), psnr=float("nan"), q8=float("nan"), ssim=float("nan")))
    return rows


def official_rr(tag, wd, step, device):
    """후보 checkpoint 로 reduced 테스트셋 추론 → results/reduced_candidate_step-<N>.mat → gspread._rr (tools/eval_dlpan SCALE/GT, crop 20:-21, eval_rr.evaluate) 의 6 지표."""
    import torch, yaml
    from scipy.io import savemat
    spec = importlib.util.spec_from_file_location("_gu", os.path.join(ROOT, "gspread", "gspread_upload.py")); gu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gu)
    spec2 = importlib.util.spec_from_file_location("_efp", os.path.join(ROOT, "tools", "eval_fr_paperset.py")); efp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(efp)
    from main import import_class
    cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml"))); ck = f"candidates/step-{step}"
    if not os.path.exists(os.path.join(wd, ck, "model.safetensors")):
        return None, f"{ck} 없음"
    mat = os.path.join(wd, "results", f"reduced_candidate_step-{step}.mat")
    if not os.path.exists(mat):
        m, fwd, how = efp.build(cfg, wd, ck); dev = torch.device(device); m = m.to(dev).eval()
        Feeder = import_class(cfg["feeder"]); fa = dict(cfg["test_reduced_feeder_args"]); ds = Feeder(**fa); mp = float(ds.max_pixel); srs = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]; gt, (lms, ms, lpan, pan) = item[0], item[1:]
                y = fwd(pan.unsqueeze(0).to(dev), lpan.unsqueeze(0).to(dev), ms.unsqueeze(0).to(dev), lms.unsqueeze(0).to(dev))
                srs.append(((y.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * mp)[0])
        os.makedirs(os.path.dirname(mat), exist_ok=True); savemat(mat, dict(sr=np.stack(srs)))
    droot = cfg["test_reduced_feeder_args"].get("dataroot", ""); dsn = next((s for s in ("wv3", "qb", "gf2", "wv2") if f"test_{s}_" in droot), "wv3")
    r = gu._rr(mat, dsn)
    return dict(scc=r.get("scc"), ergas=r.get("ergas"), psnr=r.get("psnr"), sam=r.get("sam"), q8=r.get("q2n", r.get("q8")), ssim=r.get("ssim"), mat=os.path.relpath(mat, ROOT)), "official"


def rank(cands):
    def key(c):
        return tuple((s * c[k]) if np.isfinite(c[k]) else float("inf") for k, s in ORDER)
    return sorted(cands, key=key)


def rr_pass(c):
    out = {}
    for k, (op, thr) in RR_TARGETS.items():
        v = c.get(k)
        out[k] = (None if v is None or not np.isfinite(v) else (v > thr if op == ">" else v < thr))
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
    official = False; notes = []
    if a.official and elig:
        for c in elig:
            m, how = official_rr(a.run, wd, c["step"], a.device)
            if m is None:
                notes.append(f"step {c['step']}: {how}"); continue
            c.update({k: m[k] for k in ("scc", "ergas", "psnr", "sam", "q8", "ssim")}); c["rr_mat"] = m["mat"]; c["official_rr"] = True
        official = all(c.get("official_rr") for c in elig)
    ranked = rank(elig); target = ranked[0] if ranked else None
    six = [c for c in ranked if all(v is True for v in rr_pass(c).values())]
    raw_max = max(rows, key=lambda r: (r["hqnr"] if np.isfinite(r["hqnr"]) else -1)) if rows else None
    ex50 = by_step.get(50000); late = [by_step[s]["hqnr"] for s in LATE6 if s in by_step and np.isfinite(by_step[s]["hqnr"])]
    out = dict(run=a.run, selector=SELECTOR, threshold=a.threshold, official=bool(official), n_candidates=len(rows), n_eligible=len(elig), target_feasible=bool(target is not None),
               target=(dict(target, rr_pass=rr_pass(target), numeric_pass_all6=bool(all(v is True for v in rr_pass(target).values()))) if target else None),
               six_pass_best=(dict(six[0]) if six else None), legacy_best=dict(step=leg_step, hqnr=(by_step.get(leg_step) or {}).get("hqnr"), fr_mat20_hqnr=frj.get("hqnr")), h_consistency=h_cons,
               raw_max=(dict(step=raw_max["step"], hqnr=raw_max["hqnr"]) if raw_max else None), exact50K=(dict(step=50000, hqnr=ex50["hqnr"], scc=ex50["scc"], ergas=ex50["ergas"]) if ex50 else None),
               late6=(dict(steps=list(LATE6), n=len(late), hqnr_mean=float(np.mean(late))) if late else None), eligible_ranked=[{k: v for k, v in c.items()} for c in ranked],
               notes=notes + ([] if official else ["proxy 정렬: CSV 의 학습 중 rr_scc/rr_ergas/rr_sam 만 (PSNR/Q8/SSIM 없음) — --official 로 공식 RR 6 지표를 채운다"]),
               rr_targets={k: f"{op} {thr}" for k, (op, thr) in RR_TARGETS.items()}, note="같은 checkpoint 의 raw H(mat20 전체 frame) 하한 통과 뒤 RR 정렬; legacy/raw-max/exact50K/late6 는 보존·병기 (§7.2–§7.3). FR fSCC 와 RR SCC 는 다른 값이다.")
    p = a.out or os.path.join(wd, "results", "qrecon24_target_selection.json"); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"[qrecon24-select] {a.run}: 후보 {len(rows)} · 적격(H ≥ {a.threshold}) {len(elig)} · target {'step %d (H %.6f, SCC %s, ERGAS %s)' % (target['step'], target['hqnr'], target.get('scc'), target.get('ergas')) if target else '없음 (target_feasible=false)'}"
          f" · legacy best step {leg_step} · raw-max step {raw_max['step'] if raw_max else '?'} · official {official} → {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
