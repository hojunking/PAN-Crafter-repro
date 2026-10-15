#!/usr/bin/env python
"""NA104 HQNR 조정 지시(2026-09-11)의 분석 산출물 — **읽기 전용**.

    python tools/na104_hqnr_report.py --grid 10 --baseline NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1 "NA104_*_v1"
    python tools/na104_hqnr_report.py --grid 10 --out analysis/hqnr_revision_20260911 "NA104_Q0*_v1"

research_log/PAN_NA104_S2_S3_HQNR_Experiment_Amendment_2026-09-11.md 의
  §3.2 세 가지 HQNR (hqnr_best_original · hqnr_best_common_grid · hqnr_plateau · hqnr_last)
  §3.3 공통 격자 절차 (실제 평가 시점을 읽고, 공통 후보에서 같은 selector·tie-break 로 재선택, checkpoint 연결)
  §10.1 필수 산출물 (run_inventory · common_grid_manifest · hqnr_comparison · hqnr_per_scene)
  §11  HQNR 분해 (장면별 spectral/spatial/interaction 항)
를 만든다.

**이 도구는 어떤 학습 산출물도 바꾸지 않는다.** 원래 `best_hqnr` 와 그 파일·manifest 는 그대로 두고,
공통 격자 결과는 별도 분석 파일로만 남긴다 (§3.3-6).

주 지표의 정의 (§3.1): 원본 FR 논문 세트 20장, view `raw_original`, **장면별 HQNR 을 구한 뒤 평균**.
평균 Dλ·평균 Ds 의 곱은 장면별 곱의 평균과 다르므로 분해는 반드시 장면별로 계산한다.

한계: 학습 중 평가는 checkpoint 를 모두 보존하지 않는다. 공통 격자에서 고른 시점의 가중치가 남아 있지
않으면 `score_only` 로 표시하고, 그 시점의 새 영상·분해 지표를 만들었다고 기록하지 않는다 (§3.3).
"""
import argparse, csv, glob, json, os, subprocess, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEW = "raw_original"
TIE = "HQNR → fSCC → 늦은 step (학습 중 selector 와 같은 순서)"


def _rows(run, name):
    p = os.path.join(ROOT, "work_dir", run, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def evals(run):
    """실제로 평가가 성공한 시점만 (config 주석이 아니라 산출물을 읽는다, §3.3-1)."""
    out = []
    for r in _rows(run, "checkpoint_metrics.csv"):
        h = _f(r.get(f"{VIEW}.hqnr"))
        if h is None:
            continue
        out.append(dict(step=int(r["step"]), epoch=int(r["epoch"]), hqnr=h, fscc=_f(r.get(f"{VIEW}.fscc")),
                        d_lambda=_f(r.get(f"{VIEW}.d_lambda")), d_s=_f(r.get(f"{VIEW}.d_s")),
                        rr_ergas=_f(r.get("rr_ergas")), rr_val_ergas=_f(r.get("rr_val_ergas")), rr_val_mae=_f(r.get("rr_val_mae"))))
    return sorted(out, key=lambda x: x["step"])


def best(cands):
    """같은 selector·tie-break 로 하나를 고른다 (§3.3-4)."""
    if not cands:
        return None
    return sorted(cands, key=lambda c: (round(c["hqnr"], 12), (c["fscc"] if c["fscc"] is not None else -1), c["step"]))[-1]


def saved_tags(run):
    """이 run 이 실제로 보존한 checkpoint 의 step (best_hqnr·best_rr_val·last …) — 재선택 시점과 연결한다 (§3.3-5)."""
    out = {}
    for t in ("best_hqnr", "best_aligned", "best_rr_val", "last"):
        mp = os.path.join(ROOT, "work_dir", run, f"{t}_meta.json")
        if os.path.exists(mp) and os.path.isdir(os.path.join(ROOT, "work_dir", run, t)):
            try:
                out[t] = int(json.load(open(mp)).get("step"))
            except Exception:
                pass
    return out


def inventory(run):
    wd = os.path.join(ROOT, "work_dir", run)
    cfg = os.path.join(wd, "meta", "config.yaml")
    d = dict(run_id=run, exists=os.path.isdir(wd))
    if not d["exists"]:
        return d
    import yaml
    c = yaml.safe_load(open(cfg)) if os.path.exists(cfg) else {}
    k = (c.get("kdv") or {})
    kr = os.path.join(wd, "kdv_config_resolved.json"); th = os.path.join(wd, "init_and_teacher_hashes.json")
    kj = json.load(open(kr)) if os.path.exists(kr) else {}
    tj = json.load(open(th)) if os.path.exists(th) else {}
    lm = os.path.join(wd, "last_meta.json")
    ev = evals(run)
    d.update(campaign=k.get("campaign_id"), version=k.get("version"), seed=c.get("seed"), num_iter=c.get("num_iter"),
             width=(c.get("model_args") or {}).get("hidden_size"), depth=str((c.get("model_args") or {}).get("depth")),
             rec=(k.get("rec") or {}).get("case"), stat=("OFF" if not (k.get("stat") or {}).get("enabled") else f"{k['stat'].get('kind')}-{k['stat'].get('mode')}"),
             select_primary=((kj.get("selection") or {}).get("primary")), na_protocol=kj.get("na_protocol"),
             teacher_run=((tj.get("teacher") or {}).get("run")), teacher_tag=((tj.get("teacher") or {}).get("tag")),
             teacher_sha16=((tj.get("teacher") or {}).get("tensors_sha256_16")), teacher_eval_only=((kj.get("spec") or {}).get("teacher_eval_only")),
             init_sha16=((tj.get("init_hashes") or {}).get("unet_init_sha256_16")), evaluator_hash=kj.get("evaluator_hash"),
             resumed=kj.get("resumed"), n_evals=len(ev), first_eval_step=(ev[0]["step"] if ev else None), last_eval_step=(ev[-1]["step"] if ev else None),
             eval_epoch_stride=(min({b["epoch"] - a["epoch"] for a, b in zip(ev, ev[1:])}) if len(ev) > 1 else None),
             finished_step=(json.load(open(lm)).get("step") if os.path.exists(lm) else None),
             status=("FULL_N" if os.path.exists(lm) else ("RUNNING" if ev else "STARTED")),
             saved_checkpoints=";".join(f"{t}@{s}" for t, s in sorted(saved_tags(run).items())))
    return d


def plateau(ev, lo, hi):
    xs = [e["hqnr"] for e in ev if lo <= e["step"] <= hi]
    if not xs:
        return {}
    import statistics as st
    return dict(plateau_n=len(xs), plateau_mean=sum(xs) / len(xs), plateau_std=(st.pstdev(xs) if len(xs) > 1 else 0.0),
                plateau_min=min(xs), plateau_max=max(xs))


def scenes_at(run, step):
    out = []
    for r in _rows(run, "scene_metrics.csv"):
        if int(r["step"]) != step or r.get("view") != VIEW:
            continue
        out.append(dict(scene=int(r["scene"]), hqnr=_f(r["hqnr"]), d_lambda=_f(r["d_lambda"]), d_s=_f(r["d_s"]), fscc=_f(r.get("fscc"))))
    return sorted(out, key=lambda x: x["scene"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+", help="run glob (work_dir 기준)")
    ap.add_argument("--grid", type=int, default=10, help="공통 격자 (epoch 배수). 0 이면 실제 공통 step 교집합만 쓴다")
    ap.add_argument("--baseline", default=None, help="ΔHQNR·HQNR 분해의 대조군 run")
    ap.add_argument("--plateau", default="40000:50000", help="후반 구간 (update lo:hi, §3.2)")
    ap.add_argument("--out", default="analysis/hqnr_revision_20260911")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.plateau.split(":"))
    runs = sorted({os.path.basename(p) for pat in a.pattern for p in glob.glob(os.path.join(ROOT, "work_dir", pat)) if os.path.isdir(p)})
    if not runs:
        print("대상 run 이 없다"); return 2
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    server = open(os.path.join(ROOT, "gspread", "server.txt")).read().strip() if os.path.exists(os.path.join(ROOT, "gspread", "server.txt")) else "?"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()[:12]

    inv = [dict(server=server, commit=commit, **inventory(r)) for r in runs]
    with open(os.path.join(out, "run_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys())); w.writeheader(); w.writerows(inv)

    E = {r: evals(r) for r in runs}
    E = {r: v for r, v in E.items() if v}
    if not E:
        print("평가 기록이 있는 run 이 없다"); return 2
    # 공통 후보: 격자 배수의 epoch 중 **모든 run 이 실제로 평가한** 시점 (보간하지 않는다, §3.3-3)
    per = {r: {e["epoch"] for e in v if (a.grid <= 0 or e["epoch"] % a.grid == 0)} for r, v in E.items()}
    common = set.intersection(*per.values()) if per else set()
    own_last = {r: max(x["epoch"] for x in v) for r, v in E.items()}
    trunc = (max(common) if common else 0) < 0.8 * min(own_last.values()) if common else True
    man = dict(created_by="tools/na104_hqnr_report.py", commit=commit, server=server, view=VIEW, grid=a.grid, tie_break=TIE,
               horizon=dict(own_last_epoch=own_last, common_last_epoch=(max(common) if common else None), truncated=bool(trunc),
                            warning=("공통 구간이 각 run 의 학습 horizon 보다 크게 짧다 — 진행 중 run 이 섞여 있으면 "
                                     "완료 run 끼리만 다시 돌리거나 같은 horizon 의 block 으로 묶어 비교한다" if trunc else None)),
               plateau_window=[lo, hi], runs={r: dict(evaluated_epochs=sorted(x["epoch"] for x in E[r]), on_grid=sorted(per[r])) for r in E},
               common_epochs=sorted(common), n_common=len(common),
               note=("공통 격자는 선택 '기회' 만 맞춘다 — 평가 간격이 달랐다는 사실 자체가 학습 궤적을 바꾸지는 않지만, "
                     "평가/진단이 전역 RNG·데이터 순서를 소비했다면 궤적 차이는 남는다 (§3.3)."))
    json.dump(man, open(os.path.join(out, "common_grid_manifest.json"), "w"), indent=1, ensure_ascii=False)

    base = a.baseline if (a.baseline and a.baseline in E) else None
    rows, per_scene = [], []
    sel_step = {}
    for r in sorted(E):
        ev = E[r]
        bo = best(ev)
        cg = best([e for e in ev if e["epoch"] in common]) if common else None
        last = ev[-1]
        tags = saved_tags(r)
        row = dict(server=server, run_id=r, n_evals=len(ev),
                   hqnr_best_original=bo["hqnr"], best_original_epoch=bo["epoch"], best_original_step=bo["step"],
                   d_lambda_at_best_original=bo["d_lambda"], d_s_at_best_original=bo["d_s"],
                   hqnr_best_common_grid=(cg["hqnr"] if cg else None), common_epoch=(cg["epoch"] if cg else None), common_step=(cg["step"] if cg else None),
                   d_lambda_at_common=(cg["d_lambda"] if cg else None), d_s_at_common=(cg["d_s"] if cg else None), fscc_at_common=(cg["fscc"] if cg else None),
                   rr_ergas_at_common=(cg["rr_ergas"] if cg else None), rr_val_ergas_at_common=(cg["rr_val_ergas"] if cg else None),
                   hqnr_last=last["hqnr"], last_step=last["step"], **plateau(ev, lo, hi))
        row["checkpoint_at_common"] = next((t for t, st in tags.items() if cg and st == cg["step"]), "score_only")
        row["selection_note"] = "재선택은 분석 산출물이다 — 원래 best_hqnr 파일·manifest 는 그대로 둔다 (§3.3-6)"
        sel_step[r] = cg["step"] if cg else None
        rows.append(row)
    # ΔHQNR·분해 (§11): 같은 checkpoint 정책(공통 격자 선택 시점) 의 **장면별** 값으로만 계산한다
    if base and sel_step.get(base):
        bs = {x["scene"]: x for x in scenes_at(base, sel_step[base])}
        for row in rows:
            r = row["run_id"]
            row["delta_hqnr_vs_baseline"] = (row["hqnr_best_common_grid"] - next(x["hqnr_best_common_grid"] for x in rows if x["run_id"] == base)) if row["hqnr_best_common_grid"] is not None else None
            ms = {x["scene"]: x for x in scenes_at(r, sel_step[r] or -1)}
            sc = sorted(set(bs) & set(ms))
            if not sc or r == base:
                continue
            sp_, sa_, ix_, win = 0.0, 0.0, 0.0, [0, 0, 0]
            for i in sc:
                dl = ms[i]["d_lambda"] - bs[i]["d_lambda"]; ds = ms[i]["d_s"] - bs[i]["d_s"]
                sp_ += -(1 - bs[i]["d_s"]) * dl; sa_ += -(1 - bs[i]["d_lambda"]) * ds; ix_ += dl * ds
                dq = ms[i]["hqnr"] - bs[i]["hqnr"]; win[0 if dq > 1e-9 else (1 if dq < -1e-9 else 2)] += 1
                per_scene.append(dict(server=server, run_id=r, baseline=base, scene=i, step=sel_step[r], baseline_step=sel_step[base],
                                      hqnr=ms[i]["hqnr"], d_lambda=ms[i]["d_lambda"], d_s=ms[i]["d_s"],
                                      hqnr_baseline=bs[i]["hqnr"], d_lambda_baseline=bs[i]["d_lambda"], d_s_baseline=bs[i]["d_s"],
                                      delta_hqnr=dq, spectral_contribution=-(1 - bs[i]["d_s"]) * dl, spatial_contribution=-(1 - bs[i]["d_lambda"]) * ds, interaction=dl * ds))
            n = len(sc)
            row.update(n_scenes=n, spectral_contribution=sp_ / n, spatial_contribution=sa_ / n, interaction=ix_ / n,
                       scene_win=win[0], scene_loss=win[1], scene_tie=win[2])
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "run_id", k))
    with open(os.path.join(out, "hqnr_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    if per_scene:
        with open(os.path.join(out, "hqnr_per_scene.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scene[0].keys())); w.writeheader(); w.writerows(per_scene)

    print(f"공통 격자(grid {a.grid}) epoch {sorted(common)[:6]}{'…' if len(common) > 6 else ''} · {len(common)}개 · run {len(rows)}개")
    if trunc:
        print(f"  !! 공통 구간이 짧다 (공통 마지막 epoch {max(common) if common else '-'} vs run 별 마지막 {own_last}) — "
              "진행 중 run 이 섞이면 선택 기회가 아니라 horizon 이 잘린다. 완료 run 끼리 다시 낼 것")
    print(f"{'run':44s} {'orig':>8s} {'common':>8s} {'Δ':>8s} {'plateau':>8s} {'last':>8s}  ckpt")
    for r in rows:
        d = r.get("delta_hqnr_vs_baseline")
        print(f"{r['run_id'][:44]:44s} {r['hqnr_best_original']:.5f} {(r['hqnr_best_common_grid'] or float('nan')):8.5f} "
              f"{(f'{d:+.5f}' if d is not None else '—'):>8s} {(r.get('plateau_mean') or float('nan')):8.5f} {r['hqnr_last']:.5f}  {r['checkpoint_at_common']}")
    print(f"→ {os.path.relpath(out, ROOT)}/ (run_inventory.csv · common_grid_manifest.json · hqnr_comparison.csv"
          + (" · hqnr_per_scene.csv" if per_scene else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
