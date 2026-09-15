"""B0/B1 결과 집계 (계획 §13: checkpoint_metrics.csv · paired_results.csv). 선택 asset 의 모든 metric 은 같은 checkpoint 에서; full precision paired Δ (B1 − B0, seed 별); plain GT L1(DISC/CONF)·C 변화는 sample_metrics 에서."""
import json, os
import numpy as np, pandas as pd
from tools.dcr12 import common as C
PLATEAU = [45450, 46460, 47470, 48480, 49490, 50000]


def run_metrics(ck, seed):
    rd = C.run_dir(ck, seed); out = dict(case=ck, alias=C.CASES[ck], seed=seed, run=C.run_name(ck, seed), status=("complete" if C.run_complete(ck, seed) else "incomplete"), trained_on=C.trained_on(ck, seed),
                                        finished_at=(open(os.path.join(rd, "meta", "finished_at.txt")).read().strip() if os.path.exists(os.path.join(rd, "meta", "finished_at.txt")) else None))
    if not C.run_complete(ck, seed):
        return out
    meta = C.load_json(os.path.join(rd, "best_hqnr_meta.json"), {}); fr = C.load_json(os.path.join(rd, "results", "fr_mat20.json"), {}); cm = pd.read_csv(os.path.join(rd, "checkpoint_metrics.csv"))
    col = "raw_original.hqnr"; last = cm[cm.step == 50000]; pl = cm[cm.step.isin(PLATEAU)]
    out.update(selected_step=meta.get("step"), selected_hqnr=meta.get("hqnr"), selected_fscc=meta.get("fscc"), selected_rr_ergas=meta.get("rr_ergas"), selected_rr_scc=meta.get("rr_scc"), fr20_hqnr=fr.get("hqnr"), fr20_d_lambda=fr.get("d_lambda"), fr20_d_s=fr.get("d_s"),
               per_scene_hqnr=fr.get("per_scene_hqnr"), numerical_max_hqnr=float(cm[col].max()), numerical_max_step=int(cm.loc[cm[col].idxmax(), "step"]), plateau_mean_hqnr=(float(pl[col].mean()) if len(pl) else None), plateau_n=int(len(pl)),
               last_hqnr=(float(last[col].iloc[0]) if len(last) else None), n_candidates=int(len(cm)), train_hours=_hours(rd))
    return out


def _hours(rd):
    import time
    try:
        f = lambda n: time.mktime(time.strptime(open(os.path.join(rd, "meta", n)).read().strip()[:19], "%Y-%m-%dT%H:%M:%S")); return round((f("finished_at.txt") - f("started_at.txt")) / 3600, 3)
    except Exception:
        return None


def main(profile=False):
    with C.Stage("RESULTS", "checkpoint metrics / paired B1−B0"):
        rows = [run_metrics(ck, seed) for seed in C.SEEDS for ck in ("B0", "B1")]; C.dump_json(os.path.join(C.CAMP, "run_metrics.json"), rows)
        cmrows = []
        for r in rows:
            if r["status"] == "complete":
                cm = pd.read_csv(os.path.join(C.run_dir(r["case"], r["seed"]), "checkpoint_metrics.csv"))
                for _, x in cm.iterrows():
                    cmrows.append(dict(run=r["run"], case=r["case"], seed=r["seed"], step=int(x.step), raw_original_hqnr=float(x["raw_original.hqnr"]), raw_original_d_lambda=float(x["raw_original.d_lambda"]), raw_original_d_s=float(x["raw_original.d_s"]), raw_original_fscc=float(x["raw_original.fscc"])))
        C.write_csv(os.path.join(C.CAMP, "checkpoint_metrics.csv"), cmrows)
        sm = pd.read_csv(os.path.join(C.CAMP, "sample_metrics.csv")) if os.path.exists(os.path.join(C.CAMP, "sample_metrics.csv")) else pd.DataFrame(); prs = []
        for seed in C.SEEDS:
            b0 = next(r for r in rows if r["case"] == "B0" and r["seed"] == seed); b1 = next(r for r in rows if r["case"] == "B1" and r["seed"] == seed)
            p = dict(seed=seed, B0=b0["run"], B1=b1["run"], status=("paired" if b0["status"] == b1["status"] == "complete" else f"B0 {b0['status']} / B1 {b1['status']}"))
            if p["status"] == "paired":
                for k in ("selected_hqnr", "plateau_mean_hqnr", "last_hqnr", "numerical_max_hqnr", "fr20_d_lambda", "fr20_d_s", "selected_fscc", "selected_rr_ergas"):
                    p[f"d_{k}"] = (b1[k] - b0[k]) if (b1.get(k) is not None and b0.get(k) is not None) else None
                if b0.get("per_scene_hqnr") and b1.get("per_scene_hqnr"):
                    d = np.array(b1["per_scene_hqnr"]) - np.array(b0["per_scene_hqnr"]); bs = C.block_bootstrap(d, np.arange(len(d))); p.update(d_hqnr_scene_mean=float(d.mean()), d_hqnr_scene_ci95=bs["ci95"], d_hqnr_scene_pos=int((d > 0).sum()), n_scenes=int(len(d)))
                for pn in ("DISC", "CONF", "CAL"):
                    a = sm[(sm.pipeline == C.pipe_key("S", "B0", seed, "best_hqnr")) & (sm.panel == pn)].set_index("sample_id"); b = sm[(sm.pipeline == C.pipe_key("S", "B1", seed, "best_hqnr")) & (sm.panel == pn)].set_index("sample_id")
                    if len(a) and len(b):
                        j = a.join(b, lsuffix="_b0", rsuffix="_b1", how="inner"); p[f"d_R_plain_{pn}"] = float((j.R_plain_b1 - j.R_plain_b0).mean()); p[f"d_C_median_{pn}"] = float(j.C_b1.median() - j.C_b0.median()); p[f"C_median_B0_{pn}"] = float(j.C_b0.median()); p[f"C_median_B1_{pn}"] = float(j.C_b1.median())
            prs.append(p)
        C.write_csv(os.path.join(C.CAMP, "paired_results.csv"), prs)
        for p in prs:
            print(f"[results] seed {p['seed']}: {p['status']}" + (f" · ΔHQNR(selected) {p['d_selected_hqnr']:+.5f} · plateau {p['d_plateau_mean_hqnr']:+.5f} · scene CI {p.get('d_hqnr_scene_ci95')}" if p["status"] == "paired" else ""), flush=True)
