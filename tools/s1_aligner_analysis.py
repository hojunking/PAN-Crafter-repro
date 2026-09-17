#!/usr/bin/env python
"""s1 전용 Aligner 분석 — 크기 진단 · 알려진 이동에 대한 반응(q_size) · 네 모드 비교
(계획 research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §7, protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918).

    python tools/s1_aligner_analysis.py --run <run> [--split rr|fr] [--scenes 0-19] [--device cuda]
    python tools/s1_aligner_analysis.py --assets            # 계획 §7.1 의 네 자산을 순서대로
    python tools/s1_aligner_analysis.py --modes --run <run> # 네 모드(A_ON/ZERO/NOA/CROP64_MED) RR·FR 쌍 지표만

무엇을 재는가 (§7.2–§7.4)
  크기 패널   RR256: 64×16개 · 128×4개 · full(256) · (256 crop 은 full 과 같아 중복 제외)
              FR512: 64×64개 · 128×16개 · 256×4개 · full(512)
              crop origin 은 stride-4 좌표의 non-overlap 격자. M(=bicubic×4)은 **전체에서 한 번** 만들어 같은 좌표로 crop 한다(crop 마다 재확대하지 않는다).
              aligner margin4 는 crop 마다 정확히 한 번 적용된다(PAModel._view).
  probe       AXIS16 = 반경 0.25/0.5/1/2 HR px × (±y, ±x) 네 축. **원 PAN 전체**를 warp 한 뒤 같은 좌표의 crop 을 읽는다. M 은 이동시키지 않는다.
              q_size = (1/2N_pr) Σ_j ‖ĉ_j + ε_j − ĉ_0‖₁ — 사후 측정값이며 학습 q cache/q_ref 와 **다른 값**이다(섞지 않는다).
  모드        A_ON / A_ZERO_WARP / A_BYPASS_RAW / A_CROP64_MED 의 같은 checkpoint RR·FR 지표(tools/noa_eval.py 와 같은 공식 경로).

학습하지 않는다. GT 나 장면 점수를 보고 shift·crop 을 고르지 않는다(§7.4).
산출물: work_dir/<run>/results/aligner_eval_v2/<sha16>/analysis/{size_probe.csv, size_probe.json, modes.json}
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kdv.eval_modes import CallCounter, consensus_delta, state_hash  # noqa: E402
from tools.noa_eval import PROTOCOL_ID, EVAL_DIR, _load, _json, sheet_checkpoint, sha256_file, evaluate_run, FR_H5  # noqa: E402

PROBE_RADII = (0.25, 0.5, 1.0, 2.0)                       # 계획 §7.3 = 기존 AXIS16
PROBE_AXES = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))    # ±y, ±x
SIZES = {"rr": (64, 128), "fr": (64, 128, 256)}           # full 은 따로 (§7.2: RR 의 256 crop 은 full 과 같아 제외)
FULL = {"rr": 256, "fr": 512}
S1_ASSETS = ("PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1", "PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S3407_FRESH50_v1",
             "PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1")     # §7.1 (T0 는 --teacher 로 따로)


def probes():
    """AXIS16: (radius, (uy, ux)) 16 개 — ε_j = radius × 축."""
    return [(r, (r * a[0], r * a[1])) for r in PROBE_RADII for a in PROBE_AXES]


def crop_origins(n, size):
    return [(y, x) for y in range(0, n - size + 1, size) for x in range(0, n - size + 1, size)]


def analyse_scene(model, pan, ms, split, device):
    """장면 하나: 크기별 crop 의 native Δ̂ 와 probe 반응 q_size. pan [1,1,N,N], ms [1,8,N/4,N/4]."""
    import torch
    from pa.warp import warp_pan
    N = pan.shape[-1]
    ms_base = torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic")      # M 은 전체에서 한 번 (§7.2)
    rows = []
    panels = [(s, crop_origins(N, s)) for s in SIZES[split]] + [(N, [(0, 0)])]
    warped = {}
    for j, (r, eps) in enumerate(probes()):
        d = torch.tensor([[float(eps[0]), float(eps[1])]], device=pan.device)
        warped[j] = warp_pan(pan, d)                                                    # 원 PAN 전체를 warp (§7.3)
    for size, origins in panels:
        for u, (y, x) in enumerate(origins):
            sl = (slice(None), slice(None), slice(y, y + size), slice(x, x + size))
            m_c = ms_base[sl]
            c0 = model.predict_delta(pan[sl], m_c)[0]
            acc = 0.0
            for j, (r, eps) in enumerate(probes()):
                cj = model.predict_delta(warped[j][sl], m_c)[0]
                acc += float((cj + torch.tensor([eps[0], eps[1]], device=cj.device) - c0).abs().sum())
            q = acc / (2.0 * len(probes()))
            rows.append(dict(size=int(size), crop_index=int(u), y0=int(y), x0=int(x), is_full=bool(size == N),
                             dy=float(c0[0]), dx=float(c0[1]), q_size=float(q)))
    return rows


def run_size_probe(run, split, scenes, device):
    import torch, yaml
    from main import import_class
    efp = _load(os.path.join("tools", "eval_fr_paperset.py"), "_efp_s1a")
    wd = os.path.join(ROOT, "work_dir", run); cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    ck, ckf, meta = sheet_checkpoint(run); sha = sha256_file(ckf)
    m, _f, how = efp.build(cfg, wd, ck); dev = torch.device(device); m = m.to(dev).eval(); h0 = state_hash(m)
    fa = dict(cfg["test_reduced_feeder_args"] if split == "rr" else cfg["test_full_feeder_args"])
    if split == "fr":
        fa["dataroot"] = os.path.join(ROOT, FR_H5)
    ds = import_class(cfg["feeder"])(**fa)
    out, t0 = [], time.time()
    with CallCounter(m) as cc, torch.no_grad():
        for i in scenes:
            item = ds[i]; lms, ms, lpan, pan = item[1:] if split == "rr" else item
            rows = analyse_scene(m, pan.unsqueeze(0).to(dev), ms.unsqueeze(0).to(dev), split, dev)
            for r in rows:
                r["scene"] = int(i)
            out += rows
            print(f"   scene {i:>2}: full Δ=({rows[-1]['dy']:+.4f},{rows[-1]['dx']:+.4f}) q_size {rows[-1]['q_size']:.4f} · crop 수 {len(rows)}")
    assert state_hash(m) == h0, "분석 중 모델 state 가 변했다 (V06)"
    od = os.path.join(wd, "results", EVAL_DIR, sha[:16], "analysis"); os.makedirs(od, exist_ok=True)
    keys = ["scene", "size", "crop_index", "y0", "x0", "is_full", "dy", "dx", "q_size"]
    with open(os.path.join(od, f"size_probe_{split}.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in out:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    summ = summarise(out, split)
    json.dump(dict(protocol_id=PROTOCOL_ID, run=run, checkpoint=ck, checkpoint_sha256=sha, step=meta.get("step"), split=split, forward=how,
                   scenes=[int(s) for s in scenes], probes=[dict(radius=r, eps=list(e)) for r, e in probes()], n_probe=len(probes()),
                   panels={str(s): len(crop_origins(FULL[split], s)) for s in SIZES[split]}, aligner_calls=cc.aligner, warp_calls=cc.warp,
                   seconds=round(time.time() - t0, 1), evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summ, rows=out),
              open(os.path.join(od, f"size_probe_{split}.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[s1a] {run} {split}: {len(out)} crop · {time.time() - t0:.0f}s → {os.path.relpath(od, ROOT)}")
    for s, v in summ["by_size"].items():
        print(f"   {s:>4} px: q_size 중앙 {v['q_size_median']:.4f} (평균 {v['q_size_mean']:.4f}, p90 {v['q_size_p90']:.4f}) · |Δ| 중앙 ({v['abs_dy_median']:.4f},{v['abs_dx_median']:.4f}) · crop 간 표준편차 ({v['dy_std']:.4f},{v['dx_std']:.4f})")
    print(f"   full 과 64-crop 중앙값의 차: dy {summ['full_minus_crop64_median']['dy']:+.4f} · dx {summ['full_minus_crop64_median']['dx']:+.4f} (장면 평균)")
    return summ


def summarise(rows, split):
    by = {}
    for s in sorted({r["size"] for r in rows}):
        sub = [r for r in rows if r["size"] == s]
        q = np.array([r["q_size"] for r in sub]); dy = np.array([r["dy"] for r in sub]); dx = np.array([r["dx"] for r in sub])
        by[str(s)] = dict(n=len(sub), q_size_mean=float(q.mean()), q_size_median=float(np.median(q)), q_size_p90=float(np.percentile(q, 90)),
                          abs_dy_median=float(np.median(np.abs(dy))), abs_dx_median=float(np.median(np.abs(dx))),
                          dy_std=float(dy.std(ddof=1)) if len(dy) > 1 else 0.0, dx_std=float(dx.std(ddof=1)) if len(dx) > 1 else 0.0)
    full = FULL[split]; diffs = []
    for sc in sorted({r["scene"] for r in rows}):
        f = [r for r in rows if r["scene"] == sc and r["size"] == full]
        c = [r for r in rows if r["scene"] == sc and r["size"] == 64]
        if f and c:
            diffs.append((f[0]["dy"] - float(np.median([r["dy"] for r in c])), f[0]["dx"] - float(np.median([r["dx"] for r in c]))))
    d = np.array(diffs) if diffs else np.zeros((1, 2))
    return dict(by_size=by, full_minus_crop64_median=dict(dy=float(d[:, 0].mean()), dx=float(d[:, 1].mean()), n_scenes=len(diffs)),
                note="q_size 는 사후 반응 측정값이고 학습 q cache·q_ref 와 다른 값이다 (§7.3). 작다고 절대 정합이 정확하다는 뜻이 아니다 (§7.5).")


def run_modes(run, device):
    """네 모드의 같은 checkpoint RR·FR 지표 (§7.4). 전수 평가에서 나온 ON/NOA 는 identity 가 맞으면 재사용된다."""
    rec = evaluate_run(run, modes=("A_ON", "A_ZERO_WARP", "A_BYPASS_RAW", "A_CROP64_MED"), device=device)
    ck, ckf, _ = sheet_checkpoint(run); sha = sha256_file(ckf)
    od = os.path.join(ROOT, "work_dir", run, "results", EVAL_DIR, sha[:16], "analysis"); os.makedirs(od, exist_ok=True)
    json.dump(rec, open(os.path.join(od, "modes.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[s1a] 네 모드 → {os.path.relpath(od, ROOT)}/modes.json")
    for mode, v in rec["modes"].items():
        rr, fr = v.get("rr") or {}, v.get("fr") or {}
        print(f"   {mode:<14} RR ERGAS {rr.get('ergas', float('nan')):.4f} · FR HQNR {fr.get('hqnr_raw', float('nan')):.4f}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=S1_ASSETS[0]); ap.add_argument("--assets", action="store_true")
    ap.add_argument("--split", default="rr", choices=("rr", "fr")); ap.add_argument("--scenes", default="0-4")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--modes", action="store_true")
    a = ap.parse_args()
    sc = []
    for part in a.scenes.split(","):
        if "-" in part:
            x, y = part.split("-"); sc += list(range(int(x), int(y) + 1))
        elif part.strip():
            sc.append(int(part))
    runs = list(S1_ASSETS) if a.assets else [a.run]
    for r in runs:
        if not os.path.exists(os.path.join(ROOT, "work_dir", r, "meta", "config.yaml")):
            print(f"[s1a] 건너뜀(없음): {r}"); continue
        print(f"[s1a] {r}")
        if a.modes:
            run_modes(r, a.device)
        else:
            run_size_probe(r, a.split, sc, a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
