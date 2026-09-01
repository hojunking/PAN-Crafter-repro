#!/usr/bin/env python
"""plateau(동일 epoch) HQNR 병기 — 확정 규칙 대응 도구.

best-checkpoint 하나로 실행을 비교하면 **선택 편향**이 들어간다. HQNR 곡선은
중후반에 요동치므로, best 값의 차이가 동급 band 안일 때는 그 차이가 실력인지
선택 운인지 구분되지 않는다. 그래서 이 저장소는 best 와 **plateau 평균을 함께**
보고하도록 정해져 있다.

이 도구는 각 실행의 metrics.csv(hqnr_official 열)에서
  - best      : 실제 선택된 값 (best_state.json)
  - plateau   : 후반부 eval epoch 들의 평균 ± 표준편차 (기본 마지막 40%)
  - same-epoch: 모든 실행에 공통으로 존재하는 eval epoch 들만 골라 낸 평균
을 계산해 한 표로 낸다.

  python tools/plateau_report.py S1_T00_W160_D122_MS2 S1_A0_W128_D124_MS2 ...
  python tools/plateau_report.py --json out.json  <runs...>
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAND = 0.011      # HQNR 동급 band (확정 2σ)


def curve(tag):
    """{epoch: hqnr_official} — 값이 있는 eval epoch 만."""
    p = os.path.join(ROOT, "work_dir", tag, "metrics.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            v = (row.get("hqnr_official") or "").strip()
            if v:
                try:
                    out[int(row["epoch"])] = float(v)
                except ValueError:
                    pass
    return out


def best_of(tag):
    p = os.path.join(ROOT, "work_dir", tag, "best_state.json")
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    return d.get("best_hqnr"), d.get("best_epoch_hqnr")


def mean_sd(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tail", type=float, default=0.4, help="plateau 로 볼 후반 비율 (기본 0.4)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    curves = {t: curve(t) for t in a.runs}
    have = [t for t in a.runs if curves[t]]
    missing = [t for t in a.runs if not curves[t]]
    if missing:
        print(f"[plateau] metrics 없음: {' '.join(missing)}", file=sys.stderr)
    if not have:
        return 1
    common = set.intersection(*[set(curves[t]) for t in have])
    common_tail = sorted(common)[-max(1, int(len(common) * a.tail)):] if common else []

    rows = []
    print(f"{'run':34s} {'best':>8s} {'@ep':>5s} {'plateau(후반)':>16s} {'공통ep 평균':>12s}")
    for t in have:
        c = curves[t]
        eps = sorted(c)
        tail = eps[-max(1, int(len(eps) * a.tail)):]
        pm, ps = mean_sd([c[e] for e in tail])
        cm, _ = mean_sd([c[e] for e in common_tail if e in c])
        b, be = best_of(t)
        rows.append({"run": t, "best": b, "best_epoch": be, "plateau_mean": pm,
                     "plateau_sd": ps, "common_epoch_mean": cm,
                     "n_eval": len(eps), "n_common": len(common_tail)})
        print(f"{t:34s} {b if b is None else round(b,4):>8} {be if be else '-':>5} "
              f"{pm:.4f}±{ps:.4f}  {cm:>12.4f}")

    if len(rows) >= 2:
        print(f"\n[plateau] 공통 eval epoch {len(common_tail)}개 기준 (선택 편향 없음)")
        base = rows[0]
        for r in rows[1:]:
            db = (r["best"] or 0) - (base["best"] or 0)
            dp = r["common_epoch_mean"] - base["common_epoch_mean"]
            flip = "  ← best 와 부호 반대" if db * dp < 0 else ""
            print(f"  {r['run']} vs {base['run']}: best Δ{db:+.4f} · 공통ep Δ{dp:+.4f}"
                  f"{'  (둘 다 band 안 — 구분되지 않음)' if max(abs(db), abs(dp)) <= BAND else ''}"
                  f"{flip}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
