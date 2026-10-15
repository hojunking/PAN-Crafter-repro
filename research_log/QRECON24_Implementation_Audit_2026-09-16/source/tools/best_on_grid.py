#!/usr/bin/env python
"""평가 격자를 맞춰 best HQNR 을 다시 고른다 — eval_epoch 가 다른 run 끼리 대조할 때.

best 는 "평가된 epoch 중 최대" 라, eval_epoch 5 로 돈 run 은 10 으로 돈 run 보다
후보가 2배여서 미세하게 유리하다. 같은 격자로 다시 골라 맞춘다.

    python tools/best_on_grid.py --grid 10 NF16_P0_W112_D123_WV3_S1234_N2LAST_v1
    python tools/best_on_grid.py --grid 10 "S2W112D123_*"      # glob

근거(2026-09-11 실측, 기존 65 run): 10 격자 재선택 시 best HQNR 중앙 0.00000 ·
평균 -0.00030(판정선 0.0027 의 1/9). 순위는 판정선을 넘는 142쌍 중 2쌍만 뒤집혔고,
그 2쌍 모두 "차이 있음 -> 구분되지 않음" 방향이라 주장이 반대로 뒤집힌 사례는 없었다.
"""
import argparse, csv, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick(run, grid):
    m = os.path.join(ROOT, "work_dir", run, "metrics.csv")
    if not os.path.exists(m):
        return None
    try:
        d = {int(r["epoch"]): float(r["hqnr_official"])
             for r in csv.DictReader(open(m)) if r.get("hqnr_official")}
    except Exception:
        return None
    if not d:
        return None
    g = {e: v for e, v in d.items() if e % grid == 0}
    if not g:
        return None
    b, bg = max(d, key=d.get), max(g, key=g.get)
    st = os.path.join(ROOT, "work_dir", run, "best_state.json")
    rec = json.load(open(st)) if os.path.exists(st) else {}
    return dict(run=run, rec_hqnr=rec.get("best_hqnr", d[b]), rec_ep=rec.get("best_epoch_hqnr", b),
                grid_hqnr=g[bg], grid_ep=bg, n_all=len(d), n_grid=len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+")
    ap.add_argument("--grid", type=int, default=10, help="맞출 평가 격자 (기본 10)")
    a = ap.parse_args()

    names = []
    for p in a.pattern:
        names += [os.path.basename(x) for x in
                  sorted(glob.glob(os.path.join(ROOT, "work_dir", p)))] or [p]
    rows = [r for r in (pick(n, a.grid) for n in dict.fromkeys(names)) if r]
    if not rows:
        sys.exit("대상 run 없음 (metrics.csv 에 hqnr_official 이 있어야 한다)")
    print(f"{'run':<46}{'기록 best':>11}{'ep':>6}{'%d격자' % a.grid:>10}{'ep':>6}{'Δ':>10}")
    for r in rows:
        print(f"{r['run'][:46]:<46}{r['rec_hqnr']:>11.5f}{r['rec_ep']:>6}"
              f"{r['grid_hqnr']:>10.5f}{r['grid_ep']:>6}{r['grid_hqnr']-r['rec_hqnr']:>+10.5f}")
    print(f"\n  대조표에는 '{a.grid}격자' 열을 쓴다 — eval_epoch 가 다른 run 끼리 공정해진다.")


if __name__ == "__main__":
    main()
