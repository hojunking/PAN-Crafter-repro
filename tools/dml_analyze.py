"""DML 실행 분석 — peer 별·앙상블 지표, 대응표본 검정, peer 다양성.

  python tools/dml_analyze.py --m0 work_dir/dml_m0                    # M0 다양성 진단만
  python tools/dml_analyze.py --m0 work_dir/dml_m0 --m1 work_dir/dml_m1

계획 11절의 성공 기준을 **대응표본 t-검정**으로 판정한다.
원안의 절대 임계(ΔHQNR ≥ +0.002, ΔSCC ≥ +0.0002)는 시드 잡음(ERGAS 0.8%,
SCC 전체 폭 0.52%) 아래라 잡음에 그냥 통과한다. 이 저장소가 이미 쓰는 대응표본
검정은 대응 SE 0.006~0.043 (비대응 0.129) 으로 검정력이 실제로 있다.

peer 다양성(오차 상관·오라클 이득)도 함께 낸다 — M0 단계에서 DML 헤드룸이
있는지 조기 판정하기 위한 값이며, 정의는 8/20 go/no-go 와 같다.
"""
import argparse
import os
import sys

import numpy as np
import h5py
from scipy.io import loadmat
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.metrics.eval_rr import evaluate          # noqa: E402
from tools.eval_dlpan import scc_dlpan              # noqa: E402

SCALE = {"wv3": 2047.0, "qb": 2047.0, "gf2": 1023.0, "wv2": 2047.0}
CUT, BLK = 21, 32


def load_sr(path):
    sr = loadmat(path)["sr"].astype(np.float64)
    return sr.transpose(0, 2, 3, 1) if sr.shape[1] in (4, 8) else sr


def per_image(sr, gtc, scale):
    """장면별 ERGAS/SAM/SCC. 대응표본 검정을 하려면 평균이 아니라 장면별 값이 필요하다."""
    sl = slice(CUT - 1, -CUT)
    s = sr[:, sl, sl, :]
    E, S, C = [], [], []
    for i in range(len(gtc)):
        m = evaluate(s[i:i + 1], gtc[i:i + 1], scale, BLK)
        E.append(m["ERGAS"][0])
        S.append(m["SAM"][0])
        C.append(scc_dlpan(s[i], gtc[i]))
    return np.array(E), np.array(S), np.array(C)


def diversity(sr_a, sr_b, gt):
    """8/20 go/no-go 와 같은 정의 — 오차 상관, 승률, 픽셀별 오라클 합성."""
    ea, eb = np.abs(sr_a - gt), np.abs(sr_b - gt)
    rho = float(np.corrcoef(ea.ravel(), eb.ravel())[0, 1])
    win_a = float((ea < eb).mean())
    oracle = np.where(ea < eb, sr_a, sr_b)
    return rho, win_a, oracle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m0", required=True, help="대조군 work_dir (lambda=0)")
    ap.add_argument("--m1", default=None, help="mutual work_dir. 없으면 M0 진단만 낸다")
    ap.add_argument("--preset", default="wv3", choices=list(SCALE))
    ap.add_argument("--mat", default="results/reduced_best_val.mat")
    a = ap.parse_args()

    scale = SCALE[a.preset]
    h5 = (f"data/PanCollection/{a.preset.upper()}/reduced_examples_h5/"
          f"test_{a.preset}_multiExm1.h5")
    with h5py.File(h5) as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(CUT - 1, -CUT)
    gtc = gt[:, sl, sl, :]

    runs = {}
    for tag, root in (("M0", a.m0), ("M1", a.m1)):
        if not root:
            continue
        sr = {}
        for peer in ("peerA", "peerB"):
            p = os.path.join(root, peer, a.mat)
            if not os.path.exists(p):
                print(f"  !! 없음: {p}")
                return 1
            sr[peer] = load_sr(p)
        sr["ens"] = (sr["peerA"] + sr["peerB"]) / 2.0
        runs[tag] = sr

    print(f"{'':24}{'ERGAS↓':>10}{'SAM↓':>10}{'SCC↑':>10}")
    res = {}
    for tag, sr in runs.items():
        res[tag] = {}
        for k in ("peerA", "peerB", "ens"):
            E, S, C = per_image(sr[k], gtc, scale)
            res[tag][k] = (E, S, C)
            print(f"  {tag} {k:<18}{E.mean():>10.4f}{S.mean():>10.4f}{C.mean():>10.4f}")

    # ---------------- peer 다양성 = DML 로 얻을 수 있는 상한 ----------------
    print("\npeer 다양성 — DML 헤드룸")
    for tag, sr in runs.items():
        rho, win_a, oracle = diversity(sr["peerA"], sr["peerB"], gt)
        Eo, _, _ = per_image(oracle, gtc, scale)
        base = min(res[tag]["peerA"][0].mean(), res[tag]["peerB"][0].mean())
        print(f"  {tag}  오차 상관 {rho:.4f}   A 승률 {win_a * 100:.1f}%   "
              f"오라클 ERGAS {Eo.mean():.4f}  (단일 최선 대비 {(1 - Eo.mean() / base) * 100:+.2f}%)")
    print("  8/20 go/no-go 기준: 상관 ≤ 0.85 이고 오라클 이득 ≥ +15% 여야 상보성이 있다고 본다.")
    print("  상관 ≥ 0.97 이고 오라클 < +5% 면 M1 에 시간을 쓰기 전에 λ·다양성 설계를 먼저 본다.")

    # ---------------- M1 vs M0 대응표본 검정 ----------------
    if "M1" in res:
        print("\nM1 vs M0 대응표본 t-검정 (20장)")
        print(f"{'':22}{'Δ%':>9}{'p':>10}{'개선 장면':>10}")
        verdict = {}
        for k in ("peerA", "peerB", "ens"):
            for j, nm in enumerate(("ERGAS", "SAM", "SCC")):
                v1, v0 = res["M1"][k][j], res["M0"][k][j]
                better = (v1 > v0) if nm == "SCC" else (v1 < v0)
                d = (v1.mean() / v0.mean() - 1) * 100
                p = stats.ttest_rel(v1, v0).pvalue
                verdict[(k, nm)] = (d, p, int(better.sum()))
                print(f"  {k:<10}{nm:<8}{d:>+8.2f}%{p:>10.5f}{str(int(better.sum())) + '/20':>10}")
        print("  ERGAS·SAM 은 음수가 개선, SCC 는 양수가 개선이다.")

        # 방향 일관성 — 두 peer 모두에서 같은 방향이어야 효과로 인정한다
        print("\n판정")
        for nm in ("ERGAS", "SAM", "SCC"):
            da, _, _ = verdict[("peerA", nm)]
            db, _, _ = verdict[("peerB", nm)]
            good = (da > 0 and db > 0) if nm == "SCC" else (da < 0 and db < 0)
            print(f"  {nm:<8} 두 peer 방향 일관: {'예' if good else '아니오'}"
                  f"  (A {da:+.2f}% / B {db:+.2f}%)")
        print("  대응표본 p < 0.05 이고 두 peer 방향이 일관될 때만 효과로 인정한다.")
        print("  0.8% 미만 차이는 시드 3벌 이상에서 확인한 뒤에만 결론에 넣는다 (CLAUDE.md).")
        print("  ensemble 단독 향상은 DML 성공으로 보지 않는다 (계획 9절 Case D).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
