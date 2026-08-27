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
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s   # noqa: E402
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


def per_image_fr(sr, lms, pan, preset, wald, indices):
    """장면별 D_lambda / D_s / HQNR. 계획 11절의 필수 판정이 HQNR 대응검정이다."""
    bit = int(round(np.log2(SCALE[preset] + 1)))
    dl, ds, hq = [], [], []
    for i in indices:
        fused = np.clip(sr[i], 0.0, float(2 ** bit))
        a = d_lambda_k(fused, lms[i], preset, 4, BLK, wald)
        b = d_s(fused, lms[i], pan[i], 4, BLK, wald)
        dl.append(a); ds.append(b); hq.append((1 - a) * (1 - b))
    return np.array(dl), np.array(ds), np.array(hq)


def diversity(sr_a, sr_b, gt):
    """peer 다양성 — **픽셀 단위** (8/18·8/20 정의: 밴드 평균 오차 지도에서 픽셀마다 승자 선택).

    [자가검증 2026-08-27 교정] 이전 구현은 밴드-원소 단위(np.where(ea<eb))였고, 이는
    8/20 go/no-go 의 픽셀 단위 정의와 달라 게이트 임계(≥0.97/<+5%)와 스케일이 어긋났다.
    실측: dml_m0 에서 원소 단위 ρ=0.9557/오라클 +8.16% vs 픽셀 단위 ρ=0.9797/+4.31% —
    정의 선택만으로 M1 진행/보류 판정이 뒤집힌다. 8/20 임계와 비교하려면 픽셀 단위여야 한다.
    원소 단위 값도 참고로 함께 돌려준다.
    """
    ea, eb = np.abs(sr_a - gt), np.abs(sr_b - gt)
    pa, pb = ea.mean(axis=-1), eb.mean(axis=-1)          # (N,H,W) 픽셀 오차 지도
    rho_px = float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])
    win_px = float((pa < pb).mean())
    oracle = np.where((pa < pb)[..., None], sr_a, sr_b)  # 픽셀 승자를 전 밴드에 적용
    rho_el = float(np.corrcoef(ea.ravel(), eb.ravel())[0, 1])
    return rho_px, win_px, oracle, rho_el


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m0", required=True, help="대조군 work_dir (lambda=0)")
    ap.add_argument("--m1", default=None, help="mutual work_dir. 없으면 M0 진단만 낸다")
    ap.add_argument("--preset", default="wv3", choices=list(SCALE))
    ap.add_argument("--mat", default="results/reduced_best_val.mat")
    ap.add_argument("--mat-fr", default="results/full_best_val.mat")
    ap.add_argument("--fr-indices", default="12-19",
                    help="논문 대조 구간. all 이면 전체 20장 (HQNR-all)")
    ap.add_argument("--no-fr", action="store_true", help="FR 생략 (PANCRAFTER_DLPAN 없을 때)")
    a = ap.parse_args()

    scale = SCALE[a.preset]
    h5 = (f"data/PanCollection/{a.preset.upper()}/reduced_examples_h5/"
          f"test_{a.preset}_multiExm1.h5")
    with h5py.File(h5) as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(CUT - 1, -CUT)
    gtc = gt[:, sl, sl, :]

    wald = lms_f = pan_f = None
    if not a.no_fr:
        wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", ""))
        fh5 = (f"data/PanCollection/{a.preset.upper()}/full_examples_h5/"
               f"test_{a.preset}_OrigScale_multiExm1.h5")
        with h5py.File(fh5) as f:
            lms_f = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
            pan_f = np.asarray(f["pan"], dtype=np.float64)[:, 0]
        if a.fr_indices == "all":
            fr_idx = list(range(len(lms_f)))
        else:
            lo, hi = a.fr_indices.split("-"); fr_idx = list(range(int(lo), int(hi) + 1))

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
        if not a.no_fr:
            for peer in ("peerA", "peerB"):
                pf = os.path.join(root, peer, a.mat_fr)
                if not os.path.exists(pf):
                    print(f"  !! FR mat 없음: {pf} — --no-fr 로 생략하거나 export 를 확인할 것")
                    return 1
                sr["fr_" + peer] = load_sr(pf)
            sr["fr_ens"] = (sr["fr_peerA"] + sr["fr_peerB"]) / 2.0
        runs[tag] = sr

    hdr = f"{'':24}{'ERGAS↓':>10}{'SAM↓':>10}{'SCC↑':>10}"
    if not a.no_fr:
        hdr += f"{'D_λ↓':>10}{'D_s↓':>10}{'HQNR↑':>10}"
        print(f"  (FR 구간: {a.fr_indices})")
    print(hdr)
    res = {}
    for tag, sr in runs.items():
        res[tag] = {}
        for k in ("peerA", "peerB", "ens"):
            E, S, C = per_image(sr[k], gtc, scale)
            res[tag][k] = (E, S, C)
            line = f"  {tag} {k:<18}{E.mean():>10.4f}{S.mean():>10.4f}{C.mean():>10.4f}"
            if not a.no_fr:
                DL, DS, HQ = per_image_fr(sr["fr_" + k if k != "ens" else "fr_ens"],
                                          lms_f, pan_f, a.preset, wald, fr_idx)
                res[tag][k] = (E, S, C, DL, DS, HQ)
                line += f"{DL.mean():>10.4f}{DS.mean():>10.4f}{HQ.mean():>10.4f}"
            print(line)

    # ---------------- peer 다양성 = DML 로 얻을 수 있는 상한 ----------------
    print("\npeer 다양성 — DML 헤드룸")
    for tag, sr in runs.items():
        rho, win_a, oracle, rho_el = diversity(sr["peerA"], sr["peerB"], gt)
        Eo, _, _ = per_image(oracle, gtc, scale)[:3]
        base = min(res[tag]["peerA"][0].mean(), res[tag]["peerB"][0].mean())
        print(f"  {tag}  오차 상관(픽셀) {rho:.4f}   A 승률 {win_a * 100:.1f}%   "
              f"오라클 ERGAS {Eo.mean():.4f}  (단일 최선 대비 {(1 - Eo.mean() / base) * 100:+.2f}%)"
              f"   [참고: 원소 단위 ρ={rho_el:.4f}]")
    print("  8/20 go/no-go 기준: 상관 ≤ 0.85 이고 오라클 이득 ≥ +15% 여야 상보성이 있다고 본다.")
    print("  상관 ≥ 0.97 이고 오라클 < +5% 면 M1 에 시간을 쓰기 전에 λ·다양성 설계를 먼저 본다.")

    # ---------------- M1 vs M0 대응표본 검정 ----------------
    if "M1" in res:
        n_rr = len(res["M1"]["peerA"][0])
        fr_txt = "" if a.no_fr else f", FR {len(res['M1']['peerA'][3])}장"
        print(f"\nM1 vs M0 대응표본 t-검정 (RR {n_rr}장{fr_txt})")
        print(f"{'':22}{'Δ%':>9}{'p':>10}{'개선 장면':>10}")
        verdict = {}
        for k in ("peerA", "peerB", "ens"):
            names = ("ERGAS", "SAM", "SCC") if a.no_fr else \
                    ("ERGAS", "SAM", "SCC", "D_lambda", "D_s", "HQNR")
            for j, nm in enumerate(names):
                v1, v0 = res["M1"][k][j], res["M0"][k][j]
                hi_good = nm in ("SCC", "HQNR")     # 높을수록 좋은 지표
                better = (v1 > v0) if hi_good else (v1 < v0)
                d = (v1.mean() / v0.mean() - 1) * 100
                p = stats.ttest_rel(v1, v0).pvalue
                verdict[(k, nm)] = (d, p, int(better.sum()))
                n = len(v1)
                print(f"  {k:<10}{nm:<10}{d:>+8.2f}%{p:>10.5f}"
                      f"{str(int(better.sum())) + '/' + str(n):>10}")
        print("  ERGAS·SAM·D_lambda·D_s 는 음수가 개선, SCC·HQNR 은 양수가 개선이다.")

        # 방향 일관성 — 두 peer 모두에서 같은 방향이어야 효과로 인정한다
        print("\n판정")
        for nm in (("ERGAS", "SAM", "SCC") if a.no_fr else
                   ("ERGAS", "SAM", "SCC", "D_lambda", "D_s", "HQNR")):
            da, _, _ = verdict[("peerA", nm)]
            db, _, _ = verdict[("peerB", nm)]
            hi_good = nm in ("SCC", "HQNR")
            # [자가검증 2026-08-27 교정] 계획 §11 은 '같은 방향'이다. 이전 구현은 '양쪽 개선'만
            # 참으로 놓아, 두 peer 가 일관되게 악화(§9 Case B 신호)해도 '아니오'로 가려졌다.
            same = (da > 0) == (db > 0)
            imp = (da > 0) if hi_good else (da < 0)
            direction = ("개선" if imp else "악화") if same else "—"
            print(f"  {nm:<8} 두 peer 방향 일관: {'예(' + direction + ')' if same else '아니오'}"
                  f"  (A {da:+.2f}% / B {db:+.2f}%)")
        print("  대응표본 p < 0.05 이고 두 peer 방향이 일관될 때만 효과로 인정한다.")
        print("  0.8% 미만 차이는 시드 3벌 이상에서 확인한 뒤에만 결론에 넣는다 (CLAUDE.md).")
        print("  ensemble 단독 향상은 DML 성공으로 보지 않는다 (계획 9절 Case D).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
