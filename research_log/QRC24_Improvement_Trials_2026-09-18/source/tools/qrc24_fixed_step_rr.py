#!/usr/bin/env python
"""고정 step(50K) 공통 비교 — 시트의 ERGAS 차이가 profile 때문인지 **checkpoint 선택 때문인지** 가른다.

    python tools/qrc24_fixed_step_rr.py                 # 이 서버의 완료 QRC24 run 전부 (CPU, run 당 약 20 초)
    python tools/qrc24_fixed_step_rr.py --run <run> [--mat reduced_last.mat] [--out <csv>]

무엇을 하나: 각 run 의 **이미 저장된** `results/reduced_last.mat`(= exact 50K, 선택과 무관한 공통 지점) 을 시트와 **같은 경로**(`gspread_upload._rr` = tools/eval_dlpan 규약: crop 20:-21, peak 2047, SCC=SCC.m, SSIM Gaussian 11×11)로
다시 잰다. GPU 추론이 없다 — 저장된 출력만 읽는다. 기존 파일은 덮어쓰지 않고 `work_dir/_qrecon24/fixed_step_rr.csv` 에만 쓴다.

왜 필요한가 (s1 13 run 실측): HQNR 로 고른 checkpoint 에서 재면 profile 간 ERGAS 폭이 0.0718 인데, **같은 50K 에서 재면 0.0073 으로 줄었다**. 즉 시트 ERGAS 분산의 대부분이 "HQNR 선택기가 몇 step 에서 멈췄는가" 다
(선택 step 과 시트 ERGAS 의 상관 −0.967). 각 서버가 자기 격자에서 같은 현상을 확인하면, 그 캠페인의 ERGAS 결론을 선택 artifact 와 분리해 보고할 수 있다.

주의: 50K 는 HQNR 이 정점이 아닌 지점이라 HQNR 은 대체로 낮아진다. 이 표는 **profile 비교용 공통 지점**이지 새 판정 checkpoint 를 고르는 것이 아니다(판정은 계획의 selector 로 한다).
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _gu():
    spec = importlib.util.spec_from_file_location("_gu_fx", os.path.join(ROOT, "gspread", "gspread_upload.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["_gu_fx"] = m; spec.loader.exec_module(m); return m


def _json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def runs_here(prefix="PAKD50_QRC24_"):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "work_dir", prefix + "*"))):
        if os.path.exists(os.path.join(d, "results", "reduced_best_hqnr.mat")):
            out.append(os.path.basename(d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None); ap.add_argument("--mat", default="reduced_last.mat"); ap.add_argument("--out", default=os.path.join("work_dir", "_qrecon24", "fixed_step_rr.csv"))
    ap.add_argument("--dataset", default="wv3")
    a = ap.parse_args()
    gu = _gu(); targets = [a.run] if a.run else runs_here()
    rows = []
    print(f"[fixed-step] {len(targets)} run · 공통 지점 {a.mat} (저장된 출력만 읽는다, GPU 추론 없음)")
    for r in targets:
        wd = os.path.join(ROOT, "work_dir", r); mat = os.path.join(wd, "results", a.mat)
        if not os.path.exists(mat):
            print(f"   {r[:60]:<60} {a.mat} 없음 — 건너뜀"); continue
        sel = _json(os.path.join(wd, "best_raw_meta.json")); lm = _json(os.path.join(wd, "last_meta.json"))
        fr = _json(os.path.join(wd, "results", "fr_mat20.json"))
        t0 = time.time(); m = gu._rr(mat, a.dataset); sec = time.time() - t0
        row = dict(run=r, profile=(r.split("_QRC24_")[1].split("_W104")[0] if "_QRC24_" in r else ""), seed=(r.split("_T0_S")[1].split("_")[0] if "_T0_S" in r else ""),
                   fixed_step=lm.get("step"), fixed_ergas=m.get("ergas"), fixed_scc=m.get("scc"), fixed_psnr=m.get("psnr"), fixed_sam=m.get("sam"),
                   fixed_q8=m.get("q2n", m.get("q8")), fixed_ssim=m.get("ssim"), selected_step=sel.get("step"), sheet_hqnr=fr.get("hqnr"), seconds=round(sec, 1))
        rows.append(row)
        print(f"   {r.replace('PAKD50_QRC24_', '')[:44]:<44} 선택 step {str(row['selected_step']):<6} → 고정 {str(row['fixed_step']):<6} ERGAS {row['fixed_ergas']:.4f} · {sec:.0f}s")
    if not rows:
        print("   잰 것이 없다"); return 1
    p = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    es = sorted(x["fixed_ergas"] for x in rows if x["fixed_ergas"])
    print(f"[fixed-step] {len(rows)} run · 고정 {a.mat} ERGAS 폭 {es[-1] - es[0]:.4f} ({(es[-1] - es[0]) / (sum(es) / len(es)) * 100:.2f}%) · 최소 {es[0]:.4f} 최대 {es[-1]:.4f} → {a.out}")
    print("   해석: 이 폭이 판정선 0.8% 보다 작으면 이 서버의 profile 간 ERGAS 차이는 **선택 step 의 부수효과**로 보아야 한다(공통 지점에서 구분되지 않는다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
