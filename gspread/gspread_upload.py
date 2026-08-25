#!/usr/bin/env python
"""실험 결과를 Google Sheets 로 올린다.

    python gspread/gspread_upload.py paper_ln                # 한 실행
    python gspread/gspread_upload.py "paper*" --profile      # 여러 개 + 비용 측정
    python gspread/gspread_upload.py --all --dry-run         # 올리지 않고 표만 확인

지표는 tools/metrics/ 의 DLPan 프로토콜 구현을 그대로 쓴다. 학습 중 metrics.csv 값이
아니라 .mat 을 다시 평가한 값이라, 논문 Table 과 비교 가능한 수치다.

params 는 항상 계산한다(빠르다). FLOPs·추론시간·메모리는 --profile 일 때만 재고,
한 번 잰 값은 gspread/_profile_cache.json 에 저장해 재사용한다.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CRED = os.path.join(ROOT, "gspread", "account.json")
CACHE = os.path.join(ROOT, "gspread", "_profile_cache.json")
SHEET, TAB = "pan-cvpr27", "ex1"

# 컬럼 정의. (표시명, 키, 소수자리). 순서가 그대로 시트 열 순서가 된다.
COLUMNS = [
    ("실행",        "tag",        None), ("날짜",     "date",     None),
    ("계열",        "family",     None), ("모델",     "model",    None),
    ("seed",        "seed",       None), ("iter",     "iter",     None),
    ("학습시간(h)", "train_h",    2),
    # 구조
    ("width",       "width",      None), ("depth",    "depth",    None),
    ("AttnBlock",   "n_attn",     None), ("norm",     "norm",     None),
    ("mlp_ratio",   "mlp_ratio",  None), ("crop",     "crop",     None),
    ("Params(M)",   "params_m",   4),    ("학습Params", "train_params", None),
    # reduced-resolution (20장)
    ("PSNR",        "psnr",       4), ("SSIM",  "ssim",  4), ("SAM",  "sam",   4),
    ("ERGAS",       "ergas",      4), ("SCC",   "scc",   4), ("Q8",   "q2n",   4),
    # full-resolution (12-19, 8장)
    ("D_lambda",    "d_lambda",   4), ("D_s",   "d_s",   4), ("HQNR", "hqnr",  4),
    # 비용 (--profile)
    ("FLOPs(G)",    "flops_g",    1), ("추론(ms)", "infer_ms", 2), ("메모리(MB)", "mem_mb", 1),
    ("비고",        "note",       None),
]


# ----------------------------------------------------------------- 지표
def _rr(mat):
    """reduced: PSNR/SSIM/SAM/ERGAS/SCC/Q2n. tools/eval_dlpan.py 와 같은 경로를 쓴다."""
    import importlib.util
    from scipy.io import loadmat
    spec = importlib.util.spec_from_file_location("_ed", os.path.join(ROOT, "tools", "eval_dlpan.py"))
    ed = importlib.util.module_from_spec(spec)
    sys.modules["_ed"] = ed
    spec.loader.exec_module(ed)
    from tools.metrics.eval_rr import evaluate

    import h5py
    scale = ed.SCALE["wv3"]
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5") as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(20, -21)
    gt_c = gt[:, sl, sl, :]
    sr = loadmat(mat)["sr"].astype(np.float64)
    if sr.shape[1] in (4, 8):
        sr = sr.transpose(0, 2, 3, 1)
    sr_c = sr[:, sl, sl, :]
    m = evaluate(sr_c, gt_c, scale, 32)
    out = {"sam": m["SAM"][0], "ergas": m["ERGAS"][0], "q2n": m["Q2n"][0]}
    out["scc"] = float(np.mean([ed.scc_dlpan(sr_c[i], gt_c[i]) for i in range(len(gt_c))]))
    out["psnr"] = float(np.mean([ed.psnr_global(sr_c[i], gt_c[i], scale) for i in range(len(gt_c))]))
    out["ssim"] = float(np.mean([ed.ssim_skimage(sr_c[i], gt_c[i], scale) for i in range(len(gt_c))]))
    return out


def _fr(mat, indices="12-19"):
    """full-res: D_lambda/D_s/HQNR. 논문 Table 과 대조 가능한 12-19 부분집합 기준."""
    import h5py
    from scipy.io import loadmat
    from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/full_examples_h5/test_wv3_OrigScale_multiExm1.h5") as f:
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    a, b = (int(x) for x in indices.split("-"))
    sr = loadmat(mat)["sr"].astype(np.float64)
    if sr.shape[1] in (4, 8):
        sr = sr.transpose(0, 2, 3, 1)
    dl, ds = [], []
    for i in range(a, b + 1):
        dl.append(d_lambda_k(sr[i], lms[i], "wv3", 4, 32, wald))
        ds.append(d_s(sr[i], lms[i], pan[i], 4, 32, wald))
    dl, ds = float(np.mean(dl)), float(np.mean(ds))
    return {"d_lambda": dl, "d_s": ds, "hqnr": (1 - dl) * (1 - ds)}


# ----------------------------------------------------------------- 비용
def _profile(args_ns, key, want):
    """FLOPs/추론시간/메모리. 오래 걸리므로 캐시한다."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if key in cache and not want:
        return cache[key]
    if not want:
        return {}
    import torch
    from thop import profile as thop_profile
    from main import import_class
    Model = import_class(args_ns.model)
    m = Model(**args_ns.model_args).eval()
    inp = (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 64, 64),
           torch.randn(1, 8, 64, 64), torch.ones(1))
    f, _ = thop_profile(m, inputs=inp, verbose=False)
    out = {"flops_g": f / 1e9}
    if torch.cuda.is_available():
        m = m.cuda(); inp = tuple(t.cuda() for t in inp)
        with torch.no_grad():
            for _ in range(5):
                m(*inp)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(20):
                m(*inp)
            torch.cuda.synchronize()
        out["infer_ms"] = (time.time() - t0) / 20 * 1000
        out["mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    cache[key] = out
    json.dump(cache, open(CACHE, "w"), indent=1)
    return out


# ----------------------------------------------------------------- 한 실행 수집
def collect(tag, want_profile):
    wd = os.path.join(ROOT, "work_dir", tag)
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", f"{tag}.yaml")
    if not os.path.exists(cfg_path):
        return None
    from main import get_parser, import_class
    c = yaml.safe_load(open(cfg_path))
    p = get_parser(); p.set_defaults(**c); a = p.parse_args([])

    ma = a.model_args
    hs = ma.get("hidden_size")
    row = {
        "tag": tag,
        "model": a.model.rsplit(".", 1)[-1],
        "seed": a.seed, "iter": a.num_iter,
        "width": hs if isinstance(hs, int) else (hs[0] if hs else ""),
        "depth": str(ma.get("depth", "")),
        "n_attn": ma.get("n_attn", len(ma.get("cm3a_locations") or ["2e","3e","4","3d","2d"])),
        "norm": ma.get("norm", "gn"),
        "mlp_ratio": ma.get("mlp_ratio", 4.0),
        "crop": a.train_feeder_args.get("crop", ""),
        "family": ("paper" if "Paper" in a.model else
                   ("fixed" if ma.get("fix_key_alias") else "baseline")),
        "note": "",
    }
    # 파라미터
    Model = import_class(a.model)
    m = Model(**ma)
    row["params_m"] = sum(x.numel() for x in m.parameters()) / 1e6
    row["train_params"] = sum(x.numel() for x in m.parameters() if x.requires_grad)
    del m

    # 학습 시간 / 날짜
    for k, f in (("started_at.txt", "s"), ("finished_at.txt", "f")):
        pth = os.path.join(wd, "meta", k)
        row[f] = open(pth).read().strip() if os.path.exists(pth) else ""
    if row.get("s") and row.get("f"):
        from datetime import datetime as D
        row["train_h"] = (D.fromisoformat(row["f"]) - D.fromisoformat(row["s"])).total_seconds() / 3600
    row["date"] = (row.get("f") or row.get("s") or "")[:10]
    row.pop("s", None); row.pop("f", None)

    # 지표
    rr = os.path.join(wd, "results", "reduced_best_val.mat")
    if not os.path.exists(rr):
        rr = os.path.join(wd, "results", "reduced_best_reduced.mat")
    if os.path.exists(rr):
        row.update(_rr(rr))
    for name in ("full_frrepair.mat", "full_best_val.mat", "full_best_reduced.mat"):
        fr = os.path.join(wd, "results", name)
        if os.path.exists(fr):
            try:
                row.update(_fr(fr))
            except Exception as e:
                row["note"] = f"FR 실패: {type(e).__name__}"
            break
    row.update(_profile(a, tag, want_profile))
    return row


# ----------------------------------------------------------------- 시트
def fmt(row):
    out = []
    for _, key, nd in COLUMNS:
        v = row.get(key, "")
        if v == "" or v is None:
            out.append("")
        elif nd is not None and isinstance(v, (int, float)):
            out.append(round(float(v), nd))
        else:
            out.append(v if isinstance(v, (int, float)) else str(v))
    return out


def upload(rows):
    import gspread
    from gspread_formatting import (CellFormat, TextFormat, Color, format_cell_range,
                                    set_frozen, set_column_widths)
    gc = gspread.service_account(filename=CRED)
    ws = gc.open(SHEET).worksheet(TAB)
    header = [c[0] for c in COLUMNS]
    existing = ws.get_all_values()

    if not existing or existing[0][:len(header)] != header:
        ws.clear()
        ws.update([header], "A1")
        set_frozen(ws, rows=1)
        format_cell_range(ws, f"A1:{chr(64+len(header))}1", CellFormat(
            backgroundColor=Color(0.85, 0.89, 0.95),
            textFormat=TextFormat(bold=True),
            horizontalAlignment="CENTER"))
        set_column_widths(ws, [("A", 150), ("B", 90), ("I", 90)])
        existing = ws.get_all_values()

    # 같은 실행명이 있으면 그 행을 갈아끼우고, 없으면 덧붙인다
    tag_col = [r[0] for r in existing]
    appended = 0
    for r in rows:
        vals = fmt(r)
        if r["tag"] in tag_col:
            i = tag_col.index(r["tag"]) + 1
            ws.update([vals], f"A{i}")
        else:
            ws.append_row(vals, value_input_option="RAW")
            tag_col.append(r["tag"]); appended += 1
    return len(rows), appended


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="*", default=[], help="work_dir 이름 또는 glob")
    ap.add_argument("--all", action="store_true", help="results/*.mat 이 있는 실행 전부")
    ap.add_argument("--profile", action="store_true", help="FLOPs·추론시간·메모리도 측정 (느리다)")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 표만 출력")
    a = ap.parse_args()

    tags = []
    if a.all:
        tags = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p in glob.glob(f"{ROOT}/work_dir/*/results/reduced_*.mat")]
    for pat in a.pattern:
        tags += [os.path.basename(d) for d in glob.glob(f"{ROOT}/work_dir/{pat}") if os.path.isdir(d)]
    tags = sorted(set(tags))
    if not tags:
        print("대상이 없다. 실행명이나 glob 을 줄 것."); return 1

    rows = []
    for t in tags:
        r = collect(t, a.profile)
        if r is None:
            print(f"  건너뜀 {t} (config 없음)"); continue
        rows.append(r)
        print(f"  수집 {t}: ERGAS {r.get('ergas', float('nan')):.4f}  params {r.get('params_m', 0):.4f} M")

    if a.dry_run:
        print("\n" + " | ".join(c[0] for c in COLUMNS))
        for r in rows:
            print(" | ".join(str(v) for v in fmt(r)))
        return 0
    n, added = upload(rows)
    print(f"\n업로드 완료: {n}행 처리 ({added}행 신규, {n-added}행 갱신)")
    print(f"  https://docs.google.com/spreadsheets/d/{gspread_id()}")
    return 0


def gspread_id():
    import gspread
    return gspread.service_account(filename=CRED).open(SHEET).id


if __name__ == "__main__":
    sys.exit(main())
