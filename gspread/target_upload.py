#!/usr/bin/env python
"""Narrow R2 target 열 업로더 — v2 selector 로 고른 **목표 checkpoint** 지표를 자기 탭의 새 열 묶음으로 올린다
(계획 research_log/PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md §8.2, revision QRC24_NARROW_R2_20260917).

    python gspread/target_upload.py --dry-run
    python gspread/target_upload.py [--run <run>]

규칙: 기존 legacy 품질 열(원 RR/FR·Date·Train(h)·통합실험·NOA 그룹)을 **조용히 교체하지 않는다** — 오른쪽에 target 열 묶음을 따로 둔다.
target HQNR 과 **다른 step 의** ERGAS 를 한 결과처럼 표시하지 않는다(같은 checkpoint 의 값만 한 행에 넣는다). 나머지 지표·identity 는 sidecar(json) 에 남는다.
행 매칭은 run id(run_tag), 같은 run id 의 중복 행은 alias 로 함께 갱신, 쓰기는 자기 열 범위만, 학습 uploader 와 같은 로컬 flock.
"""
import argparse
import fcntl
import glob
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import gen_pakd50_configs as G  # noqa: E402

PROTOCOL_ID = "PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918"
SCHEMA_REVISION = "r2_target_columns_v1"
LOCK = os.path.join(ROOT, "work_dir", ".gspread_write.lock")            # gspread_upload 와 공유하는 로컬 쓰기 잠금
RECORDS = os.path.join(ROOT, "work_dir", "_eval_phase", "records")

# (그룹, 표시 라벨, 기계 키) — R2 §8.2 의 키
COLUMNS = [
    ("R2 target", "Target selector", "target_selector"), ("R2 target", "Target step", "target_step"),
    ("R2 target", "Target HQNR(raw)↑", "target_HQNR_raw"), ("R2 target", "Target ERGAS↓", "target_ERGAS"),
    ("R2 target", "Target SCC↑", "target_SCC"), ("R2 target", "Target PSNR↑", "target_PSNR"),
    ("R2 target", "Target joint pass", "target_joint_pass"), ("R2 target", "Target official", "target_official_complete"),
    ("R2 lock", "recipe_lock_id", "recipe_lock_id"), ("R2 lock", "queue_revision", "queue_revision"),
]


def _load_local(mod):
    """gspread/ 폴더의 로컬 모듈 — pip 패키지 `gspread` 와 이름이 겹치므로 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location("_noa_local_" + mod, os.path.join(ROOT, "gspread", mod + ".py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def _gu():
    spec = importlib.util.spec_from_file_location("_gu_noa_up", os.path.join(ROOT, "gspread", "gspread_upload.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["_gu_noa_up"] = m; spec.loader.exec_module(m); return m


def row_values(rec):
    """run id → v2 selector 결과(results/qrecon24_target_selection_HQNR9585_ERGAS2040_v2.json) 에서 목표 checkpoint 한 줄."""
    run = rec["run"] if isinstance(rec, dict) else rec
    p = os.path.join(ROOT, "work_dir", run, "results", f"qrecon24_target_selection_{G.QRC24_R2_SELECTOR}.json")
    j = json.load(open(p)) if os.path.exists(p) else {}
    t = j.get("target") or {}
    lk = G.qrc24_recipe_lock()
    v = dict(target_selector=j.get("selector", ""), target_step=t.get("step", ""), target_HQNR_raw=t.get("hqnr", ""), target_ERGAS=t.get("ergas", ""),
             target_SCC=t.get("scc", ""), target_PSNR=t.get("psnr", ""),
             target_joint_pass=("" if j.get("joint_pass") is None else j["joint_pass"]), target_official_complete=bool(j.get("official")),
             recipe_lock_id=(lk.get("lock_id") if lk else ""), queue_revision=G.QRC24_R2_REVISION)
    return {k: ("" if x is None else (round(x, 6) if isinstance(x, float) else x)) for k, x in v.items()}


def _server():
    from tools.gen_pakd50_configs import server_id
    return server_id(open(os.path.join(ROOT, "gspread", "server.txt")).read())


def find_first_free_col(ws, gu):
    """기존 표의 오른쪽 첫 빈 열 (헤더 2·3 행 기준). 이미 NOA 헤더가 있으면 그 위치를 재사용한다(멱등)."""
    hdr2 = ws.row_values(gu.ORIGIN_ROW); hdr3 = ws.row_values(gu.ORIGIN_ROW + 1)
    if COLUMNS[0][2] in [h.strip() for h in hdr3] or COLUMNS[0][1] in [h.strip() for h in hdr3]:
        i = [h.strip() for h in hdr3].index(COLUMNS[0][1] if COLUMNS[0][1] in [h.strip() for h in hdr3] else COLUMNS[0][2])
        return i + 1, True
    used = max(len(hdr2), len(hdr3))
    return used + 2, False                                            # 기존 마지막 열 + 한 칸 비우고 시작


def ensure_header(ws, gu, c0, existing, dry=False):
    labels = [c[1] for c in COLUMNS]; groups = [c[0] for c in COLUMNS]
    a1 = lambda r, c: gu._a1(r, c)
    rng2 = f"{a1(gu.ORIGIN_ROW, c0)}:{a1(gu.ORIGIN_ROW, c0 + len(COLUMNS) - 1)}"
    rng3 = f"{a1(gu.ORIGIN_ROW + 1, c0)}:{a1(gu.ORIGIN_ROW + 1, c0 + len(COLUMNS) - 1)}"
    if dry:
        print(f"   [dry] 헤더 {rng2} / {rng3} ({len(COLUMNS)} 열, {'재사용' if existing else '신규'})"); return
    if ws.col_count < c0 + len(COLUMNS) - 1:
        gu._retry(ws.add_cols, c0 + len(COLUMNS) - 1 - ws.col_count)
    gu._retry(ws.batch_update, [{"range": rng2, "values": [groups]}, {"range": rng3, "values": [labels]}])


def upload(records, dry=False, limit=None):
    """열 배치 계산과 쓰기를 **같은 로컬 flock** 안에서 한다 — NOA/target 두 업로더가 동시에 같은 빈 열을 잡는 것을 막는다."""
    if not dry:
        _lk = open(LOCK, "w"); fcntl.flock(_lk, fcntl.LOCK_EX)
    gu = _gu(); srv = _server()
    _sc = _load_local("sheet_categories")                             # 같은 폴더의 gspread/sheet_categories.py (패키지 gspread 와 이름이 겹친다)
    run_tag = lambda cell: _sc.canonical_run_key(cell, srv)           # 긴 표기·짧은 표기(2026-09-18 단축) 를 같은 run 으로 묶는다
    import gspread as gs
    gc = gs.service_account(filename=gu.CRED); sh = gc.open(gu.SHEET)
    name = gu.sheet_name("WV3", srv); ws = sh.worksheet(name)
    c0, existing = find_first_free_col(ws, gu)
    print(f"[target-up] 탭 {name} · NOA 열 시작 {gu._col(c0)} ({'기존 재사용' if existing else '신규'}) · 레코드 {len(records)}")
    ensure_header(ws, gu, c0, existing, dry=dry)
    tags = [r[0] if r else "" for r in ws.get(f"{gu._col(gu.ORIGIN_COL)}{gu.ORIGIN_ROW + 2}:{gu._col(gu.ORIGIN_COL)}")]
    ids = [run_tag(t) for t in tags]
    pending, touched = [], []
    for rec in records[:limit] if limit else records:
        run = rec["run"]
        if rec.get("source_train_server") and rec["source_train_server"] != srv:
            print(f"   건너뜀(소유 서버 아님): {run}"); continue
        rows = [i for i, t in enumerate(ids) if t == run]
        if not rows:
            print(f"   !! 시트에 행 없음: {run} — 학습 업로더가 먼저 올려야 한다"); continue
        vals = row_values(rec); line = [vals[c[2]] for c in COLUMNS]
        for k, i in enumerate(rows):
            r1 = gu.ORIGIN_ROW + 2 + i
            pending.append({"range": f"{gu._a1(r1, c0)}:{gu._a1(r1, c0 + len(COLUMNS) - 1)}", "values": [line]})
        touched.append((run, len(rows), vals["target_step"], vals["target_HQNR_raw"], vals["target_ERGAS"], vals["target_joint_pass"]))
    for run, n, st, h, e, jp in touched:
        print(f"   {run.replace('PAKD50_QRC24_', ''):<40} 행 {n} · target step {st} · H {h} E {e} · joint {jp}")
    if dry:
        print(f"   [dry] 쓸 범위 {len(pending)} 개 — Sheet 를 수정하지 않았다"); return 0
    if not pending:
        print("   쓸 것이 없다"); return 0
    gu._retry(ws.batch_update, pending)                               # 잠금은 함수 진입에서 이미 잡았다(학습 uploader 와 공유)
    print(f"   업로드 완료: {len(pending)} 행 범위 · {gu._col(c0)}..{gu._col(c0 + len(COLUMNS) - 1)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    pat = os.path.join(ROOT, "work_dir", "PAKD50_QRC24_*", "results", f"qrecon24_target_selection_{G.QRC24_R2_SELECTOR}.json")
    runs = [a.run] if a.run else sorted(os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(pat))
    recs = [dict(run=r, source_train_server=_server()) for r in runs]
    if not recs:
        print(f"[target-up] v2 selector 결과가 없다 — 먼저 python tools/qrecon24_select.py <run> --selector {G.QRC24_R2_SELECTOR} --official"); return 1
    return upload(recs, dry=a.dry_run, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
