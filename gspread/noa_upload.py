#!/usr/bin/env python
"""NOA 평가 열 업로더 — 자기 서버 탭의 **새 열 범위만** 갱신한다
(계획 research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §6, protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918).

    python gspread/noa_upload.py --dry-run            # 쓸 내용만 출력 (Sheet 접근은 읽기만)
    python gspread/noa_upload.py                      # work_dir/_eval_phase/records/*.json 전부 자기 탭에 반영
    python gspread/noa_upload.py --run <run>

규칙 (§6.2·§6.3):
  · 기존 B..X(원 RR/FR·Date·Train(h)·통합실험) 을 **건드리지 않는다**. 오른쪽 빈 영역에 NOA 그룹 헤더 2 줄 + 값만 쓴다.
  · 행 찾기는 장식 문자열 완전일치가 아니라 `sheet_categories.run_tag()` 로 뽑은 **run id** 로 한다. 같은 run id 행이 여러 개면 전부 같은 값으로 갱신하고(alias) canonical 을 기록한다.
  · 값 쓰기는 `ws.batch_update` 로 **내 열 범위만**. `--replace`·`batch_clear`·레이아웃 재생성은 쓰지 않는다.
  · 학습 uploader(tools/_upload.sh → gspread_upload.py) 와 같은 로컬 flock 을 공유해 동시 쓰기를 막는다.
  · 소유권: `gspread/server.txt` 의 서버 탭만 쓴다. 다른 서버 탭·다른 서버 run 은 건드리지 않는다(§6.1 V10).
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

PROTOCOL_ID = "PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918"
SCHEMA_REVISION = "noa_columns_v2"
LOCK = os.path.join(ROOT, "work_dir", ".gspread_write.lock")            # gspread_upload 와 공유하는 로컬 쓰기 잠금
RECORDS = os.path.join(ROOT, "work_dir", "_eval_phase", "records")

# (그룹, 표시 라벨, 기계 키) — 계획 §6.2 의 키를 그대로 쓴다. 순서가 곧 열 순서다.
COLUMNS = [
    ("NOA RR", "NOA ERGAS↓", "noa_rr_ergas"), ("NOA RR", "NOA SAM↓", "noa_rr_sam"), ("NOA RR", "NOA PSNR↑", "noa_rr_psnr"),
    ("NOA RR", "NOA SSIM↑", "noa_rr_ssim"), ("NOA RR", "NOA SCC↑", "noa_rr_scc"), ("NOA RR", "NOA Q8↑", "noa_rr_q8"),
    ("NOA RR 보조", "NOA RMSE↓", "noa_rr_rmse"), ("NOA RR 보조", "NOA CC↑", "noa_rr_cc"),
    ("NOA FR·paper mat20", "NOA D_lambda↓", "noa_fr_d_lambda"), ("NOA FR·paper mat20", "NOA D_s↓", "noa_fr_d_s"), ("NOA FR·paper mat20", "NOA HQNR(raw)↑", "noa_fr_hqnr_raw"),
    ("Pair reference", "A_ON ERGAS(pair)", "paired_on_rr_ergas"), ("Pair reference", "A_ON HQNR(raw,pair)", "paired_on_fr_hqnr_raw"),
    ("NOA−ON", "ΔERGAS(NOA−ON)", "delta_rr_ergas"), ("NOA−ON", "ΔHQNR(NOA−ON)", "delta_fr_hqnr_raw"),
    ("Pair identity", "Eval mode", "eval_mode"), ("Pair identity", "Eval step", "eval_step"), ("Pair identity", "Eval ckpt SHA", "eval_ckpt_sha"), ("Pair identity", "Eval selector", "source_selector"),
    ("Status", "Eval status", "eval_status"), ("Status", "A_ON check", "legacy_on_check"), ("Status", "NOA joint pass", "noa_joint_pass"),
    ("Provenance", "Eval date", "eval_date"), ("Provenance", "Eval server", "eval_server"), ("Provenance", "Eval protocol", "protocol_id"), ("Provenance", "Eval(h)", "eval_hours"),
]


def _load_local(mod):
    """gspread/ 폴더의 로컬 모듈 — pip 패키지 `gspread` 와 이름이 겹치므로 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location("_noa_local_" + mod, os.path.join(ROOT, "gspread", mod + ".py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def _gu():
    spec = importlib.util.spec_from_file_location("_gu_noa_up", os.path.join(ROOT, "gspread", "gspread_upload.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["_gu_noa_up"] = m; spec.loader.exec_module(m); return m


def row_values(rec):
    """평가 레코드 → 기계 키 dict. 값이 없으면 빈 문자열(칸을 지우지 않고 비워 둔다)."""
    on = rec.get("modes", {}).get("A_ON") or {}; noa = rec.get("modes", {}).get("A_BYPASS_RAW") or {}
    rr, fr = noa.get("rr") or {}, noa.get("fr") or {}; onrr, onfr = on.get("rr") or {}, on.get("fr") or {}
    lc = rec.get("legacy_on_check") or {}
    have = bool(rr and fr and onrr and onfr)
    sec = sum(float((rec["modes"].get(m) or {}).get("seconds") or 0.0) for m in ("A_ON", "A_BYPASS_RAW"))
    v = {
        "noa_rr_ergas": rr.get("ergas"), "noa_rr_sam": rr.get("sam"), "noa_rr_psnr": rr.get("psnr"), "noa_rr_ssim": rr.get("ssim"),
        "noa_rr_scc": rr.get("scc"), "noa_rr_q8": rr.get("q8"), "noa_rr_rmse": rr.get("rmse"), "noa_rr_cc": rr.get("cc"),
        "noa_fr_d_lambda": fr.get("d_lambda"), "noa_fr_d_s": fr.get("d_s"), "noa_fr_hqnr_raw": fr.get("hqnr_raw"),
        "paired_on_rr_ergas": onrr.get("ergas"), "paired_on_fr_hqnr_raw": onfr.get("hqnr_raw"),
        "delta_rr_ergas": rec.get("delta_rr_ergas"), "delta_fr_hqnr_raw": rec.get("delta_fr_hqnr_raw"),
        "eval_mode": "A_BYPASS_RAW", "eval_step": rec.get("checkpoint_step"), "eval_ckpt_sha": (rec.get("checkpoint_sha256") or "")[:16], "source_selector": rec.get("source_selector"),
        "eval_status": ("complete" if have else "partial"), "legacy_on_check": lc.get("status"), "noa_joint_pass": rec.get("noa_joint_pass"),
        "eval_date": (rec.get("evaluated_at") or "")[:16], "eval_server": rec.get("server") or _server(), "protocol_id": PROTOCOL_ID, "eval_hours": (round(sec / 3600.0, 4) if sec else ""),
    }
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
    print(f"[noa-up] 탭 {name} · NOA 열 시작 {gu._col(c0)} ({'기존 재사용' if existing else '신규'}) · 레코드 {len(records)}")
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
        touched.append((run, len(rows), vals["noa_rr_ergas"], vals["noa_fr_hqnr_raw"], vals["delta_rr_ergas"], vals["delta_fr_hqnr_raw"]))
    for run, n, e, h, de, dh in touched:
        print(f"   {run.replace('PAKD50_QRC24_', ''):<40} 행 {n} · NOA ERGAS {e} HQNR {h} · Δ {de} / {dh}")
    if dry:
        print(f"   [dry] 쓸 범위 {len(pending)} 개 — Sheet 를 수정하지 않았다"); return 0
    if not pending:
        print("   쓸 것이 없다"); return 0
    gu._retry(ws.batch_update, pending)                               # 잠금은 함수 진입에서 이미 잡았다(학습 uploader 와 공유)
    for run, *_ in touched:
        p = os.path.join(RECORDS, run + ".json")
        if os.path.exists(p):
            j = json.load(open(p)); j["uploaded"] = time.strftime("%Y-%m-%dT%H:%M:%S"); j["upload_columns"] = f"{gu._col(c0)}..{gu._col(c0 + len(COLUMNS) - 1)}"; json.dump(j, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"   업로드 완료: {len(pending)} 행 범위 · {gu._col(c0)}..{gu._col(c0 + len(COLUMNS) - 1)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    paths = [os.path.join(RECORDS, a.run + ".json")] if a.run else sorted(glob.glob(os.path.join(RECORDS, "*.json")))
    recs = [json.load(open(p)) for p in paths if os.path.exists(p)]
    recs = [r for r in recs if (r.get("modes", {}).get("A_BYPASS_RAW") or {}).get("rr")]
    if not recs:
        print("[noa-up] 올릴 평가 레코드가 없다 (tools/noa_eval.py 먼저)"); return 1
    return upload(recs, dry=a.dry_run, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
