#!/usr/bin/env python
"""새 탭을 옛(큐레이션된) 탭과 같은 행 구성·순서로 다시 배치하고, 옛 탭에 없던 run 은 별도 탭으로 뺀다.

    python gspread/apply_layout.py --sheet WV3-s1 --ref WV3-s1_v1 [--extra WV3-s1-extra] [--dry-run]

배경 (2026-09-07): 지표 정의를 바꾸며 `--all --replace` 로 새 탭을 썼더니 work_dir 의 모든 run(148)이 파일 순서로
올라가 옛 탭(큐레이션 65 run, 범주 구분행)과 달라 보였다. 이 스크립트는
  - 메인 탭: 옛 탭의 행 순서(구분행 포함)를 그대로 따르고, 값만 새 탭의 것으로 채운다.
    옛 탭에서 마지막 구분행 뒤에 덧붙어 있던 S1_* 격자에는 '⑮ Teacher 후보 격자' 구분행을 하나 넣는다.
  - extra 탭: 옛 탭에 없던 run 을 범주별 구분행과 함께 둔다. 한 run 도 버리지 않는다.
쓰기 전에 두 탭 원본을 gspread/_sheet_backup/ 에 저장하고, 쓴 뒤 run 집합이 보존됐는지 검증한다.
"""
import argparse, json, os, sys, time
import gspread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gspread")); sys.path.insert(0, ROOT)
from sheet_categories import classify, run_tag, NAME, DESC, SEP            # noqa: E402
from refile_sheet import datarows, regroup, ncol_of, _col, DISPLAY         # noqa: E402
import gspread_upload as gu                                                # noqa: E402

CRED = os.path.join(ROOT, "gspread", "account.json"); SHEET = "pan-cvpr27"
BK = os.path.join(ROOT, "gspread", "_sheet_backup")


def sep_row(label, ncol, width):
    row = [""] * max(width, ncol + 1); row[1] = label if label.startswith(SEP) else SEP + label
    key = next((k for k, n in NAME.items() if n == row[1][len(SEP):]), None)
    if key in DESC:
        row[ncol] = DESC[key]
    return row


def write_tab(sh, ws, header, body, seps, ncol, cols):
    """body: 3행 헤더 뒤에 올 행들(구분행 포함). seps: body 안에서 구분행의 0-based 인덱스."""
    last_col = _col(ncol)
    gu._retry(ws.batch_clear, [f"B4:{last_col}{ws.row_count}"])
    if ws.row_count < len(body) + 10:
        gu._retry(ws.add_rows, len(body) + 10 - ws.row_count)
    gu._retry(ws.update, values=[(r + [""] * (ncol + 1))[1:ncol + 1] for r in body], range_name="B4", value_input_option="RAW")
    if seps:
        gu._retry(ws.spreadsheet.batch_update, {"requests": [gu._sep_request(ws, 4 + i, ncol) for i in seps]})
    gu._retry(gu._apply_borders, ws, cols, 3 + len(body))
    gu._retry(gu._bold_best, ws, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--extra", default=None, help="기본 <sheet>-extra")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src-json", action="append", default=None,
                    help="행을 보충할 백업 JSON (gspread/_sheet_backup/*.json). 탭이 손상됐을 때")
    a = ap.parse_args()
    extra_title = a.extra or f"{a.sheet}-extra"
    gc = gspread.service_account(filename=CRED); sh = gc.open(SHEET)
    ws = sh.worksheet(a.sheet); vals = ws.get_all_values(); ref = sh.worksheet(a.ref).get_all_values()
    os.makedirs(BK, exist_ok=True)
    json.dump(vals, open(os.path.join(BK, f"{a.sheet}.before_layout.json"), "w"), ensure_ascii=False, indent=1)
    ds = a.sheet.split("-")[0]; cols = gu.columns_for(ds)

    # 행은 메인 탭 + (있으면) extra 탭에서 모으고, 열은 (그룹, 라벨) 이름으로 코드 COLUMNS 에 다시 맞춘다.
    # 그래서 시트에서 열을 지우거나(예: 2026-09-07 FR·H5 열 삭제) 코드에서 열을 바꿔도 값이 어긋나지 않는다.
    def keyed_rows(v):
        if len(v) < 3 or not any(c.strip() for c in v[2]):
            print("  [경고] 헤더가 비어 있는 원본은 건너뛴다 (값을 열에 대응시킬 수 없다)")
            return {}
        grp, hdr = v[1], v[2]
        keys, g = [], ""
        for i, h in enumerate(hdr):
            g = grp[i] if i < len(grp) and grp[i].strip() else (g if h.strip() and h not in ("Run", "Date", "Notes") else "")
            keys.append((g, h))
        out = {}
        for r in datarows(v):
            out[run_tag(r[1])] = {keys[i]: r[i] for i in range(min(len(keys), len(r)))}
        return out
    src = keyed_rows(vals)
    try:
        wx0 = sh.worksheet(extra_title)
        for t, d in keyed_rows(wx0.get_all_values()).items():
            src.setdefault(t, d)                       # 메인 탭이 우선
    except gspread.exceptions.WorksheetNotFound:
        pass
    for p in (a.src_json or []):                       # 백업 JSON 에서도 보충 (탭이 손상됐을 때)
        for t, d in keyed_rows(json.load(open(p))).items():
            src.setdefault(t, d)
    want = [("", "Run")] + [(c[0], c[1]) for c in cols[1:]]
    def to_row(d):
        r = [""] * (len(want) + 1)
        for j, k in enumerate(want):
            r[j + 1] = d.get(k, "")
        return r
    rows = {t: to_row(d) for t, d in src.items()}
    # 헤더는 코드 기준으로 다시 만든다
    header = [[""] * (len(want) + 1) for _ in range(3)]
    header[1] = [""] + [c[0] if (i == 0 or cols[i - 1][0] != c[0]) else "" for i, c in enumerate(cols)]
    header[2] = [""] + [c[1] for c in cols]
    ncol = len(cols); width = len(header[2])

    main_body, seps, used, s1_labeled = [], [], set(), False
    for r in ref[3:]:
        if len(r) < 2 or not r[1].strip():
            continue
        cell = r[1]
        if cell.startswith(SEP):
            seps.append(len(main_body)); main_body.append(sep_row(cell, ncol, width)); continue
        t = run_tag(cell)
        if t not in rows:
            print(f"  [ref 에만 있음, 새 탭에 없음] {t}"); continue
        if classify(t) == "S1GRID" and not s1_labeled:
            seps.append(len(main_body)); main_body.append(sep_row(NAME["S1GRID"], ncol, width)); s1_labeled = True
        main_body.append(list(rows[t])); used.add(t)
    leftover = [r for t, r in rows.items() if t not in used]
    ex_out, ex_sep, _, _ = regroup(header + leftover)
    ex_body = ex_out[3:]; ex_seps = [i - 1 for i in ex_sep]         # regroup 의 sep 는 1-based(헤더 포함) 행 번호

    print(f"[{a.sheet}] 메인 {len(used)} run + 구분행 {len(seps)}개 (옛 순서) ; extra 탭 {extra_title}: {len(leftover)} run, 구분행 {len(ex_seps)}개")
    if a.dry_run:
        for r in main_body: print("   " + (r[1][:70] if r[1].startswith(SEP) else "     " + run_tag(r[1])))
        print("   --- extra ---")
        for r in ex_body: print("   " + (r[1][:70] if r[1].startswith(SEP) else "     " + run_tag(r[1])))
        return 0

    def ensure_header(w):
        cur = w.get(f"{gu._a1(gu.ORIGIN_ROW + 1, gu.ORIGIN_COL)}:{gu._a1(gu.ORIGIN_ROW + 1, gu.ORIGIN_COL + len(cols) - 1)}")
        grp = w.get(f"{gu._a1(gu.ORIGIN_ROW, gu.ORIGIN_COL)}:{gu._a1(gu.ORIGIN_ROW, gu.ORIGIN_COL + len(cols) - 1)}")
        if (not cur or (cur[0] + [""] * len(cols))[:len(cols)] != [c[1] for c in cols]
                or not grp or (grp[0] + [""] * len(cols))[:len(cols)] != header[1][1:]):
            # 옛 헤더가 더 넓었으면 병합·값을 행 전체에서 먼저 푼다 (좁은 범위로 unmerge 하면 API 400)
            gu._retry(w.unmerge_cells, "A2:AZ3")
            gu._retry(w.batch_clear, [f"B2:{_col(max(len(cols), 40))}3"])
            gu._retry(gu._write_header, w, cols, gu.SHEET_COLOR.get(ds, (0.85, 0.89, 0.95)))
    ensure_header(ws)
    write_tab(sh, ws, header, main_body, seps, ncol, cols)
    wx = gu._ensure_sheet(sh, extra_title)
    json.dump(wx.get_all_values(), open(os.path.join(BK, f"{extra_title}.before_layout.json"), "w"), ensure_ascii=False, indent=1)
    ensure_header(wx)
    write_tab(sh, wx, header, ex_body, ex_seps, ncol, cols)
    time.sleep(2)
    got = {run_tag(r[1]) for r in datarows(ws.get_all_values())} | {run_tag(r[1]) for r in datarows(wx.get_all_values())}
    lost = set(rows) - got
    print("검증 " + ("OK — run 손실 0" if not lost else f"실패 — 손실 {sorted(lost)}"))
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
