#!/usr/bin/env python
"""시트 정리 (2026-09-11): 현 접근(BASE/PA/PO10/KDV)과 무관한 run 을 WV3-<server> 탭에서 WV3-<server>_v1 탭으로 옮긴다.

    python gspread/archive_to_v1.py --dry-run            # 무엇이 어디로 가는지만
    python gspread/archive_to_v1.py                      # 실제 반영 (탭별 백업 → v1 에 덧붙임 → 본 탭 재작성 → 셀 단위 검증)
    python gspread/archive_to_v1.py --sheet WV3-s1       # 한 탭만

이동 규칙: sheet_categories.KEEP 에 없는 범주(ARCHIVED) 전부. v1 탭 맨 아래에 (빈 행, 현 탭 헤더 21열, "▍이전분 …" 구분행, 범주 구분행 + run 행) 을 덧붙인다 —
v1 탭의 기존 행(지표 v1 값, 19열)은 손대지 않는다. 옮긴 행은 현 탭 값(지표 v2: FR·paper mat20, evaluator 2026-09-10.5) 그대로다.
본 탭은 남는 run 만으로 refile_sheet.regroup 을 다시 돌려 범주 구분행을 정리한다. 반영 뒤 (옮긴 행 ⊆ v1) 과 (본 탭 = 남긴 행) 을 셀 단위로 확인한다.
"""
import argparse, json, os, sys, time
from collections import Counter
import gspread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gspread"))
from sheet_categories import classify, run_tag, NAME, SEP, KEEP, ARCHIVED, ORDER   # noqa: E402
from refile_sheet import datarows, regroup, ncol_of, _col                       # noqa: E402

CRED = os.path.join(ROOT, "gspread", "account.json"); SHEET = "pan-cvpr27"; BK = os.path.join(ROOT, "gspread", "_sheet_backup")
STAMP = time.strftime("%Y-%m-%d")
NOTE = (f"▍이전분 {STAMP} — 현 접근(새 baseline BASE_W*_MSPAN · PA · PO10 · KDV)과 무관해 WV3 본 탭에서 옮긴 run. 값은 지표 v2(FR·paper mat20, evaluator 2026-10.5 계열) 그대로이며 "
        "위쪽 옛 행(지표 v1, 19열)과 열이 다르다 — 아래 헤더를 따른다. gspread/archive_to_v1.py")


def sep_style(sheet_id, r0, ncol, bold=True, rgb=(.87, .89, .93)):
    return {"repeatCell": {"range": {"sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r0 + 1, "startColumnIndex": 1, "endColumnIndex": ncol + 1},
                           "cell": {"userEnteredFormat": {"backgroundColor": {"red": rgb[0], "green": rgb[1], "blue": rgb[2]}, "textFormat": {"bold": bold}}},
                           "fields": "userEnteredFormat(backgroundColor,textFormat.bold)"}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--sheet", action="append", default=None)
    ap.add_argument("--main-from-backup", default=None, help="본 탭 원본을 이 백업 json 에서 읽는다 (재작성 실패 뒤 복구용; --sheet 하나와 같이)"); a = ap.parse_args()
    gc = gspread.service_account(filename=CRED); sh = gc.open(SHEET); os.makedirs(BK, exist_ok=True)
    titles = {w.title for w in sh.worksheets()}
    targets = a.sheet or [t for t in titles if t.startswith("WV3-") and not t.endswith("_v1") and not t.endswith("-전체")]
    for title in sorted(targets):
        v1 = title + "_v1"
        if v1 not in titles:
            print(f"[{title}] v1 탭 {v1} 없음 — 건너뜀"); continue
        ws = sh.worksheet(title); wv = sh.worksheet(v1)
        vals = json.load(open(a.main_from_backup)) if a.main_from_backup else ws.get_all_values(); vvals = wv.get_all_values()
        if not a.main_from_backup:
            json.dump(vals, open(os.path.join(BK, f"{title}.before_archive_{STAMP}.json"), "w"), ensure_ascii=False, indent=1)
            json.dump(vvals, open(os.path.join(BK, f"{v1}.before_archive_{STAMP}.json"), "w"), ensure_ascii=False, indent=1)
        already = any(len(r) > 1 and r[1].startswith(f"▍이전분 {STAMP}") for r in vvals)      # 같은 날 이미 덧붙였으면 v1 은 건너뛴다 (멱등)
        ncol = ncol_of(vals); width = ncol + 1
        data = datarows(vals)
        move = [r for r in data if classify(r[1]) in ARCHIVED]; keep = [r for r in data if classify(r[1]) in KEEP]
        assert len(move) + len(keep) == len(data)
        cnt_m = Counter(classify(r[1]) for r in move); cnt_k = Counter(classify(r[1]) for r in keep)
        print(f"[{title}] 데이터 {len(data)}행: 옮김 {len(move)} {dict(sorted(cnt_m.items(), key=lambda kv: -kv[1]))} · 남김 {len(keep)} {dict(cnt_k)}")
        if a.dry_run:
            for k in ORDER:
                g = [run_tag(r[1]) for r in move if classify(r[1]) == k]
                if g: print(f"   → v1  {NAME[k][:38]:<40} {len(g):>2}건  {', '.join(g[:5])}" + (" …" if len(g) > 5 else ""))
            print(f"   남김: {', '.join(run_tag(r[1]) for r in keep)}")
            continue
        if not move:
            print("   옮길 행 없음"); continue
        # ---- v1 탭에 덧붙임: 빈 행, 현 탭 헤더(2·3행), 안내 구분행, 범주별 (구분행 + run 행)
        hdr2 = (vals[1] + [""] * width)[:width]; hdr3 = (vals[2] + [""] * width)[:width]
        block = [[""] * width, hdr2, hdr3]; note = [""] * width; note[1] = NOTE; block.append(note); sep_idx = [len(block) - 1]
        for k in ORDER:
            g = [r for r in move if classify(r[1]) == k]
            if not g: continue
            srow = [""] * width; srow[1] = SEP + NAME[k]; sep_idx.append(len(block)); block.append(srow)
            block.extend([(r + [""] * width)[:width] for r in g])
        start = len(vvals) + 1                                   # 1-based 다음 빈 행
        if already:
            print(f"   {v1}: 오늘 이전분이 이미 있음 — 덧붙이지 않음")
        else:
            if wv.row_count < start + len(block):
                wv.add_rows(start + len(block) - wv.row_count + 5)
            if wv.col_count < width:
                wv.add_cols(width - wv.col_count)
            wv.update(values=[r[1:width] for r in block], range_name=f"B{start}", value_input_option="RAW")
            reqs = [sep_style(wv.id, start - 1 + i, ncol, rgb=((.95, .85, .70) if i == sep_idx[0] else (.87, .89, .93))) for i in sep_idx]
            reqs.append(sep_style(wv.id, start - 1 + 2, ncol, rgb=(.93, .93, .93)))     # 헤더 3행
            sh.batch_update({"requests": reqs})
        # ---- 본 탭 재작성: 남긴 행만 regroup
        kept_vals = vals[:3] + keep
        out, sep, _, _ = regroup(kept_vals, None)
        last_col = _col(ncol)
        ws.batch_clear([f"B4:{last_col}{ws.row_count}"])
        ws.update(values=[(r + [""] * width)[1:width] for r in out[3:]], range_name="B4", value_input_option="RAW")
        # 옛 구분행 서식을 지우고 새 구분행에만 서식
        clear = {"repeatCell": {"range": {"sheetId": ws.id, "startRowIndex": 3, "endRowIndex": ws.row_count, "startColumnIndex": 1, "endColumnIndex": ncol + 1},
                                "cell": {"userEnteredFormat": {"backgroundColor": {"red": 1, "green": 1, "blue": 1}, "textFormat": {"bold": False}}}, "fields": "userEnteredFormat(backgroundColor,textFormat.bold)"}}
        sh.batch_update({"requests": [clear] + [sep_style(ws.id, rr - 1, ncol) for rr in sep]})
        # ---- 검증
        after_v1 = datarows(wv.get_all_values()); after_main = datarows(ws.get_all_values())
        norm = lambda r: tuple((r + [""] * width)[:width])
        lost_v1 = Counter(norm(r) for r in move) - Counter(norm(r) for r in after_v1)
        diff_main = (Counter(norm(r) for r in keep) - Counter(norm(r) for r in after_main)) + (Counter(norm(r) for r in after_main) - Counter(norm(r) for r in keep))
        if lost_v1 or diff_main:
            print(f"  ★검증 실패 — v1 누락 {sum(lost_v1.values())}행, 본 탭 불일치 {sum(diff_main.values())}행. 백업: {BK}/{title}.before_archive_{STAMP}.json"); sys.exit(1)
        print(f"   OK: {v1} 에 {len(move)}행 {'(이미 있음)' if already else f'덧붙임(행 {start}부터, 범주 {len(sep_idx) - 1}개)'} · {title} 는 {len(after_main)}행 {len(sep)}범주로 재작성 · 검증 통과")
    print(f"  https://docs.google.com/spreadsheets/d/{sh.id}")


if __name__ == "__main__":
    main()
