#!/usr/bin/env python
"""시트 재정리 — 새 업로드분을 범주 구분행 아래로 되돌린다.

`gspread_upload.py` 는 새 실행을 **시트 맨 아래에 덧붙인다**(범주와 무관하게).
캠페인이 끝나면 이 스크립트로 다시 범주별로 묶는다.

    python gspread/refile_sheet.py --dry-run          # 무엇이 어디로 갈지만 출력
    python gspread/refile_sheet.py                    # 실제 반영 (백업 자동 저장)
    python gspread/refile_sheet.py --archive          # *-전체 시트도 함께 갱신

안전장치: 반영 전에 모든 시트를 gspread/_sheet_backup/ 에 JSON 으로 저장하고,
반영 후 "원본 데이터행 ⊆ 결과" 를 셀 단위로 검증한다. 불일치면 즉시 중단한다.
범주 정의는 gspread/sheet_categories.py 한 곳에만 있다.
"""
import argparse, json, os, sys, time
from collections import Counter
import gspread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gspread"))
from sheet_categories import classify, run_tag, NAME  # noqa: E402

CRED = os.path.join(ROOT, "gspread", "account.json")
BK = os.path.join(ROOT, "gspread", "_sheet_backup")
SHEET = "pan-cvpr27"
DISPLAY = ["REF", "SEED", "P25", "SUBMOD", "ARCH", "ATTN", "KD", "SE", "MS", "MUT", "MISC"]
NCOL = 19
SEP = "▍"


def datarows(vals):
    return [r for i, r in enumerate(vals)
            if i >= 3 and len(r) > 1 and r[1].strip() and not r[1].startswith(SEP)]


def regroup(vals):
    """헤더 3행 + 범주 구분행 + 데이터행 으로 재구성. 데이터는 하나도 버리지 않는다."""
    header = vals[0:3]
    data = datarows(vals)
    buckets = {}
    for r in data:
        buckets.setdefault(classify(r[1]), []).append(r)
    out = [list(x) for x in header]
    sep = []
    for k in DISPLAY:
        if not buckets.get(k):
            continue
        row = [""] * max(len(header[2]), NCOL + 1)
        row[1] = f"{SEP}{NAME[k]}"
        out.append(row); sep.append(len(out))
        out.extend(list(r) for r in buckets[k])
    return out, sep, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--archive", action="store_true", help="*-전체 시트도 갱신")
    a = ap.parse_args()

    gc = gspread.service_account(filename=CRED)
    sh = gc.open(SHEET)
    os.makedirs(BK, exist_ok=True)
    targets = [w.title for w in sh.worksheets()
               if w.title.startswith("WV3-") and (a.archive or not w.title.endswith("-전체"))]

    for title in targets:
        ws = sh.worksheet(title)
        vals = ws.get_all_values()
        json.dump(vals, open(os.path.join(BK, f"{title}.json"), "w"), ensure_ascii=False, indent=1)
        out, sep, data = regroup(vals)

        moved = [r[1][:44] for r in data if classify(r[1]) != "MISC"]
        print(f"[{title}] 데이터 {len(data)}행 -> 범주 {len(sep)}개")
        if a.dry_run:
            for k in DISPLAY:
                g = [run_tag(r[1]) for r in data if classify(r[1]) == k]
                if g:
                    print(f"   {NAME[k][:40]:<42} {len(g):>2}건  {', '.join(g[:6])}"
                          + (" …" if len(g) > 6 else ""))
            continue

        ws.batch_clear([f"B4:U{ws.row_count}"])
        ws.update(values=[r[1:NCOL + 1] for r in out[3:]], range_name="B4",
                  value_input_option="RAW")
        reqs = [{"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": rr - 1, "endRowIndex": rr,
                      "startColumnIndex": 1, "endColumnIndex": NCOL + 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": .87, "green": .89, "blue": .93},
                "textFormat": {"bold": True}}},
            "fields": "userEnteredFormat(backgroundColor,textFormat.bold)"}} for rr in sep]
        sh.batch_update({"requests": reqs})

        after = datarows(ws.get_all_values())
        lost = Counter(tuple(r) for r in data) - Counter(tuple(r) for r in after)
        if lost:
            print(f"  ★검증 실패 — {sum(lost.values())}행 손실. "
                  f"백업: {os.path.join(BK, title + '.json')}")
            sys.exit(1)
        print(f"  검증 OK (데이터행 {len(after)}, 손실 0)")
        time.sleep(2)


if __name__ == "__main__":
    main()
