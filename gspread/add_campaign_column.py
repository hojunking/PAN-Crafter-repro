#!/usr/bin/env python
"""시트에 '캠페인' 열을 끼워 넣는다 (Run 바로 뒤, C열).

지표를 다시 계산하지 않는다 — 시트 값을 읽어 한 칸씩 밀고 C열에 캠페인 키만 채운다.
캠페인은 `sheet_categories.classify()` 가 Run 셀에서 정한다(구분행과 같은 단일 소스).

    python gspread/add_campaign_column.py --dry-run   # 무엇이 바뀌는지만
    python gspread/add_campaign_column.py             # 반영 (백업 자동 + 무손실 검증)

안전장치: 반영 전 모든 탭을 gspread/_sheet_backup/<탭>.before_campaign.json 에 저장하고,
반영 후 "원본의 모든 셀이 한 칸 밀린 자리에 그대로 있는가" 를 셀 단위로 검증한다.
"""
import argparse, json, os, sys, time
import gspread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gspread"))
from sheet_categories import classify, SEP          # noqa: E402
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gu", os.path.join(ROOT, "gspread", "gspread_upload.py"))
gu = _ilu.module_from_spec(_spec); sys.modules["gu"] = gu; _spec.loader.exec_module(gu)

CRED = os.path.join(ROOT, "gspread", "account.json")
BK = os.path.join(ROOT, "gspread", "_sheet_backup")
SHEET = "pan-cvpr27"
RUN_COL = 1          # 0-based: B열 = Run
NEW_COL = 2          # 0-based: C열 에 캠페인을 넣는다
HDR_GROUP, HDR_NAME = 1, 2      # 0-based 행: 2행 그룹헤더, 3행 컬럼명


def transform(vals):
    """각 행의 C열 자리에 캠페인 셀을 끼워 넣은 새 2차원 배열."""
    out = []
    for i, r in enumerate(vals):
        r = list(r)
        while len(r) <= RUN_COL:
            r.append("")
        run = r[RUN_COL].strip()
        if i in (HDR_GROUP, HDR_NAME):
            cell = ""        # 헤더 2행은 뒤에서 _write_header() 로 다시 그린다 (그룹은 병합 셀이라
                             # 값만 밀면 병합 범위가 옛 자리에 남아 조용히 사라진다 — 실제로 그랬다)
        elif not run:
            cell = ""                              # 빈 행
        elif run.startswith(SEP):
            cell = ""                              # 구분행은 비운다 — 필터에 안 걸리게
        else:
            cell = classify(run)
        out.append(r[:NEW_COL] + [cell] + r[NEW_COL:])
    return out


def backoff(fn, *args, **kw):
    """Sheets 쓰기 쿼터(분당 60회)에 걸리면 기다렸다 다시 한다."""
    for k in range(6):
        try:
            return fn(*args, **kw)
        except gspread.exceptions.APIError as e:
            if "429" not in str(e) and "Quota" not in str(e):
                raise
            w = 20 * (k + 1)
            print(f"    쿼터 — {w}s 대기 후 재시도 ({k+1}/6)")
            time.sleep(w)
    raise RuntimeError("쿼터 재시도 6회 초과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tabs", nargs="*", default=None, help="특정 탭만 (기본: 전부)")
    ap.add_argument("--header-only", action="store_true",
                    help="데이터는 이미 밀렸고 헤더 2행만 다시 그린다 (쿼터로 중간에 끊겼을 때)")
    a = ap.parse_args()

    gc = gspread.service_account(filename=CRED)
    sh = gc.open(SHEET)
    os.makedirs(BK, exist_ok=True)
    targets = [w for w in sh.worksheets() if not a.tabs or w.title in a.tabs]

    for ws in targets:
        vals = ws.get_all_values()
        has = any(len(r) > NEW_COL and r[NEW_COL].strip() == "캠페인" for r in vals[:4])
        ds = ws.title.split("-")[0]
        if a.header_only:
            backoff(gu._write_header, ws, gu.columns_for(ds),
                    gu.SHEET_COLOR.get(ds, (0.85, 0.89, 0.95)))
            print(f"[{ws.title}] 헤더 재작성")
            time.sleep(8)
            continue
        if has:
            print(f"[{ws.title}] 이미 캠페인 열 있음 — 건너뜀")
            continue
        new = transform(vals)
        counts = {}
        for r in new[3:]:
            k = r[NEW_COL].strip()
            if k:
                counts[k] = counts.get(k, 0) + 1
        print(f"[{ws.title}] {sum(counts.values())}행 분류: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
        if a.dry_run:
            continue

        json.dump(vals, open(os.path.join(BK, f"{ws.title}.before_campaign.json"), "w"),
                  ensure_ascii=False, indent=1)
        need = max(len(r) for r in new)
        if need > ws.col_count:
            ws.add_cols(need - ws.col_count)
        backoff(ws.update, values=new, range_name="A1", value_input_option="RAW")

        # 헤더 2행 재작성 — 병합 범위까지 새 열 수에 맞춘다
        backoff(gu._write_header, ws, gu.columns_for(ds),
                gu.SHEET_COLOR.get(ds, (0.85, 0.89, 0.95)))

        # 검증: 데이터 행(4행~)의 모든 셀이 한 칸 밀린 자리에 그대로 있는가 (헤더는 재작성이라 제외)
        after = ws.get_all_values()
        bad = 0
        for i, r in enumerate(vals):
            if i < 3:
                continue
            for j, c in enumerate(r):
                tgt = j if j < NEW_COL else j + 1
                got = after[i][tgt] if i < len(after) and tgt < len(after[i]) else ""
                if c != got:
                    bad += 1
                    if bad <= 3:
                        print(f"    ★ ({i+1},{j+1}) '{c[:24]}' -> '{got[:24]}'")
        print(f"    검증: 데이터 셀 {sum(len(r) for r in vals[3:])}개 중 불일치 {bad}개 "
              + ("OK" if bad == 0 else "★NG★"))
        if bad:
            sys.exit(1)
        time.sleep(10)   # 쿼터 여유


if __name__ == "__main__":
    main()
