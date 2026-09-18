#!/usr/bin/env python
"""시트 정리 — Run 열을 짧게, 설명은 Notes 로, 본 결과와 NOA 블록을 라벨로 구분한다 (2026-09-18 사용자 결정).

    python gspread/sheet_cleanup.py --dry-run                 # 바뀔 내용만 출력 (읽기만 한다)
    python gspread/sheet_cleanup.py --apply                   # 자기 탭에 적용 (B·W 열 백업을 먼저 남긴다)
    python gspread/sheet_cleanup.py --apply --tab WV3-s2      # 다른 탭 (소유 서버가 직접 할 때만)

무엇이 문제였나: B(Run) 열이 **평균 480 자·최대 1710 자**였다. 실행명 뒤에 method 설명이 통째로 붙어 있고, 그 설명은 W(Notes) 와 대부분 겹친다.
그래서 어디까지가 실행명인지, 어느 열이 본 결과이고 어느 열이 NOA 인지 한눈에 보이지 않는다.

무엇을 하나
  1. **B(Run)** = 짧은 표시명만. QRC24 계열은 고정 method 를 접두어로 포괄해 `QRC24 <PROFILE> S<seed>[ v<n>] (50K)`,
     그 밖 run 은 실행명 + `(50K)`. 설명 꼬리는 지운다.
  2. **W(Notes)** = `run=<원래 실행명>` + B 에서 뺀 설명 + 기존 Notes. **아무 정보도 버리지 않는다**(중복 문구만 합친다).
  3. **행 2 그룹 라벨** — 본 결과 쪽을 `본 결과(A_ON) RR` · `본 결과(A_ON) FR·paper mat20` 로 바꿔 오른쪽 `NOA …` 블록과 대비시킨다.
  4. 적용 전에 B·W 열 전체를 `gspread/_sheet_backup/<탭>.cleanup_<시각>.json` 으로 남긴다(되돌릴 수 있게).

지표 값(D..U)·Date·통합실험·NOA 열은 **읽지도 쓰지도 않는다**. 행 순서·행 수도 바꾸지 않는다.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BACKUP = os.path.join(ROOT, "gspread", "_sheet_backup")
GROUP_RELABEL = {"RR": "본 결과(A_ON) RR", "FR·paper mat20": "본 결과(A_ON) FR·paper mat20"}


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, path)); m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m; spec.loader.exec_module(m); return m


def split_tag(cell):
    """B열 문자열 → (실행명, iter 라벨, 설명). 'RUN (50K) · desc…' 형태를 가정하되 없으면 빈 값."""
    s = (cell or "").strip()
    m = re.match(r"^(?P<run>[^(·]+?)\s*(?:\((?P<lbl>[^)]*)\))?\s*(?:·\s*(?P<desc>.*))?$", s, flags=re.S)
    if not m:
        return s, "", ""
    return (m.group("run") or "").strip(), (m.group("lbl") or "").strip(), (m.group("desc") or "").strip()


def new_cells(cell, note, server, sc):
    """(B, W) 새 값. 이미 정리된 행이면 그대로 둔다."""
    raw = (cell or "").strip()
    if not raw or raw.startswith((sc.SEP, "■", "□")) or raw[0] in "▍▎▌█":
        return None, None                                                  # 캠페인 구분행은 건드리지 않는다 (실행명이 아니다)
    run, lbl, desc = split_tag(cell)
    if not run:
        return None, None
    short = sc.short_run_name(run)
    b_new = f"{short} ({lbl})" if lbl else short
    bits = [f"run={run}"] if short != run else []
    for t in (desc, (note or "").strip()):
        if t and t not in bits and not any(t in x for x in bits):
            bits.append(t)
    w_new = " · ".join(bits)
    if b_new == (cell or "").strip() and w_new == (note or "").strip():
        return None, None                                                  # 변화 없음
    return b_new, w_new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--apply", action="store_true"); ap.add_argument("--tab", default=None); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        print("--dry-run 또는 --apply 를 줄 것"); return 1
    gu = _load("_gu_cl", "gspread/gspread_upload.py"); sc = _load("_sc_cl", "gspread/sheet_categories.py")
    from tools.gen_pakd50_configs import server_id
    srv = server_id(open(os.path.join(ROOT, "gspread", "server.txt")).read())
    import gspread
    gc = gspread.service_account(filename=gu.CRED); sh = gc.open(gu.SHEET)
    tab = a.tab or gu.sheet_name("WV3", srv); ws = sh.worksheet(tab)
    if a.tab and a.tab != gu.sheet_name("WV3", srv):
        print(f"!! {tab} 은 이 서버({srv})의 탭이 아니다 — 소유 서버에서 실행할 것(§6.1 소유권). 계속하려면 그 서버에서.")
        if a.apply:
            return 1
    hdr3 = ws.row_values(3); hdr2 = ws.row_values(2) + [""] * len(hdr3)
    ib, iw = hdr3.index("Run") + 1, hdr3.index("Notes") + 1
    col_b, col_w = gu._col(ib), gu._col(iw)
    B = [r[0] if r else "" for r in ws.get(f"{col_b}4:{col_b}")]
    W = [r[0] if r else "" for r in ws.get(f"{col_w}4:{col_w}")]
    W += [""] * (len(B) - len(W))
    changes = []
    for i, (b, w) in enumerate(zip(B, W)):
        nb, nw = new_cells(b, w, srv, sc)
        if nb is None:
            continue
        changes.append((4 + i, b, nb, w, nw))
    grp = [(i + 1, hdr2[i], GROUP_RELABEL[hdr2[i]]) for i in range(len(hdr2)) if hdr2[i] in GROUP_RELABEL]
    print(f"[cleanup] 탭 {tab} · 데이터 {len(B)} 행 · Run 열 {col_b} · Notes 열 {col_w}")
    print(f"   B 길이 평균 {sum(len(x) for x in B) // max(len(B), 1)} → {sum(len(c[2]) for c in changes) // max(len(changes), 1)} (바뀌는 {len(changes)} 행 기준)")
    for r, b, nb, w, nw in (changes[:a.limit] if a.limit else changes)[:6]:
        print(f"   행 {r}: {b[:58]}… → {nb}")
        print(f"        Notes {len(w)} 자 → {len(nw)} 자 (앞머리: {nw[:70]}…)")
    if len(changes) > 6:
        print(f"   … 그 밖 {len(changes) - 6} 행")
    for i, old, new in grp:
        print(f"   행2 {gu._col(i)}: '{old}' → '{new}'")
    if not a.apply:
        print("   [dry-run] 시트를 바꾸지 않았다"); return 0
    os.makedirs(BACKUP, exist_ok=True)
    bk = os.path.join(BACKUP, f"{tab.replace('/', '_')}.cleanup_{time.strftime('%m%d-%H%M%S')}.json")
    json.dump(dict(tab=tab, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), run_col=col_b, notes_col=col_w, run=B, notes=W, group_row2=hdr2), open(bk, "w"), indent=1, ensure_ascii=False)
    print(f"   백업 → {os.path.relpath(bk, ROOT)}")
    import fcntl
    pend = []
    for r, b, nb, w, nw in changes:
        pend.append({"range": f"{col_b}{r}", "values": [[nb]]}); pend.append({"range": f"{col_w}{r}", "values": [[nw]]})
    for i, old, new in grp:
        pend.append({"range": f"{gu._col(i)}2", "values": [[new]]})
    with open(os.path.join(ROOT, "work_dir", ".gspread_write.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for j in range(0, len(pend), 200):                                 # Sheets write 제한 — 나눠 보낸다
            gu._retry(ws.batch_update, pend[j:j + 200])
    print(f"   적용 완료: {len(changes)} 행 · 그룹 라벨 {len(grp)} 칸")
    return 0


if __name__ == "__main__":
    sys.exit(main())
