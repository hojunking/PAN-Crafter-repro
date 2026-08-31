#!/usr/bin/env python
"""결과 정리 시트("결과정리") 생성 — 대표 이정표 + Swin/TF 캠페인 전체.

WV3-s1 / WV3-s2 탭의 수치를 그대로 인용해(재평가 없음) 한 장으로 모은다.
세팅은 config 에서 파생하고, 상단에 **기준 세팅**을 못 박은 뒤 각 행은
기준과 다른 것만 열별로 적는다 (구세팅 이정표는 '추가 설정'에 전부 명시).

재실행하면 시트를 새로 만든다(기존 "결과정리" 삭제 후 재생성).

  python gspread/summary_sheet.py
"""
import os
import sys

import gspread
import yaml
from gspread_formatting import (CellFormat, Color, TextFormat, format_cell_range)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED = os.path.join(ROOT, "gspread", "account.json")
SHEET = "pan-cvpr27"
WS_NAME = "결과정리"

BASELINE = ("기준 세팅 (모든 행 공통 — 아래 열은 기준과 다른 것만 표기): "
            "논문 충실 재구성 PAN-Crafter · LayerNorm(Eq 5) · crop 없음 · 50K iter · "
            "dual MARs · best 선택 = 공식 HQNR(FR index 12-19) · AdamW 1e-4/wd 0.01 · "
            "cosine+warmup 100 · batch 48(MARs 복제로 실효 96) · 입력 11ch(PAN·↑LPAN·"
            "PAN−↑LPAN·↑MS) · width 128 · depth [1,2,4](c6 골격, full/H2/H4) · "
            "attention 없음 · Swin 없음. 이정표 구간의 구세팅(배포 코드·GN·crop·"
            "val-ERGAS 선택 등)은 '추가 설정' 열에 전부 명시. "
            "수치: RR 는 DLPan 프로토콜, HQNR 는 공식 구현 — 서버 간 절대값 비교 금지 "
            "(anchor 대비로 읽을 것).")

# (tag, server, 추가설정 override 또는 None(=config 파생), 판정 한 줄)
MILESTONES = [
    ("■ Paper", "-", "논문 보고 세팅: 7.17M·79G 주장 · best 선택 미기재", "논문 보고값 — 모든 비교의 기준행"),
    ("wv3_baseline_valsel", "s1", "배포 코드 그대로: 4-scale·CM3A5·GN·mode-token·crop(실은 scale jitter)·best=val-ERGAS", "배포본 재현 2.1633 (+6.1%) — 격차 원인은 배포코드≠논문"),
    ("paper_wv3", "s1", "논문 충실 재구성(7.17M) 첫 판: GN·crop 유지·best=val-ERGAS", "구조 재구성만으로 +0.7% 회복"),
    ("s1_A0", "s1", "재구성+LN+nocrop·9ch·best=val-ERGAS", "crop(scale jitter) 제거가 최대 원인(−3.6%)"),
    ("s1_A1", "s1", "재구성+LN+nocrop·11ch·best=val-ERGAS", "재현 완결 — 논문 2.040 초과 (2.0351)"),
    ("c0_hqnr", "s1", "전체 구조(d224·attn enc/btl/dec 3개) — 이후 모든 행과 같은 HQNR 선택", "HQNR 선택 기준의 전체 구조 anchor (7.17M)"),
    ("c6_c4d124", "s1", None, "표준 anchor — d124·attn0 (경량화 캠페인 Quality winner)"),
    ("c6_c4d124", "s2", None, "c6 서버 재현 (차이 0.53%)"),
    ("N3_9_d124_noattn", "s1", None, "9ch 성립 — c6 와 전 지표 동급"),
    ("N3_9_d124_noattn", "s2", None, "9ch 서버 독립 확정"),
    ("R1_w128_d024_noattn", "s1", None, "full-res 제거: HQNR 유지·ERGAS +2.28%·추론 2배 — 한때 KD Student 1순위"),
    ("c8_c4w96", "s1", None, "순수 CNN 압축의 첫 전지표 하락점 (2.46M)"),
    ("R6_w96_d024_noattn", "s1", None, "순수 CNN 초경량 지점 (1.69M·+6.5%)"),
    ("L1_11_lr_fuse_w64", "s1", "신규 구조 LR-Fuse: PixelUnshuffle×4·전연산 1/16 면적·conv 6블록", "LR-only 패러다임 기각 (ERGAS 4.02)"),
]
SWIN = [
    ("SW2_add", "s1", None, "Swin@btl 존재 효과 — 11ch 에서 +2.85% 유해"),
    ("SW2_add", "s2", None, "+1.53% — 방향 일치, 서버 간 1.84% 격차 단서(불안정 조합)"),
    ("SW4_add", "s1", None, "깊이 +2블록: 미미한 회복(−0.41%), 여전히 열세"),
    ("SW6_add", "s1", None, "깊이 비단조 악화 — 확대 무효 확정"),
    ("SW2_add_9ch", "s1", None, "9ch 에선 무해(+0.23%) — 입력×attention 상호작용 2.6%p"),
    ("CM3A_btl_nopan_d124", "s1", None, "기전 대조: CM3A>Swin(−1.72%), 그러나 c6 열세(+1.08%)"),
    ("c6_d126", "s1", None, "용량 등가 대조 — 순수 conv +0.59M 은 ±0σ (용량 가설 기각)"),
    ("SW2_d024", "s2", None, "full-res 제거는 Swin 으로도 유상 (+3.35%)"),
    ("SW2_d122", "s2", None, "near-budget 승자 — btl 축소가 full-res 제거보다 낫다"),
    ("SW2_d122_w112", "s2", None, "hybrid 폭 곡선 평탄 (+0.23%)"),
    ("SW2_d122_w96", "s2", None, "직전 캠페인 Efficiency winner (1.95M·+1.95%)"),
    ("LR_SW2_w128", "s2", None, "LR-only 최종 기각 — Swin 판(용량 매칭)도 3.72"),
]
COMPRESS = [
    ("d122", "s1", None, "골격 준무손실(+0.28%) — btl depth 2~6 불감 구간"),
    ("d122_9ch", "s1", None, "9ch·무Swin 골격 (+0.83%)"),
    ("d122_w96", "s1", None, "무-Swin 폭 축소 가파름(+2.86%) — 단 s2 재현과 1.67% 격차"),
    ("d122_w96", "s2", None, "s1 과 1.67% 격차 — w96 급에서 실행 간 변동 증폭의 증거"),
    ("SW2_d122_9ch", "s1", None, "anchor 동률(2.0811)·params −8.7% — Quality winner 후보"),
    ("SW2_d122_9ch", "s2", None, "서버 재현 (차이 0.16% — w128 급은 안정)"),
    ("SW2_d122_w96_9ch", "s1", None, "9ch 압축은 비쌈(+3.89%) — w80 게이트 닫힘"),
    ("SW2_d122_w96_9ch", "s2", None, "s2 재현 (+2.58%) — 역시 서버 격차"),
    ("CM3A_d122", "s1", None, "전 캠페인 첫 anchor 명목 초과 (c6 −0.25%, 유의선 경계)"),
    ("CM3A_d122_w96", "s2", None, "w96 에선 attention 유무·종류 무관 동일 (보호 없음)"),
    ("SW2_d122_w80", "s2", None, "폭 바닥 — w80 에서 붕괴 시작 (11ch)"),
    ("SW2_d122_w80_9ch", "s2", None, "폭 바닥 — w80 붕괴 (9ch)"),
]

HEADER = ["Run", "서버", "입력", "Width", "Depth", "Attention", "Swin",
          "추가 설정 (기준과 다른 것 전부)", "ERGAS↓", "SAM↓", "SCC↑", "Q2n↑",
          "HQNR↑", "Params(M)", "Infer(ms)", "Train(h)", "판정"]

BASE = dict(in_mode="released", hidden_size=128, depth=[1, 2, 4])


def load_cfg(tag):
    for p in (os.path.join(ROOT, "config", f"{tag}.yaml"),
              os.path.join(ROOT, "config", f"pancrafter_{tag}.yaml")):
        if os.path.exists(p):
            return yaml.safe_load(open(p))
    return None


def settings_of(tag, override):
    """(입력, width, depth, attention, swin, 추가설정) — 기준과 같으면 빈칸."""
    if override is not None and load_cfg(tag) is None:
        return ("", "", "", "", "", override)
    c = load_cfg(tag)
    if c is None:
        return ("", "", "", "", "", override or "config 없음")
    ma = c.get("model_args", {})
    model = c.get("model", "")
    if "LRFuse" in model or "LRTinySwin" in model:
        arch = ("LR-TinySwin" if "TinySwin" in model else "LR-Fuse")
        sw = f"{ma.get('swin_depth', 0)}블록" if "TinySwin" in model else ""
        extra = override or (f"신규 구조 {arch}: PixelUnshuffle×4·전연산 1/16 면적·"
                             f"w{ma.get('hidden_size')}"
                             + (f"·Swin {ma.get('swin_depth')}" if "TinySwin" in model
                                else f"·conv {ma.get('n_blocks', 6)}블록"))
        inp = "9ch계열" if ma.get("in_mode", "paper") == "paper" else "11ch계열"
        return (inp, str(ma.get("hidden_size", "")), "-", "-", sw, extra)
    if "pancrafter_paper" in model:
        # 재구성본: in_mode 기본값 paper = 9ch
        inp = "9ch" if ma.get("in_mode", "paper") == "paper" else ""
    else:
        # 배포 코드 계열: 입력 11ch 하드코딩 (기준과 동일 → 빈칸)
        inp = ""
    w = "" if ma.get("hidden_size", 128) == 128 else str(ma["hidden_size"])
    d = "" if list(ma.get("depth", BASE["depth"])) == BASE["depth"] else str(ma.get("depth"))
    if ma.get("dec_depth") is not None:
        d = (d or str(ma.get("depth"))) + f" dec{ma['dec_depth']}"
    loc = ma.get("attn_locations")
    if loc is None:
        att = f"CM3A {ma.get('n_attn', 3)}개(enc/btl/dec)"
    elif not loc:
        att = ""
    else:
        att = "CM3A@" + "+".join(loc) + ("" if ma.get("cm3a_pan_branch", True) else " (PAN K/V 제거)")
    sw = ""
    if ma.get("swin_depth", 0):
        sw = (f"{ma['swin_depth']}@btl h{ma.get('swin_heads', 4)}·"
              f"w{ma.get('swin_window', 8)}·mlp{ma.get('swin_mlp_ratio', 2.0):g}")
    bits = []
    if c.get("mars", "dual") == "ms":
        bits.append("MARs 단일모드(ms)")
    if c.get("select_on", "hqnr") != "hqnr":
        bits.append(f"best={c['select_on']}")
    extra = override or " · ".join(bits)
    return (inp, w, d, att, sw, extra)


def main():
    gc = gspread.service_account(filename=CRED)
    sh = gc.open(SHEET)
    tabs = {}
    for t in ("WV3-s1", "WV3-s2"):
        ws = [w for w in sh.worksheets() if w.title == t][0]
        rows = ws.get_all_values()
        hdr = rows[2]
        idx = {h: i for i, h in enumerate(hdr)}
        by_tag = {}
        for r in rows[3:]:
            if len(r) > 1 and r[1].strip():
                by_tag[r[1].split(" (")[0]] = r
        tabs[t] = (idx, by_tag)

    def metrics(tag, server):
        t = "WV3-s1" if server in ("s1", "-") else "WV3-s2"
        idx, by = tabs[t]
        r = by.get(tag)
        if r is None:
            return [""] * 8
        g = lambda col: (r[idx[col]] if col in idx and idx[col] < len(r) else "")
        return [g("ERGAS↓"), g("SAM↓"), g("SCC↑"), g("Q2n↑"), g("HQNR↑"),
                g("Params(M)"), g("Infer(ms)"), g("Train(h)")]

    out = [[BASELINE] + [""] * (len(HEADER) - 1), [""] * len(HEADER), HEADER]
    sections = [("① 이정표 — 재현·재구성·경량화 대표", MILESTONES),
                ("② Swin·CM3A 대조 캠페인 (2026-08-29~30)", SWIN),
                ("③ 압축 귀속 캠페인 (2026-08-30~31)", COMPRESS)]
    sec_rows, hdr_row = [], 3
    for title, items in sections:
        out.append([title] + [""] * (len(HEADER) - 1))
        sec_rows.append(len(out))
        for tag, server, override, verdict in items:
            inp, w, d, att, sw, extra = settings_of(tag, override)
            out.append([tag, server, inp, w, d, att, sw, extra]
                       + metrics(tag, server) + [verdict])

    try:
        old = [w for w in sh.worksheets() if w.title == WS_NAME]
        if old:
            sh.del_worksheet(old[0])
    except Exception:
        pass
    ws = sh.add_worksheet(title=WS_NAME, rows=len(out) + 10, cols=len(HEADER) + 2)
    ws.update(out, "A1")
    fmt_hdr = CellFormat(textFormat=TextFormat(bold=True),
                         backgroundColor=Color(0.88, 0.92, 0.98))
    fmt_sec = CellFormat(textFormat=TextFormat(bold=True),
                         backgroundColor=Color(0.95, 0.95, 0.90))
    fmt_base = CellFormat(textFormat=TextFormat(italic=True, fontSize=9))
    format_cell_range(ws, f"A{hdr_row}:Q{hdr_row}", fmt_hdr)
    format_cell_range(ws, "A1:Q1", fmt_base)
    for r in sec_rows:
        format_cell_range(ws, f"A{r}:Q{r}", fmt_sec)
    ws.freeze(rows=hdr_row)
    print(f"'{WS_NAME}' 생성: {len(out)}행 "
          f"(이정표 {len(MILESTONES)} · Swin {len(SWIN)} · 압축귀속 {len(COMPRESS)})")
    missing = [(t, s) for sec in (MILESTONES, SWIN, COMPRESS)
               for t, s, _, _ in sec
               if t != "■ Paper" and not metrics(t, s)[0]]
    if missing:
        print("경고 — 원본 탭에서 못 찾은 행:", missing)


if __name__ == "__main__":
    main()
