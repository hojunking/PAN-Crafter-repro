#!/usr/bin/env python
"""DCR12 gate (계획 §4.2 중 코드로 닫는 것 + 진단 도구 정의). 하나라도 실패하면 exit 1.   python tools/dcr12_unit_tests.py
X01 probe 집합 · X02 closure 불변·RNG 격리 · X03 R/edge 정의 · X04 label 규칙 · X05 panel 분리 · X06 loss 항 == trainer 분해 · X07 D03 per-sample(자산 불변) · X08 D04 episode 분리 · X09 cosine LR · X10 gate 논리 · X11 개입 정의 · X12 완료 run 만 · X13 판정 라벨"""
import copy, math, os, sys
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from tools.dcr12 import common as C
FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)
pm, pc = C.probe_set("main"), C.probe_set("confirm")
check("X01 probe: main 16 = r{0.5,1.0} × θ=kπ/4 · confirm 16 = θ+π/8 (겹치지 않음) · |ε| = r · identity 별도", len(pm) == 16 and len(pc) == 16 and all(abs(math.hypot(p["ey"], p["ex"]) - p["r"]) < 1e-9 for p in pm + pc)
      and not ({(p["ey"], p["ex"]) for p in pm} & {(p["ey"], p["ex"]) for p in pc}) and (0.0, 0.0) not in {(p["ey"], p["ex"]) for p in pm + pc} and sorted({p["r"] for p in pm}) == [0.5, 1.0])
T = C.load_pipe("T"); panels = C.make_panels(); ids = panels["ids"]["CAL"][:8]; gt, ms, lpan, pan = C.load_patches(ids)
st = torch.get_rng_state(); r0 = C.consistency(T, pan, ms); rb = C.consistency(T, pan, ms, c_bias=(0.25, 0.0)); r1 = C.consistency(T, pan, ms)
check("X02 closure 불변: 두 correction 에 같은 b 를 더해도 C 동일(≤1e-6) · 반복 동일 · 전역 RNG 미소비 · identity floor < 1e-4 · C = mean|resid| 정의", float((rb["C"] - r0["C"]).abs().max()) < 1e-6 and torch.equal(r0["C"], r1["C"]) and torch.equal(st, torch.get_rng_state())
      and float(r0["identity"].max()) < 1e-4 and torch.allclose(r0["C"], r0["resid"].abs().mean(dim=(1, 2))) and set(r0["C_by_r"]) == {"0.5", "1.0"}, f"maxΔC {float((rb['C'] - r0['C']).abs().max()):.1e}")
rm = C.r_metrics(gt, gt); rm2 = C.r_metrics(gt + 0.1, gt)
check("X03 R_plain = mean|ŷ−y| 전체, R_interior = margin 16 (32²), edge_plain: 동일 입력 0 · 상수 오프셋 R 0.1 · edge 0", float(rm["R_plain"].max()) == 0 and float(rm["edge_plain"].max()) == 0 and torch.allclose(rm2["R_plain"], torch.full_like(rm2["R_plain"], 0.1), atol=1e-6) and float(rm2["edge_plain"].max()) < 1e-6)
from tools.dcr12.d01 import label
check("X04 label: (C≤tC, R≤tR)=A · (low, high)=B · (high, low)=C · (high, high)=D (≤ 가 low)", label(1.0, 1.0, 1.0, 1.0) == "A" and label(1.0, 1.1, 1.0, 1.0) == "B" and label(1.1, 1.0, 1.0, 1.0) == "C" and label(1.1, 1.1, 1.0, 1.0) == "D")
S = panels["ids"]; blocks = {k: {C.group_of(i) for i in S[k]} for k in ("CAL", "DISC", "CONF")}
check("X05 panel: CAL 1024 / DISC 512 / CONF 512 · block 단위 서로 겹치지 않음 · DISC128 ⊂ DISC · FR8 ⊂ 0..19 · seed 314159", len(S["CAL"]) == 1024 and len(S["DISC"]) == 512 and len(S["CONF"]) == 512 and not (blocks["CAL"] & blocks["DISC"]) and not (blocks["CAL"] & blocks["CONF"]) and not (blocks["DISC"] & blocks["CONF"])
      and set(S["DISC128"]) <= set(S["DISC"]) and len(S["DISC128"]) == 128 and set(S["FR8"]) <= set(range(20)) and len(S["FR8"]) == 8 and panels["seed"] == 314159 and panels["independence"].startswith("patch_only"))
cal = C.calibration(); Sm = copy.deepcopy(T.m).train().requires_grad_(True); g7 = torch.Generator().manual_seed(7)
with torch.no_grad():
    for p_ in Sm.backbone.parameters():
        p_.add_(0.01 * torch.randn(p_.shape, generator=g7).to(p_.device))
gd, md, ld, pd_ = (t[:2].to(C.DEV) for t in (gt, ms, lpan, pan)); tr = C.trainer_stub("JK0", 1234, Sm, T.m, cal); tot, info = tr._step(gd, md, ld, pd_, 1); q = C.q12_terms(info["y"].detach(), info["y_t"], gd, cal)
check("X06 q12_terms == trainer 분해: L_H, L_K, λE·L_E 가 _step 의 _rec_hard_t/_rec_soft_t/_edge_w_t 와 같다 (1e-6) · total = L_H + L_K + λE L_E + 1e-4 L_off", abs(float(q["LH"]) - float(info["_rec_hard_t"])) < 1e-6 and abs(float(q["LK"]) - float(info["_rec_soft_t"])) < 1e-6
      and abs(cal["lambda_E"] * float(q["LE"]) - float(info["_edge_w_t"])) < 1e-6 and abs(float(tot) - float(q["LH"] + q["LK"] + cal["lambda_E"] * q["LE"] + 1e-4 * info["loss_off"])) < 1e-6)
from tools.dcr12 import d03
m3 = d03.open_A(T); A = list(m3.aligner.parameters()); h0 = C.state_hash(m3.aligner)
with torch.no_grad():
    yt = T.m(pd_[:1], md[:1], ld[:1])["y"]
row = d03.per_sample(T, m3, A, pd_[:1], md[:1], ld[:1], gd[:1], yt, cal, floor=1e-9)
check("X07 D03 per-sample: norm 유한 · cos ∈ [−1,1] · gT = gH + gE 관계 · virtual step 뒤 A 복원(hash 불변) · closure 두 값 · dL/dc 4 값", all(np.isfinite(row[k]) for k in ("norm_L0", "norm_LH", "norm_LE", "norm_LK", "norm_LC", "norm_T"))
      and all(row[k] is None or -1 <= row[k] <= 1 for k in ("cos_g0_gC", "cos_gT_gC", "cos_gT_gK")) and C.state_hash(m3.aligner) == h0 and all(k in row for k in ("closure_after_sg_fixed", "closure_after_recomputed", "dL0_dc_y", "dLE_dc_x")) and T.aligner_sha16 == C.state_hash(T.m.aligner),
      f"cos(gT,gC) {row['cos_gT_gC']}")
from tools.dcr12 import d04
rng = np.random.RandomState(0); cid = list(range(1000, 1100)); rT = rng.rand(100); sp = d04.episode_split(cid, rT, rng)
check("X08 D04 episode: support 64(4×16) 와 query 16 분리 · 같은 cell · R_T 전 구간(query 가 support 의 R_T 범위 안) · 80 미만이면 None", sp is not None and len(sp[0]) == 64 and len(sp[1]) == 16 and not (set(sp[0]) & set(sp[1])) and set(sp[0] + sp[1]) <= set(cid)
      and d04.episode_split(cid[:70], rT[:70], rng) is None and (lambda q, s: min(q) >= min(s) - 1e-9 and max(q) <= max(s) + 1e-9)(rT[[cid.index(i) for i in sp[1]]], rT[[cid.index(i) for i in sp[0]]]))
check("X09 cosine LR: warmup 100 → peak · 50000 → 0 · 25250 = peak·½(1+cos(π·25150/49900))", abs(C.cosine_lr(100, 1e-4) - 1e-4) < 1e-12 and abs(C.cosine_lr(50000, 1e-4)) < 1e-12 and abs(C.cosine_lr(25250, 1e-4) - 1e-4 * 0.5 * (1 + math.cos(math.pi * 25150 / 49900))) < 1e-15)
thr = dict(T0=dict(degenerate_metric=False, C_cv=0.3))
def _df(harm_cell="B", neg=True):
    rows = []
    for pn in ("DISC", "CONF"):
        for stp in (25250, 45450):
            for c in "ABCD":
                for ep in range(8):
                    u = (-0.002 if (c == harm_cell and neg) else 0.002) + 0.0002 * np.random.RandomState(ep + 3 * stp).randn(); rows.append(dict(panel=pn, step=stp, cell=c, episode=ep, U=u, group_key=f"g{ep}"))
    return pd.DataFrame(rows)
g_pass = d04.gate(_df("B", True), thr, None); g_fail = d04.gate(_df("B", False), thr, None)
check("X10 D04 gate: DISC/CONF·두 checkpoint 에서 한 cell 만 일관되게 U<0 이고 CONF CI 가 음이면 통과(z*=B) · 전부 양이면 NOT_OPENED", g_pass["passed"] and g_pass["z_star"] == "B" and not g_fail["passed"] and g_fail["status"] == "NOT_OPENED_NO_INCREMENTAL_EVIDENCE")
from tools.dcr12.d02 import modes_of
c0 = torch.tensor([[0.3, -0.2]]); mo = modes_of(c0)
check("X11 개입: I0 = c0 · IZ = 0 · IN = −c0 · IB±y/x = c0 ± 0.25 (HR px)", torch.equal(mo["I0"], c0) and float(mo["IZ"].abs().sum()) == 0 and torch.equal(mo["IN"], -c0) and torch.allclose(mo["IB+y"], c0 + torch.tensor([0.25, 0.0])) and torch.allclose(mo["IB-x"], c0 - torch.tensor([0.0, 0.25])))
av = C.available_students(("best_hqnr",)); inc = [(ck, s) for s in C.SEEDS for ck in ("B0", "B1") if not C.run_complete(ck, s)]
check("X12 완료 run 만 진단에 쓴다: 미완 run 은 available_students 에 없음 · B0 S1234 는 완료 재사용", all(C.run_complete(ck, s) for ck, s, _ in av) and all((ck, s, "best_hqnr") not in av for ck, s in inc) and ("B0", 1234, "best_hqnr") in av, f"available {av} · incomplete {inc}")
from tools.dcr12.report import verdicts
prA = pd.DataFrame([dict(status="paired", d_selected_hqnr=0.001), dict(status="paired", d_selected_hqnr=0.002)]); prB = pd.DataFrame([dict(status="paired", d_selected_hqnr=0.001), dict(status="paired", d_selected_hqnr=-0.002)])
vA = verdicts({}, {}, {}, prA, None); vB = verdicts({}, {}, {}, prB, None)
check("X13 판정 라벨 (§14.1): 두 seed 양 → TASK_ADAPTATION_CANDIDATE · 부호 반전 → INCONCLUSIVE", any(x.startswith("TASK_ADAPTATION_CANDIDATE") for x in vA["labels"]) and any(x.startswith("INCONCLUSIVE") for x in vB["labels"]))
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
