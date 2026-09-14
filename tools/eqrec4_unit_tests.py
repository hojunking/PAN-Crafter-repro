#!/usr/bin/env python
"""EQREC4 gate (계획 §6·§14.2 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/eqrec4_unit_tests.py
E01 probe bank 정의 · E02 warp 부호/identity · E03 분할 manifest(블록 배정·역할 n) · E04 q 정의(residual, identity 미포함, q_const) · E05 routing 수신자(실제 wrapper)
E06 bias 대수 항등(q 불변) · E07 fit_response 선형 fit · E08 EA/EAQ shrinkage·ρ 질량 · E09 conditional shuffle · E10 block bootstrap"""
import math, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from tools.eqrec4 import common as C
FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)
torch.manual_seed(0)
A, B = C.probe_bank("A"), C.probe_bank("B")
check("E01 probe bank: A 4 반경 × 4 축, B 4 반경 × 4 대각 (16+16), identity 별도, (dy,dx)=(r sinθ, r cosθ)", len(A) == 16 and len(B) == 16 and all(abs(math.hypot(p["ey"], p["ex"]) - p["r"]) < 1e-9 for p in A + B) and all(p["ey"] == 0 or p["ex"] == 0 for p in A) and all(abs(abs(p["ey"]) - abs(p["ex"])) < 1e-9 for p in B)
      and (0.0, 0.0) not in [(p["ey"], p["ex"]) for p in A + B])
qc = float(np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in A])); qcB = float(np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in B]))
check("E01 무반응 기준 q_const: 축 bank r/2 평균, 대각 bank r/√2 평균 (§4.2 L1 기하 차이)", abs(qc - np.mean(C.RADII) / 2) < 1e-9 and abs(qcB - np.mean(C.RADII) / math.sqrt(2)) < 1e-9, f"{qc:.4f} / {qcB:.4f}")
P = torch.zeros(1, 1, 33, 33, dtype=torch.float64); P[0, 0, 16, 16] = 1; w = C.warp_pan(P, torch.tensor([[0.0, 1.0]], dtype=torch.float64)); yx = np.unravel_index(int(w.argmax()), (33, 33))
check("E02 W(P,c)[y,x]=P[y+cy,x+cx]: impulse 가 c=(0,1) 에서 x−1 로 · zero identity 정확", yx == (16, 15) and float((C.warp_pan(P, torch.zeros(1, 2, dtype=torch.float64)) - P).abs().max()) == 0.0)
# E03 manifest (임시 root)
_camp = C.CAMP; C.CAMP = os.path.join(ROOT, "work_dir", "_eqrec4_unit_tmp"); os.makedirs(C.CAMP, exist_ok=True)
try:
    for f in ("data_manifest.csv", "data_manifest_meta.json"):
        p = os.path.join(C.CAMP, f); os.path.exists(p) and os.remove(p)
    man = C.make_manifest(); n = man.groupby("split_role").sample_id.count().to_dict(); blk = man.groupby("source_group_id").split_role.nunique()
    check("E03 분할: A 512 / B 2048 / C 512 / D 1024 · 블록(32) 이 한 역할에만 · seed 314159 · 중복 없음", n == dict(A=512, B=2048, C=512, D=1024) and int(blk.max()) == 1 and man.sample_id.is_unique and bool(man.seen_in_pretraining.all()), str(n))
    man2 = C.make_manifest(); check("E03 manifest 재현 (같은 파일 재사용)", man2.sample_id.tolist() == man.sample_id.tolist())
finally:
    import shutil; shutil.rmtree(C.CAMP, ignore_errors=True); C.CAMP = _camp
# E04–E06 with a real model (primary), 4 patches
L = C.load_model(*C.PRIMARY); man = C.make_manifest(); ids = man[man.split_role == "A"].sample_id.iloc[:4].tolist(); gt, ms, lpan, pan = C.load_patches(ids); pr = C.probe_responses(L, pan, ms)
r = pr["A"]["resid"]; q_manual = (r.abs().sum(2).sum(1) / (2 * 16)); check("E04 q_A = 1/(2K) Σ_k ||r_k||₁ · EPE = mean ||r_k||₂ · λ 를 곱하지 않음", torch.allclose(pr["A"]["q"], q_manual, atol=1e-7) and torch.allclose(pr["A"]["epe"], r.norm(dim=2).mean(1), atol=1e-7))
m = L.m; mb = torch.nn.functional.interpolate(ms[:1].to(C.DEV), scale_factor=4, mode="bicubic"); p = pan[:1].to(C.DEV); c0 = C.predict_c(m.aligner, p, mb, L.mg); e = torch.tensor([[A[5]["ey"], A[5]["ex"]]], device=C.DEV); ce = C.predict_c(m.aligner, C.warp_pan(p, e), mb, L.mg)
check("E04 residual 정의 r = ĉε + ε − ĉ0 (probe 하나 직접 재계산; batch 1 vs 4 의 cuDNN 차이 허용 1e-4 px)", torch.allclose((ce + e - c0)[0].cpu(), r[0, 5], atol=1e-4), f"max|Δ| {float(((ce + e - c0)[0].cpu() - r[0, 5]).abs().max()):.1e}")
v = torch.tensor([[0.5, 0.0]], device=C.DEV); check("E06 A_v = A + v: (ĉε+v)+ε−(ĉ0+v) = r (q 불변, 음성 대조의 대수 항등)", torch.allclose(((ce + v) + e - (c0 + v)), (ce + e - c0), atol=1e-7))
ft = C.fit_response(np.array([[-ey, -ex] for ey, ex in [(p_["ey"], p_["ex"]) for p_ in A]]), A); check("E07 fit_response: 이상 반응 ĉε−ĉ0 = −ε 이면 B = −I, b = 0, rmse 0", abs(ft["B_yy"] + 1) < 1e-9 and abs(ft["B_xx"] + 1) < 1e-9 and abs(ft["B_yx"]) < 1e-9 and ft["fit_rmse"] < 1e-9)
from tools.eqrec4.k10 import receiver_check
gr = receiver_check("A"); check("E05 routing 수신자(실제 wrapper, Pair A): Rec→A/U · Off→A only · extra(hard/soft)→U only (A grad 가 R0/RH/RS 에서 동일) · Teacher 불변", gr["ok"], str({a: (v["g_base_A"] > 0, v["g_extra_U"]) for a, v in gr["arms"].items() if isinstance(v, dict) and "g_base_A" in v}))
# E08 utility tables / rho mass, E09 shuffle
import pandas as pd
from tools.eqrec4 import k20 as K
ut = pd.DataFrame([dict(pair="A", repeat=0, cell=c, n_source_groups=3, U_soft_hard_rel=u) for c, u in [("EdCd_Apos", .02), ("EdCd_Apos", .03), ("EdCu_Apos", -.01), ("EuCd_Aneg", .0), ("EuCu_Aneg", -.02)]])
dfB = pd.DataFrame(dict(cell=["EdCd_Apos"] * 30 + ["EdCu_Apos"] * 30 + ["EuCd_Aneg"] * 20 + ["EuCu_Aneg"] * 20, e_T=np.random.rand(100), a_T=np.random.randn(100), e_low=[True] * 60 + [False] * 40, a_pos=[True] * 60 + [False] * 40, c_low=[True] * 30 + [False] * 30 + [True] * 20 + [False] * 20, q_A=np.random.rand(100)))
_nz = os.path.join(C.CAMP, "k10_noise_A.json"); _had = os.path.exists(_nz)
tab = K.utility_tables(ut, "A", dfB); ea = tab["EA"]["EdApos"]; eaq = tab["EAQ"]["EdCd_Apos"]
check("E08 EA 셀 = e·a 만 (q 제거), EAQ 는 부모 EA 로 shrink (n·ū + 4·ū_parent)/(n+4), 지원 없는 셀은 부모값", abs(ea["raw"] - (.02 + .03 - .01) / 3) < 1e-9 and abs(eaq["value"] - (2 * .025 + 4 * ea["value"]) / 6) < 1e-9 and tab["EAQ"]["EdCu_Aneg"]["value"] == tab["EA"]["EdAneg"]["value"])
rho, info = K.rho_for(dfB, tab["EAQ"], "EAQ", tab); rho2, info2 = K.rho_for(dfB, tab["EA"], "EA", tab)
check("E08 ρ = σ(clip(ū/s_u,−4,4)+b_mass), B 평균 0.5 (EA·EAQ 각각), 순서 보존", abs(info["mean_rho"] - 0.5) < 1e-6 and abs(info2["mean_rho"] - 0.5) < 1e-6 and rho[0] > rho[30], f"b_mass {info['b_mass']:.3f}")
dfB["source_group_id"] = "b"; dfB["sample_id"] = np.arange(len(dfB)); dfS, stt, prs = K.conditional_shuffle(dfB, 1)
check("E09 conditional shuffle: e/a 셀 유지 · q-label 만 2D rank 블록(≤16, ≥8) 안에서 derangement · partner/거리 기록 · 8 미만은 unmatched", (dfS.cell.map(K.ea_key) == dfB.cell.map(K.ea_key)).all() and (dfS.e_T.values == dfB.e_T.values).all() and (dfS.cell.values != dfB.cell.values).mean() > 0
      and (prs.status == "ok").sum() > 0 and prs[prs.status == "ok"].e_rank_dist.max() <= 32 and (prs[prs.status == "ok"].partner_sample_id >= 0).all())
tiny = dfB.iloc[:5].copy(); _, st_t, _ = K.conditional_shuffle(tiny, 1); check("E09 블록이 8 미만이면 shuffle_unmatched (라벨 유지)", (st_t == "shuffle_unmatched").all())
from tools.eqrec4.k10 import fit_pools
pB = pd.DataFrame(dict(cell=["EdCd_Aneg"] * 200, source_group_id=[f"blk{i // 10:03d}" for i in range(200)], e_T=np.random.rand(200), a_T=-np.random.rand(200), q_A=np.random.rand(200)))
pools = fit_pools(pB, 1); blocks = [set(p["source_blocks"]) for p in pools]
check("E11 K10 fit pools: source block 단위 배정, pool 간 block 공유 없음, 48 고유, ≤4 pool", len(pools) == 4 and all(p["n_unique"] == 48 and not p["cycled"] for p in pools) and all(not (blocks[i] & blocks[j]) for i in range(4) for j in range(i + 1, 4)))
# 실제 자료 형태: 한 block(32 연속 index) 의 sample 이 여러 cell 에 흩어진다 → block 을 cell 에 배타 배정 (2026-09-15 K10 실패 재현·수정)
_rng = np.random.RandomState(3); pM = pd.DataFrame(dict(source_group_id=[f"blk{i // 32:03d}" for i in range(2048)], e_T=_rng.rand(2048), a_T=-_rng.rand(2048), q_A=_rng.rand(2048)))
pM["cell"] = _rng.choice(["EdCd_Aneg", "EdCu_Aneg", "EuCd_Aneg", "EuCu_Aneg"], size=2048, p=[.22, .29, .27, .22]); pM.loc[:4, "cell"] = "EdCd_Apos"   # 5 sample 짜리 cell 은 pool 불가
poolsM = fit_pools(pM, 1); bM = [b for p in poolsM for b in p["source_blocks"]]; cellsM = {p["cell"] for p in poolsM}
check("E11 다중 cell(block 이 4 cell 에 걸침): 전역에서 block 공유 없음 · 4 cell 모두 pool 있음 · <32 sample cell 은 제외 · pool 의 row 는 자기 cell 의 자기 block 뿐",
      len(bM) == len(set(bM)) and cellsM == {"EdCd_Aneg", "EdCu_Aneg", "EuCd_Aneg", "EuCu_Aneg"} and "EdCd_Apos" not in cellsM
      and all((pM.loc[p["rows"], "cell"] == p["cell"]).all() and set(pM.loc[p["rows"], "source_group_id"]) <= set(p["source_blocks"]) for p in poolsM) and len(poolsM) >= 8, f"pools {len(poolsM)} cells {sorted(cellsM)}")
# Pair B 형태(2026-09-15 02:04 실패 재현): 큰 cell 4 + 자격은 있으나 block 당 <1 sample 인 작은 cell 2 → 작은 cell 이 block 을 삼키지 않고 큰 cell 이 pool 을 얻는다
pI = pd.DataFrame(dict(source_group_id=[f"blk{i // 32:03d}" for i in range(2048)], e_T=_rng.rand(2048), a_T=_rng.rand(2048), q_A=_rng.rand(2048)))
pI["cell"] = _rng.choice(["EdCu_Apos", "EuCd_Apos", "EdCd_Apos", "EuCu_Apos", "EdCu_Aneg", "EuCu_Aneg"], size=2048, p=[.26, .24, .23, .22, .026, .024])
poolsI = fit_pools(pI, 1); bI = [b for p in poolsI for b in p["source_blocks"]]; perI = {c: sum(1 for p in poolsI if p["cell"] == c) for c in pI.cell.unique()}
check("E11 불균형(큰 4 cell + 수율 <1/block 인 작은 cell): 큰 cell 마다 ≥2 pool · 작은 cell 0 · 전역 block 분리 · 전부 48 고유", all(perI[c] >= 2 for c in ("EdCu_Apos", "EuCd_Apos", "EdCd_Apos", "EuCu_Apos")) and perI.get("EdCu_Aneg", 0) == 0 == perI.get("EuCu_Aneg", 0)
      and len(bI) == len(set(bI)) and all(p["n_unique"] == 48 and not p["cycled"] for p in poolsI), str(perI))
bb = C.block_bootstrap(np.r_[np.ones(50), -np.ones(50)] * 0.1 + np.random.randn(100) * 0.01, np.repeat(np.arange(10), 10)); check("E10 block bootstrap: 평균 ≈ 0, CI 가 0 을 포함, n_groups 10", abs(bb["point"]) < 0.05 and bb["ci95"][0] < 0 < bb["ci95"][1] and bb["n_groups"] == 10)
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
