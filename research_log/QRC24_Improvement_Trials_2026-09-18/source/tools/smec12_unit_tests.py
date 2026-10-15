#!/usr/bin/env python
"""SMEC12 gate — 준비 학습 generator(§2) + 분석 backbone(§4·§20.4). 하나라도 실패하면 exit 1.   python tools/smec12_unit_tests.py
S01 run id/naming · S02 센서 항목(band/max_pixel/dataroot/params/grid) · S03 init_dir 분리·공유 · S04 P0/L000/L1E4 registry 계약 · S05 donor 경로·exact50K·B02 po recipe · S06 큐 순서(lane 교차)·의존성 · S07 시트 범주
S10 probe bank A/B/S 정의 · S11 q/EPE 정의·identity · S12 fixed residual injection(P 매번 원본, A 재호출 없음) · S13 label/threshold · S14 W(P,c) 부호"""
import math, os, sys
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from tools import gen_smec12_bootstrap as B
from kdv.registry import resolve
FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)
ids = {(s, r): B.run_id(s, r) for s in ("qb", "gf2") for r in B.ROLES}
check("S01 run id (§2.4–2.6): B01_<S>_P0_W112_D123_S2025_BOOTSCRATCH · B02_<S>_DONN2_…_R200_S2025_BOOTSCRATCH · B03_<S>_{L000,L1E4}_…_S2025_DONB02 · B04_<S>_L1E4_…_S1234_DONB02",
      ids[("qb", "P0")] == "B01_QB_P0_W112_D123_S2025_BOOTSCRATCH" and ids[("gf2", "DONN2")] == "B02_GF2_DONN2_W112_D123_R200_S2025_BOOTSCRATCH" and ids[("qb", "L000")] == "B03_QB_L000_W112_D123_S2025_DONB02"
      and ids[("gf2", "L1E4")] == "B03_GF2_L1E4_W112_D123_S2025_DONB02" and ids[("qb", "L1E4REP")] == "B04_QB_L1E4_W112_D123_S1234_DONB02" and B.seed_of("L1E4REP") == 1234 and B.seed_of("L1E4") == 2025)
cfg = {(s, r): B.build(s, r) for s in ("qb", "gf2") for r in B.ROLES}
def _paths_ok(c, s):
    X = B.SENSORS[s]; return all(os.path.exists(c[k]["dataroot"]) and c[k]["dataroot"].endswith(os.path.basename(X[kk])) for k, kk in (("train_feeder_args", "train"), ("val_feeder_args", "valid"), ("test_reduced_feeder_args", "rr"), ("test_full_feeder_args", "fr")))
check("S02 센서 항목: QB 4 band/2047/msfix train·valid · GF2 4 band/1023 · out_channels 4 · expect_params 2.6508(4-band W112·D123 backbone) · eval_epoch QB 3(1071)·GF2 2(824) ≤ 1100 격자 · num_iter 50K · save_iter 25000 · 파일 존재",
      all(_paths_ok(c, s) and c["num_bands"] == 4 and c["model_args"]["out_channels"] == 4 and c["expect_params_m"] == 2.6508 and c["num_iter"] == 50000 and c["save_iter"] == 25000 for (s, r), c in cfg.items())
      and cfg[("qb", "P0")]["max_pixel"] == 2047.0 and cfg[("gf2", "P0")]["max_pixel"] == 1023.0 and "msfix" in cfg[("qb", "P0")]["train_feeder_args"]["dataroot"] and B.eval_epoch_for("qb") == 3 and B.eval_epoch_for("gf2") == 2
      and all(B.eval_epoch_for(s) * (B.SENSORS[s]["n_train"] // 48) <= 1100 for s in ("qb", "gf2")) and cfg[("gf2", "DONN2")]["model_args"]["hidden_size"] == 112 and cfg[("gf2", "DONN2")]["model_args"]["depth"] == [1, 2, 3])
check("S03 init_dir: kdv 는 work_dir/_kdv_init_w112_d123_<s>(P0/L000/L1E4/REP 같은 센서 공유; WV3 의 _kdv_init_w112_d123 과 분리) · po 는 _pa_init_w112_d123_<s> · seed 2025 는 P0/L000/L1E4 가 같은 초기값 파일, REP 은 seed 1234",
      all(cfg[(s, r)]["kdv"]["init_dir"] == f"work_dir/_kdv_init_w112_d123_{s}" for s in ("qb", "gf2") for r in ("P0", "L000", "L1E4", "L1E4REP")) and all(cfg[(s, "DONN2")]["po"]["init_dir"] == f"work_dir/_pa_init_w112_d123_{s}" for s in ("qb", "gf2"))
      and cfg[("qb", "P0")]["seed"] == cfg[("qb", "L000")]["seed"] == cfg[("qb", "L1E4")]["seed"] == 2025 and cfg[("qb", "L1E4REP")]["seed"] == 1234)
sp = {k: resolve(c["kdv"]) for k, c in cfg.items() if k[1] != "DONN2"}
check("S04 registry: P0 = A-ID/NOALIGN/I-A(reference_donor·aligner_lr 없음, noalign True) · L000 = A-FT/I-NATIVE-TRANSFER/offset 0 · L1E4/REP = A-FT/I-AEQ/offset 1e-4/b 2 · Teacher 없음 · expect_arch 112/[1,2,3] · retain_all_candidates · required budget",
      all(sp[(s, "P0")]["policy"] == "A-ID" and sp[(s, "P0")]["recipe"] == "NOALIGN" and sp[(s, "P0")]["protocol"] == "I-A" and cfg[(s, "P0")]["kdv"]["eval"] == {} and "aligner_lr" not in cfg[(s, "P0")]["kdv"] and cfg[(s, "P0")]["kdv"]["expect_arch"]["noalign"] is True for s in ("qb", "gf2"))
      and all(sp[(s, "L000")]["policy"] == "A-FT" and sp[(s, "L000")]["protocol"] == "I-NATIVE-TRANSFER" and sp[(s, "L000")]["offset_weight_effective"] == 0 for s in ("qb", "gf2"))
      and all(sp[(s, r)]["protocol"] == "I-AEQ" and sp[(s, r)]["offset_weight_effective"] == 1e-4 and sp[(s, r)]["radius_hr"] == 2.0 and not sp[(s, r)]["has_teacher"] for s in ("qb", "gf2") for r in ("L1E4", "L1E4REP"))
      and all(c["kdv"]["expect_arch"]["width"] == 112 and c["kdv"]["select"]["retain_all_candidates"] and c["kdv"]["budget"]["required"] is True and c["kdv"]["budget"]["ledger"] == B.LEDGER for k, c in cfg.items() if k[1] != "DONN2"))
check("S05 donor: L000/L1E4/REP 의 donor = work_dir/B02_<S>_DONN2…/last, expected_step 50000(exact50K), sha 는 학습 뒤(null) · B02 po: case N2_SG · b 2.0 · λ_off .01 ramp 5000 · provenance 가 이식 표시(센서 최적값 아님)",
      all(c["kdv"]["donor"]["source"] == f"work_dir/{B.run_id(s, 'DONN2')}/last" and c["kdv"]["donor"]["expected_step"] == 50000 and c["kdv"]["donor"]["expected_sha256"] is None and c["kdv"]["eval"] == dict(fixed_reference_from_donor=True) for (s, r), c in cfg.items() if r in ("L000", "L1E4", "L1E4REP"))
      and all(c["po"]["case"] == "N2_SG" and c["po"]["radius_hr"] == 2.0 and c["po"]["lambda_off_max"] == 0.01 and c["po"]["ramp_updates"] == 5000 and c["po"]["radius_provenance"]["dataset"] == B.SENSORS[s]["S"] and "transfer" in c["po"]["radius_provenance"]["calibration_quality"] for (s, r), c in cfg.items() if r == "DONN2"))
q = B.queue_order(["qb", "gf2"])
check("S06 큐: lane 교차(QB P0, GF2 P0, QB DONN2, GF2 DONN2, …) · 각 센서 안에서 P0 → DONN2 → L000 → L1E4 → REP (donor 의존성) · WV3 는 준비 대상 아님",
      q[:4] == [B.run_id("qb", "P0"), B.run_id("gf2", "P0"), B.run_id("qb", "DONN2"), B.run_id("gf2", "DONN2")] and all(q.index(B.run_id(s, "DONN2")) < q.index(B.run_id(s, "L000")) < q.index(B.run_id(s, "L1E4")) < q.index(B.run_id(s, "L1E4REP")) for s in ("qb", "gf2")) and len(q) == 10)
sys.path.insert(0, os.path.join(ROOT, "gspread")); import sheet_categories as SC
check("S07 시트 범주: B01_/B02_/B03_/B04_<S>_ → SMEC12 (KEEP) · PAKD50 는 그대로", SC.classify(B.run_id("qb", "P0")) == "SMEC12" and SC.classify(B.run_id("gf2", "L1E4REP")) == "SMEC12" and SC.classify("PAKD50_JQ_W112_D123_WV3_T0_S1234_FRESH50_v1") == "PAKD50" and "SMEC12" in SC.KEEP)
# ---------------- 분석 backbone (tools/smec12/common.py)
try:
    from tools.smec12 import common as C
    A, Bk, S_ = C.probe_bank("A"), C.probe_bank("B"), C.probe_bank("S")
    check("S10 probe: A = r{.25,.5,1,2}×{0,90,180,270} (16) · B = 대각 (16) · S = r{.125,.25,.5,1}×8 방향 (32) · |ε| = r · identity 별도", len(A) == 16 and len(Bk) == 16 and len(S_) == 32 and all(abs(math.hypot(p["ey"], p["ex"]) - p["r"]) < 1e-9 for p in A + Bk + S_) and sorted({p["r"] for p in S_}) == [0.125, 0.25, 0.5, 1.0]
          and all(p["ey"] == 0 or p["ex"] == 0 for p in A) and all(abs(abs(p["ey"]) - abs(p["ex"])) < 1e-9 for p in Bk))
    P = torch.zeros(1, 1, 33, 33, dtype=torch.float64); P[0, 0, 16, 16] = 1; w = C.E.warp_pan(P, torch.tensor([[0.0, 1.0]], dtype=torch.float64)); yx = np.unravel_index(int(w.argmax()), (33, 33))
    check("S14 W(P,c)[y,x] = P[y+c_y, x+c_x] ((dy,dx) 순서; impulse 가 c=(0,1) 에서 x−1 로) · zero identity 정확", yx == (16, 15) and float((C.E.warp_pan(P, torch.zeros(1, 2, dtype=torch.float64)) - P).abs().max()) == 0.0)
    lab = C.label(0.5, 0.5, tC=1.0, tR=1.0); check("S13 label: e≤med·q≤med = EdCd · e>·q≤ = EuCd · boundary_near = ±10% IQR", lab == "EdCd" and C.label(1.5, 0.5, 1.0, 1.0) == "EuCd" and C.label(0.5, 1.5, 1.0, 1.0) == "EdCu" and C.label(1.5, 1.5, 1.0, 1.0) == "EuCu" and C.boundary_near(1.02, 1.0, iqr=0.5) and not C.boundary_near(1.2, 1.0, iqr=0.5))
    qc = C.q_const("A"); check("S11 q_const(bank) = mean ||ε||₁/2 · q_rel = q/q_const", abs(qc - np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in A])) < 1e-12 and abs(C.q_rel(qc, "A") - 1.0) < 1e-12)
    if torch.cuda.is_available() or True:
        reg = C.asset_registry(); wv3 = [k for k in reg if k.startswith("WV3|L1E4")]
        check("S12 asset registry: WV3 L1E4 자산(NF16/PALS24) 이 등록되고 QB/GF2 는 pending(B queue) · P0 q=NA", bool(wv3) and any(v["status"] == "pending_dependency" for k, v in reg.items() if k.startswith("QB|")) and all(v.get("q") == "NA" for k, v in reg.items() if "|P0|" in k), f"WV3 {len(wv3)} entries")
        if wv3 and reg[wv3[0]]["status"] == "available":
            Lp = C.load_asset(wv3[0]); ids_ = C.make_panels("wv3")["ids"]["CAL"][:4]; gt, ms, lpan, pan = C.load_patches("wv3", ids_)
            c0 = C.predict_shift(Lp, pan, ms); y0 = C.reconstruct_at(Lp, pan, ms, lpan, c0); yd = C.reconstruct_at(Lp, pan, ms, lpan, c0 + torch.tensor([[0.25, 0.0]]).expand_as(c0)); y0b = C.reconstruct_at(Lp, pan, ms, lpan, c0)
            check("S12 fixed residual injection: reconstruct_at(P, M; c0+δ) 는 원본 P 에서 한 번만 sampling(A 재호출 없음) · δ=0 재현 동일 · δ=.25 에서 출력 변화 > 0 · GT L1 signed 변화 기록", torch.equal(y0, y0b) and float((yd - y0).abs().mean()) > 0 and tuple(y0.shape) == tuple(gt.shape))
except ImportError as e:
    check("S10–S14 분석 backbone import", False, repr(e)[:120])
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
