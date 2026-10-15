#!/usr/bin/env python
"""s2 W112·D123 KDV config 생성 (plan §13.2 Q00–Q08 · §22 config 계약 · 이름 규칙 §13.1).

    python tools/gen_kdv_configs.py                    # config/S2W112_*.yaml 9벌 + config/queues/kdv_s2.txt
    python tools/gen_kdv_configs.py --updates 300 --version dry   # s1 dry run 용 (짧은 학습)

템플릿 = config/PA_A1_REC_W96_D124_9CH_S<seed>.yaml (골격·입력·optimizer·평가 주기 그대로), trainer 만 kdv, 폭 112 · depth [1,2,3] (2026-09-10 사용자 결정 — 계획 원안 [1,2,4] 는 --depth 1,2,4).
Teacher(Q01) 는 seed --teacher-seed(기본 2025) 의 독립 초기값·데이터 순서로 학습한 T112_DONORFROZEN_REC — Student 와 seed 가 같으면 Q02 와 같은 모델이 되므로 다르게 둔다.
Student(Q00·Q02–Q08) 는 seed --seed(기본 1234) 의 저장된 같은 초기값(work_dir/_kdv_init_w112_d123/init_unet_seed1234.pt) 을 공유한다 (§4.3).
"""
import argparse, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, run_name, describe

DONOR = "assets/donor_aligner/PA_A1_REC_W96_D124_9CH_S2025_best_hqnr_aligner.pt"


def build_kdv(q, seed, teacher_run, pilot_run, donor, teacher_id):
    """Q 번호 → kdv dict (§13.2)."""
    base = dict(campaign_id="S2_W112_KDV_20260910", run_kind="CONTROLLED", input_protocol="I-A", diag_every=1000,
                calibration=dict(n_patches=3072, seed=1234), corruption=dict(corruption_seed_offset=2000))
    rec = lambda c: dict(case=c, alpha=1.0, kd_weight=0.1, eps=1.0e-6, tau="calibrate", eps_scale=1.0e-6)
    stat_off = dict(enabled=False)
    stat = lambda mode: dict(enabled=True, kind="GV", mode=mode, window=5, alpha=1.0, kd_weight=0.1, tau="calibrate", outer_weight="calibrate",
                             lambda_pilot=f"{pilot_run}/last", r_grad=0.05, ramp_updates=0)
    teacher = dict(id=teacher_id, run=f"work_dir/{teacher_run}", tag="best_hqnr", expected_sha256=None, bridge=False)
    donor_d = dict(source=donor, view_margin_hr=0, expected_sha256=None)
    if q == 0:
        return dict(base, recipe="NOALIGN", aligner_policy="A-ID", rec=dict(case="N0"), stat=stat_off, geom_kd=dict(mode="G0"))
    if q == 1:
        return dict(base, recipe="T112DFR", aligner_policy="A-FR", donor=donor_d, rec=dict(case="N0"), stat=stat_off, geom_kd=dict(mode="G0"))
    if q == 2:
        return dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, rec=dict(case="N0"), stat=stat_off, geom_kd=dict(mode="G0"))
    if q == 3:
        return dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R1"), stat=stat_off, geom_kd=dict(mode="G0"))
    if q == 4:
        return dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat_off, geom_kd=dict(mode="G0"))
    if q == 5:
        return dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat("H"), geom_kd=dict(mode="G0"))
    if q == 6:
        return dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat("AD"), geom_kd=dict(mode="G0"))
    if q == 7:
        return dict(base, recipe="A1", aligner_policy="A-FT", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat("AD"), geom_kd=dict(mode="G0"))
    if q == 8:
        return dict(base, recipe="A1", aligner_policy="A-SC", aligner_view_margin_hr=0, teacher=teacher, rec=rec("R3"), stat=stat("AD"), geom_kd=dict(mode="G0"))
    # 이동량 covariance KD (계획 §13.2 Q19/Q21/Q22, §11) — A-FT + REC-R3 + STAT-OFF 위에서 G 모드만 바꾼다. 실행은 보류 큐 (PO10 N2/N3 · Q07/Q08 결과 뒤)
    gk = lambda mode, src: dict(mode=mode, covariance_source=src, outer_weight="calibrate", r_gkd=0.1, probes=dict(K=16, radius_hr=1.0, guard_margin=4),
                                geo=dict(h=0.05, sigma_min=0.05, tau_rel=1.0e-3), sigma_min=0.05, covhead_epochs=3)
    # 계획 §11.5: G1/G2/G3/G4 의 Π_T 는 G-GEO(구조비용 곡률) 가 원안 1차 출처, G-EQ(closure) 는 별도 행. 18 = 같은 trainable 정책의 G0 대조 (검토 지적)
    geom_cells = {18: ("G0", "none"), 19: ("G1", "geo_curvature"), 20: ("G2", "geo_curvature"), 21: ("G3", "geo_curvature"), 22: ("G4", "geo_curvature"), 23: ("G3", "eq_closure"), 24: ("G-STRUCT", "none"), 25: ("G5", "geo_curvature"), 26: ("G5", "eq_closure")}
    if q in geom_cells:
        mode, src = geom_cells[q]
        gkd = dict(mode="G0") if mode == "G0" else (gk(mode, src) if src != "none" else dict(gk(mode, "none"), covariance_source="none"))
        return dict(base, recipe="A1", aligner_policy="A-FT", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat_off, geom_kd=gkd)
    # TRI-A/B/C (addendum §8.2) — A/C 는 Q04(A-FR R3 OFF) 위, B 는 Q06(A-FR R3 GV-AD) 위. hard·parent 계수 불변, soft 만 변경
    q04 = dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat_off)
    q06 = dict(base, recipe="A1", aligner_policy="A-FR", donor=donor_d, teacher=teacher, rec=rec("R3"), stat=stat("AD"))
    tri_cells = {
        30: (q04, dict(a=dict(mode="sign"))), 31: (q04, dict(a=dict(mode="cosine"))), 32: (q04, dict(a=dict(mode="sign_cap"))), 33: (q04, dict(a=dict(mode="band_adv"))),
        34: (q04, dict(a=dict(mode="mass"))), 35: (q04, dict(a=dict(mode="shuffle"))),
        40: (q06, dict(b=dict(mode="sign"))), 41: (q06, dict(b=dict(mode="sign_cap"))), 42: (q06, dict(b=dict(mode="comp_adv"))), 43: (q06, dict(b=dict(mode="mass"))), 44: (q06, dict(b=dict(mode="shuffle"))),
        45: (dict(q06, stat=stat("WH")), None),                       # GV-WH 대조 (계획 Q13) — B 의 필수 기준
        50: (q04, dict(c=dict(mode="sens", h=0.05))), 51: (q04, dict(c=dict(mode="diag", h=0.05, covariance_source="eq_closure"))), 52: (q04, dict(c=dict(mode="diag", h=0.05, covariance_source="geo_curvature"))),
        54: (q04, dict(c=dict(mode="diag", h=0.05, covariance_source="eq_closure", sigma_control="rotate"))), 55: (q04, dict(c=dict(mode="diag", h=0.05, covariance_source="eq_closure", sigma_control="shuffle"))),
        56: (q04, dict(c=dict(mode="qiso", h=0.05))), 57: (q04, dict(c=dict(mode="qscalar", h=0.05, covariance_source="eq_closure"))), 58: (q04, dict(c=dict(mode="full", h=0.05, covariance_source="eq_closure"))),
        60: (q06, dict(a=dict(mode="sign"), b=dict(mode="sign"))), 61: (q04, dict(a=dict(mode="sign"), c=dict(mode="diag", h=0.05, covariance_source="eq_closure"))),
    }
    if q in tri_cells:
        basecfg, tri = tri_cells[q]
        d = dict(basecfg, geom_kd=dict(mode="G0", probes=dict(K=16, radius_hr=1.0, guard_margin=4), geo=dict(h=0.05, sigma_min=0.05, tau_rel=1.0e-3), sigma_min=0.05))   # C 의 Σ 출처 상수 (§11 과 동일)
        if tri is not None:
            d["tri"] = dict(tri, shuffle_seed=4321)
            if "c" in tri and tri["c"]["mode"] in ("full", "qiso", "qscalar"):
                d["tri"]["lambda_pilot"] = f"{pilot_run}/last"
        return d
    raise ValueError(q)


PURPOSE = {0: "Q00 새 폭(W112) 의 독립 baseline: aligner·sampler·KD 없음 (B0 와 같은 그래프)", 1: "Q01 W112 Teacher 확보: donor aligner frozen + 복원 supervised (T112_DONORFROZEN_REC), seed 2025",
           2: "Q02 A-FR + REC-N0: frozen aligner 의 GT-only 기준 (Teacher 와 seed 만 다른 대응)", 3: "Q03 A-FR + REC-R1: Teacher-error 로 hard 만 재가중", 4: "Q04 A-FR + REC-R3: adaptive hard/soft 기준",
           5: "Q05 A-FR + REC-R3 + GV-H: GT gradient-variance(5×5) 통계 추가", 6: "Q06 A-FR + REC-R3 + GV-AD: GT+Teacher adaptive 통계 (첫 주력 후보)",
           7: "Q07 A-FT + REC-R3 + GV-AD: Teacher aligner 를 초기값으로 fine-tune", 8: "Q08 A-SC + REC-R3 + GV-AD: aligner 독립 초기화 — Teacher 초기값의 제약 확인",
           18: "Q19-0 A-FT + R3 + OFF + G0: 같은 trainable 정책의 alignment-KD 없음 대조", 19: "Q19-1 A-FT + R3 + OFF + G1(GEO k0): Teacher 이동 중심 자체의 효과", 20: "Q19-2 A-FT + R3 + OFF + G2(GEO): sample 별 scalar 신뢰도", 21: "Q19-3 A-FT + R3 + OFF + G3(GEO): 방향별 full precision (구조비용 곡률)",
           22: "Q21-1 A-FT + R3 + OFF + G4(GEO): precision 대각만", 23: "Q21-2 A-FT + R3 + OFF + G-EQ(= G3, Σ from probe closure): 상대 반응 신뢰도", 24: "Q21-3 A-FT + R3 + OFF + G-STRUCT: GT 구조 텐서 가중",
           25: "Q22 A-FT + R3 + OFF + G5(GEO): Gaussian 분포 KD (cov head)", 26: "Q22-b A-FT + R3 + OFF + G5(EQ): 분포 KD, closure proxy",
           30: "TRI-A-SIGN: R3 soft 에 band 별 sign gate (주력 최소안)", 31: "TRI-A-COS: pixel cosine 감쇠 (scalar 대조)", 32: "TRI-A-CAP: sign·크기비 cap", 33: "TRI-A-BANDADV: w_K 를 band 별 (1−d_c)a_c 로",
           34: "TRI-A-MASS: A-SIGN 과 soft 총계수량만 맞춘 scalar 감쇠 (인과 대조)", 35: "TRI-A-SHUFFLE: mask 공간 순서 섞은 부정 대조",
           40: "TRI-B-SIGN: GV-AD 통계 soft 에 성분별 sign gate", 41: "TRI-B-CAP", 42: "TRI-B-COMPADV: 성분별 τ_V,j·a_j 로 soft 세분화", 43: "TRI-B-MASS", 44: "TRI-B-SHUFFLE", 45: "GV-WH: 통계 hard 재가중만 (B 의 기준, 계획 Q13)",
           50: "TRI-C-SENS: Teacher 출력의 correction 민감도 r = s/(s+|J|²) 로 rec soft 감쇠 (covariance 불필요)", 51: "TRI-C-DIAG(EQ proxy): r = s_C²/(s_C²+diag(JΣJᵀ)), Σ = probe closure", 52: "TRI-C-DIAG(GEO): Σ = A3 geometry 곡률",
           54: "TRI-C-DIAG(EQ) Σ rotate 대조", 55: "TRI-C-DIAG(EQ) Σ sample shuffle 대조", 56: "TRI-C-QISO: quadratic soft W=I", 57: "TRI-C-QSCALAR(EQ): tr W/D", 58: "TRI-C-FULL(EQ): 저랭크 Woodbury quadratic soft",
           60: "TRI-AB: A-SIGN + B-SIGN (GV-AD 위)", 61: "TRI-AC: A-SIGN + C-DIAG(EQ)"}


def dump_kdv(d, indent=2):
    import yaml
    return "\n".join(" " * indent + l for l in yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--teacher-seed", type=int, default=2025)
    ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--width", type=int, default=112); ap.add_argument("--depth", default="1,2,3")
    ap.add_argument("--donor", default=DONOR); ap.add_argument("--version", default="v01"); ap.add_argument("--teacher-id", default=None)
    ap.add_argument("--params", type=float, default=None); ap.add_argument("--queue", default=None); ap.add_argument("--only", default=None, help="예: 0,1,2")
    ap.add_argument("--geomkd", action="store_true", help="이동량 covariance KD 보류 큐(Q19–Q22 → 18..26) 를 만든다: config + config/queues/kdv_s2_geomkd.txt")
    ap.add_argument("--triabc", action="store_true", help="TRI-A/B/C 큐(30..61) 를 만든다: config + config/queues/kdv_s2_triabc.txt (P1 → 대조 → P2/P3 순)")
    a = ap.parse_args()
    depth = [int(x) for x in a.depth.split(",")]; arch = f"W{a.width}_D{''.join(map(str, depth))}"; prefix = f"S2W{a.width}D{''.join(map(str, depth))}"
    teacher_id = a.teacher_id or f"T{a.width}_{a.version}"
    if a.params is None:
        import yaml
        from main import import_class
        _c = yaml.safe_load(open(os.path.join(ROOT, "config", f"PA_A1_REC_W96_D124_9CH_S{a.seed}.yaml")))
        _m = import_class(_c["model"])(**dict(_c["model_args"], hidden_size=a.width, depth=depth)); a.params = round(sum(p.numel() for p in _m.parameters()) / 1e6, 4)
    init_dir = f"work_dir/_kdv_init_{arch.lower()}"
    names = {}
    # 이름은 spec 에서 결정 — Teacher run 이름을 먼저 알아야 Student 가 참조한다
    for q in list(range(9)) + list(range(18, 27)) + [30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61]:
        seed = a.teacher_seed if q == 1 else a.seed
        d = build_kdv(q, seed, "TBD", "TBD", a.donor, teacher_id); sp = resolve(d); names[q] = run_name(sp, seed, a.version, prefix)
    TRI_ORDER = [30, 40, 50, 51, 34, 35, 45, 43, 44, 41, 42, 31, 32, 33, 52, 54, 55, 60, 61, 56, 57, 58]
    made = []; only = [int(x) for x in a.only.split(",")] if a.only else (list(range(18, 27)) if a.geomkd else (TRI_ORDER if a.triabc else list(range(9))))
    for q in only:
        seed = a.teacher_seed if q == 1 else a.seed
        d = build_kdv(q, seed, names[1], names[2], a.donor, teacher_id); d["init_dir"] = init_dir; d["version"] = a.version
        sp = resolve(d); tag = names[q]
        tpl = open(os.path.join(ROOT, "config", f"PA_A1_REC_W96_D124_9CH_S{seed}.yaml")).read()
        t = re.sub(r"^(#.*\n)+", "", tpl)
        head = (f"# {tag} — {PURPOSE[q]}. 생성: tools/gen_kdv_configs.py (템플릿 PA_A1_REC_W96_D124_9CH_S{seed}). 손으로 고치지 말 것.\n"
                f"# 계획: research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/ · 구현 train_kdv.py / kdv/ · 노트 research_log/2026-09-10_s2-w112-kdv-implementation.md\n"
                f"# 세팅: {describe(sp)}\n"
                f"# 골격 {arch}(hidden {a.width}, depth {depth}) · 9ch·mars ms·γβ 제거 · AdamW 1e-4/wd0.01 cosine warmup100 · batch 48 · {a.updates} updates · seed {seed} · init {init_dir}\n"
                f"# donor aligner = {a.donor} (s1 PA_A1 seed 2025 best_hqnr 의 aligner.*, 전체 view, strict load) · Teacher = {names[1]}/best_hqnr (id {teacher_id})\n"
                f"# best_hqnr = best_raw(raw_original HQNR → fSCC → later step) · best_aligned · best_rr_val(검증셋 plain ERGAS) · last. 시트 = raw_original(HQNR↑) + V64(HQNR(V64)↑)\n")
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = re.sub(r"^trainer: pa\npa:\n(  .*\n)+", "", t, flags=re.M)
        t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                      "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n" + dump_kdv(d))
        t = re.sub(r"^num_iter: \d+", f"num_iter: {a.updates}", t, flags=re.M)
        t = re.sub(r"^  hidden_size: \d+", f"  hidden_size: {a.width}", t, flags=re.M)
        t = re.sub(r"^  depth: \[.*?\]", f"  depth: {depth}", t, flags=re.M)
        t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {a.params}", t)
        t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
        assert f"hidden_size: {a.width}" in t and f"depth: {depth}" in t and "trainer: kdv" in t and "trainer: pa" not in t and f"num_iter: {a.updates}" in t and f"seed: {seed}" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t); made.append(tag)
    q = a.queue or os.path.join(ROOT, "config", "queues", f"kdv_s2{'_geomkd' if a.geomkd else ('_triabc' if a.triabc else '')}{'' if a.version == 'v01' else '_' + a.version}.txt")
    with open(q, "w") as f:
        if a.triabc:
            f.write(f"# s2 TRI-A/B/C 큐 (addendum §8.2) · {arch} · seed {a.seed} · {a.updates} updates · 기동은 kdv_s2 본 큐(Q01 Teacher·Q02 pilot·Q04/Q06 base) 뒤\n"
                    "# 순서: P1(A-SIGN, B-SIGN, C-SENS, C-DIAG EQ) → 인과 대조(A-MASS/SHUFFLE, GV-WH, B-MASS/SHUFFLE) → P2(B-CAP/COMPADV, A-COS/CAP/BANDADV, C-DIAG GEO, Σ rotate/shuffle, AB, AC) → P3(QISO/QSCALAR/FULL)\n"
                    "# 주의: 현재 donor(A1) 는 무반응이라 C-DIAG(EQ) 의 Σ 는 ≈0.31·I(등방) — C-SENS 와 정보가 거의 같다. 방향 정보의 가치는 반응하는 aligner 뒤에 검증 (구현 노트 §10)\n")
        elif a.geomkd:
            f.write(f"# s2 KDV 이동량 covariance KD 큐 (plan §13.2 Q19/Q21/Q22 · §11) · {arch} · A-FT + R3 + OFF 위에서 G0(대조)/G1GEO/G2GEO/G3GEO/G4GEO/G3EQ(=G-EQ)/GSTRUCT/G5GEO/G5EQ · seed {a.seed} · {a.updates} updates\n"
                    "# **실행 보류**: Q01 Teacher 가 있어야 하고, PO10 N2/N3(반응하는 aligner) 와 Q07/Q08(trainable aligner) 결과를 본 뒤 기동한다 (구현 노트 §9). 현재 donor(A1) 는 입력 무반응이라 G-EQ precision 이 작다\n")
        else:
            f.write(f"# s2 KDV 우선 queue Q00–Q08 (plan §13.2) · {arch} · Student seed {a.seed} · Teacher seed {a.teacher_seed} · {a.updates} updates\n"
                    "# 순서 의존: Q01(Teacher) 가 Q03–Q08 의 KD target, Q02(A-FR N0) 가 Q05–Q08 의 λ_V pilot(<run>/last) — 체인이 순서대로 돈다\n")
        for i, tag in enumerate(made):
            f.write(f"{tag}\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))


if __name__ == "__main__":
    main()
