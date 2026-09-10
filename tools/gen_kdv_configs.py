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
    raise ValueError(q)


PURPOSE = {0: "Q00 새 폭(W112) 의 독립 baseline: aligner·sampler·KD 없음 (B0 와 같은 그래프)", 1: "Q01 W112 Teacher 확보: donor aligner frozen + 복원 supervised (T112_DONORFROZEN_REC), seed 2025",
           2: "Q02 A-FR + REC-N0: frozen aligner 의 GT-only 기준 (Teacher 와 seed 만 다른 대응)", 3: "Q03 A-FR + REC-R1: Teacher-error 로 hard 만 재가중", 4: "Q04 A-FR + REC-R3: adaptive hard/soft 기준",
           5: "Q05 A-FR + REC-R3 + GV-H: GT gradient-variance(5×5) 통계 추가", 6: "Q06 A-FR + REC-R3 + GV-AD: GT+Teacher adaptive 통계 (첫 주력 후보)",
           7: "Q07 A-FT + REC-R3 + GV-AD: Teacher aligner 를 초기값으로 fine-tune", 8: "Q08 A-SC + REC-R3 + GV-AD: aligner 독립 초기화 — Teacher 초기값의 제약 확인"}


def dump_kdv(d, indent=2):
    import yaml
    return "\n".join(" " * indent + l for l in yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--teacher-seed", type=int, default=2025)
    ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--width", type=int, default=112); ap.add_argument("--depth", default="1,2,3")
    ap.add_argument("--donor", default=DONOR); ap.add_argument("--version", default="v01"); ap.add_argument("--teacher-id", default=None)
    ap.add_argument("--params", type=float, default=None); ap.add_argument("--queue", default=None); ap.add_argument("--only", default=None, help="예: 0,1,2")
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
    for q in range(9):
        seed = a.teacher_seed if q == 1 else a.seed
        d = build_kdv(q, seed, "TBD", "TBD", a.donor, teacher_id); sp = resolve(d); names[q] = run_name(sp, seed, a.version, prefix)
    made = []; only = [int(x) for x in a.only.split(",")] if a.only else list(range(9))
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
    q = a.queue or os.path.join(ROOT, "config", "queues", f"kdv_s2{'' if a.version == 'v01' else '_' + a.version}.txt")
    with open(q, "w") as f:
        f.write(f"# s2 KDV 우선 queue Q00–Q08 (plan §13.2) · {arch} · Student seed {a.seed} · Teacher seed {a.teacher_seed} · {a.updates} updates\n"
                "# 순서 의존: Q01(Teacher) 가 Q03–Q08 의 KD target, Q02(A-FR N0) 가 Q05–Q08 의 λ_V pilot(<run>/last) — 체인이 순서대로 돈다\n")
        for i, tag in enumerate(made):
            f.write(f"{tag}\n")
    print("\n".join(made)); print("queue:", os.path.relpath(q, ROOT))


if __name__ == "__main__":
    main()
