#!/usr/bin/env python
"""SMEC12 준비 학습(B01–B04) config 생성 — research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md §2 (센서별 P0 · DON-N2 · L000 · L1E4 · L1E4-REP).

    python tools/gen_smec12_bootstrap.py --sensors qb,gf2            # config/B0*_<SENSOR>_*.yaml + config/queues/smec12_<server>.txt (lane 교차: 두 센서의 primary L1E4 를 먼저)
    python tools/gen_smec12_bootstrap.py --plan                       # 센서별 run id · 상태(complete/pending) · 예상 시간

recipe 는 검증된 WV3 성공 run 의 resolved config 를 template 으로 쓰고(§2.7 우선순위) 센서 항목만 바꾼다:
  B01 P0     ← NF16 P0 (kdv, A-ID/NOALIGN, I-A, plain L1; reference_donor 없음)              run  B01_<S>_P0_W112_D123_S2025_BOOTSCRATCH
  B02 DON-N2 ← PO10 N2_OFFSG R200 (po, native/corrupt 1:1, SG offset .01 ramp 5K, b=2)      run  B02_<S>_DONN2_W112_D123_R200_S2025_BOOTSCRATCH
  B03 L000   ← PALS24 L000 (kdv, A-FT, I-NATIVE-TRANSFER, offset 0, donor = B02/last)         run  B03_<S>_L000_W112_D123_S2025_DONB02
  B03 L1E4   ← PALS24 L1E4 (kdv, A-FT, I-AEQ, λ_off 1e-4 홀수 update, b=2, donor = B02/last)  run  B03_<S>_L1E4_W112_D123_S2025_DONB02
  B04 L1E4-REP ← 같은 donor, seed 1234                                                          run  B04_<S>_L1E4_W112_D123_S1234_DONB02
센서 항목: dataroot(train/valid/RR/FR mat20) · num_bands/out_channels · max_pixel(GF2 1023) · expect_params_m(4-band W112·D123 backbone 2.6508 M) · eval_epoch(≈1000 update 격자) ·
init_dir(골격·센서별 분리: work_dir/_kdv_init_w112_d123_<s>, _pa_init_w112_d123_<s>; 같은 센서 P0/L000/L1E4 는 같은 seed 의 같은 U 초기값을 공유 §2.4) · save_iter 25000(exact25K 전체 state §2.6).
b=2·λ=1e-4·.01 ramp 는 WV3 recipe 의 공통 이식이지 센서 최적값이 아니다(§2.1)."""
import argparse, json, os, re, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
PLAN = "research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md"; NOTE = "research_log/2026-09-15_smec12-implementation.md"
CAMPAIGN_ID = "SMEC12_BOOT_v1"; LINEAGE = "BOOTSCRATCH"; PRIMARY_SEED = 2025; CONFIRM_SEED = 1234; UPDATES = 50000; BATCH = 48
LEDGER = "work_dir/_smec12_budget/ledger.json"; TOTAL_HOURS = 40.0; RESERVE_HOURS = 2.0; MARGIN = 1.1; RUN_RESERVED_HOURS = 2.6
D = "data/PanCollection"
SENSORS = {"qb": dict(S="QB", bands=4, max_pixel=2047.0, train=f"{D}/QB/train_qb_msfix.h5", valid=f"{D}/QB/valid_qb_msfix.h5", rr=f"{D}/QB/reduced_examples_h5/test_qb_multiExm1.h5", fr=f"{D}/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5",
                      backbone_params=2.6508, n_train=17139, data_note="train/valid ms 복구본(KNOWN_ISSUES F-3), FR lpan 복구(F-1); mat20 논문 세트"),
           "gf2": dict(S="GF2", bands=4, max_pixel=1023.0, train=f"{D}/GF2/train_gf2.h5", valid=f"{D}/GF2/valid_gf2.h5", rr=f"{D}/GF2/reduced_examples_h5/test_gf2_multiExm1.h5", fr=f"{D}/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5",
                       backbone_params=2.6508, n_train=19809, data_note="배포본 정상(F-1 해당 없음); max_pixel 1023; mat20 논문 세트"),
           "wv3": dict(S="WV3", bands=8, max_pixel=2047.0, train=f"{D}/WV3/train_wv3.h5", valid=f"{D}/WV3/valid_wv3.h5", rr=f"{D}/WV3/reduced_examples_h5/test_wv3_multiExm1.h5", fr=f"{D}/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5",
                       backbone_params=2.6589, n_train=9714, data_note="기존 자산(NF16/PO10/PALS24) 우선 — 준비 학습 대상 아님")}
TEMPLATES = dict(P0="config/NF16_P0_W112_D123_WV3_S1234_N2LAST_v1.yaml", DONN2="config/PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT.yaml",
                 L000="config/PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1.yaml", L1E4="config/PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1.yaml", L1E4REP="config/PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1.yaml")
ROLES = ["P0", "DONN2", "L000", "L1E4", "L1E4REP"]
ROLE_DESC = {"P0": "aligner 없음(A-ID/NOALIGN), 새 U, plain L1 50K — 무정합 baseline", "DONN2": "N2 형 변위 사전학습(po: native/corrupt 1:1, SG offset .01 ramp 5K, b=2) 50K — exact50K aligner 가 donor",
             "L000": "DON-N2 의 A 복사 + 새 U(P0 와 같은 초기값), native recon 만(offset 0) 50K", "L1E4": "같은 donor·같은 U 초기값, native recon + 홀수 update 1e-4·Off(b=2) 50K — 주 분석 모델", "L1E4REP": "같은 donor, U seed 1234 — 학습 반복 확인"}


def run_id(sensor, role):
    S = SENSORS[sensor]["S"]
    return {"P0": f"B01_{S}_P0_W112_D123_S{PRIMARY_SEED}_{LINEAGE}", "DONN2": f"B02_{S}_DONN2_W112_D123_R200_S{PRIMARY_SEED}_{LINEAGE}", "L000": f"B03_{S}_L000_W112_D123_S{PRIMARY_SEED}_DONB02",
            "L1E4": f"B03_{S}_L1E4_W112_D123_S{PRIMARY_SEED}_DONB02", "L1E4REP": f"B04_{S}_L1E4_W112_D123_S{CONFIRM_SEED}_DONB02"}[role]


def seed_of(role):
    return CONFIRM_SEED if role == "L1E4REP" else PRIMARY_SEED


def eval_epoch_for(sensor):
    """후보 격자 간격을 WV3 의 1010 update(5 epoch × 202) 에 가깝게(≤ ~1100 update): epoch = n_train/48 update."""
    per = SENSORS[sensor]["n_train"] / BATCH; return max(1, int(1100 // per))


def grid_id(sensor):
    per = int(SENSORS[sensor]["n_train"] // BATCH); return f"GRID{eval_epoch_for(sensor) * per}_50K_{sensor}"


def complete(run):
    r = os.path.join(ROOT, "work_dir", run, "results"); return os.path.exists(os.path.join(r, "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(r, "full_best_hqnr.mat"))


def _sensor_paths(cfg, s):
    X = SENSORS[s]
    for key in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
        if key in cfg and "dataroot" in cfg[key]:
            base = os.path.basename(cfg[key]["dataroot"]); role = "train" if base.startswith("train_") else "valid" if base.startswith("valid_") else "rr" if "reduced" in cfg[key]["dataroot"] else "fr"
            cfg[key]["dataroot"] = os.path.join(ROOT, X[role])
    cfg["num_bands"] = X["bands"]; cfg["max_pixel"] = X["max_pixel"]; cfg["model_args"]["out_channels"] = X["bands"]; cfg["expect_params_m"] = X["backbone_params"]
    cfg["eval_epoch"] = eval_epoch_for(s); cfg["num_iter"] = UPDATES; cfg["save_iter"] = 25000
    return cfg


def budget_block(sensor, role):
    runs = [run_id(sensor, r) for r in ROLES]; me = run_id(sensor, role)
    return dict(ledger=LEDGER, total_gpu_hours=TOTAL_HOURS, reserve_hours=RESERVE_HOURS, margin=MARGIN, required=True, projected_hours=None, projected_map={r: RUN_RESERVED_HOURS for r in runs},
                remaining_mandatory=[r for r in runs if r != me])


def build(sensor, role):
    X = SENSORS[sensor]; cfg = yaml.safe_load(open(os.path.join(ROOT, TEMPLATES[role]))); me = run_id(sensor, role); seed = seed_of(role)
    cfg = _sensor_paths(cfg, sensor); cfg["seed"] = seed; cfg["work_dir"] = os.path.join(ROOT, "work_dir", me)
    if role == "DONN2":
        po = cfg["po"]; po["init_dir"] = f"work_dir/_pa_init_w112_d123_{sensor}"; po["protocol_id"] = f"pan_offset_rel_smec12_{sensor}_r200"
        prov = po.get("radius_provenance") or {}; prov.update(dataset=X["S"], calibration_quality="common_recipe_transfer_from_wv3 (b=2 는 WV3 FR 통계에서 온 공통 recipe; 이 센서의 최적값 아님 — SMEC12 §2.1/§2.5)", transferred_from="WV3 R200_FRSTAT"); po["radius_provenance"] = prov
        po["smec12"] = dict(campaign_id=CAMPAIGN_ID, role=role, lineage=LINEAGE, sensor=X["S"], donor_of=[run_id(sensor, r) for r in ("L000", "L1E4", "L1E4REP")], note="donor = exact50K(last) 의 aligner; best_raw 로 대체하지 않는다")
        return cfg
    k = cfg["kdv"]; k["campaign_id"] = CAMPAIGN_ID; k["case_id"] = role; k["version"] = "v1"; k["check_run_name"] = False
    k["experiment_branch_id"] = f"SMEC12_BOOT_{X['S']}_{LINEAGE}"; k["init_dir"] = f"work_dir/_kdv_init_w112_d123_{sensor}"; k["candidate_grid_id"] = grid_id(sensor)
    k["select"] = dict(primary="best_hqnr", secondary=["best_rr_val", "last"], retain_all_candidates=True); k["diag"] = dict(fixed_batch=True)
    k["expect_arch"] = dict(width=112, depth=[1, 2, 3], noalign=(role == "P0")); k["budget"] = budget_block(sensor, role)
    for key in ("plan_protocol_id", "document_revision"):
        k.pop(key, None)
    if role == "P0":
        k["eval"] = {}                                                       # NF16 P0 의 reference_donor(고정 donor 참조 view) 는 아직 donor 가 없어 뺀다
        k.pop("aligner_lr", None)
    else:
        k["donor"] = dict(source=f"work_dir/{run_id(sensor, 'DONN2')}/last", view_margin_hr=4, expected_sha256=None, expected_step=UPDATES)      # sha 는 B02 가 끝난 뒤(A00 이 기록); step 은 exact50K
        k["eval"] = dict(fixed_reference_from_donor=True)
    return cfg


def header(sensor, role, cfg):
    X = SENSORS[sensor]; me = run_id(sensor, role)
    return (f"# {me} — SMEC12 준비 학습 {role}: {ROLE_DESC[role]}. 생성: tools/gen_smec12_bootstrap.py (template {TEMPLATES[role]}; 센서 항목만 치환). 손으로 고치지 말 것.\n"
            f"# 계획 {PLAN} §2 · 노트 {NOTE} · 센서 {X['S']} ({X['bands']} band, max_pixel {X['max_pixel']}; {X['data_note']}) · seed {seed_of(role)} · lineage {LINEAGE}\n"
            f"# 골격 W112·D123 (backbone {X['backbone_params']} M) · 50K · batch 48 · AdamW 1e-4(U)/1e-5(A; kdv) wd 0.01 cosine warmup100 · eval_epoch {cfg['eval_epoch']} (= {grid_id(sensor)}) · save_iter 25000 (exact25K/50K 전체 state)\n"
            + (f"# donor = {run_id(sensor, 'DONN2')}/last (exact50K; sha 는 학습 뒤 A00 이 기록) · init {cfg['kdv']['init_dir']}/init_unet_seed{seed_of(role)}.pt (같은 센서 P0/L000/L1E4 공유)\n" if role in ("L000", "L1E4", "L1E4REP") else "")
            + (f"# init {cfg['kdv']['init_dir']}/init_unet_seed{seed_of(role)}.pt — 같은 센서 L000/L1E4 가 같은 파일을 쓴다 (§2.4)\n" if role == "P0" else "")
            + (f"# b=2 (HR px) · λ_off .01 ramp 5K · native/corrupt 1:1 · aligner head zero-init — WV3 N2 R200 recipe 의 공통 이식(센서 최적값 아님)\n" if role == "DONN2" else "")
            + f"# 예산 {LEDGER} {TOTAL_HOURS} h(reserve {RESERVE_HOURS}) · 준비 학습은 required(초과해도 경고만; 시간 때문에 취소하지 않는다 §2.3)\n")


def queue_order(sensors):
    """lane 교차: 두 센서의 P0 → DONN2 → L000 → L1E4 → REP 순으로 번갈아 (각 센서의 primary L1E4 가 가능한 한 일찍 완성되게; §19.3 '아직 분석하지 않은 sensor 우선')."""
    return [run_id(s, r) for r in ROLES for s in sensors]


def generate(sensors, out_dir):
    made = []
    for s in sensors:
        for r in ROLES:
            cfg = build(s, r); me = run_id(s, r); os.makedirs(out_dir, exist_ok=True)
            open(os.path.join(out_dir, me + ".yaml"), "w").write(header(s, r, cfg) + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=None, width=200)); made.append(me)
    return made


def plan(sensors):
    for s in sensors:
        X = SENSORS[s]; print(f"[{X['S']}] {X['bands']} band · max_pixel {X['max_pixel']} · train {X['n_train']} patch · eval_epoch {eval_epoch_for(s)} ({grid_id(s)}) · {X['data_note']}")
        for r in ROLES:
            me = run_id(s, r); st = "complete" if complete(me) else ("running/partial" if os.path.exists(os.path.join(ROOT, "work_dir", me, "meta", "started_at.txt")) else "pending")
            print(f"   {r:8s} {me:52s} {st:16s} config {'yes' if os.path.exists(os.path.join(ROOT, 'config', me + '.yaml')) else 'no'}")
    print(f"  예상: run 당 {RUN_RESERVED_HOURS} h (s2 W112 50K 실측 ≈2.3 h 대용) × {len(sensors) * len(ROLES)} run = {RUN_RESERVED_HOURS * len(sensors) * len(ROLES):.1f} h (순차)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--sensors", default="qb,gf2"); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config")); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--server", default=None)
    a = ap.parse_args(); sensors = [s.strip().lower() for s in a.sensors.split(",")]
    for s in sensors:
        assert s in SENSORS and s != "wv3", f"준비 학습 대상 센서는 qb/gf2 (wv3 는 기존 자산): {s}"
    if a.plan:
        return plan(sensors)
    made = generate(sensors, a.out_dir)
    if a.out_dir == os.path.join(ROOT, "config"):
        srv = a.server or open(os.path.join(ROOT, "gspread", "server.txt")).read().strip(); q = os.path.join(ROOT, "config", "queues", f"smec12_{srv}.txt")
        with open(q, "w") as f:
            f.write(f"# SMEC12 준비 학습 (계획 {PLAN} §2; 노트 {NOTE}) — {srv}: 센서 {', '.join(SENSORS[s]['S'] for s in sensors)} 의 P0 → DON-N2 → L000 → L1E4 → L1E4-REP 를 lane 교차로.\n"
                    f"# 각 50K; 예약 {RUN_RESERVED_HOURS} h/run. donor(B02/last) 가 있어야 B03/B04 smoke 가 통과한다 — 순서가 의존성이다. 기존 PAKD50 chain 이 끝난 뒤 기동 (tools/smec12_prepare.sh).\n" + "\n".join(queue_order(sensors)) + "\n")
        print("queue:", os.path.relpath(q, ROOT))
    print("made:", " ".join(made))


if __name__ == "__main__":
    main()
