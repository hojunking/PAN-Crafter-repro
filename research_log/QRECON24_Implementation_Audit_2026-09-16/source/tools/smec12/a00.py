"""A00 — 자산·데이터·source audit (§6). manifests/{asset_registry,data_registry,bootstrap_queue,probe_manifest}.json · coverage.csv · reproduction.csv · discrepancies.md · unit 검사.
없는 모델은 B queue(bootstrap_queue.json) 에 넣는다 — 데이터 부재만이 실제 blocker (§2.3)."""
import json, os, time
import numpy as np, torch
from tools.smec12 import common as C
from tools import gen_smec12_bootstrap as B


def unit_checks():
    out = {}
    P = torch.zeros(1, 1, 33, 33, dtype=torch.float64); P[0, 0, 16, 16] = 1; w = C.E.warp_pan(P, torch.tensor([[0.0, 1.0]], dtype=torch.float64)); yx = np.unravel_index(int(w.argmax()), (33, 33))
    out["warp_sign"] = dict(ok=yx == (16, 15) and float((C.E.warp_pan(P, torch.zeros(1, 2, dtype=torch.float64)) - P).abs().max()) == 0.0, impulse=list(map(int, yx)), convention="(dy,dx); W(P,c)[y,x] = P[y+c_y, x+c_x]; grid x/y 내부")
    av = C.available("wv3", roles=("L1E4",), tags=("best_hqnr",))
    if av:
        L = C.load_asset(av[0][:3], av[0][3]); ids = C.make_panels("wv3")["ids"]["CAL"][:32]; gt, ms, lpan, pan = C.load_patches("wv3", ids)
        r1 = C.measure_offset_bank(L, pan, ms, "A"); r2 = C.measure_offset_bank(L, pan, ms, "A"); y1 = C.reconstruct_at(L, pan, ms, lpan); y2 = C.reconstruct_at(L, pan, ms, lpan)
        out["identity_probe"] = dict(ok=float(r1["identity"].max()) < 1e-4, max_abs=float(r1["identity"].max()))
        out["noise_floor"] = dict(q_repeat_max_abs=float((r1["q"] - r2["q"]).abs().max()), c0_repeat_max_abs=float((r1["c0"] - r2["c0"]).abs().max()), e_repeat_max_abs=float((y1 - y2).abs().max()), note="같은 입력 반복 (q_unresolved 의 10× 기준)")
        st = torch.get_rng_state(); C.measure_offset_bank(L, pan[:4], ms[:4], "B"); out["rng_isolation"] = dict(ok=torch.equal(st, torch.get_rng_state()))
        yz = C.reconstruct_at(L, pan, ms, lpan, torch.zeros(len(ids), 2)); out["zero_correction_path"] = dict(ok=float((yz - y1).abs().mean()) > 0, note="c=0 도 같은 sampler(delta_override) — zero shortcut 없음", mean_abs_diff=float((yz - y1).abs().mean()))
    return out


def reproduction():
    """기존 primary(WV3 L1E4 S2025 best_raw) 의 공식 metric 재현: run 의 fr_mat20.json vs EQREC4 s1 보고값·PAKD50 T0 자산 meta."""
    rows = []
    for key, e in C.asset_registry().items():
        if e["status"] != "available":
            continue
        rd = os.path.join(C.ROOT, "work_dir", e["run"]); fr = C.load_json(os.path.join(rd, "results", "fr_mat20.json"), {}); sel = e.get("selected") or {}
        rows.append(dict(asset=key, run=e["run"], selected_step=sel.get("step"), selected_hqnr=sel.get("hqnr"), fr_mat20_hqnr=fr.get("hqnr"), fr_mat20_d_lambda=fr.get("d_lambda"), fr_mat20_d_s=fr.get("d_s"), agree=(abs(float(sel.get("hqnr") or 0) - float(fr.get("hqnr") or 0)) < 1e-6) if sel.get("hqnr") is not None and fr.get("hqnr") is not None else None))
    t0 = C.load_json(os.path.join(C.ROOT, "assets", "pakd50", "T0_run", "best_hqnr_meta.json"), {})
    if t0:
        rows.append(dict(asset="PAKD50 T0 asset (= WV3|L1E4|S2025 best_raw)", run="assets/pakd50/T0_run", selected_step=t0.get("step"), selected_hqnr=t0.get("hqnr"), note="EQREC4 s1 보고 0.956976 · step 24240"))
    return rows


def main(profile=False):
    with C.Stage("A00", "asset/data/source audit"):
        C.ensure_dirs(); reg = C.asset_registry(write=True); dr = C.data_registry(write=True); C.probe_manifest()
        for s in ("wv3", "qb", "gf2", "wv2"):
            if s == "wv2" or dr[C.SENSORS[s]["S"]]["data_ready"]:
                C.make_panels(s)
        queue = [dict(asset=k, run=v["run"], sensor=v["sensor"], role=v["role"], seed=v["seed"], status=v["status"], config=f"config/{v['run']}.yaml", config_exists=os.path.exists(os.path.join(C.ROOT, "config", v["run"] + ".yaml"))) for k, v in reg.items() if v["status"] != "available"]
        C.dump_json(C.p_("manifests", "bootstrap_queue.json"), dict(policy="on_missing_checkpoint: train_dependency (§2.3); 큐 = config/queues/smec12_<server>.txt (lane 교차), 기동 tools/smec12_prepare.sh", pending=queue, n_pending=len(queue)))
        cov = []
        for s, X in C.SENSORS.items():
            models = C.available(s, roles=("L1E4",)) if s != "wv2" else C.available("wv3", roles=("L1E4",))
            cov.append(dict(sensor=X["S"], role=X["role"], data_ready=bool(dr[X["S"]]["data_ready"]), models_ready=bool(models), n_models_available=len([k for k, v in reg.items() if k.startswith(X["S"] + "|") and v["status"] == "available"]) if s != "wv2" else "uses WV3 models",
                            atlas_done=False, intervention_done=False, confirm_done=False, training_dependency=[k for k, v in reg.items() if k.startswith(X["S"] + "|") and v["status"] != "available"]))
        C.write_csv(C.p_("manifests", "coverage.csv"), cov); C.write_csv(C.p_("reproduction.csv"), reproduction())
        uc = unit_checks(); C.dump_json(C.p_("manifests", "a00_unit_checks.json"), uc)
        disc = ["# discrepancies (A00)", "", "- R1(s1)/R2(s3) 의 H3b 판정 차이(Supported / Insufficient) 는 통합하지 않는다 — C50/I23 에서 재검사 (계획 §23).", "- source id: PanCollection 에 scene/strip id 가 없다 → 32 연속 index block proxy (source_group_unknown); CI 는 descriptive 로 낮춘다 (§3.3·§4.5).",
                "- WV3 자산은 full-train 모델(seen_in_pretraining) — FIT70 source-holdout cohort 는 이번 release 에 없다 (§3.3 구분 표시).", "- QB/GF2 모델은 준비 학습 대기(bootstrap_queue.json) — 완료 전까지 해당 센서 stage 는 pending_dependency.",
                f"- host: {C.host_info()}", f"- unit checks: {json.dumps({k: v.get('ok') for k, v in uc.items()}, ensure_ascii=False)}"]
        open(C.p_("analysis", "discrepancies.md"), "w").write("\n".join(disc) + "\n")
        print("[a00] assets available:", sorted(k for k, v in reg.items() if v["status"] == "available"), "| pending:", len(queue), "| unit:", {k: v.get("ok") for k, v in uc.items()}, flush=True)
        return dict(implementation_invalid=[k for k, v in uc.items() if v.get("ok") is False])
