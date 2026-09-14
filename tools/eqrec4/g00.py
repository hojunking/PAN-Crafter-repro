"""G00 — 자산·provenance·metric·support·graph gate (계획 §6). 실패는 implementation_invalid / provenance_limited 로 나눈다."""
import json, os, subprocess, time
import numpy as np, torch, torch.nn.functional as F, yaml
from tools.eqrec4 import common as C
from tools.eqrec4.common import ROOT, CAMP, DEV


def _git():
    g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(sha=g("git rev-parse HEAD"), dirty=bool(g("git status --porcelain -- ':!results_log' ':!work_dir'")))


@torch.no_grad()
def reproduce_primary(L, fd):
    """Primary Teacher best_raw 의 raw-original HQNR 을 같은 evaluator 로 재현 (기록값과 대조) + RR L1 새로 계산 (ERGAS 를 L1 로 부르지 않는다)."""
    from pa.evalviews import scene_views, evaluator_hash
    from tools.eval_fr_paperset import sensor_of, load_dlpan
    sensor = sensor_of(fd["cfg"]); wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); mp = fd["mp"]; hq = []
    for i in range(len(fd["fr"])):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["fr"][i]); o = L.m(pan, ms, lpan); d = o["delta"][0].double().cpu().numpy()
        sr = ((o["y"][0].clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp).astype(np.float64).transpose(1, 2, 0); p = fd["pan_raw"][i]
        pa_eval = C.warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d)[None])[0, 0].numpy()
        v, ok, _ = scene_views(sr, fd["lms_raw"][i].transpose(1, 2, 0), p, pa_eval, sensor, wald, d, 4, mp); hq.append(v["raw_original"]["hqnr"])
    meta = C.load_json(L.path + "_meta.json", {}); rec = meta.get("hqnr"); got = float(np.mean(hq))
    rr_l1 = []
    for i in range(len(fd["rr"])):
        gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(DEV) for t in fd["rr"][i]); y = L.m(pan, ms, lpan)["y"]; rr_l1.append(float((y - gt).abs().mean()))
    return dict(model_key=L.key, recorded_best_raw_hqnr=rec, reproduced_raw_original_hqnr=got, abs_diff=(abs(got - rec) if rec is not None else None), pass_=(rec is not None and abs(got - rec) <= 1e-4),
                per_scene=hq, evaluator_hash=evaluator_hash(), rr_l1_normalized_mean=float(np.mean(rr_l1)), rr_l1_dn_mean=float(np.mean(rr_l1) * mp / 2), note="RR L1 computed fresh (normalized [-1,1] units and DN); not the sheet ERGAS")


@torch.no_grad()
def unit_checks(L):
    """§6-4/5: W 부호 · zero identity · FP32 · coordinate ramp · z-score 는 aligner 내부(재구성 입력은 z-score 아님)."""
    out = {}
    P = torch.zeros(1, 1, 33, 33, dtype=torch.float64); P[0, 0, 16, 16] = 1.0
    w = C.warp_pan(P, torch.tensor([[0.0, 1.0]], dtype=torch.float64)); yx = np.unravel_index(int(w.argmax()), w.shape[-2:])
    out["warp_sign"] = dict(impulse_at=[16, 16], applied_c_dy_dx=[0.0, 1.0], impulse_moved_to=[int(yx[0]), int(yx[1])], expected=[16, 15], ok=bool(yx == (16, 15)), rule="W(P,c)[y,x] = P[y+c_y, x+c_x]")
    Pr = torch.rand(1, 1, 64, 64, dtype=torch.float64); out["zero_identity_max_abs"] = float((C.warp_pan(Pr, torch.zeros(1, 2, dtype=torch.float64)) - Pr).abs().max())
    ramp = torch.arange(64, dtype=torch.float64).view(1, 1, 1, 64).expand(1, 1, 64, 64).contiguous(); rs = C.warp_pan(ramp, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    out["coordinate_ramp_dx1_mean_shift"] = float((rs - ramp)[..., 8:-8, 8:-8].mean()); out["fp32_warp_dtype"] = str(C.warp_pan(torch.rand(1, 1, 8, 8), torch.zeros(1, 2)).dtype)
    if L.has_aligner:
        pan = torch.rand(2, 1, 64, 64, device=DEV) * 2 - 1; ms = torch.rand(2, 8, 16, 16, device=DEV) * 2 - 1; mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
        c1 = C.predict_c(L.m.aligner, pan, mb, L.mg); c2 = C.predict_c(L.m.aligner, 1.7 * pan + 0.3, mb, L.mg)
        out["aligner_zscore_invariance_max_abs"] = float((c1 - c2).abs().max()); out["aligner_view"] = dict(margin=L.mg, view_hw=[64 - 2 * L.mg] * 2)
        lpan = torch.rand(2, 1, 16, 16, device=DEV) * 2 - 1; y1 = L.m(pan, ms, lpan, delta_override=c1)["y"]; y2 = L.m(1.7 * pan + 0.3, ms, lpan, delta_override=c1)["y"]
        out["reconstruction_sees_raw_pan_not_zscore"] = bool(float((y1 - y2).abs().max()) > 1e-4)
        H = 64; out["bicubic_support_margin_check"] = dict(margin_roi=16, max_two_stage_shift_ok=C.two_stage_ok(H, H, (2.0, 2.0), (2.0, 2.0), 16), note="|ε|∞+|c|∞+4 ≤ 16 for patch64 ROI")
    return out


def gradient_receiver_check(pair="A"):
    """§6-7: 실제 wrapper 에서 Rec→A/U, Off→A only, 추가 hard/soft→U only (K 단계 routing 검사). k10.k_step 의 검사 모드 사용."""
    from tools.eqrec4 import k10
    return k10.receiver_check(pair)


def main(profile=False):
    C.ensure_dirs(); C.probe_manifest(); man = C.make_manifest(); status = dict(implementation_invalid=[], provenance_limited=[])
    with C.Stage("G00", "assets·provenance·metric·support·graph gate"):
        rows = []
        for fam, seed, tag in C.CORE + C.EXTRA + [C.PAIRS[p]["S"] for p in C.PAIRS]:
            d = C.ckpt_dir(fam, seed, tag)
            if d is None:
                rows.append(dict(model_key=C.mkey(fam, seed, tag), family=fam, seed=seed, checkpoint_kind=tag, run_id=C.run_of(fam, seed), exists=False)); status["provenance_limited"].append(f"missing:{C.mkey(fam, seed, tag)}"); continue
            L = C.load_model(fam, seed, tag, dev="cpu"); rows.append(dict(exists=True, **L.manifest()))
        C.write_csv(os.path.join(CAMP, "assets_manifest.csv"), rows)
        fd = C.feeders(); L = C.load_model(*C.PRIMARY); h0 = C.state_hash(L.m); rep = reproduce_primary(L, fd); h1 = C.state_hash(L.m)
        rep["param_buffer_hash_stable_after_eval"] = bool(h0 == h1)
        if not rep["pass_"]:
            status["implementation_invalid"].append(f"primary reproduce Δ={rep['abs_diff']}")
        uc = unit_checks(L)
        if not (uc["warp_sign"]["ok"] and uc["zero_identity_max_abs"] < 1e-9 and abs(uc["coordinate_ramp_dx1_mean_shift"] - 1.0) < 1e-6 and uc.get("aligner_zscore_invariance_max_abs", 0) < 1e-3 and uc.get("reconstruction_sees_raw_pan_not_zscore", True)):
            status["implementation_invalid"].append("unit_checks")
        gr = gradient_receiver_check("A")
        if not gr.get("ok"):
            status["implementation_invalid"].append("gradient_receivers")
        tok = os.path.join(ROOT, "work_dir", "campaign_gates_enabled.txt"); gates = open(tok).read().strip() if os.path.exists(tok) else ""
        chain = subprocess.run("ps -eo args | grep -c '[_]run_cases\\.sh'", shell=True, capture_output=True, text=True).stdout.strip()
        pairs = {p: dict(T=dict(key=C.mkey(*v["T"]), step=C.ckpt_step(*v["T"])), S=dict(key=C.mkey(*v["S"]), step=C.ckpt_step(*v["S"]), rule="first saved candidate ≥ 20000", late_student_fallback=False)) for p, v in C.PAIRS.items()}
        for p, v in pairs.items():
            if v["S"]["step"] is None:
                v["S"]["late_student_fallback"] = True; status["provenance_limited"].append(f"pair {p} student mid checkpoint missing")
        src = dict(campaign_id=C.CAMPAIGN_ID, plan=C.PLAN, plan_sha256=C.sha256_file(os.path.join(ROOT, C.PLAN)), git=_git(), torch=str(torch.__version__), device=str(DEV), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                   evaluator_hash=rep["evaluator_hash"], data=C.load_json(os.path.join(CAMP, "data_manifest_meta.json")), pairs=pairs, other_campaign_gate_token=gates, other_chain_running=int(chain or 0) > 0,
                   parquet_available=False, storage="csv/csv.gz + jsonl (same schema; pyarrow absent)", created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        C.dump_json(os.path.join(CAMP, "sources_manifest.json"), src); C.dump_json(os.path.join(CAMP, "g00_reproduction.json"), rep); C.dump_json(os.path.join(CAMP, "g00_unit_checks.json"), uc); C.dump_json(os.path.join(CAMP, "g00_gradient_receivers.json"), gr)
        proto = dict(campaign=dict(id=C.CAMPAIGN_ID, server="s1", nominal_wall_hours=20, hard_time_limit=False, mode="diagnostic_and_short_adaptation"),
                     baseline=dict(case="L1E4", width=112, depth=[1, 2, 3], input_channels=9, task="ms_only", lambda_offset=1e-4, offset_every_updates=2, offset_on_remainder=1, jitter_radius_hr=2.0, jitter_distribution="uniform_disk_area",
                                   jittered_pan_to_unet_during_training=False, recon_grad_to_aligner=True, offset_grad_to_unet=False),
                     assets=dict(primary_teacher=dict(run=C.run_of(*C.PRIMARY[:2]), checkpoint="best_raw (dir best_hqnr)", actual_update=L.step, file_sha256=L.file_sha, aligner_hash=L.aligner_sha, unet_hash=L.unet_sha), pairs=pairs, freeze_teacher=True, share_teacher_student_objects=False),
                     data=dict(split_seed=C.SPLIT_SEED, grouping="index_block_32_proxy (source_group_unknown)", targets=C.SPLIT_TARGET, explicitly_track_pretraining_exposure=True, canonical_views_only=True),
                     quadrants=dict(error="native_hrms_l1_full64_normalized", consistency="unweighted_offset_component_mean_l1 (bank A)", threshold_source="calibration_A", threshold_rule="median_per_model_checkpoint_and_scale", tie_goes_to="low",
                                    min_unique_patches_for_cell_analysis=32, min_source_groups_for_supported_group_claim=5, detailed_samples_per_quadrant_max=64),
                     probes=dict(radii_hr=list(C.RADII), bank_A_angles_deg=list(C.BANK_ANGLES["A"]), bank_B_angles_deg=list(C.BANK_ANGLES["B"]), identity_counted_in_error_mean=False, record_full_2d_response=True),
                     roi=C.ROI_MARGIN, k10=dict(pairs=2, trials_per_cell_max=4, unique_fit_samples_target=48, updates_per_arm=64, arms=["R0", "RH", "RS"], gamma_extra=0.1, unet_lr=1e-5, aligner_lr=1e-6, optimizer_state="fresh_identical_per_trial_arm"),
                     k20=dict(updates_per_arm=5000, arms=["R", "U", "EA", "EAQ", "EAQ_SHUF"], gamma_extra=0.1, lr_schedule="warmup100_cosine_new_5k", endpoint="exact_5000", locked_eval="D", shrinkage=4, mean_soft_fraction=0.5),
                     evaluation=dict(official_fr="raw_original_paper_mat20", stress_reference="original_native_pan_and_ms", bootstrap_repeats=2000, bootstrap_unit="source_group_proxy_block32"), status=status)
        yaml.safe_dump(proto, open(os.path.join(CAMP, "protocol_resolved.yaml"), "w"), sort_keys=False, allow_unicode=True)
        with open(os.path.join(CAMP, "protocol_changes.md"), "a") as f:
            f.write(f"# protocol changes / deviations (G00 {time.strftime('%Y-%m-%d %H:%M')})\n- parquet 미지원(pyarrow 없음) → 같은 schema 의 csv/csv.gz/jsonl.\n- 원본 scene/strip id 없음 → 연속 index 32 블록을 source group proxy 로 (source_group_unknown). patch 수를 scene 수로 세지 않는다.\n"
                    f"- Pair 학생 checkpoint: S1234 ≥20K 최초 저장 = step {pairs['A']['S']['step']} (그 run 의 best_raw 와 같은 update), S7777 = step {pairs['B']['S']['step']}.\n- 모든 patch 는 pretraining 에 노출됨 (seen_in_pretraining=True).\n- gate 상태: token='{gates}', 다른 체인 실행 중={int(chain or 0) > 0}.\n")
        print(json.dumps(dict(reproduce=dict(recorded=rep["recorded_best_raw_hqnr"], got=rep["reproduced_raw_original_hqnr"], ok=rep["pass_"]), unit_ok=not status["implementation_invalid"], status=status, pairs=pairs), ensure_ascii=False, default=str))
    return status
