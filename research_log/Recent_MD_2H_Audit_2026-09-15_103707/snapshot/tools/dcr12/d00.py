"""D00 — 자산·계측·수치 검사 (계획 §4). manifest.json / source_identity.json / d00_smoke.json. 하나라도 hard 검사가 실패하면 implementation_invalid (exit 2)."""
import copy, json, os, time
import numpy as np, torch
from tools.dcr12 import common as C
from kdv.teacher_assets import extract_aligner_sd, tensors_sha, state_hash, sha256_file


def _sha_tensors(sd):
    return tensors_sha(sd)


def run_status(ck, seed):
    rd = C.run_dir(ck, seed)
    if C.run_complete(ck, seed):
        return "existing_complete"
    if os.path.exists(os.path.join(rd, "meta", "started_at.txt")) and not os.path.exists(os.path.join(rd, "meta", "finished_at.txt")):
        return "running"
    if os.path.exists(rd):
        return "incomplete_dir"
    return "pending" if os.path.exists(os.path.join(C.ROOT, "config", C.run_name(ck, seed) + ".yaml")) else "blocked_no_config"


def manifest(T, cal):
    k = C.G.kdv_block("JK0", C.SEEDS[0], C.SERVER, cal=dict(tau_R=cal["tau_R"], lambda_E=cal["lambda_E"]))
    t0_meta = json.load(open(os.path.join(C.T0_DIR, f"{C.T0_TAG}_meta.json"))); t0_file = os.path.join(C.T0_DIR, C.T0_TAG, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(t0_file); A_sd = extract_aligner_sd(sd); U_sd = {kk: v for kk, v in sd.items() if kk.startswith("backbone.")}
    runs = {}
    for seed in C.SEEDS:
        for ck in ("B0", "B1"):
            rd = C.run_dir(ck, seed); ih = C.load_json(os.path.join(rd, "initialization_hashes.json"), None)
            init_file = os.path.join(C.ROOT, "work_dir", "_kdv_init_w112_d123", f"init_unet_seed{seed}.pt")
            runs[f"{ck}_S{seed}"] = dict(case_id=ck, existing_case_alias=C.CASES[ck], run_id=C.run_name(ck, seed), seed=seed, status=run_status(ck, seed), config=f"config/{C.run_name(ck, seed)}.yaml",
                                         init_hashes=ih, init_unet_file_exists=os.path.exists(init_file), expect_init=(C.G.init_hash_for(seed)),
                                         candidates=(sorted(int(x.split('-')[1]) for x in os.listdir(os.path.join(rd, "candidates"))) if os.path.isdir(os.path.join(rd, "candidates")) else []),
                                         selected=C.load_json(os.path.join(rd, "best_hqnr_meta.json"), None), meta_git=(open(os.path.join(rd, "meta", "git_commit.txt")).read().strip() if os.path.exists(os.path.join(rd, "meta", "git_commit.txt")) else None))
    th = C.load_json(os.path.join(C.ROOT, "work_dir", "_pakd50_budget", "ledger.json"), {}).get("throughput")
    gates = open(os.path.join(C.ROOT, "work_dir", "campaign_gates_enabled.txt")).read().split() if os.path.exists(os.path.join(C.ROOT, "work_dir", "campaign_gates_enabled.txt")) else []
    m = dict(plan_id=C.CAMPAIGN_ID, analysis_campaign_id=C.CAMPAIGN_ID, plan=C.PLAN, created=time.strftime("%Y-%m-%dT%H:%M:%S"), **C.host_info(),
             teacher=dict(logical="PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1 / best_raw(=best_hqnr)", dir=C.G.T0_ASSET_DIR, file_sha256=sha256_file(t0_file), tensors_sha256_16=_sha_tensors(sd), A_tensors_sha256_16=_sha_tensors(A_sd), U_tensors_sha256_16=_sha_tensors(U_sd),
                          source_selected_update=t0_meta.get("step"), selected_hqnr=t0_meta.get("hqnr"), n_A_params=sum(v.numel() for v in A_sd.values()), n_U_params=sum(v.numel() for v in U_sd.values())),
             calibration=cal, optimizer=dict(U="AdamW peak 1e-4 wd 0.01", A="AdamW peak 1e-5 wd 0.01 (B1 만; B0 는 optimizer 밖·WD 없음)", schedule="warmup100 + cosine, 50K horizon", batch=48, updates=50000),
             kdv_contract=dict(protocol=k["input_protocol"], corruption=k.get("corruption"), aux=k.get("aux"), routing=k.get("routing"), rec=k["rec"], stat=k["stat"], donor_view_margin=k["donor"]["view_margin_hr"], candidate_grid=k["candidate_grid_id"], select=k["select"]),
             comparator="best_hqnr = raw-original HQNR (tie band 1e-4 → fSCC); 후보 50개 전부 보존", cases=dict(B0="FQ: A frozen(T0 복사) · U ← L_H + L_K + λE L_E", B1="JK0: A trainable LR 1e-5 · A ← L_H + λE L_E + odd·1e-4 L_off (soft→A 차단, routing qK=0) · U ← Q12 전체",
                                                                                                    B1_ALL="JQ (soft→A 포함) — 이번 학습 없음, 있으면 진단만", B2="FQ + 사분면 soft multiplier — 미구현·gate 통과 시만", B3="JK0 + multiplier — 미구현·gate 통과 시만"),
             runs=runs, seeds=C.SEEDS, host_pairing="같은 호스트(s1) 안에서 B0/B1 pair 를 완성 (§8.2); s1 의 FQ S1234 v1 완료본을 B0-1234 로 재사용, 나머지 3 run 새 학습 (FQ S777 은 s2 완료본과 이름이 같지만 별개 host 학습)",
             throughput_observed=th, execution_isolation=dict(campaign_gates_enabled=gates, note="s1 의 PAKD50 chain 은 2026-09-14 22:40 사용자 지시로 중단; gate 'pakd50' 는 DCR12 학습 큐가 옛 우선순위를 재주입하지 않도록 비운다 (D00 이 기록)",
                                                              budget_ledger="config 의 kdv.budget 은 PAKD50 ledger(work_dir/_pakd50_budget/ledger.json; 50h·마감 09-16 11:22:31) 를 그대로 쓴다 — DCR12 별도 ledger 는 만들지 않고 이 manifest 에 예상 시간을 적는다"),
             panels=dict(seed=C.SPLIT_SEED, sizes=C.PANEL, grad=C.GRAD_N, disc128=C.DISC128_N, fr8=C.FR8_N, independence="patch_only"), probes=dict(r=list(C.PROBE_R), n_main=16, n_confirm=16), bias_hr=C.BIAS_HR,
             d03_steps=C.D03_STEPS, d04_steps=C.D04_STEPS, expected_hours=dict(train_per_run="≈2.2–2.5 (s1 PAKD50 FQ 2.15 h 실측)", train_total="3 run ≈ 7 h", diagnostics="≈1.5 h", total="≈ 9 h (계획 10–16 h 안)"))
    return m


def smoke(T, cal):
    """§4.2 필수 smoke — 실제 wrapper/trainer 경로로. 반환 (dict, hard_fail list)."""
    out, hard = {}, []
    # warp 부호: impulse 가 c=(0,1) 에서 x−1 로, zero identity 정확
    P = torch.zeros(1, 1, 33, 33, dtype=torch.float64); P[0, 0, 16, 16] = 1; w = C.E.warp_pan(P, torch.tensor([[0.0, 1.0]], dtype=torch.float64)); yx = np.unravel_index(int(w.argmax()), (33, 33))
    out["warp_sign"] = dict(ok=(yx == (16, 15)) and float((C.E.warp_pan(P, torch.zeros(1, 2, dtype=torch.float64)) - P).abs().max()) == 0.0, impulse_at=list(map(int, yx)), convention="W(P,c)[y,x] = P[y+c_y, x+c_x]; (dy,dx) 순서")
    # 실제 batch (CAL 앞 4 patch) 로 stub step
    panels = C.make_panels(); ids = panels["ids"]["CAL"][:4]; gt, ms, lpan, pan = (t.to(C.DEV) for t in C.load_patches(ids))
    S = copy.deepcopy(T.m).train().requires_grad_(True); g7 = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p_ in S.backbone.parameters():
            p_.add_(0.01 * torch.randn(p_.shape, generator=g7).to(p_.device))          # e_S ≠ e_T 라 soft 항이 살아 있게
    Tm = T.m
    def _grads(loss, params):
        return C.flat(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)
    # B1(JK0) 홀수 update: routing 뒤 A .grad = ∇φ(L_H + λE L_E + 1e-4 L_off); U .grad = ∇θ L_Q 전체 (= JQ 의 U grad, 같은 상태·같은 ε)
    trJ = C.trainer_stub("JK0", C.SEEDS[0], copy.deepcopy(S), Tm, cal); tot, info = trJ._step(gt, ms, lpan, pan, 1)
    ap = list(trJ.M.aligner.parameters()); bp = list(trJ.M.backbone.parameters())
    LH, LK, LEw, LO = info["_rec_hard_t"], info["_rec_soft_t"], info["_edge_w_t"], float(info["lam_off"]) * info["loss_off"]
    want_A = _grads(LH + LEw + LO, ap); want_U = _grads(tot, bp); gK_A = _grads(LK, ap); gO_U = _grads(info["loss_off"], bp); gH_A_norm = float(_grads(LH, ap).norm()); gE_A_norm = float(_grads(LEw, ap).norm())
    tot.backward(retain_graph=True); trJ._apply_routing(info, trJ.M); got_A = torch.cat([p.grad.flatten() for p in ap]); got_U = torch.cat([p.grad.flatten() for p in bp])   # routing 은 graph 를 해제한다 — 그 뒤 autograd 없음
    trQ = C.trainer_stub("JQ", C.SEEDS[0], copy.deepcopy(S), Tm, cal); totQ, infoQ = trQ._step(gt, ms, lpan, pan, 1); u_JQ = _grads(totQ, list(trQ.M.backbone.parameters()))
    eA = float((got_A - want_A).abs().max()); eU = float((got_U - want_U).abs().max()); eUQ = float((got_U - u_JQ).abs().max()); ref = float(want_A.abs().max())
    out["soft_routing"] = dict(ok=(eA <= 1e-4 * (1 + ref)) and eU == 0.0 and eUQ <= 1e-4 * (1 + float(want_U.abs().max())), tolerance="1e-4 relative (GPU FP32 reduction; K12 CPU 검사는 1e-5)", max_abs_err_A=eA, ref_A=ref, err_U_vs_total=eU, err_U_vs_JQ=eUQ,
                               soft_to_A_before_routing=float(gK_A.abs().sum()), eq_exercise=info.get("eq_exercise"), lam_off=float(info["lam_off"]), policy="qD=1,qK=0,qE=1")
    out["native_task_to_A"] = dict(ok=gH_A_norm > 0 and gE_A_norm > 0, gH_A_norm=gH_A_norm, gE_A_norm=gE_A_norm)
    out["offset_to_U"] = dict(ok=float(gO_U.abs().sum()) == 0.0, g_off_U_abs_sum=float(gO_U.abs().sum()))
    # forward 기준: stub 의 total == L_H + L_K + λE L_E + 1e-4 L_off (직접 재계산)
    q = C.q12_terms(info["y"].detach(), info["y_t"], gt, cal); direct = q["LH"] + q["LK"] + cal["lambda_E"] * q["LE"] + 1e-4 * float(info["loss_off"])
    out["forward_reference"] = dict(ok=abs(float(tot) - float(direct)) <= 1e-6 * (1 + abs(float(tot))), total_trainer=float(tot), total_direct=float(direct), L0=float(q["L0"]), LH=float(q["LH"]), LK=float(q["LK"]), LE=float(q["LE"]), L_off=float(info["loss_off"]))
    # Teacher 출력 불변: student 한 step 뒤 T 출력 동일 · Teacher hash 불변 · frozen B0 A == T0 A
    hT = state_hash(Tm)
    with torch.no_grad():
        yT0 = Tm(pan, ms, lpan)["y"].clone()
    opt = torch.optim.AdamW([dict(params=list(trJ.M.backbone.parameters())), dict(params=ap, lr=1e-5)], lr=1e-4, weight_decay=0.01); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        yT1 = Tm(pan, ms, lpan)["y"]
    out["teacher_invariant"] = dict(ok=torch.equal(yT0, yT1) and state_hash(Tm) == hT, hash=hT)
    b0 = C.reference_student(C.SEEDS[0])
    if b0:
        Pb = C.load_pipe("S", *b0); out["frozen_asset"] = dict(ok=Pb.aligner_sha16 == T.aligner_sha16, B0_A_hash=Pb.aligner_sha16, T0_A_hash=T.aligner_sha16, B0=Pb.key, note="B0 의 A == T0 의 A (frozen; weight decay 도 받지 않았다)")
    else:
        out["frozen_asset"] = dict(ok=None, note="B0 checkpoint 없음 (학습 뒤 재검사)")
    # identity probe floor · RNG 격리 · 수치 floor
    st = torch.get_rng_state(); cs = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    ids32 = panels["ids"]["CAL"][:32]; g2, m2, l2, p2 = C.load_patches(ids32); r1 = C.consistency(T, p2, m2); r2 = C.consistency(T, p2, m2)
    out["identity_probe"] = dict(ok=float(r1["identity"].max()) < 1e-4, max_abs=float(r1["identity"].max()), note="C(ε=0) 는 주 평균에 넣지 않는다")
    out["numerical_floor"] = dict(c0_repeat_max_abs=float((r1["c0"] - r2["c0"]).abs().max()), C_repeat_max_abs=float((r1["C"] - r2["C"]).abs().max()), note="같은 입력 반복 (D01 의 DEGENERATE_METRIC 기준)")
    out["rng_isolation"] = dict(ok=torch.equal(st, torch.get_rng_state()) and (cs is None or torch.equal(cs, torch.cuda.get_rng_state())), note="진단은 전역 RNG 를 소비하지 않는다; 학습 ε 는 trainer 의 별도 generator")
    # support: r ≤ 1 probe + |c0| 가 margin-4 view 를 넘지 않는지 (§5.2)
    out["probe_support"] = dict(ok=bool((r1["c0"].abs().max(1).values + 1.0 + 4 <= T.mg + 4 + 1e-9).all()) if T.mg >= 4 else None, margin=T.mg, max_c0=float(r1["c0"].abs().max()), rule="|ε|∞ + |c0|∞ ≤ margin (+ 4 guard) — 보수적")
    for k, v in out.items():
        if isinstance(v, dict) and v.get("ok") is False:
            hard.append(k)
    return out, hard


def main(profile=False):
    with C.Stage("D00", "asset/instrument/numeric checks"):
        T = C.load_pipe("T"); cal = C.calibration(); C.probe_manifest(); C.make_panels()
        m = manifest(T, cal); C.dump_json(os.path.join(C.CAMP, "manifest.json"), m)
        src = dict(T0=dict(key=T.key, sha16=T.sha16, A_sha16=T.aligner_sha16, step=T.step, file=os.path.join(C.T0_DIR, C.T0_TAG)), students={})
        for ck, seed, tag in C.available_students(("best_hqnr", "last")):
            P = C.load_pipe("S", ck, seed, tag); src["students"][P.key] = dict(run=C.run_name(ck, seed), tag=tag, step=P.step, sha16=P.sha16, A_sha16=P.aligner_sha16, A_equals_T0=(P.aligner_sha16 == T.aligner_sha16))
        C.dump_json(os.path.join(C.CAMP, "source_identity.json"), src)
        sm, hard = smoke(T, cal); sm["hard_failures"] = hard; sm["implementation_invalid"] = bool(hard); C.dump_json(os.path.join(C.CAMP, "d00_smoke.json"), sm)
        print("[d00] smoke:", {k: (v.get("ok") if isinstance(v, dict) else v) for k, v in sm.items() if k not in ("hard_failures",)}, flush=True)
        print("[d00] runs:", {k: v["status"] for k, v in m["runs"].items()}, flush=True)
        return dict(implementation_invalid=hard)
