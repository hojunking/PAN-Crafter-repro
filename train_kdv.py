"""KDVTrainer — s2 W112·D124: GT-anchored adaptive KD · 출력 통계 variance · aligner 재사용
(research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md). trainer: kdv.

  L_S = L_R(rec hard/soft, §6) + λ_V·L_V(출력 통계 H/T/FIX/WH/AD, §9) + L_aux(A2 edge / A3 geo / N offset, §10) + λ_GKD·L_G(G1 mean-KD, §11.5)
  aligner 정책: A-FR(donor frozen, Teacher 와 correction 공유) · A-FT(donor 초기화 후 학습) · A-SC(독립 초기화 학습) · A-ID(aligner·sampler 없음)
  입력: I-A(native) · I-N(native/corrupt 1:1, W(W(P,ε),c)) · I-NATIVE-TRANSFER
Teacher(F_T+A_T) 는 frozen·no_grad·optimizer 밖(gate M03: state hash 불변·parameter ID 교집합 없음). calibration(tau_R·tau_V·λ_V) 은 train-only, 캐시.
평가·선택·산출물은 train_pa.PATrainer(세 view · best_raw=best_hqnr · best_aligned · last) 에 best_rr_val(검증셋 ERGAS) 을 더한다 (§18)."""
import copy
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from kdv.calibration import calibration_batches, calibrate_rec, calibrate_stat, calibrate_lambda, cached, calibrate_covariance, teacher_precision_fn, calibrate_covhead
from kdv.alignment_kd import CovHead, kd_matrix, mean_kd_loss, gaussian_kl, precision_stats, geo_autograd_grad
from kdv.tri import (direction_mask, mass_kappa, shuffle_mask, component_weights, masked_l1, masked_l1_componentwise, routing_stats, teacher_fd_jacobian, teacher_eval,
                     sens_q, sens_risk, diag_risk, lowrank_quad, iso_quad, scalar_quad, sigma_from_precision, sigma_control, linearization_check)
from kdv.calibration import calibrate_component_tau, calibrate_sens, calibrate_lambda_q
from kdv.forward import kdv_forward
from kdv.losses_rec import GTAnchoredReconstructionKD
from kdv.losses_stat import stat_term, statistic_map, MODE_TO_CRITERION
from kdv.protocol import prepare_view
from kdv.registry import resolve, run_name, describe, stat_tag, arch_prefix
from kdv.teacher_assets import load_donor_aligner, load_run_model, load_state, freeze, state_hash, assert_param_disjoint, sha256_file, tensors_sha
from pa.aligner import PANGlobalAligner
from pa.evalviews import PROTOCOL_ID, evaluator_hash, fixed_roi, VIEWS
from pa.losses import output_edge_loss, direct_geometry_loss, lambda_ramp, geometry_support_margin, scharr
from pa.model import PAModel
from pa.offset import offset_loss, predict_c, lambda_off
from pa.selector import BestSelector
from pa.warp import support_margin_ok, warp_support_mask
from train_pa import PATrainer, ROOT, EXIT_SUPPORT_FAIL, sha256_file as sha_cached
from train_po import RNGState
from utils import Train_Report, Test_Reduced_Report, reduced_metrics

EXIT_GATE = 4
MAX_ABS_DELTA_TRAIN = 8.0
CAMPAIGN_ROOT = os.path.join(ROOT, "work_dir", "_kdv_campaign")
CALIB_ROOT = os.path.join(ROOT, "work_dir", "_kdv_calibration")


def ergas_per_sample(gt_dn, pred_dn, ratio=4):
    """plain ERGAS (dim_cut 없음) — RR-val 선택 전용, 시트 ERGAS(MATLAB 포팅)와 같은 값이 아니다."""
    mse = ((gt_dn - pred_dn) ** 2).mean(dim=(2, 3)); mu = gt_dn.mean(dim=(2, 3))
    return 100.0 / ratio * torch.sqrt((mse / mu.clamp_min(1e-12) ** 2).mean(dim=1))


class KDVTrainer(PATrainer):
    # ------------------------------------------------------------------ 구성
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']; self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']; self.test_full_data_loader = data_loader['test_full']
        assert getattr(args, "mars", "dual") == "ms" and args.res, "KDV 는 단일 HRMS task(mars: ms) · 잔차 base 고정"
        assert args.model_args.get("in_mode") == "paper" and not args.model_args.get("attn_locations"), "입력 9ch(in_mode paper) · attention 없음"
        assert args.model_args.get("mode_modulation", True) is False, "MARs γ/β 제거본 위에서만"
        k = getattr(args, "kdv", {}) or {}
        self.k = k; self.spec = resolve(k); sp = self.spec
        self.campaign_id = k.get("campaign_id", "S2_W112_KDV_20260910"); self.run_kind = k.get("run_kind", "CONTROLLED"); self.version = k.get("version", "v01")
        self.run_id = os.path.basename(args.work_dir.rstrip("/"))
        self.prefix = arch_prefix(args.model_args); expected = run_name(sp, args.seed, self.version, self.prefix)
        if k.get("check_run_name", True) and self.run_id != expected:
            raise ValueError(f"work_dir 이름 {self.run_id} ≠ 규칙 이름 {expected} (§13.1) — kdv.check_run_name: false 로 끌 수 있다")
        self.case = f"{sp['policy']}/{sp['rec_case']}/{stat_tag(sp)}/{sp['geom']}"
        hidden = int(args.model_args.get("hidden_size")); depth = "".join(str(d) for d in args.model_args.get("depth"))
        self.init_dir = os.path.join(ROOT, k.get("init_dir", f"work_dir/_kdv_init_w{hidden}_d{depth}"))
        self.diag_every = int(k.get("diag_every", 1000)); self.diag_iter = self.diag_every
        self.lam_edge, self.lam_geo = sp["edge_weight"], sp["geometry_weight_effective"]
        self.ramp = sp["aux_ramp_updates"]; self.geo_sigma = sp["geometry_sigma_hr"]; self.geo_margin = sp["geometry_margin_hr"]
        self.guard_margin = geometry_support_margin(self.geo_sigma, self.geo_margin)
        self.protocol = sp["protocol"]; self.radius_hr = sp["radius_hr"]
        st = dict(k.get("stat") or {}); self.stat_ramp = int(st.get("ramp_updates", 0))
        self._t_run0 = time.time()

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)
        self.weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.accelerator.mixed_precision, torch.float32)

        # --- 초기값 (§4.3): U-Net 은 seed 별 저장된 독립 초기값을 모든 Student 가 공유. scratch aligner 는 seed+1000 (PA 와 같은 경로).
        nb = int(args.num_bands)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed) + 1000)
            scratch = PANGlobalAligner(nb)
        self.init_hashes = self._pair_init(model, scratch)
        # --- aligner 정책 (§4.2)
        pol = sp["policy"]; self.donor_manifest = None
        if pol in ("A-FR", "A-FT"):
            d = k["donor"]; aligner, self.donor_manifest = load_donor_aligner(d["source"], nb, d.get("expected_sha256"))
            margin = int(d.get("view_margin_hr", 0) or 0)
        elif pol == "A-SC":
            aligner = scratch; margin = int(k.get("aligner_view_margin_hr", 0) or 0)
        else:
            aligner = None; margin = 0
        self.aligner_view_margin = margin; self.aligner_trainable = sp["aligner_trainable"]
        if pol == "A-FR":
            freeze(aligner)
        self.model = PAModel(model, aligner, aligner_margin=margin, sampler=(pol != "A-ID"))
        if sp["geom"] == "G5":
            self.model.cov_head = CovHead()                                # Student covariance head (§11.6) — checkpoint 에 포함, aligner group 으로 학습
        self.aligner_hash0 = state_hash(aligner) if aligner is not None else None
        # --- Teacher (§3.1): 같은 저장소 run 의 checkpoint 를 그 run 의 config 대로 strict 로드 → frozen
        self.teacher = None; self.teacher_manifest = None; self.teacher_id = sp["teacher_id"]
        if sp["has_teacher"]:
            t = k["teacher"]
            self.teacher, self.teacher_manifest = load_run_model(t["run"], t.get("tag", "best_hqnr"), type(model), t.get("expected_sha256"))
            freeze(self.teacher)
            if int(self.teacher_manifest["width"]) != hidden and not bool(t.get("bridge", False)):
                raise ValueError(f"Teacher width {self.teacher_manifest['width']} ≠ Student {hidden}: bridge cohort 는 kdv.teacher.bridge: true 로 명시 (§3.2-3)")
        self.share_correction = (pol == "A-FR" and (self.teacher is None or (self.teacher.aligner is not None and self.teacher.sampler
                                 and self.teacher.aligner_margin == margin and state_hash(self.teacher.aligner) == self.aligner_hash0)))
        # --- optimizer: backbone (+ trainable aligner, 같은 LR §17.2)
        groups = [dict(params=[p for p in self.model.backbone.parameters() if p.requires_grad], name="backbone")]
        if self.aligner_trainable:
            groups.append(dict(params=list(self.model.aligner.parameters()) + (list(self.model.cov_head.parameters()) if self.model.cov_head is not None else []), name="aligner"))
        self.optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=args.num_warmup, num_training_steps=args.num_iter)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        if self.teacher is not None:
            self.teacher.to(self.accelerator.device); assert_param_disjoint(self.optimizer, self.teacher)
            if self.M.cov_head is not None:
                assert all(id(p) in {id(q) for g in self.optimizer.param_groups for q in g["params"]} for p in self.M.cov_head.parameters()), "Student cov head 가 optimizer 에 없다"
            self.teacher_hash0 = state_hash(self.teacher)
        if pol == "A-FR":
            assert_param_disjoint(self.optimizer, self.M.aligner)
        # --- I-N corruption RNG (checkpoint 에 포함)
        self.corr_seed = int(args.seed) + int((k.get("corruption") or {}).get("corruption_seed_offset", 2000))
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(self.corr_seed)
        self.accelerator.register_for_checkpointing(RNGState(self.gen))
        # --- 평가 참조·선택기 (PATrainer)
        self.last_reduced_metrics, self.last_full_metrics, self.last_val_metrics = {}, {}, {}
        self.last_fscc_official = float("nan"); self.raw_is_best = False
        self._ema = {}; self._global_step = 0; self.step_records = {}; self.start_step = 0
        self.cand_dir = os.path.join(args.work_dir, "candidates")
        self.fr_h5 = args.test_full_feeder_args["dataroot"]; self.fr_h5_sha = sha_cached(self.fr_h5, self.init_dir)
        import h5py
        with h5py.File(self.fr_h5) as f:
            self._fr_lms = np.asarray(f["lms"], dtype=np.float64); self._fr_pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
        self._roi = None; self._rr_val_last = float("nan")
        self._init_selectors()
        # --- calibration → criteria
        self._calib = {}; self._calibrate_and_build()
        if self.accelerator.is_main_process:
            json.dump(self.init_hashes, open(os.path.join(args.work_dir, "initialization_hashes.json"), "w"), indent=1)
            self._write_manifests(); self._write_kdv_manifests()
        if k.get("phase"):
            self._warm_start(k["phase"])

    # ------------------------------------------------------------------ selectors (+ rr_val)
    def _init_selectors(self):
        super()._init_selectors()
        self.sel_rrval = BestSelector("rr_val", tol_hqnr=1e-4, tol_fscc=1.0)
        p = os.path.join(self.args.work_dir, "selector_state_rr_val.json")
        if getattr(self.args, "resume", None) and os.path.exists(p):
            loaded = BestSelector.load(p, expect=dict(protocol_id=PROTOCOL_ID))
            self.sel_rrval.max_hqnr, self.sel_rrval.cands, self.sel_rrval.best, self.sel_rrval.history = loaded.max_hqnr, loaded.cands, loaded.best, loaded.history

    # ------------------------------------------------------------------ calibration (§6.4, §9.2, §9.3)
    def _batches(self):
        if "batches" not in self._calib:
            c = self.k.get("calibration") or {}
            b, man = calibration_batches(self.args, int(c.get("n_patches", 3072)), int(c.get("seed", 1234)), int(self.args.batch_size))
            self._calib["batches"] = b; self._calib["set_manifest"] = man
        return self._calib["batches"]

    def _calibrate_and_build(self):
        sp = self.spec; dev = self.accelerator.device; k = self.k
        rec = dict(k.get("rec") or {}); st = dict(k.get("stat") or {})
        tid = self.teacher_id or "no_teacher"; croot = os.path.join(CALIB_ROOT, tid)
        tkey = dict(teacher_sha=(self.teacher_manifest or {}).get("tensors_sha256_16"), evaluator=evaluator_hash())
        self.rec_crit = None; self.stat_crit = None; self.lam_V = 0.0; self.calibration = dict(teacher_id=tid)
        if self.teacher is not None:
            tau = rec.get("tau", "calibrate")
            if tau == "calibrate":
                key = dict(tkey, set=self._calib_set_key(), kind="rec")
                r, hit = cached(os.path.join(croot, "rec.json"), key, lambda: calibrate_rec(self.teacher, self._batches(), dev, float(rec.get("eps_scale", 1e-6))))
                tau = r["tau_R"]; self.calibration["rec"] = dict(r, from_cache=hit, source="calibrate")
            else:
                self.calibration["rec"] = dict(tau_R=float(tau), source="config")
            self.rec_crit = GTAnchoredReconstructionKD(float(tau), alpha=float(rec.get("alpha", 1.0)), kd_weight=float(rec.get("kd_weight", 0.1)),
                                                       eps=float(rec.get("eps", 1e-6)), mode=sp["rec_mode"]).to(dev)
        if sp["stat_enabled"]:
            kind, w, mode = sp["stat_kind"], sp["stat_window"], sp["stat_mode"]
            sid = f"{kind}_w{w}"
            if self.teacher is not None:
                tv = st.get("tau", "calibrate")
                if tv == "calibrate":
                    key = dict(tkey, set=self._calib_set_key(), kind=kind, window=w)
                    r, hit = cached(os.path.join(croot, f"stat_{sid}.json"), key, lambda: calibrate_stat(self.teacher, self._batches(), kind, w, dev))
                    if r.get("degenerate"):
                        self._gate_fail("CALIBRATION_DEGENERATE", dict(stage="tau_V", result=r))
                    tau_v, eps_v = r["tau_V"], r["eps_V"]; self.calibration["stat"] = dict(r, from_cache=hit, source="calibrate")
                else:
                    tau_v, eps_v = float(tv), float(st.get("eps", 1e-6)); self.calibration["stat"] = dict(tau_V=tau_v, eps_V=eps_v, source="config")
                if mode in MODE_TO_CRITERION:
                    self.stat_crit = GTAnchoredReconstructionKD(tau_v, alpha=float(st.get("alpha", 1.0)), kd_weight=float(st.get("kd_weight", 0.1)), eps=eps_v,
                                                                mode=MODE_TO_CRITERION[mode]).to(dev)
            lam = st.get("outer_weight", "calibrate")
            if lam == "calibrate":
                pilot = st.get("lambda_pilot")
                if not pilot:
                    raise ValueError("stat.outer_weight: calibrate 는 stat.lambda_pilot(<run>/<tag> 또는 init) 이 필요하다 (§9.3)")
                r_grad = float(st.get("r_grad", 0.05))
                if pilot == "init":
                    pm, pkey = self.M, dict(pilot="init", init=self.init_hashes.get("unet_init_sha256_16"))
                else:
                    prun, ptag = pilot.rsplit("/", 1)
                    pm, pman = load_run_model(os.path.join("work_dir", prun), ptag, type(self.M.backbone)); freeze(pm); pm.to(dev)
                    pkey = dict(pilot=pilot, sha=pman["tensors_sha256_16"])
                key = dict(pkey, set=self._calib_set_key(), kind=kind, window=w, r_grad=r_grad)
                was_training = self.M.training; self.M.eval()
                r, hit = cached(os.path.join(croot, f"lambda_{sid}_{pkey['pilot'].replace('/', '_')}.json"), key, lambda: calibrate_lambda(pm, self._batches(), kind, w, r_grad, dev))
                self.M.train(was_training)
                if pilot != "init":
                    del pm
                if r.get("degenerate"):
                    self._gate_fail("CALIBRATION_DEGENERATE", dict(stage="lambda_V", result=r))
                self.lam_V = float(r["lambda_V"]); self.calibration["lambda"] = dict(r, from_cache=hit, source="calibrate", pilot=pilot)
            else:
                self.lam_V = float(lam); self.calibration["lambda"] = dict(lambda_V=self.lam_V, source="config")
        # 이동량 covariance KD (§11): Π_T 출처 calibration → k0·임계·C01 판정, G5 는 Teacher cov head
        self.prec_fn = None; self.cov = None; self.lam_gkd = 0.0; self.k0 = None; self.cov_head_t = None; self.cov_head_t_hash = None
        if sp["geom"] != "G0":
            g = dict(k.get("geom_kd") or {}); src = sp["cov_source"]
            key = dict(tkey, set=self._calib_set_key(), source=src, probes=sp["probes"], geo=sp["geo"], eq_sigma_min=sp["eq_sigma_min"], geometry=dict(sigma=self.geo_sigma, margin=self.geo_margin))
            was_training = self.M.training; self.M.eval()
            cov, hit = cached(os.path.join(croot, f"cov_{src}.json"), key, lambda: calibrate_covariance(self.teacher, self._batches(), sp, dev, self.geo_sigma, self.geo_margin))
            if cov.get("status") != "OK":
                self._gate_fail(cov.get("status", "COV_FAIL"), dict(stage="covariance", result={kk: v for kk, v in cov.items() if not isinstance(v, (list, dict)) or kk in ("fd_vs_autograd_rel_err", "h_consistency_rel")}))
            self.cov = dict(cov, from_cache=hit); self.k0 = float(sp["geom_k0"]) if sp["geom_k0"] is not None else float(cov["k0"])
            self.lam_gkd = float(sp["geom_outer_weight"]) if sp["geom_outer_weight"] is not None else float(sp["geom_r_gkd"]) / (1.0 if src == "struct" else self.k0)
            self.prec_fn = teacher_precision_fn(self.teacher, sp, cov, self.geo_sigma, self.geo_margin)
            self.calibration["cov"] = dict(self.cov, k0_used=self.k0, lambda_GKD=self.lam_gkd, lambda_rule=("config" if sp["geom_outer_weight"] is not None else "r_gkd/k0"), r_gkd=sp["geom_r_gkd"])
            if sp["geom"] == "G5":
                hkey = dict(key, covhead_epochs=sp["covhead_epochs"], covhead_lr=1e-2)            # 검토 지적: epochs/lr 도 key 에
                hp = os.path.join(croot, f"covhead_{src}.pt"); hj = hp.replace(".pt", ".json")
                if not (os.path.exists(hp) and os.path.exists(hj) and json.load(open(hj)).get("key") == hkey):
                    sd, hman = calibrate_covhead(self.teacher, self._batches(), sp, cov, dev, epochs=sp["covhead_epochs"], geo_sigma=self.geo_sigma, geo_margin=self.geo_margin)
                    torch.save(sd, hp); json.dump(dict(key=hkey, head=hman, computed_at=time.strftime("%Y-%m-%dT%H:%M:%S")), open(hj, "w"), indent=1)
                self.cov_head_t = CovHead(); self.cov_head_t.load_state_dict(torch.load(hp, map_location="cpu")); freeze(self.cov_head_t); self.cov_head_t.to(dev)
                self.cov_head_t_hash = state_hash(self.cov_head_t); self.calibration["covhead"] = dict(json.load(open(hj))["head"], teacher_head_sha256_16=self.cov_head_t_hash)
                if pol == "A-FT" and not self.k.get("phase"):                                        # A-FT = Teacher 복사 → cov head 도 Teacher head 에서 시작 (검토 지적)
                    self.M.cov_head.load_state_dict(self.cov_head_t.state_dict()); self.calibration["covhead"]["student_init"] = "copied_from_teacher_head"
                else:
                    self.calibration["covhead"]["student_init"] = "identity_sigma"
            self.M.train(was_training)
        # TRI-A/B/C (addendum §10): 성분별 τ · s_sens · λ_Q · C 의 covariance 출처
        tri = sp["tri"]; self.tri = tri; self.tri_state = {}; self.tri_gen = torch.Generator(device="cpu"); self.tri_gen.manual_seed(int(tri["shuffle_seed"]))
        if tri["enabled"]:
            was_training = self.M.training; self.M.eval(); cal = {}
            kind, w = sp["stat_kind"], sp["stat_window"]; sid = f"{kind}_w{w}" if kind else None
            if tri["a_mode"] == "band_adv":
                r, hit = cached(os.path.join(croot, "rec_components.json"), dict(tkey, set=self._calib_set_key(), kind="rec_components"), lambda: calibrate_component_tau(self.teacher, self._batches(), dev, kind=None))
                self.tri_state["tau_R_c"] = torch.tensor(r["tau"], dtype=torch.float64, device=dev); cal["rec_components"] = dict(r, from_cache=hit)
            if tri["b_mode"] == "comp_adv":
                r, hit = cached(os.path.join(croot, f"stat_components_{sid}.json"), dict(tkey, set=self._calib_set_key(), kind=kind, window=w), lambda: calibrate_component_tau(self.teacher, self._batches(), dev, kind=kind, window=w))
                self.tri_state["tau_V_j"] = torch.tensor(r["tau"], dtype=torch.float64, device=dev); self.tri_state["eps_V_j"] = torch.tensor(r["eps"], dtype=torch.float64, device=dev); cal["stat_components"] = dict(r, from_cache=hit)
            if tri["c_mode"] != "off":
                phi = tri["c_phi"]; ck = dict(tkey, set=self._calib_set_key(), h=tri["c_h"], phi=phi, kind=(kind if phi == "stat" else None), window=(w if phi == "stat" else None))
                sens, hit = cached(os.path.join(croot, f"sens_{phi}{'_' + sid if phi == 'stat' else ''}.json"), ck, lambda: calibrate_sens(self.teacher, self._batches(), dev, h=tri["c_h"], phi=phi, stat_kind=(kind if phi == "stat" else None), window=w or 5))
                cal["sens"] = dict(sens, from_cache=hit)
                if sens["status"] != "OK":
                    self._gate_fail(sens["status"], dict(stage="tri_c_fd", result={kk: v for kk, v in sens.items() if kk != "fd_checks"}, fd_checks=sens["fd_checks"]))
                self.tri_state["s_sens"] = float(sens["s_sens"])
                sc = tri["c_s_c"]
                self.tri_state["s_c"] = float(sc) if sc != "tau" else float(self.calibration["stat"]["tau_V"] if phi == "stat" else self.calibration["rec"]["tau_R"])
                if tri["c_mode"] in ("diag", "full", "qscalar"):
                    src = tri["c_src"]; key = dict(tkey, set=self._calib_set_key(), source=src, probes=sp["probes"], geo=sp["geo"], eq_sigma_min=sp["eq_sigma_min"], geometry=dict(sigma=self.geo_sigma, margin=self.geo_margin))
                    spec_c = dict(sp, cov_source=src)
                    cov, hit = cached(os.path.join(croot, f"cov_{src}.json"), key, lambda: calibrate_covariance(self.teacher, self._batches(), spec_c, dev, self.geo_sigma, self.geo_margin))
                    if cov.get("status") != "OK":
                        self._gate_fail("BLOCKED_COVARIANCE", dict(stage="tri_c_covariance", status=cov.get("status")))
                    self.tri_state["prec_fn"] = teacher_precision_fn(self.teacher, spec_c, cov, self.geo_sigma, self.geo_margin); cal["cov"] = dict(cov, from_cache=hit)
                if tri["c_mode"] in ("full", "qiso", "qscalar"):
                    lq = tri["c_lambda_q"]
                    if lq == "calibrate":
                        pilot = (k.get("stat") or {}).get("lambda_pilot") or (k.get("tri") or {}).get("lambda_pilot") or "init"
                        if pilot == "init":
                            pm, pkey = self.M, dict(pilot="init", init=self.init_hashes.get("unet_init_sha256_16"))
                        else:
                            prun, ptag = pilot.rsplit("/", 1); pm, pman = load_run_model(os.path.join("work_dir", prun), ptag, type(self.M.backbone)); freeze(pm); pm.to(dev); pkey = dict(pilot=pilot, sha=pman["tensors_sha256_16"])
                        r, hit = cached(os.path.join(croot, f"lambda_q_{pkey['pilot'].replace('/', '_')}.json"), dict(pkey, set=self._calib_set_key(), s_c=self.tri_state["s_c"], tau_R=self.calibration["rec"]["tau_R"]),
                                        lambda: calibrate_lambda_q(pm, self.teacher, self._batches(), dev, self.rec_crit, s_c=self.tri_state["s_c"]))
                        if r.get("degenerate"):
                            self._gate_fail("CALIBRATION_DEGENERATE", dict(stage="lambda_q", result=r))
                        self.tri_state["lambda_q"] = float(r["lambda_q"]); cal["lambda_q"] = dict(r, from_cache=hit, pilot=pilot)
                    else:
                        self.tri_state["lambda_q"] = float(lq); cal["lambda_q"] = dict(lambda_q=float(lq), source="config")
            self.calibration["tri"] = dict(cal, state={kk: (v.tolist() if torch.is_tensor(v) else v) for kk, v in self.tri_state.items() if kk != "prec_fn"})
            self.M.train(was_training)
        self._calib.pop("batches", None)
        if self.accelerator.is_main_process:
            json.dump(dict(self.calibration, calibration_set=self._calib.get("set_manifest")), open(os.path.join(self.args.work_dir, "calibration_resolved.json"), "w"), indent=1)

    def _calib_set_key(self):
        c = self.k.get("calibration") or {}
        return dict(dataroot=self.args.train_feeder_args["dataroot"], n=int(c.get("n_patches", 3072)), seed=int(c.get("seed", 1234)), batch=int(self.args.batch_size))

    def _gate_fail(self, status, info):
        json.dump(dict(status=status, **info), open(os.path.join(self.args.work_dir, "gate_fail.json"), "w"), indent=1)
        print(f"[kdv] {status}: {json.dumps(info)[:400]}"); sys.exit(EXIT_GATE)

    # ------------------------------------------------------------------ manifests (§2.2, §21)
    def _write_manifests(self):
        a = self.args; wd = a.work_dir; ds = {}
        for kk in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
            p = getattr(a, kk)["dataroot"]; ds[kk] = dict(path=p, sha256=sha_cached(p, self.init_dir))
            pp = p.replace(".h5", "_pan.h5")
            if os.path.exists(pp):
                ds[kk + "_pan"] = dict(path=pp, sha256=sha_cached(pp, self.init_dir))
        json.dump(ds, open(os.path.join(wd, "dataset_hashes.json"), "w"), indent=1)
        spec0 = dict(self.spec); spec0.update(recipe="NOALIGN", policy="A-ID", rec_case="N0", stat_enabled=False, stat_key="OFF", geom="G0")
        b0 = run_name(spec0, a.seed, self.version, self.prefix); b0d = os.path.join(ROOT, "work_dir", b0)
        json.dump(dict(baseline_run_id=b0, baseline_exists=os.path.isdir(b0d), baseline_finished=os.path.exists(os.path.join(b0d, "finished_at.txt")), init_hashes=self.init_hashes,
                       selection_scene_ids=list(range(20)), report_scene_ids=list(range(20)), fr_h5=self.fr_h5, fr_h5_sha256=self.fr_h5_sha, protocol_id=PROTOCOL_ID),
                  open(os.path.join(wd, "baseline_manifest.json"), "w"), indent=1)

    def _architecture_manifest(self):
        M = self.M; bb = M.backbone; ma = self.args.model_args
        gn = [(n, m.num_groups, m.num_channels) for n, m in bb.named_modules() if isinstance(m, torch.nn.GroupNorm)]
        norms = {}
        for n, m in bb.named_modules():
            norms[type(m).__name__] = norms.get(type(m).__name__, 0) + (1 if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d)) else 0)
        convs = [(n, m.in_channels, m.out_channels) for n, m in bb.named_modules() if isinstance(m, torch.nn.Conv2d)]
        widths = sorted({c for _, _, c in convs})
        n_all = sum(p.numel() for p in M.parameters()); n_tr = sum(p.numel() for p in M.parameters() if p.requires_grad)
        return dict(backbone_class=f"{type(bb).__module__}.{type(bb).__name__}", source_file=os.path.relpath(sys.modules[type(bb).__module__].__file__, ROOT),
                    width=ma.get("hidden_size"), depth=ma.get("depth"), in_channels_effective=(1 + int(self.args.num_bands)), out_channels=ma.get("out_channels"),
                    ms_base="added exactly once in forward (y = bicubic↑MS + backbone residual)", normalization=ma.get("norm"), norm_layer_counts={k: v for k, v in norms.items() if v},
                    groupnorm_layers_backbone=gn[:8], groupnorm_divisibility_ok=all(c % g == 0 for _, g, c in gn), conv_out_widths=widths, n_conv=len(convs),
                    mode_modulation=ma.get("mode_modulation"), attention=ma.get("attn_locations"), pan_auxiliary_forward_count=0,
                    aligner=(None if M.aligner is None else dict(cls=type(M.aligner).__name__, params=sum(p.numel() for p in M.aligner.parameters()), view_margin_hr=self.aligner_view_margin,
                                                                trainable=self.aligner_trainable, source=("donor" if self.donor_manifest else "scratch"))),
                    sampler=M.sampler, params_total=n_all, params_trainable=n_tr, params_frozen=n_all - n_tr, backbone_params=sum(p.numel() for p in bb.parameters()),
                    teacher=(None if self.teacher is None else dict(params=sum(p.numel() for p in self.teacher.parameters()), width=self.teacher_manifest["width"], depth=self.teacher_manifest["depth"])))

    def _write_kdv_manifests(self):
        wd = self.args.work_dir; sp = self.spec
        json.dump(self._architecture_manifest(), open(os.path.join(wd, "architecture_manifest.json"), "w"), indent=1)
        json.dump(dict(init_hashes=self.init_hashes, donor=self.donor_manifest, teacher=self.teacher_manifest, aligner_hash0=self.aligner_hash0,
                       teacher_hash0=getattr(self, "teacher_hash0", None), share_correction=self.share_correction),
                  open(os.path.join(wd, "init_and_teacher_hashes.json"), "w"), indent=1)
        json.dump(dict(campaign_id=self.campaign_id, run_id=self.run_id, run_kind=self.run_kind, version=self.version, spec=sp, description=describe(sp), case=self.case,
                       lambda_V=self.lam_V, stat_ramp_updates=self.stat_ramp, lambda_GKD=self.lam_gkd, k0=self.k0, cov_source=sp["cov_source"], cov_status=(self.cov or {}).get("status"),
                       tri=sp["tri"], tri_state={kk: (v.tolist() if torch.is_tensor(v) else v) for kk, v in self.tri_state.items() if kk != "prec_fn"},
                       aligner_view_margin=self.aligner_view_margin, guard_margin_hr=self.guard_margin,
                       corruption_seed=self.corr_seed, evaluator_hash=evaluator_hash(), protocol_id=PROTOCOL_ID,
                       training=dict(optimizer="AdamW", lr=self.args.learning_rate, weight_decay=self.args.weight_decay, scheduler=self.args.lr_scheduler, warmup=self.args.num_warmup,
                                     batch=self.args.batch_size, updates=self.args.num_iter, amp=str(self.accelerator.mixed_precision), seed=self.args.seed)),
                  open(os.path.join(wd, "kdv_config_resolved.json"), "w"), indent=1)
        json.dump(dict(case=self.case, lambda_edge=self.lam_edge, lambda_geo=self.lam_geo, ramp_steps=self.ramp, geometry_sigma_hr=self.geo_sigma, geometry_margin_hr=self.geo_margin,
                       support_guard_margin_hr=self.guard_margin, warp="bicubic/border/align_corners=False, no zero bypass" if self.M.sampler else "none (A-ID)",
                       loss_domain="L_rec full frame", aligner_params=(sum(p.numel() for p in self.M.aligner.parameters()) if self.M.aligner is not None else 0), evaluator_hash=evaluator_hash()),
                  open(os.path.join(wd, "pa_config_resolved.json"), "w"), indent=1)
        if not self.k.get("phase"):
            yaml.safe_dump(dict(run_kind=self.run_kind, start="from_init", parent_run=None, parent_step=0, inherited_cost_hours=0.0, teacher_id=self.teacher_id), open(os.path.join(wd, "parent_and_phase.yaml"), "w"))
        os.makedirs(os.path.join(CAMPAIGN_ROOT, self.campaign_id), exist_ok=True)
        reg_p = os.path.join(CAMPAIGN_ROOT, self.campaign_id, "teacher_registry.json")
        reg = json.load(open(reg_p)) if os.path.exists(reg_p) else {}
        if self.teacher_manifest:
            reg[self.teacher_id] = dict(run=self.teacher_manifest["run"], tag=self.teacher_manifest["tag"], file_sha256=self.teacher_manifest["file_sha256"], width=self.teacher_manifest["width"],
                                       depth=self.teacher_manifest["depth"], recipe=((self.teacher_manifest.get("kdv") or {}).get("recipe")), seed=self.teacher_manifest.get("seed"), used_by=sorted(set((reg.get(self.teacher_id, {}).get("used_by") or []) + [self.run_id])))
            json.dump(reg, open(reg_p, "w"), indent=1)
        self._runs_csv(status="RUNNING")

    def _runs_csv(self, status, extra=None):
        p = os.path.join(CAMPAIGN_ROOT, self.campaign_id, "runs.csv"); sp = self.spec
        row = dict(run_id=self.run_id, time=time.strftime("%Y-%m-%dT%H:%M:%S"), status=status, recipe=sp["recipe"], input_protocol=sp["protocol"], teacher_id=self.teacher_id or "",
                   W_T=(self.teacher_manifest or {}).get("width", ""), W_S=self.args.model_args.get("hidden_size"), aligner_policy=sp["policy"], rec=sp["rec_case"], stat_kind=sp["stat_key"],
                   stat_mode=sp["stat_mode"] or "", G_mode=sp["geom"], seed=self.args.seed, init_id=self.init_hashes.get("unet_init_sha256_16"), budget_updates=self.args.num_iter,
                   parent=(self.k.get("phase") or {}).get("parent_run", ""), last_step=int(self._global_step), elapsed_h=round((time.time() - self._t_run0) / 3600, 3), lambda_V=self.lam_V, **(extra or {}))
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------ warm start / phase change (§15)
    def _warm_start(self, ph):
        parent, tag, step = ph["parent_run"], ph.get("parent_tag", "last"), int(ph["parent_step"])
        sd, p = load_state(os.path.join("work_dir", parent, tag)); M = self.M
        M.backbone.load_state_dict({kk[len("backbone."):]: v for kk, v in sd.items() if kk.startswith("backbone.")}, strict=True)
        apol = ph.get("aligner_state", "inherit_if_present")
        if M.aligner is not None and apol == "inherit_if_present" and any(kk.startswith("aligner.") for kk in sd):
            M.aligner.load_state_dict({kk[len("aligner."):]: v for kk, v in sd.items() if kk.startswith("aligner.")}, strict=True)
        if M.cov_head is not None and any(kk.startswith("cov_head.") for kk in sd):
            M.cov_head.load_state_dict({kk[len("cov_head."):]: v for kk, v in sd.items() if kk.startswith("cov_head.")}, strict=True)
        opol = ph.get("optimizer_state_policy", "preserve_existing_add_aligner_fresh"); loaded = 0
        op = os.path.join(ROOT, "work_dir", parent, tag, "optimizer.bin")
        if opol != "fresh" and os.path.exists(op):
            osd = torch.load(op, map_location="cpu"); cur = self.optimizer.state_dict()
            nb = len(cur["param_groups"][0]["params"])
            state = {i: s for i, s in osd["state"].items() if int(i) < nb}          # backbone group 은 위치 대응 (같은 module 순서)
            cur["state"] = state; self.optimizer.load_state_dict(cur); loaded = len(state)
        for _ in range(step):
            self.lr_scheduler.step()
        self.start_step = step; self._global_step = step
        pcfg = yaml.safe_load(open(os.path.join(ROOT, "work_dir", parent, "meta", "config.yaml")))
        diff = {kk: dict(old=pcfg.get("kdv", {}).get(kk), new=self.k.get(kk)) for kk in set(pcfg.get("kdv", {})) | set(self.k) if pcfg.get("kdv", {}).get(kk) != self.k.get(kk) and kk != "phase"}
        pcost = os.path.join(ROOT, "work_dir", parent, "memory_and_throughput.json")
        yaml.safe_dump(dict(run_kind="ADAPTIVE_PATH", start="warm_start", parent_run=parent, parent_tag=tag, parent_step=step, parent_checkpoint_sha256=sha256_file(p),
                            config_diff=diff, optimizer_state_policy=opol, optimizer_states_loaded=loaded, scheduler_policy="inherit_full_N_global_step",
                            teacher_id_before=pcfg.get("kdv", {}).get("teacher", {}).get("id"), teacher_id_after=self.teacher_id, reason=ph.get("reason", "REQUIRED_OBSERVATION"),
                            matched_continuation=ph.get("matched_continuation", "REQUIRED_CONTROL_ID"), formal_one_factor_ablation=(len(diff) == 1),
                            inherited_cost_hours=(json.load(open(pcost)).get("elapsed_hours") if os.path.exists(pcost) else None)),
                       open(os.path.join(self.args.work_dir, "parent_and_phase.yaml"), "w"))

    # ------------------------------------------------------------------ 한 update 의 loss (§12)
    def _step(self, gt, ms, lpan, pan, step):
        sp = self.spec; M = self.M; dev = gt.device
        pan_view, eps, corrupted = prepare_view(pan, self.protocol, step, self.radius_hr, self.gen)
        t0 = time.time()
        o = kdv_forward(M, self.teacher, pan_view, ms, lpan, share_correction=self.share_correction, teacher_needed=sp["needs_teacher"], aligner_live=self.aligner_trainable, features=(sp["geom"] == "G5"))
        y, y_t, delta = o["y"], o["y_t"], o["delta"]
        info = dict(corrupt=corrupted, eps=eps, delta=delta, delta_t=o["delta_t"], pan_aligned=o["pan_aligned"], pan_view=pan_view, y=y, y_t=y_t, t_forward=time.time() - t0, _gt=gt, feat_t=o["feat_t"], _ms=ms, _lpan=lpan)
        # reconstruction (§6) — TRI-A/C (addendum §4·§6·§7): parent 의 w_H/w_K 그대로, soft 에 band gate m_A · risk r_C 만 곱한다
        tri = self.tri; tri_rec = tri["enabled"] and (tri["a_mode"] != "off" or (tri["c_mode"] != "off" and tri["c_phi"] == "identity"))
        if y_t is not None and self.rec_crit is not None and tri_rec:
            with torch.autocast(device_type=dev.type, enabled=False):
                r = self.rec_crit(y.float(), y_t.float(), gt.float(), return_maps=True); w_h, w_k = r.maps["hard_weight"], r.maps["soft_weight"]
                yd, ytf, gf = y.detach().float(), y_t.float(), gt.float(); mask = None; risk = None; wk_band = None; J = None
                if tri["a_mode"] in ("sign", "sign_cap", "cosine"):
                    mask = direction_mask(yd, ytf, gf, mode=tri["a_mode"], eps=tri["a_eps"])
                elif tri["a_mode"] == "mass":
                    m0 = direction_mask(yd, ytf, gf, mode="sign", eps=tri["a_eps"]); mask = mass_kappa(w_k, m0).expand_as(m0); info["tri_a_kappa"] = float(mask.flatten()[0])
                elif tri["a_mode"] == "shuffle":
                    mask = shuffle_mask(direction_mask(yd, ytf, gf, mode="sign", eps=tri["a_eps"]), self.tri_gen)
                elif tri["a_mode"] == "band_adv":
                    wk_band = component_weights(yd, ytf, gf, self.tri_state["tau_R_c"], alpha=self.rec_crit.alpha, kd_weight=self.rec_crit.kd_weight, eps=self.rec_crit.eps)
                if tri["c_mode"] != "off" and tri["c_phi"] == "identity":
                    tj = time.time(); J = teacher_fd_jacobian(self.teacher, pan_view, ms, lpan, o["delta_t"], h=tri["c_h"], phi="identity"); info["t_probe"] = time.time() - tj
                    info["_J"] = J
                    if tri["c_mode"] == "sens":
                        risk = sens_risk(sens_q(J), self.tri_state["s_sens"]); info["tri_c_valid_frac"] = 1.0
                    elif tri["c_mode"] in ("diag", "full", "qscalar"):
                        Pi, valid_p, pinfo = self.tri_state["prec_fn"](pan_view, o["ms_base"], gt, o["delta_t"]); Sig, valid_s = sigma_from_precision(Pi, pinfo); valid = valid_p & valid_s
                        Sig = sigma_control(Sig, tri["c_sigma_control"], self.tri_gen); info["_Sig"] = Sig; info["_valid"] = valid; info["tri_c_valid_frac"] = float(valid.double().mean())
                        if tri["c_mode"] == "diag":
                            dr = diag_risk(J, Sig, valid, s_c=self.tri_state["s_c"]); risk = dr["risk"]; info["_q_report"] = dr["q_report"]
                if tri["c_mode"] in ("full", "qiso", "qscalar") and tri["c_phi"] == "identity":
                    d = y.float() - ytf; s_c = self.tri_state["s_c"]
                    if tri["c_mode"] == "qiso":
                        lq = iso_quad(d, s_c=s_c)
                    elif tri["c_mode"] == "full":
                        lq = lowrank_quad(d, J, Sig, s_c=s_c); lq = torch.where(valid[:, None, None, None], lq, iso_quad(d, s_c=s_c))
                    else:
                        lq = scalar_quad(d, J, Sig, s_c=s_c); lq = torch.where(valid[:, None, None, None], lq, iso_quad(d, s_c=s_c))
                    soft = self.tri_state["lambda_q"] * (w_k * lq).mean(); loss_rec = r.hard + soft
                    info.update(rec_hard=float(r.hard), rec_soft=float(soft), rec_soft_original=float(r.soft), tri_gate_mean=1.0)
                elif wk_band is not None:
                    res = masked_l1_componentwise(y.float(), ytf, gf, w_h, wk_band, risk=risk); loss_rec = res["loss"]
                    info.update(rec_hard=float(res["hard"]), rec_soft=float(res["soft"]), rec_soft_original=float(r.soft), tri_gate_mean=float(res["gate_mean"]))
                else:
                    res = masked_l1(y.float(), ytf, gf, w_h, w_k, mask=mask, risk=risk); loss_rec = res["loss"]
                    info.update(rec_hard=float(res["hard"]), rec_soft=float(res["soft"]), rec_soft_original=float(r.soft), tri_gate_mean=float(res["gate_mean"]))
                info.update(**{f"rec_{kk}": float(v) for kk, v in r.stats.items()}); info["_tri_rec"] = (yd, ytf, gf, mask, risk, w_k)
        elif y_t is not None and self.rec_crit is not None:
            r = self.rec_crit(y, y_t, gt); loss_rec = r.loss
            info.update(rec_hard=float(r.hard), rec_soft=float(r.soft), **{f"rec_{kk}": float(v) for kk, v in r.stats.items()})
        else:
            loss_rec = (y - gt).abs().mean(); info.update(rec_hard=float(loss_rec), rec_soft=0.0, rec_plain_gt_l1=float(loss_rec))
        # output statistics (§9)
        loss_stat_raw = torch.zeros((), device=dev); lam_v = 0.0
        if sp["stat_enabled"]:
            with torch.autocast(device_type=dev.type, enabled=False):
                if sp["stat_kind"] == "edge":
                    loss_stat_raw = output_edge_loss(y.float(), gt.float()); info["stat_hard"] = float(loss_stat_raw)
                elif tri["enabled"] and (tri["b_mode"] != "off" or (tri["c_mode"] != "off" and tri["c_phi"] == "stat")):
                    kind, w = sp["stat_kind"], sp["stat_window"]
                    v_s = statistic_map(y.float(), kind, w)
                    with torch.no_grad():
                        v_t = statistic_map(y_t.float(), kind, w); v_g = statistic_map(gt.float(), kind, w)
                    if sp["stat_mode"] == "T":
                        w_hV = torch.zeros(v_s.shape[0], 1, *v_s.shape[-2:], device=dev); w_kV = torch.ones_like(w_hV); s_orig = (v_s.detach() - v_t).abs().mean()
                    else:
                        sr = self.stat_crit(v_s.detach(), v_t, v_g, return_maps=True); w_hV, w_kV = sr.maps["hard_weight"], sr.maps["soft_weight"]; s_orig = sr.soft
                        info.update(**{f"stat_{kk}": float(v) for kk, v in sr.stats.items()})
                    vsd = v_s.detach(); mask_b = None; risk_v = None; wk_comp = None
                    if tri["b_mode"] in ("sign", "sign_cap"):
                        mask_b = direction_mask(vsd, v_t, v_g, mode=tri["b_mode"], eps=float(self.calibration["stat"]["eps_V"]))
                    elif tri["b_mode"] == "mass":
                        m0 = direction_mask(vsd, v_t, v_g, mode="sign", eps=float(self.calibration["stat"]["eps_V"])); mask_b = mass_kappa(w_kV, m0).expand_as(m0); info["tri_b_kappa"] = float(mask_b.flatten()[0])
                    elif tri["b_mode"] == "shuffle":
                        mask_b = shuffle_mask(direction_mask(vsd, v_t, v_g, mode="sign", eps=float(self.calibration["stat"]["eps_V"])), self.tri_gen)
                    elif tri["b_mode"] == "comp_adv":
                        wk_comp = component_weights(vsd, v_t, v_g, self.tri_state["tau_V_j"], alpha=self.stat_crit.alpha, kd_weight=self.stat_crit.kd_weight, eps=float(self.tri_state["eps_V_j"].mean()))
                    if tri["c_mode"] != "off" and tri["c_phi"] == "stat":
                        tj = time.time(); JV = teacher_fd_jacobian(self.teacher, pan_view, ms, lpan, o["delta_t"], h=tri["c_h"], phi="stat", stat_kind=kind, window=w); info["t_probe"] = time.time() - tj; info["_JV"] = JV
                        if tri["c_mode"] == "sens":
                            risk_v = sens_risk(sens_q(JV), self.tri_state["s_sens"]); info["tri_c_valid_frac"] = 1.0
                        elif tri["c_mode"] == "diag":
                            Pi, valid_p, pinfo = self.tri_state["prec_fn"](pan_view, o["ms_base"], gt, o["delta_t"]); Sig, valid_s = sigma_from_precision(Pi, pinfo); valid = valid_p & valid_s
                            Sig = sigma_control(Sig, tri["c_sigma_control"], self.tri_gen); dr = diag_risk(JV, Sig, valid, s_c=self.tri_state["s_c"]); risk_v = dr["risk"]; info["tri_c_valid_frac"] = float(valid.double().mean()); info["_Sig"] = Sig
                        else:
                            raise NotImplementedError("통계 표현의 quadratic C(full/qiso/qscalar) 는 첫 구현 범위 밖 (addendum §6.7: B 결합은 r_C^V 곱)")
                    if wk_comp is not None:
                        res = masked_l1_componentwise(v_s, v_t, v_g, w_hV, wk_comp, risk=risk_v)
                    else:
                        res = masked_l1(v_s, v_t, v_g, w_hV, w_kV, mask=mask_b, risk=risk_v)
                    loss_stat_raw = res["loss"]; info.update(stat_hard=float(res["hard"]), stat_soft=float(res["soft"]), stat_soft_original=float(s_orig), tri_b_gate_mean=float(res["gate_mean"]))
                    info["_tri_stat"] = (vsd, v_t, v_g, mask_b, risk_v, w_kV)
                else:
                    s = stat_term(y.float(), (y_t.float() if y_t is not None else None), gt.float(), kind=sp["stat_kind"], window=sp["stat_window"], mode=sp["stat_mode"], criterion=self.stat_crit)
                    loss_stat_raw = s.loss; info.update(stat_hard=float(s.hard), stat_soft=float(s.soft), **{f"stat_{kk}": float(v) for kk, v in s.stats.items()})
            lam_v = self.lam_V * (min(1.0, (step + 1) / float(self.stat_ramp)) if self.stat_ramp > 0 else 1.0)
        # inherited auxiliary (§10)
        lam_e, lam_g = lambda_ramp(self.lam_edge, step, self.ramp), lambda_ramp(self.lam_geo, step, self.ramp)
        loss_edge = output_edge_loss(y, gt) if self.lam_edge > 0 else torch.zeros((), device=dev)
        loss_geo = torch.zeros((), device=dev); geo_info = None
        if self.lam_geo > 0:
            H, W = pan.shape[-2:]; ok = support_margin_ok(H, W, delta.detach(), self.guard_margin)
            if not bool(ok.all()):
                self._support_fail(step, float(delta.detach().abs().max()), f"geometry support(margin {self.guard_margin})")
            with torch.autocast(device_type=dev.type, enabled=False):
                loss_geo, geo_info = direct_geometry_loss(o["pan_aligned"], pan_view, gt, self.geo_sigma, self.geo_margin)
        loss_off = torch.zeros((), device=dev); lam_off = 0.0
        if corrupted and sp["offset_weight_effective"] > 0:
            if sp["offset_stop_reference"]:
                with torch.no_grad():
                    c0 = predict_c(M.aligner, pan, o["ms_base"], self.aligner_view_margin)
            else:
                c0 = predict_c(M.aligner, pan, o["ms_base"], self.aligner_view_margin)
            loss_off = offset_loss(delta, c0, eps, stop_reference=sp["offset_stop_reference"]); lam_off = lambda_off(step, sp["offset_weight_effective"], self.ramp)
            info["closure"] = float((delta.detach() + eps - c0.detach()).norm(dim=1).mean())
        # alignment KD G1–G5 / G-STRUCT (§11.5–11.7, native step · trainable aligner 만; Π_T 는 frozen Teacher 에서 no_grad)
        loss_gkd = torch.zeros((), device=dev); lam_gkd = 0.0; gk = {}
        if sp["geom"] != "G0" and y_t is not None and not corrupted:
            with torch.no_grad():
                Pi, valid, pinfo = self.prec_fn(pan_view, o["ms_base"], gt, o["delta_t"])
            if sp["geom"] == "G5":
                Sig_s = M.cov_head(o["feat"])
                with torch.no_grad():
                    Sig_t = self.cov_head_t(o["feat_t"])
                    valid = valid & (pinfo["rank"] >= 2) if "rank" in pinfo else valid       # §11.3-3: rank 부족 sample 은 분포 KD 제외 (비율 기록)
                kl = gaussian_kl(delta, Sig_s, o["delta_t"], Sig_t) * valid.double()
                loss_gkd = (kl.sum() / kl.shape[0]).float(); q = kl.detach()
                gk.update(gkd_logdet_s=float(torch.logdet(Sig_s.detach()).mean()), gkd_logdet_t=float(torch.logdet(Sig_t).mean()), gkd_rank_excluded_frac=(float((pinfo["rank"] < 2).double().mean()) if "rank" in pinfo else 0.0))
            else:
                loss_gkd_d, q = mean_kd_loss(delta, o["delta_t"], kd_matrix(Pi, sp["geom"], self.k0 or 1.0), valid); loss_gkd = loss_gkd_d.float()
            lam_gkd = self.lam_gkd
            gk.update(gkd_q_mean=float(q.mean()), gkd_valid_frac=float(valid.double().mean()), gkd_pi_trace_median=float((Pi.diagonal(dim1=1, dim2=2).sum(1) / 2).median()),
                      gkd_rank_deficient_frac=(float((pinfo["rank"] < 2).double().mean()) if "rank" in pinfo else 0.0), gkd_cap_frac=(float(pinfo["cap_hit"].double().mean()) if "cap_hit" in pinfo else 0.0))
            if "closure_norm" in pinfo:
                gk["gkd_teacher_closure"] = float(pinfo["closure_norm"].mean())
            info["_Pi"] = Pi; info["_pinfo"] = pinfo; info["_valid"] = valid
        if sp["geom"] != "G0":
            gk["gkd_applied"] = float((y_t is not None) and (not corrupted))                     # §12: native-only 적용 빈도 (EMA) — 2× 보상 없음
        loss_aux = lam_e * loss_edge + lam_g * loss_geo + lam_off * loss_off + lam_gkd * loss_gkd
        info.update(gk)
        total = loss_rec + lam_v * loss_stat_raw + loss_aux
        info.update(loss_rec=loss_rec, loss_stat_raw=loss_stat_raw, lam_v=lam_v, loss_edge=float(loss_edge), loss_geo=float(loss_geo), loss_off=float(loss_off), loss_gkd=float(loss_gkd),
                    lam_e=lam_e, lam_g=lam_g, lam_off=lam_off, lam_gkd=lam_gkd, loss_aux=loss_aux, geo_info=geo_info)
        return total, info

    def _support_fail(self, step, mx, why):
        self.train_log_ref.write(f'[SUPPORT_FAIL] step {step}: |Δ| max {mx:.2f} px — {why}')
        json.dump(dict(status="SUPPORT_FAIL", step=int(step), max_abs_delta=mx, why=why), open(os.path.join(self.args.work_dir, "support_fail.json"), "w"))
        self._runs_csv("SUPPORT_FAIL"); sys.exit(EXIT_SUPPORT_FAIL)

    # ------------------------------------------------------------------ gradient 진단 (§16.6, §21.2)
    def _gnorm(self, loss, params):
        if not params or not loss.requires_grad:
            return None, 0.0
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        v = torch.cat([x.float().flatten() if x is not None else torch.zeros_like(p).flatten() for x, p in zip(g, params)])
        return v, float(v.norm())

    def _diagnose(self, step, info, M):
        bp = [p for p in M.backbone.parameters() if p.requires_grad]; ap = list(M.aligner.parameters()) if self.aligner_trainable else []
        out = dict(step=int(step), corrupt=bool(info["corrupt"]))
        v_rec, out["grad_rec_F"] = self._gnorm(info["loss_rec"], bp)
        if self.spec["stat_enabled"]:
            v_st, n_st = self._gnorm(info["loss_stat_raw"], bp); out["grad_stat_F_raw"] = n_st; out["grad_stat_F_weighted"] = n_st * info["lam_v"]
            out["cos_rec_stat_F"] = (float((v_rec * v_st).sum() / (v_rec.norm() * v_st.norm())) if v_rec is not None and v_st is not None and n_st > 0 and out["grad_rec_F"] > 0 else None)
        if ap:
            _, out["grad_rec_A"] = self._gnorm(info["loss_rec"], ap)
            if self.spec["stat_enabled"]:
                _, out["grad_stat_A_raw"] = self._gnorm(info["loss_stat_raw"], ap)
            if info["loss_aux"].requires_grad:
                _, out["grad_aux_A"] = self._gnorm(info["loss_aux"], ap)
            if info["delta"].requires_grad:
                g = torch.autograd.grad(info["loss_rec"], info["delta"], retain_graph=True, allow_unused=True)[0]
                out["dLrec_dDelta_rms"] = (float(g.norm(dim=1).pow(2).mean().sqrt()) if g is not None else 0.0)
        if "_tri_rec" in info or "_tri_stat" in info:
            rc = dict(step=int(step), corrupt=bool(info["corrupt"]))
            for name, key in (("A", "_tri_rec"), ("B", "_tri_stat")):
                if key in info:
                    sd_, td_, gd_, mask_, risk_, wk_ = info[key]
                    m_ = mask_ if mask_ is not None else torch.ones_like(sd_)
                    rc[name] = routing_stats(sd_, td_, gd_, m_, wk_)
                    rc[name]["soft_zero_due_to_aT"] = float((wk_ <= 0).double().mean()); rc[name]["soft_zero_due_to_mask"] = float(((wk_ > 0) & (m_ <= 0).all(1, keepdim=True)).double().mean())
                    if risk_ is not None:
                        q = risk_.detach(); rc[name]["risk_mean"] = float(q.mean()); rc[name]["risk_p10"] = float(torch.quantile(q.flatten()[:2_000_000], 0.1)); rc[name]["risk_p50"] = float(torch.quantile(q.flatten()[:2_000_000], 0.5))
            rc["rec_soft_original"] = info.get("rec_soft_original"); rc["rec_soft_masked"] = info.get("rec_soft"); rc["stat_soft_original"] = info.get("stat_soft_original"); rc["stat_soft_masked"] = info.get("stat_soft")
            self._jsonl("routing_components.jsonl", rc)
        J_ = info.get("_J", info.get("_JV"))
        if J_ is not None:
            tri = self.tri; phi = tri["c_phi"]; kind, w = self.spec["stat_kind"], self.spec["stat_window"]
            ev = lambda d: teacher_eval(self.teacher, info["pan_view"], info["_ms"], info["_lpan"], d, phi=phi, stat_kind=(kind if phi == "stat" else None), window=w or 5)
            J2 = teacher_fd_jacobian(self.teacher, info["pan_view"], info["_ms"], info["_lpan"], info["delta_t"], h=tri["c_h"] / 2, phi=phi, stat_kind=(kind if phi == "stat" else None), window=w or 5)
            lin = linearization_check(ev, info["delta_t"], J_, torch.tensor([[0.1, -0.05]], device=J_.device).expand(J_.shape[0], 2))
            fd = dict(step=int(step), h=tri["c_h"], J_rms=float(J_.pow(2).mean().sqrt()), J_rms_dy=float(J_[:, :, 0].pow(2).mean().sqrt()), J_rms_dx=float(J_[:, :, 1].pow(2).mean().sqrt()),
                      rel_half=float((J2 - J_).norm() / (J_.norm() + 1e-30)), finite=bool(torch.isfinite(J_).all()), t_probe_s=info.get("t_probe"), valid_fraction=info.get("tri_c_valid_frac"), **lin)
            if "_q_report" in info:
                q = info["_q_report"]; q = q[torch.isfinite(q)]
                if q.numel():
                    fd.update(q_p50=float(torch.quantile(q.flatten()[:2_000_000], 0.5)), q_p90=float(torch.quantile(q.flatten()[:2_000_000], 0.9)), s_c=self.tri_state.get("s_c"))
            if "_Sig" in info:
                ev_ = torch.linalg.eigvalsh(info["_Sig"]); fd.update(sigma_eval_min_median=float(ev_[:, 0].median()), sigma_eval_max_median=float(ev_[:, 1].median()))
            self._jsonl("fd_and_linearity_checks.jsonl", fd)
        if "_Pi" in info:
            cd = dict(step=int(step), source=self.spec["cov_source"], mode=self.spec["geom"], **precision_stats(info["_Pi"], info["_pinfo"]), valid_fraction=float(info["_valid"].double().mean()),
                      q_mean=info.get("gkd_q_mean"), lambda_GKD=self.lam_gkd, k0=self.k0, delta_drift=float((info["delta"].detach() - info["delta_t"]).norm(dim=1).mean()))
            if self.spec["cov_source"] == "geo_curvature":                # C01 을 학습 중에도 한 번씩: FD Jᵀr ↔ autograd
                ga = geo_autograd_grad(info["pan_view"], info.get("_gt"), info["delta_t"], self.geo_sigma, self.geo_margin) if info.get("_gt") is not None else None
                if ga is not None:
                    cd["fd_vs_autograd_rel_err"] = float((info["_pinfo"]["g_fd"] - ga).norm() / (ga.norm() + 1e-30))
            if self.spec["cov_source"] == "eq_closure":
                cd["teacher_closure_norm_mean"] = float(info["_pinfo"]["closure_norm"].mean()); cd["teacher_bias_norm_mean"] = float(info["_pinfo"]["bias"].norm(dim=1).mean())
            if self.spec["geom"] == "G5":
                with torch.no_grad():
                    Sig_t = self.cov_head_t(info["feat_t"]); Sig_src = torch.linalg.inv(info["_Pi"] + 1e-9 * torch.eye(2, dtype=torch.float64, device=info["_Pi"].device))
                    cd["kl_teacher_head_vs_source"] = float((gaussian_kl(info["delta_t"], Sig_t, info["delta_t"], Sig_src) * info["_valid"].double()).sum() / max(1, int(info["_valid"].sum())))
                    cd["logdet_sigma_s_mean"] = info.get("gkd_logdet_s"); cd["logdet_sigma_t_mean"] = info.get("gkd_logdet_t")
            self._jsonl("covariance_diagnostics.jsonl", cd)
        out["delta_mean"] = info["delta"].detach().mean(0).tolist(); out["delta_std"] = info["delta"].detach().std(0).tolist()
        if info["delta_t"] is not None:
            out["delta_teacher_mean"] = info["delta_t"].mean(0).tolist(); out["delta_drift_vs_teacher"] = float((info["delta"].detach() - info["delta_t"]).norm(dim=1).mean())
        self._jsonl("gradient_diagnostics.jsonl", out); self._ema["diag_last"] = step

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.train_log_ref = train_log
        self.model.train(); self.M.backbone.requires_grad_(True)
        if self.M.aligner is not None:
            self.M.aligner.requires_grad_(self.aligner_trainable)
        if self.M.cov_head is not None:
            self.M.cov_head.requires_grad_(True)                          # 검토 지적: eval 의 requires_grad_(False) 뒤 재활성 (G5)
        if global_step >= self.args.num_iter:
            return global_step
        report = Train_Report(); start = time.time()
        B = self.args.batch_size; dev, dt = self.accelerator.device, self.weight_dtype; M = self.M
        times = dict(native=[], corrupt=[], forward=[])
        for idx, (gt, lms, ms, lpan, pan) in enumerate(self.train_data_loader):
            t0 = time.time()
            with self.accelerator.accumulate(self.model):
                gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (gt, ms, lpan, pan))
                total, info = self._step(gt, ms, lpan, pan, global_step)
                if not torch.isfinite(total):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {total.item()}'); self._runs_csv("NAN"); sys.exit(3)
                dmax = float(info["delta"].detach().abs().max())
                if dmax > MAX_ABS_DELTA_TRAIN:
                    self._support_fail(global_step, dmax, f"|Δ| > {MAX_ABS_DELTA_TRAIN} (학습 patch 에서 정의 불가능한 보정)")
                if self.accelerator.is_main_process and global_step % self.diag_every == 0:
                    self._diagnose(global_step, info, M)
                self.accelerator.backward(total)
                if self.accelerator.is_main_process and global_step % self.diag_every == 0:
                    self._ema["backbone_grad_norm"] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.backbone.parameters() if p.grad is not None)))
                    self._ema["aligner_grad_norm"] = (float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.aligner.parameters() if p.grad is not None))) if self.aligner_trainable else 0.0)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    report.update(B, total.item(), float(info["loss_rec"]), float(info["lam_v"] * float(info["loss_stat_raw"]) + float(info["loss_aux"])))
                    with torch.no_grad():
                        d = info["delta"].detach().float(); key = "corrupt" if info["corrupt"] else "native"
                        vals = dict(loss_rec=float(info["loss_rec"]), stat_raw=float(info["loss_stat_raw"]), lam_v=info["lam_v"],
                                    loss_edge=info["loss_edge"], loss_geo=info["loss_geo"], loss_off=info["loss_off"], loss_gkd=info["loss_gkd"], lam_e=info["lam_e"], lam_g=info["lam_g"],
                                    lam_off=info["lam_off"], lam_gkd=info["lam_gkd"], aux_ratio=float(info["loss_aux"]) / (float(info["loss_rec"]) + 1e-12),
                                    **{f"{key}_dy": d[:, 0].mean().item(), f"{key}_dx": d[:, 1].mean().item(), f"{key}_dnorm_p50": d.norm(dim=1).median().item(), "abs_max": d.abs().max().item()},
                                    **{kk: v for kk, v in info.items() if kk.startswith(("rec_", "stat_", "gkd_", "tri_")) and isinstance(v, float)}, t_probe=float(info.get("t_probe", 0.0)))
                        if info["delta_t"] is not None:
                            vals["delta_drift"] = float((d - info["delta_t"].float()).norm(dim=1).mean())
                        if "closure" in info:
                            vals["closure"] = info["closure"]
                        for kk, v in vals.items():
                            self._ema_update(kk, v)
                        if global_step % 50 == 0 and M.sampler:
                            H, W = pan.shape[-2:]
                            self._ema_update("valid_frac", warp_support_mask(H, W, d).float().mean().item(), 0.9)
                        if info["geo_info"] is not None:
                            self._ema_update("geo_wsum", info["geo_info"]["weight_sum"])
            times["corrupt" if info["corrupt"] else "native"].append(time.time() - t0); times["forward"].append(info["t_forward"])
            global_step += 1; self._global_step = global_step
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']; e = self._ema
                extra = (f"\t[{self.case}] rec {e.get('loss_rec', 0):.5f} (hard {e.get('rec_hard', 0):.5f} soft {e.get('rec_soft', 0):.5f}) d {e.get('rec_difficulty_mean', 0):.3f} a {e.get('rec_advantage_mean', 0):.3f} "
                         f"win {e.get('rec_student_better_fraction', 0):.3f} softpos {e.get('rec_soft_positive_fraction', 0):.3f}"
                         f"\tstat {e.get('stat_raw', 0):.4g}×λ{e.get('lam_v', 0):.3g} (d {e.get('stat_difficulty_mean', 0):.3f} win {e.get('stat_student_better_fraction', 0):.3f})"
                         f"\taux edge {e.get('loss_edge', 0):.4f} geo {e.get('loss_geo', 0):.4f} off {e.get('loss_off', 0):.4f} gkd {e.get('loss_gkd', 0):.4g}×λ{self.lam_gkd:.3g} (q {e.get('gkd_q_mean', 0):.3g} valid {e.get('gkd_valid_frac', 0):.2f} trΠ/2 {e.get('gkd_pi_trace_median', 0):.3g}) aux/rec {e.get('aux_ratio', 0):.3f}"
                         f"\tΔ nat ({e.get('native_dy', 0):+.3f},{e.get('native_dx', 0):+.3f}) |Δ| {e.get('native_dnorm_p50', 0):.3f} max {e.get('abs_max', 0):.3f} drift {e.get('delta_drift', 0):.3f}"
                         + (f"\tTRI gateA {e.get('tri_gate_mean', 0):.3f} gateB {e.get('tri_b_gate_mean', 0):.3f} soft {e.get('rec_soft_original', 0):.2e}→{e.get('rec_soft', 0):.2e} Cvalid {e.get('tri_c_valid_frac', 0):.2f} probe {e.get('t_probe', 0):.3f}s" if self.tri["enabled"] else "")
                         + f"\tgrad bb {e.get('backbone_grad_norm', 0):.3g} al {e.get('aligner_grad_norm', 0):.3g}"
                         f"\tt nat {np.mean(times['native'][-50:]) if times['native'] else 0:.3f}s cor {np.mean(times['corrupt'][-50:]) if times['corrupt'] else 0:.3f}s")
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start) + extra)
                self._jsonl("train_log.jsonl", dict(step=int(global_step), lr=lr, **{kk: (float(v) if isinstance(v, (int, float)) else v) for kk, v in e.items() if not kk.startswith("_")},
                                                     t_native_s=(float(np.mean(times["native"][-200:])) if times["native"] else None), t_corrupt_s=(float(np.mean(times["corrupt"][-200:])) if times["corrupt"] else None),
                                                     t_forward_s=float(np.mean(times["forward"][-200:])), peak_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None)))
                if self.accelerator.is_main_process:
                    json.dump(dict(step=int(global_step), t_native_s=(float(np.mean(times["native"][-200:])) if times["native"] else None), t_corrupt_s=(float(np.mean(times["corrupt"][-200:])) if times["corrupt"] else None),
                                   t_forward_s=float(np.mean(times["forward"][-200:])), peak_mem_allocated_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None),
                                   peak_mem_reserved_mb=(torch.cuda.max_memory_reserved() / 2 ** 20 if torch.cuda.is_available() else None), elapsed_hours=(time.time() - self._t_run0) / 3600.0,
                                   teacher_forward=bool(self.spec["needs_teacher"]), share_correction=self.share_correction, batch=B),
                              open(os.path.join(self.args.work_dir, "memory_and_throughput.json"), "w"), indent=1)
                start = time.time(); report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                if report.num_examples > 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start))
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'last'))
                json.dump(dict(step=int(global_step), kind="last"), open(os.path.join(self.args.work_dir, "last_meta.json"), "w"))
                if self.accelerator.is_main_process:
                    self._runs_csv("FINISHED_TRAIN")
                self.accelerator.end_training()
                return global_step
        return global_step

    # ------------------------------------------------------------------ eval: RR + fitting bins (§19.2) + RR-val (§18.1)
    def _check_fixed(self):
        if self.teacher is not None and state_hash(self.teacher) != self.teacher_hash0:
            raise RuntimeError("Teacher state 가 바뀌었다 (gate M03)")
        if self.spec["policy"] == "A-FR" and state_hash(self.M.aligner) != self.aligner_hash0:
            raise RuntimeError("frozen aligner(A-FR) state 가 바뀌었다 (gate L03)")
        if self.cov_head_t is not None and state_hash(self.cov_head_t) != self.cov_head_t_hash:
            raise RuntimeError("Teacher covariance head 가 바뀌었다 (gate M03)")

    def test_reduced(self, test_log, epoch):
        self._check_fixed()
        report = Test_Reduced_Report(); rep_v = Test_Reduced_Report(); self.model.eval(); self.model.requires_grad_(False)
        ds, eT, eS, vT, vS = [], [], [], [], []
        sp = self.spec
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            o = self._infer(pan, lpan, ms)
            gt = gt.to(self.accelerator.device, dtype=self.weight_dtype)
            self.save_test_reduced(o["pan"], gt, o["y"], o["ms_base"], idx)
            report.update(self.args.test_batch_size, reduced_metrics(x_true=gt, x_pred=o["y"], max_pixel=self.args.max_pixel))
            H, W = gt.shape[-2:]
            if H > 2 * fixed_roi(512, 512)[0]:
                y0, y1, x0, x1 = fixed_roi(H, W)
                rep_v.update(self.args.test_batch_size, reduced_metrics(x_true=gt[..., y0:y1, x0:x1], x_pred=o["y"][..., y0:y1, x0:x1], max_pixel=self.args.max_pixel))
            ds.append(o["delta"][0].float().cpu().numpy())
            if self.teacher is not None:
                with torch.no_grad():
                    y_t = self.teacher(o["pan"], o["ms"], lpan.to(self.accelerator.device, dtype=self.weight_dtype))["y"]
                    eT.append((y_t - gt).abs().mean(1).flatten().float().cpu()); eS.append((o["y"] - gt).abs().mean(1).flatten().float().cpu())
                    if sp["stat_enabled"] and sp["stat_kind"] != "edge":
                        vg = statistic_map(gt.float(), sp["stat_kind"], sp["stat_window"])
                        vT.append((statistic_map(y_t.float(), sp["stat_kind"], sp["stat_window"]) - vg).abs().mean(1).flatten().cpu())
                        vS.append((statistic_map(o["y"].float(), sp["stat_kind"], sp["stat_window"]) - vg).abs().mean(1).flatten().cpu())
        d = np.array(ds)
        rr_val = self._rr_val()
        test_log.write(f'Epoch[{epoch}]\t' + report.result_str() + f'\tRR Δ median ({np.median(d[:, 0]):+.3f},{np.median(d[:, 1]):+.3f}) |Δ| median {np.median(np.linalg.norm(d, axis=1)):.3f}'
                       + (f'\t[RR-valid] ' + rep_v.result_str() if rep_v.num_examples else '') + f'\t[RR-val] ERGAS(plain) {rr_val:.4f}')
        self.last_reduced_metrics = report.as_dict()
        self.last_reduced_metrics.update(rr_dy_median=float(np.median(d[:, 0])), rr_dx_median=float(np.median(d[:, 1])), rr_val_ergas=rr_val)
        if rep_v.num_examples:
            self.last_reduced_metrics.update({f"valid_{kk}": v for kk, v in rep_v.as_dict().items()})
        if eT and self.accelerator.is_main_process:
            self._fitting_bins(self._global_step, epoch, torch.cat(eT), torch.cat(eS), "pixel")
            if vT:
                self._fitting_bins(self._global_step, epoch, torch.cat(vT), torch.cat(vS), f"stat_{sp['stat_kind']}")
        return report.ergas

    def _fitting_bins(self, step, epoch, e_t, e_s, kind):
        q50, q90 = torch.quantile(e_t, torch.tensor([0.5, 0.9])).tolist()
        rows = []
        for name, m in (("Q0-50", e_t <= q50), ("Q50-90", (e_t > q50) & (e_t <= q90)), ("Q90-100", e_t > q90), ("ALL", torch.ones_like(e_t, dtype=torch.bool))):
            a, b = e_t[m], e_s[m]
            rows.append(dict(step=int(step), epoch=int(epoch), kind=kind, bin=name, n=int(m.sum()), e_T_mean=float(a.mean()), e_S_mean=float(b.mean()), delta_e=float((a - b).mean()),
                             win_rate=float((b < a).float().mean()), q50=q50, q90=q90))
        self._csv_append("fitting_bins.csv", rows)

    @torch.no_grad()
    def _rr_val(self):
        """검증셋(valid_*.h5) plain ERGAS 평균 — best_rr_val 선택 전용 (§18.1)."""
        tot, n = 0.0, 0; dev, dt = self.accelerator.device, self.weight_dtype; mp = float(self.args.max_pixel)
        for gt, lms, ms, lpan, pan in self.val_data_loader:
            o = self._infer(pan, lpan, ms); gt = gt.to(dev)
            pred = ((o["y"].clip(-1, 1).double() + 1) / 2 * mp); g = ((gt.double() + 1) / 2 * mp)
            e = ergas_per_sample(g, pred); tot += float(e.sum()); n += int(e.numel())
        self._rr_val_last = tot / max(1, n)
        return self._rr_val_last

    def test_full(self, test_log, epoch):
        self._check_fixed()
        return super().test_full(test_log, epoch)

    def _record(self, step, epoch, rows, agg, D, all_ok, n_bad, bad_reasons):
        self._csv_append("scene_metrics.csv", rows)
        rec = dict(step=step, epoch=epoch)
        for v in VIEWS:
            for kk in ("hqnr", "d_lambda", "d_s", "fscc"):
                rec[f"{v}.{kk}"] = agg[v][kk]
        rec.update(dy_median=float(np.median(D[:, 0])), dx_median=float(np.median(D[:, 1])), delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))),
                   delta_abs_max=float(np.abs(D).max()), aligned_eligible=int(all_ok), n_invalid_scenes=n_bad, invalid_reasons="|".join(bad_reasons),
                   region_effect=agg["raw_valid"]["hqnr"] - agg["raw_original"]["hqnr"], reference_effect=agg["aligned_valid"]["hqnr"] - agg["raw_valid"]["hqnr"],
                   rr_ergas=self.last_reduced_metrics.get("ergas"), rr_scc=self.last_reduced_metrics.get("scc"), rr_sam=self.last_reduced_metrics.get("sam"),
                   rr_valid_ergas=self.last_reduced_metrics.get("valid_ergas"), rr_valid_scc=self.last_reduced_metrics.get("valid_scc"), rr_val_ergas=self._rr_val_last,
                   lambda_V=self.lam_V, rec_difficulty_ema=self._ema.get("rec_difficulty_mean"), rec_student_win_ema=self._ema.get("rec_student_better_fraction"))
        self._csv_append("checkpoint_metrics.csv", [rec])
        self._csv_append("delta_predictions.csv", [dict(step=step, scene=i, dy_hr=float(D[i, 0]), dx_hr=float(D[i, 1])) for i in range(len(D))])

    def _select(self, step, epoch, agg, all_ok, n_bad, bad_reasons, test_log):
        cand = os.path.join(self.cand_dir, f"step-{step}")
        self.accelerator.save_state(cand)
        r_raw = self.sel_raw.update(step, epoch, agg["raw_original"]["hqnr"], agg["raw_original"]["fscc"], True, cand)
        r_al = self.sel_aligned.update(step, epoch, agg["aligned_valid"]["hqnr"], agg["aligned_valid"]["fscc"], all_ok, cand, reason=("" if all_ok else f"{n_bad} scenes: {bad_reasons}"))
        r_rr = self.sel_rrval.update(step, epoch, -self._rr_val_last, 0.0, bool(np.isfinite(self._rr_val_last)), cand)
        keep = {c["path"] for s in (self.sel_raw, self.sel_aligned, self.sel_rrval) for c in s.cands}
        for p in set(r_raw["pruned"]) | set(r_al["pruned"]) | set(r_rr["pruned"]) | ({cand} if cand not in keep else set()):
            if p not in keep and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        self.raw_is_best = r_raw["changed"]
        if r_al["changed"]:
            self._materialize(self.sel_aligned.best, "best_aligned", "aligned_valid")
        if r_rr["changed"]:
            self._materialize(self.sel_rrval.best, "best_rr_val", "rr_val")
        extra = dict(protocol_id=PROTOCOL_ID, roi_hash=(self._roi or {}).get("roi_hash"), evaluator_hash=evaluator_hash(), fr_h5_sha256=self.fr_h5_sha, n_scenes=len(self._fr_pan))
        self.sel_raw.save(os.path.join(self.args.work_dir, "selector_state_raw.json"), extra)
        self.sel_aligned.save(os.path.join(self.args.work_dir, "selector_state_aligned.json"), extra)
        self.sel_rrval.save(os.path.join(self.args.work_dir, "selector_state_rr_val.json"), dict(protocol_id=PROTOCOL_ID, metric="-ERGAS(plain, valid h5)"))
        if self.sel_aligned.best is None:
            json.dump(dict(status="no_valid_candidate", history=self.sel_aligned.history[-5:]), open(os.path.join(self.args.work_dir, "best_aligned_meta.json"), "w"), indent=1)
        b = self.sel_raw.best; a = self.sel_aligned.best; r = self.sel_rrval.best
        test_log.write(f'[select] best_raw step {b["step"]} (HQNR {b["hqnr"]:.6f} fSCC {b["fscc"]:.4f}, anchor {self.sel_raw.max_hqnr:.6f}, band {len(self.sel_raw.cands)})'
                       + (f' | best_aligned step {a["step"]} (HQNR_al {a["hqnr"]:.6f})' if a else ' | best_aligned: no valid candidate')
                       + (f' | best_rr_val step {r["step"]} (ERGAS_val {-r["hqnr"]:.4f})' if r else ''))
        if self.accelerator.is_main_process:
            self._runs_csv("EVAL", dict(best_raw_step=b["step"], best_raw_hqnr=round(b["hqnr"], 6)))

    def export_tags(self):
        return ["best_hqnr", "best_aligned", "best_rr_val", "last"]
