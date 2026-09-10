"""OffsetConsistencyTrainer — PO10: PAN 추가 변위 + offset consistency (research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md).

A1 과 같은 골격·aligner·init·optimizer 위에서
  update 짝수: native (ĉ0 = A^V(P,M), P̃0 = W(P,ĉ0), L_rec)
  update 홀수: corrupted (ε ~ disk(R), P_ε = W(P,ε), ĉ0 · ĉε, P̃ε = W(P_ε,ĉε), L_rec + λ_off(t)·|ĉε + ε − ref|)
  N1: λ_off = 0 (진단값만) · N2_SG: ref = sg(ĉ0) · N3_NOSG: ref = ĉ0 (양쪽 gradient)
aligner 는 고정 내부 crop(m_A = 4·ceil((R+2)/4), R=1 → 4) 만 본다 — 학습·평가 모두 (§5). U-Net·GT·MS base 는 전체 프레임.
평가·선택·산출물은 train_pa.PATrainer(세 view · best_raw=best_hqnr alias · best_aligned · last)를 그대로 쓴다.
N3 는 예산 gate(§11.2)를 통과할 때만 시작한다 — 불통과면 DEFERRED_BUDGET 을 남기고 exit 4 (체인은 재시도하지 않는다).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from pa.offset import CASES as PO_CASES, aligner_margin, po_step, predict_c, sample_offsets, lambda_off
from pa.warp import warp_support_mask
from train_pa import PATrainer, ROOT, EXIT_SUPPORT_FAIL, sha256_file
from utils import Train_Report

EXIT_BUDGET = 4
LEDGER = os.environ.get("PO10_LEDGER", os.path.join(ROOT, "work_dir", "_po10_budget", "ledger.json"))     # 테스트는 PO10_LEDGER 로 격리
MAX_ABS_C_TRAIN = 8.0                         # 학습 patch 에서 |ĉ| 가 이보다 크면 정의 불가능 crop/발산으로 본다 (§6.3 명시적 failure)


def _ledger_load():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else dict(total_gpu_hours=10.0, entries={})


def _ledger_save(d):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True); json.dump(d, open(LEDGER, "w"), indent=1)


def budget_used_hours(d):
    return sum(float(e.get("hours", 0.0)) for e in d["entries"].values())


class OffsetConsistencyTrainer(PATrainer):
    def _configure(self, args):
        po = getattr(args, "po", {}) or {}
        self.case = po.get("case"); assert self.case in PO_CASES, f"po.case 는 {list(PO_CASES)}"
        self.radius_hr = float(po.get("radius_hr", 1.0)); assert self.radius_hr > 0
        self.radius_source = po.get("radius_source", "audit appendix E WV3 train global |δ| P90 0.250 LR px × 4")
        self.lam_off_max = 0.0 if self.case == "N1" else float(po.get("lambda_off_max", 0.01))
        self.ramp = int(po.get("ramp_updates", 5000))
        self.diag_every = int(po.get("diag_every", 1000))
        self.diag_iter = self.diag_every                                     # PATrainer 로그 호환
        self.init_dir = os.path.join(ROOT, po.get("init_dir", "work_dir/_pa_init"))
        self.lam_edge = self.lam_geo = 0.0; self.geo_sigma = 2.0; self.geo_margin = 11; self.guard_margin = 4
        self.aligner_view_margin = aligner_margin(self.radius_hr)            # §5.1
        self.corr_seed = int(args.seed) + int(po.get("corruption_seed_offset", 2000))
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(self.corr_seed)
        self.budget_total = float(po.get("budget_total_gpu_hours", 10.0)); self.budget_reserve = float(po.get("budget_diag_reserve_hours", 1.0))
        self.required = self.case in ("N1", "N2_SG")
        self._fixed_batches = None; self._t_run0 = time.time()

    def __init__(self, args, data_loader, model):
        super().__init__(args, data_loader, model)
        if self.accelerator.is_main_process:
            self._budget_gate()
            self._write_po_manifests()

    # ------------------------------------------------------------------ 예산 gate · manifests
    def _budget_gate(self):
        d = _ledger_load(); d.setdefault("total_gpu_hours", self.budget_total)
        used = budget_used_hours(d)
        rec = dict(case=self.case, started=time.strftime("%Y-%m-%dT%H:%M:%S"), required=self.required)
        if not self.required:
            done = [e for k, e in d["entries"].items() if e.get("kind") == "run" and e.get("finished")]
            proj = float(np.mean([e["hours"] for e in done])) * 1.03 if done else float("nan")     # N3: aligner backward 한 번 더 ≈ +3%
            ok = (not np.isnan(proj)) and (used + 1.2 * proj + self.budget_reserve <= d["total_gpu_hours"])
            rec.update(projected_hours=proj, used_hours_before=used, eligible=bool(ok))
            if not ok:
                rec["status"] = "DEFERRED_BUDGET"; d["entries"][f"{self.args.work_dir.rstrip('/').split('/')[-1]}"] = rec; _ledger_save(d)
                json.dump(rec, open(os.path.join(self.args.work_dir, "budget_status.json"), "w"), indent=1)
                print(f"[po10] {self.case} DEFERRED_BUDGET: used {used:.2f}h + 1.2×{proj:.2f}h + reserve {self.budget_reserve}h > {d['total_gpu_hours']}h")
                sys.exit(EXIT_BUDGET)
        rec.update(kind="run", status="RUNNING", used_hours_before=used)
        d["entries"][self.args.work_dir.rstrip("/").split("/")[-1]] = rec; _ledger_save(d)
        json.dump(rec, open(os.path.join(self.args.work_dir, "budget_status.json"), "w"), indent=1)

    def _finish_ledger(self, status="FINISHED"):
        d = _ledger_load(); k = self.args.work_dir.rstrip("/").split("/")[-1]
        e = d["entries"].get(k, {}); e.update(status=status, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), hours=(time.time() - self._t_run0) / 3600.0)
        d["entries"][k] = e; _ledger_save(d); json.dump(e, open(os.path.join(self.args.work_dir, "budget_status.json"), "w"), indent=1)

    def _write_po_manifests(self):
        wd = self.args.work_dir; seed = int(self.args.seed)
        m = dict(protocol_id="pan_offset_rel_10h_v1", scale_source="research_log/2026-09-03_alignment-audit-s2-detail.md appendix E — WV3 train HR-grid decomposition, global |δ| P90 = 0.250 LR px (n=199)",
                 calibration_quality="train_aggregate_proxy", observed_value_lr=0.250, ratio=4, radius_hr=self.radius_hr, distribution="uniform_disk_area", center_yx=[0, 0],
                 expected_axis_sd=self.radius_hr / 2.0, native_corrupt_step_ratio=[1, 1], rng="dedicated CPU torch.Generator", corruption_seed=self.corr_seed,
                 aligner_view_margin_hr=self.aligner_view_margin, margin_rule="4*ceil((R+2)/4)", pass_epsilon_to_network=False,
                 corrupted_route="sequential_two_raster_warps", native_route="single_learned_warp",
                 cross_check_2026_09_10="results_log/2026-09-10_pa-a1-a3-s1-results.md §7: WV3 train patch GT<-PAN |δ| median 0.17 HR px (floor 0.16), p95 0.76 HR px")
        json.dump(m, open(os.path.join(wd, "corruption_scale_manifest.json"), "w"), indent=1)
        # 첫 1000 ε 의 hash — N1/N2/N3 가 같은 ε 를 받는지 대조 (§7-8). 같은 seed 로 별도 generator 를 돌려 미리 뽑는다 (학습 generator 는 소비하지 않는다)
        g = torch.Generator(device="cpu"); g.manual_seed(self.corr_seed)
        eps = torch.cat([sample_offsets(int(self.args.batch_size), self.radius_hr, g) for _ in range(1000 // int(self.args.batch_size) + 1)], 0)[:1000]
        h = hashlib.sha256(eps.numpy().tobytes()).hexdigest()[:16]
        json.dump(dict(sha256_16=h, n=1000, mean=eps.mean(0).tolist(), std=eps.std(0).tolist(), max_norm=float(eps.norm(dim=1).max())), open(os.path.join(wd, "corruption_first1000.json"), "w"), indent=1)
        a1 = os.path.join(ROOT, "work_dir", f"PA_A1_REC_W96_D124_9CH_S{seed}")
        src = dict(a1_run=a1, a1_config=os.path.join(a1, "meta", "config.yaml") if os.path.exists(os.path.join(a1, "meta", "config.yaml")) else None,
                   a1_best_hqnr_sha256=(sha256_file(os.path.join(a1, "best_hqnr", "model.safetensors")) if os.path.exists(os.path.join(a1, "best_hqnr", "model.safetensors")) else None),
                   init_hashes=self.init_hashes, evaluator_hash=json.load(open(os.path.join(wd, "pa_config_resolved.json")))["evaluator_hash"],
                   training_updates=int(self.args.num_iter), optimizer=dict(type="AdamW", lr=self.args.learning_rate, weight_decay=self.args.weight_decay, scheduler=self.args.lr_scheduler, warmup=self.args.num_warmup, batch=self.args.batch_size, amp=str(self.accelerator.mixed_precision)),
                   corruption_first1000_sha256_16=h, case=self.case, lambda_off_max=self.lam_off_max, ramp_updates=self.ramp)
        json.dump(src, open(os.path.join(wd, "source_and_init_hashes.json"), "w"), indent=1)
        cfg = json.load(open(os.path.join(wd, "pa_config_resolved.json"))); cfg.update(po_case=self.case, radius_hr=self.radius_hr, lambda_off_max=self.lam_off_max, ramp_updates=self.ramp, aligner_view_margin=self.aligner_view_margin)
        json.dump(cfg, open(os.path.join(wd, "pa_config_resolved.json"), "w"), indent=1)

    # ------------------------------------------------------------------ 진단 (§9): 고정 native/corrupt minibatch
    def _grad_norm(self, loss, params, retain=True):
        g = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        flat = [x.float().flatten() if x is not None else torch.zeros_like(p).flatten() for x, p in zip(g, params)]
        v = torch.cat(flat); return v, float(v.norm()), sum(1 for x in g if x is None)

    def _diagnose(self, step, M, dev, dt):
        if self._fixed_batches is None:
            return
        out = dict(step=int(step))
        ap = list(M.aligner.parameters()); head = [M.aligner.fc2.weight, M.aligner.fc2.bias]; stem = list(M.aligner.pan_stem.parameters()) + list(M.aligner.ms_stem.parameters())
        up = [p for p in M.backbone.parameters() if p.requires_grad]
        for kind, idx in (("native", 0), ("corrupt", 1)):
            gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in self._fixed_batches[kind])
            g = torch.Generator(device="cpu"); g.manual_seed(self.corr_seed + 777)            # 진단 ε 는 학습 generator 를 소비하지 않는다
            total, info = po_step(M, pan, ms, lpan, gt, case=self.case, update_index=idx, radius_hr=self.radius_hr, generator=g, lam_max=(self.lam_off_max or 0.01), ramp=self.ramp, margin=self.aligner_view_margin)
            c = info["c"]
            dl_dc = torch.autograd.grad(info["rec"], c, retain_graph=True, allow_unused=True)[0]
            out[f"{kind}.dLrec_dc_norm"] = float(dl_dc.norm()) if dl_dc is not None else 0.0
            out[f"{kind}.dLrec_dc_rms_per_sample"] = float(dl_dc.norm(dim=1).pow(2).mean().sqrt()) if dl_dc is not None else 0.0
            v_rec, n_rec, _ = self._grad_norm(info["rec"], ap)
            out[f"{kind}.grad_rec_phi"] = n_rec
            out[f"{kind}.grad_rec_head"] = float(self._grad_norm(info["rec"], head)[0].norm()); out[f"{kind}.grad_rec_stem"] = float(self._grad_norm(info["rec"], stem)[0].norm())
            if kind == "corrupt":
                v_off, n_off, _ = self._grad_norm(info["off"], ap)
                out["corrupt.grad_off_phi"] = n_off; out["corrupt.grad_off_weighted_phi"] = n_off * float(info["weight"])
                out["corrupt.cos_rec_off_phi"] = (float((v_rec * v_off).sum() / (v_rec.norm() * v_off.norm())) if n_rec > 0 and n_off > 0 else None)
                _, n_off_theta, n_none = self._grad_norm(info["off"], up)
                out["corrupt.grad_off_theta"] = n_off_theta; out["corrupt.grad_off_theta_unused_params"] = n_none
                out["corrupt.c0_requires_grad"] = bool(info["c0"].requires_grad)
                out["corrupt.offset_closure_mean"] = float((info["c"] + info["eps"] - info["c0"]).norm(dim=1).mean())
                out["corrupt.eps_mean_norm"] = float(info["eps"].norm(dim=1).mean())
            out[f"{kind}.c_mean"] = info["c"].detach().mean(0).tolist(); out[f"{kind}.c_std"] = info["c"].detach().std(0).tolist()
            out[f"{kind}.pan_border_use_frac"] = 1.0 - float(warp_support_mask(pan.shape[-2], pan.shape[-1], info["c"].detach()).float().mean())
            del total, info, c, dl_dc, v_rec
        self._jsonl("gradient_diagnostics.jsonl", out)
        self._ema["diag_last"] = step

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.model.train(); self.model.requires_grad_(True)
        report = Train_Report(); start = time.time()
        B = self.args.batch_size; dev, dt = self.accelerator.device, self.weight_dtype
        M = self.M
        t_native, t_corrupt = [], []
        for idx, (gt, lms, ms, lpan, pan) in enumerate(self.train_data_loader):
            t0 = time.time()
            with self.accelerator.accumulate(self.model):
                gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (gt, ms, lpan, pan))
                if self._fixed_batches is None or (global_step == 1 and "corrupt" not in self._fixed_batches):
                    fb = self._fixed_batches or {}
                    fb["native" if global_step % 2 == 0 else "corrupt"] = tuple(t[:min(B, 16)].detach().cpu() for t in (gt, ms, lpan, pan))
                    self._fixed_batches = fb
                if self.accelerator.is_main_process and global_step % self.diag_every == 0 and global_step > 0 and self._fixed_batches and len(self._fixed_batches) == 2:
                    snap = [p.detach().clone() for p in M.aligner.parameters()]
                    self._diagnose(global_step, M, dev, dt)
                total, info = po_step(M, pan, ms, lpan, gt, case=self.case, update_index=global_step, radius_hr=self.radius_hr, generator=self.gen,
                                      lam_max=self.lam_off_max, ramp=self.ramp, margin=self.aligner_view_margin)
                if not torch.isfinite(total):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {total.item()}'); sys.exit(3)
                cmax = float(info["c"].detach().abs().max())
                if cmax > MAX_ABS_C_TRAIN:
                    train_log.write(f'[SUPPORT_FAIL] step {global_step}: |ĉ| max {cmax:.2f} px > {MAX_ABS_C_TRAIN} (학습 patch 에서 정의 불가능한 보정)')
                    json.dump(dict(status="SUPPORT_FAIL", step=int(global_step), max_abs_c=cmax), open(os.path.join(self.args.work_dir, "support_fail.json"), "w"))
                    self._finish_ledger("SUPPORT_FAIL"); sys.exit(EXIT_SUPPORT_FAIL)
                self.accelerator.backward(total)
                if self.accelerator.is_main_process and global_step % self.diag_every == 0 and global_step > 0:
                    an = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.aligner.parameters() if p.grad is not None)))
                    bn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.backbone.parameters() if p.grad is not None)))
                    self._ema["aligner_grad_norm"], self._ema["backbone_grad_norm"] = an, bn
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()
                if self.accelerator.is_main_process and global_step % self.diag_every == 0 and global_step > 0 and self._fixed_batches and len(self._fixed_batches) == 2:
                    with torch.no_grad():
                        upd = float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(M.aligner.parameters(), snap))))
                        gt_f, ms_f, lpan_f, pan_f = (t.to(dev, dtype=dt) for t in self._fixed_batches["native"])
                        c_new = predict_c(M.aligner, pan_f, torch.nn.functional.interpolate(ms_f, scale_factor=4, mode="bicubic"), self.aligner_view_margin)
                        c_old = self._ema.get("_c_fixed_native")
                        self._jsonl("gradient_diagnostics.jsonl", dict(step=int(global_step), parameter_update_norm_phi=upd,
                                                                        prediction_change_norm=(float((c_new - c_old).norm(dim=1).mean()) if c_old is not None else None),
                                                                        aligner_grad_norm=self._ema.get("aligner_grad_norm"), backbone_grad_norm=self._ema.get("backbone_grad_norm")))
                        self._ema["_c_fixed_native"] = c_new
                if self.accelerator.is_main_process:
                    report.update(B, total.item(), info["rec"].item(), float(info["weight"]) * float(info["off"]))
                    with torch.no_grad():
                        c = info["c"].detach().float(); e = info["eps"].float(); key = "corrupt" if info["corrupt"] else "native"
                        for k, v in ((f"{key}_c_dy", c[:, 0].mean().item()), (f"{key}_c_dx", c[:, 1].mean().item()), (f"{key}_c_norm_p50", c.norm(dim=1).median().item()),
                                     (f"{key}_c_std", c.std(0).mean().item()), ("loss_rec", info["rec"].item()), ("loss_off", float(info["off"])), ("weight", float(info["weight"]))):
                            self._ema_update(k, v)
                        if info["corrupt"]:
                            self._ema_update("eps_norm_mean", e.norm(dim=1).mean().item()); self._ema["eps_norm_max"] = max(self._ema.get("eps_norm_max", 0.0), e.norm(dim=1).max().item())
                            self._ema_update("closure", float((c + e - info["c0"].detach().float()).norm(dim=1).mean()))
                            if global_step % 50 == 1:
                                H, W = pan.shape[-2:]
                                self._ema_update("border_eps", 1.0 - warp_support_mask(H, W, e).float().mean().item(), 0.9)
                                self._ema_update("border_c", 1.0 - warp_support_mask(H, W, c).float().mean().item(), 0.9)
            (t_corrupt if info["corrupt"] else t_native).append(time.time() - t0)
            global_step += 1; self._global_step = global_step
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']; e = self._ema
                extra = (f"\t[{self.case}] rec {e.get('loss_rec', 0):.5f} off {e.get('loss_off', 0):.4f} w {e.get('weight', 0):.4f}"
                         f"\tĉ0 ({e.get('native_c_dy', 0):+.3f},{e.get('native_c_dx', 0):+.3f}) |ĉ0| {e.get('native_c_norm_p50', 0):.3f}"
                         f"\tĉε ({e.get('corrupt_c_dy', 0):+.3f},{e.get('corrupt_c_dx', 0):+.3f}) |ĉε| {e.get('corrupt_c_norm_p50', 0):.3f} sd {e.get('corrupt_c_std', 0):.3f}"
                         f"\t|ε| {e.get('eps_norm_mean', 0):.3f} max {e.get('eps_norm_max', 0):.3f} closure {e.get('closure', 0):.3f}"
                         f"\tborder ε {e.get('border_eps', 0):.4f} ĉ {e.get('border_c', 0):.4f}"
                         f"\tgrad al {e.get('aligner_grad_norm', 0):.3g} bb {e.get('backbone_grad_norm', 0):.3g}"
                         f"\tt nat {np.mean(t_native[-50:]) if t_native else 0:.3f}s cor {np.mean(t_corrupt[-50:]) if t_corrupt else 0:.3f}s")
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start) + extra)
                self._jsonl("train_log.jsonl", dict(step=int(global_step), lr=lr, **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in e.items() if not k.startswith("_")},
                                                     t_native_s=(float(np.mean(t_native[-200:])) if t_native else None), t_corrupt_s=(float(np.mean(t_corrupt[-200:])) if t_corrupt else None),
                                                     peak_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None)))
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
                    self._finish_ledger("FINISHED_TRAIN")
                self.accelerator.end_training()
                return global_step
        return global_step
