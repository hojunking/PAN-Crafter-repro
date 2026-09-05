"""ShiftRobustTrainer — J1/J2/J3/J4/G1 (research_log/s1_w168_d123_shift_robust_alignment_30h_plan.md).

원 파이프라인 그대로: 원 PanFeeder(LR 증강 — bicubic phase 1.5 는 flip/rot 과 1e-13 이내 가환), 원 bicubic
(`F.interpolate`) 업샘플, 잔차 base·GT·출력 전부 M-frame, PAN mode 는 inverse/warp/mask 없음.
바뀌는 것은 **네트워크가 보는 MS 조건 채널** 뿐이다:
  j1 : ms_cond = T_ε(ms_base), ε~U(-r,r)² HR px, 두 mode 공통
  j2 : MS mode 만 T_ε, PAN mode 는 clean
  j3 : ms_cond = G_σ*(ms_base) (위치 이동 없음, 보정 σ*), 두 mode 공통
  j4 : MS mode 를 clean/jitter 두 branch — L_MS = ½L1(Ŷ0,Y)+½L1(Ŷε,Y), L_cons = |Rε − sg(R0)|₁, λ(t) 0→0.1 (5K warmup)
  g1 : first conv 를 PAN/MS 기여로 분리, MS mode 에서만 synthetic shift ε_g 로 global correlator 학습,
       F̃_P = F_P^syn + g[W(F_P^syn, Δ̂) − F_P^syn], L += 0.1·SmoothL1(Δ̂, −ε_g). PAN mode 는 F_P+F_M 그대로.
ε=0(또는 σ=0, 또는 g1 미적용) 경로는 원 Trainer 와 동작이 같다 (T01, 등가 실행으로 검증).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from sr.jitter import sample_jitter, grad_energy_ratio
from sr.pan_align import GlobalCorrelator
from sr.forward import sr_forward, sr_infer
from train import Trainer
from utils import (Train_Report, Test_Reduced_Report, Test_Full_Report,
                   reduced_metrics, full_metrics, tensor2img, SCC_full_numpy)

ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ("j1", "j2", "j3", "j4", "g1")


class SRModel(nn.Module):
    """backbone + (g1) correlator. checkpoint 키는 backbone.* / correlator.*"""
    def __init__(self, backbone, correlator=None):
        super().__init__()
        self.backbone = backbone
        self.correlator = correlator


def _cfg(d, k, default):
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


class ShiftRobustTrainer(Trainer):
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']
        self.test_full_data_loader = data_loader['test_full']
        assert getattr(args, "mars", "dual") == "dual" and args.res, "계획 §0: dual MARs · 잔차 고정"
        assert args.feeder == "feeders.feeder.PanFeeder", "원 feeder 를 그대로 쓴다 (§4.1)"
        sr = getattr(args, "sr", {}) or {}
        self.variant = sr.get("variant")
        assert self.variant in VARIANTS, f"sr.variant 는 {VARIANTS} 중 하나"
        jit = sr.get("jitter", {}) or {}
        self.jit_r = float(_cfg(jit, "max_abs_hr_px", 0.5))
        self.jit_p = float(_cfg(jit, "probability", 1.0))
        assert _cfg(jit, "distribution", "uniform") == "uniform"
        self.blur_sigma = None
        if self.variant == "j3":
            bl = sr.get("blur", {}) or {}
            if bl.get("sigma") is not None:
                self.blur_sigma = float(bl["sigma"])
            else:
                cal = os.path.join(ROOT, _cfg(bl, "calibration", "outputs/shift_robust/blur_calib.json"))
                assert os.path.exists(cal), f"blur 보정 파일 없음: {cal} — tools/calibrate_blur.py 먼저"
                allinfo = json.load(open(cal))
                match = _cfg(bl, "match", "mse")
                info = allinfo[match] if match in allinfo else allinfo
                assert info.get("within_tol"), f"blur 보정({match})이 tol 안에 없다: {info}"
                self.blur_sigma = float(info["sigma_star"])
                self.blur_calib = info
                print(f"[sr] J3 blur sigma* = {self.blur_sigma:.4f} (match={match}, r_jit {info['r_jit']:.5f})")
        cons = sr.get("cons", {}) or {}
        self.lam_cons = float(_cfg(cons, "lambda", 0.1))
        self.cons_warmup = int(_cfg(cons, "warmup_steps", 5000))
        g1 = sr.get("g1", {}) or {}
        self.g1_tau = float(_cfg(g1, "tau", 0.07)); self.g1_c0 = float(_cfg(g1, "gate_c0", 0.30))
        self.g1_syn_max = float(_cfg(g1, "syn_max_hr_px", 1.0)); self.g1_syn_p = float(_cfg(g1, "syn_prob", 0.75))
        self.g1_lam = float(_cfg(g1, "lambda_shift", 0.1)); self.g1_radius = float(_cfg(g1, "radius_hr_px", 1.0))
        self.g1_n = int(_cfg(g1, "n_per_axis", 5)); self.g1_desc = int(_cfg(g1, "desc_channels", 16))
        inf = sr.get("inference", {}) or {}
        self.inf_jitter = float(_cfg(inf, "jitter_hr_px", 0.0))          # 진단용 고정 ε (기본 0)
        self.inf_g1_scale = float(_cfg(inf, "g1_scale", 1.0))
        self.grad_accum = int(_cfg(sr, "grad_accum", 1))

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config,
                                       gradient_accumulation_steps=self.grad_accum)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        corr = None
        if self.variant == "g1":
            C = model.input.out_channels
            corr = GlobalCorrelator(C, self.g1_desc, self.g1_radius, self.g1_n, self.g1_tau, self.g1_c0)
        self.model = SRModel(model, corr)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=args.num_warmup, num_training_steps=args.num_iter)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        self.last_reduced_metrics, self.last_full_metrics, self.last_val_metrics = {}, {}, {}
        self.last_fscc_official = float("nan")
        self._ema = {}
        self._gen = None
        self.audit_fr = self._load_audit_fr()

    # ------------------------------------------------------------------ helpers
    @property
    def M(self):
        return self.accelerator.unwrap_model(self.model)

    def _load_audit_fr(self):
        """G1 진단용 audit Δ (FR cache, LR px). 학습 label 로는 절대 쓰지 않는다 (§11.6)."""
        p = os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")
        if not os.path.exists(p):
            return None
        import pandas as pd
        df = pd.read_csv(p).sort_values("sample_id")
        return torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)

    def _ema_update(self, k, v, m=0.98):
        self._ema[k] = v if k not in self._ema else m * self._ema[k] + (1 - m) * v

    def _lam_cons(self, step):
        if self.cons_warmup <= 0:
            return self.lam_cons
        return self.lam_cons * min(1.0, step / float(self.cons_warmup))

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.model.train(); self.model.requires_grad_(True)
        report = Train_Report(); start = time.time()
        B = self.args.batch_size
        dev, dt = self.accelerator.device, self.weight_dtype
        M = self.M
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (gt, ms, lpan, pan))
                with torch.no_grad():
                    eps = sample_jitter(B, self.jit_r, dev, dt, self.jit_p) if self.variant in ("j1", "j2", "j4") else None
                    eps_g = sample_jitter(B, self.g1_syn_max, dev, dt, self.g1_syn_p) if self.variant == "g1" else None
                lam = self._lam_cons(global_step) if self.variant == "j4" else 0.0
                o = sr_forward(M, self.variant, pan, lpan, ms, gt, eps=eps, eps_g=eps_g, w_off=self.args.w_off,
                               lam_cons=lam, blur_sigma=self.blur_sigma)
                loss = o["loss"] + (self.g1_lam * o["loss_shift"] if self.variant == "g1" else 0.0)
                if not torch.isfinite(loss):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}'); sys.exit(3)
                self.accelerator.backward(loss)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(B * 2, loss.item(), o["loss_ms"].item(), o["loss_pan"].item())
                    with torch.no_grad():
                        if eps is not None:
                            e = eps.float()
                            for k, v in (("eps_y_mean", e[:, 0].mean().item()), ("eps_y_std", e[:, 0].std().item()),
                                         ("eps_x_mean", e[:, 1].mean().item()), ("eps_x_std", e[:, 1].std().item()),
                                         ("eps_absmax", e.abs().max().item())):
                                self._ema_update(k, v)
                        if self.variant in ("j1", "j2", "j3", "j4") and global_step % 20 == 0:
                            self._ema_update("ge_ratio", grad_energy_ratio(o["cond_ms"], o["ms_base"]), 0.9)
                        if self.variant == "j4":
                            self._ema_update("loss_cons", o["loss_cons"].item()); self._ema_update("lam", lam)
                            self._ema_update("cons_ratio", (lam * o["loss_cons"] / (o["loss_ms"] + 1e-12)).item())
                        if self.variant == "g1":
                            info, d = o["info"], o["info"]["delta"].detach().float()
                            for k, v in (("loss_shift", o["loss_shift"].item()), ("g_dy", d[:, 0].mean().item()),
                                         ("g_dx", d[:, 1].mean().item()), ("g_conf", info["conf"].mean().item()),
                                         ("g_pbnd", info["p_boundary"].mean().item()), ("g_pctr", info["p_center"].mean().item()),
                                         ("g_err", (d + o["eps_g"].float()).norm(dim=1).mean().item())):
                                self._ema_update(k, v)
            global_step += 1
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                e = self._ema
                extra = ""
                if self.variant in ("j1", "j2", "j4"):
                    extra += (f"\teps y {e.get('eps_y_mean', 0):+.3f}±{e.get('eps_y_std', 0):.3f} x {e.get('eps_x_mean', 0):+.3f}±{e.get('eps_x_std', 0):.3f}"
                              f" |max| {e.get('eps_absmax', 0):.2f}\tgeR {e.get('ge_ratio', 1):.4f}")
                if self.variant == "j3":
                    extra += f"\tsigma {self.blur_sigma:.3f}\tgeR {e.get('ge_ratio', 1):.4f}"
                if self.variant == "j4":
                    extra += f"\tcons {e.get('loss_cons', 0):.5f} lam {e.get('lam', 0):.3f} ratio {e.get('cons_ratio', 0):.3f}"
                if self.variant == "g1":
                    extra += (f"\tshift {e.get('loss_shift', 0):.5f}\tdelta ({e.get('g_dy', 0):+.3f},{e.get('g_dx', 0):+.3f})"
                              f" err {e.get('g_err', 0):.3f} conf {e.get('g_conf', 0):.3f} pB {e.get('g_pbnd', 0):.3f} pC {e.get('g_pctr', 0):.3f}")
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start) + extra)
                start = time.time(); report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                if report.num_examples > 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start))
                if self.variant == "j4":
                    r = self._ema.get("cons_ratio", 0.0)
                    train_log.write(f"[j4] lam*L_cons/L_MS EMA {r:.3f} — " + ("consistency 과도(>0.30)" if r > 0.30 else "사실상 무효(<0.01)" if r < 0.01 else "정상 범위"))
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step
        return global_step

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _infer(self, pan, lpan, ms, eps=None, idx=None):
        """MS mode 추론. 기본 jitter 0 (T06). g1 은 ε_g=0 으로 correlator 를 실제 쌍에 적용."""
        dev, dt = self.accelerator.device, self.weight_dtype
        ms, lpan, pan = (t.to(dev, dtype=dt) for t in (ms, lpan, pan))
        if eps is None and self.inf_jitter != 0.0 and self.variant != "g1":
            eps = torch.full((ms.shape[0], 2), self.inf_jitter, device=dev, dtype=dt)
        o = sr_infer(self.M, self.variant, pan, lpan, ms, eps=eps, blur_sigma=self.blur_sigma, g1_scale=self.inf_g1_scale)
        return o

    def test_reduced(self, test_log, epoch):
        report = Test_Reduced_Report(); self.model.eval(); self.model.requires_grad_(False)
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            o = self._infer(pan, lpan, ms)
            gt = gt.to(self.accelerator.device, dtype=self.weight_dtype)
            self.save_test_reduced(o["pan"], gt, o["y"], o["ms_base"], idx)
            report.update(self.args.test_batch_size, reduced_metrics(x_true=gt, x_pred=o["y"], max_pixel=self.args.max_pixel))
        test_log.write(f'Epoch[{epoch}]\t' + report.result_str())
        self.last_reduced_metrics = report.as_dict()
        return report.ergas

    def _official(self, y, lms, pan, f_dl, f_ds, sensor, wald):
        def _dn(t):
            a = ((t + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy()
            return np.round(a.astype(np.float64) * self.args.max_pixel)
        sr_np, lms_np, pan_np = _dn(y[0]).transpose(1, 2, 0), _dn(lms[0]).transpose(1, 2, 0), _dn(pan[0, 0])
        return (f_dl(sr_np, lms_np, sensor, 4, 32, wald), f_ds(sr_np, lms_np, pan_np, 4, 32, wald),
                float(SCC_full_numpy(tensor2img(pan, self.args.max_pixel), tensor2img(y, self.args.max_pixel))))

    def test_full(self, test_log, epoch):
        wald, f_dl, f_ds, sensor, lo, hi = self._fr_official_setup()
        dl, ds, fs, g_d, g_c, g_b, g_ctr = [], [], [], [], [], [], []
        report = Test_Full_Report(); self.model.eval(); self.model.requires_grad_(False)
        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            o = self._infer(pan, lpan, ms, idx=idx)
            lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
            self.save_test_full(o["pan"], o["y"], o["ms_base"], idx)
            report.update(self.args.test_batch_size, full_metrics(x_pred=o["y"], pan=o["pan"], ms=o["ms"], max_pixel=self.args.max_pixel))
            if lo <= idx <= hi:
                a, b, c = self._official(o["y"], lms, o["pan"], f_dl, f_ds, sensor, wald)
                dl.append(a); ds.append(b); fs.append(c)
                if o["info"] is not None:
                    g_d.append(o["info"]["delta"][0].float().cpu()); g_c.append(o["info"]["conf"][0].item())
                    g_b.append(o["info"]["p_boundary"][0].item()); g_ctr.append(o["info"]["p_center"][0].item())
        hqnr = float(np.mean((1 - np.array(dl)) * (1 - np.array(ds))))
        fscc = float(np.mean(fs))
        line = report.result_str() + f'\tHQNR_official({lo}-{hi}): {hqnr:.6f}\tfSCC({lo}-{hi}): {fscc:.6f}\tD_l_off {np.mean(dl):.5f} D_s_off {np.mean(ds):.5f}'
        self.last_full_metrics = report.as_dict()
        self.last_full_metrics.update(hqnr_official=hqnr, d_lambda_official=float(np.mean(dl)), d_s_official=float(np.mean(ds)), fscc_official=fscc)
        if g_d:
            D = torch.stack(g_d)                                     # 예측 Δ̂ (HR px, PAN→M 이동량)
            stats = dict(g1_conf_mean=float(np.mean(g_c)), g1_p_boundary=float(np.mean(g_b)), g1_p_center=float(np.mean(g_ctr)),
                         g1_delta_mag=float(D.norm(dim=1).mean()))
            if self.audit_fr is not None:
                # audit Δ(LR px, aligned[y,x]=ms[y+dy,x+dx]) 로 MS 를 PAN 에 맞춘다 ⇔ PAN 을 M 으로 가져오려면 −4Δ HR px.
                tgt = -4.0 * self.audit_fr[lo:hi + 1]
                err = (D - tgt).norm(dim=1)
                cy = float(np.corrcoef(D[:, 0], tgt[:, 0])[0, 1]) if D.shape[0] > 2 else float("nan")
                cx = float(np.corrcoef(D[:, 1], tgt[:, 1])[0, 1]) if D.shape[0] > 2 else float("nan")
                stats.update(g1_corr_dy=cy, g1_corr_dx=cx, g1_med_err_vs_audit=float(err.median()))
                line += f"\t[G1] Δ̂|·| {stats['g1_delta_mag']:.3f} conf {stats['g1_conf_mean']:.3f} pB {stats['g1_p_boundary']:.3f} corr(dy,dx)=({cy:+.2f},{cx:+.2f}) medErr {stats['g1_med_err_vs_audit']:.3f}"
            self.last_full_metrics.update({k: v for k, v in stats.items()})
        test_log.write(f'Epoch[{epoch}]\t' + line)
        self.last_fscc_official = fscc
        return report.d_s, hqnr

    def write_best_meta(self, epoch, global_step, hqnr):
        m = dict(iteration=int(global_step), epoch=int(epoch), hqnr=float(hqnr), fscc=float(self.last_fscc_official),
                 d_lambda=self.last_full_metrics.get("d_lambda_official"), d_s=self.last_full_metrics.get("d_s_official"),
                 variant=self.variant, jitter_max_abs_hr_px=(self.jit_r if self.variant in ("j1", "j2", "j4") else 0.0),
                 blur_sigma=self.blur_sigma, lambda_cons=(self.lam_cons if self.variant == "j4" else None),
                 g1={k: v for k, v in self.last_full_metrics.items() if k.startswith("g1_")} or None)
        json.dump(m, open(os.path.join(self.args.work_dir, "best_hqnr_meta.json"), "w"), indent=1)

    def _collect(self, loader, has_gt):
        self.model.eval(); self.model.requires_grad_(False)
        acc = dict(pan=[], lms=[], ms=[], gt=[], sr=[])
        for batch in tqdm(loader):
            gt, (lms, ms, lpan, pan) = (batch[0], batch[1:]) if has_gt else (None, batch)
            o = self._infer(pan, lpan, ms)
            acc["pan"].append(o["pan"]); acc["lms"].append(lms.to(o["pan"].device)); acc["ms"].append(o["ms"]); acc["sr"].append(o["y"])
            if gt is not None:
                acc["gt"].append(gt.to(o["pan"].device))
        return acc

    def _savemat(self, name, **arrays):
        from scipy.io import savemat
        path = os.path.join(self.args.work_dir, 'results/'); os.makedirs(path, exist_ok=True)
        cv = lambda L: (torch.cat(L, 0).clip(-1, 1).detach().cpu().numpy() + 1.0) / 2 * self.args.max_pixel
        savemat(os.path.join(path, name), {k: cv(v) for k, v in arrays.items() if v})

    def test_reduced_save(self, tag='best_hqnr'):
        a = self._collect(self.test_reduced_data_loader, True)
        self._savemat(f'reduced_{tag}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], gt=a["gt"], sr=a["sr"])

    def test_full_save(self, tag='best_hqnr'):
        a = self._collect(self.test_full_data_loader, False)
        self._savemat(f'full_{tag}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], sr=a["sr"])
