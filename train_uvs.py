"""UVSTrainer — UVS-KD (research_log/2026-09-06_uvs-kd_30h_experiment-plan.md). variant:
  b0 : provided lms 입력·base, 기존 L1 (baseline)
  k0 : b0 + unweighted output KD  λ_s·|R_S − R_T|
  k1 : uncertainty routing  <(1+U_T)|R_S−R_GT|> + λ_s <(1−U_T)|R_S−R_T|>
  k2 : k1 + GT residual variance 가중 w_V
  s0 : b0 + shift-token KD + student warp (처음부터 ĉ_S δ_S)
  m1 : k2 + shift KD, student warp 처음부터
  m2 : m1 + scheduled teacher forcing  δ_use = η sg(ĉ_T δ_T) + (1−η) ĉ_S δ_S
  m3 : m2 + shift-effect loss λ_w
공통: MS mode 입력 [W(P), W(LP), W(P−LP), LMS] (PAN 3ch 강체 warp 4δ), 잔차 base = LMS, GT M-frame.
      PAN mode 는 raw PAN 11ch [P, LP, P−LP, LMS], base LP·rep, target P·rep (shift·KD 없음).
      Teacher 신호는 cache(feeder 가 증강해 줌)에서만 읽는다 — online teacher forward 없음 (§5.5).
"""
import json
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

from train import Trainer
from utils import (Train_Report, Test_Reduced_Report, Test_Full_Report, reduced_metrics, full_metrics,
                   tensor2img, SCC_full_numpy)
from uvs.shift import ShiftModule, edge_rep, warp_pan_channels, gated_delta
from uvs.losses import (gt_residual_variance, percentile_normalize, variance_weight, routed_losses,
                        shift_kd_loss, teacher_forcing_eta, warp_effect_loss)

ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ("b0", "k0", "k1", "k2", "s0", "m1", "m2", "m3")
USES_SHIFT = ("s0", "m1", "m2", "m3")
USES_U = ("k1", "k2", "m1", "m2", "m3")
USES_V = ("k2", "m1", "m2", "m3")
USES_SOFT = ("k0", "k1", "k2", "m1", "m2", "m3")


class UVSModel(nn.Module):
    def __init__(self, backbone, shift=None):
        super().__init__()
        self.backbone = backbone
        self.shift = shift


def _g(d, k, default):
    v = (d or {}).get(k, default)
    return default if v is None else v


def build_inputs(pan, lpan, lms):
    """LP = bicubic↑(lpan) (원 코드와 동일), HF = P − LP. 업샘플 MS 는 제공 lms (§1.2)."""
    lpan_u = F.interpolate(lpan, scale_factor=4, mode="bicubic")
    return lpan_u, pan - lpan_u


def x11(pan, lpan_u, pan_hf, lms):
    return torch.cat((pan, lpan_u, pan_hf, lms), dim=1)


class UVSTrainer(Trainer):
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']
        self.test_full_data_loader = data_loader['test_full']
        assert getattr(args, "mars", "dual") == "dual" and args.res
        assert args.feeder == "feeders.feeder_uvs.PanFeederUVS", "uvs 는 PanFeederUVS(cache 증강) 필요"
        u = getattr(args, "uvs", {}) or {}
        self.variant = u.get("variant")
        assert self.variant in VARIANTS, f"uvs.variant ∈ {VARIANTS}"
        lo = u.get("loss", {}) or {}
        self.lam_s = float(_g(lo, "lambda_soft", 0.1)); self.lam_pan = float(_g(lo, "lambda_pan", 1.0))
        self.lam_d = float(_g(lo, "lambda_shift", 0.25)); self.lam_w = float(_g(lo, "lambda_warp", 0.05 if self.variant == "m3" else 0.0))
        self.conf_ratio = float(_g(lo, "confidence_loss_ratio", 0.1))
        va = u.get("variance", {}) or {}
        self.alpha_v = float(_g(va, "alpha", 1.0)); self.v_window = int(_g(va, "window", 5))
        sh = u.get("shift", {}) or {}
        self.radius = int(_g(sh, "search_radius", 3)); self.T = float(_g(sh, "softmax_temperature", 0.07))
        self.conf_thr = float(_g(sh, "confidence_threshold", 0.35)); self.warp_mode = _g(sh, "warp_mode", "bicubic")
        self.s_channels = tuple(_g(sh, "student_channels", [8, 8]))
        tf = u.get("teacher_forcing", {}) or {}
        self.tf_s0 = int(_g(tf, "s0", 5000)); self.tf_s1 = int(_g(tf, "s1", 20000))
        self.need_cache = self.variant != "b0"
        # 고정 상수 (train 에서 한 번 계산, cache builder 산출)
        nrm_p = os.path.join(ROOT, _g(u, "teacher_norm", "")) if u.get("teacher_norm") else None
        self.v_q = None
        if nrm_p and os.path.exists(nrm_p):
            nrm = json.load(open(nrm_p)); self.v_q = (float(nrm["v_q10"]), float(nrm["v_q90"]))
        if self.variant in USES_V:
            assert self.v_q is not None, "V_GT 정규화 분위수(uvs.teacher_norm) 필요"
        if self.need_cache:
            tc = args.train_feeder_args.get("teacher_cache")
            assert tc and os.path.exists(os.path.join(ROOT, tc) if not os.path.isabs(tc) else tc), "train_feeder_args.teacher_cache 필요"

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        shift = ShiftModule(self.s_channels, self.radius, self.T) if self.variant in USES_SHIFT else None
        self.model = UVSModel(model, shift)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=args.num_warmup, num_training_steps=args.num_iter)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        self.last_reduced_metrics, self.last_full_metrics, self.last_val_metrics = {}, {}, {}
        self.last_fscc_official = float("nan")
        self._ema = {}
        self.audit_fr = self._load_audit_fr()

    @property
    def M(self):
        return self.accelerator.unwrap_model(self.model)

    def _load_audit_fr(self):
        p = os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")
        if not os.path.exists(p):
            return None
        import pandas as pd
        df = pd.read_csv(p).sort_values("sample_id")
        return torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)

    def _ema_update(self, k, v, m=0.98):
        self._ema[k] = v if k not in self._ema else m * self._ema[k] + (1 - m) * v

    # ------------------------------------------------------------------ student shift
    def _student_shift(self, lpan, ms):
        if self.M.shift is None:
            return None
        return self.M.shift(edge_rep(lpan), edge_rep(ms))

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.model.train(); self.model.requires_grad_(True)
        report = Train_Report(); start = time.time()
        B, C = self.args.batch_size, self.args.num_bands
        dev, dt = self.accelerator.device, self.weight_dtype
        M = self.M; v = self.variant
        for idx, batch in tqdm(enumerate(self.train_data_loader)):
            gt, lms, ms, lpan, pan, r_t, u_t, d_t, c_t, _ = batch
            with self.accelerator.accumulate(self.model):
                gt, lms, ms, lpan, pan, r_t, u_t, d_t, c_t = (t.to(dev, dtype=dt) for t in (gt, lms, ms, lpan, pan, r_t, u_t, d_t, c_t))
                c_t = c_t.view(-1)
                with torch.no_grad():
                    lpan_u, pan_hf = build_inputs(pan, lpan, lms)
                    r_gt = gt - lms
                    w_v = None
                    if v in USES_V:
                        v_gt = percentile_normalize(gt_residual_variance(gt, lms, self.v_window), *self.v_q)
                        w_v = variance_weight(v_gt, self.alpha_v)
                    dt_hat = c_t.unsqueeze(1) * d_t                                    # ĉ_T δ_T (cache, 이미 sg)
                # --- shift (student) ---
                sh = self._student_shift(lpan, ms)
                if v in USES_SHIFT:
                    ds_hat = gated_delta(sh["delta"], sh["conf"])                       # ĉ_S δ_S (grad 흐름)
                    eta = teacher_forcing_eta(global_step, self.tf_s0, self.tf_s1) if v in ("m2", "m3") else 0.0
                    d_use = eta * dt_hat + (1.0 - eta) * ds_hat
                    pan_a, lpan_a, hf_a = warp_pan_channels(pan, lpan_u, pan_hf, d_use, self.warp_mode)
                else:
                    eta = 0.0; pan_a, lpan_a, hf_a = pan, lpan_u, pan_hf
                x_ms = x11(pan_a, lpan_a, hf_a, lms)
                x_pan = x11(pan, lpan_u, pan_hf, lms)                                   # PAN mode: raw (§7.5)
                sw = torch.cat([torch.zeros(B, device=dev, dtype=dt), torch.ones(B, device=dev, dtype=dt)])
                res = M.backbone(None, None, None, sw, x_in=torch.cat([x_pan, x_ms], 0))
                r_pan, r_s = res[:B], res[B:]
                p_hat = lpan_u.repeat(1, C, 1, 1) + r_pan
                loss_pan = (pan.repeat(1, C, 1, 1) - p_hat).abs().mean()
                # --- MS losses ---
                if v in USES_U:
                    l_hard, l_soft = routed_losses(r_s, r_gt, r_t, u_t, w_v)
                else:
                    l_hard = (r_s - r_gt).abs().mean()
                    l_soft = (r_s - r_t).abs().mean() if v == "k0" else torch.zeros((), device=dev)
                loss = l_hard + (self.lam_s * l_soft if v in USES_SOFT else 0.0) + self.lam_pan * loss_pan
                l_shift = l_warp = torch.zeros((), device=dev)
                if v in USES_SHIFT:
                    l_shift, l_vec, l_cf = shift_kd_loss(sh["delta"], sh["conf"], d_t, c_t, self.conf_ratio)
                    loss = loss + self.lam_d * l_shift
                    if v == "m3" and self.lam_w > 0:
                        _, _, hf_t = warp_pan_channels(pan, lpan_u, pan_hf, dt_hat, self.warp_mode)
                        _, _, hf_s = warp_pan_channels(pan, lpan_u, pan_hf, ds_hat, self.warp_mode)
                        l_warp = warp_effect_loss(hf_s, hf_t, c_t, v_gt if w_v is not None else torch.ones_like(u_t))
                        loss = loss + self.lam_w * l_warp
                if not torch.isfinite(loss):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}'); sys.exit(3)
                self.accelerator.backward(loss)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    report.update(B * 2, loss.item(), l_hard.item(), loss_pan.item())
                    with torch.no_grad():
                        self._ema_update("l_soft", l_soft.item())
                        if v in USES_SHIFT:
                            d_s = sh["delta"].detach()
                            self._ema_update("l_shift", l_shift.item()); self._ema_update("eta", eta)
                            self._ema_update("mae_s_t", (d_s - d_t).abs().mean().item())
                            self._ema_update("c_s", sh["conf"].mean().item()); self._ema_update("c_t", c_t.mean().item())
                            self._ema_update("d_s_mag", d_s.norm(dim=1).mean().item()); self._ema_update("d_use_mag", d_use.detach().norm(dim=1).mean().item())
                            self._ema_update("pB", sh["p_boundary"].mean().item())
                            if v == "m3": self._ema_update("l_warp", l_warp.item())
                        if w_v is not None: self._ema_update("wv_max", w_v.max().item())
                        if v in USES_U: self._ema_update("u_mean", u_t.mean().item())
            global_step += 1
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']; e = self._ema
                extra = f"\tsoft {e.get('l_soft', 0):.5f}"
                if v in USES_U: extra += f"\tU {e.get('u_mean', 0):.3f}"
                if v in USES_V: extra += f"\twVmax {e.get('wv_max', 0):.2f}"
                if v in USES_SHIFT:
                    extra += (f"\tshift {e.get('l_shift', 0):.5f} eta {e.get('eta', 0):.2f} MAE(S,T) {e.get('mae_s_t', 0):.3f}"
                              f" cS {e.get('c_s', 0):.2f} cT {e.get('c_t', 0):.2f} |dS| {e.get('d_s_mag', 0):.3f} |use| {e.get('d_use_mag', 0):.3f} pB {e.get('pB', 0):.2f}")
                    if v == "m3": extra += f" warp {e.get('l_warp', 0):.5f}"
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start) + extra)
                start = time.time(); report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                if report.num_examples > 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start))
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step
        return global_step

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def infer(self, lms, ms, lpan, pan, delta_override=None, gate=True):
        """MS mode 추론. shift variant 는 δ_infer = gate(ĉ_S δ_S, 0.35). delta_override 로 외부 δ(LR px) 강제 가능."""
        dev, dt = self.accelerator.device, self.weight_dtype
        lms, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (lms, ms, lpan, pan))
        lpan_u, pan_hf = build_inputs(pan, lpan, lms)
        sh = self._student_shift(lpan, ms)
        if delta_override is not None:
            d = delta_override.to(dev, dtype=dt)
        elif sh is not None:
            d = gated_delta(sh["delta"], sh["conf"], self.conf_thr if gate else None)
        else:
            d = None
        pan_a, lpan_a, hf_a = warp_pan_channels(pan, lpan_u, pan_hf, d, self.warp_mode) if d is not None else (pan, lpan_u, pan_hf)
        sw = torch.ones(ms.shape[0], device=dev, dtype=dt)
        res = self.M.backbone(None, None, None, sw, x_in=x11(pan_a, lpan_a, hf_a, lms))
        return dict(y=lms + res, lms=lms, pan=pan, ms=ms, delta=d, shift=sh)

    def test_reduced(self, test_log, epoch):
        report = Test_Reduced_Report(); self.model.eval(); self.model.requires_grad_(False)
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            o = self.infer(lms, ms, lpan, pan)
            gt = gt.to(self.accelerator.device, dtype=self.weight_dtype)
            self.save_test_reduced(o["pan"], gt, o["y"], o["lms"], idx)
            report.update(self.args.test_batch_size, reduced_metrics(x_true=gt, x_pred=o["y"], max_pixel=self.args.max_pixel))
        test_log.write(f'Epoch[{epoch}]\t' + report.result_str())
        self.last_reduced_metrics = report.as_dict()
        return report.ergas

    def _official(self, y, lms, pan, f_dl, f_ds, sensor, wald):
        def _dn(t):
            a = ((t + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy(); return np.round(a.astype(np.float64) * self.args.max_pixel)
        sr_np, lms_np, pan_np = _dn(y[0]).transpose(1, 2, 0), _dn(lms[0]).transpose(1, 2, 0), _dn(pan[0, 0])
        return (f_dl(sr_np, lms_np, sensor, 4, 32, wald), f_ds(sr_np, lms_np, pan_np, 4, 32, wald),
                float(SCC_full_numpy(tensor2img(pan, self.args.max_pixel), tensor2img(y, self.args.max_pixel))))

    def test_full(self, test_log, epoch):
        wald, f_dl, f_ds, sensor, lo, hi = self._fr_official_setup()
        dl, ds, fs, dlt, cf, gated = [], [], [], [], [], []
        report = Test_Full_Report(); self.model.eval(); self.model.requires_grad_(False)
        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            o = self.infer(lms, ms, lpan, pan)
            self.save_test_full(o["pan"], o["y"], o["lms"], idx)
            report.update(self.args.test_batch_size, full_metrics(x_pred=o["y"], pan=o["pan"], ms=o["ms"], max_pixel=self.args.max_pixel))
            if lo <= idx <= hi:
                a, b, c = self._official(o["y"], o["lms"], o["pan"], f_dl, f_ds, sensor, wald)
                dl.append(a); ds.append(b); fs.append(c)
                if o["shift"] is not None:
                    dlt.append(o["shift"]["delta"][0].float().cpu()); cf.append(o["shift"]["conf"][0].item())
                    gated.append(float(o["shift"]["conf"][0].item() < self.conf_thr))
        hqnr = float(np.mean((1 - np.array(dl)) * (1 - np.array(ds)))); fscc = float(np.mean(fs))
        line = report.result_str() + f'\tHQNR_official({lo}-{hi}): {hqnr:.6f}\tfSCC({lo}-{hi}): {fscc:.6f}\tD_l_off {np.mean(dl):.5f} D_s_off {np.mean(ds):.5f}'
        self.last_full_metrics = report.as_dict()
        self.last_full_metrics.update(hqnr_official=hqnr, d_lambda_official=float(np.mean(dl)), d_s_official=float(np.mean(ds)), fscc_official=fscc)
        if dlt:
            D = torch.stack(dlt); st = dict(g1_conf_mean=float(np.mean(cf)), g1_delta_mag=float(D.norm(dim=1).mean()),
                                            g1_p_boundary=float(np.mean(gated)))     # p_boundary 열을 '게이트로 0 처리된 비율' 로 재사용
            if self.audit_fr is not None:
                tgt = -self.audit_fr[lo:hi + 1]                                          # δ_{MS←PAN} = −Δ_audit (LR px)
                st.update(g1_corr_dy=float(np.corrcoef(D[:, 0], tgt[:, 0])[0, 1]), g1_corr_dx=float(np.corrcoef(D[:, 1], tgt[:, 1])[0, 1]),
                          g1_med_err_vs_audit=float((D - tgt).norm(dim=1).median()))
                line += f"\t[shift] |δS| {st['g1_delta_mag']:.3f} conf {st['g1_conf_mean']:.2f} gated {st['g1_p_boundary']:.2f} corr(dy,dx)=({st['g1_corr_dy']:+.2f},{st['g1_corr_dx']:+.2f}) medErr {st['g1_med_err_vs_audit']:.3f}"
            self.last_full_metrics.update(st)
        test_log.write(f'Epoch[{epoch}]\t' + line)
        self.last_fscc_official = fscc
        return report.d_s, hqnr

    def write_best_meta(self, epoch, global_step, hqnr):
        m = dict(iteration=int(global_step), epoch=int(epoch), hqnr=float(hqnr), fscc=float(self.last_fscc_official),
                 d_lambda=self.last_full_metrics.get("d_lambda_official"), d_s=self.last_full_metrics.get("d_s_official"),
                 variant=self.variant, lambda_soft=self.lam_s, lambda_shift=self.lam_d, lambda_warp=self.lam_w, alpha_v=self.alpha_v,
                 conf_threshold=self.conf_thr, warp_mode=self.warp_mode,
                 shift={k: v for k, v in self.last_full_metrics.items() if k.startswith("g1_")} or None)
        json.dump(m, open(os.path.join(self.args.work_dir, "best_hqnr_meta.json"), "w"), indent=1)

    def _collect(self, loader, has_gt):
        self.model.eval(); self.model.requires_grad_(False)
        acc = dict(pan=[], lms=[], ms=[], gt=[], sr=[])
        for batch in tqdm(loader):
            gt, (lms, ms, lpan, pan) = (batch[0], batch[1:5]) if has_gt else (None, batch[:4])
            o = self.infer(lms, ms, lpan, pan)
            acc["pan"].append(o["pan"]); acc["lms"].append(o["lms"]); acc["ms"].append(o["ms"]); acc["sr"].append(o["y"])
            if gt is not None: acc["gt"].append(gt.to(o["pan"].device))
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
