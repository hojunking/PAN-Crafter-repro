# --------------------------------------------------------
# Teacher(uncertainty)·Student(KD) trainer — 기존 Trainer 서브클래스.
# 명세: research_log/s1_mutual_and_kd_implementation_spec.md (Part B)
#
# 기존 구조 불변 원칙:
#   - validate/test_reduced/test_full/save_best_* 전부 상속 -> best 선택은
#     기존과 동일하게 **공식 HQNR** (사용자 확정: 명세 §24 의 val-ERGAS 는 쓰지 않는다)
#   - train() 한 메서드만 오버라이드. work_dir/meta/results 규약 그대로.
#   - MARs dual anchor 유지 (§0.4): KD/aux 는 MS half 에만, PAN half 는 hard L1.
# --------------------------------------------------------

import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from train import Trainer
from utils import Train_Report
from kd.ops import MTFDownsampler, AbsoluteGradient, LocalVarianceMap, ramp_then_decay
from kd.losses import (sis_loss, edge_loss, uncertainty_nll, uknow_weights,
                       weighted_l1, spectral_kd_loss, uknow_weights_fixed, gtvar_loss)
from kd.features import FeatureTap, FeatureProj, WithUncertainty


class DualBatchMixin:
    """배치 준비 공용부 — 기존 train() 의 전처리를 그대로 옮긴 것."""

    @property
    def ms_only(self):
        """mars: ms — PAN reconstruction task·batch 복제 제거 (clean MS-only)."""
        return getattr(self.args, "mars", "dual") == "ms"

    def prep(self, batch):
        """MS-only 면 rep=1·switch=1(MS mode 만), dual 이면 기존과 동일하게 2배 복제.

        반환의 B 는 항상 'MS half 의 시작 index' 다 — MS-only 에서는 0 이라
        recon[B:] 가 전체를 가리키고 recon[:B] 는 빈 텐서가 된다 (PAN 항 자연 소멸).
        """
        gt, lms, ms, lpan, pan = batch
        dev, dt = self.accelerator.device, self.weight_dtype
        bs = self.args.batch_size
        rep = 1 if self.ms_only else 2
        t = lambda x: x.to(dev, dtype=dt).repeat(rep, 1, 1, 1)
        gt, lms, ms, lpan, pan = map(t, (gt, lms, ms, lpan, pan))
        res_pan = F.interpolate(lpan, scale_factor=4, mode="bicubic")
        res_ms = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                  else F.interpolate(ms, scale_factor=4, mode="bicubic"))
        if self.ms_only:
            switch = torch.ones(bs, device=dev).to(dtype=dt)
            B = 0
        else:
            switch = torch.cat((torch.zeros(bs, device=dev),
                                torch.ones(bs, device=dev))).to(dtype=dt)
            B = bs
        return gt, lms, ms, lpan, pan, res_pan, res_ms, switch, B

    def recon_of(self, out, res_ms, res_pan, switch):
        if not self.args.res:
            return out
        if self.ms_only:
            return out + res_ms
        return (out + res_ms * switch.view(-1, 1, 1, 1)
                + res_pan.repeat(1, self.args.num_bands, 1, 1) * (1.0 - switch).view(-1, 1, 1, 1))

    def guard_finite(self, loss, train_log, global_step):
        if not torch.isfinite(loss):
            train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}')
            sys.exit(3)


def load_teacher(cfg_path, ckpt_dir, device, dtype):
    """teacher config+checkpoint 로부터 frozen teacher 를 만든다.

    checkpoint 가 WithUncertainty 래퍼로 학습된 것(base./head. 접두)이면
    래퍼째 복원해 (module, has_uncertainty=True) 를 돌려준다.
    """
    from main import import_class
    c = yaml.safe_load(open(cfg_path))
    base = import_class(c["model"])(**c.get("model_args", {}))
    sd_path = os.path.join(ckpt_dir, "model.safetensors")
    sd = load_file(sd_path) if os.path.exists(sd_path) else torch.load(
        os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    has_unc = any(k.startswith("head.") for k in sd)
    if has_unc:
        ch = c.get("model_args", {}).get("hidden_size", 128)
        # head 출력 종류는 calibration 산출물에 기록돼 있다 (없으면 기존 softplus).
        # 이걸 틀리면 theta 의 의미가 바뀌므로 조용한 오해석을 막는다.
        head_out = "softplus"
        nrm = os.path.join(ckpt_dir, "uq_norm.json")
        if os.path.exists(nrm):
            import json as _json
            head_out = _json.load(open(nrm)).get("head_out", "softplus")
        model = WithUncertainty(base, ch, head_out=head_out)
    else:
        model = base
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"teacher 가중치 누락: {missing[:5]}"
    model.to(device, dtype=dtype).eval()
    model.requires_grad_(False)
    return model, has_unc


class TeacherTrainer(DualBatchMixin, Trainer):
    """T1/T2 — c6 + uncertainty head (+ 선택적 SiS/edge). 명세 §18.

    모델을 WithUncertainty 로 감싼 뒤 기존 Trainer 에 넘긴다. forward 계약이
    residual 반환 그대로라 평가·HQNR 선택·export 경로가 전부 상속으로 동작한다.
    """

    def __init__(self, args, data_loader, model):
        ch = args.model_args.get("hidden_size", 128)
        super().__init__(args, data_loader, WithUncertainty(model, ch))
        ta = dict(args.teacher_args or {})
        self.lam_sis = float(ta.get("lambda_sis", 0.0))
        self.lam_edge = float(ta.get("lambda_edge", 0.0))
        self.sis_radius = int(ta.get("sis_radius", 1))
        self.sis_mode = ta.get("sis_mode", "shared_vector")
        self.eta_sam = float(ta.get("eta_sam", 0.1))
        dev = self.accelerator.device
        self.mtf = MTFDownsampler(bands=args.num_bands).to(dev, dtype=self.weight_dtype)
        self.grad_op = AbsoluteGradient().to(dev, dtype=self.weight_dtype)

    def train(self, train_log, global_step):
        self.model.train()
        self.model.requires_grad_(True)
        report = Train_Report()
        start = time.time()
        for idx, batch in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    gt, lms, ms, lpan, pan, res_pan, res_ms, switch, B = self.prep(batch)
                out = self.model(pan, lpan, ms, switch)
                recon = self.recon_of(out, res_ms, res_pan, switch)
                theta = self.model.theta() if not hasattr(self.model, "module") \
                    else self.model.module.theta()
                loss_pan = (torch.zeros((), device=recon.device, dtype=recon.dtype)
                            if self.ms_only else
                            (pan[:B].repeat(1, self.args.num_bands, 1, 1)
                             - recon[:B]).abs().mean() * self.args.w_off)
                l_unc, unc_d = uncertainty_nll(recon[B:], gt[B:], theta[B:])
                loss = l_unc + loss_pan
                extra = ""
                if self.lam_sis > 0:
                    pred_lr = self.mtf(recon[B:])
                    l_sis, sd = sis_loss(pred_lr, ms[B:], self.sis_radius,
                                         self.sis_mode, self.eta_sam)
                    loss = loss + self.lam_sis * l_sis
                    extra += (f"\tSiS: {l_sis.item():.5f} "
                              f"(ctr {sd['center_shift_ratio']:.2f} bnd {sd['boundary_ratio']:.2f})")
                if self.lam_edge > 0:
                    l_edge, _ = edge_loss(recon[B:], pan[B:], self.grad_op)
                    loss = loss + self.lam_edge * l_edge
                    extra += f"\tEdge: {l_edge.item():.5f}"
                self.guard_finite(loss, train_log, global_step)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size * (1 if self.ms_only else 2), loss.item(), l_unc.item(), loss_pan.item())
            global_step += 1
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t'
                                + report.result_str(lr, time.time() - start)
                                + f"\ttheta: {unc_d['theta_mean']:.4f}" + extra)
                start = time.time()
                report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step
        return global_step


class KDTrainer(DualBatchMixin, Trainer):
    """K0~K5 — teacher frozen, student 학습. 명세 §20-§22.

    kd_args.variant: k0|k1a|k1b|k2|k3|k4|k5 (누적 사다리).
    teacher 는 MS half 에서만 online forward (no_grad). PAN half 는 기존 hard anchor.
    """

    # k* = 2026-08-31 KD 캠페인 사다리 / u_full·u_full_gtvar = s2 계획(2026-09-01)
    #   u_full      : uncertainty routing + **full-output** soft KD (spectral 아님)
    #   u_full_gtvar: u_full + GT-output local variance 정합 loss
    LADDER = ["k0", "k1a", "k1b", "k2", "k3", "k4", "k5", "u_full", "u_full_gtvar"]

    def __init__(self, args, data_loader, model):
        ka = dict(args.kd_args or {})
        self.variant = ka.get("variant", "k0")
        assert self.variant in self.LADDER, f"kd variant: {self.variant}"
        if self.variant == "k5":
            raise NotImplementedError(
                "K5 는 No-Go 판정 — proj 가 scheduler 생성 후 추가돼 warm-up/cosine 을 "
                "따르지 않고, teacher 측 stop-gradient 설계도 미확정. K4 까지 결과 확인 후 재설계")
        self.need_teacher = self.variant != "k0"
        self.need_unc = self.variant in ("k2", "k3", "k4", "k5", "u_full", "u_full_gtvar")
        self.lam_soft = float(ka.get("lambda_soft", 0.1))
        self.lam_sis = float(ka.get("lambda_sis", 0.1))
        self.lam_feat = float(ka.get("lambda_feat", 0.001))
        self.alpha_u = float(ka.get("alpha_u", 1.0))
        self.beta_v = float(ka.get("beta_v", 0.5))
        self.eta_sam = float(ka.get("eta_sam", 0.1))
        self.sis_radius = int(ka.get("sis_radius", 1))
        self.sis_mode = ka.get("sis_mode", "shared_vector")
        self.w_mode = ka.get("uncertainty_weight_mode", "robust_normalized")
        self.no_schedule = bool(ka.get("no_schedule", False))
        # s2 계획: λ_U(최대) · soft weight 바닥 · GT-variance 가중/κ
        self.lam_u = float(ka.get("lambda_u_max", self.lam_soft))
        self.soft_floor = float(ka.get("soft_floor", 0.05))
        self.lam_gtvar = float(ka.get("lambda_gtvar", 0.10))
        self.gtvar_kappa = ka.get("gtvar_kappa", None)
        self.uq_q05 = ka.get("uq_q05", None)
        self.uq_q95 = ka.get("uq_q95", None)
        super().__init__(args, data_loader, model)
        dev, dt = self.accelerator.device, self.weight_dtype
        self.mtf = MTFDownsampler(bands=args.num_bands).to(dev, dtype=dt)
        self.var_map = LocalVarianceMap().to(dev, dtype=dt)
        self.teacher, t_has_unc = (None, False)
        if self.need_teacher:
            assert args.teacher_config and args.teacher_checkpoint, \
                "kd 는 teacher_config/teacher_checkpoint 가 필요하다"
            self.teacher, t_has_unc = load_teacher(
                args.teacher_config, args.teacher_checkpoint, dev, dt)
            # head-only calibration 이 남긴 고정 상수(q05/q95/κ)를 teacher 디렉터리에서 읽는다.
            # config 로 명시하면 그쪽이 우선한다.
            nrm = os.path.join(args.teacher_checkpoint, "uq_norm.json")
            if os.path.exists(nrm):
                import json as _json
                d = _json.load(open(nrm))
                if self.uq_q05 is None:
                    self.uq_q05, self.uq_q95 = d.get("q05"), d.get("q95")
                if self.gtvar_kappa is None:
                    self.gtvar_kappa = d.get("kappa")
            # GT-variance gradient audit 결과(계획 §6)가 있으면 그 λ_V 를 쓴다 —
            # 두 seed 가 같은 파일을 읽으므로 "한 번만 조정하고 동일 고정"이 지켜진다.
            aud = os.path.join(args.teacher_checkpoint, "gtvar_audit.json")
            if self.variant == "u_full_gtvar" and os.path.exists(aud):
                import json as _json
                _a = _json.load(open(aud))
                self.lam_gtvar = float(_a["lambda_gtvar"])
                self._audit_ratio = _a.get("ratio")
            if self.variant in ("u_full", "u_full_gtvar"):
                assert self.uq_q05 is not None and self.uq_q95 is not None, (
                    "u_full 계열은 고정 분위수(q05/q95)가 필요하다 — "
                    "tools/calibrate_head.py 로 teacher 를 보정하거나 kd_args 에 직접 줄 것")
            if self.variant == "u_full_gtvar":
                assert os.path.exists(os.path.join(args.teacher_checkpoint,
                                                   "gtvar_audit.json")), (
                    "u_full_gtvar 는 계획 §6 의 gradient audit 이 선행돼야 한다 — "
                    "tools/gtvar_audit.py 를 먼저 돌릴 것 (게이트가 자동 실행한다)")
                assert self.gtvar_kappa is not None, (
                    "u_full_gtvar 는 gtvar_kappa 가 필요하다 (tools/calibrate_head.py 산출)")
            if self.need_unc and not t_has_unc:
                raise RuntimeError(
                    f"{self.variant} 는 uncertainty teacher(T1/T2 checkpoint)가 필요하다 "
                    "— k1a/k1b 를 쓰거나 teacher 를 T1 로 학습할 것")
        self.t_tap = self.s_tap = self.proj = None
        if self.variant == "k5":
            base_t = self.teacher.base if isinstance(self.teacher, WithUncertainty) else self.teacher
            t_ch = base_t.middle[-1].out_channels
            s_ch = args.model_args.get("hidden_size", 128)
            self.t_tap = FeatureTap(base_t)
            # k5 proj: prepare 로 accelerator 에 등록해 save_state 에 포함시키고,
            # param group 으로 optimizer 에 추가한다 (양쪽 사영 모두 학습).
            self.proj = self.accelerator.prepare(FeatureProj(t_ch, s_ch).to(dev, dtype=dt))
            self.optimizer.add_param_group({"params": self.proj.parameters()})
            self.s_tap = FeatureTap(self.model)

    def _sched(self, step):
        """명세 §22 의 ramp_then_decay 3종. no_schedule(스모크용)이면 최댓값 고정."""
        if getattr(self, "no_schedule", False):
            return self.lam_soft, self.lam_sis, self.lam_feat
        T = self.args.num_iter
        lam_max = self.lam_u if self.variant in ("u_full", "u_full_gtvar") else self.lam_soft
        return (ramp_then_decay(step, 5_000, 15_000, 40_000, T, lam_max),
                ramp_then_decay(step, 10_000, 20_000, 45_000, T, self.lam_sis),
                ramp_then_decay(step, 15_000, 25_000, 40_000, T, self.lam_feat))

    def train(self, train_log, global_step):
        self.model.train()
        self.model.requires_grad_(True)
        report = Train_Report()
        start = time.time()
        for idx, batch in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    gt, lms, ms, lpan, pan, res_pan, res_ms, switch, B = self.prep(batch)
                out = self.model(pan, lpan, ms, switch)
                recon = self.recon_of(out, res_ms, res_pan, switch)
                pred, gt_ms = recon[B:], gt[B:]
                loss_pan = (torch.zeros((), device=pred.device, dtype=pred.dtype)
                            if self.ms_only else
                            (pan[:B].repeat(1, self.args.num_bands, 1, 1)
                             - recon[:B]).abs().mean() * self.args.w_off)

                t_pred = theta = None
                if self.need_teacher:
                    with torch.no_grad():
                        ones = switch[B:]
                        t_out = self.teacher(pan[B:], lpan[B:], ms[B:], ones)
                        t_pred = t_out + res_ms[B:] if self.args.res else t_out
                        if self.need_unc:
                            theta = self.teacher.theta()

                w_hard = w_soft = None
                if self.need_unc:
                    if self.variant in ("k3", "k4", "k5"):
                        # 명세: normalize(1 + αu + βv) — 한 번에 정규화해야 실효 β 가 유지된다
                        from kd.ops import mean_normalize, robust_normalize_01
                        u = robust_normalize_01(theta.detach())
                        v = self.var_map(gt_ms)
                        w_hard = mean_normalize(1.0 + self.alpha_u * u + self.beta_v * v)
                        w_soft = mean_normalize(1.0 - u)
                    elif self.variant in ("u_full", "u_full_gtvar"):
                        # s2 계획 §4: 고정 train-set 분위수 정규화 + soft 바닥
                        w_hard, w_soft, _u = uknow_weights_fixed(
                            theta, self.uq_q05, self.uq_q95,
                            soft_floor=self.soft_floor, alpha_u=self.alpha_u)
                    else:
                        w_hard, w_soft = uknow_weights(theta, self.w_mode, alpha_u=self.alpha_u)

                hard = weighted_l1(pred, gt_ms, w_hard)
                lam_soft_t, lam_sis_t, lam_feat_t = self._sched(global_step)
                loss = hard + loss_pan
                extra = ""
                if self.variant == "k1a":
                    soft = F.l1_loss(pred, t_pred)
                    loss = loss + lam_soft_t * soft
                    extra += f"\tsoft(full): {soft.item():.5f}"
                elif self.variant in ("u_full", "u_full_gtvar"):
                    # uncertainty routing + full-output soft KD (s2 계획 §4)
                    soft = weighted_l1(pred, t_pred, w_soft)
                    loss = loss + lam_soft_t * soft
                    extra += (f"\tsoft(u_full): {soft.item():.5f}"
                              f"\twh {w_hard.mean().item():.2f}/{w_hard.std().item():.2f}"
                              f"\tws {w_soft.mean().item():.2f}/{w_soft.std().item():.2f}")
                    if self.variant == "u_full_gtvar":
                        l_gv, gd = gtvar_loss(pred - res_ms[B:], gt_ms - res_ms[B:], self.gtvar_kappa)
                        loss = loss + self.lam_gtvar * l_gv
                        extra += (f"\tGTVar: {l_gv.item():.5f}"
                                  f" (V_S {gd['v_s_mean']:.3f}±{gd['v_s_std']:.3f}"
                                  f" V_GT {gd['v_gt_mean']:.3f} r {gd['v_corr']:.3f})")
                elif self.variant in ("k1b", "k2", "k3", "k4", "k5"):
                    w_lr = None
                    if w_soft is not None:
                        w_lr = F.interpolate(w_soft, size=(pred.shape[-2] // 4,) * 2, mode="bilinear")
                    soft, _ = spectral_kd_loss(self.mtf(pred), self.mtf(t_pred), w_lr, self.eta_sam)
                    loss = loss + lam_soft_t * soft
                    extra += f"\tsoft(spec): {soft.item():.5f}"
                if self.variant in ("k4", "k5"):
                    l_sis, sd = sis_loss(self.mtf(pred), ms[B:], self.sis_radius,
                                         self.sis_mode, self.eta_sam)
                    loss = loss + lam_sis_t * l_sis
                    extra += f"\tSiS: {l_sis.item():.5f} (ctr {sd['center_shift_ratio']:.2f})"
                if self.variant == "k5":
                    f_s = self.s_tap.out["bottleneck_h4"]
                    f_t = self.t_tap.out["bottleneck_h4"]
                    w_b = None
                    if w_soft is not None:
                        w_b = F.interpolate(w_soft, size=f_s.shape[-2:], mode="bilinear")
                    l_feat = self.proj(f_s, f_t.detach(), w_b)
                    loss = loss + lam_feat_t * l_feat
                    extra += f"\tfeat: {l_feat.item():.5f}"

                self.guard_finite(loss, train_log, global_step)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size * (1 if self.ms_only else 2), loss.item(), hard.item(), loss_pan.item())
            global_step += 1
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t'
                                + report.result_str(lr, time.time() - start)
                                + f"\t[{self.variant}] λs {lam_soft_t:.3f}" + extra)
                start = time.time()
                report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step
        return global_step
