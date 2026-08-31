# --------------------------------------------------------
# Mutual learning trainer (M0~M3) — 기존 Trainer 서브클래스. 명세 Part A.
#
# 설계:
#   - peer_a = 기존 경로의 self.model (평가·best 선택·export 의 1차 대상)
#   - peer_b = 같은 model_args 로 다른 init 을 받아 같은 accelerator 에 prepare
#     -> save_state/load_state 가 두 peer + 두 optimizer 를 함께 저장·복원한다
#   - 같은 augmented batch 를 두 peer 가 공유 (명세 §3.1 — 좌표 정합 필수)
#   - mutual target 은 반드시 detach (명세 §14 — gradient isolation)
#   - best 선택: 두 peer HQNR 의 **평균** (test_full 오버라이드), per-peer 값은
#     로그로 남긴다. export 는 peer_a 기본 + *_peerB 접미 mat 추가.
#   - checkpoint 선택 기준은 기존과 동일한 공식 HQNR (불변).
# --------------------------------------------------------

import os
import time

import torch
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from train import Trainer
from utils import Train_Report
from train_kd import DualBatchMixin
from kd.ops import MTFDownsampler, AbsoluteGradient
from kd.losses import (sis_loss, edge_loss, mutual_residual, mutual_spectral,
                       mutual_edge)


def mutual_weight(step, total=50_000, max_w=0.05):
    """명세 §13 스케줄: 5K warm-up, 15K 까지 ramp, 40K plateau, 이후 decay."""
    if step < 5_000:
        return 0.0
    if step < 15_000:
        return max_w * (step - 5_000) / 10_000
    if step < 40_000:
        return max_w
    return max_w * max(0.0, total - step) / 10_000


class MutualTrainer(DualBatchMixin, Trainer):
    LADDER = ["m0", "m1", "m2", "m3"]

    def __init__(self, args, data_loader, model):
        ma = dict(args.mutual_args or {})
        self.variant = ma.get("variant", "m0")
        assert self.variant in self.LADDER, f"mutual variant: {self.variant}"
        self.lam_mutual = float(ma.get("lambda_mutual", 0.02))
        self.lam_edge_self = float(ma.get("lambda_edge_self", 0.05))
        self.lam_sis_self = float(ma.get("lambda_sis_self", 0.10))
        self.lam_m2p = float(ma.get("lambda_m_to_p_spec", 0.05))
        self.lam_p2m = float(ma.get("lambda_p_to_m_edge", 0.02))
        self.sis_radius = int(ma.get("sis_radius", 1))
        self.sis_mode = ma.get("sis_mode", "shared_vector")
        self.eta_sam = float(ma.get("eta_sam", 0.1))

        super().__init__(args, data_loader, model)          # peer_a 경로

        # peer_b: parameter init seed 만 다르게 (명세 §10) — DataLoader 는 공유
        from main import import_class
        torch.manual_seed(args.seed + 1)
        peer_b = import_class(args.model)(**args.model_args)
        opt_b = torch.optim.AdamW([p for p in peer_b.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
        sched_b = get_scheduler(args.lr_scheduler, optimizer=opt_b,
                                num_warmup_steps=args.num_warmup,
                                num_training_steps=args.num_iter)
        self.peer_b, self.opt_b, self.sched_b = self.accelerator.prepare(peer_b, opt_b, sched_b)
        torch.manual_seed(args.seed)                        # 이후 흐름은 원래 seed 로

        dev, dt = self.accelerator.device, self.weight_dtype
        self.mtf = MTFDownsampler(bands=args.num_bands).to(dev, dtype=dt)
        self.grad_op = AbsoluteGradient().to(dev, dtype=dt)
        self.last_reduced_metrics_b = {}
        self.last_full_metrics_b = {}

    # ---------------------------------------------------------------- 학습
    def _mars(self, recon, gt, pan, B):
        loss_pan = (pan[:B].repeat(1, self.args.num_bands, 1, 1)
                    - recon[:B]).abs().mean() * self.args.w_off
        loss_ms = (gt[B:] - recon[B:]).abs().mean()
        return loss_ms + loss_pan, loss_ms, loss_pan

    def train(self, train_log, global_step):
        self.model.train(); self.model.requires_grad_(True)
        self.peer_b.train(); self.peer_b.requires_grad_(True)
        report = Train_Report()
        start = time.time()
        for idx, batch in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    gt, lms, ms, lpan, pan, res_pan, res_ms, switch, B = self.prep(batch)
                out_a = self.model(pan, lpan, ms, switch)
                out_b = self.peer_b(pan, lpan, ms, switch)
                rec_a = self.recon_of(out_a, res_ms, res_pan, switch)
                rec_b = self.recon_of(out_b, res_ms, res_pan, switch)
                mars_a, ms_a, pan_a = self._mars(rec_a, gt, pan, B)
                mars_b, ms_b, pan_b = self._mars(rec_b, gt, pan, B)
                loss_a, loss_b = mars_a, mars_b
                lam_t = mutual_weight(global_step, self.args.num_iter, self.lam_mutual)
                extra = ""

                if self.variant in ("m2", "m3"):
                    # peer_a = P(edge), peer_b = M(SiS) — 명세 §12
                    l_edge_a, _ = edge_loss(rec_a[B:], pan[B:], self.grad_op)
                    l_sis_b, sd = sis_loss(self.mtf(rec_b[B:]), ms[B:],
                                           self.sis_radius, self.sis_mode, self.eta_sam)
                    loss_a = loss_a + self.lam_edge_self * l_edge_a
                    loss_b = loss_b + self.lam_sis_self * l_sis_b
                    extra += f"\tedgeA {l_edge_a.item():.5f}\tsisB {l_sis_b.item():.5f}"

                if self.variant == "m1" and lam_t > 0:
                    l_ab = mutual_residual(out_a[B:], out_b[B:].detach())
                    l_ba = mutual_residual(out_b[B:], out_a[B:].detach())
                    loss_a = loss_a + lam_t * l_ab
                    loss_b = loss_b + lam_t * l_ba
                    extra += f"\tmutual {l_ab.item():.5f}/{l_ba.item():.5f}"
                elif self.variant == "m3" and lam_t > 0:
                    # M(SiS peer_b) -> P(peer_a) 로 spectral, P -> M 으로 edge
                    l_m2p = mutual_spectral(self.mtf(rec_a[B:]),
                                            self.mtf(rec_b[B:]).detach(), self.eta_sam)
                    l_p2m = mutual_edge(rec_b[B:], rec_a[B:].detach(), self.grad_op)
                    loss_a = loss_a + lam_t * (self.lam_m2p / self.lam_mutual) * l_m2p
                    loss_b = loss_b + lam_t * (self.lam_p2m / self.lam_mutual) * l_p2m
                    extra += f"\tm2p {l_m2p.item():.5f}\tp2m {l_p2m.item():.5f}"

                total = loss_a + loss_b
                self.guard_finite(total, train_log, global_step)
                self.accelerator.backward(total)
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()
                self.opt_b.step(); self.sched_b.step(); self.opt_b.zero_grad()
                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size * 2, total.item(), ms_a.item(), pan_a.item())
            global_step += 1
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                with torch.no_grad():
                    dis = (out_a[B:] - out_b[B:]).abs().mean().item()   # 명세 §16 핵심 신호
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t'
                                + report.result_str(lr, time.time() - start)
                                + f"\t[{self.variant}] λm {lam_t:.4f}\tdisagree {dis:.5f}"
                                + extra)
                start = time.time()
                report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'lastest'))
                self.accelerator.end_training()
                return global_step
        return global_step

    # ------------------------------------------------- 평가: 두 peer 모두
    def _swapped(self, fn, *a, **kw):
        keep = self.model
        self.model = self.peer_b
        try:
            return fn(*a, **kw)
        finally:
            self.model = keep

    def test_reduced(self, test_log, epoch):
        test_log.write('[peer_a]')
        ergas_a = super().test_reduced(test_log, epoch)
        met_a = dict(self.last_reduced_metrics)
        test_log.write('[peer_b]')
        ergas_b = self._swapped(super().test_reduced, test_log, epoch)
        self.last_reduced_metrics_b = dict(self.last_reduced_metrics)
        self.last_reduced_metrics = met_a                 # 1차 기록은 peer_a 유지
        return (ergas_a + ergas_b) / 2

    def test_full(self, test_log, epoch):
        test_log.write('[peer_a]')
        ds_a, hqnr_a = super().test_full(test_log, epoch)
        met_a = dict(self.last_full_metrics)
        test_log.write('[peer_b]')
        ds_b, hqnr_b = self._swapped(super().test_full, test_log, epoch)
        self.last_full_metrics_b = dict(self.last_full_metrics)
        self.last_full_metrics = met_a
        test_log.write(f'[mutual] HQNR peerA {hqnr_a:.6f}  peerB {hqnr_b:.6f}  '
                       f'mean {(hqnr_a + hqnr_b) / 2:.6f}')
        # best 선택은 두 peer 평균 HQNR — save_state 가 두 peer 를 함께 저장한다
        return (ds_a + ds_b) / 2, (hqnr_a + hqnr_b) / 2

    def validate(self, test_log, epoch):
        return super().validate(test_log, epoch)

    # ------------------------------------------------- export: peer_b 병기
    def test_reduced_save(self, tag='best_reduced'):
        super().test_reduced_save(tag=tag)
        self._swapped(super().test_reduced_save, tag=tag + '_peerB')

    def test_full_save(self, tag='best_reduced'):
        super().test_full_save(tag=tag)
        self._swapped(super().test_full_save, tag=tag + '_peerB')
