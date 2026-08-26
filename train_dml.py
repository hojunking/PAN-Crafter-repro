"""Teacher–Teacher Deep Mutual Learning 조인트 트레이너.

`research_log/s2_teacher_teacher_mutual_learning_plan.md` 구현이다.

설계 원칙 — **`train.py` 를 건드리지 않는다.** s1 과 공유하는 파일이라 수정하면
그쪽 실행이 위험해진다. 대신 `Trainer` 인스턴스를 두 개 만들어 평가·저장·mat 내보내기를
그대로 재사용하고, 학습 스텝만 여기서 조인트로 돈다.

두 peer 는 **같은 배치·같은 증강**을 본다 (계획 5절). 하나의 dataloader 를 공유하므로
자동으로 성립한다. 서로 다른 rotation/flip 이 걸리면 출력 좌표계가 달라져
output-level mutual loss 자체가 성립하지 않는다.
"""
import os, time, torch
import torch.nn.functional as F
from tqdm import tqdm


def lambda_at(step, lam_max, warmup, ramp_end):
    """계획 6절의 스케줄. warmup 까지 0, ramp_end 까지 선형 증가, 이후 lam_max."""
    if step < warmup:
        return 0.0
    if step < ramp_end:
        return lam_max * (step - warmup) / max(1, ramp_end - warmup)
    return lam_max


class DMLRunner:
    """두 Trainer 를 받아 조인트 학습 스텝을 돈다. 평가는 각 Trainer 에 위임한다."""

    def __init__(self, args, peer_a, peer_b, train_loader):
        self.args = args
        self.A, self.B = peer_a, peer_b
        self.loader = train_loader
        self.lam_max = float(getattr(args, "mutual_lambda", 0.0))
        self.warmup = int(getattr(args, "mutual_warmup", 2500))
        self.ramp_end = int(getattr(args, "mutual_ramp_end", 5000))
        self.diag_every = int(getattr(args, "mutual_diag_iter", 500))
        self.dev = self.A.accelerator.device
        self.dtype = self.A.weight_dtype
        self.diag = []          # 진단 로그 (계획 12절)

    # ---------------------------------------------------------------- 손실
    def _anchor(self, recon, gt, pan):
        """배포본과 동일한 MARs 손실. 앞 절반 PAN mode, 뒤 절반 MS mode."""
        b = self.args.batch_size
        loss_pan = (pan[:b].repeat(1, self.args.num_bands, 1, 1)
                    - recon[:b]).abs().mean() * self.args.w_off
        loss_ms = (gt[b:] - recon[b:]).abs().mean()
        return loss_pan + loss_ms, loss_ms, loss_pan

    def _grad_norm(self, model):
        s = 0.0
        for p in model.parameters():
            if p.grad is not None:
                s += p.grad.detach().float().pow(2).sum().item()
        return s ** 0.5

    # ---------------------------------------------------------------- 스텝
    def train(self, train_log, global_step):
        # train.py 의 평가 메서드는 model.requires_grad_(False) 로 grad 를 영구히 끈다
        # (train.py:221 등). 단일 모델 경로는 Trainer.train() 이 매 epoch 시작에
        # requires_grad_(True) 로 되살린다(train.py:133-134). 여기서도 똑같이 해야 한다 —
        # 빠뜨리면 첫 평가 직후 손실에 grad_fn 이 없어 backward 가 죽는다.
        for tr in (self.A, self.B):
            tr.model.train()
            tr.model.requires_grad_(True)
        t_start = time.time()
        n_seen = 0
        acc = dict(la=0.0, lb=0.0, ma=0.0, mb=0.0, dis=0.0)

        for gt, lms, ms, lpan, pan in tqdm(self.loader):
            b = self.args.batch_size
            with torch.no_grad():
                gt, lms, ms, lpan, pan = [
                    t.to(self.dev, dtype=self.dtype).repeat(2, 1, 1, 1)
                    for t in (gt, lms, ms, lpan, pan)]
                res_pan = F.interpolate(lpan, scale_factor=4, mode="bicubic")
                res_ms = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                          else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                switch = torch.cat((torch.zeros(b, device=self.dev, dtype=self.dtype),
                                    torch.ones(b, device=self.dev, dtype=self.dtype)), dim=0)
                sw = switch.view(-1, 1, 1, 1)

            def head(model):
                out = model(pan, lpan, ms, switch)
                if self.args.res:
                    out = out + res_ms * sw + res_pan.repeat(1, self.args.num_bands, 1, 1) * (1.0 - sw)
                return out

            # 두 그래프가 동시에 살아 있어야 한다 — 서로의 출력을 detach 해 타깃으로 쓴다
            rec_a, rec_b = head(self.A.model), head(self.B.model)
            anc_a, ms_a, pan_a = self._anchor(rec_a, gt, pan)
            anc_b, ms_b, pan_b = self._anchor(rec_b, gt, pan)

            lam = lambda_at(global_step, self.lam_max, self.warmup, self.ramp_end)
            # 계획 5절: mutual 은 MS mode 의 HRMS 에만 건다. PAN mode 는 anchor 로만 학습.
            mut_a = (rec_a[b:] - rec_b[b:].detach()).abs().mean()
            mut_b = (rec_b[b:] - rec_a[b:].detach()).abs().mean()
            loss_a = anc_a + lam * mut_a
            loss_b = anc_b + lam * mut_b

            # gradient norm 분리 측정은 여기서 하지 않는다.
            # retain_graph 로 두 backward 동안 그래프를 살리면 peak VRAM 이 30 -> 34 GiB 를
            # 넘어 OOM 난다 (실측). r_g 는 스텝마다 필요한 값이 아니라 lambda 보정용 상수이므로
            # tools/dml_calibrate.py 에서 작은 배치로 따로 잰다 (계획 6절).
            (loss_a + loss_b).backward()

            self.A.optimizer.step(); self.A.lr_scheduler.step()
            self.B.optimizer.step(); self.B.lr_scheduler.step()
            self.A.optimizer.zero_grad(set_to_none=True)
            self.B.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                acc['la'] += float(anc_a); acc['lb'] += float(anc_b)
                acc['ma'] += float(mut_a); acc['mb'] += float(mut_b)
                acc['dis'] += float((rec_a[b:] - rec_b[b:]).abs().mean())
            if self.diag_every and global_step % self.diag_every == 0:
                self.diag.append(dict(step=global_step, lam=lam,
                                      anchor_a=float(anc_a), anchor_b=float(anc_b),
                                      mutual=float(mut_a),
                                      disagreement=float((rec_a[b:] - rec_b[b:]).abs().mean()),
                                      corr=float(torch.corrcoef(torch.stack([
                                          rec_a[b:].flatten(), rec_b[b:].flatten()]))[0, 1])))
            n_seen += 1
            global_step += 1

            if global_step % self.args.log_iter == 0:
                k = max(1, n_seen)
                train_log.write(
                    f'Iter[{global_step}/{self.args.num_iter}]\t'
                    f'A anchor {acc["la"]/k:.7f}\tB anchor {acc["lb"]/k:.7f}\t'
                    f'A mut {acc["ma"]/k:.7f}\tB mut {acc["mb"]/k:.7f}\t'
                    f'lambda {lam:.4f}\tdisagree {acc["dis"]/k:.6f}\t'
                    f'LR {self.A.lr_scheduler.get_last_lr()[0]:.7f}\t'
                    f'Time {time.time()-t_start:.2f}')
                acc = dict(la=0.0, lb=0.0, ma=0.0, mb=0.0, dis=0.0); n_seen = 0
                t_start = time.time()

            if global_step >= self.args.num_iter:
                break
        return global_step

    def save_diag(self, path):
        import csv
        if not self.diag:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.diag[0].keys()))
            w.writeheader(); w.writerows(self.diag)
