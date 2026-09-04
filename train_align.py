"""AlignTrainer — global alignment 캠페인용 trainer (계획 §9·§11~§15·§17·§21).

Trainer 와 다른 점만 담당한다:
  - AlignedModel(backbone + 선택적 ShiftNet) 로 감싼다. 코어 U-Net 은 그대로.
  - delta 를 표본당 한 번 얻어(zero/cache/trainable) MS·PAN 두 mode 에 복제한다 (§9.3).
    PAN mode 로 가는 delta 는 detach — PAN loss 가 ShiftNet 을 흔들지 못한다.
  - MS loss 는 inverse warp 가 있으면 border mask 로 정규화한다 (§7.2). PAN loss 는 마스크 없음.
  - 평가: 최종 frame 으로 공식 HQNR(12-19) + fSCC(12-19). 다른 frame 뷰(y_pan / y_ms)는
    *_alt 로 함께 기록한다 (§12.5·§13.5). RR 1차 지표는 M-frame 뷰(y_loss).
  - best 선택은 main.py: HQNR(1e-4) -> fSCC(1e-4) -> 나중 iteration (§17.2).
  - trainable case 는 ShiftNet pretrain(§15.3) 을 자동 실행/검증하고 gate 실패면 exit 3.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from align.cache import ShiftCache
from align.model import AlignCfg, AlignedModel
from align.resample import masked_l1, transform_delta
from train import Trainer
from utils import (Train_Report, Test_Reduced_Report, Test_Full_Report,
                   reduced_metrics, full_metrics, tensor2img, SCC_full_numpy, SAM_numpy, ERGAS_numpy)

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


class AlignTrainer(Trainer):
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']
        self.test_full_data_loader = data_loader['test_full']
        assert getattr(args, "mars", "dual") == "dual", "계획 §0.1: dual MARs 고정"
        assert args.res, "잔차 학습 고정"
        assert args.feeder == "feeders.feeder_align.PanFeederAlign", "feeder 는 PanFeederAlign 이어야 한다 (meta 반환)"
        self.acfg = AlignCfg.from_dict(getattr(args, "alignment", {}) or {})

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision,
                                       project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.model = AlignedModel(model, self.acfg)

        # shift cache (frozen 또는 trainable 의 pseudo-label)
        self.cache = None
        if self.acfg.delta_source in ("cache", "trainable"):
            self.cache = ShiftCache(os.path.join(ROOT, self.acfg.cache_dir))
            for code in (0, 2, 3):
                assert self.cache.has(code), f"cache 에 split {code} 가 없다"

        if self.acfg.trainable_shift_net:
            self._load_or_pretrain_shiftnet()

        # optimizer: backbone / shift_net 은 별도 param group (§15.4)
        groups = [dict(params=[p for p in self.model.backbone.parameters() if p.requires_grad],
                       lr=args.learning_rate, weight_decay=args.weight_decay)]
        if self.model.shift_net is not None:
            groups.append(dict(params=list(self.model.shift_net.parameters()),
                               lr=self.acfg.shiftnet_lr, weight_decay=self.acfg.shiftnet_wd))
        self.optimizer = torch.optim.AdamW(groups)
        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=args.num_warmup,
                                          num_training_steps=args.num_iter)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler)

        self.last_reduced_metrics, self.last_full_metrics, self.last_val_metrics = {}, {}, {}
        self.last_fscc_official = float("nan")
        self._ema = {}
        # resume provenance: 이전 best 가 다른 cache/upsampler 로 만들어졌으면 이어 돌리지 않는다
        bm = os.path.join(args.work_dir, "best_hqnr_meta.json")
        if getattr(args, "resume", None) and os.path.exists(bm):
            prev = json.load(open(bm))
            cur = self.cache.sha256_all if self.cache else None
            if prev.get("cache_sha256") != cur or prev.get("upsampler") != self.acfg.upsampler:
                print(f"[align] resume 거부 — best_hqnr_meta 의 cache/upsampler 가 현재와 다르다: "
                      f"{prev.get('cache_sha256')}/{prev.get('upsampler')} vs {cur}/{self.acfg.upsampler}")
                sys.exit(4)

    # ------------------------------------------------------------------ helpers
    @property
    def M(self):
        return self.accelerator.unwrap_model(self.model)

    def _load_or_pretrain_shiftnet(self):
        ck = os.path.join(ROOT, self.acfg.shiftnet_pretrained)
        rep = ck.replace(".pt", ".json")
        if not os.path.exists(ck):
            print(f"[align] ShiftNet pretrain 실행 -> {ck}")
            r = subprocess.run([PY, os.path.join(ROOT, "tools", "pretrain_shiftnet.py"),
                                "--cache-dir", os.path.join(ROOT, self.acfg.cache_dir), "--out", ck], cwd=ROOT)
            if r.returncode != 0 or not os.path.exists(ck):
                print("[align] ShiftNet pretrain 실패 — C4 를 시작하지 않는다 (계획 §15.3)")
                sys.exit(4)          # 4 = gate 불통과 (체인: 학습 시작 안 함·재시도 없음. 3=NaN 과 구분)
        info = json.load(open(rep)) if os.path.exists(rep) else {}
        if not info.get("pass", False):
            print(f"[align] ShiftNet pretrain gate FAIL {info} — C4 를 시작하지 않는다 (계획 §15.3)")
            sys.exit(4)
        sd = torch.load(ck, map_location="cpu")
        self.model.shift_net.load_state_dict(sd)
        print(f"[align] ShiftNet pretrained 로드: median err {info.get('val_median_err'):.4f} "
              f"P90 {info.get('val_p90_err'):.4f} sign {info.get('val_sign_acc'):.3f}")

    def _delta(self, lpan, ms, meta, train):
        """표본당 한 번. 반환 (delta[B,2], accepted[B] bool, target[B,2] or None)."""
        B, dev, dt = ms.shape[0], ms.device, ms.dtype
        src = self.acfg.delta_source
        if src == "zero":
            return torch.zeros(B, 2, dtype=dt, device=dev), torch.zeros(B, dtype=torch.bool, device=dev), None
        d, acc = self.cache.lookup(meta[:, 1], meta[:, 0], device=dev)
        d = d.to(dt)                     # 원본 frame. 증강 frame 으로의 변환은 inverse 직전에만 (train 참고)
        if src == "cache":
            return d, acc, None
        pred = self.M.predict_delta(lpan, ms).to(dt)          # trainable
        return pred, acc, d

    def _ema_update(self, k, v, m=0.98):
        self._ema[k] = v if k not in self._ema else m * self._ema[k] + (1 - m) * v

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.model.train()
        self.model.requires_grad_(True)
        report = Train_Report()
        start = time.time()
        B, C = self.args.batch_size, self.args.num_bands
        M = self.M
        for idx, (gt, lms, ms, lpan, pan, meta) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                dev, dt = self.accelerator.device, self.weight_dtype
                gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (gt, ms, lpan, pan))
                meta = meta.to(dev)
                delta, acc, tgt = self._delta(lpan, ms, meta, train=True)
                # dual MARs: [PAN mode(switch 0) | MS mode(switch 1)], shift 는 복제 (§9.3)
                ms2, lpan2, pan2 = ms.repeat(2, 1, 1, 1), lpan.repeat(2, 1, 1, 1), pan.repeat(2, 1, 1, 1)
                delta_dual = torch.cat([delta.detach(), delta], dim=0)
                switch = torch.cat([torch.zeros(B, device=dev, dtype=dt), torch.ones(B, device=dev, dtype=dt)])
                # 증강은 HR 에서, upsample(+shift) 뒤에 (feeder 는 LR 을 원본으로 준다)
                aug = (meta[:, 2], meta[:, 3], meta[:, 4])
                v = M.build_views(pan2, lpan2, ms2, delta_dual, aug=tuple(t.repeat(2) for t in aug))
                res = M.residual(v["x11"], switch)
                # PAN mode — inverse 없음, 마스크 없음, target 은 입력 PAN 복제 (§9.2)
                p_hat = v["lpan_hr"][:B].repeat(1, C, 1, 1) + res[:B]
                loss_pan = (pan.repeat(1, C, 1, 1) - p_hat).abs().mean() * self.args.w_off
                # MS mode
                # inverse/mask 는 증강된 출력 frame 에서 — Δ 를 같은 frame 으로 변환 (T05·T12)
                fin = M.finalize_ms(v["ms_base_hr"][B:], res[B:], transform_delta(delta, *aug))
                loss_ms = masked_l1(fin["y_loss"], gt, fin["mask"])
                loss = loss_pan + loss_ms
                loss_shift = loss_zero = torch.zeros((), device=dev)
                if self.acfg.trainable_shift_net:
                    if acc.any():
                        loss_shift = F.smooth_l1_loss(delta[acc], tgt[acc], beta=0.05)
                    if (~acc).any():
                        loss_zero = delta[~acc].abs().mean()
                    loss = loss + self.acfg.lambda_shift * loss_shift + self.acfg.lambda_zero * loss_zero

                if not torch.isfinite(loss):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}')
                    sys.exit(3)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(B * 2, loss.item(), loss_ms.item(), loss_pan.item())
                    with torch.no_grad():
                        dd = delta.detach().float()
                        mag = dd.norm(dim=1)
                        for k, val in (("loss_shift", loss_shift.item()), ("loss_zero", loss_zero.item()),
                                       ("dy_mean", dd[:, 0].mean().item()), ("dy_std", dd[:, 0].std().item()),
                                       ("dx_mean", dd[:, 1].mean().item()), ("dx_std", dd[:, 1].std().item()),
                                       ("mag_p50", mag.median().item()), ("mag_p90", mag.quantile(0.9).item()),
                                       ("accepted_ratio", acc.float().mean().item()),
                                       ("valid_mask_ratio", fin["mask"].mean().item())):
                            self._ema_update(k, val)
            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                e = self._ema
                extra = (f"\tshift: {e.get('loss_shift', 0):.4f}\tzero: {e.get('loss_zero', 0):.4f}"
                         f"\tdy {e.get('dy_mean', 0):+.3f}±{e.get('dy_std', 0):.3f}"
                         f"\tdx {e.get('dx_mean', 0):+.3f}±{e.get('dx_std', 0):.3f}"
                         f"\t|d| p50 {e.get('mag_p50', 0):.3f} p90 {e.get('mag_p90', 0):.3f}"
                         f"\tacc {e.get('accepted_ratio', 0):.2f}\tmask {e.get('valid_mask_ratio', 1):.3f}")
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, period_time) + extra)
                start = time.time()
                report.__init__()

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
    def _infer(self, lms, ms, lpan, pan, meta):
        dev, dt = self.accelerator.device, self.weight_dtype
        ms, lpan, pan = (t.to(dev, dtype=dt) for t in (ms, lpan, pan))
        meta = meta.to(dev)
        with torch.no_grad():
            delta, acc, _ = self._delta(lpan, ms, meta, train=False)
            out = self.M.infer_ms(pan, lpan, ms, delta)
        out["delta"], out["accepted"], out["pan"], out["ms"] = delta, acc, pan, ms
        out["alt"] = out["y_ms"] if self.acfg.output_frame == "P" else out["y_pan"]   # 최종이 아닌 frame 뷰
        return out

    def validate(self, test_log, epoch):
        # select_on=val 이 아닌 한 호출되지 않는다. cache 가 없는 split 이라 delta 는 zeros/ShiftNet.
        return super().validate.__wrapped__(self, test_log, epoch) if hasattr(super().validate, "__wrapped__") \
            else float("nan")

    def test_reduced(self, test_log, epoch):
        report, alt_e, alt_s = Test_Reduced_Report(), [], []
        self.model.eval(); self.model.requires_grad_(False)
        for idx, (gt, lms, ms, lpan, pan, meta) in tqdm(enumerate(self.test_reduced_data_loader)):
            out = self._infer(lms, ms, lpan, pan, meta)
            gt = gt.to(self.accelerator.device, dtype=self.weight_dtype)
            y = out["y_loss"]                       # RR 1차 지표는 M-frame 뷰 (§13.5)
            self.save_test_reduced(out["pan"], gt, y, out["ms_base_hr"], idx)
            report.update(self.args.test_batch_size, reduced_metrics(x_true=gt, x_pred=y, max_pixel=self.args.max_pixel))
            if out["alt"] is not None:
                a, b = tensor2img(gt, self.args.max_pixel), tensor2img(out["alt"], self.args.max_pixel)
                alt_e.append(ERGAS_numpy(a, b)); alt_s.append(SAM_numpy(a, b))
        test_log.write(f'Epoch[{epoch}]\t' + report.result_str())
        self.last_reduced_metrics = report.as_dict()
        if alt_e:
            self.last_reduced_metrics.update(ergas_alt=float(np.mean(alt_e)), sam_alt=float(np.mean(alt_s)))
        return report.ergas

    def _official(self, y, lms, pan, f_dl, f_ds, sensor, wald):
        def _dn(ten):
            a = ((ten + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy()
            return np.round(a.astype(np.float64) * self.args.max_pixel)
        sr_np, lms_np, pan_np = _dn(y[0]).transpose(1, 2, 0), _dn(lms[0]).transpose(1, 2, 0), _dn(pan[0, 0])
        dl = f_dl(sr_np, lms_np, sensor, 4, 32, wald)
        ds = f_ds(sr_np, lms_np, pan_np, 4, 32, wald)
        fscc = float(SCC_full_numpy(tensor2img(pan, self.args.max_pixel), tensor2img(y, self.args.max_pixel)))
        return dl, ds, fscc

    def test_full(self, test_log, epoch):
        wald, f_dl, f_ds, sensor, fr_lo, fr_hi = self._fr_official_setup()
        dl_off, ds_off, fs_off, dl_alt, ds_alt, fs_alt, mags, accs = [], [], [], [], [], [], [], []
        report = Test_Full_Report()
        self.model.eval(); self.model.requires_grad_(False)
        for idx, (lms, ms, lpan, pan, meta) in tqdm(enumerate(self.test_full_data_loader)):
            out = self._infer(lms, ms, lpan, pan, meta)
            lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
            y = out["y_final"]
            self.save_test_full(out["pan"], y, out["ms_base_hr"], idx)
            report.update(self.args.test_batch_size, full_metrics(x_pred=y, pan=out["pan"], ms=out["ms"], max_pixel=self.args.max_pixel))
            if fr_lo <= idx <= fr_hi:
                dl, ds, fs = self._official(y, lms, out["pan"], f_dl, f_ds, sensor, wald)
                dl_off.append(dl); ds_off.append(ds); fs_off.append(fs)
                mags.append(out["delta"].norm(dim=1).mean().item()); accs.append(out["accepted"].float().mean().item())
                if out["alt"] is not None:
                    dl2, ds2, fs2 = self._official(out["alt"], lms, out["pan"], f_dl, f_ds, sensor, wald)
                    dl_alt.append(dl2); ds_alt.append(ds2); fs_alt.append(fs2)
        # 장면별 (1-D_l)(1-D_s) 의 평균 — tools/eval_dlpan_fr.py·DLPan·계획 §17.2 와 같은 식.
        # (1-mean D_l)(1-mean D_s) 와는 ~1e-5 차이지만 tie band 1e-4 안이라 같은 식을 써야 한다.
        hqnr = float(np.mean((1 - np.array(dl_off)) * (1 - np.array(ds_off))))
        hqnr_pm = float((1 - np.mean(dl_off)) * (1 - np.mean(ds_off)))
        fscc = float(np.mean(fs_off))
        line = report.result_str() + f'\tHQNR_official({fr_lo}-{fr_hi}): {hqnr:.6f} (prod-of-means {hqnr_pm:.6f})' \
            + f'\tfSCC({fr_lo}-{fr_hi}): {fscc:.6f}' \
            + f'\tD_l_off {np.mean(dl_off):.5f} D_s_off {np.mean(ds_off):.5f}\t|delta| {np.mean(mags):.3f} acc {np.mean(accs):.2f}'
        if dl_alt:
            h_alt = float(np.mean((1 - np.array(dl_alt)) * (1 - np.array(ds_alt))))
            line += f'\t[alt frame] HQNR {h_alt:.6f} fSCC {np.mean(fs_alt):.6f}'
        test_log.write(f'Epoch[{epoch}]\t' + line)
        self.last_full_metrics = report.as_dict()
        self.last_full_metrics.update(hqnr_official=hqnr, d_lambda_official=float(np.mean(dl_off)),
                                      d_s_official=float(np.mean(ds_off)), fscc_official=fscc,
                                      delta_mag_mean=float(np.mean(mags)), accepted_ratio=float(np.mean(accs)))
        if dl_alt:
            self.last_full_metrics.update(hqnr_alt=h_alt, fscc_alt=float(np.mean(fs_alt)))
        self.last_fscc_official = fscc
        return report.d_s, hqnr

    # ------------------------------------------------------------------ export
    def write_best_meta(self, epoch, global_step, hqnr):
        m = dict(iteration=int(global_step), epoch=int(epoch), hqnr=float(hqnr),
                 fscc=float(self.last_fscc_official),
                 d_lambda=self.last_full_metrics.get("d_lambda_official"),
                 d_s=self.last_full_metrics.get("d_s_official"),
                 hqnr_alt=self.last_full_metrics.get("hqnr_alt"), fscc_alt=self.last_full_metrics.get("fscc_alt"),
                 final_frame=self.acfg.output_frame, shift_source=self.acfg.delta_source,
                 alpha=float(self.acfg.alpha), inverse_location=self.acfg.inverse_location,
                 upsampler=self.acfg.upsampler,
                 cache_sha256=(self.cache.sha256_all if self.cache else None))
        json.dump(m, open(os.path.join(self.args.work_dir, "best_hqnr_meta.json"), "w"), indent=1)

    def _collect(self, loader, has_gt):
        self.model.eval(); self.model.requires_grad_(False)
        acc = dict(pan=[], lms=[], ms=[], gt=[], final=[], msframe=[], alt=[])
        for batch in tqdm(loader):
            if has_gt:
                gt, lms, ms, lpan, pan, meta = batch
            else:
                lms, ms, lpan, pan, meta = batch; gt = None
            out = self._infer(lms, ms, lpan, pan, meta)
            acc["pan"].append(out["pan"]); acc["lms"].append(lms.to(out["pan"].device)); acc["ms"].append(out["ms"])
            acc["final"].append(out["y_final"]); acc["msframe"].append(out["y_loss"])
            if out["alt"] is not None:
                acc["alt"].append(out["alt"])
            if gt is not None:
                acc["gt"].append(gt.to(out["pan"].device))
        return acc

    def _savemat(self, name, **arrays):
        from scipy.io import savemat
        path = os.path.join(self.args.work_dir, 'results/')
        os.makedirs(path, exist_ok=True)
        cv = lambda L: (torch.cat(L, 0).clip(-1, 1).detach().cpu().numpy() + 1.0) / 2 * self.args.max_pixel
        savemat(os.path.join(path, name), {k: cv(v) for k, v in arrays.items() if v})

    def test_reduced_save(self, tag='best_hqnr'):
        a = self._collect(self.test_reduced_data_loader, has_gt=True)
        # 1차(시트·평가) 파일의 sr 는 M-frame 뷰 — GT 와 같은 좌표계 (§13.5·§18)
        self._savemat(f'reduced_{tag}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], gt=a["gt"], sr=a["msframe"])
        if self.acfg.full_shift:
            other = "panframe" if self.acfg.output_frame == "M" else "panframe"
            self._savemat(f'reduced_{tag}_{other}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], gt=a["gt"],
                          sr=(a["final"] if self.acfg.output_frame == "P" else a["alt"]))

    def test_full_save(self, tag='best_hqnr'):
        a = self._collect(self.test_full_data_loader, has_gt=False)
        # 1차 파일의 sr 는 배포(최종) frame — 선택에 쓴 HQNR 과 같은 텐서
        self._savemat(f'full_{tag}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], sr=a["final"])
        if self.acfg.full_shift and a["alt"]:
            other = "msframe" if self.acfg.output_frame == "P" else "panframe"
            self._savemat(f'full_{tag}_{other}.mat', ms=a["ms"], lms=a["lms"], pan=a["pan"], sr=a["alt"])
