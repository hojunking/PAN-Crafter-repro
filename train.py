# --------------------------------------------------------
# PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
# Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee†, and Munchurl Kim†
#
# This code is released under the MIT License (see LICENSE file for details).
#
# This software is licensed for **non-commercial research and educational use only**.
# For commercial use, please contact: mkimee@kaist.ac.kr
# --------------------------------------------------------

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
from safetensors.torch import load_file

from tqdm import tqdm
import numpy as np
from scipy.io import savemat

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler

from utils import to_rgb, reduced_metrics, full_metrics, Train_Report, Test_Reduced_Report, Test_Full_Report


class Trainer:
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']
        self.test_full_data_loader = data_loader['test_full']

        # Accelerator
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            project_config=self.accelerator_project_config)

        if self.accelerator.is_main_process:
            if args.work_dir is not None:
                os.makedirs(args.work_dir, exist_ok=True)

        # Weight data type
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Model
        self.model = model

        # Optimizer
        # fix_local_attn 을 켜면 dep_conv 가 requires_grad=False 로 고정되므로 걸러낸다.
        # 전부 학습 대상이면 배포본과 동일한 param 목록이 된다.
        params_to_opt = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        # Scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup,
            num_training_steps=self.args.num_iter)

        # Accelerator
        # KNOWN_ISSUES.md C-1: 배포본은 model 만 prepare 해서 save_state 에 optimizer/scheduler
        # state 가 들어가지 않았고, 그래서 중간 재개가 불가능했다. 셋을 함께 등록하면
        # checkpoint 가 완전해진다. 단일 GPU 에서 학습 수학은 달라지지 않는다.
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler)

        self.last_reduced_metrics = {}
        self.last_full_metrics = {}
        self.last_val_metrics = {}

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.args.work_dir, f"epoch-{epoch}")
        self.accelerator.save_state(save_path)

    def save_best_model_reduced(self):
        save_path = os.path.join(self.args.work_dir, 'best_reduced')
        self.accelerator.save_state(save_path)

    def save_best_model_full(self):
        save_path = os.path.join(self.args.work_dir, 'best_full')
        self.accelerator.save_state(save_path)

    def save_val(self, pan, gt, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/val/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gt, gen, lms = to_rgb(pan), to_rgb(gt), to_rgb(gen), to_rgb(lms)
        generated_1 = np.concatenate((pan, lms), axis=2)
        generated_2 = np.concatenate((gen, gt), axis=2)
        generated_image = np.concatenate((generated_1, generated_2), axis=1)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def save_test_reduced(self, pan, gt, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/reduced/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gt, gen, lms = to_rgb(pan), to_rgb(gt), to_rgb(gen), to_rgb(lms)
        generated_1 = np.concatenate((pan, lms), axis=2)
        generated_2 = np.concatenate((gen, gt), axis=2)
        generated_image = np.concatenate((generated_1, generated_2), axis=1)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def save_test_full(self, pan, gen, lms, idx):
        path = os.path.join(self.args.work_dir, 'results/full/')
        if not os.path.exists(path):
            os.makedirs(path)

        pan, gen, lms = to_rgb(pan), to_rgb(gen), to_rgb(lms)
        generated_image = np.concatenate((pan, lms, gen), axis=2)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(f'{path}/{idx:04}.png')

    def train(self, train_log, global_step):
        self.model.train()
        self.model.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        # mars='ms' 는 PAN back-reconstruction mode 를 끈 단일 mode 학습이다 (M1 실험).
        # batch 복제가 없어져 step 당 연산이 절반이 된다. 논문 Table 16 은 이 제거가
        # 성능을 크게 해친다고 주장한다 — 그 주장의 재검증용이다.
        single_mode = getattr(self.args, "mars", "dual") == "ms"
        rep = 1 if single_mode else 2
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    gt = gt.to(self.accelerator.device, dtype=self.weight_dtype).repeat(rep, 1, 1, 1)
                    lms = lms.to(self.accelerator.device, dtype=self.weight_dtype).repeat(rep, 1, 1, 1)
                    ms = ms.to(self.accelerator.device, dtype=self.weight_dtype).repeat(rep, 1, 1, 1)
                    lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype).repeat(rep, 1, 1, 1)
                    pan = pan.to(self.accelerator.device, dtype=self.weight_dtype).repeat(rep, 1, 1, 1)

                    res_pan = F.interpolate(lpan, scale_factor=4, mode="bicubic")
                    # 배포본은 bicubic(ms,x4) 를 잔차 기준선으로 쓴다. 데이터셋이 제공하는
                    # lms(PanCollection 공식 보간)가 단독 기준선으로는 2.5% 낫다 — 진단용 옵션.
                    res_ms = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                              else F.interpolate(ms, scale_factor=4, mode="bicubic"))

                    switch_on = torch.ones((self.args.batch_size,), device=self.accelerator.device).to(dtype=self.weight_dtype)
                    if single_mode:
                        switch = switch_on
                    else:
                        switch_off = torch.zeros((self.args.batch_size,), device=self.accelerator.device).to(dtype=self.weight_dtype)
                        switch = torch.cat((switch_off, switch_on), dim=0)

                objective_recon = self.model(pan, lpan, ms, switch)

                if self.args.res:
                    objective_recon = objective_recon + res_ms * switch.view(-1, 1, 1, 1) + res_pan.repeat(1, self.args.num_bands, 1, 1) * (1.0 - switch).view(-1, 1, 1, 1)

                if self.args.loss_type == 'l1':
                    if single_mode:
                        loss_ms = (gt - objective_recon).abs().mean()
                        loss_pan = torch.zeros_like(loss_ms)   # 로그 호환용. PAN mode 없음
                        loss = loss_ms
                    else:
                        loss_pan = (pan[:self.args.batch_size].repeat(1, self.args.num_bands, 1, 1) - objective_recon[:self.args.batch_size]).abs().mean() * self.args.w_off
                        loss_ms = (gt[self.args.batch_size:] - objective_recon[self.args.batch_size:]).abs().mean()
                        loss = loss_pan + loss_ms
                else:
                    raise NotImplementedError()

                reduced_loss = self.accelerator.gather(loss).mean()
                reduced_loss_ms = self.accelerator.gather(loss_ms).mean()
                reduced_loss_pan = self.accelerator.gather(loss_pan).mean()

                # NaN/Inf 는 이후 모든 step 을 오염시킨다. 몇 시간 낭비하지 말고 즉시 죽는다.
                # exit 3 은 체인이 '재시도 무의미' 로 해석하는 코드다 (재개해도 다시 발산한다).
                if not torch.isfinite(loss):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}')
                    sys.exit(3)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size * rep, reduced_loss.item(), reduced_loss_ms.item(), reduced_loss_pan.item())

            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str(lr, period_time)
                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                save_path = os.path.join(self.args.work_dir, f'checkpoint-{global_step}')
                self.accelerator.save_state(save_path)

            if global_step >= self.args.num_iter:
                # KNOWN_ISSUES.md C-3: 바로 위 log_iter 블록이 report 를 리셋했다면
                # num_examples 가 0 이라 재출력 시 0/0 -> nan 이 찍힌다. 비어 있으면 건너뛴다.
                if report.num_examples > 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    period_time = time.time() - start
                    prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                    result_str = report.result_str(lr, period_time)
                    train_log.write(prefix_str + result_str)
                save_path = os.path.join(self.args.work_dir, 'lastest')
                self.accelerator.save_state(save_path)
                self.accelerator.end_training()
                return global_step

        return global_step

    def validate(self, test_log, epoch):
        """검증셋(valid_*.h5)으로 지표를 낸다. 체크포인트 선택 전용이다.

        배포본은 검증셋을 만들어만 두고 쓰지 않았고, best 를 테스트셋으로 골랐다
        (KNOWN_ISSUES.md E-1). 그러면 테스트 수치가 낙관적으로 부풀려진다.
        여기서는 테스트셋을 보지 않고 고른다.
        """
        report = Test_Reduced_Report()
        self.model.eval()
        self.model.requires_grad_(False)

        for gt, lms, ms, lpan, pan in tqdm(self.val_data_loader):
            with torch.no_grad():
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                b = pan.shape[0]
                switch = torch.ones(b, device=self.accelerator.device).to(dtype=self.weight_dtype)
                generated = self.model(pan, lpan, ms, switch)
                if self.args.res:
                    generated = generated + (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                                             else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                # tensor2img 가 batch=1 을 가정하므로 표본별로 계산한다
                for i in range(b):
                    metrics = reduced_metrics(x_true=gt[i:i + 1], x_pred=generated[i:i + 1],
                                              max_pixel=self.args.max_pixel)
                    report.update(1, metrics)

        prefix_str = f'Epoch[{epoch}] (val)\t'
        test_log.write(prefix_str + report.result_str())
        self.last_val_metrics = report.as_dict()
        return report.ergas

    def save_best_model_hqnr(self):
        save_path = os.path.join(self.args.work_dir, 'best_hqnr')
        self.accelerator.save_state(save_path)

    def save_best_model_val(self):
        save_path = os.path.join(self.args.work_dir, 'best_val')
        self.accelerator.save_state(save_path)

    def test_reduced(self, test_log, epoch):
        report = Test_Reduced_Report()
        self.model.eval()
        self.model.requires_grad_(False)

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                self.save_test_reduced(pan, gt, generated, F.interpolate(ms, scale_factor=4, mode="bicubic"), idx)
                metrics = reduced_metrics(x_true=gt, x_pred=generated, max_pixel=self.args.max_pixel)
                report.update(self.args.test_batch_size, metrics)

        prefix_str = f'Epoch[{epoch}]\t'
        result_str = report.result_str()   # compute_mean() 이 여기서 수행된다
        test_log.write(prefix_str + result_str)
        self.last_reduced_metrics = report.as_dict()
        return report.ergas

    def _fr_official_setup(self):
        """공식 DLPan 프로토콜(D_lambda_K + block-UQI D_s)을 학습 중 선택에 쓰기 위한 준비.

        utils 의 QNR 은 global-UIQI 근사라 공식 HQNR 과 순위가 다르다 (실측: paper_ln
        proxy 0.9265/공식 0.9360 vs paper_ln_mlp1 proxy 0.9170/공식 0.9388 — 역전).
        선택 기준은 반드시 공식 구현이어야 한다. 실패 시 조용히 proxy 로 떨어지지 않고
        죽는다 — 선택 기준이 뒤바뀐 채 5시간을 돌리는 것보다 낫다.
        """
        if hasattr(self, "_fr_official"):
            return self._fr_official
        import os as _os
        from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
        wald = load_dlpan(_os.environ.get("PANCRAFTER_DLPAN",
                                          "/home/knuvi/Desktop/song/DLPan-Toolbox"))
        root = self.args.test_full_feeder_args.get("dataroot", "")
        sensor = next((s for s in ("wv3", "qb", "gf2", "wv2") if s in _os.path.basename(root)), "wv3")
        lo, hi = (int(x) for x in getattr(self.args, "fr_select_indices", "12-19").split("-"))
        self._fr_official = (wald, d_lambda_k, d_s, sensor, lo, hi)
        return self._fr_official

    def test_full(self, test_log, epoch):
        wald, f_dl, f_ds, sensor, fr_lo, fr_hi = self._fr_official_setup()
        dl_off, ds_off = [], []
        report = Test_Full_Report()
        self.model.eval()
        self.model.requires_grad_(False)
        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                self.save_test_full(pan, generated, F.interpolate(ms, scale_factor=4, mode="bicubic"), idx)
                metrics = full_metrics(x_pred=generated, pan=pan, ms=ms, max_pixel=self.args.max_pixel)
                report.update(self.args.test_batch_size, metrics)

                # 공식 HQNR: 논문 대조 프로토콜과 같은 index 부분집합만 계산한다
                if fr_lo <= idx <= fr_hi:
                    # feeder 는 [-1,1] 정규화다. mat 내보내기(tensor2img)와 동일한
                    # 역변환((x+1)/2 -> clamp -> x max_pixel -> round)을 써야
                    # 사후 평가(eval_dlpan_fr)와 같은 값이 나온다.
                    def _dn(ten):
                        a = ((ten + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy()
                        return np.round(a.astype(np.float64) * self.args.max_pixel)
                    sr_np = _dn(generated[0]).transpose(1, 2, 0)
                    lms_np = _dn(lms[0]).transpose(1, 2, 0)
                    pan_np = _dn(pan[0, 0])
                    dl_off.append(f_dl(sr_np, lms_np, sensor, 4, 32, wald))
                    ds_off.append(f_ds(sr_np, lms_np, pan_np, 4, 32, wald))

        hqnr_official = float((1 - np.mean(dl_off)) * (1 - np.mean(ds_off)))
        prefix_str = f'Epoch[{epoch}]\t'
        result_str = report.result_str()   # compute_mean() 이 여기서 수행된다
        test_log.write(prefix_str + result_str
                       + f'\tHQNR_official({fr_lo}-{fr_hi}): {hqnr_official:.6f}')
        self.last_full_metrics = report.as_dict()
        self.last_full_metrics['hqnr_official'] = hqnr_official
        return report.d_s, hqnr_official

    def test_reduced_save(self, tag='best_reduced'):
        self.model.eval()
        self.model.requires_grad_(False)

        gt_list = []
        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                gt_list.append(gt)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        gt_save = (torch.cat(gt_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=gt_save * self.args.max_pixel,
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/reduced_{tag}.mat', d)

    def test_full_save(self, tag='best_reduced'):
        self.model.eval()
        self.model.requires_grad_(False)

        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/full_{tag}.mat', d)

    def test_reduced_save_full(self, tag='best_full'):
        self.model.eval()
        self.model.requires_grad_(False)

        gt_list = []
        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                gt_list.append(gt)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        gt_save = (torch.cat(gt_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=gt_save * self.args.max_pixel,
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/reduced_{tag}.mat', d)

    def test_full_save_full(self, tag='best_full'):
        self.model.eval()
        self.model.requires_grad_(False)

        pan_list = []
        lms_list = []
        ms_list = []
        sr_list = []

        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            with torch.no_grad():
                lms = lms.to(self.accelerator.device, dtype=self.weight_dtype)
                ms = ms.to(self.accelerator.device, dtype=self.weight_dtype)
                lpan = lpan.to(self.accelerator.device, dtype=self.weight_dtype)
                pan = pan.to(self.accelerator.device, dtype=self.weight_dtype)
                switch = torch.ones(self.args.test_batch_size, device=self.accelerator.device).to(dtype=self.weight_dtype)
                if self.args.res:
                    base = (lms if getattr(self.args, "residual_base", "bicubic") == "lms"
                            else F.interpolate(ms, scale_factor=4, mode="bicubic"))
                    generated = self.model(pan, lpan, ms, switch) + base
                else:
                    generated = self.model(pan, lpan, ms, switch)
                pan_list.append(pan)
                lms_list.append(lms)
                ms_list.append(ms)
                sr_list.append(generated)

        pan_save = (torch.cat(pan_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        lms_save = (torch.cat(lms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        ms_save = (torch.cat(ms_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2
        sr_save = (torch.cat(sr_list, dim=0).clip(-1.0, 1.0).detach().cpu().numpy() + 1.0) / 2

        path = os.path.join(self.args.work_dir, 'results/')
        if not os.path.exists(path):
            os.makedirs(path)

        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            ms=ms_save * self.args.max_pixel,
            lms=lms_save * self.args.max_pixel,
            pan=pan_save * self.args.max_pixel,
            sr=sr_save * self.args.max_pixel
        )

        savemat(f'{path}/full_{tag}.mat', d)