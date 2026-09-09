"""PATrainer — PAN 앞단 전역 정합 A1/A2/A3 (research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md).

B0(W96·D124 · MS+PAN 9ch · mars ms · mode_modulation false) 의 학습 경로에 aligner 하나를 앞에 붙인다.
  Ŷ = M + Fθ(concat(P̃, M)),  M = bicubic↑S,  P̃ = W(P, Aφ(P, M))          (§2)
  A1: L_rec / A2: L_rec + λE(t) L_edge / A3: L_rec + λG(t) L_geo,  λ(t) = λ·min(t/5000, 1)   (§5.4)
학습 loss 영역은 B0 와 같은 전 영역(L_rec), 고정 interior(L_edge 1px, L_geo margin 11). 평가 mask 는 loss 에 닿지 않는다 (§6·§10.7).
support guard 는 L_geo 가 실제로 참조하는 P̃ 영역(margin − Gaussian radius − Scharr radius = [4:60])까지 검사한다 (검토 지적 1).
평가(test_full): 같은 forward 의 같은 SR 에 세 view(raw_original / raw_valid / aligned_valid) + RR(original·valid), 두 선택기(best_raw → best_hqnr alias,
best_aligned) 와 last 를 보존한다 (§10). 참조(lms·pan)는 h5 원본 float64, P̃_eval = W(P_raw, Δ̂) (float64) — 시트 evaluator 와 같은 경로.
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from pa.aligner import PANGlobalAligner
from pa.losses import output_edge_loss, direct_geometry_loss, lambda_ramp, scharr, geometry_support_margin
from pa.model import PAModel
from pa.warp import warp_support_mask, support_margin_ok, warp_pan
from pa.evalviews import scene_views, roi_manifest, PROTOCOL_ID, MAX_ELIGIBLE_SHIFT, VIEWS, evaluator_hash, fixed_roi
from pa.selector import BestSelector
from train import Trainer
from utils import Train_Report, Test_Reduced_Report, Test_Full_Report, reduced_metrics, full_metrics

ROOT = os.path.dirname(os.path.abspath(__file__))
CASES = {"A1": dict(lambda_edge=0.0, lambda_geo=0.0), "A2": dict(lambda_edge=0.1, lambda_geo=0.0), "A3": dict(lambda_edge=0.0, lambda_geo=0.01)}
EXIT_SUPPORT_FAIL = 4


def _sha_tensors(t):
    h = hashlib.sha256()
    for k in sorted(t):
        h.update(k.encode()); h.update(t[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def sha256_file(path, cache_dir=None):
    """파일 sha256 (경로·크기·mtime 으로 캐시 — 학습셋 h5 는 GB 단위)."""
    st = os.stat(path); key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    cp = os.path.join(cache_dir, "sha256_cache.json") if cache_dir else None
    cache = json.load(open(cp)) if cp and os.path.exists(cp) else {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    cache[key] = h.hexdigest()
    if cp:
        json.dump(cache, open(cp, "w"), indent=1)
    return cache[key]


class PATrainer(Trainer):
    def __init__(self, args, data_loader, model):
        self.args = args
        self.train_data_loader = data_loader['train']; self.val_data_loader = data_loader['val']
        self.test_reduced_data_loader = data_loader['test_reduced']; self.test_full_data_loader = data_loader['test_full']
        assert getattr(args, "mars", "dual") == "ms" and args.res, "PA 는 단일 HRMS task(mars: ms) · 잔차 base 고정 (§0.1)"
        assert args.model_args.get("in_mode") == "paper" and not args.model_args.get("attn_locations"), "입력 9ch(in_mode paper) · attention 없음"
        assert args.model_args.get("mode_modulation", True) is False, "MARs γ/β 제거본(mode_modulation false) 위에서만 (§9.1)"
        pa = getattr(args, "pa", {}) or {}
        self.case = pa.get("case"); assert self.case in CASES, f"pa.case 는 {list(CASES)}"
        self.lam_edge = float(pa.get("lambda_edge", CASES[self.case]["lambda_edge"]))
        self.lam_geo = float(pa.get("lambda_geo", CASES[self.case]["lambda_geo"]))
        assert (self.case == "A1") == (self.lam_edge == 0.0 and self.lam_geo == 0.0), "A1 은 보조 loss 없음"
        assert not (self.lam_edge > 0 and self.lam_geo > 0), "A3 에 A2 loss 를 함께 넣지 않는다 (§5.4)"
        self.ramp = int(pa.get("ramp_steps", 5000))
        self.geo_sigma = float(pa.get("geometry_sigma_hr", 2.0))            # r/2, WV3 r=4
        self.geo_margin = int(pa.get("geometry_margin_hr", 11))
        self.guard_margin = geometry_support_margin(self.geo_sigma, self.geo_margin)     # 11 − (6+1) = 4
        self.diag_iter = int(pa.get("diag_iter", 500))
        self.init_dir = os.path.join(ROOT, pa.get("init_dir", "work_dir/_pa_init"))

        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process and args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # §7.3 초기 가중치 대응: U-Net 은 main.py 가 init_seed(seed) 뒤 만든 그 tensor(=B0 와 같은 경로). aligner 는 RNG 를 소비하지
        # 않도록 fork_rng 안에서 seed+1000 으로 만든다(data/augmentation RNG 순서 보존). seed 별 init 파일을 저장·대조한다.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed) + 1000)
            aligner = PANGlobalAligner(int(args.num_bands))
        self.init_hashes = self._pair_init(model, aligner)
        self.model = PAModel(model, aligner)
        groups = [dict(params=[p for p in self.model.backbone.parameters() if p.requires_grad], name="backbone"),
                  dict(params=list(self.model.aligner.parameters()), name="aligner")]      # 같은 LR/scheduler (§7.2)
        self.optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=args.num_warmup, num_training_steps=args.num_iter)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        self.last_reduced_metrics, self.last_full_metrics, self.last_val_metrics = {}, {}, {}
        self.last_fscc_official = float("nan"); self.raw_is_best = False
        self._ema = {}; self._global_step = 0; self.step_records = {}
        self.cand_dir = os.path.join(args.work_dir, "candidates")
        # 평가 참조: h5 원본 float64 (시트 evaluator 와 같은 값). feeder 의 float32 정규화 왕복을 쓰지 않는다 (검토 지적 3).
        self.fr_h5 = args.test_full_feeder_args["dataroot"]; self.fr_h5_sha = sha256_file(self.fr_h5, self.init_dir)
        with h5py.File(self.fr_h5) as f:
            self._fr_lms = np.asarray(f["lms"], dtype=np.float64); self._fr_pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
        self._roi = None
        self._init_selectors()
        if self.accelerator.is_main_process:
            json.dump(self.init_hashes, open(os.path.join(args.work_dir, "initialization_hashes.json"), "w"), indent=1)
            json.dump(dict(case=self.case, lambda_edge=self.lam_edge, lambda_geo=self.lam_geo, ramp_steps=self.ramp, geometry_sigma_hr=self.geo_sigma,
                           geometry_margin_hr=self.geo_margin, support_guard_margin_hr=self.guard_margin, warp="bicubic/border/align_corners=False, no zero bypass",
                           loss_domain="L_rec full frame (B0)", aligner_params=sum(p.numel() for p in aligner.parameters()), evaluator_hash=evaluator_hash()),
                      open(os.path.join(args.work_dir, "pa_config_resolved.json"), "w"), indent=1)
            self._write_manifests()

    # ------------------------------------------------------------------ init pairing · manifests · selectors
    def _pair_init(self, unet, aligner):
        os.makedirs(self.init_dir, exist_ok=True)
        seed = int(self.args.seed); out = {}
        for name, mod in (("unet", unet), ("aligner", aligner)):
            p = os.path.join(self.init_dir, f"init_{name}_seed{seed}.pt")
            sd = {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
            h = _sha_tensors(sd)
            if os.path.exists(p):
                ref = torch.load(p, map_location="cpu"); hr = _sha_tensors(ref)
                if hr != h:
                    mod.load_state_dict(ref); h = hr; out[f"{name}_loaded_from_file"] = True     # 같은 seed 의 다른 case 와 맞춘다 (T19)
            else:
                torch.save(sd, p)
            out[f"{name}_init_sha256_16"] = h; out[f"{name}_init_file"] = p
        out["seed"] = seed
        return out

    def _write_manifests(self):
        a = self.args; wd = a.work_dir
        ds = {}
        for k in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
            p = getattr(a, k)["dataroot"]; ds[k] = dict(path=p, sha256=sha256_file(p, self.init_dir))
            pp = p.replace(".h5", "_pan.h5")
            if os.path.exists(pp):
                ds[k + "_pan"] = dict(path=pp, sha256=sha256_file(pp, self.init_dir))
        json.dump(ds, open(os.path.join(wd, "dataset_hashes.json"), "w"), indent=1)
        b0 = f"BASE_W96_D124_MSPAN_WV3_S{int(a.seed)}"; b0d = os.path.join(ROOT, "work_dir", b0)
        m = dict(b0_run_id=b0, b0_config=os.path.join(ROOT, "config", b0 + ".yaml"), b0_exists=os.path.isdir(b0d),
                 b0_finished=os.path.exists(os.path.join(b0d, "finished_at.txt")), init_hashes=self.init_hashes,
                 b0_best_hqnr_model_sha256=(sha256_file(os.path.join(b0d, "best_hqnr", "model.safetensors")) if os.path.exists(os.path.join(b0d, "best_hqnr", "model.safetensors")) else None),
                 b0_best_state=(json.load(open(os.path.join(b0d, "best_state.json"))) if os.path.exists(os.path.join(b0d, "best_state.json")) else None),
                 note="U-Net init 은 B0 와 같은 경로(init_seed(seed) 뒤 build)로 만들었다; B0 는 초기 tensor 를 저장하지 않아 동치는 경로로만 대응한다",
                 selection_scene_ids=list(range(20)), report_scene_ids=list(range(20)), fr_h5=self.fr_h5, fr_h5_sha256=self.fr_h5_sha, protocol_id=PROTOCOL_ID)
        json.dump(m, open(os.path.join(wd, "baseline_manifest.json"), "w"), indent=1)

    def _init_selectors(self):
        wd = self.args.work_dir
        self.sel_raw = BestSelector("raw"); self.sel_aligned = BestSelector("aligned")
        paths = [os.path.join(wd, f"selector_state_{n}.json") for n in ("raw", "aligned")]
        if getattr(self.args, "resume", None):
            expect = dict(protocol_id=PROTOCOL_ID, evaluator_hash=evaluator_hash(), fr_h5_sha256=self.fr_h5_sha, n_scenes=len(self._fr_pan))
            for s, p in zip((self.sel_raw, self.sel_aligned), paths):
                if os.path.exists(p):
                    loaded = BestSelector.load(p, expect=expect)
                    s.max_hqnr, s.cands, s.best, s.history = loaded.max_hqnr, loaded.cands, loaded.best, loaded.history
            rp = os.path.join(wd, "selection_roi_manifest.json")
            if os.path.exists(rp):
                self._roi = json.load(open(rp))
                if self._roi.get("evaluator_hash") != evaluator_hash() or self._roi.get("input_sha256") != self.fr_h5_sha:
                    raise RuntimeError("selection_roi_manifest 가 현재 evaluator/데이터와 다르다 — 새 protocol 로 처음부터 돌릴 것 (§13)")
        elif self.accelerator.is_main_process:
            stale = [p for p in paths + [self.cand_dir, os.path.join(wd, "selection_roi_manifest.json")] if os.path.exists(p)]
            if stale:                                    # --resume 없이 같은 work_dir 를 다시 쓰는 경우: 이전 선택 상태를 옆으로 치운다 (검토 지적 5)
                d = os.path.join(wd, f"_stale_selection_{time.strftime('%Y%m%d-%H%M%S')}"); os.makedirs(d, exist_ok=True)
                for p in stale:
                    shutil.move(p, os.path.join(d, os.path.basename(p)))

    # ------------------------------------------------------------------ helpers
    @property
    def M(self):
        return self.accelerator.unwrap_model(self.model)

    def _ema_update(self, k, v, m=0.98):
        self._ema[k] = v if k not in self._ema else m * self._ema[k] + (1 - m) * v

    def _lams(self, step):
        return lambda_ramp(self.lam_edge, step, self.ramp), lambda_ramp(self.lam_geo, step, self.ramp)

    @staticmethod
    def _grad_energy(x):
        gx, gy = scharr(x.float())
        return float((gx ** 2 + gy ** 2).mean())

    def _jsonl(self, name, rec):
        with open(os.path.join(self.args.work_dir, name), "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ train
    def train(self, train_log, global_step):
        self.model.train(); self.model.requires_grad_(True)
        report = Train_Report(); start = time.time()
        B = self.args.batch_size; dev, dt = self.accelerator.device, self.weight_dtype
        M = self.M
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.model):
                gt, ms, lpan, pan = (t.to(dev, dtype=dt) for t in (gt, ms, lpan, pan))
                o = M(pan, ms, lpan)
                y, delta, pan_al = o["y"], o["delta"], o["pan_aligned"]
                H, W = pan.shape[-2:]
                # §6.2 support guard — 모든 case 동일. L_geo 가 참조하는 [guard_margin:H-guard_margin] 의 sampling 이웃이 전부 영상 안이어야 한다.
                ok = support_margin_ok(H, W, delta.detach(), self.guard_margin)
                if not bool(ok.all()):
                    d = delta.detach().float()
                    train_log.write(f'[SUPPORT_FAIL] step {global_step}: |Δ| max {d.abs().max().item():.2f} px 가 geometry support(margin {self.guard_margin}) 를 깼다 ({int((~ok).sum())}/{B} sample)')
                    json.dump(dict(status="SUPPORT_FAIL", step=int(global_step), max_abs_delta=float(d.abs().max()), guard_margin=self.guard_margin),
                              open(os.path.join(self.args.work_dir, "support_fail.json"), "w"))
                    sys.exit(EXIT_SUPPORT_FAIL)
                loss_rec = (gt - y).abs().mean()                                     # §5.1 B0 와 같은 전 영역 L1
                lam_e, lam_g = self._lams(global_step)
                loss_edge = output_edge_loss(y, gt) if self.lam_edge > 0 else torch.zeros((), device=dev)
                geo_info = None
                if self.lam_geo > 0:
                    with torch.autocast(device_type=dev.type, enabled=False):
                        loss_geo, geo_info = direct_geometry_loss(pan_al, pan, gt, self.geo_sigma, self.geo_margin)
                else:
                    loss_geo = torch.zeros((), device=dev)
                loss_aux = lam_e * loss_edge + lam_g * loss_geo
                loss = loss_rec + loss_aux
                if not torch.isfinite(loss):
                    train_log.write(f'[abort] non-finite loss at step {global_step}: {loss.item()}'); sys.exit(3)
                if self.accelerator.is_main_process and global_step % self.diag_iter == 0:     # §13 aligner 가 rec / aux 에서 받는 gradient
                    ap = [p for p in M.aligner.parameters()]
                    g_rec = torch.autograd.grad(loss_rec, ap, retain_graph=True, allow_unused=True)
                    n_rec = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_rec if g is not None)))
                    n_aux = 0.0
                    if self.lam_edge > 0 or self.lam_geo > 0:
                        g_aux = torch.autograd.grad(loss_aux, ap, retain_graph=True, allow_unused=True)
                        n_aux = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_aux if g is not None)))
                    self._jsonl("gradient_diagnostics.jsonl", dict(step=int(global_step), aligner_grad_from_rec=n_rec, aligner_grad_from_aux=n_aux,
                                                                    ratio_aux_over_rec=(n_aux / n_rec if n_rec > 0 else None)))
                    self._ema["g_rec"], self._ema["g_aux"] = n_rec, n_aux
                self.accelerator.backward(loss)
                if self.accelerator.is_main_process and global_step % self.diag_iter == 0:
                    bn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.backbone.parameters() if p.grad is not None)))
                    an = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in M.aligner.parameters() if p.grad is not None)))
                    self._ema["backbone_grad_norm"], self._ema["aligner_grad_norm"] = bn, an
                self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(B, loss.item(), loss_rec.item(), loss_aux.item())        # 'Loss PAN' 칸 = 보조 loss (PA 에는 PAN task 없음)
                    with torch.no_grad():
                        d = delta.detach().float(); n = d.norm(dim=1)
                        for k, v in (("dy_mean", d[:, 0].mean().item()), ("dy_std", d[:, 0].std().item()), ("dx_mean", d[:, 1].mean().item()),
                                     ("dx_std", d[:, 1].std().item()), ("abs_max", d.abs().max().item()), ("norm_p50", n.median().item()),
                                     ("norm_p95", torch.quantile(n, 0.95).item()), ("loss_rec", loss_rec.item()), ("loss_edge", float(loss_edge)),
                                     ("loss_geo", float(loss_geo)), ("aux_ratio", (loss_aux / (loss_rec + 1e-12)).item()), ("lam_e", lam_e), ("lam_g", lam_g)):
                            self._ema_update(k, v)
                        if global_step % 50 == 0:
                            self._ema_update("valid_frac", warp_support_mask(H, W, d).float().mean().item(), 0.9)
                            self._ema_update("ge_before", self._grad_energy(pan), 0.9); self._ema_update("ge_after", self._grad_energy(pan_al), 0.9)
                        if geo_info is not None:
                            self._ema_update("geo_wsum", geo_info["weight_sum"]); self._ema["geo_zero"] = self._ema.get("geo_zero", 0) + geo_info["zero_weight_samples"]
            global_step += 1; self._global_step = global_step
            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']; e = self._ema
                extra = (f"\t[{self.case}] rec {e.get('loss_rec', 0):.5f} edge {e.get('loss_edge', 0):.5f} geo {e.get('loss_geo', 0):.5f} "
                         f"lamE {e.get('lam_e', 0):.3f} lamG {e.get('lam_g', 0):.4f} aux/rec {e.get('aux_ratio', 0):.3f}"
                         f"\tΔ dy {e.get('dy_mean', 0):+.3f}±{e.get('dy_std', 0):.3f} dx {e.get('dx_mean', 0):+.3f}±{e.get('dx_std', 0):.3f} "
                         f"|Δ| p50 {e.get('norm_p50', 0):.3f} p95 {e.get('norm_p95', 0):.3f} max {e.get('abs_max', 0):.3f}"
                         f"\tvalid {e.get('valid_frac', 1):.4f} gE {e.get('ge_before', 0):.4g}->{e.get('ge_after', 0):.4g}"
                         f"\tgrad bb {e.get('backbone_grad_norm', 0):.3g} al {e.get('aligner_grad_norm', 0):.3g} (rec {e.get('g_rec', 0):.3g} aux {e.get('g_aux', 0):.3g})")
                train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start) + extra)
                self._jsonl("train_log.jsonl", dict(step=int(global_step), lr=lr, **{k: float(v) for k, v in e.items()}, step_time_s=(time.time() - start) / max(1, self.args.log_iter),
                                                     peak_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None)))
                start = time.time(); report.__init__()
            if global_step % self.args.save_iter == 0:
                self.accelerator.save_state(os.path.join(self.args.work_dir, f'checkpoint-{global_step}'))
            if global_step >= self.args.num_iter:
                if report.num_examples > 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    train_log.write(f'Iter[{global_step}/{self.args.num_iter}]\t' + report.result_str(lr, time.time() - start))
                self.accelerator.save_state(os.path.join(self.args.work_dir, 'last'))       # §10.8 정확한 마지막 update
                json.dump(dict(step=int(global_step), kind="last"), open(os.path.join(self.args.work_dir, "last_meta.json"), "w"))
                self.accelerator.end_training()
                return global_step
        return global_step

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _infer(self, pan, lpan, ms, delta_override=None):
        dev, dt = self.accelerator.device, self.weight_dtype
        ms, lpan, pan = (t.to(dev, dtype=dt) for t in (ms, lpan, pan))
        o = self.M(pan, ms, lpan, delta_override=delta_override)
        o["pan"] = pan; o["ms"] = ms
        return o

    def test_reduced(self, test_log, epoch):
        report = Test_Reduced_Report(); rep_v = Test_Reduced_Report(); self.model.eval(); self.model.requires_grad_(False)
        ds = []
        for idx, (gt, lms, ms, lpan, pan) in tqdm(enumerate(self.test_reduced_data_loader)):
            o = self._infer(pan, lpan, ms)
            gt = gt.to(self.accelerator.device, dtype=self.weight_dtype)
            self.save_test_reduced(o["pan"], gt, o["y"], o["ms_base"], idx)
            report.update(self.args.test_batch_size, reduced_metrics(x_true=gt, x_pred=o["y"], max_pixel=self.args.max_pixel))
            H, W = gt.shape[-2:]
            if H > 2 * fixed_roi(512, 512)[0]:                                       # RR-valid: 같은 margin 규칙(64px) — 256² 이면 128² (§10.7)
                y0, y1, x0, x1 = fixed_roi(H, W)
                rep_v.update(self.args.test_batch_size, reduced_metrics(x_true=gt[..., y0:y1, x0:x1], x_pred=o["y"][..., y0:y1, x0:x1], max_pixel=self.args.max_pixel))
            ds.append(o["delta"][0].float().cpu().numpy())
        d = np.array(ds)
        test_log.write(f'Epoch[{epoch}]\t' + report.result_str() + f'\tRR Δ median ({np.median(d[:, 0]):+.3f},{np.median(d[:, 1]):+.3f}) |Δ| median {np.median(np.linalg.norm(d, axis=1)):.3f}'
                       + (f'\t[RR-valid] ' + rep_v.result_str() if rep_v.num_examples else ''))
        self.last_reduced_metrics = report.as_dict()
        self.last_reduced_metrics.update(rr_dy_median=float(np.median(d[:, 0])), rr_dx_median=float(np.median(d[:, 1])))
        if rep_v.num_examples:
            self.last_reduced_metrics.update({f"valid_{k}": v for k, v in rep_v.as_dict().items()})
        return report.ergas

    def _dn_sr(self, t):
        return ((t.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * self.args.max_pixel).astype(np.float64)

    def test_full(self, test_log, epoch):
        wald, _, _, sensor, lo, hi = self._fr_official_setup()
        step = self._global_step
        report = Test_Full_Report(); self.model.eval(); self.model.requires_grad_(False)
        per = {v: dict(d_lambda=[], d_s=[], hqnr=[], fscc=[]) for v in VIEWS}
        deltas, elig, reasons, rows = [], [], [], []
        for idx, (lms, ms, lpan, pan) in tqdm(enumerate(self.test_full_data_loader)):
            o = self._infer(pan, lpan, ms)
            self.save_test_full(o["pan"], o["y"], o["ms_base"], idx)
            report.update(self.args.test_batch_size, full_metrics(x_pred=o["y"], pan=o["pan"], ms=o["ms"], max_pixel=self.args.max_pixel))
            if not (lo <= idx <= hi):
                continue
            sr = self._dn_sr(o["y"][0]).transpose(1, 2, 0)
            lm = self._fr_lms[idx].transpose(1, 2, 0); p = self._fr_pan[idx]                # h5 원본 float64 참조
            d = o["delta"][0].float().cpu().numpy()
            pa_eval = warp_pan(torch.from_numpy(p)[None, None], torch.from_numpy(d.astype(np.float64))[None])[0, 0].numpy()   # P̃_eval = W(P_raw, Δ̂), float64
            if self._roi is None:
                self._roi = roi_manifest(p.shape[0], p.shape[1], hi - lo + 1, self.fr_h5, self.fr_h5_sha)
                if self.accelerator.is_main_process:
                    json.dump(self._roi, open(os.path.join(self.args.work_dir, "selection_roi_manifest.json"), "w"), indent=1)
            views, ok, reason = scene_views(sr, lm, p, pa_eval, sensor, wald, d, 4, float(self.args.max_pixel))
            deltas.append(d); elig.append(ok); reasons.append(reason)
            for v in VIEWS:
                for k in per[v]:
                    per[v][k].append(views[v][k])
                rows.append(dict(step=step, epoch=epoch, scene=idx, view=v, roi_scope=("original" if v == "raw_original" else "selection_fixed_V"),
                                 pan_reference=("P_aligned" if v == "aligned_valid" else "P"), d_lambda=views[v]["d_lambda"], d_s=views[v]["d_s"],
                                 hqnr=views[v]["hqnr"], fscc=views[v]["fscc"], dy_hr=float(d[0]), dx_hr=float(d[1]), delta_norm_hr=float(np.hypot(*d)),
                                 selection_eligible=bool(ok), invalid_reason=reason, roi_hash=self._roi["roi_hash"]))
        agg = {v: {k: float(np.mean(per[v][k])) for k in per[v]} for v in VIEWS}       # HQNR = 장면별 곱의 평균 (§10.4)
        D = np.array(deltas); all_ok = bool(np.all(elig)); n_bad = int(np.sum(~np.array(elig)))
        bad_reasons = sorted({r for r in reasons if r})
        hqnr, fscc = agg["raw_original"]["hqnr"], agg["raw_original"]["fscc"]
        line = (report.result_str() + f'\tHQNR_official({lo}-{hi}): {hqnr:.6f}\tfSCC({lo}-{hi}): {fscc:.6f}\tD_l_off {agg["raw_original"]["d_lambda"]:.5f} D_s_off {agg["raw_original"]["d_s"]:.5f}'
                f'\t[views] raw_valid HQNR {agg["raw_valid"]["hqnr"]:.6f} fSCC {agg["raw_valid"]["fscc"]:.4f} | aligned_valid HQNR {agg["aligned_valid"]["hqnr"]:.6f} fSCC {agg["aligned_valid"]["fscc"]:.4f}'
                f' eligible {"all" if all_ok else f"INVALID({n_bad} scenes: {bad_reasons})"}'
                f'\tΔ median ({np.median(D[:, 0]):+.3f},{np.median(D[:, 1]):+.3f}) |Δ| median {np.median(np.linalg.norm(D, axis=1)):.3f} max {np.abs(D).max():.3f}')
        self.last_full_metrics = report.as_dict()
        self.last_full_metrics.update(hqnr_official=hqnr, d_lambda_official=agg["raw_original"]["d_lambda"], d_s_official=agg["raw_original"]["d_s"], fscc_official=fscc,
                                      hqnr_raw_valid=agg["raw_valid"]["hqnr"], hqnr_aligned_valid=agg["aligned_valid"]["hqnr"], fscc_raw_valid=agg["raw_valid"]["fscc"],
                                      fscc_aligned_valid=agg["aligned_valid"]["fscc"], d_s_aligned_valid=agg["aligned_valid"]["d_s"], aligned_eligible=float(all_ok),
                                      fr_dy_median=float(np.median(D[:, 0])), fr_dx_median=float(np.median(D[:, 1])), fr_delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))))
        self.last_fscc_official = fscc
        self.step_records[step] = dict(step=step, epoch=epoch, hqnr=hqnr, fscc=fscc, scc=self.last_reduced_metrics.get("scc"), ergas=self.last_reduced_metrics.get("ergas"),
                                       sam=self.last_reduced_metrics.get("sam"), hqnr_aligned_valid=agg["aligned_valid"]["hqnr"])
        test_log.write(f'Epoch[{epoch}]\t' + line)
        if self.accelerator.is_main_process:
            self._record(step, epoch, rows, agg, D, all_ok, n_bad, bad_reasons)
            self._select(step, epoch, agg, all_ok, n_bad, bad_reasons, test_log)
        return report.d_s, hqnr

    # ------------------------------------------------------------------ records · selection
    def _csv_append(self, name, rows):
        if not rows:
            return
        p = os.path.join(self.args.work_dir, name); new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    def _record(self, step, epoch, rows, agg, D, all_ok, n_bad, bad_reasons):
        self._csv_append("scene_metrics.csv", rows)
        rec = dict(step=step, epoch=epoch)
        for v in VIEWS:
            for k in ("hqnr", "d_lambda", "d_s", "fscc"):
                rec[f"{v}.{k}"] = agg[v][k]
        rec.update(dy_median=float(np.median(D[:, 0])), dx_median=float(np.median(D[:, 1])), delta_norm_median=float(np.median(np.linalg.norm(D, axis=1))),
                   delta_abs_max=float(np.abs(D).max()), aligned_eligible=int(all_ok), n_invalid_scenes=n_bad, invalid_reasons="|".join(bad_reasons),
                   region_effect=agg["raw_valid"]["hqnr"] - agg["raw_original"]["hqnr"], reference_effect=agg["aligned_valid"]["hqnr"] - agg["raw_valid"]["hqnr"],
                   rr_ergas=self.last_reduced_metrics.get("ergas"), rr_scc=self.last_reduced_metrics.get("scc"), rr_sam=self.last_reduced_metrics.get("sam"),
                   rr_valid_ergas=self.last_reduced_metrics.get("valid_ergas"), rr_valid_scc=self.last_reduced_metrics.get("valid_scc"))
        self._csv_append("checkpoint_metrics.csv", [rec])
        self._csv_append("delta_predictions.csv", [dict(step=step, scene=i, dy_hr=float(D[i, 0]), dx_hr=float(D[i, 1])) for i in range(len(D))])

    def _select(self, step, epoch, agg, all_ok, n_bad, bad_reasons, test_log):
        cand = os.path.join(self.cand_dir, f"step-{step}")
        self.accelerator.save_state(cand)                                            # band 판정 전 저장, 밖이면 아래서 정리
        r_raw = self.sel_raw.update(step, epoch, agg["raw_original"]["hqnr"], agg["raw_original"]["fscc"], True, cand)
        r_al = self.sel_aligned.update(step, epoch, agg["aligned_valid"]["hqnr"], agg["aligned_valid"]["fscc"], all_ok, cand,
                                       reason=("" if all_ok else f"{n_bad} scenes: {bad_reasons}"))
        keep = {c["path"] for c in self.sel_raw.cands} | {c["path"] for c in self.sel_aligned.cands}
        for p in set(r_raw["pruned"]) | set(r_al["pruned"]) | ({cand} if cand not in keep else set()):
            if p not in keep and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        self.raw_is_best = r_raw["changed"]
        if r_al["changed"]:
            self._materialize(self.sel_aligned.best, "best_aligned", "aligned_valid")
        extra = dict(protocol_id=PROTOCOL_ID, roi_hash=(self._roi or {}).get("roi_hash"), evaluator_hash=evaluator_hash(), fr_h5_sha256=self.fr_h5_sha, n_scenes=len(self._fr_pan))
        self.sel_raw.save(os.path.join(self.args.work_dir, "selector_state_raw.json"), extra)
        self.sel_aligned.save(os.path.join(self.args.work_dir, "selector_state_aligned.json"), extra)
        if self.sel_aligned.best is None:
            json.dump(dict(status="no_valid_candidate", history=self.sel_aligned.history[-5:]), open(os.path.join(self.args.work_dir, "best_aligned_meta.json"), "w"), indent=1)
        b = self.sel_raw.best; a = self.sel_aligned.best
        test_log.write(f'[select] best_raw step {b["step"]} (HQNR {b["hqnr"]:.6f} fSCC {b["fscc"]:.4f}, anchor {self.sel_raw.max_hqnr:.6f}, band {len(self.sel_raw.cands)})'
                       + (f' | best_aligned step {a["step"]} (HQNR_al {a["hqnr"]:.6f} fSCC_al {a["fscc"]:.4f}, band {len(self.sel_aligned.cands)})' if a else ' | best_aligned: no valid candidate'))

    def _materialize(self, best, tag, view):
        dst = os.path.join(self.args.work_dir, tag)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(best["path"], dst)
        rec = self.step_records.get(best["step"], {})
        json.dump(dict(step=best["step"], epoch=best["epoch"], selection_view=view, hqnr=best["hqnr"], fscc=best["fscc"], protocol_id=PROTOCOL_ID,
                       roi_hash=(self._roi or {}).get("roi_hash"), alias=("best_hqnr" if tag == "best_hqnr" else None), case=self.case,
                       rr_scc=rec.get("scc"), rr_ergas=rec.get("ergas"), rr_sam=rec.get("sam"), hqnr_aligned_valid=rec.get("hqnr_aligned_valid")),
                  open(os.path.join(self.args.work_dir, f"{tag}_meta.json"), "w"), indent=1)

    def best_raw_record(self):
        """main.py 가 best_state.json 에 쓸 값 — 선택된 step 의 기록 (현재 step 이 아닐 수 있다, 검토 지적 5)."""
        b = self.sel_raw.best
        return dict(self.step_records.get(b["step"], {}), **b) if b else None

    def save_best_model_hqnr(self):
        """main.py 가 raw_is_best 일 때 부른다. best_raw = 선택기가 고른 후보(현재 step 이 아닐 수 있다) → best_hqnr(alias) 로 복사."""
        self._materialize(self.sel_raw.best, "best_hqnr", "raw_original")
        json.dump(json.load(open(os.path.join(self.args.work_dir, "best_hqnr_meta.json"))), open(os.path.join(self.args.work_dir, "best_raw_meta.json"), "w"), indent=1)

    def write_best_meta(self, epoch, global_step, hqnr):
        pass                                                                        # best_raw_meta.json 이 이미 선택기 값으로 쓰였다

    def export_tags(self):
        return ["best_hqnr", "best_aligned", "last"]

    # ------------------------------------------------------------------ export (§13 predictions: sr · pan_aligned · delta)
    def _collect(self, loader, has_gt):
        self.model.eval(); self.model.requires_grad_(False)
        acc = dict(pan=[], lms=[], ms=[], gt=[], sr=[], pan_aligned=[], delta=[])
        for batch in tqdm(loader):
            gt, (lms, ms, lpan, pan) = (batch[0], batch[1:]) if has_gt else (None, batch)
            o = self._infer(pan, lpan, ms)
            acc["pan"].append(o["pan"]); acc["lms"].append(lms.to(o["pan"].device)); acc["ms"].append(o["ms"]); acc["sr"].append(o["y"])
            acc["pan_aligned"].append(o["pan_aligned"].to(o["pan"].dtype)); acc["delta"].append(o["delta"].float().cpu())
            if gt is not None:
                acc["gt"].append(gt.to(o["pan"].device))
        return acc

    def _savemat(self, name, acc):
        from scipy.io import savemat
        path = os.path.join(self.args.work_dir, 'results/'); os.makedirs(path, exist_ok=True)
        cv = lambda L: (torch.cat(L, 0).clip(-1, 1).detach().cpu().numpy() + 1.0) / 2 * self.args.max_pixel
        out = {k: cv(v) for k, v in acc.items() if v and k not in ("delta",)}
        out["delta"] = torch.cat(acc["delta"], 0).numpy()
        savemat(os.path.join(path, name), out)

    def test_reduced_save(self, tag='best_hqnr'):
        self._savemat(f'reduced_{tag}.mat', self._collect(self.test_reduced_data_loader, True))

    def test_full_save(self, tag='best_hqnr'):
        self._savemat(f'full_{tag}.mat', self._collect(self.test_full_data_loader, False))
