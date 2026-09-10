#!/usr/bin/env python
"""아키텍처 사전 smoke 검증 — 체인이 2시간짜리 슬롯을 태우기 전에 분 단위로 잡는다.

계획서 §8.1 (research_log/2026-08-28_architecture-search-24h-plan.md) 의 점검 항목:
  1. config 로드 -> 모델 build -> params 실측
  2. 학습 형상 forward + backward (dual 이면 switch 0/1 혼합, mars: ms 면 1만)
     -> 출력 shape (B,8,64,64) · finite · grad finite
  3. FR 형상(512², WV3 full-res) no-grad forward -> shape · finite
     (PixelUnshuffle 배수 문제·해상도 의존 버그 검출)
  4. 실배치 OOM 검사: batch_size x MARs 복제(dual 이면 2배)로 forward+backward+
     AdamW step 1회 — 학습 시작 후에야 OOM 나는 사고(s1_A2 d444) 재발 방지
  5. peak VRAM · 대략적인 step 시간

zero_module 함정: 출력 conv 가 0 으로 초기화돼 있어 그대로면 상류 grad 가 전부 0
이라 backward 검사가 공허하다. 검사 전에 0 파라미터를 난수화한다 (CLAUDE.md 함정 목록).

사용:
  python tools/smoke_cases.py <config이름> [<config이름> ...]
종료코드: 하나라도 실패면 1 (체인은 이걸 보고 해당 case 를 학습 없이 FAILED 처리)
"""
import os
import sys
import time
import importlib

import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


_ALIGN_PEAK = 0.0      # align wrapper 실배치 peak (check_trainer_extras 가 채운다)


class ConfigMissing(Exception):
    """config 파일 부재 — 빌드 실패와 달리 일시 사유다 (예: git pull 전).

    체인은 이 경우(exit 2)를 실패 원장에 남기지 않고 이번 패스만 건너뛴다.
    """


def load_cfg(name):
    for p in (os.path.join(ROOT, "config", f"{name}.yaml"),
              os.path.join(ROOT, "config", f"pancrafter_{name}.yaml")):
        if os.path.exists(p):
            return yaml.safe_load(open(p))
    raise ConfigMissing(f"config 없음: {name}")


def build(cfg):
    mod, cls = cfg["model"].rsplit(".", 1)
    M = getattr(importlib.import_module(mod), cls)
    return M(**cfg["model_args"])


def randomize_zero_params(m):
    """zero_module 로 0 초기화된 파라미터를 난수화 — backward 가 실제로 흐르게."""
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dtype.is_floating_point and p.abs().sum() == 0:
                p.normal_(0, 1e-3)


def check_trainer_extras(cfg):
    global _ALIGN_PEAK
    """신규 trainer 의 실제 실패 지점을 학습 전에 재현한다.

    kd: teacher 를 **실제로 로딩**(CPU) — 존재/키/uncertainty 호환을 전부 검증.
    mutual: peer 2벌 build — 구성 오류·대략적 2x 메모리 사실을 드러낸다.
    """
    tr = cfg.get("trainer")
    if tr == "kd":
        variant = (cfg.get("kd_args") or {}).get("variant", "k0")
        if variant == "k0":
            return ""
        ck = cfg.get("teacher_checkpoint")
        assert ck and os.path.isdir(os.path.join(ROOT, ck)), f"teacher checkpoint 없음: {ck}"
        from train_kd import load_teacher
        t, has_unc = load_teacher(os.path.join(ROOT, cfg["teacher_config"]),
                                  os.path.join(ROOT, ck),
                                  torch.device("cpu"), torch.float32)
        del t
        if variant in ("k2", "k3", "k4", "k5"):
            assert has_unc, f"{variant} 는 uncertainty teacher 필요 — {ck} 에 head 없음"
        return f" teacher={'unc' if has_unc else 'plain'}"
    if tr == "align":
        from align.model import AlignCfg, AlignedModel
        from align.cache import ShiftCache
        cfg_a = AlignCfg.from_dict(cfg.get("alignment") or {})
        assert cfg.get("feeder") == "feeders.feeder_align.PanFeederAlign", "align 은 PanFeederAlign 필요"
        assert cfg.get("mars", "dual") == "dual" and cfg.get("res", True), "align 은 dual MARs·잔차 고정"
        assert not (cfg.get("train_feeder_args") or {}).get("crop", False), "align 캠페인은 crop 없음"
        note = f" GA:{cfg_a.delta_source}/a{cfg_a.alpha:g}/{cfg_a.output_frame}/{cfg_a.inverse_location}/{cfg_a.upsampler}"
        if cfg_a.delta_source in ("cache", "trainable"):
            c = ShiftCache(os.path.join(ROOT, cfg_a.cache_dir))
            assert all(c.has(k) for k in (0, 2, 3)), "shift cache 3 split 필요"
            note += f" cache={c.sha256_all[:8]}"
        if cfg_a.trainable_shift_net:
            rep = os.path.join(ROOT, cfg_a.shiftnet_pretrained.replace(".pt", ".json"))
            note += (" shiftnet=pretrained" if os.path.exists(rep) else " shiftnet=미학습(체인이 pretrain 실행)")
        # wrapper 로 **실배치** forward+backward+AdamW 1 step — C2 의 이중 view, C4 의 ShiftNet graph,
        # HR 증강, inverse warp/mask 까지 실제 학습 그래프 그대로 peak VRAM 을 잰다 (bare backbone 만으로는 부족).
        from align.resample import transform_delta
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w = AlignedModel(build(cfg), cfg_a).to(dev)
        randomize_zero_params(w)
        if w.shift_net is not None:
            with torch.no_grad():
                w.shift_net.head.weight.normal_(0, 0.05)
        B = int(cfg.get("batch_size", 48))
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        opt = torch.optim.AdamW(w.parameters(), lr=1e-4)
        pan = torch.randn(B, 1, 64, 64, device=dev); lpan = torch.randn(B, 1, 16, 16, device=dev)
        ms = torch.randn(B, 8, 16, 16, device=dev); gt = torch.randn(B, 8, 64, 64, device=dev)
        aug = (torch.ones(B, device=dev, dtype=torch.long), torch.ones(B, device=dev, dtype=torch.long),
               torch.randint(0, 4, (B,), device=dev))
        if cfg_a.delta_source == "zero":
            d = torch.zeros(B, 2, device=dev)
        elif cfg_a.delta_source == "trainable":
            d = w.predict_delta(lpan, ms)
        else:
            d = torch.randn(B, 2, device=dev) * 0.3
        dd = torch.cat([d.detach(), d]); sw = torch.cat([torch.zeros(B, device=dev), torch.ones(B, device=dev)])
        v = w.build_views(pan.repeat(2, 1, 1, 1), lpan.repeat(2, 1, 1, 1), ms.repeat(2, 1, 1, 1), dd,
                          aug=tuple(t.repeat(2) for t in aug))
        res = w.residual(v["x11"], sw)
        fin = w.finalize_ms(v["ms_base_hr"][B:], res[B:], transform_delta(d, *aug))
        assert fin["y_final"].shape == (B, 8, 64, 64) and torch.isfinite(fin["y_final"]).all()
        loss = ((fin["y_loss"] - gt).abs() * fin["mask"]).sum() / (fin["mask"].sum() * 8) \
            + (v["lpan_hr"][:B].repeat(1, 8, 1, 1) + res[:B] - pan.repeat(1, 8, 1, 1)).abs().mean()
        loss.backward(); opt.step()
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20
            note += f" wrapperPeak {_ALIGN_PEAK:.0f}MB"
        with torch.no_grad():
            o = w.infer_ms(torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, 1, 128, 128, device=dev),
                           torch.randn(1, 8, 128, 128, device=dev), d[:1].detach())
        assert o["y_final"].shape[-2:] == (512, 512) and torch.isfinite(o["y_final"]).all()
        del w, opt, v, res, fin, loss
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return note
    if tr == "sr":
        from train_sr import SRModel, VARIANTS
        from sr.forward import sr_forward, sr_infer
        from sr.jitter import sample_jitter
        from sr.pan_align import GlobalCorrelator
        sr = cfg.get("sr") or {}; v = sr.get("variant")
        assert v in VARIANTS, f"sr.variant {v}"
        assert cfg.get("feeder") == "feeders.feeder.PanFeeder" and cfg.get("mars", "dual") == "dual", "sr 는 원 feeder·dual"
        note = f" SR:{v}"
        blur = None
        if v == "j3":
            import json
            cal = os.path.join(ROOT, (sr.get("blur") or {}).get("calibration", "outputs/shift_robust/blur_calib.json"))
            assert os.path.exists(cal), "blur 보정 파일 없음 — tools/calibrate_blur.py"
            j = json.load(open(cal)); mt = (sr.get("blur") or {}).get("match", "mse"); j = j.get(mt, j)
            assert j["within_tol"]; blur = j["sigma_star"]; note += f" sigma={blur:.3f}({mt})"
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bb = build(cfg).to(dev); randomize_zero_params(bb)
        g1 = sr.get("g1") or {}
        corr = GlobalCorrelator(bb.input.out_channels, int(g1.get("desc_channels", 16)), float(g1.get("radius_hr_px", 1.0)),
                                int(g1.get("n_per_axis", 5)), float(g1.get("tau", 0.07)), float(g1.get("gate_c0", 0.30))).to(dev) if v == "g1" else None
        m = SRModel(bb, corr)
        B = int(cfg.get("batch_size", 48)); r = float((sr.get("jitter") or {}).get("max_abs_hr_px", 0.5))
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        pan, lpan = torch.randn(B, 1, 64, 64, device=dev), torch.randn(B, 1, 16, 16, device=dev)
        ms, gt = torch.randn(B, 8, 16, 16, device=dev), torch.randn(B, 8, 64, 64, device=dev)
        eps = sample_jitter(B, r, dev, torch.float32) if v in ("j1", "j2", "j4") else None
        eps_g = sample_jitter(B, float(g1.get("syn_max_hr_px", 1.0)), dev, torch.float32, float(g1.get("syn_prob", 0.75))) if v == "g1" else None
        o = sr_forward(m, v, pan, lpan, ms, gt, eps=eps, eps_g=eps_g, lam_cons=0.1, blur_sigma=blur)
        loss = o["loss"] + 0.1 * o["loss_shift"]
        assert torch.isfinite(loss) and o["loss_ms"].item() > 0 and o["loss_pan"].item() > 0
        assert o["y"].shape == (B, 8, 64, 64)
        loss.backward(); opt.step()
        if eps is not None:
            assert eps.abs().max() <= r + 1e-6 and eps.abs().mean() > 0.05 * r, "jitter 통계 이상"
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20
            note += f" trainPeak {_ALIGN_PEAK:.0f}MB" + (" (3x48 forward)" if v == "j4" else "")
        m.eval()
        with torch.no_grad():
            of = sr_infer(m, v, torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, 1, 128, 128, device=dev),
                          torch.randn(1, 8, 128, 128, device=dev), blur_sigma=blur)
        assert of["y"].shape[-2:] == (512, 512) and torch.isfinite(of["y"]).all()
        del m, bb, corr, opt, o, loss, of
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return note
    if tr == "po":
        from pa.aligner import PANGlobalAligner
        from pa.model import PAModel
        from pa.offset import po_step, aligner_margin, CASES as PO_CASES
        po = cfg.get("po") or {}; case = po.get("case"); assert case in PO_CASES, f"po.case {case}"
        assert cfg.get("mars") == "ms" and cfg["model_args"].get("in_mode") == "paper" and cfg["model_args"].get("mode_modulation") is False, "po 는 B0(9ch·mars ms·γβ 제거) 위"
        R = float(po.get("radius_hr", 1.0)); mg = aligner_margin(R)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bb = build(cfg).to(dev); randomize_zero_params(bb)
        al = PANGlobalAligner(int(cfg["num_bands"])).to(dev); m = PAModel(bb, al, aligner_margin=mg)
        B = int(cfg.get("batch_size", 48)); nb = int(cfg["num_bands"]); note = f" PO:{case} R={R} view-margin {mg}"
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4); g = torch.Generator(device="cpu"); g.manual_seed(0)
        pan, lpan = torch.randn(B, 1, 64, 64, device=dev), torch.randn(B, 1, 16, 16, device=dev)
        ms, gt = torch.randn(B, nb, 16, 16, device=dev), torch.randn(B, nb, 64, 64, device=dev)
        import time as _t; times = {}
        for k, upd in (("native", 0), ("corrupt", 1)):
            for rep in range(3):
                t0 = _t.time(); total, info = po_step(m, pan, ms, lpan, gt, case=case, update_index=upd, radius_hr=R, generator=g, lam_max=0.01, ramp=1)
                assert torch.isfinite(total) and info["pred"].shape == (B, nb, 64, 64)
                opt.zero_grad(); total.backward(); opt.step()
                if dev.type == "cuda": torch.cuda.synchronize()
                times[k] = _t.time() - t0
            if k == "corrupt":
                assert info["corrupt"] and float(info["eps"].norm(dim=1).max()) <= R + 1e-6
                if case == "N1": assert info["weight"] == 0.0
        assert al.fc2.weight.grad is not None
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20
            note += f" trainPeak {_ALIGN_PEAK:.0f}MB t_native {times['native']*1000:.0f}ms t_corrupt {times['corrupt']*1000:.0f}ms"
        m.eval()
        with torch.no_grad():
            of = m(torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, nb, 128, 128, device=dev), torch.randn(1, 1, 128, 128, device=dev))
        assert of["y"].shape[-2:] == (512, 512) and torch.isfinite(of["y"]).all()
        del m, bb, al, opt, total, info, of
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return note
    if tr == "kdv":
        # s2 W112 KDV: skeleton(정책별 aligner/sampler) + donor strict + Teacher strict(존재해야 한다) + 실제 loss 조합으로 실배치 1 step (peak·시간) + FR 512² forward
        from kdv.registry import resolve, run_name, arch_prefix
        from kdv.teacher_assets import skeleton_from_cfg, load_donor_aligner, load_run_model, freeze
        from kdv.forward import kdv_forward
        from kdv.losses_rec import GTAnchoredReconstructionKD
        from kdv.losses_stat import stat_term, MODE_TO_CRITERION
        from kdv.protocol import prepare_view
        k = cfg.get("kdv") or {}; sp = resolve(k)
        assert cfg.get("mars") == "ms" and cfg["model_args"].get("in_mode") == "paper" and cfg["model_args"].get("mode_modulation") is False, "kdv 는 9ch·mars ms·γβ 제거 위"
        exp_name = run_name(sp, int(cfg["seed"]), k.get("version", "v01"), arch_prefix(cfg["model_args"]))
        assert not k.get("check_run_name", True) or os.path.basename(cfg["work_dir"].rstrip("/")) == exp_name, f"run 이름 규칙 불일치: {exp_name}"
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mod, cls = cfg["model"].rsplit(".", 1); Model = getattr(importlib.import_module(mod), cls)
        m, info = skeleton_from_cfg(cfg, Model); m.to(dev); randomize_zero_params(m.backbone)
        note = f" KDV:{sp['policy']}/{sp['rec_case']}/{'OFF' if not sp['stat_enabled'] else sp['stat_key'] + sp['stat_mode']}/{sp['geom']}"
        if sp["policy"] in ("A-FR", "A-FT"):
            al, dman = load_donor_aligner(k["donor"]["source"], int(cfg["num_bands"]), (k["donor"] or {}).get("expected_sha256")); m.aligner.load_state_dict(al.state_dict(), strict=True)
            note += f" donor={dman['aligner_tensors_sha256_16'][:8]}"
        if sp["policy"] == "A-FR":
            freeze(m.aligner)
        teacher = None
        if sp["needs_teacher"]:
            t = k["teacher"]; teacher, tman = load_run_model(t["run"], t.get("tag", "best_hqnr"), Model, t.get("expected_sha256")); freeze(teacher); teacher.to(dev)
            assert int(tman["width"]) == int(cfg["model_args"]["hidden_size"]) or t.get("bridge"), f"Teacher width {tman['width']} ≠ Student"
            note += f" teacher={tman['run'][:24]}…/{tman['tag']}"
        B = int(cfg.get("batch_size", 48)); nb = int(cfg["num_bands"])
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        params = [p for p in m.backbone.parameters()] + (list(m.aligner.parameters()) if sp["aligner_trainable"] else [])
        opt = torch.optim.AdamW(params, lr=1e-4); g = torch.Generator(device="cpu"); g.manual_seed(0)
        pan, lpan = torch.randn(B, 1, 64, 64, device=dev), torch.randn(B, 1, 16, 16, device=dev)
        ms, gt = torch.randn(B, nb, 16, 16, device=dev), torch.randn(B, nb, 64, 64, device=dev)
        crit = GTAnchoredReconstructionKD(0.05, mode=sp["rec_mode"]).to(dev) if teacher is not None else None
        scrit = GTAnchoredReconstructionKD(0.05, mode=MODE_TO_CRITERION[sp["stat_mode"]]).to(dev) if (sp["stat_enabled"] and sp["stat_mode"] in MODE_TO_CRITERION and teacher is not None) else None
        import time as _t; times = {}
        for kind, upd in (("native", 0), ("corrupt", 1)):
            if kind == "corrupt" and sp["protocol"] != "I-N":
                continue
            for rep in range(3):
                t0 = _t.time(); pv, eps, cor = prepare_view(pan, sp["protocol"], upd, sp["radius_hr"], g)
                o = kdv_forward(m, teacher, pv, ms, lpan, share_correction=(sp["policy"] == "A-FR"), teacher_needed=sp["needs_teacher"], aligner_live=sp["aligner_trainable"])
                loss = crit(o["y"], o["y_t"], gt).loss if crit is not None else (o["y"] - gt).abs().mean()
                if sp["stat_enabled"]:
                    from pa.losses import output_edge_loss
                    loss = loss + 0.1 * (output_edge_loss(o["y"], gt) if sp["stat_kind"] == "edge" else stat_term(o["y"], (o["y_t"] if o["y_t"] is not None else None), gt, kind=sp["stat_kind"], window=sp["stat_window"], mode=sp["stat_mode"], criterion=scrit).loss)
                assert torch.isfinite(loss) and o["y"].shape == (B, nb, 64, 64)
                opt.zero_grad(); loss.backward(); opt.step()
                if dev.type == "cuda": torch.cuda.synchronize()
                times[kind] = _t.time() - t0
        if sp["aligner_trainable"]:
            assert m.aligner.fc2.weight.grad is not None, "trainable aligner 에 gradient 가 없다"
        if sp["policy"] == "A-FR":
            assert all(p.grad is None for p in m.aligner.parameters()), "frozen aligner 에 gradient 가 생겼다"
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20
            note += f" trainPeak {_ALIGN_PEAK:.0f}MB t_native {times['native']*1000:.0f}ms" + (f" t_corrupt {times['corrupt']*1000:.0f}ms" if "corrupt" in times else "")
        m.eval()
        with torch.no_grad():
            of = m(torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, nb, 128, 128, device=dev), torch.randn(1, 1, 128, 128, device=dev))
        assert of["y"].shape[-2:] == (512, 512) and torch.isfinite(of["y"]).all()
        del m, opt, o, loss, of, teacher
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return note
    if tr == "pa":
        from pa.aligner import PANGlobalAligner
        from pa.model import PAModel
        from pa.losses import output_edge_loss, direct_geometry_loss
        from pa.warp import support_margin_ok
        pa = cfg.get("pa") or {}; case = pa.get("case"); assert case in ("A1", "A2", "A3"), f"pa.case {case}"
        assert cfg.get("mars") == "ms" and cfg["model_args"].get("in_mode") == "paper" and cfg["model_args"].get("mode_modulation") is False, "pa 는 B0(9ch·mars ms·γβ 제거) 위"
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bb = build(cfg).to(dev); randomize_zero_params(bb)
        al = PANGlobalAligner(int(cfg["num_bands"])).to(dev)
        n_al = sum(p.numel() for p in al.parameters()); assert n_al == 105330 or int(cfg["num_bands"]) != 8, f"aligner params {n_al}"
        m = PAModel(bb, al); note = f" PA:{case} aligner {n_al / 1e6:.4f}M"
        B = int(cfg.get("batch_size", 48)); nb = int(cfg["num_bands"])
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        pan, lpan = torch.randn(B, 1, 64, 64, device=dev), torch.randn(B, 1, 16, 16, device=dev)
        ms, gt = torch.randn(B, nb, 16, 16, device=dev), torch.randn(B, nb, 64, 64, device=dev)
        o = m(pan, ms, lpan)
        assert o["y"].shape == (B, nb, 64, 64) and o["delta"].shape == (B, 2)
        assert bool(support_margin_ok(64, 64, o["delta"].detach(), int(pa.get("geometry_margin_hr", 11))).all())
        loss = (gt - o["y"]).abs().mean()
        if case == "A2":
            loss = loss + 0.1 * output_edge_loss(o["y"], gt)
        if case == "A3":
            lg, _ = direct_geometry_loss(o["pan_aligned"], pan, gt, float(pa.get("geometry_sigma_hr", 2.0)), int(pa.get("geometry_margin_hr", 11))); loss = loss + 0.01 * lg
        assert torch.isfinite(loss)
        loss.backward()
        assert al.fc2.weight.grad is not None and torch.isfinite(al.fc2.weight.grad).all(), "zero-head 에 gradient 가 없다 (§4.2 bypass?)"
        opt.step()
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20
            note += f" trainPeak {_ALIGN_PEAK:.0f}MB"
        m.eval()
        with torch.no_grad():
            of = m(torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, nb, 128, 128, device=dev), torch.randn(1, 1, 128, 128, device=dev))
        assert of["y"].shape[-2:] == (512, 512) and torch.isfinite(of["y"]).all() and torch.isfinite(of["delta"]).all()
        del m, bb, al, opt, o, loss, of
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return note
    if tr == "uvs":
        import json
        from train_uvs import UVSModel, VARIANTS, USES_SHIFT, USES_V, build_inputs, x11
        from uvs.shift import ShiftModule, edge_rep, warp_pan_channels, gated_delta
        from uvs.losses import gt_residual_variance, percentile_normalize, variance_weight, routed_losses, shift_kd_loss
        u = cfg.get("uvs") or {}; v = u.get("variant"); assert v in VARIANTS, f"uvs.variant {v}"
        assert cfg.get("feeder") == "feeders.feeder_uvs.PanFeederUVS"
        note = f" UVS:{v}"
        if v != "b0":
            tc = (cfg.get("train_feeder_args") or {}).get("teacher_cache"); tcp = os.path.join(ROOT, tc)
            assert tc and os.path.exists(tcp), f"teacher cache 없음: {tc} — tools/uvs_build_cache.py"
            meta = json.load(open(tcp.replace(".npz", ".json"))); note += f" cache={meta['sha256'][:8]} gate={'PASS' if meta.get('gate_pass') else 'FAIL'}"
            if v in USES_SHIFT:
                tg = json.load(open(os.path.join(ROOT, "work_dir", meta["teacher"], "uvs_teacher", "gate.json")))
                assert tg.get("pass_shift"), "teacher shift gate FAIL — S0/M1/M2/M3 는 열지 않는다 (계획 §10.1)"
        if v in USES_V:
            nrm = os.path.join(ROOT, u.get("teacher_norm", "")); assert os.path.exists(nrm), "uvs.teacher_norm 없음"
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bb = build(cfg).to(dev); randomize_zero_params(bb)
        sh = u.get("shift") or {}
        shift = ShiftModule(tuple(sh.get("student_channels", [8, 8])), int(sh.get("search_radius", 3)), float(sh.get("softmax_temperature", 0.07))).to(dev) if v in USES_SHIFT else None
        m = UVSModel(bb, shift); B = int(cfg.get("batch_size", 48))
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        pan, lpan, ms = torch.randn(B, 1, 64, 64, device=dev), torch.randn(B, 1, 16, 16, device=dev), torch.randn(B, 8, 16, 16, device=dev)
        lms, gt = torch.randn(B, 8, 64, 64, device=dev), torch.randn(B, 8, 64, 64, device=dev)
        r_t, u_t = torch.randn(B, 8, 64, 64, device=dev) * 0.05, torch.rand(B, 1, 64, 64, device=dev)
        d_t, c_t = torch.randn(B, 2, device=dev) * 0.3, torch.rand(B, device=dev)
        lpan_u, pan_hf = build_inputs(pan, lpan, lms)
        d = None
        if shift is not None:
            o = shift(edge_rep(lpan), edge_rep(ms)); d = gated_delta(o["delta"], o["conf"])
        pa, la, ha = warp_pan_channels(pan, lpan_u, pan_hf, d) if d is not None else (pan, lpan_u, pan_hf)
        sw = torch.cat([torch.zeros(B, device=dev), torch.ones(B, device=dev)])
        res = bb(None, None, None, sw, x_in=torch.cat([x11(pan, lpan_u, pan_hf, lms), x11(pa, la, ha, lms)], 0))
        r_s = res[B:]
        w_v = variance_weight(percentile_normalize(gt_residual_variance(gt, lms), 0.001, 0.05)) if v in USES_V else None
        hard, soft = routed_losses(r_s, gt - lms, r_t, u_t, w_v)
        loss = hard + 0.1 * soft + (lpan_u.repeat(1, 8, 1, 1) + res[:B] - pan.repeat(1, 8, 1, 1)).abs().mean()
        if shift is not None:
            loss = loss + 0.25 * shift_kd_loss(o["delta"], o["conf"], d_t, c_t)[0]
        assert torch.isfinite(loss); loss.backward(); opt.step()
        if dev.type == "cuda":
            _ALIGN_PEAK = torch.cuda.max_memory_allocated(dev) / 2**20; note += f" trainPeak {_ALIGN_PEAK:.0f}MB"
        with torch.no_grad():
            P, LP, M, L = torch.randn(1, 1, 512, 512, device=dev), torch.randn(1, 1, 128, 128, device=dev), torch.randn(1, 8, 128, 128, device=dev), torch.randn(1, 8, 512, 512, device=dev)
            lu, hf = build_inputs(P, LP, L); y = L + bb(None, None, None, torch.ones(1, device=dev), x_in=x11(P, lu, hf, L))
        assert y.shape[-2:] == (512, 512) and torch.isfinite(y).all()
        del m, bb, shift, opt, res, loss
        if dev.type == "cuda": torch.cuda.empty_cache()
        return note
    if tr == "mutual":
        b = build(cfg)     # peer_b 구성 재현 (같은 model_args)
        del b
        return " 2-peer"
    return ""


def check_identity(name, cfg, model):
    """config 이름·work_dir·구조 스위치의 자기정합 검사.

    - work_dir 이 config 이름과 다르면 남의 실행을 덮어쓴다 (복사로 config 를 만들 때
      실제로 나는 사고).
    - mode_modulation 은 이 캠페인의 정의적 스위치인데 params 차이가 0.2% 뿐이라
      expect_params_m(허용 0.5%)로는 절대 잡히지 않는다 -> 구조로 직접 확인한다.
    """
    wd = os.path.basename(str(cfg.get("work_dir", "")).rstrip("/"))
    assert wd == name, f"work_dir 이름 불일치: config {name} vs work_dir {wd}"
    want = cfg.get("model_args", {}).get("mode_modulation", True)
    has = any("mod." in k or k.endswith(".gamma") or k.endswith(".beta")
              for k in model.state_dict())
    assert bool(want) == bool(has), (
        f"mode_modulation={want} 인데 실제 모델은 {'있음' if has else '없음'} — "
        "MS2 plain 을 의도하고 MS1(γβ 유지)을 학습하는 사고")


def smoke_one(name, dev):
    global _ALIGN_PEAK
    _ALIGN_PEAK = 0.0
    cfg = load_cfg(name)
    extra_note = check_trainer_extras(cfg)
    m = build(cfg).to(dev)
    check_identity(name, cfg, m)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    # config 에 expect_params_m 이 있으면 실측과 대조한다 — 옵션 하나(예:
    # cm3a_pan_branch) 빠뜨려 딴 모델을 학습하는 사고를 학습 전에 잡는다.
    exp = cfg.get("expect_params_m")
    if exp is not None:
        assert abs(n_params - float(exp)) / float(exp) < 0.005, \
            f"params {n_params:.4f}M ≠ 기대 {exp}M — config 옵션 누락 의심"
    randomize_zero_params(m)
    dual = cfg.get("mars", "dual") != "ms"

    # 학습 형상: PAN 64², LPAN/MS 16² (feeder ms_size=16 기준)
    B = 4
    pan = torch.randn(B, 1, 64, 64, device=dev)
    lpan = torch.randn(B, 1, 16, 16, device=dev)
    ms = torch.randn(B, cfg.get("num_bands", 8), 16, 16, device=dev)
    switch = (torch.tensor([0., 0., 1., 1.], device=dev) if dual
              else torch.ones(B, device=dev))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    out = m(pan, lpan, ms, switch)
    assert out.shape == (B, cfg.get("num_bands", 8), 64, 64), f"학습 출력 shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "학습 출력에 NaN/Inf"
    out.abs().mean().backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"grad NaN/Inf: {bad[:3]}"
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "0 이 아닌 grad 가 없다 — backward 가 흐르지 않음"
    step_ms = (time.time() - t0) * 1000

    # FR 형상: WV3 full-res 는 PAN 512², LPAN/MS 128²
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_fr = m(torch.randn(1, 1, 512, 512, device=dev),
                   torch.randn(1, 1, 128, 128, device=dev),
                   torch.randn(1, cfg.get("num_bands", 8), 128, 128, device=dev),
                   torch.ones(1, device=dev))
    assert out_fr.shape[-2:] == (512, 512), f"FR 출력 shape {tuple(out_fr.shape)}"
    assert torch.isfinite(out_fr).all(), "FR 출력에 NaN/Inf"

    # 실배치 OOM 검사: 학습과 같은 실효 배치(batch_size x MARs 복제)로
    # forward+backward+AdamW step 을 1회 돌려 peak VRAM 을 잰다.
    # s1_A2 가 d444·batch48 에서 학습 시작 후에야 OOM 난 사고의 재발 방지 —
    # GPU 가 유휴한 case 시작 직전에 미리 터뜨린다. optimizer state 까지 잡는다.
    peak_train = 0.0
    if dev.type == "cuda":
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        Bt = int(cfg.get("batch_size", 48)) * (2 if dual else 1)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
        pan_t = torch.randn(Bt, 1, 64, 64, device=dev)
        lpan_t = torch.randn(Bt, 1, 16, 16, device=dev)
        ms_t = torch.randn(Bt, cfg.get("num_bands", 8), 16, 16, device=dev)
        sw_t = (torch.arange(Bt, device=dev) >= Bt // 2).float() if dual \
            else torch.ones(Bt, device=dev)
        out_t = m(pan_t, lpan_t, ms_t, sw_t)
        out_t.abs().mean().backward()
        opt.step()
        peak_train = torch.cuda.max_memory_allocated(dev) / 2**20
        del opt, pan_t, lpan_t, ms_t, sw_t, out_t

    peak = (torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else 0
    peak = max(peak, _ALIGN_PEAK)          # align: wrapper 실배치 peak 가 진짜 학습 peak
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    mm = cfg.get("model_args", {}).get("mode_modulation", True)
    return (n_params, step_ms, max(peak, peak_train),
            ("dual" if dual else "ms") + ("" if mm else "+plain") + extra_note)


def main():
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        sys.exit(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failed, missing = [], []
    print(f"[smoke] device={dev}  cases={len(names)}")
    for name in names:
        try:
            p, ms_t, peak, mars = smoke_one(name, dev)
            print(f"[smoke] OK   {name:26s} {p:7.4f}M  step≈{ms_t:6.0f}ms  "
                  f"peakVRAM {peak:6.0f}MB  mars={mars}")
        except ConfigMissing as e:
            print(f"[smoke] MISS {name:26s} {e}")
            missing.append(name)
        except Exception as e:
            print(f"[smoke] FAIL {name:26s} {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print(f"[smoke] 실패 {len(failed)}: {' '.join(failed)}")
        sys.exit(1)
    if missing:
        print(f"[smoke] config 없음 {len(missing)}: {' '.join(missing)}")
        sys.exit(2)
    print("[smoke] 전부 통과")


if __name__ == "__main__":
    main()
