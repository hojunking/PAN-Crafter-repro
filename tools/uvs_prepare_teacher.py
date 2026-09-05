#!/usr/bin/env python
"""Teacher 준비 (계획 §5): 기존 checkpoint(예: c0_hqnr)에 T_shift·T_unc 를 붙여 짧게 학습하고 gate(§5.4) 를 판정한다.

  python tools/uvs_prepare_teacher.py --teacher c0_hqnr [--shift-iters 8000] [--unc-iters 5000]

산출: work_dir/<teacher>/uvs_teacher/{shift.safetensors, unc.safetensors, norm.json, gate.json}
- T_shift: ShiftModule(16,32,32) — 입력 E_P=Norm|∇lpan|, E_M=Norm|∇mean(ms)| (feeder 의 lpan = MTF↓PAN 레시피).
  손실 = q_A·SmoothL1(δ, δ_A)  [audit pseudo-label δ_A = −Δ_cache(P←M), q_A = accepted·min(1, margin/0.3)]
       + SmoothL1(δ(aug), δ_A + a)  [합성: lpan 을 −a 로 LR 강체 warp, a ~ 0.2·0 / 0.6·U[−1,1]² / 0.2·U[−3,3]²]
- T_unc: WithUncertainty(head logvar) 를 aligned PAN 입력(ĉ_T δ_T)에서 5K head-only 학습(NLL). θ=exp(s).
- gate: synthetic MAE(|a|≤1) ≤0.12 · 전체 ≤0.25 · identity ≤0.05 · Spearman(θ, err) ≥0.30 (val)
        · aligned teacher RR ERGAS 악화 ≤0.5% · FR 12-19 예측 vs audit(−Δ) median ≤0.20 LR px.
"""
import argparse, json, os, sys
import h5py, numpy as np, torch, torch.nn.functional as F, yaml
from safetensors.torch import save_file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                         # noqa: E402
from uvs.shift import ShiftModule, edge_rep, warp, warp_pan_channels, gated_delta   # noqa: E402
from kd.features import WithUncertainty                               # noqa: E402
from kd.losses import local_error_map, logvar_nll                     # noqa: E402
from utils import reduced_metrics                                     # noqa: E402

R = os.path.join(ROOT, "data/PanCollection/WV3")
SRC = dict(train=f"{R}/train_wv3.h5", val=f"{R}/valid_wv3.h5", rr=f"{R}/reduced_examples_h5/test_wv3_multiExm1.h5",
           fr=f"{R}/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5")


def load_split(name, n=None, keys=("ms", "lms", "pan", "gt")):
    out = {}
    with h5py.File(SRC[name]) as f:
        for k in keys:
            if k in f: out[k] = torch.tensor(f[k][:n] / 1023.5 - 1.0, dtype=torch.float32)
    with h5py.File(SRC[name].replace(".h5", "_pan.h5")) as f:
        out["lpan"] = torch.tensor(f["lpan"][:n] / 1023.5 - 1.0, dtype=torch.float32)
    return out


def sample_syn(B, dev):
    u = torch.rand(B, device=dev)
    a = torch.zeros(B, 2, device=dev)
    m1 = (u >= 0.2) & (u < 0.8); m3 = u >= 0.8
    a[m1] = (torch.rand(int(m1.sum()), 2, device=dev) * 2 - 1) * 1.0
    a[m3] = (torch.rand(int(m3.sum()), 2, device=dev) * 2 - 1) * 3.0
    return a


def teacher_forward(bb, pan, lpan, ms, lms=None):
    """native 경로(내부 bicubic). 반환 Y_T = res + bicubic(ms) (teacher 는 그 base 로 학습됐다)."""
    sw = torch.ones(ms.shape[0], device=ms.device)
    return bb(pan, lpan, ms, sw) + F.interpolate(ms, scale_factor=4, mode="bicubic")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="c0_hqnr"); ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--shift-iters", type=int, default=8000); ap.add_argument("--unc-iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--real-weight", type=float, default=1.0)
    ap.add_argument("--radius", type=int, default=3); ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--conf-thr", type=float, default=0.35); ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd = os.path.join(ROOT, "work_dir", a.teacher); out = os.path.join(wd, "uvs_teacher"); os.makedirs(out, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    bb = import_class(cfg["model"])(**cfg["model_args"])
    from safetensors.torch import load_file
    bb.load_state_dict(load_file(os.path.join(wd, a.ckpt, "model.safetensors")), strict=True)
    bb = bb.to(dev).eval().requires_grad_(False)

    # ---------------- T_shift ----------------
    tr = load_split("train", keys=("ms",)); N = tr["ms"].shape[0]
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "outputs/global_shift_cache/wv3_train.csv")).sort_values("sample_id")
    assert len(df) == N
    d_a = -torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)       # δ_{MS←PAN} = −Δ(P←M)
    q_a = torch.tensor((df["accepted"].values.astype(float) * np.minimum(1.0, df["peak_margin"].values / 0.3)), dtype=torch.float32)
    shift = ShiftModule((16, 32, 32), a.radius, a.temperature).to(dev)
    opt = torch.optim.AdamW(shift.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.shift_iters)
    ms_all, lp_all = tr["ms"], tr["lpan"]
    for it in range(1, a.shift_iters + 1):
        idx = torch.randint(N, (a.batch,))
        ms, lp = ms_all[idx].to(dev), lp_all[idx].to(dev); da, qa = d_a[idx].to(dev), q_a[idx].to(dev)
        e_m = edge_rep(ms)
        # real (audit pseudo-label)
        o = shift(edge_rep(lp), e_m)
        l_real = (qa.unsqueeze(1) * F.smooth_l1_loss(o["delta"], da, reduction="none")).mean()
        # synthetic: P^aug = W(P, −4a)  ⇔  lpan^aug = W_LR(lpan, −a);  target δ = δ_A + a
        s = sample_syn(a.batch, dev)
        o2 = shift(edge_rep(warp(lp, -s, 1.0, "bicubic")), e_m)
        l_syn = F.smooth_l1_loss(o2["delta"], da + s)
        loss = a.real_weight * l_real + l_syn
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == 1:
            print(f"[T_shift] {it}/{a.shift_iters} real {l_real.item():.4f} syn {l_syn.item():.4f} conf {o2['conf'].mean().item():.2f}")
    shift.eval()
    # gate: synthetic MAE on val (identity, |a|<=1, all)
    va = load_split("val", keys=("ms",)); g = torch.Generator(device=dev).manual_seed(0)
    def syn_mae(rng, n=1024):
        errs = []
        with torch.no_grad():
            for i in range(0, min(n, va["ms"].shape[0]), 128):
                ms, lp = va["ms"][i:i + 128].to(dev), va["lpan"][i:i + 128].to(dev)
                s = (torch.rand(ms.shape[0], 2, device=dev, generator=g) * 2 - 1) * rng if rng > 0 else torch.zeros(ms.shape[0], 2, device=dev)
                base = shift(edge_rep(lp), edge_rep(ms))["delta"]                        # 고유 shift 추정(노이즈)
                o = shift(edge_rep(warp(lp, -s, 1.0, "bicubic")), edge_rep(ms))
                errs.append((o["delta"] - (base + s)).abs().mean(dim=1))
        return float(torch.cat(errs).mean())
    mae_id, mae_1, mae_3 = syn_mae(0.0), syn_mae(1.0), syn_mae(3.0)
    # FR agreement vs audit (12-19)
    fr = load_split("fr", keys=("ms",)); af = pd.read_csv(os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")).sort_values("sample_id")
    tgt = -torch.tensor(af[["dy_lr_raw", "dx_lr_raw"]].values[12:20], dtype=torch.float32)
    with torch.no_grad():
        of = shift(edge_rep(fr["lpan"][12:20].to(dev)), edge_rep(fr["ms"][12:20].to(dev)))
    fr_med = float((of["delta"].cpu() - tgt).norm(dim=1).median()); fr_conf = float(of["conf"].mean())
    save_file({k: v.cpu() for k, v in shift.state_dict().items()}, os.path.join(out, "shift.safetensors"))
    print(f"[T_shift gate] identity MAE {mae_id:.4f} (≤0.05) | |a|≤1 MAE {mae_1:.4f} (≤0.12) | all MAE {mae_3:.4f} (≤0.25) | FR medErr vs audit {fr_med:.3f} (≤0.20) conf {fr_conf:.2f}")

    # ---------------- aligned teacher no-harm (RR) ----------------
    rr = load_split("rr")
    def rr_ergas(aligned):
        vals = []
        with torch.no_grad():
            for i in range(rr["ms"].shape[0]):
                ms, lp, pan, gt = (rr[k][i:i + 1].to(dev) for k in ("ms", "lpan", "pan", "gt"))
                if aligned:
                    o = shift(edge_rep(lp), edge_rep(ms)); d = gated_delta(o["delta"], o["conf"], a.conf_thr)
                    lpu = F.interpolate(lp, scale_factor=4, mode="bicubic")
                    pan_a, _, _ = warp_pan_channels(pan, lpu, pan - lpu, d, "bicubic"); lp_a = warp(lp, d, 1.0, "bicubic")
                else:
                    pan_a, lp_a = pan, lp
                y = teacher_forward(bb, pan_a, lp_a, ms)
                vals.append(reduced_metrics(x_true=gt, x_pred=y, max_pixel=2047.0)["ergas"])
        return float(np.mean(vals))
    e_raw, e_al = rr_ergas(False), rr_ergas(True)
    print(f"[teacher RR] ERGAS raw {e_raw:.4f} aligned {e_al:.4f} (악화 {100 * (e_al - e_raw) / e_raw:+.2f}% ≤ 0.5%)")

    # ---------------- T_unc (head-only, aligned input, logvar NLL) ----------------
    unc = WithUncertainty(bb, cfg["model_args"].get("hidden_size", 128), head_out="logvar").to(dev)
    unc.base.requires_grad_(False); unc.head.requires_grad_(True)
    trn = load_split("train", keys=("ms", "gt"))
    def aligned_inputs(pan, lp, ms):
        with torch.no_grad():
            o = shift(edge_rep(lp), edge_rep(ms)); d = gated_delta(o["delta"], o["conf"], a.conf_thr)
            lpu = F.interpolate(lp, scale_factor=4, mode="bicubic")
            pan_a, _, _ = warp_pan_channels(pan, lpu, pan - lpu, d, "bicubic"); lp_a = warp(lp, d, 1.0, "bicubic")
        return pan_a, lp_a, o["delta"], o["conf"]
    with torch.no_grad():                    # warm start (전역 상수 분산)
        idx = torch.randint(N, (32,)); pan_a, lp_a, _, _ = aligned_inputs(trn["pan"][idx].to(dev), trn["lpan"][idx].to(dev), trn["ms"][idx].to(dev))
        y = teacher_forward(unc.base, pan_a, lp_a, trn["ms"][idx].to(dev)); e0 = local_error_map(y, trn["gt"][idx].to(dev)).mean().item()
        unc.head.net[-1].weight.zero_(); unc.head.net[-1].bias.fill_(float(np.log(e0 + 1e-12)))
    opt = torch.optim.AdamW(unc.head.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.unc_iters)
    for it in range(1, a.unc_iters + 1):
        idx = torch.randint(N, (24,)); ms, gt, pan, lp = (trn[k][idx].to(dev) for k in ("ms", "gt", "pan", "lpan"))
        pan_a, lp_a, _, _ = aligned_inputs(pan, lp, ms)
        with torch.no_grad():
            y = teacher_forward(unc.base, pan_a, lp_a, ms); e_loc = local_error_map(y, gt)
        s = unc.theta(); loss, d = logvar_nll(e_loc, s)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == 1: print(f"[T_unc] {it}/{a.unc_iters} NLL {loss.item():.5f} s {d['s_mean']:+.3f}")
    unc.eval()
    from scipy.stats import spearmanr
    S, E, TH = [], [], []
    with torch.no_grad():
        for i in range(0, min(1024, va["ms"].shape[0]), 32):
            ms, lp = va["ms"][i:i + 32].to(dev), va["lpan"][i:i + 32].to(dev)
            vgt = load_split("val", keys=("gt", "pan"))  # 소량이라 재로딩 허용
            gt, pan = vgt["gt"][i:i + 32].to(dev), vgt["pan"][i:i + 32].to(dev)
            pan_a, lp_a, _, _ = aligned_inputs(pan, lp, ms)
            y = teacher_forward(unc.base, pan_a, lp_a, ms); e = local_error_map(y, gt); s = unc.theta()
            S.append(s[..., ::2, ::2].flatten().cpu()); E.append(e[..., ::2, ::2].flatten().cpu())
    S, E = torch.cat(S).numpy(), torch.cat(E).numpy(); rho = float(spearmanr(S[::7], E[::7]).statistic)
    # θ 분위수 (train) — U_T 정규화 상수, V_GT 분위수도 여기서
    TH, V = [], []
    from uvs.losses import gt_residual_variance
    with torch.no_grad():
        for i in range(0, 2048, 32):
            idx = torch.arange(i, i + 32); ms, gt, pan, lp, lms = (trn[k][idx].to(dev) for k in ("ms", "gt", "pan", "lpan", "lms"))
            pan_a, lp_a, _, _ = aligned_inputs(pan, lp, ms); teacher_forward(unc.base, pan_a, lp_a, ms)
            TH.append(torch.exp(unc.theta()).flatten().cpu()); V.append(gt_residual_variance(gt, lms, 5).flatten().cpu())
    TH, V = torch.cat(TH).numpy(), torch.cat(V).numpy()
    norm = dict(theta_q10=float(np.quantile(TH, .10)), theta_q90=float(np.quantile(TH, .90)),
                v_q10=float(np.quantile(V, .10)), v_q90=float(np.quantile(V, .90)), head_out="logvar", conf_threshold=a.conf_thr)
    save_file({k: v.cpu() for k, v in unc.head.state_dict().items()}, os.path.join(out, "unc.safetensors"))
    gate = dict(identity_mae=mae_id, syn_mae_le1=mae_1, syn_mae_all=mae_3, fr_med_err_vs_audit=fr_med, fr_conf=fr_conf,
                rr_ergas_raw=e_raw, rr_ergas_aligned=e_al, rr_ergas_rel=(e_al - e_raw) / e_raw, spearman=rho,
                checks=dict(identity=mae_id <= 0.05, syn_le1=mae_1 <= 0.12, syn_all=mae_3 <= 0.25, fr_audit=fr_med <= 0.20,
                            rr_noharm=(e_al - e_raw) / e_raw <= 0.005, spearman=rho >= 0.30))
    gate["pass_shift"] = all(gate["checks"][k] for k in ("identity", "syn_le1", "syn_all", "fr_audit", "rr_noharm"))
    gate["pass_unc"] = gate["checks"]["spearman"]; gate["pass"] = gate["pass_shift"] and gate["pass_unc"]
    json.dump(norm, open(os.path.join(out, "norm.json"), "w"), indent=1); json.dump(gate, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print(f"[T_unc gate] Spearman {rho:.3f} (≥0.30)"); print("[gate]", gate["checks"], "-> shift", gate["pass_shift"], "unc", gate["pass_unc"])
    sys.exit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()
