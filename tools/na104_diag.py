#!/usr/bin/env python
"""NA104 artifact 진단 — 계획 §14.3 "Artifact: noise/ringing, 과도한 smoothing, 특정 band bias, 평탄·어두운 영역의 악화".

    python tools/na104_diag.py --run NA104_Q10_..._v1 [--ckpt best_rr_val] [--out na104_diag]

RR 테스트셋(GT 있음)에서만 잰다 — 통계 loss 가 줄었을 때 그것이 **구조 보존**인지 **노이즈/링잉으로 통계만 맞춘 것**인지 가르기 위한 진단이다.
지표는 전부 GT 대비 비율·차이이고, 선택(selector)에는 쓰지 않는다. 값 하나로 판정하지 않는다 (판정은 HQNR→SCC, §14.3).

기록하는 것
  band_bias        : band 별 평균 부호 오차 (DN) — 특정 밴드 편향
  grad_ratio       : mean|∇ŷ| / mean|∇GT| (Scharr, band 평균) — 1 보다 작으면 과도한 smoothing, 크면 과도한 대비/링잉
  hf_ratio         : 고주파 에너지 비 (라플라시안 RMS) — grad_ratio 와 함께 본다
  ringing_flat     : GT 국소분산 하위 20% (평탄) 영역에서 **오차의 국소분산** — 링잉/노이즈가 평탄부에 실리는 정도
  mae_by_var_q     : GT 국소분산 4분위별 MAE — 평탄부 악화 여부
  mae_by_lum_q     : GT 밝기 4분위별 MAE — 어두운 영역 악화 여부
  edge_l1, stat_l1 : signed Scharr edge 오차, GT 통계(GV) 오차 — fitting 지표와 같은 정의
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                              # noqa: E402
from kdv.teacher_assets import skeleton_from_cfg, load_state               # noqa: E402
from kdv.losses_stat import statistic_map                                  # noqa: E402
from pa.losses import output_edge_loss                                     # noqa: E402

KX = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def _filt(x, k):
    c = x.shape[1]
    return F.conv2d(x, k.to(x)[None, None].repeat(c, 1, 1, 1), groups=c)


def grad_mag(x):
    return (_filt(x, KX) ** 2 + _filt(x, KX.T.contiguous()) ** 2).sqrt()


def local_var(x, w=7):
    mu = F.avg_pool2d(x, w, 1, 0)
    return (F.avg_pool2d(x * x, w, 1, 0) - mu * mu).clamp_min(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_rr_val")
    ap.add_argument("--out", default="na104_diag"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    wd = os.path.join(ROOT, "work_dir", a.run)
    cfg = yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))
    dev = torch.device(a.device)
    Model = import_class(cfg["model"])
    with torch.random.fork_rng(devices=[]):
        m, info = skeleton_from_cfg(cfg, Model)
    sd, ckpt_path = load_state(os.path.join(wd, a.ckpt))
    m.load_state_dict(sd, strict=True) if info["kind"] != "b0" else m.backbone.load_state_dict(sd, strict=True)
    m.to(dev).eval()
    Feeder = import_class(cfg["feeder"]); ds = Feeder(**cfg["test_reduced_feeder_args"]); mp = float(ds.max_pixel)

    rows, agg = [], {}
    with torch.no_grad():
        for i in range(len(ds)):
            gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            y = m(pan, ms, lpan)["y"].clamp(-1, 1).float()
            g = gt.float()
            yd = (y + 1) / 2 * mp; gd = (g + 1) / 2 * mp                       # DN 단위 (보고용)
            err = yd - gd
            gm_y, gm_g = grad_mag(y), grad_mag(g)
            lv = local_var(g[:, :1] if g.shape[1] == 1 else g.mean(1, keepdim=True))
            lum = F.avg_pool2d(gd.mean(1, keepdim=True), 7, 1, 0)
            e_c = err[..., 3:-3, 3:-3].abs().mean(1, keepdim=True)             # local_var 와 같은 중심
            ev = local_var(err.mean(1, keepdim=True))
            q = torch.quantile(lv.flatten(), torch.tensor([0.2, 0.25, 0.5, 0.75], device=dev))
            ql = torch.quantile(lum.flatten(), torch.tensor([0.25, 0.5, 0.75], device=dev))
            flat = lv <= q[0]
            r = dict(scene=i,
                     mae_dn=float(err.abs().mean()),
                     band_bias_dn=[float(err[:, c].mean()) for c in range(err.shape[1])],
                     band_bias_absmax_dn=float(max(abs(float(err[:, c].mean())) for c in range(err.shape[1]))),
                     grad_ratio=float(gm_y.mean() / gm_g.mean().clamp_min(1e-12)),
                     hf_ratio=float(_filt(y, LAP).pow(2).mean().sqrt() / _filt(g, LAP).pow(2).mean().sqrt().clamp_min(1e-12)),
                     ringing_flat=float(ev[flat].mean()) if bool(flat.any()) else float("nan"),
                     ringing_textured=float(ev[~flat].mean()) if bool((~flat).any()) else float("nan"),
                     edge_l1=float(output_edge_loss(y, g)),
                     stat_l1_gv=float((statistic_map(y, "grad_var", 5) - statistic_map(g, "grad_var", 5)).abs().mean()))
            for name, key, qs in (("var", lv, q[1:]), ("lum", lum, ql)):
                bins = torch.bucketize(key.flatten(), qs)
                for b in range(4):
                    msk = (bins == b)
                    r[f"mae_by_{name}_q{b + 1}"] = float(e_c.flatten()[msk].mean()) if bool(msk.any()) else float("nan")
            rows.append(r)
    keys = [k for k in rows[0] if k not in ("scene", "band_bias_dn")]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    agg["band_bias_dn"] = [float(np.mean([r["band_bias_dn"][c] for r in rows])) for c in range(len(rows[0]["band_bias_dn"]))]
    out = dict(run=a.run, ckpt=a.ckpt, ckpt_file=os.path.relpath(ckpt_path, ROOT), n_scenes=len(rows), max_pixel=mp,
               note="RR 테스트셋 진단. 통계만 맞고 구조가 상하는 경우를 가르기 위한 보조 지표이며 선택(selector)·판정에 쓰지 않는다 (§14.3).",
               reading=dict(grad_ratio="<1 이면 과도한 smoothing, >1 이면 과도한 대비/링잉 의심",
                            ringing_flat="평탄부에 실린 오차의 국소분산 — 커지면 노이즈/링잉으로 통계를 맞춘 신호",
                            mae_by_var_q1="가장 평탄한 사분위의 MAE (평탄부 악화 확인)", mae_by_lum_q1="가장 어두운 사분위의 MAE"),
               mean=agg, per_scene=rows)
    os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    p = os.path.join(wd, "results", f"{a.out}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"{a.run}/{a.ckpt}: MAE {agg['mae_dn']:.2f} DN · grad_ratio {agg['grad_ratio']:.4f} · hf_ratio {agg['hf_ratio']:.4f} · "
          f"ringing(flat/text) {agg['ringing_flat']:.4g}/{agg['ringing_textured']:.4g} · band bias |max| {agg['band_bias_absmax_dn']:.2f} DN · "
          f"MAE var-q1 {agg['mae_by_var_q1']:.2f} lum-q1 {agg['mae_by_lum_q1']:.2f} → {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
