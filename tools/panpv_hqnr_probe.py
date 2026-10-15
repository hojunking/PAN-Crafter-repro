"""HQNR 이 제대로 쟀는지 · 제로샷 무늬에 둔감한지 확인.

1. 재현: 평가 입력(512 문맥)을 그대로 재구성 → 모델 재추론 → PANPV fr_tile_metrics → 저장값과 비교
2. 교차 구현: 같은 창 · 같은 식을 저장소 검증 부품(tools/metrics + DLPan wald interp23tap)으로 다시 계산
3. 민감도:
   a. 제로샷에서 4화소 주기 성분만 제거(notch) → HQNR 변화
   b. Brovey 에 제로샷의 주기 성분을 더함 → HQNR 변화
   c. 씬 적응 출력의 잔차(출력 − 4배 확대 MS)를 α 배 → HQNR · D_s 변화
   d. 제로샷 잔차를 2 배 → HQNR · D_s 변화
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from safetensors.torch import load_file  # noqa: E402
from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet  # noqa: E402
from panpv.campaign_eval import _context, fr_tile_metrics, valid_windows  # noqa: E402
from panpv.v3_data import SceneV3  # noqa: E402
import h5py  # noqa: E402
import panpv_qualitative as Q  # noqa: E402
from metrics import eval_fr as TM  # noqa: E402  (저장소 검증 구현)
from metrics.q2n import q2n as tq2n  # noqa: E402

torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
C = Q.CAMPAIGN
WALD = TM.load_dlpan(os.environ.get("PANCRAFTER_DLPAN", str(ROOT.parent / "DLPan-Toolbox")))
PROTO = json.load(open(C / "protocol.json"))["scenes"]
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def ckpt(sid, key):
    if key == "ZS_WV3":
        return C / "donor_pretrain/run/checkpoints/step_000050000/model.safetensors"
    meth = "SC_SCENE" if key.startswith("SC") else "FT_WV3_SCENE"
    run = C / f"runs/PV_S1_{sid}_{meth}_R01_SE261002"
    return run / json.load(open(run / "exact_endpoint.json"))["directory"] / "model.safetensors"


def model_for(sid, key):
    m = SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=261002)
    m.load_state_dict(load_file(str(ckpt(sid, key)), device="cpu"), strict=True)
    return m.cuda().eval()


@torch.no_grad()
def infer(model, s):
    t = [torch.from_numpy(s[k] * np.float32(2 / 2047) - np.float32(1))[None].cuda() for k in ("pan", "ms", "lpan")]
    return ((model("WV3", *t)[0].float() + 1) * np.float32(2047 / 2)).cpu().numpy()


def up4(ms):
    return F.interpolate(torch.from_numpy(ms)[None], scale_factor=4, mode="bicubic",
                         align_corners=False)[0].numpy()


def brovey_hybrid(sample, stored_core):
    """가운데 256 은 저장된 GDAL Brovey 그대로, 주변 문맥(필터용)만 같은 식의 numpy Brovey(등가중치)."""
    ms_up = up4(sample["ms"].astype(np.float32))
    pseudo = ms_up.mean(0)
    out = np.where(pseudo > 0, ms_up * sample["pan"][0] / np.maximum(pseudo, 1e-6), 0).astype(np.float32)
    out[:, 128:384, 128:384] = stored_core
    return out


def notch4(x):
    """4화소 주기 성분(주파수 (k/4, l/4), (k,l)≠(0,0)) 만 Gaussian notch 로 제거."""
    n = x.shape[-1]
    fy, fx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
    mask = np.ones((n, n))
    for k in range(4):
        for l in range(4):
            if k == l == 0:
                continue
            cy, cx = k * n // 4, l * n // 4
            dy = np.minimum(np.abs(fy % n - cy), n - np.abs(fy % n - cy))
            dx = np.minimum(np.abs(fx % n - cx), n - np.abs(fx % n - cx))
            mask *= 1 - np.exp(-(dy ** 2 + dx ** 2) / (2 * 1.5 ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * mask, axes=(-2, -1)))


def p4_ratio(pred512):
    core = np.clip(pred512, 0, 2047).mean(0)[128:384, 128:384]
    N = 256; c = N // 2; k = N // 4
    yy, xx = np.mgrid[:N, :N] - c
    ring = (np.hypot(yy, xx) > k - 2) & (np.hypot(yy, xx) < k + 2)
    P = Q._power(core)
    return float(np.mean([P[c, c + k], P[c, c - k], P[c + k, c], P[c - k, c]]) / np.median(P[ring]))


def alt_metrics(prediction, sample, origin, halo=128):
    """같은 창 · 같은 식, 저장소 검증 부품(tools/metrics, DLPan interp23tap)."""
    pred = np.clip(prediction.astype(np.float64), 0, 2047).transpose(1, 2, 0)
    pan = sample["pan"][0].astype(np.float64)
    ms = sample["ms"].astype(np.float64).transpose(1, 2, 0)
    lms = WALD.interp23tap(ms, 4)
    filt = TM.mtf_filter(pred, "wv2", 4)
    lp = WALD.interp23tap(TM.imresize_matlab(pan, .25)[..., None], 4)[..., 0]
    w = list(valid_windows(np.ones((256, 256), bool), origin))
    reg = np.s_[w[0][0].start + halo:w[-1][0].stop + halo, w[0][1].start + halo:w[-1][1].stop + halo]
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        dl = 1 - tq2n(lms[reg], filt[reg], 32, 32)[0]
        ds = np.mean([abs(TM._blockproc_uqi(pred[reg][..., b], pan[reg], 32)
                          - TM._blockproc_uqi(lms[reg][..., b], lp[reg], 32)) for b in range(8)])
    return dict(hqnr=(1 - dl) * (1 - ds), d_lambda=dl, d_s=ds)


def M(pred, sample, origin):
    m = fr_tile_metrics(pred, sample, origin)
    return dict(hqnr=m["metrics"]["hqnr"], d_lambda=m["metrics"]["d_lambda"], d_s=m["metrics"]["d_s"],
                signed_ds=float(np.mean(m["signed_ds"])))


out = {}
for sid, _ in Q.SCENES:
    scene = SceneV3(Q.SCENES_DIR / f"{sid}.h5", verify_bytes=False)
    audit = PROTO[sid]["audit"]
    sel = [audit[i] for i in np.linspace(0, len(audit) - 1, NPER).astype(int)]
    models = {k: model_for(sid, k) for k in ("ZS_WV3", "SC_SCENE_EXACT20K", "FT_WV3_SCENE_EXACT20K")}
    stored = {}
    for key in ("BROVEY8", "ZS_WV3", "SC_SCENE_EXACT20K", "FT_WV3_SCENE_EXACT20K"):
        rep = json.load(open(C / f"evaluations/{sid}_{key}/report.json"))
        stored[key] = {t["raw_index"]: t["metrics"] for t in rep["fr"]["per_tile"]}
    with h5py.File(C / f"evaluations/{sid}_BROVEY8/predictions.h5", "r") as ph:
        bidx = {int(x): j for j, x in enumerate(ph["raw_index"][:])}
        bcore = {e["raw_index"]: ph["fr/prediction"][bidx[e["raw_index"]]] for e in sel}
    rows = []
    for e in sel:
        s = _context(scene, e, "native")
        o = tuple(e["fr_origin"])
        preds = {"BROVEY8": brovey_hybrid(s, bcore[e["raw_index"]])}
        for k, m in models.items():
            preds[k] = infer(m, s)
        base = up4(s["ms"])
        r = {"raw_index": e["raw_index"]}
        for k, p in preds.items():
            r[k] = M(p, s, o)
            r[k]["stored_hqnr"] = stored[k][e["raw_index"]]["hqnr"]
            r[k]["alt"] = alt_metrics(p, s, o)
            r[k]["p4"] = p4_ratio(p)
        zs = preds["ZS_WV3"]
        zs_n = notch4(zs)
        grid = zs - zs_n
        r["ZS_notch"] = M(zs_n, s, o); r["ZS_notch"]["p4"] = p4_ratio(zs_n)
        r["BROVEY_plus_ZSgrid"] = M(preds["BROVEY8"] + grid, s, o)
        r["BROVEY_plus_ZSgrid"]["p4"] = p4_ratio(preds["BROVEY8"] + grid)
        r["ZS_resid_x2"] = M(base + 2.0 * (zs - base), s, o)
        for k in ("SC_SCENE_EXACT20K", "FT_WV3_SCENE_EXACT20K"):
            for a in (0.75, 0.5, 0.35):
                r[f"{k}_resid_x{a}"] = M(base + a * (preds[k] - base), s, o)
        rows.append(r)
    out[sid] = rows
    print(sid, "done", flush=True)
json.dump(out, open(Path(sys.argv[1]) / "hqnr_probe.json", "w"), indent=1)
