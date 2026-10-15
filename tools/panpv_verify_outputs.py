"""PANPV 신경망 출력의 격자 무늬 · 고주파 잡음이 실제 모델 출력인지 검증.

A. 저장 예측 재현: 같은 체크포인트 · 같은 입력으로 다시 추론 → 저장된 fr/prediction 과 비교
B. 타일 없이 한 번 추론: 영역 전체(+128 문맥)를 한 번에 넣고 타일 결합본과 비교
C. 입력 저주파 PAN(native/lpan) 재계산 일치
D. 고주파 이득: 출력 밝기 고주파 / PAN 고주파 (저주파로 척도 맞춘 뒤) — Brovey 는 정의상 1
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from safetensors.torch import load_file  # noqa: E402
from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet  # noqa: E402
from deploy.pandep_shared.project.src.pan_shared.frontend import make_lpan  # noqa: E402
import panpv_qualitative as Q  # noqa: E402

C = Q.CAMPAIGN
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def ckpt(sid, key):
    if key == "ZS_WV3":
        return C / "donor_pretrain/run/checkpoints/step_000050000/model.safetensors"
    meth = "SC_SCENE" if key.startswith("SC") else "FT_WV3_SCENE"
    run = C / f"runs/PV_S1_{sid}_{meth}_R01_SE261002"
    ep = json.load(open(run / "exact_endpoint.json"))
    return run / ep["directory"] / "model.safetensors"


def model_for(sid, key):
    m = SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=261002)
    m.load_state_dict(load_file(str(ckpt(sid, key)), device="cpu"), strict=True)
    return m.cuda().eval()


def read_native(sh, r, c, h, w):
    pan = sh["native/pan"][0, :, 4 * r:4 * (r + h), 4 * c:4 * (c + w)].astype(np.float32)
    ms = sh["native/ms"][0, :, r:r + h, c:c + w].astype(np.float32)
    lp = sh["native/lpan"][0, :, r:r + h, c:c + w].astype(np.float32)
    return pan, ms, lp


@torch.no_grad()
def infer(model, pan, ms, lp):
    t = [torch.from_numpy(a * np.float32(2 / 2047) - np.float32(1))[None].cuda() for a in (pan, ms, lp)]
    y = model("WV3", *t)
    return ((y[0].float() + 1) * np.float32(2047 / 2)).cpu().numpy()


def detail_gain(out_int, pan):
    """저주파(σ4) 로 척도 a 를 맞춘 뒤 고주파 회귀 기울기 / a."""
    import cv2
    lo_o, lo_p = cv2.GaussianBlur(out_int, (0, 0), 4), cv2.GaussianBlur(pan, (0, 0), 4)
    a = np.polyfit(lo_p.ravel(), lo_o.ravel(), 1)[0]
    hp_o, hp_p = out_int - cv2.GaussianBlur(out_int, (0, 0), 1.0), pan - cv2.GaussianBlur(pan, (0, 0), 1.0)
    g = float((hp_o * hp_p).sum() / (hp_p * hp_p).sum())
    resid = hp_o - g * hp_p
    return g / a, float(resid.std() / (a * hp_p.std()))  # 이득, PAN 과 무관한 고주파 크기(PAN 고주파 대비)


REGIONS = [("200014103764_01", (2816, 7296)), ("200014103525_01", (2688, 4608))]
KEYS = ["ZS_WV3", "SC_SCENE_EXACT20K", "FT_WV3_SCENE_EXACT20K"]
report = {}
for sid, (r0, c0) in REGIONS:
    sh = h5py.File(Q.SCENES_DIR / f"{sid}.h5", "r")
    rep = report.setdefault(sid, {})
    # C. LPAN 재계산 (영역 + 32 MS 문맥, 가운데만 비교)
    pan_c, _, lp_c = read_native(sh, r0 - 32, c0 - 32, 192, 192)
    lp_re = make_lpan(pan_c[None])[0]
    rep["lpan_recompute_maxabs_center"] = float(np.abs(lp_re - lp_c)[:, 16:-16, 16:-16].max())
    for key in KEYS:
        model = model_for(sid, key)
        with h5py.File(C / f"evaluations/{sid}_{key}/predictions.h5", "r") as ph:
            idx = {(int(a), int(b)): k for k, (a, b) in enumerate(ph["canonical_ms_row_col"][:])}
            stored = Q.stitch(ph, idx, r0, c0, 2)
        # A. 타일 단위 재현 (평가와 같은 512 입력 = 256 + 128 문맥)
        tile_max = 0.0
        for i in range(2):
            for j in range(2):
                r, c = r0 + 64 * i, c0 + 64 * j
                y = infer(model, *read_native(sh, r - 32, c - 32, 128, 128))[:, 128:384, 128:384]
                tile_max = max(tile_max, float(np.abs(y - stored[:, 256 * i:256 * (i + 1), 256 * j:256 * (j + 1)]).max()))
        # B. 영역 한 번에 (512 + 문맥 128)
        y_full = infer(model, *read_native(sh, r0 - 32, c0 - 32, 192, 192))[:, 128:640, 128:640]
        d = np.abs(y_full - stored)
        pan = sh["native/pan"][0, 0, 4 * r0:4 * r0 + 512, 4 * c0:4 * c0 + 512].astype(np.float32)
        g_s, n_s = detail_gain(np.clip(stored, 0, 2047).mean(0), pan)
        g_f, n_f = detail_gain(np.clip(y_full, 0, 2047).mean(0), pan)
        rep[key] = dict(tile_repro_maxabs_DN=tile_max, single_pass_vs_tiles_maxabs_DN=float(d.max()),
                        single_pass_vs_tiles_meanabs_DN=float(d.mean()),
                        detail_gain_tiles=g_s, detail_gain_single=g_f, unrelated_hf_tiles=n_s,
                        unrelated_hf_single=n_f)
        np.save(Path(sys.argv[1]) / f"single_{sid}_{key}.npy", y_full)
        print(sid, key, json.dumps({k: round(v, 4) for k, v in rep[key].items()}), flush=True)
    with h5py.File(C / f"evaluations/{sid}_BROVEY8/predictions.h5", "r") as ph:
        idx = {(int(a), int(b)): k for k, (a, b) in enumerate(ph["canonical_ms_row_col"][:])}
        b = Q.stitch(ph, idx, r0, c0, 2)
    pan = sh["native/pan"][0, 0, 4 * r0:4 * r0 + 512, 4 * c0:4 * c0 + 512].astype(np.float32)
    rep["BROVEY8"] = dict(zip(("detail_gain", "unrelated_hf"), detail_gain(np.clip(b, 0, 2047).mean(0), pan)))
    print(sid, "lpan", rep["lpan_recompute_maxabs_center"], "brovey", rep["BROVEY8"], flush=True)
json.dump(report, open(Path(sys.argv[1]) / "verify_report.json", "w"), indent=1)
