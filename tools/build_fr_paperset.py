"""PanCollection 의 **.mat 형식** WV3 full-resolution 테스트셋(20장)을 PAN-Crafter 입력 h5 로 만든다.

왜 필요한가 (results_log/2026-09-07_metric-comparability-audit.md):
  배포 H5 `test_wv3_OrigScale_multiExm1.h5` 의 20장과 .mat 형식 `Test(HxWxC)_wv3_data_fr{1..20}.mat`
  의 20장은 **다른 장면 집합**이다 (겹치는 장면 6장뿐). PAN-Crafter / U-Know-DiffPAN / CANConv 가
  보고한 FR 수치는 .mat 쪽(MATLAB DLPan-Toolbox 입력)과 일치한다 — EXP 기준선이 평균·표준편차까지
  맞는다(D_λ 0.0231±0.0064 / D_s 0.0814±0.0310 / HQNR 0.8976±0.0353 vs 논문 0.0232±0.0066 /
  0.0813±0.0318 / 0.897±0.036). 논문 표와 비교할 FR 수치는 이 세트로 내야 한다.

    python tools/build_fr_paperset.py --src <mat 폴더> [--sensor wv3]

출력: data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5   (lms, ms, pan — 원 h5 와 같은 NCHW float64)
      data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20_pan.h5 (lpan — tools/repair_lpan.py 레시피로 생성)
      같은 폴더의 provenance.json (원본 파일명·sha256·생성 시각)

lpan 은 배포본에 없다. 배포 train/valid/test 파일에서 역추정한 레시피(Gaussian σ=1.98, 41탭,
replicate, [2::4] 데시메이션, RMSE 0.2~0.5 DN)로 만든다. 경로에 'full' 과 센서명이 들어가야
feeder 가 FR 모드·max_pixel 을 맞게 잡는다.
"""
import os, re, sys, glob, json, hashlib, argparse, datetime
import numpy as np, h5py
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.repair_lpan import make_lpan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Test(HxWxC)_<sensor>_data_fr*.mat 이 있는 폴더")
    ap.add_argument("--sensor", default="wv3")
    a = ap.parse_args()
    s, D = a.sensor, a.sensor.upper()
    files = sorted(glob.glob(os.path.join(a.src, f"Test(HxWxC)_{s}_data_fr*.mat")),
                   key=lambda p: int(re.search(r"fr(\d+)\.mat", p).group(1)))
    assert len(files) == 20, f"{len(files)}개 — 20장이어야 한다"
    lms, ms, pan, prov = [], [], [], []
    for p in files:
        m = loadmat(p)
        lms.append(m["lms"].astype(np.float64).transpose(2, 0, 1))
        ms.append(m["ms"].astype(np.float64).transpose(2, 0, 1))
        pn = m["pan"].astype(np.float64)
        pan.append(pn[None] if pn.ndim == 2 else pn.transpose(2, 0, 1))
        prov.append({"file": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    lms, ms, pan = np.stack(lms), np.stack(ms), np.stack(pan)
    out_dir = os.path.join(ROOT, "data", "PanCollection", D, "full_examples_mat20")
    os.makedirs(out_dir, exist_ok=True)
    name = f"test_{s}_OrigScale_mat20"
    with h5py.File(os.path.join(out_dir, name + ".h5"), "w") as f:
        f.create_dataset("lms", data=lms); f.create_dataset("ms", data=ms); f.create_dataset("pan", data=pan)
        f.attrs["source"] = "PanCollection Testing Dataset (FullData, mat Format) — Google Drive folder 16pGIqvwWfyQVvkk3s1xrwLpavqQd0Bv7"
    lpan = make_lpan(pan)
    with h5py.File(os.path.join(out_dir, name + "_pan.h5"), "w") as f:
        f.create_dataset("lpan", data=lpan.astype(np.float64))
    json.dump({"built_at": datetime.datetime.now().isoformat(timespec="seconds"), "files": prov,
               "lpan": "tools/repair_lpan.make_lpan (Gaussian s1.98 k41 replicate, [2::4])"},
              open(os.path.join(out_dir, "provenance.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{out_dir}: lms{lms.shape} ms{ms.shape} pan{pan.shape} lpan{lpan.shape}  "
          f"pan[{pan.min():.0f},{pan.max():.0f}] ms[{ms.min():.0f},{ms.max():.0f}]")


if __name__ == "__main__":
    main()
