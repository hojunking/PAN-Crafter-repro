"""배포 pan_h5.zip 의 손상된 full-resolution lpan 을 재생성한다.

WV3·QB 의 test_*_OrigScale_multiExm1_pan.h5 는 짝이 되는 PAN 과 무관한 다른 장면이다
(상관 0.01). GF2 만 정상이다. results_log/2026-08-14_wv3-baseline-vs-fixed.md 3절 참고.

생성 레시피는 정상인 파일 10개(train/valid/test_reduced x 3센서 + test_gf2_OrigScale)에서
역추정했다: Gaussian(sigma=1.98, N=41, BORDER_REPLICATE) 후 [2::4, 2::4] 데시메이션.
RMSE 0.23~0.53 (값범위 0~2047) 으로 배포본을 재현한다.

  python tools/repair_lpan.py --sensor wv3

원본은 건드리지 않고 full_examples_h5_repaired/ 에 새로 만든다.
"""
import os, argparse, shutil
import numpy as np, h5py, cv2

SIGMA, KSIZE, OFFSET = 1.98, 41, 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_lpan(pan):
    k = cv2.getGaussianKernel(KSIZE, SIGMA); K = k @ k.T
    return np.stack([cv2.filter2D(p[0], -1, K, borderType=cv2.BORDER_REPLICATE)
                     [OFFSET::4, OFFSET::4][None] for p in pan])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", default="wv3", choices=["wv3", "qb", "gf2"])
    a = ap.parse_args()
    s, D = a.sensor, a.sensor.upper()

    src_dir = f"{ROOT}/data/PanCollection/{D}/full_examples_h5"
    dst_dir = f"{ROOT}/data/PanCollection/{D}/full_examples_h5_repaired"
    os.makedirs(dst_dir, exist_ok=True)
    name = f"test_{s}_OrigScale_multiExm1"

    # 본체 h5 는 심볼릭 링크 (수 백 MB 복사 불필요)
    link = f"{dst_dir}/{name}.h5"
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.realpath(f"{src_dir}/{name}.h5"), link)

    with h5py.File(f"{src_dir}/{name}.h5") as f:
        pan = f["pan"][:].astype(np.float64)
        ms_hw = f["ms"].shape[-2:]
    with h5py.File(f"{src_dir}/{name}_pan.h5") as f:
        old = f["lpan"][:].astype(np.float64)

    new = make_lpan(pan)
    assert new.shape[-2:] == ms_hw, f"해상도 불일치 {new.shape[-2:]} != ms {ms_hw}"

    cor = lambda a, b: float(np.mean([np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1]
                                      for i in range(len(a))]))
    dn = make_lpan(pan)
    print(f"[{s}] pan{pan.shape[-2:]} -> lpan{new.shape[-2:]}  (ms{ms_hw})")
    print(f"  배포본 lpan  vs PAN 다운샘플 : 상관 {cor(old, dn):+.4f}   {'<-- 손상' if cor(old,dn)<0.9 else ''}")
    print(f"  재생성 lpan  vs PAN 다운샘플 : 상관 {cor(new, dn):+.4f}")

    out = f"{dst_dir}/{name}_pan.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("lpan", data=new.astype(np.float32))
    print(f"  저장: {os.path.relpath(out, ROOT)}  ({os.path.getsize(out)/1e6:.1f} MB)")

    # reduced 쪽은 정상이므로 그대로 링크해 둔다 (config 가 한 곳만 보게)
    print(f"  원본은 그대로 둔다: {os.path.relpath(src_dir, ROOT)}/")


if __name__ == "__main__":
    main()
