"""WV2 zero-shot 평가용 데이터 배치.

WV2 는 학습에 쓰지 않은 위성이다(논문 Table 3 의 zero-shot 벤치마크).
CANConv 에 테스트셋은 있으나 저자 배포 `*_pan.h5`(lpan) 가 없어서 직접 만든다.

생성 레시피는 WV3/QB/GF2 배포본 10개 파일에서 역추정한 것과 동일하다
(Gaussian sigma=1.98, N=41, REPLICATE -> [2::4, 2::4]). 센서에 무관하게 세 센서 모두
RMSE 0.23~0.53 으로 재현했으므로 WV2 에도 같은 것을 적용한다. 대조할 배포본이 없어
**검증은 불가능하다** — 이 가정은 결과 해석 시 명시할 것.

  python tools/setup_wv2.py
"""
import os, sys
import numpy as np, h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from repair_lpan import make_lpan  # noqa: E402

SRC = os.path.join(os.environ.get("PANCRAFTER_CANCONV", "/home/knuvi/Desktop/song/CANConv"), "data/datasets/wv2")
DST = f"{ROOT}/data/PanCollection/WV2"
JOBS = [("reduced_examples_h5", "test_wv2_multiExm1"),
        ("full_examples_h5",    "test_wv2_OrigScale_multiExm1")]

for sub, name in JOBS:
    d = f"{DST}/{sub}"; os.makedirs(d, exist_ok=True)
    link = f"{d}/{name}.h5"
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(f"{SRC}/{name}.h5", link)

    with h5py.File(link) as f:
        pan = f["pan"][:].astype(np.float64)
        ms_hw = f["ms"].shape[-2:]
    lpan = make_lpan(pan)
    assert lpan.shape[-2:] == ms_hw, f"{lpan.shape[-2:]} != ms {ms_hw}"
    with h5py.File(f"{d}/{name}_pan.h5", "w") as f:
        f.create_dataset("lpan", data=lpan.astype(np.float32))
    print(f"  {sub}/{name}: pan{pan.shape[-2:]} -> lpan{lpan.shape[-2:]}  "
          f"(ms{ms_hw})  생성 완료")

print(f"\n배치 위치: {os.path.relpath(DST, ROOT)}")
print("주의: WV2 lpan 은 저자 배포본이 없어 대조 검증이 불가능하다(레시피 이식).")
