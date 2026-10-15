"""PAN-Crafter 데이터 배치 검증.

config/*.yaml 의 4개 feeder dataroot 에 대해
  - 본체 h5 와 짝이 되는 _pan.h5 가 모두 존재하는지
  - 샘플 수(N)가 일치하는지
  - lpan 의 공간 해상도가 ms 와 같은지 (PAN 을 4배 다운샘플한 것)
  - feeder 의 문자열 기반 split / max_pixel 추론이 의도대로 동작하는지
를 확인한다.

usage: python tools/check_data.py [config/pancrafter_wv3.yaml ...]
"""

import sys
import os
import glob

import h5py
import yaml

KEYS = ["train_feeder_args", "val_feeder_args",
        "test_reduced_feeder_args", "test_full_feeder_args"]


def infer_split(dataroot):
    # feeders/feeder.py 와 동일한 분기
    for token, split in (("train", "train"), ("valid", "val"),
                         ("reduced", "test_reduced"), ("full", "test_full")):
        if token in dataroot:
            return split
    return None


def infer_max_pixel(dataroot):
    # feeders/feeder.py 와 동일한 분기 (max_pixel 인자를 넘기지 않을 때)
    if "wv3" in dataroot or "qb" in dataroot or "wv2" in dataroot:
        return 2047.
    if "gf2" in dataroot:
        return 1023.
    return None


def check(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n=== {cfg_path} ===")
    print(f"num_bands={cfg['num_bands']}  max_pixel={cfg['max_pixel']}  "
          f"work_dir={cfg['work_dir']}")
    ok = True

    for key in KEYS:
        root = cfg[key]["dataroot"]
        pan_root = root.replace(".h5", "_pan.h5")
        print(f"\n[{key}]")
        print(f"  {root}")

        for p in (root, pan_root):
            if not os.path.exists(p):
                print(f"  !! MISSING: {p}")
                ok = False
        if not (os.path.exists(root) and os.path.exists(pan_root)):
            continue

        with h5py.File(root, "r") as h:
            shapes = {k: h[k].shape for k in h.keys()}
            dtypes = {k: str(h[k].dtype) for k in h.keys()}
        with h5py.File(pan_root, "r") as h:
            shapes["lpan"] = h["lpan"].shape
            dtypes["lpan"] = str(h["lpan"].dtype)

        for k in ("gt", "lms", "ms", "pan", "lpan"):
            if k in shapes:
                print(f"    {k:<5} {str(shapes[k]):<26} {dtypes[k]}")

        n = shapes["pan"][0]
        if shapes["lpan"][0] != n:
            print(f"  !! N mismatch: pan={n} vs lpan={shapes['lpan'][0]}")
            ok = False
        if shapes["lpan"][2:] != shapes["ms"][2:]:
            print(f"  !! lpan HW {shapes['lpan'][2:]} != ms HW {shapes['ms'][2:]}")
            ok = False
        if shapes["pan"][2] != 4 * shapes["ms"][2]:
            print(f"  !! pan HW is not 4x ms HW")
            ok = False

        split, mp = infer_split(root), infer_max_pixel(root)
        expect = {"train_feeder_args": "train", "val_feeder_args": "val",
                  "test_reduced_feeder_args": "test_reduced",
                  "test_full_feeder_args": "test_full"}[key]
        print(f"    split={split} (expect {expect})   inferred max_pixel={mp}")
        if split != expect:
            print(f"  !! split 추론 실패")
            ok = False
        if mp != cfg["max_pixel"]:
            print(f"  !! feeder 추론 max_pixel({mp}) != config max_pixel({cfg['max_pixel']})")
            ok = False
        if "gt" in shapes and shapes["gt"][1] != cfg["num_bands"]:
            print(f"  !! band 수 {shapes['gt'][1]} != num_bands {cfg['num_bands']}")
            ok = False

    print(f"\n>>> {os.path.basename(cfg_path)}: {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("config/pancrafter_*.yaml"))
    results = [check(p) for p in paths]
    print("\n" + "=" * 60)
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
