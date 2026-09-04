"""Global shift cache (계획 §5.3–5.4). parquet 대신 CSV — 이 환경에 pyarrow 가 없고, 형식은
본질이 아니다. 열 이름은 계획 그대로다.

    outputs/global_shift_cache/
    ├── wv3_train.csv   (9714 patch, 16x16 LR 단위 추정 — parent scene ID 가 h5 에 없어 §5.2 우선순위 2)
    ├── wv3_rr.csv      (20 scene, 64x64 LR)
    ├── wv3_fr.csv      (20 scene, 128x128 LR)
    └── cache_meta.json (부호 규약·게이트·estimator_version·각 CSV 의 SHA256)

lookup 은 (split_code, sample_id) 로 한다. split_code: 0 train · 1 val · 2 test_reduced · 3 test_full.
val 은 선택에 쓰지 않으므로 cache 가 없고 zeros 를 돌려준다.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

SPLIT_FILE = {0: "wv3_train.csv", 2: "wv3_rr.csv", 3: "wv3_fr.csv"}
ESTIMATOR_VERSION = "scharr_zncc_q3x3+census5_hamming/v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_h5(path):
    """수 GB h5 를 통째로 해시하지 않는다 — (size, mtime, 앞/뒤 1MB) 지문."""
    st = os.stat(path)
    h = hashlib.sha256(f"{st.st_size}:{int(st.st_mtime)}".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, st.st_size - (1 << 20)))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


class ShiftCache:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        meta_p = os.path.join(cache_dir, "cache_meta.json")
        assert os.path.exists(meta_p), f"shift cache 없음: {meta_p} — tools/build_shift_cache.py 먼저"
        self.meta = json.load(open(meta_p))
        self.tables, self.sha = {}, {}
        for code, fn in SPLIT_FILE.items():
            p = os.path.join(cache_dir, fn)
            if not os.path.exists(p):
                continue
            sha = sha256_file(p)
            assert self.meta["sha256"].get(fn) == sha, f"cache 변조/불일치: {fn}"
            df = pd.read_csv(p).sort_values("sample_id")
            assert (df["sample_id"].values == np.arange(len(df))).all(), f"{fn}: sample_id 가 0..N-1 이 아니다"
            self.tables[code] = dict(
                applied=torch.tensor(df[["dy_lr_applied", "dx_lr_applied"]].values, dtype=torch.float32),
                raw=torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32),
                accepted=torch.tensor(df["accepted"].values.astype(bool)))
            self.sha[fn] = sha
        self.sha256_all = hashlib.sha256("".join(sorted(self.sha.values())).encode()).hexdigest()

    def has(self, code):
        return int(code) in self.tables

    def lookup(self, split_code, idx, device=None, raw=False):
        """split_code: int 또는 [B] tensor(동일 split 가정). idx: [B] long. -> (delta[B,2], accepted[B])"""
        code = int(split_code if not torch.is_tensor(split_code) else split_code.flatten()[0])
        idx = idx.detach().cpu().long()
        if code not in self.tables:                       # val 등 — zeros, accepted False
            z = torch.zeros(len(idx), 2)
            return (z.to(device) if device else z), torch.zeros(len(idx), dtype=torch.bool)
        t = self.tables[code]
        d = (t["raw"] if raw else t["applied"])[idx]
        a = t["accepted"][idx]
        if device is not None:
            d, a = d.to(device), a.to(device)
        return d, a
