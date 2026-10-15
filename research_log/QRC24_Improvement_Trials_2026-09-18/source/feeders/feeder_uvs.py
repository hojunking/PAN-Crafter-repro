"""UVS-KD feeder: 원 PanFeeder + Teacher cache(R_T, U_T, δ_T, c_T) 를 **같은 flip/rot 로** 증강해 함께 돌려준다 (계획 §5.5).

- 영상(gt, lms, ms, lpan, pan) 증강은 원 feeder 와 동일(hflip/vflip 은 플래그면 무조건, rot 은 randint).
  bicubic 경로는 flip 대칭이라 LR 증강이 안전하고, lms 는 제공 HR 텐서라 HR 에서 flip 된다.
- R_T, U_T 는 HR 맵이므로 영상과 같이 flip/rot. δ_T 는 벡터 규칙 (§5.5): hflip (dy,−dx), vflip (−dy,dx),
  rot90 CCW k: (dy,dx)→(−dx,dy) 를 k 번. c_T 는 불변. 이 변환은 tools/test_uvs.py 에서 impulse 로 검증한다.
- cache 가 없으면(B0 등) teacher 항목은 0 으로 채운다 (형상 유지).
반환(train): gt, lms, ms, lpan, pan, r_t[8,64,64], u_t[1,64,64], delta_t[2], c_t[1], idx[1]
"""
import os
import random

import numpy as np
import torch

from feeders.feeder import PanFeeder


def transform_delta_np(d, hflip, vflip, rot):
    dy, dx = float(d[0]), float(d[1])
    if hflip:
        dx = -dx
    if vflip:
        dy = -dy
    for _ in range(rot % 4):
        dy, dx = -dx, dy
    return np.array([dy, dx], dtype=np.float32)


class PanFeederUVS(PanFeeder):
    def __init__(self, *a, teacher_cache=None, **k):
        super().__init__(*a, **k)
        assert not self.crop, "UVS 캠페인은 crop 없음 (§11)"
        self.cache = None
        if teacher_cache and self.split == "train":
            assert os.path.exists(teacher_cache), f"teacher cache 없음: {teacher_cache}"
            z = np.load(teacher_cache)
            n = self.pan.shape[0]
            assert z["r_t"].shape[0] == n and z["u_t"].shape[0] == n, "cache 표본 수가 train h5 와 다르다"
            self.cache = dict(r_t=z["r_t"], u_t=z["u_t"], delta_t=z["delta_t"].astype(np.float32),
                              c_t=z["c_t"].astype(np.float32))

    def _aug_np(self, x, hflip, vflip, rot):       # x: HWC
        if hflip:
            x = x[:, ::-1, :]
        if vflip:
            x = x[::-1, :, :]
        if rot:
            x = np.rot90(x, rot, (0, 1))
        return np.ascontiguousarray(x)

    def __getitem__(self, index):
        lms, ms = np.array(self.lms[index]), np.array(self.ms[index])
        lpan, pan = np.array(self.lpan[index]), np.array(self.pan[index])
        gt = np.array(self.gt[index]) if self.has_gt else None
        hflip = vflip = rot = 0
        if self.split == "train":
            hflip, vflip = int(bool(self.hflip)), int(bool(self.vflip))
            rot = random.randint(0, 3) if self.rot else 0        # 원 feeder 와 같은 난수 호출
            gt, lms, ms, lpan, pan = (self._aug_np(x, hflip, vflip, rot) for x in (gt, lms, ms, lpan, pan))
        out = [self.np2tensor(x) for x in ((gt,) if gt is not None else ()) + (lms, ms, lpan, pan)]
        if self.split != "train":
            return tuple(out)
        H = pan.shape[0]
        if self.cache is not None:
            r_t = self._aug_np(self.cache["r_t"][index].transpose(1, 2, 0).astype(np.float32), hflip, vflip, rot)
            u_t = self._aug_np(self.cache["u_t"][index].transpose(1, 2, 0).astype(np.float32), hflip, vflip, rot)
            d_t = transform_delta_np(self.cache["delta_t"][index], hflip, vflip, rot)
            c_t = self.cache["c_t"][index:index + 1]
            r_t = torch.from_numpy(r_t.transpose(2, 0, 1).copy())     # 이미 정규화 단위([-1,1] 잔차) 로 저장됨
            u_t = torch.from_numpy(u_t.transpose(2, 0, 1).copy())
        else:
            r_t = torch.zeros(self.ms.shape[-1], H, H); u_t = torch.zeros(1, H, H)
            d_t = np.zeros(2, dtype=np.float32); c_t = np.zeros(1, dtype=np.float32)
        return tuple(out) + (r_t, u_t, torch.from_numpy(d_t), torch.from_numpy(np.asarray(c_t, dtype=np.float32)),
                             torch.tensor([index], dtype=torch.long))
