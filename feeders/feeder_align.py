"""PanFeeder + (sample_id, split, augmentation 파라미터) 메타 반환.

shift cache 를 표본별로 찾고(§5), augmentation 이 걸리면 shift vector 도 같이 변환해야(§6)
하므로 feeder 가 무엇을 했는지 밖으로 알려야 한다. 원본 feeder 의 난수 호출 순서
(crop 없음 -> hflip/vflip 은 플래그면 무조건 -> rot 은 randint) 를 그대로 재현한다.
meta = LongTensor [sample_id, split_code, hflip, vflip, rot]  (split_code 0 train·1 val·2 RR·3 FR)
"""
import random

import numpy as np
import torch

from feeders.feeder import PanFeeder

SPLIT_CODE = {"train": 0, "val": 1, "test_reduced": 2, "test_full": 3}


class PanFeederAlign(PanFeeder):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        assert not self.crop, "이 캠페인은 crop/scale jitter 를 쓰지 않는다 (계획 §0.1)"

    def _apply(self, arrs, hflip, vflip, rot):
        out = []
        for x in arrs:
            if hflip:
                x = x[:, ::-1, :]
            if vflip:
                x = x[::-1, :, :]
            if rot:
                x = np.rot90(x, rot, (0, 1))
            out.append(np.ascontiguousarray(x))
        return out

    def __getitem__(self, index):
        lms, ms = np.array(self.lms[index]), np.array(self.ms[index])
        lpan, pan = np.array(self.lpan[index]), np.array(self.pan[index])
        hflip = vflip = rot = 0
        if self.split == "train":
            hflip, vflip = int(bool(self.hflip)), int(bool(self.vflip))
            rot = random.randint(0, 3) if self.rot else 0      # 원본과 같은 난수 호출
        meta = torch.tensor([index, SPLIT_CODE[self.split], hflip, vflip, rot], dtype=torch.long)
        if self.has_gt:
            gt = np.array(self.gt[index])
            if self.split == "train":
                gt, lms, ms, lpan, pan = self._apply((gt, lms, ms, lpan, pan), hflip, vflip, rot)
            return (self.np2tensor(gt), self.np2tensor(lms), self.np2tensor(ms),
                    self.np2tensor(lpan), self.np2tensor(pan), meta)
        if self.split == "train":
            lms, ms, lpan, pan = self._apply((lms, ms, lpan, pan), hflip, vflip, rot)
        return self.np2tensor(lms), self.np2tensor(ms), self.np2tensor(lpan), self.np2tensor(pan), meta
