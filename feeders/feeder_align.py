"""PanFeeder + (sample_id, split, augmentation 파라미터) 메타 반환.

**LR(ms, lpan) 은 증강하지 않는다.** HR(gt, lms, pan) 만 flip/rot 하고, LR 은 원본 격자 그대로
돌려준다 — wrapper(align/model.py)가 LR 을 interp23tap 으로 올린 *뒤에* 같은 flip/rot 을 HR 에서 건다.

이유 (2026-09-05 검증 지적): phase-2 격자(LR j -> HR 4j+2)는 flip 대칭이 아니다. LR 을 먼저 flip 하고
올리면 62-4j 인데 HR 을 flip 하면 61-4j 라 **축마다 1 HR px 어긋난다.** 실측: hflip+vflip 고정 + rot 0/1/3
에서 interp23tap(aug_LR(ms)) 와 aug_HR(lms) 의 MAD 22.7 / 17.0 / 14.8 DN (1px roll 로 0.000) — 표본의 75%.
원본 feeder 의 bicubic(phase 1.5)은 flip 대칭이라 이 문제가 없었다.

원본 feeder 의 난수 호출 순서(crop 없음 -> hflip/vflip 은 플래그면 무조건 -> rot 은 randint)를 그대로 재현한다.
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
                gt, lms, pan = self._apply((gt, lms, pan), hflip, vflip, rot)    # HR 만. LR 은 원본 격자
            return (self.np2tensor(gt), self.np2tensor(lms), self.np2tensor(ms),
                    self.np2tensor(lpan), self.np2tensor(pan), meta)
        if self.split == "train":
            lms, pan = self._apply((lms, pan), hflip, vflip, rot)
        return self.np2tensor(lms), self.np2tensor(ms), self.np2tensor(lpan), self.np2tensor(pan), meta
