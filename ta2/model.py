"""TA2 Teacher-only P0: native observations in, no GT/inference dependency.

The U implementation is reused unchanged. Only its explicitly supplied input
and residual reference differ in the registered U00--U03 cases. Construction
uses separate, restored CPU RNG streams, so removing A never changes U.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from model.pancrafter_paper import PANCrafterPaper
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan


POLICIES = frozenset(("BYPASS", "ZERO", "GLOBAL_FIXED", "PER_IMAGE_GRAD",
                      "PER_IMAGE_INTENSITY", "PER_IMAGE_BICUBIC",
                      "TRAIN_GT_PROXY", "GLOBAL_LEARNED", "SHUFFLED", "LEARNED"))
FIXED_POLICIES = POLICIES - {"BYPASS", "ZERO", "GLOBAL_LEARNED", "LEARNED"}
REFERENCES = frozenset(("NATIVE_LMS", "BICUBIC_MS", "NA"))


def _role_seed(seed, role):
    payload = f"TA2_FRESH_SPLIT_RNG_v1|{int(seed)}|{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class TeacherModel(nn.Module):
    """Fixed W112/D123/LN/attention-free U plus optional global PAN aligner.

    Aligner views use the full provided P/reference fields: no unregistered
    margin crop. No LPAN, HPAN, mode modulation, MARs, KD or donor lookup exists.
    ``shift`` is an explicit current-PAN-pixel correction (dy,dx), never GT.
    Offline policies require the caller's provenance-checked cached shift.
    """

    def __init__(self, bands, seed, config):
        super().__init__()
        if bands not in (4, 8):
            raise ValueError("TA2 supports 4-band QB/GF2 and 8-band WV3 only")
        self.bands, self.seed, self.config = int(bands), int(seed), dict(config)
        self.policy = config["aligner_policy"]
        self.a_reference = config["a_reference"]
        self.u_reference = config["u_reference"]
        if self.policy not in POLICIES or self.a_reference not in REFERENCES:
            raise ValueError("Unregistered aligner policy/reference")
        if self.u_reference not in REFERENCES - {"NA"}:
            raise ValueError("U requires a registered MS reference")
        if self.policy == "LEARNED" and self.a_reference == "NA":
            raise ValueError("Learned A requires an MS observation reference")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_role_seed(seed, "U"))
            self.backbone = PANCrafterPaper(
                out_channels=bands, hidden_size=112, depth=(1, 2, 3),
                dropout=0., n_attn=0, attn_locations=[], norm="ln",
                in_mode="paper", mode_modulation=False)
        self.aligner = None
        if self.policy == "LEARNED":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(_role_seed(seed, "A"))
                self.aligner = PANGlobalAligner(ms_bands=bands)
        self.global_shift = (nn.Parameter(torch.zeros(2))
                             if self.policy == "GLOBAL_LEARNED" else None)

    def u_parameters(self):
        return self.backbone.parameters()

    def a_parameters(self):
        if self.aligner is not None:
            return self.aligner.parameters()
        return iter(()) if self.global_shift is None else iter((self.global_shift,))

    def references(self, pan, ms, lms):
        if (pan.ndim != 4 or pan.shape[1] != 1 or ms.ndim != 4
                or ms.shape[:2] != (pan.shape[0], self.bands)
                or tuple(pan.shape[-2:]) != tuple(4 * d for d in ms.shape[-2:])):
            raise ValueError("Expected PAN[B,1,H,W], MS[B,C,H/4,W/4]")
        need_lms = "NATIVE_LMS" in (self.a_reference, self.u_reference)
        if need_lms and (lms is None or lms.shape != (pan.shape[0], self.bands, *pan.shape[-2:])):
            raise ValueError("BLOCKED_LMS_LINEAGE: explicit matching native LMS is mandatory")
        bicubic = None
        if "BICUBIC_MS" in (self.a_reference, self.u_reference):
            # B is generated after corresponding LR augmentation, as in PAModel.
            bicubic = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
        references = {"BICUBIC_MS": bicubic, "NATIVE_LMS": lms, "NA": None}
        return references[self.a_reference], references[self.u_reference]

    def predict_correction(self, pan, reference, shift=None):
        if shift is not None:
            correction = shift.to(device=pan.device, dtype=pan.dtype)
            if correction.shape == (2,):
                correction = correction[None].expand(pan.shape[0], -1)
        elif self.policy == "LEARNED":
            if reference is None or reference.shape[-2:] != pan.shape[-2:]:
                raise ValueError("A requires matching full-resolution reference")
            correction = self.aligner(pan, reference)
        elif self.policy == "GLOBAL_LEARNED":
            correction = self.global_shift[None].expand(pan.shape[0], -1)
        elif self.policy in FIXED_POLICIES:
            raise ValueError(f"{self.policy} requires an explicit verified per-input shift")
        else:
            correction = pan.new_zeros((pan.shape[0], 2))
        if correction.shape != (pan.shape[0], 2) or not bool(torch.isfinite(correction).all()):
            raise ValueError("NONFINITE_OR_INVALID_CORRECTION")
        return correction

    def correction(self, pan, ms, lms, shift=None):
        reference, _ = self.references(pan, ms, lms)
        return self.predict_correction(pan, reference, shift)

    def reconstruction(self, pan, ms, lms, correction, detach_correction=False,
                       force_warp=False):
        reference, base = self.references(pan, ms, lms)
        rec_correction = correction.detach() if detach_correction else correction
        aligned = (pan if self.policy == "BYPASS" and not force_warp
                   else warp_pan(pan, rec_correction))
        switches = pan.new_ones(pan.shape[0])
        residual = self.backbone(aligned, None, ms, switches,
                                 x_in=torch.cat((aligned, base), dim=1))
        prediction = base + residual
        return dict(prediction=prediction, correction=correction, warped_pan=aligned,
                    reference=reference, u_reference=base, y=prediction,
                    delta=correction, pan_aligned=aligned, ms_base=base)

    def forward(self, pan, ms, lms, shift=None):
        reference, _ = self.references(pan, ms, lms)
        correction = self.predict_correction(pan, reference, shift)
        return self.reconstruction(pan, ms, lms, correction,
                                   force_warp=shift is not None)

    def initial_manifest(self):
        return dict(policy="TA2_FRESH_SPLIT_RNG_v1", seed=self.seed,
                    U_sha256=state_hash(self.backbone),
                    A_sha256=None if self.aligner is None else state_hash(self.aligner),
                    global_parameters=0 if self.global_shift is None else 2,
                    width=112, depth=[1, 2, 3], bands=self.bands,
                    aligner_view_margin=0, input_layout="P0", donor_loads=0)
