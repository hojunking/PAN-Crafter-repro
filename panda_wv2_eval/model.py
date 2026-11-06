"""Reconstruct the exact frozen A+S from each same-state WV3 checkpoint."""
import torch
from torch import nn
from panda_wv3_ablation.model import Stage2 as Operators, build_aligner
from panda_tune30.reconstruction import StudentReconstructionSpec, build_reconstructor
from panda_method.calibration import state_dict_sha256
from .common import ROOT, OUT, digest, sha, source_evidence


class Deployment(nn.Module):
    corrected = Operators.corrected
    reconstruct = Operators.reconstruct
    native_prediction = Operators.native_prediction

    def __init__(self, case):
        super().__init__()
        cfg, ev = source_evidence(case)
        self.cfg = cfg
        path = OUT / 'sources' / case['case_id'] / 'model.pt'
        if sha(path) != case['checkpoint_sha256']:
            raise ValueError('Preserved evaluation checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['config'] != cfg or saved['updates'] != case['source_step']:
            raise ValueError('Wrong saved source configuration/step')
        pick = lambda role: {k[len(role)+1:]: v for k, v in saved['model'].items() if k.startswith(role+'.')}
        roles = {r: state_dict_sha256(pick(r)) for r in ('aligner', 'teacher', 'student')}
        if digest(roles) != ev['model_sha256'] or saved['model_sha256'] != ev['model_sha256']:
            raise ValueError('A/T/S state hash differs from official WV3 evaluation')
        self.aligner = build_aligner(cfg['stream_seeds']['A_init'])
        self.student = build_reconstructor(StudentReconstructionSpec(8, case['width'], (1, 2, 3), 'P0', 'BICUBIC_MS', 2),
            seed=cfg['stream_seeds']['S_init'], role='S', sampling_variant='UBIL', input_variant='CORRECTED_DIFF')
        for role in ('aligner', 'student'):
            getattr(self, role).load_state_dict(pick(role), strict=True)
            if state_dict_sha256(getattr(self, role)) != roles[role]:
                raise ValueError('Loaded deployment weights differ')
        self.requires_grad_(False).eval()
        self.source_evaluator = saved['pins']['evaluator']

    @property
    def teacher(self):
        return self.student  # historical profiler role alias; no Teacher inference

    def weight_hash(self):
        return digest({r: state_dict_sha256(getattr(self, r)) for r in ('aligner', 'student')})
