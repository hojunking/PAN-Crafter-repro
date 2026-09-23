"""The original G23 P0 graph and constructor-order initialization, isolated.

This intentionally does NOT use the later FH12 named-tensor initializer or the
PLH frontend. The Aligner views an internal margin of four; the U-Net and the
reported prediction still cover the entire native image.
"""
from __future__ import annotations

from pathlib import Path

import torch

from fh12.model import state_hash, tensor_hash
from model.pancrafter_paper import PANCrafterPaper
from pa.aligner import PANGlobalAligner
from pa.model import PAModel


INIT_POLICY = 'G23_LEGACY_CONSTRUCTOR_ORDER_T0_CLONE_v1'


def validate_architecture(cfg):
    args = cfg['model_args']
    if (cfg.get('model') != 'model.pancrafter_paper.PANCrafterPaper'
            or args.get('hidden_size') != 104 or list(args.get('depth', [])) != [1, 2, 1]
            or args.get('in_mode') != 'paper' or args.get('norm') != 'ln'
            or args.get('attn_locations') != [] or args.get('mode_modulation') is not False
            or args.get('out_channels') != 8 or cfg.get('num_bands') != 8
            or float(cfg.get('max_pixel', 0)) != 2047.0
            or cfg.get('mars') != 'ms' or cfg.get('res') is not True):
        raise ValueError('G23 SENS requires native WV3 P0 W104D121 LN, no attention/MARs')
    if int((cfg['kdv'].get('donor') or {}).get('view_margin_hr', -1)) != 4:
        raise ValueError('T0 Aligner margin must remain four pixels')


def fresh_backbone(cfg):
    """Return the legacy fresh U plus the exact post-constructor CPU RNG state.

    main.py seeds Torch, constructs its PanFeeders (which consume no Torch RNG),
    then constructs U normally. A/donor/Teacher constructors are forked there.
    Keeping that order matters: named-tensor reinitialization is not equivalent.
    """
    validate_architecture(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(cfg['seed']))
        backbone = PANCrafterPaper(**cfg['model_args'])
        after = torch.get_rng_state().clone()
    return backbone, after


def initial_snapshot(cfg, teacher):
    backbone, after = fresh_backbone(cfg)
    if teacher.aligner is None or int(teacher.aligner_margin) != 4:
        raise ValueError('The frozen T0 must provide its margin-four Aligner')
    u = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    a = {k: v.detach().cpu().clone() for k, v in teacher.aligner.state_dict().items()}
    return dict(policy=INIT_POLICY, seed=int(cfg['seed']), U=u, A=a,
                post_constructor_torch_rng=after,
                hashes=dict(U=state_hash(u), A=state_hash(a),
                            post_constructor_rng=tensor_hash(after)))


def build_model(cfg, snapshot):
    """Independently own every U/A tensor; no state from the previous arm."""
    backbone, after = fresh_backbone(cfg)
    if (snapshot.get('policy') != INIT_POLICY or snapshot.get('seed') != int(cfg['seed'])
            or snapshot['hashes']['U'] != state_hash(backbone.state_dict())
            or snapshot['hashes']['U'] != state_hash(snapshot['U'])
            or snapshot['hashes']['A'] != state_hash(snapshot['A'])
            or not torch.equal(after, snapshot['post_constructor_torch_rng'])
            or snapshot['hashes']['post_constructor_rng'] != tensor_hash(after)):
        raise ValueError('Initial U/A snapshot does not match this fresh legacy seed')
    backbone.load_state_dict(snapshot['U'], strict=True)
    with torch.random.fork_rng(devices=[]):
        aligner = PANGlobalAligner(8)
    aligner.load_state_dict(snapshot['A'], strict=True)
    model = PAModel(backbone, aligner, aligner_margin=4, sampler=True)
    model.requires_grad_(True)
    return model


def load_model(cfg, checkpoint, device='cpu'):
    """Strictly load a native prediction model from an immutable candidate."""
    from safetensors.torch import load_file
    from g23sens.common import read_json, sha256
    validate_architecture(cfg)
    folder = Path(checkpoint)
    identity = read_json(folder / 'identity.json')
    weights = folder / 'model.safetensors'
    if identity.get('model_sha256') != sha256(weights):
        raise ValueError('Candidate model file checksum mismatch')
    backbone, _ = fresh_backbone(cfg)
    with torch.random.fork_rng(devices=[]):
        model = PAModel(backbone, PANGlobalAligner(8), aligner_margin=4, sampler=True)
    state = load_file(str(weights), device='cpu')
    if identity.get('state_hash') != state_hash(state):
        raise ValueError('Candidate tensor checksum mismatch')
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity
