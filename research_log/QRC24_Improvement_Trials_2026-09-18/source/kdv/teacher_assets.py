"""Teacher/donor asset 검증·strict load·freeze (plan §2.1, §3.1, §4.2, gate M02/M03).

- donor aligner: run 폴더(<run>/<tag>/model.safetensors 의 aligner.*) 또는 aligner 단독 .pt/.safetensors. strict 검사 (key·shape).
- KD Teacher: run 폴더의 config 로 같은 skeleton(PAModel) 을 만들고 strict=True 로 싣는다 — W96 backbone 을 W112 에 partial load 하는 일은 구조적으로 불가능하다.
- freeze: requires_grad False + eval. state hash 로 학습 중 불변 검사, optimizer parameter ID 교집합 검사."""
import hashlib
import json
import os

import torch
import yaml

from pa.aligner import PANGlobalAligner
from pa.model import PAModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def tensors_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def state_hash(module):
    return tensors_sha({k: v for k, v in module.state_dict().items()})


def _resolve(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.isdir(p):
        p = os.path.join(p, 'model.safetensors')
    if not os.path.exists(p):
        raise FileNotFoundError(f"checkpoint 없음: {p}")
    return p


def load_state(path):
    p = _resolve(path)
    if p.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(p), p
    obj = torch.load(p, map_location='cpu')
    return (obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj), p


def extract_aligner_sd(sd):
    a = {k[len('aligner.'):]: v for k, v in sd.items() if k.startswith('aligner.')}
    if a:
        return a
    with torch.random.fork_rng(devices=[]):
        ref = set(PANGlobalAligner(8).state_dict())
    if set(sd) == ref:
        return dict(sd)
    raise KeyError("aligner.* 키가 없고 aligner 단독 state_dict 도 아니다")


def load_donor_aligner(source, ms_bands, expected_sha256=None):
    """strict: 모든 key 존재·shape 일치. 반환 (aligner, manifest)."""
    sd, p = load_state(source)
    asd = extract_aligner_sd(sd)
    with torch.random.fork_rng(devices=[]):                 # 모듈 생성의 난수 초기화가 전역 RNG(DataLoader 순서)를 바꾸지 않게 (검토 지적 4)
        al = PANGlobalAligner(int(ms_bands))
    al.load_state_dict(asd, strict=True)                    # key/shape 불일치는 여기서 즉시 오류
    fsha = sha256_file(p)
    man = dict(source=source, resolved_file=p, file_sha256=fsha, aligner_tensors_sha256_16=tensors_sha(asd), n_params=sum(v.numel() for v in asd.values()),
               n_keys=len(asd), expected_sha256=expected_sha256, expected_match=(None if not expected_sha256 else fsha == expected_sha256))
    if expected_sha256 and fsha != expected_sha256:
        raise RuntimeError(f"donor sha256 불일치: {fsha} ≠ 기대 {expected_sha256} ({p})")
    mp = p.replace('.pt', '.json').replace('.safetensors', '.json')
    if os.path.exists(mp) and mp != p:
        try:
            man['source_manifest'] = json.load(open(mp))
        except Exception:
            pass
    return al, man


def skeleton_from_cfg(cfg, Model):
    """run config → (PAModel skeleton, info). trainer 종류별로 실제 학습 그래프와 같은 wrapper 를 만든다 (가중치는 싣지 않는다)."""
    bb = Model(**cfg['model_args']); nb = int(cfg.get('num_bands', 8)); tr = cfg.get('trainer', 'default')
    if tr == 'kdv':
        k = cfg.get('kdv') or {}; pol = k.get('aligner_policy')
        if pol == 'A-ID':
            return PAModel(bb, None, aligner_margin=0, sampler=False), dict(kind='kdv', policy=pol, margin=0, has_aligner=False)
        mg = int((k.get('donor') or {}).get('view_margin_hr', k.get('aligner_view_margin_hr', 0)) or 0)
        m = PAModel(bb, PANGlobalAligner(nb), aligner_margin=mg, sampler=True)
        if (k.get('geom_kd') or {}).get('mode') == 'G5':
            from kdv.alignment_kd import CovHead
            m.cov_head = CovHead()
        return m, dict(kind='kdv', policy=pol, margin=mg, has_aligner=True)
    if tr in ('pa', 'po'):
        from pa.offset import aligner_margin
        mg = aligner_margin(float((cfg.get('po') or {}).get('radius_hr', 1.0))) if tr == 'po' else 0
        return PAModel(bb, PANGlobalAligner(nb), aligner_margin=mg, sampler=True), dict(kind=tr, margin=mg, has_aligner=True)
    if tr == 'default':                                      # B0 계열: aligner 없음, sampling 없음
        return PAModel(bb, None, aligner_margin=0, sampler=False), dict(kind='b0', margin=0, has_aligner=False)
    raise NotImplementedError(f"trainer {tr} 의 checkpoint 는 KDV Teacher/pilot 으로 쓰지 않는다")


def load_run_model(run_dir, tag, Model, expected_sha256=None):
    """run 폴더의 <tag> checkpoint 를 그 run 의 config 대로 strict 하게 싣는다. 반환 (PAModel, manifest)."""
    rd = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    cfg_p = os.path.join(rd, 'meta', 'config.yaml')
    if not os.path.exists(cfg_p):
        raise FileNotFoundError(f"run config 없음: {cfg_p}")
    cfg = yaml.safe_load(open(cfg_p))
    with torch.random.fork_rng(devices=[]):
        m, info = skeleton_from_cfg(cfg, Model)
    sd, p = load_state(os.path.join(rd, tag))
    if info['kind'] == 'b0':
        m.backbone.load_state_dict(sd, strict=True)
    else:
        m.load_state_dict(sd, strict=True)                   # W96 ↔ W112 shape 불일치는 RuntimeError — partial load 없음
    fsha = sha256_file(p)
    if expected_sha256 and fsha != expected_sha256:
        raise RuntimeError(f"Teacher sha256 불일치: {fsha} ≠ 기대 {expected_sha256} ({p})")
    ma = cfg['model_args']
    meta_p = os.path.join(rd, f'{tag}_meta.json')
    man = dict(run=os.path.basename(rd.rstrip('/')), run_dir=rd, tag=tag, file=p, file_sha256=fsha, tensors_sha256_16=tensors_sha(sd),
               trainer=cfg.get('trainer', 'default'), width=ma.get('hidden_size'), depth=ma.get('depth'), in_mode=ma.get('in_mode'), norm=ma.get('norm'),
               backbone_params=sum(q.numel() for q in m.backbone.parameters()), aligner_params=(sum(q.numel() for q in m.aligner.parameters()) if m.aligner is not None else 0),
               aligner_view_margin=info['margin'], sampler=m.sampler, seed=cfg.get('seed'), kdv=(cfg.get('kdv') if cfg.get('trainer') == 'kdv' else None),
               tag_meta=(json.load(open(meta_p)) if os.path.exists(meta_p) else None), expected_sha256=expected_sha256)
    return m, man


def freeze(module):
    module.requires_grad_(False); module.eval()
    return module


def assert_param_disjoint(optimizer, *frozen_modules):
    """optimizer 에 frozen module 의 parameter 가 들어 있으면 오류 (gate M03)."""
    opt_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    for m in frozen_modules:
        if m is None:
            continue
        inter = [n for n, p in m.named_parameters() if id(p) in opt_ids]
        if inter:
            raise RuntimeError(f"optimizer 가 frozen parameter 를 포함한다: {inter[:5]}")
    return True
