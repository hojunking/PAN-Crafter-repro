import importlib
import json
import torch
from .common import source_namespace


def build(cfg, checkpoint, step):
    package = source_namespace(cfg)
    method = importlib.import_module(package + '.method')
    specification = importlib.import_module(package + '.spec')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if json.loads(json.dumps(state['config'])) != cfg or state['trainer']['updates'] != step:
        raise ValueError('Serialized config or update differs from authenticated metadata')
    spec = specification.MatureSpec(**cfg['stage2_spec'])
    if package == 'panda_sensor_focus':
        from panda_sensor_focus.aligner import aligner_class
        aligner = aligner_class(cfg['dataset'])(spec.aligner_outputs)
    else:
        aligner = method.BandPANAligner(spec.aligner_outputs)
    student = method.build_reconstructor(spec.reconstruction_spec, seed=spec.seed, role='S',
        sampling_variant=spec.sampling_variant, input_variant=spec.input_variant)
    for role, module in [('aligner', aligner), ('student', student)]:
        module.load_state_dict({key[len(role)+1:]: value for key, value in state['trainer']['stage'].items()
                               if key.startswith(role + '.')}, strict=True)
    model = method.MatureInference(aligner, student, spec, completed_update=step)
    return model.eval().requires_grad_(False)
