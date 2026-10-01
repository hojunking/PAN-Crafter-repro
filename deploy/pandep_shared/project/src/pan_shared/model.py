"""Sensor stems/heads around exactly one mode-free PAN-Crafter paper trunk."""
import math
import torch
from torch import nn
from .blocks import SharedTrunk, ChannelLayerNorm, ResBlock
from .frontend import SENSOR_PROFILES, plh_inputs
from .common import seed_for


def canonical_initialization(model, seed):
    """Initialization is keyed by canonical tensor name AND shape, never case."""
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            module_name, field = name.rsplit('.', 1)
            module = modules[module_name]
            key = name.replace('.', '/')
            generator = torch.Generator(device='cpu').manual_seed(seed_for(seed, 'model-init', key, tuple(parameter.shape)))
            if isinstance(module, nn.LayerNorm):
                parameter.fill_(1.0 if field == 'weight' else 0.0)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if field == 'weight':
                    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    nn.init.uniform_(parameter, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), generator=generator)
            else:
                raise ValueError('Unregistered initializable tensor: ' + name)
        for module in model.modules():
            if isinstance(module, ResBlock):
                module.out_layers[3].weight.zero_()
                module.out_layers[3].bias.zero_()
        for stem in model.stems.values():
            stem.weight[:, 1:3].zero_()
        for head in model.heads.values():
            head.weight.zero_()
            head.bias.zero_()


class SharedPLHUNet(nn.Module):
    def __init__(self, width=104, depth=(1, 2, 2), seed=271001, *, sensors=('WV3', 'GF2', 'QB')):
        super().__init__()
        if not sensors or len(set(sensors)) != len(sensors) or set(sensors) - set(SENSOR_PROFILES):
            raise ValueError('Invalid sensor set')
        self.width, self.depth, self.seed, self.sensors = int(width), tuple(depth), int(seed), tuple(sensors)
        if self.width <= 0:
            raise ValueError('Positive width required')
        # Constructor defaults must not consume the caller's model/data RNG.
        with torch.random.fork_rng(devices=[]):
            self.stems = nn.ModuleDict({s: nn.Conv2d(3 + SENSOR_PROFILES[s]['bands'], width, 3, padding=1) for s in sensors})
            self.trunk = SharedTrunk(width, depth)
            self.heads = nn.ModuleDict({s: nn.Conv2d(width, SENSOR_PROFILES[s]['bands'], 3, padding=1) for s in sensors})
            canonical_initialization(self, seed)

    def forward(self, sensor_id, pan, ms, lpan):
        if sensor_id not in self.sensors:
            raise ValueError('Inactive/unregistered sensor: ' + str(sensor_id))
        features, base = plh_inputs(sensor_id, pan, ms, lpan)
        residual = self.heads[sensor_id](self.trunk(self.stems[sensor_id](features)))
        return base + residual

    def architecture_manifest(self):
        trunk = sum(p.numel() for p in self.trunk.parameters())
        stems = {s: sum(p.numel() for p in m.parameters()) for s, m in self.stems.items()}
        heads = {s: sum(p.numel() for p in m.parameters()) for s, m in self.heads.items()}
        return dict(schema='PANDEP_ARCHITECTURE_v1', width=self.width, depth=list(self.depth),
                    sensors=list(self.sensors), total_parameters=sum(p.numel() for p in self.parameters()),
                    shared_parameters=trunk, stem_parameters=stems, head_parameters=heads,
                    active_parameters={s: trunk + stems[s] + heads[s] for s in self.sensors},
                    input_order=['PAN', 'LPAN_UP', 'PAN_MINUS_LPAN_UP', 'MS_UP'],
                    initializer='canonical_tensor_name_shape_seed_v1', seed=self.seed,
                    aligner=False, warp=False, attention=False, mode_modulation=False,
                    output='MS_UP+residual_once', norm='channel_LayerNorm_eps1e-5')


class SingleSensorPLHUNet(SharedPLHUNet):
    def __init__(self, sensor_id, width=104, depth=(1, 2, 2), seed=271001):
        super().__init__(width, depth, seed, sensors=(sensor_id,))


def build_model(run):
    kwargs = dict(width=run['width'], depth=run['depth'], seed=run['seed'])
    if run.get('mode', run.get('runmode')) == 'SHARED':
        return SharedPLHUNet(**kwargs)
    return SingleSensorPLHUNet(run['sensors'][0], **kwargs)


def reference_parameter_mapping(sensor_id):
    """Explicit original PANCrafterPaper key -> new path key mapping rule."""
    return {'input.': f'stems.{sensor_id}.', 'output.0.': 'trunk.output_norm.',
            'output.2.': f'heads.{sensor_id}.', '*trunk_blocks*': 'trunk.<original_key>'}


def to_reference_state(model, sensor_id):
    result = {}
    for name, tensor in model.state_dict().items():
        if name.startswith('trunk.output_norm.'):
            key = 'output.0.' + name[len('trunk.output_norm.'):]
        elif name.startswith('trunk.'):
            key = name[len('trunk.'):]
        elif name.startswith('stems.' + sensor_id + '.'):
            key = 'input.' + name[len('stems.' + sensor_id + '.'):]
        elif name.startswith('heads.' + sensor_id + '.'):
            key = 'output.2.' + name[len('heads.' + sensor_id + '.'):]
        else:
            continue
        result[key] = tensor.clone()
    return result
