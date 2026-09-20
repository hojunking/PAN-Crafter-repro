"""Unmodified numerical model, with complete paired-initialization evidence."""
from qg40.model import build_model as _build_model, state_hash, sync_frontend
from g20.common import object_sha


def build_model(*args, **kwargs):
    model, manifest = _build_model(*args, **kwargs)
    manifest['hashes']['U'] = state_hash(model.backbone.state_dict())
    manifest['architecture_sha256'] = object_sha({
        'state_shapes': {key: list(value.shape) for key, value in model.state_dict().items()},
        'modules': {key: type(value).__module__ + '.' + type(value).__qualname__
                    for key, value in model.named_modules()},
        'frontend': 'native_MSbase_PAN_LP_sync_bicubic_signedHP_no_clamp_margin4',
        'layout': manifest['input_layout'] if 'input_layout' in manifest else kwargs.get('layout', args[0] if args else None),
    })
    return model, manifest
