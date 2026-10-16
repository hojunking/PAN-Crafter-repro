"""Atomic, no-overwrite local Stage 2 state; no legacy checkpoint fallback."""
from pathlib import Path
import os
import tempfile

import torch

from ta2.evaluation import file_sha
from .training import TRAINER_SCHEMA

SCHEMA = 'PANDA_S2_CONDQ_CHECKPOINT_V1'


def _validate_attachment(evaluation, trainer, weights, update):
    from .evaluation import validate_result
    from panda_method.calibration import state_dict_sha256, canonical_sha256
    validate_result(evaluation)
    identity = evaluation['state_after']
    def role(prefix):
        return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    if (identity['completed_update'] != update
            or identity['student_state_sha256'] != state_dict_sha256(role('student.'))
            or identity['aligner_state_sha256'] != state_dict_sha256(role('aligner.'))
            or identity['training_contract_sha256'] != canonical_sha256(trainer.contract())
            or trainer.evaluator_pin != evaluation['eval_protocol']
            or evaluation['checkpoint_selection'] != trainer.selection_policy.manifest()
            or evaluation['sensor'] != trainer.stage.spec.sensor
            or evaluation['pipeline'] != trainer.stage.pipeline_manifest()):
        raise ValueError('Cannot attach another model/update evaluation to saved weights')


def save_checkpoint(path, trainer, *, stream_state=None, evaluation=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing to overwrite Stage 2 checkpoint: ' + str(path))
    if evaluation is not None:
        _validate_attachment(evaluation, trainer, trainer.stage.state_dict(), trainer.updates)
    if trainer.stage.stats is not None:
        trainer.stage.stats.assert_integrity(full_hash=True)
    payload = {'schema': SCHEMA, 'trainer': trainer.state_dict(stream_state=stream_state),
               'evaluation': evaluation}
    fd, tmp = tempfile.mkstemp(prefix='.condq-checkpoint-', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        # Hard link publication is atomic and exclusive; existing files are
        # never replaced even if another process wins the final-name race.
        os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {'path': str(path.resolve()), 'sha256': file_sha(path), 'update': trainer.updates}


def load_checkpoint(path, expected_sha256, trainer):
    from panda_method.data import require_sha256
    require_sha256(expected_sha256, 'Stage 2 checkpoint file')
    path = Path(path).resolve(strict=True)
    if file_sha(path) != expected_sha256:
        raise ValueError('Stage 2 checkpoint file SHA mismatch')
    # Only trusted, locally saved SHA-pinned checkpoints, not downloaded pickle.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if file_sha(path) != expected_sha256:
        raise ValueError('Stage 2 checkpoint changed while loading')
    if payload.get('schema') != SCHEMA or payload.get('trainer', {}).get('schema') != TRAINER_SCHEMA:
        raise ValueError('New conditional-q Stage 2 checkpoint required; legacy donor30 E/EQ/MASK2 rejected')
    if trainer.stage.stats is not None:
        trainer.stage.stats.assert_integrity(full_hash=True)
    if payload.get('evaluation') is not None:
        _validate_attachment(payload['evaluation'], trainer, payload['trainer']['stage'],
                             payload['trainer']['updates'])
    stream_state = trainer.load_state_dict(payload['trainer'])
    return {'stream_state': stream_state, 'evaluation': payload.get('evaluation')}
