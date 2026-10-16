"""One immutable, local WV3 Direct/MASK2 A/T binding for every s3 trial."""
import json
from pathlib import Path

from donor24.plan import CAMPAIGN_ID as SOURCE_CAMPAIGN
from donor30.plan import LOSS_MASK_POLICY
from ta2.evaluation import file_sha
from . import plan


def local_candidates(work_dir, server='s3'):
    """Read only completed donor24 attempts; reconstruct paths locally.

    Container absolute paths in old JSON are NOT trusted as host paths. No
    downloaded checkpoints, no GF2/s1 donor, no partial/missing-weight metric.
    """
    root = Path(work_dir)/'donor24'/SOURCE_CAMPAIGN/server/'WV3'/'runs'
    found, rejected = [], []
    for config_path in sorted(root.glob('*/attempt_*/config.json')):
        directory = config_path.parent
        try:
            cfg = json.loads(config_path.read_text())
            status = json.loads((directory/'run_state.json').read_text())
            if status['status'] != 'COMPLETED':
                continue
            if (cfg['campaign_id'] != SOURCE_CAMPAIGN or cfg['server'] != server
                    or cfg['dataset'] != 'WV3' or cfg['phase'] != 'donor'
                    or cfg['loss_mask_policy'] != LOSS_MASK_POLICY):
                continue
            registry = json.loads((directory/'models'/'selection.json').read_text())
            for row in registry['saved']:
                if not all(row.get(k) is True for k in ('checkpoint_verified','metrics_complete',
                                                      'metrics_finite','same_state_verified')):
                    continue
                step = row['completed_update']
                plan.rank(row)  # physical finite metrics; SCC never ranks
                path = directory/'models'/f'step_{step:07d}.pt'
                evaluation = directory/f'evaluation_{step:07d}.json'
                if not path.is_file() or not evaluation.is_file():
                    rejected.append({'run_id':cfg['run_id'],'step':step,'reason':'missing_saved_artifact'})
                    continue
                evidence = json.loads(evaluation.read_text())
                identity = evidence.get('state_before',{})
                gate = evidence.get('qualification',{})
                if (any(evidence.get(k)!=identity for k in ('rr_state','fr_state','state_after'))
                        or any(identity.get(k)!=cfg[k] for k in ('run_id','attempt','seed','dataset'))
                        or identity.get('completed_update')!=step
                        or identity.get('model_state_sha256')!=row.get('state_hash')
                        or any(gate.get(k) is not True for k in ('metrics_complete','metrics_finite','same_state_verified'))
                        or evidence['fr']['metrics']['HQNR']!=row['HQNR']
                        or evidence['rr']['metrics']['ERGAS']!=row['ERGAS']
                        or any(len(evidence[s]['per_scene'])!=20 for s in ('rr','fr'))):
                    raise ValueError('Selector and stored full40 evaluation disagree')
                found.append({**row,'checkpoint_path':str(path.resolve()),
                    'evaluation_path':str(evaluation.resolve()),'source_config':cfg,
                    'source':'COMPLETED_LOCAL_DONOR24','source_server':server})
        except (KeyError,ValueError,OSError) as exc:
            rejected.append({'path':str(config_path),'reason':repr(exc)})
    return found, rejected


def verify_file(candidate):
    path = Path(candidate['checkpoint_path']).resolve(strict=True)
    sha = file_sha(path)
    expected = candidate.get('file_sha256') or candidate.get('checkpoint_sha256')
    if expected is None or sha != expected:
        raise ValueError('Selected donor file differs from its saved verification record')
    return sha


def bind(candidate, *, replay_verified):
    if replay_verified is not True:
        raise ValueError('Donor must be replayed on the full RR20/FR20 before binding')
    sha = verify_file(candidate)
    return {**candidate,'checkpoint_sha256':sha,
        'same_checkpoint_A_T':True,'reloaded_full40_verified':True,
        'selection_policy':plan.POLICY.manifest(),
        'exploratory':candidate['HQNR'] < plan.POLICY.hqnr_target,
        'selection_scope':'ONE_FIXED_LOCAL_S3_DONOR_FOR_ALL_STAGE2_CASES',
        'cross_server_absolute_metric_comparison':False}
