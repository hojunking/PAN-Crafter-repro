"""Exact train sample/view exposure accounting; never touches sampler RNG."""
import torch


def validate_counts(counts, count, updates, batch_size):
    if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int64
            or tuple(counts.shape) != (count, 4) or bool((counts < 0).any())
            or int(counts.sum()) != updates * batch_size):
        raise ValueError('Exact sample exposure differs from completed optimizer updates')
    return counts.detach().cpu().clone()


def record_batch(counts, metadata):
    ids = metadata[:, :2].detach().cpu().long()
    if bool(((ids[:, 0] < 0) | (ids[:, 0] >= len(counts)) | (ids[:, 1] < 0) | (ids[:, 1] > 3)).any()):
        raise ValueError('Invalid sample/view IDs for exposure accounting')
    counts.view(-1).index_add_(0, ids[:, 0] * 4 + ids[:, 1], torch.ones(len(ids), dtype=torch.int64))


def exposure_report(counts, batch_size, epoch, cursor):
    per_patch = counts.sum(1)
    return dict(scope='actual completed training batches only; no validation/test exposure',
        n_base=len(counts), batch_size=batch_size, drop_last=True,
        dropped_per_epoch=len(counts) % batch_size, current_epoch=epoch, batch_cursor=cursor,
        total_sample_presentations=int(counts.sum()), nominal_presentations_per_patch=float(counts.sum()) / len(counts),
        unique_base_seen=int((per_patch > 0).sum()), unseen_base=int((per_patch == 0).sum()),
        per_patch_quantile_levels=[0., .01, .5, .99, 1.],
        per_patch_quantiles=torch.quantile(per_patch.double(), torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist(),
        per_view_presentations=counts.sum(0).tolist(), full_counts_location='last/training_state.pt:exposure_counts')
