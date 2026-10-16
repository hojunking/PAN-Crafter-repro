"""HQNR feasibility gate, then minimum ERGAS. SCC is monitoring only."""
from dataclasses import dataclass
import math


SELECTION_RULE = 'HQNR_GATE_ERGAS_ASC_BELOW_GATE_HQNR_DESC_NO_SCC_V1'


@dataclass(frozen=True)
class SelectionPolicy:
    hqnr_target: float

    def __post_init__(self):
        if (type(self.hqnr_target) not in (int, float)
                or not math.isfinite(self.hqnr_target) or not 0 < self.hqnr_target <= 1):
            raise ValueError('Explicit finite HQNR target in (0,1] required')

    def manifest(self):
        return {'rule': SELECTION_RULE, 'hqnr_target': self.hqnr_target,
                'qualification': 'scene_mean_HQNR_GE_target_without_rounding',
                'qualified': 'ERGAS_ASC_then_HQNR_DESC_then_earlier_update',
                'fallback': 'HQNR_DESC_then_ERGAS_ASC_then_earlier_update_EXPLORATORY',
                'SCC': 'monitor_only_never_rank'}

    @classmethod
    def from_manifest(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Explicit checkpoint selection manifest required')
        policy = cls(value.get('hqnr_target'))
        if value != policy.manifest():
            raise ValueError('Checkpoint selection rule/target metadata mismatch')
        return policy

    def rank(self, hqnr, ergas, update):
        """Maximize this tuple. A qualified candidate ALWAYS beats a fallback."""
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (hqnr, ergas))
                or not 0 <= hqnr <= 1 or ergas < 0
                or type(update) is not int or update < 0):
            raise ValueError('Finite physical metrics and nonnegative update required')
        if hqnr >= self.hqnr_target:
            return (1, -ergas, hqnr, -update)
        return (0, hqnr, -ergas, -update)


def resolve_policy(sensor, hqnr_target=None):
    # Only the current WV3 goal is established by the user. Do not silently
    # transplant it to a different dataset or revive the old GF2 campaign goal.
    if hqnr_target is None:
        if sensor != 'WV3':
            raise ValueError('A non-WV3 HQNR target must be explicitly configured')
        hqnr_target = .959
    return SelectionPolicy(hqnr_target)
