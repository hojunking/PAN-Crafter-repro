"""Stdlib-only scientific and publication identity; no implicit launch."""
import math

CAMPAIGN_ID = 'PANDA_S3_WARP_STAGE12_30H_20261019_v1'
METHOD_REVISION = 'PANDA-WARP30-STAGE12-v1'
SELECTION = {'rule': 'HQNR_DESC_THEN_ERGAS_ASC_THEN_EARLIER_UPDATE_V1',
             'primary': 'official_FR20_unmasked_HQNR', 'secondary': 'official_RR20_ERGAS',
             'SCC': 'monitor_only', 'thresholds': 'report_only_not_checkpoint_ranking'}
BUDGET = {'total_seconds': 30*3600, 'closeout_seconds': 1800,
          'safety_factor': 1.2, 'calibration_reserve_seconds': 600,
          'scope': 'persistent_wall_time_including_setup_train_eval_profile_calibration',
          'restart': 'never_reset_deadline'}


class HQNRPolicy:
    def manifest(self):
        return dict(SELECTION)

    def rank(self, hqnr, ergas, update):
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (hqnr, ergas))
                or not 0 <= hqnr <= 1 or ergas < 0 or type(update) is not int or update < 0):
            raise ValueError('Finite original-reference HQNR/ERGAS and nonnegative update required')
        return (hqnr, -ergas, -update)


POLICY = HQNRPolicy()
