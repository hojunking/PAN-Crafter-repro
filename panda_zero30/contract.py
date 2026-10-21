"""New exploratory campaign; historical Mature30 sources remain immutable."""
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = "PANDA_S13_ZERO_ROTCONS_HARDONLY_30H_20261021_v1"
METHOD_REVISION = "PANDA-ZERO30-HARDONLY-ROTCONS-v1"
BUDGET = {
    "total_seconds": 30 * 3600, "closeout_seconds": 1800,
    "safety_factor": 1.12, "calibration_reserve_seconds": 600,
    "scope": "persistent_wall_time_including_import_setup_train_eval_calibration",
    "restart": "never_reset_deadline",
}
