"""Scientific identity; Stage1 geometry selection is separate from Stage2."""
from panda_band30.contract import file_identity, HQNRPolicy

CAMPAIGN_ID = "PANDA_S13_MATURE_ALIGN_SCHEDULE_30H_20261020_v1"
METHOD_REVISION = "PANDA-MATURE30-v1"
POLICY = HQNRPolicy()
STAGE1_SELECTION = {
    "rule": "VAL_RRE50_ASC_RRE90_ASC_VAL_ERGAS_ASC_EARLIER_UPDATE",
    "primary": "fixed_RR_validation_RRE50",
    "secondary": "fixed_RR_validation_RRE90",
    "tertiary": "fixed_RR_validation_ERGAS",
    "test_metrics": "REPORT_ONLY_NOT_USED_FOR_DONOR_SELECTION",
}
BUDGET = {
    "total_seconds": 30 * 3600, "closeout_seconds": 1800,
    "safety_factor": 1.2, "calibration_reserve_seconds": 600,
    "scope": "persistent_wall_time_including_selection_setup_train_eval_calibration",
    "restart": "never_reset_deadline",
}
