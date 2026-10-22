"""Registered positive-scale exploration; historical campaign sources remain immutable."""
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = "PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1"
METHOD_REVISION = "PANDA-SCALE30-HARDONLY-TUNE-v1"
BUDGET = {
    "total_seconds": 30 * 3600, "closeout_seconds": 1800,
    "safety_factor": 1.12, "calibration_reserve_seconds": 600,
    "scope": "persistent_wall_time_including_import_setup_train_eval_calibration",
    "restart": "never_reset_deadline",
}

# Pure-data registry shared by training and dependency-light evidence readers.
SCALE_POLICIES = {"S010": 0.1, "S020": 0.2, "S030": 0.3, "S040": 0.4, "S050": 0.5}
CASES = {
    "S010BASE": dict(policy_id="S010", scale=0.1, alpha=0.25, gamma=0.5, lr_S=1e-4),
    "S020BASE": dict(policy_id="S020", scale=0.2, alpha=0.25, gamma=0.5, lr_S=1e-4),
    "S030BASE": dict(policy_id="S030", scale=0.3, alpha=0.25, gamma=0.5, lr_S=1e-4),
    "S040BASE": dict(policy_id="S040", scale=0.4, alpha=0.25, gamma=0.5, lr_S=1e-4),
    "S050BASE": dict(policy_id="S050", scale=0.5, alpha=0.25, gamma=0.5, lr_S=1e-4),
    "S020LR075": dict(policy_id="S020", scale=0.2, alpha=0.25, gamma=0.5, lr_S=7.5e-5),
    "S020LR150": dict(policy_id="S020", scale=0.2, alpha=0.25, gamma=0.5, lr_S=1.5e-4),
    "S020H050": dict(policy_id="S020", scale=0.2, alpha=0.125, gamma=0.25, lr_S=1e-4),
    "S020H200": dict(policy_id="S020", scale=0.2, alpha=0.5, gamma=1.0, lr_S=1e-4),
    "S020LR150H050": dict(policy_id="S020", scale=0.2, alpha=0.125, gamma=0.25, lr_S=1.5e-4),
    "S010LR150H050": dict(policy_id="S010", scale=0.1, alpha=0.125, gamma=0.25, lr_S=1.5e-4),
}
