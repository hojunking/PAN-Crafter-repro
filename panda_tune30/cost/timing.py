"""Measured training suffix, excluding the preserved continuation prefix."""
from panda_schedule30.cost.timing import timing_for_config as original_timing
from panda_tune30.reporting import validate_raw_model


def timing_for_config(config, selected_step, root, history_roots=None, cost_root=None):
    validate_raw_model(config)
    return original_timing(config, selected_step, root, history_roots, cost_root)
