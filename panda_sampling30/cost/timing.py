"""Observed fresh-Student and unchanged source-Teacher training timing."""
from experiment_cost.timing import timing_for_config as original_timing


def timing_for_config(config, selected_step, root, history_roots=None, cost_root=None):
    if config.get("inherited_updates", 0) or config.get("source_binding"):
        raise ValueError("Sampling30 timing admits no inherited Student training")
    return original_timing(config, selected_step, root, history_roots, cost_root)
