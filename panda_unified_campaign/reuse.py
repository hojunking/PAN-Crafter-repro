"""Both stages train afresh under Lanczos4; historical weights cannot substitute."""
def load_reused_records(cfg,directory):
    raise ValueError("Unified RGB/RGBN experiments require fresh Stage1/Stage2; no historical donor reuse")
