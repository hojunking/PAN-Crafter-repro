"""Source fingerprints for the isolated implementation, not a campaign runner."""
from pathlib import Path

from ta2.evaluation import file_sha


def code_pin():
    from panda_stage2_condq.runtime import code_pin as inherited
    root = Path(__file__).resolve().parents[1]
    return {**inherited(), **{str(p.relative_to(root)): file_sha(p)
                             for p in (root / 's2dshift48').rglob('*.py')}}
