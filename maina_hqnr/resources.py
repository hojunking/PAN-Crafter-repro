"""Local disk headroom: never delete candidates or change the recipe."""
from pathlib import Path
import shutil


def ensure_space(path, required_bytes=8 * 1024 ** 3):
    from maina_hqnr.common import RuntimePaused
    parent = Path(path)
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required_bytes:
        raise RuntimePaused(f'PAUSED_STORAGE: required={required_bytes}, available={free}')
    return free
