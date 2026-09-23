"""Storage pause only; no silent cleanup, batch reduction, or deadline."""
import shutil
from pathlib import Path


def ensure_space(path, required_bytes=8*1024**3):
    from g23sens.common import RuntimePaused
    parent = Path(path)
    while not parent.exists(): parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required_bytes:
        raise RuntimePaused(f'STORAGE_PAUSE: need {required_bytes} free bytes; available {free}')
    return free
