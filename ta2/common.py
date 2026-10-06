"""Small local provenance/atomic-I/O helpers; no legacy controller side effects."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'PANDA_ALIGNER_TEACHER_LMS_GT_S135_20261006_v2'
BUNDLE = ROOT / 'research_log/PAN_Aligner_TeacherOnly_v2'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return json.load(handle)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = stream.name
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    if Path(path).exists():
        if load_json(path) != value:
            raise ValueError('Immutable TA2 identity differs: ' + str(path))
    else:
        atomic_json(path, value)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def source_manifest(repo=ROOT):
    repo = Path(repo).resolve()
    paths = set((repo / 'ta2').rglob('*.py')) | set((repo / 'ta2').glob('*.sh'))
    for module in ('model', 'pa', 'align', 'fh12'):
        paths.update((repo / module).glob('*.py'))
    paths.update((repo / 'deploy/pandep_shared/project/src/pan_shared/metrics').glob('*.py'))
    for relative in ('ablr2/sensor_sources.json', 'fh20r1/training.py', 'utils.py'):
        paths.add(repo / relative)
    paths.update((repo/'tools/metrics').glob('*.py'))
    paths.add(repo/'tools/eval_dlpan.py')
    paths.add(repo/'feeders/feeder.py')
    # Approved destination IDs are part of the immutable execution contract,
    # not an untracked sidecar that can silently reroute a resumed campaign.
    registry = repo/'ta2/detail_books.json'
    if registry.is_file():
        paths.add(registry)
    return {str(p.relative_to(repo)): file_sha(p) for p in sorted(paths)}


def owned(root, path):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError('TA2 output leaves its owned run directory')
    return path
