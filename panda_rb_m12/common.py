"""Campaign-local paths and identity; no mutation of legacy sources or outputs."""
from pathlib import Path
from fh12.common import atomic_json, object_sha, read_json, sha256, utcnow
from panda_rb.common import locked, read, immutable_json, append_event, verify_files, docker_required
from panda_rb_m12.plan import ROOT, CAMPAIGN_ID, SERVERS, campaign_dir, run_dir


def server_dir(server, root=ROOT):
    if server not in SERVERS:
        raise ValueError('Unknown M12 server')
    return campaign_dir(root) / 'control' / server


def source_identity(root=ROOT):
    from fh12.common import source_identity as inherited
    root = Path(root)
    result = inherited(root)
    files = dict(result['files'])
    for folder in ('panda_rb_m12', 'fh20r1', 'model', 'pa', 'tools/metrics', 'reporting_extra'):
        for path in sorted((root / folder).rglob('*.py')):
            if not path.name.startswith('test_') and path.name not in ('upload.py', 'worker.py'):
                files[str(path.relative_to(root))] = sha256(path)
    for name in ('__init__.py', 'common.py', 'bindings.py', 'plan.py', 'weights.py', 'stress.py'):
        path = root / 'panda_rb' / name
        files[str(path.relative_to(root))] = sha256(path)
    for pattern in ('rb_m12_*.py',):
        for path in sorted((root / 'tools').glob(pattern)):
            files[str(path.relative_to(root))] = sha256(path)
        for path in sorted((root / 'reporting_bridge').glob(pattern)):
            files[str(path.relative_to(root))] = sha256(path)
    result.update(files=files, content_sha256=object_sha(files), campaign_id=CAMPAIGN_ID)
    release = root / 'runtime_release.json'
    if release.exists():
        frozen = read_json(release)
        verify_files(frozen['files'], root)
        result['git_release'] = frozen['git_origin']
        result['frozen_release_sha256'] = frozen['release_sha256']
    return result
