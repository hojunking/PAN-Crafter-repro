"""Verified local s3/s5 result-package intake. Never fetches or executes code."""
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile
import json
from panda_rb_m12.common import ROOT, atomic_json, sha256, locked, object_sha
from panda_rb_m12.plan import CAMPAIGN, campaign_dir, training_runs, run_dir


def import_package(path, expected_sha256, root=ROOT, activate=False):
    if not activate:
        raise PermissionError('Result import requires --activate; package/report remain read-only')
    root = Path(root).resolve(); path = Path(path).resolve()
    if len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise ValueError('Result ZIP differs from the supplied sender receipt SHA256')
    with locked(campaign_dir(root) / 'control/import.lock'), zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [i.filename for i in infos]
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Duplicate ZIP members or missing manifest')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema') != 'PANDA_M12_RETURN_v1' or manifest.get('campaign_id') != CAMPAIGN:
            raise ValueError('Not an M12 measured-evidence package')
        report = manifest['report']; servers = report['servers']
        if not servers or not set(servers) <= {'s3', 's5'} or report.get('completed_students', 0) < 1:
            raise ValueError('s1 intake accepts measured s3/s5 packages only')
        files = manifest['files']
        if set(names) != {'manifest.json', *files}:
            raise ValueError('Unlisted/missing package members')
        roots = [run_dir(r['run_id'], root) for r in training_runs(root=root) if r['server'] in servers]
        extra_roots = [campaign_dir(root) / base / server for server in servers for base in ('common/response', 'control')]
        added = []; total = 0
        for info in infos:
            if info.filename == 'manifest.json':
                continue
            relative = Path(info.filename); target = root / relative
            if (relative.is_absolute() or '..' in relative.parts or stat.S_ISLNK(info.external_attr >> 16)
                    or not target.resolve().is_relative_to(root)):
                raise ValueError('Unsafe result package member')
            owners = [base for base in roots + extra_roots if target.is_relative_to(base)]
            if len(owners) != 1 or target.resolve() != target:
                raise ValueError('Package member escapes registered remote run or traverses a symlink')
            sub = target.relative_to(owners[0])
            extra = owners[0] in extra_roots
            # This is the small consumed sample/view COUNT tensor, not model
            # weights or a resume state. Intake copies bytes and never unpickles.
            count_tensor = sub == Path('diagnostics/consumed_sample_view_counts.pt')
            if extra and sub.suffix not in ('.json', '.jsonl', '.csv', '.md', '.txt', '.npy', '.npz'):
                raise ValueError('Unsupported diagnostic artifact')
            if not extra and (sub.parts[0] not in ('meta', 'native', 'stress', 'diagnostics', 'checkpoints',
                                     'init_manifest.json', 'stream_manifest.json', 'bindings.json', 'case.json')
                    or (sub.suffix not in ('.json', '.jsonl', '.csv', '.yaml', '.npy') and not count_tensor)
                    or (sub.parts[0] == 'checkpoints' and sub.name not in ('selection_manifest.json', 'identity.json'))):
                raise ValueError('Result intake excludes executable files, training data, weights and credentials')
            h = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    h.update(chunk)
            if h.hexdigest() != files[info.filename]:
                raise ValueError('Result member hash mismatch')
            if target.exists():
                if sha256(target) != files[info.filename]:
                    raise ValueError('Conflicting existing evidence; will not overwrite: ' + info.filename)
            else:
                total += info.file_size; added.append(info.filename)
        existing = campaign_dir(root)
        while not existing.exists():
            existing = existing.parent
        if shutil.disk_usage(existing).free < total + 1024**3:
            raise OSError('Insufficient space for verified result intake')
        for name in added:
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix='.m12-import-', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as dest, archive.open(name) as source:
                    shutil.copyfileobj(source, dest); dest.flush(); os.fsync(dest.fileno())
                if sha256(temp) != files[name]:
                    raise ValueError('Result changed during intake')
                os.link(temp, target)
            finally:
                os.unlink(temp)
        receipt = dict(schema='PANDA_M12_IMPORT_v1', package_sha256=expected_sha256,
                       manifest_sha256=object_sha(manifest), servers=servers, added_files=len(added),
                       existing_files=len(files)-len(added), training_updates=0, sheets_uploaded=False)
        atomic_json(campaign_dir(root) / 'reporting/imports' / (expected_sha256 + '.json'), receipt)
        return receipt
