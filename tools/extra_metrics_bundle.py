#!/usr/bin/env python
"""Package only the additive reporting overlay, never credentials or training code."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted((ROOT / 'reporting_extra').glob('*.py'))
    files += [ROOT / 'reporting_extra/README.md', ROOT / 'tools/extra_metrics.py',
              ROOT / 'tools/extra_metrics_start.sh', ROOT / 'tools/extra_metrics_bundle.py']
    entries = {str(p.relative_to(ROOT)): p.read_bytes() for p in files}
    manifest = {'schema': 'PAN_EXTRA_OVERLAY_v1', 'git_head_must_remain_unchanged': True,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}}
    entries['reporting_extra/overlay_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output = ROOT / 'work_dir/_extra_metrics/deployment/pan-extra-metrics-overlay.tar.gz'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name, data in entries.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(data))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'sha256': digest, 'files': list(entries)}, indent=2))


if __name__ == '__main__':
    main()
