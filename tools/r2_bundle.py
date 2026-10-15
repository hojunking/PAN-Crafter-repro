#!/usr/bin/env python
"""Build a HEAD-preserving R2 overlay with derived, validated queue specifications."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from campaign_r2.plan import canonical_overlay, validate_definition, SOURCE_PLAN
    from fh20r1.plan import REFERENCE_ASSETS
    overlay, validation = canonical_overlay(ROOT), validate_definition(ROOT)
    paths = sorted((ROOT / 'campaign_r2').glob('*.py'))
    paths += [ROOT / 'campaign_r2/README.md', ROOT / SOURCE_PLAN,
              ROOT / 'tools/r2_runner.py', ROOT / 'tools/r2_start.sh', ROOT / 'tools/r2_bundle.py']
    entries = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    specs = {'priority_overlay.json': overlay, 'definition_validation.json': validation,
             'reference_assets.json': {'source': 'unchanged fh20r1.plan', 'new_references': False,
                                       'references': REFERENCE_ASSETS}}
    for name, value in specs.items():
        entries['campaign_r2/spec/' + name] = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['server', 'priority', 'block_id', 'tier', 'run_order', 'run_id', 'local_status'])
    for server, spec in overlay['servers'].items():
        for priority, block in enumerate(spec['blocks'], 1):
            for order, run in enumerate(block['run_ids'], 1):
                writer.writerow([server, priority, block['block_id'], block['tier'], order, run,
                                 'RECONCILE_LOCAL_DO_NOT_ASSUME_FRESH'])
    entries['campaign_r2/spec/queue_overlay.csv'] = out.getvalue().encode()
    manifest = dict(schema='FH20R1_R2_OVERLAY_PACKAGE_v1', keep_git_head=True,
                    files={name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()})
    entries['campaign_r2/spec/package_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output = ROOT / 'work_dir/_fh20r1/r2_deployment/pan-fh20r1-r2-overlay.tar.gz'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name, raw in entries.items():
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(raw), 0o644
            archive.addfile(item, io.BytesIO(raw))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps(dict(archive=str(output), sha256=digest, files=list(entries)), indent=2))


if __name__ == '__main__':
    main()
