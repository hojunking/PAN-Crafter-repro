"""Download one public native PAN/MS pair for provenance audit, never training.

Fixed public URLs only, no credentials, no archive extraction, no overwrite.
The S3 ETag is retained as metadata; SHA256 is computed locally, not confused
with a remotely published checksum. Re-running verifies existing downloads.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/external_audit/spacenet_paris_20261013'
BASE = 'https://spacenet-dataset.s3.amazonaws.com/'
PREFIX = 'AOIs/AOI_3_Paris/'
KEYS = [
    PREFIX + 'MS/16FEB29111913-M2AS_R08C3-055649178040_01_P001.TIF',
    PREFIX + 'PAN/16FEB29111913-P2AS_R08C3-055649178040_01_P001.TIF',
    PREFIX + 'metadata/16FEB29111913-M2AS-055649178040_01_P001.XML',
    PREFIX + 'metadata/16FEB29111913-P2AS-055649178040_01_P001.XML',
    PREFIX + 'metadata/AOI_3_Paris_MUL.vrt',
    PREFIX + 'metadata/AOI_3_Paris_PAN.vrt',
]


def fetch(item):
    key = item['key']
    dest = OUT / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    if dest.exists():
        if dest.stat().st_size != item['size']:
            raise ValueError('Existing incomplete/wrong-size file: ' + str(dest))
        with dest.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
    else:
        partial = dest.with_suffix(dest.suffix + '.partial')
        start = time.monotonic()
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['size']:
            raise ValueError('Oversized existing partial download')
        parts = dest.parent / (dest.name + '.ranges')
        parts.mkdir(exist_ok=True)

        def fetch_range(bounds):
            first, last = bounds
            part = parts / f'{first}-{last}'
            if part.exists():
                if part.stat().st_size != last-first+1:
                    raise ValueError('Invalid existing range')
                return part
            # A unique temporary file avoids trusting an interrupted range.
            import tempfile
            import os
            with requests.get(BASE+key, headers={
                    'Range': f'bytes={first}-{last}', 'If-Match': item['etag']},
                    stream=True, timeout=(15, 60)) as r:
                if (r.status_code != 206 or r.headers.get('ETag') != item['etag']
                        or r.headers.get('Content-Range') != f'bytes {first}-{last}/{item["size"]}'):
                    raise ValueError('Range response/object identity mismatch')
                with tempfile.NamedTemporaryFile(dir=parts, prefix='pending-', delete=False) as f:
                    temp = Path(f.name)
                    count = 0
                    for chunk in r.iter_content(1024*1024):
                        count += len(chunk)
                        if count > last-first+1:
                            raise ValueError('Oversized range response')
                        f.write(chunk)
                if count != last-first+1:
                    raise ValueError('Truncated range response')
                os.link(temp, part)
                temp.unlink()
            return part

        bounds = [(i, min(i+8*1024*1024, item['size'])-1)
                  for i in range(offset, item['size'], 8*1024*1024)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            # Commit contiguous completed ranges only; partial remains resumable.
            with partial.open('ab') as f:
                for part in pool.map(fetch_range, bounds):
                    with part.open('rb') as src:
                        for chunk in iter(lambda: src.read(1024*1024), b''):
                            f.write(chunk)
                    f.flush()
                    print('progress', dest.name, f.tell(), '/', item['size'], flush=True)
        count = partial.stat().st_size
        if count != item['size']:
            raise ValueError('Truncated download')
        with partial.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        # No overwrite of an existing final destination, even under concurrency.
        import os
        os.link(partial, dest)
        partial.unlink()
        print('downloaded', key, count, 'bytes', round(time.monotonic()-start, 2), 's', flush=True)
    return dict(item, local_path=str(dest.relative_to(ROOT)), sha256=h.hexdigest())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    inventory = []
    for section in ('MS/', 'PAN/', 'metadata/'):
        params = {'list-type': '2', 'prefix': PREFIX+section, 'max-keys': 1000}
        while True:
            r = requests.get(BASE, params=params, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for c in root.findall('s:Contents', ns):
                inventory.append({
                    'key': c.findtext('s:Key', namespaces=ns),
                    'size': int(c.findtext('s:Size', namespaces=ns)),
                    'etag': c.findtext('s:ETag', namespaces=ns),
                    'last_modified': c.findtext('s:LastModified', namespaces=ns),
                })
            if root.findtext('s:IsTruncated', namespaces=ns) == 'false':
                break
            params['continuation-token'] = root.findtext('s:NextContinuationToken', namespaces=ns)
    by_key = {x['key']: x for x in inventory}
    selected = [by_key[k] for k in KEYS]
    if sum(x['size'] for x in selected) > 850_000_000:
        raise ValueError('Audit download exceeds fixed 850 MB ceiling')
    manifest = OUT / 'download_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        checked = [fetch(x) for x in selected]
        if checked != old['downloaded']:
            raise ValueError('Existing manifest/file identity mismatch')
        print('Existing manifest verified', manifest)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        downloaded = list(pool.map(fetch, selected))
    result = {'purpose': 'READ_ONLY_PROVENANCE_AUDIT_NOT_ADMITTED_FOR_TRAINING',
              'catalog_id': '1040010018805C00',
              'source_page': 'https://spacenet.ai/paris/',
              'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'bucket': BASE, 'inventory': inventory, 'downloaded': downloaded,
              'bytes_downloaded': sum(x['size'] for x in downloaded)}
    with manifest.open('x') as f:
        json.dump(result, f, indent=2)
    print('manifest', manifest, flush=True)


if __name__ == '__main__':
    main()
