"""Resolve registered H5 symlink targets for read-only Docker file mounts.

Only stdlib: the host does not need the training Python environment. A common
PanCollection installation symlinks H5 files into a sibling CANConv checkout.
Mounting the repository alone does not make those absolute targets visible.
"""
import json
from pathlib import Path
import sys


def targets(root, server):
    root=Path(root).resolve(strict=True)
    sensor={'s1':'WV3','s2':'QB','s3':'GF2'}[server]
    sources=json.loads((root/'ablr2/sensor_sources.json').read_text())['sensors'][sensor]['splits']
    paths=[root/item['path'] for item in sources.values()]
    if sensor=='QB':
        paths.extend(root/item['path'] for item in json.loads(
            (root/'config/bglobal_rr/qb_original_lms.json').read_text())['splits'].values())
    resolved=sorted({p.resolve(strict=True) for p in paths})
    if any(not p.is_file() or any(c in str(p) for c in ('\n','\r',',')) for p in resolved):
        raise ValueError('Docker mounts require existing regular files and delimiter-safe paths')
    return resolved


if __name__=='__main__':
    for path in targets(Path(__file__).resolve().parents[1],sys.argv[1]):
        print(path)
