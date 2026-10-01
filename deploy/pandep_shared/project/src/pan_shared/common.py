"""Small deterministic identities and campaign-owned atomic persistence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

CAMPAIGN = 'PANDEP_S2_SHARED_PLH_20261001_v4'
SENSORS = ('WV3', 'GF2', 'QB')
SOURCE_COMMIT = 'ea71b68be637a1d1f4d61bf89cb4e023cbf00385'

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def seed_for(master, *parts):
    return int(canonical_sha(['PANDEP_RNG_v1', int(master), *parts])[:16], 16) % (2**63-1)

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        with open(temporary, encoding='utf-8') as stream:
            if json.load(stream) != value:
                raise ValueError('Atomic JSON readback failed')
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
