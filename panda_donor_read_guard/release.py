"""Keep the IO overlay hash separate from frozen training-source pins."""
import hashlib
from pathlib import Path

def identity():
    folder=Path(__file__).resolve().parent
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.py'))}
    import json
    return {'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'files':files}
