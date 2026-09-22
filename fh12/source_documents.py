"""Read a pinned research document after a user-owned archive move.

Only the original path or the same relative path beneath research_log/past is
accepted. An existing but changed original is never bypassed by an archive.
"""
import hashlib
from pathlib import Path

def pinned_document(root,name,expected_sha256):
    root=Path(root);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='research_log':
        raise ValueError('Expected repository research document')
    original=root/relative
    path=original if original.exists() else root/'research_log/past'/Path(*relative.parts[1:])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Pinned research document differs: '+str(path))
    return path
