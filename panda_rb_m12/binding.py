"""F1 binding adapter. Reuse immutable original assets, never a local Teacher."""
from pathlib import Path
from panda_rb.bindings import load_binding, validate_binding, prepare_binding
from panda_rb_m12.common import ROOT, immutable_json, read_json, sha256
from panda_rb_m12.plan import binding_path


def bind_existing(root=ROOT, source=None, destination=None):
    root = Path(root)
    source = Path(source or root / 'work_dir/_panda_rb/20260928/B01/common/bindings.json')
    destination = Path(destination or binding_path(root))
    # Existing B01's validated portable mapping is copied, not rewritten.
    _, value, _, _ = validate_binding(source)
    immutable_json(destination, value)
    immutable_json(destination.with_name('binding_origin.json'), dict(
        source_B01_binding_path=str(source.resolve()), source_B01_binding_sha256=sha256(source),
        common_sha256=value['common_sha256'], e_bar_policy='reuse exact original B01 cache; never recompute'))
    return destination
