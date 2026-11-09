"""Main-matched sensitivity; share the frozen hyper engine under this namespace.

Relative imports of inherited modules bind to this package's registered plan,
release and reuse policy. The original campaign and its source files are never
modified. The release manifest pins both local overrides and inherited files.
"""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "panda_wv3_hyper"))
