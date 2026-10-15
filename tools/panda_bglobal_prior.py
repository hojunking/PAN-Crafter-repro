#!/usr/bin/env python3
"""Explicit train-only b_global preprocessing CLI; see --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher48.prepare_global_prior import main


if __name__ == "__main__":
    main()
