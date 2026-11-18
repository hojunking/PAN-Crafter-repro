#!/usr/bin/env python3
"""s5 operator entrypoint: start | status | pause | resume."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcrepro.s5_runtime import main

if __name__=='__main__':
    try:print(json.dumps(main(),indent=2,ensure_ascii=False,default=str,allow_nan=False))
    except (InterruptedError,KeyboardInterrupt):sys.exit(75)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
