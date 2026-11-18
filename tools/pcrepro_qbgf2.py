#!/usr/bin/env python3
"""Finite s5 QB/GF2 paper reproduction; use plan/prepare before run."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcrepro.finite import main

if __name__=='__main__':
    try:
        result=main();print(json.dumps(result,indent=2,ensure_ascii=False,default=str,allow_nan=False))
        sys.exit(result.get('exit_code',0))
    except (InterruptedError,KeyboardInterrupt):sys.exit(75)
    except OSError as exc:print(str(exc),file=sys.stderr);sys.exit(74)
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
