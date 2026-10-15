#!/usr/bin/env python
"""EQREC4-S1-v1 CLI (research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md).

    python tools/eqrec4.py g00|d10|d20|d30|d40|d50|k10|k20|report [--profile] [--pairs A,B] [--pair A]
    python tools/eqrec4.py all            # G00 → … → report (tools/eqrec4_run.sh 가 detached 로 부른다)
--profile: 첫 128 patch·1 pool·짧은 update 로 시간 측정 (§17.3). 출력 root work_dir/_eqrec4_s1_campaign/."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
STAGES = ("g00", "d10", "d20", "d30", "d40", "d50", "k10", "k20", "report")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("stage", choices=STAGES + ("all",)); ap.add_argument("--profile", action="store_true"); ap.add_argument("--pairs", default="A,B"); ap.add_argument("--pair", default="A")
    a = ap.parse_args(); import importlib
    for st in (STAGES if a.stage == "all" else (a.stage,)):
        mod = importlib.import_module(f"tools.eqrec4.{st}")
        if st == "k10":
            mod.main(profile=a.profile, pairs=tuple(a.pairs.split(",")))
        elif st == "k20":
            mod.main(profile=a.profile, pair=a.pair)
        else:
            r = mod.main(profile=a.profile)
            if st == "g00" and isinstance(r, dict) and r.get("implementation_invalid") and a.stage == "all":
                print("!! G00 implementation_invalid — 중단:", r["implementation_invalid"]); sys.exit(2)


if __name__ == "__main__":
    main()
