#!/usr/bin/env python
"""DCR12 CLI (research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md; s1 실행판).

    python tools/dcr12.py d00|d01|d02|d03|d04|results|report [--profile]
    python tools/dcr12.py check --of d01
D01–D03 은 다시 부르면 새로 생긴 checkpoint 분만 덧붙인다(학습 전 T0/B0-1234, 학습 뒤 전부). 출력 root work_dir/_dcr12_<server>_campaign/."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
STAGES = ("d00", "d01", "d02", "d03", "d04", "results", "report")
REQUIRED = {"d00": ["manifest.json", "source_identity.json", "d00_smoke.json", "panel_ids.json", "probe_manifest.json"], "d01": ["sample_metrics.csv", "quadrant_thresholds.json", "quadrant_summary.csv", "d01_stats.json", "scene_metrics.csv"],
            "d02": ["correction_interventions.csv", "closure_invariance.csv", "synthetic_composition.csv", "d02_stats.json"], "d03": ["gradient_conflicts.csv", "d03_summary.json"], "d04": ["micro_update_utility.csv", "routing_gate.json"],
            "results": ["checkpoint_metrics.csv", "paired_results.csv"], "report": ["report.md", "run_status.json"]}


def check(stage):
    from tools.dcr12 import common as C
    miss = [f for f in REQUIRED.get(stage, []) if not os.path.exists(os.path.join(C.CAMP, f))]
    print(f"[check] {stage}: " + ("complete" if not miss else "missing " + " ".join(miss))); return not miss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("stage", choices=STAGES + ("check",)); ap.add_argument("--profile", action="store_true"); ap.add_argument("--of", default=None)
    a = ap.parse_args(); import importlib
    if a.stage == "check":
        sys.exit(0 if check(a.of) else 1)
    mod = importlib.import_module(f"tools.dcr12.{a.stage}"); r = mod.main(profile=a.profile)
    if a.stage == "d00" and isinstance(r, dict) and r.get("implementation_invalid"):
        print("!! D00 implementation_invalid — 중단:", r["implementation_invalid"]); sys.exit(2)


if __name__ == "__main__":
    main()
