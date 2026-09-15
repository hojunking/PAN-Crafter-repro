#!/usr/bin/env python
"""SMEC12 CLI (research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md). 준비 학습은 tools/gen_smec12_bootstrap.py + tools/smec12_prepare.sh.

    python tools/smec12.py a00|d10|i20|i23|i24|i25|x40|report [--profile]
    python tools/smec12.py check --of d10
분석 stage 는 **있는 자산만** 처리하고 다시 부르면 새로 완료된 모델 분만 덧붙인다 (QB/GF2 준비 학습이 끝날 때마다 재실행). 출력 root work_dir/_smec12_<server>_campaign/."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
STAGES = ("a00", "d10", "i20", "i23", "i24", "i25", "x40", "report")
REQUIRED = {"a00": ["manifests/asset_registry.json", "manifests/data_registry.json", "manifests/bootstrap_queue.json", "manifests/coverage.csv", "manifests/probe_manifest.json"], "d10": ["raw/sample_metrics.csv", "manifests/thresholds.json", "analysis/d10_stats.json"],
            "i20": ["raw/fixed_residual_response.csv", "analysis/i20_stats.json"], "i23": ["raw/correction_interventions.csv", "analysis/i23_stats.json"], "i24": ["raw/interpolation_controls.csv", "analysis/i24_stats.json"], "i25": ["raw/gradient_interventions.csv", "analysis/i25_stats.json"],
            "x40": ["analysis/cross_sensor_results.csv"], "report": ["report_SMEC12.md", "analysis/hypothesis_verdicts.csv"]}


def check(stage):
    from tools.smec12 import common as C
    miss = [f for f in REQUIRED.get(stage, []) if not os.path.exists(os.path.join(C.CAMP, f))]
    print(f"[check] {stage}: " + ("complete" if not miss else "missing " + " ".join(miss))); return not miss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("stage", choices=STAGES + ("check",)); ap.add_argument("--profile", action="store_true"); ap.add_argument("--of", default=None)
    a = ap.parse_args(); import importlib
    if a.stage == "check":
        sys.exit(0 if check(a.of) else 1)
    mod = importlib.import_module(f"tools.smec12.{a.stage}"); r = mod.main(profile=a.profile)
    if a.stage == "a00" and isinstance(r, dict) and r.get("implementation_invalid"):
        print("!! A00 implementation_invalid:", r["implementation_invalid"]); sys.exit(2)


if __name__ == "__main__":
    main()
