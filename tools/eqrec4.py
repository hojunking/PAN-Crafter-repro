#!/usr/bin/env python
"""EQREC4-S1-v1 CLI (research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md).

    python tools/eqrec4.py g00|d10|d20|d30|d40|d50|k10|k20|report [--profile] [--pairs A,B] [--pair A]
    python tools/eqrec4.py all            # G00 → … → report (tools/eqrec4_run.sh 가 detached 로 부른다)
--profile: 첫 128 patch·1 pool·짧은 update 로 시간 측정 (§17.3). 출력 root work_dir/_eqrec4_s1_campaign/."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
STAGES = ("g00", "d10", "d20", "d30", "d30b", "d40", "d50", "k10", "k20", "report")
# stage 완료 인정에 필요한 산출물 (감사 Q11: ledger 의 done 문자열만으로 건너뛰지 않는다)
REQUIRED = {"g00": ["sources_manifest.json", "protocol_resolved.yaml", "g00_reproduction.json", "data_manifest.csv", "probe_bank_A.json"], "d10": ["native_sample_metrics.csv", "quadrant_assignments.csv", "calibration_thresholds.json", "h1_stats.json", "detail_subsets.json"],
            "d20": ["geometry_modality.csv", "response_matrices.csv", "geometry_controls.csv"], "d30": ["correction_interventions.csv", "h3_stats.json", "correction_landscape.csv"], "d30b": ["correction_interventions_core.csv", "h3_core_stats.json", "blur_control_status_A.json"],
            "d40": ["output_stress.csv", "h2_stats.json", "band_edge_profiles.csv"], "d50": ["gradient_interventions.csv", "gradient_summary.json"], "k10": ["k10_utility.csv", "k10_trial_manifest.json"], "k20": ["k20_gate.json"], "report": ["report_EQREC4.md", "hypothesis_verdicts.json"]}


def check(stage):
    from tools.eqrec4 import common as C
    miss = [f for f in REQUIRED.get(stage, []) if not os.path.exists(os.path.join(C.CAMP, f))]
    print(f"[check] {stage}: " + ("complete" if not miss else "missing " + " ".join(miss))); return not miss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("stage", choices=STAGES + ("all", "check")); ap.add_argument("--profile", action="store_true"); ap.add_argument("--pairs", default="A,B"); ap.add_argument("--pair", default="A"); ap.add_argument("--of", default=None)
    a = ap.parse_args(); import importlib
    if a.stage == "check":
        sys.exit(0 if check(a.of) else 1)
    for st in (STAGES if a.stage == "all" else (a.stage,)):
        mod = importlib.import_module(f"tools.eqrec4.{st}")
        if st == "k10":
            mod.main(profile=a.profile, pairs=tuple(a.pairs.split(",")))
        elif st == "k20":
            mod.main(profile=a.profile, pair=a.pair)
        else:
            r = mod.main(profile=a.profile)
            if st == "g00" and isinstance(r, dict) and r.get("implementation_invalid"):       # 감사 Q11: 단독 호출에서도 실패를 exit code 로 전파
                print("!! G00 implementation_invalid — 중단:", r["implementation_invalid"]); sys.exit(2)


if __name__ == "__main__":
    main()
