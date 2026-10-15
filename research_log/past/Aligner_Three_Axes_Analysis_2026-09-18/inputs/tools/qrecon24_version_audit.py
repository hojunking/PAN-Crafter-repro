#!/usr/bin/env python
"""QRECON24 run 의 **학습 정의 버전 감사** (ADJ-R1 계획 research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md §8.1·§8.2).

    python tools/qrecon24_version_audit.py <run> [<run> ...] [--out <json>] [--no-write]
    python tools/qrecon24_version_audit.py --all [--out work_dir/_qrecon24/version_audit.json]

실제 시작 시 저장된 manifest/로그가 우선이다 (Sheet 설명이 아니다): kdv_config_resolved.json 의 qrecon 요약(formula·numerator_factor·uniform_weight), spec 의 λE(절대),
meta/config.yaml 스냅샷(stat.outer_weight·qrecon.uniform_weight), meta/git_commit.txt(release), meta/command.txt(--resume 여부), 같은 id 의 이전 시도(ledger '<run>#N', PREV_CRASHED 등).
판정(§8.1 표):
  single_definition_factor1  — 분자 1(numerator_factor 1.0)·uniform .5·λE == 스냅샷 outer_weight, 재개 없음(또는 exact_resume 이고 비정확 재개 아님) → 유효한 새 방식 결과 (근거 manifest 연결)
  factor2_or_old_definition  — numerator_factor 2 / 옛 formula(2·qref) / uniform_weight 없음 → 구버전 결과로 보존, 새 방식 대응 분석에서 제외
  unverified                 — manifest 부족 · 비정확 재개 · λE 불일치 등 → 버전 미확정(별도 표기; 정상 비교 기준으로 쓰지 않음)
이 도구는 run 디렉터리·ledger 를 읽기만 하고(기본으로 results/qrecon24_version_audit.json 을 남긴다; --no-write 면 안 남김) 과거 metadata 를 덮어쓰지 않는다.
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import gen_pakd50_configs as G  # noqa: E402

AUDIT_VERSION = "QRC24_VERSION_AUDIT_v1"


def _j(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _yaml_kdv(p):
    try:
        import yaml
        return (yaml.safe_load(open(p)) or {}).get("kdv") or {}
    except Exception:
        return {}


def audit_run(run, write=True):
    rd = os.path.join(ROOT, "work_dir", run); out = dict(run=run, audit=AUDIT_VERSION, checked=time.strftime("%Y-%m-%dT%H:%M:%S"), evidence={}, notes=[])
    if not os.path.isdir(rd):
        out["verdict"] = "unverified"; out["notes"].append("run 디렉터리 없음"); return out
    kc = _j(os.path.join(rd, "kdv_config_resolved.json")); qr = kc.get("qrecon") or {}; sp = (kc.get("spec") or {}).get("qrecon") or {}
    kb = _yaml_kdv(os.path.join(rd, "meta", "config.yaml")); st = kb.get("stat") or {}; qk = kb.get("qrecon") or {}; qc = kb.get("qrc24") or {}
    gc = os.path.join(rd, "meta", "git_commit.txt"); cmd = os.path.join(rd, "meta", "command.txt"); started = os.path.join(rd, "meta", "started_at.txt")
    ev = out["evidence"]
    ev["release_sha"] = open(gc).read().strip() if os.path.exists(gc) else None
    ev["started_at"] = open(started).read().strip() if os.path.exists(started) else None
    ev["command"] = open(cmd).read().strip().splitlines() if os.path.exists(cmd) else []
    ev["resumed_cli"] = any("--resume" in l for l in ev["command"])
    ev["formula"] = qr.get("formula"); ev["numerator_factor"] = qr.get("numerator_factor"); ev["uniform_weight_manifest"] = qr.get("uniform_weight"); ev["qref"] = qr.get("qref")
    ev["lambda_E_spec"] = sp.get("lambda_E"); ev["uniform_weight_spec"] = sp.get("uniform_weight")
    ev["outer_weight_config"] = st.get("outer_weight"); ev["uniform_weight_config"] = qk.get("uniform_weight"); ev["queue_revision"] = qc.get("queue_revision"); ev["profile"] = qc.get("profile")
    ev["exact_resume"] = kc.get("exact_resume"); ev["resumed_nonexact"] = kc.get("resumed_nonexact")
    led = _j(os.path.join(ROOT, G.QRC24_LEDGER)); ent = led.get("entries") or {}
    ev["ledger_status"] = (ent.get(run) or {}).get("status"); ev["train_hours"] = (ent.get(run) or {}).get("train_hours")
    ev["previous_attempts"] = [dict(id=k, status=v.get("status"), hours=v.get("hours"), note="같은 id 의 이전 시도(작업 디렉터리는 옮겨졌거나 지워졌다; 현재 결과와 무관)") for k, v in ent.items() if k.startswith(run + "#")]
    # 판정
    if not kc:
        out["verdict"] = "unverified"; out["notes"].append("kdv_config_resolved.json 없음 (시작 manifest 부재)")
    elif (ev["numerator_factor"] == 2.0) or (isinstance(ev["formula"], str) and ("2·qref" in ev["formula"] or "2qref" in ev["formula"] or "2*qref" in ev["formula"])) or (ev["numerator_factor"] is None and ev["uniform_weight_manifest"] is None):
        out["verdict"] = "factor2_or_old_definition"; out["notes"].append("분자 2 / uniform 1 시절의 정의로 학습 — 새 방식 대응 분석에서 제외, 결과는 구버전으로 보존")
    else:
        ok = (ev["numerator_factor"] == 1.0) and (ev["uniform_weight_manifest"] == 0.5) and (ev["lambda_E_spec"] is not None) and (ev["outer_weight_config"] is not None) and abs(float(ev["lambda_E_spec"]) - float(ev["outer_weight_config"])) < 1e-12
        if not ok:
            out["verdict"] = "unverified"; out["notes"].append("numerator/uniform/λE 증거 불일치 또는 부족")
        elif ev["resumed_cli"] and not (ev["exact_resume"] and ev["resumed_nonexact"] is False):
            out["verdict"] = "unverified"; out["notes"].append("재개된 run 인데 exact resume 증거가 없다 (정의가 섞였을 수 있음)")
        else:
            out["verdict"] = "single_definition_factor1"
            if ev["resumed_cli"]:
                out["notes"].append("exact resume 로 재개(비정확 재개 아님) — 같은 id 의 단일 정의로 본다")
    if ev["previous_attempts"]:
        out["notes"].append(f"이전 시도 {len(ev['previous_attempts'])} 건이 ledger 에 있다(현재 디렉터리와 별개; 예: 옛 코드로 시작했다가 버린 run)")
    if write and os.path.isdir(os.path.join(rd, "results")):
        json.dump(out, open(os.path.join(rd, "results", "qrecon24_version_audit.json"), "w"), indent=1, ensure_ascii=False)
    return out


def completed_qrc24_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "work_dir", "PAKD50_QRC24_*"))):
        r = os.path.basename(d)
        if os.path.exists(os.path.join(d, "results", "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(d, "results", "full_best_hqnr.mat")):
            runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*"); ap.add_argument("--all", action="store_true", help="이 서버의 완료 QRC24 run 전부"); ap.add_argument("--out", default=None); ap.add_argument("--no-write", action="store_true", help="run results/ 에 per-run json 을 남기지 않는다 (dry-run)")
    a = ap.parse_args()
    runs = completed_qrc24_runs() if a.all else list(a.runs)
    if not runs:
        print("[qrecon24-audit] 대상 run 없음"); return 0
    res = [audit_run(r, write=not a.no_write) for r in runs]
    for r in res:
        e = r["evidence"]; print(f"  {r['run']:<62} {r['verdict']:<28} factor {e.get('numerator_factor')} uniform {e.get('uniform_weight_manifest')} λE {e.get('lambda_E_spec')} release {(e.get('release_sha') or '?')[:12]} resume {e.get('resumed_cli')} prev {len(e.get('previous_attempts') or [])}" + (f" — {'; '.join(r['notes'])}" if r["notes"] else ""))
    out = a.out or (os.path.join(ROOT, "work_dir", "_qrecon24", "version_audit.json") if a.all else None)
    if out and not a.no_write:
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(dict(audit=AUDIT_VERSION, checked=time.strftime("%Y-%m-%dT%H:%M:%S"), runs=res), open(out, "w"), indent=1, ensure_ascii=False); print(f"[qrecon24-audit] → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
