#!/usr/bin/env python
"""QRECON24 자산 보존 (ADJ-R1 계획 research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md §7.1): 완료 run 의 **선택 checkpoint(A/U 한 파일)·시작 manifest·config·git·Teacher/cue/init hash·fr_mat20 결과**
를 work_dir/_qrecon24/preserved/<run>/ 에 복사하고 sha256 목록(preserve_manifest.json) 을 남긴다. 원본은 건드리지 않는다(prune 대상이 아닌 별도 사본; git 밖).

    python tools/qrecon24_preserve.py <run> [--dest work_dir/_qrecon24/preserved] [--verify]

보존 대상: best_hqnr/model.safetensors(= best_raw alias; selected raw-original step 의 Student A+U) · candidates/step-<selected>/model.safetensors(있으면) · best_raw_meta/best_hqnr_meta/best_state ·
checkpoint_metrics.csv · kdv_config_resolved/calibration_resolved/pa_config_resolved · init_and_teacher_hashes/initialization_hashes/dataset_hashes · architecture/baseline/cost manifest · run_key · meta/* ·
results/fr_mat20.json · results/qrecon24_target_selection.json(+ reduced_candidate_step-*.json sidecar) · results/qrecon24_version_audit.json (있는 것만).
Sheet 의 HQNR 은 selected raw-original 값이다 — '50K run' 표기만 보고 exact50K 점수로 쓰지 않는다 (§7.1).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESERVE_VERSION = "QRC24_PRESERVE_v1"
FILES = ["best_hqnr/model.safetensors", "best_raw_meta.json", "best_hqnr_meta.json", "best_state.json", "checkpoint_metrics.csv", "kdv_config_resolved.json", "calibration_resolved.json", "pa_config_resolved.json",
         "init_and_teacher_hashes.json", "initialization_hashes.json", "dataset_hashes.json", "architecture_manifest.json", "baseline_manifest.json", "cost_manifest.json", "run_key.json", "last_meta.json",
         "results/fr_mat20.json", "results/qrecon24_target_selection.json", "results/qrecon24_version_audit.json", "results/export_convention.json"]


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def preserve(run, dest_root, verify=False):
    rd = os.path.join(ROOT, "work_dir", run)
    if not os.path.isdir(rd):
        raise SystemExit(f"!! run 디렉터리 없음: {rd}")
    dest = os.path.join(ROOT, dest_root, run); mp = os.path.join(dest, "preserve_manifest.json")
    if verify:
        if not os.path.exists(mp):
            raise SystemExit(f"!! 보존 manifest 없음: {mp}")
        man = json.load(open(mp)); bad = []
        for rel, sha in man["files"].items():
            p = os.path.join(dest, rel)
            if not os.path.exists(p) or sha256_file(p) != sha:
                bad.append(rel)
        print(f"[qrecon24-preserve] verify {run}: {len(man['files'])} 파일 중 불일치 {len(bad)}" + (f" — {bad[:5]}" if bad else " (전부 일치)"))
        return 1 if bad else 0
    rel_files = list(FILES)
    meta = json.load(open(os.path.join(rd, "best_raw_meta.json"))) if os.path.exists(os.path.join(rd, "best_raw_meta.json")) else {}
    step = meta.get("step")
    if step is not None and os.path.exists(os.path.join(rd, "candidates", f"step-{int(step)}", "model.safetensors")):
        rel_files.append(f"candidates/step-{int(step)}/model.safetensors")
    rel_files += [os.path.relpath(p, rd) for p in sorted(glob.glob(os.path.join(rd, "meta", "*")))] + [os.path.relpath(p, rd) for p in sorted(glob.glob(os.path.join(rd, "results", "reduced_candidate_step-*.json")))]
    os.makedirs(dest, exist_ok=True); files = {}; skipped = []; copied = 0
    prev = json.load(open(mp))["files"] if os.path.exists(mp) else {}
    for rel in rel_files:
        src = os.path.join(rd, rel)
        if not os.path.isfile(src):
            skipped.append(rel); continue
        sha = sha256_file(src); dst = os.path.join(dest, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and prev.get(rel) == sha and sha256_file(dst) == sha):
            shutil.copy2(src, dst); copied += 1
        files[rel] = sha
    ith = json.load(open(os.path.join(rd, "init_and_teacher_hashes.json"))) if os.path.exists(os.path.join(rd, "init_and_teacher_hashes.json")) else {}
    cal = json.load(open(os.path.join(rd, "calibration_resolved.json"))) if os.path.exists(os.path.join(rd, "calibration_resolved.json")) else {}
    gc = os.path.join(rd, "meta", "git_commit.txt")
    man = dict(preserve=PRESERVE_VERSION, run=run, preserved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), source_dir=rd, selected=dict(step=step, epoch=meta.get("epoch"), selection_view=meta.get("selection_view"), hqnr=meta.get("hqnr"), rr_scc=meta.get("rr_scc"), rr_ergas=meta.get("rr_ergas")),
               release_sha=(open(gc).read().strip() if os.path.exists(gc) else None), teacher_sha256=((ith.get("teacher") or {}).get("file_sha256")), cue_asset_id=((cal.get("qrecon") or {}).get("asset_id")),
               note="Sheet HQNR = selected raw-original(best_raw alias best_hqnr); exact50K 점수가 아니다 (§7.1). 이 사본은 git 밖(work_dir/_qrecon24/preserved) — prune 과 무관하게 남긴다",
               files=files, skipped=skipped)
    json.dump(man, open(mp, "w"), indent=1, ensure_ascii=False)
    print(f"[qrecon24-preserve] {run}: {len(files)} 파일 보존({copied} 복사, {len(files) - copied} 이미 일치) · 없음 {len(skipped)} · selected step {step} H {meta.get('hqnr')} · → {dest}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--dest", default=os.path.join("work_dir", "_qrecon24", "preserved")); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(); return preserve(a.run, a.dest, verify=a.verify)


if __name__ == "__main__":
    sys.exit(main())
