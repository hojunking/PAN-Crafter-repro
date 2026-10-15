#!/usr/bin/env python
"""DCR12 pair bundle — 한 호스트에서 완성한 B0/B1 pair(FQ/JK0, 같은 seed) 를 다른 서버의 같은 work_dir 배치로 옮긴다.
계획 research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md §8.2 의 host A/B 배치(pair 안은 같은 호스트; seed 간 호스트가 다르면 seed×host 결합을 D00/REPORT 가 적는다).

    (학습 호스트)  python tools/dcr12_bundle.py pack --seed 1234 [--out work_dir/_dcr12_bundle]     # 파일 복사 + bundle_manifest.json (sha256, 크기, source_server)
                   → 전송은 사람이: rsync -a <host>:/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_dcr12_bundle/ work_dir/_dcr12_bundle/
    (분석 호스트)  python tools/dcr12_bundle.py verify  [--bundle work_dir/_dcr12_bundle]           # sha256 대조
                   python tools/dcr12_bundle.py install [--bundle ...]                              # work_dir/<run>/... 배치 (기존 파일이 있으면 sha 가 같을 때만 통과) + bundle_provenance.json
bundle 에 들어가는 것(run 마다): meta/{config.yaml,started_at.txt,finished_at.txt,git_commit.txt} · best_hqnr/model.safetensors + best_hqnr_meta.json · last/model.safetensors + last_meta.json
· results/fr_mat20.json · checkpoint_metrics.csv · initialization_hashes.json · candidates/step-{5050,25250,45450,50000}/model.safetensors (D03/D04 격자; optimizer.bin 은
D04 가 fresh AdamW 를 쓰므로 넣지 않는다). results/{reduced,full}_best_hqnr.mat(run 당 0.8 GB, 진단이 읽지 않음) 은 sha/size 만 manifest 에 남기고 --with-mats 일 때만 복사 — install 된 run 은 provenance 의 complete_on_source 로 완료 판정. 가상 checkpoint 를 만들지 않는다 — 없는 필수 파일이 있으면 pack 이 실패한다."""
import argparse, json, os, shutil, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.teacher_assets import sha256_file                                            # noqa: E402
from tools.dcr12 import common as C                                                   # noqa: E402

PROVENANCE = "bundle_provenance.json"
REQUIRED = ["meta/config.yaml", "best_hqnr/model.safetensors", "best_hqnr_meta.json", "last/model.safetensors", "last_meta.json", "checkpoint_metrics.csv", "results/fr_mat20.json"]
OPTIONAL = ["meta/started_at.txt", "meta/finished_at.txt", "meta/git_commit.txt", "initialization_hashes.json"]
MATS = ["results/reduced_best_hqnr.mat", "results/full_best_hqnr.mat"]   # 완료 판정 파일(run 당 0.8 GB; 진단 stage 는 읽지 않는다) — 기본은 sha/size 만 manifest 에 남기고 --with-mats 일 때만 복사
CAND_STEPS = sorted(set(C.D03_STEPS + C.D04_STEPS))


def entries(seed, with_mats=False):
    """seed pair 의 두 run 에 대해 (run, file, required) 목록 — 존재 여부는 pack 이 본다."""
    out = []
    for ck in ("B0", "B1"):
        run = C.run_name(ck, seed)
        for f in REQUIRED + [f"candidates/step-{s}/model.safetensors" for s in CAND_STEPS] + (MATS if with_mats else []):
            out.append(dict(run=run, case=ck, file=f, required=True))
        for f in OPTIONAL:
            out.append(dict(run=run, case=ck, file=f, required=False))
    return out


def pack(seed, out, with_mats=False):
    for ck in ("B0", "B1"):
        if not C.run_complete(ck, seed):
            sys.exit(f"!! {ck}/{C.CASES[ck]} S{seed} ({C.run_name(ck, seed)}) 가 완료가 아니다 — 학습 중/미완 run 은 묶지 않는다 (X12 와 같은 규칙)")
    os.makedirs(out, exist_ok=True); man = dict(campaign=C.CAMPAIGN_ID, kind="DCR12 pair bundle", seed=seed, packed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), source_server=C.SERVER, **{k: v for k, v in C.host_info().items() if k != "server_id"}, files=[], skipped_optional=[], with_mats=with_mats,
                                        completion_evidence={C.run_name(ck, seed): {f: dict(sha256=sha256_file(os.path.join(C.run_dir(ck, seed), f)), bytes=os.path.getsize(os.path.join(C.run_dir(ck, seed), f))) for f in MATS} for ck in ("B0", "B1")}); total = 0; missing = []
    for f in [os.path.join(out, C.run_name(ck, seed), m) for ck in ("B0", "B1") for m in MATS]:
        if not with_mats and os.path.exists(f):
            os.remove(f)                                                    # 이전 pack 이 남긴 .mat 사본 (bundle 안의 복사본만; work_dir 원본은 손대지 않는다)
    for e in entries(seed, with_mats):
        src = os.path.join(ROOT, "work_dir", e["run"], e["file"])
        if not os.path.exists(src):
            (missing if e["required"] else man["skipped_optional"]).append(f"{e['run']}/{e['file']}"); continue
        dst = os.path.join(out, e["run"], e["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or sha256_file(dst) != sha256_file(src):
            shutil.copy2(src, dst)
        sz = os.path.getsize(dst); total += sz; man["files"].append(dict(run=e["run"], case=e["case"], file=e["file"], sha256=sha256_file(dst), bytes=sz))
    if missing:
        sys.exit("!! 필수 파일 없음 — bundle 을 만들지 않는다:\n   " + "\n   ".join(missing))
    man["total_mb"] = round(total / 2 ** 20, 1); json.dump(man, open(os.path.join(out, "bundle_manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[bundle] seed {seed}: {len(man['files'])} files, {man['total_mb']} MB → {out} (source {C.SERVER}; optional 생략 {man['skipped_optional'] or '없음'})\n[bundle] 전송: rsync -a {C.SERVER}:{os.path.abspath(out)}/ work_dir/_dcr12_bundle/   (그 뒤 분석 호스트에서 verify → install)")
    return man


def verify(bundle):
    man = json.load(open(os.path.join(bundle, "bundle_manifest.json"))); bad = []
    for f in man["files"]:
        p = os.path.join(bundle, f["run"], f["file"])
        if not os.path.exists(p) or sha256_file(p) != f["sha256"]:
            bad.append(f"{f['run']}/{f['file']}")
    print(f"[bundle] verify (seed {man.get('seed')}, source {man.get('source_server')}): {len(man['files']) - len(bad)}/{len(man['files'])} OK" + (f"; BAD {bad}" if bad else "")); return not bad


def install(bundle):
    man = json.load(open(os.path.join(bundle, "bundle_manifest.json"))); n_new = n_same = 0; conflicts = []
    if man.get("source_server") == C.SERVER:
        print(f"[bundle] source_server 가 이 서버({C.SERVER}) 와 같다 — install 할 것이 없다"); return True
    for f in man["files"]:
        src = os.path.join(bundle, f["run"], f["file"]); dst = os.path.join(ROOT, "work_dir", f["run"], f["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if sha256_file(dst) == f["sha256"]:
                n_same += 1; continue
            conflicts.append(dst); continue                                   # 이 서버의 다른 산출물을 덮어쓰지 않는다
        shutil.copy2(src, dst); n_new += 1
    for run in sorted({f["run"] for f in man["files"]}):
        json.dump(dict(source_server=man.get("source_server"), hostname=man.get("hostname"), gpu=man.get("gpu"), packed_at=man.get("packed_at"), installed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), installed_on=C.SERVER, seed=man.get("seed"),
                       bundle_files=sum(1 for f in man["files"] if f["run"] == run), complete_on_source=bool((man.get("completion_evidence") or {}).get(run)), completion_evidence=(man.get("completion_evidence") or {}).get(run),
                       note="다른 호스트에서 학습한 run 을 그대로 옮긴 것 — 이 서버가 학습하지 않았다 (§8.2 host A/B 배치). results/*.mat 은 기본 미포함(sha/size 만) — common.run_complete 가 complete_on_source 로 완료로 본다"),
                  open(os.path.join(ROOT, "work_dir", run, PROVENANCE), "w"), indent=1, ensure_ascii=False)
    seed = man.get("seed"); ok = all(C.run_complete(ck, seed) and C.ckpt_path(ck, seed, "best_hqnr") and C.ckpt_path(ck, seed, "last") for ck in ("B0", "B1"))
    print(f"[bundle] install: new {n_new}, already identical {n_same}, conflicts {len(conflicts)}" + (f" — 다른 내용의 파일이 이미 있다 (손대지 않음): {conflicts[:5]}" if conflicts else "") + f" · seed {seed} pair 완비: {ok} · provenance {PROVENANCE}")
    return ok and not conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=("pack", "verify", "install")); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "work_dir", "_dcr12_bundle")); ap.add_argument("--bundle", default=os.path.join(ROOT, "work_dir", "_dcr12_bundle")); ap.add_argument("--with-mats", action="store_true", help="results/{reduced,full}_best_hqnr.mat 도 복사 (run 당 0.8 GB; FR 재평가가 필요할 때만)")
    a = ap.parse_args()
    if a.cmd == "pack":
        seed = a.seed if a.seed is not None else C.G.SERVER_SEED.get(C.SERVER)
        if seed not in C.SEEDS:
            sys.exit(f"!! seed {seed} 는 DCR12 seed {C.SEEDS} 가 아니다 (--seed)")
        pack(seed, a.out, a.with_mats)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.bundle) else 1)
    else:
        sys.exit(0 if install(a.bundle) else 1)


if __name__ == "__main__":
    main()
