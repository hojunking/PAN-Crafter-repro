#!/usr/bin/env python
"""EQREC4 registry bundle — 계획 §2.2 의 frozen checkpoint 24벌 + pair 학생 2벌을 s1 work_dir 에서 꺼내(pack) 다른 서버(s3 등)의 같은 work_dir 배치로 넣는다(install). 가상 checkpoint 를 만들지 않는다.

    (s1)  python tools/eqrec4_bundle.py pack   [--out work_dir/_eqrec4_bundle]        # 파일 복사 + bundle_manifest.json (sha256, 크기)
          → 전송은 사람이: rsync -a s1:/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_eqrec4_bundle/ work_dir/_eqrec4_bundle/
    (s3)  python tools/eqrec4_bundle.py verify [--bundle work_dir/_eqrec4_bundle]     # sha256 대조
          python tools/eqrec4_bundle.py install [--bundle ...]                        # work_dir/<run>/{meta/config.yaml, <ckpt>/model.safetensors, <tag>_meta.json} 로 배치 (기존 파일이 있으면 sha 가 같을 때만 통과)
bundle 에 들어가는 것: run 마다 meta/config.yaml · best_hqnr/model.safetensors + best_hqnr_meta.json · last/model.safetensors + last_meta.json · (pair 학생) candidates/step-N/model.safetensors.
model.safetensors 는 A+U 한 쌍(같은 checkpoint) — best U-Net 과 last aligner 를 섞지 않는다 (§2.2)."""
import argparse, json, os, shutil, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.teacher_assets import sha256_file                                            # noqa: E402
from tools.eqrec4 import common as C                                                  # noqa: E402

TAG_DIR = dict(best_raw="best_hqnr", last="last")


def files_for(fam, seed, tag):
    run = C.run_of(fam, seed); d = C.ckpt_dir(fam, seed, tag)
    if d is None:
        return run, None
    rel = os.path.relpath(d, os.path.join(ROOT, "work_dir", run)); out = [os.path.join("meta", "config.yaml"), os.path.join(rel, "model.safetensors")]
    if tag in TAG_DIR:
        out.append(f"{TAG_DIR[tag]}_meta.json")
    return run, out


def entries():
    seen = set(); out = []
    for fam, seed, tag in C.CORE + C.EXTRA + [C.PAIRS[p]["S"] for p in C.PAIRS]:
        run, fl = files_for(fam, seed, tag)
        if fl is None:
            out.append(dict(model_key=C.mkey(fam, seed, tag), run=run, missing=True)); continue
        for f in fl:
            if (run, f) not in seen:
                seen.add((run, f)); out.append(dict(model_key=C.mkey(fam, seed, tag), run=run, file=f, missing=False))
    return out


def pack(out):
    os.makedirs(out, exist_ok=True); man = dict(campaign="EQREC4 registry bundle", packed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), source_server=C.SERVER, files=[], missing=[]); total = 0
    for e in entries():
        if e["missing"]:
            man["missing"].append(e["model_key"]); continue
        src = os.path.join(ROOT, "work_dir", e["run"], e["file"]); dst = os.path.join(out, e["run"], e["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or sha256_file(dst) != sha256_file(src):
            shutil.copy2(src, dst)
        sz = os.path.getsize(dst); total += sz; man["files"].append(dict(run=e["run"], file=e["file"], sha256=sha256_file(dst), bytes=sz, model_key=e["model_key"]))
    man["total_mb"] = round(total / 2 ** 20, 1); json.dump(man, open(os.path.join(out, "bundle_manifest.json"), "w"), indent=1)
    print(f"[bundle] {len(man['files'])} files, {man['total_mb']} MB → {out}; missing: {man['missing']}\n[bundle] 전송: rsync -a <s1>:{os.path.abspath(out)}/ work_dir/_eqrec4_bundle/   (그 뒤 s3 에서 verify → install)")
    return man


def verify(bundle):
    man = json.load(open(os.path.join(bundle, "bundle_manifest.json"))); bad = []
    for f in man["files"]:
        p = os.path.join(bundle, f["run"], f["file"])
        if not os.path.exists(p) or sha256_file(p) != f["sha256"]:
            bad.append(f"{f['run']}/{f['file']}")
    print(f"[bundle] verify: {len(man['files']) - len(bad)}/{len(man['files'])} OK" + (f"; BAD {bad}" if bad else "")); return not bad


def install(bundle):
    man = json.load(open(os.path.join(bundle, "bundle_manifest.json"))); n_new = n_same = 0; conflicts = []
    for f in man["files"]:
        src = os.path.join(bundle, f["run"], f["file"]); dst = os.path.join(ROOT, "work_dir", f["run"], f["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if sha256_file(dst) == f["sha256"]:
                n_same += 1; continue
            conflicts.append(dst); continue                                   # 이 서버의 다른 run 산출물을 덮어쓰지 않는다 (§17.3)
        shutil.copy2(src, dst); n_new += 1
    print(f"[bundle] install: new {n_new}, already identical {n_same}, conflicts {len(conflicts)}" + (f" — 다른 내용의 파일이 이미 있다 (손대지 않음): {conflicts[:5]}" if conflicts else ""))
    ok = all(C.ckpt_dir(fam, seed, tag) is not None for fam, seed, tag in C.CORE + C.EXTRA + [C.PAIRS[p]["S"] for p in C.PAIRS]); print(f"[bundle] registry 완비: {ok}"); return ok and not conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=("pack", "verify", "install")); ap.add_argument("--out", default=os.path.join(ROOT, "work_dir", "_eqrec4_bundle")); ap.add_argument("--bundle", default=os.path.join(ROOT, "work_dir", "_eqrec4_bundle"))
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.out)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.bundle) else 1)
    else:
        sys.exit(0 if install(a.bundle) else 1)


if __name__ == "__main__":
    main()
