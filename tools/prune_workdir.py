"""work_dir 정리 — 재생성 가능한 산출물을 지워 디스크를 회수한다.

run 을 지우지 않는다. **산출물 종류로만** 자르므로 어느 run 이든 판정 수치
(results/*.json · metrics.csv · 시트)와 체크포인트(best_*/last)가 남아
언제든 재평가할 수 있다.

  T2  완료 run 의 epoch-* / checkpoint-*      — accelerate 재개 전용. run 이 끝나면 쓸모없다
  T1  results/*.mat 안의 테스트셋 입력 키      — data/PanCollection 의 h5 에 원본이 있는 복사본

T1 이 지우는 키는 lms/gt/pan/ms 넷이다. 전부 데이터셋을 그대로 복사해 둔 것이라
run 마다 값이 같고, h5 에서 되살릴 수 있다(아래 "복원" 참조). 보존하는 키는
sr · pan_aligned · pan_aligned_forward_fp32 · delta — run 이 실제로 만들어낸 것 전부다.
저장소에서 .mat 을 읽는 곳은 전부 ["sr"] 만 꺼낸다. 유일한 예외가 EXCLUDE 다.

복원:
    reduced_*.mat  <- data/PanCollection/<DS>/reduced_examples_h5/test_<ds>_multiExm1.h5
    full_*.mat     <- run 의 meta/ 에 기록된 입력 h5 (mat20 세트면 full_examples_mat20/)
  현 프로토콜 run 은 h5 와 비트 동일하고, 구 run 은 float32 저장 반올림 때문에
  2047 스케일에서 1e-4 미만 차이가 난다(h5 쪽이 clip 되지 않은 원본이라 정보량이 많다).

  ** 따라서 data/PanCollection/ 을 지우면 안 된다. T1 이후로는 필수 자산이다. **

안전장치:
  - 실행 중(ps)·1시간 내 갱신·완료 표식 없는 run 은 건드리지 않는다
  - T1 은 임시파일 작성 -> 보존 키 비트 단위 검증 -> os.replace 원자적 교체.
    검증에 실패하면 임시파일을 지우고 원본을 그대로 둔다
  - 기본이 dry-run 이다. 실제 삭제는 --apply

사용:
    python tools/prune_workdir.py                 # 계획만 출력 (아무것도 지우지 않음)
    python tools/prune_workdir.py --apply         # T2 + T1 실행
    python tools/prune_workdir.py --tier t2 --apply
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "work_dir")
G = 1073741824

# T1 이 .mat 에서 빼는 키 — 전부 데이터셋 복사본
DROP = {"lms", "gt", "pan", "ms"}
# tools/make_report_figures.py 가 gt/ms/pan 을 직접 읽는 유일한 예외. 정성 비교 그림용
EXCLUDE = {
    os.path.join(WD, "wv3_baseline", "results", "reduced_best_reduced.mat"),
    os.path.join(WD, "wv3_fixed", "results", "reduced_best_reduced.mat"),
}


def du(path):
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return 0


def live_runs():
    """건드리면 안 되는 run -> 이유."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    now = time.time()
    out = {}
    for r in sorted(os.listdir(WD)):
        p = os.path.join(WD, r)
        if not os.path.isdir(p) or r.startswith("_"):
            continue
        if r in ps:
            out[r] = "실행 중(ps)"
            continue
        done = (os.path.exists(os.path.join(p, "results", "reduced_best_val.mat"))
                or os.path.exists(os.path.join(p, "results", "reduced_best_hqnr.mat")))
        if not done:
            out[r] = "완료 표식 없음"
            continue
        try:
            if now - max(os.path.getmtime(os.path.join(p, x)) for x in os.listdir(p)) < 3600:
                out[r] = "1시간 내 갱신"
        except ValueError:
            out[r] = "비어 있음"
    return out


def runs_to_touch(skip):
    return [r for r in sorted(os.listdir(WD))
            if os.path.isdir(os.path.join(WD, r)) and not r.startswith("_") and r not in skip]


def tier2(runs, apply_):
    """완료 run 의 epoch-* / checkpoint-* 삭제."""
    freed = n = 0
    for r in runs:
        for d in sorted(glob.glob(os.path.join(WD, r, "epoch-*"))) + \
                 sorted(glob.glob(os.path.join(WD, r, "checkpoint-*"))):
            if not os.path.isdir(d):
                continue
            freed += du(d)
            n += 1
            if apply_:
                shutil.rmtree(d)
    return n, freed


def tier1(runs, apply_, manifest, cutoff=None):
    """results/*.mat 에서 DROP 키를 빼고 재저장."""
    import numpy as np
    from scipy.io import loadmat, savemat

    freed = n = 0
    err = []
    for r in runs:
        src = None
        for m in glob.glob(os.path.join(WD, r, "meta", "*")):
            try:
                txt = open(m, errors="ignore").read()
            except Exception:
                continue
            for line in txt.splitlines():
                if "OrigScale" in line and ".h5" in line:
                    src = line.strip()[:200]
                    break
            if src:
                break
        manifest["runs"][r] = {"source_h5": src}

        for p in sorted(glob.glob(os.path.join(WD, r, "results", "*.mat"))):
            if p in EXCLUDE:
                continue
            if cutoff is not None and os.path.getmtime(p) >= cutoff:
                continue                                   # --before 이후에 만들어진 것은 건드리지 않는다
            tmp = p + ".prunetmp"
            try:
                if not apply_:
                    from scipy.io import whosmat
                    info = whosmat(p)
                    if not any(k in DROP for k, _, _ in info):
                        continue
                    sz = {"double": 8, "single": 4, "int32": 4, "uint8": 1, "int64": 8, "uint16": 2}
                    for k, shape, cls in info:
                        if k in DROP:
                            nb = sz.get(cls, 4)
                            for s in shape:
                                nb *= s
                            freed += nb
                    n += 1
                    continue

                d = loadmat(p)
                keep = {k: v for k, v in d.items() if not k.startswith("__")}
                drop = {k for k in keep if k in DROP}
                if not drop:
                    continue
                keep = {k: v for k, v in keep.items() if k not in DROP}
                if not keep:                       # 보존할 게 없으면 손대지 않는다
                    err.append((p, "보존 키 없음"))
                    continue
                before = os.path.getsize(p)
                savemat(tmp, keep, do_compression=False, format="5")
                chk = loadmat(tmp)                 # 보존 키가 비트 단위로 같은지 확인
                bad = [k for k, v in keep.items()
                       if k not in chk or chk[k].dtype != v.dtype or chk[k].shape != v.shape
                       or not np.array_equal(chk[k], v)]
                if bad:
                    os.remove(tmp)
                    err.append((p, "검증 실패 %s" % bad))
                    continue
                del d, chk, keep
                os.replace(tmp, p)
                after = os.path.getsize(p)
                freed += before - after
                n += 1
                manifest["files"].append({"path": os.path.relpath(p, ROOT), "dropped": sorted(drop),
                                          "bytes_before": before, "bytes_after": after})
                if n % 25 == 0:
                    print("    %d 파일, %.1fG 회수" % (n, freed / G), flush=True)
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                err.append((p, repr(e)[:120]))
    manifest["errors"] = err
    return n, freed, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["t1", "t2", "all"], default="all")
    ap.add_argument("--apply", action="store_true", help="실제로 지운다 (기본은 계획만 출력)")
    ap.add_argument("--before", default=None, metavar="YYYY-MM-DD",
                    help="T1: 이 날짜 00:00 **이전**에 만들어진 .mat 만 대상 (이후 파일은 제외)")
    a = ap.parse_args()

    if not os.path.isdir(WD):
        sys.exit("work_dir 없음: %s" % WD)

    cutoff = None
    if a.before:
        import datetime
        cutoff = datetime.datetime.strptime(a.before, "%Y-%m-%d").timestamp()
        print(f"T1 대상 제한: {a.before} 00:00 이전 생성분만 (mtime < {cutoff:.0f})\n")

    skip = live_runs()
    runs = runs_to_touch(skip)
    mode = "실행" if a.apply else "DRY-RUN (아무것도 지우지 않는다)"
    print("work_dir %s\n대상 run %d · 제외 %d · %s\n" % (WD, len(runs), len(skip), mode))
    if skip:
        print("제외된 run:")
        for r, w in sorted(skip.items()):
            print("  %-58s <- %s" % (r[:58], w))
        print()

    total = 0
    if a.tier in ("t2", "all"):
        n, f = tier2(runs, a.apply)
        total += f
        print("T2  epoch-*/checkpoint-*  %4d 디렉토리  %7.1fG" % (n, f / G))

    manifest = {"created": time.strftime("%F %T"), "tier": a.tier, "applied": a.apply,
                "before": a.before, "dropped_keys": sorted(DROP), "skipped_runs": skip,
                "runs": {}, "files": []}
    if a.tier in ("t1", "all"):
        n, f, err = tier1(runs, a.apply, manifest, cutoff=cutoff)
        total += f
        print("T1  .mat 입력 키 제거     %4d 파일      %7.1fG%s"
              % (n, f / G, ("  (오류 %d)" % len(err)) if err else ""))
        for p, e in err[:10]:
            print("      ! %s  %s" % (os.path.relpath(p, ROOT), e))

    print("\n합계 %.1fG%s" % (total / G, "" if a.apply else "  — 실제로 지우려면 --apply"))

    if a.apply:
        out = os.path.join(WD, "_prune")
        os.makedirs(out, exist_ok=True)
        mp = os.path.join(out, "prune_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        manifest["freed_bytes"] = total
        json.dump(manifest, open(mp, "w"), indent=1, ensure_ascii=False)
        print("기록: %s" % os.path.relpath(mp, ROOT))
        df = subprocess.run(["df", "-h", ROOT], capture_output=True, text=True).stdout.splitlines()
        print(df[-1] if len(df) > 1 else "")


if __name__ == "__main__":
    main()
