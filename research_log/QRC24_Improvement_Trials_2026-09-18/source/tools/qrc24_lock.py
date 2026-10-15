#!/usr/bin/env python
"""Narrow R2 recipe lock — 마지막 β 비교를 닫고 공통 설정 C* 하나를 동결한다
(계획 research_log/PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md §4.3·§6.1, revision QRC24_NARROW_R2_20260917).

    python tools/qrc24_lock.py --status                 # 현재 lock 과 β 비교표(시트에서 읽는다)
    python tools/qrc24_lock.py --decide                 # §4.3 규칙으로 C* 를 계산해 보여준다 (파일은 쓰지 않는다)
    python tools/qrc24_lock.py --write --profile G23    # 동결 (되돌리기 어렵다 — 확인을 받는다)
    python tools/qrc24_lock.py --verify                 # lock 과 현재 생성기 정의가 일치하는지

**2026-09-18 사용자 결정: 이 도구는 더 이상 아무것도 막지 않는다.** seed 단계는 lock 없이 바로 돈다(설정은 `gen_pakd50_configs.qrc24_seed_profile()`, 기본 G23).
공동 목표 통과 seed 수를 확인해 seed 실행을 보류하던 게이트를 없앴다. 아래 비교표·`--write` 는 **기록용**으로만 남는다 — 쓰더라도 seed 편성은 바뀌지 않는다.

동결 규칙(§4.3, 사전 의사결정 정책이지 통계적 유의성 판정이 아니다):
  1) n_joint(H≥.9585 **그리고** E<2.040 을 같은 checkpoint 에서 만족한 seed 수) 큰 후보
  2) 동률이면 n_H(H 하한 통과 seed 수) 큰 후보
  3) 동률이고 적격 값이 있으면 med_E_H(H 통과 seed 의 선택 ERGAS 중앙값) 낮은 후보
  4) 여전히 같거나 둘 다 적격이 없으면 **G23 유지**
β=.15·rA=.02 같은 제3 후보를 추가하지 않는다. lock 이 없으면 seed 단계(41001–41020) config 는 생성되지 않는다.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import gen_pakd50_configs as G  # noqa: E402

CANDIDATES = G.QRC24_LOCK_CANDIDATES                      # ("G23", "B20A03")
THRESH_H, THRESH_E = 0.9585, 2.040


def _load_local(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, path)); m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m; spec.loader.exec_module(m); return m


def sheet_rows():
    """시트의 QRC24 행 → [(server, profile, seed, H, E)]. 읽기 전용."""
    gu = _load_local("_gu_lock", "gspread/gspread_upload.py"); sc = _load_local("_sc_lock", "gspread/sheet_categories.py")
    import gspread
    gc = gspread.service_account(filename=gu.CRED); sh = gc.open(gu.SHEET)
    pat = re.compile(r"^PAKD50_QRC24_(S\d)_(.+?)_W104_D121_WV3_T0_S(\d+)_FRESH50_v\d+$")
    out = []
    for w in sh.worksheets():
        if not w.title.startswith("WV3-") or w.title.endswith("_v1"):
            continue
        hdr = w.row_values(3)
        try:
            ih, ie = hdr.index("HQNR↑"), hdr.index("ERGAS↓")
        except ValueError:
            continue
        for r in w.get("B4:W"):
            if not r or not r[0]:
                continue
            m = pat.match(sc.run_tag(r[0]) or "")
            if not m:
                continue
            f = lambda i: (float(r[i - 1]) if len(r) >= i and r[i - 1] not in (None, "") else float("nan"))
            try:
                h, e = f(ih), f(ie)
            except ValueError:
                continue
            out.append((m.group(1).lower(), m.group(2), int(m.group(3)), h, e))
    return out


def compare(rows):
    """§4.3 집계. **같은 서버·같은 seed 의 대응 block** 에서만 두 후보를 비교한다."""
    by = {}
    for srv, prof, seed, h, e in rows:
        if prof in CANDIDATES:
            by.setdefault((srv, seed), {})[prof] = (h, e)
    blocks = {k: v for k, v in by.items() if len(v) == len(CANDIDATES)}          # 짝이 있는 block 만
    agg = {}
    for c in CANDIDATES:
        vals = [v[c] for v in blocks.values()]
        hs = [x for x in vals if x[0] == x[0] and x[0] >= THRESH_H]
        joint = [x for x in vals if x[0] == x[0] and x[1] == x[1] and x[0] >= THRESH_H and x[1] < THRESH_E]
        med = None
        if hs:
            es = sorted(x[1] for x in hs if x[1] == x[1])
            med = (es[len(es) // 2] if len(es) % 2 else (es[len(es) // 2 - 1] + es[len(es) // 2]) / 2) if es else None
        agg[c] = dict(n_seed=len(vals), n_joint=len(joint), n_H=len(hs), med_E_H=med)
    return blocks, agg


def decide(agg):
    """§4.3 1→4 순서. 반환 (profile, reason)."""
    a, b = CANDIDATES[0], CANDIDATES[1]
    if agg[a]["n_joint"] != agg[b]["n_joint"]:
        w = a if agg[a]["n_joint"] > agg[b]["n_joint"] else b
        return w, f"1) n_joint {agg[a]['n_joint']} vs {agg[b]['n_joint']}"
    if agg[a]["n_H"] != agg[b]["n_H"]:
        w = a if agg[a]["n_H"] > agg[b]["n_H"] else b
        return w, f"2) n_joint 동률({agg[a]['n_joint']}) · n_H {agg[a]['n_H']} vs {agg[b]['n_H']}"
    ma, mb = agg[a]["med_E_H"], agg[b]["med_E_H"]
    if ma is not None and mb is not None and ma != mb:
        w = a if ma < mb else b
        return w, f"3) n_joint·n_H 동률 · med_E_H {ma:.4f} vs {mb:.4f}"
    return "G23", f"4) 동률이거나 적격 결과 없음 → G23 유지 (n_joint {agg[a]['n_joint']}/{agg[b]['n_joint']}, n_H {agg[a]['n_H']}/{agg[b]['n_H']})"


def recipe_fields(profile):
    """seed·경로 metadata 를 뺀 **학습 정의** (§6.1 recipe hash 의 대상). 모델·loss·LR·전처리는 절대 제외하지 않는다."""
    P = G.qrc24_profile(profile)
    return dict(method="qrecon_continuous_v1", profile=profile, lambda_E=P["lam"], rA=P["rA"], alpha=P["alpha"], beta=P["beta"], U_lr=P["ulr"], A_lr=P["alr"],
                a_weight=P["a"], e_weight=P["e"], A_frozen=P["frozen"], q_formula="qref/(qref+q_T)", q_ref=G.QRC24_QREF, uniform_weight=G.QRC24_UNIFORM_W,
                tau_R=0.012463942170143127, arch=G.QRC24_ARCH, updates=50000, protocol="FRESH50", grid="GRID1010_50K_v1",
                teacher=dict(run="assets/pakd50/T0_run", tag="best_hqnr", sha256=G.t0_identity(None)[0] if hasattr(G, "t0_identity") else None),
                cue_asset=G.QEDGE9_CUE_ASSET, selector=G.QRC24_R2_SELECTOR, gradient_routing="U ← H+K+λE·s_q·E · A ← s_q·H 만",
                excluded="Student offset/jitter · soft/edge→A 직접 gradient · 새 loss/정규화/head · U-Net FT/tail")


def write_lock(profile, reason, agg, blocks, force=False):
    p = os.path.join(ROOT, G.QRC24_LOCK_FILE)
    if os.path.exists(p) and not force:
        raise SystemExit(f"!! 이미 lock 이 있다: {p} (불변이 원칙 — 바꾸려면 --force 와 사유를 남긴다)")
    rec = recipe_fields(profile)
    lk = dict(lock_id=G.QRC24_LOCK_ID, revision=G.QRC24_R2_REVISION, plan=G.QRC24_R2_PLAN, profile=profile, decided_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
              decision_rule=reason, comparison=dict(aggregate=agg, blocks={f"{k[0]}_S{k[1]}": {c: dict(hqnr=v[c][0], ergas=v[c][1]) for c in v} for k, v in blocks.items()}),
              recipe=rec, recipe_sha256=hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              selector=G.QRC24_R2_SELECTOR, seeds={s: list(v) for s, v in G.QRC24_R2_SEEDS.items()},
              note="seed 와 허용된 식별/경로 metadata 만 run 마다 달라진다. 중간 결과로 β·LR·loss 를 바꾸지 않는다 (R2 §6.2).")
    os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".tmp"; json.dump(lk, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, p)
    return lk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true"); ap.add_argument("--decide", action="store_true"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--profile", default=None, choices=CANDIDATES); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cur = G.qrc24_recipe_lock()
    if a.verify:
        if not cur:
            print("[lock] 없음 — seed 단계는 생성되지 않는다"); return 1
        rec = recipe_fields(cur["profile"]); sha = hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        ok = sha == cur.get("recipe_sha256")
        print(f"[lock] {cur['lock_id']} · C* {cur['profile']} · recipe sha {'일치' if ok else '불일치(생성기 정의가 바뀌었다)'}")
        if not ok:
            print(f"   lock {cur.get('recipe_sha256')[:16]} vs 현재 {sha[:16]} — 정의가 바뀌었으면 lock 을 다시 결정해야 한다")
        print(f"   seed 단계: " + ", ".join(f"{s} {G.QRC24_R2_SEEDS[s]}" for s in sorted(G.QRC24_R2_SEEDS)))
        return 0 if ok else 1
    if cur and not (a.decide or a.status):
        print(f"[lock] 이미 동결됨: {cur['lock_id']} · C* {cur['profile']} ({cur['decided_at']}) · 근거 {cur['decision_rule']}"); return 0
    rows = sheet_rows(); blocks, agg = compare(rows); prof, reason = decide(agg)
    print(f"[lock] β 비교 — 짝이 맞는 block {len(blocks)} 개 (같은 서버·같은 seed 에 두 후보가 다 있는 경우만)")
    for (srv, seed), v in sorted(blocks.items()):
        line = " · ".join(f"{c} H {v[c][0]:.4f} E {v[c][1]:.4f}" for c in CANDIDATES)
        print(f"   {srv} S{seed}: {line}")
    for c in CANDIDATES:
        m = agg[c]["med_E_H"]
        print(f"   {c:<8} n_seed {agg[c]['n_seed']} · n_joint {agg[c]['n_joint']} · n_H {agg[c]['n_H']} · med_E_H {('%.4f' % m) if m is not None else '—'}")
    print(f"[lock] §4.3 규칙 → C* = **{prof}** ({reason})")
    if not blocks:
        print("   !! 짝이 맞는 block 이 하나도 없다 — B20A03 을 기존 G23 의 seed 에 맞춰 먼저 돌려야 한다 (R2 §4.1: s1 1234/3407 · s2 777 · s3 2026)")
    if a.status or a.decide:
        print("   (파일을 쓰지 않았다. 동결하려면 --write --profile <C*>)"); return 0
    if a.write:
        if not a.profile:
            print("!! --profile 을 명시할 것 (권고 계산값과 다를 수 있다 — 사람이 결정한다)"); return 1
        lk = write_lock(a.profile, reason if a.profile == prof else f"사용자 지정 (규칙 계산값 {prof}: {reason})", agg, blocks, force=a.force)
        print(f"[lock] 동결: {lk['lock_id']} · C* {lk['profile']} · recipe sha {lk['recipe_sha256'][:16]} → {G.QRC24_LOCK_FILE}")
        print(f"   이제 seed 단계 config 가 생성된다: " + ", ".join(f"{s}{G.QRC24_R2_SEEDS[s]}" for s in sorted(G.QRC24_R2_SEEDS)))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
