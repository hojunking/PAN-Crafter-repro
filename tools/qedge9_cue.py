#!/usr/bin/env python
"""QEDGE9 cue 자산 (계획 research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md §2·§3.1·§6·§7) — 고정 T0 aligner 의 offset-consistency q(AXIS16 = EQREC4 bank A), θq, gate, T0 native e_T_roi32,
QES 셔플 assignment 를 **train view 전체**(N base patch × 4 augmentation state = 현재 feeder 계약: hflip/vflip 고정 + rot 0..3, crop 없음, §7.2) 에 대해 한 번 만들고 hash 로 고정한다. Student/seed/update 와 무관(§7.1).

    python tools/qedge9_cue.py build  [--config config/PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml] [--name cue_T0_AXIS16_v1] [--chunk 128] [--bench 256]   # (s1) → assets/qedge9/<name>.{json,npz}
    python tools/qedge9_cue.py verify [--n 96] [--seed 0]           # 어느 서버든: 고정 view 재계산 vs cache — |Δq| 최대, gate label 차이, θq 근처 수 → work_dir/_qedge9/verify_<server>.json (§7.4 서버 간 대조)
    python tools/qedge9_cue.py pilot  --run PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1 [--tag last] [--all-views]   # (s4) QEC §6.1: c_E = Σ g E_pilot / Σ E_pilot (calibration view) → work_dir/_qedge9/qec_cE.json
    python tools/qedge9_cue.py status
Teacher A 의 입력 adapter(view margin 4, M = bicubic↑4, z-score 는 aligner 내부) 는 Teacher forward 와 같다(§7.3). e_T_roi32 = T0 native 복원의 중앙 [16:48]² L1(정규화 단위) — 학습 loss support 는 바꾸지 않는다(§2.2).
θq = calibration view(기존 train calibration seed 1234 · 3072 base id × 4 state) 의 q 중앙값 (§7.3); FR20/RR 로 정하지 않는다. T_cue 는 먼저 bench 뒤 실제 총시간으로 기록한다(§7.4)."""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, torch, h5py, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv import edge_gate as EG                                                      # noqa: E402
from kdv.teacher_assets import load_run_model, freeze, state_hash, sha256_file     # noqa: E402
from main import import_class                                                        # noqa: E402
from tools import gen_pakd50_configs as G                                            # noqa: E402

ASSET_DIR = "assets/qedge9"; CAMP = "work_dir/_qedge9"; DEFAULT_CONFIG = "config/PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); CAL_N, CAL_SEED = 3072, 1234; ROTS = list(EG.ROT_STATES)


def server():
    p = os.path.join(ROOT, "gspread", "server.txt"); return open(p).read().strip() if os.path.exists(p) else "unknown"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def sha_cached(path):
    """train h5 sha256 (train_pa 의 캐시 규약; 캐시 디렉토리는 work_dir/_qedge9)."""
    from train_pa import sha256_file as _s
    d = os.path.join(ROOT, CAMP); os.makedirs(d, exist_ok=True); return _s(path, d)


def contract_from(cfg):
    fa = cfg["train_feeder_args"]; c = {k: bool(fa.get(k, False)) for k in EG.FEEDER_CONTRACT_KEYS}
    if c != dict(hflip=True, vflip=True, rot=True, crop=False):
        sys.exit(f"!! feeder 계약 {c} 는 계획 §7.2(고정 flip + 4 rotation, crop 없음) 와 다르다 — 상태 수·입력 경로를 다시 정의하고 cache 를 새로 만든다")
    tr = fa["dataroot"]; tr = tr if os.path.isabs(tr) else os.path.join(ROOT, tr)
    return c, tr, tr.replace(".h5", "_pan.h5")


def max_pixel_of(path):
    """feeders.feeder.PanFeeder 의 문자열 규칙 그대로 (max_pixel 인자 없이 만들어지는 값)."""
    if "wv3" in path or "qb" in path or "wv2" in path:
        return 2047.0
    if "gf2" in path:
        return 1023.0
    raise ValueError(path)


def calibration_ids(N):
    """kdv.calibration.calibration_batches 와 같은 선택: randperm(N, seed 1234)[:3072] 정렬 (학생 seed 와 무관)."""
    g = torch.Generator(device="cpu"); g.manual_seed(CAL_SEED); n = min(CAL_N, N)
    idx = torch.randperm(N, generator=g)[:n].sort().values.tolist()
    return idx, hashlib.sha256(np.array(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


def load_T0():
    srv = server(); tdir = os.path.join(ROOT, G.t0_dir(srv)); cfg = yaml.safe_load(open(os.path.join(tdir, "meta", "config.yaml"))); Model = import_class(cfg["model"])
    m, man = load_run_model(tdir, G.T0_TAG, Model); freeze(m); m.to(DEV).eval()
    return m, man


def read_rows(f, fp, ids):
    ids = np.asarray(ids, dtype=np.int64); order = np.argsort(ids); sidx = ids[order]; inv = np.argsort(order)
    gt = f["gt"][sidx][inv]; ms = f["ms"][sidx][inv]; pan = f["pan"][sidx][inv]; lpan = fp["lpan"][sidx][inv]
    return dict(gt=gt, ms=ms, pan=pan, lpan=lpan)


@torch.no_grad()
def compute_views(T0, margin, rows, contract, mp, rots=ROTS, with_e=True):
    """rows(dict numpy [n,C,H,W]) × rots → (q [n*len(rots)] float64, c0 [.,2], e [.] float32) — row 순서: base 순 × rot 순."""
    n = rows["pan"].shape[0]; qs, cs, es = [], [], []
    for r in rots:
        v = {k: EG.apply_view(EG.to_feeder_tensor(a, mp), contract["hflip"], contract["vflip"], r) for k, a in rows.items()}
        q, c0 = EG.teacher_q(T0.aligner, v["pan"], v["ms"], margin, chunk=n)
        if with_e:
            y = T0(v["pan"].to(DEV), v["ms"].to(DEV), v["lpan"].to(DEV))["y"].float()
            e = (y - v["gt"].to(DEV).float()).abs()[..., EG.ROI[0]:EG.ROI[1], EG.ROI[0]:EG.ROI[1]].mean(dim=(1, 2, 3)).cpu()
        else:
            e = torch.full((n,), float("nan"))
        qs.append(q); cs.append(c0); es.append(e)
    Q = torch.stack(qs, 1).reshape(-1); C = torch.stack(cs, 1).reshape(-1, 2); E = torch.stack(es, 1).reshape(-1)     # [n, R] → base 순 × rot 순
    return Q.numpy(), C.numpy(), E.numpy()


def build(a):
    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config))); contract, h5, h5pan = contract_from(cfg); mp = max_pixel_of(h5)
    T0, tman = load_T0(); margin = int(tman["aligner_view_margin"]); t_all = time.time()
    with h5py.File(h5) as f:
        N = int(f["pan"].shape[0]); bands = int(f["gt"].shape[1])
    cal_ids, cal_sha = calibration_ids(N); cal_set = set(cal_ids)
    pk = os.path.join(ROOT, G.CAL_PATH); pm = (json.load(open(pk)).get("patch_manifest") or {}) if os.path.exists(pk) else {}
    R = len(ROTS); V = N * R; index = np.repeat(np.arange(N, dtype=np.int32), R); rot = np.tile(np.array(ROTS, dtype=np.int8), N)
    q = np.zeros(V, dtype=np.float64); c0 = np.zeros((V, 2), dtype=np.float32); e = np.zeros(V, dtype=np.float32)
    print(f"[cue] {server()} · T0 {tman['run']}/{tman['tag']} (file {tman['file_sha256'][:16]}…, A hash {state_hash(T0.aligner)}, margin {margin}) · train {os.path.relpath(h5, ROOT)} N {N} × {R} state = {V} view · calibration {len(cal_ids)} base ({cal_sha}{' == pakd50' if pm.get('index_sha256_16') == cal_sha else ''})")
    bench = None
    with h5py.File(h5) as f, h5py.File(h5pan) as fp:
        for s in range(0, N, a.chunk):
            ids = list(range(s, min(N, s + a.chunk))); t0 = time.time(); rows = read_rows(f, fp, ids)
            Q, C, E = compute_views(T0, margin, rows, contract, mp); sl = slice(s * R, (s + len(ids)) * R); q[sl] = Q; c0[sl] = C; e[sl] = E
            if bench is None and (s + len(ids)) * R >= a.bench:
                bench = dict(views=(s + len(ids)) * R, seconds=time.time() - t_all); rate = bench["views"] / bench["seconds"]
                print(f"[cue] bench {bench['views']} view / {bench['seconds']:.1f}s → {rate:.0f} view/s · 전체 {V} view 예상 {V / rate / 60:.1f} min (§7.4 T_cue; 실제 총시간으로 교체)")
            if (s // a.chunk) % 10 == 0:
                print(f"[cue] {s + len(ids)}/{N} base · {time.time() - t0:.2f}s/chunk", flush=True)
    assert np.isfinite(q).all() and np.isfinite(e).all(), "q/e 에 비유한 값"
    cal_mask = np.isin(index, list(cal_set)); theta = EG.theta_from(q[cal_mask]); g = EG.gate_low_q(q, theta)
    ties_cal = int((q[cal_mask] == theta).sum()); ties_all = int((q == theta).sum())
    edges = EG.decile_edges(e[cal_mask]); strata, dec = EG.strata_of(e, rot, edges); g_sh, sh_stats = EG.shuffle_gate(g, strata, EG.QES_SEED)
    os.makedirs(os.path.join(ROOT, ASSET_DIR), exist_ok=True); npz_rel = f"{ASSET_DIR}/{a.name}.npz"; json_rel = f"{ASSET_DIR}/{a.name}.json"
    np.savez_compressed(os.path.join(ROOT, npz_rel), index=index, rot=rot, q=q, c0=c0, e_roi32=e, gate_low_q=g.astype(np.int8), gate_shuffle=g_sh.astype(np.int8), e_decile=dec, calib_mask=cal_mask)
    total_s = time.time() - t_all
    man = dict(name=a.name, version="cue_v1", plan=G.QEDGE9_PLAN, campaign_id=G.QEDGE9_CAMPAIGN_ID, computed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), server=server(), git_commit=git_commit(), device=str(DEV),
               gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), torch=torch.__version__,
               teacher=dict(run=tman["run"], tag=tman["tag"], file_sha256=tman["file_sha256"], tensors_sha256_16=tman["tensors_sha256_16"], aligner_state_hash=state_hash(T0.aligner), aligner_view_margin=margin,
                            selected_step=(tman.get("tag_meta") or {}).get("step"), adapter="predict_c(aligner, view(P, margin), view(bicubic↑4 M, margin)) — Teacher forward 와 동일; U 는 q 에 쓰지 않음"),
               dataset=dict(train=os.path.relpath(h5, ROOT), train_sha256=sha_cached(h5), train_pan=os.path.relpath(h5pan, ROOT), train_pan_sha256=sha_cached(h5pan), n_train=N, bands=bands, max_pixel=mp,
                            normalization="feeder.np2tensor: float64→float32, 2x/max_pixel−1 (float32)"),
               feeder_contract=dict(contract, states=R, rot_states=ROTS, note="hflip/vflip 는 조건 없이 적용(feeder), rot 만 random.randint(0,3) → 유효 상태 4 (§7.2)"),
               bank=dict(id=EG.BANK_ID, radii_hr=list(EG.RADII), angles_deg=list(EG.ANGLES_DEG), K=EG.K_PROBES, q_const_component=EG.q_const_component(), epsilon="(dy,dx) = (r sinθ, r cosθ) HR px; W(P,ε)[y,x] = P[y+ε_y, x+ε_x] bicubic/border",
                         q="(1/2K) Σ_k ‖ĉ_k + ε_k − ĉ_0‖_1 (HR px; λoff 없음; EPE/정규화 아님)"),
               calibration=dict(n_base=len(cal_ids), seed=CAL_SEED, index_sha256_16=cal_sha, n_views=int(cal_mask.sum()), matches_pakd50_calibration=(pm.get("index_sha256_16") == cal_sha), rule="θq = median q over calibration views; FR20/RR 미사용"),
               theta_q=float(theta), selection=dict(calib_frac=float(g[cal_mask].mean()), all_frac=float(g.mean()), ties_calib=ties_cal, ties_all=ties_all, rule="g = 1[q < θq]; q == θq → 0"),
               q_stats=dict(all={k: float(v) for k, v in zip(("p10", "p50", "p90", "mean"), (*np.quantile(q, [0.1, 0.5, 0.9]), q.mean()))}, calib={k: float(v) for k, v in zip(("p10", "p50", "p90", "mean"), (*np.quantile(q[cal_mask], [0.1, 0.5, 0.9]), q[cal_mask].mean()))},
                            by_rot={str(r): float(q[rot == r].mean()) for r in ROTS}),
               e_roi32=dict(roi=list(EG.ROI), decile_edges=edges, calib_p50=float(np.median(e[cal_mask])), note="T0 native 복원 중앙 ROI L1 (정규화 단위); 학습 loss 는 full-domain 그대로"),
               qes=dict(seed=EG.QES_SEED, bins=EG.QES_E_BINS, strata="e_T_roi32 decile(calibration edges) × aug state(rot)", **sh_stats, changed_frac=float((g_sh != g).mean()), rule="stratum 내 고정 permutation; active 수 보존; 원소<2·단일 라벨 stratum 은 그대로"),
               timing=dict(bench=bench, total_seconds=total_s, views_per_second=V / total_s, n_views=V, note="T_cue 실측 (§7.4) — 학습 시간표 밖의 일회성 비용"),
               npz=npz_rel, npz_sha256=sha256_file(os.path.join(ROOT, npz_rel)), npz_arrays="index int32 · rot int8 · q float64 · c0 float32[2] · e_roi32 float32 · gate_low_q/gate_shuffle int8 · e_decile int8 · calib_mask bool (row = base×4 + rot)")
    json.dump(man, open(os.path.join(ROOT, json_rel), "w"), indent=1, ensure_ascii=False)
    print(f"[cue] θq {theta:.6f} (calibration {len(cal_ids)} base × {R}) · 선택 비율 calib {man['selection']['calib_frac']:.4f} / 전체 {man['selection']['all_frac']:.4f} · 동일값 {ties_cal}/{ties_all} · q p50 {man['q_stats']['all']['p50']:.4f} (무반응 상수 {EG.q_const_component():.4f})")
    print(f"[cue] QES: strata {sh_stats['n_strata']} · 셔플 {sh_stats['n_shuffled']} · 단일라벨 {sh_stats['n_degenerate']} · 소형 {sh_stats['n_small']} · 라벨 변경 비율 {man['qes']['changed_frac']:.4f}")
    print(f"[cue] 총 {total_s / 60:.1f} min ({V / total_s:.0f} view/s) → {json_rel} + {npz_rel} ({os.path.getsize(os.path.join(ROOT, npz_rel)) / 2 ** 20:.2f} MB, sha {man['npz_sha256'][:16]}…) — git 에 넣어 s4/s5 로 전달")


def verify(a):
    man, z = EG.read_asset(a.asset); cfg = yaml.safe_load(open(os.path.join(ROOT, a.config))); contract, h5, h5pan = contract_from(cfg); mp = max_pixel_of(h5)
    T0, tman = load_T0(); margin = int(tman["aligner_view_margin"]); out = dict(server=server(), asset=a.asset, npz_sha256=man["npz_sha256"], theta_q=man["theta_q"], checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    out["teacher_aligner_hash_ok"] = (state_hash(T0.aligner) == man["teacher"]["aligner_state_hash"]); out["dataset_sha_ok"] = (sha_cached(h5) == man["dataset"]["train_sha256"])
    out["feeder_contract_ok"] = ({k: bool(v) for k, v in contract.items()} == {k: bool(man["feeder_contract"][k]) for k in contract})
    rng = np.random.RandomState(a.seed); N = int(man["dataset"]["n_train"]); base = np.sort(rng.choice(N, size=min(a.n, N), replace=False))
    with h5py.File(h5) as f, h5py.File(h5pan) as fp:
        rows = read_rows(f, fp, base)
    Q, C, E = compute_views(T0, margin, rows, contract, mp); rows_idx = (np.repeat(base, len(ROTS)) * len(ROTS) + np.tile(np.array(ROTS), len(base)))
    qc = z["q"][rows_idx]; gc = z["gate_low_q"][rows_idx]; g_new = EG.gate_low_q(Q, man["theta_q"]); dq = np.abs(Q - qc); de = np.abs(E - z["e_roi32"][rows_idx])
    out.update(n_views=int(len(rows_idx)), max_abs_dq=float(dq.max()), mean_abs_dq=float(dq.mean()), max_abs_de=float(de.max()), label_flips=int((g_new != gc).sum()), near_threshold=int((np.abs(qc - man["theta_q"]) < 1e-3).sum()),
               pass_=bool(out["teacher_aligner_hash_ok"] and out["dataset_sha_ok"] and out["feeder_contract_ok"] and float(dq.max()) < a.tol))
    os.makedirs(os.path.join(ROOT, CAMP), exist_ok=True); json.dump(out, open(os.path.join(ROOT, CAMP, f"verify_{server()}.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[cue] verify {server()}: T0 A hash {'OK' if out['teacher_aligner_hash_ok'] else 'MISMATCH'} · dataset sha {'OK' if out['dataset_sha_ok'] else 'MISMATCH'} · feeder 계약 {'OK' if out['feeder_contract_ok'] else 'MISMATCH'} · {out['n_views']} view 재계산: max|Δq| {out['max_abs_dq']:.2e} mean {out['mean_abs_dq']:.2e} · max|Δe| {out['max_abs_de']:.2e} · gate 라벨 차이 {out['label_flips']} · θq 근처(<1e-3) {out['near_threshold']} → {'PASS' if out['pass_'] else 'FAIL'} (허용 |Δq| < {a.tol:g})")
    sys.exit(0 if out["pass_"] else 1)


def pilot(a):
    man, z = EG.read_asset(a.asset); cfg = yaml.safe_load(open(os.path.join(ROOT, a.config))); contract, h5, h5pan = contract_from(cfg); mp = max_pixel_of(h5)
    rd = os.path.join(ROOT, "work_dir", a.run); lm = os.path.join(rd, f"{a.tag}_meta.json")
    if a.tag == "last" and not (os.path.exists(lm) and json.load(open(lm)).get("step") == 50000):
        sys.exit(f"!! pilot {a.run} 의 exact-50K last 가 없다 (§6.1: s4 W104 J0 S1234 exact50K)")
    Model = import_class(cfg["model"]); pm, pman = load_run_model(rd, a.tag, Model); freeze(pm); pm.to(DEV).eval()
    if int(pman["width"]) != 104:
        sys.exit(f"!! pilot 골격 W{pman['width']} — QEC 의 c_E 는 W104·D121 pilot 에서 (§6.1)")
    N = int(man["dataset"]["n_train"]); cal_ids, cal_sha = calibration_ids(N)
    if cal_sha != man["calibration"]["index_sha256_16"]:
        sys.exit("!! calibration id 가 cue 자산과 다르다")
    base = np.array(range(N), dtype=np.int64) if a.all_views else np.array(cal_ids, dtype=np.int64); Es = []
    with torch.no_grad(), h5py.File(h5) as f, h5py.File(h5pan) as fp:
        for s in range(0, len(base), a.chunk):
            ids = base[s:s + a.chunk]; rows = read_rows(f, fp, ids); E_r = []
            for r in ROTS:
                v = {k: EG.apply_view(EG.to_feeder_tensor(x, mp), contract["hflip"], contract["vflip"], r) for k, x in rows.items()}
                y = pm(v["pan"].to(DEV), v["ms"].to(DEV), v["lpan"].to(DEV))["y"].float(); E_r.append(EG.edge_per_sample(y, v["gt"].to(DEV).float()).double().cpu())
            Es.append(torch.stack(E_r, 1).reshape(-1))
    E = torch.cat(Es).numpy(); rows_idx = (np.repeat(base, len(ROTS)) * len(ROTS) + np.tile(np.array(ROTS), len(base))); g = z["gate_low_q"][rows_idx].astype(np.float64)
    den = float(E.sum()); num = float((g * E).sum())
    if not (np.isfinite(den) and den > 0):
        sys.exit("!! c_E 분모 퇴화 (Σ E_pilot ≤ 0) — calibration 실패로 처리, 임의 상수를 채우지 않는다 (§6.1)")
    cE = num / den
    out = dict(c_E=cE, sum_gE=num, sum_E=den, n_views=int(len(E)), views=("all" if a.all_views else "calibration"), gate_frac=float(g.mean()), mean_E_gated=float((g * E).sum() / max(g.sum(), 1)), mean_E_all=float(E.mean()),
               pilot_run=a.run, pilot_tag=a.tag, pilot_sha256_16=pman["tensors_sha256_16"], pilot_file_sha256=pman["file_sha256"], pilot_step=(pman.get("tag_meta") or {}).get("step"), pilot_width=pman["width"], pilot_depth=pman["depth"],
               cue=a.asset, cue_npz_sha256=man["npz_sha256"], theta_q=man["theta_q"], computed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), server=server(),
               rule="c_E = Σ_i g_i E_pilot(i) / Σ_i E_pilot(i) (train calibration view, 고정 pilot; §6.1) — 고정 pilot 에서 edge loss 평균을 맞춘 대조이지 전 학습 gradient 일치 대조가 아니다",
               note="0.5 로 두지 않는다: 선택 patch 의 edge 오차가 크면 c_E > 0.5")
    os.makedirs(os.path.join(ROOT, CAMP), exist_ok=True); json.dump(out, open(os.path.join(ROOT, G.QEDGE9_CE_FILE), "w"), indent=1, ensure_ascii=False)
    print(f"[cue] pilot {a.run}/{a.tag} (W{pman['width']}, step {out['pilot_step']}, sha {pman['tensors_sha256_16']}): c_E = {cE:.6f} (gate 비율 {out['gate_frac']:.4f}; E gated 평균 {out['mean_E_gated']:.5f} vs 전체 {out['mean_E_all']:.5f}; {out['n_views']} view) → {G.QEDGE9_CE_FILE}")


def status(a):
    j = os.path.join(ROOT, a.asset)
    if not os.path.exists(j):
        print(f"[cue] 자산 없음: {a.asset} (build 필요)"); return
    man = json.load(open(j)); ok = os.path.exists(os.path.join(ROOT, man["npz"])) and sha256_file(os.path.join(ROOT, man["npz"])) == man["npz_sha256"]
    print(f"[cue] {a.asset}: θq {man['theta_q']:.6f} · 선택 calib {man['selection']['calib_frac']:.4f} / 전체 {man['selection']['all_frac']:.4f} · view {man['timing']['n_views']} · T0 A {man['teacher']['aligner_state_hash']} · npz {'OK' if ok else 'BAD'} ({man['npz_sha256'][:16]}…) · 만든 곳 {man['server']} {man['computed_at']}")
    c = os.path.join(ROOT, G.QEDGE9_CE_FILE)
    print(f"[cue] c_E: " + (f"{json.load(open(c))['c_E']:.6f} ({json.load(open(c))['pilot_run']})" if os.path.exists(c) else f"없음 ({G.QEDGE9_CE_FILE}; s4 에서 pilot)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=("build", "verify", "pilot", "status"))
    ap.add_argument("--config", default=DEFAULT_CONFIG); ap.add_argument("--name", default="cue_T0_AXIS16_v1"); ap.add_argument("--asset", default=G.QEDGE9_CUE_ASSET); ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--bench", type=int, default=256)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tol", type=float, default=1e-4); ap.add_argument("--run", default=None); ap.add_argument("--tag", default="last"); ap.add_argument("--all-views", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pilot" and not a.run:
        sys.exit("--run <pilot run> 필요")
    {"build": build, "verify": verify, "pilot": pilot, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
