"""EQREC4 공통: 경로·ledger·모델 registry·데이터 분할(A/B/C/D)·probe bank·ROI·배치 유틸. 계획 §2–§4·§16."""
import csv, gzip, hashlib, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F, h5py, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from main import import_class                                            # noqa: E402
from pa.model import PAModel                                             # noqa: E402
from pa.aligner import PANGlobalAligner                                  # noqa: E402
from pa.offset import predict_c, aligner_margin, sample_offsets, offset_loss   # noqa: E402,F401
from pa.warp import warp_pan                                             # noqa: E402
from pa.losses import scharr as scharr_t                                 # noqa: E402
from kdv.teacher_assets import sha256_file, tensors_sha, state_hash      # noqa: E402,F401

PLAN = "research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md"
_sv = os.path.join(ROOT, "gspread", "server.txt"); SERVER = (open(_sv).read().strip() if os.path.exists(_sv) else "s1") or "s1"     # 실행 서버 (계획은 s1 기준; s3 등에서는 같은 registry bundle 로 돈다)
CAMPAIGN_ID = f"EQREC4_{SERVER.upper()}_v1"
CAMP = os.path.join(ROOT, "work_dir", f"_eqrec4_{SERVER}_campaign"); FIG = os.path.join(CAMP, "figures")     # 서버별 별도 root (§6-8: 다른 캠페인·다른 서버 결과를 덮어쓰지 않는다)


def localize_path(p):
    """run config 에 박힌 s1 절대 dataroot 를 이 저장소의 data/ 경로로 (setup_paths.sh 는 config/*.yaml 만 고치고 work_dir/<run>/meta 는 안 고친다)."""
    if not p or os.path.exists(p) or "/data/" not in p:
        return p
    return os.path.join(ROOT, "data", p.split("/data/", 1)[1])


def localize_cfg(cfg):
    cfg = dict(cfg)
    for k in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
        if k in cfg and isinstance(cfg[k], dict) and "dataroot" in cfg[k]:
            cfg[k] = dict(cfg[k], dataroot=localize_path(cfg[k]["dataroot"]))
    return cfg
SPLIT_SEED = 314159; BLOCK = 32                                          # source group proxy = 연속 index 32개 (원본 scene/strip id 없음 → source_group_unknown)
SPLIT_TARGET = dict(A=512, B=2048, C=512, D=1024)                        # calibration / adaptation fit / policy validation / locked holdout (§3.2)
MAX_PIXEL = 2047.0
ROI_MARGIN = dict(native64=16, rr256=32, fr512=96)                       # §4.3: 64→32², 256→192², 512→V96 320²
RADII = (0.25, 0.5, 1.0, 2.0); BANK_ANGLES = dict(A=(0, 90, 180, 270), B=(45, 135, 225, 315))   # §4.2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUADS = ("EdCd", "EdCu", "EuCd", "EuCu")                                 # E↓C↓ · E↓C↑ · E↑C↓ · E↑C↑
DETAIL_PER_QUAD = 64; GRAD_PER_QUAD = 32; LANDSCAPE_PER_QUAD = 16

# ---------------------------------------------------------------- registry (§2.2)
RUNS = {
    ("L1E4", 1234): "PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1", ("L1E4", 7777): "PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1", ("L1E4", 2025): "PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1",
    ("L000", 1234): "NF16_P2_W112_D123_WV3_S1234_N2LAST_v1", ("L000", 7777): "PALS24_L000_W112_D123_WV3_S7777_N2LAST_R200_v1", ("L000", 2025): "PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1",
    ("L3E4", 1234): "PALSV18_L3E4_W112_D123_WV3_S1234_N2LAST_R200_v1", ("L3E4", 7777): "PALSV18_L3E4_W112_D123_WV3_S7777_N2LAST_R200_v1", ("L3E4", 2025): "PALSV18_L3E4_W112_D123_WV3_S2025_N2LAST_R200_v1",
    ("P0", 1234): "NF16_P0_W112_D123_WV3_S1234_N2LAST_v1", ("P0", 7777): "PALS24_CTRLP0_W112_D123_WV3_S7777_N2LAST_R200_v1", ("P0", 2025): "PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1",
    ("N2", 2025): "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT", ("L1E2", 1234): "NF16_P3_W112_D123_WV3_S1234_N2LAST_v1",
}
# (family, seed, tag) — tag: best_raw (= 저장소 alias best_hqnr) / last (exact 50K) / first_ge_20k (candidates 중 ≥20000 최초; §2.3)
CORE = [("L1E4", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("L000", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")]
EXTRA = [("L3E4", s, "best_raw") for s in (1234, 7777, 2025)] + [("P0", s, t) for s in (1234, 7777, 2025) for t in ("best_raw", "last")] + [("N2", 2025, "last"), ("L1E2", 1234, "best_raw"), ("L1E2", 1234, "last")]
PRIMARY = ("L1E4", 2025, "best_raw")
PAIRS = dict(A=dict(T=("L1E4", 2025, "best_raw"), S=("L1E4", 1234, "first_ge_20k")), B=dict(T=("L1E4", 1234, "best_raw"), S=("L1E4", 7777, "first_ge_20k")))


def mkey(fam, seed, tag):
    return f"{fam}_S{seed}_{tag}"


def run_of(fam, seed):
    return RUNS[(fam, seed)]


def ckpt_dir(fam, seed, tag):
    """실제 checkpoint 폴더 (없으면 None). best_raw → best_hqnr (저장소 alias), last → last, first_ge_20k → candidates/step-<min ≥ 20000>."""
    wd = os.path.join(ROOT, "work_dir", run_of(fam, seed))
    if tag == "best_raw":
        d = os.path.join(wd, "best_hqnr")
    elif tag == "last":
        d = os.path.join(wd, "last")
    elif tag == "first_ge_20k":
        cd = os.path.join(wd, "candidates")
        steps = sorted(int(x.split("-")[1]) for x in os.listdir(cd) if x.startswith("step-")) if os.path.isdir(cd) else []
        ok = [s for s in steps if s >= 20000]
        d = os.path.join(cd, f"step-{ok[0]}") if ok else None
    elif tag.startswith("cand:"):
        d = os.path.join(wd, "candidates", f"step-{int(tag[5:])}")
    else:
        raise ValueError(tag)
    return d if (d and os.path.exists(os.path.join(d, "model.safetensors"))) else None


def ckpt_step(fam, seed, tag):
    d = ckpt_dir(fam, seed, tag)
    if d is None:
        return None
    if os.path.basename(os.path.dirname(d)) == "candidates":
        return int(os.path.basename(d).split("-")[1])
    mp = d + "_meta.json"
    return json.load(open(mp)).get("step") if os.path.exists(mp) else None


class Loaded:
    def __init__(self, fam, seed, tag, m, cfg, R, mg, path):
        self.fam, self.seed, self.tag, self.m, self.cfg, self.R, self.mg, self.path = fam, seed, tag, m, cfg, R, mg, path
        self.key = mkey(fam, seed, tag); self.run = run_of(fam, seed); self.step = ckpt_step(fam, seed, tag)
        self.has_aligner = m.aligner is not None; self.file_sha = sha256_file(os.path.join(path, "model.safetensors"))
        self.aligner_sha = state_hash(m.aligner) if m.aligner is not None else None; self.unet_sha = state_hash(m.backbone)

    def manifest(self):
        return dict(model_key=self.key, family=self.fam, seed=self.seed, checkpoint_kind=self.tag, run_id=self.run, actual_update=self.step, checkpoint_dir=os.path.relpath(self.path, ROOT),
                    model_hash=self.file_sha, aligner_hash=self.aligner_sha, unet_hash=self.unet_sha, has_aligner=self.has_aligner, radius_hr=self.R, view_margin=self.mg)


def load_model(fam, seed, tag, dev=DEV):
    """run config 대로 skeleton 을 만들고 strict load (po10_diag.load_run 과 같은 규약; A-ID 는 aligner 없음·sampler 없음)."""
    from safetensors.torch import load_file
    d = ckpt_dir(fam, seed, tag)
    if d is None:
        raise FileNotFoundError(f"{mkey(fam, seed, tag)}: checkpoint 없음")
    wd = os.path.join(ROOT, "work_dir", run_of(fam, seed)); cfg = localize_cfg(yaml.safe_load(open(os.path.join(wd, "meta", "config.yaml")))); tr = cfg.get("trainer")
    if tr == "kdv":
        from kdv.teacher_assets import skeleton_from_cfg
        m, info = skeleton_from_cfg(cfg, import_class(cfg["model"])); k = cfg.get("kdv") or {}; R = float((k.get("corruption") or {}).get("radius_hr", 0) or 2.0); mg = info["margin"]
    elif tr == "po":
        R = float((cfg.get("po") or {}).get("radius_hr", 1.0)); mg = aligner_margin(R); m = PAModel(import_class(cfg["model"])(**cfg["model_args"]), PANGlobalAligner(int(cfg["num_bands"])), aligner_margin=mg)
    else:
        raise ValueError(f"trainer {tr}")
    m.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=True)
    return Loaded(fam, seed, tag, m.to(dev).eval().requires_grad_(False), cfg, R, mg, d)


# ---------------------------------------------------------------- ledger / io
def ensure_dirs():
    os.makedirs(CAMP, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def ledger(stage, status, wall_s, note="", **kw):
    ensure_dirs()
    rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, status=status, wall_hours=round(wall_s / 3600.0, 4), gpu=torch.cuda.is_available(),
               gpu_peak_mem_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None), note=note, **kw)
    with open(os.path.join(CAMP, "time_ledger.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


class Stage:
    def __init__(self, name, note=""):
        self.name, self.note = name, note

    def __enter__(self):
        self.t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ledger(self.name, "start", 0.0, self.note); print(f"[eqrec4] {self.name} start {time.strftime('%H:%M:%S')}", flush=True); return self

    def __exit__(self, et, ev, tb):
        ledger(self.name, "done" if et is None else "error", time.time() - self.t0, self.note if et is None else f"{self.note} !! {ev!r}"[:400])
        print(f"[eqrec4] {self.name} {'done' if et is None else 'ERROR'} {(time.time() - self.t0) / 60:.1f} min", flush=True)


def write_csv(path, rows, gz=False):
    if not rows:
        open(path, "w").close(); return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    op = gzip.open(path, "wt", newline="") if gz else open(path, "w", newline="")
    with op as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def dump_json(path, obj):
    json.dump(obj, open(path, "w"), indent=1, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


# ---------------------------------------------------------------- data (§3)
def data_paths():
    cfg = localize_cfg(yaml.safe_load(open(os.path.join(ROOT, "work_dir", run_of(*PRIMARY[:2]), "meta", "config.yaml"))))
    tr = cfg["train_feeder_args"]["dataroot"]
    return dict(train=tr, train_pan=tr.replace(".h5", "_pan.h5"), rr=cfg["test_reduced_feeder_args"]["dataroot"], fr=cfg["test_full_feeder_args"]["dataroot"], cfg=cfg)


def make_manifest():
    """4,096 patch 를 연속 index 32 블록 단위로 A/B/C/D 에 배정 (블록 = source group proxy; 인접 patch 가 다른 분할로 가지 않게). seed 314159. 이미 있으면 재사용."""
    ensure_dirs(); p = os.path.join(CAMP, "data_manifest.csv")
    if os.path.exists(p):
        return read_csv(p)
    dp = data_paths()
    with h5py.File(dp["train"]) as f:
        N = f["pan"].shape[0]
    nb = N // BLOCK; rng = np.random.RandomState(SPLIT_SEED); order = rng.permutation(nb)
    need = {k: v // BLOCK for k, v in SPLIT_TARGET.items()}; rows = []; pos = 0
    for role in ("A", "B", "C", "D"):
        for b in order[pos:pos + need[role]]:
            for i in range(b * BLOCK, (b + 1) * BLOCK):
                rows.append(dict(sample_id=int(i), source_group_id=f"blk{int(b):03d}", split_role=role, scale_hw="64x64", seen_in_pretraining=True, source_group_note="source_group_unknown: index-block proxy (32 consecutive train patches)"))
        pos += need[role]
    write_csv(p, rows); dump_json(os.path.join(CAMP, "data_manifest_meta.json"), dict(train_h5=dp["train"], train_sha256=sha256_file(dp["train"]), n_total=int(N), block=BLOCK, n_blocks=int(nb), seed=SPLIT_SEED, targets=SPLIT_TARGET,
                                                                                    note="patches were seen in pretraining (all registry models trained on the full train split); A/B/C/D separate roles of this campaign only (§3.2)"))
    return read_csv(p)


_PATCH_CACHE = {}


def load_patches(ids):
    """train h5 → (gt, ms, lpan, pan) float32 정규화([-1,1]) CPU 텐서, ids 순서 유지."""
    ids = [int(i) for i in ids]; key = tuple(ids)
    if key in _PATCH_CACHE:
        return _PATCH_CACHE[key]
    dp = data_paths(); order = np.argsort(ids); sidx = np.array(ids)[order]
    with h5py.File(dp["train"]) as f:
        gt = np.asarray(f["gt"][sidx], dtype=np.float32); ms = np.asarray(f["ms"][sidx], dtype=np.float32); pan = np.asarray(f["pan"][sidx], dtype=np.float32)
    with h5py.File(dp["train_pan"]) as f:
        lpan = np.asarray(f["lpan"][sidx], dtype=np.float32)
    inv = np.argsort(order); out = tuple(torch.from_numpy(a[inv]) * 2 / MAX_PIXEL - 1 for a in (gt, ms, lpan, pan))
    if len(ids) <= 4096:
        _PATCH_CACHE[key] = out
    return out


def feeders():
    """RR / FR Feeder (다른 진단 도구와 같은 경로) + 원시 DN 배열 (FR 평가용)."""
    dp = data_paths(); cfg = dp["cfg"]; Feeder = import_class(cfg["feeder"])
    rr = Feeder(**cfg["test_reduced_feeder_args"]); fr = Feeder(**cfg["test_full_feeder_args"])
    with h5py.File(dp["fr"]) as f:
        lms_raw = np.asarray(f["lms"], dtype=np.float64); pan_raw = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    return dict(rr=rr, fr=fr, lms_raw=lms_raw, pan_raw=pan_raw, cfg=cfg, mp=float(fr.max_pixel))


# ---------------------------------------------------------------- probes (§4.2), ROI (§4.3)
def probe_bank(bank):
    out = []
    for r in RADII:
        for th in BANK_ANGLES[bank]:
            a = math.radians(th); out.append(dict(probe_id=f"{bank}_r{r}_t{th}", bank=bank, r=r, theta_deg=th, ey=round(r * math.sin(a), 12), ex=round(r * math.cos(a), 12)))
    return out


def probe_manifest():
    ensure_dirs()
    for b in ("A", "B"):
        dump_json(os.path.join(CAMP, f"probe_bank_{b}.json"), dict(bank=b, epsilon="(dy,dx) = (r sinθ, r cosθ) HR px; W(P,ε)[y,x] = P[y+ε_y, x+ε_x] bicubic/border", radii_hr=list(RADII), angles_deg=list(BANK_ANGLES[b]),
                                                                    q_const_component=float(np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in probe_bank(b)])), n=len(probe_bank(b)), identity_counted=False, probes=probe_bank(b)))
    dump_json(os.path.join(CAMP, "roi_contract.json"), dict(native64=dict(margin=16, roi="[16:48]²", note="L1/edge only"), rr256=dict(margin=32, roi="[32:224]²"), fr512=dict(margin=96, roi="V96 [96:416]² (pa.evalviews STRESS_MARGIN)"),
                                                            rule="fixed before outcomes; native and stress use the same ROI; invalid_support reported, corrections never clamped"))


def roi_slice(scale):
    m = ROI_MARGIN[scale]; return (slice(m, -m), slice(m, -m))


def l1_roi(y, gt, scale, per_band=False):
    """[B,C,H,W] → 공통 ROI 안 sample 별 L1 (정규화 단위; DN = ×MAX_PIXEL/2)."""
    sy, sx = roi_slice(scale); d = (y - gt).abs()[..., sy, sx]
    return d.mean(dim=(1, 2, 3)) if not per_band else d.mean(dim=(2, 3))


def edge_l1_roi(y, gt, scale):
    sy, sx = roi_slice(scale); gx, gy = scharr_t(y.float()); hx, hy = scharr_t(gt.float())
    return (0.5 * ((gx - hx).abs() + (gy - hy).abs()))[..., sy, sx].mean(dim=(1, 2, 3))


def edge_mask_gt(gt, top=0.30):
    """GT 의 Scharr 크기 상위 30% (band 별) mask [B,C,H,W]."""
    hx, hy = scharr_t(gt.float()); mag = (hx ** 2 + hy ** 2).sqrt(); B, C = mag.shape[:2]
    thr = mag.flatten(2).quantile(1 - top, dim=2).view(B, C, 1, 1); return mag >= thr


@torch.no_grad()
def two_stage_ok(H, W, eps, c, margin):
    """두 번 warp(ε 뒤 c) 의 sampling support 가 ROI 안에서 관측만 읽는가 — |ε|∞ + |c|∞ + 2 taps ≤ margin (보수적)."""
    return bool(max(abs(float(eps[0])), abs(float(eps[1]))) + max(abs(float(c[0])), abs(float(c[1]))) + 4 <= margin)


# ---------------------------------------------------------------- aligner probes (§4.1)
@torch.no_grad()
def probe_responses(L, pan, ms, banks=("A", "B"), kernel="bicubic", padding="border", pan_mode="normal", ms_mode="own", chunk=256):
    """pan [N,1,H,W], ms [N,8,h,w] (정규화, CPU) → dict(c0 [N,2], per bank: records list[dict], q [N], epe [N], resp [N,K,2]) — aligner 만 호출.
    pan_mode: normal | affine (1.3·P+0.1) · ms_mode: own | swap (다음 sample 의 MS) | const (band 별 공간 평균)."""
    from tools.po10_diag import warp_generic
    m = L.m; mg = L.mg; N = pan.shape[0]; probes = {b: probe_bank(b) for b in banks}
    c0_all = torch.zeros(N, 2); resp = {b: torch.zeros(N, len(probes[b]), 2) for b in banks}
    for s in range(0, N, chunk):
        p = pan[s:s + chunk].to(DEV); q = ms[s:s + chunk].to(DEV)
        if pan_mode == "affine":
            p = 1.3 * p + 0.1
        if ms_mode == "swap":
            q = torch.roll(q, 1, dims=0)
        elif ms_mode == "const":
            q = q.mean(dim=(2, 3), keepdim=True).expand_as(q).contiguous()
        mb = F.interpolate(q, scale_factor=4, mode="bicubic"); c0 = predict_c(m.aligner, p, mb, mg); c0_all[s:s + chunk] = c0.cpu()
        for b in banks:
            for k, pr in enumerate(probes[b]):
                e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV).expand(p.shape[0], 2)
                pe = warp_pan(p, e) if (kernel == "bicubic" and padding == "border") else warp_generic(p, e, mode=kernel, padding=padding)
                ce = predict_c(m.aligner, pe, mb, mg); resp[b][s:s + chunk, k] = (ce - c0).cpu()
    out = dict(c0=c0_all)
    for b in banks:
        E = torch.tensor([[pr["ey"], pr["ex"]] for pr in probes[b]]); r = resp[b] + E[None]          # r = ĉε − ĉ0 + ε (이상 0)
        out[b] = dict(resp=resp[b], resid=r, q=r.abs().mean(dim=(1, 2)), epe=r.norm(dim=2).mean(dim=1), probes=probes[b],
                      q_by_radius={str(rad): r[:, [k for k, pr in enumerate(probes[b]) if pr["r"] == rad]].abs().mean(dim=(1, 2)) for rad in RADII})
    return out


def fit_response(resp, probes):
    """sample 별 ĉε−ĉ0 = B ε + b 최소제곱 (resp [K,2]) → B(2×2), b, rmse, sv."""
    E = np.array([[p["ey"], p["ex"], 1.0] for p in probes]); Q = np.asarray(resp, dtype=np.float64); coef, *_ = np.linalg.lstsq(E, Q, rcond=None); Bm = coef[:2].T; res = Q - E @ coef
    return dict(B_yy=float(Bm[0, 0]), B_yx=float(Bm[0, 1]), B_xy=float(Bm[1, 0]), B_xx=float(Bm[1, 1]), b_y=float(coef[2, 0]), b_x=float(coef[2, 1]), fit_rmse=float(np.sqrt((res ** 2).mean())),
                sv_max=float(np.linalg.svd(Bm, compute_uv=False)[0]), sv_min=float(np.linalg.svd(Bm, compute_uv=False)[1]))


@torch.no_grad()
def native_forward(L, pan, ms, lpan, chunk=64, delta_override=None, aligner_enabled=True):
    """[N,…] CPU → y [N,8,H,W] CPU float32, delta [N,2]. delta_override: [N,2] 텐서(CPU) 면 sample 별 치환."""
    ys, ds = [], []
    for s in range(0, pan.shape[0], chunk):
        p, q, l = pan[s:s + chunk].to(DEV), ms[s:s + chunk].to(DEV), lpan[s:s + chunk].to(DEV)
        kw = {}
        if delta_override is not None:
            kw["delta_override"] = delta_override[s:s + chunk].to(DEV).float()
        elif not aligner_enabled:
            kw["aligner_enabled"] = False
        o = L.m(p, q, l, **kw); ys.append(o["y"].float().cpu()); ds.append(o["delta"].float().cpu())
    return torch.cat(ys), torch.cat(ds)


def input_stats(pan, ms):
    """입력 통계 (§7): PAN 범위·contrast·Scharr energy·방향 다양성, MS band variance, PAN–MS 구조 상관."""
    gx, gy = scharr_t(pan.float()); mag = (gx ** 2 + gy ** 2).sqrt(); ang = torch.atan2(gy, gx)
    w = mag.flatten(1); ent = []; idx = ((ang.flatten(1) + math.pi) / (2 * math.pi) * 8).long().clamp(0, 7)
    for i in range(pan.shape[0]):
        h = torch.bincount(idx[i], weights=w[i], minlength=8).float(); pp = h / (h.sum() + 1e-12); ent.append(float(-(pp * (pp + 1e-12).log()).sum()))
    mb = F.interpolate(ms, scale_factor=4, mode="bicubic").mean(1, keepdim=True); px = pan - pan.mean(dim=(1, 2, 3), keepdim=True); mx = mb - mb.mean(dim=(1, 2, 3), keepdim=True)
    corr = (px * mx).sum(dim=(1, 2, 3)) / ((px ** 2).sum(dim=(1, 2, 3)).sqrt() * (mx ** 2).sum(dim=(1, 2, 3)).sqrt() + 1e-12)
    return dict(pan_mean=pan.mean(dim=(1, 2, 3)), pan_std=pan.std(dim=(1, 2, 3)), pan_range=(pan.amax(dim=(1, 2, 3)) - pan.amin(dim=(1, 2, 3))), pan_scharr_energy=(mag ** 2).mean(dim=(1, 2, 3)),
                pan_orient_entropy=torch.tensor(ent), ms_band_var_mean=ms.var(dim=(2, 3)).mean(1), pan_msmean_corr=corr)


def texture_tertile(vals):
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3]); return np.where(vals <= q1, "low", np.where(vals <= q2, "mid", "high"))


def stratified_pick(df, n, seed, strata_cols=("source_group_id", "texture")):
    """층화 무작위 (source × texture 라운드로빈) — 손실이 유리한 예만 고르지 않는다 (§7)."""
    rng = np.random.RandomState(seed); df = df.copy(); df["_r"] = rng.rand(len(df)); df = df.sort_values("_r")
    groups = [g for _, g in df.groupby(list(strata_cols), sort=False)]; rng.shuffle(groups); out = []; i = 0
    while len(out) < n and any(len(g) > i for g in groups):
        for g in groups:
            if len(g) > i and len(out) < n:
                out.append(g.iloc[i])
        i += 1
    import pandas as pd
    return pd.DataFrame(out) if out else df.iloc[:0]


def block_bootstrap(values, groups, fn=np.mean, n=2000, seed=0):
    """source-group block bootstrap 95% CI (§15.2). groups 가 1개 이하면 None."""
    values = np.asarray(values, dtype=np.float64); groups = np.asarray(groups); ug = np.unique(groups)
    if len(ug) < 2 or len(values) == 0:
        return dict(point=float(fn(values)) if len(values) else None, ci95=None, n_groups=int(len(ug)), note="insufficient groups for block bootstrap")
    rng = np.random.RandomState(seed); idx_by = {g: np.where(groups == g)[0] for g in ug}; stats = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True); sel = np.concatenate([idx_by[g] for g in pick]); stats.append(fn(values[sel]))
    return dict(point=float(fn(values)), ci95=[float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))], n_groups=int(len(ug)), n=int(len(values)))


def spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return dict(rho=None, p=None, n=int(ok.sum()))
    r = spearmanr(x[ok], y[ok]); return dict(rho=float(r.correlation), p=float(r.pvalue), n=int(ok.sum()))


def spearman_boot(x, y, groups, n=2000, seed=0):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); groups = np.asarray(groups); ok = np.isfinite(x) & np.isfinite(y); x, y, groups = x[ok], y[ok], groups[ok]
    base = spearman(x, y); ug = np.unique(groups)
    if len(ug) < 2 or base["rho"] is None:
        return dict(base, ci95=None, n_groups=int(len(ug)))
    rng = np.random.RandomState(seed); idx_by = {g: np.where(groups == g)[0] for g in ug}; st = []
    for _ in range(n):
        sel = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug), replace=True)]); st.append(spearmanr(x[sel], y[sel]).correlation)
    st = np.array(st, float); st = st[np.isfinite(st)]
    return dict(base, ci95=[float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], n_groups=int(len(ug)))
