"""SMEC12 공통 — 센서/자산 registry(§2–§3), panel(§3.3–3.4), probe A/B/S(§4.1), e/q/label(§4.2), adapter(§20.1: load_asset · read_sample · predict_shift · reconstruct_at · measure_offset_bank), ledger/io.
기존 검증 코드 재사용: tools/eqrec4/common.py (warp_pan · predict_c · ROI L1 · Scharr · bootstrap · Spearman), kdv/teacher_assets (strict load), feeders (RR/FR scene)."""
import copy, hashlib, json, math, os, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F, h5py, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from tools.eqrec4 import common as E                                      # noqa: E402
from tools import gen_smec12_bootstrap as B                               # noqa: E402
from kdv.teacher_assets import load_run_model, load_state, sha256_file, tensors_sha, state_hash   # noqa: E402
from main import import_class                                             # noqa: E402

PLAN = B.PLAN; NOTE = B.NOTE
SERVER = open(os.path.join(ROOT, "gspread", "server.txt")).read().strip() if os.path.exists(os.path.join(ROOT, "gspread", "server.txt")) else "s1"
CAMPAIGN_ID = "SMEC12_v1"; CAMP = os.path.join(ROOT, "work_dir", f"_smec12_{SERVER}_campaign")
for _d in ("manifests", "raw", "figures", "analysis", "models_prepared"):
    pass
DEV = E.DEV
MANIFEST_SEED = 271828; BLOCK = 32                                        # §20.2 sampling.manifest_seed; source group proxy = 연속 index 32 (원 scene id 없음 → source_group_unknown)
PANEL = dict(CAL=512, DISC=1536, CONF=1024, FIT=1024)                      # §3.4 model atlas (≤4,096/센서)
DETAIL_PER_QUAD = 32; DETAIL_PARTS = ("DISC", "CONF")                      # §3.4 상세 intervention: 네 집단당 ≤32 × DISC/CONF
TILES_PER_SCENE = 16; TILE = 64                                            # §3.4 RR/FR tile (겹치지 않는 64², deterministic)
RADII_AB = (0.25, 0.5, 1.0, 2.0); RADII_S = (0.125, 0.25, 0.5, 1.0); ANG = dict(A=(0, 90, 180, 270), B=(45, 135, 225, 315), S=(0, 45, 90, 135, 180, 225, 270, 315))
ROI_MARGIN = dict(native64=16, rr256=32, fr512=96); QUADS = ("EdCd", "EdCu", "EuCd", "EuCu"); BOUNDARY_FRAC = 0.10; Q_UNRESOLVED_X = 10
ROLES = ("P0", "DONN2", "L000", "L1E4", "L1E4REP")
SENSORS = dict(B.SENSORS); SENSORS["wv2"] = dict(S="WV2", bands=8, max_pixel=2047.0, train=None, valid=None, rr="data/PanCollection/WV2/reduced_examples_h5/test_wv2_multiExm1.h5", fr="data/PanCollection/WV2/full_examples_mat20/test_wv2_OrigScale_mat20.h5",
                                                 backbone_params=2.6589, n_train=0, data_note="공식 학습셋 없음 — WV3 모델의 zero-shot RR/FR + tile 만 (test 학습 금지 §3.1)")
for _s in SENSORS:
    SENSORS[_s]["role"] = "zero_shot" if _s == "wv2" else "in_domain"
# 기존 WV3 자산 (§2.2 대응; NF16/PO10/PALS24 의 검증 run) — 새로 학습하지 않는다 (§2.3 규칙 2·9)
WV3_ASSETS = {("P0", 1234): "NF16_P0_W112_D123_WV3_S1234_N2LAST_v1", ("P0", 2025): "PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1", ("P0", 7777): "PALS24_CTRLP0_W112_D123_WV3_S7777_N2LAST_R200_v1",
              ("DONN2", 2025): "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT", ("L000", 1234): "NF16_P2_W112_D123_WV3_S1234_N2LAST_v1", ("L000", 2025): "PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1", ("L000", 7777): "PALS24_L000_W112_D123_WV3_S7777_N2LAST_R200_v1",
              ("L1E4", 2025): "PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1", ("L1E4REP", 1234): "PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1", ("L1E4REP", 7777): "PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1"}
PRIMARY = dict(wv3=("L1E4", 2025), qb=("L1E4", 2025), gf2=("L1E4", 2025))


# ---------------------------------------------------------------- io / ledger
def ensure_dirs():
    for d in ("manifests", "raw", "figures", "analysis", "models_prepared"):
        os.makedirs(os.path.join(CAMP, d), exist_ok=True)


def p_(*a):
    return os.path.join(CAMP, *a)


def dump_json(path, obj):
    E.dump_json(path, obj)


def load_json(path, default=None):
    return E.load_json(path, default)


def write_csv(path, rows, gz=False):
    E.write_csv(path, rows, gz)


def read_csv(path):
    return E.read_csv(path)


def append_rows(path, rows, key_cols):
    import pandas as pd
    new = pd.DataFrame(rows)
    if os.path.exists(path) and len(new):
        old = pd.read_csv(path); keys = [k for k in key_cols if k in old.columns and k in new.columns]
        if keys:
            newk = set(map(tuple, new[keys].astype(str).values.tolist())); old = old[~old[keys].astype(str).apply(tuple, axis=1).isin(newk)]
        new = pd.concat([old, new], ignore_index=True)
    elif os.path.exists(path):
        return pd.read_csv(path)
    new.to_csv(path, index=False); return new


def ledger(stage, status, wall_s, note="", **kw):
    ensure_dirs()
    with open(p_("ledger.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, status=status, wall_hours=round(wall_s / 3600, 4), gpu=torch.cuda.is_available(), gpu_peak_mem_mb=(int(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0), note=note, code=code_fingerprint(), **kw), ensure_ascii=False) + "\n")


class Stage:
    def __init__(self, name, note=""):
        self.name, self.note, self.t = name, note, None

    def __enter__(self):
        ensure_dirs(); self.t = time.time(); ledger(self.name, "start", 0.0, self.note)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[smec12] {self.name} start {time.strftime('%H:%M:%S')}", flush=True); return self

    def __exit__(self, et, ev, tb):
        w = time.time() - self.t; ledger(self.name, "done" if et is None else "error", w, self.note if et is None else f"{self.note} | {ev!r}"[:300])
        print(f"[smec12] {self.name} {'done' if et is None else 'ERROR'} {w / 60:.1f} min", flush=True); return False


def code_fingerprint():
    g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip(); d = os.path.dirname(os.path.abspath(__file__)); h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return dict(git=g("git rev-parse HEAD")[:12], smec12_sha16=h.hexdigest()[:16])


def host_info():
    g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(server_id=SERVER, hostname=g("hostname"), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), driver=g("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"), torch=torch.__version__, git_commit=g("git rev-parse HEAD"))


# ---------------------------------------------------------------- asset registry (§2.3)
def run_complete(run):
    r = os.path.join(ROOT, "work_dir", run, "results"); return os.path.exists(os.path.join(r, "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(r, "full_best_hqnr.mat"))


def akey(sensor, role, seed):
    return f"{SENSORS[sensor]['S']}|{role}|S{seed}"


def asset_runs(sensor):
    """(role, seed) → run 이름: WV3 는 기존 자산, QB/GF2 는 준비 학습 id, WV2 는 없음(zero-shot 은 WV3 자산으로)."""
    if sensor == "wv3":
        return dict(WV3_ASSETS)
    if sensor in ("qb", "gf2"):
        return {(r, B.seed_of(r)): B.run_id(sensor, r) for r in B.ROLES}
    return {}


def asset_registry(write=False):
    reg = {}
    for s in ("wv3", "qb", "gf2"):
        for (role, seed), run in asset_runs(s).items():
            rd = os.path.join(ROOT, "work_dir", run); ok = run_complete(run); tags = [t for t in ("best_hqnr", "last") if os.path.exists(os.path.join(rd, t, "model.safetensors"))]
            e = dict(sensor=SENSORS[s]["S"], role=role, seed=seed, run=run, status=("available" if ok else ("running_or_partial" if os.path.exists(os.path.join(rd, "meta", "started_at.txt")) else "pending_dependency")), tags=tags, q=("NA" if role == "P0" else "measured"),
                     bands=SENSORS[s]["bands"], lineage=("historical" if s == "wv3" else B.LINEAGE), seen_in_pretraining=True, source_split=("full_train (historical)" if s == "wv3" else "full_train (bootstrap; FIT/CAL/DISC/CONF 분할은 analysis panel 만)"))
            if ok:
                bh = os.path.join(rd, "best_hqnr", "model.safetensors"); e.update(best_hqnr_sha256_16=(sha256_file(bh)[:16] if os.path.exists(bh) else None), selected=load_json(os.path.join(rd, "best_hqnr_meta.json"), None), last_meta=load_json(os.path.join(rd, "last_meta.json"), None))
            reg[akey(s, role, seed)] = e
    if write:
        ensure_dirs(); dump_json(p_("manifests", "asset_registry.json"), reg)
    return reg


def available(sensor, roles=("L1E4", "L1E4REP", "L000", "P0"), tags=("best_hqnr",)):
    out = []
    for (role, seed), run in asset_runs(sensor).items():
        if role in roles and run_complete(run):
            for t in tags:
                if os.path.exists(os.path.join(ROOT, "work_dir", run, t, "model.safetensors")):
                    out.append((sensor, role, seed, t))
    return out


class Loaded:
    def __init__(self, sensor, role, seed, tag, m, man, cfg, run):
        self.sensor, self.role, self.seed, self.tag, self.m, self.man, self.cfg, self.run = sensor, role, seed, tag, m, man, cfg, run
        self.key = f"{akey(sensor, role, seed)}|{tag}"; self.mg = int(man["aligner_view_margin"]); self.has_aligner = m.aligner is not None; self.sha16 = man["tensors_sha256_16"]
        self.step = (man.get("tag_meta") or {}).get("step") or (50000 if tag == "last" else None); self.bands = SENSORS[sensor]["bands"]

    def __repr__(self):
        return f"Loaded({self.key}, step {self.step}, sha {self.sha16})"


_CACHE = {}


def load_asset(key_or_tuple, tag="best_hqnr", dev=DEV):
    """registry id → model(eval, frozen), resolved config, hash (§20.1 load_asset)."""
    if isinstance(key_or_tuple, str):
        S, role, sd = key_or_tuple.split("|")[:3]; sensor = next(k for k, v in SENSORS.items() if v["S"] == S); seed = int(sd[1:])
    else:
        sensor, role, seed = key_or_tuple[:3]
    ck = (sensor, role, seed, tag)
    if ck in _CACHE:
        return _CACHE[ck]
    run = asset_runs(sensor)[(role, seed)]; rd = os.path.join(ROOT, "work_dir", run); cfg = E.localize_cfg(yaml.safe_load(open(os.path.join(rd, "meta", "config.yaml"))))
    m, man = load_run_model(rd, tag, import_class(cfg["model"])); L = Loaded(sensor, role, seed, tag, m.to(dev).eval().requires_grad_(False), man, cfg, run); _CACHE[ck] = L; return L


# ---------------------------------------------------------------- data (§3.3–3.4)
def data_file(sensor, part):
    p = SENSORS[sensor].get(part); return os.path.join(ROOT, p) if p else None


def data_registry(write=False):
    out = {}
    for s, X in SENSORS.items():
        d = dict(sensor=X["S"], bands=X["bands"], max_pixel=X["max_pixel"], role=X["role"], note=X["data_note"], files={})
        for part in ("train", "valid", "rr", "fr"):
            f = data_file(s, part)
            if f and os.path.exists(f):
                with h5py.File(f) as h:
                    d["files"][part] = dict(path=os.path.relpath(f, ROOT), shapes={k: list(h[k].shape) for k in h.keys()}, sha256_16=sha256_file(f)[:16] if os.path.getsize(f) < 3e9 else "skipped(>3GB)")
                pp = f.replace(".h5", "_pan.h5"); d["files"][part + "_pan"] = dict(path=os.path.relpath(pp, ROOT), exists=os.path.exists(pp))
            else:
                d["files"][part] = dict(path=(os.path.relpath(f, ROOT) if f else None), exists=False)
        d["data_ready"] = all(d["files"].get(k, {}).get("shapes") for k in (("rr", "fr") if s == "wv2" else ("train", "rr", "fr")))
        out[X["S"]] = d
    if write:
        ensure_dirs(); dump_json(p_("manifests", "data_registry.json"), out)
    return out


def make_panels(sensor):
    """in-domain: train patch 를 32-block 단위로 CAL/DISC/CONF/FIT 에 배정(seed 271828; block = source proxy · source_group_unknown). zero-shot(WV2): RR/FR tile 목록. 이미 있으면 재사용."""
    ensure_dirs(); p = p_("manifests", f"panels_{sensor}.json")
    if os.path.exists(p):
        return load_json(p)
    X = SENSORS[sensor]
    if sensor == "wv2":
        out = dict(sensor=X["S"], role="zero_shot", ids={"RRTILE": list(range(20 * TILES_PER_SCENE)), "FRTILE": list(range(20 * TILES_PER_SCENE))}, tiles_per_scene=TILES_PER_SCENE, tile=TILE, note="공식 RR/FR 20 scene × 16 tile(겹치지 않는 64², 격자 순); train 없음"); dump_json(p, out); return out
    with h5py.File(data_file(sensor, "train")) as f:
        N = f["pan"].shape[0]
    nb = N // BLOCK; rng = np.random.RandomState(MANIFEST_SEED); order = rng.permutation(nb); pos = 0; ids = {}; blocks = {}
    for name in ("CAL", "DISC", "CONF", "FIT"):
        k = PANEL[name] // BLOCK; bs = order[pos:pos + k]; pos += k; ids[name] = [int(i) for b in bs for i in range(b * BLOCK, (b + 1) * BLOCK)]; blocks[name] = [f"blk{int(b):03d}" for b in bs]
    out = dict(sensor=X["S"], role="in_domain", seed=MANIFEST_SEED, block=BLOCK, n_train=int(N), sizes={k: len(v) for k, v in ids.items()}, ids=ids, blocks=blocks, independence="patch_only (source_group_unknown: 32 연속 index block proxy; 원 scene/strip id 없음 §3.3)",
               seen_in_pretraining=True, note="기존 full-train 모델은 이 patch 를 학습에 봤다(seen_in_pretraining). 준비 학습(QB/GF2) 도 full train 으로 돌아 같은 상태 — FIT70 source-holdout cohort 는 이번 release 에 없다(§3.3 구분 표시)")
    dump_json(p, out); return out


def group_of(sensor, sample_id, part=None):
    return f"{SENSORS[sensor]['S']}:blk{int(sample_id) // BLOCK:03d}" if part in (None, "CAL", "DISC", "CONF", "FIT") else f"{SENSORS[sensor]['S']}:{part}:scene{int(sample_id) // TILES_PER_SCENE:02d}"


_PATCH_CACHE = {}


def load_patches(sensor, ids):
    """train h5 → (gt, ms, lpan, pan) float32 [-1,1] (센서 max_pixel) CPU, ids 순서 유지 (§20.1 read_sample; make_ms_base 는 model 안의 bicubic ×4)."""
    ids = [int(i) for i in ids]; key = (sensor, tuple(ids))
    if key in _PATCH_CACHE:
        return _PATCH_CACHE[key]
    mp = SENSORS[sensor]["max_pixel"]; f0 = data_file(sensor, "train"); order = np.argsort(ids); sidx = np.array(ids)[order]
    with h5py.File(f0) as f:
        gt = np.asarray(f["gt"][sidx], dtype=np.float32); ms = np.asarray(f["ms"][sidx], dtype=np.float32); pan = np.asarray(f["pan"][sidx], dtype=np.float32)
    with h5py.File(f0.replace(".h5", "_pan.h5")) as f:
        lpan = np.asarray(f["lpan"][sidx], dtype=np.float32)
    inv = np.argsort(order); out = tuple(torch.from_numpy(a[inv]) * 2 / mp - 1 for a in (gt, ms, lpan, pan))
    if len(ids) <= 4096:
        _PATCH_CACHE[key] = out
    return out


def feeders(sensor):
    """RR/FR Feeder (센서 max_pixel) + FR 원시 DN (scene_views 용)."""
    from feeders.feeder import PanFeeder
    rr = PanFeeder(dataroot=data_file(sensor, "rr")); fr = PanFeeder(dataroot=data_file(sensor, "fr")); mp = float(fr.max_pixel)      # max_pixel 은 feeder 가 dataroot 이름으로 정한다 (인자로 주면 속성이 안 잡힌다)
    assert abs(mp - SENSORS[sensor]["max_pixel"]) < 1e-6, f"{sensor}: feeder max_pixel {mp} ≠ registry {SENSORS[sensor]['max_pixel']}"
    with h5py.File(data_file(sensor, "fr")) as f:
        lms_raw = np.asarray(f["lms"], dtype=np.float64); pan_raw = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    return dict(rr=rr, fr=fr, lms_raw=lms_raw, pan_raw=pan_raw, mp=mp, sensor=sensor)


def scene_tiles(t, part):
    """[C,H,W] scene → 16 tile (64², 겹치지 않음; RR 256² 는 전부, FR 512² 는 4 픽셀 간격 격자에서 deterministic 16)."""
    C_, H, W = t.shape; n = H // TILE; coords = [(i * TILE, j * TILE) for i in range(n) for j in range(n)]
    if len(coords) > TILES_PER_SCENE:
        coords = coords[::max(1, len(coords) // TILES_PER_SCENE)][:TILES_PER_SCENE]
    return coords


# ---------------------------------------------------------------- probes (§4.1)
def probe_bank(bank):
    out = []
    for r in (RADII_S if bank == "S" else RADII_AB):
        for th in ANG[bank]:
            a = math.radians(th); out.append(dict(probe_id=f"{bank}_r{r}_t{th}", bank=bank, r=r, theta_deg=th, ey=round(r * math.sin(a), 12), ex=round(r * math.cos(a), 12)))
    return out


def q_const(bank):
    return float(np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in probe_bank(bank)]))


def q_rel(q, bank):
    return float(q) / q_const(bank)


def probe_manifest():
    ensure_dirs(); m = {b: dict(n=len(probe_bank(b)), probes=probe_bank(b), q_const=q_const(b)) for b in ("A", "B", "S")}
    m["definition"] = "r = A(W(P,ε),M) + ε − A(P,M); q = 1/(2K) Σ||r||₁ (HR px); EPE = 1/K Σ||r||₂; ε = (r sinθ, r cosθ) = (dy, dx); W(P,c)[y,x] = P[y+c_y, x+c_x] bicubic/border; identity(ε=0) 는 q 평균에 넣지 않음"
    m["training_disk"] = "학습 jitter 는 b=2 원판 면적 uniform — probe 분포와 다르다 (§4.1)"; dump_json(p_("manifests", "probe_manifest.json"), m); return m


# ---------------------------------------------------------------- adapter (§20.1)
@torch.no_grad()
def predict_shift(L, pan, ms, chunk=256):
    """c0 [N,2] — margin4/z-score 는 predict_c 안에서 정확히 한 번. A 없는 모델(P0) 은 None."""
    if not L.has_aligner:
        return None
    out = []
    for s in range(0, pan.shape[0], chunk):
        p = pan[s:s + chunk].to(DEV); mb = F.interpolate(ms[s:s + chunk].to(DEV), scale_factor=4, mode="bicubic"); out.append(E.predict_c(L.m.aligner, p, mb, L.mg).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruct_at(L, pan, ms, lpan, c=None, chunk=64):
    """Ŷ = M + F([W(P,c), M]) — 원본 P 에서 한 번 sampling (delta_override). c None 이면 모델 자체(c0 또는 A 없음)."""
    ys = []
    for s in range(0, pan.shape[0], chunk):
        p, q, l = pan[s:s + chunk].to(DEV), ms[s:s + chunk].to(DEV), lpan[s:s + chunk].to(DEV); kw = {}
        if c is not None and L.has_aligner:
            kw["delta_override"] = c[s:s + chunk].to(DEV).float()
        ys.append(L.m(p, q, l, **kw)["y"].float().cpu())
    return torch.cat(ys)


@torch.no_grad()
def measure_offset_bank(L, pan, ms, bank="A", chunk=256, view_fn=None):
    """→ dict(c0 [N,2], c_eps [N,K,2], resid [N,K,2], q [N], epe [N], q_by_r {r: [N]}, identity [N]). view_fn(p) 로 A 전용 view 변환(I21) 을 넣을 수 있다 (U-Net 경로는 바꾸지 않는다)."""
    probes = probe_bank(bank); K = len(probes); N = pan.shape[0]; c0_all = torch.zeros(N, 2); ce_all = torch.zeros(N, K, 2); ident = torch.zeros(N); Et = torch.tensor([[p["ey"], p["ex"]] for p in probes])
    for s in range(0, N, chunk):
        p = pan[s:s + chunk].to(DEV); mb = F.interpolate(ms[s:s + chunk].to(DEV), scale_factor=4, mode="bicubic"); pv = view_fn(p) if view_fn else p
        c0 = E.predict_c(L.m.aligner, pv, mb, L.mg).cpu(); c0_all[s:s + chunk] = c0
        ident[s:s + chunk] = (E.predict_c(L.m.aligner, E.warp_pan(pv, torch.zeros(p.shape[0], 2, device=DEV)), mb, L.mg).cpu() - c0).abs().sum(1)
        for k, pr in enumerate(probes):
            e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV).expand(p.shape[0], 2); ce_all[s:s + chunk, k] = E.predict_c(L.m.aligner, E.warp_pan(pv, e), mb, L.mg).cpu()
    resid = ce_all + Et[None] - c0_all[:, None]; radii = RADII_S if bank == "S" else RADII_AB
    return dict(c0=c0_all, c_eps=ce_all, resid=resid, q=resid.abs().mean(dim=(1, 2)), epe=resid.norm(dim=2).mean(1), q_by_r={str(r): resid[:, [k for k, pr in enumerate(probes) if pr["r"] == r]].abs().mean(dim=(1, 2)) for r in radii}, identity=ident, probes=probes)


# ---------------------------------------------------------------- e / ROI / label (§4.2–4.4)
def roi(t, scale):
    m = ROI_MARGIN[scale]; return t[:, :, m:-m, m:-m]


def e_metrics(y, gt, scale="native64"):
    d = (y - gt).abs(); out = dict(e_full=d.mean(dim=(1, 2, 3)), e_roi=roi(d, scale).mean(dim=(1, 2, 3)), e_band=d.mean(dim=(2, 3)))
    gx_h, gy_h = E.scharr_t(y); gx, gy = E.scharr_t(gt); s = (slice(None), slice(None), slice(1, -1), slice(1, -1)); out["edge_l1"] = 0.5 * (gx_h - gx)[s].abs().mean(dim=(1, 2, 3)) + 0.5 * (gy_h - gy)[s].abs().mean(dim=(1, 2, 3)); return out


def contrast(gt, floor=1e-6):
    """band c 의 robust contrast s_ic = P95 − P5 (§D11-A; GT 기반 진단)."""
    flat = gt.flatten(2); return (torch.quantile(flat, 0.95, dim=2) - torch.quantile(flat, 0.05, dim=2)).clamp_min(floor)


def e_contrast(y, gt, s_floor):
    d = (y - gt).abs().mean(dim=(2, 3)); s = contrast(gt); return (d / torch.maximum(s, s_floor[None])).mean(1)


def label(e, q, tC, tR):
    """§4.2: (e ≤ median e, q ≤ median q) → EdCd 등. 인자 순서 (e, q, tC(q median), tR(e median))."""
    return ("Ed" if e <= tR else "Eu") + ("Cd" if q <= tC else "Cu")


def boundary_near(v, thr, iqr):
    return abs(float(v) - float(thr)) <= BOUNDARY_FRAC * float(iqr)


def thresholds_from(cal_e, cal_q, noise_q=0.0):
    cal_e = np.asarray(cal_e, float); cal_q = np.asarray(cal_q, float); iq = float(np.quantile(cal_q, .75) - np.quantile(cal_q, .25)); ie = float(np.quantile(cal_e, .75) - np.quantile(cal_e, .25))
    return dict(tR=float(np.median(cal_e)), tC=float(np.median(cal_q)), e_iqr=ie, q_iqr=iq, q_noise=float(noise_q), q_unresolved=bool(iq < Q_UNRESOLVED_X * max(noise_q, 1e-9) or iq == 0.0), n=int(len(cal_e)), rule="CAL median; boundary_near = ±10% IQR (신규 sensitivity 설정)")


# ---------------------------------------------------------------- stats (§4.5)
def block_bootstrap(values, groups, fn=np.mean, n=2000, seed=0):
    return E.block_bootstrap(values, groups, fn=fn, n=n, seed=seed)


def spearman(x, y):
    return E.spearman(x, y)


def spearman_boot(x, y, groups, n=2000, seed=0):
    return E.spearman_boot(x, y, groups, n=n, seed=seed)


def loso(x, y, groups):
    """leave-one-source-out Spearman 범위 (§D10-A): source(block) 하나씩 뺀 ρ 의 min/max."""
    x = np.asarray(x, float); y = np.asarray(y, float); g = np.asarray(groups); rs = []
    for u in np.unique(g):
        m = g != u; r = spearman(x[m], y[m])
        if r["rho"] is not None:
            rs.append(r["rho"])
    return dict(min=float(min(rs)), max=float(max(rs)), n_sources=int(len(np.unique(g)))) if rs else None
