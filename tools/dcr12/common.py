"""DCR12 공통 — registry(T0 · B0/FQ · B1/JK0 by seed), panel(CAL/DISC/CONF/GRAD · RR20 · FR20), probe 집합 E16(+confirm), C·R 정의, 모델 적재, ledger/io.
계획 research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md (§2 계약 · §5 panel/probe/label · §13 산출물). 기존 EQREC4 helper(tools/eqrec4/common.py) 를 재사용한다."""
import copy, hashlib, json, math, os, subprocess, sys, time, types
import numpy as np, torch, torch.nn.functional as F, h5py, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
from tools.eqrec4 import common as E                                      # noqa: E402  (warp_pan · predict_c · load_patches · feeders · l1_roi · scharr_t · block_bootstrap · spearman …)
from tools import gen_pakd50_configs as G                                 # noqa: E402
from kdv.teacher_assets import load_run_model, load_state, sha256_file, tensors_sha, state_hash   # noqa: E402
from main import import_class                                             # noqa: E402

PLAN = "research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md"
SERVER = open(os.path.join(ROOT, "gspread", "server.txt")).read().strip() if os.path.exists(os.path.join(ROOT, "gspread", "server.txt")) else "s1"
CAMPAIGN_ID = "DCR12_20260914_v1"; CAMP = os.path.join(ROOT, "work_dir", f"_dcr12_{SERVER}_campaign"); FIG = os.path.join(CAMP, "figures")
DEV = E.DEV; MAX_PIXEL = E.MAX_PIXEL
SEEDS = [1234, 777]                                                       # §8.2: 단일 호스트 두 seed (같은 호스트 안에서 pair 를 완성한다)
CASES = {"B0": "FQ", "B1": "JK0"}                                         # §0.2: B0 = FQ(A frozen, U Q12) · B1 = JK0(A trainable: L_H + λE L_E + odd·1e-4 L_off; soft→A 차단)
T0_DIR = os.path.join(ROOT, G.T0_ASSET_DIR); T0_TAG = G.T0_TAG
D03_STEPS = [5050, 25250, 45450, 50000]                                   # §7.3 (공통 격자 GRID1010 의 후보)
D04_STEPS = [25250, 45450]                                                # §9.1
PANEL = dict(CAL=1024, DISC=512, CONF=512); GRAD_N = 64; DISC128_N = 128; FR8_N = 8; SPLIT_SEED = 314159; BLOCK = 32
PROBE_R = (0.5, 1.0); BIAS_HR = 0.25; SMALL_CELL_N = 32; SMALL_CELL_GROUPS = 4
LABELS = {("low", "low"): "A", ("low", "high"): "B", ("high", "low"): "C", ("high", "high"): "D"}     # §5.3: (C, R)
CELL_DESC = {"A": "C low · R low", "B": "C low · R high", "C": "C high · R low", "D": "C high · R high"}
LR_U, LR_A, WD = 1e-4, 1e-5, 0.01; TOTAL_UPDATES, WARMUP = 50000, 100


# ---------------------------------------------------------------- io / ledger
def ensure_dirs():
    os.makedirs(CAMP, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def dump_json(path, obj):
    E.dump_json(path, obj)


def load_json(path, default=None):
    return E.load_json(path, default)


def write_csv(path, rows, gz=False):
    E.write_csv(path, rows, gz)


def read_csv(path):
    return E.read_csv(path)


def append_rows(path, rows, key_cols):
    """CSV 에 행을 덧붙인다 — key_cols 가 같은 기존 행은 새 행으로 교체 (stage 를 학습 뒤 다시 돌려도 중복 없이 갱신)."""
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
    with open(os.path.join(CAMP, "time_ledger.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, status=status, wall_hours=round(wall_s / 3600, 4), gpu=torch.cuda.is_available(),
                                gpu_peak_mem_mb=(int(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0), note=note, code=code_fingerprint(), **kw), ensure_ascii=False) + "\n")


class Stage:
    def __init__(self, name, note=""):
        self.name, self.note, self.t = name, note, None

    def __enter__(self):
        ensure_dirs(); self.t = time.time(); ledger(self.name, "start", 0.0, self.note)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[dcr12] {self.name} start {time.strftime('%H:%M:%S')}", flush=True); return self

    def __exit__(self, et, ev, tb):
        w = time.time() - self.t; ledger(self.name, "done" if et is None else "error", w, self.note if et is None else f"{self.note} | {ev!r}"[:300])
        print(f"[dcr12] {self.name} {'done' if et is None else 'ERROR'} {w / 60:.1f} min", flush=True); return False


def code_fingerprint():
    g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    d = os.path.dirname(os.path.abspath(__file__)); h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(open(os.path.join(d, f), "rb").read())
    return dict(git=g("git rev-parse HEAD")[:12], dirty=bool(g("git status --porcelain -- tools/dcr12 train_kdv.py kdv pa")), dcr12_sha16=h.hexdigest()[:16])


# ---------------------------------------------------------------- registry
def run_name(case_key, seed):
    return G.run_name(CASES[case_key], seed)


def run_dir(case_key, seed):
    return os.path.join(ROOT, "work_dir", run_name(case_key, seed))


def ckpt_path(case_key, seed, tag):
    """tag: best_hqnr | last | cand:<step> → 폴더 (없으면 None)."""
    rd = run_dir(case_key, seed); d = os.path.join(rd, "candidates", f"step-{int(tag[5:])}") if tag.startswith("cand:") else os.path.join(rd, tag)
    return d if os.path.exists(os.path.join(d, "model.safetensors")) else None


def ckpt_step(case_key, seed, tag):
    if tag.startswith("cand:"):
        return int(tag[5:])
    p = os.path.join(run_dir(case_key, seed), f"{tag}_meta.json")
    if os.path.exists(p):
        return int(json.load(open(p)).get("step", -1))
    if tag == "last":
        return TOTAL_UPDATES
    return None


def run_complete(case_key, seed):
    r = os.path.join(run_dir(case_key, seed), "results")
    return os.path.exists(os.path.join(r, "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(r, "full_best_hqnr.mat"))


def provenance(case_key, seed):
    """다른 호스트에서 옮겨 온 run 의 출처 (tools/dcr12_bundle.py install 이 쓰는 work_dir/<run>/bundle_provenance.json) — 없으면 None(이 서버 학습)."""
    return load_json(os.path.join(run_dir(case_key, seed), "bundle_provenance.json"), None)


def trained_on(case_key, seed):
    """run 을 학습한 서버 id: bundle provenance 의 source_server, 아니면 이 서버(run 디렉토리가 있을 때), 없으면 None."""
    p = provenance(case_key, seed)
    if p and p.get("source_server"):
        return p["source_server"]
    return SERVER if os.path.exists(run_dir(case_key, seed)) else None


def host_pairing():
    """§8.2: pair(B0/B1, 같은 seed) 안은 같은 호스트여야 한다; seed 간 호스트가 다르면 host A/B 배치(seed×host 결합)."""
    by = {f"S{s}": {ck: trained_on(ck, s) for ck in ("B0", "B1")} for s in SEEDS}
    same = {k: ((v["B0"] == v["B1"]) if (v["B0"] and v["B1"]) else None) for k, v in by.items()}
    hosts = sorted({h for v in by.values() for h in v.values() if h})
    return dict(by_seed=by, pair_same_host=same, hosts=hosts, seeds_same_host=(len(hosts) <= 1), seed_host_coupled=(len(hosts) > 1),
                note="§8.2: pair 안은 같은 호스트에서 완성. seed 간 호스트가 다르면(host A/B 배치) seed 와 host 가 결합돼 있다 — 최종 재현 주장은 한 호스트의 두 seed 또는 양쪽 pair 복제로 보강")


def pipe_key(role, case_key=None, seed=None, tag=None):
    return "T0" if role == "T" else f"{case_key}_S{seed}_{tag.replace('cand:', 'cand')}"


class Pipe:
    """적재된 pipeline (Teacher 또는 Student checkpoint): m(eval·no-grad), cfg, aligner margin, hash."""

    def __init__(self, role, case_key, seed, tag, m, man, cfg, rd):
        self.role, self.case_key, self.seed, self.tag, self.m, self.man, self.cfg, self.rd = role, case_key, seed, tag, m, man, cfg, rd
        self.key = pipe_key(role, case_key, seed, tag); self.mg = int(man["aligner_view_margin"]); self.step = (man.get("tag_meta") or {}).get("step") if role == "T" else ckpt_step(case_key, seed, tag)
        self.sha16 = man["tensors_sha256_16"]; self.aligner_sha16 = state_hash(m.aligner) if m.aligner is not None else None
        self.case = (CASES[case_key] if case_key else "T0"); self.has_aligner = m.aligner is not None

    def __repr__(self):
        return f"Pipe({self.key}, step {self.step}, sha {self.sha16})"


_PIPES = {}


def load_pipe(role, case_key=None, seed=None, tag="best_hqnr", dev=DEV):
    key = pipe_key(role, case_key, seed, tag)
    if key in _PIPES:
        return _PIPES[key]
    if role == "T":
        rd = T0_DIR; cfg = E.localize_cfg(yaml.safe_load(open(os.path.join(rd, "meta", "config.yaml")))); m, man = load_run_model(rd, T0_TAG, import_class(cfg["model"]))
    else:
        rd = run_dir(case_key, seed); d = ckpt_path(case_key, seed, tag)
        if d is None:
            raise FileNotFoundError(f"{key}: checkpoint 없음 ({rd}/{tag})")
        cfg = E.localize_cfg(yaml.safe_load(open(os.path.join(rd, "meta", "config.yaml")))); m, man = load_run_model(rd, os.path.relpath(d, rd), import_class(cfg["model"]))
    p = Pipe(role, case_key, seed, tag, m.to(dev).eval().requires_grad_(False), man, cfg, rd); _PIPES[key] = p; return p


def available_students(tags=("best_hqnr", "last")):
    """**완료된 run** 의 checkpoint 만 (학습 중인 run 의 best_hqnr/last 는 바뀐다 — 진단에 쓰지 않는다)."""
    out = []
    for seed in SEEDS:
        for ck in ("B0", "B1"):
            if not run_complete(ck, seed):
                continue
            for tag in tags:
                if ckpt_path(ck, seed, tag):
                    out.append((ck, seed, tag))
    return out


def reference_student(seed):
    """§5.3: 같은 seed B0 의 고정 reference checkpoint(best_hqnr; 완료 run) — Student bin 정의에 쓴다 (B1 에도 같은 bin 적용)."""
    return ("B0", seed, "best_hqnr") if (run_complete("B0", seed) and ckpt_path("B0", seed, "best_hqnr")) else None


def calibration():
    cal = G.calibration(); assert cal.get("tau_R") and cal.get("lambda_E"), "τR/λE 자산 없음 (assets/pakd50/calibration_resolved.json) — §2.2: 없으면 BLOCKED"
    return dict(tau_R=float(cal["tau_R"]), lambda_E=float(cal["lambda_E"]), source=G.ASSET_CAL_PATH, sha256=sha256_file(os.path.join(ROOT, G.ASSET_CAL_PATH)))


# ---------------------------------------------------------------- data / panels (§5.1)
def data_paths():
    cfg = E.localize_cfg(yaml.safe_load(open(os.path.join(T0_DIR, "meta", "config.yaml")))); tr = cfg["train_feeder_args"]["dataroot"]
    return dict(train=tr, train_pan=tr.replace(".h5", "_pan.h5"), rr=cfg["test_reduced_feeder_args"]["dataroot"], fr=cfg["test_full_feeder_args"]["dataroot"], cfg=cfg)


def make_panels():
    """CAL 1024 / DISC 512 / CONF 512 를 32-index block 단위로 서로 겹치지 않게 (seed 314159; 결과를 보기 전에 고정). DISC128 = DISC 의 고정 부분, FR8 = FR20 중 고정 8 scene. 이미 있으면 재사용."""
    ensure_dirs(); p = os.path.join(CAMP, "panel_ids.json")
    if os.path.exists(p):
        return load_json(p)
    dp = data_paths()
    with h5py.File(dp["train"]) as f:
        N = f["pan"].shape[0]
    nb = N // BLOCK; rng = np.random.RandomState(SPLIT_SEED); order = rng.permutation(nb); pos = 0; ids = {}; blocks = {}
    for name in ("CAL", "DISC", "CONF"):
        k = PANEL[name] // BLOCK; bs = order[pos:pos + k]; pos += k
        ids[name] = [int(i) for b in bs for i in range(b * BLOCK, (b + 1) * BLOCK)]; blocks[name] = [f"blk{int(b):03d}" for b in bs]
    perm = np.random.RandomState(SPLIT_SEED + 1).permutation(len(ids["DISC"])); ids["DISC128"] = [ids["DISC"][i] for i in sorted(perm[:DISC128_N])]
    ids["FR8"] = sorted(int(i) for i in np.random.RandomState(SPLIT_SEED + 2).permutation(20)[:FR8_N])
    out = dict(seed=SPLIT_SEED, block=BLOCK, n_train_total=int(N), train_h5=dp["train"], train_sha256=sha256_file(dp["train"]), independence="patch_only (원 scene id 없음; 32 연속 index block 을 source group proxy 로)",
               sizes={k: len(v) for k, v in ids.items()}, blocks=blocks, ids=ids, GRAD="D01 이 Teacher cell 균형으로 DISC 에서 64 를 뽑아 grad_ids 에 기록", grad_ids=None,
               note="CAL/DISC/CONF 는 학습 train split 의 patch(모든 model 이 학습에 봤다) — 분석 규칙의 개발/확인 분리이지 미사용 test 가 아니다 (§5.1). augmentation OFF.")
    dump_json(p, out); return out


def group_of(sample_id):
    return f"blk{int(sample_id) // BLOCK:03d}"


def load_patches(ids):
    return E.load_patches(ids)


# ---------------------------------------------------------------- probes (§5.2) and C/R (§5.3)
def probe_set(kind="main"):
    """E = {r(sinθ, cosθ): r ∈ {0.5, 1.0}, θ = kπ/4} (16); confirm = θ + π/8 (16). (dy, dx) = (r sinθ, r cosθ) HR px."""
    out = []; rot = math.pi / 8 if kind == "confirm" else 0.0
    for r in PROBE_R:
        for k in range(8):
            th = k * math.pi / 4 + rot; out.append(dict(probe_id=f"{kind}_r{r}_k{k}", kind=kind, r=r, theta_deg=round(math.degrees(th), 3), ey=round(r * math.sin(th), 12), ex=round(r * math.cos(th), 12)))
    return out


def probe_manifest():
    ensure_dirs(); m = dict(main=probe_set("main"), confirm=probe_set("confirm"), identity=dict(ey=0.0, ex=0.0, use="identity floor only (주 평균에 넣지 않음)"),
                            definition="C = 1/(2K) Σ_ε Σ_q |c_{ε,q} + ε_q − c_{0,q}|, K=16, HR px; W(P,ε)[y,x] = P[y+ε_y, x+ε_x] (bicubic, border); 학습 sampler(radius 2 disk-uniform) 는 바꾸지 않는다",
                            support_note="margin-4 view 가 r ≤ 1 probe 의 support 를 만족한다 (|ε|∞ + |c|∞ + 4 ≤ margin 검사는 sample 별 support_ok)")
    dump_json(os.path.join(CAMP, "probe_manifest.json"), m); return m


@torch.no_grad()
def consistency(P, pan, ms, probes=None, chunk=256, c_bias=None):
    """pan [N,1,H,W], ms [N,8,h,w] (정규화 CPU) → dict(c0 [N,2], resid [N,K,2] (= c_ε + ε − c_0), C [N], C_by_r {r: [N]}, C_axis [N,2], identity [N]).
    c_bias: (dy, dx) 를 **예측된 두 correction 모두에** 더하는 wrapper 개입 (§6.3 closure 불변 대조; 입력은 바꾸지 않는다)."""
    probes = probes or probe_set("main"); m = P.m; mg = P.mg; N = pan.shape[0]; K = len(probes)
    c0_all = torch.zeros(N, 2); resid = torch.zeros(N, K, 2); ident = torch.zeros(N); Et = torch.tensor([[pr["ey"], pr["ex"]] for pr in probes])
    b = torch.tensor(c_bias, dtype=torch.float32) if c_bias is not None else torch.zeros(2)
    for s in range(0, N, chunk):
        p = pan[s:s + chunk].to(DEV); q = ms[s:s + chunk].to(DEV); mb = F.interpolate(q, scale_factor=4, mode="bicubic")
        c0 = E.predict_c(m.aligner, p, mb, mg).cpu() + b; c0_all[s:s + chunk] = c0
        cz = E.predict_c(m.aligner, E.warp_pan(p, torch.zeros(p.shape[0], 2, device=DEV)), mb, mg).cpu() + b; ident[s:s + chunk] = (cz - c0).abs().sum(1)
        for k, pr in enumerate(probes):
            e = torch.tensor([[pr["ey"], pr["ex"]]], device=DEV).expand(p.shape[0], 2); ce = E.predict_c(m.aligner, E.warp_pan(p, e), mb, mg).cpu() + b
            resid[s:s + chunk, k] = ce + Et[k][None] - c0
    C = resid.abs().mean(dim=(1, 2)); by_r = {str(r): resid[:, [k for k, pr in enumerate(probes) if pr["r"] == r]].abs().mean(dim=(1, 2)) for r in PROBE_R}
    return dict(c0=c0_all, resid=resid, C=C, C_by_r=by_r, C_axis=resid.abs().mean(1), identity=ident, probes=probes)


@torch.no_grad()
def reconstruct(P, pan, ms, lpan, chunk=64, delta_override=None):
    """→ y [N,8,H,W] CPU, delta [N,2]. delta_override [N,2] 면 correction 치환 (같은 sampler 를 정확히 한 번; §6.1)."""
    ys, ds = [], []
    for s in range(0, pan.shape[0], chunk):
        p, q, l = pan[s:s + chunk].to(DEV), ms[s:s + chunk].to(DEV), lpan[s:s + chunk].to(DEV); kw = {}
        if delta_override is not None:
            kw["delta_override"] = delta_override[s:s + chunk].to(DEV).float()
        o = P.m(p, q, l, **kw); ys.append(o["y"].float().cpu()); ds.append(o["delta"].float().cpu())
    return torch.cat(ys), torch.cat(ds)


def r_metrics(y, gt):
    """R_plain(전체 64²) · R_interior(margin 16) · edge_plain(signed Scharr L1, 전체; 경계 1px 제외) — sample 별 (정규화 [-1,1] 단위)."""
    d = (y - gt).abs(); full = d.mean(dim=(1, 2, 3)); mgn = E.ROI_MARGIN["native64"]; inner = d[:, :, mgn:-mgn, mgn:-mgn].mean(dim=(1, 2, 3))
    gx_h, gy_h = E.scharr_t(y); gx, gy = E.scharr_t(gt); s = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    edge = 0.5 * (gx_h - gx)[s].abs().mean(dim=(1, 2, 3)) + 0.5 * (gy_h - gy)[s].abs().mean(dim=(1, 2, 3))
    return dict(R_plain=full, R_interior=inner, edge_plain=edge)


def saturation(pan, ms):
    return dict(pan_sat_frac=(pan >= 0.999).float().mean(dim=(1, 2, 3)), ms_sat_frac=(ms >= 0.999).float().mean(dim=(1, 2, 3)))


# ---------------------------------------------------------------- losses (§3.1) for D03/D04 — 학습과 같은 객체
def q12_terms(y, y_t, gt, cal, tau=None):
    """L0 = ⟨e_S⟩ · L_H = ⟨(1+d_T) e_S⟩ · L_K = 0.1⟨(1−d_T) a_T k⟩ · L_E(무가중) — GTAnchoredReconstructionKD(adaptive) 와 output_edge_loss 그대로 (Teacher/GT/gate detach)."""
    from kdv.losses_rec import GTAnchoredReconstructionKD
    from pa.losses import output_edge_loss
    crit = q12_terms._crit.get(float(tau or cal["tau_R"]))
    if crit is None:
        crit = GTAnchoredReconstructionKD(float(tau or cal["tau_R"]), alpha=1.0, kd_weight=0.1, eps=1e-6, mode="adaptive"); q12_terms._crit[float(tau or cal["tau_R"])] = crit
    r = crit(y.float(), y_t.float(), gt.float(), return_maps=True)
    return dict(L0=(y.float() - gt.float()).abs().mean(), LH=r.hard, LK=r.soft, LE=output_edge_loss(y.float(), gt.float()), maps=r.maps)
q12_terms._crit = {}


def cosine_lr(step, peak):
    """warmup100 + cosine (50K horizon) — transformers get_scheduler('cosine') 와 같은 식 (D04 의 checkpoint LR)."""
    if step < WARMUP:
        return peak * step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, TOTAL_UPDATES - WARMUP); return peak * 0.5 * (1.0 + math.cos(math.pi * prog))


# ---------------------------------------------------------------- trainer stub (실제 KDVTrainer._step / _apply_routing; tools/pakd50_unit_tests._stub 과 같은 방식)
def trainer_stub(case, seed, student_model, teacher_model, cal, seed_gen=3234):
    from train_kdv import KDVTrainer
    from kdv.registry import resolve
    from kdv.losses_rec import GTAnchoredReconstructionKD
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, seed, SERVER, cal=dict(tau_R=cal["tau_R"], lambda_E=cal["lambda_E"])); sp = tr.spec = resolve(tr.k)
    tr.model = student_model; tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = teacher_model; tr.aligner_trainable = sp["aligner_trainable"]; tr.aligner_view_margin = 4; tr.share_correction = False
    tr.protocol = sp["protocol"]; tr.radius_hr = sp["radius_hr"]; tr.diag_every = 10 ** 9
    tr.rec_crit = (GTAnchoredReconstructionKD(cal["tau_R"], alpha=float(tr.k["rec"].get("alpha", 1.0)), kd_weight=float(tr.k["rec"].get("kd_weight", 0.0)), eps=1e-6, mode=sp["rec_mode"]) if sp["rec_case"] != "N0" else None)
    tr.tri = sp["tri"]; tr.stat_extra = []; tr.lam_V = (cal["lambda_E"] if sp["stat_enabled"] else 0.0); tr.stat_ramp = 0; tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0
    tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=TOTAL_UPDATES)
    tr.freeze_until, tr.freeze_from = sp["aligner_freeze_until"], sp["aligner_freeze_from"]; tr.route_A = tuple(sp["route_A"]); tr._routed = any(q != 1.0 for q in tr.route_A); tr._sched_last = None
    return tr


def flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(gs, params)])


def cos(a, b, eps=1e-30):
    return float((a @ b) / (a.norm() * b.norm() + eps))


# ---------------------------------------------------------------- stats
def block_bootstrap(values, groups, fn=np.mean, n=1000, seed=0):
    return E.block_bootstrap(values, groups, fn=fn, n=n, seed=seed)


def spearman(x, y):
    return E.spearman(x, y)


def spearman_boot(x, y, groups, n=1000, seed=0):
    return E.spearman_boot(x, y, groups, n=n, seed=seed)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 8 else None


def host_info():
    g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(host=SERVER, hostname=g("hostname"), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), driver=g("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
                torch=torch.__version__, cuda=torch.version.cuda, git_commit=g("git rev-parse HEAD"), git_dirty=g("git status --porcelain | wc -l"), dirty_diff_sha16=hashlib.sha256(g("git diff").encode()).hexdigest()[:16])
