"""QEDGE9 — 고정 Teacher aligner 의 offset-consistency q 로 GT edge loss 의 적용 patch 를 고르는 gate (계획 research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md §2–§3·§6–§7).

q_T(i) = (1/2K) Σ_k ‖c^T_{i,k} + ε_k − c^T_{i,0}‖_1, K=16 = EQREC4 axis bank A (r ∈ {.25,.5,1,2} HR px × θ ∈ {0°,90°,180°,270°}); frozen T0 aligner 만 호출, 입력 adapter 는 Teacher forward 와 같다(view margin 4, M = bicubic↑4).
θq = train-only calibration view(3072 base patch × 4 augmentation state) 의 q 중앙값 · g_i = 1[q_T(i) < θq] (동일값은 0).
cue cache key: Teacher/A hash · dataset sha · sample index · 실제 augmentation state(hflip/vflip 고정 + rot 0..3) · probe bank · dtype — Student/seed/update 와 무관. 학습 loader 는 feeder(return_meta) 가 (index, rot) 를 그대로 준다(RNG 추가 소비 없음).
loss: L = L_H + L_K + λE·(1/B) Σ_i g_i E_i + m_t λoff L_off — Q12 의 edge 항을 **교체**(sum(g·E)/B; active 수로 재정규화하지 않는다). QEC: λE·c_E·mean_i E_i (c_E = Σ g E_pilot / Σ E_pilot, 고정 pilot). QES: g 를 (e_T_roi32 decile, aug state) stratum 안에서 고정 permutation(seed 51515) 으로 바꾼 것(stratum 별 active 수 보존).
자산: assets/qedge9/<name>.json(manifest) + .npz(view 별 q·c0·e_roi32·gate) — tools/qedge9_cue.py 가 만들고 hash 로 서버 간 전달한다. θq/c_E 가 없으면 학습을 시작하지 않는다(placeholder 0 금지)."""
import hashlib, json, math, os
import numpy as np, torch, torch.nn.functional as F
from pa.offset import predict_c
from pa.warp import warp_pan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ("low_q", "const", "shuffle")
RADII = (0.25, 0.5, 1.0, 2.0); ANGLES_DEG = (0, 90, 180, 270); BANK_ID = "EQREC4_A_AXIS16"; K_PROBES = 16
ROT_STATES = (0, 1, 2, 3); ROI = (16, 48); QES_SEED = 51515; QES_E_BINS = 10
FEEDER_CONTRACT_KEYS = ("hflip", "vflip", "rot", "crop")


def probe_bank():
    out = []
    for r in RADII:
        for th in ANGLES_DEG:
            a = math.radians(th); out.append(dict(probe_id=f"A_r{r}_t{th}", r=r, theta_deg=th, ey=round(r * math.sin(a), 12), ex=round(r * math.cos(a), 12)))
    return out


def q_const_component():
    """입력에 반응하지 않는 aligner(ĉε = ĉ0) 의 q = mean_k (|ε_y|+|ε_x|)/2 — 단위 확인용 상수 (HR px)."""
    return float(np.mean([(abs(p["ey"]) + abs(p["ex"])) / 2 for p in probe_bank()]))


def apply_view(x, hflip, vflip, rot):
    """[..., H, W] 텐서에 feeders.feeder.PanFeeder.augment 와 같은 순서·규약: hflip = W 축 반전([:, ::-1]), vflip = H 축 반전([::-1]), rot = np.rot90(k, (H, W)) ≡ torch.rot90(k, dims=(-2, -1)). 값은 바뀌지 않는다(위치만)."""
    if hflip:
        x = x.flip(-1)
    if vflip:
        x = x.flip(-2)
    if int(rot) % 4:
        x = torch.rot90(x, int(rot) % 4, dims=(-2, -1))
    return x.contiguous()


def to_feeder_tensor(arr, max_pixel):
    """feeder.np2tensor 와 같은 수치 경로: float64 배열 → float32 텐서 → 2x/max_pixel − 1 (float32)."""
    x = torch.Tensor(np.asarray(arr, dtype=np.float64)).mul_(1.0)
    return 2.0 * x / float(max_pixel) - 1.0


def edge_per_sample(y_hat, y):
    from pa.losses import output_edge_loss_per_sample
    return output_edge_loss_per_sample(y_hat, y)


@torch.no_grad()
def teacher_q(aligner, pan, ms, margin, chunk=256):
    """frozen Teacher aligner 의 q (bank AXIS16). pan [N,1,H,W], ms [N,C,h,w] (정규화; 임의 device) → (q [N] float64 cpu, c0 [N,2] float32 cpu). aligner 만 호출한다(U 없음)."""
    dev = next(aligner.parameters()).device; probes = probe_bank(); N = int(pan.shape[0]); q = torch.zeros(N, dtype=torch.float64); c0_all = torch.zeros(N, 2)
    E = torch.tensor([[p["ey"], p["ex"]] for p in probes], dtype=torch.float32, device=dev)
    for s in range(0, N, chunk):
        p = pan[s:s + chunk].to(dev).float(); m = ms[s:s + chunk].to(dev).float(); mb = F.interpolate(m, scale_factor=4, mode="bicubic")
        c0 = predict_c(aligner, p, mb, margin); acc = torch.zeros(p.shape[0], dtype=torch.float64, device=dev)
        for k in range(len(probes)):
            e = E[k][None].expand(p.shape[0], 2); ce = predict_c(aligner, warp_pan(p, e), mb, margin)
            acc += (ce - c0 + e).abs().sum(dim=1).double()                      # ‖ĉε + ε − ĉ0‖_1 (이상 aligner 면 0; λoff 없음)
        q[s:s + chunk] = (acc / (2.0 * len(probes))).cpu(); c0_all[s:s + chunk] = c0.float().cpu()
    return q, c0_all


def theta_from(q_cal):
    return float(np.median(np.asarray(q_cal, dtype=np.float64)))


def gate_low_q(q, theta):
    """g = 1[q < θq]; q == θq 는 0 (계획 §3.1)."""
    return (np.asarray(q, dtype=np.float64) < float(theta)).astype(np.int8)


def decile_edges(e_cal, bins=QES_E_BINS):
    return np.quantile(np.asarray(e_cal, dtype=np.float64), np.linspace(0, 1, bins + 1)[1:-1]).tolist()


def strata_of(e, rot, edges):
    d = np.searchsorted(np.asarray(edges, dtype=np.float64), np.asarray(e, dtype=np.float64), side="right").astype(np.int64)
    return d * len(ROT_STATES) + np.asarray(rot, dtype=np.int64), d.astype(np.int8)


def shuffle_gate(gate, strata, seed=QES_SEED):
    """stratum(e decile × aug state) 안에서 gate 를 고정 permutation 으로 교환 — active 수 보존; 원소 < 2 인 stratum 과 라벨이 하나뿐인 stratum 은 그대로(비율 기록)."""
    rng = np.random.RandomState(int(seed)); out = np.asarray(gate, dtype=np.int8).copy(); stats = dict(n_strata=0, n_shuffled=0, n_degenerate=0, n_small=0, n_views_shuffled=0)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]; stats["n_strata"] += 1
        if len(idx) < 2:
            stats["n_small"] += 1; continue
        if out[idx].min() == out[idx].max():
            stats["n_degenerate"] += 1; continue
        out[idx] = out[idx][rng.permutation(len(idx))]; stats["n_shuffled"] += 1; stats["n_views_shuffled"] += int(len(idx))
    return out, stats


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


NORMALIZATION = "feeder.np2tensor: float64→float32, 2x/max_pixel−1 (float32)"
ASSET_ID_FIELDS = ("npz_sha256", "teacher.aligner_state_hash", "teacher.file_sha256", "teacher.aligner_view_margin", "dataset.train_sha256", "dataset.train_pan_sha256", "dataset.n_train", "dataset.normalization",
                   "feeder_contract.hflip", "feeder_contract.vflip", "feeder_contract.rot", "feeder_contract.crop", "bank.id", "bank.K", "theta_q", "calibration.index_sha256_16", "calibration.seed", "qes.seed", "qes.bins")


def _get(man, dotted):
    v = man
    for k in dotted.split("."):
        v = (v or {}).get(k) if isinstance(v, dict) else None
    return v


def asset_id(man):
    """cue 자산의 불변 ID (감사 F03): q 의 출처(Teacher A/파일/margin)·데이터·정규화·feeder 계약·bank·θq·calibration id·QES seed·npz sha 를 묶은 sha256[:32]. 재개 시 checkpoint 의 값과 대조한다."""
    payload = {k: _get(man, k) for k in ASSET_ID_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def check_asset(man, z, perm_seed=QES_SEED):
    """자산의 내부 일관성(§7.1 key·§3.1·§6.2): bank 정의 · θq 유한·gate == 1[q<θq] · calibration id 재계산 · coverage(index×rot 완비) · QES 재현(고정 permutation) · 정규화 문자열 · asset_id 일치. 위반 목록을 돌려준다(비면 통과)."""
    bad = []
    b = man.get("bank") or {}
    if b.get("id") != BANK_ID or int(b.get("K", 0)) != K_PROBES or [float(x) for x in (b.get("radii_hr") or [])] != list(RADII) or [int(x) for x in (b.get("angles_deg") or [])] != list(ANGLES_DEG):
        bad.append(f"bank {b.get('id')}/{b.get('K')}/{b.get('radii_hr')}/{b.get('angles_deg')} ≠ {BANK_ID}/{K_PROBES}/{RADII}/{ANGLES_DEG}")
    th = man.get("theta_q")
    if th is None or not math.isfinite(float(th)) or not (0.0 < float(th) < 10.0):
        bad.append(f"theta_q {th} 가 유한한 HR px 값이 아니다")
    q = z["q"]; g = z["gate_low_q"]; idx = z["index"].astype(np.int64); rot = z["rot"].astype(np.int64); N = int(_get(man, "dataset.n_train") or 0)
    if th is not None and math.isfinite(float(th)) and not np.array_equal(g, (q < float(th)).astype(np.int8)):
        bad.append("gate_low_q ≠ 1[q < theta_q] (θq·q·label 내부 모순)")
    if len(idx) != N * len(ROT_STATES) or set(np.unique(rot).tolist()) != set(ROT_STATES) or len(set(zip(idx.tolist(), rot.tolist()))) != N * len(ROT_STATES):
        bad.append(f"coverage: {len(idx)} view ≠ N {N} × {len(ROT_STATES)} state 또는 (index, rot) 중복")
    fc = man.get("feeder_contract") or {}
    if int(fc.get("states", 0)) != len(ROT_STATES) or [int(x) for x in (fc.get("rot_states") or [])] != list(ROT_STATES):
        bad.append(f"feeder_contract states {fc.get('states')}/{fc.get('rot_states')}")
    if _get(man, "dataset.normalization") != NORMALIZATION:
        bad.append(f"normalization {_get(man, 'dataset.normalization')!r} ≠ {NORMALIZATION!r}")
    cal = man.get("calibration") or {}
    if N > 0:
        gen = torch.Generator(device="cpu"); gen.manual_seed(int(cal.get("seed")) if cal.get("seed") is not None else 0); n_b = min(int(cal.get("n_base") or 0), N)
        ids = torch.randperm(N, generator=gen)[:n_b].sort().values.numpy().astype(np.int64); sha = hashlib.sha256(ids.tobytes()).hexdigest()[:16]
        if cal.get("seed") != 1234 or sha != cal.get("index_sha256_16"):
            bad.append(f"calibration id 재계산 {sha} ≠ manifest {cal.get('index_sha256_16')} (seed {cal.get('seed')})")
        cm = z["calib_mask"].astype(bool)
        if int(cm.sum()) != n_b * len(ROT_STATES) or not np.array_equal(np.unique(idx[cm]), ids):
            bad.append("calib_mask 가 calibration id × 4 state 와 다르다")
    qes = man.get("qes") or {}; edges = (man.get("e_roi32") or {}).get("decile_edges")
    if int(qes.get("seed", -1)) != int(perm_seed) or edges is None:
        bad.append(f"qes seed {qes.get('seed')} ≠ {perm_seed} 또는 decile edges 없음")
    else:
        strata, dec = strata_of(z["e_roi32"], rot, edges); g_sh, _ = shuffle_gate(g, strata, int(perm_seed))
        if not np.array_equal(g_sh, z["gate_shuffle"]) or not np.array_equal(dec, z["e_decile"].astype(np.int8)):
            bad.append("gate_shuffle ≠ 고정 permutation 재현 (또는 e_decile 불일치)")
    if man.get("asset_id") != asset_id(man):
        bad.append(f"asset_id {man.get('asset_id')} ≠ 재계산 {asset_id(man)}")
    return bad


def read_asset(asset_json, strict=True, perm_seed=QES_SEED):
    """manifest + npz. strict: npz sha + 내부 일관성(check_asset) 위반이면 ValueError."""
    man = json.load(open(_abs(asset_json))); npz_p = _abs(man["npz"])
    if sha256_file(npz_p) != man["npz_sha256"]:
        raise ValueError(f"cue npz sha 불일치: {npz_p}")
    z = np.load(npz_p)
    if strict:
        bad = check_asset(man, z, perm_seed)
        if bad:
            raise ValueError("cue 자산 내부 일관성 위반: " + " · ".join(bad))
    return man, z


class EdgeGate:
    """학습 시 (index, rot) → g 조회 (low_q/shuffle) 또는 상수 c_E (const). 자산 검증은 load() 가 한다."""

    def __init__(self, mode, table=None, c_E=None, manifest=None, asset=None, ce_info=None, checks=None):
        if mode not in MODES:
            raise ValueError(f"edge_gate mode {mode!r} ∉ {MODES}")
        self.mode = mode; self.table = table; self.c_E = c_E; self.manifest = manifest or {}; self.asset = asset; self.ce_info = ce_info or {}; self.checks = checks or {}
        self._dev_table = {}

    @property
    def needs_meta(self):
        return self.mode in ("low_q", "shuffle")

    @property
    def theta_q(self):
        return (self.manifest.get("theta_q") if self.manifest else None)

    @classmethod
    def load(cls, cfg, teacher=None, train_h5=None, feeder_args=None, sha_fn=None, previous=None):
        """cfg = registry 의 edge_gate spec. teacher/train_h5/feeder_args 가 있으면 대조한다. previous: 재개 시 이 run 이 전에 기록한 edge_gate 요약(kdv_config_resolved.json) — asset_id/c_E 가 다르면 거부 (감사 F03)."""
        mode = cfg["mode"]; asset = cfg.get("asset"); checks = {}
        man, z = read_asset(asset, strict=True, perm_seed=int(cfg.get("perm_seed") or QES_SEED)); aid = asset_id(man); checks["asset_id"] = aid
        if teacher is not None:
            from kdv.teacher_assets import state_hash
            if teacher.aligner is None:
                raise ValueError("edge_gate: Teacher 에 aligner 가 없다 — q 는 T0 aligner 의 것")
            got = state_hash(teacher.aligner); want = man["teacher"]["aligner_state_hash"]
            if got != want:
                raise ValueError(f"edge_gate: cue 의 Teacher aligner hash {want} ≠ 이 run 의 Teacher {got} — q 출처가 다르다(§2.1 q 는 고정 T0 A)")
            checks["teacher_aligner_hash"] = got
            mg = int(getattr(teacher, "aligner_margin", -1))
            if mg != int(man["teacher"]["aligner_view_margin"]):
                raise ValueError(f"edge_gate: cue 의 Teacher view margin {man['teacher']['aligner_view_margin']} ≠ 이 run 의 Teacher {mg} — 같은 A 라도 입력 adapter 가 다르면 q 가 다르다(§7.3)")
            checks["teacher_view_margin"] = mg
        if previous is not None:
            pa = (previous or {}).get("asset_id")
            if pa and pa != aid:
                raise ValueError(f"edge_gate: 재개 run 의 이전 cue asset_id {pa} ≠ 현재 {aid} — 재개 중 gate 가 바뀌었다 (§8.2 Resume)")
            checks["previous_asset_id"] = pa
        if train_h5 is not None and sha_fn is not None:
            got = sha_fn(train_h5); want = man["dataset"]["train_sha256"]
            if got != want:
                raise ValueError(f"edge_gate: cue 의 train h5 sha {want[:16]} ≠ 이 run 의 {got[:16]}")
            checks["train_sha256"] = got
        if feeder_args is not None:
            fa = {k: bool(feeder_args.get(k, False)) for k in FEEDER_CONTRACT_KEYS}; want = {k: bool(man["feeder_contract"].get(k, False)) for k in FEEDER_CONTRACT_KEYS}
            if fa != want:
                raise ValueError(f"edge_gate: feeder augmentation 계약 {fa} ≠ cue {want} (상태 수·입력 경로가 다르면 cache 를 다시 만든다, §7.2)")
            checks["feeder_contract"] = fa
        idx = z["index"].astype(np.int64); rot = z["rot"].astype(np.int64); N = int(man["dataset"]["n_train"])
        if mode in ("low_q", "shuffle"):
            g = z["gate_low_q" if mode == "low_q" else "gate_shuffle"].astype(np.float32)
            table = torch.full((N, len(ROT_STATES)), float("nan"), dtype=torch.float32); table[torch.from_numpy(idx), torch.from_numpy(rot)] = torch.from_numpy(g)
            if bool(torch.isnan(table).any()):
                raise ValueError("edge_gate: cue 가 모든 (index, rot) view 를 덮지 않는다")
            return cls(mode, table=table, manifest=man, asset=asset, checks=checks)
        ce_info = {}
        if cfg.get("c_E_file"):
            p = _abs(cfg["c_E_file"])
            if not os.path.exists(p):
                raise FileNotFoundError(f"edge_gate const: c_E 파일 없음 {cfg['c_E_file']} — tools/qedge9_cue.py pilot 으로 먼저 만든다 (placeholder 금지)")
            ce_info = json.load(open(p))
            if ce_info.get("cue_npz_sha256") != man["npz_sha256"] or ce_info.get("cue_asset_id") != aid:
                raise ValueError(f"edge_gate const: c_E 가 다른 cue 자산에서 계산됐다 (npz {str(ce_info.get('cue_npz_sha256'))[:16]} / asset_id {ce_info.get('cue_asset_id')} ≠ {aid})")
            for key, want in (("pilot_run", cfg.get("pilot_run")), ("pilot_tag", cfg.get("pilot_tag")), ("pilot_step", cfg.get("pilot_step"))):     # 감사 F02: pilot identity 필수 대조
                if want is not None and ce_info.get(key) != want:
                    raise ValueError(f"edge_gate const: c_E 의 {key} {ce_info.get(key)!r} ≠ config {want!r} (§6.1 pilot = W104 J0 S1234 exact50K)")
            if not ce_info.get("pilot_file_sha256") or not ce_info.get("pilot_sha256_16") or ce_info.get("views") != "calibration":
                raise ValueError("edge_gate const: c_E 파일에 pilot checkpoint hash / calibration view 표기가 없다")
            c_E = float(ce_info["c_E"])
        else:
            c_E = float(cfg["c_E"])
        if not (math.isfinite(c_E) and 0.0 < c_E <= 1.0):
            raise ValueError(f"edge_gate const: c_E {c_E} 는 (0, 1] 이어야 한다 (분모 퇴화면 calibration 실패로 처리)")
        if previous is not None and (previous or {}).get("c_E") is not None and abs(float(previous["c_E"]) - c_E) > 1e-12:
            raise ValueError(f"edge_gate const: 재개 run 의 이전 c_E {previous['c_E']} ≠ 현재 {c_E}")
        return cls(mode, c_E=c_E, manifest=man, asset=asset, ce_info=ce_info, checks=checks)

    @classmethod
    def synthetic(cls, mode, table=None, c_E=None):
        """검사용: 자산 없이 직접 준 표/상수."""
        return cls(mode, table=table, c_E=c_E, manifest=dict(theta_q=None, synthetic=True))

    def gate_for(self, meta, device):
        """meta [B,4] int64 = (index, rot, hflip, vflip) → g [B] float32 (device). low_q/shuffle 전용."""
        if meta is None:
            raise ValueError("edge_gate low_q/shuffle 는 batch meta(index, rot) 가 필요하다 — train_feeder_args.return_meta: true")
        key = str(device)
        if key not in self._dev_table:
            self._dev_table[key] = self.table.to(device)
        t = self._dev_table[key]; m = meta.to(device).long()
        return t[m[:, 0], m[:, 1]]

    def summary(self):
        man = self.manifest or {}
        return dict(mode=self.mode, asset=self.asset, asset_id=man.get("asset_id"), npz_sha256=man.get("npz_sha256"), theta_q=man.get("theta_q"), bank=man.get("bank", {}).get("id") if isinstance(man.get("bank"), dict) else man.get("bank"),
                    q_source="T0 aligner (frozen)", q_cut="median(train calibration views)", edge_gate=("low_q" if self.mode == "low_q" else ("shuffle(perm seed %s)" % man.get("qes", {}).get("seed") if self.mode == "shuffle" else f"const c_E={self.c_E}")),
                    hard_always=1, selection=man.get("selection"), qes=man.get("qes"), c_E=self.c_E, c_E_source=({k: self.ce_info.get(k) for k in ("pilot_run", "pilot_tag", "pilot_step", "pilot_sha256_16", "pilot_file_sha256", "n_views", "views", "computed_at", "server")} if self.ce_info else None),
                    checks=self.checks, teacher=man.get("teacher"), computed_at=man.get("computed_at"), source_server=man.get("server"))
