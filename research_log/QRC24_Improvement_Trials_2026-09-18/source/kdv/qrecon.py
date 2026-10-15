"""QRECON24 — 연속 q 가중 (계획 research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §2.3–§2.5, §4.2, §6).

w_i = sg( qref / (qref + q_T(i)) ),  qref = 0.3276133416220546 (T0/train/AXIS16 calibration 중앙값을 scale 로 재사용; threshold 아님).
**2026-09-16 저녁 사용자 결정: 분자의 2 를 뺀다** — 계획서(§2.3 "2qref/(qref+q)", case_registry 의 twice_median_over_median_plus_q) 에는 2 가 남아 있지만 구현은 qref/(qref+q) 다.
(w(qref) = 0.5, w(0) = 1, 0 < w ≤ 1; 실제 자산에서 w 0.397–0.547, 평균 0.498. s1 첫 run PAKD50_QRC24_S1_G22_…_S1234 은 2 가 있던 코드(15204de) 로 시작했다 — 노트 §9.)
같은 결정의 환산(사용자 §2–§3): λE 는 종전 값의 2 배(3e-4/1e-3/3e-3 → 6e-4/2e-3/6e-3; λE_new·s_q = λE_old·w_q 로 같은 출력에서 edge loss/gradient 보존) · rA/α/β/qref/seed 유지 · A 는 L_A = mean(s_q·H)(보상 계수 없음)
· **uniform 대조의 가중치는 1 이 아니라 s_q(qref) = 0.5**(q 를 뺀 대조에서 총 강도가 두 배가 되지 않게) · shuffle 은 새 연속 가중치를 그대로 섞는다.
q_T 는 cue 자산(assets/qedge9/cue_T0_AXIS16_v1.{json,npz}) 의 **raw q**(sample index × 실제 rot state) — 기존 gate_low_q 0/1 표에 함수를 씌우지 않는다.
q ≥ 0·유한이 아니면 fail-fast, w(qref) = 0.5, w(0) = 1, 0 < w ≤ 1. batch 정규화·threshold·temperature·clip 없음.

두 목적함수(§2.5):  L_U = mean_i[H_i + K_i + λE·w^E_i·E_i]  (U 만)   ·   L_A = mean_i[w^A_i·H_i]  (A 만; soft/edge/offset 없음)
w^A / w^E 는 각각 q | uniform(=1) | shuffle(연속 w 를 (Teacher e_roi32 decile × rot) stratum 안에서 고정 permutation 51515 로 교환; 값 multiset 보존, sample 연결만 제거).
"""
import hashlib, os
import numpy as np, torch
from kdv.edge_gate import EdgeGate, read_asset, asset_id, ROT_STATES, QES_SEED

MODES = ("q", "uniform", "shuffle")
QREF_DEFAULT = 0.3276133416220546
UNIFORM_WEIGHT = 0.5                                                       # uniform 대조 = s_q(qref) = qref/(qref+qref) = 0.5 (사용자 §3; 종전 계획의 1 아님)


def q_weight(q, qref):
    """w = qref/(qref + q) (float64; 분자의 2 없음 — 2026-09-16 저녁 결정). q 는 유한·비음, qref > 0 — 아니면 ValueError (조용히 clip 하지 않는다)."""
    qref = float(qref)
    if not (np.isfinite(qref) and qref > 0):
        raise ValueError(f"qref 는 양의 유한값이어야 한다 — {qref!r}")
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("q 에 NaN/Inf 가 있다 — cue 자산을 다시 만든다 (placeholder 금지)")
    if np.any(q < 0):
        raise ValueError("q < 0 인 view 가 있다 — q 는 L1 잔차 평균이라 비음이어야 한다")
    return qref / (qref + q)


def shuffle_values(vals, strata, seed=QES_SEED):
    """연속값을 stratum 안에서 고정 permutation 으로 교환 (kdv.edge_gate.shuffle_gate 와 같은 규약; 값 multiset 은 stratum 마다 보존).
    원소 < 2 인 stratum 과 모든 값이 같은(degenerate) stratum 은 그대로. 반환 (out, stats)."""
    rng = np.random.RandomState(int(seed)); out = np.asarray(vals, dtype=np.float64).copy(); strata = np.asarray(strata)
    stats = dict(n_strata=0, n_shuffled=0, n_degenerate=0, n_small=0, n_views_shuffled=0, seed=int(seed))
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]; stats["n_strata"] += 1
        if len(idx) < 2:
            stats["n_small"] += 1; continue
        if out[idx].min() == out[idx].max():
            stats["n_degenerate"] += 1; continue
        out[idx] = out[idx][rng.permutation(len(idx))]; stats["n_shuffled"] += 1; stats["n_views_shuffled"] += int(len(idx))
    return out, stats


def _sha_full(arr, dtype=np.float64):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, dtype=dtype)).tobytes()).hexdigest()


def _sha16(arr):
    return _sha_full(arr)[:16]


class QWeight:
    """(index, rot) → w 조회. tables: {"q": [N,4] float32, "shuffle": [N,4]} · uniform 은 1. 자산 검증(asset_id·Teacher A hash/margin·train sha·feeder 계약·재개 대조) 은 EdgeGate.load 와 같다."""

    def __init__(self, qref, tables, a_mode, e_mode, manifest=None, asset=None, checks=None, stats=None, uniform_w=UNIFORM_WEIGHT):
        for m in (a_mode, e_mode):
            if m not in MODES:
                raise ValueError(f"qrecon weight mode {m!r} ∉ {MODES}")
        if abs(float(uniform_w) - UNIFORM_WEIGHT) > 1e-12:
            raise ValueError(f"qrecon uniform 가중치는 s_q(qref) = {UNIFORM_WEIGHT} 로 고정 (현재 {uniform_w!r})")
        self.qref = float(qref); self.tables = tables; self.a_mode = a_mode; self.e_mode = e_mode; self.manifest = manifest or {}; self.asset = asset; self.checks = checks or {}; self.stats = stats or {}; self.uniform_w = float(uniform_w)
        self._dev = {}

    @classmethod
    def load(cls, cfg, teacher=None, train_h5=None, feeder_args=None, sha_fn=None, previous=None):
        """cfg = registry 의 qrecon spec (q_ref, asset, a_weight, e_weight, perm_seed). previous: 재개 시 이 run 이 전에 기록한 qrecon 요약 — asset_id / w 표 sha / qref 가 다르면 거부."""
        asset = cfg["asset"]; qref = float(cfg.get("q_ref", QREF_DEFAULT)); seed = int(cfg.get("perm_seed", QES_SEED) or QES_SEED)
        prev_gate = None if previous is None else dict(asset_id=(previous or {}).get("asset_id"))
        gate = EdgeGate.load(dict(mode="low_q", asset=asset), teacher=teacher, train_h5=train_h5, feeder_args=feeder_args, sha_fn=sha_fn, previous=prev_gate)     # identity checks only
        man, z = read_asset(asset, strict=True, perm_seed=seed)
        idx = z["index"].astype(np.int64); rot = z["rot"].astype(np.int64); N = int(man["dataset"]["n_train"])
        w = q_weight(z["q"], qref)
        strata = (z["e_decile"].astype(np.int64) * len(ROT_STATES) + rot)                     # (Teacher e_roi32 decile × 실제 rot state) — QES 와 같은 stratum
        w_sh, sh_stats = shuffle_values(w, strata, seed)
        perm_map = shuffle_values(np.arange(len(w), dtype=np.float64), strata, seed)[0].astype(np.int64)      # 같은 permutation 의 index 사상(view i ← view perm_map[i]) — full sha 기록용 (감사 F10)
        tables = {}
        for name, arr in (("q", w), ("shuffle", w_sh)):
            t = torch.full((N, len(ROT_STATES)), float("nan"), dtype=torch.float32); t[torch.from_numpy(idx), torch.from_numpy(rot)] = torch.from_numpy(arr.astype(np.float32))
            if bool(torch.isnan(t).any()):
                raise ValueError("qrecon: cue 가 모든 (index, rot) view 를 덮지 않는다")
            tables[name] = t
        cm = z["calib_mask"].astype(bool)
        stats = dict(qref=qref, theta_q_asset=man.get("theta_q"), q_min=float(z["q"].min()), q_max=float(z["q"].max()), q_mean=float(z["q"].mean()), w_min=float(w.min()), w_max=float(w.max()), w_mean_all=float(w.mean()),
                     w_mean_calib=float(w[cm].mean()) if cm.any() else None, w_at_qref=0.5, numerator_factor=1.0, w_shuffle_mean_all=float(w_sh.mean()), w_sha256_16=_sha16(w), w_shuffle_sha256_16=_sha16(w_sh), shuffle=sh_stats,
                     w_sha256=_sha_full(w), w_shuffle_sha256=_sha_full(w_sh), permutation_sha256=_sha_full(perm_map, np.int64), q_raw_sha256=_sha_full(z["q"]),
                     multiset_preserved=bool(np.allclose(np.sort(w), np.sort(w_sh))), n_views=int(len(w)))
        checks = dict(gate.checks, asset_id=asset_id(man))
        if previous:
            for key, want in (("w_sha256_16", stats["w_sha256_16"]), ("w_shuffle_sha256_16", stats["w_shuffle_sha256_16"])):
                pv = ((previous.get("stats") or {}).get(key)) if isinstance(previous.get("stats"), dict) else previous.get(key)
                if pv and pv != want:
                    raise ValueError(f"qrecon: 재개 run 의 이전 {key} {pv} ≠ 현재 {want} — 재개 중 w 표가 바뀌었다")
            if previous.get("qref") is not None and abs(float(previous["qref"]) - qref) > 1e-15:
                raise ValueError(f"qrecon: 재개 run 의 이전 qref {previous['qref']} ≠ 현재 {qref}")
        return cls(qref, tables, cfg.get("a_weight", "q"), cfg.get("e_weight", "q"), manifest=man, asset=asset, checks=checks, stats=stats, uniform_w=float(cfg.get("uniform_weight", UNIFORM_WEIGHT)))

    @classmethod
    def synthetic(cls, table_q, table_shuffle=None, qref=QREF_DEFAULT, a_mode="q", e_mode="q", uniform_w=UNIFORM_WEIGHT):
        """검사용: (index, rot) 표를 직접 준다 (w 값 자체)."""
        return cls(qref, {"q": table_q, "shuffle": (table_shuffle if table_shuffle is not None else table_q)}, a_mode, e_mode, manifest=dict(synthetic=True), uniform_w=uniform_w)

    def _table(self, name, device):
        key = (name, str(device))
        if key not in self._dev:
            self._dev[key] = self.tables[name].to(device)
        return self._dev[key]

    def weight_for(self, meta, device, mode):
        """meta [B,4] int64 = (index, rot, hflip, vflip) → w [B] float32 (sg). mode ∈ MODES; uniform 은 s_q(qref) = 0.5 (사용자 §3)."""
        if mode == "uniform":
            if meta is None:
                raise ValueError("qrecon: batch meta(index, rot) 가 필요하다 — train_feeder_args.return_meta: true")
            return torch.full((int(meta.shape[0]),), self.uniform_w, dtype=torch.float32, device=device)
        if meta is None:
            raise ValueError("qrecon: batch meta(index, rot) 가 필요하다 — train_feeder_args.return_meta: true")
        t = self._table("q" if mode == "q" else "shuffle", device); m = meta.to(device).long()
        return t[m[:, 0], m[:, 1]].detach()

    def weights(self, meta, device):
        return self.weight_for(meta, device, self.a_mode), self.weight_for(meta, device, self.e_mode)

    def summary(self):
        man = self.manifest or {}
        return dict(mode="continuous_v1", formula="w_i = qref/(qref + q_T(i)); q_T = frozen T0 aligner AXIS16 raw q (cue asset), sg", formula_note="2026-09-16 저녁 사용자 결정: 분자의 2 제거 (계획 §2.3 의 2qref/(qref+q) 아님; w(qref)=0.5); λE 는 2 배 환산(6e-4/2e-3/6e-3), uniform 대조 = s_q(qref) = 0.5", numerator_factor=1.0, uniform_weight=self.uniform_w, qref=self.qref, a_weight=self.a_mode, e_weight=self.e_mode, asset=self.asset,
                    asset_id=man.get("asset_id"), npz_sha256=man.get("npz_sha256"), q_source="T0 aligner (frozen) raw q", A_loss="weighted_H_only", A_soft=0, A_edge=0, student_offset=0, stats=self.stats, checks=self.checks,
                    teacher=man.get("teacher"), computed_at=man.get("computed_at"), source_server=man.get("server"))
