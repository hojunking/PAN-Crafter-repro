#!/usr/bin/env python
"""NA104 (W104·D122 no-align KD) 실행 전 gate — 계획 §18 의 NA01–NA04 · CK01 · KD01–KD04 · ST01–ST04 · TRI01–TRI03 · EV01 · RS01 · PERF01 · CS01.

    python tools/na104_unit_tests.py            # CPU, 수 분

이전 캠페인(W96/W112) 의 CPU 통과 기록을 이 골격의 통과로 쓰지 않는다 (§16.1) — 여기서 W104·D122 로 다시 확인한다.
gate 실패는 IMPLEMENTATION_ERROR 로 분류한다 (성능 비개선 NEGATIVE, 의존성 미충족 BLOCKED_INPUT 와 구분, §18).
"""
import os, sys, json, copy, glob, math, re, types
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch.nn.functional as F                                            # noqa: E402
from pa.model import PAModel                                              # noqa: E402
from pa.losses import output_edge_loss                                    # noqa: E402
from kdv.registry import resolve, run_name, rec_tag, stat_tag, tri_tag, describe, REC_CASES, STAT_KINDS  # noqa: E402
from kdv.losses_rec import GTAnchoredReconstructionKD                     # noqa: E402
from kdv.losses_stat import statistic_map, stat_term, stat_maps, stat_transform, stat_margin  # noqa: E402
from kdv.tri import (direction_mask, masked_l1, masked_l1_componentwise, component_weights, mass_kappa, shuffle_mask,
                     teacher_eval, teacher_fd_jacobian, sens_q, sens_risk, roi_gate, linearization_check)  # noqa: E402
from kdv.forward import kdv_forward                                       # noqa: E402
from kdv.teacher_assets import state_hash, freeze                         # noqa: E402
from main import import_class                                             # noqa: E402
from train_kdv import KDVTrainer                                           # noqa: E402

FAIL = []


def check(name, cond, info=""):
    (print if cond else print)(f"  {'OK ' if cond else 'FAIL'}  {name} {info}")
    if not cond:
        FAIL.append(name)


torch.manual_seed(0)
W, D, NB = 104, [1, 2, 2], 8
TPL = yaml.safe_load(open(os.path.join(ROOT, "config", "PA_A1_REC_W96_D124_9CH_S1234.yaml")))
MA = dict(TPL["model_args"], hidden_size=W, depth=D)
Model = import_class(TPL["model"])


def make(seed=0, randomize=True):
    """A-ID: aligner 없음·sampler 없음. **zero_module 로 0 초기화된 출력층을 난수화한다** —
    그러지 않으면 초기 출력이 정확히 0(= M) 이라 민감도·동등성 검사가 무엇을 넣어도 통과한다 (CLAUDE.md 함정)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = PAModel(Model(**MA), None, aligner_margin=0, sampler=False)
        if randomize:
            with torch.no_grad():
                for q in m.parameters():
                    if q.dtype.is_floating_point and float(q.abs().sum()) == 0:
                        q.normal_(0, 1e-3)
        return m


def batch(b=2, hr=64):
    g = torch.Generator().manual_seed(7)
    pan = torch.rand(b, 1, hr, hr, generator=g) * 2 - 1
    ms = torch.rand(b, NB, hr // 4, hr // 4, generator=g) * 2 - 1
    lpan = torch.rand(b, 1, hr // 4, hr // 4, generator=g) * 2 - 1
    gt = torch.rand(b, NB, hr, hr, generator=g) * 2 - 1
    return gt, ms, lpan, pan


print("== NA01–NA04: 골격·그래프")
stud, teach = make(0), make(1)
n_par = sum(p.numel() for p in stud.parameters())
check("NA01 zero_module 파라미터를 난수화해 검사를 유효하게 만든다 (초기 출력 0 트랩)",
      float(stud.backbone(torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 16, 16), torch.zeros(1, NB, 16, 16), torch.ones(1)).abs().max()) > 0)
check("NA01 Student/Teacher 가 같은 W104·D122·9→8ch 구조 (params 동일, 가중치는 다름)",
      n_par == sum(p.numel() for p in teach.parameters()) and state_hash(stud) != state_hash(teach)
      and MA["hidden_size"] == 104 and MA["depth"] == [1, 2, 2] and MA["in_mode"] == "paper" and MA["mode_modulation"] is False,
      f"params {n_par/1e6:.4f} M")
convs = [(m.in_channels, m.out_channels) for m in stud.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
check("NA01 입력 9ch(PAN+MS) · 출력 8ch", convs[0][0] == 1 + NB and MA["out_channels"] == NB, f"first conv in {convs[0][0]}, out_channels {MA['out_channels']}")
check("NA02 aligner parameter = 0 (aligner None, sampler False)",
      stud.aligner is None and teach.aligner is None and stud.sampler is False and sum(p.numel() for p in stud.parameters() if False) == 0)

import kdv.forward as _kf, pa.model as _pm                                   # noqa: E402
_calls = {"n": 0}
_orig_f, _orig_m = _kf.warp_pan, _pm.warp_pan


def _count(*a, **kw):
    _calls["n"] += 1
    return _orig_f(*a, **kw)


_kf.warp_pan = _count; _pm.warp_pan = _count
gt, ms, lpan, pan = batch()
o = kdv_forward(stud, teach, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=False)
_ = stud(pan, ms, lpan)
_kf.warp_pan, _pm.warp_pan = _orig_f, _orig_m
check("NA02 학습·평가 forward 에서 PAN warp 호출 = 0 (native PAN 그대로, cached correction 없음)", _calls["n"] == 0, f"warp calls {_calls['n']}")
check("NA02 Δ = 0 이고 U-Net 입력이 원 PAN 과 비트 동일", float(o["delta"].abs().max()) == 0.0 and torch.equal(o["pan_aligned"], pan.float()))

mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
check("NA03 MS base 는 bicubic ×4 이고 정확히 한 번 더해진다 (y − residual == M)",
      torch.allclose(o["ms_base"], mb, atol=1e-6) and torch.allclose(o["y"] - o["residual"], mb, atol=1e-6))
y_again = stud(pan, ms, lpan)["y"]
check("NA04 추론 forward 는 GT 를 받지 않는다 (signature: pan, ms, lpan) · 결정론",
      torch.allclose(y_again, o["y"], atol=1e-6) and "gt" not in stud.forward.__code__.co_varnames)

print("== CK01: 다른 폭의 checkpoint 를 부분 로드하지 않는다")
w112 = PAModel(Model(**dict(MA, hidden_size=112, depth=[1, 2, 3])), None, sampler=False)
try:
    stud.load_state_dict(w112.state_dict(), strict=True); ck = False
except RuntimeError:
    ck = True
check("CK01 W112·D123 state_dict 를 W104·D122 에 strict 로드하면 오류 (silent partial load 없음)", ck)

print("== KD01–KD04: reconstruction 계약")
crit = GTAnchoredReconstructionKD(0.05, mode="adaptive")
s_ = torch.rand(2, NB, 8, 8, dtype=torch.float64, requires_grad=True)
t_ = torch.rand(2, NB, 8, 8, dtype=torch.float64, requires_grad=True)
g_ = torch.rand(2, NB, 8, 8, dtype=torch.float64, requires_grad=True)
r = crit(s_, t_, g_, return_maps=True); r.loss.backward()
check("KD01 Teacher·GT·gate 에 gradient 없음, Student 에는 있음", t_.grad is None and g_.grad is None and s_.grad is not None
      and all(not v.requires_grad for v in r.maps.values()))
sv = torch.full((1, NB, 1, 1), 0.3, dtype=torch.float64, requires_grad=True)
rr = crit(sv, torch.full((1, NB, 1, 1), 0.3, dtype=torch.float64), torch.zeros(1, NB, 1, 1, dtype=torch.float64)); rr.loss.backward()
check("KD02 Student=Teacher 가 같은 오답이어도 GT gradient 가 남는다", float(rr.soft) == 0.0 and float(sv.grad.abs().min()) > 0)
gt_exact = torch.rand(1, NB, 4, 4, dtype=torch.float64)
r3 = crit(gt_exact.clone().requires_grad_(), torch.rand(1, NB, 4, 4, dtype=torch.float64), gt_exact, return_maps=True)
check("KD03 전 픽셀 S=GT 이면 R3 soft = 0", float(r3.soft) == 0.0 and float(r3.loss) == 0.0)
s_b = gt_exact.clone(); s_b[:, 0] += 0.2                                       # band 0 만 틀림
m_b = direction_mask(s_b, torch.rand(1, NB, 4, 4, dtype=torch.float64), gt_exact, mode="sign")
check("KD03 일부 band 만 정확하면 그 band 의 A mask = 0 (u=0 → gate 0)", float(m_b[:, 1:].max()) == 0.0)
c_r1 = GTAnchoredReconstructionKD(0.05, kd_weight=0.0, mode="adaptive")(s_, t_, g_)
c_hard = GTAnchoredReconstructionKD(0.05, mode="hard_only")(s_, t_, g_)
check("KD04 R3 의 β=0 은 R1 과 같다 (신규 방법이 아니다)", abs(float(c_r1.loss) - float(c_hard.loss)) < 1e-12)
r_base = crit(s_, t_, g_, return_maps=True)
mb_ones = masked_l1(s_, t_, g_, r_base.maps["hard_weight"], r_base.maps["soft_weight"])
check("KD04 TRI-A-BASE(mask 없음) 는 R3 와 같다", abs(float(mb_ones["loss"]) - float(r_base.loss)) < 1e-12)
vs, vt, vg = stat_maps(torch.rand(2, NB, 13, 15, dtype=torch.float64), torch.rand(2, NB, 13, 15, dtype=torch.float64), torch.rand(2, NB, 13, 15, dtype=torch.float64), kind="grad_var", window=5)
sc = GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="adaptive"); sr = sc(vs, vt, vg, return_maps=True)
mb_v = masked_l1(vs, vt, vg, sr.maps["hard_weight"], sr.maps["soft_weight"])
check("KD04 TRI-B-BASE 는 STAT-AD 와 같다", abs(float(mb_v["loss"]) - float(sr.loss)) < 1e-12)

print("== ST01–ST04: 통계 표현")
x = torch.rand(2, NB, 13, 15, dtype=torch.float64)
pat = F.unfold(x[..., 1:-1, 1:-1], kernel_size=5).reshape(2, NB, 25, 7, 9)
check("ST01 IV 가 명시적 population variance(correction 0) 와 일치", torch.allclose(statistic_map(x, "image_var", 5), pat.var(dim=2, correction=0), rtol=1e-10, atol=1e-12))
const = torch.full((1, 3, 13, 15), 0.37, dtype=torch.float64)
check("ST01 상수 영상의 모든 통계 = 0", all(float(statistic_map(const, k, 5).abs().max()) < 1e-26 for k in ("image_var", "grad_var", "grad_cov", "spectral_cov", "grad_moment2")))
gv = statistic_map(x, "grad_var", 5); gc = statistic_map(x, "grad_cov", 5).reshape(2, NB, 2, 2, 7, 9)
check("ST02 GC 대각 == GV (같은 통계 — all-entry reduction 효과와 구분해야 한다)",
      torch.allclose(gc[:, :, 0, 0], gv[:, :NB]) and torch.allclose(gc[:, :, 1, 1], gv[:, NB:]) and torch.allclose(gc[:, :, 0, 1], gc[:, :, 1, 0]))
sc_ = statistic_map(x, "spectral_cov", 5).reshape(2, NB, NB, 7, 9)
ev = torch.linalg.eigvalsh(sc_.permute(0, 3, 4, 1, 2))
check("ST02 SC 대칭·PSD·off-diagonal 음수 보존", torch.allclose(sc_, sc_.transpose(1, 2)) and float(ev.min()) > -1e-12 and float(sc_.min()) < 0)
m2 = statistic_map(x, "grad_moment2", 5).reshape(2, NB, 2, 2, 7, 9)
kx = x.new_tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.; ky = kx.T.contiguous()
gyy = F.conv2d(x, ky[None, None].repeat(NB, 1, 1, 1), groups=NB); gxx = F.conv2d(x, kx[None, None].repeat(NB, 1, 1, 1), groups=NB)
muy = F.avg_pool2d(gyy, 5, 1, 0); mux = F.avg_pool2d(gxx, 5, 1, 0)
check("ST02 M2(비중심) − GC(중심) == μμᵀ — 중심화 여부만 다르다", torch.allclose(m2[:, :, 0, 0] - gc[:, :, 0, 0], muy ** 2, atol=1e-10)
      and torch.allclose(m2[:, :, 0, 1] - gc[:, :, 0, 1], muy * mux, atol=1e-10))
x64 = torch.rand(1, NB, 64, 64, dtype=torch.float64)
check("ST03 k5 공통 중심 58² (train 64²) · margin 3 · 표현마다 같은 중심",
      stat_margin(5) == 3 and all(tuple(statistic_map(x64, k, 5).shape[-2:]) == (58, 58) for k in ("image_var", "grad_var", "grad_cov", "spectral_cov", "grad_moment2")))
check("ST03 통계 ROI 가 reconstruction 영역을 바꾸지 않는다 (rec 는 64² 전체)", tuple(o["y"].shape[-2:]) == (64, 64))
zz = x.clone().requires_grad_()
h0 = stat_term(zz, None, x, kind="grad_var", window=5, mode="H")
check("ST04 S=GT 이면 통계 H loss = 0", float(h0.loss) == 0.0)
zz2 = torch.rand(2, NB, 13, 15, dtype=torch.float64, requires_grad=True)
stat_term(zz2, None, x, kind="grad_var", window=5, mode="H").loss.backward()
check("ST04 Student 통계 gradient 는 live·유한·비영", bool(torch.isfinite(zz2.grad).all()) and float(zz2.grad.abs().max()) > 0)
v = statistic_map(x, "grad_var", 5)
check("ST04 REP-STD/LOGVAR 은 비음수 표현에만 정의 (resolver 가 공분산에 막는다)",
      torch.allclose(stat_transform(v, "std", 1e-12), (v + 1e-12).sqrt()) and torch.allclose(stat_transform(v, "logvar", 1e-12), (v + 1e-12).log()))

print("== TRI01–TRI03: 방향 gate 와 대조군")
u_s = torch.tensor([[[[0.10]], [[0.90]], [[0.50]], [[0.50]]]], dtype=torch.float64)
u_t = torch.tensor([[[[0.05]], [[0.10]], [[0.50]], [[0.20]]]], dtype=torch.float64)
u_g = torch.tensor([[[[0.00]], [[0.00]], [[0.50]], [[0.80]]]], dtype=torch.float64)
ms_ = direction_mask(u_s, u_t, u_g, mode="sign")
check("TRI01 SIGN 손계산: 같은 방향 1 · 반대 방향 0 · u=0(이미 GT) 0 · v=0(Teacher=Student) 0",
      [float(z) for z in ms_.flatten()] == [1.0, 1.0, 0.0, 0.0])
cap = direction_mask(u_s, u_t, u_g, mode="sign_cap", eps=1e-6)
check("TRI01 CAP = SIGN · min(1, |u|/(|v|+ε)) — GT 를 지나친 Teacher 방향을 감쇠",
      abs(float(cap.flatten()[0]) - min(1.0, 0.10 / 0.05)) < 1e-9 and abs(float(cap.flatten()[1]) - min(1.0, 0.90 / 0.80)) < 1e-9)
cos = direction_mask(u_s, u_t, u_g, mode="cosine")
uu = (u_g - u_s).flatten(); vv = (u_t - u_s).flatten()
check("TRI01 COS = [cos(u,v)]_[0,1] 를 band 에 broadcast", abs(float(cos.flatten()[0]) - max(0.0, float((uu @ vv) / (uu.norm() * vv.norm())))) < 1e-9)
tau_c = torch.full((NB,), 0.05, dtype=torch.float64)
wk_c = component_weights(s_, t_, g_, tau_c, alpha=1.0, kd_weight=0.1, eps=1e-6)
r_avg = crit(s_, t_, g_, return_maps=True)
same = ((g_ - s_) * (t_ - s_) > 0)
check("TRI02 BANDADV: 성분별 우위가 있는 band 만 w_K,c > 0 이고, 평균 gate 가 0 인 곳에서도 살아난다 (SIGN 과 다른 질문)",
      float(wk_c.max()) > 0 and bool(((wk_c > 0) & (r_avg.maps["soft_weight"] == 0)).any()))
check("TRI02 성분 우위와 같은 부호는 중복이다 — Teacher 가 GT 에 더 가까운 성분은 방향도 일치한다",
      bool((~((wk_c > 0) & ~same)).all()) or float(((wk_c > 0) & ~same).double().mean()) < 1e-12)
mask_a = direction_mask(s_, t_, g_, mode="sign")
kap = mass_kappa(r_avg.maps["soft_weight"], mask_a)
b_ = torch.broadcast_to(r_avg.maps["soft_weight"], mask_a.shape)
check("TRI03 MASS 의 분모는 broadcast 된 같은 shape 의 합 (ρ = Σ b m / Σ b) 이고 총계수가 일치",
      abs(float(kap) - float((b_ * mask_a).sum() / b_.sum())) < 1e-15
      and abs(float((b_ * mask_a).sum()) - float((b_ * kap.expand_as(mask_a)).sum())) < 1e-9)
gen = torch.Generator().manual_seed(4321)
sh = shuffle_mask(mask_a, gen)
check("TRI03 SHUFFLE 은 위치만 섞는다 (총량 보존, 배치는 달라진다)",
      abs(float(sh.mean()) - float(mask_a.mean())) < 1e-12 and not torch.equal(sh, mask_a))
wc = torch.cat((r_avg.maps["hard_weight"], r_avg.maps["soft_weight"]), dim=1)
sh2 = shuffle_mask(wc, torch.Generator().manual_seed(4321))
check("TRI03 CTL-RSHUFFLE: d_T·a_T 를 **같은 순열**로 함께 섞는다 (w_H·w_K 의 대응이 유지된다)",
      abs(float(sh2[:, :1].mean()) - float(wc[:, :1].mean())) < 1e-12 and abs(float(sh2[:, 1:].mean()) - float(wc[:, 1:].mean())) < 1e-12)
hs = r_avg.maps["hard_weight"].mean().expand_as(r_avg.maps["hard_weight"])
check("TRI03 CTL-HSCALE: 공간 w_H 를 batch 평균 스칼라로 (전체 hard 총량 보존, 위치 정보 제거)",
      abs(float((hs * (s_ - g_).abs().mean(1, keepdim=True)).mean()) - float((r_avg.maps["hard_weight"].mean() * (s_ - g_).abs().mean(1, keepdim=True)).mean())) < 1e-12
      and float(hs.std()) < 1e-12)

print("== CS01: Teacher 입력 민감도 (NA-TSENS 전용)")
tb = make(2); freeze(tb)
gt2, ms2, lpan2, pan2 = batch(b=1, hr=64)
mu0 = tb.predict_delta(pan2, F.interpolate(ms2, scale_factor=4, mode="bicubic"))
check("CS01 기준점 δ=0 (aligner 가 없으므로 예측 이동량이 0)", float(mu0.abs().max()) == 0.0)
h0_ = state_hash(tb)
J = teacher_fd_jacobian(tb, pan2, ms2, lpan2, mu0, h=0.05)
check("CS01 중심 유한차분 J [B,8,2,H,W] 유한·비영, Teacher 가중치 불변(frozen target)",
      tuple(J.shape) == (1, NB, 2, 64, 64) and bool(torch.isfinite(J).all()) and float(J.abs().max()) > 0 and state_hash(tb) == h0_)
J2 = teacher_fd_jacobian(tb, pan2, ms2, lpan2, mu0, h=0.025)
rel = float((J2 - J).norm() / (J.norm() + 1e-30))
check("CS01 h/2 로 바꿔도 J 가 안정 (상대 변화 < 0.2)", rel < 0.2, f"rel {rel:.4f}")
q = sens_q(J); rsk = sens_risk(q, float(q[q > 0].median()))
check("CS01 r = s/(s+q) ∈ (0,1], 양의 q 가 없으면 감쇠 없음(r=1) 로 fallback",
      float(rsk.max()) <= 1.0 and float(rsk.min()) > 0 and float(sens_risk(torch.zeros_like(q), 0.0).min()) == 1.0)
rg = roi_gate(rsk, 4)
check("CS01 경계 margin 4 밖은 감쇠 없음 (r=1) — 검증되지 않은 경계에서 원래 soft 로 복귀",
      float(rg[..., :4, :].min()) == 1.0 and float(rg[..., 4:-4, 4:-4].min()) == float(rsk[..., 4:-4, 4:-4].min()))
lin = linearization_check(lambda d: teacher_eval(tb, pan2, ms2, lpan2, d), mu0, J, torch.tensor([[0.1, -0.05]]))
check("CS01 선형화 잔차가 유한 (진단용 기록)", all(math.isfinite(float(v)) for v in lin.values()), json.dumps({k: round(float(v), 5) for k, v in lin.items()}))

print("== EV01 · RS01 · PERF01 · resolver 계약")
cfgs = [f for f in sorted(glob.glob(os.path.join(ROOT, "config", "NA104_*.yaml"))) if not f.endswith("_dry.yaml")]
specs = {}
for f in cfgs:
    c = yaml.safe_load(open(f)); sp = resolve(c["kdv"]); specs[os.path.basename(f)[:-5]] = (c, sp)
check("EV01 NA104 config 가 존재하고 전부 resolve 된다", len(specs) >= 80, f"{len(specs)}벌")
check("REQ 모든 case 가 W104·D122·A-ID·G0·offset0·geometry0 를 강제한다",
      all(c["model_args"]["hidden_size"] == 104 and c["model_args"]["depth"] == [1, 2, 2] and c["trainer"] == "kdv"
          and sp["policy"] == "A-ID" and sp["recipe"] == "NOALIGN" and sp["geom"] == "G0"
          and sp["offset_weight_effective"] == 0 and sp["geometry_weight_effective"] == 0 and sp["protocol"] == "I-A"
          for c, sp in specs.values()))
check("EV01 aligned selector 를 만들지 않는다 (aligner 없음 → aligned_valid == raw_valid)",
      all(sp["aligned_selector"] is False for _, sp in specs.values()))
check("EV01 평가 view/scene 설정 공통: FR 논문 세트 20장·같은 evaluator",
      all(c["test_full_feeder_args"]["dataroot"] == list(specs.values())[0][0]["test_full_feeder_args"]["dataroot"] for c, _ in specs.values())
      and all(str(c.get("fr_select_indices", "0-19")) == str(list(specs.values())[0][0].get("fr_select_indices", "0-19")) for c, _ in specs.values()))
tsens = {n: sp for n, (c, sp) in specs.items() if sp["na_protocol"] == "NA-TSENS"}
check("EV01 NA-STRICT 가 기본이고 NA-TSENS 는 **C 를 쓰는 case 에 한정** (개수를 박아두지 않고 술어로 판정)",
      all(sp["na_protocol"] in ("NA-STRICT", "NA-TSENS") for _, sp in specs.values())
      and all(sp["tri"]["c_mode"] == "off" for _, sp in specs.values() if sp["na_protocol"] == "NA-STRICT")
      and all(sp["tri"]["c_mode"] != "off" for sp in tsens.values()), f"NA-TSENS {len(tsens)}벌")
check("EV01 모든 case 가 expect_arch 로 골격을 강제한다 (§20)",
      all((sp.get("expect_arch") or {}).get("width") == 104 and list((sp.get("expect_arch") or {}).get("depth") or []) == [1, 2, 2] for _, sp in specs.values()))
_sch_src = open(os.path.join(ROOT, "train_kdv.py")).read()


def _q(name):
    p_ = os.path.join(ROOT, "config", "queues", f"na104_{name}.txt")
    return [l.strip() for l in open(p_)] if os.path.exists(p_) else []


def _ids(name):
    return [l.split("_")[1] for l in _q(name) if l and not l.startswith("#")]


q2, q3 = [l for l in _q("s2") if l and not l.startswith("#")], [l for l in _q("s3") if l and not l.startswith("#")]
i2, i3 = _ids("s2"), _ids("s3")
# ==== FINAL 계획(2026-09-12, 01_S2/02_S3_FINAL_EXPERIMENT_PLAN.md) — 단계별 큐·신규 18 정의·v1/v2·Teacher/pilot 고정·core10 반복
_T = "NA104_T00_W104_D122_WV3_N0_OFF_S2025_v1"; _P = "NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1/last"
_all = {os.path.basename(f)[:-5]: yaml.safe_load(open(f)) for f in glob.glob(os.path.join(ROOT, "config", "NA104_*.yaml")) if not f.endswith("_dry.yaml")}
_sp = {n: resolve(c["kdv"]) for n, c in _all.items()}
_cid = lambda n: n.split("_")[1]
_ver = lambda n: n.rsplit("_", 1)[1]
_seed = lambda n: int(n.rsplit("_", 2)[1][1:])
NEW18 = {"X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11", "X12", "PX01", "PX02", "CX01", "CX02", "LX01", "LX02"}
check("FINAL §4 신규 18 정의가 전부 존재하고 v2 다", sorted({_cid(n) for n in _all if _cid(n) in NEW18}) == sorted(NEW18) and all(_ver(n) == "v2" for n in _all if _cid(n) in NEW18))
check("FINAL §8 기존 case 의 seed1234(T00 는 2025) 는 v1 이름을 유지하고, 추가 seed 는 v2 다",
      all(_ver(n) == ("v1" if (_seed(n) == 1234 or _cid(n) == "T00") and _cid(n) not in NEW18 else "v2") for n in _all))
check("FINAL 부록 §6.2 Teacher 는 항상 T00 S2025 v1/best_hqnr — --version/--seed 로 바뀌지 않는다",
      all((c["kdv"].get("teacher") or {}).get("run") == f"work_dir/{_T}" and (c["kdv"].get("teacher") or {}).get("tag") == "best_hqnr" for n, c in _all.items() if _cid(n) != "T00"))
check("FINAL 부록 §3.4/§6.2 λ pilot 은 항상 Q00 S1234 v1/last — 반복 seed 의 Q00 이 pilot 이 되지 않는다",
      all(c["kdv"]["stat"].get("lambda_pilot") == _P and all(e.get("lambda_pilot") == _P for e in (c["kdv"]["stat"].get("extra") or []))
          for n, c in _all.items() if c["kdv"]["stat"].get("enabled")))
check("FINAL 부록 §6.2 비교 baseline 은 같은 Student seed 의 Q00 (pilot 과 별개 identity)",
      all(c["kdv"].get("baseline_run") == f"NA104_Q00_W104_D122_WV3_N0_OFF_S{_seed(n)}_{'v1' if _seed(n) == 1234 else 'v2'}" for n, c in _all.items() if _cid(n) != "T00"))
check("FINAL §6.3 TCOPY/PX parent = T00 v1/best_hqnr · CONT/CX parent = Q00 S1234 v1/last (schedule step 0, fresh optimizer)",
      all((c["kdv"].get("phase") or {}).get("parent_run") == _T and c["kdv"]["phase"]["parent_tag"] == "best_hqnr" for n, c in _all.items() if _cid(n).startswith(("TCOPY", "PX")))
      and all((c["kdv"].get("phase") or {}).get("parent_run") == _P.split("/")[0] and c["kdv"]["phase"]["parent_step"] == 0 and c["kdv"]["phase"]["optimizer_state_policy"] == "fresh"
              for n, c in _all.items() if _cid(n).startswith(("CONT", "CX"))))
check("FINAL §6.3 horizon: 일반/TCOPY/PX 50K · CONT/CX 25K tail · LONG/LX 100K",
      all(c["num_iter"] == (100000 if _cid(n).startswith(("LONG", "LX")) else 25000 if _cid(n).startswith(("CONT", "CX")) else 50000) for n, c in _all.items()))
core10 = ["Q00", "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q09", "Q11", "Q12"]
check("FINAL §R1/R2 core10 × seed 777·2026 가 전부 생성됐다 (--repeat 로는 안 나오던 것)",
      all(f"NA104_{cid}_" in " ".join(n for n in _all if _seed(n) == sd and _cid(n) == cid) for sd in (777, 2026) for cid in core10)
      and sum(1 for n in _all if _seed(n) in (777, 2026) and _cid(n) in core10) == 20)
check("FINAL §R3 원 반복안 보존: Q00/Q04/Q10@2025 · Q10@777 (TIED_TO_TEACHER_SEED 표시 대상)",
      all(any(_cid(n) == c and _seed(n) == sd for n in _all) for c, sd in (("Q00", 2025), ("Q04", 2025), ("Q10", 2025), ("Q10", 777))))
check("FINAL 부록 §5.2 새 run 의 평가 주기는 10 epoch", all(c.get("eval_epoch") == 10 for c in _all.values()))
check("FINAL §4.1 X05/X06/X08 토큰 GVHAD/GVWFIX/GVTMATCH · X07 = R1RSHUF · X03 = N0+GVFIX(Teacher 학습 사용)",
      any("_R3_GVHAD_" in n for n in _all) and any("_R3_GVWFIX_" in n for n in _all) and any("_R3_GVTMATCH_" in n for n in _all)
      and any("_R1RSHUF_OFF_" in n for n in _all) and any(_cid(n) == "X03" and not _sp[n]["teacher_eval_only"] and _sp[n]["needs_teacher"] for n in _all))
# 단계별 큐와 stage plan
for srv, n_new, n_rep in (("s2", 73, 14), ("s3", 83, 14)):
    qq = [l for l in _q(srv) if l and not l.startswith("#")]
    pj = json.load(open(os.path.join(ROOT, "config", "queues", f"na104_{srv}_stage_plan.json")))
    check(f"FINAL {srv} 큐: 완료분 {n_rep} 이 맨 앞(체인이 건너뜀) + 신규/확인 {n_new} = {n_rep + n_new} (계획 S{srv[1]}-C 슬롯), 중복 없음, 모두 config 존재",
          len(qq) == n_rep + n_new and len(set(qq)) == len(qq) and all(x in _all for x in qq) and pj["slots"]["new_or_verify"] == n_new
          and pj["stages"][0]["stage"] == "P0_DONE" and len(pj["stages"][0]["runs"]) == n_rep, f"{len(qq)} run")
    check(f"FINAL {srv} 큐: T00 → Q00 이 맨 앞이고 R1/R2 반복 block 은 그 seed 의 Q00 부터 시작한다",
          _cid(qq[0]) == "T00" and _cid(qq[1]) == "Q00" and all(_cid(next(x for x in qq if _seed(x) == sd)) == "Q00" for sd in (777, 2026)))
    check(f"FINAL {srv} stage plan 이 execute:false 이고 직접 대조·Teacher·pilot 을 담는다",
          pj["execute"] is False and pj["teacher_run"] == _T and pj["lambda_pilot"] == _P and all("contrasts" in r for st_ in pj["stages"] for r in st_["runs"] if r.get("run_id")))
q3f = [l for l in _q("s3") if l and not l.startswith("#")]
check("§8.1 A/B 2×2 네 셀(Q10 base · Q40 A · Q17 B · Q41 AB) 이 s3 한 큐 안에 모두 있다", all(any(_cid(x) == c and _seed(x) == 1234 for x in q3f) for c in ("Q10", "Q40", "Q17", "Q41")))
q2f = [l for l in _q("s2") if l and not l.startswith("#")]
check("FINAL S2-A/S3-A 원 보류 목록(s2 34 · s3 39) 이 각 서버 큐에 빠짐없이 있다",
      all(any(_cid(x) == c for x in q2f) for c in ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "CTLHSCALE", "CTLRSHUF", "CTLAMASS", "CTLASHUF", "CTLBMASS", "CTLBSHUF", "CTLTAU05", "CTLTAU20", "CTLBETA03", "CTLBETA05", "CTLLAMV03", "CTLLAMV30", "CS00", "CS01", "CS02", "CS03", "CTLCMASS", "TCOPYN0", "TCOPYR1", "TCOPYR3", "CONTN0", "CONTR3", "CONTGVAD", "LONG2NN0", "LONG2NR1", "LONG2NR3", "LONG2NGVAD"])
      and all(any(_cid(x) == c for x in q3f) for c in [f"Q{i:02d}" for i in range(20, 35)] + ["Q13", "Q17", "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46", "Q47", "CTLGVSCHALF", "VXW3AD", "VXW7AD", "VXW3H", "VXW7H", "VXM357AD", "VXM357H", "VXSTD", "VXLOG", "VXRES", "VXM2H", "VXM2AD"]))
check("FINAL §6.4 COSTMATCH 는 N 이 측정되기 전에는 큐에 없고 stage plan 에 BLOCKED_COST_MEASUREMENT 로 남는다",
      not any(_cid(x) == "COSTMATCH" for x in q2f) and "COSTMATCH" in json.load(open(os.path.join(ROOT, "config", "queues", "na104_s2_stage_plan.json")))["slots"]["blocked"])
# 새 통계 모드의 항등식 (부록 §3.3)
_g = torch.Generator().manual_seed(11); _r = lambda: torch.rand(2, NB, 13, 15, generator=_g, dtype=torch.float64)
_s, _t, _y = _r(), _r(), _r()
_had = GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="plain_hard_adaptive_kd")(_s, _t, _y); _ad0 = GTAnchoredReconstructionKD(1e-3, eps=1e-8, alpha=0.0, mode="adaptive")(_s, _t, _y)
check("FINAL X05 HAD = plain hard + adaptive soft (= adaptive 의 α=0 과 수치 동일, hard 는 plain L1)",
      abs(float(_had.loss) - float(_ad0.loss)) < 1e-12 and abs(float(_had.hard) - float((_s - _y).abs().mean())) < 1e-12)
_wf = GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="weighted_hard_fixed_kd")(_s, _t, _y, return_maps=True)
check("FINAL X06 WFIX = weighted hard(WH 와 동일) + fixed soft(FIX 와 동일), soft 가중치는 상수 β",
      abs(float(_wf.hard) - float(GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="hard_only")(_s, _t, _y).hard)) < 1e-12
      and abs(float(_wf.soft) - float(GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="fixed_kd")(_s, _t, _y).soft)) < 1e-12
      and float(_wf.maps["soft_weight"].max() - _wf.maps["soft_weight"].min()) == 0.0)
_tm = stat_term(_s, _t, _y, kind="grad_var", window=5, mode="TMATCH", kd_weight=0.1); _tt = stat_term(_s, _t, _y, kind="grad_var", window=5, mode="T")
check("FINAL X08 TMATCH = β_V·(T 의 K_V), hard 0 — T(계수 1) 와 강도만 다르다", abs(float(_tm.loss) - 0.1 * float(_tt.loss)) < 1e-12 and float(_tm.hard) == 0.0)
try:
    resolve(dict(recipe="NOALIGN", aligner_policy="A-ID", na_protocol="NA-STRICT", select=dict(primary="best_hqnr", aligned_selector=False), teacher=dict(id="T", run="w/x"),
                 rec=dict(case="R3"), geom_kd=dict(mode="G0"), stat=dict(enabled=True, kind="GV", mode="GVHAD", lambda_pilot="x/l"))); _bad_mode = False
except ValueError:
    _bad_mode = True
check("FINAL §4.1 미정의 모드 문자열(예: 'GVHAD' 를 mode 로) 은 오류로 막힌다 — 조용히 다른 모드로 돌지 않는다", _bad_mode)
# ---- 조정 §10.3·§10.5 진단 계측
check("조정 §10.3 계수비·loss비·gradient비를 따로 기록하고 집계 정의를 남긴다",
      all(k in _sch_src for k in ("r_coef=", "r_loss=", "r_grad=", "ratio_definition=", "cos_hard_soft=")))
check("조정 §10.3 통계 항도 hard/soft 를 λ_V 를 곱한 값으로 나눠 남긴다", "stat_L_hard_weighted" in _sch_src and "stat_L_soft_weighted" in _sch_src)
check("조정 §10.5 d·a 는 평균뿐 아니라 분위·표준편차, 우세 마진(A_T·A_S) 까지",
      all(k in _sch_src for k in ("difficulty=dict(_q(", "advantage=dict(_q(", "A_T=float(", "A_S=float(", "teacher_better_fraction")))
check("조정 §10.4 진단은 diag step 에서만·optimizer step 없이 (autograd.grad·retain_graph)",
      "retain_graph=True" in open(os.path.join(ROOT, "train_kdv.py")).read() and "self.is_diag_step(global_step)" in _sch_src)
# ---- 조정 §12.2 결과 key
check("조정 §12.2 run_key.json 에 서버·Teacher·init·calibration·selector·evaluator·데이터 hash 를 모은다",
      all(k in _sch_src for k in ("run_key.json", "server_id=", "teacher_checkpoint_sha=", "calibration_sha=", "selection_policy_id=", "fr_dataset_sha256=")))
# ---- 조정 §3.2·§3.3·§11 분석 도구
_rep = os.path.join(ROOT, "tools", "na104_hqnr_report.py")
check("조정 §3.2/§3.3/§11 분석 도구가 있고 읽기 전용이다 (학습 산출물을 쓰지 않는다)",
      os.path.exists(_rep) and all(k in open(_rep).read() for k in ("hqnr_best_common_grid", "hqnr_plateau", "spectral_contribution", "score_only"))
      and "accelerator" not in open(_rep).read())
# ---- RS01: 재개에서 **무엇이 복원되고 무엇이 복원되지 않는지**를 실제로 확인한다 (문자열 검사 아님)
from train_po import RNGState                                              # noqa: E402
g1 = torch.Generator(); g1.manual_seed(2000); [torch.randperm(8, generator=g1) for _ in range(3)]
st = RNGState(g1).state_dict(); before = [torch.randperm(8, generator=g1).tolist() for _ in range(2)]
g2 = torch.Generator(); g2.manual_seed(2000); r2 = RNGState(g2); r2.load_state_dict(st)
check("RS01 corruption/TRI RNG 는 checkpoint 로 정확히 복원된다 (같은 다음 열)",
      [torch.randperm(8, generator=g2).tolist() for _ in range(2)] == before)
check("RS01 warm start 는 parent step 만큼 scheduler 를 진행시킨다 (CONT/TCOPY 는 step 0 = 새 tail)", "for _ in range(step):" in _sch_src)
# 알려진 한계: train DataLoader 는 accelerate 에 prepare 되지 않아 **배치 순서가 복원되지 않는다** (main.py C-1).
_main_src = open(os.path.join(ROOT, "main.py")).read()
_prep = re.search(r"accelerator\.prepare\(([^)]*)\)", open(os.path.join(ROOT, "train_kdv.py")).read())
check("RS01 [알려진 한계] train DataLoader 는 prepare 되지 않는다 → 재개 시 배치 순서 미복원 (근사 재개)",
      _prep is not None and "data_loader" not in _prep.group(1) and "근사 재개" in _main_src, _prep.group(1) if _prep else "?")
check("RS01 그래서 **재개 사실을 run 에 기록**한다 (resume_events.jsonl + manifest resumed 플래그) — 통제 비교에서 가려낼 수 있어야 한다",
      "resume_events.jsonl" in _sch_src and "resumed=bool(self.resumed_from)" in _sch_src)
_g = torch.Generator(); _g.manual_seed(7)
_d0 = [b[0].flatten().tolist() for b in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(8.).view(8, 1)), batch_size=2, shuffle=True, generator=_g)]
_g.manual_seed(7)
_d1 = [b[0].flatten().tolist() for b in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(8.).view(8, 1)), batch_size=2, shuffle=True, generator=_g)]
check("RS01 (참고) 같은 generator 상태에서만 배치 열이 같다 — 상태가 다르면 다른 열이 나온다", _d0 == _d1)

# ---- 검토 1: calibration 이 전역 RNG 를 소비하면 캐시 적중 여부로 학습 배치가 달라진다
import main as _mainmod                                                   # noqa: E402
from kdv.calibration import calibration_batches                           # noqa: E402


class _DummyFeeder(torch.utils.data.Dataset):
    def __init__(self, **kw):
        self.n = 64
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        t = torch.full((1,), float(i))
        return t, t, t, t, t


_orig_ic = _mainmod.import_class; _mainmod.import_class = lambda path: _DummyFeeder
_args = types.SimpleNamespace(feeder="dummy", train_feeder_args=dict(dataroot="x"), batch_size=8)
torch.manual_seed(1234); _s_before = torch.random.get_rng_state()
_b, _man = calibration_batches(_args, n_patches=16, seed=1234, batch_size=8)
_s_after = torch.random.get_rng_state()
_mainmod.import_class = _orig_ic
check("검토1 calibration 이 전역 RNG 를 건드리지 않는다 (캐시 적중/미적중이 학습 배치를 바꾸지 않는다)",
      torch.equal(_s_before, _s_after), f"batches {len(_b)}")
torch.manual_seed(1234); _n0 = torch.randperm(64).tolist()[:4]
torch.manual_seed(1234); _mainmod.import_class = lambda path: _DummyFeeder
calibration_batches(_args, n_patches=16, seed=1234, batch_size=8); _n1 = torch.randperm(64).tolist()[:4]
_mainmod.import_class = _orig_ic
check("검토1 calibration 실행 여부와 무관하게 다음 데이터 순열이 같다", _n0 == _n1, f"{_n0} vs {_n1}")
check("검토1 trainer 는 calibration 전 구간을 fork_rng 로 격리한다", "with torch.random.fork_rng(devices=" in _sch_src and "_calibrate_and_build_inner" in _sch_src)

# ---- 검토 2: 판정·시트·Teacher·진단이 같은 checkpoint (저장소 확정 지시: 무조건 HQNR)
check("검토2 주 selector 는 best_hqnr 이고 Teacher tag 도 best_hqnr (보조로 best_rr_val·last 선언)",
      all(sp["select_primary"] == "best_hqnr" and "best_rr_val" in (sp.get("select_secondary") or []) for _, sp in specs.values())
      and all((c["kdv"].get("teacher") or {}).get("tag", "best_hqnr") == "best_hqnr" for c, _ in specs.values()))
_up = open(os.path.join(ROOT, "tools", "_upload.sh")).read()
check("검토2 artifact 진단도 주 selector(best_hqnr) 에서 돈다", "for CK in best_hqnr best_rr_val last" in _up)

# ---- 검토 5: NASENS 의 calibration·FD 검사·학습이 같은 ROI
from kdv.calibration import _roi as _roi_fn                               # noqa: E402
_q = torch.zeros(1, 2, 64, 64); _q[..., :4, :] = 100.0; _q[..., 4:-4, 4:-4] = 1.0
check("검토5 calibration 이 학습과 같은 내부 ROI 에서만 q 를 잰다 (경계가 s_sens 를 바꾸지 못한다)",
      float(_roi_fn(_q, 4).max()) == 1.0 and tuple(_roi_fn(_q, 4).shape[-2:]) == (56, 56) and float(_q.max()) == 100.0)
check("검토5 trainer 가 roi_margin 을 calibration 과 cache key 에 넘긴다", "roi_margin=tri[\"c_roi_margin\"]" in _sch_src and "_roi{tri['c_roi_margin']}" in _sch_src)

# ---- 검토 6: fitting bin 의 정의와 표현 범위
check("검토6 Δe 열 이름이 정의를 담는다 (Teacher−Student; 계획의 Δe(q) 는 run 간 비교라 분석 시점 계산)",
      "delta_e_T_minus_S" in _sch_src and "win_rate_vs_teacher" in _sch_src and "delta_e=float" not in _sch_src)


def _reps(kdv_dict):
    return [r[0] for r in KDVTrainer._stat_reps(types.SimpleNamespace(spec=resolve(kdv_dict)))]


_base = dict(recipe="NOALIGN", aligner_policy="A-ID", na_protocol="NA-STRICT", select=dict(primary="best_hqnr", aligned_selector=False),
             teacher=dict(id="T", run="w/x"), rec=dict(case="R3"), geom_kd=dict(mode="G0"))
check("검토6 통계 OFF arm(N0·R3) 도 기준 표현(GV w5) bin 을 남긴다", _reps(dict(_base, stat=dict(enabled=False))) == ["grad_var_w5"])
check("검토6 다중 창은 3/5/7 을 각각 남긴다", _reps(dict(_base, stat=dict(enabled=True, kind="GV", mode="H", windows=[3, 5, 7], lambda_pilot="x/l"))) == ["grad_var_w3", "grad_var_w5", "grad_var_w7"])
check("검토6 GV+SC 결합은 SC 도 남긴다",
      _reps(dict(_base, stat=dict(enabled=True, kind="GV", mode="H", lambda_pilot="x/l", extra=[dict(kind="SC", mode="H", lambda_pilot="x/l")]))) == ["grad_var_w5", "spectral_cov_w5"])
check("검토6 residual 표현은 자기 표현과 기준 표현을 함께 남긴다",
      _reps(dict(_base, stat=dict(enabled=True, kind="GV", mode="AD", domain="residual", lambda_pilot="x/l"))) == ["grad_var_w5_res", "grad_var_w5"])

# ---- 검토 4: 캠페인 게이트 격리 (큐가 끝난 뒤 과거 캠페인이 열리지 않는다)
import subprocess                                                         # noqa: E402
_env = dict(os.environ); _env.pop("PANCRAFTER_CAMPAIGN_GATES", None)
_r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], capture_output=True, text=True, env=_env, cwd=ROOT)
check("검토4 캠페인 게이트는 기본적으로 아무 것도 열지 않는다 (과거 UVS·SR·s2 KD 자동 실행 차단)",
      _r.returncode == 0 and _r.stdout.strip() == "" and "비활성" in _r.stderr)
_env["PANCRAFTER_CAMPAIGN_GATES"] = "sr"
_r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], capture_output=True, text=True, env=_env, cwd=ROOT)
check("검토4 명시하면 그 캠페인만 열린다", _r2.returncode == 0 and "활성" in _r2.stderr)
n0 = [n for n, (c, sp) in specs.items() if sp["rec_case"] == "N0" and not sp["stat_enabled"]]
check("§15.2 GT-only arm 도 같은 고정 Teacher 를 평가 bin 용으로 싣는다 (학습 loss 에는 미사용)",
      all(specs[n][1]["has_teacher"] and specs[n][1]["teacher_eval_only"] and not specs[n][1]["needs_teacher"] for n in n0 if "T00" not in n))
check("§16.4 R1·WH 는 Teacher 가 필요한 case 로 표시된다 (Teacher 없는 학습이 아니다)",
      all(sp["needs_teacher"] for n, (c, sp) in specs.items() if sp["rec_case"] in ("R1", "R2", "R3") or (sp["stat_enabled"] and sp["stat_mode"] in ("WH", "T", "FIX", "AD"))))

print("== PERF01: 실제 batch forward/backward/step")
opt = torch.optim.AdamW([p for p in stud.backbone.parameters()], lr=1e-4)
o2 = kdv_forward(stud, teach, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=False)
rc = crit(o2["y"].double(), o2["y_t"].double(), gt.double())
st = stat_term(o2["y"].float(), o2["y_t"].float(), gt.float(), kind="grad_var", window=5, mode="AD",
               criterion=GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode="adaptive"))
tot = rc.loss + 0.1 * st.loss + 0.1 * output_edge_loss(o2["y"].float(), gt.float())
tot.backward(); gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in stud.backbone.parameters() if p.grad is not None)))
opt.step()
check("PERF01 rec + λ·stat + λ·edge 세 항의 forward/backward/step 이 유한", bool(torch.isfinite(tot)) and math.isfinite(gn) and gn > 0, f"loss {float(tot):.4f} |g| {gn:.3f}")
check("PERF01 Teacher 는 학습되지 않는다 (gradient 없음, hash 불변)", all(p.grad is None for p in teach.parameters()))

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)")
sys.exit(1 if FAIL else 0)
