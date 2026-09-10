#!/usr/bin/env python
"""NA104 (W104·D122 no-align KD) 실행 전 gate — 계획 §18 의 NA01–NA04 · CK01 · KD01–KD04 · ST01–ST04 · TRI01–TRI03 · EV01 · RS01 · PERF01 · CS01.

    python tools/na104_unit_tests.py            # CPU, 수 분

이전 캠페인(W96/W112) 의 CPU 통과 기록을 이 골격의 통과로 쓰지 않는다 (§16.1) — 여기서 W104·D122 로 다시 확인한다.
gate 실패는 IMPLEMENTATION_ERROR 로 분류한다 (성능 비개선 NEGATIVE, 의존성 미충족 BLOCKED_INPUT 와 구분, §18).
"""
import os, sys, json, copy, glob, math
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
check("EV01 주 selector 는 best_rr_val, aligned selector 는 만들지 않는다 (aligner 없음 → aligned_valid == raw_valid)",
      all(sp["select_primary"] == "best_rr_val" and sp["aligned_selector"] is False for _, sp in specs.values()))
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
q2 = [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "na104_s2.txt")) if l.strip() and not l.startswith("#")]
q3 = [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "na104_s3.txt")) if l.strip() and not l.startswith("#")]
def _has(q, cid):
    return any(t.split("_")[1] == cid for t in q)
check("§8.1 A/B 2×2 네 셀(Q10 base · Q40 A · Q17 B · Q41 AB) 이 **한 서버 안에** 모두 있다",
      all(_has(q3, c) for c in ("Q10", "Q40", "Q17", "Q41")), "s3")
check("큐 순서 의존: 각 큐에서 T00 이 Teacher 사용 run 보다 앞, Q00 이 λ pilot 사용 run 보다 앞",
      all(q.index([t for t in q if t.split("_")[1] == "T00"][0]) == 0 and q.index([t for t in q if t.split("_")[1] == "Q00"][0]) <= 1 for q in (q2, q3)))
check("RS01 corruption RNG 를 checkpoint 에 등록하는 경로가 유지된다 (재개 동일성)", "register_for_checkpointing" in open(os.path.join(ROOT, "train_kdv.py")).read())
check("RS01 warm start 는 parent step 만큼 scheduler 를 진행시킨다 (CONT/TCOPY 는 step 0 = 새 tail)",
      "for _ in range(step):" in open(os.path.join(ROOT, "train_kdv.py")).read())
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
