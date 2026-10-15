#!/usr/bin/env python
"""KDV 실행 전 gate (plan §20.2 중 코드로 닫을 수 있는 것: M01–M04 · W01/W03 · L01–L04 · R01 · 참조 파일의 10+12 검사 · resolver · calibration 규칙). 하나라도 실패하면 exit 1.

    python tools/kdv_unit_tests.py
"""
import os, sys, math, copy, json, tempfile
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.losses_rec import GTAnchoredReconstructionKD, MODES
from kdv.losses_stat import statistic_map, statistic_loss, stat_term, KINDS, positive_median
from kdv.registry import resolve, run_name
from kdv.teacher_assets import load_donor_aligner, skeleton_from_cfg, freeze, state_hash, assert_param_disjoint, extract_aligner_sd
from kdv.forward import kdv_forward
from kdv.protocol import prepare_view, is_corrupt_update
from kdv.calibration import calibrate_lambda
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.warp import warp_pan
from train_po import RNGState
from main import import_class

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

torch.manual_seed(0); torch.set_num_threads(4)
T = lambda v, grad=False: torch.tensor([[[[v]]]], dtype=torch.float64, requires_grad=grad)

# ---------------- L01 reconstruction criterion (참조 10 tests)
s, t, y = T(.1, True), T(.08), T(.0); r = GTAnchoredReconstructionKD(.01)(s, t, y)
d = .08 / (.08 + .01); a = (.10 - .08) / (.10 + 1e-6); exp = (1 + d) * .10 + .1 * (1 - d) * a * .02
check("L01 main formula = hand calculation", abs(r.loss.item() - exp) < 1e-12)
s = T(.08, True); r = GTAnchoredReconstructionKD(.01)(s, T(.08), T(0)); r.loss.backward(); check("L01 both wrong → soft 0, GT gradient > 1", r.soft.item() == 0 and s.grad.item() > 1)
r = GTAnchoredReconstructionKD(.01)(T(.04), T(.08), T(0), return_maps=True); check("L01 student better → soft weight 0", r.maps['soft_weight'].item() == 0 and r.hard.item() > .04)
s = T(0, True); r = GTAnchoredReconstructionKD(.01)(s, T(.08), T(0)); r.loss.backward(); check("L01 exact GT → loss 0, grad 0", r.loss.item() == 0 and s.grad.item() == 0)
s, t, y = T(.1, True), T(.02, True), T(0, True); r = GTAnchoredReconstructionKD(.01)(s, t, y, return_maps=True); r.loss.backward()
check("L01 teacher/GT/routing detached", t.grad is None and y.grad is None and s.grad is not None and all(not m.requires_grad for m in r.maps.values()))
s = torch.tensor([[[[.1]], [[.9]]]], dtype=torch.float64, requires_grad=True); r = GTAnchoredReconstructionKD(.01, kd_weight=.5)(s, torch.tensor([[[[.2]], [[.1]]]], dtype=torch.float64), torch.zeros(1, 2, 1, 1, dtype=torch.float64)); r.loss.backward()
check("L01 elementwise GT direction with conflicting bands", bool((s.grad > 0).all()))
g = torch.Generator().manual_seed(2025); s = torch.randn(2, 8, 7, 7, generator=g, requires_grad=True); t, y = torch.randn(2, 8, 7, 7, generator=g), torch.randn(2, 8, 7, 7, generator=g)
r = GTAnchoredReconstructionKD(.05)(s, t, y, return_maps=True); r.loss.backward()
check("L01 coefficient bounds (8 band, random)", bool((r.maps['hard_weight'] >= 1).all() and (r.maps['hard_weight'] < 2).all() and (r.maps['soft_weight'] >= 0).all()) and r.maps['soft_weight'].max().item() <= .100001 and bool((s.grad * (s.detach() - y) > 0).all()))
ok = True
for mode in MODES:
    ok &= bool(torch.isfinite(GTAnchoredReconstructionKD(.01, mode=mode)(T(.1), T(.08), T(0)).loss))
check("L01 ablation modes finite; gt=.1, fixed_kd=.102", ok and abs(GTAnchoredReconstructionKD(.01, mode='gt')(T(.1), T(.08), T(0)).loss.item() - .1) < 1e-12 and abs(GTAnchoredReconstructionKD(.01, mode='fixed_kd')(T(.1), T(.08), T(0)).loss.item() - .102) < 1e-12)
bad = 0
for f in (lambda: GTAnchoredReconstructionKD(0), lambda: GTAnchoredReconstructionKD(.01, kd_weight=1), lambda: GTAnchoredReconstructionKD(.01)(torch.zeros(1, 8, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 8, 4, 4))):
    try: f()
    except ValueError: bad += 1
check("L01 config/input validation raises", bad == 3)
s = torch.ones(1, 8, 2, 2, dtype=torch.float16, requires_grad=True); r = GTAnchoredReconstructionKD(.01)(s, torch.zeros_like(s), torch.zeros_like(s)); r.loss.backward()
check("L01 fp16 → float32 loss, finite grad", r.loss.dtype == torch.float32 and bool(torch.isfinite(s.grad).all()))

# ---------------- L02 statistics (참조 12 tests) + L04 spectral chunking
rng = torch.Generator().manual_seed(1234); R = lambda c=3: torch.rand(2, c, 13, 15, generator=rng, dtype=torch.float64)
x = R(); check("L02 shapes share centers", all(statistic_map(x, k, 5).shape == (2, dd, 7, 9) for k, dd in dict(image_var=3, grad_var=6, grad_cov=12, spectral_cov=9).items()))
xc = torch.full((1, 3, 13, 15), .37, dtype=torch.float64); check("L02 constants → 0", all(statistic_map(xc, k).abs().max().item() < 1e-26 for k in KINDS))
x = R(); patches = F.unfold(x[..., 1:-1, 1:-1], kernel_size=5).reshape(2, 3, 25, 7, 9)
check("L02 population variance explicit", torch.allclose(statistic_map(x, 'image_var'), patches.var(dim=2, correction=0), rtol=1e-10, atol=1e-12))
x = R(); check("L02 intensity scaling squared", all(torch.allclose(statistic_map(2.3 * x, k), 2.3 ** 2 * statistic_map(x, k), rtol=1e-9, atol=1e-11) for k in KINDS))
sh = x + x.new_tensor([.2, -.4, .7])[None, :, None, None]; check("L02 bandwise brightness offset invariant", all(torch.allclose(statistic_map(sh, k), statistic_map(x, k), rtol=1e-9, atol=1e-11) for k in KINDS))
gv = statistic_map(x, 'grad_var'); gc = statistic_map(x, 'grad_cov').reshape(2, 3, 2, 2, 7, 9)
check("L02 grad_cov diagonal = grad_var, symmetric", torch.allclose(gc[:, :, 0, 0], gv[:, :3]) and torch.allclose(gc[:, :, 1, 1], gv[:, 3:]) and torch.allclose(gc[:, :, 0, 1], gc[:, :, 1, 0]))
cv = statistic_map(x, 'spectral_cov').reshape(2, 3, 3, 7, 9); ev = torch.linalg.eigvalsh(cv.permute(0, 3, 4, 1, 2))
check("L02 spectral_cov symmetric PSD", torch.allclose(cv, cv.transpose(1, 2)) and ev.min().item() >= -1e-12)
xg = R().requires_grad_(); ok = True
for k in KINDS:
    gr, = torch.autograd.grad(statistic_map(xg, k).square().mean(), (xg,), retain_graph=True); ok &= bool(torch.isfinite(gr).all()) and gr.abs().max().item() > 0
check("L02 direct student gradient", ok)
s = R().requires_grad_(); t, y = R().requires_grad_(), R().requires_grad_()
lo = statistic_loss(s, t, y, kind='grad_var', window=5, criterion=GTAnchoredReconstructionKD(1e-3, eps=1e-8)); lo.loss.backward()
check("L02 target detach, live student", t.grad is None and y.grad is None and s.grad.abs().max().item() > 0 and not lo.maps['difficulty'].requires_grad)
y, t = R(), R(); s = y.clone().requires_grad_()
check("L02 exact GT → stat loss 0 (all kinds)", all(statistic_loss(s, t, y, kind=k, window=5, criterion=GTAnchoredReconstructionKD(1e-3, eps=1e-8)).loss.item() == 0 for k in KINDS))
xh = R().half().requires_grad_(); yh = statistic_map(xh, 'grad_var'); yh.sum().backward(); check("L02 half → float32, finite grad", yh.dtype == torch.float32 and bool(torch.isfinite(xh.grad).all()))
bad = 0
for k in [0, 2, 4, 21]:
    try: statistic_map(x, 'grad_var', k)
    except ValueError: bad += 1
try: statistic_map(x, 'unknown', 5)
except ValueError: bad += 1
check("L02 bad config raises", bad == 5)
# L04: spectral covariance row-chunking vs explicit unfold (작은 tensor), 음의 off-diagonal 보존
x8 = torch.rand(1, 4, 9, 9, generator=rng, dtype=torch.float64); x8[:, 1] = 1.0 - x8[:, 0]      # band1 = −band0 → 음의 공분산
sc = statistic_map(x8, 'spectral_cov', 3).reshape(1, 4, 4, 5, 5)
P = F.unfold(x8[..., 1:-1, 1:-1], kernel_size=3).reshape(1, 4, 9, 5, 5); Pc = P - P.mean(dim=2, keepdim=True)
exp_cov = torch.einsum('bikhw,bjkhw->bijhw', Pc, Pc) / 9
check("L04 spectral cov chunking = explicit unfold; negative off-diagonal kept", torch.allclose(sc, exp_cov, rtol=1e-9, atol=1e-12) and sc[0, 0, 1].max().item() < 0)
# stat_term modes
s = R().requires_grad_(); t, y = R(), R(); crit = {m: GTAnchoredReconstructionKD(1e-3, eps=1e-8, mode=c) for m, c in dict(H='gt', FIX='fixed_kd', WH='hard_only', AD='adaptive').items()}
vals = {m: stat_term(s, t, y, kind='grad_var', window=5, mode=m, criterion=crit.get(m)).loss.item() for m in ('H', 'T', 'FIX', 'WH', 'AD')}
h_no_t = stat_term(s, None, y, kind='grad_var', window=5, mode='H').loss.item()
vs, vt, vg = statistic_map(s.detach(), 'grad_var'), statistic_map(t, 'grad_var'), statistic_map(y, 'grad_var')
check("stat_term H(no teacher) = <|V_S−V_G|>, T = <|V_S−V_T|>, FIX = H + 0.1·T", abs(h_no_t - (vs - vg).abs().mean().item()) < 1e-12 and abs(vals['T'] - (vs - vt).abs().mean().item()) < 1e-12 and abs(vals['FIX'] - (vals['H'] + 0.1 * vals['T'])) < 1e-12)
bad = 0
try: stat_term(s, None, y, kind='grad_var', window=5, mode='AD')
except ValueError: bad += 1
try: stat_term(s, t, y, kind='grad_var', window=5, mode='AD', criterion=crit['H'])
except ValueError: bad += 1
check("stat_term guards (no teacher AD / wrong criterion mode)", bad == 2)

# ---------------- registry / resolver
def ok_spec(**kw):
    base = dict(recipe='A1', input_protocol='I-A', aligner_policy='A-FR', donor=dict(source='x'), teacher=dict(run='y', id='T112_v01'), rec=dict(case='R3'), stat=dict(enabled=True, kind='GV', mode='AD'))
    base.update(kw); return base
check("resolver run name", run_name(resolve(ok_spec()), 1234) == "S2W112D123_A1_IA_AFR_R3_GVAD_G0_s1234_v01")
bad = 0
for kw in (dict(geom_kd=dict(mode='G1', outer_weight=1.0)), dict(teacher={}), dict(aligner_policy='A-SC'), dict(recipe='N2_SG'), dict(input_protocol='I-N'), dict(geom_kd=dict(mode='G3')),
           dict(aligner_policy='A-ID'), dict(stat=dict(enabled=True, kind='EDGE', mode='AD')), dict(aux=dict(offset_weight=0.01))):
    try: resolve(ok_spec(**kw)); print("   not caught:", kw)
    except ValueError: bad += 1
check("resolver rejects 9 invalid combos", bad == 9)
sp = resolve(ok_spec(recipe='A3', aligner_policy='A-FR'))
check("resolver: A-FR disables geometry term (diag only)", sp['geometry_weight'] == 0.01 and sp['geometry_weight_effective'] == 0.0 and 'geometry' in sp['disabled_terms'])
sp = resolve(dict(recipe='NOALIGN', aligner_policy='A-ID', rec=dict(case='N0'))); check("resolver: Q00 no teacher needed", not sp['needs_teacher'] and run_name(sp, 1234) == "S2W112D123_NOALIGN_IA_AID_N0_OFF_G0_s1234_v01")

# ---------------- M01/M02/M04: W112·D124 골격 · donor strict · W96→W112 partial load 불가 · residual base 1회
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
MA = dict(in_channels=1, out_channels=8, hidden_size=112, depth=[1, 2, 3], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False, n_attn=3)
bb = Model(**MA); n_bb = sum(p.numel() for p in bb.parameters())
gn = [m for m in bb.modules() if isinstance(m, torch.nn.GroupNorm)]
check("M01 W112·D123 backbone 2.6589 M (캠페인 골격), no GroupNorm (LN) → divisibility n/a", abs(n_bb / 1e6 - 2.6589) < 1e-3 and not gn, f"{n_bb / 1e6:.4f} M, GN layers {len(gn)}")
n124 = sum(p.numel() for p in Model(**dict(MA, depth=[1, 2, 4])).parameters()); check("M01 (참고) W112·D124 = 2.8854 M (계획 원안)", abs(n124 / 1e6 - 2.8854) < 1e-3, f"{n124 / 1e6:.4f} M")
from kdv.registry import arch_prefix; check("M01 run prefix from model_args = S2W112D123", arch_prefix(MA) == "S2W112D123" and run_name(resolve(ok_spec()), 1234, 'v01', arch_prefix(MA)) == "S2W112D123_A1_IA_AFR_R3_GVAD_G0_s1234_v01")
first = next(m for m in bb.modules() if isinstance(m, torch.nn.Conv2d)); check("M01 first conv 9→? (PAN 1 + MS 8)", first.in_channels == 9, f"in {first.in_channels}")
donor = os.path.join(ROOT, "assets/donor_aligner/PA_A1_REC_W96_D124_9CH_S2025_best_hqnr_aligner.pt")
al, man = load_donor_aligner(donor, 8); check("M02 donor aligner strict load (105,330 params, 22 keys)", man['n_params'] == 105330 and man['n_keys'] == 22, man['aligner_tensors_sha256_16'])
w96 = Model(**dict(MA, hidden_size=96)).state_dict(); bad = False
try: bb.load_state_dict(w96, strict=True)
except RuntimeError: bad = True
check("M02 W96 backbone → W112 strict load raises (partial load 불가)", bad)
try: extract_aligner_sd({'backbone.x': torch.zeros(1)}); bad = False
except KeyError: bad = True
check("M02 non-aligner state → KeyError", bad)
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)
m = PAModel(bb, al, aligner_margin=0, sampler=True)
pan, ms, lpan = torch.rand(2, 1, 64, 64), torch.rand(2, 8, 16, 16), torch.rand(2, 1, 16, 16)
o = m(pan, ms, lpan); mb = F.interpolate(ms, scale_factor=4, mode="bicubic")
res_explicit = bb(o['pan_aligned'], o['pan_aligned'], ms, torch.ones(2))
ok_ = kdv_forward(m, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True)
check("M04 y == bicubic MS base + backbone residual (base 정확히 1회)", torch.equal(o['ms_base'], mb) and torch.allclose(o['y'], mb + res_explicit, atol=1e-6) and torch.allclose(ok_['y'], ok_['ms_base'] + ok_['residual']))
m_id = PAModel(copy.deepcopy(bb), None, sampler=False); o_id = m_id(pan, ms, lpan)
sw = torch.ones(2); y_b0 = mb + bb(pan, pan, ms, sw)
check("M04 A-ID (no aligner/sampler) == B0 forward, Δ=0", torch.allclose(o_id['y'], y_b0, atol=1e-6) and float(o_id['delta'].abs().max()) == 0 and torch.equal(o_id['pan_aligned'], pan))
o0 = m(pan, ms, lpan, aligner_enabled=False); check("W01 sampler identity at Δ=0 (|W(P,0)−P| < 1e-6)", float((o0['pan_aligned'] - pan).abs().max()) < 1e-6)
check("W01 donor Δ on random input finite, |Δ| < 8", bool(torch.isfinite(o['delta']).all()) and float(o['delta'].abs().max()) < 8, f"Δ {o['delta'][0].tolist()}")
skel, info = skeleton_from_cfg(dict(model_args=MA, num_bands=8, trainer='kdv', kdv=dict(aligner_policy='A-ID')), Model)
check("skeleton A-ID has no aligner keys", skel.aligner is None and not any(k.startswith('aligner.') for k in skel.state_dict()) and info['has_aligner'] is False)

# ---------------- M03 / L03 routing: A-FR shares correction, aligner no grad, F_S grads; A-FT live aligner grads; Teacher fixed
teacher = PAModel(copy.deepcopy(bb), copy.deepcopy(al)); freeze(teacher); th = state_hash(teacher)
stud_fr = PAModel(copy.deepcopy(bb), copy.deepcopy(al)); freeze(stud_fr.aligner)
for p_ in stud_fr.backbone.parameters(): p_.data.add_(0.01 * torch.randn_like(p_))
gt = torch.rand(2, 8, 64, 64)
o = kdv_forward(stud_fr, teacher, pan, ms, lpan, share_correction=True, teacher_needed=True, aligner_live=False)
check("L03 A-FR: shared correction (Δ_S == Δ_T), teacher output detached, no graph through aligner", torch.equal(o['delta'], o['delta_t']) and not o['y_t'].requires_grad and not o['delta'].requires_grad and o['y'].requires_grad)
crit = GTAnchoredReconstructionKD(0.02, mode='adaptive'); r = crit(o['y'], o['y_t'], gt); r.loss.backward()
check("L03 A-FR: F_S grads present, aligner grads absent, teacher grads absent", all(p.grad is not None for p in stud_fr.backbone.parameters() if p.requires_grad) and all(p.grad is None for p in stud_fr.aligner.parameters()) and all(p.grad is None for p in teacher.parameters()))
opt = torch.optim.AdamW([p for p in stud_fr.backbone.parameters()], lr=1e-4); opt.step()
check("M03 teacher state hash unchanged after student step; aligner frozen hash unchanged", state_hash(teacher) == th and state_hash(stud_fr.aligner) == state_hash(al))
try: assert_param_disjoint(torch.optim.AdamW(list(stud_fr.backbone.parameters()) + list(teacher.parameters()), lr=1e-4), teacher); bad = False
except RuntimeError: bad = True
check("M03 optimizer ∩ teacher params → RuntimeError", bad and assert_param_disjoint(opt, teacher, stud_fr.aligner))
stud_ft = PAModel(copy.deepcopy(bb), copy.deepcopy(al))
with torch.no_grad(): stud_ft.aligner.fc2.weight.add_(0.05 * torch.randn_like(stud_ft.aligner.fc2.weight))
o = kdv_forward(stud_ft, teacher, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=True)
r = crit(o['y'], o['y_t'], gt); r.loss.backward()
check("L03 A-FT: live aligner receives gradient through sampler; Δ_S ≠ Δ_T", all(p.grad is not None and torch.isfinite(p.grad).all() for p in stud_ft.aligner.parameters()) and not torch.equal(o['delta'].detach(), o['delta_t']) and o['delta'].requires_grad)
# geo-only backward → F_S 에 gradient 없음
from pa.losses import direct_geometry_loss
stud_ft.zero_grad(); o = kdv_forward(stud_ft, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True)
lg, _ = direct_geometry_loss(o['pan_aligned'], pan, gt, 2.0, 11); lg.backward()
check("L03 geometry-only backward: aligner grads, no F_S grads", any(p.grad is not None for p in stud_ft.aligner.parameters()) and all(p.grad is None for p in stud_ft.backbone.parameters()))
# equivalence: kdv_forward student path == PAModel.forward (평가 경로)
o1 = kdv_forward(stud_ft, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True); o2 = stud_ft(pan, ms, lpan)
check("E01 train forward == eval forward (same y, Δ, P~)", torch.allclose(o1['y'], o2['y']) and torch.equal(o1['delta'].detach(), o2['delta'].detach()) and torch.allclose(o1['pan_aligned'], o2['pan_aligned']))

# ---------------- W03 / R01: I-N view, parity, RNG checkpoint
g = torch.Generator().manual_seed(3); pv, eps, c = prepare_view(pan, 'I-N', 1, 1.0, g); pv0, eps0, c0 = prepare_view(pan, 'I-N', 0, 1.0, g)
check("W03 I-N parity: odd update corrupt (|ε|≤R, P_view ≠ P), even native (P_view is P, ε=0)", c and float(eps.norm(dim=1).max()) <= 1.0 and not torch.equal(pv, pan) and not c0 and pv0 is pan and float(eps0.abs().max()) == 0)
o = kdv_forward(stud_ft, None, pv, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True)
two = warp_pan(warp_pan(pan.float(), eps), o['delta'].detach()); one = warp_pan(pan.float(), eps + o['delta'].detach())
check("W03 sequential two warps W(W(P,ε),c) ≠ single W(P,ε+c) (composition not collapsed)", torch.allclose(o['pan_aligned'], two, atol=1e-6) and float((two - one).abs().max()) > 0)
g1 = torch.Generator().manual_seed(7); _ = prepare_view(pan, 'I-N', 1, 1.0, g1); st = RNGState(g1).state_dict(); e1 = prepare_view(pan, 'I-N', 1, 1.0, g1)[1]
g2 = torch.Generator().manual_seed(0); RNGState(g2).load_state_dict(st); e2 = prepare_view(pan, 'I-N', 1, 1.0, g2)[1]
check("R01 corruption RNG state roundtrip", torch.equal(e1, e2))

# ---------------- calibration rules
pilot = PAModel(copy.deepcopy(bb), copy.deepcopy(al)); freeze(pilot)
batches = [(torch.rand(4, 8, 64, 64), None, torch.rand(4, 8, 16, 16), torch.rand(4, 1, 16, 16), torch.rand(4, 1, 64, 64)) for _ in range(3)]
r = calibrate_lambda(pilot, batches, 'grad_var', 5, 0.05, torch.device('cpu'))
check("calibration λ_V = r_grad·median(g_rec/g_V), finite, OK", r['status'] == 'OK' and abs(r['lambda_V'] - 0.05 * r['ratio_median']) < 1e-12 and r['lambda_V'] > 0, f"λ {r['lambda_V']:.4g} ratio {r['ratio_median']:.3g}")
check("calibration v_scale positive median (constant → None)", positive_median(torch.zeros(3)) is None and abs(positive_median(torch.tensor([0., 1., 3., 2.])) - 2.0) < 1e-12)


# ---------------- 이동량 covariance KD (계획 §11): C01 · C02 · G-STRUCT · G2–G4 · resolver
from kdv.alignment_kd import (structure_tensor_mean, probe_set, eq_closure_moment, precision_from_moment, geo_curvature, geo_autograd_grad, precision_from_curvature,
                              kd_matrix, mean_kd_loss, CovHead, gaussian_kl)
from kdv.registry import geom_tag
from pa.losses import direct_geometry_loss, geometry_residual
g_ = torch.Generator().manual_seed(11)
# 부드러운 합성 영상 (Gaussian blob 들) — 유한차분 검증은 잡음 영상이 아니라 실제와 비슷한 구조에서
yy, xx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
def blobs(n, seed):
    gg = torch.Generator().manual_seed(seed); img = torch.zeros(n, 1, 64, 64)
    for b in range(n):
        for _ in range(12):
            cy, cx, sg, amp = 8 + 48 * torch.rand(1, generator=gg), 8 + 48 * torch.rand(1, generator=gg), 2 + 4 * torch.rand(1, generator=gg), torch.rand(1, generator=gg)
            img[b, 0] += amp * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
    return img
pan_s = blobs(3, 1); gt_s = (pan_s * torch.linspace(0.5, 1.5, 8)[None, :, None, None]).clone() + 0.01 * torch.randn(3, 8, 64, 64, generator=g_)
mu_t = torch.tensor([[0.2, -0.1], [0.0, 0.0], [-0.3, 0.4]])
r, info = geometry_residual(warp_pan(pan_s, mu_t), pan_s, gt_s, 2.0, 11); L, _ = direct_geometry_loss(warp_pan(pan_s, mu_t), pan_s, gt_s, 2.0, 11)
check("C01 ||r||² == direct_geometry_loss (같은 가중·정규화)", abs(float((r ** 2).sum(1).mean()) - float(L)) < 1e-6)
gc = geo_curvature(pan_s, gt_s, mu_t, 2.0, 11, 0.05); ga = geo_autograd_grad(pan_s, gt_s, mu_t, 2.0, 11)
rel = float((gc['g_fd'] - ga).norm() / ga.norm()); gc2 = geo_curvature(pan_s, gt_s, mu_t, 2.0, 11, 0.025); hc = float((gc2['H'] - gc['H']).norm() / gc['H'].norm())
check("C01 FD Jᵀr ↔ autograd ∇½||r||² (rel err < 5%) · h/2 일관성 (< 5%)", rel < 0.05 and hc < 0.05, f"rel {rel:.4f} h-cons {hc:.4f}")
H = gc['H']; ev = torch.linalg.eigvalsh(H); check("C01 H = JᵀJ symmetric PSD", torch.allclose(H, H.transpose(1, 2)) and float(ev.min()) >= -1e-12)
flat = torch.full((2, 1, 64, 64), 0.3); gflat = geo_curvature(flat, flat.expand(2, 8, 64, 64).clone(), torch.zeros(2, 2), 2.0, 11, 0.05)
Pi_f, inf_f = precision_from_curvature(gflat['H'], gflat['s2'], 1e-3, 0.05, 1e-6, 1e-3, 1e-6, 1.0)
check("C01 평탄 영상 → H≈0 → rank 0, precision 0 (damping 확신 금지)", float(Pi_f.abs().max()) == 0.0 and int(inf_f['rank'].max()) == 0)
Pi_g, inf_g = precision_from_curvature(H, gc['s2'], 1e-3, 0.05, 1e-6, 1e-3, 1e-6, 1.0)
check("C01 구조 영상 → rank 2, precision ≤ 1/σ_min² (cap)", int(inf_g['rank'].min()) == 2 and float(torch.linalg.eigvalsh(Pi_g).max()) <= 1 / 0.05 ** 2 + 1e-6)
# eq_closure: 이상적 반응(μ_ε = b − ε) 은 closure 0 → cap, 무반응(상수) 은 closure = ε → Π ≈ 1/(mean|ε|²+σ²)
ms_b = torch.nn.functional.interpolate(torch.rand(3, 8, 16, 16, generator=g_), scale_factor=4, mode="bicubic"); pr = probe_set(16, 1.0)
check("eq probe set: K=16, 두 반경 {0.5,1.0}, 평균 0", pr.shape == (16, 2) and set(round(float(x), 3) for x in pr.norm(dim=1)) == {0.5, 1.0} and float(pr.mean(0).abs().max()) < 1e-6)
const = PANGlobalAligner(8)
with torch.no_grad():
    const.fc2.weight.zero_(); const.fc2.bias.copy_(torch.tensor([0.2, -0.1]))
eqc = eq_closure_moment(const, pan_s, ms_b, pr, 0); Pi_c, inf_c = precision_from_moment(eqc['Q'], 0.05)
exp_q = float((pr ** 2).sum(1).mean() / 2)
check("eq 무반응(상수) aligner: Q = mean εεᵀ (= R²·5/16 I), Π ≈ 1/(Q+σ²) ≪ cap, valid", torch.allclose(eqc['Q'][0], exp_q * torch.eye(2, dtype=torch.float64), atol=1e-6) and float(torch.linalg.eigvalsh(Pi_c).max()) < 5 and bool(eqc['valid'].all()), f"Q00 {float(eqc['Q'][0,0,0]):.4f} exp {exp_q:.4f}")
Pi_i, inf_i = precision_from_moment(torch.zeros(3, 2, 2, dtype=torch.float64), 0.05)
check("eq 완벽 반응(closure 0): Π = cap 1/σ_min² 전부", bool(inf_i['cap_hit'].all()) and abs(float(Pi_i[0, 0, 0]) - 400.0) < 1e-6)
# G-STRUCT / G2–G4 대수
J = structure_tensor_mean(gt_s); check("G-STRUCT J_Y symmetric PSD, (dy,dx) 순서", torch.allclose(J, J.transpose(1, 2)) and float(torch.linalg.eigvalsh(J).min()) >= 0)
Pi = Pi_g; mu_s = mu_t + torch.tensor([[0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
l3, q3 = mean_kd_loss(mu_s, mu_t, kd_matrix(Pi, 'G3')); l2, q2 = mean_kd_loss(mu_s, mu_t, kd_matrix(Pi, 'G2')); l4, q4 = mean_kd_loss(mu_s, mu_t, kd_matrix(Pi, 'G4')); l1, q1 = mean_kd_loss(mu_s, mu_t, kd_matrix(Pi, 'G1', k0=2.0))
d = (mu_s - mu_t).double()
rel = lambda a, b: abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
check("G3 = ½ dᵀΠd · G2 trace 동일 · G4 = diag(Π) · G1 = ½k0||d||²", rel(q3[0], d[0] @ Pi[0] @ d[0]) and rel(kd_matrix(Pi, 'G2').diagonal(dim1=1, dim2=2).sum(1)[0], Pi[0].trace())
      and rel(q4[2], Pi[2, 0, 0] * d[2, 0] ** 2 + Pi[2, 1, 1] * d[2, 1] ** 2) and rel(l1, 0.5 * 2.0 * float((d ** 2).sum(1).mean())), f"q3 {float(q3[0]):.6g} vs {float(d[0] @ Pi[0] @ d[0]):.6g}")
mu_live = mu_s.clone().requires_grad_(); lg, _ = mean_kd_loss(mu_live, mu_t.clone().requires_grad_(), kd_matrix(Pi, 'G3'), torch.tensor([True, False, True])); lg.backward()
check("mean-KD: invalid sample gradient 0, Teacher μ 로 gradient 없음, 분모는 batch 전체", float(mu_live.grad[1].abs().max()) == 0 and float(mu_live.grad[0].abs().max()) > 0)
# C02: G5 KL
head = CovHead(); feat = torch.randn(3, 64, generator=g_); S = head(feat)
check("C02 CovHead 초기 Σ = I(+eps), SPD", torch.allclose(S[0], torch.eye(2, dtype=torch.float64) * (1 + 1e-4), atol=1e-6) and float(torch.linalg.eigvalsh(S).min()) > 0)
St = torch.tensor([[[2.0, 0.3], [0.3, 0.5]]]).double().expand(3, 2, 2); kl0 = gaussian_kl(mu_t, St, mu_t, St)
check("C02 같은 분포 → KL 0; 방향 KL(q_S‖q_T) 는 Teacher precision 으로 mean 오차 가중", float(kl0.abs().max()) < 1e-12 and abs(float(gaussian_kl(mu_s, St, mu_t, St)[0]) - 0.5 * float(d[0] @ torch.linalg.inv(St[0]) @ d[0])) < 1e-10)
kl = gaussian_kl(mu_t, S, mu_t, St); kl.sum().backward()
check("C02 head 가 gradient 를 받고 KL ≥ 0", head.fc.weight.grad is not None and float(kl.min()) >= 0)
kl_ts = gaussian_kl(mu_t, St, mu_t, S.detach()); check("C02 KL 비대칭 (KL(S‖T) ≠ KL(T‖S))", float((kl - kl_ts).abs().max()) > 1e-6)
# resolver: G 모드 조합
def gs(**kw): return ok_spec(aligner_policy='A-FT', **kw)
check("resolver G3EQ/G3GEO/GSTRUCT/G5EQ/G1K names", geom_tag(resolve(gs(geom_kd=dict(mode='G3', covariance_source='eq_closure', outer_weight=1.0)))) == 'G3EQ' and geom_tag(resolve(gs(geom_kd=dict(mode='G3', covariance_source='geo_curvature', outer_weight=1.0)))) == 'G3GEO'
      and geom_tag(resolve(gs(geom_kd=dict(mode='G-STRUCT', outer_weight=1.0)))) == 'GSTRUCT' and geom_tag(resolve(gs(geom_kd=dict(mode='G5', covariance_source='eq_closure', outer_weight=1.0)))) == 'G5EQ' and geom_tag(resolve(gs(geom_kd=dict(mode='G1', k0=1.0, outer_weight=1.0)))) == 'G1K')
bad = 0
for kw in (dict(geom_kd=dict(mode='G3', outer_weight=1.0)), dict(geom_kd=dict(mode='G3', covariance_source='struct', outer_weight=1.0)), dict(geom_kd=dict(mode='G1', outer_weight=1.0)),
           dict(geom_kd=dict(mode='G5', covariance_source='eq_closure', outer_weight=0.0)), dict(geom_kd=dict(mode='G-CORR', outer_weight=1.0)), dict(geom_kd=dict(mode='G0', covariance_source='eq_closure'))):
    try: resolve(gs(**kw)); print("   not caught:", kw)
    except ValueError: bad += 1
try: resolve(ok_spec(geom_kd=dict(mode='G3', covariance_source='eq_closure', outer_weight=1.0))); print("   not caught: A-FR + G3")
except ValueError: bad += 1
check("resolver rejects 7 invalid geom combos (source 없음·struct 오용·k0 없음·weight 0·PENDING·G0+source·A-FR)", bad == 7)


# ---------------- TRI-A/B/C (addendum §12.1–12.3)
from kdv.tri import (direction_mask, mass_kappa, shuffle_mask, component_weights, masked_l1, masked_l1_componentwise, routing_stats, fd_jacobian, teacher_fd_jacobian, teacher_eval,
                     sens_q, sens_risk, diag_risk, lowrank_quad, iso_quad, scalar_quad, sigma_from_precision, sigma_control, linearization_check)
from kdv.registry import tri_tag
T2 = lambda *rows: torch.tensor(rows, dtype=torch.float64).view(1, -1, 1, 1)
G_, S_ = T2(0.5, 0.5), T2(0.6, 0.5)
check("TRI-A §4.5 T=(0.55,0.50) → m=(1,0); T=(0.50,0.55) → (1,0); T=S → (0,0)",
      direction_mask(S_, T2(0.55, 0.5), G_).flatten().tolist() == [1.0, 0.0] and direction_mask(S_, T2(0.5, 0.55), G_).flatten().tolist() == [1.0, 0.0] and direction_mask(S_, S_.clone(), G_).flatten().tolist() == [0.0, 0.0])
check("TRI-A G=S, T 만 다름 → mask 0", float(direction_mask(G_.clone(), T2(0.4, 0.7), G_).abs().max()) == 0)
mc = direction_mask(S_, T2(0.2, 0.5), G_, mode="sign_cap"); ms_ = direction_mask(S_, T2(0.2, 0.5), G_, mode="sign")
check("TRI-A cap < sign when Teacher overshoots GT (u=−0.1, v=−0.4 → 0.25)", abs(float(mc[0, 0]) - 0.25) < 1e-6 and float(ms_[0, 0]) == 1.0)
mcos = direction_mask(S_, T2(0.55, 0.5), G_, mode="cosine"); check("TRI-A cosine: parallel → 1 broadcast; T=S → 0", mcos.flatten().tolist() == [1.0, 1.0] and float(direction_mask(S_, S_.clone(), G_, mode="cosine").abs().max()) == 0)
g8 = torch.Generator().manual_seed(5); s8 = torch.rand(2, 8, 6, 6, generator=g8, dtype=torch.float64).requires_grad_(); t8 = torch.rand(2, 8, 6, 6, generator=g8, dtype=torch.float64); y8 = torch.rand(2, 8, 6, 6, generator=g8, dtype=torch.float64)
crit8 = GTAnchoredReconstructionKD(0.05); r8 = crit8(s8, t8, y8, return_maps=True)
ml = masked_l1(s8, t8, y8, r8.maps['hard_weight'], r8.maps['soft_weight'], mask=torch.ones_like(s8))
ga = torch.autograd.grad(r8.loss, s8, retain_graph=True)[0]; gb = torch.autograd.grad(ml['loss'], s8, retain_graph=True)[0]
check("TRI-A mask=ones ⇒ R3 loss·gradient 동일", abs(float(ml['loss']) - float(r8.loss)) < 1e-12 and torch.allclose(ga, gb, atol=1e-12))
m0 = masked_l1(s8, t8, y8, r8.maps['hard_weight'], r8.maps['soft_weight'], mask=torch.zeros_like(s8))
check("TRI-A mask=0 ⇒ weighted GT-only (= hard), soft 0, hard gradient 존재", abs(float(m0['loss']) - float(r8.hard)) < 1e-12 and float(m0['soft']) == 0 and float(torch.autograd.grad(m0['loss'], s8, retain_graph=True)[0].abs().max()) > 0)
tl, yl = t8.clone().requires_grad_(), y8.clone().requires_grad_(); mk = direction_mask(s8.detach(), tl, yl)
mm = masked_l1(s8, tl, yl, r8.maps['hard_weight'], r8.maps['soft_weight'], mask=mk); mm['loss'].backward()
check("TRI-A Teacher/GT/mask 에 gradient 없음, Student live", tl.grad is None and yl.grad is None and not mk.requires_grad and s8.grad is not None)
s8.grad = None
same = ((y8 - s8.detach()) * (t8 - s8.detach()) > 0).double(); check("TRI-A sign == 1[u·v>0] on random", torch.equal(direction_mask(s8.detach(), t8, y8), same))
wb = component_weights(s8.detach(), t8, y8, torch.full((8,), 0.05, dtype=torch.float64), alpha=1.0, kd_weight=0.1, eps=1e-6)
check("TRI-A BANDADV: 성분 우위(w_K,c>0) ⇒ 같은 부호 (sign 재곱은 중복)", bool(((wb > 0) <= (same > 0)).all()))
kap = mass_kappa(r8.maps['soft_weight'], same); check("TRI-A MASS κ ∈ [0,1], Σb=0 → 0", 0 <= float(kap) <= 1 and float(mass_kappa(torch.zeros_like(r8.maps['soft_weight']), same)) == 0)
shf = shuffle_mask(same, torch.Generator().manual_seed(1)); check("TRI-A SHUFFLE: 성분별 합 보존, 공간 순서 변경", torch.allclose(shf.sum(dim=(2, 3)), same.sum(dim=(2, 3))) and not torch.equal(shf, same))
sB = torch.rand(2, 8, 13, 15, generator=g8, dtype=torch.float64); tB = torch.rand(2, 8, 13, 15, generator=g8, dtype=torch.float64); yB = torch.rand(2, 8, 13, 15, generator=g8, dtype=torch.float64)
vs_ = statistic_map(sB, 'grad_var', 5); vt_ = statistic_map(tB, 'grad_var', 5); vg_ = statistic_map(yB, 'grad_var', 5)
mb = direction_mask(vs_, vt_, vg_); ub = vg_ - vs_; vb = vt_ - vs_
check("TRI-B mask shape == stat map [B,16,·,·], 성분별 독립 (dy band c ↔ 채널 c, dx ↔ 8+c)", mb.shape == vs_.shape and torch.equal(mb, ((ub * vb) > 0).double()))
crit_v = GTAnchoredReconstructionKD(1e-3, eps=1e-8); s8v = sB.clone().requires_grad_(); vs_live = statistic_map(s8v, 'grad_var', 5); rv = crit_v(vs_live.detach(), vt_, vg_, return_maps=True)
lb = masked_l1(vs_live, vt_, vg_, rv.maps['hard_weight'], rv.maps['soft_weight'], mask=mb); lb['loss'].backward()
check("TRI-B 통계 gate 를 통한 Student gradient 존재, hard 유지", s8v.grad is not None and float(s8v.grad.abs().max()) > 0 and float(lb['hard']) > 0)
cst = torch.full((1, 8, 13, 15), 0.3, dtype=torch.float64).requires_grad_(); vc = statistic_map(cst, 'grad_var', 5); vc.sum().backward()
check("TRI-B constant Student → variance gradient 0 (정상; recon 이 탈출 신호)", float(cst.grad.abs().max()) == 0)
Jt = torch.randn(2, 8, 2, 4, 4, generator=g8, dtype=torch.float64); bt = torch.randn(2, 8, 4, 4, generator=g8, dtype=torch.float64)
evl = lambda d: bt + torch.einsum('bdihw,bi->bdhw', Jt, d)
Jfd = fd_jacobian(evl, torch.zeros(2, 2, dtype=torch.float64), h=0.05); check("TRI-C 선형 toy FD == J", torch.allclose(Jfd, Jt, atol=1e-9))
Sg = torch.tensor([[[0.04, 0.01], [0.01, 0.02]], [[0.09, 0.0], [0.0, 0.01]]], dtype=torch.float64); val = torch.tensor([True, True])
dr = diag_risk(Jt, Sg, val, s_c=0.02); qan = torch.einsum('bdihw,bij,bdjhw->bdhw', Jt, Sg, Jt)
check("TRI-C diag(JΣJᵀ) 해석식 일치, q ≥ 0, r ∈ (0,1]", torch.allclose(dr['q_report'], qan) and float(qan.min()) >= 0 and float(dr['risk'].min()) > 0 and float(dr['risk'].max()) <= 1)
dr0 = diag_risk(Jt, torch.zeros(2, 2, 2, dtype=torch.float64), val, s_c=0.02); check("TRI-C Σ=0 (명시적 유효) → r=1", float(dr0['risk'].min()) == 1.0)
dri = diag_risk(Jt, Sg, torch.tensor([True, False]), s_c=0.02); check("TRI-C invalid sample → r=1 fallback, q NaN 표시", float(dri['risk'][1].min()) == 1.0 and bool(torch.isnan(dri['q_report'][1]).all()) and not bool(torch.isnan(dri['q_report'][0]).any()))
dr2 = diag_risk(Jt, Sg + 0.05 * torch.eye(2, dtype=torch.float64), val, s_c=0.02); check("TRI-C Σ 를 PSD 순서로 키우면 q↑ r↓", bool((dr2['q_report'] >= dr['q_report'] - 1e-12).all()) and bool((dr2['risk'] <= dr['risk'] + 1e-12).all()))
Jsw = fd_jacobian(lambda d: bt + torch.einsum('bdihw,bi->bdhw', Jt, d.flip(1)), torch.zeros(2, 2, dtype=torch.float64), h=0.05)
check("TRI-C (dy,dx) 순서 뒤집힘 검출 (열 교환)", not torch.allclose(Jsw, Jt) and torch.allclose(Jsw.flip(2), Jt, atol=1e-9))
check("TRI-C h 단위: Σ_HR = r²Σ_LR, J_HR = J_LR/r ⇒ q 불변", torch.allclose(diag_risk(Jt / 4, 16 * Sg, val, s_c=0.02)['q_report'], qan, atol=1e-9))
dq = torch.randn(2, 8, 4, 4, generator=g8, dtype=torch.float64).requires_grad_()
lq = lowrank_quad(dq, Jt, Sg, s_c=0.02)
Ls = torch.linalg.cholesky(Sg); U = torch.einsum('bdihw,bij->bhwdj', Jt, Ls) / 0.02; Wd = torch.linalg.inv(torch.eye(8, dtype=torch.float64) + U @ U.transpose(-1, -2))
dqp = dq.detach().permute(0, 2, 3, 1).unsqueeze(-1); direct = ((dqp.transpose(-1, -2) @ Wd @ dqp).squeeze(-1).squeeze(-1) / (8 * 0.02)).unsqueeze(1)
evw = torch.linalg.eigvalsh(Wd)
check("TRI-C Woodbury 2×2 == 직접 8×8 inverse, W 고유값 ∈ (0,1]", torch.allclose(lq.detach(), direct, atol=1e-9) and float(evw.min()) > 0 and float(evw.max()) <= 1 + 1e-12)
lq.sum().backward(); check("TRI-C quadratic soft: d 로 gradient, J/Σ 는 detach", dq.grad is not None and float(dq.grad.abs().max()) > 0)
qi = iso_quad(dq.detach(), s_c=0.02); qs = scalar_quad(dq.detach(), Jt, Sg, s_c=0.02)
check("TRI-C QISO = ‖d‖²/(D s_C) · QSCALAR ≤ QISO (tr W/D ≤ 1)", torch.allclose(qi, (dq.detach() ** 2).sum(1, keepdim=True) / (8 * 0.02)) and bool((qs <= qi + 1e-12).all()))
check("TRI-C J=0 → sens fallback r=1 (s_sens=0 은 무반응 표시)", float(sens_risk(sens_q(torch.zeros_like(Jt)), 0.0).min()) == 1.0 and float(sens_risk(sens_q(Jt), 1.0).max()) < 1.0)
Pi_r = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]], dtype=torch.float64); Sg_r, val_r = sigma_from_precision(Pi_r)
check("TRI-C precision 0 방향 → invalid (null direction ≠ certainty)", not bool(val_r[0]))
Sg_rot = sigma_control(Sg, 'rotate'); check("TRI-C Σ rotate control: 고유값 유지·행렬 변경", torch.allclose(torch.linalg.eigvalsh(Sg_rot), torch.linalg.eigvalsh(Sg)) and not torch.allclose(Sg_rot, Sg))
teach = PAModel(copy.deepcopy(bb), copy.deepcopy(al)); freeze(teach); h0 = state_hash(teach)
ms_lr = torch.nn.functional.interpolate(ms_b, scale_factor=0.25, mode='bicubic')
mu_tt = teach.predict_delta(pan_s, ms_b); Jr = teacher_fd_jacobian(teach, pan_s, ms_lr, None, mu_tt, h=0.05)
check("TRI-C 실제 Teacher J 유한·비영 [B,8,2,64,64], Teacher hash 불변", Jr.shape == (3, 8, 2, 64, 64) and bool(torch.isfinite(Jr).all()) and float(Jr.abs().max()) > 0 and state_hash(teach) == h0)
z0 = teacher_eval(teach, pan_s, ms_lr, None, mu_tt); z1 = teacher_eval(teach, pan_s, ms_lr, None, mu_tt + torch.tensor([[0.5, 0.0]]))
check("TRI-C probe 값이 δ 에 따라 실제로 변함", float((z1 - z0).abs().max()) > 0)
sp_t = resolve(ok_spec(tri=dict(a=dict(mode='sign'), b=dict(mode='sign'), c=dict(mode='diag', covariance_source='eq_closure'))))
check("TRI 이름 토큰 TRI_ASIGN_BSIGN_CDIAGEQ", tri_tag(sp_t) == 'TRI_ASIGN_BSIGN_CDIAGEQ' and tri_tag(resolve(ok_spec())) == '')


# ---------------- 검토(2026-09-10 adversarial review) 확정 결함의 gate
from kdv.calibration import calibrate_covariance, teacher_precision_fn
sp_k = resolve(gs(geom_kd=dict(mode='G1', k0=2.5, outer_weight=1.0)))
cov_k = calibrate_covariance(None, [], sp_k, torch.device('cpu')); fnk = teacher_precision_fn(None, sp_k, cov_k)
Pi_k, val_k, _ = fnk(pan_s, ms_b, gt_s, mu_t)
check("review: G1K(explicit k0, source none) → k0·I precision, no crash", cov_k['status'] == 'OK' and cov_k['k0'] == 2.5 and torch.allclose(Pi_k[0], 2.5 * torch.eye(2, dtype=torch.float64)) and bool(val_k.all()))
mu_live = mu_t.clone().requires_grad_(); mu_tl = mu_t.clone().requires_grad_()
lgk, _ = mean_kd_loss(mu_live, mu_tl, kd_matrix(Pi_g, 'G3')); lgk.backward()
check("review: mean-KD gives no gradient to Teacher μ (detach) and gradient to Student μ", mu_tl.grad is None and mu_live.grad is not None)
o_f = kdv_forward(stud_ft, teacher, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=True, features=True)
check("review: features path — Student feat has graph, Teacher feat_t detached", o_f['feat'].requires_grad and not o_f['feat_t'].requires_grad and o_f['feat'].shape[1] == 64)
Sg_r2, val_r2 = sigma_from_precision(torch.stack([Pi_g[0], torch.tensor([[4.0, 0.0], [0.0, 0.0]], dtype=torch.float64)]))
check("review: sigma_from_precision marks rank-deficient sample invalid, keeps full-rank", bool(val_r2[0]) and not bool(val_r2[1]))
try: resolve(gs(geom_kd=dict(mode='G3', covariance_source='eq_closure', outer_weight=1.0, probes=dict(radius_hr=3.0)))); bad_r = False
except ValueError: bad_r = True
check("review: resolver rejects probes.radius_hr > 2.5 (would invalidate every sample)", bad_r)
ch = CovHead(); ch.requires_grad_(False); ch.requires_grad_(True); check("review: CovHead requires_grad can be re-enabled after eval", all(p.requires_grad for p in ch.parameters()))

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)")
sys.exit(1 if FAIL else 0)
