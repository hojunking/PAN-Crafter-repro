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
check("resolver run name", run_name(resolve(ok_spec()), 1234) == "S2W112_A1_IA_AFR_R3_GVAD_G0_s1234_v01")
bad = 0
for kw in (dict(geom_kd=dict(mode='G1', outer_weight=1.0)), dict(teacher={}), dict(aligner_policy='A-SC'), dict(recipe='N2_SG'), dict(input_protocol='I-N'), dict(geom_kd=dict(mode='G3')),
           dict(aligner_policy='A-ID'), dict(stat=dict(enabled=True, kind='EDGE', mode='AD')), dict(aux=dict(offset_weight=0.01))):
    try: resolve(ok_spec(**kw)); print("   not caught:", kw)
    except ValueError: bad += 1
check("resolver rejects 9 invalid combos", bad == 9)
sp = resolve(ok_spec(recipe='A3', aligner_policy='A-FR'))
check("resolver: A-FR disables geometry term (diag only)", sp['geometry_weight'] == 0.01 and sp['geometry_weight_effective'] == 0.0 and 'geometry' in sp['disabled_terms'])
sp = resolve(dict(recipe='NOALIGN', aligner_policy='A-ID', rec=dict(case='N0'))); check("resolver: Q00 no teacher needed", not sp['needs_teacher'] and run_name(sp, 1234) == "S2W112_NOALIGN_IA_AID_N0_OFF_G0_s1234_v01")

# ---------------- M01/M02/M04: W112·D124 골격 · donor strict · W96→W112 partial load 불가 · residual base 1회
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
MA = dict(in_channels=1, out_channels=8, hidden_size=112, depth=[1, 2, 4], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False, n_attn=3)
bb = Model(**MA); n_bb = sum(p.numel() for p in bb.parameters())
gn = [m for m in bb.modules() if isinstance(m, torch.nn.GroupNorm)]
check("M01 W112·D124 backbone 2.8854 M, no GroupNorm (LN) → divisibility n/a", abs(n_bb / 1e6 - 2.8854) < 1e-3 and not gn, f"{n_bb / 1e6:.4f} M, GN layers {len(gn)}")
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

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)")
sys.exit(1 if FAIL else 0)
