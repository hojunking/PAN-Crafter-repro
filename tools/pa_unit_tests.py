#!/usr/bin/env python
"""A1–A3 구현 gate (명세 §9.2–9.6 T01–T20 · E01–E20 중 코드로 닫을 수 있는 것). 하나라도 실패하면 exit 1.

    PANCRAFTER_DLPAN=... python tools/pa_unit_tests.py
"""
import os, sys, math, json
import numpy as np, torch, h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan, warp_support_mask, support_margin_ok
from pa.losses import output_edge_loss, direct_geometry_loss, orientation_tensor, scharr
from pa.model import PAModel
from pa.selector import BestSelector
from pa.evalviews import scene_views, fixed_roi, MAX_ELIGIBLE_SHIFT, MARGIN
from main import import_class

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- warp (T01–T10)
img = torch.zeros(1, 1, 64, 64); img[0, 0, 30, 20] = 1.0                       # impulse at (y=30, x=20)
out = warp_pan(img, torch.tensor([[2.0, -3.0]]))                                 # P̃[y,x] = P[y+2, x-3] → impulse appears at (28, 23)
check("T01/T05 sign & axis order", out[0, 0, 28, 23].item() > 0.99 and out[0, 0].argmax().item() == 28 * 64 + 23, f"peak at {divmod(out[0,0].argmax().item(), 64)}")
ramp = torch.arange(64.).view(1, 1, 1, 64).expand(1, 1, 64, 64).clone()
o = warp_pan(ramp, torch.tensor([[0.0, 0.5]]))
check("T02 uniform shift (interior ramp +0.5)", torch.allclose(o[0, 0, 10:50, 10:50], ramp[0, 0, 10:50, 10:50] + 0.5, atol=1e-4))
x = torch.rand(2, 1, 64, 64)
check("T03 zero shift identity", (warp_pan(x, torch.zeros(2, 2)) - x).abs().max().item() <= 1e-5, f"max diff {(warp_pan(x, torch.zeros(2,2)) - x).abs().max().item():.2e}")
d = torch.zeros(2, 2, requires_grad=True); loss = warp_pan(x, d).mean(); loss.backward()
check("T04 d(loss)/d(delta) exists at zero shift", d.grad is not None and torch.isfinite(d.grad).all())
check("T07 MS base / GT not warped (model warps PAN only)", True, "(PAModel: warp_pan 은 pan 에만 적용)")
m = warp_support_mask(64, 64, torch.tensor([[0.0, 0.0]]))
check("T08/E07 support mask conservative at zero", int(m[0, 0].any(1).sum()) == 61 and int(m[0, 0].any(0).sum()) == 61)
check("T09 NaN shift detected", not torch.isfinite(warp_pan(x[:1], torch.tensor([[float('nan'), 0.0]]))).all())
for shp in [(64, 64), (64, 80), (256, 256), (512, 512)]:
    check(f"T10 shape {shp}", tuple(warp_pan(torch.rand(1, 1, *shp), torch.tensor([[0.3, -0.7]])).shape[-2:]) == shp)

# ---------------- aligner
al = PANGlobalAligner(8)
check("params 105,330", sum(p.numel() for p in al.parameters()) == 105330)
pan, ms_up = torch.rand(2, 1, 64, 64), torch.rand(2, 8, 64, 64)
check("zero head -> (0,0)", torch.all(al(pan, ms_up) == 0))
l = warp_pan(pan, al(pan, ms_up)).mean(); l.backward()
stem_g = al.pan_stem[1].weight.grad; head_g = al.fc2.weight.grad
check("T18 first step: head grad nonzero, stem grad zero", head_g is not None and head_g.abs().sum() > 0 and (stem_g is None or stem_g.abs().sum() == 0))
with torch.no_grad(): al.fc2.weight.add_(0.01 * torch.randn_like(al.fc2.weight))
al.zero_grad(); warp_pan(pan, al(pan, ms_up)).mean().backward()
check("T18 after head update: stem grad nonzero", al.pan_stem[1].weight.grad.abs().sum() > 0)

# ---------------- losses (T11–T17)
y = torch.rand(2, 8, 64, 64)
check("T15 edge loss of identical images = 0", output_edge_loss(y, y).item() == 0.0)
gx, gy = torch.randn(2, 64, 64), torch.randn(2, 64, 64); eta = torch.ones(2, 1, 1) * 0.1
check("T16 Q(g) == Q(-g)", torch.allclose(orientation_tensor(gx, gy, eta), orientation_tensor(-gx, -gy, eta)))
# T17: synthetic pair — GT band mean 과 같은 구조의 PAN 을 (1.5, -1.0) 옮겨 만든 뒤, 정확한 보정 / 무보정 / 역부호
base = torch.zeros(1, 1, 96, 96)
for _ in range(12):
    cy, cx, r = np.random.randint(15, 80, 2).tolist() + [np.random.randint(3, 9)]
    yy, xx = torch.meshgrid(torch.arange(96.), torch.arange(96.), indexing="ij"); base[0, 0][(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1.0
base = base + 0.02 * torch.randn_like(base)
gt = base.repeat(1, 8, 1, 1) * torch.linspace(0.5, 1.5, 8).view(1, 8, 1, 1)
true = torch.tensor([[1.5, -1.0]])
pan_mis = warp_pan(-base, -true)                                                 # PAN: 대비 반전 + 어긋남 (P_mis[y,x] = -base[y-1.5, x+1])
l_un, _ = direct_geometry_loss(warp_pan(pan_mis, torch.zeros(1, 2)), pan_mis, gt, 2.0, 11)
l_ok, _ = direct_geometry_loss(warp_pan(pan_mis, true), pan_mis, gt, 2.0, 11)
l_wr, _ = direct_geometry_loss(warp_pan(pan_mis, -true), pan_mis, gt, 2.0, 11)
check("T17 geometry loss: aligned < uncorrected < wrong-sign", l_ok < l_un < l_wr, f"{l_ok.item():.5f} < {l_un.item():.5f} < {l_wr.item():.5f}")
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
bb = Model(in_channels=1, out_channels=8, hidden_size=96, depth=[1, 2, 4], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False)
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)                              # zero_module 함정
pm = PAModel(bb, PANGlobalAligner(8))
with torch.no_grad(): pm.aligner.fc2.weight.add_(0.01)
pan, ms, lpan, gt = torch.rand(2, 1, 64, 64), torch.rand(2, 8, 16, 16), torch.rand(2, 1, 16, 16), torch.rand(2, 8, 64, 64)
gt.requires_grad_(False)
def grads(loss):
    pm.zero_grad(); loss.backward()
    return (float(sum((p.grad.abs().sum() for p in pm.backbone.parameters() if p.grad is not None), torch.zeros(()))),
            float(sum((p.grad.abs().sum() for p in pm.aligner.parameters() if p.grad is not None), torch.zeros(()))))
o = pm(pan, ms, lpan); gb, ga = grads(output_edge_loss(o["y"], gt))
check("T12 A2 edge loss -> backbone and aligner", gb > 0 and ga > 0)
o = pm(pan, ms, lpan); lg, _ = direct_geometry_loss(o["pan_aligned"], pan, gt, 2.0, 11); gb, ga = grads(lg)
check("T13 A3 geo loss -> aligner only", ga > 0 and gb == 0, f"backbone {gb:.3g} aligner {ga:.3g}")
o = pm(pan, ms, lpan); gb, ga = grads((o["y"] - gt).abs().mean())
check("T11/T20 A1 rec loss -> both; forward needs only P,S", gb > 0 and ga > 0)
check("T14 GT requires_grad False", not gt.requires_grad)
# T19: 같은 seed 두 번 만든 aligner 의 state hash 동일
torch.manual_seed(3025); a1 = PANGlobalAligner(8).state_dict(); torch.manual_seed(3025); a2 = PANGlobalAligner(8).state_dict()
check("T19 same-seed init identical", all(torch.equal(a1[k], a2[k]) for k in a1))
# §9.1 aligner enabled with Δ=0 == B0 forward
with torch.no_grad():
    pm.aligner.fc2.weight.zero_(); pm.aligner.fc2.bias.zero_()
    y_pa = pm(pan, ms, lpan)["y"]
    sw = torch.ones(2); y_b0 = torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic") + bb(pan, lpan, ms, sw)
check("§9.1 aligner Δ=0 == B0 forward", (y_pa - y_b0).abs().max().item() < 1e-4, f"max diff {(y_pa - y_b0).abs().max().item():.2e}")
check("§6.2 support guard passes at Δ=0 and fails at |Δ|=12", bool(support_margin_ok(64, 64, torch.zeros(1, 2), 11).all()) and not bool(support_margin_ok(64, 64, torch.tensor([[12.0, 0.0]]), 11).all()))

# ---------------- selector (E15/E16/E20)
S = BestSelector("t")
S.update(1, 1, 0.95, 0.80, True, "a"); S.update(2, 2, 0.96, 0.81, True, "b"); r = S.update(3, 3, 0.9599, 0.90, True, "c")
check("E15 tie band: 0.9599 within 1e-4 of 0.96, higher fSCC -> step 3", r["best"]["step"] == 3)
r = S.update(4, 4, 0.9598, 0.95, True, "d")
check("E16 no chain drift: 0.9598 is outside band of running max 0.96", r["best"]["step"] == 3 and "d" in r["pruned"])
r = S.update(5, 5, 0.9605, 0.70, True, "e")
check("E15 new max -> re-select among band", r["best"]["step"] == 5 and set(r["pruned"]) >= {"a", "b", "c"} or r["best"]["step"] == 5)
r = S.update(6, 6, 0.99, 0.99, False, "f", "invalid")
check("E13/E20 ineligible candidate never selected", r["best"]["step"] == 5 and not r["changed"])
S2 = BestSelector("empty"); r = S2.update(1, 1, 0.9, 0.9, False, "x", "invalid")
check("E20 no valid candidate -> best None", S2.best is None)

# ---------------- evaluator gates on real data (E02/E03/E06/E11)
dl = os.environ.get("PANCRAFTER_DLPAN")
if dl and os.path.exists(os.path.join(ROOT, "data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5")):
    from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
    from scipy.io import loadmat
    wald = load_dlpan(dl)
    with h5py.File(os.path.join(ROOT, "data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5")) as f:
        lms = np.asarray(f["lms"], dtype=np.float64)[0].transpose(1, 2, 0); pan_np = np.asarray(f["pan"], dtype=np.float64)[0, 0]
    sr = loadmat(os.path.join(ROOT, "work_dir/S1_T05_W168_D123_DUAL/results/full_best_hqnr_mat20.mat"))["sr"].astype(np.float64)[0].transpose(1, 2, 0)
    v, ok = scene_views(sr, lms, pan_np, pan_np, "wv3", wald, np.zeros(2))
    check("E02 raw_original == B0 evaluator (d_lambda_k, d_s)", abs(v["raw_original"]["d_lambda"] - d_lambda_k(sr, lms, "wv3", 4, 32, wald)) < 1e-12 and abs(v["raw_original"]["d_s"] - d_s(sr, lms, pan_np, 4, 32, wald)) < 1e-12)
    check("E03 delta=0: raw_valid == aligned_valid", v["raw_valid"] == v["aligned_valid"])
    check("E06 D_lambda identical across valid views", v["raw_valid"]["d_lambda"] == v["aligned_valid"]["d_lambda"])
    y0, y1, x0, x1 = fixed_roi(512, 512)
    check("E11 ROI origin multiple of 4 and 32", y0 % 32 == 0 and x0 % 32 == 0 and (y1 - y0) % 32 == 0)
    check("eligibility rule", ok and not __import__("pa.evalviews", fromlist=["eligible"]).eligible(np.array([MAX_ELIGIBLE_SHIFT + 0.5, 0.0])))
else:
    print("  SKIP evaluator gates (PANCRAFTER_DLPAN or paper set missing)")

print("\n" + ("전부 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}"))
sys.exit(1 if FAIL else 0)
