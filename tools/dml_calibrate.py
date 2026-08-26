"""lambda 보정 — anchor 대비 mutual gradient 비 r_g 를 잰다 (계획 6절).

  python tools/dml_calibrate.py --config config/dml_m1.yaml [--batch 12]

학습 루프 안에서 재면 retain_graph 로 peak VRAM 이 30->34 GiB 를 넘어 OOM 난다.
r_g 는 스텝마다 필요한 값이 아니라 lambda 를 정하기 위한 상수이므로 여기서 작은 배치로
따로 잰다. 목표는 실제 mutual 기여 lambda x r_g 가 0.1~0.2 가 되는 lambda 다.
"""
import argparse, os, sys, yaml, importlib, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def grad_norm(m):
    return sum(p.grad.detach().float().pow(2).sum().item()
               for p in m.parameters() if p.grad is not None) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=12, help="nominal batch (복제 후 2배)")
    ap.add_argument("--steps", type=int, default=30, help="이 스텝만큼 anchor 로 워밍업 후 측정")
    ap.add_argument("--random", action="store_true",
                    help="난수 입력으로 잰다(기본은 실제 학습 데이터). 절대값 신뢰도가 낮다")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config))
    mod, cls = c["model"].rsplit(".", 1)
    M = getattr(importlib.import_module(mod), cls)
    bands, dev = c["num_bands"], "cuda"
    B = a.batch * 2

    def peer(seed):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        return M(**c["model_args"]).to(dev).train()

    A, Bp = peer(c["seed"]), peer(c.get("seed_b", 2026))
    lr = float(c["learning_rate"])          # YAML 은 1e-4 를 str 로 읽는다
    oa = torch.optim.AdamW(A.parameters(), lr)
    ob = torch.optim.AdamW(Bp.parameters(), lr)
    if a.random:
        g = torch.Generator(device="cpu").manual_seed(0)
        def batch():
            t = lambda *s: torch.randn(*s, generator=g).to(dev)
            return (t(B,1,64,64), t(B,1,16,16), t(B,bands,16,16), t(B,bands,64,64))
    else:
        # 실제 학습 데이터로 잰다 — 계획 6절이 요구하는 것은 1K smoke(실데이터) 기준이다.
        # 난수 입력은 두 peer 가 과도하게 불일치해 r_g 가 부풀려진다.
        from main import import_class
        Feeder = import_class(c["feeder"])
        ds = Feeder(**c["train_feeder_args"])
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True,
                                         num_workers=2, drop_last=True)
        it = iter(dl)
        def batch():
            nonlocal it
            try:
                gt_, lms_, ms_, lpan_, pan_ = next(it)
            except StopIteration:
                it = iter(dl); gt_, lms_, ms_, lpan_, pan_ = next(it)
            rep = lambda t: t.to(dev).repeat(2, 1, 1, 1)
            return rep(pan_), rep(lpan_), rep(ms_), rep(gt_)
    sw = torch.cat([torch.zeros(a.batch, device=dev), torch.ones(a.batch, device=dev)])

    def fwd(m, pan, lpan, ms):
        out = m(pan, lpan, ms, sw)
        base_ms = F.interpolate(ms, scale_factor=4, mode="bicubic")
        base_pan = F.interpolate(lpan, scale_factor=4, mode="bicubic").repeat(1, bands, 1, 1)
        s4 = sw.view(-1,1,1,1)
        return out + base_ms * s4 + base_pan * (1 - s4)

    # 난수 입력으로는 gradient 크기가 실제와 다르므로 몇 스텝 anchor 로 굴려 안정화한다
    for _ in range(a.steps):
        pan, lpan, ms, gt = batch()
        for m, o in ((A, oa), (Bp, ob)):
            r = fwd(m, pan, lpan, ms)
            ((gt[a.batch:] - r[a.batch:]).abs().mean()
             + (pan[:a.batch].repeat(1,bands,1,1) - r[:a.batch]).abs().mean() * float(c["w_off"])).backward()
            o.step(); o.zero_grad(set_to_none=True)

    pan, lpan, ms, gt = batch()
    with torch.no_grad():
        rb = fwd(Bp, pan, lpan, ms)
    ra = fwd(A, pan, lpan, ms)
    anchor = ((gt[a.batch:] - ra[a.batch:]).abs().mean()
              + (pan[:a.batch].repeat(1,bands,1,1) - ra[:a.batch]).abs().mean() * float(c["w_off"]))
    A.zero_grad(set_to_none=True); anchor.backward(retain_graph=True); gA = grad_norm(A)
    A.zero_grad(set_to_none=True)
    ((ra[a.batch:] - rb[a.batch:]).abs().mean()).backward(); gM = grad_norm(A)

    rg = gM / gA
    print(f"config          : {a.config}")
    print(f"nominal batch   : {a.batch}  (MARs 복제 후 {B})")
    print(f"|grad anchor|   : {gA:.5f}")
    print(f"|grad mutual|   : {gM:.5f}")
    print(f"r_g             : {rg:.4f}")
    print(f"\n목표: lambda x r_g 가 0.1~0.2  ->  lambda 권장 구간 "
          f"{0.1/rg:.4f} ~ {0.2/rg:.4f}")
    lam = float(c.get("mutual_lambda") or 0.05)
    print(f"현재 config lambda {lam}  ->  lambda x r_g = {lam*rg:.4f}  "
          + ("적정" if 0.1 <= lam*rg <= 0.2 else ("너무 강함" if lam*rg > 0.2 else "너무 약함")))
    src = "난수 입력 — 절대값 신뢰도 낮음" if a.random else "실제 학습 데이터"
    print(f"\n측정 입력: {src}")


if __name__ == "__main__":
    sys.exit(main())
