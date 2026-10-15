"""Teacher / Student 후보의 파라미터·FLOPs·추론시간·메모리·학습비용을 한 번에 잰다.

  python tools/profile_models.py                 # 기본 세트
  python tools/profile_models.py --bands 4       # QB/GF2

논문 보충 Table 10 은 reduced-resolution 256x256xC_MS 출력 기준이므로 추론은 그 조건으로 잰다.
학습 비용은 64x64 patch, MARs 복제 후 실효 배치 기준이다.
"""
import argparse, time, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pancrafter import PANCrafter

CFGS = [
    ("Teacher (배포 config)", [128]*4, [2,2,2,2], 128),
    ("S96-Mid",               [96]*4,  [1,1,2,1], 96),
    ("S96-Detail",            [96]*4,  [2,1,1,1], 96),
]

def build(h, d, se, bands):
    return PANCrafter(in_channels=1, out_channels=bands, hidden_size=h, s_embed_size=se,
                      dropout=0.2, depth=d, num_heads=8, pan_ks=3, ms_ks=3, ka=3)

def infer_inputs(b, pan_hw, bands, dev):
    return (torch.randn(b,1,pan_hw,pan_hw,device=dev),
            torch.randn(b,1,pan_hw//4,pan_hw//4,device=dev),
            torch.randn(b,bands,pan_hw//4,pan_hw//4,device=dev),
            torch.ones(b,device=dev))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48, help="MARs 복제 전 nominal batch")
    a = ap.parse_args()
    dev = "cuda"
    print(f"밴드 {a.bands} / 추론 256x256 / 학습 64x64, nominal batch {a.batch} (MARs 복제 후 {a.batch*2})\n")
    hdr = (f"{'모델':<22}{'Params(M)':>10}{'FLOPs(G)':>10}{'추론(ms)':>10}"
           f"{'추론mem(MB)':>12}{'학습mem(GB)':>12}{'s/iter':>9}{'50K(h)':>8}")
    print(hdr); print("-"*len(hdr)+"-"*10)
    base = None
    for name, h, d, se in CFGS:
        net = build(h, d, se, a.bands).to(dev).eval()
        n = sum(p.numel() for p in net.parameters())

        from thop import profile
        with torch.no_grad():
            f, _ = profile(net, inputs=infer_inputs(1,256,a.bands,dev), verbose=False)

        x = infer_inputs(1,256,a.bands,dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3): net(*x)
            torch.cuda.synchronize(); t0=time.time()
            for _ in range(20): net(*x)
            torch.cuda.synchronize(); ms=(time.time()-t0)/20*1000
        imem = torch.cuda.max_memory_allocated()/1024**2

        net.train()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-4)
        B = a.batch*2
        xt = infer_inputs(B,64,a.bands,dev); gt = torch.randn(B,a.bands,64,64,device=dev)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for i in range(6):
            if i==2: torch.cuda.synchronize(); t0=time.time()
            (net(*xt)-gt).abs().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); sit=(time.time()-t0)/4
        tmem = torch.cuda.max_memory_allocated()/1024**3

        print(f"{name:<22}{n/1e6:>10.3f}{f/1e9:>10.2f}{ms:>10.2f}{imem:>12.1f}"
              f"{tmem:>12.2f}{sit:>9.3f}{sit*50000/3600:>8.2f}")
        if base is None: base = (n, f, ms, imem, tmem, sit)
        else:
            print(f"{'  → Teacher 대비':<22}{n/base[0]:>10.3f}{f/base[1]:>10.3f}"
                  f"{base[2]/ms:>9.2f}x{imem/base[3]:>12.3f}{tmem/base[4]:>12.3f}{base[5]/sit:>8.2f}x")
        del net, opt, xt, gt, x; torch.cuda.empty_cache()
    print("\n논문 보충 Table 10 (PAN-Crafter): Params 7.17 M / FLOPs 79.03 G / 0.009 s / 1751.9 MB")
    print("※ 배포 config 실측 파라미터는 9.969 M 으로 논문 값과 다르다. 비율 계산은 실측 Teacher 기준.")

if __name__ == "__main__":
    main()
