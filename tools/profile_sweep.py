"""config/sweep_*.yaml 각 구성의 정적 비용을 측정해 JSON 으로 저장한다.

학습이 시작되기 전에 한 번만 돌린다 (GPU 를 점유하므로).
  python tools/profile_sweep.py
출력: results_log/assets/sweep_profile.json
"""
import os, sys, glob, json, time, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pancrafter import PANCrafter
from thop import profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_log/assets/sweep_profile.json")
dev = "cuda"

def run(cfg_path, nominal_batch=48):
    c = yaml.safe_load(open(cfg_path))
    ma, bands = c["model_args"], c["num_bands"]
    net = PANCrafter(**ma).to(dev)
    n = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    net.eval()
    x = (torch.randn(1,1,256,256,device=dev), torch.randn(1,1,64,64,device=dev),
         torch.randn(1,bands,64,64,device=dev), torch.ones(1,device=dev))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        f, _ = profile(net, inputs=x, verbose=False)
        for _ in range(5): net(*x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20): net(*x)
        torch.cuda.synchronize(); ms = (time.time()-t0)/20*1000
    imem = torch.cuda.max_memory_allocated()/1024**2

    net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-4)
    B = nominal_batch*2
    xt = (torch.randn(B,1,64,64,device=dev), torch.randn(B,1,16,16,device=dev),
          torch.randn(B,bands,16,16,device=dev), torch.ones(B,device=dev))
    gt = torch.randn(B,bands,64,64,device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for i in range(6):
        if i == 2: torch.cuda.synchronize(); t0 = time.time()
        (net(*xt)-gt).abs().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); sit = (time.time()-t0)/4
    tmem = torch.cuda.max_memory_allocated()/1024**3
    del net, opt, xt, gt, x; torch.cuda.empty_cache()

    return dict(tag=os.path.basename(cfg_path)[6:-5], params=n, trainable=trainable,
                width=ma["hidden_size"][0], depth=ma["depth"], cm3a=len(ma["cm3a_locations"]),
                flops_g=f/1e9, infer_ms=ms, infer_mem_mb=imem, train_mem_gb=tmem,
                s_iter=sit, hours_25k=sit*c["num_iter"]/3600, num_iter=c["num_iter"])

if __name__ == "__main__":
    res = [run(p) for p in sorted(glob.glob(os.path.join(ROOT, "config/sweep_*.yaml")))]
    res.sort(key=lambda r: -r["params"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{'구성':<16}{'params(M)':>10}{'FLOPs(G)':>10}{'추론ms':>8}{'학습GB':>8}{'s/iter':>8}{'25K(h)':>8}")
    print("-"*68)
    for r in res:
        print(f"{r['tag']:<16}{r['params']/1e6:>10.3f}{r['flops_g']:>10.1f}{r['infer_ms']:>8.1f}"
              f"{r['train_mem_gb']:>8.2f}{r['s_iter']:>8.3f}{r['hours_25k']:>8.2f}")
    print(f"\n총 학습 예상 {sum(r['hours_25k'] for r in res):.2f} h  → {OUT}")
