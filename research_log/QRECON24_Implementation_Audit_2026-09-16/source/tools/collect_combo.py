"""Student 아키텍처 스윕 결과를 모아 results_log 문서와 그림을 만든다.

끝난 것만 모으므로 학습 도중 아무 때나 돌려도 된다. 모델별 평가 결과는
work_dir/<exp>/eval_summary.json 에 캐시되어 재계산하지 않는다.

  python tools/collect_sweep.py
"""
import os, sys, glob, json
import numpy as np, h5py
from scipy.io import loadmat
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.metrics.eval_rr import evaluate as rr_eval          # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s  # noqa: E402

SCALE, CUT, BLK = 2047.0, slice(20, -21), 32
DOC = os.path.join(ROOT, "results_log/2026-08-21_WIP_combo-ablation.md")
ASSET = os.path.join(ROOT, "results_log/assets")


def _scc(F, G):
    lf = np.sqrt(sobel(F[1:-1,1:-1,:],0)**2 + sobel(F[1:-1,1:-1,:],1)**2)
    lg = np.sqrt(sobel(G[1:-1,1:-1,:],0)**2 + sobel(G[1:-1,1:-1,:],1)**2)
    return float((lf*lg).sum()/np.sqrt((lf**2).sum())/np.sqrt((lg**2).sum()))


def eval_one(wd, gt, lms_r, lms_f, pan_f, wald):
    """한 실행의 reduced/full 지표. 이미 계산돼 있으면 캐시를 읽는다."""
    cache = os.path.join(wd, "eval_summary.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    rp, fp = f"{wd}/results/reduced_best_val.mat", f"{wd}/results/full_best_val.mat"
    if not (os.path.exists(rp) and os.path.exists(fp)):
        return None
    sr = loadmat(rp)["sr"].astype(np.float64)
    sr = sr.transpose(0,2,3,1) if sr.shape[1] in (4,8) else sr
    sr = sr[:, CUT, CUT]
    m = rr_eval(sr, gt, SCALE, BLK)
    out = dict(ergas=m["ERGAS"][0], sam=m["SAM"][0], q2n=m["Q2n"][0],
               psnr=float(np.mean([10*np.log10(SCALE**2/((sr[i]-gt[i])**2).mean()) for i in range(len(gt))])),
               ssim=float(np.mean([structural_similarity(gt[i]/SCALE, sr[i]/SCALE, data_range=1.0,
                                                         channel_axis=-1) for i in range(len(gt))])),
               scc=float(np.mean([_scc(sr[i], gt[i]) for i in range(len(gt))])))
    fs = loadmat(fp)["sr"].astype(np.float64)
    fs = fs.transpose(0,2,3,1) if fs.shape[1] in (4,8) else fs
    dl = [d_lambda_k(np.clip(fs[i],0,2**11), lms_f[i], "wv3", 4, BLK, wald) for i in range(len(fs))]
    ds = [d_s(np.clip(fs[i],0,2**11), lms_f[i], pan_f[i], 4, BLK, wald) for i in range(len(fs))]
    out.update(d_lambda=float(np.mean(dl)), d_s=float(np.mean(ds)),
               hqnr=float(np.mean([(1-a)*(1-b) for a,b in zip(dl,ds)])))
    json.dump(out, open(cache, "w"), indent=1)
    return out


def main():
    prof = {r["tag"]: r for r in json.load(open(f"{ASSET}/combo_profile.json"))}
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5") as f:
        gt = f["gt"][:].astype(np.float64).transpose(0,2,3,1)[:, CUT, CUT]
        lms_r = f["lms"][:].astype(np.float64).transpose(0,2,3,1)[:, CUT, CUT]
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/full_examples_h5/test_wv3_OrigScale_multiExm1.h5") as f:
        lms_f = f["lms"][:].astype(np.float64).transpose(0,2,3,1); pan_f = f["pan"][:].astype(np.float64)[:,0]
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))

    rows = []
    for tag, p in prof.items():
        r = eval_one(f"{ROOT}/work_dir/combo_{tag}", gt, lms_r, lms_f, pan_f, wald)
        rows.append({**p, **(r or {}), "done": r is not None})
    done = [r for r in rows if r["done"]]
    rows.sort(key=lambda r: -r["params"])

    # 기준선: 보간 입력
    base = rr_eval(lms_r, gt, SCALE, BLK)
    print(f"완료 {len(done)}/{len(rows)}\n")
    hdr = (f"{'구성':<15}{'params(M)':>10}{'비율':>6}{'ms':>6}{'GB':>6}"
           f"{'ERGAS':>8}{'SAM':>7}{'Q2n':>7}{'HQNR':>7}{'ERGAS/ms':>10}")
    print(hdr); print("-"*len(hdr))
    T = rows[0]
    for r in rows:
        if not r["done"]:
            print(f"{r['tag']:<15}{r['params']/1e6:>10.3f}{r['params']/T['params']:>6.3f}"
                  f"{r['infer_ms']:>6.1f}{r['train_mem_gb']:>6.2f}{'  (학습 중/대기)':>32}")
            continue
        gain = ((base["ERGAS"][0] - r["ergas"]) / r["infer_ms"]) if r.get("infer_ms") else float("nan")
        print(f"{r['tag']:<15}{r['params']/1e6:>10.3f}{r['params']/T['params']:>6.3f}"
              f"{r['infer_ms']:>6.1f}{r['train_mem_gb']:>6.2f}{r['ergas']:>8.4f}{r['sam']:>7.4f}"
              f"{r['q2n']:>7.4f}{r['hqnr']:>7.4f}{gain:>10.4f}")
    json.dump(rows, open(f"{ASSET}/combo_results.json","w"), indent=1)
    write_doc(rows, base, len(done))
    if len(done) >= 2: make_fig(done, base)
    print(f"\n문서: {os.path.relpath(DOC, ROOT)}")


def write_doc(rows, base, ndone):
    T = rows[0]
    L = ["# [WIP] Student 아키텍처 스윕 — 가성비 탐색 (2026-08-19)", "",
         f"> **진행 중 — {ndone}/{len(rows)} 완료.** 결과가 나오는 대로 이 문서가 갱신된다.", "",
         "PAN-Crafter 를 Teacher 후보로 두고 **width(W) × depth(D) × CM3A 개수(A)** 세 축으로 12개 구성을",
         "동일 조건에서 학습해 성능/비용 파레토를 찾는다. 근거: "
         "[research_log/pancrafter_student_first_strategy_v0.1.md](../research_log/pancrafter_student_first_strategy_v0.1.md)", "",
         "**공통 설정**: WV3, seed 2025, nominal batch 48(실효 96), AdamW 1e-4 / wd 0.01, cosine + warmup 100,",
         "`select_on: val`, `fix_key_alias/fix_local_attn: True`, full-res 는 복구 `lpan` 사용.",
         "**스윕 전용**: `num_iter 25000` (원 설정의 50%), `eval_epoch 10`.", "",
         "> **주의**: 25K 는 완전 수렴이 아니다(Teacher 기준 최종 대비 1.9% 미수렴). 순위 판정에는 충분하나",
         "> **작은 모델이 빨리 수렴하므로 소형 쪽에 약간 유리**하다. 우승 후보는 50K 로 재확인해야 한다.", "",
         f"기준선(보간 입력 lms): ERGAS {base['ERGAS'][0]:.4f} / SAM {base['SAM'][0]:.4f} / Q2n {base['Q2n'][0]:.4f}", "",
         "| 구성 | W | D | CM3A | Params(M) | 비율 | FLOPs(G) | 추론(ms) | 학습(GB) | ERGAS↓ | SAM↓ | Q2n↑ | SSIM↑ | HQNR↑ |",
         "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        d = "".join(map(str, r["depth"]))
        head = (f"| {r['tag']} | {r['width']} | {d} | {r['cm3a']} | {r['params']/1e6:.3f} | "
                f"{r['params']/T['params']:.3f} | {r['flops_g']:.1f} | {r['infer_ms']:.1f} | {r['train_mem_gb']:.2f} |")
        L.append(head + (f" {r['ergas']:.4f} | {r['sam']:.4f} | {r['q2n']:.4f} | {r['ssim']:.4f} | {r['hqnr']:.4f} |"
                         if r["done"] else " — | — | — | — | — |"))
    L += ["", "지표는 DLPan 프로토콜(`tools/eval_dlpan.py` / `eval_dlpan_fr.py`), full-res 는 전체 20장 기준이다.",
          "PSNR·SSIM 은 표준 프로토콜 밖이라 논문 절대값과 비교하지 말 것.", ""]
    if ndone >= 2:
        L += ["![](assets/combo_plot.png)", "*성능 대비 비용. 좌하단이 유리하다.*", ""]
    L += ["## 다음", "", "1. 전체 완료 후 파레토 무릎점 선정",
          "2. 우승 후보 1~2개를 50K 로 재학습해 확정 (각 1~3시간)",
          "3. Teacher–Student 오차 상보성 측정 → mutual learning go/no-go", ""]
    open(DOC, "w").write("\n".join(L))


def make_fig(done, base):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (xk, xl) in zip(ax, [("infer_ms","inference time (ms)"),
                                ("params","parameters"), ("train_mem_gb","training memory (GB)")]):
        xs = [r[xk]/1e6 if xk=="params" else r[xk] for r in done]
        ys = [r["ergas"] for r in done]
        CMAP = {5:"#dc2626",4:"#ea580c",3:"#2563eb",2:"#7c3aed",1:"#16a34a",0:"#64748b"}
        cs = [CMAP.get(r["cm3a"], "#64748b") for r in done]
        a.scatter(xs, ys, c=cs, s=70, zorder=3)
        for r, x, y in zip(done, xs, ys):
            a.annotate(r["tag"].replace("D1121","").replace("D2222","*"), (x, y), fontsize=7,
                       xytext=(4,4), textcoords="offset points")
        a.axhline(base["ERGAS"][0], ls=":", c="gray", lw=1)
        a.set_xlabel(xl + ("  (M)" if xk=="params" else "")); a.set_ylabel("ERGAS (lower better)")
        a.grid(alpha=.3)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=f"CM3A {k}")
                          for k,c in sorted({r["cm3a"]: CMAP.get(r["cm3a"],"#64748b") for r in done}.items(), reverse=True)], fontsize=8)
    fig.suptitle("PAN-Crafter student sweep - cost vs quality (WV3, 25K iters).  * = depth 2222", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{ASSET}/combo_plot.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
