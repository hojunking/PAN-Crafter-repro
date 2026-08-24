"""임의 실험군의 결과를 모아 표와 마크다운을 만든다.

  python tools/collect_exp.py "x*" results_log/2026-08-22_WIP_extended-ablation.md

첫 인자는 work_dir 아래 glob 패턴, 둘째는 출력 문서. 평가 결과는 각 work_dir 의
eval_summary.json 에 캐시되어 재계산하지 않는다.
"""
import os, sys, glob, json
import numpy as np, h5py
from scipy.io import loadmat
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.environ.get("PANCRAFTER_CANCONV", "/home/knuvi/Desktop/song/CANConv"))
from tools.eval_rr import evaluate as rr_eval          # noqa: E402
from tools.eval_fr import load_dlpan, d_lambda_k, d_s  # noqa: E402
SCALE, CUT, BLK = 2047.0, slice(20, -21), 32


def _scc(F, G):
    lf = np.sqrt(sobel(F[1:-1,1:-1,:],0)**2 + sobel(F[1:-1,1:-1,:],1)**2)
    lg = np.sqrt(sobel(G[1:-1,1:-1,:],0)**2 + sobel(G[1:-1,1:-1,:],1)**2)
    return float((lf*lg).sum()/np.sqrt((lf**2).sum())/np.sqrt((lg**2).sum()))


def eval_one(wd, gt, lms_f, pan_f, wald):
    cache = os.path.join(wd, "eval_summary.json")
    if os.path.exists(cache):
        try: return json.load(open(cache))
        except Exception: pass
    rp = f"{wd}/results/reduced_best_val.mat"
    # full-res 는 복구 lpan 으로 만든 것을 우선한다. 구 실행(valsel)은 손상본으로 만들어져 있어
    # full_best_val.mat 을 그대로 쓰면 HQNR 이 무의미해진다 (KNOWN_ISSUES F-1).
    fp = next((q for q in (f"{wd}/results/full_frrepair.mat", f"{wd}/results/full_best_val.mat")
               if os.path.exists(q)), None)
    if not (os.path.exists(rp) and fp): return None
    sr = loadmat(rp)["sr"].astype(np.float64)
    sr = (sr.transpose(0,2,3,1) if sr.shape[1] in (4,8) else sr)[:, CUT, CUT]
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


# 문서별 머리말. WIP 문서는 매번 통째로 덮어쓰이므로, CONVENTION.md §2 가 요구하는
# "이 실험으로 확인하려는 것" 을 여기 둬야 갱신돼도 살아남는다. 키는 문서 파일명이다.
INTRO = {
    "2026-08-24_WIP_running.md": (
        "[WIP] 진행 중인 실험 — 두 축 결합 + 논문 격차 진단 (2026-08-24)",
        ["두 갈래가 순차로 돈다. 결과는 [재현 감사 보고서](2026-08-24_reproduction-audit.md)에 합류한다.",
         "",
         "### A. 두 축(폭 × CM3A 개수) 결합 — `y_*`, `z_*` (8종, ≈16:28 종료)",
         "",
         "[8/22 확장 실험](2026-08-22_extended-ablation-and-kd-target.md) §4 의 미측정 구간이다.",
         "폭 축소와 CM3A 축소를 각각은 재봤지만 **결합했을 때가 가산인지 초가산인지** 모른다.",
         "`z_seed1234`/`z_seed7777` 은 권장 구성(6.694M) 시드 반복으로, 지금까지의 ±0.4% 차이가",
         "의미 있는 값인지 판정할 **시드 변동 폭**을 준다. 손실이 나타나는 지점이 **KD 의 실제 작업 대상**이다.",
         "",
         "### B. 논문 격차 원인 진단 — `d1_nocrop`, `d2_lmsbase` (2종, ≈21:10 종료)",
         "",
         "[재현 감사](2026-08-24_reproduction-audit.md) 의 후보 B·C 검증이다.",
         "대조군은 아래 **[기준] A5 Teacher (9.969M)** 행(`sweep_W128D2222A5`, 25K, ERGAS 2.2598)이고,",
         "두 실행은 거기서 argparse 파싱 기준 **정확히 한 항목만** 다르다.",
         "",
         "| 실행 | 대조군 대비 차이 | 검증하는 후보 |",
         "|---|---|---|",
         "| `d1_nocrop` | `train_feeder_args.crop: True → False` | B — 배포본 `crop` 이 실은 scale jitter라 MS→GT 열화관계를 깬다 |",
         "| `d2_lmsbase` | `residual_base: bicubic → lms` | C — 잔차 기준선 선택이 최종 품질에 남는가 |",
         "",
         "**무엇을 보면 되는가.** 2.2598 보다 유의하게 낮아지면 그 후보가 격차의 일부다.",
         "**둘 다 무차별이면 후보 A(배포 코드 ≠ 논문 모델)만 남고**, 이쪽 조사는 종료한다.",
         "",
         "> 두 실행 모두 `fix_*: True`(수정본) 계열이라 대조군과 조건이 맞는다.",
         "> 배포본 그대로의 재현 수치는 `wv3_baseline`(2.1633) 이다 — 감사 보고서 참고."],
        ["y_W96D1121_A1", "y_W96D1121_A2", "y_W128D1121_A2", "y_W96D2222_A2",
         "y_W64D1121_A1", "y_W64D1121_A2", "z_seed1234", "z_seed7777",
         "d1_nocrop", "d2_lmsbase"]),
}


# 08-24 두 체인(_run_twoaxis.sh, _run_diag.sh)은 각자 다른 문서를 쓰도록 짜여 있는데,
# 실행 중이라 스크립트를 고칠 수 없다. 여기서 경로/패턴을 하나로 돌려 문서를 합친다.
# 두 체인은 순차 실행(diag 가 twoaxis 종료를 기다림)이라 경합하지 않는다.
MERGE = {"2026-08-24_WIP_twoaxis.md": ("2026-08-24_WIP_running.md", "[yzd]*"),
         "2026-08-24_WIP_diag.md":    ("2026-08-24_WIP_running.md", "[yzd]*")}


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "x*"
    doc = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results_log/2026-08-22_WIP_extended-ablation.md"
    if os.path.basename(doc) in MERGE:
        newdoc, pat = MERGE[os.path.basename(doc)]
        doc = os.path.join(os.path.dirname(doc) or ".", newdoc)
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5") as f:
        gt = f["gt"][:].astype(np.float64).transpose(0,2,3,1)[:, CUT, CUT]
    with h5py.File(f"{ROOT}/data/PanCollection/WV3/full_examples_h5/test_wv3_OrigScale_multiExm1.h5") as f:
        lms_f = f["lms"][:].astype(np.float64).transpose(0,2,3,1); pan_f = f["pan"][:].astype(np.float64)[:,0]
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))

    # 기준선 두 개
    refs = {}
    for nm, wd in [("A3 (8.030M)", "sweep_W128D2222A3"), ("A5 Teacher (9.969M)", "sweep_W128D2222A5"),
                   ("A5 Teacher 50K", "wv3_fixed_valsel")]:
        r = eval_one(f"{ROOT}/work_dir/{wd}", gt, lms_f, pan_f, wald)
        if r: refs[nm] = r

    rows = []
    for d in sorted(glob.glob(f"{ROOT}/work_dir/{pat}")):
        if not os.path.isdir(d): continue
        r = eval_one(d, gt, lms_f, pan_f, wald)
        rows.append((os.path.basename(d), r))
    done = [(t, r) for t, r in rows if r]

    print(f"완료 {len(done)}/{len(rows)}\n")
    hdr = f"{'구성':<24}{'ERGAS':>9}{'SAM':>8}{'Q2n':>8}{'SSIM':>8}{'HQNR':>8}"
    print(hdr); print("-"*len(hdr))
    for nm, r in refs.items():
        print(f"{'[기준] '+nm:<24}{r['ergas']:>9.4f}{r['sam']:>8.4f}{r['q2n']:>8.4f}{r['ssim']:>8.4f}{r['hqnr']:>8.4f}")
    print("-"*len(hdr))
    for t, r in rows:
        if r: print(f"{t:<24}{r['ergas']:>9.4f}{r['sam']:>8.4f}{r['q2n']:>8.4f}{r['ssim']:>8.4f}{r['hqnr']:>8.4f}")
        else: print(f"{t:<24}{'(학습 중/대기)':>41}")

    title, body, planned = (INTRO.get(os.path.basename(doc)) + (None,))[:3] if \
        os.path.basename(doc) in INTRO else (
        "[WIP] 확장 서브모듈 실험 (2026-08-22)",
        ["[8/21 서브모듈 제거 실험](2026-08-21_submodule-ablation.md)의 후속. 세 가지를 확인한다.", "",
         "1. **x1** — 비싼 원자(PAN 브랜치·encoder H/4)의 조합. 8/21 은 안전한 원자 위주였다.",
         "2. **x2** — 배포본(A5, CM3A 5개) 기준 재확인. 8/21 결론이 A3 조건부인지 검증.",
         "3. **x3** — 최대 감축 구성 50K. 25K 미수렴 편향 제거."], None)
    # planned 가 있으면 진행률은 그 목록만으로 센다. glob 패턴이 과거 실험 디렉터리까지
    # 잡는 경우(예: "[xyz]*" 가 8/22 의 x* 를 포함) 완료 수가 부풀려지기 때문이다.
    # 잡힌 나머지 행은 표에 대조용으로 남긴다.
    ok = {t for t, r in rows if r}
    track = list(planned) if planned else [t for t, _ in rows]
    pend = [t for t in track if t not in ok]
    total = len(track)
    done = [(t, r) for t, r in rows if r and t in set(track)]
    L = [f"# {title}", "",
         f"> **진행 중 — {len(done)}/{total} 완료.** 결과가 나오는 대로 자동 갱신된다.",
         f"> 전부 끝나면 정식 문서로 대체하고 이 파일은 지운다 (`CONVENTION.md` §2).", ""]
    L += body
    L += ["", f"**진행 상태** — 완료 {len(done)}/{total}"
              + (f", 대기/학습 중: {', '.join('`'+x+'`' for x in pend)}" if pend else ", 전부 완료"), "",
          "| 구성 | ERGAS↓ | SAM↓ | Q2n↑ | SSIM↑ | HQNR↑ |", "|---|--:|--:|--:|--:|--:|"]
    for nm, r in refs.items():
        L.append(f"| **[기준] {nm}** | {r['ergas']:.4f} | {r['sam']:.4f} | {r['q2n']:.4f} | {r['ssim']:.4f} | {r['hqnr']:.4f} |")
    for t, r in rows:
        L.append(f"| {t} | {r['ergas']:.4f} | {r['sam']:.4f} | {r['q2n']:.4f} | {r['ssim']:.4f} | {r['hqnr']:.4f} |"
                 if r else f"| {t} | — | — | — | — | — |")
    L += ["", "지표는 DLPan 프로토콜(`tools/eval_dlpan.py`), full-res 는 복구 `lpan` 전체 20장 기준이다.", ""]
    open(doc, "w").write("\n".join(L))
    print(f"\n문서: {os.path.relpath(doc, ROOT)}")


if __name__ == "__main__":
    main()
