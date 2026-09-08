#!/usr/bin/env python
"""학습된 run 의 best checkpoint 를 다른 센서 테스트셋에 그대로 적용하는 zero-shot 평가 run 을 만든다.

    python tools/make_zeroshot_run.py --run S1_T05_W168_D123_DUAL --sensor wv2
      -> work_dir/S1_T05_W168_D123_DUAL_zs_wv2/  (meta/config.yaml · best_hqnr -> 원 run 심볼릭 링크 ·
         results/reduced_best_hqnr.mat · results/fr_mat20.json)

논문(PAN-Crafter Table 3)의 "WV3 학습 → WV2 unseen" 과 같은 절차다. 밴드 수가 같은 센서끼리만 된다(WV3↔WV2 8밴드).
만들어진 디렉터리는 보통 run 처럼 시트에 올린다(gspread 가 test dataroot 로 센서를 읽어 WV2 탭에 둔다).
lpan 은 배포본이 없어 F-1 레시피로 생성한 것(tools/setup_wv2.py) — 검증 불가라는 점을 Notes 에 남긴다.
"""
import os, sys, json, argparse, datetime
import numpy as np, yaml, torch
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from main import import_class                                   # noqa: E402
import tools.eval_fr_paperset as efp                            # noqa: E402

DATA = {  # sensor -> (reduced, full, bands)
    "wv2": ("data/PanCollection/WV2/reduced_examples_h5/test_wv2_multiExm1.h5",
            "data/PanCollection/WV2/full_examples_h5/test_wv2_OrigScale_multiExm1.h5", 8),
    "wv3": ("data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5",
            "data/PanCollection/WV3/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5", 8),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--sensor", required=True, choices=list(DATA))
    ap.add_argument("--force", action="store_true"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    src = os.path.join(ROOT, "work_dir", a.run); cfg = yaml.safe_load(open(os.path.join(src, "meta", "config.yaml")))
    red, full, nb = DATA[a.sensor]
    assert int(cfg.get("num_bands", 8)) == nb, f"밴드 수가 다르다: run {cfg.get('num_bands')} vs {a.sensor} {nb}"
    for p in (red, full, full.replace(".h5", "_pan.h5")):
        assert os.path.exists(os.path.join(ROOT, p)), f"{p} 없음 — tools/setup_wv2.py"
    ckpt = efp.pick_ckpt(src); assert ckpt, "checkpoint 없음"
    tag = f"{a.run}_zs_{a.sensor}"; wd = os.path.join(ROOT, "work_dir", tag)
    os.makedirs(os.path.join(wd, "meta"), exist_ok=True); os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    c2 = dict(cfg); c2["work_dir"] = wd
    c2["test_reduced_feeder_args"] = dict(cfg["test_reduced_feeder_args"], dataroot=os.path.join(ROOT, red))
    c2["test_full_feeder_args"] = dict(cfg["test_full_feeder_args"], dataroot=os.path.join(ROOT, full))
    c2["zero_shot_of"] = a.run
    yaml.safe_dump(c2, open(os.path.join(wd, "meta", "config.yaml"), "w"), allow_unicode=True, sort_keys=False)
    link = os.path.join(wd, ckpt)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link) if os.path.islink(link) else None
    if not os.path.exists(link):
        os.symlink(os.path.relpath(os.path.join(src, ckpt), wd), link)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for k in ("started_at.txt", "finished_at.txt"):        # 학습이 아니므로 둘 다 지금 (eval_fr_paperset 의 진행중 판정 회피)
        open(os.path.join(wd, "meta", k), "w").write(now + "\n")
    open(os.path.join(wd, "meta", "zero_shot.txt"), "w").write(f"source run: {a.run}\ncheckpoint: {ckpt} (symlink)\nsensor: {a.sensor}\n")
    # RR 추론 -> results/reduced_best_hqnr.mat (train.py test_reduced_save 와 같은 저장 규약: clip(-1,1) -> DN)
    mat = os.path.join(wd, "results", "reduced_best_hqnr.mat")
    if not os.path.exists(mat) or a.force:
        m, fwd, how = efp.build(c2, wd, ckpt); m = m.to(a.device).eval()
        Feeder = import_class(c2["feeder"]); ds = Feeder(**c2["test_reduced_feeder_args"]); mp = float(ds.max_pixel)
        srs = []
        with torch.no_grad():
            for i in range(len(ds)):
                gt, lms, ms, lpan, pan = (t.unsqueeze(0).to(a.device) for t in ds[i])
                y = fwd(pan, lpan, ms, lms); srs.append(((y.clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp)[0])
        savemat(mat, dict(sr=np.stack(srs))); print(f"[zs] RR {len(srs)}장 -> {os.path.relpath(mat, ROOT)}  ({how})")
    # FR (논문 세트)
    wald = efp.load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    j, st = efp.run_one(tag, None, wald, a.device, a.force)
    if j is None:
        print(f"[zs] FR 건너뜀 ({st})")
    else:
        print(f"[zs] FR {a.sensor} paper set: HQNR {j['hqnr']:.4f}±{j['hqnr_sd']:.4f}  D_l {j['d_lambda']:.4f}  D_s {j['d_s']:.4f}  [{st}]")
    print(f"[zs] 완료: work_dir/{tag}  -> 시트: ./tools/_upload.sh {tag}")


if __name__ == "__main__":
    main()
