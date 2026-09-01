#!/usr/bin/env python
"""실험 결과를 Google Sheets 로 올린다.

    python gspread/gspread_upload.py paper_ln                # 한 실행
    python gspread/gspread_upload.py "paper*" --profile      # 여러 개 + 비용 측정
    python gspread/gspread_upload.py --all --dry-run         # 올리지 않고 표만 확인

지표는 tools/metrics/ 의 DLPan 프로토콜 구현을 그대로 쓴다. 학습 중 metrics.csv 값이
아니라 .mat 을 다시 평가한 값이라, 논문 Table 과 비교 가능한 수치다.

params 는 항상 계산한다(빠르다). FLOPs·추론시간·메모리는 --profile 일 때만 재고,
한 번 잰 값은 gspread/_profile_cache.json 에 저장해 재사용한다.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CRED = os.path.join(ROOT, "gspread", "account.json")
CACHE = os.path.join(ROOT, "gspread", "_profile_cache.json")
SHEET = "pan-cvpr27"

# 시트는 "<데이터셋>-<서버>" 로 나눈다 (예: WV3-s1, WV3-s2).
# 서버끼리 결과를 섞지 않기 위한 것이다. 같은 config 를 두 서버가 돌리면 실행명이
# 같아지는데, 시트가 갈려 있으면 애초에 충돌하지 않는다.
# 비용(resource) 열은 각 서버의 첫 데이터셋 시트에만 둔다 —
# params·FLOPs·추론시간은 데이터셋과 무관해 한 번만 적으면 된다.
SHEET_ORDER = ["WV3", "QB", "GF2", "WV2"]
SHEET_COLOR = {                       # 시트마다 표 색을 달리한다
    "WV3": (0.82, 0.88, 0.96),        # 파랑
    "QB":  (0.84, 0.93, 0.84),        # 초록
    "GF2": (0.99, 0.90, 0.78),        # 주황
    "WV2": (0.91, 0.86, 0.96),        # 보라
}

# 표는 B열에서 시작한다. 2행 = 그룹 헤더, 3행 = 컬럼명, 4행부터 데이터.
ORIGIN_ROW, ORIGIN_COL = 2, 2

# (그룹, 표시명, 키, 소수자리)
COLUMNS = [
    ("", "Run", "tag", None),
    # reduced-resolution (테스트 20장)
    ("RR", "ERGAS↓", "ergas", 4), ("RR", "SAM↓", "sam", 4),
    ("RR", "PSNR↑", "psnr", 4),   ("RR", "SSIM↑", "ssim", 4),
    ("RR", "SCC↑", "scc", 4),     ("RR", "Q2n↑", "q2n", 4),
    ("RR", "RMSE↓", "rmse", 4),   ("RR", "CC↑", "cc", 4),
    # full-resolution — 논문 대조가 가능한 12-19(8장) 기준.
    # 전체 0-19 는 쓰지 않는다. 0-11 은 12-19 보다 크게 어려운 장면이라(D_lambda 2.4배)
    # 논문 수치와 맞지 않고, 어떤 논문도 20장 기준으로 보고하지 않는다.
    # 근거: ../CANConv/RUNBOOK.md 8.5 — D_lambda 는 EXP 에서 msexp 에만 의존하는데도
    # 두 구간이 2.4배 차이나므로 코드가 아니라 데이터 특성이다.
    ("FR", "D_lambda↓", "d_lambda", 4), ("FR", "D_s↓", "d_s", 4), ("FR", "HQNR↑", "hqnr", 4),
    # 비용 — 첫 시트에만
    ("Cost", "Params(M)", "params_m", 4), ("Cost", "FLOPs(G)", "flops_g", 1),
    ("Cost", "Infer(ms)", "infer_ms", 2), ("Cost", "Mem(MB)", "mem_mb", 1),
    ("Cost", "Train(h)", "train_h", 2),
    ("", "Date", "date", None),
    ("", "Notes", "note", None),
]


# 논문이 보고한 수치. 항상 표 맨 위에 둔다. RMSE/CC/FR(20장)은 논문에 없다.
PAPER_ROW = {
    "WV3": dict(tag="■ Paper (reported)", ergas=2.040, sam=2.787, psnr=37.956,
                ssim=0.976, scc=0.988, q2n=0.922,
                d_lambda=0.016, d_s=0.027, hqnr=0.958,
                params_m=7.170, flops_g=79.03, infer_ms=9.0, mem_mb=1751.9,
                note="[기준] w128 · depth 총12블록(배분 미기재; 우리는 2,2,4) · AttnBlock 3 · PAN K/V 유지 · LN(Eq 5) · 입력 9ch · crop 명시(구현은 scale jitter) · 50K · seed 2025 · AdamW 1e-4/wd0.01 cosine warmup100 · batch48(실효96) · k=3 · λ=1.0 · best 선택 미기재. 아래 행 Notes 는 이 기준 대비 바뀐 부분만 적는다"),
    "QB": dict(tag="■ Paper (reported)", ergas=4.169, sam=5.078, psnr=29.276,
               q2n=0.846, d_lambda=0.036, d_s=0.022, hqnr=0.942, note="논문 Table. 세팅은 WV3 행과 동일"),
    "GF2": dict(tag="■ Paper (reported)", ergas=0.552, sam=0.596, psnr=45.076,
                ssim=0.988, scc=0.994, q2n=0.988,
                d_lambda=0.017, d_s=0.020, hqnr=0.964, note="논문 Table. 세팅은 WV3 행과 동일"),
}

# work_dir 에 config 없이 결과 mat 만 있는 참조 (외부 모델의 배포 가중치 등)
EXTERNAL = {"_ref_cannet": ("□ CANConv (released weights)", "wv3",
                            "CANConv 배포 가중치 실측. 논문 Table 3 의 CANConv 행과 6지표 0.5% 이내 일치 — 평가기 검증용",
                            {"params_m": 0.7874})}


def sheet_name(ds, server):
    return f"{ds}-{server}"


def columns_for(ds):
    """첫 데이터셋(WV3)이 아니면 비용 열을 뺀다. ds 는 서버 접미사 없는 이름이다."""
    if ds == SHEET_ORDER[0]:
        return COLUMNS
    return [c for c in COLUMNS if c[0] != "Cost"]


# ----------------------------------------------------------------- 지표
def _rr(mat, ds):
    """reduced: PSNR/SSIM/SAM/ERGAS/SCC/Q2n. tools/eval_dlpan.py 와 같은 경로를 쓴다."""
    import importlib.util
    from scipy.io import loadmat
    spec = importlib.util.spec_from_file_location("_ed", os.path.join(ROOT, "tools", "eval_dlpan.py"))
    ed = importlib.util.module_from_spec(spec)
    sys.modules["_ed"] = ed
    spec.loader.exec_module(ed)
    from tools.metrics.eval_rr import evaluate

    import h5py
    scale = ed.SCALE[ds]
    with h5py.File(os.path.join(ROOT, ed.GT_H5[ds])) as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(20, -21)
    gt_c = gt[:, sl, sl, :]
    sr = loadmat(mat)["sr"].astype(np.float64)
    if sr.shape[1] in (4, 8):
        sr = sr.transpose(0, 2, 3, 1)
    sr_c = sr[:, sl, sl, :]
    m = evaluate(sr_c, gt_c, scale, 32)
    out = {"sam": m["SAM"][0], "sam_sd": m["SAM"][1],
           "ergas": m["ERGAS"][0], "ergas_sd": m["ERGAS"][1],
           "q2n": m["Q2n"][0], "q2n_sd": m["Q2n"][1]}
    # RMSE 와 CC 는 evaluate() 에 없어 여기서 영상별로 낸다
    rm, cc = [], []
    for i in range(len(gt_c)):
        a, b = sr_c[i].ravel(), gt_c[i].ravel()
        rm.append(float(np.sqrt(np.mean((a - b) ** 2))))
        cc.append(float(np.corrcoef(a, b)[0, 1]))
    out["rmse"] = float(np.mean(rm)); out["cc"] = float(np.mean(cc))
    out["scc"] = float(np.mean([ed.scc_dlpan(sr_c[i], gt_c[i]) for i in range(len(gt_c))]))
    out["psnr"] = float(np.mean([ed.psnr_global(sr_c[i], gt_c[i], scale) for i in range(len(gt_c))]))
    out["ssim"] = float(np.mean([ed.ssim_skimage(sr_c[i], gt_c[i], scale) for i in range(len(gt_c))]))
    return out


def _fr(mat, ds, indices="12-19"):
    """full-res: D_lambda/D_s/HQNR. 논문 Table 과 대조 가능한 12-19 부분집합 기준."""
    import h5py
    from scipy.io import loadmat
    from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    fr_h5 = os.path.join(ROOT, "data", "PanCollection", ds.upper(),
                         "full_examples_h5", f"test_{ds}_OrigScale_multiExm1.h5")
    with h5py.File(fr_h5) as f:
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    a, b = (int(x) for x in indices.split("-"))
    sr = loadmat(mat)["sr"].astype(np.float64)
    if sr.shape[1] in (4, 8):
        sr = sr.transpose(0, 2, 3, 1)
    # 누적 리스트 이름을 ds 와 겹치지 않게 둔다 (데이터셋 인자를 덮어쓰면 안 된다)
    dl_all, dsv_all = [], []
    for i in range(a, b + 1):
        dl_all.append(d_lambda_k(sr[i], lms[i], ds, 4, 32, wald))
        dsv_all.append(d_s(sr[i], lms[i], pan[i], 4, 32, wald))
    dl, dsv = float(np.mean(dl_all)), float(np.mean(dsv_all))
    return {"d_lambda": dl, "d_s": dsv, "hqnr": (1 - dl) * (1 - dsv)}


# ----------------------------------------------------------------- 비용
def _gpu_busy(threshold=30):
    """다른 프로세스가 GPU 를 쓰고 있는가. nvidia-smi 사용률로 판단한다."""
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().split("\n")[0]) >= threshold
    except Exception:
        return False


def _profile(args_ns, key, want_flops):
    """비용 측정.

    추론시간·메모리는 몇 초면 끝나므로 항상 잰다. FLOPs 는 thop 이 모델 전체를
    훑어야 해 상대적으로 느리고 구조가 같으면 값이 같으므로 --profile 일 때만 재고
    캐시한다.

    주의: GPU 가 학습 중이면 추론시간이 경합으로 부풀려진다. 절대값이 필요하면
    유휴 상태에서 --profile 로 다시 잴 것.
    """
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    hit = cache.get(key, {})
    out = {}
    import torch
    from main import import_class
    Model = import_class(args_ns.model)

    # --profile 이면 강제 재측정. 캐시 미스(신규 구조)면 자동 업로드에서도 한 번
    # 측정해 채운다 — 업로드는 case 사이(GPU 유휴)에 돌아서 학습을 방해하지 않고,
    # 값은 캐시돼 다음부터는 재지 않는다. thop 이 새 구조에서 실패해도 업로드는 계속.
    if want_flops or "flops_g" not in hit:
        try:
            from thop import profile as thop_profile
            m = Model(**args_ns.model_args).eval()
            inp = (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 64, 64),
                   torch.randn(1, 8, 64, 64), torch.ones(1))
            f, _ = thop_profile(m, inputs=inp, verbose=False)
            out["flops_g"] = f / 1e9
            del m
        except Exception:
            if "flops_g" in hit:
                out["flops_g"] = hit["flops_g"]
    else:
        out["flops_g"] = hit["flops_g"]

    # GPU 가 학습으로 바쁘면 추론시간이 경합으로 부풀려진다(실측 16 -> 36 ms).
    # 그럴 때는 재지 않고 캐시된 값을 쓴다. 오염된 수치를 올리는 것보다 낫다.
    busy = _gpu_busy()
    if busy:
        # 메모리는 경합 영향이 작아 캐시를 그대로 쓴다. 추론시간은 캐시가 있을 때만.
        if "mem_mb" in hit:
            out["mem_mb"] = hit["mem_mb"]
        if "infer_ms" in hit:
            out["infer_ms"] = hit["infer_ms"]
    elif torch.cuda.is_available():
        try:
            m = Model(**args_ns.model_args).eval().cuda()
            inp = tuple(x.cuda() for x in (torch.randn(1, 1, 256, 256), torch.randn(1, 1, 64, 64),
                                           torch.randn(1, 8, 64, 64), torch.ones(1)))
            with torch.no_grad():
                for _ in range(5):
                    m(*inp)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(20):
                    m(*inp)
                torch.cuda.synchronize()
            out["infer_ms"] = (time.time() - t0) / 20 * 1000
            out["mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            del m, inp
            torch.cuda.empty_cache()
        except Exception:
            pass                      # 메모리 부족 등으로 실패해도 나머지는 올린다


    cache[key] = {**hit, **out}
    json.dump(cache, open(CACHE, "w"), indent=1)
    return out


# ----------------------------------------------------------------- 한 실행 수집
SERVER_FILE = os.path.join(ROOT, "gspread", "server.txt")


def resolve_server(cli):
    """서버 식별자. --server > PANCRAFTER_SERVER > gspread/server.txt 순.

    지정이 없으면 올리지 않는다. 두 서버가 같은 config 를 돌리면 실행명이 같아져,
    suffix 없이 올리면 상대 서버 값을 조용히 덮어쓴다. 그러면 한 행에 어느 서버
    수치인지 알 수 없는 값이 남는다 (CLAUDE.md 가 금지하는 서버 간 수치 혼용).
    """
    s = cli or os.environ.get("PANCRAFTER_SERVER") or (
        open(SERVER_FILE).read().strip() if os.path.exists(SERVER_FILE) else "")
    if not s:
        raise SystemExit(
            "서버 식별자가 없다. 다음 중 하나로 지정할 것:\n"
            "  --server s1\n"
            "  export PANCRAFTER_SERVER=s1\n"
            f"  echo s1 > {SERVER_FILE}   (그 서버에 한 번만 해두면 된다)\n\n"
            "두 서버가 같은 config 를 돌리면 실행명이 같아진다. suffix 가 없으면\n"
            "상대 서버 값을 덮어써서 어느 쪽 수치인지 알 수 없게 된다.")
    return s.strip()


# 재구성본의 표준 설정. 실행명에는 여기서 벗어난 항목만 붙인다.
REBUILD_DEFAULT = {"hidden_size": 128, "depth": [2, 2, 4], "n_attn": 3,
                   "norm": "ln", "in_mode": "paper", "crop": True}


def _descriptor(ma, crop, family):
    """실행명에 붙일 짧은 설명. 표준에서 벗어난 축만 보여준다.

    s1_A0 같은 ID 만으로는 시트에서 무엇을 시험한 실행인지 알 수 없다.
    family: "paper"(재구성본) / "released"(배포 구조) / "lrfuse"(LR-Fuse)
    """
    if family in ("lrfuse", "lrtinyswin"):   # 저해상도 초경량 구조들
        head = (f"LR-TinySwin w{ma.get('hidden_size', 64)} sw{ma.get('swin_depth', 2)}"
                if family == "lrtinyswin"
                else f"LR-Fuse w{ma.get('hidden_size', 64)} {ma.get('n_blocks', 6)}blk")
        bits = [head, "9ch계열" if ma.get("in_mode", "paper") == "paper" else "11ch계열"]
        return " ".join(bits + ([] if crop else ["nocrop"]))
    is_rebuild = family == "paper"
    if not is_rebuild:                       # 배포 구조 계열
        bits = [f"w{ma['hidden_size'][0]}", f"d{''.join(map(str, ma['depth']))}",
                f"a{ma.get('n_attn', 5)}", "gn"]
        return " ".join(bits + ([] if crop else ["nocrop"]))
    D, bits = REBUILD_DEFAULT, []
    w = ma.get("hidden_size")
    w = w[0] if isinstance(w, (list, tuple)) else w
    if w != D["hidden_size"]:
        bits.append(f"w{w}")
    if list(ma.get("depth", [])) != D["depth"]:
        bits.append("d" + "".join(map(str, ma["depth"])))
    if ma.get("n_attn", 3) != D["n_attn"]:
        bits.append(f"a{ma['n_attn']}")
    # norm 키가 없는 meta 스냅샷은 옵션 도입 전에 찍힌 것이다. 그때 동작은
    # 배포 코드에서 물려받은 GroupNorm 이었으므로 기본값을 gn 으로 본다.
    nrm = ma.get("norm", "gn")
    if nrm != D["norm"]:
        bits.append(nrm)
    if ma.get("in_mode", "paper") != D["in_mode"]:
        bits.append("11ch")
    if ma.get("mlp_ratio", 4.0) != 4.0:
        bits.append(f"mlp{ma['mlp_ratio']:g}")
    if ma.get("cm3a_pan_branch", True) is False:
        bits.append("noPANkv")
    loc = ma.get("attn_locations")
    if loc is not None and tuple(loc) != ("enc", "btl", "dec"):
        bits.append("attn:" + ("+".join(loc) if loc else "0"))
    if ma.get("dec_depth") is not None:
        bits.append("dd" + "".join(map(str, ma["dec_depth"])))
    if ma.get("mode_modulation", True) is False:
        bits.append("plain")          # γβ 조건화 제거 — MS1 과 시트에서 구분되도록
    if ma.get("swin_depth", 0):
        bits.append(f"sw{ma['swin_depth']}@btl")
    if ma.get("swin_mid", 0):
        bits.append(f"sw{ma['swin_mid']}@H2")
    if not crop:
        bits.append("nocrop")
    return " ".join(bits) if bits else "표준"


def _iter_label(n):
    """25000 -> '25K'. 실행명에 붙여 iteration 이 다른 실행을 한눈에 구분한다."""
    return f"{n // 1000}K" if n and n % 1000 == 0 else str(n)


def collect(tag, want_profile, server, peer=None):
    wd = os.path.join(ROOT, "work_dir", tag)
    if tag in EXTERNAL:                       # config 가 없는 외부 참조
        label, ds, note, extra = EXTERNAL[tag]   # 외부 참조는 서버와 무관하다
        row = {"tag": label, "_ds": ds.upper(), "note": note, "date": ""}
        row.update(extra)
        rr = os.path.join(wd, "results", "reduced_best_val.mat")
        if os.path.exists(rr):
            row.update(_rr(rr, ds))
        return row
    cfg_path = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "config", f"{tag}.yaml")
    if not os.path.exists(cfg_path):
        return None
    from main import get_parser, import_class
    c = yaml.safe_load(open(cfg_path))
    p = get_parser(); p.set_defaults(**c); a = p.parse_args([])

    # 데이터셋은 테스트 dataroot 에서 읽는다
    droot = a.test_reduced_feeder_args.get("dataroot", "")
    ds = next((s for s in ("wv3", "qb", "gf2", "wv2") if f"test_{s}_" in droot), "wv3")

    ma = a.model_args
    hs = ma.get("hidden_size")
    row = {
        "tag": tag, "_ds": ds.upper(),
        "model": a.model.rsplit(".", 1)[-1],
        "seed": a.seed, "iter": a.num_iter,   # iter 는 비고와 실행명 양쪽에 들어간다
        "width": hs if isinstance(hs, int) else (hs[0] if hs else ""),
        "depth": str(ma.get("depth", "")),
        "n_attn": (0 if ("LRFuse" in a.model or "LRTinySwin" in a.model) else
                   ma.get("n_attn", len(ma.get("cm3a_locations") or ["2e","3e","4","3d","2d"]))),
        "norm": ma.get("norm", "gn"),
        "mlp_ratio": ma.get("mlp_ratio", 4.0),
        "crop": a.train_feeder_args.get("crop", ""),
        "family": ("lrtinyswin" if "LRTinySwin" in a.model
                   else "lrfuse" if "LRFuse" in a.model
                   else "paper" if "Paper" in a.model else "released"),
        "fix": "True" if ma.get("fix_key_alias") else "False",
    }
    # 파라미터
    Model = import_class(a.model)
    m = Model(**ma)
    row["params_m"] = sum(x.numel() for x in m.parameters()) / 1e6
    row["train_params"] = sum(x.numel() for x in m.parameters() if x.requires_grad)
    del m

    # 학습 시간 / 날짜
    for k, f in (("started_at.txt", "s"), ("finished_at.txt", "f")):
        pth = os.path.join(wd, "meta", k)
        row[f] = open(pth).read().strip() if os.path.exists(pth) else ""
    if row.get("s") and row.get("f"):
        from datetime import datetime as D
        row["train_h"] = (D.fromisoformat(row["f"]) - D.fromisoformat(row["s"])).total_seconds() / 3600
    row["date"] = (row.get("f") or row.get("s") or "")[:10]
    row.pop("s", None); row.pop("f", None)

    # 지표
    _sfx = "_peerB" if peer == "B" else ""
    rr = next((q for q in (os.path.join(wd, "results", f"reduced_{k}{_sfx}.mat")
                           for k in ("best_hqnr", "best_val", "best_reduced"))
               if os.path.exists(q)), "")
    if os.path.exists(rr):
        row.update(_rr(rr, ds))
    for name in (("full_best_hqnr_peerB.mat",) if peer == "B" else
                 ("full_frrepair.mat", "full_best_hqnr.mat", "full_best_val.mat", "full_best_reduced.mat")):
        fr = os.path.join(wd, "results", name)
        if os.path.exists(fr):
            try:
                row.update(_fr(fr, ds))
            except Exception as e:
                row["note_err"] = f"FR 실패: {type(e).__name__}"
            break
    row.update(_profile(a, tag, want_profile))

    family = row["family"]
    is_rebuild = family == "paper"
    desc = _descriptor(ma, row["crop"], family)
    # KD·mutual trainer 표기 — run 명과 Notes 양쪽에 남긴다
    _tr = getattr(a, "trainer", "default")
    if _tr == "kd":
        _v = (getattr(a, "kd_args", {}) or {}).get("variant", "?")
        _tc = os.path.basename(getattr(a, "teacher_config", "") or "").replace(".yaml", "")
        desc = (desc + f" KD:{_v}(T={_tc})").strip()
    elif _tr == "mutual":
        _v = (getattr(a, "mutual_args", {}) or {}).get("variant", "?")
        desc = (desc + f" MUT:{_v}").strip()
    elif _tr == "teacher":
        desc = (desc + " +unc.head").strip()
    if getattr(a, "mars", "dual") == "ms":
        desc = (desc + " singleMARs").strip().removeprefix("표준 ")

    # Notes 는 기준(논문 충실 PAN-Crafter) 대비 "바뀐 부분만" 적는다. 같으면 안 쓴다.
    # 기준: w128 · depth(2,2,4) · AttnBlock 3(enc+btl+dec) · PAN K/V 유지 · LN ·
    #       mlp 4.0 · 입력 9ch · crop · MARs dual · seed 2025 · best선택 HQNR(공식 12-19)
    n_iter = row["iter"]
    _loc = ma.get("attn_locations")
    bits = []
    if family in ("lrfuse", "lrtinyswin"):
        if family == "lrtinyswin":
            bits.append(f"arch=LR-TinySwin (PixelUnshuffle×4 · 전연산 1/16 면적 "
                        f"· w{ma.get('hidden_size', 64)} · Swin {ma.get('swin_depth', 2)} "
                        f"(h{ma.get('num_heads', 4)}·w{ma.get('window_size', 8)}"
                        f"·mlp{ma.get('mlp_ratio', 2.0):g}) · residual group)")
        else:
            bits.append(f"arch=LR-Fuse (PixelUnshuffle×4 · 전연산 1/16 면적 "
                        f"· w{ma.get('hidden_size', 64)} · ResBlock {ma.get('n_blocks', 6)} "
                        f"· attention 없음)")
        bits.append("in=" + ("unshuffle(PAN)+MS (9ch 철학)"
                             if ma.get("in_mode", "paper") == "paper"
                             else "unshuffle(PAN)+LPAN+고주파+MS (11ch 철학)"))
        if not row["crop"]:
            bits.append("crop=False")
        if getattr(a, "mars", "dual") == "ms":
            bits.append("mars=ms (PAN task·복제 제거)")
        if ma.get("mode_modulation", True) is False:
            bits.append("mode_modulation=False (γβ 조건화 제거 — 순수 residual U-Net)")
    elif not is_rebuild:
        bits.append("arch=배포코드 (4-scale · CM3A5 · GroupNorm · mode-token · 11ch)")
        bits.append(f"fix_A1A2={row.get('fix', '')}")
        if not row["crop"]:
            bits.append("crop=False")
    else:
        if row["width"] != 128:
            bits.append(f"width={row['width']}")
        if list(ma.get("depth", [])) != [2, 2, 4]:
            bits.append(f"depth={row['depth']}")
        if ma.get("cm3a_pan_branch", True) is False:
            bits.append("cm3a_pan_branch=False (PAN K/V 제거)")
        if _loc is not None and tuple(_loc) != ("enc", "btl", "dec"):
            bits.append("attn_locations=없음" if not _loc else f"attn_locations={'+'.join(_loc)}")
        if ma.get("dec_depth") is not None:
            bits.append(f"dec_depth={list(ma['dec_depth'])} (decoder 비대칭; 0=해당 해상도 생략)")
        if ma.get("swin_depth", 0):
            bits.append(f"swin={ma['swin_depth']}@btl (표준 Swin, W→SW 교대 · "
                        f"h{ma.get('swin_heads', 4)}·w{ma.get('swin_window', 8)}"
                        f"·mlp{ma.get('swin_mlp_ratio', 2.0):g})")
        if ma.get("swin_mid", 0):
            bits.append(f"swin_mid={ma['swin_mid']}@H/2enc (표준 Swin)")
        if ma.get("norm", "gn") != "ln":
            bits.append("norm=gn (논문은 LN)")
        if ma.get("mlp_ratio", 4.0) != 4.0:
            bits.append(f"mlp_ratio={ma['mlp_ratio']:g}")
        if ma.get("in_mode", "paper") == "released":
            bits.append("in=11ch (↑LPAN·PAN−↑LPAN 추가)")
        if not row["crop"]:
            bits.append("crop=False")
        if getattr(a, "mars", "dual") == "ms":
            bits.append("mars=ms (PAN task·복제 제거)")
        if ma.get("mode_modulation", True) is False:
            bits.append("mode_modulation=False (γβ 조건화 제거 — 순수 residual U-Net)")
    if _tr == "kd":
        _ka = getattr(a, "kd_args", {}) or {}
        _tck = getattr(a, "teacher_checkpoint", "") or ""
        bits.append(f"trainer=kd({_ka.get('variant', '?')}) · teacher={_tck}")
    elif _tr == "mutual":
        _ma2 = getattr(a, "mutual_args", {}) or {}
        bits.append(f"trainer=mutual({_ma2.get('variant', '?')}) · 2-peer · "
                    "선택=pair 평균 HQNR, 이 행 지표=개별 peer mat (A/B 두 행 병기)")
    elif _tr == "teacher":
        _ta = getattr(a, "teacher_args", {}) or {}
        bits.append("trainer=teacher (uncertainty head"
                    + (f" + SiS r{_ta.get('sis_radius')}" if _ta.get("lambda_sis") else "") + ")")
    if row["seed"] != 2025:
        bits.append(f"seed={row['seed']}")
    if a.select_on != "hqnr":
        bits.append(f"select={'val-ERGAS' if a.select_on == 'val' else 'test-ERGAS/D_s'}")
    for k in ("width", "depth", "n_attn", "norm", "mlp_ratio", "crop", "iter", "seed",
              "model", "train_params", "fix", "family"):
        row.pop(k, None)
    if not bits:
        bits = ["기준 구조 그대로"]
    # 서버는 시트 이름("<데이터셋>-<서버>")이 이미 담고 있으므로 Notes 에 넣지 않는다.
    if row.get("note_err"):
        bits.append(row.pop("note_err"))
    row["note"] = " · ".join(bits)

    if peer == "B":
        row["tag"] = row["tag"] + "·peerB"
        desc = (desc + " peerB").strip()
    lbl = _iter_label(n_iter)
    base = row["tag"] if lbl.lower() in row["tag"].lower() else f"{row['tag']} ({lbl})"
    row["tag"] = f"{base} · {desc}" if desc else base
    return row


# ----------------------------------------------------------------- 시트
def fmt(row, cols):
    out = []
    for _, _, key, nd in cols:
        v = row.get(key, "")
        if v == "" or v is None:
            out.append("")
        elif nd is not None and isinstance(v, (int, float)):
            out.append(round(float(v), nd))
        else:
            out.append(v if isinstance(v, (int, float)) else str(v))
    return out


def _a1(row, col):
    """(1-based row, col) -> A1 표기."""
    s = ""
    while col:
        col, r = divmod(col - 1, 26)
        s = chr(65 + r) + s
    return f"{s}{row}"


def _col(col):
    return _a1(1, col)[:-1]


def _ensure_sheet(sh, name):
    try:
        return sh.worksheet(name)
    except Exception:
        return sh.add_worksheet(title=name, rows=200, cols=40)


def _group_edges(cols):
    """그룹이 바뀌는 지점의 0-based 열 인덱스. 여기에 세로 실선을 놓는다."""
    return [i for i in range(1, len(cols)) if cols[i][0] != cols[i - 1][0]]


def _apply_borders(ws, cols, last_row):
    """RR / FR / 비용 사이에 세로 실선을 긋는다. 데이터 행까지 이어지게 한다."""
    from gspread_formatting import (CellFormat, Border, Borders, Color,
                                    format_cell_range, batch_updater)
    line = Border("SOLID_MEDIUM", Color(0.25, 0.25, 0.25))
    n = len(cols)
    with batch_updater(ws.spreadsheet) as batch:
        for i in _group_edges(cols):                    # 그룹 시작 열의 왼쪽에 선
            c = _col(ORIGIN_COL + i)
            batch.format_cell_range(ws, f"{c}{ORIGIN_ROW}:{c}{last_row}",
                                    CellFormat(borders=Borders(left=line)))
        # 표 바깥 테두리
        c0, c1 = _col(ORIGIN_COL), _col(ORIGIN_COL + n - 1)
        batch.format_cell_range(ws, f"{c0}{ORIGIN_ROW}:{c0}{last_row}",
                                CellFormat(borders=Borders(left=line)))
        batch.format_cell_range(ws, f"{c1}{ORIGIN_ROW}:{c1}{last_row}",
                                CellFormat(borders=Borders(right=line)))
        # 헤더와 데이터 사이 가로선
        batch.format_cell_range(ws, f"{c0}{ORIGIN_ROW + 1}:{c1}{ORIGIN_ROW + 1}",
                                CellFormat(borders=Borders(bottom=line)))


def _write_header(ws, cols, color):
    """2행에 그룹(RR/FR/비용), 3행에 컬럼명. 그룹은 병합한다."""
    from gspread_formatting import (CellFormat, TextFormat, Color, format_cell_range,
                                    set_frozen, set_column_width)
    n = len(cols)
    c0 = ORIGIN_COL
    grp_a1 = f"{_a1(ORIGIN_ROW, c0)}:{_a1(ORIGIN_ROW, c0 + n - 1)}"
    hdr_a1 = f"{_a1(ORIGIN_ROW + 1, c0)}:{_a1(ORIGIN_ROW + 1, c0 + n - 1)}"

    ws.update([[c[0] for c in cols]], grp_a1)
    ws.update([[c[1] for c in cols]], hdr_a1)

    # 같은 그룹이 이어지는 구간을 병합한다.
    # merge_cells/unmerge_cells 는 A1 문자열을 받는다. 정수를 넘기면 조용히 실패해
    # 이전 레이아웃의 병합이 그대로 남는다 (실제로 그렇게 어긋나 있었다).
    ws.unmerge_cells(grp_a1)
    i = 0
    while i < n:
        g = cols[i][0]
        j = i
        while j + 1 < n and cols[j + 1][0] == g:
            j += 1
        if g and j > i:
            ws.merge_cells(f"{_a1(ORIGIN_ROW, c0 + i)}:{_a1(ORIGIN_ROW, c0 + j)}")
        i = j + 1

    base = Color(*color)
    dark = Color(*[max(0.0, v - 0.12) for v in color])
    format_cell_range(ws, grp_a1, CellFormat(
        backgroundColor=dark, textFormat=TextFormat(bold=True),
        horizontalAlignment="CENTER"))
    format_cell_range(ws, hdr_a1, CellFormat(
        backgroundColor=base, textFormat=TextFormat(bold=True),
        horizontalAlignment="CENTER"))
    set_frozen(ws, rows=ORIGIN_ROW + 1)
    set_column_width(ws, _col(c0), 170)
    set_column_width(ws, _col(c0 + n - 1), 620)


def _bold_best(ws, cols):
    """각 metric 컬럼의 최고(↑)/최저(↓)값 셀을 bold 로. 매 업로드마다 전체를 다시 계산한다.

    ■ Paper (reported) 행은 제외한다. 이전 bold 는 컬럼 전체를 평문으로 되돌린 뒤
    다시 칠하므로 행이 갱신되어 최고값이 바뀌어도 남은 bold 가 없다.
    """
    from gspread_formatting import CellFormat, TextFormat, format_cell_range, batch_updater
    data = ws.get_all_values()
    r0 = ORIGIN_ROW + 2                     # 1-based 첫 데이터 행
    rows_ = data[r0 - 1:]
    if not rows_:
        return
    tag_i = ORIGIN_COL - 1
    with batch_updater(ws.spreadsheet) as batch:
        for ci, (grp, hdr, _, _) in enumerate(cols):
            if "↓" not in hdr and "↑" not in hdr:
                continue
            col = ORIGIN_COL + ci
            vals = []
            for ri, r in enumerate(rows_):
                tag = r[tag_i] if len(r) > tag_i else ""
                if not tag or tag.startswith("■"):
                    continue
                try:
                    vals.append((float(r[col - 1]), r0 + ri))
                except (ValueError, IndexError):
                    continue
            if not vals:
                continue
            best_row = (min if "↓" in hdr else max)(vals)[1]
            rng = f"{_col(col)}{r0}:{_col(col)}{r0 + len(rows_) - 1}"
            batch.format_cell_range(ws, rng, CellFormat(textFormat=TextFormat(bold=False)))
            batch.format_cell_range(ws, f"{_col(col)}{best_row}",
                                    CellFormat(textFormat=TextFormat(bold=True)))


def upload(rows, server, replace=False):
    import gspread
    gc = gspread.service_account(filename=CRED)
    sh = gc.open(SHEET)

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r.pop("_ds"), []).append(r)

    total = added = 0
    for ds, rs in by_ds.items():
        cols = columns_for(ds)
        n = len(cols)
        ws = _ensure_sheet(sh, sheet_name(ds, server))
        cur = ws.get(f"{_a1(ORIGIN_ROW + 1, ORIGIN_COL)}:{_a1(ORIGIN_ROW + 1, ORIGIN_COL + n - 1)}")
        if not cur or cur[0][:n] != [c[1] for c in cols]:
            _write_header(ws, cols, SHEET_COLOR.get(ds, (0.85, 0.89, 0.95)))

        # 논문 수치를 맨 위에 놓는다
        if ds in PAPER_ROW:
            rs = [dict(PAPER_ROW[ds])] + [r for r in rs if not r["tag"].startswith("■")]

        tcol = _col(ORIGIN_COL)
        if replace:
            last = ws.row_count
            ws.batch_clear([f"{_a1(ORIGIN_ROW + 2, ORIGIN_COL)}:{_a1(last, ORIGIN_COL + n - 1)}"])
            tags = []
        else:
            vals = ws.get(f"{tcol}{ORIGIN_ROW + 2}:{tcol}")
            tags = [v[0] if v else "" for v in vals]
            while tags and not tags[-1]:  # 빈 범위에서 gspread 가 빈 행을 돌려주는 경우가 있다
                tags.pop()

        last = ORIGIN_ROW + 1
        for r in rs:
            v = fmt(r, cols)
            if r["tag"] in tags:
                i = ORIGIN_ROW + 2 + tags.index(r["tag"])
            else:
                i = ORIGIN_ROW + 2 + len(tags)
                tags.append(r["tag"]); added += 1
            ws.update([v], f"{_a1(i, ORIGIN_COL)}:{_a1(i, ORIGIN_COL + n - 1)}")
            last = max(last, i)
            total += 1
        _apply_borders(ws, cols, last)      # 새로 쓴 행까지 선을 이어준다
        _bold_best(ws, cols)
        print(f"  [{sheet_name(ds, server)}] {len(rs)}행 (best bold 갱신)")
    return total, added


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="*", default=[], help="work_dir 이름 또는 glob")
    ap.add_argument("--all", action="store_true", help="results/*.mat 이 있는 실행 전부")
    ap.add_argument("--profile", action="store_true", help="FLOPs·추론시간·메모리도 측정 (느리다)")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 표만 출력")
    ap.add_argument("--server", default=None,
                    help="서버 식별자 (예: s1, s2). 미지정 시 PANCRAFTER_SERVER 또는 "
                         "gspread/server.txt 를 본다. 셋 다 없으면 올리지 않는다")
    ap.add_argument("--replace", action="store_true",
                    help="기존 데이터 행을 비우고 주어진 순서대로 다시 쓴다")
    a = ap.parse_args()

    tags = []
    if a.all:
        tags = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p in glob.glob(f"{ROOT}/work_dir/*/results/reduced_*.mat")]
    for pat in a.pattern:
        tags += [os.path.basename(d) for d in glob.glob(f"{ROOT}/work_dir/{pat}") if os.path.isdir(d)]
    seen = set(); ordered = []
    for x in tags:                 # 인자로 준 순서를 유지한다 (표 정렬에 그대로 반영된다)
        if x not in seen:
            seen.add(x); ordered.append(x)
    tags = ordered
    server = resolve_server(a.server)
    print(f"  서버 식별자: [{server}]")
    if not tags:
        print("대상이 없다. 실행명이나 glob 을 줄 것."); return 1

    rows = []
    for t in tags:
        r = collect(t, a.profile, server)
        if r is None:
            print(f"  건너뜀 {t} (config 없음)"); continue
        rows.append(r)
        # mutual 실행: peer_b 의 mat 이 있으면 별도 행으로 병기한다 — 선택은
        # pair 평균 HQNR 이지만, 시트의 지표는 mat(=개별 peer)에서 나오므로
        # 두 peer 를 모두 올려야 모집단이 일관된다.
        pb = os.path.join(ROOT, "work_dir", t, "results", "reduced_best_hqnr_peerB.mat")
        if os.path.exists(pb):
            rb = collect(t, False, server, peer="B")
            if rb is not None:
                rows.append(rb)
        pm = r.get("params_m")
        print(f"  수집 {t}: ERGAS {r.get('ergas', float('nan')):.4f}"
              + (f"  params {pm:.4f} M" if pm else ""))

    if a.dry_run:
        for ds in sorted({r["_ds"] for r in rows}):
            cols = columns_for(ds)
            rs = [r for r in rows if r["_ds"] == ds]
            if ds in PAPER_ROW:
                rs = [dict(PAPER_ROW[ds])] + rs
            print(f"\n[{ds}]  " + " | ".join(f"{g}:{h}" if g else h for g, h, _, _ in cols))
            for r in rs:
                print("       " + " | ".join(str(v) for v in fmt(r, cols)))
        return 0
    n, added = upload(rows, server, replace=a.replace)
    print(f"\n업로드 완료: {n}행 처리 ({added}행 신규, {n-added}행 갱신)")
    print(f"  https://docs.google.com/spreadsheets/d/{gspread_id()}")
    return 0


def gspread_id():
    import gspread
    return gspread.service_account(filename=CRED).open(SHEET).id


if __name__ == "__main__":
    sys.exit(main())
