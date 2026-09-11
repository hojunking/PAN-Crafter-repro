#!/usr/bin/env python
"""실험 결과를 Google Sheets 로 올린다.

    python gspread/gspread_upload.py paper_ln                # 한 실행
    python gspread/gspread_upload.py "paper*" --profile      # 여러 개 + 비용 측정
    python gspread/gspread_upload.py --all --dry-run         # 올리지 않고 표만 확인

지표는 tools/metrics/ 의 DLPan 프로토콜 구현을 그대로 쓴다. 학습 중 metrics.csv 값이
아니라 .mat 을 다시 평가한 값이라, 논문 Table 과 비교 가능한 수치다.

측정 규약 (results_log/2026-09-07_alignment-shift-robust-and-metric-v2.md — 논문 표와 맞춘 근거):
  RR  : 20장, dim_cut=21, thvalues=0. SAM/ERGAS/Q2n 은 MATLAB 원본 포팅, SCC 는 SCC.m(zero-padding Sobel),
        PSNR 은 전 밴드 통합 MSE, SSIM 은 11×11 Gaussian σ1.5 (Wang/MATLAB). RMSE/CC 는 논문에 없는 자체 정의.
  FR·paper mat20 : PanCollection **.mat 형식** FR 20장 (= 논문들이 MATLAB DLPan 으로 평가한 세트).
        work_dir/<run>/results/fr_mat20.json (tools/eval_fr_paperset.py). **시트의 FR 은 이 열뿐이다.**
        배포 H5 12-19 (best checkpoint 선택 기준) 는 시트에 올리지 않는다 (2026-09-07 결정).
  HQNR 은 장면별 (1-D_λ)(1-D_s) 의 평균(MATLAB 관례). 새 서버 준비는 tools/metric_v2_prepare.sh.
  JQM↑ (2026-09-08 추가): Palubinskas 2015 Joint Quality Measure, tools/metrics/jqm.py. 논문 미보고 추가 지표, 판정 기준 아님.
2026-09-07 이전 시트의 SCC 는 reflect 패딩(약 +0.004), SSIM 은 skimage 기본창(약 +0.002)이었다 —
그 시트는 `<데이터셋>-<서버>_v1` 탭으로 남겨 두었고, 접미사 없는 탭이 새 정의다.
`--all --replace` 는 work_dir 의 run 을 전부 파일 순서로 올린다. 큐레이션된 순서·구분행으로 되돌리려면
`python gspread/apply_layout.py --sheet WV3-s1 --ref WV3-s1_v1` (옛 탭에 없던 run 은 `<탭>-extra` 로).

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
    # 캠페인 키 — sheet_categories.classify() 가 실행명에서 정한다(구분행과 같은 단일 소스).
    # 시트에서 필터·정렬로 "지금 무슨 실험인가" 를 열 하나로 가른다. 구분행은 그대로 둔다(둘은 보완).
    ("", "캠페인", "campaign", None),
    # reduced-resolution (테스트 20장)
    ("RR", "ERGAS↓", "ergas", 4), ("RR", "SAM↓", "sam", 4),
    ("RR", "PSNR↑", "psnr", 4),   ("RR", "SSIM↑", "ssim", 4),
    ("RR", "SCC↑", "scc", 4),     ("RR", "Q2n↑", "q2n", 4),   # 표시명은 columns_for() 가 데이터셋에 맞춰 Q4↑/Q8↑ 로 바꾼다
    ("RR", "RMSE↓", "rmse", 4),   ("RR", "CC↑", "cc", 4),
    # full-resolution (1) — 논문 세트. PanCollection .mat 형식 FR 20장. CANConv 배포 가중치가
    # 논문 CANConv 행과 D_λ/D_s/HQNR 평균·표준편차까지 일치한다(0.9513±0.0122 vs 0.951±0.013).
    # 값은 tools/eval_fr_paperset.py 가 쓴 results/fr_mat20.json 에서 읽는다. 없으면 빈 칸.
    ("FR·paper mat20", "D_lambda↓", "p_d_lambda", 4), ("FR·paper mat20", "D_s↓", "p_d_s", 4),
    ("FR·paper mat20", "HQNR↑", "p_hqnr", 4),
    # masking 있는 HQNR (2026-09-10 사용자 요청): 가장자리 64 px(블록 2개) 를 뺀 고정 영역 [64:H-64,64:W-64] 의 (1-D_λ)(1-D_s) 장면 평균.
    # 필터(MTF·imresize·interp23tap·Sobel)는 전체 프레임에서 계산한 뒤 자른다. 논문 프로토콜은 전체 프레임(HQNR↑) — 비교표에는 HQNR↑ 를 쓴다.
    ("FR·paper mat20", "HQNR(V64)↑", "p_hqnr_valid", 4),
    # JQM (Palubinskas 2015, tools/metrics/jqm.py) — 두 논문이 보고하지 않는 추가 지표. 판정 기준이 아니다(HQNR→SCC 유지).
    ("FR·paper mat20", "JQM↑", "p_jqm", 4),
    # 배포 H5 의 12-19(8장) FR 열은 2026-09-07 시트에서 뺐다 (사용자 결정 — 논문 세트만 보고한다).
    # H5 12-19 는 학습 중 best checkpoint 선택(train.py `fr_select_indices`)에만 쓰이고 시트에는 오르지 않는다.
    # 0-11 은 12-19 보다 크게 어려운 장면(D_lambda 2.4배)이고 논문 세트와는 6장만 겹친다 (KNOWN_ISSUES F-2).
    # 필요하면 work_dir/<run>/best_state.json 의 best_hqnr 이나 tools/eval_dlpan_fr.py --indices 12-19 로 본다.
    # 비용 — 첫 시트에만
    ("Cost", "Params(M)", "params_m", 4), ("Cost", "FLOPs(G)", "flops_g", 1),
    ("Cost", "Infer(ms)", "infer_ms", 2), ("Cost", "Mem(MB)", "mem_mb", 1),
    ("Cost", "Train(h)", "train_h", 2),
    ("", "Date", "date", None),
    ("", "Notes", "note", None),
]


# 논문이 보고한 수치. 항상 표 맨 위에 둔다. RMSE/CC/FR(20장)은 논문에 없다.
PAPER_ROW = {
    # 논문 FR 수치는 .mat 형식 20장 세트의 값이므로 FR·paper 열(p_*)에 둔다. FR·H5 12-19 열은 비운다.
    "WV3": dict(tag="■ Paper (reported)", ergas=2.040, sam=2.787, psnr=37.956,
                ssim=0.976, scc=0.988, q2n=0.922,
                p_d_lambda=0.016, p_d_s=0.027, p_hqnr=0.958,
                params_m=7.170, flops_g=79.03, infer_ms=9.0, mem_mb=1751.9,
                note="[기준] w128 · depth 총12블록(배분 미기재; 우리는 2,2,4) · AttnBlock 3 · PAN K/V 유지 · LN(Eq 5) · 입력 9ch · crop 명시(구현은 scale jitter) · 50K · seed 2025 · AdamW 1e-4/wd0.01 cosine warmup100 · batch48(실효96) · k=3 · λ=1.0 · best 선택 미기재. FR 은 .mat 20장 세트(FR·paper 열) 기준. 아래 행 Notes 는 이 기준 대비 바뀐 부분만 적는다"),
    # 2026-09-07 검증 지적으로 교정: 종전 QB 행은 WV2(unseen) 값이었고 GF2 는 D_λ/D_s 가 뒤바뀌어 있었다.
    # 출처: 논문 Table 2 (GF2/QB) · Table 3 (WV2) · 보충자료 표(±std, Q4/Q8, SSIM).
    "QB": dict(tag="■ Paper (reported)", ergas=3.570, sam=4.426, psnr=38.195,
               ssim=0.963, scc=0.984, q2n=0.938,
               p_d_lambda=0.043, p_d_s=0.039, p_hqnr=0.920, note="논문 Table 2 + 보충자료. 세팅은 WV3 행과 동일"),
    # GF2 ERGAS 는 논문 내부가 불일치한다 — 본문 Table 2 는 0.522, 보충 Table 7 은 0.552±0.093.
    # 다른 9개 방법의 GF2 ERGAS 는 두 표가 완전히 일치하고 PAN-Crafter 행만 다르다(둘 중 하나가 오기).
    # 우리는 ±std 가 있는 보충자료 값을 쓴다. ERGAS 는 참고 지표라 판정에 영향 없음.
    "GF2": dict(tag="■ Paper (reported)", ergas=0.552, sam=0.596, psnr=45.076,
                ssim=0.988, scc=0.994, q2n=0.988,
                p_d_lambda=0.020, p_d_s=0.017, p_hqnr=0.964,
                note="논문 보충자료 Table 7(±std) 기준. 세팅은 WV3 행과 동일. "
                     "주의: ERGAS 는 본문 Table 2 가 0.522, 보충 Table 7 이 0.552 로 논문 내부 불일치 — "
                     "보충값 채택(다른 9개 방법은 두 표 일치). "
                     "검증: CANConv 배포 가중치 실측 HQNR 0.9189 / D_s 0.0629 / D_λ 0.0194 가 "
                     "논문 CANConv 행 0.919±0.011 / 0.063±0.009 / 0.019±0.010 과 소수 셋째 자리까지 일치"),
    "WV2": dict(tag="■ Paper (reported, WV3 학습 → WV2 zero-shot)", ergas=4.169, sam=5.078, psnr=29.276,
                ssim=0.839, scc=0.924, q2n=0.846,
                p_d_lambda=0.022, p_d_s=0.036, p_hqnr=0.942, note="논문 Table 3 (unseen WV2) + 보충자료. WV3 로 학습한 모델을 WV2 에 그대로 적용"),
}

# work_dir 에 config 없이 결과 mat 만 있는 참조 (외부 모델의 배포 가중치 등)
EXTERNAL = {"_ref_cannet": ("□ CANConv (released weights)", "wv3",
                            "CANConv 배포 가중치 실측 — 평가기 검증용. RR 6지표는 논문 CANConv 행과 0.5% 이내, "
                            "FR·paper(.mat 20장)는 HQNR 0.9511±0.0126 vs 논문 0.951±0.013 으로 표준편차까지 일치",
                            {"params_m": 0.7874}),
            # 센서별 anchor (tools/make_cannet_reference.py): 두 논문의 CANConv 행과 대조해 평가기·데이터가 같은지 확인
            "_ref_cannet_qb": ("□ CANConv (released cannet_qb.pth)", "qb",
                               "CANConv 배포 QB 가중치 실측 — 평가기·데이터 anchor (논문 CANConv 행과 대조)", {"params_m": 0.7874}),
            "_ref_cannet_gf2": ("□ CANConv (released cannet_gf2.pth)", "gf2",
                                "CANConv 배포 GF2 가중치 실측 — 평가기·데이터 anchor (논문 CANConv 행과 대조)", {"params_m": 0.7874}),
            "_ref_cannet_wv2": ("□ CANConv (released cannet_wv3.pth → WV2 zero-shot)", "wv2",
                                "CANConv WV3 가중치의 WV2 zero-shot 실측 — 논문 Table 3 CANConv 행과 대조", {"params_m": 0.7874})}


def _fr_paper(wd, peer=None):
    """논문 세트(.mat FR 20장) 결과 — tools/eval_fr_paperset.py 가 쓴 results/fr_mat20[_peerB].json.

    JSON 의 provenance(평가기 버전·입력 h5 해시)가 지금 코드·데이터와 다르면 **쓰지 않는다**(빈 칸 + 경고).
    검증 지적(2026-09-07): 이전에는 검증 없이 읽어 옛 정의의 값이 새 열에 섞일 수 있었다.
    """
    p = os.path.join(wd, "results", "fr_mat20_peerB.json" if peer == "B" else "fr_mat20.json")
    if not os.path.exists(p):
        return {}
    from tools.eval_fr_paperset import EVAL_VERSION, sha256_of
    j = json.load(open(p))
    src_h5 = j.get("input_h5", "")
    want_sha = sha256_of(src_h5) if src_h5 and os.path.exists(src_h5) else None    # 센서별 논문 세트 h5
    if j.get("eval_version") != EVAL_VERSION or want_sha is None or j.get("input_sha256") != want_sha:
        print(f"  [fr_paper] {os.path.basename(wd)}: JSON 이 옛 평가기/데이터({j.get('eval_version')}) — "
              f"FR·paper 열 비움. tools/eval_fr_paperset.py 로 다시 잴 것")
        return {}
    out = {"p_hqnr": j["hqnr"], "p_d_lambda": j["d_lambda"], "p_d_s": j["d_s"]}
    if "jqm" in j:
        out["p_jqm"] = j["jqm"]
    if "hqnr_valid" in j:
        out["p_hqnr_valid"] = j["hqnr_valid"]
    return out


def sheet_name(ds, server):
    return f"{ds}-{server}"


def columns_for(ds):
    """첫 데이터셋(WV3)만 비용 열 전부. 다른 데이터셋은 Params(M)·Train(h) 만 남긴다 — FLOPs·추론시간·메모리는
    데이터셋과 무관하지만 학습 시간은 데이터셋마다 다르다 (2026-09-08 요청). ds 는 서버 접미사 없는 이름이다."""
    cols = COLUMNS if ds == SHEET_ORDER[0] else [c for c in COLUMNS if c[0] != "Cost" or c[2] in ("params_m", "train_h")]
    # Q2n 열 이름은 데이터셋의 밴드 수를 따른다 — 8밴드(WV3·WV2) Q8, 4밴드(QB·GF2) Q4 (논문 표기)
    q = "Q8↑" if ds in ("WV3", "WV2") else "Q4↑"
    return [(g, q if k == "q2n" else h, k, nd) for g, h, k, nd in cols]


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
    dl_all, dsv_all = np.array(dl_all), np.array(dsv_all)
    # HQNR 은 장면별 (1-D_λ)(1-D_s) 의 평균 — MATLAB 관례(indexes_evaluation_FS 를 장면마다 부른 뒤 평균)이자
    # train.py 의 선택 지표와 같은 식. 2026-09-07 이전의 (1-mean D_λ)(1-mean D_s) 와는 ~1e-5 차이.
    return {"d_lambda": float(dl_all.mean()), "d_s": float(dsv_all.mean()),
            "hqnr": float(((1 - dl_all) * (1 - dsv_all)).mean())}


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


def _cost_model(args_ns):
    """비용 측정용 모듈. trainer pa 면 aligner+warp 를 포함한 PAModel 을 (pan, lpan, ms, s) 서명으로 감싼다 (검토 지적 6)."""
    import torch
    from main import import_class
    Model = import_class(args_ns.model)
    m = Model(**args_ns.model_args)
    tr = getattr(args_ns, "trainer", "default")
    if tr in ("pa", "po", "kdv"):
        from pa.aligner import PANGlobalAligner
        from pa.model import PAModel
        from pa.offset import aligner_margin
        mg = aligner_margin(float((getattr(args_ns, "po", {}) or {}).get("radius_hr", 1.0))) if tr == "po" else 0

        class _PA(torch.nn.Module):
            def __init__(self, pm):
                super().__init__(); self.pm = pm

            def forward(self, pan, lpan, ms, s):
                return self.pm(pan, ms, lpan)["y"]
        if tr == "kdv":
            from kdv.teacher_assets import skeleton_from_cfg
            cfg = dict(model=args_ns.model, model_args=args_ns.model_args, num_bands=getattr(args_ns, "num_bands", 8), trainer="kdv", kdv=(getattr(args_ns, "kdv", {}) or {}))
            pm, _ = skeleton_from_cfg(cfg, Model)
            return _PA(pm)                       # A-ID 면 aligner 없음(= backbone 비용), 그 외 aligner+warp 포함
        return _PA(PAModel(m, PANGlobalAligner(int(getattr(args_ns, "num_bands", 8))), aligner_margin=mg))
    return m


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
            m = _cost_model(args_ns).eval()
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
            m = _cost_model(args_ns).eval().cuda()
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
sys.path.insert(0, os.path.join(ROOT, "gspread"))
from sheet_categories import classify, DESC, separator_cell, SEP  # noqa: E402

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


def _ga_case(al):
    """alignment config 5-key -> (case 표기, 계획이 그 case 로 답하려는 질문)."""
    src = al.get("delta_source", "zero"); a = float(al.get("alpha", 0))
    inv = al.get("inverse_location", "none")
    if src == "zero":
        return "P0(phase fix)", "bicubic(phase 1.5) 을 interp23tap(=데이터셋 lms) 로 바꾸면 성능이 달라지는가"
    if src == "cache" and inv == "final_output":
        return "C1(frozen round-trip)", "PAN frame 에서 정렬된 MS 를 내부 처리한 뒤 최종 출력을 M-frame 으로 되돌려도 이득이 남는가"
    if src == "cache" and inv == "loss_branch":
        return "C3(frozen dual-frame)", "최종 출력은 P-frame 에 두고 GT loss 만 inverse 뷰로 계산하면 좌표 충돌이 줄어드는가"
    if src == "cache":
        return f"C2(input-only α={a:g})", "조건 입력 MS 에만 부분 shift 를 주면 HQNR–fSCC 절충점이 있는가"
    if inv == "final_output":
        return "C4-A(trainable round-trip)", "작은 trainable ShiftNet 이 외부 추정기의 효과를 재현하는가"
    return "C4-B(trainable dual-frame)", "trainable ShiftNet + dual-frame 이 최종 method 로 성립하는가"


def _ga_notes(al):
    """GA 실행의 맞춤 세팅 설명 — 시트만 보고도 무엇을 어떻게 돌렸는지 알 수 있어야 한다."""
    case, q = _ga_case(al)
    src = al.get("delta_source", "zero"); a = float(al.get("alpha", 0))
    fr = al.get("output_frame", "M"); inv = al.get("inverse_location", "none")
    src_d = {"zero": "없음",
             "cache": "frozen cache(Scharr-ZNCC audit, GT 미사용; FR 12-19 Δ≈(−0.16,+0.18) LR px, "
                      "train 16² patch 는 추정 노이즈 jitter sd 0.076)",
             "trainable": "trainable GlobalShiftNet(입력=Scharr 구조맵; pseudo-label pretrain, "
                          "gate 결과 outputs/global_shift_cache/shiftnet_pretrained.json)"}.get(src, "?")
    inv_d = {"none": "inverse 없음", "final_output": "최종 출력을 M-frame 으로 inverse(round-trip, border mask 로 loss 정규화)",
             "loss_branch": "P-frame 출력 유지 · GT loss 만 inverse 뷰(M-frame, border mask)"}.get(inv, "?")
    full = inv != "none"
    return [
        f"GA {case} — 질문: {q}",
        f"세팅: up={al.get('upsampler', 'interp23tap')}(데이터셋 lms 정확 재현; 기존 bicubic phase 1.5 대체) · Δ={src_d}"
        f" · alpha={a:g} · 잔차 base={'P-frame(shift 된 MS)' if full else 'M-frame'} · 출력 frame={fr} · {inv_d}"
        " · PAN mode 는 inverse·마스크 없음(target=입력 PAN 복제)",
        "판정: best=HQNR(12-19, 동률 1e-4)→fSCC(12-19)→나중 iteration · 이 행 FR 지표=최종 frame(mat sr), "
        "RR 지표=M-frame 뷰(GT 좌표) · 다른 frame 뷰는 work_dir/<run>/best_hqnr_meta.json 의 hqnr_alt/fscc_alt",
        "anchor: S1_T05_W152_D123_DUAL(bicubic phase 1.5, HQNR 0.9546) — case 간 직접 비교는 P0 기준",
    ]


def _sr_notes(sr):
    """shift-robust 실행의 맞춤 세팅 설명 (계획 §5–§11)."""
    v = sr.get("variant", "?"); jit = sr.get("jitter") or {}; r = float(jit.get("max_abs_hr_px", 0.5))
    q = {"j1": "noisy cache 없이 통제된 무작위 jitter 만으로 C2 의 HQNR·후반 D_s 안정화가 재현되는가",
         "j2": "같은 jitter 를 MS mode 에만 주는 것과 두 mode 에 주는 것 중 무엇이 나은가 (PAN mode 의 shared-backbone 정규화 기여)",
         "j3": "C2 의 이득이 위치 perturbation 인가, sub-pixel 보간의 약한 smoothing 인가",
         "j4": "clean/jitter MS 가 같은 M-frame 잔차를 내도록 직접 제약하면 C2 보다 안정적인가",
         "g1": "MS 를 옮기지 않고 PAN 의 shallow feature 를 M-frame 으로 가져오면 HQNR·fSCC 를 함께 지키는가"}.get(v, "")
    st = {"j1": f"조건 MS = T_ε(bicubic ↑MS), ε~U(−{r:g},{r:g})² HR px(표본당 1개, 8밴드 동일), MS·PAN 두 mode 공통 입력",
          "j2": f"조건 MS = T_ε(bicubic ↑MS), ε~U(−{r:g},{r:g})² HR px, MS mode 만. PAN mode 는 clean 조건",
          "j3": "조건 MS = depthwise Gaussian(σ*) — σ* 는 J1 jitter 와 gradient-energy 비를 500 표본에서 맞춘 값(outputs/shift_robust/blur_calib.json). 위치 이동 없음, 두 mode 공통",
          "j4": f"MS mode 를 clean/jitter(±{r:g}px) 두 branch(가중치 공유): L_MS=½L1(Ŷ0)+½L1(Ŷε), L_cons=|Rε−sg(R0)|₁, λ 0→{(sr.get('cons') or {}).get('lambda', 0.1)} ({(sr.get('cons') or {}).get('warmup_steps', 5000)} step warmup). PAN mode clean. 3×48 forward",
          "g1": "first conv 를 PAN(0-2ch)/MS(3-10ch) 기여로 분리(합은 원 conv 와 동일). MS mode 에서 synthetic shift ε_g~U(−1,1)² (p 0.75) 를 PAN feature 에 주고 25 후보(±1 HR px, 0.5 step) edge-weighted descriptor 상관 → softmax(τ0.07) soft-argmax Δ̂, gate clip(conf/0.3), F̃_P=F_P+g[W(F_P,Δ̂)−F_P], L+=0.1·SmoothL1(Δ̂,−ε_g). PAN mode 는 원 경로"}.get(v, "")
    return [f"SR {v.upper()} — 질문: {q}", f"세팅: {st}",
            "공통: 원 bicubic(F.interpolate)·원 feeder·잔차 base/GT/출력 M-frame·PAN mode inverse 없음·추론 시 jitter 0 · 판정 best=HQNR(장면별 평균, 1e-4)→fSCC→나중 iteration",
            "anchor: S1_T05_W168_D123_DUAL(HQNR 0.9571 / fSCC 0.8785 / D_λ 0.0231 / D_s 0.0202)"]


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
        row = {"tag": label, "_ds": ds.upper(), "note": note, "date": "",
               "campaign": classify(label)}
        row.update(extra)
        rr = os.path.join(wd, "results", "reduced_best_val.mat")
        if os.path.exists(rr):
            row.update(_rr(rr, ds))
        row.update(_fr_paper(wd))
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
        # 캠페인은 **꾸미기 전 실행명**으로 정한다 — tag 는 뒤에서 "(50K) · w96 …"·"·peerB" 가 붙는다
        "campaign": classify(tag),
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
    m = _cost_model(a)                      # pa 면 aligner(0.1053M) 포함 — 시트 비용은 실제 추론 모델 전체
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
    # FR 은 논문 세트(.mat 20장)만 올린다. 배포 H5 12-19 (`_fr`) 는 2026-09-07 시트에서 뺐다 — 선택 전용.
    row.update(_fr_paper(wd, peer))          # tools/eval_fr_paperset.py 산출물이 유효(버전·해시 일치)할 때만
    if not any(k in row for k in ("p_hqnr",)):
        row["note_err"] = "FR·paper 없음 — tools/eval_fr_paperset.py <run> 을 돌릴 것 (KNOWN_ISSUES F-2)"
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
    elif _tr == "sr":
        _sr = getattr(a, "sr", {}) or {}
        _v = _sr.get("variant", "?"); _r = (_sr.get("jitter") or {}).get("max_abs_hr_px", 0.5)
        desc = (desc + " SR " + {"j1": f"J1(random jitter ±{_r:g}px, 두 mode)", "j2": f"J2(random jitter ±{_r:g}px, MS mode 만)",
                                  "j3": "J3(matched blur control)", "j4": f"J4(clean+jitter ±{_r:g}px consistency)",
                                  "g1": "G1(global PAN-feature correlator)"}.get(_v, _v)).strip()
    elif _tr == "pa":
        _p = getattr(a, "pa", {}) or {}; _c = _p.get("case", "?")
        desc = (desc + " PA " + {"A1": "A1(aligner, L_rec)", "A2": "A2(aligner, L_rec+edge)", "A3": "A3(aligner, L_rec+geo)"}.get(_c, _c)).strip()
    elif _tr == "po":
        _p = getattr(a, "po", {}) or {}; _c = _p.get("case", "?")
        desc = (desc + " PO10 " + {"N1": "N1(PAN corrupt, L_rec)", "N2_SG": "N2(corrupt + offset loss, sg)", "N3_NOSG": "N3(corrupt + offset loss, no-sg)"}.get(_c, _c) + f" R={_p.get('radius_hr', 1.0)}").strip()
    elif _tr == "kdv":
        from kdv.registry import resolve as _kdv_resolve, describe as _kdv_describe
        try:
            desc = (desc + " KDV " + _kdv_describe(_kdv_resolve(getattr(a, "kdv", {}) or {}))).strip()
        except Exception as _e:                      # 설명 실패가 업로드를 막지 않게
            desc = (desc + f" KDV (spec 해석 실패: {_e})").strip()
    elif _tr == "uvs":
        _u = getattr(a, "uvs", {}) or {}; _v = _u.get("variant", "?")
        desc = (desc + " UVS " + {"b0": "B0(lms baseline)", "k0": "K0(output KD)", "k1": "K1(U routing)", "k2": "K2(U+GT var)",
                                   "s0": "S0(shift KD)", "m1": "M1(K2+shift)", "m2": "M2(+teacher forcing)", "m3": "M3(+warp loss)"}.get(_v, _v)).strip()
    elif _tr == "align":
        _al = getattr(a, "alignment", {}) or {}
        _case, _ = _ga_case(_al)
        desc = (desc + f" GA {_case}: Δ={_al.get('delta_source', 'zero')} α={float(_al.get('alpha', 0)):g}"
                f" out={_al.get('output_frame', 'M')} inv={_al.get('inverse_location', 'none')}").strip()
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
        # task 축(PAN reconstruction 유무·mode 조건화)은 **항상** 적는다.
        # 2026-09-02 발견: 4-6M 대역에서 MS-only 는 학습 후반 D_lambda 가 계속 악화돼
        # HQNR 을 잃고(같은 구조 dual 대비 plateau -0.0112), 붕괴 폭이 용량에 비례한다.
        # 그래서 두 계열을 시트에서 반드시 구분할 수 있어야 한다.
        _ms = getattr(a, "mars", "dual") == "ms"
        _mm = ma.get("mode_modulation", True)
        if _ms and not _mm:
            bits.append("task=MS-only plain (PAN 재구성 없음 · γβ 조건화 없음 — "
                        "4-6M 에서 후반 D_lambda 악화 주의)")
        elif _ms:
            bits.append("task=MS-only (PAN 재구성 없음 · γβ 조건화 유지)")
        else:
            bits.append("task=dual MARs (MS+PAN 재구성 · γβ mode 조건화)"
                        + ("" if _mm else " · mode_modulation=False"))
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
        # task 축(PAN reconstruction 유무·mode 조건화)은 **항상** 적는다.
        # 2026-09-02 발견: 4-6M 대역에서 MS-only 는 학습 후반 D_lambda 가 계속 악화돼
        # HQNR 을 잃고(같은 구조 dual 대비 plateau -0.0112), 붕괴 폭이 용량에 비례한다.
        # 그래서 두 계열을 시트에서 반드시 구분할 수 있어야 한다.
        _ms = getattr(a, "mars", "dual") == "ms"
        _mm = ma.get("mode_modulation", True)
        if _ms and not _mm:
            bits.append("task=MS-only plain (PAN 재구성 없음 · γβ 조건화 없음 — "
                        "4-6M 에서 후반 D_lambda 악화 주의)")
        elif _ms:
            bits.append("task=MS-only (PAN 재구성 없음 · γβ 조건화 유지)")
        else:
            bits.append("task=dual MARs (MS+PAN 재구성 · γβ mode 조건화)"
                        + ("" if _mm else " · mode_modulation=False"))
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
    elif _tr == "align":
        bits.extend(_ga_notes(getattr(a, "alignment", {}) or {}))   # 맞춤 세팅 설명 (family 무관)
    elif _tr == "sr":
        bits.extend(_sr_notes(getattr(a, "sr", {}) or {}))
    elif _tr == "pa":
        _p = getattr(a, "pa", {}) or {}; _c = _p.get("case", "?")
        desc = (desc + " PA " + {"A1": "A1(aligner, L_rec)", "A2": "A2(aligner, L_rec+edge)", "A3": "A3(aligner, L_rec+geo)"}.get(_c, _c)).strip()
    elif _tr == "kdv":
        _k = getattr(a, "kdv", {}) or {}
        try:
            from kdv.registry import resolve as _kdv_resolve, describe as _kdv_describe
            _sp = _kdv_resolve(_k)
            bits.append(f"KDV {_sp['policy']} · REC-{_sp['rec_case']} · STAT-{('OFF' if not _sp['stat_enabled'] else _sp['stat_key'] + '-' + _sp['stat_mode'])} · {_sp['geom']} — {_kdv_describe(_sp)}")
            bits.append(f"donor {((_k.get('donor') or {}).get('source') or 'none')} · Teacher {((_k.get('teacher') or {}).get('run') or 'none')}/{((_k.get('teacher') or {}).get('tag') or '')} · run_kind {_k.get('run_kind', 'CONTROLLED')}")
        except Exception as _e:
            bits.append(f"KDV (spec 해석 실패: {_e})")
    elif _tr == "uvs":
        _u = getattr(a, "uvs", {}) or {}; _v = _u.get("variant", "?"); _l = _u.get("loss") or {}; _s = _u.get("shift") or {}
        _q = {"b0": "phase-correct 공통 baseline(제공 lms)", "k0": "일반 output KD 대조", "k1": "U-KD > plain KD?", "k2": "UV-KD > U-KD?",
              "s0": "shift cue 단독 기전", "m1": "UV-KD 위에 shift 를 더한 효과", "m2": "early teacher forcing 효과 (주력)", "m3": "vector KD 와 shift-effect KD 차이"}.get(_v, "")
        bits.append(f"UVS {_v.upper()} — 질문: {_q}")
        bits.append(f"세팅: 입력 [P, LP, P−LP, LMS(제공)] · 잔차 base LMS · teacher c0_hqnr cache(R_T=Y_T−LMS, U_T=θ Q10/Q90 정규화, δ_T,c_T) · "
                    f"λ_soft {_l.get('lambda_soft', 0.1)} λ_shift {_l.get('lambda_shift', 0.25)} λ_warp {_l.get('lambda_warp', 0)} α_V {(_u.get('variance') or {}).get('alpha', 1.0)} · "
                    f"shift: MS 격자 ±{_s.get('search_radius', 3)} cost-volume, T {_s.get('softmax_temperature', 0.07)}, 추론 gate c<{_s.get('confidence_threshold', 0.35)}→0, PAN 3ch {_s.get('warp_mode', 'bicubic')} warp ×4 · "
                    f"teacher forcing η 1→0 ({(_u.get('teacher_forcing') or {}).get('s0', 5000)}–{(_u.get('teacher_forcing') or {}).get('s1', 20000)})")
        bits.append("판정: best=HQNR(12-19, 장면별 평균)→fSCC · controlled-shift 는 tools/uvs_controlled_shift.py (results/controlled_shift.csv)")
    if "_zs_" in tag:
        bits.append("zero-shot: 학습 센서(WV3) best checkpoint 를 이 센서 테스트셋에 그대로 적용 · "
                    "lpan 은 F-1 레시피 생성(배포본 없어 검증 불가) · tools/make_zeroshot_run.py")
        row["train_h"] = ""                  # 학습이 없다 — 생성 시각으로 계산된 0h 를 지운다
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


def _needs_separator(tags, tag_cell):
    """이 행의 범주에 캠페인 설명(DESC)이 있고 시트에 그 구분행이 아직 없으면 구분행 문자열을 돌려준다."""
    key = classify(tag_cell)
    if key not in DESC:
        return None
    cell = separator_cell(key)
    return None if any(t.strip().startswith(cell) for t in tags) else cell


def _sep_request(ws, row_idx, n):
    """구분행 서식(굵게 · 연회색 배경) repeatCell 요청 하나. refile_sheet.py 의 구분행과 같은 모양."""
    return {"repeatCell": {
        "range": {"sheetId": ws.id, "startRowIndex": row_idx - 1, "endRowIndex": row_idx,
                  "startColumnIndex": ORIGIN_COL - 1, "endColumnIndex": ORIGIN_COL - 1 + n},
        "cell": {"userEnteredFormat": {
            "backgroundColor": {"red": .87, "green": .89, "blue": .93},
            "textFormat": {"bold": True}}},
        "fields": "userEnteredFormat(backgroundColor,textFormat.bold)"}}


def _format_separator(ws, row_idx, n):
    ws.spreadsheet.batch_update({"requests": [_sep_request(ws, row_idx, n)]})


def _retry(fn, *a, **kw):
    """Sheets API 429(분당 write 한도)·5xx 는 잠시 쉬고 다시 시도한다. 그 밖의 오류는 그대로 올린다."""
    import gspread
    for k in range(6):
        try:
            return fn(*a, **kw)
        except gspread.exceptions.APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code not in (429, 500, 502, 503) or k == 5:
                raise
            print(f"  [sheets] API {code} — 65초 후 재시도 ({k + 1}/5)", flush=True)
            time.sleep(65)


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
        grp = ws.get(f"{_a1(ORIGIN_ROW, ORIGIN_COL)}:{_a1(ORIGIN_ROW, ORIGIN_COL + n - 1)}")
        want_hdr = [c[1] for c in cols]
        want_grp = [c[0] if (i == 0 or cols[i - 1][0] != c[0]) else "" for i, c in enumerate(cols)]
        cur_hdr = (cur[0] + [""] * n)[:n] if cur else []
        cur_grp = (grp[0] + [""] * n)[:n] if grp else []
        if cur and cur_hdr != want_hdr or cur_grp != want_grp:
            got = ws.get(f"{_col(ORIGIN_COL)}{ORIGIN_ROW + 2}:{_col(ORIGIN_COL)}")
            # gspread 는 빈 범위에도 [[]] 를 돌려줄 수 있다 — 실제 값이 있는지 본다 (2026-09-08 QB-s1 빈 탭에서 오탐)
            has_rows = any(any(c.strip() for c in r) for r in (got or []))
            if has_rows and not replace:
                # 열 배치가 바뀌었는데 기존 행이 있다. 헤더만 다시 쓰면 기존 행의 셀이 새 열과 어긋난다
                # (2026-09-07 FR·paper 열 추가 때 생긴 상황). 단건 업로드는 건너뛰고 전체 재작성을 요구한다.
                print(f"  [{sheet_name(ds, server)}] !! 시트의 열 배치가 코드와 다르다 — 단건 업로드를 건너뛴다. "
                      f"`python gspread/gspread_upload.py --all --replace` 로 전체를 다시 쓸 것")
                continue
            _write_header(ws, cols, SHEET_COLOR.get(ds, (0.85, 0.89, 0.95)))
        elif not cur:
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

        # 행 쓰기는 모아서 한 번에 보낸다. Sheets API 는 분당 write 60회 제한이 있어 행마다 update 를
        # 부르면 --all --replace(150행)에서 429 로 죽는다 (2026-09-07 실제 발생).
        last = ORIGIN_ROW + 1
        pending, sep_rows = [], []
        for r in rs:
            v = fmt(r, cols)
            if r["tag"] in tags:
                i = ORIGIN_ROW + 2 + tags.index(r["tag"])
            else:
                # 새 캠페인(DESC 가 정의된 범주)의 첫 행이면 그 위에 구분행을 먼저 넣는다 —
                # 업로드는 맨 아래에 덧붙이므로 지난 실험과 섞여 보이지 않게. Notes 에 캠페인 설명.
                sep_cell = _needs_separator(tags, r["tag"])
                if sep_cell:
                    j = ORIGIN_ROW + 2 + len(tags)
                    tags.append(sep_cell)
                    srow = [""] * n; srow[0] = sep_cell; srow[-1] = DESC[classify(r["tag"])]
                    pending.append({"range": f"{_a1(j, ORIGIN_COL)}:{_a1(j, ORIGIN_COL + n - 1)}", "values": [srow]})
                    sep_rows.append(j)
                    last = max(last, j)
                i = ORIGIN_ROW + 2 + len(tags)
                tags.append(r["tag"]); added += 1
            pending.append({"range": f"{_a1(i, ORIGIN_COL)}:{_a1(i, ORIGIN_COL + n - 1)}", "values": [v]})
            last = max(last, i)
            total += 1
        if pending:
            _retry(ws.batch_update, pending)                       # 값 전체를 한 요청으로
        if sep_rows:
            _retry(ws.spreadsheet.batch_update, {"requests": [_sep_request(ws, j, n) for j in sep_rows]})
        _retry(_apply_borders, ws, cols, last)      # 새로 쓴 행까지 선을 이어준다
        _retry(_bold_best, ws, cols)
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
    ap.add_argument("--include-archived", action="store_true",
                    help="2026-09-11 에 _v1 탭으로 옮긴 범주(sheet_categories.ARCHIVED)의 run 도 --all 에 포함한다 (기본: 뺀다)")
    a = ap.parse_args()

    tags = []
    if a.all:
        tags = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p in glob.glob(f"{ROOT}/work_dir/*/results/reduced_*.mat")]
        # 무효·옛 프로토콜 run 은 --all 에서 뺀다: _INVALID_*(증강 위상 버그), *_msbug(QB 배포 ms 결함, F-3), *_sel1219(12-19 선택, 2026-09-09 폐기)
        tags = [t for t in tags if not t.startswith("_INVALID") and not t.endswith(("_msbug", "_sel1219"))]   # _sel1219: H5 12-19 로 선택한 옛 프로토콜 run
        if not a.include_archived:                       # 2026-09-11 시트 정리: 현 접근과 무관한 범주는 본 탭에 다시 올리지 않는다 (gspread/archive_to_v1.py)
            from sheet_categories import classify as _cls, ARCHIVED as _arch
            n0 = len(tags); tags = [t for t in tags if _cls(t) not in _arch]
            print(f"  --all: 이전 범주(_v1 탭) run {n0 - len(tags)}개 제외 (--include-archived 로 포함)")
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
