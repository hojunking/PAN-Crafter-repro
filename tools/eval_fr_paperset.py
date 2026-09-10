"""논문 비교용 FR 평가 — PanCollection **.mat 형식** WV3 full-resolution 20장(= 논문들이 MATLAB DLPan-Toolbox 로
평가한 세트)에 run 의 best checkpoint 를 추론해 D_λ / D_s / HQNR 을 낸다.

    python tools/eval_fr_paperset.py c0_hqnr d122 "SR_J4*"      # work_dir 이름 또는 glob
    python tools/eval_fr_paperset.py --all                       # results/ 가 있는 run 전부 (없는 것만)

배경: results_log/2026-09-07_metric-comparability-audit.md. 배포 H5 의 FR 20장은 논문 세트와
다른 장면 집합이라(겹침 6장) 시트의 FR(12-19) 열은 논문 표와 직접 비교할 수 없다. 이 스크립트가
내는 값이 논문 Table 의 FR 행과 같은 데이터·같은 프로토콜(D_lambda_K + block-UQI D_s, S=32,
HQNR=장면별 (1-D_λ)(1-D_s) 평균)이다. 입력 h5 는 tools/build_fr_paperset.py 로 만든다.

산출: work_dir/<run>/results/full_<ckpt>_mat20.mat (sr, NCHW DN — 기존 mat 과 같은 저장 규약)
      work_dir/<run>/results/fr_mat20.json   (hqnr, d_lambda, d_s, 표준편차, 장면별, checkpoint, 입력 h5)

지원 trainer: default / kd / teacher(base.* 만 사용) / sr(j1~j4: 추론은 jitter 0 = 원 forward, g1: correlator).
align 은 frozen cache Δ 가 필요해 여기서 다루지 않는다 (GA 캠페인은 종료됐다).
"""
import os, re, sys, glob, json, argparse, datetime, hashlib
import numpy as np, yaml, torch
import torch.nn.functional as F
from scipy.io import savemat, loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from main import import_class                                             # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s             # noqa: E402

SENSORS = ("wv3", "qb", "gf2", "wv2")


def h5_for(sensor):
    """센서별 논문 세트 h5 (tools/build_fr_paperset.py --sensor 가 만든다)."""
    return os.path.join(ROOT, "data", "PanCollection", sensor.upper(), "full_examples_mat20", f"test_{sensor}_OrigScale_mat20.h5")


def sensor_of(cfg):
    """run 의 FR 테스트 dataroot 파일명에서 센서를 읽는다 (test_<sensor>_OrigScale_...)."""
    base = os.path.basename(str(cfg.get("test_full_feeder_args", {}).get("dataroot", "")))
    return next((s for s in SENSORS if base.startswith(f"test_{s}_")), "wv3")


H5_DEFAULT = h5_for("wv3")           # 하위 호환 (gspread 등)
CKPTS = ("best_hqnr", "best_val", "best_reduced")

# 평가기 버전. tools/metrics/eval_fr.py 나 이 파일의 추론·집계가 바뀌면 올린다 — JSON 의 eval_version 이
# 다르면 캐시를 버리고 다시 잰다 (검증 지적 2026-09-07: 캐시가 코드·데이터 변경을 몰랐다).
#   2026-09-07.1  최초 (DLPan 파이썬 포트 MTF 커널, imresize replicate, std N)
#   2026-09-07.2  genMTF.m 충실 커널(정규화 없음)·imresize symmetric·std N-1·provenance 필드·uvs/mutual 지원
#   2026-09-08.3  JQM(Palubinskas 2015; tools/metrics/jqm.py) 추가 — D_λ/D_s/HQNR 은 .2 와 같으므로 .2 JSON 은 저장된
#                 mat 에서 JQM 만 계산해 올린다(재추론 없음)
#   2026-09-08.4  JQM 을 SIPSA-Net 규약(QLR 균등평균·QHR 볼록 가중)으로, 범위 정책 추가 — .2/.3 JSON 도 mat 에서 JQM 만 갱신
EVAL_VERSION = "2026-09-10.5"            # .5: HQNR(V64) — 가장자리 64 px(블록 2개) 제외 고정 영역의 D_λ/D_s/HQNR/fSCC 를 함께 기록 (masking 유무 두 가지, 2026-09-10 사용자 요청)
JQM_COMPATIBLE = ("2026-09-07.2", "2026-09-08.3")
VIEWS_COMPATIBLE = ("2026-09-08.4",)      # JQM 까지 있는 JSON: 저장 mat 에서 V64 필드만 더한다 (재추론 없음)

_SHA = {}


def sha256_of(path):
    """파일 sha256 (프로세스 안에서 캐시). 400MB h5 도 1~2초."""
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def provenance(h5, cfgp):
    lp = h5.replace(".h5", "_pan.h5")
    return dict(eval_version=EVAL_VERSION, input_h5=h5, input_sha256=sha256_of(h5),
                lpan_sha256=sha256_of(lp) if os.path.exists(lp) else "", config_sha256=sha256_of(cfgp))


def pick_ckpt(wd):
    return next((c for c in CKPTS if os.path.exists(os.path.join(wd, c, "model.safetensors"))), None)


def build(cfg, wd, ckpt, peer=None):
    """(model, callable(pan,lpan,ms,lms)->y[-1,1], 설명) — trainer 별 추론 경로.
    peer="B" 는 mutual trainer 의 두 번째 peer (accelerate 가 model_1.safetensors 로 저장)."""
    from safetensors.torch import load_file
    fn = "model_1.safetensors" if peer == "B" else "model.safetensors"
    sd = load_file(os.path.join(wd, ckpt, fn))
    tr = cfg.get("trainer", "default")
    Model = import_class(cfg["model"])
    res = bool(cfg.get("res", True)); rbase = cfg.get("residual_base", "bicubic")

    def base_of(ms, lms):
        return lms if rbase == "lms" else F.interpolate(ms, scale_factor=4, mode="bicubic")

    if tr == "uvs":
        # train_uvs.UVSTrainer.infer 와 같은 경로: 제공 lms 잔차, student shift 를 gate 한 δ 로 PAN 3채널 rigid warp
        from train_uvs import UVSModel, build_inputs, x11, USES_SHIFT
        from uvs.shift import ShiftModule, edge_rep, warp_pan_channels, gated_delta
        u = cfg.get("uvs") or {}; v = u.get("variant"); sh_cfg = u.get("shift") or {}
        g = lambda d, k, dflt: dflt if (d or {}).get(k) is None else d[k]
        shift = (ShiftModule(tuple(g(sh_cfg, "student_channels", [8, 8])), int(g(sh_cfg, "search_radius", 3)),
                             float(g(sh_cfg, "softmax_temperature", 0.07))) if v in USES_SHIFT else None)
        conf_thr = float(g(sh_cfg, "confidence_threshold", 0.35)); warp_mode = g(sh_cfg, "warp_mode", "bicubic")
        m = UVSModel(Model(**cfg["model_args"]), shift); m.load_state_dict(sd, strict=True)

        def fwd(pan, lpan, ms, lms):
            lpan_u, pan_hf = build_inputs(pan, lpan, lms)
            if m.shift is not None:
                s = m.shift(edge_rep(lpan), edge_rep(ms)); d = gated_delta(s["delta"], s["conf"], conf_thr)
                pan, lpan_u, pan_hf = warp_pan_channels(pan, lpan_u, pan_hf, d, warp_mode)
            sw = torch.ones(ms.shape[0], device=ms.device, dtype=ms.dtype)
            return lms + m.backbone(None, None, None, sw, x_in=x11(pan, lpan_u, pan_hf, lms))
        return m, fwd, f"uvs/{v}: lms + backbone(x11(warp(PAN ch, gate(δ_S)), lms))"

    if tr in ("default", "kd", "teacher", "mutual"):
        m = Model(**cfg["model_args"])
        if tr == "teacher":                       # TeacherModel(base=backbone, head=unc) — backbone 만
            sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        try:
            m.load_state_dict(sd, strict=True)
        except RuntimeError:
            # norm 옵션 도입 전 meta 스냅샷(예: paper_wv3)은 'norm' 키가 없고 그때 동작은 GroupNorm 이었다
            # (gspread_upload._descriptor 와 같은 해석). 기본값 ln 으로 만든 모델에는 안 실린다.
            if "norm" in cfg["model_args"]:
                raise
            m = Model(**dict(cfg["model_args"], norm="gn"))
            m.load_state_dict(sd, strict=True)

        def fwd(pan, lpan, ms, lms):
            sw = torch.ones(pan.shape[0], device=pan.device, dtype=pan.dtype)
            y = m(pan, lpan, ms, sw)
            return y + base_of(ms, lms) if res else y
        return m, fwd, f"{tr}: model(pan,lpan,ms,1)" + (f" + {rbase}" if res else "")
    if tr == "sr":
        from sr.forward import sr_infer
        from sr.pan_align import GlobalCorrelator
        from train_sr import SRModel
        bb = Model(**cfg["model_args"]); s = cfg.get("sr") or {}; v = s["variant"]; g1 = s.get("g1") or {}
        corr = GlobalCorrelator(bb.input.out_channels, int(g1.get("desc_channels", 16)), float(g1.get("radius_hr_px", 1.0)),
                                int(g1.get("n_per_axis", 5)), float(g1.get("tau", 0.07)), float(g1.get("gate_c0", 0.30))) if v == "g1" else None
        m = SRModel(bb, corr); m.load_state_dict(sd, strict=True)

        def fwd(pan, lpan, ms, lms):
            return sr_infer(m, v, pan, lpan, ms)["y"]
        return m, fwd, f"sr/{v}: sr_infer (jitter 0)"
    if tr in ("pa", "po", "kdv"):
        from pa.aligner import PANGlobalAligner
        from pa.model import PAModel
        from pa.offset import aligner_margin
        if tr == "kdv":
            from kdv.teacher_assets import skeleton_from_cfg
            m, _info = skeleton_from_cfg(cfg, Model); mg = _info["margin"]; m.load_state_dict(sd, strict=True)
        else:
            mg = aligner_margin(float((cfg.get("po") or {}).get("radius_hr", 1.0))) if tr == "po" else 0
            bb = Model(**cfg["model_args"]); m = PAModel(bb, PANGlobalAligner(int(cfg["num_bands"])), aligner_margin=mg); m.load_state_dict(sd, strict=True)

        def fwd(pan, lpan, ms, lms):
            return m(pan, ms, lpan)["y"]
        _case = (cfg.get(tr) or {}).get('case') if tr != "kdv" else f"{(cfg.get('kdv') or {}).get('aligner_policy')}/{((cfg.get('kdv') or {}).get('rec') or {}).get('case', 'N0')}"
        return m, fwd, f"{tr}/{_case}: " + ("no aligner/sampler (A-ID)" if m.aligner is None else f"aligner on (learned Δ, view margin {mg})") + ", raw_original view"
    raise NotImplementedError(f"trainer={tr} 는 이 스크립트가 다루지 않는다")


def cache_valid(j, ckpt, ckpt_mtime, prov, ignore_version=False):
    """캐시 유효 조건: 평가기 버전·입력 h5·lpan·config 해시·checkpoint 이름·checkpoint 수정시각이 전부 같다."""
    keys = ("input_sha256", "lpan_sha256", "config_sha256") + (() if ignore_version else ("eval_version",))
    return (j.get("checkpoint") == ckpt and abs(float(j.get("ckpt_mtime", -1)) - ckpt_mtime) < 1e-6
            and all(j.get(k) == prov[k] for k in keys))


def jqm_fields(sr_hwc_list, ms_all, pan_all, sensor, R):
    """장면별 JQM/QLR/QHR (tools/metrics/jqm.py). sr_hwc_list: (H,W,C) DN 리스트."""
    from tools.metrics.jqm import jqm
    r = [jqm(s, ms_all[i], pan_all[i], sensor.upper(), 4, R) for i, s in enumerate(sr_hwc_list)]
    J = np.array([x["JQM"] for x in r]); L = np.array([x["QLR"] for x in r]); H = np.array([x["QHR"] for x in r])
    sd = (lambda v: float(v.std(ddof=1))) if len(J) > 1 else (lambda v: 0.0)
    return dict(jqm=float(J.mean()), jqm_sd=sd(J), qlr=float(L.mean()), qhr=float(H.mean()),
                per_scene_jqm=[round(float(x), 6) for x in J], jqm_w_source=r[0]["w_source"],
                jqm_protocol="SIPSA-Net supp Eq.3-9 규약: CMSC 전역 통계(Palubinskas Eq.4), QLR 밴드 균등평균, QHR 볼록 가중 intensity"
                             "(w=" + r[0]["w_source"] + "; SRF 없어 NNLS 정규화 대체), lpf=genMTF(sensor)+(2,2) 데시메이션, 입력 [0,R] 클립, v1=v2=0.5, R=2^L-1")


def valid_fields(sr_hwc_list, lms_all, pan_all, sensor, wald, R):
    """전체 프레임(raw_original)·고정 V64(raw_valid) 두 view. 전체 프레임 값은 d_lambda_k/d_s 와 동일(E02) — 여기서는 V64 와 fSCC 만 취한다."""
    from pa.evalviews import raw_views, MARGIN
    r = [raw_views(s, lms_all[i], pan_all[i], sensor, wald, 4, R) for i, s in enumerate(sr_hwc_list)]
    hv = np.array([x["raw_valid"]["hqnr"] for x in r]); dlv = np.array([x["raw_valid"]["d_lambda"] for x in r]); dsv = np.array([x["raw_valid"]["d_s"] for x in r])
    f0 = np.array([x["raw_original"]["fscc"] for x in r]); fv = np.array([x["raw_valid"]["fscc"] for x in r])
    sd = (lambda v: float(v.std(ddof=1))) if len(hv) > 1 else (lambda v: 0.0)
    return dict(hqnr_valid=float(hv.mean()), hqnr_valid_sd=sd(hv), d_lambda_valid=float(dlv.mean()), d_s_valid=float(dsv.mean()), fscc=float(f0.mean()), fscc_valid=float(fv.mean()),
                per_scene_hqnr_valid=[round(float(x), 6) for x in hv], valid_roi=f"[{MARGIN}:H-{MARGIN}, {MARGIN}:W-{MARGIN}] (block-aligned; filters on full frame, then crop)",
                valid_protocol="pa/evalviews.raw_views — masking 유무 두 HQNR: HQNR(전체 프레임, 논문 프로토콜) · HQNR(V64)(가장자리 64px 제외)")


def run_one(tag, h5, wald, dev, force, peer=None):
    wd = os.path.join(ROOT, "work_dir", tag)
    sfx = "_peerB" if peer == "B" else ""
    out_json = os.path.join(wd, "results", f"fr_mat20{sfx}.json")
    cfgp = os.path.join(wd, "meta", "config.yaml")
    if not os.path.exists(cfgp):
        return None, "config 없음"
    ckpt = pick_ckpt(wd)
    if ckpt is None:
        return None, "checkpoint 없음"
    ckpt_file = os.path.join(wd, ckpt, "model_1.safetensors" if peer == "B" else "model.safetensors")
    if not os.path.exists(ckpt_file):
        return None, f"{os.path.basename(ckpt_file)} 없음"
    ckpt_mtime = os.path.getmtime(ckpt_file)
    # 학습 중인 run 은 best checkpoint 가 아직 바뀐다 — 부분 결과를 남기지 않는다 (--force 로 강제 가능).
    meta = os.path.join(wd, "meta")
    if not force and os.path.exists(os.path.join(meta, "started_at.txt")) and not os.path.exists(os.path.join(meta, "finished_at.txt")):
        return None, "학습 진행 중 (finished_at 없음)"
    cfg = yaml.safe_load(open(cfgp))
    sensor = sensor_of(cfg)
    if h5 is None:
        h5 = h5_for(sensor)
    if not os.path.exists(h5):
        return None, f"논문 세트 h5 없음 ({os.path.relpath(h5, ROOT)}) — tools/build_paperset_all.sh {sensor}"
    prov = provenance(h5, cfgp)
    R = float(2 ** int(round(np.log2(2048.0 if sensor != "gf2" else 1024.0))) - 1)     # 2047 / 1023
    if os.path.exists(out_json) and not force:
        j = json.load(open(out_json))
        if cache_valid(j, ckpt, ckpt_mtime, prov):
            return j, "cached"
        # 이전 호환 버전 JSON: D_λ/D_s/HQNR 은 그대로 두고 저장된 mat 에서 JQM 만 더한다 (재추론 없음)
        mat_prev = os.path.join(ROOT, j.get("mat", "")) if j.get("mat") else ""
        if j.get("eval_version") in JQM_COMPATIBLE and cache_valid(j, ckpt, ckpt_mtime, prov, ignore_version=True) and os.path.exists(mat_prev):
            import h5py
            with h5py.File(h5) as f:
                ms_all = np.asarray(f["ms"], dtype=np.float64).transpose(0, 2, 3, 1); pan_all = np.asarray(f["pan"], dtype=np.float64)[:, 0]
            srm = loadmat(mat_prev)["sr"].astype(np.float64)
            srl = [srm[i].transpose(1, 2, 0) for i in range(len(srm))]
            with h5py.File(h5) as f:
                lms_all = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
            j.update(jqm_fields(srl, ms_all, pan_all, sensor, R)); j.update(valid_fields(srl, lms_all, pan_all, sensor, wald, R)); j["eval_version"] = EVAL_VERSION
            json.dump(j, open(out_json, "w"), indent=1)
            return j, "jqm+valid-added"
        # JQM 까지 있는 JSON: 저장 mat 에서 V64 view 만 더한다 (재추론 없음)
        if j.get("eval_version") in VIEWS_COMPATIBLE and cache_valid(j, ckpt, ckpt_mtime, prov, ignore_version=True) and os.path.exists(mat_prev):
            import h5py
            with h5py.File(h5) as f:
                lms_all = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1); pan_all = np.asarray(f["pan"], dtype=np.float64)[:, 0]
            srm = loadmat(mat_prev)["sr"].astype(np.float64)
            srl = [srm[i].transpose(1, 2, 0) for i in range(len(srm))]
            j.update(valid_fields(srl, lms_all, pan_all, sensor, wald, R)); j["eval_version"] = EVAL_VERSION
            json.dump(j, open(out_json, "w"), indent=1)
            return j, "valid-added"
    try:
        m, fwd, how = build(cfg, wd, ckpt, peer)
    except NotImplementedError as e:
        return None, str(e)
    m = m.to(dev).eval()
    Feeder = import_class(cfg["feeder"])
    fargs = dict(cfg["test_full_feeder_args"]); fargs["dataroot"] = h5
    ds = Feeder(**fargs)
    mp = float(ds.max_pixel)
    srs = []
    with torch.no_grad():
        for i in range(len(ds)):
            lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i])
            y = fwd(pan, lpan, ms, lms)
            # train.py save_test_full 과 같은 역변환: clip(-1,1) -> (x+1)/2 -> * max_pixel (반올림 없음)
            srs.append(((y.clip(-1.0, 1.0).float().cpu().numpy() + 1.0) / 2.0 * mp)[0])
    sr = np.stack(srs)                                     # (N, C, H, W)
    os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    mat_path = os.path.join(wd, "results", f"full_{ckpt}_mat20{sfx}.mat")
    savemat(mat_path, dict(sr=sr))
    import h5py
    with h5py.File(h5) as f:
        lms_all = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        ms_all = np.asarray(f["ms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan_all = np.asarray(f["pan"], dtype=np.float64)[:, 0]
    dl, dsv, srl = [], [], []
    for i in range(len(sr)):
        s = sr[i].astype(np.float64).transpose(1, 2, 0); srl.append(s)
        # 센서별 MTF(genMTF.m: WV3/WV2/QB 표, GF2 는 otherwise 0.3) — eval_fr.SENSOR_NAME 이 preset 키로 매핑
        dl.append(d_lambda_k(s, lms_all[i], sensor, 4, 32, wald)); dsv.append(d_s(s, lms_all[i], pan_all[i], 4, 32, wald))
    jq = jqm_fields(srl, ms_all, pan_all, sensor, R); jq.update(valid_fields(srl, lms_all, pan_all, sensor, wald, R))
    dl, dsv = np.array(dl), np.array(dsv); h = (1 - dl) * (1 - dsv)
    sd = (lambda v: float(v.std(ddof=1))) if len(h) > 1 else (lambda v: 0.0)     # MATLAB std (N-1)
    j = dict(hqnr=float(h.mean()), hqnr_sd=sd(h), d_lambda=float(dl.mean()), d_lambda_sd=sd(dl),
             d_s=float(dsv.mean()), d_s_sd=sd(dsv), per_scene_hqnr=[round(float(x), 6) for x in h],
             per_scene_d_lambda=[round(float(x), 6) for x in dl], per_scene_d_s=[round(float(x), 6) for x in dsv],
             n=int(len(sr)), checkpoint=ckpt, ckpt_mtime=ckpt_mtime, peer=peer or "", forward=how, mat=os.path.relpath(mat_path, ROOT),
             sensor=sensor, std="ddof=1 (MATLAB std)",
             protocol="DLPan HQNR: D_lambda_K(genMTF.m 충실 커널, q2n S=32) + D_s(block-UQI S=32, interp23tap(imresize symmetric(PAN,1/4))) ; HQNR = mean_i (1-Dl_i)(1-Ds_i)",
             evaluated_at=datetime.datetime.now().isoformat(timespec="seconds"), **prov, **jq)
    json.dump(j, open(out_json, "w"), indent=1)
    return j, "new"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--h5", default=None, help="논문 세트 h5 (기본: run 의 센서에 맞는 data/PanCollection/<DS>/full_examples_mat20/)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default=None, help="i/n — 대상 run 을 n 조각으로 나눠 i 번째만 (병렬 배치용)")
    a = ap.parse_args()
    tags = []
    if a.all:
        tags = sorted({os.path.basename(os.path.dirname(d)) for d in glob.glob(os.path.join(ROOT, "work_dir", "*", "results"))})
    for pat in a.pattern:
        tags += [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "work_dir", pat))) if os.path.isdir(d)]
    seen = set(); tags = [t for t in tags if not (t in seen or seen.add(t)) and not t.startswith("_INVALID") and not t.endswith(("_msbug", "_sel1219"))]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/")); tags = tags[i::n]
    if not tags:
        print("대상 없음"); return 1
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    print(f"입력 {a.h5 or '센서별 full_examples_mat20'}  device={a.device}  {len(tags)} run")
    for t in tags:
        peers = [None]
        ck = pick_ckpt(os.path.join(ROOT, "work_dir", t))
        if ck and os.path.exists(os.path.join(ROOT, "work_dir", t, ck, "model_1.safetensors")):
            peers.append("B")                      # mutual trainer: peer_b 도 병기
        for peer in peers:
            lab = t + ("·peerB" if peer == "B" else "")
            try:
                j, st = run_one(t, a.h5, wald, a.device, a.force, peer)
            except Exception as e:                 # 한 run 이 죽어도 나머지는 계속
                print(f"  {lab:52s} 실패: {type(e).__name__}: {e}", flush=True); continue
            if j is None:
                print(f"  {lab:52s} 건너뜀 ({st})", flush=True); continue
            print(f"  {lab:52s} HQNR {j['hqnr']:.4f}±{j['hqnr_sd']:.4f}  HQNR(V64) {j.get('hqnr_valid', float('nan')):.4f}  D_l {j['d_lambda']:.4f}  D_s {j['d_s']:.4f}  JQM {j.get('jqm', float('nan')):.4f}  [{j['checkpoint']}, {st}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
