"""D10/D11 — 네 집단 atlas(§7): 유효 model/sample 의 e_full/e_roi · q_A/q_B/EPE(반경별) · c0 · label(CAL median 고정) · boundary_near · e_base/gain(D11-B) · contrast-normalized e(D11-A) · band/edge(D11-C 일부).
raw/sample_metrics.csv · manifests/thresholds.json · analysis/d10_stats.json. in-domain 은 CAL/DISC/CONF, WV2 zero-shot 은 RR tile(GT 있음)/FR tile(GT 없음 → e 없음)."""
import os
import numpy as pd_np, numpy as np, pandas as pd, torch, torch.nn.functional as F
from tools.smec12 import common as C
PHASE = os.environ.get("SMEC12_PHASE", "")


def rows_patches(L, sensor, part, ids):
    gt, ms, lpan, pan = C.load_patches(sensor, ids); y0 = C.reconstruct_at(L, pan, ms, lpan); em = C.e_metrics(y0, gt); mb = F.interpolate(ms, scale_factor=4, mode="bicubic"); e_base = (mb - gt).abs().mean(dim=(1, 2, 3))
    st = C.E.input_stats(pan, ms); sat = (pan >= 0.999).float().mean(dim=(1, 2, 3)); ctr = C.contrast(gt).mean(1); hf = (gt - F.avg_pool2d(F.pad(gt, (2, 2, 2, 2), mode="reflect"), 5, stride=1)).abs().mean(dim=(1, 2, 3))
    qa = C.measure_offset_bank(L, pan, ms, "A") if L.has_aligner else None; qb = C.measure_offset_bank(L, pan, ms, "B") if L.has_aligner else None
    rows = []
    for i, sid in enumerate(ids):
        r = dict(sensor=C.SENSORS[sensor]["S"], model=L.key, role=L.role, seed=L.seed, tag=L.tag, step=L.step, scale="native64", part=part, sample_id=int(sid), source_group=C.group_of(sensor, sid, part), seen_in_pretraining=True,
                 e_full=float(em["e_full"][i]), e_roi=float(em["e_roi"][i]), edge_l1=float(em["edge_l1"][i]), e_band_max=float(em["e_band"][i].max()), e_band_iqr=float(em["e_band"][i].quantile(.75) - em["e_band"][i].quantile(.25)),
                 e_base=float(e_base[i]), g_restore=float(e_base[i] - em["e_full"][i]), contrast_gt=float(ctr[i]), gt_hf_energy=float(hf[i]), pan_texture=float(st["pan_scharr_energy"][i]), pan_std=float(st["pan_std"][i]), pan_msmean_corr=float(st["pan_msmean_corr"][i]), pan_sat_frac=float(sat[i]))
        if qa is not None:
            r.update(q_A=float(qa["q"][i]), epe_A=float(qa["epe"][i]), q_B=float(qb["q"][i]), epe_B=float(qb["epe"][i]), c0_y=float(qa["c0"][i, 0]), c0_x=float(qa["c0"][i, 1]), c0_norm=float(qa["c0"][i].norm()), identity_floor=float(qa["identity"][i]),
                     **{f"q_A_r{rr}": float(qa["q_by_r"][str(rr)][i]) for rr in C.RADII_AB}, **{f"q_B_r{rr}": float(qb["q_by_r"][str(rr)][i]) for rr in C.RADII_AB}, q_A_rel=C.q_rel(float(qa["q"][i]), "A"))
        else:
            r.update(q_A="NA", q_B="NA", c0_y=0.0, c0_x=0.0, c0_norm=0.0)
        rows.append(r)
    return rows


def rows_tiles(L, sensor, fd, part):
    """zero-shot(WV2) / scene tile: RR tile 은 GT L1, FR tile 은 e 없음 (§4.4: 1−HQNR 을 L1 로 치환하지 않는다)."""
    ds = fd["rr"] if part == "RRTILE" else fd["fr"]; rows = []
    for si in range(len(ds)):
        t = [x.unsqueeze(0) for x in ds[si]]; pan, ms, lpan = t[-1], t[-3], t[-2]; gt = t[0] if part == "RRTILE" else None; coords = C.scene_tiles(pan[0], part); T = C.TILE
        gts, mss, lps, pns = [], [], [], []
        for (y, x) in coords:
            pns.append(pan[:, :, y:y + T, x:x + T]); lps.append(lpan[:, :, y // 4:(y + T) // 4, x // 4:(x + T) // 4]); mss.append(ms[:, :, y // 4:(y + T) // 4, x // 4:(x + T) // 4]); gts.append(gt[:, :, y:y + T, x:x + T] if gt is not None else None)
        pn = torch.cat(pns); mq = torch.cat(mss); lp = torch.cat(lps); y0 = C.reconstruct_at(L, pn, mq, lp); qa = C.measure_offset_bank(L, pn, mq, "A") if L.has_aligner else None; qb = C.measure_offset_bank(L, pn, mq, "B") if L.has_aligner else None
        em = C.e_metrics(y0, torch.cat(gts)) if gt is not None else None; st = C.E.input_stats(pn, mq)
        for k, (y, x) in enumerate(coords):
            r = dict(sensor=C.SENSORS[sensor]["S"], model=L.key, role=L.role, seed=L.seed, tag=L.tag, step=L.step, scale=("rr256tile" if part == "RRTILE" else "fr512tile"), part=part, sample_id=si * C.TILES_PER_SCENE + k, source_group=C.group_of(sensor, si * C.TILES_PER_SCENE + k, part),
                     scene_id=si, crop_box=f"{y},{x},{T},{T}", seen_in_pretraining=False, pan_texture=float(st["pan_scharr_energy"][k]), pan_std=float(st["pan_std"][k]), pan_msmean_corr=float(st["pan_msmean_corr"][k]))
            if em is not None:
                r.update(e_full=float(em["e_full"][k]), e_roi=float(em["e_roi"][k]), edge_l1=float(em["edge_l1"][k]), e_base=float((F.interpolate(mq[k:k + 1], scale_factor=4, mode="bicubic") - gts[k]).abs().mean()), contrast_gt=float(C.contrast(gts[k]).mean()))
            if qa is not None:
                r.update(q_A=float(qa["q"][k]), epe_A=float(qa["epe"][k]), q_B=float(qb["q"][k]), c0_y=float(qa["c0"][k, 0]), c0_x=float(qa["c0"][k, 1]), c0_norm=float(qa["c0"][k].norm()), **{f"q_A_r{rr}": float(qa["q_by_r"][str(rr)][k]) for rr in C.RADII_AB})
            rows.append(r)
    return rows


def assign(sm, thr):
    sm = sm.copy(); sm["quadrant"] = None; sm["boundary_near"] = False
    for (sensor, model, scale), g in sm.groupby(["sensor", "model", "scale"]):
        t = thr.get(f"{sensor}|{model}|{scale}")
        if not t or "e_full" not in g or (g.q_A.astype(str) == "NA").all():
            continue
        q = pd.to_numeric(g.q_A, errors="coerce"); e = g.e_full
        sm.loc[g.index, "quadrant"] = [C.label(ee, qq, t["tC"], t["tR"]) if pd.notna(qq) else None for ee, qq in zip(e, q)]
        sm.loc[g.index, "boundary_near"] = [bool(pd.notna(qq) and (C.boundary_near(qq, t["tC"], t["q_iqr"]) or C.boundary_near(ee, t["tR"], t["e_iqr"]))) for ee, qq in zip(e, q)]
    return sm


def stats(sm, thr):
    out = {}
    for (sensor, model, scale, part), g in sm.groupby(["sensor", "model", "scale", "part"]):
        if "e_full" not in g or g.e_full.isna().all():
            continue
        q = pd.to_numeric(g.q_A, errors="coerce"); ok = q.notna()
        d = dict(n=int(len(g)), n_sources=int(g.source_group.nunique()), e_median=float(g.e_full.median()), e_base_median=float(g.e_base.median()) if "e_base" in g else None)
        if ok.sum() >= 8:
            gg = g[ok]; qq = q[ok]
            d.update(spearman_q_e=C.spearman_boot(qq, gg.e_full, gg.source_group), spearman_q_e_roi=C.spearman(qq, gg.e_roi), loso_q_e=C.loso(qq, gg.e_full, gg.source_group),
                     spearman_q_e_by_r={str(r): C.spearman(gg[f"q_A_r{r}"], gg.e_full)["rho"] for r in C.RADII_AB if f"q_A_r{r}" in gg}, spearman_qB_e=C.spearman(gg.q_B, gg.e_full)["rho"], spearman_qA_qB=C.spearman(qq, gg.q_B)["rho"],
                     spearman_q_ebase=C.spearman(qq, gg.e_base)["rho"] if "e_base" in gg else None, spearman_q_contrast=C.spearman(qq, gg.contrast_gt)["rho"] if "contrast_gt" in gg else None, spearman_q_texture=C.spearman(qq, gg.pan_texture)["rho"],
                     spearman_e_ebase=C.spearman(gg.e_full, gg.e_base)["rho"] if "e_base" in gg else None, spearman_e_contrast=C.spearman(gg.e_full, gg.contrast_gt)["rho"] if "contrast_gt" in gg else None)
            if "e_contrast" in gg:
                d["spearman_q_e_contrastnorm"] = C.spearman_boot(qq, gg.e_contrast, gg.source_group)
            if "quadrant" in gg and gg.quadrant.notna().any():
                d["quadrants"] = gg.quadrant.value_counts().to_dict(); d["boundary_near_frac"] = float(gg.boundary_near.mean())
                d["by_quadrant"] = {qd: dict(n=int(len(h)), e=float(h.e_full.median()), q=float(pd.to_numeric(h.q_A).median()), e_base=float(h.e_base.median()) if "e_base" in h else None, g_restore=float(h.g_restore.median()) if "g_restore" in h else None, texture=float(h.pan_texture.median()), contrast=float(h.contrast_gt.median()) if "contrast_gt" in h else None, edge=float(h.edge_l1.median())) for qd, h in gg.groupby("quadrant")}
        out[f"{sensor}|{model}|{scale}|{part}"] = d
    out["thresholds"] = thr; return out


def main(profile=False):
    with C.Stage("D10" + (f"-{PHASE}" if PHASE else ""), "quadrant atlas / thresholds / D11 scale-difficulty"):
        smp = C.p_("raw", "sample_metrics.csv"); have = set(pd.read_csv(smp).model.unique()) if os.path.exists(smp) else set(); rows = []
        for s in ("wv3", "qb", "gf2"):
            if not C.data_registry()[C.SENSORS[s]["S"]]["data_ready"]:
                continue
            panels = C.make_panels(s)
            for (sensor, role, seed, tag) in C.available(s, roles=("L1E4", "L1E4REP", "L000", "P0"), tags=("best_hqnr", "last")):
                L = C.load_asset((sensor, role, seed), tag)
                if L.key in have:
                    continue
                mrows = []
                for part in ("CAL", "DISC", "CONF"):
                    ids = panels["ids"][part][:64] if profile else panels["ids"][part]; mrows += rows_patches(L, s, part, ids); print(f"[d10] {L.key} {part} {len(ids)}", flush=True)
                C.append_rows(smp, mrows, ["model", "part", "sample_id"]); rows += mrows           # model 단위로 바로 기록 (중간 실패해도 잃지 않게)
        if C.available("wv3", roles=("L1E4",)):                                          # WV2 zero-shot: WV3 L1E4(primary + REP) 를 RR/FR tile 에
            fd = C.feeders("wv2")
            for (sensor, role, seed, tag) in C.available("wv3", roles=("L1E4", "L1E4REP"), tags=("best_hqnr",)):
                L = C.load_asset((sensor, role, seed), tag)
                if f"{L.key}@wv2" in have:
                    continue
                Lz = L; rr = rows_tiles(Lz, "wv2", fd, "RRTILE") + ([] if profile else rows_tiles(Lz, "wv2", fd, "FRTILE"))
                for r in rr:
                    r["model"] = f"{L.key}@wv2"; r["zero_shot"] = True
                C.append_rows(smp, rr, ["model", "part", "sample_id"]); rows += rr; print(f"[d10] WV2 zero-shot {L.key}", flush=True)
        sm = pd.read_csv(smp)
        thr = C.load_json(C.p_("manifests", "thresholds.json"), {}); noise = float((C.load_json(C.p_("manifests", "a00_unit_checks.json"), {}).get("noise_floor") or {}).get("q_repeat_max_abs") or 0.0)
        for (sensor, model, scale), g in sm.groupby(["sensor", "model", "scale"]):
            cal = g[g.part == ("CAL" if scale == "native64" else "RRTILE")]; q = pd.to_numeric(cal.q_A, errors="coerce")
            if len(cal) and q.notna().sum() >= 8 and "e_full" in cal and cal.e_full.notna().any():
                thr[f"{sensor}|{model}|{scale}"] = dict(C.thresholds_from(cal.e_full[q.notna()], q[q.notna()], noise), part=("CAL" if scale == "native64" else "RRTILE(zero-shot, test-descriptive)"))
        C.dump_json(C.p_("manifests", "thresholds.json"), thr); sm = assign(sm, thr)
        floors = {}
        for (sensor,), g in sm[sm.part == "CAL"].groupby(["sensor"]):
            floors[sensor] = float(np.quantile(g.contrast_gt[g.contrast_gt > 0], .05)) if (g.contrast_gt > 0).any() else 1e-6
        sm["e_contrast"] = sm.apply(lambda r: (r.e_full / max(r.contrast_gt, floors.get(r.sensor, 1e-6))) if pd.notna(r.get("contrast_gt")) and pd.notna(r.get("e_full")) else np.nan, axis=1)
        sm.to_csv(smp, index=False); st = stats(sm, thr); st["contrast_floor_P5_CAL"] = floors; C.dump_json(C.p_("analysis", "d10_stats.json"), st)
        for k, v in st.items():
            if "|native64|DISC" in k and v.get("spearman_q_e"):
                print(f"[d10] {k}: ρ(q,e) {v['spearman_q_e']['rho']:+.3f} CI {v['spearman_q_e'].get('ci95')} · contrast-norm {v.get('spearman_q_e_contrastnorm', {}).get('rho')} · quadrants {v.get('quadrants')}", flush=True)
