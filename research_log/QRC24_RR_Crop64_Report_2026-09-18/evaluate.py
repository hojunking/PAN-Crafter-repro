"""Report-only paired RR evaluation. No training, Sheet import, network, or work_dir writes.

Capture completed QRC24 assets, then evaluate the SAME checkpoint in A_ON and
A_CROP64_MED. Primary: existing best_hqnr. G23 last@50K is a separate robustness panel.
Usage: python -B evaluate.py --capture; python -B evaluate.py --device cuda [--limit 1]
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')


def no_network(*_a, **_kw):
    raise RuntimeError('Network disabled: report-only evaluation, no Sheet access')


socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from scipy.io import loadmat
from kdv.eval_modes import CallCounter, forward_mode, state_hash
from kdv.teacher_assets import skeleton_from_cfg
from tools.metrics.eval_rr import sam, ergas
from tools.metrics.q2n import q2n
from tools.eval_dlpan import psnr_global, ssim_skimage, scc_dlpan
from tools.qrecon24_version_audit import audit_run

PROTOCOL = 'QRC24_RR_FULL_VS_CROP64_REPORT_v1_20260918'
MODES = ('A_ON', 'A_CROP64_MED')
METRICS = ('ergas', 'sam', 'psnr', 'ssim', 'scc', 'q8', 'rmse', 'cc')
SOURCE_FILES = ('pa/model.py', 'pa/aligner.py', 'pa/warp.py', 'kdv/eval_modes.py',
                'kdv/teacher_assets.py', 'feeders/feeder.py', 'model/pancrafter_paper.py',
                'tools/eval_dlpan.py', 'tools/metrics/eval_rr.py', 'tools/metrics/q2n.py')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jread(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return default


def jwrite(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)


def source_identity():
    files = list(SOURCE_FILES) + [str(Path(__file__).relative_to(ROOT))]
    return {p: sha(ROOT/p) for p in files}


def capture():
    path = OUT/'cohort.json'
    if path.exists():
        print('Existing frozen cohort:', path, flush=True)
        return
    assets = []; excluded = []; runs = []
    for cfgp in sorted((ROOT/'work_dir').glob('PAKD50_QRC24_*/meta/config.yaml')):
        wd = cfgp.parent.parent; run = wd.name
        cfg = yaml.safe_load(cfgp.read_text())
        profile = cfg['kdv']['qrc24']['profile']; last = jread(wd/'last_meta.json', {})
        if last.get('step') != 50000 or not (wd/'results/reduced_best_hqnr.mat').exists():
            excluded.append(dict(run=run, reason='not completed at capture', last_step=last.get('step')))
            continue
        audit = audit_run(run, write=False)
        if audit['verdict'] != 'single_definition_factor1':
            excluded.append(dict(run=run, reason=audit['verdict']))
            continue
        runs.append(run)
        for tag in (('best_hqnr', 'last') if profile == 'G23' else ('best_hqnr',)):
            ck = wd/tag/'model.safetensors'
            if not ck.exists():
                excluded.append(dict(run=run, checkpoint=tag, reason='missing checkpoint'))
                continue
            meta = jread(wd/f'{tag}_meta.json', {})
            snap = OUT/'assets'/run/tag
            snap.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, snap/'model.safetensors')
            shutil.copy2(cfgp, snap/'config.yaml')
            jwrite(snap/'checkpoint_meta.json', meta)
            jwrite(snap/'definition_audit.json', audit)
            selection = {}
            for p in (wd/'results').glob('qrecon24_target_selection*.json'):
                # Preserve legacy records verbatim; some contain nonstandard NaN.
                shutil.copy2(p, snap/p.name)
                selection[p.name] = dict(sha256=sha(p), source=str(p))
            jwrite(snap/'selection_records.json', selection)
            assets.append(dict(run=run, server=run.split('_')[2].lower(), profile=profile,
                               seed=int(cfg['seed']), checkpoint=tag, step=meta['step'],
                               panel=('existing_selected' if tag == 'best_hqnr' else 'G23_exact50K'),
                               snapshot=str(snap.relative_to(OUT)), checkpoint_sha256=sha(ck), config_sha256=sha(cfgp),
                               legacy_rr_mat=str(wd/'results/reduced_best_hqnr.mat') if tag == 'best_hqnr' else None,
                               selected_hqnr=meta.get('hqnr'), definition=audit['verdict']))
    assets.sort(key=lambda r: (r['profile'] != 'G23', r['checkpoint'] != 'best_hqnr', r['seed'], r['run']))
    for rel in SOURCE_FILES:
        p=OUT/'source'/rel; p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/rel, p)
    jwrite(path, dict(protocol=PROTOCOL, captured_at=datetime.datetime.now().astimezone().isoformat(),
                      head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      scope='All completed locally available factor1 QRC24 runs at capture; no remote assets available',
                      primary_selector='Existing best_hqnr checkpoint; never reselected per inference mode',
                      supplemental='G23 last@50000 only; kept separate from selected panel',
                      n_runs=len(runs), n_assets=len(assets), assets=assets, excluded=excluded,
                      missing_source_servers=['s2','s3','s4','s5'], sources=source_identity()))
    print(f'Captured {len(runs)} completed runs / {len(assets)} checkpoint assets; excluded {len(excluded)}', flush=True)


def metrics(sr, gt):
    sl = slice(20, -21)
    a = sr.astype(np.float64).transpose(1, 2, 0)[sl, sl]
    b = gt.astype(np.float64).transpose(1, 2, 0)[sl, sl]
    return dict(ergas=ergas(a,b), sam=sam(a,b), psnr=psnr_global(a,b,2047.),
                ssim=ssim_skimage(a,b,2047.), scc=scc_dlpan(a,b), q8=float(q2n(b,a,32,32)[0]),
                rmse=float(np.sqrt(np.mean((a-b)**2))), cc=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def summary(rows):
    return {k: dict(mean=float(np.mean([r[k] for r in rows])), sd=float(np.std([r[k] for r in rows],ddof=1))) for k in METRICS}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate(asset, args, data_cache):
    snap = OUT/asset['snapshot']; cfg = yaml.safe_load((snap/'config.yaml').read_text())
    assert sha(snap/'model.safetensors') == asset['checkpoint_sha256']
    assert sha(snap/'config.yaml') == asset['config_sha256']
    # Relocate recorded absolute dataset paths by their repository-relative suffix, when necessary.
    src = Path(cfg['test_reduced_feeder_args']['dataroot'])
    if not src.exists():
        src = ROOT/'data'/str(src).split('/data/',1)[1]
    lp = src.with_name(src.stem+'_pan.h5')
    key = str(src)
    if key not in data_cache:
        with h5py.File(src,'r') as f:
            d = {k:np.asarray(f[k]) for k in ('pan','ms','gt')}
        with h5py.File(lp,'r') as f:
            d['lpan'] = np.asarray(f['lpan'])
        assert d['pan'].shape == (20,1,256,256) and d['ms'].shape == (20,8,64,64)
        d['identity'] = dict(rr_h5_sha256=sha(src), lpan_h5_sha256=sha(lp), rr_h5=str(src), lpan_h5=str(lp))
        data_cache[key] = d
    data = data_cache[key]
    ident = dict(protocol=PROTOCOL, checkpoint_sha256=asset['checkpoint_sha256'], config_sha256=asset['config_sha256'],
                 data=data['identity'], sources=source_identity(), device=args.device, torch=torch.__version__,
                 dtype='float32', autocast=False, cudnn_benchmark=torch.backends.cudnn.benchmark,
                 cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, crop='20:-21', peak=2047.,
                 q8_block=32, sr_conversion='clip[-1,1] then float32 (y+1)/2*2047', full_frame_unet=True,
                 crop64_policy='16 nonoverlap crops, M upsample once, margin4 once, component median midpoint',
                 no_sheet_import=True, network_disabled=True)
    od = OUT/'results'/asset['run']/asset['checkpoint']; od.mkdir(parents=True,exist_ok=True)
    prev = jread(od/'result.json')
    if prev and prev.get('identity') == ident and prev.get('complete'):
        print('CACHED',asset['run'],asset['checkpoint'],flush=True); return prev
    mod, cls = cfg['model'].rsplit('.',1)
    Model = getattr(importlib.import_module(mod),cls)
    model, info = skeleton_from_cfg(cfg,Model)
    model.load_state_dict(load_file(str(snap/'model.safetensors'),device='cpu'),strict=True)
    dev=torch.device(args.device)
    if dev.type=='cuda':
        free,_=torch.cuda.mem_get_info(dev)
        if free < 2*1024**3: raise RuntimeError('Less than 2 GiB free GPU memory; do not compete with training')
        torch.cuda.reset_peak_memory_stats(dev)
    model=model.to(dev).eval(); before=state_hash(model)
    assert info['margin']==4 and model.aligner is not None
    rows={mode:[] for mode in MODES}; extras={mode:[] for mode in MODES}
    inference={mode:0. for mode in MODES}; calls={mode:dict(aligner_calls=0,warp_calls=0) for mode in MODES}
    comparisons=[]; legacy=None; legacy_diffs=[]; probes=[]; repeat_check={}
    if asset['legacy_rr_mat'] and Path(asset['legacy_rr_mat']).exists():
        legacy=loadmat(asset['legacy_rr_mat'])['sr']
        if legacy.shape[1] != 8:legacy=legacy.transpose(0,3,1,2)
        assert legacy.shape==(20,8,256,256)
    sr_saved={m:[] for m in MODES}
    started=time.time()
    with torch.no_grad():
        for i in range(20):
            inputs={k:(torch.from_numpy(data[k][i].astype(np.float32)).unsqueeze(0).to(dev)*2/2047.-1) for k in ('pan','ms','lpan')}
            pair={}
            for mode in MODES:
                delta_outputs=[]
                hook=model.aligner.register_forward_hook(lambda _m,_i,o:delta_outputs.append(o.detach().cpu().tolist()))
                sync(dev);t=time.perf_counter()
                with CallCounter(model) as counter:
                    y, ex=forward_mode(model,inputs['pan'],inputs['ms'],inputs['lpan'],mode,return_extra=True)
                sync(dev);inference[mode]+=time.perf_counter()-t;hook.remove()
                expected=1 if mode=='A_ON' else 16
                assert counter.aligner==expected and counter.warp==1, (mode,counter.as_dict())
                for k in calls[mode]:calls[mode][k]+=counter.as_dict()[k]
                if mode=='A_CROP64_MED':ex['crop_deltas']=[v[0] for v in delta_outputs]
                ex['scene']=i;extras[mode].append(ex)
                sr=((y.clip(-1.,1.).float().cpu().numpy()+1.)/2.*2047.)[0]
                pair[mode]=sr
                rows[mode].append(dict(scene=i,**metrics(sr,data['gt'][i])))
                if asset['profile']=='G23':sr_saved[mode].append(sr)
                if i==0:
                    yr,_=forward_mode(model,inputs['pan'],inputs['ms'],inputs['lpan'],mode,return_extra=True)
                    sr_repeat=((yr.clip(-1.,1.).float().cpu().numpy()+1.)/2.*2047.)[0]
                    repeat_check[mode]=dict(max_abs_dn=float(np.abs(sr-sr_repeat).max()),
                        abs_ergas_delta=abs(metrics(sr_repeat,data['gt'][i])['ergas']-rows[mode][-1]['ergas']))
            diff=pair['A_CROP64_MED']-pair['A_ON']
            d0=np.asarray(extras['A_ON'][-1]['delta']);dm=np.asarray(extras['A_CROP64_MED'][-1]['delta'])
            comparisons.append(dict(scene=i,**{f'delta_{k}':rows['A_CROP64_MED'][-1][k]-rows['A_ON'][-1][k] for k in METRICS},
                                    full_dy=float(d0[0]),full_dx=float(d0[1]),crop_dy=float(dm[0]),crop_dx=float(dm[1]),
                                    delta_shift_l2=float(np.linalg.norm(dm-d0)),max_abs_dn=float(np.abs(diff).max()),
                                    mean_abs_dn=float(np.abs(diff).mean())))
            if legacy is not None:
                legacy_diffs.append(dict(scene=i,max_abs_dn=float(np.abs(pair['A_ON']-legacy[i]).max()),
                                          ergas_delta=rows['A_ON'][-1]['ergas']-metrics(legacy[i],data['gt'][i])['ergas']))
            if (i+1)%5==0:print(f"  {asset['profile']} s{asset['seed']} {asset['checkpoint']} scene {i+1}/20 ({time.time()-started:.1f}s)",flush=True)
            if args.pause>0:time.sleep(args.pause)
    state_ok=state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert state_ok and model.sampler
    assert not any(k=='gspread' or k.startswith('gspread.') or 'gspread_upload' in k for k in sys.modules)
    sums={m:summary(rows[m]) for m in MODES}
    changes={k:sums['A_CROP64_MED'][k]['mean']-sums['A_ON'][k]['mean'] for k in METRICS}
    de=np.array([r['delta_ergas'] for r in comparisons]);rng=np.random.default_rng(20260918)
    ci=np.percentile(de[rng.integers(0,len(de),size=(10000,len(de)))].mean(1),[2.5,97.5]).tolist()
    result=dict(asset=asset,identity=ident,complete=True,state_unchanged=state_ok,
                evaluated_at=datetime.datetime.now().astimezone().isoformat(),seconds=time.time()-started,
                modes={m:dict(summary=sums[m],per_scene=rows[m],extras=extras[m],calls=calls[m],inference_seconds=inference[m]) for m in MODES},
                delta_crop_minus_on=changes,scene_comparisons=comparisons,
                ergas_improved_scenes=int((de<0).sum()),ergas_worsened_scenes=int((de>0).sum()),
                ergas_scene_bootstrap_ci95=ci,bootstrap_note='Descriptive resampling of the same 20 test images; no independent seed or sensor-population significance claim',
                mean_shift_difference=float(np.mean([r['delta_shift_l2'] for r in comparisons])),
                max_shift_difference=float(max(r['delta_shift_l2'] for r in comparisons)),
                repeated_scene0=repeat_check,legacy_on_comparison=legacy_diffs,
                gpu_peak_allocated_mib=float(torch.cuda.max_memory_allocated(dev)/1024**2) if dev.type=='cuda' else None)
    for mode in MODES:
        with (od/f'{mode}_per_scene.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[mode][0]));w.writeheader();w.writerows(rows[mode])
    with (od/'paired_deltas.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    if asset['profile']=='G23':
        np.savez_compressed(od/'sr_pair.npz',**{m:np.stack(sr_saved[m]) for m in MODES})
    jwrite(od/'result.json',result)
    print(f"DONE {asset['profile']} s{asset['seed']} {asset['checkpoint']}: ERGAS {sums['A_ON']['ergas']['mean']:.8f} -> {sums['A_CROP64_MED']['ergas']['mean']:.8f} (d={changes['ergas']:+.8f})",flush=True)
    del model
    if dev.type=='cuda':torch.cuda.empty_cache()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',action='store_true');ap.add_argument('--device',default='cpu')
    ap.add_argument('--limit',type=int);ap.add_argument('--profile');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--pause',type=float,default=.05)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    if args.capture:capture();return
    cohort=jread(OUT/'cohort.json');assert cohort,'Run --capture first'
    assets=[a for a in cohort['assets'] if not args.profile or a['profile']==args.profile]
    if args.limit:assets=assets[:args.limit]
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available(),'CUDA unavailable'
        torch.cuda.set_per_process_memory_fraction(.20)
    jwrite(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
                                      cuda=torch.version.cuda,device=args.device,cpu_threads=args.threads,
                                      gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                                      cuda_memory_limit_fraction=.20 if args.device.startswith('cuda') else None))
    cache={}
    for i,a in enumerate(assets):
        print(f"[{i+1}/{len(assets)}] {a['run']} {a['checkpoint']}",flush=True)
        try:evaluate(a,args,cache)
        except Exception as e:
            jwrite(OUT/'errors'/f"{a['run']}__{a['checkpoint']}.json",dict(asset=a,error=repr(e),at=datetime.datetime.now().isoformat()))
            raise
    print('FINISHED',len(assets),'checkpoint assets; no Sheet/network access',flush=True)


if __name__=='__main__':main()
