"""Export exact official four-band predictions, references, metadata and optional ZIP."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .common import ROOT, read, write, sha, checkpoint_evidence
from .metrics import rgb_display, pan_display, measure, PROTOCOL


def save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def reference_display(cube, bounds):
    lo, hi = np.asarray(bounds).T
    x = np.asarray(cube, np.float64)[[2, 1, 0]].transpose(1, 2, 0)
    return np.rint(np.clip((x-lo)/np.maximum(hi-lo, 1e-12), 0, 1)*255).astype(np.uint8)


def package_zip(out):
    target = out.with_suffix('.zip')
    if target.exists():
        receipt = read(Path(str(target)+'.sha256.json'))
        if sha(target) != receipt['sha256']:
            raise ValueError('Existing archive checksum differs')
        return target
    partial = Path(str(target)+'.partial')
    if partial.exists():
        raise FileExistsError('Incomplete archive retained: ' + str(partial))
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    write(out/'files_sha256.json', {str(p.relative_to(out)):sha(p) for p in paths if p.name!='files_sha256.json'})
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    with zipfile.ZipFile(partial, 'x', allowZip64=True) as archive:
        for path in paths:
            archive.write(path, str(Path(out.name)/path.relative_to(out)),
                compress_type=zipfile.ZIP_STORED if path.suffix=='.png' else zipfile.ZIP_DEFLATED,
                compresslevel=6)
    manifest = read(out/'manifest.json')
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        for scene in manifest['scenes']:
            arr = np.load(io.BytesIO(archive.read(str(Path(out.name)/scene['prediction_npy']))), allow_pickle=False)
            if (arr.dtype != np.float32 or list(arr.shape) != scene['shape'] or
                    hashlib.sha256(arr.tobytes()).hexdigest() != scene['raw_prediction_sha256']):
                raise ValueError('Archived prediction differs from original evaluation')
    partial.replace(target)
    write(Path(str(target)+'.sha256.json'), dict(sha256=sha(target), bytes=target.stat().st_size,
        files=len(paths), verified_predictions=len(manifest['scenes']), ZIP_CRC_verified=True))
    return target


@torch.no_grad()
def export(run, step, out, *, splits=('rr','fr'), device='cuda', references=True, make_zip=False):
    from panda_stage2_condq.training import configure_precision
    from panda_method.edge_data import EdgeDataSources
    from panda_sensor_refine.plan import lms_overrides
    from .model import build
    run, out = Path(run).resolve(), Path(out).resolve()
    cfg, ev, ck, side = checkpoint_evidence(run, step)
    binding = dict(checkpoint_sha256=sha(ck), evaluation_sha256=sha(run/f'evaluation_{step:07d}.json'),
                   splits=list(splits), references=references, schema='SENSOR_VISUAL_EXPORT_V1')
    if out.exists():
        if not (out/'manifest.json').exists() or read(out/'manifest.json').get('export_binding') != binding:
            raise ValueError('Existing output is incomplete or belongs to another export; preserve it and use a new output path')
        for path, expected in read(out/'files_sha256.json').items():
            if sha(out/path) != expected: raise ValueError('Previously exported file changed: '+path)
        return package_zip(out) if make_zip else out
    configure_precision(); torch.set_num_threads(2)
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.20)
    model = build(cfg, ck, step).to(device)
    sensor = cfg['dataset']
    source = EdgeDataSources(ROOT/'ablr2/sensor_sources.json', root=ROOT, sensor=sensor,
                             lms_overrides=lms_overrides(sensor))
    out.mkdir(parents=True)
    write(out/'INCOMPLETE.json', binding)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    scenes, metrics, visual = [], [], []
    for split in splits:
        folder=out/split.upper()
        for i, expected in enumerate(ev[split]['per_scene']):
            pred=model(source.batch(split,[i],device=device))
            arr=((pred['y'][0].float()+1)*(source.peak/2)).cpu().numpy()
            digest=hashlib.sha256(arr.tobytes()).hexdigest()
            size=256 if split=='rr' else 512
            if (arr.dtype!=np.float32 or arr.shape!=(4,size,size) or not np.isfinite(arr).all()
                    or digest!=expected['raw_prediction_sha256']):
                raise ValueError(f'{split} scene {i}: output differs from official evaluation')
            name=f'scene_{i:02d}'
            npy=folder/'prediction_npy'/(name+'.npy');npy.parent.mkdir(parents=True,exist_ok=True);np.save(npy,arr)
            raw={k:v[0] for k,v in source.raw(split,[i]).items()}
            rgb=rgb_display(arr,source.peak);pan=pan_display(raw['pan'])
            save_image(folder/'prediction'/(name+'.png'),rgb)
            save_image(folder/'PAN'/(name+'.png'),pan)
            save_image(folder/'prediction_fixed_DN'/(name+'.png'),
                np.rint(np.clip(arr[[2,1,0]].transpose(1,2,0)/source.peak,0,1)*255).astype(np.uint8))
            ref=raw['gt'] if split=='rr' else raw['lms']
            bounds=[np.percentile(ref[b],[1,99]).tolist() for b in [2,1,0]]
            shared=reference_display(arr,bounds)
            save_image(folder/'prediction_shared_reference'/(name+'.png'),shared)
            panels=[('PAN',pan),('Stored LMS',reference_display(raw['lms'],bounds)),('Prediction',shared)]
            if references:
                for k,label in [('lms','LMS'),('ms','LRMS'),('gt','GT')]:
                    if k in raw:save_image(folder/label/(name+'.png'),reference_display(raw[k],bounds))
                refs=folder/'references_npz'/(name+'.npz');refs.parent.mkdir(exist_ok=True)
                np.savez_compressed(refs,**raw)
                if split=='rr':panels.append(('GT',reference_display(raw['gt'],bounds)))
            canvas=Image.new('RGB',(len(panels)*size,size+28),'white');draw=ImageDraw.Draw(canvas)
            for c,(label,img) in enumerate(panels):
                draw.text((c*size+6,5),label,font=font,fill='black');canvas.paste(Image.fromarray(img),(c*size,28))
            save_image(folder/'comparison'/(name+'.png'),np.asarray(canvas))
            write(folder/'render_metadata'/(name+'.json'),dict(scene=i,split=split,rgb_indices=[2,1,0],
                prediction_display='own sorted 1/99% stretch',shared_reference='GT' if split=='rr' else 'stored LMS, not GT',
                rgb_bounds_dn=bounds,pan_bounds_dn=np.percentile(raw['pan'],[1,99]).tolist(),peak_DN=source.peak))
            if split=='fr':visual.append(dict(scene=i,**measure(rgb,pan)))
            scenes.append(dict(split=split,scene=i,prediction_npy=str(npy.relative_to(out)),shape=list(arr.shape),
                dtype='float32',raw_prediction_sha256=digest,npy_sha256=sha(npy),matches_official_prediction=True))
            metrics.append(dict(split=split,scene=i,**expected['metrics']))
            print(split,i,'exact official SHA match',flush=True)
        canvas=Image.new('RGB',(5*256,4*280),'white');draw=ImageDraw.Draw(canvas)
        for i in range(20):
            with Image.open(folder/'prediction'/f'scene_{i:02d}.png') as image:
                thumb=image.resize((256,256),Image.Resampling.LANCZOS)
            x,y=(i%5)*256,(i//5)*280;draw.text((x+4,y+3),f'{split.upper()} {i:02d}',font=font,fill='black');canvas.paste(thumb,(x,y+24))
        canvas.save(folder/'overview.png')
    model.cpu();del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    prov=out/'provenance';prov.mkdir()
    for path in [run/'config.json',run/f'evaluation_{step:07d}.json',ck.with_suffix('.json')]:shutil.copy2(path,prov/path.name)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,prov/path.name)
    write(out/'metrics_summary.json',{s.upper():ev[s]['metrics'] for s in splits})
    if visual:
        write(out/'FR_sharpness.json',dict(protocol=PROTOCOL,mean={k:float(np.mean([r[k] for r in visual])) for k in visual[0] if k!='scene'},per_scene=visual))
    fields=['split','scene']+sorted({k for r in metrics for k in r}-{'split','scene'})
    with (out/'metrics_per_scene.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    write(out/'manifest.json',dict(export_binding=binding,run_id=cfg['run_id'],sensor=sensor,step=step,
        checkpoint_path=str(ck),checkpoint_sha256=sha(ck),original_run_directory=str(run),
        bands=['Blue','Green','Red','NIR'],layout='CHW',units='unclipped float32 DN',
        original_peak_DN=source.peak,all_predictions_match_original_evaluation=True,
        RR_FR_scene_numbers_are_independent=True,scenes=scenes))
    (out/'README.md').write_text(f'''# {sensor} {cfg['run_id']} — {step:,} steps

RR/FR 원본 예측 및 표시 이미지입니다. NPY는 **4밴드 CHW float32, clipping하지 않은 DN**이며
밴드 순서는 Blue, Green, Red, NIR입니다. 모든 예측은 원래 평가 당시 raw SHA-256과 일치합니다.

- `RR/FR/prediction_npy/scene_XX.npy`: 원본 예측; RR (4,256,256), FR (4,512,512).
- `prediction/`: 선명도 분석과 같은 각 출력별 RGB [2,1,0] 정렬 1–99% 대비 표시, gamma1.
- `prediction_shared_reference/`, `comparison/`: RR GT / FR stored LMS 범위를 공통 적용한 비교.
- `prediction_fixed_DN/`: 동일 DN 0–{source.peak:g} 범위. 대비 변환은 NPY에 적용되지 않습니다.
- `references_npz/`: 요청된 경우 PAN, MS, LMS, RR GT 원본 참조. FR의 LMS는 GT가 아닙니다.
- `PAN/`, `LMS/`, `LRMS/`, `GT/`: 표시용 참조 PNG. RR와 FR scene 번호의 지리적 대응은 가정하지 않습니다.
- `metrics_summary.json`, `FR_sharpness.json`, `manifest.json`, `provenance/`: 지표·정의·모델 경로·원본 근거.

모델 weight는 ZIP에 포함하지 않습니다. 학습·Sheet·원본 weight 변경은 없습니다.

```python
import numpy as np
x = np.load('FR/prediction_npy/scene_00.npy')
x_hwc = x.transpose(1, 2, 0)
```
''',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Prediction export</title>'+
        '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}</style><h1>'+cfg['run_id']+'</h1>'+
        ''.join(f'<h2>{s.upper()}</h2><img src="{s.upper()}/overview.png">'+
            ''.join(f'<p>Scene {i:02d}: <a href="{s.upper()}/prediction/scene_{i:02d}.png">RGB</a> · '
                    f'<a href="{s.upper()}/prediction_npy/scene_{i:02d}.npy">NPY</a> · '
                    f'<a href="{s.upper()}/comparison/scene_{i:02d}.png">Compare</a></p>' for i in range(20)) for s in splits))
    (out/'INCOMPLETE.json').unlink()
    write(out/'files_sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return package_zip(out) if make_zip else out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--splits',nargs='+',choices=['rr','fr'],default=['rr','fr'])
    parser.add_argument('--no-references',action='store_true');parser.add_argument('--zip',action='store_true')
    args=parser.parse_args()
    print(export(args.run_dir,args.step,args.output,splits=tuple(args.splits),device=args.device,
                 references=not args.no_references,make_zip=args.zip))


if __name__=='__main__':main()
