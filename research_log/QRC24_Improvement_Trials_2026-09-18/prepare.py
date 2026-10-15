from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
P=Path(__file__).resolve().parent;R=P.parents[1]
if (P/'manifest.json').exists():print('Already captured');raise SystemExit
h={}
for folder in ('model','pa','kdv','feeders','tools'):
 for src in (R/folder).rglob('*.py'):
  rel=src.relative_to(R);dst=P/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);h[str(rel)]=hashlib.sha256(dst.read_bytes()).hexdigest()
c=json.loads((R/'research_log/QRC24_RR_Crop64_Report_2026-09-18/cohort.json').read_text())
manifest=dict(at=datetime.datetime.now().astimezone().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),source_hashes=h,
 assets=[a for a in c['assets'] if a['profile']=='G23'],seeds=[1234,3407,41001],network_disabled=True,
 protocol=dict(cross=['AbUb','AlUl','AbUl','AlUb'],interpolation=[.25,.5,.75],bias_axis_px=[-.1,0,.1],
 validation='all1080 native64 center32; select parameters before RR/FR scoring',rr='20 images crop20:-21',fr='mat20 all20 raw_original',
 finetune=dict(seed=1234,updates=1000,batch_size=8,U_lr=1e-5,A_lr=3e-7,optimizer='AdamW reset betas(.9,.999) eps1e-8 weight_decay.01',scheduler='constant',
 arms=['bicubic_joint','lms_Aref_joint','bicubic_Afreeze'],same_order_seed=20260918,teacher='frozen T0',note='small matched pilot, not exact resume or fresh50K')))
(P/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print('Frozen',len(h),'source files')
