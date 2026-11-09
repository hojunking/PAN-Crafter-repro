"""Synthetic saved-output fixtures; none of these values are scientific results."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from panda_sensor_visuals.common import read, sha, write
from panda_gf2_sharpness import select as S
from panda_gf2_sharpness.selection_metrics import classify, measure_saved, pareto


def fixture(folder, *, step=47000, smooth=1.0, rr=True):
    folder.mkdir(parents=True)
    cfg = dict(run_id=S.QUAL_RUN, dataset='GF2', server='s3', stage2_spec={'student_width':112})
    identity = dict(run_id=S.QUAL_RUN, dataset='GF2', server='s3', completed_update=step)
    checkpoint_sha = hashlib.sha256(f'SYNTHETIC-FIXTURE-{step}'.encode()).hexdigest()
    ev = dict(state_before=identity, state_after=identity, rr_state=identity, fr_state=identity)
    manifest_scenes = []
    for split, size in [('fr',512), ('rr',256)]:
        y,x = np.mgrid[:size,:size]
        lum = .4 + .13*np.sin(x/13) + .10*np.cos(y/17) + .17*((x>size//3)&(x<size*2//3))
        pan = np.repeat(np.rint(np.clip(lum,0,1)*255).astype(np.uint8)[...,None],3,axis=2)
        truth = np.stack([100+700*lum,90+650*lum,80+600*lum,150+500*lum]).astype(np.float32)
        raw = np.stack([gaussian_filter(b,smooth) for b in truth]).astype(np.float32)
        raw_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        ev[split] = dict(metrics=(dict(HQNR=.965,D_s=.02,D_lambda=.015,JQM=.98) if split=='fr'
                         else dict(ERGAS=.52,PSNR=45.1,SAM=.7,SCC=.99,SSIM=.99,Q4=.98,RMSE=2.,CC=.99)),
                         per_scene=[dict(scene_id=f'GF2:{split}:{i}',raw_prediction_sha256=raw_sha) for i in range(20)])
        if split=='rr' and not rr:
            continue
        for i in range(20):
            stem=f'scene_{i:02d}';prefix=folder/split.upper()
            for name in ('prediction_npy','PAN','render_metadata'):
                (prefix/name).mkdir(exist_ok=True,parents=True)
            path=prefix/'prediction_npy'/(stem+'.npy')
            if i:
                os.link(prefix/'prediction_npy/scene_00.npy',path)
            else:
                np.save(path,raw)
            Image.fromarray(pan).save(prefix/'PAN'/(stem+'.png'))
            write(prefix/'render_metadata'/(stem+'.json'),dict(scene=i,split=split,rgb_indices=[2,1,0],
                  peak_DN=1023,shared_reference='stored LMS, not GT' if split=='fr' else 'GT',
                  rgb_bounds_dn=[[0,1023]]*3,pan_bounds_dn=[0,1023]))
            if split=='rr':
                (prefix/'references_npz').mkdir(exist_ok=True)
                np.savez_compressed(prefix/'references_npz'/(stem+'.npz'),gt=truth)
            manifest_scenes.append(dict(split=split,scene=i,prediction_npy=str(path.relative_to(folder)),
                  raw_prediction_sha256=raw_sha,npy_sha256=sha(path),shape=list(raw.shape)))
    write(folder/'provenance/config.json',cfg)
    write(folder/f'provenance/evaluation_{step:07d}.json',ev)
    write(folder/f'provenance/step_{step:07d}.json',dict(config=cfg,artifact=dict(identity=identity,
          completed_update=step,file_sha256=checkpoint_sha)))
    write(folder/'manifest.json',dict(sensor='GF2',layout='CHW',bands=['Blue','Green','Red','NIR'],
          original_peak_DN=1023,step=step,run_id=S.QUAL_RUN,checkpoint_sha256=checkpoint_sha,
          checkpoint_path='/SYNTHETIC_NO_WEIGHT_REQUIRED.pt',scenes=manifest_scenes,
          export_binding=dict(checkpoint_sha256=checkpoint_sha,
              evaluation_sha256=sha(folder/f'provenance/evaluation_{step:07d}.json'))))
    refresh_ledger(folder)
    return folder


def refresh_ledger(folder):
    write(folder/'files_sha256.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*')
                                    if p.is_file() and p.name!='files_sha256.json'})


def score_row(cid, edge, grain, *, legacy_edge=1.2, legacy_grain=.009, ergas=.52):
    return dict(candidate_id=cid,shared_edge_pm2=edge,raw_edge_pm2=edge,
                shared_flat_grain=grain,raw_flat_grain=grain,
                legacy_edge_pm2=legacy_edge,legacy_flat_grain=legacy_grain,HQNR=.965,ERGAS=ergas,PSNR=45.1)


class SelectionTests(unittest.TestCase):
    def test_absent_qualitative_reference_uses_fixed_quantitative_reference_only(self):
        quant=dict(manifest=dict(run_id=S.QUANT_RUN,step=47000))
        arbitrary=dict(manifest=dict(run_id='arbitrary',step=1000))
        candidate,role=S.choose_baseline([arbitrary,quant])
        self.assertIs(candidate,quant)
        self.assertIn('FALLBACK',role)
        with self.assertRaisesRegex(ValueError,'No verified'):
            S.choose_baseline([arbitrary])

    def test_pareto_does_not_reward_noisy_edges_as_unique_winner(self):
        rows=[dict(edge=1,grain=1),dict(edge=2,grain=2),dict(edge=.5,grain=1.5)]
        self.assertEqual(pareto(rows,[('edge',True),('grain',False)]),[0,1])

    def test_gates_use_paired_scenes_raw_noise_and_unrounded_targets(self):
        rows=[score_row('base',1,.01,legacy_edge=1.02),score_row('clean',1.2,.009),
              score_row('noisy',1.8,.02),score_row('rounding',1.2,.009,ergas=.52200001)]
        scenes=[dict(candidate_id=r['candidate_id'],scene=i,shared_edge_pm2=r['shared_edge_pm2'],
                     raw_edge_pm2=r['raw_edge_pm2']) for r in rows for i in range(20)]
        refs={'U-Know':{'edge_pm2':1.15},'Ours':{'flat_grain':.00943},'PAN-Crafter (s5)':{'edge_pm2':1.228}}
        lists=classify(rows,scenes,'base',refs)
        self.assertIn('clean',lists['visual'])
        self.assertNotIn('noisy',lists['visual'])
        self.assertNotIn('rounding',lists['quantitative'])
        self.assertEqual(rows[1]['improvement_scenes'],20)
        self.assertEqual(rows[0]['improvement_scenes'],0)
        self.assertEqual(rows[1]['human_review'],'PENDING')

    def test_no_passing_model_is_not_promoted_from_frontier(self):
        rows=[score_row('base',1,.01,legacy_edge=1.02),score_row('bad',1.3,.02)]
        scenes=[dict(candidate_id=r['candidate_id'],scene=i,shared_edge_pm2=r['shared_edge_pm2'],
                     raw_edge_pm2=r['raw_edge_pm2']) for r in rows for i in range(20)]
        refs={'U-Know':{'edge_pm2':1.15},'Ours':{'flat_grain':.00943},'PAN-Crafter (s5)':{'edge_pm2':1.228}}
        lists=classify(rows,scenes,'base',refs)
        self.assertFalse(lists['visual'])
        self.assertFalse(lists['compromise'])
        self.assertTrue(lists['provisional_frontier'])

    def test_raw_and_shared_metrics_retain_gain_hidden_by_own_stretch(self):
        y,x=np.mgrid[:512,:512]
        lum=.2+.15*np.sin(x/10)+.1*np.cos(y/17)
        raw=np.stack([150+500*lum]*4).astype(np.float32)
        pan=np.repeat(np.rint((lum+.3)*255).astype(np.uint8)[...,None],3,axis=2)
        a=measure_saved(raw,pan,[[0,1023]]*3,1023)
        b=measure_saved(raw*1.1,pan,[[0,1023]]*3,1023)
        self.assertAlmostEqual(a['legacy_edge_pm2'],b['legacy_edge_pm2'],delta=.002)
        self.assertAlmostEqual(b['raw_edge_pm2']/a['raw_edge_pm2'],1.1,places=5)
        self.assertAlmostEqual(b['shared_flat_grain']/a['shared_flat_grain'],1.1,places=5)

    def test_export_integrity_and_reference_bounds_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=fixture(Path(tmp)/'export',rr=False)
            item=S.inspect(folder)
            context=S.fr_context(item)
            meta=folder/'FR/render_metadata/scene_00.json'
            changed=read(meta);changed['rgb_bounds_dn'][0][1]=1022;write(meta,changed);refresh_ledger(folder)
            with self.assertRaisesRegex(ValueError,'bounds differ'):
                S.measure_candidate(S.inspect(folder),context)
            # Updating the ledger cannot disguise a raw output differing from the original evaluation.
            p=folder/'FR/prediction_npy/scene_00.npy';arr=np.load(p);arr[0,0,0]+=1
            p.unlink();np.save(p,arr);refresh_ledger(folder)
            with self.assertRaisesRegex(ValueError,'modified NPY'):
                S.load_prediction(S.inspect(folder),'fr',0)
            # A wrong scene order is rejected before metrics are averaged.
            path=folder/'provenance/evaluation_0047000.json';ev=read(path);ev['fr']['per_scene'].reverse();write(path,ev)
            manifest=read(folder/'manifest.json');manifest['export_binding']['evaluation_sha256']=sha(path);write(folder/'manifest.json',manifest)
            refresh_ledger(folder)
            with self.assertRaisesRegex(ValueError,'reordered'):
                S.inspect(folder)

    def test_end_to_end_saved_outputs_only_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exports=root/'exports'
            fixture(exports/'baseline',rr=True)
            fixture(exports/'early',step=1000,smooth=.5,rr=False)
            before={str(p.relative_to(exports)):sha(p) for p in exports.rglob('*') if p.is_file()}
            roi=root/'rois.json';write(roi,[dict(scene=0,x=32,y=32,size=96,label='fixture')])
            args=S.parser().parse_args(['--exports',str(exports),'--output',str(root/'selection'),'--rois',str(roi)])
            with patch.object(S,'verify_release',return_value={'test_fixture':True}):
                result=S.run(args)
            self.assertEqual(read(result/'completion.json')['verified'],2)
            self.assertEqual(read(result/'shortlist.json')['final_selection'],'PENDING_HUMAN_REVIEW')
            self.assertEqual(len(list((result/'FR_all20').glob('*.png'))),20)
            self.assertEqual(len(list((result/'RR_all20').glob('*.png'))),20)
            self.assertTrue(any(r['status']=='RR_NPY_NOT_EXPORTED' for r in read(result/'RR_availability.json')))
            self.assertEqual(before,{str(p.relative_to(exports)):sha(p) for p in exports.rglob('*') if p.is_file()})
            self.assertFalse((result/'INCOMPLETE.json').exists())
            self.assertTrue(all(r['human_review']=='PENDING' for r in read(result/'ranking.json')['rows']))
            # Explicitly preserve only synthetic screenshots when a test reviewer requests them.
            if os.environ.get('GF2_SELECTOR_TEST_ARTIFACTS'):
                out=Path(os.environ['GF2_SELECTOR_TEST_ARTIFACTS']);out.mkdir(parents=True,exist_ok=True)
                for name in ('FR_blind.png','report.md','completion.json'):
                    shutil.copy2(result/name,out/name)
            with patch.object(S,'verify_release',return_value={'test_fixture':True}):
                with self.assertRaises(FileExistsError):
                    S.run(args)


if __name__=='__main__':
    unittest.main()
