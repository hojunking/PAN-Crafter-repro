"""Offline cross-module selection/report checks; no actual checkpoint training."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from fh12.common import atomic_json,object_sha,read_json,sha256
from pcrepro import postrun
from pcrepro.common import run_dir
from pcrepro.data import NativeDataset,validate_manifest
from pcrepro.evaluation import evaluate_model
from pcrepro.plan import CAMPAIGN_ID,RECIPE_ID,RECIPE_SHA256,GRID,Case,build_config
from pcrepro.test_data import make_manifest
from pcrepro.test_evaluation import ToyModel,fake_wald


class SelectedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.case=Case('s3',1,'WV3')
        self.source=dict(files={'test':'a'*64},content_sha256='b'*64,git_release='offline-fixture')
        self.wd=run_dir(self.case,self.root)
        self.manifest=dict(schema='OFFLINE_SELECTION_FIXTURE')
        self.path=self.root/'manifest.json';atomic_json(self.path,self.manifest)
        self.cfg=build_config(self.case,str(self.path),object_sha(self.manifest),self.source)
        atomic_json(self.wd/'meta/config.resolved.yaml',self.cfg)
        self.context=dict(campaign_id=CAMPAIGN_ID,run_id=self.case.run_id,recipe_id=RECIPE_ID,
            recipe_sha256=RECIPE_SHA256,source_identity=self.source,data_sha=object_sha(self.manifest),
            config_sha=object_sha(self.cfg),model_seed=self.case.seed,num_bands=8,max_pixel=2047)
        self.rows=[dict(update=step,ergas=1. if step in (2000,3000) else 3.,model_state_hash=f'{step:064x}') for step in GRID]
        self.grid=dict(self.context,complete=True,test_metrics_used=False,expected_steps=list(GRID),records=self.rows)
        self.status=dict(self.context,training_complete=True,actual_updates=50000,scheduler_end=50000)
        self._write()
        self.identities={label:dict(self.context,update=step,model_state_hash=f'{step:064x}',
            model_sha256=('c' if label=='EXACT_50000' else 'd')*64,
            full_state=label=='EXACT_50000',val_ergas=1. if step==2000 else 3.)
            for label,step in (('EXACT_50000',50000),('RR_VAL_ERGAS_MIN',2000))}
        self.paths={label:self.wd/'checkpoints'/label for label in self.identities}
        self.fullstate=dict(update=50000,precision='fp32',scheduler={'last_epoch':50000},
            sampler={'completed_updates':50000},validation_records=self.rows,rng={'torch':'fixture'},
            optimizer={'state':{0:{'step':50000.}}})
        def loader(path,expected=None,**kwargs):
            identity=self.identities[Path(path).name]
            if expected and any(identity.get(k)!=v for k,v in expected.items()):
                raise ValueError('Fixture provenance differs')
            return torch.nn.Identity(),identity
        for patch in (mock.patch.object(postrun,'source_identity',return_value=self.source),
                mock.patch('pcrepro.checkpoint.selection_checkpoints',return_value=self.paths),
                mock.patch('pcrepro.checkpoint.load_model_checkpoint',side_effect=loader),
                mock.patch('torch.load',side_effect=lambda *a,**k:self.fullstate)):
            patch.start();self.addCleanup(patch.stop)

    def _write(self):
        atomic_json(self.wd/'official/validation_grid.json',self.grid)
        atomic_json(self.wd/'meta/training_status.json',self.status)

    def test_fixed_fifty_point_grid_earliest_tie_and_exact_primary(self):
        result=postrun.selected_evidence(self.case,self.root)
        self.assertEqual(result['RR_VAL_ERGAS_MIN']['identity']['update'],2000)
        self.assertEqual(result['EXACT_50000']['identity']['update'],50000)

    def test_later_equal_validation_candidate_is_rejected(self):
        self.identities['RR_VAL_ERGAS_MIN'].update(update=3000,model_state_hash=f'{3000:064x}')
        with self.assertRaisesRegex(ValueError,'selection'):postrun.selected_evidence(self.case,self.root)

    def test_missing_grid_or_test_selection_is_rejected(self):
        self.grid['records']=self.rows[:-1];self._write()
        with self.assertRaisesRegex(ValueError,'grid'):postrun.selected_evidence(self.case,self.root)
        self.grid['records']=self.rows;self.grid['test_metrics_used']=True;self._write()
        with self.assertRaisesRegex(ValueError,'validation-only'):postrun.selected_evidence(self.case,self.root)

    def test_scheduler_or_optimizer_endpoint_is_not_merely_a_label(self):
        self.fullstate['optimizer']['state'][0]['step']=49999
        with self.assertRaisesRegex(ValueError,'endpoint'):postrun.selected_evidence(self.case,self.root)

    def test_empty_optimizer_state_is_not_an_exact_training_proof(self):
        self.fullstate['optimizer']['state']={}
        with self.assertRaises(ValueError):postrun.selected_evidence(self.case,self.root)

    def test_wv2_uses_same_server_cycle_wv3_and_no_fallback(self):
        wv2=Case('s3',1,'WV2')
        self.assertEqual(postrun.selected_evidence(wv2,self.root)['EXACT_50000']['identity']['run_id'],self.case.run_id)
        other=build_config(Case('s4',1,'WV3'),str(self.path),object_sha(self.manifest),self.source)
        atomic_json(self.wd/'meta/config.resolved.yaml',other)
        with self.assertRaisesRegex(ValueError,'Source config'):postrun.selected_evidence(wv2,self.root)


class ReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.case=Case('s3',0,'GF2');self.wd=run_dir(self.case,self.root)
        self.source=dict(files={'test':'a'*64},content_sha256='b'*64,git_release='offline-fixture')
        self.manifest=validate_manifest(make_manifest(self.root,'GF2',rr_count=2),self.root)
        path=self.root/'manifest.json';atomic_json(path,self.manifest)
        self.cfg=build_config(self.case,str(path),object_sha(self.manifest),self.source)
        atomic_json(self.wd/'meta/config.resolved.yaml',self.cfg)
        self.digest='c'*64;self.evaldir=self.wd/'official/evaluations'/self.digest
        self.wald=fake_wald()
        datasets={s:NativeDataset(self.manifest,s,self.root) for s in ('rr','fr')}
        # Real metric primitives on native64 synthetic fixtures, but test interp23tap.
        self.metrics=evaluate_model(ToyModel(),datasets,'cpu',self.digest,self.evaldir,wald=self.wald)
        self.selected={label:dict(path=self.root/label,
            identity=dict(model_sha256=self.digest,update=50000),val_ergas=1.) for label in postrun.LABELS}
        selections={label:dict(update=50000,checkpoint_sha256=self.digest,
            checkpoint_identity=self.selected[label]['identity'],val_ergas=1.,data_sha256=object_sha(self.manifest),
            evaluation_path=str((self.evaldir/'metrics.json').relative_to(self.wd)),
            evaluation_manifest_sha256=sha256(self.evaldir/'metrics.json'),
            rr=deepcopy(self.metrics['rr']),fr=deepcopy(self.metrics['fr']),
            alias_of='EXACT_50000' if label=='RR_VAL_ERGAS_MIN' else None) for label in postrun.LABELS}
        self.summary=dict(campaign_id=CAMPAIGN_ID,recipe_id=RECIPE_ID,run_id=self.case.run_id,
            case=self.case.to_dict(),complete=True,actual_updates=50000,source_identity=self.source,
            data_sha256=object_sha(self.manifest),config_sha256=object_sha(self.cfg),
            recipe_manifest_sha256=RECIPE_SHA256,selections=selections,paper_identity_status='UNVERIFIED')
        self._save()
        for patch in (mock.patch.object(postrun,'source_identity',return_value=self.source),
                mock.patch.object(postrun,'selected_evidence',return_value=self.selected),
                mock.patch('tools.metrics.eval_fr.load_dlpan',return_value=self.wald)):
            patch.start();self.addCleanup(patch.stop)

    def _save(self):
        atomic_json(self.evaldir/'metrics.json',self.metrics)
        for record in self.summary['selections'].values():
            record.update(rr=deepcopy(self.metrics['rr']),fr=deepcopy(self.metrics['fr']),
                evaluation_manifest_sha256=sha256(self.evaldir/'metrics.json'))
        self.summary['evaluation_manifest_sha256']=object_sha({k:v['evaluation_manifest_sha256']
            for k,v in self.summary['selections'].items()})
        atomic_json(self.wd/'official/summary.json',self.summary)

    def test_valid_same_checkpoint_alias_one_metric_source(self):
        self.assertEqual(postrun.verify_summary_for_upload(self.case.run_id,self.root),self.summary)

    def test_alias_cannot_combine_other_checkpoint_metrics(self):
        self.summary['selections']['RR_VAL_ERGAS_MIN']['rr']['ergas']+=1
        atomic_json(self.wd/'official/summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'aliases'):postrun.verify_summary_for_upload(self.case.run_id,self.root)

    def test_declared_population_cannot_be_silently_shortened(self):
        self.metrics['rr']['n_scenes']=1;self._save()
        with self.assertRaisesRegex(ValueError,'population'):postrun.verify_summary_for_upload(self.case.run_id,self.root)

    def test_cursor_and_evaluator_must_be_complete_and_identical(self):
        path=self.evaldir/'evaluation_cursor.json';cursor=read_json(path);cursor['complete']=False;atomic_json(path,cursor)
        with self.assertRaisesRegex(ValueError,'cursor'):postrun.verify_summary_for_upload(self.case.run_id,self.root)
        cursor['complete']=True;atomic_json(path,cursor)
        self.metrics['metadata']['evaluator']['sha256']='d'*64;self._save()
        with self.assertRaisesRegex(ValueError,'provenance'):postrun.verify_summary_for_upload(self.case.run_id,self.root)

    def test_mean_hqnr_cannot_replace_scene_hqnr_products(self):
        path=self.evaldir/'evaluation_cursor.json';cursor=read_json(path)
        for row in self.metrics['fr']['per_scene']:
            row['hqnr']=.8
            row['row_sha256']=object_sha({k:v for k,v in row.items() if k!='row_sha256'})
        cursor['rows']['fr']=deepcopy(self.metrics['fr']['per_scene']);atomic_json(path,cursor)
        self.metrics['fr']['hqnr']=.8;self._save()
        with self.assertRaises(ValueError):postrun.verify_summary_for_upload(self.case.run_id,self.root)

    def test_unknown_paper_set_cannot_be_relabelled_verified(self):
        self.metrics['metadata']['paper_comparable']=True
        self.summary['paper_identity_status']='VERIFIED';self._save()
        with self.assertRaises(ValueError):postrun.verify_summary_for_upload(self.case.run_id,self.root)
