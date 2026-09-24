"""Safe-cutover tests using mock inventories and temporary preserved artifacts."""
from contextlib import ExitStack
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from maina_hqnr import handoff as h
from maina_hqnr.common import atomic_json,read_json,object_sha,sha256,camp,RuntimePaused
from fh12.model import state_hash


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=self.root/'work_dir/g23sens/s4'
        self.old.mkdir(parents=True);self.gpu='GPU-fixture'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.inventory=self.stack.enter_context(patch.object(h,'inventory',return_value=[]))
        self.docker=self.stack.enter_context(patch.object(h,'docker_inventory',return_value=[]))
        self.idle=self.stack.enter_context(patch.object(h,'ensure_gpu_idle'))
        self.support=self.stack.enter_context(patch.object(h,'_runner_support',return_value={'stop':True,'safe_now':True}))

    def local(self,server='s4',script='g23sens_runner.py'):
        return dict(server=server,script=script,script_path='/fixture/tools/'+script,pid=1234)

    def request(self):return h.request_boundary(self.root,'s4',self.gpu,activated=True)

    def test_authority_and_foreign_servers(self):
        with self.assertRaises(PermissionError):h.request_boundary(self.root,'s4',self.gpu)
        with self.assertRaises(ValueError):h.request_boundary(self.root,'s1',self.gpu,activated=True)
        self.inventory.return_value=[self.local('s1'),self.local('s4')]
        request=self.request()
        self.assertEqual(request['policy'],'SAFE_UPDATE_BOUNDARY')
        self.assertEqual(len(request['protected_other_lane_processes']),1)
        self.assertTrue((self.old/'STOP_NOW_SAFE').is_file())
        self.assertFalse((self.root/'work_dir/g23sens/s1').exists())
        self.assertFalse(request['signals_sent'])

    def test_unknown_watchdog_never_signaled(self):
        self.inventory.return_value=[self.local(None,'_watchdog.sh')]
        with self.assertRaises(RuntimePaused):self.request()
        self.assertFalse((self.old/'STOP_AFTER_RUN').exists())

    def test_wrong_docker_mount_or_restart_pauses(self):
        base=dict(id='fixture',mounts=[dict(source='/wrong/work_dir')],restart_policy='no',labels={})
        self.docker.return_value=[base]
        with self.assertRaises(RuntimePaused):self.request()
        base.update(mounts=[dict(source=str(self.root/'work_dir'))],restart_policy='always')
        with self.assertRaises(RuntimePaused):self.request()
        self.assertFalse((self.old/'STOP_AFTER_RUN').exists())

    def test_unsupported_safe_now_uses_run_boundary_only(self):
        self.inventory.return_value=[self.local()]
        self.support.return_value={'stop':True,'safe_now':False}
        request=self.request()
        self.assertEqual(request['policy'],'CURRENT_RUN_BOUNDARY')
        self.assertTrue((self.old/'STOP_AFTER_RUN').is_file())
        self.assertFalse((self.old/'STOP_NOW_SAFE').exists())

    def test_missing_stop_support_fails_closed(self):
        self.inventory.return_value=[self.local()]
        self.support.return_value={'stop':False,'safe_now':False}
        with self.assertRaises(RuntimePaused):self.request()
        self.assertFalse((self.old/'STOP_AFTER_RUN').exists())

    def test_active_old_worker_blocks_completion(self):
        self.inventory.return_value=[self.local()];request=self.request()
        with self.assertRaises(RuntimePaused):h.verify_boundary(self.root,'s4',request,self.gpu)
        self.assertFalse((camp(self.root,'s4')/'handoff/complete.json').exists())

    def fullstate(self,mutate=None):
        run='fixture_old_run';work=self.old/'runs'/run/'attempt000';work.mkdir(parents=True,exist_ok=True)
        cfg={'fixture':'config'};bindings={'fixture':'bindings'}
        manifest={'manifest_sha256':'a'*64,'metadata_sha256':'b'*64}
        atomic_json(work/'stream_manifest.json',manifest)
        atomic_json(work/'meta/config.resolved.yaml',cfg);atomic_json(work/'meta/consumed_bindings.json',bindings)
        model={'aligner.weight':torch.ones(1),'backbone.weight':torch.ones(1)}
        state=dict(full_state=True,precision='fp32',update=7,model_state=model,
            optimizer=dict(state={i:dict(step=torch.tensor(7.),exp_avg=torch.ones(1),exp_avg_sq=torch.ones(1))for i in (0,1)},
                param_groups=[dict(name='U',params=[0]),dict(name='A',params=[1])]),
            scheduler={'last_epoch':7},rng=dict(python=random.Random(1).getstate(),numpy=np.random.RandomState(1).get_state(),
                torch=torch.Generator().manual_seed(1).get_state(),cuda=[torch.ones(16,dtype=torch.uint8)]),
            sampler=dict(schema='G23SENS_STREAM_CURSOR_v1',cursor=7,completed_updates=7,
                manifest_sha256='a'*64,metadata_sha256='b'*64,consumed_prefix_sha256='c'*64),
            run_id=run,attempt=0,source_identity={'files':{'fixture':'source'}},
            bindings_sha256=object_sha(bindings),config_sha256=object_sha(cfg),stream_sha256=object_sha(manifest))
        if mutate:mutate(state)
        target=work/'last/training_state.pt';target.parent.mkdir();torch.save(state,target)
        identity={key:state[key] for key in ('full_state','update','source_identity','bindings_sha256',
            'config_sha256','stream_sha256','run_id','attempt')}
        identity.update(training_state_sha256=sha256(target),state_hash=state_hash(model))
        atomic_json(work/'last/identity.json',identity)
        oldstate=dict(campaign_id=h.OLD_CAMPAIGN,server='s4',active_run_id=run,runs={run:{'attempt':0}})
        atomic_json(self.old/'state.json',oldstate)
        return oldstate,work

    def test_verified_full_state_and_receipt_preserve_original(self):
        state,work=self.fullstate();before=sha256(work/'last/training_state.pt')
        request=self.request();receipt=h.verify_boundary(self.root,'s4',request,self.gpu)
        self.assertEqual(receipt['preserved_run']['actual_updates'],7)
        self.assertEqual(receipt['preserved_run']['status'],'PRESERVED')
        self.assertEqual(sha256(work/'last/training_state.pt'),before)
        self.assertEqual(h.verify_local_receipt(self.root,'s4',self.gpu),receipt)
        with self.assertRaises(RuntimePaused):h.verify_local_receipt(self.root,'s4','GPU-other')

    def test_incomplete_nested_rng_rejected(self):
        state,_=self.fullstate(lambda s:s.update(rng={}))
        with self.assertRaises(RuntimePaused):h._preserved_run(self.root,'s4',state)

    def test_mismatched_sampler_cursor_rejected(self):
        state,_=self.fullstate(lambda s:s['sampler'].update(cursor=6))
        with self.assertRaises(RuntimePaused):h._preserved_run(self.root,'s4',state)

    def test_missing_optimizer_moments_rejected(self):
        state,_=self.fullstate(lambda s:s['optimizer'].update(state={}))
        with self.assertRaises(RuntimePaused):h._preserved_run(self.root,'s4',state)

    def test_hash_corruption_and_missing_candidate_state_fail(self):
        state,work=self.fullstate();(work/'last/training_state.pt').write_bytes(b'corrupted')
        with self.assertRaises(RuntimePaused):h._preserved_run(self.root,'s4',state)
        with self.assertRaises(ValueError):h._preserved_run(self.root,'s4',{'active_run_id':'../other'})

    def test_readback_detects_reenabled_old_admission(self):
        request=self.request();h.verify_boundary(self.root,'s4',request,self.gpu)
        (self.old/'STOP_AFTER_RUN').unlink()
        with self.assertRaises(RuntimePaused):h.verify_local_receipt(self.root,'s4',self.gpu)


if __name__=='__main__':unittest.main()
