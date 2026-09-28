import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from fh12.data import AUGMENTATION, RECIPE, canonical_sha, sha256_file
from panda_rb import bindings as b


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.paths={k:str(self.root/k) for k in (*b.PAIRS,'origin_manifest','init_manifest')}
        self.cfg=dict(seed=71001,fh12=dict(role='T',server_id='s1',input_layout='P0'),
                      model_args=dict(hidden_size=112,depth=[1,2,3]))
        Path(self.paths['teacher_config']).write_text(yaml.safe_dump(self.cfg))
        Path(self.paths['teacher_checkpoint']).write_bytes(b'fixture-model-not-live-F1')
        model_sha=sha256_file(self.paths['teacher_checkpoint'])
        source=dict(git_release='a'*40,files={'fh12/model.py':'b'*64})
        source['content_sha256']=canonical_sha(source['files'])
        origin=dict(schema='FH12_REFERENCE_v1',server='s1',teacher_run_id=b.TEACHER_RUN,
                    teacher_update=50000,teacher_layout='P0',teacher_checkpoint_sha256=model_sha,
                    teacher_config_sha256=canonical_sha(self.cfg),source_identity=source,
                    LP_recipe=RECIPE,augmentation=AUGMENTATION,tau_R=.012,q_ref=.25,q_shape=[9714,4])
        ids=np.arange(3072,dtype=np.int64);q=np.full((9714,4),.25,np.float32)
        with open(self.paths['q_cache_path'],'wb') as f:np.savez(f,q=q,calibration_indices=ids)
        origin['calibration_indices_sha256']=canonical_sha(ids.tolist())
        data=dict(schema='FH12_DATA_v1',sensor='WV3',server='s1',max_pixel=2047.,
                  recipe=RECIPE,augmentation=AUGMENTATION,splits={})
        self.data_paths={}
        for split in ('train','val','rr','fr'):
            datafile=self.root/(split+'.h5');datafile.write_bytes(('native '+split).encode())
            lpfile=self.root/(split+'_lp.h5');lpfile.write_bytes(('LP '+split).encode())
            self.data_paths[split]=dict(dataroot=str(datafile),lpan_path=str(lpfile))
            data['splits'][split]=dict(**self.data_paths[split],sha256=sha256_file(datafile),
                                      lpan_sha256=sha256_file(lpfile),sample_order_sha256='c'*64)
        origin.update(train_sha256=data['splits']['train']['sha256'],
                      train_lpan_sha256=data['splits']['train']['lpan_sha256'],train_sample_order_sha256='c'*64)
        Path(self.paths['dataset_manifest_path']).write_text(json.dumps(data))
        Path(self.paths['lpan_manifest_path']).write_text(json.dumps(dict(schema='FH12_LPAN_v1',server='s1',
                                                                       recipe=RECIPE,splits=data['splits'])))
        Path(self.paths['init_manifest']).write_text('{}')
        state=dict(update=50000,model_sha256=model_sha,config_sha256=origin['teacher_config_sha256'],source_identity=source)
        torch.save(state,self.paths['teacher_training_state'])
        identity=dict(**state,training_state_sha256=sha256_file(self.paths['teacher_training_state']))
        Path(self.paths['teacher_checkpoint_identity']).write_text(json.dumps(identity))
        Path(self.paths['calibration_path']).write_text(json.dumps(origin))
        for key,field in b.PAIRS.items():origin[field]=sha256_file(self.paths[key])
        self.origin=origin;Path(self.paths['origin_manifest']).write_text(json.dumps(origin))
        self.pin=patch.object(b,'EXPECTED_F1_SHA',model_sha);self.pin.start()
        # Synthetic tiny files test the reference-chain logic; real H5 geometry
        # validation is covered by the unchanged FH20/FH12 reference tests.
        self.data_mock=patch('fh20r1.references._validate_data',side_effect=lambda data,lp,root:data)
        self.data_mock.start()
        self.destination=self.root/'bindings.json'

    def tearDown(self):
        self.data_mock.stop();self.pin.stop();self.temp.cleanup()

    def prepare(self):return b.prepare_binding(self.destination,self.root,self.paths,self.data_paths)

    def test_binding_full_chain_readonly_and_not_ready(self):
        original={k:sha256_file(v) for k,v in self.paths.items()}
        self.prepare();cfg,binding,data,q=b.validate_binding(self.destination)
        self.assertEqual(binding['state'],'BOUND_REQUIRES_RUNTIME_PARITY')
        self.assertFalse(binding['ready_to_train']);self.assertEqual(q.shape,(9714,4))
        self.assertEqual(original,{k:sha256_file(v) for k,v in self.paths.items()})
        self.assertEqual(binding['tau_R'],.012);self.assertEqual(binding['q_ref'],.25)

    def test_wrong_teacher_server_rejected(self):
        for field,value in [('server','s3'),('teacher_run_id','F3'),('teacher_update',49999),('teacher_layout','PH')]:
            bad=dict(self.origin,**{field:value})
            with self.assertRaises(ValueError):b._origin_valid(bad)

    def test_byte_tamper_rejected(self):
        self.prepare();Path(self.paths['teacher_checkpoint']).write_bytes(b'changed')
        with self.assertRaises(ValueError):b.validate_binding(self.destination)

    def test_q_calibration_mismatch_rejected(self):
        qpath=Path(self.paths['q_cache_path'])
        with open(qpath,'wb') as f:np.savez(f,q=np.ones((9714,4),np.float32),calibration_indices=np.arange(3072))
        self.origin['q_cache_sha256']=sha256_file(qpath)
        Path(self.paths['origin_manifest']).write_text(json.dumps(self.origin))
        with self.assertRaises(ValueError):self.prepare()

    def test_manifest_payload_tamper_rejected(self):
        self.prepare();payload=json.loads(self.destination.read_text());payload['q_ref']=.5
        self.destination.chmod(0o644);self.destination.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):b.validate_binding(self.destination)

    def test_export_import_same_common_identity_different_paths(self):
        self.prepare();old=json.loads(self.destination.read_text())
        package=b.export_binding(self.destination,self.root/'portable')
        native={s:p['dataroot'] for s,p in self.data_paths.items()}
        target=self.root/'imported.json';b.import_binding(package,target,native,self.root)
        _,new,_,_=b.validate_binding(target)
        self.assertEqual(old['common_sha256'],new['common_sha256'])
        self.assertNotEqual(old['paths']['teacher_checkpoint'],new['paths']['teacher_checkpoint'])
        self.assertFalse((package/'native').exists())

    def test_incomplete_import_explicit_paths_rejected(self):
        self.prepare();package=b.export_binding(self.destination,self.root/'portable')
        with self.assertRaises(ValueError):b.import_binding(package,self.root/'x.json',{'train':'x'},self.root)

    def test_package_traversal_rejected(self):
        self.prepare();package=b.export_binding(self.destination,self.root/'portable')
        path=package/'package.json';meta=json.loads(path.read_text())
        meta['artifacts']['teacher_checkpoint']['path']='../bad'
        path.chmod(0o644);path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):b.import_binding(package,self.root/'x.json',{},self.root)

    def test_bare_package_not_allowed_as_common_weights(self):
        self.prepare();package=b.export_binding(self.destination,self.root/'portable')
        native={s:p['dataroot'] for s,p in self.data_paths.items()}
        with self.assertRaises(ValueError):b.import_binding(package,self.root/'x.json',native,self.root,weights_dir=self.root/'weights')

    def test_immutable_rebind(self):
        self.prepare();self.prepare()
        changed=copy.deepcopy(self.data_paths);changed['train']['dataroot']='relative/file'
        with self.assertRaises(ValueError):b.prepare_binding(self.destination,self.root,self.paths,changed)

    def test_full_six_weight_cache_roundtrip_and_e_tamper(self):
        from panda_rb import weights as w
        from panda_rb.plan import registry
        self.prepare();bound=json.loads(self.destination.read_text());weightdir=self.root/'weights'
        q=np.full((9714,4),.25,np.float32);e=np.zeros_like(q)
        w._save_npz(weightdir/'train_e_bar.npz',{'e_bar':e})
        identity=dict(schema=w.REVISION,binding_common_sha256=bound['common_sha256'],
                      q_sha256=w.array_sha(q),e_definition=w.E_DEFINITION,source_identity={'content_sha256':'fixture'},
                      train_views=[0,1,2,3])
        esha=sha256_file(weightdir/'train_e_bar.npz')
        (weightdir/'train_e_bar.json').write_text(json.dumps(dict(identity=identity,cache_sha256=esha)))
        seeds=[s for pair in registry()['independent_student_seeds'].values() for s in pair]
        for seed in seeds:
            arrays,meta=w.build_weights(q,e,.25,seed);arrays['calibration_indices']=np.arange(3072,dtype=np.int64)
            meta['arrays_sha256']['calibration_indices']=w.array_sha(arrays['calibration_indices'])
            file=weightdir/f'seed_{seed}.npz';w._save_npz(file,arrays)
            meta.update(cache_sha256=sha256_file(file),binding_common_sha256=bound['common_sha256'],
                        cache_identity=identity,train_e_cache_sha256=esha)
            (weightdir/f'seed_{seed}.json').write_text(json.dumps(meta))
        package=b.export_binding(self.destination,self.root/'portable',weightdir)
        imported=self.root/'imported.json';target_weights=self.root/'imported_weights'
        native={s:p['dataroot'] for s,p in self.data_paths.items()}
        b.import_binding(package,imported,native,b.ROOT,weights_dir=target_weights)
        new=json.loads(imported.read_text())
        for seed in seeds:
            arrays,manifest=w.load_weights(target_weights,seed,new)
            self.assertEqual(manifest['binding_common_sha256'],bound['common_sha256'])
        ecache=target_weights/'train_e_bar.npz';ecache.chmod(0o644);ecache.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):w.load_weights(target_weights,seeds[0],new)

if __name__=='__main__':unittest.main()
