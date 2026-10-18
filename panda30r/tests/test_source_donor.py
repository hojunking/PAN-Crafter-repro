"""Real architecture/source import of the historical minimal resume identity."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from panda30r.tests.test_runtime_engines import source_checkpoint
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda30r.source_donor import load_donor
from ta2.evaluation import file_sha


class SourceDonorTests(unittest.TestCase):
    def test_minimal_resume_dataset_is_verified_from_config_and_source(self):
        with tempfile.TemporaryDirectory() as root:
            sources=synthetic_sources(root)
            original,payload=source_checkpoint(root,sources)
            explicit=load_donor(original['checkpoint_path'],original['checkpoint_sha256'],sources)
            payload['identity'].pop('dataset')
            payload.update(resume_only=True,evaluation=None)
            path=Path(root)/'resume.pt'; torch.save(payload,path)
            imported=load_donor(path,file_sha(path),sources)
            self.assertEqual(imported.checkpoint_metadata['source_identity_schema'],'VERIFIED_MINIMAL_RESUME')
            for key in ('aligner_state_sha256','teacher_state_sha256','source_manifest_sha256','donor_step'):
                self.assertEqual(imported.binding[key],explicit.binding[key])
            imported.assert_frozen()

    def test_missing_dataset_is_not_generic_metadata_bypass(self):
        with tempfile.TemporaryDirectory() as root:
            sources=synthetic_sources(root)
            original,payload=source_checkpoint(root,sources)
            for mode in ('not_resume','wrong_explicit_dataset','wrong_source','wrong_hash'):
                value=deepcopy(payload)
                value['identity'].pop('dataset')
                value.update(resume_only=True,evaluation=None)
                if mode=='not_resume': value['resume_only']=False
                if mode=='wrong_explicit_dataset': value['identity']['dataset']='GF2'
                if mode=='wrong_source': value['trainer']['stage']['_extra_state']['sources']['sensor']='GF2'
                if mode=='wrong_hash': value['identity']['model_state_sha256']='0'*64
                path=Path(root)/(mode+'.pt'); torch.save(value,path)
                with self.subTest(mode=mode),self.assertRaises(ValueError):
                    load_donor(path,file_sha(path),sources)
