import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from pcrepro.model import build_model,state_hash
from pcrepro.checkpoint import save_resume,load_resume,save_model,load_model_checkpoint,selection_checkpoints
from fh12.common import read_json


class CheckpointTests(unittest.TestCase):
    def model(self):return build_model(4,seed=2,max_pixel=1023,hidden_size=8,depth=(1,1,1),num_heads=2)
    def test_two_resume_snapshots_and_checksum(self):
        model=self.model()
        with tempfile.TemporaryDirectory() as tmp:
            for update in (0,1,2):
                save_resume(tmp,dict(full_state=True,update=update,model_state=model.state_dict()),dict(update=update,source='fixed'))
            self.assertEqual(len(list((Path(tmp)/'resume').glob('state_*'))),2)
            state,identity=load_resume(tmp,dict(source='fixed'))
            self.assertEqual(state['update'],2)
            with self.assertRaises(ValueError):load_resume(tmp,dict(source='changed'))
            index=read_json(Path(tmp)/'resume/index.json')
            path=Path(tmp)/'resume'/index['snapshots'][0]/'training_state.pt'
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):load_resume(tmp)

    def test_best_replacement_exact_preservation_and_alias_sha(self):
        model=self.model()
        with tempfile.TemporaryDirectory() as tmp:
            save_model(tmp,'best_val',model,dict(update=1000))
            best=save_model(tmp,'best_val',model,dict(update=50000))
            exact=save_model(tmp,'exact_50000',model,dict(update=50000),dict(full_state=True,update=50000,model_state=model.state_dict()))
            locations=selection_checkpoints(tmp)
            self.assertEqual(best['model_sha256'],exact['model_sha256'])
            self.assertEqual(len(list((Path(tmp)/'checkpoints').glob('best_val_*'))),1)
            loaded,identity=load_model_checkpoint(locations['EXACT_50000'])
            self.assertEqual(state_hash(loaded.state_dict()),state_hash(model.state_dict()))
            self.assertEqual(identity['update'],50000)
            save_model(tmp,'best_val',model,dict(update=40000))
            self.assertTrue(locations['EXACT_50000'].is_dir())
            with torch.no_grad():next(model.parameters()).add_(.01)
            with self.assertRaises(ValueError):save_model(tmp,'exact_50000',model,dict(update=50000),dict(update=50000))

    def test_final_requires_actual_horizon_and_matching_state(self):
        model=self.model()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):save_model(tmp,'exact_50000',model,dict(update=49000),{})
            with self.assertRaises(ValueError):save_model(tmp,'exact_50000',model,dict(update=50000))
            with self.assertRaises(ValueError):save_model(tmp,'random_best',model,dict(update=50000))

    def test_failed_pointer_publication_keeps_previous_resume(self):
        from pcrepro.checkpoint import atomic_json
        model=self.model()
        with tempfile.TemporaryDirectory() as tmp:
            save_resume(tmp,dict(full_state=True,update=0,model_state=model.state_dict()),dict(update=0))
            def fail_index(path,value):
                if Path(path).name=='index.json':raise OSError('fixture I/O failure')
                return atomic_json(path,value)
            with patch('pcrepro.checkpoint.atomic_json',side_effect=fail_index):
                with self.assertRaises(OSError):
                    save_resume(tmp,dict(full_state=True,update=1,model_state=model.state_dict()),dict(update=1))
            saved,_=load_resume(tmp)
            self.assertEqual(saved['update'],0)


if __name__=='__main__':unittest.main()
