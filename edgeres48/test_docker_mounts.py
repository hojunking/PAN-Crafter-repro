import json
from pathlib import Path
import tempfile
import unittest
from .docker_mounts import targets


class DockerMountTests(unittest.TestCase):
    def fixture(self, root):
        (root/'ablr2').mkdir()
        paths={}
        for split in ('train','val','rr','fr'):
            p=root/(split+'.h5');p.touch()
            paths[split]={'path':p.name}
        (root/'ablr2/sensor_sources.json').write_text(json.dumps({'sensors':{
            'WV3':{'splits':paths},'QB':{'splits':paths}}}))

    def test_absolute_symlink_target_is_mounted(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root=Path(directory); self.fixture(root)
            target=Path(external)/'original.h5';target.touch()
            (root/'train.h5').unlink();(root/'train.h5').symlink_to(target)
            self.assertIn(target,targets(root,'s1'))
            self.assertNotIn(root/'train.h5',targets(root,'s1'))

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root);(root/'val.h5').unlink()
            with self.assertRaises(FileNotFoundError):targets(root,'s1')

    def test_qb_original_lms_sources_also_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'config/bglobal_rr').mkdir(parents=True)
            original=root/'original.h5';original.touch()
            (root/'config/bglobal_rr/qb_original_lms.json').write_text(json.dumps({
                'splits':{'train':{'path':original.name},'val':{'path':'val.h5'}}}))
            self.assertEqual(len(targets(root,'s2')),5)
            self.assertIn(original,targets(root,'s2'))


if __name__=='__main__':unittest.main()
