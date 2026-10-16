import contextlib
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from . import plan
from .__main__ import main


class CLITests(unittest.TestCase):
    def test_catalog_is_offline_and_s3_is_wv3(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(['catalog','--server','s3']),0)
        obj=json.loads(out.getvalue())
        self.assertEqual(obj['campaign']['dataset'],'WV3')
        self.assertEqual(len(obj['search']),6)
        self.assertTrue(all(c['phase']=='donor' for c in obj['search']))

    def test_mounts_explicitly_resolve_wv3_not_s3_gf2(self):
        with patch('edgeres48.docker_mounts.targets',return_value=[Path('/tmp/WV3.h5')]) as target,\
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['mounts']),0)
        target.assert_called_once_with(plan.ROOT,'s1')
        self.assertIn('WV3.h5',out.getvalue())

    def test_launchers_parse_and_do_not_stop_training(self):
        for file in ('scripts/run_donor24.sh','scripts/run_donor24_docker.sh'):
            path=plan.ROOT/file
            subprocess.run(['bash','-n',str(path)],check=True)
            text=path.read_text()
            for forbidden in ('docker stop','docker rm','kill -9','pkill','GF2'):
                self.assertNotIn(forbidden,text)
        self.assertIn('-m donor24 mounts',(plan.ROOT/'scripts/run_donor24_docker.sh').read_text())


if __name__=='__main__':unittest.main()
