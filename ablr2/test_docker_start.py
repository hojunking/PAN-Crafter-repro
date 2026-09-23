"""Docker launch contract; no daemon, GPU, worktree or lease writes."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.ablr2_docker_start import build_command, build_mounts, matching_running, LABEL


class DockerStartTests(unittest.TestCase):
    def command(self, **overrides):
        kwargs = dict(root='/repo', frozen='/frozen', server='s1', image_id='sha256:'+'a'*64,
            commit='b'*40, name='test', mounts=[('/repo','ro'),('/repo/work_dir','rw')],
            dlpan='/DLPan', lease_hours=72, uid=1000, gid=1000)
        kwargs.update(overrides)
        return build_command(**kwargs)

    def test_pinned_nonroot_foreground_no_restart_or_privilege(self):
        cmd = self.command()
        self.assertEqual(cmd[cmd.index('--user')+1], '1000:1000')
        self.assertIn('--pid=host', cmd)
        self.assertIn('--foreground', cmd)
        self.assertEqual(cmd[cmd.index('--restart')+1], 'no')
        self.assertIn('sha256:'+'a'*64, cmd)
        self.assertNotIn('--privileged', cmd)
        self.assertNotIn('--rm', cmd)
        self.assertNotIn('/var/run/docker.sock', ' '.join(cmd))
        self.assertFalse(any(arg.startswith('HOME=') for arg in cmd))
        self.assertIn('HF_HOME=/tmp/ablr2-huggingface', cmd)
        self.assertIn('type=bind,src=/repo,dst=/repo,readonly', cmd)
        self.assertIn('type=bind,src=/repo/work_dir,dst=/repo/work_dir', cmd)

    def test_invalid_gpu_lane_image_lease_rejected(self):
        for override in ({'server':'s4'}, {'gpu':'all'}, {'gpu':'0,1'},
                         {'image_id':'mutable:tag'}, {'lease_hours':73}, {'lease_hours':0}):
            with self.assertRaises(ValueError):
                self.command(**override)

    def test_gpu_uuid_and_no_upload(self):
        cmd = self.command(gpu='GPU-1234-abcd', no_upload=True)
        self.assertIn('device=GPU-1234-abcd', cmd)
        self.assertEqual(cmd[-1], '--no-upload')

    def test_gf2_until_stop_is_not_an_infinite_or_auto_renewed_lease(self):
        cmd=self.command(server='s3',lease_hours=None,until_operator_stop=True)
        self.assertIn('--until-operator-stop',cmd)
        self.assertNotIn('--lease-hours',cmd)
        with self.assertRaises(ValueError):self.command(until_operator_stop=True)
        with self.assertRaises(ValueError):self.command(lease_hours=None)

    def test_only_authenticated_original_container_may_drain(self):
        original=dict(path='/original',git_commit='d'*40)
        row=self.container(Config=dict(Labels={LABEL+'.server':'s1',LABEL+'.commit':'d'*40,LABEL+'.runtime':'/original'}))
        self.assertIsNone(matching_running([row],server='s1',commit='b'*40,frozen='/frozen',
            image_id='sha256:'+'a'*64,draining_release=original))
        with self.assertRaises(ValueError):matching_running([row],server='s1',commit='b'*40,frozen='/frozen',
            image_id='sha256:'+'c'*64,draining_release=original)

    def test_mounts_external_native_sources_and_raw_qb(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root, native, frozen, dlpan = [base/name for name in ('repo','native','frozen','DLPan')]
            for path in (root/'ablr2',root/'work_dir',root/'data',native,dlpan):
                path.mkdir(parents=True)
            for name in ('train','val','rr','fr','raw_train','raw_val'):
                (native/(name+'.h5')).touch()
                (root/'data'/(name+'.h5')).symlink_to(native/(name+'.h5'))
            (root/'ablr2/sensor_sources.json').write_text(json.dumps(dict(sensors=dict(QB=dict(
                splits={s:dict(path='data/'+s+'.h5') for s in ('train','val','rr','fr')},
                source_provenance=dict(raw_train_path='data/raw_train.h5',raw_val_path='data/raw_val.h5'))))))
            mounts = dict(build_mounts(root,frozen,'s2',dlpan))
            self.assertEqual(mounts[str(native)], 'ro')
            self.assertEqual(mounts[str(root)], 'ro')
            self.assertEqual(mounts[str(frozen)], 'ro')
            self.assertEqual(mounts[str(dlpan)], 'ro')
            self.assertEqual([p for p,a in mounts.items() if a=='rw'], [str(root/'work_dir')])

    def container(self, **changes):
        row = dict(Id='one',Image='sha256:'+'a'*64,State=dict(Running=True),Config=dict(Labels={
            LABEL+'.server':'s1',LABEL+'.commit':'b'*40,LABEL+'.runtime':'/frozen'}))
        row.update(changes)
        return row

    def match(self, rows):
        return matching_running(rows,server='s1',commit='b'*40,frozen='/frozen',image_id='sha256:'+'a'*64)

    def test_duplicate_start_reuses_only_exact_running_release(self):
        self.assertEqual(self.match([self.container()])['Id'], 'one')
        self.assertIsNone(self.match([self.container(State=dict(Running=False))]))
        with self.assertRaises(ValueError):
            self.match([self.container(Image='sha256:'+'c'*64)])
        with self.assertRaises(ValueError):
            self.match([self.container(),self.container(Id='two')])


if __name__ == '__main__':
    unittest.main()
