"""Offline takeover tests. All OS signals, cron, GPU and worker discovery are mocked."""
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pcrepro import handoff as H
from pcrepro.common import atomic_json, camp, read, sha256


def job(root, *, pid=919191, server='s3', action='run', script='gfp40_runner.py'):
    return dict(pid=pid,args=['python',str(Path(root)/'tools'/script),action,'--server',server],
        cwd=str(Path(root).resolve()),ticks=str(pid+13),uid=os.getuid(),script=script,
        action=action,server=server,campaign=H.LEGACY.get(script),config=None,
        shared_work_dir=str((Path(root)/'work_dir').resolve()))


def receipt(root, *, server='s3', jobs=None):
    value=dict(server=server,status='STOP_REQUESTED',jobs=jobs or [],
        shared_work_dir=str((Path(root)/'work_dir').resolve()))
    atomic_json(camp(root,server)/'handoff/transition.json',value)
    return value


@contextmanager
def offline_world(rows=(), gpu=()):
    """Even negative tests must never accidentally execute real system operations."""
    with ExitStack() as stack:
        inventory=stack.enter_context(patch('pcrepro.handoff.inventory',return_value=list(rows)))
        stack.enter_context(patch('pcrepro.handoff.gpu_processes',return_value=list(gpu)))
        cron=stack.enter_context(patch('pcrepro.handoff.disable_known_watchdog',return_value=dict(removed=0)))
        kill=stack.enter_context(patch('pcrepro.handoff.signal_verified',return_value=True))
        stack.enter_context(patch('pcrepro.handoff.subprocess.run',side_effect=AssertionError('unmocked OS process')))
        stack.enter_context(patch('pcrepro.handoff.subprocess.check_output',side_effect=AssertionError('unmocked GPU query')))
        stack.enter_context(patch('pcrepro.handoff.os.kill',side_effect=AssertionError('unmocked signal')))
        yield inventory,cron,kill


class ProcessIdentityTests(unittest.TestCase):
    def test_signal_only_exact_current_pid_and_sigterm(self):
        row=job('/tmp/fixture')
        with patch('pcrepro.handoff.process_record',return_value=row.copy()),patch('pcrepro.handoff.os.kill') as kill:
            self.assertTrue(H.signal_verified(row))
            kill.assert_called_once_with(row['pid'],signal.SIGTERM)

    def test_pid_reuse_or_changed_command_cwd_uid_is_never_signaled(self):
        row=job('/tmp/fixture')
        for field,value in [('args',['python','other.py']),('cwd','/tmp/other'),('ticks','new'),('uid',os.getuid()+1)]:
            with self.subTest(field=field):
                changed=dict(row,**{field:value})
                with patch('pcrepro.handoff.process_record',return_value=changed),patch('pcrepro.handoff.os.kill') as kill:
                    with self.assertRaises(ValueError):H.signal_verified(row)
                    kill.assert_not_called()
        with patch('pcrepro.handoff.process_record',return_value=None),patch('pcrepro.handoff.os.kill') as kill:
            self.assertFalse(H.signal_verified(row));kill.assert_not_called()

    def test_even_matching_foreign_uid_cannot_be_signaled(self):
        row=dict(job('/tmp/fixture'),uid=os.getuid()+1)
        with patch('pcrepro.handoff.process_record',return_value=row),patch('pcrepro.handoff.os.kill') as kill:
            with self.assertRaises(ValueError):H.signal_verified(row)
            kill.assert_not_called()

    def test_absolute_repository_watchdog_is_found_when_cwd_is_home(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'repo';root.mkdir();home=Path(temporary)/'home';home.mkdir()
            proc=Path(temporary)/'proc';base=proc/'812345';base.mkdir(parents=True)
            (base/'cmdline').write_bytes(('bash\0'+str(root/'tools/_watchdog.sh')+'\0').encode())
            fields=['S']+['0']*18+['123456']
            (base/'stat').write_text('812345 (bash) '+' '.join(fields))
            (base/'cwd').symlink_to(home,target_is_directory=True)
            rows=H.inventory(root,proc)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['script'],'_watchdog.sh')
            self.assertEqual(rows[0]['ticks'],'123456');self.assertIsNone(rows[0]['server'])
            self.assertEqual(rows[0]['cwd'],str(home))
            # Ownership cannot be inferred solely from the repository path.
            with offline_world(rows) as (_,cron,kill):
                with self.assertRaises(ValueError):H.begin(root,'s3',activated=True)
                cron.assert_not_called();kill.assert_not_called()


class TakeoverSafetyTests(unittest.TestCase):
    def test_explicit_activation_and_supported_server_required(self):
        with tempfile.TemporaryDirectory() as root,offline_world() as (inventory,cron,kill):
            with self.assertRaises(PermissionError):H.begin(root,'s3')
            for server in ('s1','s2','s6'):
                with self.assertRaises(ValueError):H.begin(root,server,activated=True)
            inventory.assert_not_called();cron.assert_not_called();kill.assert_not_called()

    def test_other_server_and_unknown_launchers_block_before_cron_or_signal(self):
        with tempfile.TemporaryDirectory() as root:
            for row in (job(root,server='s1'),job(root,server='s4'),job(root,script='train.py')):
                with self.subTest(row=row),offline_world([row]) as (_,cron,kill):
                    with self.assertRaises(ValueError):H.begin(root,'s3',activated=True)
                    cron.assert_not_called();kill.assert_not_called()

    def test_foreign_receipt_or_job_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            path=camp(root,'s3')/'handoff/transition.json'
            for corrupt in (dict(server='s4',jobs=[]),
                    dict(server='s3',shared_work_dir='/other/work_dir',jobs=[]),
                    dict(server='s3',jobs=[job(root,server='s4')]),
                    dict(server='s3',jobs=[dict(job(root),shared_work_dir='/other/work_dir')])):
                with self.subTest(receipt=corrupt):
                    atomic_json(path,corrupt)
                    with offline_world() as (_,cron,kill):
                        with self.assertRaises(ValueError):H.begin(root,'s3',activated=True)
                        with self.assertRaises(ValueError):H.verify_stopped(root,'s3')
                        cron.assert_not_called();kill.assert_not_called()

    def test_missing_handoff_receipt_cannot_authorize_new_work(self):
        with tempfile.TemporaryDirectory() as root,offline_world():
            with self.assertRaises(ValueError):H.verify_stopped(root,'s3')

    def test_protected_local_route_blocks_before_watchdog_side_effects(self):
        for name in ('_fh20r1/r2_local_server.txt','_fh20r1/local_server.txt','_fh12/local_server.txt'):
            for owner in ('s1','s2','s4'):
                with self.subTest(name=name,owner=owner),tempfile.TemporaryDirectory() as root:
                    path=Path(root)/'work_dir'/name;path.parent.mkdir(parents=True);path.write_text(owner+'\n')
                    with patch('pcrepro.handoff.subprocess.run') as cron:
                        with self.assertRaises(ValueError):H.disable_known_watchdog(root,'s3')
                        cron.assert_not_called()
                    with offline_world() as (_,cron,kill):
                        with self.assertRaises(ValueError):H.begin(root,'s3',activated=True)
                        cron.assert_not_called();kill.assert_not_called()

    def test_only_tagged_exact_repository_watchdog_cron_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            own=f'* * * * * bash {root}/tools/_watchdog.sh # PANCRAFTER-WATCHDOG'
            foreign='* * * * * bash /other/repo/tools/_watchdog.sh # PANCRAFTER-WATCHDOG'
            prefix_collision=f'* * * * * bash /another{root}/tools/_watchdog.sh # PANCRAFTER-WATCHDOG'
            untagged=f'* * * * * bash {root}/tools/_watchdog.sh'
            original='\n'.join([own,foreign,prefix_collision,untagged,'# ordinary task'])+'\n'
            response=subprocess.CompletedProcess(['crontab','-l'],0,original,'')
            with patch('pcrepro.handoff.subprocess.run',return_value=response) as cron:
                result=H.disable_known_watchdog(root,'s3')
            self.assertEqual(result['removed'],1);self.assertEqual(cron.call_count,2)
            self.assertEqual(cron.call_args.args[0],['crontab','-'])
            self.assertEqual(cron.call_args.kwargs['input'],'\n'.join([foreign,prefix_collision,untagged,'# ordinary task'])+'\n')
            backup=camp(root,'s3')/'handoff/removed_watchdog_cron.json'
            self.assertEqual(read(backup)['entries'],[own]);self.assertEqual(backup.stat().st_mode & 0o777,0o600)

    def test_training_metadata_requires_explicit_gf2_and_same_server(self):
        with tempfile.TemporaryDirectory() as root:
            row=job(root,action='train');cfg=Path(root)/'config.json';row['config']=str(cfg)
            for field in ({},{'server':'s3'},{'sensor':'WV3','server':'s3'},{'sensor':'GF2','server':'s4'}):
                atomic_json(cfg,dict(work_dir='work_dir/OLD',gfp40=field))
                with self.assertRaises(ValueError):H._training_run(row)
            atomic_json(cfg,dict(work_dir='work_dir/OLD',gfp40=dict(sensor='GF2',server='s3')))
            self.assertEqual(H._training_run(row)['work_dir'],str(Path(root)/'work_dir/OLD'))

    def test_controllers_stop_before_training_and_cooperative_request_is_local(self):
        with tempfile.TemporaryDirectory() as root:
            trainer=job(root,pid=919193,action='train');runner=job(root,pid=919192)
            watcher=job(root,pid=919191,script='_watchdog.sh',action='')
            old=Path(root)/'work_dir/_gfp40/s3';old.mkdir(parents=True)
            other=Path(root)/'work_dir/_gfp40/s4';other.mkdir(parents=True)
            with offline_world([trainer,runner,watcher]) as (_,_,kill):
                result=H.begin(root,'s3',activated=True)
            self.assertEqual([c.args[0]['pid'] for c in kill.call_args_list],[919191,919192,919193])
            self.assertEqual(read(old/'control.json')['command'],'STOP_NOW_SAFE')
            self.assertFalse((other/'control.json').exists());self.assertTrue(result['old_final_results_preserved'])

    def test_restarted_watchdog_after_completed_handoff_is_stopped_again(self):
        with tempfile.TemporaryDirectory() as root:
            value=receipt(root);value.update(status='COMPLETE',complete=True)
            atomic_json(camp(root,'s3')/'handoff/transition.json',value)
            watcher=job(root,script='_watchdog.sh',action='')
            with offline_world([watcher]) as (_,_,kill):
                result=H.begin(root,'s3',activated=True)
            kill.assert_called_once();self.assertEqual(result['status'],'STOP_REQUESTED')
            self.assertEqual(result['jobs'][0]['pid'],watcher['pid'])

    def test_gpu_or_unfinished_legacy_prevents_completion(self):
        with tempfile.TemporaryDirectory() as root:
            receipt(root)
            for rows,gpu in (([job(root)],[]),([],[555])):
                with offline_world(rows,gpu):self.assertFalse(H.verify_stopped(root,'s3')['complete'])
            with offline_world(),patch('pcrepro.handoff.gpu_processes',return_value=None):
                self.assertFalse(H.verify_stopped(root,'s3')['complete'])


class PreservationTests(unittest.TestCase):
    def fullstate(self,root,*,missing=None,wrong_update=False,wrong_sha=False):
        import torch
        from pcrepro.model import state_hash
        wd=Path(root)/'work_dir/OLD';path=wd/'last/training_state.pt';path.parent.mkdir(parents=True)
        model={'weight':torch.ones(1)}
        saved=dict(model_state=model,full_state=True,optimizer={},scheduler={'last_epoch':123},rng={},sampler={},update=123)
        if missing:saved.pop(missing)
        torch.save(saved,path)
        atomic_json(wd/'last/identity.json',dict(full_state=True,state_hash=state_hash(model),update=124 if wrong_update else 123,
            training_state_sha256='wrong' if wrong_sha else sha256(path)))
        row=dict(job(root,action='train'),training=dict(run_id='OLD',work_dir=str(wd)))
        receipt(root,jobs=[row]);return path

    def test_partial_fullstate_preserved_and_never_called_completed_training(self):
        with tempfile.TemporaryDirectory() as root:
            path=self.fullstate(root);before=sha256(path)
            with offline_world():result=H.verify_stopped(root,'s3')
            self.assertTrue(result['complete']);self.assertFalse(result['old_campaign_files_deleted'])
            self.assertEqual(result['preserved_runs'][0]['actual_updates'],123)
            self.assertEqual(result['preserved_runs'][0]['status'],'STOPPED_BY_USER_DIRECTION')
            self.assertEqual(sha256(path),before)

    def test_incomplete_or_corrupt_saved_state_blocks_new_gpu_work(self):
        options=[dict(missing=k) for k in ('optimizer','scheduler','rng','sampler')]+[dict(wrong_update=True),dict(wrong_sha=True)]
        for option in options:
            with self.subTest(option=option),tempfile.TemporaryDirectory() as root:
                path=self.fullstate(root,**option);before=sha256(path)
                with offline_world():
                    with self.assertRaises(ValueError):H.verify_stopped(root,'s3')
                self.assertEqual(sha256(path),before)
                self.assertEqual(read(camp(root,'s3')/'handoff/transition.json')['status'],'STOP_REQUESTED')

    def test_unstarted_queues_and_admissions_are_cancelled_without_deletion(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            atomic_json(root/'config/gfp40/s3_queue.json',[dict(server='s3',run_ids=['P40_A','P40_STARTED']),dict(server='s4',run_ids=['OTHER'])])
            atomic_json(root/'config/gfb20/s3_queue.json',dict(blocks=[dict(run_ids=['B20_A'])]))
            atomic_json(root/'work_dir/_gfp40/s3/status.json',dict(unstarted_run_ids=['STATE_A']))
            atomic_json(root/'work_dir/_gfp40/s3/admissions/block.json',dict(run_ids=['ADMIT_A']))
            atomic_json(root/'work_dir/P40_STARTED/meta/training_start_manifest.json',dict(actual_start='fixture'))
            receipt(root)
            with offline_world():result=H.verify_stopped(root,'s3')
            rows=result['cancelled_unstarted']
            self.assertEqual({r['run_id'] for r in rows},{'P40_A','B20_A','STATE_A','ADMIT_A'})
            self.assertTrue(all(r['status']=='CANCELLED_BY_USER_DIRECTION' for r in rows))
            self.assertTrue((root/'config/gfp40/s3_queue.json').is_file())
            self.assertTrue((root/'work_dir/P40_STARTED/meta/training_start_manifest.json').is_file())


if __name__=='__main__':unittest.main()
