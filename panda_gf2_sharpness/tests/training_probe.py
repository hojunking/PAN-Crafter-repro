"""Fresh-process CPU check of both original registries, including complete resume."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import torch

if sys.argv[1]=='psnr':
    from panda_gf2_psnr.adapter import install
    install()

from panda_sensor_focus import plan,engine
from panda_sensor_focus.tests.test_core import sources,donor,bank,PINS
from panda_sensor_focus.calibration import bind_donor
from panda_sensor_focus.spec import MatureSpec
from panda_stage2_condq.stream import TrainStream,StreamSpec
from panda_method.calibration import state_dict_sha256
from donor30.artifacts import _same
from panda_gf2_sharpness import plan as P
from panda_sensor_visuals.common import write

torch.set_num_threads(1)
case_name='GF2P_C60LR200' if sys.argv[1]=='psnr' else 'GF2_C60LR150'
for rep in ('R1','R2'):
    cfg=plan.resolve_case('s3',case_name,rep)
    with tempfile.TemporaryDirectory() as temp:
        reg=sources(temp,'GF2',count=1)
        d=donor('s3',reg);spec=MatureSpec(**cfg['stage2_spec']);d=bind_donor(d,spec)
        cfg['donor_binding']=d.binding;cfg['donor_checkpoint_sha256']=P.DONOR_SHA
        # Production receives JSON-normalized original config, not a new recipe.
        cfg=json.loads(json.dumps(cfg))
        case=next(c for c in P.cases() if c['case']==case_name and c['replica']==rep)
        P.validate_case(cfg,case)
        for key,value in [('planned_horizon',5000),('donor_checkpoint_sha256','changed')]:
            bad=deepcopy(cfg);bad[key]=value
            try:P.validate_case(bad,case)
            except ValueError:pass
            else:raise AssertionError('Changed original recipe accepted')
        schedule=engine.ScheduleTrainingSpec(**cfg['training_spec'])
        assert schedule.total_updates==60000 and schedule.rate(5000)>.97*case['lr']
        # Small synthetic batch only for CPU checks; registered model and loss unchanged.
        cfg['training_spec'].update(effective_batch_size=1,microbatch_size=1)
        stats=bank(d,spec,reg)
        def build():
            trainer=engine.build_fresh_stage2(deepcopy(cfg),d,stats,evaluator_pin=PINS['evaluator_pin'])
            stream=TrainStream(reg,StreamSpec(cfg['stream_seeds']['rr_data'],cfg['stream_seeds']['rr_augmentation'],**cfg['augmentation']),
                microbatch_size=1,effective_batch_size=1)
            return trainer,stream
        trainer,stream=build()
        frozen={k:state_dict_sha256(getattr(trainer.stage,k)) for k in ('aligner','teacher')}
        first=engine.advance(trainer,stream)
        assert first['loss']==first['hard_weighted'] and first['soft_weighted']==0
        path=Path(temp)/'resume.pt';saved=engine.save_checkpoint(path,trainer,stream,**PINS)
        from panda_sensor_visuals.model import build as build_export_model
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            batch=reg.sources.batch('train',[0],device='cpu')
            exported=build_export_model(cfg,path,1)(batch)['y']
            native=trainer.stage.inference_model()(batch)['y']
            assert torch.equal(exported,native), 'Export graph differs from original GF2 graph'
        expected=engine.advance(trainer,stream);expected_state=deepcopy(trainer.state_dict())
        fresh,new_stream=build()
        restored,new_stream=engine.load_checkpoint(path,saved['file_sha256'],fresh,new_stream,**PINS)
        actual=engine.advance(restored,new_stream)
        assert _same(actual,expected) and _same(restored.state_dict(),expected_state)
        assert all(state_dict_sha256(getattr(restored.stage,k))==v for k,v in frozen.items())
        print(case_name,rep,'schedule/frozen A+T/Hard-only/exact resume OK')
