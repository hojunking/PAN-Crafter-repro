"""Read-only native FR operator test; these are not trained-model results."""
import argparse,json
import numpy as np
import torch
from directshift48.evaluation import metric_functions_for,metric_values
from directshift48.reporting import atomic_json
from panda_unified.core import warp_pan,base_from_batch
from ta2.evaluation import evaluator_identity
from .assets import sources
from .evaluation import dual_fr_metrics
from . import plan

def run(server):
    torch.set_num_threads(4)
    source,_=sources(server);batch=source.batch('fr',[0],device='cpu')
    prediction=((base_from_batch(batch,'interp23')[0]+1)*(source.peak/2)).numpy()
    delta=torch.tensor([[.625,-.375]],dtype=torch.float32)
    pan=((warp_pan(batch.pan,delta)[0]+1)*(source.peak/2)).numpy()
    arrays={k:v[0] for k,v in source.raw('fr',[0]).items()}
    original_pan=arrays['pan'].copy()
    fn=metric_functions_for(source,evaluator_identity())['fr']
    original=metric_values(fn(prediction,arrays),'fr',source.bands)
    corrected=dual_fr_metrics(fn,prediction,arrays,pan,original,source.bands)
    if not np.array_equal(arrays['pan'],original_pan):raise AssertionError('Original PAN mutated')
    result=dict(sensor=source.sensor,scope='OPERATOR_TEST_ONLY_INTERP23_BASE_NOT_TRAINED_MODEL',
        original=original,corrected=corrected,prediction_shape=list(prediction.shape),
        PAN_DN_peak=source.peak,D_lambda_equal=True,original_pan_unchanged=True,
        corrected_pan_max_abs_change=float(np.abs(pan-original_pan).max()))
    atomic_json(plan.campaign_root(server)/'checks/native_fr_probe.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--server',choices=('s2','s3'),required=True)
    print(json.dumps(run(p.parse_args().server)))
