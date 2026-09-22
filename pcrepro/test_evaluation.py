from copy import deepcopy
from types import SimpleNamespace
from contextlib import ExitStack
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fh12.common import atomic_json,read_json
from pcrepro.data import CANONICAL_BANDS,NativeDataset,SENSORS,validate_manifest
from pcrepro import evaluation as ev
from pcrepro.test_data import make_manifest,raises


class ToyModel(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,pan,ms):
        self.calls+=1
        return F.interpolate(ms,scale_factor=4,mode='nearest')+.025*pan


def fake_wald():
    # Test-only identity explicitly disables paper-comparable claims.
    return SimpleNamespace(interp23tap=lambda x,r:np.repeat(np.repeat(x,r,0),r,1))


def test_rr_matches_existing_primitives_and_sensor_dn(sensor):
    bands,dn=SENSORS[sensor];rng=np.random.default_rng(301)
    gt=rng.uniform(dn*.2,dn*.7,(bands,64,64));sr=gt+rng.normal(0,1,gt.shape)
    actual=ev.rr_scene_metrics(sr,gt,sensor)
    a,b=sr.transpose(1,2,0)[20:-21,20:-21],gt.transpose(1,2,0)[20:-21,20:-21]
    expected=dict(ergas=ev.ergas(a,b),sam=ev.sam(a,b),scc=ev.scc_dlpan(a,b),
        psnr=ev.psnr_global(a,b,dn),ssim=ev.ssim_skimage(a,b,dn),q2n=ev.q2n(b,a,32,32)[0])
    for key,value in expected.items():assert actual[key]==value
    assert actual['q4' if bands==4 else 'q8']==actual['q2n']
    sr[:,0]=np.nan
    with raises(FloatingPointError):ev.rr_scene_metrics(sr,gt,sensor)


def test_fr_matches_native_primitives_and_jqm(sensor):
    from tools.metrics.eval_fr import d_lambda_k,d_s
    from tools.metrics.jqm import jqm
    bands,dn=SENSORS[sensor];rng=np.random.default_rng(21)
    sr=rng.uniform(dn*.1,dn*.8,(bands,64,64));raw=dict(lms=sr*.98,
        ms=sr[:,2::4,2::4]*.99,pan=sr.mean(0,keepdims=True))
    wald=fake_wald();a=sr.transpose(1,2,0);lms=raw['lms'].transpose(1,2,0)
    actual=ev.fr_scene_metrics(sr,raw,sensor,CANONICAL_BANDS[sensor],wald)
    assert actual['d_lambda']==d_lambda_k(a,lms,sensor.lower(),4,32,wald)
    assert np.isclose(actual['d_s'],d_s(a,lms,raw['pan'][0],4,32,wald),rtol=0,atol=1e-15)
    assert actual['hqnr']==(1-actual['d_lambda'])*(1-actual['d_s'])
    extra=jqm(a,raw['ms'].transpose(1,2,0),raw['pan'][0],sensor,ratio=4,R=dn,lpf='mtf',window=None,v1=.5)
    assert actual['jqm']==extra['JQM'] and actual['jqm_w_source']=='nnls-normalized'
    assert actual['d_s']==np.abs(np.asarray(actual['Q_high'])-actual['Q_low']).mean()


def test_full_support_rejects_silent_partial_block_crop():
    sr=np.ones((4,68,64));raw=dict(lms=sr,ms=np.ones((4,17,16)),pan=np.ones((1,68,64)))
    with raises(ValueError,match='partial-edge'):ev.fr_scene_metrics(sr,raw,'GF2',CANONICAL_BANDS['GF2'],fake_wald())


def test_sensor_mtf_uses_declared_band_order():
    from tools.metrics.eval_fr import mtf_filter
    rng=np.random.default_rng(54);sr=rng.uniform(30,1600,(4,64,64));raw=dict(lms=sr*.9,
        ms=sr[:,2::4,2::4]*.99,pan=sr.mean(0,keepdims=True))
    actual=ev.fr_scene_metrics(sr,raw,'QB',['NIR','R','G','B'],fake_wald(),False)
    filtered=mtf_filter(sr[::-1].transpose(1,2,0),'qb',4)[...,::-1]
    assert actual['d_lambda']==1-ev.q2n(raw['lms'].transpose(1,2,0),filtered,32,32)[0]


def test_prediction_clip_then_dn_has_no_extra_rounding():
    class Values(nn.Module):
        def forward(self,pan,ms):return torch.tensor([-.99,-2.,2.,.1111]).reshape(1,4,1,1).expand(1,4,4,4)
    batch=dict(pan=torch.zeros(1,1,4,4),ms=torch.zeros(1,4,1,1))
    result=ev._prediction(Values(),batch,'cpu',1023)
    assert result[0,0,0,0]!=round(float(result[0,0,0,0]))
    assert result[0,1,0,0]==0 and result[0,2,0,0]==1023


def test_validation_is_native_ergas_only_and_restores_mode_rng(tmp_path):
    manifest=validate_manifest(make_manifest(tmp_path),tmp_path);data=NativeDataset(manifest,'val',tmp_path)
    model=ToyModel();model.train();torch.manual_seed(98);before=torch.get_rng_state().clone()
    result=ev.validation_metrics(model,data,'cpu')
    assert model.training and torch.equal(before,torch.get_rng_state())
    assert result['hqnr'] is None and result['n_scenes']==len(data) and result['support']=='native64'
    prediction=ev._prediction(model,{k:torch.stack([data[i][k] for i in range(len(data))]) for k in ('pan','ms')},'cpu',1023)
    expected=np.mean([ev.ergas(a.astype(np.float64).transpose(1,2,0),
        data.raw(i)['gt'].astype(np.float64).transpose(1,2,0)) for i,a in enumerate(prediction)])
    assert result['ergas']==expected
    with raises(InterruptedError):ev.validation_metrics(model,data,'cpu',stopcheck=lambda:True)
    assert model.training


def fast_metrics(monkeypatch):
    def rr(sr,gt,sensor):
        value=float(sr.mean()/SENSORS[sensor][1])
        return {**{k:value for k in ev.RR_KEYS},'q4' if SENSORS[sensor][0]==4 else 'q8':value}
    def fr(sr,raw,sensor,order,wald,include_jqm):
        dl=float(sr.mean()/SENSORS[sensor][1]);ds=dl*dl
        row=dict(d_lambda=dl,d_s=ds,hqnr=(1-dl)*(1-ds))
        if include_jqm:row.update(jqm=.4,qlr=.3,qhr=.5)
        return row
    monkeypatch.setattr(ev,'rr_scene_metrics',rr);monkeypatch.setattr(ev,'fr_scene_metrics',fr)


def test_declared_wv3_all20_mean_per_scene_and_resume_identity(tmp_path,monkeypatch):
    fast_metrics(monkeypatch)
    manifest=validate_manifest(make_manifest(tmp_path,'WV3',rr_count=20),tmp_path)
    datasets={s:NativeDataset(manifest,s,tmp_path) for s in ('rr','fr')};model=ToyModel()
    calls=[0]
    def stop():calls[0]+=1;return calls[0]==5
    out=tmp_path/'evaluation'
    with raises(InterruptedError):ev.evaluate_model(model,datasets,'cpu','a'*64,out,stopcheck=stop,wald=fake_wald())
    state=read_json(out/'evaluation_cursor.json');assert len(state['rows']['rr'])==2
    result=ev.evaluate_model(model,datasets,'cpu','a'*64,out,wald=fake_wald())
    assert result['rr']['n_scenes']==20 and result['fr']['n_scenes']==20 and model.calls==40
    assert result['rr']['q_name']=='Q8' and result['rr']['discarded_samples']==0
    assert result['fr']['hqnr']==np.mean([r['hqnr'] for r in result['fr']['per_scene']])
    assert result['fr']['hqnr']!=(1-result['fr']['d_lambda'])*(1-result['fr']['d_s'])
    assert result['fr']['masking'] is False and result['fr']['alignment'] is False
    assert not result['metadata']['paper_comparable'] # RR provenance and test-double wald are unverified.
    repeated=ev.evaluate_model(model,datasets,'cpu','a'*64,out,wald=fake_wald())
    assert repeated==result and model.calls==40
    with raises(ValueError,match='resume'):ev.evaluate_model(model,datasets,'cpu','b'*64,out,wald=fake_wald())
    tampered=read_json(out/'evaluation_cursor.json');tampered['rows']['rr'][0]['ergas']=0
    atomic_json(out/'evaluation_cursor.json',tampered)
    with raises(ValueError,match='checksum'):ev.evaluate_model(model,datasets,'cpu','a'*64,out,wald=fake_wald())


def test_wv2_unknown_paper_identity_is_computed_but_separately_labeled(tmp_path,monkeypatch):
    fast_metrics(monkeypatch)
    manifest=validate_manifest(make_manifest(tmp_path,'WV2',rr_count=2,fr_count=3),tmp_path)
    datasets={s:NativeDataset(manifest,s,tmp_path) for s in ('rr','fr')}
    result=ev.evaluate_model(ToyModel(),datasets,'cpu','c'*64,wald=fake_wald())
    assert result['metadata']['status']=='PAPERSET_IDENTITY_UNVERIFIED'
    assert result['metadata']['optimizer_updates_during_evaluation']==0
    assert result['rr']['n_scenes']==2 and result['fr']['n_scenes']==3
    assert not result['metadata']['paper_comparable']
    with raises(ValueError,match='checkpoint'):ev.evaluate_model(ToyModel(),datasets,'cpu','unknown',wald=fake_wald())


class EvaluationTests(unittest.TestCase):
    pass


def _install_tests():
    for name,fn in list(globals().items()):
        if not name.startswith('test_') or not callable(fn):continue
        parameters=inspect.signature(fn).parameters
        for sensor in (list(SENSORS) if 'sensor' in parameters else [None]):
            def run(self,fn=fn,sensor=sensor,parameters=parameters):
                with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
                    kwargs={}
                    if 'tmp_path' in parameters:kwargs['tmp_path']=Path(directory)
                    if sensor is not None:kwargs['sensor']=sensor
                    if 'monkeypatch' in parameters:
                        kwargs['monkeypatch']=SimpleNamespace(setattr=lambda obj,key,value:
                            stack.enter_context(mock.patch.object(obj,key,value)))
                    fn(**kwargs)
            setattr(EvaluationTests,name+('_'+sensor if sensor else ''),run)


_install_tests()
