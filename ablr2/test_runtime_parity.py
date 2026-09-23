"""Parity receipt integrity only; fixtures are explicitly not measured bridges."""
import copy
from pathlib import Path
import unittest

from ablr2.common import ROOT
from ablr2.runtime_parity import (SCHEMA,PROBE_VERSION,NUMERIC_KEYS,UNCHANGED_CORE,
    _sha,_hash,validate_runtime_parity)


def fixture():
    origin=dict(files={k:'a'*64 for k in UNCHANGED_CORE},git_release='old',torch='same')
    consumer=dict(origin,git_release='new')
    values={k:{} for k in NUMERIC_KEYS}
    values.update(teacher_forward='b'*64,teacher_gradient='c'*64,teacher_optimizer='d'*64,
        native_views='e'*64,components={f'C{i:02}':{} for i in range(17)},evaluation={'n_scenes':20},validation=1.)
    branch=dict(initial={'U':'a'*64,'A':'b'*64,'full':'c'*64},final='d'*64,
        observed=[dict(output='e'*64,gradient='f'*64,gradient_l1=1.)]*2)
    values.update(C03=branch,C07=copy.deepcopy(branch))
    def result(identity,path):return dict(schema=SCHEMA,probe_version=PROBE_VERSION,
        source_identity=identity,imported_root=str(path),server='s1',sensor='WV3',device='cpu',
        nondegenerate=True,numerics=copy.deepcopy(values),numerics_sha256=_hash(values))
    old,new=result(origin,'/fixture/old'),result(consumer,ROOT)
    receipt=dict(schema=SCHEMA,probe_version=PROBE_VERSION,producer_sha256=_sha(ROOT/'ablr2/runtime_parity.py'),
        passed=True,origin_source_identity=origin,consumer_source_identity=consumer,
        origin_root='/fixture/old',consumer_root=str(ROOT),original=old,consumer=new,
        original_sha256=_hash(old),consumer_sha256=_hash(new),
        measurements=dict(original_sha256=_hash(values),consumer_sha256=_hash(values)))
    return receipt,origin,consumer


class RuntimeParityTests(unittest.TestCase):
    def test_matching_complete_measurement_structure(self):
        receipt,old,new=fixture()
        self.assertIs(validate_runtime_parity(receipt,old,new,ROOT),receipt)

    def test_pass_only_wrong_sources_or_producer_are_rejected(self):
        receipt,old,new=fixture()
        for corrupt in ({'passed':True},dict(receipt,producer_sha256='x'*64),
                dict(receipt,origin_source_identity=new),dict(receipt,consumer_root='/other'),
                dict(receipt,original_sha256='x'*64)):
            with self.subTest(corrupt=corrupt.keys()),self.assertRaises(ValueError):
                validate_runtime_parity(corrupt,old,new,ROOT)

    def test_changed_measurements_cannot_hide_behind_pass_flag(self):
        receipt,old,new=fixture()
        receipt['consumer']['numerics']['C03']['observed'][0]['gradient']='a'*64
        with self.assertRaises(ValueError):validate_runtime_parity(receipt,old,new,ROOT)

    def test_self_consistent_but_different_behavior_or_runtime_rejected(self):
        for key in ('behavior','runtime','core'):
            receipt,old,new=fixture()
            if key=='behavior':receipt['consumer']['numerics']['validation']=2.
            if key=='runtime':new['torch']='different'
            if key=='core':new['files']=dict(new['files'],**{'ablr2/model.py':'b'*64})
            result=receipt['consumer'];result['numerics_sha256']=_hash(result['numerics'])
            receipt['consumer_sha256']=_hash(result);receipt['measurements']['consumer_sha256']=result['numerics_sha256']
            with self.subTest(key=key),self.assertRaises(ValueError):validate_runtime_parity(receipt,old,new,ROOT)

    def test_no_optimizer_update_or_zero_gradient_is_not_numerical_evidence(self):
        for mode in ('zero_gradient','unchanged_weights','missing_component'):
            receipt,old,new=fixture()
            for label in ('original','consumer'):
                value=receipt[label]['numerics']
                if mode=='zero_gradient':value['C03']['observed'][0]['gradient_l1']=0.
                if mode=='unchanged_weights':value['C03']['final']=value['C03']['initial']['full']
                if mode=='missing_component':value['components'].pop('C16')
                receipt[label]['numerics_sha256']=_hash(value)
                receipt[label+'_sha256']=_hash(receipt[label])
                receipt['measurements'][label+'_sha256']=_hash(value)
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_runtime_parity(receipt,old,new,ROOT)


if __name__=='__main__':unittest.main()
