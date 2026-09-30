"""Synthetic fixtures only. Never upload these values to a real spreadsheet."""
import copy
import math
import unittest

from reporting_bridge import rb_b01_contract as c
from reporting_bridge.rb_b01_summary import summarize, stats


def synthetic_entry(case=None):
    case = dict(case or c.registry()[0])
    source = dict(content_sha256='1'*64, torch='test', cuda='test', cudnn=0,
                  numpy='test', scipy='test', skimage='test', tf32_matmul=False, tf32_cudnn=False)
    context = dict(case=case, source_identity=source, evaluator_identity={'synthetic': True},
                   binding_common_sha256='2'*64, data_content_identity={'synthetic': True})
    selected = dict(selection_id=c.SELECTIONS[0], update=50000, checkpoint_sha256='a'*64,
                    rr=dict(ergas=2., scc=.9, sam=3., psnr=30., ssim=.8, q8=.7),
                    fr=dict(hqnr=.95, d_s=.03, d_lambda=.02), alias_of=None)
    alias = dict(copy.deepcopy(selected), selection_id=c.SELECTIONS[1], alias_of=c.SELECTIONS[0])
    report = dict(schema='PANDA_RB01_NATIVE_v1', complete=True, context=context,
                  selections={c.SELECTIONS[0]: selected, c.SELECTIONS[1]: alias}, elapsed_seconds=72.,
                  payload_sha256='3'*64, completed_at_utc='2026-09-30T00:00:00+00:00')
    return dict(case=case, report=report,
        config=dict(model_args=dict(hidden_size=104, depth=[1,2,2]), learning_rate=.0001,
                    panda_rb=dict(input_layout='PLH', alpha=1., beta=.1, lambda_E=.002, aligner_lr=.000003)),
        binding=dict(q_ref=.4, tau_R=.01, teacher_checkpoint_sha256='4'*64),
        training_status=dict(timings=dict(train=3600.)),
        provenance=dict(raw_map_identity=dict(q='5'*64,e_bar='6'*64)),
        verification=dict(status='EVIDENCE_VERIFIED', synthetic=True))


def synthetic_curve(entry, mode='A_ON', value=2.):
    points = []
    for i in range(49):
        radius = 0. if i == 0 else c.RADII[(i-1)//8+1]
        angle = None if i == 0 else ((i-1)%8)*45
        rad = math.radians(angle or 0)
        point = dict(id=f'D{i:03d}', radius_hr=radius, angle_degrees=angle,
                     dy=radius*math.sin(rad), dx=radius*math.cos(rad), n_scenes=20, n_failures=0,
                     **{key: value for key in c.STRESS_METRICS})
        point['coverage_all_pan_paths'] = 1.
        for key in c.STRESS_METRICS[:4]:
            point[key+'_scene_std'] = .5
            point['delta_from_zero_'+key] = 0.
        points.append(point)
    report = dict(schema='PANDA_RB02_CURVE_v1', complete=True,
                  identity=dict(context=copy.deepcopy(entry['report']['context']), mode=mode,
                                source_selection=c.SELECTIONS[0], update=50000,
                                checkpoint_sha256='a'*64, grid_sha256='b'*64),
                  n_scenes=20, n_shifts=49, n_observations=980, n_numerical_failures=0,
                  n_invalid_geometry=0, curve=points, payload_sha256='c'*64,
                  completed_at_utc='2026-09-30T01:00:00+00:00')
    return dict(entry, report=report)


class ContractTests(unittest.TestCase):
    def test_headers_and_registry(self):
        self.assertEqual(len(c.NATIVE_HEADERS),64)
        self.assertEqual(len(c.STRESS_HEADERS),43)
        self.assertEqual(c.NATIVE_HEADERS[59], 'Result_ID')
        self.assertEqual(len(c.registry()),24)
        self.assertEqual(len({row['run_id'] for row in c.registry()}),24)
        self.assertEqual(c.registry()[16]['seed'],9281501)

    def test_header_drift_blocks(self):
        headers=list(c.NATIVE_HEADERS); headers[3]='HQNR'
        with self.assertRaises(c.ContractError): c.validate_headers(headers)

    def test_native_slots_and_alias(self):
        slots=[c.source_row(repeat,case,selector) for repeat in (1,2) for case in c.CASES for selector in c.SELECTIONS]
        self.assertEqual(slots,list(range(2,18)))
        records=c.canonical_native(synthetic_entry())
        self.assertEqual(len(records),2)
        self.assertNotEqual(c.native_result_id(records[0]),c.native_result_id(records[1]))
        self.assertEqual(records[1]['alias_of'], c.SELECTIONS[0])

    def test_bad_registry_blocks(self):
        entry=synthetic_entry(); entry['case']['seed']=999
        with self.assertRaises(c.ContractError): c.canonical_native(entry)

    def test_numbers_reject_nonfinite_bool_and_string(self):
        for value in (True,False,float('nan'),float('inf'),'0','2.0'):
            with self.subTest(value=value),self.assertRaises(c.ContractError): c.finite_number(value)
        self.assertEqual(c.finite_number(0),0)
        self.assertIsNone(c.finite_number(None,optional=True))

    def test_optional_blank_real_zero_bool_and_units(self):
        records=c.canonical_native(synthetic_entry())
        records[0]['metrics']['jqm']=0.
        row=dict(zip(c.NATIVE_HEADERS,c.build_native_row(records[0],gid=123)))
        self.assertEqual(row['JQM↑'],0.)
        self.assertEqual(row['RMSE↓'],'')
        self.assertEqual(row['Infer(ms)'],'')
        self.assertEqual(row['Params(M)'],'')
        self.assertIs(row['Test_aware'],False)
        self.assertEqual(row['Train(h)'],1.)
        self.assertEqual(row['Eval(h)'],.02)
        self.assertIn('gid=123&range=A2:BL2',row['Source_URL'])
        self.assertIn('BOTH selectors',row['Cost_scope'])

    def test_alias_conflict_and_no_q4_fallback(self):
        for field,update in [('fr',{'ds':.99}),('rr',{'q2n':.99})]:
            entry=synthetic_entry();entry['report']['selections'][c.SELECTIONS[0]][field].update(update)
            with self.subTest(field=field), self.assertRaises(c.ContractError): c.canonical_native(entry)
        entry=synthetic_entry();rr=entry['report']['selections'][c.SELECTIONS[0]]['rr'];rr['q4']=rr.pop('q8')
        with self.assertRaises(c.ContractError):c.canonical_native(entry)

    def test_same_sha_alias_metrics_must_match(self):
        entry=synthetic_entry();entry['report']['selections'][c.SELECTIONS[1]]['rr']['ergas']=2.1
        with self.assertRaises(c.ContractError):c.canonical_native(entry)

    def test_idempotency_and_same_logical_key_conflict(self):
        records=c.canonical_native(synthetic_entry())
        self.assertEqual(len(c.assert_no_conflicts(records+copy.deepcopy(records))),2)
        different=copy.deepcopy(records[0]);different['checkpoint_sha256']='f'*64
        with self.assertRaises(c.ContractError):c.assert_no_conflicts([records[0],different])
        transport=dict(records[0],readback='READBACK_VERIFIED')
        self.assertEqual(len(c.assert_no_conflicts([records[0],transport])),1)

    def test_stress_width_grid_failure_and_slots(self):
        entry=synthetic_curve(synthetic_entry());entry['report']['n_invalid_geometry']=40
        rows=c.canonical_stress(entry)
        self.assertEqual(len(rows),49)
        self.assertEqual(c.stress_source_row(rows[0]),2)
        self.assertEqual(c.stress_source_row(rows[-1]),50)
        row=dict(zip(c.STRESS_HEADERS,c.build_stress_row(rows[0])))
        self.assertEqual(row['Curve_invalid_geometry'],40)
        self.assertEqual(row['N_failures'],0)
        self.assertEqual(row['Angle_deg'],'')
        self.assertEqual(row['Status'],'COLLECTED_WITH_FAILURES')
        self.assertEqual(row['ROI'],'32:-32 / fixed192')

    def test_stress_null_is_blank_not_zero(self):
        entry=synthetic_curve(synthetic_entry());entry['report']['curve'][4]['ergas']=None
        point=c.canonical_stress(entry)[4]
        row=dict(zip(c.STRESS_HEADERS,c.build_stress_row(point)))
        self.assertEqual(row['ERGAS'],'')
        self.assertIn('MISSING',row['Review'])

    def test_stress_alias_and_wrong_grid_fail(self):
        entry=synthetic_curve(synthetic_entry());entry['report']['curve'][1]['angle_deg']=45
        with self.assertRaises(c.ContractError):c.canonical_stress(entry)
        entry=synthetic_curve(synthetic_entry());entry['report']['curve'][1]['dx']=9.
        with self.assertRaises(c.ContractError):c.canonical_stress(entry)

    def test_status_nested_missing_not_complete(self):
        status=dict(c.registry()[0],training_status='FILE_NOT_FOUND',native_status='FILE_NOT_FOUND',
                    native_selectors=0,curve_status={mode:dict(status='FILE_NOT_FOUND',shifts=0) for mode in c.MODES},
                    errors=[{'error':'not copied'}])
        row=dict(zip(c.STATUS_HEADERS,c.build_status_row(status)))
        self.assertEqual(row['Actual_updates'],'')
        self.assertEqual(row['Selector_count'],0)
        self.assertEqual(row['A_ON_status'],'FILE_NOT_FOUND')
        self.assertIn('not copied',row['Error'])

    def test_formula_looking_provenance_stays_text(self):
        record=c.canonical_native(synthetic_entry())[0];record['review']='=SUM(A1:A2)'
        self.assertEqual(c.build_native_row(record)[61],'=SUM(A1:A2)')


class SummaryTests(unittest.TestCase):
    def test_ddof1_and_n1(self):
        self.assertIsNone(stats([1.])['sample_std'])
        self.assertAlmostEqual(stats([1.,3.])['sample_std'],math.sqrt(2))
        with self.assertRaises(c.ContractError):stats([float('nan')])

    def test_alias_n_and_partial_s1_campaign(self):
        entries=[synthetic_entry(row) for row in c.registry() if row['server']=='s1']
        rows=[record for entry in entries for record in c.canonical_native(entry)]
        result=summarize(rows,[])
        self.assertEqual(result['counts']['verified_students'],8)
        self.assertEqual(result['counts']['native_observations'],16)
        panel=next(p for p in result['native'] if p['case_id']=='QFULL' and p['selection_id']==c.SELECTIONS[0] and p['metric']=='ergas')
        self.assertEqual(panel['n'],2)
        self.assertEqual(panel['missing_n'],4)
        self.assertFalse(panel['full_six'])
        self.assertEqual(result['server_status'][1]['verified_students'],0)

    def test_paired_case_difference_with_improvements(self):
        entries=[synthetic_entry(row) for row in c.registry()[:2]]
        rows=[record for entry in entries for record in c.canonical_native(entry)]
        for record in rows:
            if record['case_id']=='QMEAN':record['metrics']['ergas']=1.5
        panels=summarize(rows,[])['paired']
        p=next(p for p in panels if p['case_id']=='QMEAN' and p['selection_id']==c.SELECTIONS[0] and p['metric']=='ergas')
        self.assertEqual(p['n'],1);self.assertEqual(p['mean'],-.5);self.assertEqual(p['improved'],1)
        self.assertEqual(len(p['member_pairs']),1)

    def test_new_evaluator_splits_cohorts(self):
        first=c.canonical_native(synthetic_entry(c.registry()[0]))
        entry=synthetic_entry(c.registry()[4]);entry['report']['context']['evaluator_identity']={'different':True}
        second=c.canonical_native(entry)
        result=summarize(first+second,[])
        self.assertEqual(result['counts']['cohort_count'],2)
        panels=[p for p in result['native'] if p['case_id']=='QFULL' and p['metric']=='ergas' and p['selection_id']==c.SELECTIONS[0]]
        self.assertEqual([p['n'] for p in panels],[1,1])

    def test_raw_maps_split_but_case_specific_transforms_do_not(self):
        first = synthetic_entry(c.registry()[0])
        second = synthetic_entry(c.registry()[4])
        first['provenance']['q_weight_map_sha256'] = '7'*64
        second['provenance']['q_weight_map_sha256'] = '8'*64
        a = c.canonical_native(first)
        b = c.canonical_native(second)
        self.assertEqual(a[0]['cohort_id'], b[0]['cohort_id'])
        second['provenance']['raw_map_identity']['e_bar'] = '9'*64
        changed = c.canonical_native(second)
        self.assertNotEqual(a[0]['cohort_id'], changed[0]['cohort_id'])

    def test_unknown_cohort_fails_instead_of_pooling(self):
        record = c.canonical_native(synthetic_entry())[0]
        del record['provenance']['teacher_sha256']
        with self.assertRaises(c.ContractError): summarize([record], [])

    def test_radius_directions_and_modes_not_students(self):
        entry=synthetic_entry();native=c.canonical_native(entry)
        stress=c.canonical_stress(synthetic_curve(entry,value=3.))+c.canonical_stress(synthetic_curve(entry,mode=c.MODES[1],value=2.))
        result=summarize(native,stress)
        panel=next(p for p in result['stress_radius'] if p['mode']=='A_ON_MINUS_A_ZERO' and p['radius_hr']==1. and p['metric']=='ergas')
        self.assertEqual(panel['n'],1);self.assertEqual(panel['mean'],1.)
        self.assertEqual(panel['direction_count'],8);self.assertIsNone(panel['sample_std'])
        self.assertEqual(len(panel['student_values'][0]['member_pairs']),8)

    def test_invalid_direction_explicit_failed_student(self):
        entry=synthetic_entry();native=c.canonical_native(entry)
        curve=synthetic_curve(entry);curve['report']['curve'][1]['coverage_all_pan_paths']=.98
        stress=c.canonical_stress(curve)
        result=summarize(native,stress)
        panel=next(p for p in result['stress_radius'] if p['mode']=='A_ON' and p['radius_hr']==.25 and p['metric']=='ergas')
        self.assertEqual(panel['n'],0);self.assertEqual(panel['failed_n'],1);self.assertEqual(panel['missing_n'],5)
        self.assertIsNone(panel['mean'])
        self.assertEqual(stress[1]['ergas'],2.)

    def test_missing_curve_point_or_native_pair_is_error(self):
        entry=synthetic_entry();stress=c.canonical_stress(synthetic_curve(entry))
        with self.assertRaises(c.ContractError):summarize(c.canonical_native(entry),stress[:-1])
        with self.assertRaises(c.ContractError):summarize([],stress)


if __name__=='__main__':
    unittest.main()
