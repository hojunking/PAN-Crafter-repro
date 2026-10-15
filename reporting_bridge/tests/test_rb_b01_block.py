"""Synthetic-only tests for the existing-ablations wide display renderer."""
import copy
import hashlib
import json
import unittest

from reporting_bridge import rb_b01_contract as c
from reporting_bridge.rb_b01_block import (
    ALL_NATIVE_METRICS, OWNER, PRIMARY_METRICS, SECTION_IDS, render_block,
)
from reporting_bridge.rb_b01_summary import summarize
from reporting_bridge.tests.test_rb_b01_contract_summary import synthetic_entry, synthetic_curve


def prepared_fixture(servers=('s1',), *, include_stress=False, geometry=False):
    native, stress, statuses = [], [], []
    for case in c.registry():
        present = case['server'] in servers
        statuses.append(dict(case, native_status='NATIVE_EVAL_COMPLETE' if present else 'FILES_NOT_PROVIDED',
            verification_status='EVIDENCE_VERIFIED' if present else 'REMOTE_EVIDENCE_NOT_SUPPLIED',
            upload_status='UPLOAD_PENDING' if present else 'READBACK_VERIFIED',
            last_checked_at_utc='2026-10-01T00:00:00Z', errors=[]))
        if not present:
            continue
        entry = synthetic_entry(case)
        value = 2. + .1*case['repeat'] - .1*c.CASES.index(case['case_id'])
        for selected in entry['report']['selections'].values():
            selected['rr']['ergas'] = value
        native.extend(c.canonical_native(entry))
        if include_stress:
            for mode in c.MODES:
                curve = synthetic_curve(entry, mode=mode, value=value+(.2 if mode == c.MODES[0] else 0.))
                if geometry:
                    curve['report']['n_invalid_geometry'] = 40
                    curve['report']['curve'][41]['coverage_all_pan_paths'] = .98
                stress.extend(c.canonical_stress(curve))
    return dict(schema='RB_B01_PREPARED_v1', prepared_at_utc='2026-10-01T00:00:00Z',
                native_records=native, stress_records=stress, statuses=statuses,
                summary=summarize(native,stress,statuses), synthetic=True)


def seal(prepared):
    value = copy.deepcopy(prepared)
    value['payload_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                       separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


def first_panel_row(block, section, metrics=None):
    panel = next(p for p in block['panels'] if p['section_id']==section
                 and (metrics is None or p['metrics']==list(metrics)))
    return block['expected'][panel['start_row']+1]


class BlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partial = prepared_fixture()
        cls.with_stress = prepared_fixture(include_stress=True, geometry=True)

    def test_eight_ordered_sections_rectangular_raw_cells(self):
        block = render_block(self.partial)
        self.assertEqual([s['id'] for s in block['sections']], list(SECTION_IDS))
        self.assertEqual(block['column_count'],31)
        self.assertEqual(block['row_count'],len(block['expected']))
        self.assertEqual(block['new_sheet_count'],0)
        self.assertFalse(block['original_metrics_recomputed'])
        for row, expected in zip(block['rows'], block['expected']):
            self.assertEqual(len(row['values']),31)
            self.assertEqual(len(expected),31)
            for cell, value in zip(row['values'], expected):
                entered=cell['userEnteredValue']
                self.assertNotIn('formulaValue',entered)
                self.assertEqual(next(iter(entered.values())),value)

    def test_title_overflows_without_changing_sheet_layout(self):
        block=render_block(self.partial)
        self.assertEqual(block['rows'][0]['values'][0]['userEnteredFormat']['wrapStrategy'],'OVERFLOW_CELL')
        title=next(s for s in block['sections'] if s['id']=='rb01_primary')['start_row']
        self.assertEqual(block['rows'][title-1]['values'][0]['userEnteredFormat']['wrapStrategy'],'OVERFLOW_CELL')
        self.assertNotIn('merges',block)
        self.assertNotIn('columnWidths',block)

    def test_exactly_one_ownership_anchor(self):
        block=render_block(self.partial)
        owner_notes=[cell['note'] for row in block['rows'] for cell in row['values'] if OWNER in cell.get('note','')]
        self.assertEqual(len(owner_notes),1)
        marker=json.loads(owner_notes[0])
        self.assertEqual(marker['section_id'],'block')
        self.assertEqual(marker['relative_range'],f"A1:AE{block['row_count']}")

    def test_all48_slots_reserved_and_missing_metrics_empty(self):
        block=render_block(self.partial)
        self.assertEqual(len(block['native_slots']),48)
        self.assertEqual(sum(s['measured'] for s in block['native_slots']),16)
        absent=next(s for s in block['native_slots'] if '_S3_' in s['run_id'])
        row=block['expected'][absent['relative_row']-1]
        self.assertEqual(row[5],'')
        self.assertEqual(row[7:19],['']*12)
        self.assertEqual(row[19],'')
        self.assertIn('FILES_NOT_PROVIDED',row)

    def test_s1_two_not_six_alias_not_extra_students(self):
        block=render_block(self.partial)
        row=first_panel_row(block,'rb01_primary',PRIMARY_METRICS)
        self.assertEqual(row[3],2)
        self.assertEqual(row[4],6)
        self.assertEqual(row[7],2)
        self.assertEqual(row[23],4)
        secondary=first_panel_row(block,'rb01_secondary',PRIMARY_METRICS)
        self.assertEqual(secondary[3],2)
        self.assertEqual(secondary[25],2)
        self.assertFalse(secondary[26])

    def test_each_mean_sd_n_preserved_as_separate_numeric_cells(self):
        block=render_block(self.partial)
        row=first_panel_row(block,'rb01_primary',PRIMARY_METRICS)
        metric_index=PRIMARY_METRICS.index('ergas')
        mean,sd,n=row[5+3*metric_index:8+3*metric_index]
        self.assertAlmostEqual(mean,2.15)
        self.assertAlmostEqual(sd,2**.5*.05)
        self.assertEqual(n,2)
        self.assertIsInstance(mean,float)
        self.assertIsInstance(sd,float)

    def test_paired_improvement_counts_and_ties_in_notes(self):
        block=render_block(self.partial)
        row=first_panel_row(block,'rb01_paired',PRIMARY_METRICS)
        offset=4+4*PRIMARY_METRICS.index('ergas')
        self.assertAlmostEqual(row[offset],-.1)
        self.assertEqual(row[offset+2],2)
        self.assertEqual(row[offset+3],2)
        panel=next(p for p in block['panels'] if p['section_id']=='rb01_paired')
        note=json.loads(block['rows'][panel['start_row']+1]['values'][0]['note'])
        self.assertEqual(note['metric_evidence']['hqnr']['ties'],2)

    def test_full_sha_runid_source_links_preserved(self):
        block=render_block(self.partial)
        slot=block['native_slots'][0]
        row=block['expected'][slot['relative_row']-1]
        notes=block['rows'][slot['relative_row']-1]['values']
        self.assertEqual(notes[19]['note'],'a'*64)
        self.assertEqual(row[20],c.registry()[0]['run_id'])
        self.assertIn('gid=269300000&range=A2:BL2',row[23])
        self.assertTrue(row[24].startswith('RB01|'))
        self.assertEqual(len(ALL_NATIVE_METRICS),12)

    def test_geometry_numerical_flags_stay_separate(self):
        block=render_block(self.with_stress)
        total=next(row for row in block['expected'] if row[0]=='ALL')
        self.assertEqual(total[5],16)
        self.assertEqual(total[7],784)
        self.assertEqual(total[11],0)
        self.assertEqual(total[12],640)
        radius=next(row for row in block['expected'] if row[0:3]==['QFULL','A_ON',4.])
        self.assertEqual(radius[4],2)
        self.assertEqual(radius[7],0)
        self.assertEqual(radius[25],2)
        self.assertEqual(radius[26],0)
        self.assertEqual(radius[27],2)
        self.assertEqual(radius[28],80)
        paired=next(row for row in block['expected'] if row[0:3]==['QFULL','A_ON_MINUS_A_ZERO',4.])
        self.assertEqual(paired[28],160)
        self.assertEqual(paired[4],2)

    def test_native_only_and_unknown_remote_are_not_failed_runs(self):
        block=render_block(self.partial)
        s3=next(row for row in block['expected'] if row[0]=='s3')
        self.assertEqual(s3[14],'REMOTE_EVIDENCE_NOT_SUPPLIED')
        self.assertEqual(s3[11],'')
        self.assertEqual(s3[12],'')
        self.assertEqual(s3[18],8)
        self.assertEqual(s3[1],0)
        stress=next(row for row in block['expected'] if row[0:3]==['QFULL','A_ON',0.])
        self.assertEqual(stress[5:8],['','',0])
        self.assertEqual(stress[28],'')

    def test_same_cohort_late_arrivals_keep_capacity_and_slot_coordinates(self):
        first=render_block(self.partial)
        full=render_block(prepared_fixture(c.SERVERS))
        self.assertEqual(first['row_count'],full['row_count'])
        self.assertEqual([(r['run_id'],r['selection_id'],r['relative_row']) for r in first['native_slots']],
                         [(r['run_id'],r['selection_id'],r['relative_row']) for r in full['native_slots']])
        self.assertEqual(sum(s['measured'] for s in full['native_slots']),48)
        self.assertEqual(first_panel_row(full,'rb01_primary',PRIMARY_METRICS)[3],6)

    def test_new_cohort_not_forced_into_pooled_six(self):
        prepared=copy.deepcopy(self.partial)
        for row in prepared['native_records']:
            if row['repeat']==2:
                row['provenance']['evaluator_sha256']='f'*64
                row['cohort_id']=c.cohort_id(row['provenance'])
        prepared['summary']=summarize(prepared['native_records'],[],prepared['statuses'])
        block=render_block(prepared)
        self.assertEqual(block['cohort_count'],2)
        panels=[p for p in block['panels'] if p['section_id']=='rb01_primary' and p['metrics']==list(PRIMARY_METRICS)]
        self.assertEqual(len(panels),2)
        for panel in panels:
            row=block['expected'][panel['start_row']+1]
            self.assertEqual(row[3],1)
            self.assertEqual(row[6],'')
        self.assertEqual(len(block['native_slots']),48)

    def test_retries_deterministic_and_no_input_mutation(self):
        prepared=seal(self.partial)
        before=copy.deepcopy(prepared)
        delivery=dict(source_readback_verified=True,checked_at_utc=prepared['prepared_at_utc'])
        a=render_block(prepared,delivery=delivery)
        self.assertEqual(a,render_block(prepared,delivery=delivery))
        self.assertEqual(prepared,before)
        self.assertEqual(a['source_hash'],prepared['payload_sha256'])
        s1=next(row for row in a['expected'] if row[0]=='s1')
        self.assertEqual(s1[15],'SOURCE_READBACK_VERIFIED')
        self.assertEqual(s1[16],0)
        self.assertEqual(s1[17],0)

    def test_formula_injection_is_plain_string_value(self):
        prepared=copy.deepcopy(self.partial)
        prepared['native_records'][0]['review']='=IMPORTXML("example","x")'
        block=render_block(prepared)
        line=block['native_slots'][0]['relative_row']
        cell=block['rows'][line-1]['values'][29]
        self.assertEqual(cell['userEnteredValue'],{'stringValue':'=IMPORTXML("example","x")'})

    def test_stale_stats_tampered_hash_and_false_boolean_rejected(self):
        prepared=copy.deepcopy(self.partial)
        prepared['summary']['native'][0]['mean']+=.1
        with self.assertRaises(c.ContractError):render_block(prepared)
        prepared=seal(self.partial);prepared['prepared_at_utc']='wrong'
        with self.assertRaises(c.ContractError):render_block(prepared)
        with self.assertRaises(c.ContractError):render_block(self.partial,delivery={'source_readback_verified':'true'})

    def test_empty_prepared_still_reserves_registry(self):
        block=render_block(prepared_fixture(()))
        self.assertEqual(len(block['native_slots']),48)
        self.assertEqual(block['cohort_count'],0)
        primary=first_panel_row(block,'rb01_primary',PRIMARY_METRICS)
        self.assertEqual(primary[5:8],['','',0])

    def test_notes_remain_bounded(self):
        block=render_block(self.with_stress)
        self.assertLess(max(len(cell.get('note','')) for row in block['rows'] for cell in row['values']),40000)


if __name__=='__main__':
    unittest.main()
