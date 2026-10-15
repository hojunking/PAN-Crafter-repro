"""Pure synthetic grids: placement tests never authenticate or touch Sheets."""
import copy
import json
import unittest

from reporting_bridge import rb_b01_placement as p


def fixture(cells=None, rows=1020, cols=31, merges=None, protected=None, named=None):
    cells = cells or {}
    row_data = []
    for row in range(max((r for r, _ in cells), default=-1) + 1):
        values = [{} for _ in range(max((c for r, c in cells if r == row), default=-1) + 1)]
        for (r, c), cell in cells.items():
            if r == row: values[c] = copy.deepcopy(cell)
        row_data.append({'values': values})
    props = dict(sheetId=p.SHEET_ID, title=p.TITLE, index=1,
                 gridProperties=dict(rowCount=rows, columnCount=cols, frozenRowCount=2))
    sheet = dict(properties=props, merges=merges or [], protectedRanges=protected or [])
    metadata = dict(sheets=[sheet], namedRanges=named or [])
    grid = dict(sheets=[dict(properties=copy.deepcopy(props), data=[dict(startRow=0, startColumn=0, rowData=row_data)])],
                _read_bounds=dict(sheetId=p.SHEET_ID, startRowIndex=0, endRowIndex=rows,
                                  startColumnIndex=0, endColumnIndex=cols))
    return grid, metadata


def region(r0, r1, c0=0, c1=31):
    return dict(sheetId=p.SHEET_ID, startRowIndex=r0, endRowIndex=r1,
                startColumnIndex=c0, endColumnIndex=c1)


def marker(start=501, end=700, sequence=1, **extra):
    value = dict(owner=p.OWNER, campaign_id=p.CAMPAIGN, section_id='B01', schema=p.DEFAULT_SCHEMA,
                 source_hash='a' * 64, range=f"'ablations'!A{start}:AE{end}", sequence=sequence)
    value.update(extra)
    return {'note': json.dumps(value)}


class PlacementTests(unittest.TestCase):
    def plan(self, grid, metadata, height=150, **kwargs):
        return p.plan_placement(grid, metadata, height, source_hash='b' * 64, **kwargs)

    def test_used_row_not_capacity_and_two_blank_rows(self):
        grid, meta = fixture({(486, 0): {'userEnteredValue': {'stringValue': 'historical'}}})
        result = self.plan(grid, meta)
        self.assertEqual((result['start_row'], result['end_row']), (490, 639))
        self.assertEqual(result['mode'], 'APPEND')
        self.assertEqual(result['range'], "'ablations'!A490:AE639")
        self.assertEqual(result['new_sheet_count'], 0)
        self.assertEqual(result['append_grid_rows'], 0)

    def test_empty_sheet_still_leaves_two_rows(self):
        grid, meta = fixture()
        self.assertEqual(self.plan(grid, meta)['start_row'], 3)

    def test_all_columns_note_validation_chips_effective_and_zero(self):
        fields = [dict(note='reserved'), dict(dataValidation={'condition': {'type': 'BOOLEAN'}}),
                  dict(chipRuns=[{'startIndex': 0, 'chip': {'richLinkProperties': {'uri': 'https://example.com'}}}]),
                  dict(effectiveValue={'numberValue': 0}), dict(userEnteredValue={'boolValue': False})]
        for cell in fields:
            with self.subTest(cell=cell):
                grid, meta = fixture({(800, 39): cell}, cols=40)
                self.assertEqual(self.plan(grid, meta)['start_row'], 804)

    def test_empty_format_only_grid_does_not_consume_every_styled_row(self):
        grid, meta = fixture({(1019, 30): {'userEnteredFormat': {'backgroundColor': {'red': 1}}}})
        self.assertEqual(self.plan(grid, meta)['start_row'], 3)
        self.assertIn('1020:31', p.outside_owned_snapshot(grid, meta)['cells'])

    def test_complete_bounded_read_required(self):
        grid, meta = fixture()
        for bounds in (None, dict(grid['_read_bounds'], endColumnIndex=1), dict(grid['_read_bounds'], endRowIndex=500)):
            mutated = dict(grid, _read_bounds=bounds)
            with self.assertRaisesRegex(p.PlacementError, 'FULL_GRID_READ_REQUIRED'):
                self.plan(mutated, meta)

    def test_fixed_title_id(self):
        grid, meta = fixture(); meta['sheets'][0]['properties']['sheetId'] = 123
        with self.assertRaisesRegex(p.PlacementError, 'TARGET_MISSING_OR_SCHEMA_CONFLICT'):
            self.plan(grid, meta)

    def test_merges_named_protected_and_reserved_count_blank_extent(self):
        for kind in ('merge', 'named', 'protected', 'reserved'):
            with self.subTest(kind=kind):
                grid, meta = fixture()
                if kind == 'merge': meta['sheets'][0]['merges'] = [region(600, 650)]
                if kind == 'named': meta['namedRanges'] = [dict(name='future_manual', range=region(600, 650))]
                if kind == 'protected': meta['sheets'][0]['protectedRanges'] = [dict(range=region(600, 650))]
                if kind == 'reserved': meta['reservedRanges'] = [dict(reason='user future section', range=region(600, 650))]
                self.assertEqual(self.plan(grid, meta)['start_row'], 653)

    def test_unbounded_whole_column_reservation_blocks_append(self):
        grid, meta = fixture(named=[dict(name='forever', range=dict(sheetId=p.SHEET_ID, startColumnIndex=0, endColumnIndex=1))])
        with self.assertRaisesRegex(p.PlacementError, 'UNBOUNDED_RESERVED_RANGE'):
            self.plan(grid, meta)

    def test_bounded_choosecols_single_row_not_unbounded_spill(self):
        formula = "=CHOOSECOLS('_source'!$Z$48:$AG$48,1,2,4)"
        grid, meta = fixture({(486, 0): {'userEnteredValue': {'formulaValue': formula}}})
        self.assertEqual(self.plan(grid, meta)['start_row'], 490)
        self.assertEqual(p.discover(grid, meta)['array_formula_extents'][0]['height_bound'], 1)

    def test_iferror_choosecols_bounded_rows(self):
        formula = '=IFERROR(CHOOSECOLS(\'source\'!Z48:AG50,1,2),"")'
        grid, meta = fixture({(486, 0): {'userEnteredValue': {'formulaValue': formula}}})
        self.assertEqual(self.plan(grid, meta)['start_row'], 492)

    def test_future_blank_array_spill_counts_proven_bound(self):
        grid, meta = fixture({(99, 5): {'userEnteredValue': {'formulaValue': '=SEQUENCE(500,1)'}}})
        self.assertEqual(self.plan(grid, meta)['start_row'], 602)

    def test_dynamic_array_requires_reserved_extent(self):
        grid, meta = fixture({(5, 2): {'userEnteredValue': {'formulaValue': '=FILTER(Source!A:A,Source!B:B=1)'}}})
        with self.assertRaisesRegex(p.PlacementError, 'ARRAY_SPILL_EXTENT_REQUIRED'):
            self.plan(grid, meta)
        meta['arraySpillRanges'] = [dict(range=region(5, 100, 2, 3), reason='externally established bounded spill')]
        self.assertEqual(self.plan(grid, meta)['start_row'], 103)

    def test_unknown_transposed_split_query_and_pivot_require_extent(self):
        for cell in (
                {'userEnteredValue': {'formulaValue': '=TRANSPOSE(SPLIT(A1,","))'}},
                {'userEnteredValue': {'formulaValue': '=QUERY(Source!A1:B1,"select sum(A)",0)'}},
                {'pivotTable': {'source': region(0, 10)}}):
            with self.subTest(cell=cell):
                grid, meta = fixture({(200, 0): cell})
                with self.assertRaisesRegex(p.PlacementError, 'ARRAY_SPILL_EXTENT_REQUIRED'):
                    self.plan(grid, meta)

    def test_title_without_owner_marker_cannot_create_duplicate(self):
        for value_field in ('userEnteredValue', 'effectiveValue'):
            for title in (p.BLOCK_TITLE, p.BLOCK_TITLE + ' — continuation 2'):
                grid, meta = fixture({(500, 0): {value_field: {'stringValue': title}}})
                with self.assertRaisesRegex(p.PlacementError, 'title exists without'):
                    self.plan(grid, meta)

    def test_tables_basic_filter_and_filter_view_reserve_rows(self):
        for key in ('tables', 'basicFilter', 'filterViews'):
            grid, meta = fixture()
            value = dict(range=region(600, 650))
            meta['sheets'][0][key] = value if key == 'basicFilter' else [value]
            self.assertEqual(self.plan(grid, meta)['start_row'], 653)

    def test_owner_reuses_reserved_empty_tail(self):
        grid, meta = fixture({(500, 0): marker(), (800, 20): {'note': 'another user later'}})
        result = self.plan(grid, meta)
        self.assertEqual(result['mode'], 'REUSE')
        self.assertEqual((result['start_row'], result['end_row']), (501, 700))
        self.assertEqual(result['capacity_rows'], 200)
        self.assertEqual(result['owner_marker']['source_hash'], 'b' * 64)

    def test_insufficient_space_uses_full_continuation_below_other_content(self):
        grid, meta = fixture({(500, 0): marker(), (799, 25): {'userEnteredValue': {'stringValue': 'other user'}}})
        result = self.plan(grid, meta, height=250)
        self.assertEqual(result['mode'], 'CONTINUATION')
        self.assertEqual((result['start_row'], result['end_row']), (803, 1052))
        self.assertEqual(result['append_grid_rows'], 32)
        self.assertEqual(result['prior_ranges'], ["'ablations'!A501:AE700"])
        self.assertEqual(result['owner_marker']['continuation_of'], "'ablations'!A501:AE700")
        self.assertEqual(result['owner_marker']['previous_range'], "'ablations'!A501:AE700")

    def test_existing_continuation_latest_only_reused(self):
        primary = "'ablations'!A501:AE700"
        grid, meta = fixture({(500, 0): marker(), (802, 0): marker(803, 1102, 2,
            continuation_of=primary, previous_range=primary)}, rows=1200)
        result = self.plan(grid, meta, height=250)
        self.assertEqual(result['mode'], 'REUSE')
        self.assertEqual(result['start_row'], 803)
        self.assertEqual(result['prior_ranges'], [primary])
        self.assertEqual(result['owner_marker']['sequence'], 2)

    def test_duplicate_primary_broken_chain_and_schema_fail(self):
        cases = [
            {(500, 0): marker(), (802, 0): marker(803, 1000)},
            {(500, 0): marker(), (802, 0): marker(803, 1000, 3, continuation_of="'ablations'!A501:AE700")},
            {(500, 0): marker(schema='OTHER_SCHEMA')},
            {(500, 0): marker(campaign_id='OTHER_CAMPAIGN')},
        ]
        for cells in cases:
            with self.subTest(cells=cells):
                grid, meta = fixture(cells)
                with self.assertRaises(p.PlacementError): self.plan(grid, meta)

    def test_marker_not_at_declared_start_fails(self):
        grid, meta = fixture({(501, 0): marker()})
        with self.assertRaisesRegex(p.PlacementError, 'marker not at range start'):
            self.plan(grid, meta)

    def test_user_protection_inside_owned_region_is_conflict(self):
        grid, meta = fixture({(500, 0): marker()}, protected=[dict(range=region(550, 551, 0, 1))])
        with self.assertRaisesRegex(p.PlacementError, 'PLACEMENT_RESERVED_RANGE_CONFLICT'):
            self.plan(grid, meta)

    def test_cross_boundary_merge_blocked_but_own_merge_allowed(self):
        grid, meta = fixture({(500, 0): marker()}, merges=[region(500, 501, 0, 31)])
        self.assertEqual(self.plan(grid, meta)['mode'], 'REUSE')
        meta['sheets'][0]['merges'] = [region(699, 701)]
        with self.assertRaisesRegex(p.PlacementError, 'PLACEMENT_RESERVED_RANGE_CONFLICT'):
            self.plan(grid, meta)

    def test_outside_snapshot_preserves_prior_block_notes_formats_and_user_cells(self):
        primary = "'ablations'!A501:AE700"
        cells = {(500, 0): marker(), (799, 30): {'note': 'manual', 'userEnteredFormat': {'wrapStrategy': 'WRAP'}},
                 (802, 0): marker(803, 1102, 2, continuation_of=primary, previous_range=primary)}
        grid, meta = fixture(cells, rows=1200)
        result = self.plan(grid, meta, height=250)
        snapshot = p.outside_owned_snapshot(grid, meta, result['range'])
        self.assertIn('501:1', snapshot['cells'])
        self.assertIn('800:31', snapshot['cells'])
        self.assertNotIn('803:1', snapshot['cells'])
        self.assertEqual(snapshot['snapshot_sha256'], result['protected_outside_sha256'])
        changed = copy.deepcopy(grid)
        changed['sheets'][0]['data'][0]['rowData'][799]['values'][30]['note'] = 'changed'
        self.assertNotEqual(p.outside_owned_snapshot(changed, meta, result['range'])['snapshot_sha256'], snapshot['snapshot_sha256'])

    def test_grid_extension_does_not_change_outside_snapshot(self):
        grid, meta = fixture({(486, 0): {'note': 'existing'}}, rows=600)
        placement = self.plan(grid, meta, height=300)
        before = p.outside_owned_snapshot(grid, meta, placement['range'])
        after_grid, after_meta = fixture({(486, 0): {'note': 'existing'}}, rows=789)
        after = p.outside_owned_snapshot(after_grid, after_meta, placement['range'])
        self.assertEqual(before['snapshot_sha256'], after['snapshot_sha256'])


if __name__ == '__main__':
    unittest.main()
