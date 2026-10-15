"""Existing-sheet-only B01 presentation; no research-runtime imports or writes.

The v1 collector, immutable source upsert and numerical summaries remain intact.
Only the presentation destination and its ownership/snapshot policy change.
"""
from __future__ import annotations

import copy
import json

from reporting_bridge import rb_b01_sheets as legacy

POLICY = 'existing_sheets_append_only'
OWNER = 'B01_EXISTING_SHEETS_APPEND_ONLY_v2'
PRESERVED_PUBLIC = ('RB-B01', 'RB02-curves')
FIXED = dict(legacy.FIXED_IDS, **{
    '_rb01_s1': 269300000, '_rb01_s3': 269300001, '_rb01_s5': 269300002,
    '_rb02_points': 269300003, '_rb_b01_status': 269300004,
    'RB02-curves': 269300005, 'RB-B01': 269300006})
BridgeError = legacy.BridgeError


def _cells(grid, sheet_id):
    result = {}
    for sheet in grid.get('sheets', []):
        if sheet['properties']['sheetId'] != sheet_id:
            continue
        for data in sheet.get('data', []):
            for i, row in enumerate(data.get('rowData', [])):
                for j, cell in enumerate(row.get('values', [])):
                    result[(data.get('startRow', 0)+i, data.get('startColumn', 0)+j)] = cell
    return result


def _content_hash(cells, start_row, end_row, width):
    """Google canonicalizes RAW empty strings to absent CellData. Ignore that.

    Formatting is preserved on updates, so manual formatting is not overwritten.
    The first note contains this hash and therefore cannot hash itself.
    """
    result = {}
    for (r, c), cell in cells.items():
        if not start_row-1 <= r < end_row or not 0 <= c < width:
            continue
        value = cell.get('userEnteredValue', {})
        entry = {}
        if value and value != {'stringValue': ''}:
            # Sheets may serialize an integral double (0.0) as JSON integer 0.
            # Normalize only the seal representation, never the stored metric.
            entry['userEnteredValue'] = ({'numberValue': float(value['numberValue'])}
                                         if 'numberValue' in value else value)
        if cell.get('note') and (r, c) != (start_row-1, 0):
            entry['note'] = cell['note']
        if entry:
            result[f'{r-start_row+1}:{c}'] = entry
    return legacy.digest(result)


def _exclude(grid, placement):
    result = copy.deepcopy(legacy._without_layout(grid))
    if not placement:
        return result
    sheet = result.get(str(FIXED['ablations']), {})
    for key in list(sheet.get('cells', {})):
        r, c = map(int, key.split(':'))
        if placement['start_row']-1 <= r < placement['end_row'] and 0 <= c < placement['end_col']:
            del sheet['cells'][key]
    # The writer does not create/change merges, so never exclude merge metadata.
    return result


class AppendOnlySheetsBridge(legacy.SheetsBridge):
    output_policy = POLICY

    def _tabs(self):
        metadata = self.adapter.metadata()
        if metadata.get('spreadsheetId', self.spreadsheet_id) != self.spreadsheet_id:
            raise BridgeError('SCHEMA_CONFLICT: returned spreadsheet ID')
        tabs = {s['properties']['title']: s['properties'] for s in metadata['sheets']}
        for title, sid in FIXED.items():
            if title not in tabs:
                raise BridgeError('TARGET_MISSING: ' + title)
            if tabs[title]['sheetId'] != sid:
                raise BridgeError('SCHEMA_CONFLICT: fixed tab ID/title ' + title)
        return metadata, tabs

    def _full_ablations(self, metadata=None):
        metadata = metadata or self.adapter.metadata()
        shape = next(s['properties']['gridProperties'] for s in metadata['sheets']
                     if s['properties']['sheetId'] == FIXED['ablations'])
        grid = self.adapter.grid([f"'ablations'!A1:{legacy.column(shape['columnCount'])}{shape['rowCount']}"])
        grid['_read_bounds'] = dict(sheetId=FIXED['ablations'], startRowIndex=0,
            endRowIndex=shape['rowCount'], startColumnIndex=0, endColumnIndex=shape['columnCount'])
        return grid

    @staticmethod
    def _require_union(cell):
        formula = cell.get('userEnteredValue', {}).get('formulaValue', '')
        if not formula or legacy.patch_formula(formula) != formula:
            raise BridgeError('SCHEMA_CONFLICT: existing native union missing; v2 never edits _records!A2')

    def inspect(self):
        snapshot = super().inspect()
        self._require_union(snapshot['records_a2'])
        snapshot.update(output_policy=POLICY,
            preserved_public=self.adapter.grid([legacy.quoted(t) for t in PRESERVED_PUBLIC]),
            ablations=self._full_ablations(snapshot['metadata']))
        snapshot.pop('snapshot_sha256', None)
        snapshot['snapshot_sha256'] = legacy.digest(snapshot)
        return snapshot

    def _block_plan(self, prepared):
        from reporting_bridge.rb_b01_block import render_block
        from reporting_bridge.rb_b01_placement import plan_placement
        block = render_block(prepared, delivery={'source_readback_verified': True})
        metadata, _ = self._tabs()
        grid = self._full_ablations(metadata)
        live = _cells(grid, FIXED['ablations'])
        # Validate all markers BEFORE choosing reuse versus continuation. A
        # manually shortened range must not manufacture a new continuation.
        for cell in live.values():
            note = cell.get('note', '')
            if OWNER not in note:
                continue
            try:
                marker = json.loads(note)
            except (TypeError, ValueError) as exc:
                raise BridgeError('SCHEMA_CONFLICT: invalid ownership note') from exc
            if not isinstance(marker, dict) or marker.get('owner') != OWNER:
                raise BridgeError('SCHEMA_CONFLICT: invalid ownership note')
            unsigned = {k: v for k, v in marker.items() if k != 'marker_sha256'}
            if marker.get('marker_sha256') != legacy.digest(unsigned):
                raise BridgeError('SCHEMA_CONFLICT: ownership note was manually edited')
        placement = plan_placement(grid, metadata, block['row_count'], width=block['column_count'],
            source_hash=block['source_hash'], schema=block['schema'])
        if placement['mode'] == 'REUSE':
            sheet = next(s for s in metadata['sheets'] if s['properties']['sheetId'] == FIXED['ablations'])
            for merge in sheet.get('merges', []):
                if (merge.get('startRowIndex', 0) < placement['end_row'] and
                        merge['endRowIndex'] > placement['start_row']-1 and
                        merge.get('startColumnIndex', 0) < placement['end_col']):
                    raise BridgeError('SCHEMA_CONFLICT: manual merge inside owned block')
        if placement['mode'] == 'REUSE':
            note = live.get((placement['start_row']-1, 0), {}).get('note', '')
            marker = json.loads(note)
            unsigned = {k: v for k, v in marker.items() if k != 'marker_sha256'}
            if marker.get('marker_sha256') != legacy.digest(unsigned):
                raise BridgeError('SCHEMA_CONFLICT: ownership note was manually edited')
            actual_hash = _content_hash(live, placement['start_row'], placement['end_row'], placement['end_col'])
            if marker.get('content_sha256') != actual_hash:
                raise BridgeError('SCHEMA_CONFLICT: owned block was manually edited; no overwrite')
        rows = copy.deepcopy(block['rows'])
        expected = copy.deepcopy(block['expected'])
        capacity = placement['end_row']-placement['start_row']+1
        while len(rows) < capacity:
            rows.append({'values': [{} for _ in range(block['column_count'])]})
            expected.append([''] * block['column_count'])
        desired = {(placement['start_row']-1+i, j): cell
                   for i, row in enumerate(rows) for j, cell in enumerate(row['values'])}
        content_hash = _content_hash(desired, placement['start_row'], placement['end_row'], placement['end_col'])
        marker = dict(placement['owner_marker'], content_sha256=content_hash)
        marker['marker_sha256'] = legacy.digest(marker)
        rows[0]['values'][0]['note'] = json.dumps(marker, sort_keys=True, ensure_ascii=False)
        placement['owner_marker'] = marker
        placement['marker_note'] = rows[0]['values'][0]['note']
        same = (placement['mode'] == 'REUSE' and
                _content_hash(live, placement['start_row'], placement['end_row'], placement['end_col']) == content_hash and
                live.get((placement['start_row']-1, 0), {}).get('note') == placement['marker_note'])
        return dict(block=block, placement=placement, rows=rows, expected=expected,
                    action='NO_OP' if same else placement['mode'], content_sha256=content_hash)

    def plan(self, prepared, snapshot):
        self._require_union(snapshot['records_a2'])
        result = super().plan(prepared, snapshot)
        if result['create_tabs'] or result['formula_changed']:
            raise BridgeError('SCHEMA_CONFLICT: v2 requires pre-existing sources/union')
        target = self._block_plan(prepared)
        _, tabs = self._tabs()
        result.update(output_policy=POLICY, new_sheet_count=0, existing_sheet_count=len(tabs),
            existing_tab_ids={t: p['sheetId'] for t, p in tabs.items()},
            placement=target['placement'], block_action=target['action'],
            append_ranges=[target['placement']['range']] if target['placement']['mode'] != 'REUSE' else [],
            update_ranges=[target['placement']['range']] if target['placement']['mode'] == 'REUSE' else [],
            protected_ranges=['paper', '유의미한결과', 'RB-B01', 'RB02-curves',
                              'ablations outside ' + target['placement']['range'], '_records!A2'],
            source_payload_sha256=target['block']['source_hash'])
        return result

    def setup(self, plan, snapshot, *, apply=False, writer_server='s1'):
        """Compatibility command: validate only. Never creates or migrates anything."""
        self._writer(writer_server)
        self._check_snapshot(snapshot)
        if (plan.get('output_policy') != POLICY or plan.get('new_sheet_count') != 0 or
                plan.get('create_tabs') or plan.get('formula_changed') or
                plan.get('snapshot_sha256') != snapshot['snapshot_sha256']):
            raise BridgeError('SCHEMA_CONFLICT: not an existing-sheets-only plan')
        self._regression(snapshot)
        return dict(status='EXISTING_TARGETS_VERIFIED', output_policy=POLICY,
                    new_sheet_count=0, created_tabs=[], formula_changed=False,
                    writes=0, placement=plan['placement'])

    def _regression(self, snapshot, placement=None):
        metadata, tabs = self._tabs()
        previous = snapshot['metadata']['sheets']
        old_props = {s['properties']['sheetId']: s['properties'] for s in previous}
        new_props = {p['sheetId']: p for p in tabs.values()}
        if old_props.keys() != new_props.keys():
            raise BridgeError('HISTORICAL_REGRESSION: worksheet count/IDs changed')
        for sid, prop in old_props.items():
            current = copy.deepcopy(new_props[sid])
            old = copy.deepcopy(prop)
            if sid == FIXED['ablations']:
                current['gridProperties'].pop('rowCount', None)
                old['gridProperties'].pop('rowCount', None)
                if new_props[sid]['gridProperties']['rowCount'] < prop['gridProperties']['rowCount']:
                    raise BridgeError('HISTORICAL_REGRESSION: ablations grid shrunk')
            if current != old:
                raise BridgeError('HISTORICAL_REGRESSION: existing sheet properties')
        for field in ('namedRanges',):
            if metadata.get(field, []) != snapshot['metadata'].get(field, []):
                raise BridgeError('HISTORICAL_REGRESSION: ' + field)
        for old, now in zip(previous, metadata['sheets']):
            for field in ('merges','protectedRanges','basicFilter','tables'):
                if old.get(field, []) != now.get(field, []):
                    raise BridgeError('HISTORICAL_REGRESSION: ' + field)
        if self.adapter.values("'_records'!A1:BL1") != [snapshot['headers']]:
            raise BridgeError('HISTORICAL_REGRESSION: canonical header')
        a2 = legacy.cell_from_grid(self.adapter.grid(["'_records'!A2"]))
        for field in ('userEnteredValue','note','userEnteredFormat','dataValidation'):
            if a2.get(field) != snapshot['records_a2'].get(field):
                raise BridgeError('HISTORICAL_REGRESSION: _records!A2 ' + field)
        protected = self.adapter.grid([legacy.quoted(t) for t in legacy.PROTECTED])
        if _exclude(protected, placement) != _exclude(snapshot['protected'], placement):
            raise BridgeError('HISTORICAL_REGRESSION: protected table/outside owned block')
        if legacy._without_layout(self.adapter.grid([legacy.quoted(t) for t in PRESERVED_PUBLIC])) != legacy._without_layout(snapshot['preserved_public']):
            raise BridgeError('HISTORICAL_REGRESSION: preserved public B01 tabs')
        views = self.adapter.grid([f'{legacy.quoted(t)}!A5:BL6' for t in legacy.VIEW_NAMES if t in tabs])
        if legacy._without_layout(views) != legacy._without_layout(snapshot['views']):
            raise BridgeError('HISTORICAL_REGRESSION: existing display formulas')
        records = self.adapter.values("'_records'!A2:BL")
        now = legacy.observation_map(records)
        for rid, old in snapshot['observations'].items():
            if rid not in now or not legacy.rows_equal(now[rid], old):
                raise BridgeError('HISTORICAL_REGRESSION: existing Result_ID ' + rid)
        for key, rows in [('records', records), ('main', self.adapter.values("'WV3-main'!A6:BL"))]:
            if set(legacy._errors(rows)) - set(snapshot['errors'][key]):
                raise BridgeError('VIEW_REVIEW_REQUIRED: new spreadsheet error ' + key)
        return dict(historical_observations=len(snapshot['observations']), protected_tables_unchanged=True,
            existing_view_formulas_unchanged=True, historical_ids_preserved=True,
            preserved_public_tabs_unchanged=True, outside_owned_block_unchanged=True,
            new_sheet_count=0, existing_sheet_count=len(tabs), records_union_unchanged=True)

    def _update_summary(self, prepared, tabs):
        # Re-read placement immediately before the atomic content + owner marker write.
        target = self._block_plan(prepared)
        placement = target['placement']
        planned = getattr(self, '_upload_target', target)['placement']
        if (placement['range'], placement['mode']) != (planned['range'], planned['mode']):
            raise BridgeError('CONCURRENT_EDIT: ablations placement changed after source upsert')
        self._last_target = target
        if target['action'] == 'NO_OP':
            return False
        sid = FIXED['ablations']
        requests = []
        have = tabs['ablations']['gridProperties']['rowCount']
        if placement['end_row'] > have:
            requests.append({'appendDimension': dict(sheetId=sid, dimension='ROWS', length=placement['end_row']-have)})
        requests.append({'updateCells': {
            'range': dict(sheetId=sid, startRowIndex=placement['start_row']-1,
                          endRowIndex=placement['end_row'], startColumnIndex=0, endColumnIndex=placement['end_col']),
            'rows': target['rows'],
            'fields': 'userEnteredValue,note' + (',userEnteredFormat' if placement['mode'] != 'REUSE' else '')}})
        try:
            self.adapter.batch_update(requests)
        except Exception:
            # A lost reply can follow an already committed atomic batch. Verify it;
            # never blindly repeat a placement or create a second block.
            self._verify_target(target)
        self._progress('ABLATIONS_BLOCK_WRITTEN', {'placement': placement, 'content_sha256': target['content_sha256']})
        return True

    def _verify_target(self, target):
        place = target['placement']
        actual = self.adapter.values(place['range'])
        for i, row in enumerate(target['expected']):
            if not legacy.rows_equal(actual[i] if i < len(actual) else [], row):
                raise BridgeError('SUMMARY_READBACK_MISMATCH: ablations row ' + str(place['start_row']+i))
        grid = self.adapter.grid([place['range']])
        cells = _cells(grid, FIXED['ablations'])
        if cells.get((place['start_row']-1, 0), {}).get('note') != place['marker_note']:
            raise BridgeError('SUMMARY_READBACK_MISMATCH: ownership note')
        if _content_hash(cells, place['start_row'], place['end_row'], place['end_col']) != target['content_sha256']:
            raise BridgeError('SUMMARY_READBACK_MISMATCH: block literals/notes')
        if legacy._errors(actual):
            raise BridgeError('SUMMARY_READBACK_MISMATCH: new formula error')
        return True

    def _verify_summary(self, prepared):
        target = self._block_plan(prepared)
        if target['placement']['mode'] != 'REUSE':
            raise BridgeError('SUMMARY_READBACK_MISMATCH: owned block missing')
        self._last_target = target
        return self._verify_target(target)

    def upload(self, prepared, *, apply=False, writer_server='s1'):
        self._writer(writer_server)
        before = self.inspect()
        self._upload_target = self._block_plan(prepared)
        p = self._upload_target['placement']
        # Preflight known placement/manual-edit conflict before any source writes.
        if apply:
            self._progress('BEFORE_SNAPSHOT', before)
            self._progress('PLACEMENT_PLANNED', p)
        result = super().upload(prepared, apply=apply, writer_server=writer_server)
        result.update(output_policy=POLICY, new_sheet_count=0, placement=p,
            existing_tab_ids={s['properties']['title']: s['properties']['sheetId'] for s in before['metadata']['sheets']},
            append_ranges=[p['range']] if p['mode'] != 'REUSE' else [],
            update_ranges=[p['range']] if p['mode'] == 'REUSE' else [],
            protected_ranges=['paper','유의미한결과','RB-B01','RB02-curves','ablations outside '+p['range']],
            source_payload_sha256=self._upload_target['block']['source_hash'],
            block_action=self._upload_target['action'], conflict_count=0)
        if apply:
            result['regression'] = self._regression(before, p)
            result['placement_manifest'] = self._last_target['placement']['owner_marker']
            result['block_written'] = self._upload_target['action'] != 'NO_OP'
            after = self.inspect()
            self._progress('AFTER_SNAPSHOT', after)
            result['before_snapshot_sha256'] = before['snapshot_sha256']
            result['after_snapshot_sha256'] = after['snapshot_sha256']
        return result

    def verify(self, prepared, snapshot):
        self._check_snapshot(snapshot)
        _, tabs = self._tabs()
        rows = self._rows(prepared, tabs)
        existing = self._existing_rows({r[0] for r in rows})
        native = []; stress = []
        for title, number, row, idcol, readcol in rows:
            if readcol is not None:
                row[readcol] = 'READBACK_VERIFIED'
            if not legacy.rows_equal(self._actual(existing, title, number), row):
                raise BridgeError('SOURCE_READBACK_MISMATCH: ' + title + ' ' + str(number))
            if title in legacy.NATIVE_TABS:
                native.append(row)
            if title == '_rb02_points':
                stress.append(row)
        view = self.adapter.values("'WV3-main'!A6:BL")
        maps = [legacy.observation_map(self.adapter.values("'_records'!A2:BL")), legacy.observation_map(view)]
        for row in native:
            if any(row[59] not in data or not legacy.rows_equal(data[row[59]], row) for data in maps):
                raise BridgeError('VIEW_READBACK_MISMATCH: native Result_ID ' + row[59])
        self._verify_summary(prepared)
        placement = self._last_target['placement']
        displayed = [i for i, row in enumerate(view, 6) if len(row) > 59 and row[59] in {r[59] for r in native}]
        result = dict(schema=OWNER, status='VIEW_READBACK_VERIFIED', output_policy=POLICY,
            new_sheet_count=0, native_rows=len(native), students=len({r[29] for r in native}),
            stress_points=len(stress), source_readback_verified=True, formula_views_readback_verified=True,
            summary_readback_verified=True, placement=placement,
            wv3_main_native_range=f"'WV3-main'!A{min(displayed)}:BL{max(displayed)}" if displayed else None,
            regression=self._regression(snapshot, placement))
        self._progress('VIEW_READBACK_VERIFIED', result)
        return result
