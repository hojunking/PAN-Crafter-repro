"""Explicit s1-only M12 writer; native and stress RAW sources are separate protocols.

No public view/body write API exists here. Legacy B01 and its 43-column stress
schema remain untouched. Network failure is upload debt, not a training retry.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path

from reporting_bridge.rb_m12_contract import (
    CAMPAIGN, SCHEMA, SPREADSHEET_ID, SERVERS, NATIVE_HEADERS, POINT_HEADERS, STATUS_HEADERS,
    build_native_row, build_point_row, build_status_row, source_row, point_row, status_row,
    digest, result_id,
)

FIXED_IDS = {'_records': 260928001, 'WV3-ablations': 261001010, 'WV3-main': 260928010}
SOURCE_IDS = {'_rb_m12_s1': 271001100, '_rb_m12_s3': 271001101, '_rb_m12_s5': 271001102,
              '_rb_m12_points': 271001103, '_rb_m12_status': 271001104}
SPECS = {**{f'_rb_m12_{s}': (81, NATIVE_HEADERS) for s in SERVERS},
         '_rb_m12_points': (14113, POINT_HEADERS), '_rb_m12_status': (121, STATUS_HEADERS)}
OWNER_KEY = 'PAN_M12_SOURCE_OWNER'
OWNER_VALUE = CAMPAIGN + ';' + SCHEMA
BASE_UNION = 'joined,VSTACK(allrows,rbload("_rb01_s1"),rbload("_rb01_s3"),rbload("_rb01_s5"))'
M12_UNION = BASE_UNION[:-1] + ',rbload("_rb_m12_s1"),rbload("_rb_m12_s3"),rbload("_rb_m12_s5"))'


class SheetError(ValueError):
    pass


def column(number):
    text = ''
    while number:
        number, rem = divmod(number - 1, 26)
        text = chr(65 + rem) + text
    return text


def quoted(title):
    return "'" + title.replace("'", "''") + "'"


def patch_formula(formula):
    """Change one live VSTACK only; never restore an old complete parser/formula."""
    if not isinstance(formula, str) or not formula.startswith('='):
        raise SheetError('FORMULA_REVIEW_REQUIRED: _records A2 is not its live formula')
    for title in ('_rb01_s1', '_rb01_s3', '_rb01_s5'):
        if formula.count(f'rbload("{title}")') != 1:
            raise SheetError('FORMULA_REVIEW_REQUIRED: original B01 union changed')
    required = ('routebucket,IF(REGEXMATCH(TO_TEXT(CHOOSECOLS(kept,28)),"^(02|03|04|06|10) [|] "),"Ablations",CHOOSECOLS(kept,27))',
                'HSTACK(CHOOSECOLS(kept,SEQUENCE(1,26)),routebucket,CHOOSECOLS(kept,SEQUENCE(1,37,28)))')
    if any(formula.count(part) != 1 for part in required):
        raise SheetError('FORMULA_REVIEW_REQUIRED: AA-only dataset routing is not recognized')
    if M12_UNION in formula:
        if formula.count(M12_UNION) != 1 or any(formula.count(f'rbload("_rb_m12_{s}")') != 1 for s in SERVERS):
            raise SheetError('FORMULA_REVIEW_REQUIRED: duplicate M12 union')
        return formula
    if '_rb_m12_' in formula or formula.count(BASE_UNION) != 1:
        raise SheetError('FORMULA_REVIEW_REQUIRED: unknown source union; no replacement')
    return formula.replace(BASE_UNION, M12_UNION, 1)


def _equal(actual, expected):
    a = list(actual) + ['']*max(0, len(expected)-len(actual))
    if any(a[len(expected):]):
        return False
    for x, y in zip(a, expected):
        x = '' if x is None else x
        y = '' if y is None else y
        if isinstance(x, bool) or isinstance(y, bool):
            if type(x) is not type(y) or x != y:
                return False
        elif type(y) in (int, float):
            if type(x) not in (int, float) or not math.isfinite(x) or abs(x-y) > 1e-12*max(1., abs(y)):
                return False
        elif x != y:
            return False
    return True


def _owners(metadata):
    # Sheet-scoped developer metadata can be nested under sheets, not at workbook root.
    entries = list(metadata.get('developerMetadata', []))
    for sheet in metadata.get('sheets', []):
        entries.extend(sheet.get('developerMetadata', []))
    owners = {}
    for entry in entries:
        if entry.get('metadataKey') != OWNER_KEY or 'sheetId' not in entry.get('location', {}):
            continue
        sid = entry['location']['sheetId']; value = entry.get('metadataValue')
        if sid in owners and owners[sid] != value:
            raise SheetError('Conflicting M12 ownership developer metadata')
        owners[sid] = value
    return owners


def _atomic(path, value):
    import os
    import tempfile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


@contextmanager
def writer_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class GoogleSheetsAdapter:
    """Lazy auth; fixed workbook and a strict network-boundary write allowlist."""
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self._book = None

    @property
    def book(self):
        if self._book is None:
            import gspread
            client = gspread.service_account(filename=str(self.root / 'gspread/account.json'))
            client.http_client.set_timeout(30)
            self._book = client.open_by_key(SPREADSHEET_ID)
        return self._book

    def _writer(self):
        if (self.root / 'gspread/server.txt').read_text().strip() != 's1':
            raise SheetError('Only actual s1 may mutate M12 Sheet sources')

    def metadata(self):
        return self.book.fetch_sheet_metadata(params={'fields': 'spreadsheetId,sheets(properties,developerMetadata),developerMetadata'})

    def values(self, name):
        return self.book.values_get(name, params={'valueRenderOption': 'UNFORMATTED_VALUE'}).get('values', [])

    def formulas(self, name):
        return self.book.values_get(name, params={'valueRenderOption': 'FORMULA'}).get('values', [])

    def formula(self):
        rows = self.book.values_get("'_records'!A2", params={'valueRenderOption': 'FORMULA'}).get('values', [])
        return rows[0][0] if rows and rows[0] else ''

    def structural(self, requests):
        self._writer()
        for request in requests:
            if len(request) != 1:
                raise SheetError('Exactly one approved structural operation per request')
            kind, body = next(iter(request.items()))
            if kind == 'addSheet':
                props = body['properties']; title = props['title']
                if title not in SOURCE_IDS or props.get('sheetId') != SOURCE_IDS[title] or props.get('hidden') is not True:
                    raise SheetError('Only five fixed owned hidden M12 sources may be created')
            elif kind == 'createDeveloperMetadata':
                meta = body['developerMetadata']
                if meta.get('metadataKey') != OWNER_KEY or meta.get('metadataValue') != OWNER_VALUE or meta.get('location', {}).get('sheetId') not in SOURCE_IDS.values():
                    raise SheetError('Unexpected ownership metadata')
            elif kind == 'appendDimension':
                if body.get('sheetId') not in SOURCE_IDS.values() or body.get('dimension') != 'ROWS':
                    raise SheetError('Only owned source row-capacity growth is allowed')
            else:
                raise SheetError('Forbidden structural mutation: ' + kind)
        if requests:
            return self.book.batch_update({'requests': requests})

    def write_raw(self, entries):
        self._writer()
        for name, values in entries:
            title, span = name.split('!', 1); title = title.strip("'")
            if title not in SPECS or not span.startswith('A'):
                raise SheetError('RAW writes only to bounded owned M12 source rows')
            import re
            match = re.fullmatch(r'A([0-9]+):([A-Z]+)([0-9]+)', span)
            rows, headers = SPECS[title]
            if not match or match[2] != column(len(headers)) or int(match[1]) < 1 or int(match[3]) > rows:
                raise SheetError('Unbounded/wrong-width source write rejected')
            if len(values) != int(match[3])-int(match[1])+1 or any(len(r) != len(headers) for r in values):
                raise SheetError('Payload dimensions mismatch')
        if entries:
            return self.book.values_batch_update({'valueInputOption': 'RAW', 'data': [{'range': n, 'values': v} for n, v in entries]})

    def write_formula(self, before, after):
        self._writer()
        if self.formula() != before or patch_formula(before) != after:
            raise SheetError('Formula changed concurrently or exceeds minimal M12 patch')
        return self.book.values_update("'_records'!A2", params={'valueInputOption': 'USER_ENTERED'}, body={'values': [[after]]})


class Uploader:
    def __init__(self, adapter, work_root, *, writer_server='s1'):
        if writer_server != 's1':
            raise SheetError('s3/s5 produce packages; only s1 is a Sheet writer')
        self.adapter = adapter
        self.work_root = Path(work_root)

    def _inspect(self):
        meta = self.adapter.metadata()
        if meta.get('spreadsheetId') != SPREADSHEET_ID:
            raise SheetError('Wrong spreadsheet')
        props = {s['properties']['title']: s['properties'] for s in meta['sheets']}
        for name, sid in FIXED_IDS.items():
            if name not in props or props[name]['sheetId'] != sid:
                raise SheetError('Live view identity changed: ' + name)
        for name, row in (('_records', 1), ('WV3-ablations', 5), ('WV3-main', 5)):
            header = self.adapter.values(f'{quoted(name)}!A{row}:BL{row}')
            if not header or not _equal(header[0], NATIVE_HEADERS):
                raise SheetError('Canonical native header drift: ' + name)
        return meta, props

    def setup(self, *, activate=False):
        """Dry-run by default. Stress public extension remains pending; no old 43-col edits."""
        with writer_lock(self.work_root / 'writer.lock'):
            meta, props = self._inspect()
            formula = self.adapter.formula(); patched = patch_formula(formula)
            occupied = {p['sheetId']: p['title'] for p in props.values()}
            owner = _owners(meta)
            requests = []; headers = []; extra_cells = 0
            for name, (rows, labels) in SPECS.items():
                sid = SOURCE_IDS[name]
                if name not in props:
                    if sid in occupied:
                        raise SheetError('Reserved M12 sheet ID belongs to another tab')
                    requests.extend([
                        {'addSheet': {'properties': {'sheetId': sid, 'title': name, 'hidden': True,
                                                    'gridProperties': {'rowCount': rows, 'columnCount': len(labels), 'frozenRowCount': 1}}}},
                        {'createDeveloperMetadata': {'developerMetadata': {'metadataKey': OWNER_KEY, 'metadataValue': OWNER_VALUE,
                                                                          'location': {'sheetId': sid}, 'visibility': 'DOCUMENT'}}}])
                    headers.append((f'{quoted(name)}!A1:{column(len(labels))}1', [list(labels)]))
                    extra_cells += rows*len(labels)
                else:
                    p = props[name]
                    if p['sheetId'] != sid or p.get('hidden') is not True or owner.get(sid) != OWNER_VALUE:
                        raise SheetError('Existing source ownership/schema is not M12: ' + name)
                    if p['gridProperties']['columnCount'] != len(labels):
                        raise SheetError('Source schema width changed')
                    header = self.adapter.values(f'{quoted(name)}!A1:{column(len(labels))}1')
                    if not header or not any(header[0]):
                        headers.append((f'{quoted(name)}!A1:{column(len(labels))}1', [list(labels)]))
                    elif not _equal(header[0], labels):
                        raise SheetError('Existing owned header differs; never overwrite')
                    missing = rows - p['gridProperties']['rowCount']
                    if missing > 0:
                        requests.append({'appendDimension': {'sheetId': sid, 'dimension': 'ROWS', 'length': missing}})
                        extra_cells += missing*len(labels)
            existing_cells = sum(p['gridProperties']['rowCount']*p['gridProperties']['columnCount'] for p in props.values() if 'gridProperties' in p)
            pending_stress = False
            if existing_cells + extra_cells > 9500000 and '_rb_m12_points' not in props:
                # Large stress source is optional infrastructure, never a native-upload gate.
                sid = SOURCE_IDS['_rb_m12_points']
                requests = [r for r in requests if not (
                    r.get('addSheet', {}).get('properties', {}).get('sheetId') == sid or
                    r.get('createDeveloperMetadata', {}).get('developerMetadata', {}).get('location', {}).get('sheetId') == sid)]
                headers = [(span, rows) for span, rows in headers if not span.startswith("'_rb_m12_points'!")]
                extra_cells -= SPECS['_rb_m12_points'][0]*len(POINT_HEADERS)
                pending_stress = True
            if existing_cells + extra_cells > 9500000:
                raise SheetError('Workbook capacity reserve exceeded even for native/status setup')
            result = dict(schema=SCHEMA, campaign_id=CAMPAIGN, activate=activate, structural_requests=requests,
                          header_writes=len(headers), formula_changed=patched != formula,
                          formula_before_sha256=hashlib.sha256(formula.encode()).hexdigest(),
                          formula_after_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                          workbook_cells_after=existing_cells+extra_cells,
                          stress_source_pending_capacity=pending_stress,
                          stress_public_extension='PENDING_NOT_BLOCKING_NATIVE', protected_public_tabs_unchanged=True)
            if not activate:
                return result
            snapshot = dict(result, formula_before=formula, formula_after=patched, source_metadata=meta)
            _atomic(self.work_root / ('setup_snapshot_' + result['formula_before_sha256'] + '.json'), snapshot)
            if self.adapter.formula() != formula:
                raise SheetError('Live formula changed after inspection; retry from a fresh snapshot')
            self.adapter.structural(requests)
            self.adapter.write_raw(headers)
            if patched != formula:
                self.adapter.write_formula(formula, patched)
            if self.adapter.formula() != patched:
                raise SheetError('Patched formula readback differs')
            for name, (_rows, labels) in SPECS.items():
                if pending_stress and name == '_rb_m12_points':
                    continue
                actual = self.adapter.values(f'{quoted(name)}!A1:{column(len(labels))}1')
                if not actual or not _equal(actual[0], labels):
                    raise SheetError('Source header readback failed')
            result['status'] = 'SETUP_VERIFIED'
            _atomic(self.work_root / 'setup_receipt.json', result)
            return result

    def sync(self, native, points=(), statuses=(), *, activate=False):
        """Retry observes already-written sources before writing; no append_row calls."""
        with writer_lock(self.work_root / 'writer.lock'):
            meta, props = self._inspect()
            if patch_formula(self.adapter.formula()) != self.adapter.formula():
                raise SheetError('Run explicit M12 setup before upload')
            entries = []; expected = []; native_expected = []; unique = set()
            pending_points = list(points) if '_rb_m12_points' not in props else []
            if pending_points:
                points = ()
            batches = [(r, '_rb_m12_' + r['server'], source_row(r), build_native_row(r, SOURCE_IDS['_rb_m12_' + r['server']]), 61) for r in native]
            batches += [(r, '_rb_m12_points', point_row(r), build_point_row(r), len(POINT_HEADERS)-1) for r in points]
            batches += [(r, '_rb_m12_status', status_row(r), build_status_row(r), None) for r in statuses]
            # Read each occupied source chunk once, not once per 14,112 point.
            sources = {name for _, name, _, _, _ in batches}
            ownership = _owners(meta)
            for name in sources:
                if name not in props or props[name]['sheetId'] != SOURCE_IDS[name] or ownership.get(SOURCE_IDS[name]) != OWNER_VALUE or props[name].get('hidden') is not True:
                    raise SheetError('Missing/foreign M12 source; explicit setup is required')
                header = self.adapter.values(f'{quoted(name)}!A1:{column(len(SPECS[name][1]))}1')
                if not header or not _equal(header[0], SPECS[name][1]):
                    raise SheetError('Source header changed')
            chunks = {(name, 2 + ((index-2)//256)*256) for _, name, index, _, _ in batches}
            before = self._source_chunks(chunks, formulas=True)
            for record, name, index, row, review in batches:
                key = (name, index)
                if key in unique:
                    raise SheetError('Duplicate logical slot in upload batch')
                unique.add(key)
                span = f'{quoted(name)}!A{index}:{column(len(row))}{index}'
                old = before.get((name, index), [])
                if any(isinstance(v, str) and v.startswith('=') for v in old):
                    raise SheetError('Unexpected formula/spill in RAW source slot')
                if review is not None and len(old) > review:
                    row[review] = old[review]  # Human Review is never reset by retries.
                if name in ('_rb_m12_s1', '_rb_m12_s3', '_rb_m12_s5') and len(old) > 51 and old[51] == 'VERIFIED':
                    row[51] = 'VERIFIED'  # Never demote a previously verified immutable payload on retry.
                if name == '_rb_m12_status' and any(old) and (old[:2] != row[:2]):
                    raise SheetError('Foreign status row cannot be replaced')
                if any(old) and not _equal(old, row) and name != '_rb_m12_status':
                    raise SheetError('EVIDENCE_CONFLICT: occupied immutable slot ' + span)
                if not _equal(old, row):
                    entries.append((span, [row]))
                expected.append((span, row))
                if name in ('_rb_m12_s1', '_rb_m12_s3', '_rb_m12_s5'):
                    native_expected.append(row)
            result = dict(schema=SCHEMA, campaign_id=CAMPAIGN, activate=activate, changed_rows=len(entries),
                          native_observations=len(native), stress_points=len(points), status_rows=len(statuses),
                          stress_source_pending_points=len(pending_points),
                          status='DRY_RUN', stress_public_extension='PENDING_NOT_BLOCKING_NATIVE')
            if not activate:
                return result
            debt_path = self.work_root / ('upload_' + digest(expected) + '.json')
            _atomic(debt_path, dict(result, status='UPLOAD_PENDING', expected=expected))
            if pending_points:
                _atomic(self.work_root / ('stress_pending_' + digest(pending_points) + '.json'),
                        dict(schema=SCHEMA, status='UPLOAD_PENDING_CAPACITY', points=pending_points))
            # Bounded chunks avoid a 14k point one-request payload; source readback remains required.
            for start in range(0, len(entries), 100):
                self.adapter.write_raw(entries[start:start+100])
            after = self._source_chunks(chunks)
            for (_, name, index, _, _), (span, row) in zip(batches, expected):
                if not _equal(after.get((name, index), []), row):
                    raise SheetError('Source RAW numeric readback mismatch: ' + span)
            self._verify_views(props, native_expected)
            confirmed = []
            for row in native_expected:
                if row[51] != 'VERIFIED':
                    row[51] = 'VERIFIED'
                    confirmed.append((f'{quoted(row[56])}!A{row[57]}:BL{row[57]}', [row]))
            # Only after RAW -> normalized -> public readback may AZ claim VERIFIED.
            for start in range(0, len(confirmed), 100):
                self.adapter.write_raw(confirmed[start:start+100])
            if confirmed:
                actual = self._source_chunks({(row[56], 2+((row[57]-2)//256)*256) for row in native_expected})
                for row in native_expected:
                    if not _equal(actual.get((row[56], row[57]), []), row):
                        raise SheetError('Final verified-marker readback failed')
                self._verify_views(props, native_expected)
            result['status'] = 'VERIFIED'
            _atomic(debt_path, dict(result, expected=expected))
            return result

    def _verify_views(self, props, native_expected):
        if not native_expected:
            return
        for title in ('_records', 'WV3-ablations', 'WV3-main'):
            p = props[title]
            values = self.adapter.values(f'{quoted(title)}!A2:BL{p["gridProperties"]["rowCount"]}')
            for expected_row in native_expected:
                matches = [r for r in values if len(r) > 59 and r[59] == expected_row[59]]
                if title == 'WV3-main':
                    if matches:
                        raise SheetError('M12 native appeared in WV3-main')
                elif len(matches) != 1 or not _equal(matches[0], expected_row):
                    raise SheetError('Normalized/view readback pending or conflicting: ' + title)

    def _source_chunks(self, chunks, *, formulas=False):
        result = {}
        read = self.adapter.formulas if formulas else self.adapter.values
        for name, start in sorted(chunks):
            end = min(start+255, SPECS[name][0])
            rows = read(f'{quoted(name)}!A{start}:{column(len(SPECS[name][1]))}{end}')
            for offset, row in enumerate(rows):
                result[(name, start+offset)] = row
        return result
