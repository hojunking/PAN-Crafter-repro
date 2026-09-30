#!/usr/bin/env python3
"""B01 evidence -> normalized Sheets, without training or metric recomputation.

Only setup/upload with --apply may write to Google Sheets. Collection reads the
original assets and writes new evidence under a separate reporting workspace.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def safe_output(path, source_root=ROOT):
    path = Path(path).resolve()
    for root in {ROOT.resolve(), Path(source_root).resolve()}:
        reporting = root / 'work_dir/_rb_sheet_upload'
        if (path == root or root in path.parents) and not (path == reporting or reporting in path.parents):
            raise ValueError('In-repository reporting output must stay under work_dir/_rb_sheet_upload: ' + str(path))
    return path


def write(path, value):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def object_at(path, filename):
    path = Path(path)
    return read(path / filename if path.is_dir() else path)


def prepared_at(path):
    document = object_at(path, 'prepared.json')
    payload = {k: v for k, v in document.items() if k != 'payload_sha256'}
    if document.get('schema') != 'RB_B01_PREPARED_v1' or document.get('payload_sha256') != digest(payload):
        raise ValueError('Prepared payload schema/hash mismatch')
    return document


def prepare(evidence_root):
    from reporting_bridge.rb_b01_collect import load_packages
    from reporting_bridge.rb_b01_contract import canonical_native, canonical_stress, registry
    from reporting_bridge.rb_b01_summary import summarize
    packages = load_packages(Path(evidence_root))
    native = []
    stress = []
    statuses = {}
    seen_servers = set()
    hashes = []
    errors = []
    for package in packages:
        server = package['source_server']
        if server in seen_servers:
            raise ValueError('Provide one verified evidence package per server: ' + server)
        seen_servers.add(server)
        hashes.append({'server': server, 'payload_sha256': package['package_payload_sha256']})
        for entry in package['native']:
            native.extend(canonical_native(entry))
        for entry in package['curves']:
            stress.extend(canonical_stress(entry))
        for status in package['status']:
            if status['server'] == server:
                statuses[status['run_id']] = status
        errors.extend(package.get('errors', []))
    for case in registry():
        if case['run_id'] not in statuses:
            statuses[case['run_id']] = dict(case, verification_status='REMOTE_EVIDENCE_NOT_SUPPLIED',
                training_status='UNKNOWN', native_status='FILES_NOT_PROVIDED',
                upload_status='NOT_AVAILABLE', actual_updates=None,
                native_selectors=None, last_checked_at_utc=stamp(),
                curve_status={mode: dict(status='FILES_NOT_PROVIDED', shifts=None,
                    numerical_failures=None, invalid_geometry=None)
                    for mode in ('A_ON', 'A_ZERO_INFERENCE_ONLY')},
                error='No verified evidence package supplied; execution state is unknown.')
    status_rows = [statuses[case['run_id']] for case in registry()]
    document = dict(schema='RB_B01_PREPARED_v1', prepared_at_utc=stamp(),
        native_records=native, stress_records=stress, statuses=status_rows,
        summary=summarize(native, stress, status_rows), package_hashes=hashes,
        errors=errors, collected_servers=sorted(seen_servers))
    document['payload_sha256'] = digest(document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect', 'inspect', 'prepare', 'plan', 'setup', 'upload', 'verify'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--server', choices=('s1', 's3', 's5'))
    parser.add_argument('--writer-server', choices=('s1',), default='s1')
    parser.add_argument('--frozen-report', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--spreadsheet-id', default=SHEET_ID)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if args.apply and args.command not in ('setup', 'upload'):
        parser.error('--apply is accepted only for setup/upload')
    required = {'collect': ('server', 'output_dir'), 'inspect': ('output_dir',),
                'prepare': ('evidence_root', 'output_dir'),
                'plan': ('prepared', 'snapshot', 'output_dir'), 'setup': ('plan',),
                'upload': ('prepared', 'output_dir'),
                'verify': ('prepared', 'snapshot', 'output_dir')}
    for name in required[args.command]:
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required')
    output = safe_output(args.output_dir or args.plan.parent / 'setup', args.root)
    output.mkdir(parents=True, exist_ok=True)
    event_file = output / ('events-' + args.command + '.jsonl')
    # Local serialization complements (does not replace) the declared s1 writer.
    lock_path = safe_output(args.root / 'work_dir/_rb_sheet_upload/B01/bridge.lock', args.root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        lock_mode = fcntl.LOCK_EX if args.apply else fcntl.LOCK_SH
        fcntl.flock(lock, lock_mode | fcntl.LOCK_NB)
        attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex
        last_successful_stage = 'STARTED'

        def progress(state, details=None):
            nonlocal last_successful_stage
            record = dict(at_utc=stamp(), command=args.command, attempt_id=attempt_id,
                          state=state, details=details or {})
            write(output / 'history' / (args.command + '_' + attempt_id + '_' + state + '.json'), record)
            with event_file.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            last_successful_stage = state
        try:
            if args.command == 'collect':
                from reporting_bridge.rb_b01_collect import collect_local
                result = collect_local(args.root, args.server, frozen_report_path=args.frozen_report)
                destination = output / (args.server + '.evidence.json')
            elif args.command == 'prepare':
                result = prepare(args.evidence_root)
                destination = output / 'prepared.json'
            else:
                from reporting_bridge.rb_b01_sheets import GoogleSheetsAdapter, SheetsBridge
                bridge = SheetsBridge(GoogleSheetsAdapter(args.spreadsheet_id, root=args.root))
                bridge.progress_callback = progress
                if args.command == 'inspect':
                    result = bridge.inspect()
                    destination = output / 'snapshot.json'
                elif args.command == 'plan':
                    result = bridge.plan(prepared_at(args.prepared), object_at(args.snapshot, 'snapshot.json'))
                    result['snapshot_path'] = str(args.snapshot.resolve())
                    result['prepared_path'] = str(args.prepared.resolve())
                    destination = output / 'plan.json'
                elif args.command == 'setup':
                    plan = object_at(args.plan, 'plan.json')
                    snapshot = object_at(args.snapshot or Path(plan['snapshot_path']), 'snapshot.json')
                    result = bridge.setup(plan, snapshot, apply=args.apply, writer_server=args.writer_server)
                    destination = output / 'setup_receipt.json'
                elif args.command == 'upload':
                    prepared = prepared_at(args.prepared)
                    immutable_spool = output / 'spool' / (prepared['payload_sha256'] + '.json')
                    if immutable_spool.exists():
                        if read(immutable_spool) != prepared:
                            raise ValueError('Existing immutable spool payload differs')
                    else:
                        write(immutable_spool, prepared)
                    write(output / 'spooled_payload.json', prepared)
                    progress('SPOOLED', {'payload_sha256': prepared['payload_sha256']})
                    result = bridge.upload(prepared, apply=args.apply, writer_server=args.writer_server)
                    destination = output / 'upload_receipt.json'
                else:
                    result = bridge.verify(prepared_at(args.prepared), object_at(args.snapshot, 'snapshot.json'))
                    destination = output / 'verification.json'
            archive = output / 'history' / (args.command + '_' + attempt_id + '.json')
            write(archive, result)
            write(destination, result)
            event = dict(at_utc=stamp(), command=args.command, apply=args.apply,
                         state='RETURNED', artifact=str(destination), immutable_receipt=str(archive),
                         last_successful_stage=last_successful_stage,
                         result_sha256=digest(result))
        except Exception as exc:
            event = dict(at_utc=stamp(), command=args.command, apply=args.apply,
                         state='FAILED', last_successful_stage=last_successful_stage,
                         error_type=type(exc).__name__, error=str(exc))
            write(output / 'history' / (args.command + '_' + attempt_id + '_failure.json'), event)
            write(output / ('last_' + args.command + '_failure.json'), event)
            with event_file.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            raise
        with event_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        print(json.dumps(event, ensure_ascii=False))
        return result


if __name__ == '__main__':
    main()
