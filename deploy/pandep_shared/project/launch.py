#!/usr/bin/env python3
"""Host-stdlib launcher for the isolated, immutable-image S2 experiment.

Never pulls an image, installs packages, stops another container, or imports the
legacy project. CPU control commands use new short-lived, network-free containers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

IMAGE = 'sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887'
CAMPAIGN = 'PANDEP_S2_SHARED_PLH_20261001_v4'
SHEET_ID = 1198707876
PROJECT = Path(__file__).resolve().parent
COMMANDS = ('start', 'status', 'pause', 'resume', 'stop', 'sync-sheet')


def _under(path, parent):
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def _bind(source, target, readonly):
    # --mount comma syntax cannot represent these paths safely.
    if any(character in str(source) + str(target) for character in (',', '\n', '\r')):
        raise ValueError('Mount paths cannot contain commas or newlines')
    return ['--mount', 'type=bind,src=' + str(source) + ',dst=' + str(target) + (',readonly' if readonly else '')]


def resolve_layout(legacy_root, credentials=None, *, project=PROJECT, require_sources=True):
    project, legacy = Path(project).resolve(strict=True), Path(legacy_root).resolve(strict=True)
    if _under(project, legacy) or _under(legacy, project):
        raise ValueError('BLOCKED_PATH: new and legacy project roots must be physically separate')
    identity = legacy / 'gspread/server.txt'
    if not identity.is_file() or identity.read_text(encoding='utf-8').strip() != 's2':
        raise ValueError('BLOCKED_SERVER: gspread/server.txt must explicitly identify this host as s2')
    catalog_path = project / 'vendor_reference/model_source/ablr2/sensor_sources.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if catalog.get('schema') != 'ABLR2_SENSOR_SOURCES_v1' or set(catalog.get('sensors', {})) != {'WV3', 'GF2', 'QB'}:
        raise ValueError('BLOCKED_DATA_IDENTITY: frozen three-sensor catalog required')
    source_paths = set()
    for sensor, entry in catalog['sensors'].items() if require_sources else []:
        if set(entry['splits']) != {'train', 'val', 'rr', 'fr'}:
            raise ValueError('BLOCKED_DATA_IDENTITY: expected four splits for ' + sensor)
        for binding in entry['splits'].values():
            source_paths.add((legacy / binding['path']).resolve(strict=True))
    qb = catalog['sensors']['QB']['source_provenance']
    if require_sources:
        source_paths.update((legacy / qb['raw_' + split + '_path']).resolve(strict=True) for split in ('train', 'val'))
    if require_sources and (len(source_paths) != 14 or any(not path.is_file() or _under(path, project) for path in source_paths)):
        raise ValueError('BLOCKED_DATA_IDENTITY: 12 benchmark and two raw-QB read-only source files required')
    credential_path = Path(credentials).resolve(strict=True) if credentials is not None else None
    if credential_path is not None and (not credential_path.is_file() or _under(credential_path, project)):
        raise ValueError('BLOCKED_CREDENTIAL_PATH: use an existing credential outside the new project')
    work = (project / 'work_dir' / CAMPAIGN / 's2').resolve()
    if not _under(work, project / 'work_dir'):
        raise ValueError('BLOCKED_PATH: campaign work root escapes project via symlink')
    return dict(project=project, legacy=legacy, credentials=credential_path, identity=identity.resolve(strict=True),
                catalog=catalog_path, work=work, sources=sorted(source_paths))


def build_command(command, layout, *, microbatch=48, stamp=None):
    if command not in COMMANDS or (microbatch not in (12, 24, 48) and not (command == 'resume' and microbatch is None)):
        raise ValueError('Unregistered launcher command or microbatch')
    gpu = command in ('start', 'resume')
    network = gpu or command == 'sync-sheet'
    if network and layout['credentials'] is None:
        raise ValueError('Explicit --credentials path required; secrets are never copied into the project')
    name = 'pandep-shared-s2-' + (stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')) + '-' + str(os.getpid())
    args = ['docker', 'run', '--pull=never', '--label', 'pan_shared.campaign=' + CAMPAIGN,
            '--label', 'pan_shared.owner=pan_deploy_shared', '--name', name,
            '--hostname', socket.gethostname(),
            '--user', str(os.getuid()) + ':' + str(os.getgid()), '--pid', 'host',
            '--workdir', str(layout['project']), '--env', 'PYTHONPATH=' + str(layout['project'] / 'src'),
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'TMPDIR=' + str(layout['work'] / 'tmp'),
            '--env', 'PYTHONHASHSEED=0', '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '--env', 'OMP_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
            '--env', 'XDG_CACHE_HOME=' + str(layout['work'] / 'cache'),
            '--env', 'TORCH_HOME=' + str(layout['work'] / 'cache/torch')]
    args += ['--detach', '--gpus', 'all'] if gpu else ['--rm']
    if not network:
        args += ['--network', 'none']
    args += _bind(layout['project'], layout['project'], False)
    args += _bind(layout['legacy'], layout['legacy'], True)
    mounted = {layout['legacy'], layout['project']}
    # Resolve every raw symlink target, including files outside the legacy tree.
    # They retain identical absolute paths inside the container and remain RO.
    for path in [*layout['sources'], layout['identity'], layout['credentials']]:
        if path is not None and path not in mounted:
            args += _bind(path, path, True)
            mounted.add(path)
    args += [IMAGE, 'python', '-m', 'pan_shared.cli', command, '--work-root', str(layout['work'])]
    if gpu:
        args += ['--server', 's2', '--until-operator-stop', '--legacy-root', str(layout['legacy']),
                 '--data-root', str(layout['legacy']), '--data-catalog', str(layout['catalog']),
                 '--credentials', str(layout['credentials']),
                 '--server-identity', str(layout['identity'])]
        if microbatch is not None:
            args += ['--microbatch', str(microbatch)]
    elif command == 'sync-sheet':
        args += ['--sheet-id', str(SHEET_ID), '--credentials', str(layout['credentials'])]
    elif command == 'stop':
        args += ['--after-current-update']
    return args


def verify_image():
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE],
                            check=False, capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != IMAGE:
        raise RuntimeError('BLOCKED_ENVIRONMENT: immutable image is absent/mismatched; no automatic pull or package installation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True, type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='print argv only; no Docker execution or directory writes')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in COMMANDS:
        item = sub.add_parser(command)
        if command in ('start', 'resume'):
            item.add_argument('--microbatch', type=int, choices=(48, 24, 12), default=48 if command == 'start' else None)
    args = parser.parse_args(argv)
    try:
        layout = resolve_layout(args.legacy_root, args.credentials, require_sources=args.command in ('start', 'resume'))
        command = build_command(args.command, layout, microbatch=getattr(args, 'microbatch', 48))
        if args.dry_run:
            print(json.dumps({'status': 'DRY_RUN', 'docker_argv': command, 'gpu_requested': args.command in ('start', 'resume'),
                              'source_mounts_read_only': len(layout['sources']), 'legacy_written': False}, ensure_ascii=False, indent=2))
            return 0
        verify_image()
        for suffix in ('tmp', 'cache/torch'):
            (layout['work'] / suffix).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
        if args.command in ('start', 'resume'):
            print('Container launched; this is NOT a training-success receipt. Check status/logs for preflight, verification, WAIT_RESOURCE or RUNNING.', flush=True)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
