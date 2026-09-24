#!/usr/bin/env python3
"""Local s4/s5 safe G23 handoff, pinned Docker and independent MAIN-A launch."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL = 'org.pancrafter.maina_hqnr'
DEFAULT_IMAGE = 'hojunqueen/pancrafter-env:torch2.4.0-cu118'
PINNED_IMAGE = 'sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887'


def output(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def build_mounts(root, frozen, dlpan, bindings, asset_map=None):
    root = Path(root).resolve(); frozen = Path(frozen).resolve()
    work = (root / 'work_dir').resolve(strict=True)
    mounts = {root: 'ro', frozen: 'ro', work: 'rw', Path(dlpan).resolve(strict=True): 'ro'}
    paths = [bindings['canonical_bridge_path'], *bindings['resolved_artifacts'].values()]
    for item in bindings['dataset_manifest']['splits'].values():
        paths.extend((item['dataroot'], item['lpan_path']))
    paths.extend((root / 'gspread').glob('*.json'))
    if asset_map:
        paths.append(asset_map)
    for value in paths:
        path = Path(value).resolve(strict=True)
        if root not in path.parents and work not in path.parents:
            mounts[path] = 'ro'
    return [(str(path), access) for path, access in sorted(mounts.items(), key=lambda x: (len(x[0].parts), str(x[0])))]


def build_command(*, frozen, server, image_id, commit, name, mounts, dlpan, gpu,
                  no_upload=False, asset_map=None):
    from maina_hqnr.common import verify_server
    verify_server(server)
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', gpu):
        raise ValueError('One host-resolved physical GPU UUID is required')
    if image_id != PINNED_IMAGE:
        raise ValueError('Original main Docker image differs; do not silently change the numerical environment')
    command = ['docker', 'run', '--detach', '--name', name, '--gpus', 'device=' + gpu,
        '--pid=host', '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g',
        '--restart', 'no', '--workdir', str(frozen),
        '--label', LABEL + '.server=' + server, '--label', LABEL + '.commit=' + commit,
        '--label', LABEL + '.runtime=' + str(frozen), '--label', LABEL + '.gpu_uuid=' + gpu]
    for key, value in dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/maina-hqnr-mpl', XDG_CACHE_HOME='/tmp/maina-hqnr-cache',
        HF_HOME='/tmp/maina-hqnr-hf', PANCRAFTER_DLPAN=str(dlpan), PANCRAFTER_MAINA_GPU_UUID=gpu).items():
        command += ['--env', key + '=' + value]
    for path, access in mounts:
        if ',' in path or access not in ('ro', 'rw'):
            raise ValueError('Invalid exact bind mount')
        command += ['--mount', 'type=bind,src=' + path + ',dst=' + path + (',readonly' if access == 'ro' else '')]
    command += ['--entrypoint', 'python', image_id, str(Path(frozen) / 'tools/maina_hqnr_runner.py'),
                'start', '--server', server, '--in-place', '--foreground', '--host-cutover-done']
    if no_upload:
        command.append('--no-upload')
    if asset_map:
        command += ['--asset-map', str(Path(asset_map).resolve())]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s4', 's5'), required=True)
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--asset-map', type=Path)
    parser.add_argument('--no-upload', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Read-only inspection; never request old stop or create a worktree')
    args = parser.parse_args(argv)
    from maina_hqnr.common import camp, read, read_json, locked, RuntimePaused
    from maina_hqnr.assets import verify_assets
    from maina_hqnr.deployment import frozen_checkout
    from maina_hqnr.handoff import wait_boundary
    image_id = output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image])
    if image_id != PINNED_IMAGE:
        raise RuntimePaused('PAUSED_RUNTIME_MISMATCH: use the original main image ' + PINNED_IMAGE)
    if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)', args.gpu):
        raise ValueError('Select one GPU index or UUID')
    gpu = output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader,nounits'])
    commit = output(['git', 'rev-parse', 'HEAD'], cwd=ROOT)
    receipt = read(camp(ROOT, args.server) / 'runtime_release.json')
    frozen = Path(receipt['path']) if receipt else ROOT.parent / f'{ROOT.name}-runtime-maina-hqnr-{args.server}-{commit[:12]}'
    dlpan = Path(os.environ.get('PANCRAFTER_DLPAN', str(ROOT.parent / 'DLPan-Toolbox'))).resolve(strict=True)
    # Fail before asking the old campaign to stop if a required local F1/data file is absent.
    bindings = verify_assets(args.server, root=ROOT,
        overrides=read_json(args.asset_map) if args.asset_map else None, persist=False)

    def command():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        return build_command(frozen=frozen, server=args.server, image_id=image_id, commit=commit,
            name=f'pancrafter-maina-hqnr-{args.server}-{stamp}',
            mounts=build_mounts(ROOT, frozen, dlpan, bindings, args.asset_map), dlpan=dlpan,
            gpu=gpu, no_upload=args.no_upload, asset_map=args.asset_map)

    if args.dry_run:
        print(shlex.join(command())); return 0
    folder = camp(ROOT, args.server)
    with locked(folder / 'docker_launch.lock'):
        ids = output(['docker', 'ps', '-q', '--filter', 'label=' + LABEL + '.server=' + args.server]).split()
        if ids:
            rows = json.loads(output(['docker', 'inspect', *ids]))
            if len(rows) != 1:
                raise RuntimePaused('Multiple active local MAIN-A owners')
            row = rows[0]; labels = row['Config'].get('Labels') or {}
            if row['Image'] != image_id or labels.get(LABEL + '.runtime') != str(frozen):
                raise RuntimePaused('Different local MAIN-A runtime already owns this lane')
            if labels.get(LABEL + '.gpu_uuid') != gpu:
                raise RuntimePaused('Active MAIN-A GPU identity is missing or differs from the requested GPU; inspect the existing owner')
            print(json.dumps(dict(status='ALREADY_RUNNING', container_id=ids[0], gpu_uuid=gpu))); return 0
        if (folder / 'STOP_AFTER_RUN').exists() or (folder / 'STOP_NOW_SAFE').exists():
            raise RuntimePaused('Operator stop markers retained; explicitly inspect before resuming')
        frozen = frozen_checkout(ROOT, args.server)
        commit = output(['git', 'rev-parse', 'HEAD'], cwd=frozen)
        wait_boundary(ROOT, args.server, gpu, activated=True)
        container = output(command())
        print(json.dumps(dict(status='START_REQUESTED', container_id=container, runtime=str(frozen), gpu_uuid=gpu,
            log_command='docker logs --tail 80 ' + container,
            note='Not RUNNING yet: CUDA acceptance and the first optimizer update must succeed.')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
