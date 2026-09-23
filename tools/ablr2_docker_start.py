#!/usr/bin/env python3
"""Start one frozen ABLR2 lane in a pinned local Docker image.

Docker is invoked only after the committed runtime and local assets are checked.
This launcher never pulls an image, removes a container, or renews a running lease.
"""
import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL = 'org.pancrafter.ablr2'
DEFAULT_IMAGE = 'hojunqueen/pancrafter-env:torch2.4.0-cu118'


def checked_output(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def lane_sensor(server):
    if server not in ('s1', 's2','s3'):
        raise ValueError('Docker ABLR2X supports s1/WV3,s2/QB,s3/GF2; s4/s5 protected')
    return {'s1': 'WV3', 's2': 'QB','s3':'GF2'}[server]


def build_mounts(root, frozen, server, dlpan):
    """Resolve symlink targets on the host; preserve their absolute container paths."""
    root, frozen, dlpan = Path(root).resolve(), Path(frozen).resolve(), Path(dlpan).resolve(strict=True)
    sensor = lane_sensor(server)
    catalog = json.loads((root / 'ablr2/sensor_sources.json').read_text())['sensors'][sensor]
    work = (root / 'work_dir').resolve()
    if not work.is_dir():
        raise ValueError('Missing work_dir; create the frozen runtime before launching')
    mounts = {root: 'ro', frozen: 'ro', work: 'rw', dlpan: 'ro'}
    original=root/'work_dir/ablr2'/sensor/server/'runtime_release.json'
    if original.is_file():
        old=Path(json.loads(original.read_text())['path']).resolve(strict=True)
        if (old/'work_dir').resolve()!=work:raise ValueError('Original runtime does not share the owned work_dir')
        mounts[old]='ro'
    source_paths = [row['path'] for row in catalog['splits'].values()]
    provenance = catalog['source_provenance']
    source_paths += [provenance[k] for k in ('raw_train_path', 'raw_val_path') if k in provenance]
    for source in source_paths:
        path = (root / source).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Dataset source must be a regular file: ' + str(path))
        # Native source directories contain associated evidence/provenance files too.
        if root not in path.parents:
            if path.parent in (Path('/'), Path('/home'), Path('/tmp')):
                mounts[path] = 'ro'
            else:
                mounts[path.parent] = 'ro'
    for credential in (root / 'gspread').glob('*.json'):
        path = credential.resolve(strict=True)
        if root not in path.parents:
            mounts[path] = 'ro'
    # Parent mounts precede nested mounts; work_dir remains the sole writable bind.
    return [(str(path), access) for path, access in sorted(mounts.items(), key=lambda x: (len(x[0].parts), str(x[0])))]


def build_command(*, root, frozen, server, image_id, commit, name, mounts,
                  dlpan, lease_hours=None,until_operator_stop=False,boundary='RUN',gpu='0', no_upload=False, uid=None, gid=None):
    lane_sensor(server)
    if not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', str(gpu)):
        raise ValueError('--gpu must select exactly one GPU index or UUID')
    if until_operator_stop and lease_hours is not None:raise ValueError('Choose continuous authorization or finite lease')
    if not until_operator_stop and (lease_hours is None or not 0<float(lease_hours)<=72):
        raise ValueError('Explicit finite lease in (0,72] or --until-operator-stop is required')
    if boundary not in ('RUN','SAFE'):raise ValueError('Unsupported handover boundary')
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', image_id):
        raise ValueError('Docker image must be pinned to the inspected content ID')
    argv = ['docker', 'run', '--detach', '--name', name, '--gpus', 'device=' + str(gpu),
            '--pid=host', '--user', f'{os.getuid() if uid is None else uid}:{os.getgid() if gid is None else gid}',
            '--shm-size', '8g', '--restart', 'no', '--workdir', str(frozen),
            '--label', LABEL + '.server=' + server,
            '--label', LABEL + '.commit=' + commit,
            '--label', LABEL + '.runtime=' + str(frozen)]
    for key, value in dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            MPLCONFIGDIR='/tmp/ablr2-matplotlib', XDG_CACHE_HOME='/tmp/ablr2-cache',
            HF_HOME='/tmp/ablr2-huggingface',
            PANCRAFTER_DLPAN=str(dlpan)).items():
        argv += ['--env', key + '=' + value]
    for path, access in mounts:
        if ',' in path:
            raise ValueError('Docker bind path contains unsupported comma: ' + path)
        argv += ['--mount', 'type=bind,src=' + path + ',dst=' + path + (',readonly' if access == 'ro' else '')]
    argv += ['--entrypoint', 'python', image_id,
             str(Path(frozen) / 'tools/ablr2_runner.py'), 'start', '--in-place', '--foreground',
             '--server', server,'--boundary',boundary]
    argv+=(['--until-operator-stop'] if until_operator_stop else ['--lease-hours',str(float(lease_hours))])
    if no_upload:
        argv += ['--no-upload']
    return argv


def matching_running(containers, *, server, commit, frozen, image_id,draining_release=None):
    matches = []
    for item in containers:
        if not item.get('State', {}).get('Running'):
            continue
        labels = item.get('Config', {}).get('Labels', {}) or {}
        if labels.get(LABEL + '.server') != server:
            continue
        if (labels.get(LABEL + '.commit') != commit or labels.get(LABEL + '.runtime') != str(frozen)
                or item.get('Image') != image_id):
            if (draining_release and labels.get(LABEL+'.commit')==draining_release.get('git_commit')
                    and labels.get(LABEL+'.runtime')==draining_release.get('path') and item.get('Image')==image_id):
                continue # This known original run remains the GPU owner; new entrypoint only waits.
            raise ValueError('Another ABLR2 container already runs this lane with a different release/image')
        matches.append(item)
    if len(matches) > 1:
        raise ValueError('Multiple running containers for one lane; inspect without starting another')
    return matches[0] if matches else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's2','s3'), required=True)
    authorization=parser.add_mutually_exclusive_group(required=True)
    authorization.add_argument('--lease-hours', type=float)
    authorization.add_argument('--until-operator-stop',action='store_true')
    parser.add_argument('--boundary',choices=('RUN','SAFE'),default='RUN')
    parser.add_argument('--image', default=DEFAULT_IMAGE, help='Already installed local image; never pulled implicitly')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--no-upload', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Read-only preview; no runtime, lease or container is created')
    args = parser.parse_args(argv)
    lane_sensor(args.server)
    if args.lease_hours is not None and not 0 < args.lease_hours <= 72:
        parser.error('--lease-hours must be in (0,72]')
    dlpan = Path(os.environ.get('PANCRAFTER_DLPAN', str(ROOT.parent / 'DLPan-Toolbox'))).resolve(strict=True)
    image_id = checked_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image])
    commit = checked_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT)
    release_path = ROOT / 'work_dir/ablr2' / lane_sensor(args.server) / args.server / 'runtime_release_ablr2x.json'
    original_path=release_path.with_name('runtime_release.json')
    original=json.loads(original_path.read_text()) if original_path.is_file() else None
    if args.dry_run:
        frozen = (Path(json.loads(release_path.read_text())['path']) if release_path.is_file()
                  else ROOT.parent / f'{ROOT.name}-runtime-ablr2x-{args.server}-{commit[:12]}')
        if frozen.exists():
            commit = checked_output(['git', 'rev-parse', 'HEAD'], cwd=frozen)
    else:
        from ablr2.deployment import frozen_checkout
        frozen = frozen_checkout(ROOT, args.server,extension=True)
        commit = checked_output(['git', 'rev-parse', 'HEAD'], cwd=frozen)
    with contextlib.ExitStack() as stack:
        if not args.dry_run:
            from ablr2.common import locked
            stack.enter_context(locked(release_path.parent / 'docker_launch.lock'))
        mounts = build_mounts(ROOT, frozen, args.server, dlpan)
        ids = checked_output(['docker', 'ps', '-aq', '--filter', 'label=' + LABEL + '.server=' + args.server]).split()
        containers = json.loads(checked_output(['docker', 'inspect', *ids])) if ids else []
        active = matching_running(containers, server=args.server, commit=commit, frozen=frozen, image_id=image_id,
                                  draining_release=original)
        if active:
            print(json.dumps(dict(status='ALREADY_RUNNING', container_id=active['Id'],
                runtime=str(frozen), image_id=image_id, lease_renewed=False), indent=2))
            return 0
        # Stopped containers are evidence, never removed/restarted behind the operator.
        suffix = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        name = f'pancrafter-ablr2-{args.server}-{commit[:12]}-{suffix}'
        command = build_command(root=ROOT, frozen=frozen, server=args.server, image_id=image_id,
            commit=commit, name=name, mounts=mounts, dlpan=dlpan, lease_hours=args.lease_hours,
            until_operator_stop=args.until_operator_stop,boundary=args.boundary,gpu=args.gpu, no_upload=args.no_upload)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        container_id = checked_output(command)
        launched = json.loads(checked_output(['docker', 'inspect', container_id]))[0]
        if launched['Image'] != image_id:
            raise RuntimeError('Docker image identity differs after launch; inspect container ' + container_id)
        print(json.dumps(dict(status='CONTAINER_STARTED', container_id=container_id, name=name,
            image_id=image_id, runtime=str(frozen), lease_hours=args.lease_hours,
            service_mode='UNTIL_OPERATOR_STOP' if args.until_operator_stop else 'FINITE_LEASE',
            automatic_restart=False, automatic_lease_renewal=False,
            log_command='docker logs --tail 80 ' + name,
            note='GPU/data preflight and actual training admission run inside the container.'), indent=2))
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
