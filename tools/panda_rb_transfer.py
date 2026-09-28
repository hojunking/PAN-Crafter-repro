#!/usr/bin/env python3
"""Local F1 handoff using pinned CPU Docker; no SSH/network/training launch.

S1: python3 tools/panda_rb_transfer.py export work_dir/RB_B01_F1_package
After copying that directory by an explicitly chosen transport:
S3/S5: python3 tools/panda_rb_transfer.py import work_dir/RB_B01_F1_package --data-paths local_data_paths.json
The JSON must map train/val/rr/fr to explicit local absolute H5 filenames.
Use --dry-run to inspect the Docker command without running it.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb.deployment import DEFAULT_IMAGE, asset_mounts, git_mounts
from panda_rb.plan import ROOT, binding_path, weights_dir


def _path(value, root):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _within_work(path, root, label):
    work = (root / 'work_dir').resolve()
    if path == work or not path.is_relative_to(work):
        raise ValueError(label + ' must be a specific path under work_dir (no symlink escape)')
    return path


def transfer(operation, package=None, root=ROOT, binding=None, weights=None, data_paths=None, dry_run=False):
    """Only local asset export/import/check; stdlib-only host imports."""
    root = Path(root).resolve()
    if operation not in ('export', 'import', 'check'):
        raise ValueError('Only local export/import/check are supported')
    bind = _within_work(_path(binding or binding_path(root), root), root, 'Binding')
    cache = _within_work(_path(weights or weights_dir(root), root), root, 'Weights')
    extra = []
    if operation in ('export', 'import'):
        if package is None:
            raise ValueError(operation + ' requires a package directory under work_dir')
        package = _within_work(_path(package, root), root, 'Package')
        if operation == 'import' and not (package / 'package.json').is_file():
            raise FileNotFoundError('Missing complete F1 package manifest: ' + str(package / 'package.json'))
    elif package is not None:
        raise ValueError('check takes no package directory')
    asset_args = ['--root', str(root), operation]
    if operation == 'export':
        asset_args += [str(package), '--binding', str(bind), '--weights', str(cache)]
    elif operation == 'import':
        if data_paths is None:
            raise ValueError('Import requires --data-paths JSON with all four explicit native H5 paths')
        data_paths = _path(data_paths, root)
        mapping = json.loads(data_paths.read_text())
        if not isinstance(mapping, dict) or set(mapping) != {'train', 'val', 'rr', 'fr'}:
            raise ValueError('Data JSON must contain exactly train, val, rr, fr')
        for split, filename in mapping.items():
            if not isinstance(filename, str) or not Path(filename).is_absolute() or not Path(filename).is_file():
                raise ValueError('Explicit existing absolute H5 filename required for ' + split)
            # Mount the named symlink path as well as its real storage target;
            # source bytes are validated inside Docker before publication.
            extra.extend([Path(filename), Path(filename).resolve()])
        extra.append(data_paths)
        asset_args += [str(package), '--data-paths', str(data_paths), '--destination', str(bind), '--weights', str(cache)]
    else:
        asset_args += ['--binding', str(bind)]
    command = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2',
               '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'HF_HOME=/tmp/panda_rb_hf',
               '-e', 'MPLCONFIGDIR=/tmp/panda_rb_mpl',
               '-v', f'{root}:{root}:ro',
               '-v', f'{root / "work_dir"}:{root / "work_dir"}:rw']
    mounts = set(git_mounts(root)) | set(asset_mounts(root, bind, extra))
    # asset_mounts resolves symlinks. An explicitly external symlink also needs
    # its original name available at the exact path supplied by the user.
    for path in extra:
        if path.is_symlink() and not path.is_relative_to(root):
            mounts.add(path)
    for path in sorted(mounts):
        if ':' in str(path) or '\n' in str(path):
            raise ValueError('Unsupported Docker bind path')
        command += ['-v', f'{path}:{path}:ro']
    command += ['-w', str(root), '--entrypoint', 'python', DEFAULT_IMAGE,
                'tools/panda_rb_assets.py', *asset_args]
    result = dict(operation=operation, image=DEFAULT_IMAGE, command=command,
                  training_started=False, remote_actions=False, dry_run=bool(dry_run))
    if dry_run:
        return result
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', DEFAULT_IMAGE], text=True))[0]['Id']
    if image != DEFAULT_IMAGE:
        raise ValueError('Local image differs from the pinned common B01 Docker image')
    subprocess.run(command, check=True)
    result['asset_operation_complete'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('operation', choices=['export', 'import', 'check'])
    parser.add_argument('package', nargs='?', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--data-paths', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = transfer(args.operation, args.package, args.root, args.binding, args.weights,
                      args.data_paths, args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
