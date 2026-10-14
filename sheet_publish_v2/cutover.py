"""Explicit operator-only backup/verified publisher stop. Never starts training.

The normal publisher does not import this module or signal any process.
Linux pidfd is required for --stop-pid, preventing PID-reuse races.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import tarfile
import time

from .transport import atomic_json


def process_identity(pid):
    base = Path('/proc') / str(pid)
    # Fields following the parenthesized comm start with field 3 (state).
    stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(stat[19]), 'ppid': int(stat[1]),
            'uid': base.stat().st_uid, 'cwd': os.readlink(str(base / 'cwd')),
            'argv': [s.decode() for s in (base / 'cmdline').read_bytes().split(b'\0') if s]}


def verify_publisher(identity, server, repo):
    argv = identity['argv']
    expected = ['-m', 'edgeres48', 'publish', '--server', server, '--watch']
    if (len(argv) != 7 or argv[1:] != expected or
            not Path(argv[0]).name.startswith('python') or
            identity['uid'] != os.getuid() or identity['cwd'] != str(Path(repo).resolve())):
        raise RuntimeError('PID is not the exact same-user host legacy publisher; no signal sent')


def code_proof(root, repo):
    pin = json.loads((Path(root) / 'admission.json').read_text())['code']
    actual = {name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest() for name in pin}
    if actual != pin:
        raise RuntimeError('Training admission code pin changed; stop cutover and investigate')
    return actual


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def backup_sources(root, target, repo):
    """Immutable outbox/evals plus bounded log prefixes; original files untouched."""
    root, target = Path(root).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(os.getpid())
    archive = target / ('sources-' + stamp + '.tar.gz')
    manifest = {'root': str(root), 'archive': str(archive), 'created_unix': time.time(),
                'code_pin': code_proof(root, repo), 'files': [],
                'snapshot_semantics': 'Per-file snapshot; growing JSONL retained through captured byte length'}
    paths = list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    for directory in ('outbox', 'receipts'):
        paths.extend((root / directory).glob('*.json'))
    paths.extend((root / 'runs').glob('*/*/*.json'))
    # Source logs train.jsonl contain per-update history but are not publication
    # evidence; they remain in place and are deliberately not recopied here.
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1) as tar:
            for path in sorted(set(paths)):
                if path.is_symlink() or root not in path.resolve().parents:
                    raise RuntimeError('Source backup path escapes lane')
                with path.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    info = tarfile.TarInfo(str(path.relative_to(root)))
                    info.size, info.mtime, info.mode = stat.st_size, stat.st_mtime, 0o600
                    reader = HashingReader(stream)
                    tar.addfile(info, reader)
                    manifest['files'].append({'path': info.name, 'bytes': info.size,
                                              'sha256': reader.sha.hexdigest()})
        raw.flush()
        os.fsync(raw.fileno())
    sha = hashlib.sha256()
    with archive.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(data)
    manifest['archive_sha256'] = sha.hexdigest()
    manifest['archive_bytes'] = archive.stat().st_size
    path = target / ('sources-' + stamp + '.json')
    atomic_json(path, manifest)
    return path, manifest


def _pidfd_syscall(name, *args):
    # Older conda Python builds omit the wrappers despite a supporting kernel.
    # Resolve ABI numbers from the installed architecture header, never guess.
    if os.uname().machine != 'x86_64':
        raise RuntimeError('Python pidfd unavailable on this ABI; no signal sent')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    match = re.search(r'^#define __NR_' + re.escape(name) + r'\s+(\d+)\s*$', header, re.M)
    if match is None:
        raise RuntimeError('Installed Linux headers lack ' + name)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(int(match.group(1))), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    return _pidfd_syscall('pidfd_open', ctypes.c_int(pid), ctypes.c_uint(0))


def _signal_pidfd(fd):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, signal.SIGTERM)
    return _pidfd_syscall('pidfd_send_signal', ctypes.c_int(fd), ctypes.c_int(signal.SIGTERM),
                         ctypes.c_void_p(0), ctypes.c_uint(0))


def stop_verified(pid, expected, *, server, repo):
    verify_publisher(expected, server, repo)
    fd = _open_pidfd(pid)
    try:
        current = process_identity(pid)
        if current != expected:
            raise RuntimeError('Publisher PID identity changed; no signal sent')
        verify_publisher(current, server, repo)
        _signal_pidfd(fd)
        if not select.select([fd], [], [], 15)[0]:
            raise RuntimeError('Publisher did not terminate after SIGTERM; no force kill attempted')
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's2', 's3'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--publisher-pid', type=int, required=True)
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--stop-pid', action='store_true', help='Explicit SIGTERM after validated local backup')
    args = parser.parse_args(argv)
    if args.stop_pid and not args.backup:
        parser.error('--stop-pid requires --backup')
    identity = process_identity(args.publisher_pid)
    verify_publisher(identity, args.server, args.repo)
    proof = {'publisher_before': identity, 'code_pin_before': code_proof(args.root, args.repo),
             'training_signalled': False, 'stopped': False}
    if args.backup:
        path, manifest = backup_sources(args.root, args.audit_dir, args.repo)
        proof.update(backup_manifest=str(path), archive_sha256=manifest['archive_sha256'])
    if args.stop_pid:
        stop_verified(args.publisher_pid, identity, server=args.server, repo=args.repo)
        proof['stopped'] = True
    proof['code_pin_after'] = code_proof(args.root, args.repo)
    proof['at_unix'] = time.time()
    atomic_json(Path(args.audit_dir) / 'cutover.json', proof)
    print(json.dumps({k: v for k, v in proof.items() if not k.startswith('code_pin')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
