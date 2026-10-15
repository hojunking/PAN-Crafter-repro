"""Offline shared-code/document fingerprints; never pull, reset, or launch jobs.

Run on each server and compare the two JSON inventories. Same HEAD alone does
not establish working-tree equality. No credentials/data/model bytes are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SCHEMA = 'PAN_SHARED_TREE_V1'
LOCAL_ROOTS = {'data', 'work_dir', 'runs', 'outputs', 'analysis_runs', 'temp',
               'pan_h5', '.git', '.agents', '.codex', '.aws', '.ssh'}
TEXT_EXTENSIONS = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.toml',
                   '.ini', '.cfg', '.csv', '.tsv', '.cpp', '.cu', '.h', '.hpp', '.pyx',
                   '.jsonl', '.patch', '.sha256', '.html'}
SECRET_NAMES = {'account.json', 'credentials.json', 'token.json', '.env', 'id_rsa', 'id_ed25519'}
SECRET_PATTERNS = (
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'"(?:private_key|refresh_token|client_secret)"\s*:\s*"[^"\s]{16,}'),
)


def git(root, *args, binary=False):
    return subprocess.check_output(['git', '-C', str(root), *args], text=not binary)


def paths(root, *args):
    return [p.decode('utf-8') for p in git(root, *args, '-z', binary=True).split(b'\0') if p]


def shared_kind(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] in LOCAL_ROOTS:
        return None
    if (p.name.lower() in SECRET_NAMES or p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}
            or relative in {'gspread/server.txt', 'gspread/_profile_cache.json', 'config/_po10_gate_N3.yaml'}
            or any(part in {'__pycache__', '.ipynb_checkpoints'} for part in p.parts)):
        return None
    if (p.suffix.lower() not in TEXT_EXTENSIONS
            and p.name not in {'.gitignore', '.gitattributes', '.dockerignore', 'LICENSE', 'Dockerfile', 'Makefile'}
            and not p.name.startswith('Dockerfile.')):
        return None
    if p.parts[0] in {'research_log', 'results_log'} or p.suffix.lower() == '.md':
        return 'documents'
    if p.parts[0] == 'storage':
        return 'storage_catalog'
    return 'code_config'


def fingerprint(path):
    # Never dereference a symlink: a .json/.py alias can point at a credential,
    # a model, a multi-GB dataset or outside this repository.
    if path.is_symlink():
        raw = os.readlink(path).encode()
        return {'kind': 'symlink', 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'mode': '120000'}
    if not path.exists():
        return {'kind': 'missing', 'sha256': None, 'bytes': 0, 'mode': None}
    if not path.is_file():
        raise ValueError('Not a regular source file: ' + str(path))
    content = path.read_bytes()
    return {'kind': 'file', 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
            'mode': '100755' if path.stat().st_mode & 0o111 else '100644',
            'secret_pattern_flag': any(p.search(content) is not None for p in SECRET_PATTERNS)}


def digest_files(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inventory(root):
    root = Path(root).resolve()
    tracked = set(paths(root, 'ls-files', '--cached'))
    untracked = set(paths(root, 'ls-files', '--others', '--exclude-standard'))
    files, excluded = {}, Counter()
    for relative in sorted(tracked | untracked):
        category = shared_kind(relative)
        if category is None:
            excluded[relative.split('/')[0]] += 1
            continue
        if root not in (root / relative).parent.resolve().parents and (root / relative).parent.resolve() != root:
            raise ValueError('Shared path traverses an external directory symlink: ' + relative)
        item = fingerprint(root / relative)
        files[relative] = {**item, 'category': category, 'tracked': relative in tracked}
    groups = {}
    for category in ('code_config', 'documents', 'storage_catalog'):
        content = {p: {k: row[k] for k in ('kind', 'sha256', 'mode')}
                   for p, row in files.items() if row['category'] == category}
        groups[category] = {'count': len(content), 'sha256': digest_files(content)}
    return {'schema': SCHEMA, 'head': git(root, 'rev-parse', 'HEAD').strip(),
        'branch': git(root, 'branch', '--show-current').strip(), 'files': files, 'groups': groups,
        'status': git(root, 'status', '--porcelain=v1', '--untracked-files=no').splitlines(),
        'summary': {'shared_paths': len(files), 'shared_untracked': sum(not r['tracked'] for r in files.values()),
                    'missing_shared': sum(r['kind'] == 'missing' for r in files.values()),
                    'secret_pattern_flags': [p for p, r in files.items() if r.get('secret_pattern_flag')],
                    'excluded_counts': dict(excluded)},
        'limits': 'No remote access. Excludes credentials, datasets, weights, local run state and non-text assets. '
                  'Symlink hashes cover link text only, never target contents. Same code is not the same environment/data.'}


def compare(left, right):
    if any(value.get('schema') != SCHEMA for value in (left, right)):
        raise ValueError('Both inputs must use ' + SCHEMA)
    a, b = left['files'], right['files']
    changed = [p for p in sorted(a.keys() & b.keys())
               if any(a[p][k] != b[p][k] for k in ('kind', 'sha256', 'mode'))]
    left_only, right_only = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    return {'same_head': left['head'] == right['head'], 'left_head': left['head'], 'right_head': right['head'],
        'same_shared_files': not (changed or left_only or right_only),
        'changed': changed, 'left_only': left_only, 'right_only': right_only,
        'tracking_differs': [p for p in sorted(a.keys() & b.keys()) if a[p]['tracked'] != b[p]['tracked']],
        'counts': {'changed': len(changed), 'left_only': len(left_only), 'right_only': len(right_only)}}


def compatibility_links(root):
    """Explicit operator-approved bulk link creation; never replace a file.

    Only deleted tracked research/result paths whose exact Git blob survives
    at another shared path qualify. No content copies, resets or Git writes.
    """
    root = Path(root).resolve()
    report = review(root)
    planned = []
    for old, candidates in report['exact_content_relocations'].items():
        if not old.startswith(('research_log/', 'results_log/')):
            continue
        preferred = 'research_log/past/' + old[len('research_log/'):] if old.startswith('research_log/') else None
        chosen = (preferred if preferred in candidates else
                  min(candidates, key=lambda p: (Path(p).name != Path(old).name, len(Path(p).parts), p)))
        source, target = root / old, root / chosen
        if root not in source.parent.resolve().parents and source.parent.resolve() != root:
            raise ValueError('Compatibility parent escapes repository: ' + old)
        if os.path.lexists(source):
            raise ValueError('Refusing to replace existing path: ' + old)
        if target.is_symlink() or not target.is_file() or root not in target.resolve().parents:
            raise ValueError('Archive target must be an in-repository regular file: ' + chosen)
        planned.append((old, chosen, os.path.relpath(target, source.parent)))
    # All paths checked first; exclusive symlink creation never overwrites a
    # concurrently recreated file. Every source remains recoverable in Git.
    for old, chosen, relative in planned:
        source = root / old
        source.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(relative, source)
    return {'created': len(planned), 'links': [{'original': old, 'archive': chosen, 'link': link}
                                             for old, chosen, link in planned],
            'unmatched': report['deletions_without_exact_untracked_copy']}


def review(root):
    """Classify existing changes, including exact-content archived relocations."""
    root = Path(root).resolve()
    value = inventory(root)
    deleted = paths(root, 'ls-files', '--deleted')
    untracked = paths(root, 'ls-files', '--others', '--exclude-standard')
    index = {}
    for line in git(root, 'ls-files', '--stage', '-z', binary=True).split(b'\0'):
        if line:
            meta, name = line.split(b'\t', 1)
            index[name.decode()] = meta.split()[1].decode()
    candidates = [p for p in untracked if shared_kind(p) and (root / p).is_file() and not (root / p).is_symlink()]
    # Hash bytes directly without storing Git objects or applying conversion filters.
    by_blob = defaultdict(list)
    for p in candidates:
        data = (root / p).read_bytes()
        key = hashlib.sha1(('blob ' + str(len(data))).encode() + b'\0' + data).hexdigest()
        by_blob[key].append(p)
    moves = {old: by_blob[index[old]] for old in deleted if by_blob.get(index.get(old))}
    groups = defaultdict(list)
    for p in candidates:
        groups[shared_kind(p)].append(p)
    return {'head': value['head'], 'summary': value['summary'],
            'untracked_shared_by_kind': dict(groups), 'exact_content_relocations': moves,
            'deletions_without_exact_untracked_copy': sorted(set(deleted) - set(moves)),
            'tracked_modified': paths(root, 'diff', '--name-only', '--diff-filter=M'),
            'local_or_binary_untracked': [p for p in untracked if shared_kind(p) is None]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inventory', 'review'):
        sub.add_parser(name).add_argument('--output')
    links = sub.add_parser('compat-links', help='Explicitly create exact-content archive compatibility links')
    links.add_argument('--apply', action='store_true')
    links.add_argument('--output', required=True)
    diff = sub.add_parser('compare')
    diff.add_argument('left'); diff.add_argument('right')
    diff.add_argument('--output')
    args = parser.parse_args(argv)
    if args.command == 'compat-links' and not args.apply:
        parser.error('compat-links requires explicit --apply; inventory/review never change repository files')
    if args.output and Path(args.output).exists():
        parser.error('Output already exists; never overwrite a prior inventory or receipt')
    result = (compatibility_links(args.root) if args.command == 'compat-links' else
              inventory(args.root) if args.command == 'inventory' else review(args.root)
              if args.command == 'review' else compare(json.loads(Path(args.left).read_text()),
                                                       json.loads(Path(args.right).read_text())))
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reports are explicit outputs, never overwrite code or an existing report.
        with target.open('x') as stream:
            stream.write(encoded)
        print(json.dumps({'output': str(target), 'summary': result.get('summary', result.get('counts'))}, ensure_ascii=False))
    else:
        print(encoded, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
