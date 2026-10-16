"""Reuse pinned selection/evidence rules, replace only host upload transport."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'restart':
        from .restart import main as restart
        return restart(argv[1:])
    if not argv or argv[0] not in ('publish', 'publish-auto'):
        raise SystemExit('Use publish, publish-auto, or restart --server s1|s3 --apply')
    from donor24 import publication
    from donor24.__main__ import main as original
    from .transport import reconcile_rows
    # In-memory substitution in this uploader process only. The running GPU
    # process and pinned donor24/*.py files are not modified.
    previous = publication.reconcile_rows
    try:
        publication.reconcile_rows = reconcile_rows
        return original(argv)
    finally:
        publication.reconcile_rows = previous


if __name__ == '__main__':
    raise SystemExit(main())
