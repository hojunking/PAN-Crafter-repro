"""Explicit R2 deployment helpers; importing this module never mutates runtime.

The legacy scheduler is source-bound by active checkpoints.  Only its unhashed
shell watchdog receives a routing hook, and only when explicitly requested by
the R2 start command.  All original shell bytes outside that hook are preserved.
"""
from pathlib import Path
import hashlib
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile


SERVERS = frozenset({"s1", "s2", "s3", "s4", "s5"})
BEGIN = "# BEGIN PANCRAFTER FH20R1 R2 ROUTE"
END = "# END PANCRAFTER FH20R1 R2 ROUTE"
ANCHOR = 'LOG="$REPO/work_dir/cases_chain.log"'
ROUTE = '''# BEGIN PANCRAFTER FH20R1 R2 ROUTE
# An explicit R2 registration owns recovery; never fall back to the old queue.
R2_POINTER="$REPO/work_dir/_fh20r1/r2_local_server.txt"
if [ -e "$R2_POINTER" ] || [ -L "$R2_POINTER" ]; then
  [ -f "$R2_POINTER" ] && [ -r "$R2_POINTER" ] || {
    echo "FH20R1 R2 registration unreadable; no legacy fallback" >&2; exit 2;
  }
  R2_SERVER="$(cat "$R2_POINTER")" || exit 2
  case "$R2_SERVER" in s1|s2|s3|s4|s5) ;; *)
    echo "FH20R1 R2 registration invalid; no legacy fallback" >&2; exit 2;;
  esac
  [ -f "$REPO/tools/r2_runner.py" ] && [ -r "$REPO/tools/r2_runner.py" ] || {
    echo "FH20R1 R2 controller missing; no legacy fallback" >&2; exit 2;
  }
  R2_PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
  [ -x "$R2_PY" ] || R2_PY=python
  exec "$R2_PY" "$REPO/tools/r2_runner.py" ensure --server "$R2_SERVER"
fi
# END PANCRAFTER FH20R1 R2 ROUTE
'''


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def install_watchdog(root):
    """Insert the fail-closed R2 route atomically; never install cron or start work.

    A pre-existing nonidentical/partial R2 hook is an integrity error, not an
    invitation to overwrite a local edit.  Caller serializes explicit activation.
    """
    root = Path(root).resolve()
    path = root / "tools/_watchdog.sh"
    if path.is_symlink() or not path.is_file():
        raise ValueError("R2 needs an existing regular, nonsymlink watchdog")
    original = path.read_bytes()
    mode = stat.S_IMODE(path.stat().st_mode)
    route = ROUTE.encode()
    begin, end, anchor = BEGIN.encode(), END.encode(), ANCHOR.encode()
    if begin in original or end in original:
        if (original.count(begin) != 1 or original.count(end) != 1
                or original.count(route) != 1 or original.count(anchor) != 1
                or route + anchor not in original):
            raise ValueError("Existing R2 watchdog hook is partial or modified; preserved")
        prefix = original[:original.index(route)]
        if b'REPO=' not in prefix or b'"--install"' not in prefix:
            raise ValueError("Existing R2 watchdog hook precedes required setup; preserved")
        updated = original
    else:
        if original.count(anchor) != 1:
            raise ValueError("Watchdog insertion anchor is missing/ambiguous; preserved")
        offset = original.index(anchor)
        if offset and original[offset - 1:offset] != b"\n":
            raise ValueError("Watchdog insertion anchor must start its own line")
        # Do not bypass the shell's --install behavior or its repository setup.
        prefix = original[:offset]
        if b'REPO=' not in prefix or b'"--install"' not in prefix:
            raise ValueError("Watchdog setup/install prefix is not recognized; preserved")
        updated = prefix + route + original[offset:]
    changed = updated != original
    if changed:
        fd, temporary_name = tempfile.mkstemp(prefix="._watchdog.r2.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(updated)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
            if path.is_symlink() or path.read_bytes() != original:
                raise ValueError("Watchdog changed during installation; refusing overwrite")
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {"path": str(path), "changed": changed,
            "before_sha256": _sha(original), "after_sha256": _sha(updated)}


def spawn(root, server):
    """Submit the R2 controller in the background; its own lock deduplicates it.

    This function does not acquire/release the legacy runner lock, restore holds,
    or claim that training has started.  Those are the controller's guarded work.
    """
    if server not in SERVERS:
        raise ValueError("R2 server must be one of s1..s5")
    root = Path(root).resolve()
    script = root / "tools/r2_runner.py"
    if not script.is_file():
        raise ValueError("R2 controller is missing; nothing launched")
    log_path = root / "work_dir/_fh20r1" / server / "priority_r2/runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with log_path.open("a") as stream:
        process = subprocess.Popen(
            [sys.executable, str(script), "run", "--server", server],
            cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        )
    return {"status": "R2_RUNNER_SUBMITTED", "pid": process.pid,
            "log_path": str(log_path)}


def _read_crontab():
    """Distinguish an absent crontab from an unreadable one before any write."""
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True, check=False,
        timeout=15, env=dict(os.environ, LC_ALL="C"),
    )
    if result.returncode == 0:
        return True, result.stdout
    no_table = re.fullmatch(r"(?:crontab:\s*)?no crontab for [^\r\n]+", result.stderr.strip())
    if result.returncode == 1 and not result.stdout.strip() and no_table:
        return False, ""
    raise RuntimeError("Cannot read existing crontab; no jobs changed: " + result.stderr.strip()[:240])


def _managed_cron_kind(line, watchdog):
    """Match only this repository's tagged watchdog jobs, never text mentions."""
    if line.lstrip().startswith("#"):
        return None
    tag = re.search(r"\s+#\s*PANCRAFTER-WATCHDOG\s*$", line.rstrip("\r\n"))
    if not tag:
        return None
    try:
        tokens = shlex.split(line[:tag.start()], comments=False)
    except ValueError:
        return None
    if tokens == ["*/15", "*", "*", "*", "*", watchdog]:
        return "periodic"
    if tokens in (["@reboot", "sleep", "120", "&&", watchdog], ["@reboot", watchdog]):
        return "reboot"
    return None


def install_watchdog_cron(root):
    """Explicit-start-only recovery registration, preserving unrelated cron text.

    Only this repo's tagged 15-minute/reboot jobs are managed.  A failed listing
    is not treated as an empty table.  An optimistic second listing detects edits
    before replacement; failed/mismatched readback is reported without rollback
    (which could overwrite a concurrent human edit).  No operation runs at import.
    """
    root = Path(root).resolve()
    watchdog = root / "tools/_watchdog.sh"
    if watchdog.is_symlink() or not watchdog.is_file():
        raise ValueError("Existing nonsymlink watchdog is required before cron registration")
    # Cron treats percent specially even inside shell quotes.  Avoid silently
    # installing a syntactically different command for such unusual paths.
    if any(char in str(watchdog) for char in ("\n", "\r", "%")):
        raise ValueError("Watchdog path contains unsupported cron metacharacters")
    command = shlex.quote(str(watchdog))
    expected = {
        "periodic": f"*/15 * * * * {command} # PANCRAFTER-WATCHDOG\n",
        "reboot": f"@reboot sleep 120 && {command} # PANCRAFTER-WATCHDOG\n",
    }
    existed, original = _read_crontab()
    lines = original.splitlines(keepends=True)
    managed = [(line, _managed_cron_kind(line, str(watchdog))) for line in lines]
    own = [(line, kind) for line, kind in managed if kind is not None]
    already = len(own) == 2 and {kind for _, kind in own} == set(expected) and all(
        line == expected[kind] for line, kind in own)
    if already:
        return {"status": "ALREADY_INSTALLED", "changed": False,
                "before_sha256": _sha(original.encode()), "after_sha256": _sha(original.encode()),
                "managed_entries": list(expected.values()), "recover_on_reboot": True,
                "readback_verified": True, "unrelated_entries_preserved": True}
    retained = "".join(line for line, kind in managed if kind is None)
    if retained and not retained.endswith("\n"):
        retained += "\n"
    updated = retained + expected["periodic"] + expected["reboot"]
    if _read_crontab() != (existed, original):
        raise RuntimeError("Crontab changed during R2 registration; no jobs overwritten")
    result = subprocess.run(
        ["crontab", "-"], input=updated, capture_output=True, text=True, check=False,
        timeout=15, env=dict(os.environ, LC_ALL="C"),
    )
    if result.returncode:
        raise RuntimeError("Watchdog cron registration failed: " + result.stderr.strip()[:240])
    readback_exists, readback = _read_crontab()
    if not readback_exists or readback != updated:
        raise RuntimeError("Watchdog cron readback differs; inspect current crontab (no rollback attempted)")
    return {"status": "INSTALLED", "changed": True,
            "before_sha256": _sha(original.encode()), "after_sha256": _sha(updated.encode()),
            "managed_entries": list(expected.values()), "recover_on_reboot": True,
            "readback_verified": True, "unrelated_entries_preserved": True,
            "created_new_crontab": not existed}
