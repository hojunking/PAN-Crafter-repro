"""Process-shared, host-local Google request pacing for the three-server run.

Each host gets at most 16 requests of each type in any 60-second interval
(4-second minimum spacing), so three hosts leave headroom below the shared
service-account limit of 60 reads and 60 writes/minute. This is not a distributed
quota coordinator: additional hosts or unrelated clients need separate budgeting.
State is in the launcher-mounted lane, outside frozen source snapshots, and
shared by manual/background containers. Never silently use container-local /tmp.
No request, particularly an uncertain append, is blindly retried here. A failed
request extends the shared cooldown; Outbox retries via immutable Record_ID.
"""
import email.utils
from contextlib import contextmanager
import os
from pathlib import Path
import random
import sqlite3
import time
from urllib.parse import urlparse

from gspread.http_client import HTTPClient


MIN_INTERVAL = 4.0


def request_bucket(method, endpoint):
    """Read-only Sheets POST methods consume the read, not write, budget."""
    parsed = urlparse(endpoint)
    service = 'sheets' if parsed.hostname == 'sheets.googleapis.com' else 'drive'
    read = method.upper() in ('GET', 'HEAD')
    if service == 'sheets' and parsed.path.endswith((':getByDataFilter', ':batchGetByDataFilter', ':batchGet')):
        read = True
    return service + (':read' if read else ':write')


class QuotaGate:
    def __init__(self, directory=None, *, clock=None, sleep=None, jitter=None):
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.jitter = jitter or random.SystemRandom().random
        # The launcher/controller binds this to the writable persistent lane.
        # /tmp is private per Docker invocation, so it cannot coordinate quotas.
        # No credential, principal name or scientific data is stored here.
        selected = directory or os.environ.get('TA2_GOOGLE_QUOTA_DIR')
        if not selected:
            raise ValueError('Set TA2_GOOGLE_QUOTA_DIR to the shared persistent lane/google_quota directory')
        self.root = Path(selected)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.root / 'quota.sqlite3'
        with self._db() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS quota '
                               '(bucket TEXT PRIMARY KEY, next_at REAL NOT NULL, failures INTEGER NOT NULL)')

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(str(self.database), timeout=30)
        try:
            connection.execute('PRAGMA synchronous=FULL')
            with connection:
                yield connection
        finally:
            connection.close()

    def wait(self, bucket):
        """Claim only a currently available slot; no lock is held while waiting."""
        waited = 0.0
        while True:
            with self._db() as connection:
                connection.execute('BEGIN IMMEDIATE')
                row = connection.execute('SELECT next_at, failures FROM quota WHERE bucket=?', (bucket,)).fetchone()
                current = self.clock()
                delay = max(0.0, row[0] - current) if row else 0.0
                if delay == 0:
                    connection.execute('INSERT OR REPLACE INTO quota VALUES (?, ?, ?)',
                                       (bucket, current + MIN_INTERVAL, row[1] if row else 0))
                    return waited
            # Bounded sleeps also honor a cooldown extended by another process.
            duration = min(delay, 30.0)
            self.sleep(duration)
            waited += duration

    def success(self, bucket):
        with self._db() as connection:
            connection.execute('UPDATE quota SET failures=0 WHERE bucket=?', (bucket,))

    def failure(self, bucket, retry_after=None):
        with self._db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT next_at, failures FROM quota WHERE bucket=?', (bucket,)).fetchone()
            failures = (row[1] if row else 0) + 1
            cooldown = max(MIN_INTERVAL, min(64.0, 2.0 ** min(failures, 6)) + self.jitter(),
                           float(retry_after or 0.0))
            deadline = max(row[0] if row else 0.0, self.clock() + cooldown)
            connection.execute('INSERT OR REPLACE INTO quota VALUES (?, ?, ?)', (bucket, deadline, failures))
        return cooldown


def _retry_after(response, current):
    raw = getattr(response, 'headers', {}).get('Retry-After')
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        try:
            return max(0.0, email.utils.parsedate_to_datetime(raw).timestamp() - current)
        except (ValueError, TypeError, OverflowError):
            return None


class LimitedHTTPClient(HTTPClient):
    """Injected into both live gspread constructors; fake Sheet tests stay fast."""
    def __init__(self, auth, session=None, *, quota=None):
        super().__init__(auth, session=session)
        self.quota = quota or QuotaGate()
        self.timeout = (10, 60)

    def request(self, method, endpoint, **kwargs):
        bucket = request_bucket(method, endpoint)
        self.quota.wait(bucket)
        try:
            response = super().request(method, endpoint, **kwargs)
        except Exception as exc:
            response = getattr(exc, 'response', None)
            code = getattr(response, 'status_code', None)
            # 403 is paced conservatively as well: Drive quota often uses 403.
            # Transport errors may hide a successful write, so propagate once.
            if code is None or code in (403, 429, 500, 502, 503, 504):
                self.quota.failure(bucket, _retry_after(response, self.quota.clock()))
            raise
        self.quota.success(bucket)
        return response
