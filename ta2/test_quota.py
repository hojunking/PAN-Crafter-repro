import tempfile
import unittest
import os
from unittest.mock import patch

from ta2.quota import LimitedHTTPClient, QuotaGate, request_bucket


class Clock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class Response:
    def __init__(self, code=200, retry=None):
        self.status_code = code
        self.ok = code < 400
        self.headers = {} if retry is None else {'Retry-After': str(retry)}
        self.text = 'fake'

    def json(self):
        return {'error': {'code': self.status_code, 'message': 'fake quota failure', 'status': 'RESOURCE_EXHAUSTED'}}


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class QuotaTests(unittest.TestCase):
    def gate(self, folder, clock):
        return QuotaGate(folder, clock=clock.now, sleep=clock.sleep, jitter=lambda: .25)

    def test_same_host_snapshot_processes_share_spacing(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock()
            first, second = self.gate(folder, clock), self.gate(folder, clock)
            self.assertEqual(first.wait('sheets:read'), 0)
            self.assertEqual(second.wait('sheets:read'), 4)
            self.assertEqual(first.wait('sheets:write'), 0)
            restarted = self.gate(folder, clock)
            self.assertEqual(restarted.wait('sheets:read'), 4)
            self.assertEqual(clock.value, 1008)

    def test_no_silent_private_container_tmp_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'shared persistent'):
                QuotaGate()

    def test_budget_does_not_exceed_sixteen_per_minute_per_host(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); gate = self.gate(folder, clock)
            sent = []
            for _ in range(35):
                gate.wait('sheets:read'); sent.append(clock.value)
            for start in sent:
                self.assertLessEqual(sum(start <= value <= start + 60 for value in sent), 16)
            self.assertTrue(all(value <= 30 for value in clock.sleeps))

    def test_classifies_read_post_and_drive(self):
        self.assertEqual(request_bucket('POST', 'https://sheets.googleapis.com/v4/spreadsheets/id/values:batchGetByDataFilter'), 'sheets:read')
        self.assertEqual(request_bucket('POST', 'https://sheets.googleapis.com/v4/spreadsheets/id:getByDataFilter'), 'sheets:read')
        self.assertEqual(request_bucket('POST', 'https://sheets.googleapis.com/v4/spreadsheets/id/values/A1:append'), 'sheets:write')
        self.assertEqual(request_bucket('GET', 'https://www.googleapis.com/drive/v3/files'), 'drive:read')

    def test_shared_exponential_backoff_retry_after_and_bounded_sleeps(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); gate = self.gate(folder, clock)
            gate.wait('sheets:read')
            self.assertEqual(gate.failure('sheets:read'), 4)
            self.assertEqual(gate.failure('sheets:read'), 4.25)
            self.assertEqual(gate.failure('sheets:read'), 8.25)
            self.assertEqual(gate.failure('sheets:read', 75), 75)
            restarted = self.gate(folder, clock)
            self.assertEqual(restarted.wait('sheets:read'), 75)
            self.assertEqual(clock.sleeps, [30, 30, 15])
            restarted.success('sheets:read')
            self.assertEqual(restarted.failure('sheets:read'), 4)

    def test_uncertain_append_is_not_blindly_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); session = Session([TimeoutError('uncertain successful append'), Response()])
            client = LimitedHTTPClient(None, session=session, quota=self.gate(folder, clock))
            endpoint = 'https://sheets.googleapis.com/v4/spreadsheets/id/values/A1:append'
            with self.assertRaises(TimeoutError):
                client.request('post', endpoint, json={'values': [[1]]})
            self.assertEqual(len(session.calls), 1)
            client.request('post', endpoint, json={'values': [[1]]})
            self.assertEqual(clock.value, 1004)
            self.assertEqual(session.calls[-1]['timeout'], (10, 60))

    def test_429_cooldown_honors_server_retry_after(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); session = Session([Response(429, 61), Response()])
            client = LimitedHTTPClient(None, session=session, quota=self.gate(folder, clock))
            with self.assertRaises(Exception):
                client.request('get', 'https://sheets.googleapis.com/v4/spreadsheets/id')
            self.assertEqual(len(session.calls), 1)
            client.request('get', 'https://sheets.googleapis.com/v4/spreadsheets/id')
            self.assertEqual(clock.value, 1061)
            self.assertEqual(clock.sleeps, [30, 30, 1])

    def test_both_live_constructors_use_pacing_transport(self):
        from ta2.publication import DriveDetailFactory
        from ta2.reporting import connect, SPREADSHEET_ID
        with patch('gspread.service_account') as service:
            service.return_value.open_by_key.return_value.id = SPREADSHEET_ID
            connect('fake.json')
            self.assertIs(service.call_args.kwargs['http_client'], LimitedHTTPClient)
        with tempfile.TemporaryDirectory() as folder:
            with patch('google.auth.load_credentials_from_file', return_value=(object(), None)), patch('gspread.authorize') as authorize:
                DriveDetailFactory(folder, credentials='fake.json')._client()
                self.assertIs(authorize.call_args.kwargs['http_client'], LimitedHTTPClient)


if __name__ == '__main__':
    unittest.main()
