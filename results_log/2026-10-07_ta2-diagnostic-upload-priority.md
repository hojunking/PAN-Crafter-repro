# TA2 diagnostic publication scheduling — 2026-10-07

RESPONSE/GRAD measurements were generated but remained behind large, repeatedly
produced paired/seed aggregate queues. Earlier formula-boundary failures are
retained as history; the appendCells/readback repair is preserved.

The outbox now uses a durable weighted round-robin cursor. When all classes are
due, 70% of slots cover native correction, gain, alignment proxy and gradient;
the remaining slots preserve progress for quality, status, aggregates, assets
and other diagnostics. Empty classes donate their slots immediately.

An indexed, bounded delivery_class migration also classifies rows from the old
immutable training producer. Only transport metadata changes. Payload JSON/CSV,
scientific hashes, IDs, failure history, backoff, routes and receipts are kept.
65 Docker regression tests pass, including continuous aggregate load, one-row
calls across restart, legacy producer arrivals and unchanged payload/backoff.

The CPU upload worker runs in the existing training Docker. Its predecessor
finishes its current upload batch before handing over the same publication lock.
Training PID/container/source, data, checkpoints and scientific settings are
unchanged. Authoritative state remains upload_status_operator.json in this lane.
Worker evidence: operator_diagnostic_priority_20261007/worker_owner.json,
worker.log and installation.json. Future launches use ta2/reporting.py.

## Live verification

First diagnostic batch completed without errors. Correction, gain, alignment
proxy and measured gradient rows were independently read through Google Sheets
CellData: numeric values match local originals and VERIFIED/receipt hashes match.
The native Google-rendered rows were visually checked. Full backlog remains in
the running queue; this is not a claim that every pending record is delivered.
Evidence: operator_diagnostic_priority_20261007/verification.json.
