# TA2 publication boundary fix (2026-10-07)

The 78 registered workbooks in Drive folder 1lH61W1Enb0BGJYCzurnNIolOHaJFsx9e
receive detailed records. Core summaries and workbook links go to pan-cvpr27/analysis.
The existing admission row CU110 contains ="LANE:"&C110. Inspection of that
untouched boundary rejected all publication, including detail-manifest links.

The uploader now permits only this exact scalar Record_ID formula on a STATUS /
ADMISSION_V2 row when it is the preserved append boundary. Other formulas, all
validation/chips, headers, target rows and existing measured-row readback remain
guarded. Existing formulas are neither replaced nor copied. Atomic append,
duplicate-ID checks, numeric readback and durable receipts remain in force.

Validation: 58 publication/outbox tests including single/batch formula preservation,
retry idempotency, target/header/foreign formula rejection and marker protection.
No model, loss, data, seed, queue or active immutable training source was changed.
A short-lived Docker upload process uses the tested patch to pass the initial
boundary. Existing background publishers can then continue against appended rows.
Installation backup and runtime evidence: work_dir/aligner_teacher_lms_v2/s1/operator_upload_fix_20261007/.

## Sparse-row append correction and active deployment

Google values.append inferred a right-hand table from sparse rows and placed an
s5 detail manifest at DP114:JK114. The header remained intact, but the enlarged
grid correctly blocked subsequent publication. The proven misplaced row is
recovered to A114:EV114 with its original values, then the extra blank columns
are removed. Existing scientific results and CU108:CU110 formulas are preserved.

All new records now use numeric sheet-ID appendCells with complete typed CellData
and explicit formatting. This always starts in column A. Immutable Record_IDs
are located after atomic append; duplicate checks and numeric readback still gate
receipts. Timeout retry does not append a second copy. Sixty regression tests pass.

Active training uses an immutable older snapshot. A dedicated CPU-only upload
process inside the same Docker owns the existing outbox publication lock, so the
older worker cannot write. The producer continues enqueueing while learning.
Authoritative upload state: upload_status_operator.json in this server's TA2 lane.
The older upload_status.json may report lock contention and is superseded while
operator_upload_fix_20261007/worker_owner.json identifies the active owner.
The helper ends with the training container; future launches use repository code.
