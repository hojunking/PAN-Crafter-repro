# QG40 Sheet upload audit — 2026-09-20

Verified at 16:46 KST (07:46 UTC). Scope: current PANDA_QG40_20260920_v1 campaign, not a new audit of historical WV3 campaigns.

Spreadsheet: [pan-cvpr27](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit).

## Result

30 completed experiments, 30 verified Sheet uploads. Compared server-local training/postrun status and upload receipts against the actual Sheet rows, including run identity, sensor/server, actual updates (50,000), number evaluated (50), READBACK_VERIFIED, and metric presence. All five servers continued training.

| Server | Host | Dataset | Sheet tab | Completed / verified uploads | Latest upload KST | Active student seed / logged updates at 16:46 |
|---|---|---|---|---:|---|---|
| s1 | knuvi-hojun-local | QB (QuickBird), C4 | QB-s1 | 7 / 7 | 16:01:08 | 82005 / 25,700 |
| s2 | honbul | QB (QuickBird), C4 | QB-s2 | 1 / 1 | 16:44:47 | 82006 / 25,500 |
| s3 | song | GF2 (Gaofen-2), C4 | GF2-s3(5090) | 7 / 7 | 15:50:48 | 92010 / 44,900 |
| s4 | minte | QB (QuickBird), C4 | QB-s4 | 9 / 9 | 15:48:18 | 82016 / 44,100 |
| s5 | jeje | GF2 (Gaofen-2), C4 | GF2-s5 | 6 / 6 | 15:57:04 | 92015 / 39,100 |

Active runs use BASE student profiles. s1 uses QB_TB teacher seed 81002; s2/s4 use QB_TA seed 81001; s3/s5 use GF2_TA seed 91001. GPU utilization at this snapshot: s1 90%, s2 93%, s3 94%, s4 95%, s5 94%.

## s2 upload recovery

Initial audit found one completed but unuploaded experiment:
QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82003_BASE_FRESH50_v1.

It had completed training and official evaluation, but postrun reported:
ValueError: Duplicate Sheet header: Notes.

QB-s2!T3 and U3 both contained Notes. The additional U column contained historical notes; no historical row was removed or rewritten. Changed only QB-s2!U3 userEnteredValue from Notes to Legacy Notes, retaining its formatting. Existing Q4:U8 cell values and formats were compared before/after and were identical. Original native cell metadata is saved locally in 2026-09-20_qb-s2-header-before-repair.json.

Re-ran the existing official uploader inside pan-qg40-s2-20260920:
python tools/qg40_runner.py upload --run QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82003_BASE_FRESH50_v1

Command exited 0. Receipt verified row 9, gid 678743053, uploaded_at_utc 2026-09-20T07:44:47.714302+00:00. Live Sheet row 9 independently reports the exact run, QB, s2, 50,000 updates, 50 evaluations, READBACK_VERIFIED, ERGAS 3.97240546021252 and HQNR 0.9321191961347608. Header duplicates are now absent. The normal uploader extended the sheet to 159 columns for the existing QG40 schema. Google Sheets rendered view also verified the header and appended row.

The controller's in-memory state still contains the previous upload_pending=true for this run while the next training is active. This is a stale delivery flag: official/postrun_status.json says sheet_uploaded=true and official/upload_receipt.json says readback_verified=true. Existing qg40/controller.py retry_uploads updates it at the next controller loop/block boundary. No training was interrupted to force the bookkeeping refresh.

No numerical training code, source pin, dataset, checkpoint, or campaign clock was changed. Nothing was pushed to GitHub.

Machine-readable audit: 2026-09-20_qg40-sheet-audit.json.
