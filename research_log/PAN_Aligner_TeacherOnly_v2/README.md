# Teacher-only native LMS / direct structure experiments v2

먼저 `PAN_Aligner_TeacherOnly_NativeLMS_GTStructure_S135_3Seeds_2026-10-06_v2.md`의 §0–6과 §9–10을 읽는다.

52조건 × 3 seed × 3 dataset = 468개 학습 정의이며, 실제 기동은 수행되지 않았다. s1=WV3, s3=QB, s5=GF2만 대상이다.

`python validate_plan.py`는 등록 수·seed·case·schema의 정적 검사일 뿐 학습 실행기가 아니다.

- 전체 registry: `TeacherOnly_Run_Registry_468.csv`
- 서버별 queue: `s1_WV3_Queue_156.csv`, `s3_QB_Queue_156.csv`, `s5_GF2_Queue_156.csv`
- 업로드: `analysis_upload_template_v2.csv`와 column dictionary
- 시트 변경 영수증: `sheet_update_receipt.json`
- 근거: `sources/`의 보고서와 superseded v1 계획

어떤 plan/placeholder 값도 실측 성능으로 집계하지 않는다. GT는 학습 구조 지도와 사후 진단에만 사용하고 공식 inference에는 주지 않는다.
