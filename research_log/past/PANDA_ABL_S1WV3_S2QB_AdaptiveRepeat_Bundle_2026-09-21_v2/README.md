# ABLR2 — s1 WV3 / s2 QB

먼저 `PANDA_ABL_S1_WV3_S2_QB_AdaptiveRepeat_ExperimentPlan_2026-09-21_v2.md`를 읽는다.

기본 실행은 센서별 전체 구성 5회이며 두 센서 합계 Teacher 20 + Student 170 = 190개 학습이다.
각 Teacher의 calibration 20개는 별도 작업이다. 초기 구성과 seed는 BOOT5 CSV에 고정했다.

이 번들은 실험 **설계 자료**다. 원격 실행이나 trainer 연결을 수행하지 않았으며, JSON/CSV는 기존 runner에 바로 넣는 실행 파일이 아니다.
상시 controller에는 운영자 자원 허가, 모든 결과 보존, paired 재시험, 기술 오류 복구와 중지·재개 구현이 필요하다.

기존의 s1/s2 모두 QB 계획에서 s1 WV3/s2 QB로 변경했다. s3–s5 GF2에는 적용하지 않는다.
