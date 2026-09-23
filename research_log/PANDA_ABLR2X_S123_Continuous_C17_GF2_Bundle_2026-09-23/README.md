# ABLR2X s1/WV3 · s2/QB · s3/GF2

메인 문서: `PANDA_ABLR2X_S1_WV3_S2_QB_S3_GF2_Continuous_ExperimentPlan_2026-09-23.md`

**신규 C17 = C03-TZERO**를 진행 중인 s1/s2 cycle에 append하고, s3에는 GF2 native/fresh50K component cycle을 시작하는 설계입니다. 모든 서버는 독립적으로 사용자 중지까지 반복합니다. s4/s5는 보호합니다.

## 바로 볼 파일

- `registries/ABLR2X_S1S2_Append_C17_10.csv`: 원 최초5회에 C17만 5개씩 추가합니다.
- `registries/ABLR2X_GF2_S3_BOOT5_100.csv`: 새 GF2 최초5회, matched Teacher10 + Student90입니다.
- `registries/ABLR2X_First5_ReferenceUniverse_300.csv`: 전체 coverage 참고용입니다. **기존190개를 재시작하는 queue가 아닙니다.**

CSV의 `queue_rank`에 들어 있는 `.5`는 원 C03 뒤에 삽입하는 설계용 위치입니다. 원 runner의 integer rank 필드에 그대로 넣지 말고 `insert_after_run_id`와 새 extension scheduler를 사용합니다. 기존 source 검증과 frozen runtime을 우회하지 않습니다.

## 로컬 정적 검증

```bash
python build_registry.py --out registries
python -m unittest test_registry.py -v
```

이 스크립트는 manifest 생성·검증만 합니다. SSH, GPU 학습, Google Sheets 쓰기, live queue 수정, lease 생성은 하지 않습니다. 실행 binder에 필요한 실제 source/data/reference/runtime/authorization identity는 아직 null이며, 원격 배포·수치 parity·safe migration 확인 후 채워야 합니다.

읽은 기준코드에서는 s3/GF2가 금지되고 일부 maxDN=2047이 하드코딩되어 있으며 기본 lease는72시간입니다. 메인문서 §1·§6·§9의 구현 변경 없이 기존런처로 무한GF2학습이 가능하다고 해석하지 않습니다.
