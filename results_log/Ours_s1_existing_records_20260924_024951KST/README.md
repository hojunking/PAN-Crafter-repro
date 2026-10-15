# s1 기존 결과 기록 복사 — RAW 예측 미포함

상태: **METADATA_ONLY_RAW_MISSING**

2026-09-24 KST, 사용자의 "찾아서 복사만" 요청에 따라 원본 평가 기록과 출처 파일을
이 디렉터리에 복사했다. 11개 복사 파일(165,201 bytes)을 원본과 byte 단위 대조했다.
이것은 RR/FR full-band 예측 패키지가 아니다. 학습·추론·Sheet 쓰기는 수행하지 않았다.

## 확인된 대상

| 데이터 | selection | step | 원래 HQNR | 원래 ERGAS |
| --- | --- | ---: | ---: | ---: |
| WV3 | TARGET | 38380 | 0.9587325529698706 | 2.056060193721197 |
| QB | TARGET | 47470 | 0.9251135521548385 | 3.5612873217098007 |

두 평가 기록 모두 RR20/FR20이며 해당 checkpoint의 실제 파일 SHA256이 아래 요구값과
일치한다. 이는 기존 평가 기록의 확인이며 이번 작업에서 재평가한 수치가 아니다.

WV3 원 run:
`work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1`

WV3 checkpoint SHA256:
`3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50`

QB 원 run:
`work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1`

QB checkpoint SHA256:
`548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d`

## 복사된 파일

- 각 데이터셋의 `metrics/target_selection.json`, 선택 step의 `candidate_<step>.json`.
- WV3의 `metrics/step_38380.json`: 기존 supplemental metric 기록.
- 각 데이터셋의 `provenance/identity.json`, `config.resolved.yaml`, `dataset_manifest.json`.
- 원 source identity는 복사된 checkpoint identity와 selection JSON 안에 포함되어 있다.
- 설정 및 데이터 manifest의 canonical SHA가 checkpoint 기록과 일치함을 확인했다.

데이터 manifest 원본은 WV3 `work_dir/_fh12/s1/dataset_manifest.json`,
QB `work_dir/_qg40/s1/dataset_manifest.json`이다. H5 데이터 자체의 재해싱이나
논문 scene correspondence 재인증은 이번 복사 작업에서 수행하지 않았다.

## 없는 항목 / 진행하지 않은 작업

- 요청한 step의 full-band 예측 `.npy/.npz/.mat` 및 raw ZIP은 대상 run·기존 결과
  저장 위치에서 발견되지 않았다. QB의 다른 step 학습 diagnostic NPZ는 대상 예측이
  아니므로 복사하지 않았다. PNG도 지정 checkpoint provenance가 없는 것은 제외했다.
- raw RR20/FR20을 만들려면 지정 checkpoint로 **추론만 추가 실행**해야 한다.
  이번에는 "복사만" 범위를 유지하여 실행하지 않았다.
- 모델 weight, optimizer state, train 데이터, credential은 복사하지 않았다.
- s4 GF2 run `GFP40_GF2_S4_B09_MIX_G050_FT20K_RS98202_v1` 및 s4 운영 상태는
  이 s1 저장소에 없다. 원격 접속·s4 pause/inference/resume은 수행하지 않았다.
- s1/s4를 비롯한 기존 학습과 원본 파일은 변경하지 않았다.

`COPIED_FILES.sha256`은 위 11개 원본 복사 파일만 검증한다.
README와 checksum 파일은 안내용 신규 파일이며 raw 예측을 대신하지 않는다.
