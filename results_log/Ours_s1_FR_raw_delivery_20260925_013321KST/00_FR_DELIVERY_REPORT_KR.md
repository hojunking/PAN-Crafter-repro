# WV3·QB FR raw 보충 — 로컬 저장 완료

작업 ID: `PANDA_S1_WV3_QB_FR_RAW_SUPPLEMENT_v1`

- Raw: **FR_RAW_COMPLETE**
- 렌더링: **RENDERING_SKIPPED_BY_SCOPE**
- 전달: **READY_LOCAL_NOT_DELIVERED** — 이 서버에 저장했으며 외부 업로드는 하지 않았습니다.
- 장면 대응: **USER_CONFIRMED_PREVIEW_ORDER**. 사용자가 확인했다고 전달한 순서를 보존했으며 외부 PNG 대응을 새로 검증한 것은 아닙니다.

## 결과 파일

[FR 원본 예측 40개 ZIP](Ours_s1_WV3_QB_FR_raw_20260925_013740KST.zip)

실제 경로:
`/home/knuvi/Desktop/song/PAN-Crafter/results_log/Ours_s1_FR_raw_delivery_20260925_013321KST/Ours_s1_WV3_QB_FR_raw_20260925_013740KST.zip`

- 크기: **222,467,851 bytes**
- SHA256: `c71bbb928f844c6a1924d8e0786222f2e2737fe4f8130f9331036bf3fe29f7a8`
- ZIP CRC 및 모든 ZIP member SHA 검증: 통과.
- [기계 판독용 영수증](DELIVERY_RECEIPT.json)

ZIP 최상위에 바로 `Ours_raw/`, `manifest.json`, `scene_manifest.json`, `provenance/`, `validation/`, `checksums.sha256` 등이 있습니다. 감싸는 상위 폴더는 넣지 않았습니다.

## 고정 모델과 파일

| 센서 | 선택 | 파일 수 | shape | 형식 |
| --- | --- | ---: | --- | --- |
| WV3 | TARGET38380 | 20 | `(8,512,512)` | float32 / CHW / DN0–2047 |
| QB | TARGET47470 | 20 | `(4,512,512)` | float32 / CHW / DN0–2047 |

WV3 checkpoint SHA256:
`3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50`

QB checkpoint SHA256:
`548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d`

배열 위치:

- [WV3/full/pred](Ours_s1_WV3_QB_FR_raw_20260925_013740KST/Ours_raw/WV3/full/pred/)
- [QB/full/pred](Ours_s1_WV3_QB_FR_raw_20260925_013740KST/Ours_raw/QB/full/pred/)

각각 `testimg0.npy`부터 `testimg19.npy`까지이며, 원 FR H5의 행 index0–19와 동일합니다. 재정렬·crop·resize·재정합·추가 정규화·rounding·RGB 변환은 하지 않았습니다.

## 재사용·무결성 검증

2026-09-24 03시 export의 검증된 CPU FR 예측을 그대로 재사용했습니다. 이번 작업은 신규 모델 추론이 아니며, 새 package와 기존 NPY 사이의 바이트 및 배열 값 일치를 확인했습니다.

- 실제 checkpoint와 identity, TARGET selection, 원 candidate, resolved config 및 dataset manifest 연결 검증.
- 실제 FR H5와 LP 파일 SHA256 검증. QB 입력은 실행 시 resolved config의 실제 바인딩을 사용.
- 원 frozen source WV3 42개·QB 58개 해시 대조.
- NPY40개를 allow_pickle=False로 재읽어 번호·float32·CHW·전체 밴드·512×512·finite·DN 범위 검사.
- 원 source H5 SHA 및 각 prediction의 source index와 export index 일치 검사.
- 모든 파일 SHA와 ZIP 내부 배열 존재·member SHA·CRC 검사.
- WV3는 원 band index0..7을 보존하고 미확정 밴드 명칭을 새로 부여하지 않았습니다. QB는 manifest의 B/G/R/NIR 순서를 유지했습니다.

RR·GT·PAN/LMS 참조 배열·PNG·오차맵·모델 weight·optimizer·train dataset은 새 ZIP에 포함하지 않았습니다. 원 provenance JSON은 변조 없이 보존했으므로 기존 RR/FR 평가 메타데이터가 함께 기록되어 있을 수 있습니다.

## 저장 배열의 FR 재평가

저장된 NPY를 다시 읽어 원 frozen FR evaluator로 1회 재평가했습니다. Native PAN/LMS, full512, masking 없음, 장면별 HQNR 평균을 유지했습니다.

| 데이터 | 지표 | 원 GPU 선택 기록 | 저장 배열 재평가 | 차이 |
| --- | --- | ---: | ---: | ---: |
| WV3 | HQNR | 0.958732552970 | 0.958735308631 | +0.000002755661 |
| WV3 | D_lambda | 0.015799093758 | 0.015798772096 | -0.000000321663 |
| WV3 | D_s | 0.025886448899 | 0.025883962076 | -0.000002486823 |
| QB | HQNR | 0.925113552155 | 0.925101552319 | -0.000011999835 |
| QB | D_lambda | 0.044236486527 | 0.044298040608 | +0.000061554081 |
| QB | D_s | 0.032175615007 | 0.032125904888 | -0.000049710118 |

원 기록은 GPU 추론, 재사용 배열은 CPU 추론 결과입니다. 사전 허용 오차가 없으므로 **PARITY_REVIEW_REQUIRED**를 유지합니다. 이는 raw 무결성 통과와 별개이며, 다른 checkpoint로 바꾸거나 숫자를 맞추지 않았습니다.

[전체 정밀도·장면별 metric 기록](Ours_s1_WV3_QB_FR_raw_20260925_013740KST/validation/fr_metric_comparison.json)

## 실행과 보존

- 이번 복사·검증·재평가·압축: **2026-09-25 01:37:40~01:38:16 KST**, CPU 전용 Docker.
- 원 라이브러리 이미지: `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`.
- 원본·frozen checkout은 읽기 전용, 신규 결과 폴더만 쓰기 허용. 네트워크/GPU 접근 없음.
- s1 기존 학습 컨테이너 실행 유지. 중단·queue/watchdog 변경 없음.
- 이전 RR ZIP SHA256이 `4e7fec59623d48911aea3cc258264b062e1dc414cab385ad06cf84c750fd9c5f`로 그대로임을 확인했습니다.
- 이전 결과·스프레드시트·체크포인트 선택 변경 없음. Git commit/push 없음.
- 외부 전송만 미수행입니다. 로컬 ZIP을 사용자가 회수할 수 있으며, 로컬 생성만으로 외부 다운로드/전달 완료를 주장하지 않습니다.
