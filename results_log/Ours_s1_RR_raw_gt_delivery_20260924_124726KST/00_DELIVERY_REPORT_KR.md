# WV3·QB RR raw prediction + GT — 로컬 저장 완료

작업 ID: `PANDA_S1_WV3_QB_RR_RAW_GT_v2`

- 원시 배열 상태: **COMPLETE**
- 전달 상태: **READY_LOCAL_NOT_DELIVERED** — 이 서버에 저장했으며 외부 업로드/공개 공유는 하지 않았습니다.
- 렌더링: **SKIPPED_BY_USER** — PNG 생성·renderer 검색·calibration 없음.
- 외부 PNG 장면 대응: **NOT_CHECKED_BY_SCOPE**.

## 결과 파일

[RR prediction + GT 통합 ZIP](Ours_s1_WV3_QB_RR_raw_gt_20260924_125135KST.zip)

실제 경로:
`/home/knuvi/Desktop/song/PAN-Crafter/results_log/Ours_s1_RR_raw_gt_delivery_20260924_124726KST/Ours_s1_WV3_QB_RR_raw_gt_20260924_125135KST.zip`

- 크기: **81,541,515 bytes**
- SHA256: `4e7fec59623d48911aea3cc258264b062e1dc414cab385ad06cf84c750fd9c5f`
- ZIP CRC 및 ZIP 내부 모든 파일 SHA 검증: 통과.
- [기계 판독용 영수증](DELIVERY_RECEIPT.json)

## 대상 및 저장 내용

| 데이터 | 선택 | checkpoint SHA256 | 필수 NPY | 장면별 shape |
| --- | --- | --- | --- | --- |
| WV3 | TARGET 38380 | `3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50` | pred20 + GT20 | `(8,256,256)` |
| QB | TARGET 47470 | `548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d` | pred20 + GT20 | `(4,256,256)` |

총 **80개 NPY**이며, 모두 **float32 / CHW / DN / maxDN2047 / 256×256 전체 영상**입니다. 예측은 2026-09-24 03시의 검증된 CPU 추론 export를 바이트 그대로 재사용했습니다. 신규 모델 추론이나 학습은 실행하지 않았습니다.

GT는 원 RR H5의 `gt` key와 동일 index0..19에서 다시 읽고 float32로 저장했습니다. 두 데이터셋 모두 float64→float32 변환의 최대 절대오차가 **0**으로, 모든 화소의 값이 보존됐습니다. GT에 clipping·warp·crop·resize·정규화·색변환을 적용하지 않았습니다.

예측·GT의 동일 장면 여부는 원 H5 SHA·GT key·원 index와 기존 prediction scene manifest를 연결해 확인했습니다. 실제 checkpoint SHA, selection/candidate identity, config, 원 코드 파일, RR H5 및 LP SHA도 검증했습니다. WV3 밴드 명칭은 원 데이터에서 독립 확인하지 못했으므로 명칭을 새로 부여하지 않고 원 index0..7 보존으로 명시했습니다. QB는 원 manifest의 B/G/R/NIR 순서입니다.

PNG·FR·PAN/MS/LMS 추가 배열·모델 weight·optimizer·train dataset은 새 package에 넣지 않았습니다. 단, 원 provenance 기록은 변형하지 않았으므로 이전 FR 평가에 관한 메타데이터가 포함될 수 있습니다.

## 저장 배열 RR 재평가

저장된 float32 pred/GT를 다시 읽고 원 frozen evaluator 및 동일 라이브러리 버전으로 재계산했습니다. metric crop `[20:-21,20:-21]`, Q block32는 원 규칙대로이며 저장 배열은 전체 256×256을 유지했습니다.

| 데이터 | 원 GPU ERGAS | 저장 배열 ERGAS | 차이 |
| --- | ---: | ---: | ---: |
| WV3 | 2.056060193721197 | 2.055864339967293 | -0.00019585375390374082 |
| QB | 3.5612873217098007 | 3.560985665075832 | -0.0003016566339688609 |

SAM·PSNR·SSIM·SCC·Q8/Q4를 포함한 상세 비교는 [validation/rr_metric_comparison.json](Ours_s1_WV3_QB_RR_raw_gt_20260924_125135KST/validation/rr_metric_comparison.json)에 있습니다.

재사용 예측이 CPU 추론 결과이고 원 기록은 CUDA 결과이며, 사전 허용 오차가 없으므로 **PARITY_REVIEW_REQUIRED**를 유지했습니다. 원 숫자를 맞추기 위해 배열이나 checkpoint를 바꾸지 않았습니다. Raw 무결성 통과와 정량 재현성 인증은 별도 상태입니다.

## 실행 및 보존

- 이번 추출·검증·재평가: 2026-09-24 12:51:35~12:51:45 KST, CPU 전용 Docker.
- 원 실행 Docker 이미지: `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`.
- 원본 입력과 frozen checkout은 읽기 전용, 신규 저장 폴더만 쓰기 허용.
- 기존 s1 학습 컨테이너 및 GPU 실행 유지. 중단·queue/watchdog 변경 없음.
- 기존 결과/스프레드시트/체크포인트 선택 변경 없음. Git commit/push 없음.
- 두 sensor 모두 raw 추출 미완료 항목 없음. 외부 파일 전달만 미수행이며, 사용자가 위 로컬 ZIP을 회수할 수 있습니다.
