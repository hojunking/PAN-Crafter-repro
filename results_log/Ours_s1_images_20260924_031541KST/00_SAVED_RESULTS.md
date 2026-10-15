# 실제 이미지 저장 완료

2026-09-24 KST, 지정 TARGET 체크포인트로 추론을 완료했습니다. 원 코드·Docker 라이브러리 버전을 검증하고 CPU에서 실행했으며, 기존 GPU 학습은 중단하거나 변경하지 않았습니다.

## 바로 확인

- [WV3 RR 예측 PNG](Ours_s1_WV3_raw_20260924_031541KST/Ours/WV3/reduced/output/) / [WV3 FR 예측 PNG](Ours_s1_WV3_raw_20260924_031541KST/Ours/WV3/full/output/)
- [QB RR 예측 PNG](Ours_s1_QB_raw_20260924_031541KST/Ours/QB/reduced/output/) / [QB FR 예측 PNG](Ours_s1_QB_raw_20260924_031541KST/Ours/QB/full/output/)
- [WV3 비교 패널 예시](Ours_s1_WV3_raw_20260924_031541KST/Ours/WV3/reduced/compare_output/testimg0.png)
- [QB 비교 패널 예시](Ours_s1_QB_raw_20260924_031541KST/Ours/QB/reduced/compare_output/testimg0.png)
- [WV3 ZIP](Ours_s1_WV3_raw_20260924_031541KST.zip): 700,458,970 bytes
- [QB ZIP](Ours_s1_QB_raw_20260924_031541KST.zip): 403,967,357 bytes

## 저장·검증 결과

| 항목 | WV3 | QB |
| --- | --- | --- |
| TARGET step | 38380 | 47470 |
| 예측 배열 | RR20 + FR20 | RR20 + FR20 |
| dtype / 단위 / 축 | float32 / DN / CHW | float32 / DN / CHW |
| RR 크기 | 8×256×256 | 4×256×256 |
| FR 크기 | 8×512×512 | 4×512×512 |
| 예측 PNG | 40장 | 40장 |
| 참조·오차·비교 패널 포함 PNG | 220장 | 220장 |
| 원 H5/LP와 완전 일치한 참조 NPY | 180개 | 180개 |
| SHA256 검증 대상 내부 파일 | 499개 | 498개 |
| ZIP CRC 및 실제 SHA256 | 통과 | 통과 |

모든 예측 배열의 shape·dtype·유한값·DN 범위 [0,2047]를 검증했습니다. 각 예측 PNG는 전체 native frame을 보존합니다. H5의 원래 scene 인덱스와 대응하며, RR과 FR 사이의 동일 장면 대응을 주장하지 않습니다. 체크포인트, config, source, 데이터 및 LP 해시를 검증했고, selection/candidate의 checkpoint identity도 서로 일치했습니다.

WV3 ZIP SHA256:
`bd3f589ed6baa7b6d85f13942872a767df14957a81deace7b763953231d84a59`

QB ZIP SHA256:
`af6323c499ffdff33551bd429085885eed49bbd935b41eb5b9e477238c618cc5`

각 package의 `checksums.sha256`는 package 내부 파일을 검증하며, 위 ZIP SHA256 및 [EXPORT_RECEIPTS.json](EXPORT_RECEIPTS.json)은 실제 ZIP 전체를 검증합니다.

## CPU 재추론에 따른 차이

| 데이터 | 원 GPU HQNR | 저장 배열 HQNR | 원 GPU ERGAS | 저장 배열 ERGAS |
| --- | ---: | ---: | ---: | ---: |
| WV3 | 0.958732553 | 0.958735309 | 2.056060194 | 2.055864340 |
| QB | 0.925113552 | 0.925101552 | 3.561287322 | 3.560985665 |

지표는 저장된 NPY를 다시 읽어 원 evaluator로 계산했습니다. RR20은 원래 metric crop20:-21, FR20은 native PAN/LMS와 full512·masking 없음·장면별 HQNR 평균을 유지했습니다. Raw/PNG에는 crop을 적용하지 않았습니다.

원 GPU 실행과 CPU 재추론 사이에 차이가 있으며, 사전 허용 오차가 없어 비트 단위 재현·논문용 동등성 PASS를 선언하지 않았습니다(`PARITY_REVIEW_REQUIRED`). 각 package의 `metrics/metric_comparison.json`에 상세 차이를 기록했습니다. 기존 평가·스프레드시트·체크포인트 선택은 변경하지 않았습니다.

## PNG 용도

실제 모델 출력 배열을 직접 수치 변환한 확인용 미리보기입니다. 생성형 이미지는 사용하지 않았습니다. WV3 RGB는 [4,2,1], QB는 manifest의 R/G/B 밴드를 사용합니다. RR GT 또는 FR LMS에서 계산한 장면별 밴드별 1–99% stretch를 예측·참조에 공통 적용하고 gamma=1을 유지합니다.

CANConv 논문 비교 renderer/ROI 설정이 없으므로 최종 논문 비교 그림과 동일하다고 주장하지 않습니다. 전체 밴드 원본은 `Ours_raw/`, 원 test 참조는 `references/`, 렌더링 설정과 실행 provenance는 각 package에 포함되어 있습니다.

두 ZIP은 이 서버의 results_log에 저장되어 있습니다. 외부 업로드, s4 원격 작업, 학습 중단, Git commit/push는 이번 저장 작업에서 수행하지 않았습니다.
