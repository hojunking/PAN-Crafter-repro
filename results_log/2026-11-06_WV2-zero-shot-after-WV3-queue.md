# s1 WV3 완료 후 WV2 zero-shot 평가 3건

사용자 요청: M5-LR110 R3 29K와 WV3 우수 모델 두 개를 WV2에서 평가하고
`WV2-ablations`에 게시한다. 현재 진행 중인 실험 큐가 전부 끝난 뒤 실행한다.

| Case | 모델 | Step | Width | WV3 HQNR | WV3 ERGAS |
|---|---|---:|---:|---:|---:|
| Z1_M5_R3_29K | M5-LR110 R3 (RGB) | 29K | 112 | 0.9606112 | 2.0394382 |
| Z2_W128_W1_R1_30K | W128 W1-G025-LR100 R1 (RGB) | 30K | 128 | 0.9597951 | 2.0327496 |
| Z3_M0_R2_28K | M0-OURS R2 (RGB) | 28K | 112 | 0.9591662 | 2.0393692 |

표의 수치는 **모델 선정 근거인 WV3 결과**이며 WV2 결과가 아니다.
W128은 보존된 main 목표 통과 결과 중 낮은 ERGAS를 기준으로, M0 R2는 다른
seed/donor의 W112 기본 recipe 비교를 위해 선정했다. WV2를 본 뒤 선택하지 않는다.

현재 W128 18-run 큐의 전체 완료 및 GPU 유휴를 확인한 뒤 세 평가를 순서대로 실행한다.
WV2 추가 학습·통계 재보정·TTA 없이 source의 Aligner+Student를 그대로 쓴다.
RR20/FR20 원본 데이터와 8밴드 순서를 고정하며 FR 지표에는 WV2 MTF를 적용한다.
FR H5와 MAT20의 PAN/MS/LMS 전체 장면이 동일한 것도 확인했다.

결과는 고정한 세 행에 4자리로 기록하고, 설정·원본 checkpoint SHA·원래 WV3 성능은
note에 남긴다. WV2 목표선이 지정되지 않아 세 결과 모두 게시한다.
실제 WV2 성능 평가는 대기 상태이며, 이번 구현 검증은 CPU 합성 입력과 원본 증거 확인이다.

2026-11-06 14:01 KST에 대기 컨테이너
`panda-wv2-zero-shot-after-wv3-s1-v1`을 등록했다. 등록 시 WV3는 2/18 완료,
WV2는 `WAIT_WV3_QUEUE`였으며 GPU에는 기존 WV3 학습 프로세스만 존재했다.
`WV2-ablations` 6–8행의 대기 표시·빈 성능 셀·4자리 표시 형식을 재조회해 확인했다.
CPU 검증 5개와 실제 실행 이미지의 데이터·모델·Sheet 사전 검사가 통과했다.

[실행·검증·출력 안내](../panda_wv2_eval/README.md)

원본 및 source 사본을 삭제하지 않는다. source 사본은 새 평가 root 아래 `sources/`에 있으며
동일 s1 서버 안에서만 보존한다. 다른 서버의 weight를 가져오지 않는다.
