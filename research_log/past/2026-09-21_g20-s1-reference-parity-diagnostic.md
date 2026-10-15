# s1 G20 참조 재현성 검증 중단 진단

확인 시각: 2026-09-21 02:36 KST. 상태: **BLOCKED_INTEGRITY, 미복구**.

s1은 00:39 KST에 R0 참조를 불러오다가 종료 코드 2로 중단됐다. G20의 `runs`와 `blocks`는 모두 비어 있으며 신규 학습은 시작되지 않았다. 기존 QG40 seed 82008·82009의 50,000-step 결과와 QB-s1 Sheet 20·21행은 앞선 점검에서 완료·업로드 확인됐으며 이번 진단에서 변경하지 않았다.

## 확인된 중단 조건

- RTX 5090에서 생성된 GF2_TA의 고정 R0 q-cache를 RTX 4090에서 계산한 값과 비교할 때 실패한다.
- 기본 실행에서 64개 중 9개 값이 `atol=3e-6, rtol=2e-5` 범위를 벗어났다. 최대 절대 차이는 `2.0056962966918945e-05`이고 가장 큰 허용 범위 대비 비율은 약 1.977이다.
- 원본 portable probe의 모델 출력도 동일한 원소별 기준을 통과하지 못한다. 최대 차이 `0.0001119375228881836`, 32,768개 중 10,632개 실패. 정렬 delta는 최대 차이 `2.864748239517212e-05`, 4개 중 4개 실패했다. 따라서 q-cache 검사만 우회해도 다음 검증에서 다시 중단된다.
- 모든 데이터 split과 LP의 동일성 검사가 통과했다. 원본 probe의 PAN/MS/LP 입력도 로컬 정규화 입력과 바이트 단위로 일치했다. 참조 체크포인트와 q-cache의 고정 해시는 유지됐다.

원인은 현재 GPU 간 수치 재현성 검증으로 좁혀졌다. 정확히 어느 GPU 연산이 차이를 만드는지는 아직 분리하지 못했다. 데이터 손상 또는 Torch 버전 차이만으로 설명할 근거는 없다.

## 수행한 검증

기존 운영 환경은 Torch 2.4.0 / CUDA 11.8이다. s3의 Torch 2.8.0+cu128 / cuDNN 91002 환경을 s1의 `/raid/PAN-Crafter-runtime/g20-s1-torch280-cu128`에 복사해 별도 컨테이너에서 시험했다. s1 RTX 4090을 정상 인식했고 공식 preflight 64개 테스트와 실제 CUDA forward가 통과했지만, R0 import에서는 **동일한 최대 q-cache 차이**가 재현됐다. TF32 해제, cuDNN 결정론 모드, cuDNN 해제, batch 1·2·4·8·32·64도 기존 기준을 통과하지 못했다.

별도 공식 same-checkpoint 전체 평가 재현 검사는 모두 PASS였다.

| 참조 | HQNR 절대 차이 | ERGAS 절대 차이 | 결과 |
|---|---:|---:|---|
| TA | 2.886676352398254e-7 | 2.986026676410347e-5 | PASS |
| S92001 | 5.276229537720667e-7 | 7.04394933603858e-7 | PASS |

공식 평가 기준은 HQNR `1e-5`, ERGAS `1e-4` 이하다. 이 통과 결과는 q-cache/모델 출력 검증 통과를 대신하지 않는다.

## 현재 상태와 재개 조건

운영 컨테이너는 종료 상태이고, 원래 `status.json`과 `registration.json`은 진단 전 백업과 바이트 단위로 동일하다. 실험 코드·검증 기준·τ·qref·R0 아티팩트·캠페인 시각을 변경하지 않았다. 새 라이브러리 환경은 진단용이며 운영 환경으로 승격하지 않았다. 진단 컨테이너는 모두 종료됐다. GitHub push는 하지 않았다.

배포 전 preflight가 통과했어도 서버별 R0 import의 수치 검증 통과가 보장되지 않았다. 이후 실행기가 참조를 불러오는 단계에서 실패하면서 s1 전체 블록이 시작되지 못했다. 같은 명령을 재실행하거나 같은 코드를 pull하는 것만으로는 이 문제를 해결할 수 없다.

재개하려면 원본 참조와 실험 정의를 보존하면서 RTX 4090에서도 기존 원소별 기준을 통과하는 실행 경로를 찾거나, GPU 간 검증 정책을 근거와 함께 개정해야 한다. 현재 증거만으로 임의로 허용 오차를 확대하거나 PASS로 기록하지 않았다. 원본 R0를 재보정하면 고정된 q-cache/τ/qref를 바꾸므로 복구 방법으로 적용하지 않았다.

공통 캠페인 시각은 유지돼 있다: 신규 블록 진입 마감 2026-09-21 14:05:13 KST, 학습 마감 16:05:13, 전체 마감 18:05:13. 지연된 시간을 임의로 연장하지 않았다.

## 코드 기준과 증거

- G20 배포 소스: `e2d838a3f9ea45170fb71052710b38acc9992c61`
- s1 고정 worktree HEAD: `8a63d10604901609facfd22b2b195724ea0122f6`
- 수치 소스 content SHA256: `cdb98d2c18dcfa3225102aab1846cdc5db812b283de15a73c7244924fd65bdab`
- R0 TA checkpoint SHA256: `0fb973376f9f00947d295620d96062b621145607aee311e28c889caefb656dab`
- R0 q-cache SHA256: `ddbb74a2a97661098a10db7bd9d17f5b7f7c5cb92b65ce39dd35751ed6b6d8ac`
- 운영 오류: `/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_g20/s1/reference_import.log`
- 서버 종합 진단: `/home/knuvi/Desktop/song/PAN-Crafter/.git/codex-g20-activation-20260920/runtime-repair-20260921/s1-final-diagnostic.json`
- 분리 진단 폴더: `/raid/PAN-Crafter-runtime/g20-s1-validation-workdir/_g20/s1/`
- 로컬 상세 증거: `2026-09-21_g20-s1-reference-parity-diagnostic.json`
