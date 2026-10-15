# PCREPRO 구현·검증 기록

기준 계획: `PANCRAFTER_Reproduction_CasePlan_KR_2026-09-22.md`.
이 문서는 구현 결과이며 본 학습 결과가 아니다. 기존 실험 중지, GPU 학습, live Sheet 쓰기,
commit/push는 이 구현 작업에서 수행하지 않았다. 기존 사용자의 수정 및 로그 이동은 보존했다.

## 반영 범위

- `pcrepro/`: corrected Eq11 model, native data/augmentation, fresh50K trainer,
  atomic full-state, selection/evaluation, independent controller, safe handoff, reporting/upload.
- `tools/pcrepro_runner.py`, `tools/pcrepro_start.sh`: s3/s4/s5 전용 진입점.
- `config/pcrepro/s{3,4,5}_C000000.json`: 최초 cycle12개 case export. 최대 cycle 수가 아니다.
- 실행/자산 연결 설명: `pcrepro/README.md`, `pcrepro/DATA_BINDINGS.md`.

원본 MD에 기술된 recipe/bindings/acceptance CSV 등의 동봉 bundle은 저장소에서 찾지 못했다.
MD의 고정 설정을 구현했으며 `pcrepro/acceptance_derived.csv`는 **이 구현에서 작성한20개 점검표**다.
원본 acceptance CSV를 실행·통과했다고 주장하지 않는다. 실제 데이터 경로/SHA를 추정하여
검증 완료로 등록하지 않았으며, 빈 binding template을 제공했다.

## 중요한 구분

기존 PANDA/Aligner/KD/PANMIX 학습 경로는 변경하지 않았다. Eq11 attention의 query와
joint PAN K/V는 raw PAN repeated bands를 사용한다. PAN-mode residual의 Gaussian LP는
같은 augmented PAN에서 만들며 MS inference에는 해당 LP 경로가 없다.

실측 total=trainable parameters: **C4 7,129,476 / C8 7,207,816**.
기존 fixed gather convolution을 `unfold`로 바꿔 불필요한 고정 parameter가 없어졌다.
7.17M에 맞추는 depth/FFN 변경은 하지 않았다. heads/depth/filter 등 논문 미기재 선택은
recipe에 명시했으며 bit-exact 저자 재현이라고 주장하지 않는다.

이번 계획의 primary는 exact50K, secondary는 native val-ERGAS minimum이다.
기존 실험의 HQNR best 정책을 전역 변경하지 않았다. test RR/FR은 학습 중 선택에 쓰지 않는다.
WV2는 같은 서버·cycle의 WV3 두 선택점만 평가한다. 실패 시 이전 cycle Teacher로 대체하지 않는다.

## Sheet

전용 `PC-Repro-s3/s4/s5` 탭에 run당1행·두 선택점 별도 열을 사용한다.
UTC/KST 날짜, 실제 update, seed/cycle, source/data/checkpoint/evaluator SHA, 비용을 기록한다.
Q4/Q8은 band 수에 맞게 분리하며 RMSE·CC와 SRF-substitute JQM도 포함한다.
소수4자리는 표시 형식이고 원값은 보존한다. 전송 전 로컬 spool, idempotent upsert,
unformatted readback을 적용하며 Sheet 장애 때문에 학습을 실패 처리하지 않는다.
기존 WV3/GF2 탭에는 접근하지 않았다.

## 검증 및 운영 보강

CPU C128D224 C4/C8 full-model MARs forward/backward finite smoke를 통과했다.
오프라인 테스트는 모델, 데이터, 평가, 재개, 선택 증명, cursor, 안전한 전환,
Sheet 중복·형식·실패 처리와 실제 CLI 분기를 검사한다. PCREPRO 테스트 **128개 통과**를 확인했다.
기존 GFP40 테스트 **123개 통과**도 확인했다.

검토에서 발견해 수정한 항목:

- 마지막 case cursor 저장 직후 crash 시 index4에서 재개 불가 → idempotent cycle 마무리.
- 평가 실패가 이후 cycle에서 잊히는 문제 → durable pending evaluation 큐와 제한된 재시도.
- 제어 로그 timestamp 중복 및 CLI 분기 변수 scope 오류.
- 동명 다른 저장소 cron·보호 서버 route 오인식 및 재시작한 watchdog 누락.
- 서로 다른 server label의 같은 GPU 중복 admission → 물리 호스트 local lock.
- release와 work_dir가 다른 mount일 때 disk guard 오측정 → 실제 저장 볼륨 검사.
- 공식 FR 의존성 누락을50K 이후 발견 → DLPan import를 preflight로 이동.

## 실행 전 남은 실제 검증

각 서버의 원본 train/val/RR/FR binding 및 파일/tensor parity를 확정해야 한다.
WV3 RR20 중19 MAT만 대응되는 경우와 WV2 미확인 paper identity는 미검증 상태로 분리한다.
실제 GPU batch48 메모리, stop/resume, full50K, live Sheet 권한/readback은 아직 검증하지 않았다.
preflight batch1 smoke가 본 학습·batch48 검증을 대신하지는 않는다.
256/512 profile MAC×2는 포함 연산을 명시한 부분 집계이며 논문 전체 FLOPs와 동일하지 않다.

사용자가 실행을 승인한 뒤 commit된 코드와 로컬 binding으로 `pcrepro_start.sh`를 실행하면
해당 서버에서만 안전 전환 및 독립 cycle이 시작된다. source/data가 바뀌면 기존 진행 run을
조용히 이어가지 않고 차단한다.
