# s3 PAN Aligner scale와 Hard 튜닝 실행 인계

사용자 요청에 따라 Git으로 전달하는 실행 인계다. 이 문서 작성 세션에서는
s3에 원격 접속하거나 실험을 시작하지 않았다. s3 에이전트가 로컬 상태를 확인하고
아래 30시간 큐를 시작한다. [전체 계획](PANDA_Scale30_HardTune_S13_2026-10-22.md)이
case·seed·평가의 기준이다. **Sheet 게시 규칙은 2026-10-23 사용자 지시로 아래와 같이 변경됐다.**

## 실행 중인 s3 게시기만 갱신할 때

Git 변경을 받은 뒤 다음 명령을 실행한다. 학습 컨테이너를 재시작하지 않는다.

```bash
bash scripts/start_panda_scale30_best_publisher.sh s3 --restart
```

새 정책은 `PANDA_SCALE30_RRE_STAGE2_H9885_PASS_ALL_OR_REPRESENTATIVE_V4`다.
`sheet_publish_panda_scale30/last_poll.json`의 `publication_policy`와
`readback_verified=true`를 확인한다. 상세 조건과 검증 절차는
[게시 규칙 변경 인계](gspread_rules/SCALE30_H9885_E2040_UPLOAD_2026-10-23.md)를 따른다.

## 로컬 준비

1. 이 커밋을 가져오되 사용자 작업과 실행 중인 소스를 덮어쓰지 않는다.
2. `gspread/server.txt`가 `s3`인지 확인한다. 인증 파일 내용은 출력·커밋하지 않는다.
3. 현재 GPU/Docker 작업을 확인한다. s1의 정지 지시를 s3의 다른 작업에 임의 적용하지
   않는다. ZERO30 정상 종료를 기다릴 경우 아래 대기 launcher를 쓸 수 있다.
4. 자체 원본 receipt와 해당 donor checkpoint가 존재해야 한다:
   `work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/s3/WV3/donors/rre_selection.json`.
   receipt의 path/SHA와 같은 A/T가 필요하다. s1 receipt/weight로 대체하지 않는다.
5. 기존 WV3 train/val/RR20/FR20 데이터, DLPan checkout, NVIDIA Docker 환경과
   검증된 기존 s3 이미지가 필요하다. 자동 탐색되지 않으면 `PAN_SCALE30_IMAGE`를
   해당 로컬 이미지로 지정한다. 새 이미지나 패키지를 임의 설치하지 않는다.
6. host Python은 기존 torch·google-auth·requests 환경을 사용한다. 필요하면
   `PAN_HOST_PYTHON`에 해당 실행 파일을 지정한다. `work_dir` 여유 공간도 확인한다.

## 실행

```bash
# 등록된 8개 case를 검토:
python -m panda_scale30 catalog --server s3 --csv

# GPU가 비어 있고 자체 donor가 준비됐을 때:
bash scripts/start_panda_scale30_costed.sh s3

# 대안: 기존 ZERO30 컨테이너와 campaign이 존재하고 정상 종료를 기다릴 때만:
# bash scripts/start_panda_scale30_after_zero30.sh s3
```

두 시작 명령을 동시에 실행하지 않는다. 대기 launcher는 기존 실험을 정지하지 않으며,
기존 실행이 비정상 종료되면 자동 시작을 거부한다. 대기시간은 새 예산에 포함하지
않는다. 새 cost 시작 시점부터 s3 자체 30시간이 시작된다. s1 날짜를 복사하지 않는다.

큐: `S020BASE R1 → S020H050 R1 → S020LR150 R1 → S020H200 R1 →
S020LR150H050 R1 → S010LR150H050 R1 → S020BASE R2 → [예비 S020LR150H050 R2]`.
모두 fresh50K, 단일 view, Soft=0, 회전 일관성=0, 같은 scale을 학습·추론에 적용한다.
예산 부족은 명시적 skip이며 30시간을 연장하거나 짧은 run으로 바꾸지 않는다.

## 확인과 종료

```bash
docker logs --tail 30 panda-scale30-hardonly-wv3-s3-v1
```

Campaign root는 `work_dir/panda_scale30/PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1/s3/WV3`.
`preflight/passed.json`의 모든 case PASS, `campaign_state.json`의 RUNNING과 증가하는
train step, `sheet_publish_panda_scale30/last_poll.json`의 readback을 확인한다.
게시 로그는 `work_dir/panda_scale30/publisher-s3-best-hqnr-v1.log`다.

시트는 `WV3-ablations`의 run/seed/attempt별로 두 게시 규칙을 적용한다.
HQNR ≥ **0.9885**, ERGAS ≤ **2.040**인 결과는 모두 게시한다.
HQNR ≥ **0.9885**인 평가 중 ERGAS 최소 1개는 **ERGAS 상한과 관계없이 대표로 유지**한다.
대표가 공동 조건도 만족하면 한 행으로 합쳐 중복 게시하지 않는다.
ERGAS 동률이면 HQNR 높은 결과 → 이른 step → source ID 순이다.
HQNR 하한을 만족하는 후보가 없으면 대표도 없다. V3의 대표 ERGAS gate는 사용자 확인에 따라 정정됐다.
Stage1 donor의 Val-RRE 선정·게시 규칙은 유지한다. 원본 config의 .959/2.045는
당시 기록으로 보존하며, 게시 판정은 새 기준으로 별도 계산한다.
동일 checkpoint의 모든 metric과 cost identity를 묶는다. 기존 다른 서버 행을
삭제·수정하거나 final50K 전용 행을 추가하지 않는다.

실패하면 로그/receipt/checkpoint를 보존한다. 자동 deadline 초기화, donor 재선정,
loss/seed/평가 변경 또는 원본 weight 삭제는 하지 않는다. 실행 시작·deadline·
최초 실제 학습 step·시트 readback을 s3 WIP에 기록하고 결과를 같은 서버의
S020BASE 대응 case와 비교한다.
