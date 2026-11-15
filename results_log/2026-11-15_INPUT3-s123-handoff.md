# INPUT3: RGBN 고정 donor에서 Student PAN 입력 비교

사용자 확정: 이전 QB donor3×schedule3 실험은 보류한다. s1=WV3,
s2=QB, s3=GF2로 돌아가 각 서버의 기존 main donor 하나를 고정한다.
새 Stage 1 학습 및 서버 간 weight 이동은 없다. 현재 실행은 checkpoint
저장 후 즉시 전환하고 기존 결과를 보존한다. 각 case는 1회씩이다.

| 순서 | Case | Student 입력 |
|---|---|---|
| 1 | A_RAW | 원본 PAN + interp23 I_lrms |
| 2 | B_RAW_CORRECTED | 원본 PAN + corrected PAN + interp23 I_lrms |
| 3 | C_CORRECTED_DIFF | corrected PAN + (corrected PAN−원본 PAN) + interp23 I_lrms |

세 case 모두 기존 `w=1+0.25*d_e*r_q`와 corrected PAN으로 계산하는 q를
유지한다. α=β=0, γ=.25, q=0일 때 r_q=0. A에서도 학습용 Teacher/q는
같은 Aligner를 사용한다. RGBN·Lanczos-4·공통 유효영역·D123/UBIL 유지.

| 서버 | Dataset | 기존 donor | 폭 | LR | Step |
|---|---|---|---:|---:|---:|
| s1 | WV3 | UNIFIED_WV3_S1_S1_RGBN_FULL_R1_S261016101 @27K | 112 | 1.1e-4 | 40K |
| s2 | QB | UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301 @32K | 128 | 1e-4 | 50K |
| s3 | GF2 | SEQ40_GF2_S3_S1_RGBN_FULL_R2_S261016302 @42K | 128 | 2.5e-4 | 50K |

GF2 LR은 앞서 목표 통과한 G3 LR250 기준. 각 데이터셋의 세 case는 동일
Student seed261018101, batch48/micro12, AdamW WD.01, warmup100와
전체 horizon cosine을 사용한다. 평가 지표에 따른 조기 중단은 없다.
총9회이며 새 Stage 1 재학습 없이 바로 Stage 2를 진행한다.

B와 C는 정보상 같은 두 PAN의 선형 재표현이다. 초기 예측까지 같게
맞추었으므로 결과 차이는 입력 표현과 학습 과정의 차이로 해석한다.
B의 첫 Conv PAN 가중치 `[0,W_pan]`, C는 `[W_pan,0]`으로 시작한다.
case별 입력 순서/정체성은 checkpoint에 저장하며 상호 로딩은 거부한다.

Sheet: 각 dataset의 `*-ablations`, Bucket=ablation, 소수4자리.
원본 PAN FR D:F 및 corrected PAN FR H:J를 동일 결과 행에 기록한다.
3개 모두 같은 donor의 corrected PAN이 평가 참조이며, 입력이 RAW라고
평가 참조까지 RAW로 바뀌지 않는다. RR/FR 매1K, weight 전부 보관.
기존 donor×schedule 큐는 자동 재개하지 않는다.

사용자 push 이후 s2/s3 에이전트 실행 명령:

```bash
git pull --ff-only
python3 -m panda_input3.release
python3 -m panda_input3.enqueue --server s2
# s3는 --server s3
```

`--check-only`로 사전 점검 가능. 신규 INPUT3 재개에만 `--resume` 사용.
기존 큐는 SIGTERM으로 정상 저장 후 미완료 행을 HELD_BY_USER로 표시하며
기존 weight/평가/log를 삭제하지 않는다. 다른 GPU 작업은 종료하지 않는다.
자세한 설정은 `panda_input3/README.md`, donor hash는
`config/panda_input3/donors.json`, 서버별 case는 같은 폴더의 CSV 참조.

검증: production Docker CPU 통합검증 5개 통과(23.534초). 세 데이터셋의
실제 구조를 직렬화/로드해 입력 채널, 기존 복원 함수와의 초기 예측 일치,
공통 guidance/mask, A/T 동결, checkpoint 재개, B↔C 잘못된 모델 로딩
거부를 확인했다. 3개 시트 경로와 원본/보정 FR 동일 행 기록도 확인했다.
호스트 Python에 PyTorch가 없는 경우의 큐 보류 경로도 별도로 검증했다.

s1 기존 실행은 R3 D_RQ_ONLY 26,250 step에서 정상 저장 후 보류했다.
남은 E_OURS R3도 HELD_BY_USER이며 이전 결과는 모두 보존했다.
신규 실행의 CUDA preflight/진행 상태는 `work_dir/panda_input3/` 아래
`preflight/verified.json`, `campaign_state.json`, `publication/last_poll.json`
으로 확인한다. s2/s3는 push·pull 후 각 서버의 로컬 donor 검증을 통과해야 시작된다.

s1 실행 확인: `panda-input3-s1-v1` 컨테이너에서 A_RAW 100 step 진행을
확인했다. 실제 CUDA preflight는 VERIFIED이며 모델/optimizer/RNG/stream
checkpoint 경계 복원이 모두 일치했다. UBIL CUDA backward 재실행은
기존 수치 허용오차 기준을 통과했다. 세 case의 guidance·공통 mask 일치,
B/C 초기 예측 일치, 동일 출력의 두 PAN 기준 FR 측정도 확인했다.
`WV3-ablations` 917·918·919행에 세 case 등록 및 readback을 확인했다.
이는 초기 상태 행이며 본 평가 수치는 첫1K 평가 후 게시된다.

배포 release SHA-256:
`7e26aef437b91d516f83092ecbe5c06e719a6f07e01325a388bd6068c5c74117` (375 files).
