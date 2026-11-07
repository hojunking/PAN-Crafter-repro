# WV3 하이퍼파라미터 ablation 실행·s3 인계

2026-11-08. 사용자 승인 기준은 **기존 WV3 RGB W112, 현재 warp 경로 유지, 계수 효과만 비교**다.
실행 패키지: `panda_wv3_hyper`. 등록: `config/panda_wv3_hyper/plan.json`.

## 서버별 큐

| 서버 | 역할 | 설정·반복 | 신규 학습 | 주 비교 |
|---|---|---:|---:|---|
| s1 | frozen local A/T + Stage2 α·γ | 10설정×3=30 | 기존9회 재사용 검증 완료 →21회 | horizon30K, fixed29K |
| s3 | Stage1 λ_spa·λ_ε | 11설정×3=33 | 로컬 대조군 검증 후24–33회 | horizon/fixed50K |

모든 case는 **R1 전체 → R2 전체 → R3 전체** 순서다.

s1 순서: `H2_REF`, `H2_A0125`, `H2_G0125`, `H2_A0`, `H2_G0`, `H2_GT`,
`H2_A050`, `H2_G050`, `H2_A100`, `H2_G100`.
각각 α/γ는 (.25,.25),(.125,.25),(.25,.125),(0,.25),(.25,0),(0,0),(.5,.25),(.25,.5),(1,.25),(.25,1).
처음3개는 기존 M0/M2/M1 결과를 재사용한다.

s3 순서: `H1_REF`, `H1_SPA0`, `H1_EPS0`, `H1_SPA1`, `H1_SPA4`, `H1_E001`,
`H1_E010`, `H1_SPA05`, `H1_SPA8`, `H1_E0003`, `H1_E030`.
λ_spa/λ_ε는 (2,.003),(0,.003),(2,0),(1,.003),(4,.003),(2,.001),(2,.01),(.5,.003),(8,.003),(2,.0003),(2,.03).
처음3개는 s3의 기존 RGB A5/A4/A3 원본 전체 곡선·50K·seed·설정·검증 근거가 일치할 때만 재사용한다.
누락되면 그 case/seed를 fresh로 실행한다. s3 기존 결과의 실제 완료 여부를 s1에서 추정하지 않는다.

공통: scale1, β0, RGB L_spa/q, Aligner LMS 전체8밴드, interp23 복원 base,
Stage1/Teacher bicubic-border, Stage2 Student Lanczos4-reflection,
W112/D123 (Teacher BASE, Student UBIL), AdamW WD.01, batch48/micro12,
A LR1e-5 / T·S LR1e-4, warmup100→cosine0, 기존 paired augmentation.
Stage1 R_ε=3px disk. 회전 loss/TTA/SWA/밴드 가중 loss/width 변경은 없다.

s1 donor는 로컬 mainRGB의 R1@43K / R2@49K / R3@38K와 해당 train calibration이다.
같은 반복 안에서는 donor·Student 초기화·데이터 순서를 공유한다. 서버 간 weight를 이동하지 않는다.
s3은 독립 Stage1이므로 s1 donor 파일을 요구하지 않는다.

## s3 실행

사용자가 s1 commit을 push한 다음, s3 에이전트가 저장소에서 pull한다.
환경 이미지에 소스를 마운트하므로 이번 코드 변경을 위해 이미지를 재빌드할 필요는 없다.

```bash
git pull --ff-only
python3 -m panda_wv3_hyper.enqueue --server s3 --check-only
python3 -m panda_wv3_hyper.enqueue --server s3 --wait
docker logs --tail 40 panda-wv3-hyper-s3-v1
```

이미지는 기본 `hojunqueen/pancrafter-env:torch2.4.0-cu118`이다. 다른 이름으로 같은 환경을 보관했다면
`--image IMAGE`를 지정한다. 환경·metric hash가 일치해야 한다. 기존 작업이 있으면 `--wait`가 기다리고
다른 큐를 중단하지 않는다. 필요한 데이터는 WV3 train/val/RR20/FR20, 자동 게시 자격증명은 기존
`gspread/account.json`이다. 자격증명·weight·dataset을 Git에 넣지 않는다.

사전 검사는 소스482개와 데이터/평가기/환경, Sheet, CUDA update 및 완전한 resume을 확인한다.
실제 s3 GPU에서 사전 검사를 통과해야 학습이 시작된다. s1에서 s3 Stage1 코드의 CUDA 동작을
시험했으나, 이는 s3에서 실제 학습을 시작했다는 뜻이 아니다.

## s1 실행 및 보관

컨테이너: `panda-wv3-hyper-s1-v1`.
첫 신규 case: `WHYP_WV3_S1_H2_A0_R1_S261018101` (α0, γ.25, β0, LR1e-4, 30K).
출력: `work_dir/panda_wv3_hyper/PANDA_WV3_HYPER_S13_20261108_v1/s1/WV3/`.

```bash
python3 -m panda_wv3_hyper.enqueue --server s1 --status
docker logs --tail 30 panda-wv3-hyper-s1-v1
# 중단한 동일 소스/환경의 컨테이너 재개
python3 -m panda_wv3_hyper.enqueue --server s1 --resume
```

새로 학습하는 모든1K 평가 weight와 최근2세대 complete resume 상태를 보관한다.
29K/30K/50K endpoint는 목표 미달이어도 삭제하지 않는다. 기존 파일을 자동 정리하지 않는다.
재사용한 옛 모델8개는29K weight가 이미 삭제됐으므로 수치만 사용하며,
`METRICS_ONLY_WEIGHT_PRUNED`로 표시한다. 남은 M0 R2@29K weight는 실제 SHA 검증했다.

## Sheet 및 시작 시 게시기 수정

`WV3-ablations`에 case명·계수·donor·warp를 표시하고 소수점4자리로 게시한다.
Stage2 동시 조건(HQNR≥.9585 AND ERGAS≤2.040) 충족 checkpoint와 대표1개를 유지하며,
Stage1 validation 대표 및 **fixed29K/fixed50K**를 함께 게시한다. 동일 checkpoint는 한 행에 합친다.
고정 시점 결과는 미달이어도 게시된다. 수치 재사용 표시는 별도이며 원본 정밀도·SHA가 note에 남는다.
W112의 동일 추론 구조 실측 cost는 범위를 명시해 공유하고, 학습시간은 실제 trace에서 계산한다.

첫 시작 중 Google `values.append`가 중간 표를 선택하여 기존381–386행6개와 원래 빈387–411행에
신규 데이터를 썼다. 새 학습을67step에서 정상 정지하고, 원본 백업과 현재 셀을 대조한 뒤
**A381:BL411의 값·노트를 모두 원래대로 복구**, readback 검증했다.
게시기는 `appendCells`로 변경하고 기존 행이 바뀌지 않았는지 검증한다.
수정 후 신규33행은9550–9582에 연속으로 게시됐으며 readback을 통과했다.
최초67step과 관련 증거는 `s1/startup_attempt_01_transport_fix/WV3/`에 보관하고
동일 seed·설정으로 새 학습을0step부터 다시 시작했다. 평가된 신규 checkpoint는 정지 당시 없었다.

## 검증 근거

- 커밋 대상 소스만 추출한 환경에서 **26개 테스트 통과**: positive 경로 loss/gradient 동일,
  zero endpoint 식/support, Stage1 ε RNG 및 Stage2 완전 재개, 기존 결과 변조 거부,
  round-major, 서버 구분, 고정 시점·반올림·대표 행·중간 표가 있는 Sheet와 복구 범위.
- s1/s3 모델 경로의 실제 CUDA update, 원본 RGB 기준 RR/FR 출력 오차0, checkpoint 경계의
  model/optimizer/stream/RNG 정확 복구. 기존 bilinear CUDA backward의 다음 update는 rtol1e-6/atol1e-8 검사.
- s1 donor/calibration3쌍과 기존 control9개 원본 기록 검증 완료.
- 소스 manifest에 미추적 외부 패키지나 다른 미커밋 변경 의존 없음.

증거: `results_log/assets/2026-11-08-wv3-hyperparameter-implementation/`.
예상 신규 학습량은 s1 약36–40h, s3은 s1속도 환산 약62–67h(대조군 모두 신규이면86–92h)다.
실제 runtime/validation/게시 비용 및 s3 하드웨어에 따라 달라진다.
