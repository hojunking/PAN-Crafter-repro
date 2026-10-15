# bGlobal · 제공 LMS · RR-only 구현/검증 기록

- 기준: `research_log/PANDA_bGlobal_RR_Implementation_Changes_2026-10-13 (1).md`
- 추가 결정: FR **학습 loss 제거**, QB **원배포 LMS 사용**.
- 범위: method 모듈, prior 준비 도구, 실제 H5 연결, CPU 검증. 새 실험 case/queue/학습 실행/Sheets 갱신/commit은 하지 않았다.

## 1. 판정

**새 method의 수치 경로와 데이터 연결은 구현했다. 실험 시작 준비까지 완료된 것은 아니다.**

특히 명세 §10.1이 요구하는 WV3/QB/GF2의 실제 frozen b asset은 아직 없다.
명세 §10.2가 미정으로 남긴 matcher/consensus/near-zero 수치를 임의로 확정하지 않았기 때문이다.
현재 자산 생성 도구는 준비되어 있지만 실제 전체 train matching이나 실측 b 생성은 실행하지 않았다.

기존 scene-v1.3/RR+FR 코드와 체크포인트는 별도 보존했다. 새 method는
`panda_method.global_*` + `teacher48.global_training` 경로다. **기존 Teacher48
실행 명령을 돌리면 자동으로 새 RR-only method가 되는 것은 아니다.**
이는 method만 변경하고 실험 case는 후속 전달한다는 범위를 따른 것이다.

## 2. 구현 대조

| 요구 | 구현/확인 |
|---|---|
| scene mapping 제거 | `GlobalBatchLineage`와 `RRTrainingSources`는 dataset/sensor/source SHA만 사용. patch 번호를 scene ID로 위장하지 않음 |
| train-only b | `prepare_global_prior`는 등록 train H5의 PAN/LMS **전체 행**을 읽음. GT/MS는 읽지 않음 |
| 제공 LMS | stored LMS를 그대로 읽고, matcher에서 band mean. 온라인 모델 입력은 기존 `2*DN/peak-1` scaling만 적용 |
| signed 2D 집계 | 원 signed `(dy,dx)`를 weighted translation consensus → component-wise weighted median으로 집계. invalid는 weight=0, 후보/기각 사유 보존 |
| 부호/단위 | `aligned[y,x]=PAN[y+dy,x+dx]`; 양·음 y/x synthetic translation과 실제 warp로 검증. 64 PAN-grid 추정에 ×4 없음 |
| 증강 | Hflip → Vflip → CCW rot90를 PAN/MS/LMS/GT 및 b에 동일 적용. 기록된 변환과 b가 다르면 거부 |
| Head/골격 | 기존 LN common ResBlock, FC 66→32→2 유지. 마지막 2개 입력 raw signed b, FC2 zero-init → 최초 correction=b |
| RR-only S1 | `L_rec + L_abs_RR + 0.001*L_eps_RR`; Teacher 1회, Aligner 2회. FR 인자/loader 없음 |
| gradient | rec→A/T, abs→A, epsilon→shifted A; epsilon anchor에서만 native correction detach |
| epsilon | RR PAN pixels 기준 radius2 uniform-area disk; native/epsilon branch의 b 동일 |
| Stage2 | 하나의 frozen shared A/T, 동일 corrected PAN, train 전체 pixel e/q median, Student만 gradient |
| RR256 | whole-image Aligner. 자동 64 tile 분해나 prior 단위 변경 없음 |
| checkpoint | scene-v1.3과 새 bGlobal identity 분리. source/asset/conditioning/epsilon RNG/optimizer 순서 검증 |

실제 데이터→builder→optimizer update→checkpoint restore→다음 step 동일성까지
테스트했다. 정확한 calibration은 기존 complete train population 계약을 재사용하며,
실제 전체 train calibration은 실행하지 않았다.

## 3. QB 사용자 결정 반영

원배포 LMS와 현행 `msfix` LMS는 실제로 다르다. 요청은 LMS 교체이므로
**MS 복원까지 임의로 확대하지 않았다.**

- b 추정: 원배포 `train_qb.h5`의 PAN + LMS.
- 학습: 기존 `train_qb_msfix.h5`의 PAN/MS/GT + 원배포 `train_qb.h5`의 LMS.
- RR validation: 같은 원칙으로 원배포 `valid_qb.h5` LMS 사용.
- 두 파일을 각각 hash pin하고, 읽는 모든 행의 PAN/GT가 정확히 같아야 LMS를 합침.
- 원 LMS의 음수 보간 링잉은 clipping하지 않음.
- 공식 RR/FR test source를 원 train/validation으로 대체하지 않음.

추가 읽기 전용 표본 검사: train/valid 각각 균등 32행.

| 원배포 vs msfix | Train 32행 | Valid 32행 |
|---|---:|---:|
| PAN/GT | 모두 exact 일치 | 모두 exact 일치 |
| LMS RMSE, DN | 7.826775 | 6.671750 |
| LMS MAE, DN | 2.464015 | 2.246078 |
| LMS 최대 절대차, DN | 307.807337 | 325.521126 |
| MS RMSE, DN | 11.965468 | 11.029363 |

표본 원 LMS 최소값은 train −31.0736 DN, valid −20.6656 DN이었다.
이는 **32행 화소 차이 검사**이며 전체 행 동일성 또는 misalignment/phase 추정이 아니다.
원 LMS가 `interp23tap(msfix MS)`와 같다는 주장은 하지 않는다.

Aligner가 보는 제공 LMS와 U-Net reconstruction base는 별개다. 기존에 선택한
`BICUBIC_MS` base를 이번에 `NATIVE_LMS`로 몰래 바꾸지 않았다. 실제 case에서
base/layout을 명시해야 한다. 현 H5-to-Teacher builder는 LPAN cache가 없는
P0 경로를 지원하며 PL/PH/PLH는 빌드 시 설명 가능한 오류로 거부한다. Method
core는 명시적으로 공급된 LPAN을 이미 지원하지만 실제 cache 연결은 후속이다.

## 4. 실제 원본 파일 검사

다음 네 파일은 **전체 bytes SHA256**를 읽어 등록값과 비교했다. 모두 일치,
해시 전후 inode/크기/mtime/ctime 변화 없음. 총 17,346,904,736 bytes를 검사했다.

| 파일 | 행 수 | bytes | SHA256 |
|---|---:|---:|---|
| WV3/train_wv3.h5 | 9,714 | 5,570,398,376 | `f9e7ec381390ff36ddffd0714df7af07a3c5f456078b219a25f3e18372228f82` |
| QB/train_qb.h5 | 17,139 | 5,194,901,672 | `7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371` |
| GF2/train_gf2.h5 | 19,809 | 6,004,189,352 | `243a0bc8a4cc0a2740ed24409f9e87afe531d87e9d57524c6bcbec2b5f937621` |
| QB/valid_qb.h5 | 1,905 | 577,415,336 | `b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470` |

모두 PAN `[N,1,64,64]`, LMS/GT `[N,C,64,64]`, MS `[N,C,16,16]`.
C는 WV3=8, QB/GF2=4. 파일 hash는 출처 bytes를 고정하지만 geographic split
독립성이나 알려지지 않은 offline augmentation을 증명하지는 않는다.
QB msfix 파일의 전체 hash는 이 표의 재검사 대상이 아니며 데이터 adapter가
실사용 시 별도 검증한다.

## 5. 구현 감사 중 보완한 사항

1. Optimizer가 동일 파라미터 집합을 **다른 순서**로 받을 때 checkpoint state가
   엉뚱한 파라미터로 연결될 수 있었다. parameter-name/group 순서 저장·로드 전
   대조·매 step 재검증을 추가했다. 잘못된 재개를 모델 변경 전에 차단한다.
2. CPU Aligner 초기화의 `torch.manual_seed`가 CUDA RNG에도 영향을 줄 수 있어
   CPU default generator만 seed하고 CPU RNG를 복구하도록 변경했다.
3. canonical `(0.3,0.4)` magnitude 0.5를 FP32 변환 후 다시 norm 검사하면
   inclusive 경계에서 오판할 수 있었다. canonical asset에서 magnitude를
   검증하고 runtime에는 그 b의 정확한 D4 orbit만 허용하도록 했다.
4. 실제 prior source의 행 수/PAN·LMS shape, band order, 파일 변경을 추가 검사했다.
5. QB negative LMS를 실제 포함한 synthetic fixture로 clipping 부재를 확인했다.

## 6. 검증 결과

- `python -m unittest discover -s panda_method -t . -q`: **94 PASS**.
- `python -m unittest discover -s teacher48 -t . -q`: **206 PASS**.
- `python -m unittest ta2.test_model_loss -v`: **19 PASS**.
- 합계 **319 PASS**, 이 변경의 새 테스트는 **55개**(method16 + data9 + prior30).
- Python compileall, prior CLI help, 기존 tracked diff whitespace 검사 통과.
- 환경: 기존 `pancrafter` Python3.9/PyTorch2.4 CPU. 실제 GPU 학습·성능 비교 없음.

테스트에는 source-only 전체 행 방문, 외부 H5 링크/GT/MS 접근 차단,
signed vector·epsilon gradient·Stage2 freeze, wrong split/hash/legacy checkpoint,
원 LMS/MS 분리, negative ringing, 실제 adapter+optimizer 재개가 포함된다.
Synthetic 테스트 결과는 실제 b가 유효하거나 품질이 개선된다는 증거가 아니다.

## 7. FR loss와 기존 FR 평가의 구분

사용자 확인대로 **FR loss는 완전히 제외**한다. 이는 기존 pansharpening
FR 평가(HQNR/Ds/Dlambda)를 삭제하라는 뜻이 아니다. Metric 공식·평가 기준
이미지·클리핑/마스킹·HQNR 우선 선택 규칙은 이번 변경에서 수정하지 않았다.

별도 남은 것은 **새 Aligner가 FR512 입력을 받을 때 사용할 b**다. 명세 §7.2는
정책 기록 없이 b_RR 또는 4*b_RR를 공급하지 못하게 한다. 특히 raw b가 FC1
입력이므로 단위 변환은 conditioning 분포도 바꾼다. 현재 새 method의 FR
추론은 명확하게 거부하며 기존 모델의 공식 FR 평가는 그대로 남아 있다.
이 정책을 정하기 전에는 새 모델의 **HQNR-selected 전체 campaign**을 시작할
준비가 완료된 것으로 간주하지 않는다.

## 8. 추가 결정이 필요한 것

1. **b 추정/채택의 수치 정책**: search range, matcher confidence, 2D consensus,
   유효 후보 비율/개수, 방향 집중도, dispersion, near-zero 기준. 과거 LR16
   설정을 PAN64/제공 LMS에 검증 없이 이월하지 않았다. 먼저 train-only 전체
   분포를 진단한 뒤 결정하는 방식을 권한다. Test/HQNR 성적을 보고 prior
   기준을 고르는 절차로 바꾸어서는 안 된다.
2. **near-zero/다봉 분포 처리**: 현재는 frozen asset 생성 실패. offline 증강이나
   서로 반대 방향 patch 분포가 있으면 b가 상쇄될 수 있다. 임의 축/0 이동을
   넣지 않는다. 이때 global prior 가설 자체를 재검토할지 별도 결정해야 한다.
3. **실행용 spatial ROI/view/boundary 정책**: 기존 미정 값을 임의 채우지 않음.
   동적으로 loss pixel을 masking하는 방식도 추가하지 않았다.
4. **FR inference b 정책**: FR loss 삭제와 독립적인 결정. 자동 ×4나 무변환을
   어느 쪽도 대신 확정하지 않았다.
5. **후속 실험 case**: capacity/layout/base, LPAN 자산 필요 여부, optimizer,
   augmentation sampling·data order·scheduler·평가/업로드 연결. 현 변경에
   추가 실험을 임의로 생성하지 않았다.

사용법·모듈 진입점: `config/bglobal_rr/README.md`.
