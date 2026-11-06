# 2026-11-06: WV3-main 4자리 / s3 GF2 W128 후속 실험

## WV3-main 적용

사용자 요청에 따라 **실제 업로드 숫자와 표시 형식 모두 소수점4자리**로 전환했다.
원본 JSON 및 note의 정밀도, 목표 판정, 대표 모델 선택 기준은 유지한다.
기존 W11222행을 원본 metric/cost/timing에서 재구성했고, W12818행(진행1·대기17)도4자리로 게시했다.
Sheet connector readback에서40행의4자리 형식과 기존 W11222행의 값·note 일치를 확인했다.

학습 파일은 source pin 때문에 수정하지 않고 `panda_wv3_reporting4`를 별도로 구현했다.
W128은20,442step에서 완전한 재개 checkpoint를 저장하고 새 게시기를 사용해 이어서 학습한다.
469개 원본 source hash는 `43009d0b92cb9a1b6b9144aabdf97f37412d4119ecd5620b47c160736b816317`로 동일하다.
원본 optimizer/scheduler/RNG/stream/config를 유지하며, 기존 W112 삭제 weight는 retirement ledger로 검증한다.
이제 기존 RGB main을 재게시할 때도 [4자리 entrypoint](../panda_wv3_reporting4/README.md)를 사용한다.

## GF2 설계 근거

2026-11-06 live Sheet의 GF2/ QB 원본 configuration과 기존 보존 문서를 대조했다.

- GF2 최종 목표 달성: **GF2P_C60LR200 R2 @47K**, W112/LR2e-4/alpha0.25/gamma0.50/60K.
  HQNR0.9646694883, ERGAS0.5148830159, PSNR45.0841380957.
- QB 유력 폭 설정: **Q2_W128 R1**, W128/alpha0.25/gamma0.50.
  QB의30K·LR1e-4를 GF2에 그대로 복사하지 않고 GF2의 검증된60K·LR2e-4를 중심으로 설계한다.
- GF2 W128/LR1e-4는 이미 R1/R2 실험이 있으며 게시 대표의 PSNR은 약45.00/45.02다.
  이 조합을 같은 seed로 반복하지 않고 LR1.5e-4·2e-4·2.5e-4를 확인한다.
- 기본 adaptive Hard 가중과 두 계수를 함께 절반으로 줄이는 H050을 비교한다.
  W128이 GF2에서도 개선한다고 결론 내린 상태는 아니며, seed별 비교로 확인할 실험이다.

## 큐

| 순서 | Case | Width | LR | alpha | gamma | Steps |
|---|---|---:|---:|---:|---:|---:|
| 1 | G1_LR200 | 128 | 2e-4 | 0.25 | 0.50 | 60K |
| 2 | G2_LR150 | 128 | 1.5e-4 | 0.25 | 0.50 | 60K |
| 3 | G3_LR250 | 128 | 2.5e-4 | 0.25 | 0.50 | 60K |
| 4 | G4_LR200_H050 | 128 | 2e-4 | 0.125 | 0.25 | 60K |
| 5 | G5_LR150_H050 | 128 | 1.5e-4 | 0.125 | 0.25 | 60K |
| 6 | G6_LR250_H050 | 128 | 2.5e-4 | 0.125 | 0.25 | 60K |

R1의6개를 순회한 뒤 R2의6개를 순회한다. **6 recipe / 12 run / 720K updates**, 시간 상한 없음.
기존 s3 GF2 `RGB_MATURE_U23`42K Aligner/Teacher를 재사용한다.
새 Stage1 학습·s1/s2 weight 이동은 없다. Aligner scale1, beta0, 회전 loss/TTA/SWA/EMA 없음.
GF24밴드·Student6입력/4출력·W128/D123/UBIL, TeacherW112, 양쪽Lanczos4,
batch48/microbatch12 및 warmup100/cosine0을 유지한다.
세 목표는 **HQNR≥.9645, ERGAS≤.522, PSNR≥45.08**, 동일 checkpoint 기준이다.

## s3 인계

사용자가 push한 뒤 s3에서 전체 커밋을 pull한다.
[실행 README](../panda_gf2_w128/README.md)와 [12-run CSV](../config/panda_gf2_w128/cases_s3.csv)를 따른다.

```bash
docker build -f Dockerfile.env -t pancrafter-env:gf2-w128 .
python3 -m panda_gf2_w128.enqueue --server s3 --image pancrafter-env:gf2-w128 --wait
```

기존 GPU 작업을 종료하지 않고 유휴 시점에 시작한다. 이전 ablation 큐가 계속 GPU를 쓰면 대기한다.
원본 donor·목표 달성47K weight·기존 데이터는 read-only이며 새 출력만 별도 저장한다.
기존 weight 정리 시 이 두 source와 selection/calibration 증거를 보존해야 한다.
`GF2-ablations`에 새 Experiment로 게시하고 기존 결과는 유지한다.
H/E 통과 step 전부와 기존 기준 대표행을 게시하며 PSNR 포함3목표 통과를 별도로 표시한다.
지금 s1에서 s3 실험을 시작한 것은 아니며, 실제 파일/SHA/환경/Sheet/GPU 검증은 launcher가 s3에서 수행한다.

## 구현 검증

Git index의 파일만 별도 임시 디렉터리에 내보내 검증했다.
555개 실행 의존 파일의 SHA 및 기준 환경·평가기 검증이 통과했다.
GF2 테스트 5개와 WV3 게시 정밀도 테스트 2개가 모두 통과했다.
GF2 검증에는 6개 recipe 각각의 CPU optimizer 업데이트, frozen Aligner/Teacher 유지,
6입력/4출력, checkpoint 저장 후 동일한 다음 업데이트 재현이 포함된다.
이는 합성 입력을 사용한 구현 검증이며, 실제 GF2 성능 결과는 s3 실험에서 확인한다.
