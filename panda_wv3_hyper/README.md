# WV3 coefficient ablation — s1 / s3

2026-11-08 구현. 기존 WV3 RGB W112 경로를 유지한 계수 민감도 실험이다.
Teacher와 Stage1 warp는 bicubic-border, Stage2 Student warp는 Lanczos4-reflection이다.
scale1, β0, batch48/micro12, AdamW WD.01, warmup100/cosine0, 기존 paired augmentation을 유지한다.
회전 loss/TTA/SWA/밴드 가중 loss/width 변경은 없다.

| 서버 | 계수 | 설정 | 학습/주 비교 | 반복 |
|---|---|---|---|---|
| s1 | α, γ | α={0,.125,.25,.5,1} at γ=.25; γ={0,.125,.25,.5,1} at α=.25; (0,0) | Stage2 30K / fixed29K | 10설정×3 |
| s3 | λ_spa, λ_ε | λ_spa={0,.5,1,2,4,8} at λ_ε=.003; λ_ε={0,.0003,.001,.003,.01,.03} at λ_spa=2 | Stage1 50K / fixed50K | 11설정×3 |

공통 기준점은 중복하지 않는다. **R1의 모든 case → R2의 모든 case → R3의 모든 case** 순서다.
추가 R4 shift 범위·rank 그룹 수·α/γ 교차 실험은 이번 실행 큐에 포함하지 않는다.
계수 전체 목록과 순서는 `config/panda_wv3_hyper/cases_s1.csv`, `cases_s3.csv`에 있다.

s1은 로컬 mainRGB R1/R2/R3의 validation-selected 43K/49K/38K A/T와 정확한 calibration을
SHA 검증 후 읽는다. Stage1 재학습이나 서버 간 weight 이동은 없다.
s3은 각 Stage1 계수·반복을 독립 fresh 초기화하며 다른 서버의 checkpoint가 필요 없다.
L_spa는 GT RGB [4,2,1] 평균의 Scharr이며, Aligner MS 입력과 복원 출력은 전체8밴드다.
α=0,γ>0에도 γd_e r_q가 남으며, (0,0)만 adaptive weight 전체를 제거한다.
모든 계수에서 Teacher/Student support 교집합과 유효 pixel 수 정규화를 유지한다.

## 기존 결과 재사용

기존 mainRGB M0/M2/M1과 s3 RGB A5/A4/A3의 동일 설정·seed·데이터·평가기·고정 시점 평가를 검증한다.
검증된 경우 `REUSED_COMPLETE`로 수치와 출처를 기록하고, 누락/불일치면 동일 case/seed를 새로 학습한다.
s1은 최대9회 재사용하여21회 신규, s3은 최대9회 재사용하여24회 신규다.
Student weight가 과거 정리로 삭제됐으면 `METRICS_ONLY_WEIGHT_PRUNED`로 명시한다.
기존 수치를 재사용하는 것이며, 삭제된 모델의 재추론이나 새로운 학습 결과로 표시하지 않는다.
기존 donor·calibration·결과 파일은 read-only로 사용한다.

## 실행

레포 루트에서 서버에 맞는 한 명령을 실행한다. 기존 환경 이미지에 코드를 마운트하므로 이미지 재빌드는 필요 없다.

```bash
# s1: Stage2 α/γ
python3 -m panda_wv3_hyper.enqueue --server s1

# s3: Stage1 λ_spa/λ_ε; s1의 commit을 push한 뒤 s3에서 pull
python3 -m panda_wv3_hyper.enqueue --server s3 --wait
```

다른 GPU 작업이 있으면 `--wait`는 종료될 때까지 기다린다. 기존 작업이나 큐를 자동 중단하지 않는다.
소스·환경·데이터·donor·Sheet를 먼저 검사하고, 실제 CUDA에서 disposable update와 완전한
checkpoint 복구를 검증한 후 detached 학습 컨테이너를 시작한다. production step에는 포함하지 않는다.
필요시 `--image IMAGE`와 `--dlpan /실제/DLPan-Toolbox`를 명시할 수 있다.

```bash
python3 -m panda_wv3_hyper.enqueue --server s3 --check-only
python3 -m panda_wv3_hyper.enqueue --server s1 --status
docker logs --tail 40 panda-wv3-hyper-s1-v1
# 동일 소스/환경/큐의 중단된 컨테이너만 재개
python3 -m panda_wv3_hyper.enqueue --server s1 --resume
```

출력은 `work_dir/panda_wv3_hyper/PANDA_WV3_HYPER_S13_20261108_v1/{server}/WV3/`다.
새로운 모든1K 평가 model과 최근2개의 complete resume 상태를 저장한다.
고정29K/30K/50K 및 선택 donor는 목표 미달이어도 보존한다. 자동 weight 정리는 하지 않는다.
수치/support 실패는 해당 반복의 실패로 남기고 독립 다음 case를 계속한다.
근거/데이터/코드 오류는 조용히 우회하지 않고 ERROR로 남긴다.

## 게시 및 해석

`WV3-ablations`에 `WV3-HYP-H2-A050 R1 (RGB)`와 같이 case를 드러내고 소수점4자리로 게시한다.
원본 정밀도의 metrics·source hash는 JSON과 Sheet note에 남긴다.
Stage2의 목표 동시 충족 checkpoint와 run 대표1개, Stage1의 validation 대표를 유지하며,
**fixed29K/fixed50K 결과는 목표 미달이어도 게시**한다. 같은 checkpoint의 용도는 합친다.
다른 캠페인 행과 확장 열을 보존하며 write 후 readback을 확인한다.
historical graph cost를 재사용하면 측정 범위를 명시하고 학습시간은 해당 run 기록에서 계산한다.

계수 곡선은 fixed29K/fixed50K의3회 평균±표본표준편차와 같은 반복 기준점 대비 차이로 판단한다.
run별 test 대표와 고정 시점 비교를 섞지 않는다. Stage1 결과만으로 최종 Student 개선을 주장하지 않는다.

## 검증

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s panda_wv3_hyper/tests -v
python -m panda_wv3_hyper.release --environment
python -m panda_wv3_hyper catalog --server s1
python -m panda_wv3_hyper catalog --server s3
```
