# PAN Aligner — 입력 영역 크기·정규화·GAP/head 독립 검증 실행 명세

**작성일:** 2026-09-18  
**진단 ID:** `PAN_ASV_READONLY_v1_20260918`  
**기본 수행:** s1의 기존 자산, 별도 평가 프로세스·작업공간  
**작업 종류:** 이미 저장된 checkpoint의 사후 분석. 신규 학습 실험이 아니다.  
**산출 상태:** case와 입출력·검사·실행 순서를 정의한 MD. 이 파일 작성으로 저장소 코드, 학습 큐, GPU 작업, Sheet가 변경된 것은 아니다.

> **현재 진행 중인 학습과 방법론적으로 독립적으로 실행한다.**
> 기존 완료 checkpoint를 읽어서 평가하며, 현재 학습의 종료·β 비교·다른 서버 결과·공통 recipe lock을 기다리지 않는다. 다만 같은 GPU에서 학습과 동시에 실행해도 자원 충돌이 없다는 뜻은 아니다. 별도 가용 GPU, CPU 또는 승인된 평가 자원을 사용한다. 자원이 없으면 **진단 작업만** 대기하고 본 학습은 계속한다.

## 0. 결정 요약과 기존 계획에서 달라지는 점

이번 작업은 아래 세 질문을 분리해 확인한다.

1. **작은 patch용 global shift 추정이 큰 영역에서도 유효한가?** 같은 pixel scale에서 crop 크기에 따른 shift·probe 반응과 최종 복원을 비교한다.
2. **입력 z-score와 GroupNorm의 통계 범위가 차이를 만드는가?** 같은 crop에서 통계 출처만 바꾸는 사후 개입을 수행한다.
3. **공간 feature를 평균한 뒤 head를 적용하는 것과 shift를 먼저 예측해 평균하는 것이 얼마나 다른가?** 같은 feature 집합을 사용해 비선형 head의 집계 순서를 분리한다.

기존 `A_ON ↔ NOA`는 정합 경로의 제거 효과를 측정한 실험이다. NOA 악화는 위 세 질문의 해결을 뜻하지 않는다. 기존 통합본도 NOA의 결과와 입력 크기 분석을 별개로 정의한다. [S1 §5.4·§7]

**이번 독립 진단에 적용할 운영 규칙**

| 항목 | 이번 규칙 |
|---|---|
| 새 50K 학습 / fine-tuning | 없음 |
| optimizer / backward / 가중치 업데이트 | 없음 |
| 공통 `recipe_lock` / `--decide` / `--write` | 필요 없음. 진단의 의존성으로 사용하지 않음 |
| 학습 큐·mandatory·waiter·학습 hold 변경 | 없음 |
| s2–s5의 파라미터 범위 × seed 학습 | 이 진단을 기다리지 않음 |
| s1 학습 | 별도 자원이 있으면 계속. 같은 GPU 자원 배분은 로컬 운영 정책에 따름 |
| 진단 중 모델 선택 | 기존 checkpoint를 사전 고정. 모드별 best 재선택 없음 |
| 진단 결과에 따른 본 방법 자동 변경 | 없음 |
| 기존 NOA Sheet 열 | 읽기·캐시 참고만. 다른 모드 결과로 덮어쓰지 않음 |

이 명세는 **이번 추가 크기·정규화·head 진단을 학습 복귀의 장벽으로 삼지 않는다.** 과거 통합본의 `s1 분석 완료 후 본 학습 복귀`라는 운영 의존성을 이번 독립 진단에 다시 적용하지 않는다. 기존 운영 hold가 실제로 남아 있다면 담당자가 별도로 상태를 확인한다. 진단 도구가 그 파일을 자동 생성·삭제하거나 학습을 기동해서는 안 된다.

---

## 1. 근거, 구현 상태, 새로 구체화하는 부분

이 명세의 근거는 기존 두 진단 계획 [S1·S2], 사용자와 직전 대화에서 정리한 세 가지 가설, 그리고 아래 저장소 코드를 읽은 결과다. 실험 결과 자체를 새로 측정한 것은 아니다.

| 항목 | 조회한 코드에서 확인한 것 | 이 명세의 처리 |
|---|---|---|
| 크기 패널·AXIS16 | `s1_aligner_analysis.py`에 구현 | 기존 결과 검사 후 재사용/보완 |
| A_ON / ZERO / NOA / CROP64_MED | 네 모드 평가 호출 구현 | 같은 checkpoint로 재사용/보완 |
| GAP feature 반환 | `PANGlobalAligner(..., return_features=True)` 구현 | C03에서 이용 |
| z-score/GN 통계 기록·재사용 | 현재 크기 분석 스크립트에 없음 | C04 신규 **평가 분기** 구현 필요 |
| feature-mean / shift-mean 비교 | 현재 크기 분석 스크립트에 없음 | C03 신규 **평가 분기** 구현 필요 |
| scene 내부 shift 분산 | 기존 요약은 여러 scene의 crop을 섞어 std 계산 | C05에서 scene별로 다시 집계 |
| probe별 원 응답 | 기존 코드는 누적 `q_size`만 저장 | C01 확장 산출물에 개별 응답 저장 |
| T0 포함 | `--assets` 실제 목록은 Student 3개. 주석의 `--teacher` 옵션은 parser에 없음 | T0 전용 로더를 명시적으로 추가. 실행됐다고 간주하지 않음 |
| 출력 격리 | 기존 네 모드 평가는 `_eval_phase/records`·진단 ledger도 갱신 | 독립 작업공간 또는 분리된 writer 필수 |

**‘case 정의’와 ‘구현 완료’의 구분:** 아래 C00–C05는 이번 문서에서 확정하는 평가 case 명세다. C01·C02에는 재사용할 실행 코드가 있지만, C03·C04와 독립 출력 어댑터까지 저장소에 구현·등록됐다는 뜻은 아니다. 구현자는 §12의 완료표와 검사를 통해 상태를 남긴다. 이 문서를 학습용 `main.py --config`에 넣어 실행하지 않는다.

---

## 2. 자산과 checkpoint 고정

### 2.1 대상

| asset_id | 실제 자산 | 역할 |
|---|---|---|
| `S_G23_1234` | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1` | 기존 좋은 Student 자산 |
| `S_G23_3407` | `PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S3407_FRESH50_v1` | 동일 profile, 다른 seed |
| `S_G22_1234` | `PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1` | aligner LR 비율이 다른 대응 자산 |
| `T0` | 기존 `assets/pakd50/T0_run/best_hqnr` 및 그 원 config | Teacher 자체 반응 확인; Student 집계와 분리 |

Student 3개를 주 분석 대상으로 둔다. T0도 기존 계획의 비교 대상이므로 로더를 연결해 같은 분석을 수행하되, T0 연결이 미완료라는 이유로 Student 진단을 멈추지는 않는다. 전체 완료표에는 T0 미완료를 숨기지 않는다. 다른 서버의 자산 수집이나 신규 seed 결과를 선행 조건으로 추가하지 않는다. [S1 §7.1; C1]

### 2.2 사용할 checkpoint

각 asset의 기존 보고 checkpoint 하나를 선정해 **tag·step·전체 SHA256**을 실행 전에 고정한다. `S_G23_1234`의 과거 기록 step31310은 참고값이며 실제 파일과 metadata로 재확인한다. 다른 asset의 step을 그 값으로 대체하지 않는다.

현재 코드의 `sheet_checkpoint()`는 live Sheet를 직접 해석하는 함수가 아니라 로컬 `best_hqnr → best_val → best_reduced` 순으로 파일을 찾는다. 함수 이름만으로 원본 Sheet와 연결이 검증됐다고 간주하지 않는다. 이번 분석에서는 source metadata와 정확한 checkpoint 파일을 먼저 연결한다. [C3]

- 가변 `best_*`·`last` alias를 case마다 다시 읽지 않는다. 선정 후 독립 디렉터리에 실제 bytes를 복사하고 hash를 검증한다.
- 현재 저장 중인 checkpoint를 복사하지 않는다. 완료된 이전 자산 또는 원자적으로 저장 완료된 고정 checkpoint를 사용한다.
- RR/FR, 모든 추론 모드, 통계 개입에서 **같은 asset의 동일 checkpoint**를 사용한다.
- 모드별 best 재선택, 전체 50개 checkpoint 탐색, 새 학습 재개를 하지 않는다.
- Teacher는 Teacher의 자체 backbone과 aligner를 함께 읽는다. Student U-Net에 T0 aligner를 교환 삽입하는 실험은 이번 범위가 아니다.

### 2.3 필요한 파일

크기·통계·head만 볼 때는 aligner weight와 구조/config, PAN·LRMS가 필요하다. 최종 RR/FR까지 평가하려면 **동일 checkpoint의 Student U-Net도 필요**하다. RR GT는 metric 계산에서만 읽는다.

필수 identity: checkpoint 전체 hash, 원 config hash, 경로 매핑 후 config hash, model/evaluator/진단 코드 revision, PAN·LRMS·필요한 부가 feeder 파일과 RR GT/FR mat20 데이터 hash, source selector·step, device/precision. 학습 optimizer state나 새로운 Teacher 추론·q cache 재생성은 필요 없다.

---

## 3. 독립 실행: 학습을 건드리지 않되 자원은 분리

### 3.1 허용하는 실행 방식

| 자원 상태 | 실행 방식 |
|---|---|
| 학습 GPU와 다른 가용 GPU가 있음 | 별도 프로세스에 해당 장치를 명시해 실행 |
| 가용 GPU가 없지만 CPU·메모리 여유가 있음 | `--device cpu`; thread·메모리 사용 제한, 예상 시간은 실측 |
| 같은 GPU를 써야 함 | 담당자가 배정한 빈 시간/평가 슬롯에서 실행. 자동 동시 투입 금지 |
| 어느 자원도 안전하게 쓸 수 없음 | **진단 작업만** `WAIT_RESOURCE`. 다른 학습·서버를 정지하지 않음 |

`torch.no_grad()`는 추가 학습을 하지 않는다는 뜻이지 GPU 메모리·연산을 사용하지 않는다는 뜻은 아니다. 같은 GPU 동시 실행은 메모리 부족과 속도 저하를 일으킬 수 있으므로 ‘실험적으로 독립’과 ‘자원적으로 무관’을 구분한다. 별도 GPU여도 CPU RAM·스토리지 대역폭은 공유할 수 있다.

### 3.2 별도 작업공간

권장 구조는 **독립 코드 사본 + 독립 `work_dir` + 읽기 전용 데이터/자산**이다.

```text
<DIAG_REPO>/                 # 학습 worktree와 다른 경로
  tools/, kdv/, pa/, ...     # 확인한 revision의 코드와 진단 전용 확장
  work_dir/                 # 학습 work_dir의 symlink로 만들지 않음
    <source_run>/           # 복사한 immutable checkpoint/config, 자체 결과
  analysis_runs/
    PAN_ASV_READONLY_v1_20260918/
```

원 데이터 파일은 읽기 전용 경로로 연결할 수 있다. 원 config의 절대 경로를 변환할 때 원본 config와 매핑표를 별도로 보존한다. 원 checkpoint/원 config를 직접 수정하지 않는다.

특히 기존 `--modes`는 `evaluate_run()`을 통해 `work_dir/_eval_phase/records/<run>.json`과 진단 ledger도 기록한다. 따라서 **학습 worktree에서 그대로 실행하면서 ‘파일도 전혀 안 바꾼다’고 표현하지 않는다.** 독립 사본에서 실행하거나, metric 추론과 writer를 분리한 진단 어댑터를 사용한다. [C1; C3]

### 3.3 금지하는 동작

- `noa_eval_switch.sh --hold/--release`, `eval_phase.py hold/release/resume`, 학습 campaign switch 호출.
- `recipe_lock.json`, active/effective/mandatory queue, 학습 reservation·watchdog 설정 변경.
- 실행 중 학습 worktree에 코드 pull/patch, 동일 Python 프로세스에서 모델·GN hook 공유.
- 원 결과 파일·NOA Sheet 열 덮어쓰기, 평가 비용을 새 Train(h)로 합산.
- 전 서버 결과 취합 전까지 학습 대기, 새로운 파라미터/seed 코호트의 완료를 진단 조건으로 사용.

진단 산출물의 충돌을 막는 **로컬 단일 writer 보호**는 사용할 수 있다. 이는 전 서버 학습을 막는 recipe lock과 전혀 다른 용도다.

---

## 4. 실행 case 목록 — C00–C05

모든 ID는 **진단 job ID**이며 50K 학습 run ID가 아니다. case 세부 key는 `(protocol, asset_id, ckpt_sha, case_id, split, mode, scene_panel)`이다.

| case_id | 작업 | 데이터 | 사용 모듈 | 구현 상태/진입점 |
|---|---|---|---|---|
| **ASV-C00** | 자산·캐시·무변경·원 forward 검사 | 고정 RR/FR 1장씩 | A/U | 기존 sanity 재사용 + 독립 manifest/출력 보완 |
| **ASV-C01-RR** | 64/128/full shift와 AXIS16 반응 | RR20, scenes0–19 | A만 | 기존 `--split rr`; probe별 응답 저장 보완 |
| **ASV-C01-FR** | 64/128/256/full shift와 AXIS16 반응 | FR mat20, scenes0–19 | A만 | 기존 `--split fr`; probe별 응답 저장 보완 |
| **ASV-C02** | ON/ZERO/NOA/CROP64_MED 전체 복원 | RR20 + FR20 | A/U | 기존 `--modes`; 결과 격리 |
| **ASV-C03** | feature-mean / shift-mean / full head 분해 + 복원 | RR20 + FR20 | A/U | 신규 평가 분기; 기존 `return_features=True` 활용 |
| **ASV-C04** | z-score×GN 통계 출처 2×2 + 복원·경계 점검 | RR20 + FR20 | A/U | 신규 평가 분기; 가중치·정규화 종류는 유지 |
| **ASV-C05** | scene별 집계·기여 구분·완료표 | C00–C04 산출물 | CPU | 신규 집계. 재학습·재선택 없음 |

job ID 예: `ASV_S_G23_1234_C01_RR_v1`, `ASV_S_G23_1234_C04_v1`. 전체 SHA는 manifest에 필수로 넣으며 ID를 같은 이름으로 다시 쓸 때 다른 checkpoint 결과를 덮어쓰지 않는다.

**순서:** 기존 결과 읽기 → C00 → C01/C02 → C03/C04 → C05. C03/C04는 앞선 결과가 ‘좋아야만’ 실행하는 성능 gate가 아니다. 동작·수치 검사를 통과하면 사전 정의대로 수행한다. 단일 asset의 누락은 다른 asset과 다른 서버 학습을 막지 않는다.

---

## 5. 공통 수학·데이터 규약

LRMS를 X, 원 PAN을 P, bicubic base를 M=U4(X), 학습된 aligner/U-Net을 Aφ/Fθ로 둔다. A 호출은 원 config의 margin4 view와 원 정규화를 포함한다. 출력 c=(dy,dx)는 **현재 PAN pixel 단위**이며 입력 크기 때문에 4배/1/4배 환산하지 않는다.

```math
Z(c)=M+F_\theta([\mathcal W(P,c),M]),\qquad
\mathcal W(P,c)(y,x)=P(y+c_y,x+c_x).
```

실제 backbone API는 원 코드대로 `(pan, lpan, ms, sw)`를 사용한다. MS base를 LRMS 인자에 넣지 않고, residual base는 최종 출력에 한 번만 더한다. 모든 복원은 RR256/FR512 전체 프레임이며 U-Net tiling을 하지 않는다. [S1 §4; C4]

| PAN crop 한 변 | margin4 후 | 두 stride-2 이후 | 64 crop 수(전체 장면당) |
|---:|---:|---:|---:|
| 64 | 56 | 14 | — |
| 128 | 120 | 30 | — |
| 256(RR) | 248 | 62 | 16 |
| 512(FR) | 504 | 126 | 64 |

위 내부 크기는 소스 구조에서 계산한 값이다. 실행 시 실제 tensor shape도 기록한다. 다른 shape 데이터가 들어오면 조용히 resize/잘라내지 않고 불일치로 보고한다.

crop origin은 64 또는 해당 crop size 간격의 non-overlap grid다. M은 전체 X에서 한 번 만든 뒤 P와 같은 좌표로 crop한다. margin4는 정확히 한 번 적용한다. crop 개수는 독립 scene 수가 아니다.

**RR/FR:** 원 공식 evaluator·crop·clamp·intensity 변환을 유지한다. RR의 ERGAS/SAM/PSNR/SSIM/SCC/Q8, FR의 Dλ/Ds/raw-original HQNR를 기록한다. FR reference는 항상 원 PAN/원 MS이며 결과에 맞춰 이동시키지 않는다. HQNR는 기존 장면별 곱 평균 규약을 유지한다. [S1 §5; S2 §7; C3]

---

## 6. ASV-C00 — 자산 고정·기존 결과 재사용·구현 검사

1. §2 자산을 읽고 checkpoint bytes·config·데이터 identity를 고정한다. source의 종료·저장완료 여부를 확인한다.
2. 기존 `size_probe_rr/fr.json`, `modes.json`, ON/NOA metric을 **읽기 전용**으로 조사한다. 코드 파일 존재나 exit code0만으로 완료 판정하지 않는다.
3. 같은 checkpoint/데이터/모드/precision/장면 목록의 유효한 결과는 복사·링크로 provenance를 남겨 재사용한다. 예비 3/5장면을 20장면 완료로 취급하지 않는다.
4. 원 A_ON과 분리한 진단 A_ON의 출력을 비교한다. `.eval()`, `no_grad()`, parameter/buffer hash 전후 불변, `.grad is None`, sampler·hook 복원을 검사한다.
5. NOA의 aligner/warp 호출 수0/0, ZERO의0/1, ON의1/1을 확인한다. ZERO와 NOA가 bitwise 같다고 사전 가정하지 않고 출력 max/mean absolute difference를 기록한다.
6. C03 feature-return 경로에서 `g(z)`가 원 aligner 출력과 재현되는지 검사한다. C04 LOCAL/LOCAL custom forward도 원 crop forward를 재현해야 한다.

수치 허용 오차는 실행 전에 manifest에 기록한다. 제안 기본값은 **FP32 중간 feature/출력 동일성 검사에 atol=1e−5, rtol=1e−4**이며, 실제 backend 차이가 있으면 원인을 기록한다. 이는 수치 sanity용 기준이고 성능 개선의 유의성·채택 기준이 아니다. state/원 파일 hash는 정확 일치가 필요하다.

원본 legacy reference가 다른 step이거나 proxy RR이면 그 비교의 상태를 `reference_not_comparable`로 남긴다. 같은 새 checkpoint의 진단 내부 대응 비교는 별도로 진행할 수 있지만, 과거 Sheet 재현 완료라고 쓰지 않는다.

---

## 7. ASV-C01 — 크기별 shift·AXIS16 반응

### 7.1 패널

RR 장면당 64×16개 + 128×4개 + full256×1개 =21개 view. FR 장면당 64×64개 +128×16개 +256×4개 +full512×1개 =85개 view다. 같은 scene 내에서 크기별 결과를 비교한다.

native shift c0와 AXIS16 반응을 저장한다. ε는 반경0.25/0.5/1/2 각각의 ±y,±x로 총16개다. 원 PAN 전체에 W(P,ε)를 적용한 뒤 같은 crop을 읽고 M은 이동시키지 않는다.

```math
r_j^{(s,k)}=c_j^{(s,k)}+\epsilon_j-c_0^{(s,k)},\qquad
q_{\rm size}^{(s,k)}=\frac1{2N_{\rm pr}}\sum_j\|r_j^{(s,k)}\|_1.
```

학습 q cache·q_ref를 수정하지 않는다. native absolute shift의 정답 오차를 측정한 값도 아니다.

### 7.2 저장할 원값과 해석

`native_views.csv`: asset/scene/split/size/crop origin/dy/dx/q_size.  
`probe_responses.csv`: 위 key + radius/epsilon_y/x/c_native_y/x/c_probe_y/x/residual_y/x/support flag.

기존 코드의 누적 q_size만으로는 radius별 gain·부호 반응을 복원할 수 없다. 개별 응답이 없으면 해당 항목만 추가 계산하고, 구체 응답을 추정해서 채우지 않는다.

장면별로 crop shift 분산, full과 64-crop median의 차이 벡터·L2거리, q_size 중앙값/상위 분위, radius별 반응을 기록한다. **부호 있는 장면 평균만 보고 full/crop이 같다고 결론내리지 않는다.** 양·음 차이가 상쇄될 수 있다.

해석을 돕는 계산 기준으로, 모든 probe에 c_j=c_0인 **무반응 모델**의 본 AXIS16 q는0.46875이다. 이는 `(4×(0.25+0.5+1+2))/(2×16)`에서 계산한 참고값이며 실제 관측이 아니다. 낮은 native crop 분산이나 작은 full−crop 차이가 무반응/상수 예측 때문인지도 개별 probe 반응과 함께 확인한다.

probe의 원본 경계·보간 support 영향은 별도 flag와 보조 요약으로 남긴다. 공식 RR/FR metric 영역을 case마다 바꾸거나 예측 shift를 clipping하지 않는다.

**이 case의 한계:** crop 간 예측 차이는 실제 공간 변위 차이, 내용 의존적 추정 오차, 정규화·경계 효과가 혼합된 관측이다. 실제 기하학적 변위의 정답으로 해석하지 않는다.

---

## 8. ASV-C02 — 같은 전체 U-Net에서 네 모드 복원

| mode | shift/입력 | warp | 목적 |
|---|---|---|---|
| `A_ON` | 전체 영상에서 c_full | 1회 | 현재 방식 |
| `A_ZERO_WARP` | c=0, aligner 호출 없음 | 1회 | 영점 sampler 수치 확인 |
| `A_BYPASS_RAW` | 원 PAN, aligner 호출 없음 | 없음 | 기존 NOA 재사용 |
| `A_CROP64_MED` | 64 crop별 c의 성분별 중앙값 | 전체 PAN 1회 | 작은 영역 추정의 대표값 효과 |

짝수 crop 수의 median은 가운데 두 값 평균이다. `torch.median`의 lower median과 혼용하지 않는다. U-Net은 모든 모드에서 전체 입력을 받는다. patch마다 다른 warp, output blending, GT로 shift를 고르는 oracle은 넣지 않는다. [C4]

최종값은 같은 checkpoint·모드에서 계산하고 차이는 `해당 모드 − A_ON`으로 통일한다. ΔERGAS<0은 개선, ΔHQNR>0은 개선이다. 공동목표는 `HQNR_raw≥0.9585 AND ERGAS<2.040`으로만 판정한다. 이 진단은 모드별 후보 step을 다시 고르는 실험이 아니다.

**해석:** CROP64_MED가 좋아지면 ‘보정량의 관측 영역·집계 방식이 영향을 준다’는 근거다. 이 결과만으로 GN이나 GAP 하나를 원인으로 특정하지 않는다.

---

## 9. ASV-C03 — GAP feature와 비선형 head의 집계 순서 분리

**이번에 추가할 평가 분기다.** 기존 aligner는 GAP 후64차원 z와 shift를 반환할 수 있고 head는 `fc2(SiLU(fc1(z)))`다. 기존 weight를 그대로 사용한다. [C2]

### 9.1 동일 feature 집합의 세 비교

같은64 crop에서 얻은 GAP feature를 z_k, shift를 c_k=g(z_k), 전체 영상의 feature를 z_F로 둔다. 실제 margin4 view를 먼저 한 번 적용한 뒤 `return_features=True`를 호출한다. crop z-score/GN은 이 case에서는 원래 방식 그대로다.

```math
\bar z=\frac1K\sum_kz_k,\qquad
c_{\rm FM}=g(\bar z),\qquad
c_{\rm SM}=\frac1K\sum_kg(z_k),\qquad
c_F=g(z_F).
```

| mode | 정의 | 복원 |
|---|---|---|
| `A_CROP64_FEAT_MEAN` | c_FM = g(mean z_k) | 전체 PAN 1warp + 전체 U-Net |
| `A_CROP64_SHIFT_MEAN` | c_SM = mean g(z_k) | 동일 |
| `A_CROP64_MED` | median c_k | C02 재사용 |
| `A_ON` | c_F | C02 재사용 |

c_FM과 c_SM에는 **같은 z_k tensor bank**를 재사용한다. 두 분기에서 crop/정규화/weight를 다르게 실행하지 않는다. 그러면 이 두 값의 차이는 head와 평균의 순서에서 발생한다.

참고 계산:

```python
# 진단 분기의 핵심 계산. 실제 실행기는 manifest/crop/metric writer를 별도로 연결한다.
z_bar = crop_features.mean(dim=0, keepdim=True)
c_fm = aligner.fc2(torch.nn.functional.silu(aligner.fc1(z_bar)))
c_sm = crop_shifts.mean(dim=0, keepdim=True)
```

### 9.2 정확한 벡터 분해

```math
c_F-c_{\rm SM}
=\underbrace{g(z_F)-g(\bar z)}_{\text{feature 생성·문맥·통계 차이}}
+\underbrace{g(\bar z)-\operatorname{mean}_kg(z_k)}_{\text{동일 feature에서 head/평균 순서 차이}}.
```

이는 shift 벡터의 항등식이다. 두 항의 norm이나 ERGAS/HQNR 기여율이 단순히 더해진다는 뜻이 아니다. 각 항의 방향·크기와 실제 복원 지표를 함께 기록한다. median−mean 차이는 별도다.

### 9.3 추가 forward가 거의 필요 없는 full-feature 확인

full forward의 마지막 pre-GAP feature map을 hook으로 읽고, 겹치지 않는 영역으로 **모든 위치를 빠짐없이 분할**한다. 영역 k의 feature 평균 z_k^F, 픽셀 수 n_k에 대해 a_k=n_k/Σn_k를 사용한다.

```math
z_F=\sum_k a_kz_k^F.
```

`g(Σa_k z_k^F)`가 원 c_F를 재현하는지 sanity로 확인한다. `Σa_k g(z_k^F)`와의 차이는 full forward의 동일 feature를 사용하므로 crop 재실행의 padding/정규화 차이를 새로 끼우지 않는다. feature map62×62를 crop14×14 묶음으로 단순 대체해 위치를 누락시키면 안 된다.

이 검사는 보조 feature/shift 통계만 기록한다. 새 복원 모드를 추가로 무한히 늘리지 않는다.

---

## 10. ASV-C04 — 입력 z-score × GN 통계 범위의 2×2 평가

**이번에 추가할 평가 분기다.** 여기의 2×2는 하이퍼파라미터 학습 grid가 아니라 동일 checkpoint에서 정규화 통계의 출처만 바꾸는 네 번의 평가 조건이다.

### 10.1 full-reference 통계 기록

각 asset·split·scene의 **native full A_ON forward**에서 다음을 기록한다.

- margin4 이후 PAN과 M의 sample/channel별 평균·분산. `znorm`은 population variance(`unbiased=False`)와 기존 epsilon을 사용한다. [C2]
- 각 GN 바로 **직전 tensor**의 group별 평균·분산과 GN의 실제 groups/epsilon/affine weight·bias.
- 마지막 pre-GAP feature map, GAP feature, c_F.

현재 코드의 GN 위치는 `pan_stem.2`, `ms_stem.2`, `joint1.2`, `joint2.2`, `res.a.2`, `res.b.2`다. 실행 시 named_modules로 실제 목록을 확인하고 누락/추가가 있으면 보고한다. layer마다 `[B,G,C/G,H_l,W_l]`로 묶어 그룹의 채널·공간 축 통계를 읽는다. [C2]

이 통계 bank는 **같은 asset·checkpoint·scene·split의 동일 full native forward에서만** 가져온다. 다른 장면·Teacher·seed의 통계를 사용하지 않는다. 학습 running statistics로 저장하거나 모델 buffer를 바꾸지 않는다.

### 10.2 네 조건

각64 crop의 픽셀, margin, conv/head weight, padding은 고정한다.

| norm_mode | 입력 z-score 통계 | GN 통계 | 상태 |
|---|---|---|---|
| `ZL_GL` | 해당 crop | 해당 crop의 현재 layer 입력 | 원 crop forward와 같아야 함 |
| `ZF_GL` | full-reference | 해당 분기에서 계산한 crop 통계 | 입력 정규화 범위 개입 |
| `ZL_GF` | 해당 crop | full-reference의 모든 GN 통계 | 내부 정규화 범위 개입 |
| `ZF_GF` | full-reference | full-reference의 모든 GN 통계 | 두 범위 개입 |

GN full-reference 재사용의 계산 형태:

```math
\operatorname{GN}_{F,\ell}(x)_{c,p}
=\gamma_{\ell,c}\frac{x_{c,p}-\mu^F_{\ell,g(c)}}
 {\sqrt{v^F_{\ell,g(c)}+\epsilon_\ell}}+\beta_{\ell,c}.
```

같은2×2 비교에서 GN reference bank는 원 full A_ON의 한 bank로 고정한다. 특정 분기에 유리하도록 뒤에서 다시 선택하지 않는다. `ZF_GL`의 GN만 그 분기의 실제 crop activation에서 계산하는 원래 방식이다.

**주의:** crop을 미리 정규화한 뒤 원 `aligner.forward()`를 그대로 부르면 내부 `znorm`이 다시 적용될 수 있다. 입력 통계 대체는 원 z-score 위치를 정확히 한 번 바꾸는 어댑터로 구현한다. `.eval()`만 호출하거나 GN을 LN으로 교체하는 것은 이 실험이 아니다.

### 10.3 복원과 통계

각 조건에서64 crop shift들의 **같은 성분별 median**을 한 쌍으로 모아 전체 PAN을1회 warp한 뒤 전체 U-Net에 넣는다.

| 복원 mode | 대응 |
|---|---|
| `A_NORM_ZL_GL_MED64` | 기존 `A_CROP64_MED`와 동일하여 재사용 |
| `A_NORM_ZF_GL_MED64` | 입력 통계만 full |
| `A_NORM_ZL_GF_MED64` | GN 통계만 full |
| `A_NORM_ZF_GF_MED64` | 양쪽 통계 full |

crop별/scene별 c 차이, feature 차이, RR/FR를 기록한다. `c11−c10−c01+c00`는 통계 개입의 상호작용 벡터로 보고할 수 있지만, 이것을 곧바로 성능의 가산적 기여율로 해석하지 않는다.

본2×2는 native 영상만 평가한다. AXIS16×모든 통계 조건까지 늘리는 것은 기본 범위가 아니다. 추후 probe에 적용한다면 reference bank의 출처도 그 입력에 맞춰 별도로 정의해야 하며 native/probe 통계를 조용히 혼합하지 않는다.

### 10.4 경계·문맥을 함께 분리하는 구현 sanity

통계 출처를 바꿔도 crop 경계의 reflection padding 영향은 남는다. 따라서 `ZF_GF` 조건에서 **같은 좌표의 내부 feature**를 full forward와 대조한다.

현재 conv 경로만 계산하면 최대 receptive field는25×25, feature stride는4다. 계산: stem3 → joint1 5 → joint2 9 → res.a 17 → res.b 25. **이는 z-score와 모든 GN 통계를 기록된 상수로 고정한 이 진단 조건에서의 공간 conv 경로 계산**이다. 원 GN까지 포함한 모델 전체가25×25 문맥만 본다는 뜻은 아니다. [C2에서 도출]

제안 sanity 영역은 pre-GAP crop feature map의 각 경계4칸(입력16px)을 제외한 core다. stride4에 정렬된 origin `(y0,x0)`에서 crop feature `(j,i)`는 full feature `(y0/4+j,x0/4+i)`에 대응한다. margin4 적용 횟수와 grid phase를 실제 tensor로 검사한다.

이 내부 feature는 고정 full 통계·동일 연산하에서 수치적으로 재현되는지 검사한다. 큰 불일치가 있으면 norm hook·좌표·padding·dtype 구현을 먼저 확인한다. core 비교는 **feature sanity용**이며 공식 RR/FR crop을 바꾸지 않는다. core만 pooling한 보조 shift를 보고하려면 `core_diagnostic`으로 분리하고 기존 full pooling 결과를 대체하지 않는다.

**해석 한계:** full 통계 개입이 성능을 바꾸면 정규화 범위의 영향은 확인되지만, 일반화 성능이 보장되거나 실제 지상 변위가 규명된 것은 아니다. 추론 통계 정책을 바꾼 진단이며 주 메서드의 자동 변경이 아니다.

---

## 11. ASV-C05 — 집계, 판단, 완료 기준

### 11.1 집계 단위

- crop/probe를 먼저 **같은 scene 안에서** 요약하고, scene별 요약을20장에 동일 가중으로 보고한다.
- 현재 구현 `by_size.dy_std/dx_std`는 여러 scene의 crop을 합친 값이므로 `pooled_std`로만 부른다. scene 내부 std를 따로 계산한다. [C1]
- full−crop 차이는 signed mean뿐 아니라 scene별 L2거리의 평균/중앙값/p90을 보고한다.
- RR/FR 주 집계는 기존 공식 evaluator를 전체 데이터에 적용한 값이다. per-scene 보조값을 만들었다고 PSNR/Q8 등 집계 정의를 임의로 바꾸지 않는다.
- asset/profile/seed/Teacher는 분리한다. 같은 seed의 G22/G23을 독립 seed2개로 세지 않는다.
- 이미 개발·checkpoint 선택에 사용한 RR/FR 세트를 새 미사용 test set이라고 부르지 않는다.

### 11.2 결과표

```text
asset / source_step / source_sha / eval_mode
RR ERGAS / SAM / PSNR / SSIM / SCC / Q8
FR D_lambda / D_s / HQNR_raw
delta_ERGAS_vs_ON / delta_HQNR_vs_ON / joint_pass
q_size_by_size / within_scene_shift_std / full_crop_distance
feature_generation_delta / head_order_delta / norm_intervention_delta
identity_status / metric_status / n_scenes / notes
```

| 관측 | 가능한 결론 | 금지할 확대 해석 |
|---|---|---|
| NOA만 악화 | 정합 경로 제거가 현재 학습된 U에 불리 | 큰 입력 정합 문제가 전혀 없음 |
| CROP64_MED가 ON보다 좋음 | 영역·집계 정책이 성능에 영향 | GN 하나가 원인이라고 확정 |
| 동일 z에서 c_FM과 c_SM 차이 큼 | 비선형 head/평균 순서 영향 | 그 차이가 모두 RR 손실의 원인 |
| 통계 재사용에서 shift·metric 변화 | 해당 정규화 범위 개입의 영향 | 실제 기하 변위의 정답 규명 |
| full/crop shift 차이가 작음 | 현재 패널의 출력이 비슷함 | input-insensitive 예측 여부를 안 봐도 됨 |
| RR만 개선, FR 하한 미달 | RR/FR trade-off | 모드를 섞어 공동목표 통과 |
| 차이가 작거나 asset별 방향 다름 | 현재 패널에서 우선 병목 근거 약함 | 어떤 조건에서도 문제가 없음 |

주 관심은 기존 HQNR 하한을 유지하면서 ERGAS가 개선되는지다. 단, 상대 개선의 존재 자체는 하한 미달 asset에서도 관측값으로 남긴다. 상대 개선과 목표 달성을 혼동하지 않는다.

### 11.3 완료 체크

각 asset마다 C00, C01-RR, C01-FR, C02, C03, C04, C05 상태를 남긴다. `implemented / pending / running / complete / blocked_missing_asset / failed / reused_verified`를 구분한다.

- RR/FR 각각 scenes0–19가 정확히 포함돼야20장 전수 완료다.
- 모드별 RR와FR가 같은 asset/step/hash를 공유해야 한다.
- 기존 결과를 재사용한 case는 원 경로·identity와 부족한 항목을 기록한다.
- 새 C03/C04가 아직 구현되지 않았으면 ‘전체 검증 완료’가 아니다.
- `--assets`에서 누락 자산을 건너뛰어도 exit0일 수 있으므로 기대 목록과 대조한다.
- T0가 미연결이면 Student 분석 완료와 T0 미완료를 별도로 보고한다.
- 진단 미완료/실패가 본 학습의 자동 정지 조건이 되어서는 안 된다.

---

## 12. 구현 인계와 실제 명령의 범위

### 12.1 기존 코드로 가능한 실행

아래는 조회한 **현재 코드가 지원하는 명령**이다. 새 norm/head 분기를 포함하지 않는다. §3의 **독립 작업공간**을 준비하고, 사용할 device를 명시한 뒤 실행한다.

```bash
# 학습 저장소가 아니라 진단 사본의 root에서 실행한다.
# DIAG_DEVICE는 cpu 또는 실제로 배정된 cuda:N이다. 학습 중 GPU를 자동 선택하지 않는다.
: "${DIAG_DEVICE:?DIAG_DEVICE를 cpu 또는 배정된 cuda:N으로 지정하세요}"

python tools/s1_aligner_analysis.py --assets --split rr --scenes 0-19 --device "$DIAG_DEVICE"
python tools/s1_aligner_analysis.py --assets --split fr --scenes 0-19 --device "$DIAG_DEVICE"
python tools/s1_aligner_analysis.py --assets --modes --device "$DIAG_DEVICE"
```

특정 asset만 실행하려면 `--assets` 대신 표의 정확한 source run을 `--run`에 넣는다. 기존 default `--scenes 0-4`를 그대로 두지 않는다. `--modes`는 두 split의 전체 평가 함수로 들어가며 size/probe 모드를 대신하지 않는다. [C1]

기존 코드의 실제 결과 경로:

```text
work_dir/<run>/results/aligner_eval_v2/<sha16>/analysis/
  size_probe_rr.csv
  size_probe_rr.json
  size_probe_fr.csv
  size_probe_fr.json
  modes.json
```

같은 이름의 예비 결과를 덮어쓰지 않도록 실행 전 보존한다. 신규 실행의 scene panel·code identity와 결과를 별도 analysis session에 기록한다.

### 12.2 새로 구현할 독립 분석 어댑터

제안 파일명은 `tools/aligner_scope_audit.py`다. **이 이름의 도구가 현재 저장소에 존재한다는 뜻은 아니다.** 역할은 아래 계약으로 제한한다.

| 기능 | 필요한 구현 |
|---|---|
| C00 | 명시적 immutable ckpt/config manifest, source/data 검증, 캐시 조사 |
| C01 | 기존 crop/probe 연산 재사용, 개별 응답과 scene별 통계 추가 |
| C02 | 기존 forward-mode/공식 metric 연산 재사용, 운영 writer 호출 분리 |
| C03 | GAP feature 추출, 동일 bank의 두 집계, full feature partition sanity |
| C04 | z-score/GN pre-normalization 통계 trace와 재사용, 원 LOCAL 재현 검사 |
| C05 | source/scene/mode별 집계, 상태표, Markdown 결과 보고 |
| 공통 | 별도 결과 root, 원 파일/학습 상태 무변경, T0 직접 로더, 명시적 device |

새 도구의 CLI를 구현할 경우 `--manifest`, `--case`, `--device`, `--output-root`, `--inventory-only`를 지원하도록 설계한다. **현재 기존 스크립트에 이 옵션들이 있다고 가정해 호출하지 않는다.** 구현 뒤 `--help`·로컬 fixture 검사·실제1장 sanity를 확인한 다음 명령을 실행 기록에 남긴다.

학습 generator/CASES/50K queue에 이 진단 ID를 등록할 필요는 없다. 진단 전용 case registry와 dispatcher만 두면 된다.

### 12.3 manifest 계약 예시

다음 YAML은 **새 진단 어댑터용 명세 예시**이며 현재 학습 config가 아니다. null 필드는 구현자가 기존 파일에서 해소한다. 공통 lock이나 새로운 승인 장벽으로 사용하지 않는다.

```yaml
protocol_id: PAN_ASV_READONLY_v1_20260918
job_kind: checkpoint_diagnostic
training_control: none
requires_recipe_lock: false
requires_other_server_results: false
requires_current_training_completion: false
changes_training_queue: false
source_policy: immutable_existing_checkpoint
output_root: null                 # 학습 work_dir와 분리된 절대경로
device: null                      # 배정된 cuda:N 또는 cpu
scenes:
  rr: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  fr: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
cases: [ASV-C00, ASV-C01-RR, ASV-C01-FR, ASV-C02, ASV-C03, ASV-C04, ASV-C05]
assets:
  - asset_id: S_G23_1234
    source_run: PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1
    checkpoint_file: null
    checkpoint_sha256: null
    checkpoint_step: null
    source_selector: null
    source_config: null
    mapped_config: null
  - asset_id: S_G23_3407
    source_run: PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S3407_FRESH50_v1
    checkpoint_file: null
    checkpoint_sha256: null
    checkpoint_step: null
    source_selector: null
    source_config: null
    mapped_config: null
  - asset_id: S_G22_1234
    source_run: PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1
    checkpoint_file: null
    checkpoint_sha256: null
    checkpoint_step: null
    source_selector: null
    source_config: null
    mapped_config: null
teacher_panel:
  asset_id: T0
  source_asset_dir: assets/pakd50/T0_run
  source_selector: best_hqnr
  loader_status: requires_explicit_asset_loader
  blocks_student_cases: false
```

### 12.4 최소 회귀 검사

| 검사 | 합격 조건 |
|---|---|
| 자산 불변 | 원본과 복사본 hash 일치, 진단 전후 state 불변 |
| 원 forward 재현 | A_ON, C03 g(z), C04 ZL_GL이 각 원 경로를 수치 재현 |
| 크기·좌표 | margin4 한 번, native-scale crop, M full 생성, 기대 panel 수 |
| probe 부호 | c_eps+eps−c0, 분모2Npr, 모든16 probe 저장 |
| NOA | A/warp0회, M1회, sampler 복원 |
| 집계 정의 | mean/median 구분, 같은 z bank 사용, full partition 가중합 재현 |
| 정규화 재사용 | GN 모든 위치 trace, scene/asset 혼합 없음, 이중 znorm 없음 |
| 경계 sanity | 고정 full 통계 조건의 matched interior feature 재현 |
| 결과 격리 | 원 results·Sheet·학습 ledger/queue/hold 무변경 |
| 부분 완료 | 미구현/누락/5장 결과가20장·전체 완료로 오인되지 않음 |

---

## 13. 결과 저장·보고·계산량

### 13.1 권장 결과 구조

```text
analysis_runs/PAN_ASV_READONLY_v1_20260918/<session_id>/
  manifest.json
  source_inventory.json
  case_status.csv
  timing.json
  <asset_id>/<checkpoint_sha>/
    C00/sanity.json
    C01/<rr|fr>/native_views.csv
    C01/<rr|fr>/probe_responses.csv
    C01/<rr|fr>/scene_summary.csv
    C02/modes.json
    C03/<rr|fr>/feature_bank.*
    C03/<rr|fr>/head_decomposition.csv
    C04/<rr|fr>/norm_reference.*
    C04/<rr|fr>/crop_norm_results.csv
    C04/<rr|fr>/interior_parity.json
    metrics/<mode>/rr.json
    metrics/<mode>/fr.json
    metrics/<mode>/per_scene_rr.csv
    metrics/<mode>/per_scene_fr.csv
  report.md
```

각 파일에 checkpoint/data/code/모드/precision/probe·crop 정책을 연결한다. full hash는 manifest에 보존하며 표시용16자 hash만으로 검증하지 않는다. 정규화 trace도 정책과 weight가 같아야 재사용한다. 기존 캐시에 RR data hash 등 필요한 identity가 없으면 ‘verified reuse’로 간주하지 않는다.

이번 산출물은 기본적으로 로컬 CSV/JSON/MD다. 기존 NOA Sheet 열에 CROP/NORM/HEAD 결과를 넣지 않는다. 추후 업로드가 필요하면 별도 진단 영역에서 source run/step/mode를 모두 구분하고, 본 학습 행이나 통계에 새 run으로 더하지 않는다.

### 13.2 계산량의 성격

C01은 aligner만 실행한다. asset당20장 기준 native+16probe의 **view 계산 수**는 RR `20×21×17=7,140`, FR `20×85×17=28,900`, 합계36,040이다. 이는 고정된 패널로부터 계산한 workload이며 실제 wall time은 아니다. 동일 크기의 crop batching이나 캐시로 호출 횟수/시간을 줄일 수 있다.

C02의4모드, C03의새2모드, C04의새3모드를 모두 복원 평가하면 중복 제외9모드다. asset당40장×9모드=360개의 full-image U-Net 예측이 최대 기본 범위이며, 유효한 ON/NOA/CROP 캐시 재사용 시 줄어든다. C03의full-feature sanity나 C04의core sanity를 별도 새 복원모드로 늘리지 않는다.

3개의 Student 기준1,080개 full-image 예측, T0 포함 시1,440개다. **이는 새 학습 run 수가 아니다.** 이전 NOA 한 run의 평가시간을 수만 개 crop-probe 분석시간으로 그대로 환산하지 않는다. C00과 RR/FR 각1장 pilot으로 실제 A forward/U inference/metric/I/O 시간을 나누어 측정한다.

메모리가 부족하면 crop batch·데이터 로딩 동시성만 낮춘다. 비교 크기·장면 수·정규화 종류·정밀도는 조용히 바꾸지 않는다. 진단 시간이 길어진다는 이유로 s2–s5 학습을 기다리게 하지 않는다.

---

## 14. 근거 자료와 이번 작성의 확인 범위

### 기존 계획

- **[S1]** `PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md`, 특히 §4–7: NOA 정의, full-U 유지, 입력 크기/probe, 정오와 해석 범위. 운영상의 선행 학습 대기는 이번 문서 §0·§3으로 대체한다.
- **[S2]** `PAN_S1_Aligner_Validation_NOA_RR_FR_Protocol_v1.md`, 특히 §2·§5·§7: 네 모드·동일 checkpoint·crop/probe·metric. 중앙 s1 평가 운영은 이번 추가 진단의 의존성이 아니다.
- **[U]** 직전 사용자 지시: 현재 checkpoint로 위 검증을 진행할 수 있도록 MD 작성. 현재 학습과 독립 진행. 공통 설정 lock을 기다리기보다 좁은 파라미터 범위와 seed를 섞어 본 실험 진행.

### 이번에 다시 읽은 저장소 코드

- **[C1]** [tools/s1_aligner_analysis.py](https://github.com/hojunking/PAN-Crafter-repro/blob/main/tools/s1_aligner_analysis.py), 조회 blob SHA `0da5059a2af99ef062944a2d74955e837ce4ba8a`: parser, Student3 목록, size/probe, modes, 실제 출력·요약.
- **[C2]** [pa/aligner.py](https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/aligner.py), 조회 blob SHA `e9dd406ae7f0a2dc491273e21b38f8e450dfa28d`: z-score, GN, conv, GAP, feature 반환, head.
- **[C3]** [tools/noa_eval.py](https://github.com/hojunking/PAN-Crafter-repro/blob/main/tools/noa_eval.py), 조회 blob SHA `d372aa64f36005f8edd92ba56168c0e39a7899f5`, 이번 읽기1–150·240–342행: checkpoint 선택 helper, identity, 운영 record/ledger writer, sanity.
- **[C4]** 앞선 조회의 `kdv/eval_modes.py` 및 `pa/model.py`에 대한[S1·S2]의 명시적 forward 계약: NOA는 aligner+sampler 우회, crop median은 전체 PAN1warp, U-Net/residual API 유지. 실제 실행 시 해당 파일들의 revision도 확인한다.

C03/C04의 통계 개입·head 분해·경계 sanity·case registry·독립 작업공간은 **이 문서에서 기존 가설을 실험 가능하게 구체화한 새 평가 설계**다. 소스에 이미 구현됐다거나 그 원인에 대한 결과가 나왔다는 주장과 구분한다.

**이번 작성에서 하지 않은 것:** 원격 학습 프로세스 조회·중단, 서버 코드/queue 변경, 실제 GPU/CPU metric 실행, C03/C04 코드 구현, checkpoint bytes 감사, live Sheet 업데이트. 따라서 현재 s1의 전수 진단 완료 상태를 이 문서로 인증하지 않는다.

---

## 실행 담당자에게 전달할 최종 지시

**기존 완료 Student checkpoint3개와 연결 가능한 T0를 별도 진단 사본에서 읽고, C00–C05를 평가 job으로 수행한다. 기존 결과를 검증해 재사용하며, 크기/probe와 네 모드는 현 코드를 활용하고, head 집계·정규화 통계 개입은 명시된 새 평가 분기로 추가한다. 학습을 새로 하지 않고, 학습 큐·공통 lock·다른 서버 결과를 의존성으로 두지 않는다. GPU/CPU 자원을 별도로 확보하고, 같은 GPU를 무계획하게 공유하지 않는다. 결과는 별도 JSON/CSV/MD에 저장하고 본 방법을 자동 교체하지 않는다.**
