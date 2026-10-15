# PAN — 전 서버 Student 평가 · s1 Aligner 분석 · 현재 Method 통합 인계서

**작성일:** 2026-09-18  
**운영 revision:** `PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918`  
**학습 Method:** `QRECON24 / qrecon_continuous_v1` — 기존 수식 유지  
**평가 모드:** `A_ON`, `A_BYPASS_RAW`(표시명 NOA); s1 분석에만 `A_ZERO_WARP`, `A_CROP64_MED` 추가  
**본 실험 선택 목표:** raw-original HQNR ≥ 0.9585 → 공식 RR ERGAS 최소, 목표 < 2.040  
**문서 상태:** 사용자 최신 운영 지시를 반영한 실행 명세. 이 문서 작성으로 원격 코드·서버 큐·GPU 실행·Sheet가 변경된 것은 아니다.

> **가장 먼저 적용할 운영 순서**
>
> - **s1:** 자기 서버의 기존 Student RR/FR 평가·업로드 → 대표 Aligner 분석 실험 → 자기 본 실험 복귀.
> - **s2–s5:** 각자 자기 서버의 기존 Student RR/FR 평가·업로드 → 각자 자기 본 실험 복귀.
> - **s2–s5는 s1의 평가·분석 완료나 다른 서버의 평가 완료를 기다리지 않는다.**
> - s1은 다른 서버의 전체 checkpoint를 모아 대신 평가하는 중앙 평가 서버가 아니다.
> - 모든 서버에서 이미 실행 중인 학습은 안전한 case 경계까지 원 정의로 완료한다. 평가 대상을 고정한 뒤 신규 학습보다 평가를 우선한다.

**읽는 순서:** 서버 운영·평가·Sheet는 §1–7, Method 본문과 정오는 §8–9, 본 실험 복귀는 §10, 구현 검사·완료 판정은 §11–12를 참조한다.

## 문서의 근거와 통합 범위

이 문서는 아래 자료와 사용자 최신 지시를 통합했다. 문서 간 운영 차이는 §1에서 명시적으로 변경하며, Method 수식은 handoff의 확정 계산을 따른다.

| 표기 | 자료 | 이 문서에서의 용도 |
|---|---|---|
| [U] | 최신 사용자 지시: s1 `Student 평가 → 분석 → 복귀`, s2–s5 `Student 평가 → 복귀` | 서버별 독립 진행 순서의 최상위 근거 |
| [E1] | `PAN_S1_Aligner_Validation_NOA_RR_FR_Protocol_v1.md` | 정확한 NOA 정의, 동일 checkpoint 비교, 대표 크기 분석, RR/FR·Sheet 명세 |
| [H] | `PAN_New_Session_Handoff_2026-09-17.md` | 확정 Method 수식, Teacher 계보, 표기 보완, 전처리·평가 기준 |
| [R2] | `PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md` | 본 실험 복귀 후 마지막 β 비교, 공통 recipe lock, seed 반복, ERGAS 우선 selector |
| [P] | `PAN_Current_Method_PaperStyle_Detailed_2026-09-17.md` | 사용자가 통합을 요청한 논문형 초안의 이름 |

**[P] 원문 확보 상태:** 이번 작업에서 Project/Library 파일 검색과 작업 디렉터리 확인으로 별도 원문을 확보하지 못했다. 따라서 §8은 **[H] §2–4의 확정 계산을 [H] §12의 논문형 순서에 맞추어 재서술한 Method 통합본**이다. [P]의 모든 문장·절·그림을 그대로 병합하거나 원문과 독립 대조를 끝냈다는 뜻은 아니다. §9의 정오 목록도 [H]에 기록된 보완 사항을 출처와 함께 계승한다. 확보되지 않은 원문의 내용을 새로 만들어 채우지 않는다.

**확인 범위:** 이번 작성에서는 첨부 명세를 읽고 새 MD를 만들었다. 현재 서버 프로세스·실행 중 step·최신 완료 건수·현재 worktree·현재 Sheet 헤더를 실시간 감사한 것은 아니다. 과거 39/62/66건을 현재 평가 건수로 하드코딩하지 않는다. 실제 적용 시 각 서버가 로컬 자산과 live Sheet를 대조한다.

---

## 1. 이번 변경: 중앙 평가가 아니라 서버별 선행 평가

### 1.1 실행 순서

| 서버 | 1단계: 공통 선행 작업 | 2단계 | 3단계 | 복귀에 필요한 평가 범위 |
|---|---|---|---|---|
| s1 | s1 소유 기존 Student 전수 A_ON/NOA RR·FR 및 업로드 | s1 대표 Aligner 크기·probe·consensus 분석 | s1의 승인된 본 실험 큐 | s1의 전수 평가 + s1 분석 |
| s2 | s2 소유 기존 Student 전수 A_ON/NOA RR·FR 및 업로드 | s2의 승인된 본 실험 큐로 복귀 | — | s2의 전수 평가만 |
| s3 | s3 소유 기존 Student 전수 A_ON/NOA RR·FR 및 업로드 | s3의 승인된 본 실험 큐로 복귀 | — | s3의 전수 평가만 |
| s4 | s4 소유 기존 Student 전수 A_ON/NOA RR·FR 및 업로드 | s4의 승인된 본 실험 큐로 복귀 | — | s4의 전수 평가만 |
| s5 | s5 소유 기존 Student 전수 A_ON/NOA RR·FR 및 업로드 | s5의 승인된 본 실험 큐로 복귀 | — | s5의 전수 평가만 |

s1도 **기존 Student 전수 평가를 먼저** 끝낸다. 이전 [E1]의 `대표 크기 분석 → 전수 평가` 순서는 이번 지시로 바꾼다. 전수 평가 전에 수행하는 짧은 우회 경로·identity sanity check는 필수 구현 검사이며, 전체 크기/probe 분석을 앞당겨 진행한다는 뜻이 아니다.

### 1.2 [E1]에서 교체하는 운영 규칙

| [E1]의 이전 설계 | 이번 v2 설계 |
|---|---|
| 타 서버 selected checkpoint를 s1으로 모아 평가 | 각 학습 서버가 자기 checkpoint를 로컬 평가 |
| s1 진단 writer가 모든 탭의 단일 평가 writer | 각 서버가 자기 탭의 NOA 그룹만 쓰는 소유권 분리 |
| s1에서 대표 분석 후 전체 Student 평가 | s1에서 자기 Student 전수 평가 후 대표 분석 |
| 평가 검토 전 전 서버 신규 seed 시작 보류 | 자기 평가 종료 후 독립 복귀; s1 분석은 타 서버 복귀 조건이 아님 |
| 중앙 s1 평가 완료를 기준으로 전체 복귀 | 서버별 완료 manifest를 기준으로 로컬 복귀 |

NOA 정의, 같은 checkpoint 비교, 원 PAN 참조, 기존 열 보존, 본 방법 자동 변경 금지 등은 [E1]에서 그대로 유지한다.

### 1.3 두 종류의 대기를 혼동하지 않는다

**평가 대기:** 이 문서는 전 서버 공통 평가 장벽을 만들지 않는다. s2 평가가 끝났으면 s3–s5 평가나 s1 분석을 기다리지 않고 s2 본 실험 큐로 돌아간다.

**본 실험의 recipe lock:** [R2]에서 최종 seed-only 단계는 공통 설정 C*의 lock이 있어야 한다. 이 기존 조건은 평가 장벽과 별개다. lock이 아직 없으면 복귀한 서버는 승인된 lock 전 작업을 진행하고, 그 작업도 없다면 `MAIN_WAIT_RECIPE_LOCK`을 표시한다. 서버마다 임의 β를 정해 새 seed를 돌리지 않는다. s5처럼 lock 전 신규 학습이 없는 경우도 같은 규칙을 따른다. 원래 본 실험에 공통 lock이 이미 있으면 바로 그 설정의 미완 seed 작업을 진행한다.

---

## 2. 서버별 상태 전환과 복귀 큐 보호

### 2.1 상태 전환

```text
s1:
MAIN_RUNNING
  → DRAIN_CURRENT_CASE
  → LOCAL_STUDENT_EVAL
  → LOCAL_EVAL_UPLOAD_CHECK
  → S1_ALIGNER_ANALYSIS
  → S1_ANALYSIS_COMPLETE
  → MAIN_RESUME

s2–s5 각각:
MAIN_RUNNING
  → DRAIN_CURRENT_CASE
  → LOCAL_STUDENT_EVAL
  → LOCAL_EVAL_UPLOAD_CHECK
  → MAIN_RESUME
```

실행 중 학습이 없으면 `DRAIN_CURRENT_CASE`는 건너뛴다. `MAIN_RESUME`의 의미는 **각 서버의 저장된 승인 본 실험 큐를 복원·연결한다는 것**이지, 새 학습을 첫 step부터 다시 하거나 완료 case를 다시 돌린다는 뜻이 아니다.

### 2.2 전환 전에 저장할 것

각 서버는 평가 phase를 끼워 넣기 전에 아래의 로컬 복귀 정보를 별도 manifest로 보존한다. 다음 경로명·상태명은 구현 계약으로 제안하며 현재 도구에 이미 존재한다고 가정하지 않는다.

- `source_train_server`, 원래 campaign/queue revision, active/effective/persistent queue 파일 내용과 hash.
- 현재 run ID·진행/후처리/중단 상태, 다음 미완 case, 기존 mandatory/held/reservation 정보.
- 본 실험의 승인된 profile·seed 순서, recipe_lock 경로/ID/존재 상태.
- 평가 release/worktree와 복귀할 학습 release/worktree. 평가 코드 수정이 실행 중 trainer의 정의를 바꾸지 않도록 분리한다.
- 이번 평가 cohort ID, 적용 시각, 복귀 결정의 근거와 완료 상태.

기존 runner·waiter가 평가 중 새 학습을 자동 기동하지 않도록 **해당 서버의 phase-aware 로컬 실행 잠금**을 공유해야 한다. 평가용 큐로 원래 학습 큐를 덮어써 잃어버리지 않는다. 기존 `mandatory`가 평가 job을 학습 50K case로 해석하거나, 옛 held run을 다시 살리지 않도록 한다.

### 2.3 진행 중·중단 run 처리

이미 실행 중인 run은 원 정의로 완료하고 checkpoint와 기존 후처리를 보존한다. 성능이 낮다는 이유로 중간에 종료하지 않는다. 신규 학습은 그 안전 경계 이후 평가 뒤로 넘긴다.

checkpoint를 남긴 `started_interrupted`는 완전 미시작으로 오인하지 않는다. 복귀 큐에 원 resume 정보를 보존한다. 아직 완료되지 않은 중단 run을 전수 **완료 Student 평가**의 숫자에 넣거나, 미완 checkpoint를 최종 결과로 대신 쓰지 않는다. [R2]의 원 정의 완료 정책은 유지한다.

### 2.4 로컬 복귀 조건

각 서버의 필수 Student 평가 건수 N_required에 대해 다음이 모두 완료되어야 한다.

1. paired checkpoint/데이터/평가 정의 확인.
2. 같은 checkpoint의 A_ON RR·FR, NOA RR·FR 산출 또는 검증 가능한 캐시 재사용.
3. metric·identity·per-scene 결과 저장과 자기 탭 업로드 확인.
4. 재시작 시 중복 실행·중복 업로드를 막는 완료 기록 저장.

s1에만 §7 분석 완료가 추가된다. s2–s5는 s1 결과에 대한 중앙 승인이나 전체 취합 보고서를 복귀 전제로 두지 않는다.

누락 자산·오류는 `blocked`/`failed`로 남긴다. 이를 완료로 바꾸거나 숫자 0을 채워 통과시키지 않는다. 해결할 수 없는 항목을 제외하고 복귀할 때는 명시적 제외 사유·승인 기록을 남기고, `complete_with_waiver`와 완전 완료를 구분한다. 한 서버의 blocked 상태를 다른 서버의 자동 정지 조건으로 사용하지 않는다.

명확한 공통 구현 오류가 발견되면 영향받는 평가를 무효화하고 원인·대상·revision을 알린다. 이것은 s1 분석이 끝날 때까지 무조건 전 서버를 기다리게 하는 일반 장벽과 다르다.

---

## 3. 전 서버 공통: 기존 Student 평가 대상 확정

### 3.1 범위

각 서버의 로컬 source manifest와 자기 Sheet 탭을 대조하여 **최근 고정 메서드 계열의 누적 완료 Student 전체**를 잡는다. 특정 날짜·상위 성능·최종 후보 두 개만으로 제한하지 않는다. [E1 §3]

포함 범위는 `qrecon_continuous_v1`의 현재 q 분자1 계열이다. G/H/L, B20A03, A_UNIF/A_SHUF/A_FREEZE, A03_UNIF/A03_SHUF, E/ALL 대조, α=0·β=0 대조와 실제 완료된 Narrow R2 run을 포함한다. 성분 대조군은 `variant_kind`로 구분하되 그 이유만으로 제외하지 않는다.

factor2 구버전·다른 architecture/Teacher·다른 학습 정의는 별도 cohort 또는 제외 사유로 분리한다. 이름·현재 config 주석만으로 과거 실행을 인증하지 말고 training-start metadata를 우선한다. 정의 미확정은 `definition_unverified`, checkpoint 유실은 `missing_checkpoint`, 기존 Sheet checkpoint 연결 미확정은 `source_unresolved`로 남긴다.

### 3.2 유한한 로컬 cohort

평가 phase를 예약하는 시점의 완료 run 목록을 고정한다. 그 시점에 이미 실행 중인 현재 고정-method run도 `pending_current_run_id`로 기록하고, 정상 완료 시 한 번만 합류시킨다. 이후 복귀해 새로 학습할 run이 계속 평가 대상에 추가되어 phase가 끝나지 않는 구조를 만들지 않는다.

신규 본 학습 이후의 NOA 자동 postrun은 이번 선행 cohort의 필수 범위가 아니다. 추후 확장하더라도 이미 완료한 평가 phase를 다시 열지 않는다.

### 3.3 소유권과 key

평가 레코드의 기본 식별자는 다음과 같다.

```text
(source_train_server, original_run_id,
 checkpoint_sha256, checkpoint_step, source_selector,
 dataset_identity, eval_definition_id, eval_mode)
```

source server는 실행 manifest로 확인한다. 현재 평가 프로세스의 host만으로 source를 정하지 않는다. 기본은 `source_train_server == eval_server`이며, 예외 이동은 명시적으로 기록해야 한다.

같은 run/checkpoint의 중복 Sheet 행은 계산을 한 번만 수행하고 canonical row 및 alias row를 기록한다. 서로 다른 selector/step의 행은 합치지 않는다. 원본 run ID만 같더라도 실제 서로 다른 서버 실행 자산이면 별도 key를 사용한다.

### 3.4 checkpoint 고정

**주 비교는 현재 Sheet 본 행이 가리키는 checkpoint**다. 이를 실행 metadata와 연결하여 hash를 고정한다. 다음을 금지한다.

- A_ON과 NOA에서 각각 가장 좋은 checkpoint를 새로 고르기.
- A_ON의 FR checkpoint와 다른 step의 NOA RR를 같은 비교로 넣기.
- 특정 run만 후보 step을 더 촘촘히 늘리기.
- 모든 완료 run × 50개 후보 × 모든 모드의 탐색으로 작업 범위를 확대하기.

기존 Sheet의 RR/FR가 다른 checkpoint에 연결된 것으로 확인되면 같은 checkpoint의 paired result를 별도로 재구성한다. 과거 본 열은 조용히 덮어쓰지 않는다. 새 ERGAS 우선 selector가 다른 step을 고르면 `target_pair`로 분리하며 legacy 행 옆에 다른 step의 NOA 값을 쓰지 않는다. [E1 §4; R2 §2]

---

## 4. 전수 평가의 추론 경로: Aligner와 sampler를 함께 우회

여기서는 고정된 Student checkpoint의 파라미터를 θ, φ로 쓴다. LRMS X의 기존 bicubic base를 M=U4(X), 원 PAN을 P로 둔다. A는 기존 margin4 view와 그 정규화를 포함한 보정량 추정 호출의 약칭이다.

### 4.1 A_ON

$$
\mathbf c=A_\phi(P,M),\qquad
Z_{\mathrm{ON}}=M+F_\theta([\mathcal W(P,\mathbf c),M]).
$$

기존 full-image PAModel 경로를 사용한다. RR에서는 U-Net 전체 입력 256×256, FR에서는 512×512를 유지한다. 실제 입력 shape는 데이터에서 확인해 manifest에 기록한다.

### 4.2 A_BYPASS_RAW — NOA

$$
\boxed{Z_{\mathrm{NOA}}=M+F_\theta([P,M])}.
$$

**Aligner를 호출하지 않고, warp_pan/grid_sample도 호출하지 않는다.** 원 PAN과 원 LRMS를 동일한 Student backbone에 직접 전달한다. U-Net 가중치는 그대로이며 optimizer·backward·추가 학습은 없다. Teacher와 q probe도 전수 NOA 추론에는 필요 없다. GT는 RR metric 계산에만 사용한다.

실제 backbone API는 앞선 코드 확인본의 `backbone(pan, lpan, ms, sw)`다. backbone 내부의 MS upsampling을 유지하며, 수학식의 concat을 이유로 이미 확대된 M을 LRMS 인자로 잘못 넣지 않는다. MS residual base는 최종 출력에 정확히 한 번 더한다. [E1 §2]

full checkpoint를 원래 skeleton에 strict load한 뒤 평가 분기를 선택한다. aligner key를 제거해 새 모델 파일로 저장하거나 원 checkpoint를 수정하지 않는다. 평가용 우회 설정이 본 학습 복귀 모델에 남지 않도록 독립 모델 인스턴스 또는 명시적 복원 검사를 사용한다.

### 4.3 A_ZERO_WARP는 NOA와 다르다

$$
Z_{\mathrm{ZERO}}=M+F_\theta([\mathcal W(P,\mathbf0),M]).
$$

A는 호출하지 않지만 sampler는 사용한다. 앞선 `pa/model.py` 확인본에서 `aligner_enabled=False`만 주면 sampler가 True인 경우 이 경로로 간다. `delta_override=0`도 warp한다. `sampler=False`만 바꾸면 aligner 예측 호출이 남을 수 있다. 따라서 정확한 NOA 구현은 **두 호출이 모두 0회**임을 검사해야 한다. [E1 §2; 앞선 확인 blob `cffe4f6ff9caa93877afa6a2a4d61d6d7a37d4d3`]

### 4.4 각 서버의 짧은 사전 sanity check

각 서버는 자기 대표 checkpoint 하나와 고정 입력으로 다음을 확인한 뒤 전수 평가를 시작한다. 이것은 s1 전용 크기 분석과 별개인 구현 검사다.

- A_ON의 aligner·sampler 사용, NOA의 두 호출 0회, ZERO의 A 호출 0회와 sampler 사용을 확인한다.
- M·입력 intensity·출력 clamp·band order·dtype/precision 정책이 두 모드에서 같다.
- ZERO−NOA 출력의 max/mean absolute difference를 기록한다. 예상 밖 차이는 구현 검토 대상으로 두며 bitwise 동일성을 선험적으로 가정하지 않는다.
- 전후 모델 state hash가 같고 `.grad`/optimizer update가 없으며 eval mode가 유지된다.
- full-frame U-Net을 유지하고, NOA 경로가 PAN 자체나 평가 reference를 crop/resize/warp하지 않는다.

### 4.5 로컬 paired A_ON 재평가 및 캐시

각 서버에서 A_ON과 NOA를 같은 고정 평가 환경으로 계산하는 것이 기본이다. 기존 A_ON 캐시가 checkpoint·data·evaluator·forward·precision·환경 identity와 연결되어 있으면 재사용할 수 있다. 과거 Sheet 숫자만 보고 재평가를 생략하지 않는다.

본 행 A_ON과 이번 paired A_ON이 다르면 `legacy_on_discrepancy`를 저장한다. NOA−ON 차이는 **이번 verified paired ON**을 기준으로 계산한다. 불일치가 미해결이면 기존 열과 직접 비교 가능하다고 표시하지 않는다.

이전 v1 평가가 이미 존재하면 재사용 가능성을 확인한다. 운영 protocol 문자열만 v2로 바뀌었다고 같은 계산을 불필요하게 반복할 필요는 없지만, 계산 동등성과 identity 증거가 있어야 한다. 특히 s1에서 타 서버 자산을 평가한 결과는 새로운 로컬 paired 평가와 자동으로 동일시하지 않는다. 재사용하면 원래 `computed_with_protocol`, eval host와 hash를 보존한다.

---

## 5. RR/FR 규약과 해석

### 5.1 기존 공식 평가 규약 유지

[E1]과 앞선 uploader 확인본의 기준은 RR20장, 기존 `_rr`, `dim_cut=21`, `thvalues=0`이며 FR는 **paper mat20 전체 프레임**이다. 실제 적용 시 source config·dataset/evaluator hash를 고정하고 설치본이 이 정의와 연결되는지 확인한다. 이 문서 작성에서 데이터 bytes나 GPU 출력을 새로 검사한 것은 아니다.

| 구분 | 필수 지표 | 주의 |
|---|---|---|
| RR | ERGAS, SAM, PSNR, SSIM, SCC, Q8 | 기존 GT·crop·metric 구현 유지 |
| RR 보조 | RMSE, CC | 기존 공식 함수가 반환하는 동일 정의 사용 |
| FR | D_lambda, D_s, raw-original HQNR | ON/NOA 모두 원 PAN·원 MS 참조 |
| FR 보조 | V64, JQM 등 기존 보조값 | 주 HQNR이나 완료 조건을 대신하지 않음 |

출력 clamp와 DN 변환, MTF·다운샘플링·평가 crop을 모드별로 바꾸지 않는다. FR에서 NOA에 유리하도록 reference PAN을 바꾸거나, ON만 aligned-self 점수를 넣지 않는다.

### 5.2 평균과 차이

FR 장면 j의 HQNR를 구한 뒤 장면 평균을 낸다.

$$
\overline{\mathrm{HQNR}}
=\frac1{n_{\mathrm{FR}}}\sum_j
(1-D_{\lambda,j})(1-D_{s,j}).
$$

평균 D_lambda와 평균 D_s를 먼저 곱해 HQNR을 재구성하지 않는다. 차이의 부호는 전 서버에서 다음과 같이 통일한다.

$$
\Delta E=E_{\mathrm{NOA}}-E_{\mathrm{ON}},\qquad
\Delta H=H_{\mathrm{NOA}}-H_{\mathrm{ON}}.
$$

ΔE<0이면 NOA ERGAS 개선, ΔH>0이면 NOA HQNR 개선이다. Δ는 반올림한 Sheet 표시값이 아니라 원본 metric에서 계산한다.

### 5.3 공동목표

$$
\operatorname{joint\_pass}(k,m)
=\mathbf1[H(k,m)\ge0.9585]\,
 \mathbf1[E(k,m)<2.040].
$$

k는 같은 checkpoint, m은 같은 평가 모드다. A_ON의 HQNR과 NOA의 ERGAS를 조합해 통과로 표시하지 않는다. 표시 ERGAS 2.040이나 HQNR 0.9585의 반올림만으로 strict 판정을 하지 않는다.

### 5.4 이 실험으로 주장할 수 있는 범위

전수 ON/NOA 비교는 **현재 방식으로 학습된 특정 Student checkpoint에서 추론 정합 경로를 사용하는 것이 유리한가**를 측정한다. 처음부터 aligner 없이 학습한 모델과의 비교가 아니다.

NOA가 좋아도 학습 중 aligner가 불필요했다는 결론은 나오지 않는다. NOA가 나빠도 native absolute alignment가 물리적으로 정확하다고 증명되는 것은 아니다. 작은 입력과 큰 입력에서의 반응 문제는 §7의 별도 분석으로 확인한다.

run별 paired difference와 scene별 차이를 보존한다. profile별 결과와 seed별 결과를 분리하고, 같은 seed의 다른 서버 반복이나 한 장면의 여러 crop을 독립 seed/scene으로 세지 않는다. 기존 RR/FR 개발·선택 세트를 새로운 독립 test set이라고 소개하지 않는다.

---

## 6. Sheet: 각 서버가 자기 탭의 NOA 새 열을 채운다

### 6.1 탭 소유권

아래 이름과 gid는 [H]의 식별 정보다. 적용할 때 metadata를 다시 읽어 연결한다. 행·열 위치는 고정하지 않는다.

| 평가 및 학습 소유 서버 | 원본 탭 | 기록된 gid | 새 열 writer |
|---|---|---:|---|
| s1 | WV3-s1 | 994031662 | s1 로컬 gspread 평가 업로더 |
| s2 | WV3-s2 | 991648123 | s2 로컬 gspread 평가 업로더 |
| s3 | WV3-s3(5090) | 284220763 | s3 로컬 gspread 평가 업로더 |
| s4 | WV3-s4 | 2026091404 | s4 로컬 gspread 평가 업로더 |
| s5 | WV3-s5 | 823586191 | s5 로컬 gspread 평가 업로더 |

s1의 분석 산출물은 로컬 CSV/JSON/MD로 분리한다. s1이 s2–s5 원본 탭의 평가 숫자를 다시 쓰거나, 모든 run을 s1 탭에 새 학습 행처럼 추가하지 않는다.

### 6.2 공통 컬럼 계약

기존 RR/FR·통합실험 열을 유지하고 우측의 사용되지 않은 영역에 **동일한 라벨·키의 NOA 그룹**을 추가한다. 기존 Y열부터라고 가정하지 않는다.

| 그룹 | 표시 라벨 | 기계 판독 키 예시 |
|---|---|---|
| NOA RR | NOA ERGAS↓ / NOA SAM↓ / NOA PSNR↑ / NOA SSIM↑ / NOA SCC↑ / NOA Q8↑ | noa_rr_ergas / noa_rr_sam / noa_rr_psnr / noa_rr_ssim / noa_rr_scc / noa_rr_q8 |
| NOA RR 보조 | NOA RMSE↓ / NOA CC↑ | noa_rr_rmse / noa_rr_cc |
| NOA FR·paper mat20 | NOA D_lambda↓ / NOA D_s↓ / NOA HQNR(raw)↑ | noa_fr_d_lambda / noa_fr_d_s / noa_fr_hqnr_raw |
| Pair reference | A_ON ERGAS(pair) / A_ON HQNR(raw,pair) | paired_on_rr_ergas / paired_on_fr_hqnr_raw |
| NOA−ON | ΔERGAS(NOA−ON) / ΔHQNR(NOA−ON) | delta_rr_ergas / delta_fr_hqnr_raw |
| Pair identity | Eval mode / Eval step / Eval ckpt SHA / Eval selector | eval_mode / eval_step / eval_ckpt_sha / source_selector |
| Status | Eval status / A_ON check / NOA joint pass | eval_status / legacy_on_check / noa_joint_pass |
| Provenance | Eval date / Eval server / Eval protocol / Eval(h) | eval_date / eval_server / protocol_id / eval_hours |

`Eval mode`의 값은 `A_BYPASS_RAW`이고, 그룹 표시명은 NOA로 쓴다. 전체 metric·identity·원본 숫자는 sidecar에 보존하고, Sheet에는 필요한 표시만 반영한다.

### 6.3 다중 서버 동시 쓰기 보호 — v2 추가 요구

이번 변경으로 writer가 5개가 되므로, uploader 확장은 **공통 schema + 탭별 단일 writer** 구조여야 한다.

- schema 정의는 버전 관리된 동일 코드에서 가져온다. 각 서버는 자기 탭의 헤더만 멱등적으로 준비·검증한다. 같은 workbook을 동시에 전체 재배치하지 않는다.
- 자기 탭에서도 학습 uploader와 NOA uploader가 동시에 행을 쓰지 않도록 같은 로컬 쓰기 잠금을 사용한다. 기존 runner/waiter의 중복 writer도 이 잠금을 공유한다.
- 긴 설명 문자열·mutable row 번호만으로 매칭하지 않는다. 쓰기 직전에 원본 ID와 checkpoint 의미를 다시 확인한다.
- 전체 행을 빈 칸 포함 덮어쓰지 않는다. NOA writer는 자기 새 컬럼 범위만 갱신하고, 학습 writer는 NOA 컬럼을 보존한다.
- `--all --replace`나 전체 Sheet clear/layout 재생성으로 구현하지 않는다. 기존 Date/Train(h)/Cost/원 RR·FR를 보호한다.
- API 속도 제한·일시 오류에는 로컬 재시도와 backoff를 적용한다. 업로드 실패를 GPU 재학습이나 metric 재추론으로 처리하지 않는다.
- 동일 run/step 중복행의 alias 갱신은 identity가 같을 때만 허용한다. 계산·집계는 한 번이다.
- 복귀 후 새 본 실험 행을 기록해도 이전 NOA 결과가 소실되지 않는 회귀 검사를 수행한다.

공통 코드는 s1에서 준비할 수 있지만, 각 서버의 header 준비·평가 실행·완료·복귀는 독립적이다. 대화 도구로 Sheet를 직접 편집하지 않고 기존 gspread 운영 경로로 반영한다.

### 6.4 저장과 캐시

제안 결과 경로:

```text
work_dir/<run>/results/aligner_eval_v2/<checkpoint_sha>/<mode>/
  rr_metrics.json
  fr_metrics.json
  per_scene_rr.csv
  per_scene_fr.csv
  identity.json
  [필요한 RR/FR 예측 출력 파일]
```

기존 `reduced.mat`, `fr_mat20.json`, best_hqnr metadata를 덮어쓰지 않는다. 캐시 identity는 run/ckpt/config·dataset·evaluator·forward implementation·mode·precision·환경·crop/probe 정책을 포함한다. ON의 출력을 checkpoint hash가 같다는 이유로 NOA 캐시로 사용하지 않는다.

RR·FR 완료, paired identity 완료, upload 완료는 서로 다른 상태로 기록한다. 일부 수치만 있는 행을 `complete`로 표시하지 않는다. source config의 절대 경로를 로컬 경로에 매핑하더라도 원 config와 path mapping을 따로 보존한다.

---

## 7. s1 전용: 자기 Student 평가 후 Aligner 분석 실험

### 7.1 대상과 순서

s1 전수 Student 평가·업로드가 끝난 뒤 다음 네 자산으로 분석한다. 아래 selected step·hash는 source에서 다시 확인한다. [E1 §5]

| 대상 | 역할 |
|---|---|
| T0의 기존 selected A/U | Teacher 자체 반응 확인; 전수 Student 건수에 포함하지 않음 |
| s1 G23·1234 | 기존 좋은 selected 자산; 이전 기록은 step31310 |
| s1 G23·3407 | 같은 profile의 다른 seed |
| s1 G22·1234의 유효한 factor1 실행 | rA 차이가 있는 비교 자산 |

s1의 전수 평가에서 나온 ON/NOA 결과는 identity를 확인한 뒤 재사용한다. 분석을 위해 같은 forward를 다시 계산했다면 기존 결과와 일치 여부를 검사한다. 이 네 자산으로 우선 분석을 끝내며, 타 서버 checkpoint 수집을 선행 의존성으로 추가하지 않는다.

### 7.2 크기 진단 — 원래 pixel scale 유지

RR에서는 64/128 crop과 256 전체, FR에서는 64/128/256 crop과 512 전체를 사용한다. M은 전체 X에서 기존 bicubic ×4로 한 번 계산하고 P와 같은 좌표로 crop한다. crop마다 LRMS를 다시 확대해 보간 경계까지 바꾸지 않는다. resize/scale jitter를 넣지 않는다.

crop마다 기존 margin4를 정확히 한 번 적용한다. 다음 값은 [H] 구조·margin에서 계산되는 내부 shape이며 새 GPU 측정값은 아니다.

| PAN view 크기 | Aligner crop 후 | 두 stride-2 이후 | GAP 후 | 최종 출력 |
|---:|---:|---:|---:|---:|
| 64×64 | 56×56 | 14×14 | 64차원 | dy, dx |
| 128×128 | 120×120 | 30×30 | 64차원 | dy, dx |
| 256×256 | 248×248 | 62×62 | 64차원 | dy, dx |
| 512×512 | 504×504 | 126×126 | 64차원 | dy, dx |

§7.4의 64 crop bank는 [E1]에서 지정한 non-overlap grid를 유지한다. 다른 크기도 **이번 v2의 구체화로** 같은 non-overlap 규칙을 미리 고정한다. crop origin은 stride-4 좌표에 놓인다. 실제 shape가 예상과 다르면 임의 resize하지 말고 panel 정의를 명시적으로 갱신한다.

| 원본 크기 | 64 crop | 128 crop | 256 crop | full |
|---|---:|---:|---:|---:|
| RR256 | 16개 | 4개 | 전체와 동일하여 중복 계산 제외 | 1개 |
| FR512 | 64개 | 16개 | 4개 | 1개 |

crop 전체 수가 늘어도 독립 scene 수는 늘지 않는다. 크기별 결과는 scene 안에서 요약한 뒤 scene 간 집계한다.

### 7.3 알려진 PAN 추가 이동에 대한 반응

기존 AXIS16 probe를 유지한다: 반경 0.25/0.5/1/2 HR pixel 각각에 대해 ±y, ±x 네 축 방향이다. 원 PAN 전체에 probe warp를 한 뒤 같은 좌표의 crop을 읽고, M은 이동시키지 않는다.

$$
\mathbf c^{(\ell,u)}_0=A_\phi(P^{(\ell,u)},M^{(\ell,u)}),
$$

$$
\mathbf c^{(\ell,u)}_j
=A_\phi\big((\mathcal W(P,\boldsymbol\epsilon_j))^{(\ell,u)},M^{(\ell,u)}\big),
$$

$$
q_{\mathrm{size}}^{(\ell,u)}
=\frac{1}{2N_{\mathrm{pr}}}\sum_{j=1}^{N_{\mathrm{pr}}}
\|\mathbf c^{(\ell,u)}_j+\boldsymbol\epsilon_j-\mathbf c^{(\ell,u)}_0\|_1.
$$

ℓ은 crop의 한 변 크기, u는 같은 scene 내 crop 인덱스다. 이 q_size는 사후 측정값이며 학습 q cache/q_ref와 분리한다. Student에 offset loss를 다시 넣지 않는다.

보간 support·원본 경계의 영향을 함께 기록하고, 필요하면 고정된 동일 support 비교를 보조값으로 남긴다. 평가마다 달라지는 mask로 공식 RR/FR 영역을 변경하지 않는다. shift가 커져 support가 깨졌다는 이유로 예측 shift를 임의 clipping해 결과를 개선하지 않는다.

native dy/dx, probe별 반응, q_size의 평균·중앙값·상위 분위, crop 간 offset 분산, full offset과 crop-consensus 차이를 scene별로 저장한다. 반응 차이가 발견됐다고 곧바로 실제 지역별 기하 변위가 다르다거나 native 정합이 틀렸다고 단정하지 않는다.

### 7.4 같은 전체 U-Net에서 네 모드 비교

| 모드 | Aligner 입력·shift | 복원 |
|---|---|---|
| A_ON | full input에서 예측한 한 쌍 | 전체 PAN warp 후 full U-Net |
| A_ZERO_WARP | 0 shift; A 미호출 | identity warp 후 full U-Net |
| A_BYPASS_RAW | A/warp 모두 미호출 | 원 PAN으로 full U-Net |
| A_CROP64_MED | non-overlap 64 crop별 shift의 성분별 중앙값 한 쌍 | 그 한 쌍으로 전체 PAN을 한 번 warp 후 full U-Net |

A_CROP64_MED의 median은 각 축의 0.5 quantile로 정의하고, 짝수 개수에서는 가운데 두 값을 평균한다. 구현 함수가 짝수 배열에서 작은 가운데 값만 반환하는지 확인한다. GT나 scene별 점수를 보고 mean/median·crop 위치를 선택하지 않는다.

모든 모드에서 U-Net은 RR256/FR512 전체를 받는다. patch별 서로 다른 warp, dense flow, U-Net tiling, GT-guided shift search는 이번 기본 분석에 없다. T0 분석도 같은 T0 checkpoint의 U/A를 사용한다.

### 7.5 완료와 복귀

s1 분석 산출물은 고정 panel의 크기/probe 반응 CSV·JSON, 네 모드의 RR/FR paired metric, scene별 비교 및 해석 MD다. **좋은 결론이 나와야 완료되는 것이 아니라 지정한 비교가 끝나고 해석 범위가 정리되면 완료**다.

q_size가 작은 것과 absolute alignment의 정확성은 구분한다. NOA나 consensus가 유리해도 본 방법을 자동 변경하지 않는다. 명확한 구현 오류가 없다면 s1은 승인된 본 실험 큐로 복귀한다. 타 서버가 이미 본 실험에 돌아갔다면 분석 결과를 전달하되 그들의 실행 중 학습을 소급 변경하지 않는다.


---

## 8. 현재 Method — 논문형 수식과 설명의 통합본

**출처와 범위:** 이 절은 [H] §2–4 및 §12의 서술 순서를 따른다. [P] 원문 전체를 확보한 상태에서 복사·병합한 것이 아니다. 학습 목적함수와 실제 gradient 경로는 변경하지 않는다. 아래의 q 기반 주 방법과 §3에 포함한 uniform/shuffle/frozen 등의 성분 대조는 구분한다.

### 8.1 문제 설정과 표기

PAN-sharpening의 입력은 LRMS \(X_i\in\mathbb R^{C\times H/4\times W/4}\)와 PAN \(P_i\in\mathbb R^{1\times H\times W}\)이며, 목표는 MS 좌표계에 놓인 HRMS \(Y_i\in\mathbb R^{C\times H\times W}\)다. WV3에서는 C=8이다. Bicubic ×4 연산 \(\mathcal U_4\)로 MS base를 만든다.

$$
M_i=\mathcal U_4(X_i).
\tag{1}
$$

모델은 PAN의 global 보정량을 예측하는 aligner \(A_\phi\)와 HRMS residual을 복원하는 U-Net \(F_\theta\)로 구성된다. Teacher는 T0 W112–D123, Student는 W104–D121이다. 연구 목적은 단순 압축에 한정되지 않고, 학습된 PAN 정합과 Teacher의 복원 정보를 이용해 HRMS 복원을 개선하는 것이다. [H §0–3]

| 기호 | 의미 |
|---|---|
| B, i | batch의 patch 수, patch 인덱스 |
| C, b | MS band 수, band 인덱스 |
| H, W, Ω, N=HW | 복원 영상 높이·너비, 전체 공간 영역, 공간 위치 수 |
| p | 공간 위치 |
| φ_T, θ_T / φ_S, θ_S | Teacher / Student의 aligner·U-Net 파라미터 |
| c_T,i, c_S,i | global 보정 벡터 (dy,dx), 현재 PAN/HR 좌표의 pixel 단위 |
| Z_T,i, Z_S,i | M을 더한 최종 Teacher/Student HRMS 출력 |
| e_T, e_S, k | Teacher–GT, Student–GT, Student–Teacher의 band 평균 pixel L1 |
| d_T, a_T | Teacher difficulty, Student 대비 Teacher advantage |
| q_T,i, q_ref, s_i | Teacher consistency residual, 고정 기준 scale, 연속 q 가중치 |
| ε_j, N_pr | 알려진 2D probe 이동량, probe 수 16 |
| ε_R, ε_τ, ε_z | advantage 분모, error scale 하한, z-score 분산 안정화 상수 |
| ℓ_H,i, ℓ_K,i, ℓ_E,i | patch hard, β가 포함된 soft, 계수 미포함 edge 오차 |
| L_U, L_A | Student U와 A에 각각 미분할 목적함수 |
| sg | stop-gradient |

높이 H와 hard loss, probe 수와 soft loss 기호가 겹치지 않도록 patch loss를 ℓ_H, ℓ_K, ℓ_E로 표기한다. 이는 [H]의 표기 별칭이며 계산 변경이 아니다.

### 8.2 PAN 정합과 좌표계

Aligner는 PAN과 MS base에서 같은 위치의 정보를 결합해 전역 보정 벡터를 추정한다. 실제 구현은 **aligner 전용 margin4 crop → sample·channel별 공간 z-score → dual-stem CNN** 순서다. 이 절에서 \(A_\phi(P,M)\)는 그 view 전처리를 포함한 전체 보정량 추정 호출을 뜻한다.

공간 표준화는 crop된 입력 V에 대해 다음과 같다.

$$
\operatorname{znorm}(V)
=\frac{V-\operatorname{sg}(\mu(V))}
{\sqrt{\operatorname{sg}(\operatorname{var}(V))+\epsilon_z}},
\qquad \epsilon_z=10^{-6}.
\tag{2}
$$

통계의 detach는 통계가 모든 입력에 대해 상수라는 뜻이 아니다. U-Net과 sampler에는 원래 intensity scale의 입력을 사용하며, aligner 전용 표준화 영상을 복원 입력으로 바꾸지 않는다. [H §3.1]

CNN 구조는 PAN stem 1→16과 MS stem 8→16의 feature를 concat하여 32 channel로 만든 뒤, stride-2 conv 32→32, stride-2 conv 32→64, residual64, GAP64, linear64→32→2를 통과하는 구성이다. [H]에 기록된 WV3 aligner 파라미터 수는 105,330개이며 이번 문서에서 다시 계측한 값은 아니다. Dense flow·attention·confidence head를 추가하지 않는다.

Warp의 보정량은 sampling할 원본 위치의 offset이다.

$$
\mathcal W(P,\mathbf c)(y,x)=P(y+c_y,x+c_x).
\tag{3}
$$

CNN 출력 순서는 (dy,dx), sampler grid 순서는 (x,y)다. 기존 `bicubic / border / align_corners=False`를 유지한다. **PAN만 이동하고 M·GT·출력 목표 frame은 고정**한다. 최종 HRMS에 inverse warp를 적용하지 않는다. [H §2.2]

### 8.3 Teacher 구축과 Student 초기화

실제 자산 계보는 **N2 donor → L1E4 Teacher T0 → Student**다. N2 donor의 exact50K last에서 aligner만 복사하여 fresh W112–D123 U-Net과 L1E4 Teacher를 구성한다. N2의 U-Net·optimizer를 그대로 이어받는 것이 아니다. [H §3.2]

L1E4에서는 native 입력과 알려진 추가 PAN 이동을 비교한다.

$$
P_i^\epsilon=\mathcal W(P_i,\boldsymbol\epsilon_i),\quad
\mathbf c_{i,0}=A_\phi(P_i,M_i),\quad
\mathbf c_{i,\epsilon}=A_\phi(P_i^\epsilon,M_i).
\tag{4}
$$

$$
L_{\mathrm{off}}
=\frac1{2B}\sum_{i=1}^B
\|\mathbf c_{i,\epsilon}+\boldsymbol\epsilon_i
-\operatorname{sg}(\mathbf c_{i,0})\|_1.
\tag{5}
$$

Vector L1을 두 좌표 절댓값의 합으로 정의했으므로 1/(2B)는 두 성분과 batch의 평균이다. 추가 변위는 반경 2 HR pixel 원판에서 면적 균등으로 뽑는다.

$$
L_{T,t}=L_{\mathrm{rec},T}+m_t\,10^{-4}L_{\mathrm{off}},
\qquad m_t=\mathbf1[t\bmod2=1].
\tag{6}
$$

Native reconstruction은 매 update 수행하고 offset 보조항은 기존 update 인덱스의 홀수 step에서만 활성화된다. 추가로 이동시킨 PAN은 보조 aligner 경로에만 들어가며, Teacher U-Net은 매번 native PAN을 보정한 영상으로 복원한다. 식 (5)의 native target 경로는 stop-gradient지만 reconstruction→aligner 경로는 살아 있다. Teacher A는 recon+off, Teacher U는 recon을 받는다. [H §3.3]

선택된 T0는 다음 자산으로 인계되어 있다. 이는 현재 서버에 파일이 존재한다는 확인이 아니라 **검증할 자산 identity**다.

```text
T0 run: PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
selected tag: best_hqnr
selected step: 24240
logical source: assets/pakd50/T0_run/best_hqnr
checkpoint SHA256:
16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32
```

Student 학습에서 Teacher의 A/U는 모두 frozen이다. Student A는 T0 A 전체 state의 **별도 clone**으로 초기화하고, Student U는 fresh 초기화한다. Teacher와 Student A를 같은 module object로 공유하지 않는다. 이후 Student A는 작게 적응할 수 있다. 이번 기존 Student 평가는 이 초기화를 다시 하는 것이 아니라 이미 완료된 Student checkpoint를 그대로 읽는다.

### 8.4 Teacher와 Student의 최종 출력

$$
\mathbf c_{T,i}=A_{\phi_T}(P_i,M_i),\qquad
\mathbf c_{S,i}=A_{\phi_S}(P_i,M_i),
\tag{7}
$$

$$
Z_{T,i}=M_i+F_{\theta_T}
\left([\mathcal W(P_i,\mathbf c_{T,i}),M_i]\right),
\tag{8}
$$

$$
Z_{S,i}=M_i+F_{\theta_S}
\left([\mathcal W(P_i,\mathbf c_{S,i}),M_i]\right).
\tag{9}
$$

Teacher와 Student는 각각 자기 aligner를 이용한다. Z는 residual 자체가 아니라 MS base를 한 번 더한 최종 HRMS다. Student의 native 입력은 추가 synthetic jitter가 없다는 뜻이며 aligner 보정을 끈다는 뜻이 아니다. [H §2.2–2.3]

### 8.5 복원 오차, Teacher difficulty와 advantage

복원 오차는 band 평균 pixel L1으로 정의한다.

$$
e_{T,i}(p)=\frac1C\sum_{b=1}^C
|Z_{T,i,b}(p)-Y_{i,b}(p)|,
\tag{10}
$$

$$
e_{S,i}(p)=\frac1C\sum_{b=1}^C
|Z_{S,i,b}(p)-Y_{i,b}(p)|,
\tag{11}
$$

$$
k_i(p)=\frac1C\sum_{b=1}^C
|Z_{S,i,b}(p)-\operatorname{sg}(Z_{T,i,b}(p))|.
\tag{12}
$$

세 오차는 [B,1,H,W] map이다. k는 두 출력의 차이이지 |e_S−e_T|가 아니다. Teacher 오류 e_T는 GT와 비교한 실제 복원 오차이며 별도 uncertainty head의 예측값으로 설명하지 않는다.

Teacher error scale은 학습 calibration에서 고정한다.

$$
\tau_R=\max\!\left(
\operatorname{median}_{(i,p)\in\mathcal D_{\mathrm{cal},R}}
 e_{T,i}(p),\epsilon_\tau\right),
\qquad \tau_R=0.012463942170143127.
\tag{13}
$$

Difficulty와 advantage는 다음과 같다.

$$
d_{T,i}(p)=\operatorname{sg}\!\left(
\frac{e_{T,i}(p)}{e_{T,i}(p)+\tau_R}\right),
\tag{14}
$$

$$
a_{T,i}(p)=\operatorname{sg}\!\left[
\operatorname{clip}_{[0,1]}
\frac{[e_{S,i}(p)-e_{T,i}(p)]_+}{e_{S,i}(p)+\epsilon_R}
\right],\qquad \epsilon_R=10^{-6}.
\tag{15}
$$

Difficulty는 Teacher가 GT에서 벗어난 정도를, advantage는 Teacher가 현재 Student보다 나은 정도를 나타낸다. 분자는 **Student error − Teacher error**다. Student가 같거나 더 정확하면 advantage는 0이다.

d_T와 a_T의 계산 경로는 detach한다. 그러나 실제 hard loss의 e_S와 soft loss의 k는 live로 유지한다. a_T 계산에 쓰는 Student 오차 복사본만 멈추어야 하며 e_S 전체를 detach하면 안 된다. Teacher·GT에는 gradient가 없다. [H §2.3–2.4]

### 8.6 GT hard와 adaptive output soft

$$
\ell_{H,i}=\frac1N\sum_{p\in\Omega}
(1+\alpha d_{T,i}(p))e_{S,i}(p),
\tag{16}
$$

$$
\ell_{K,i}=\frac1N\sum_{p\in\Omega}
\beta(1-d_{T,i}(p))a_{T,i}(p)k_i(p).
\tag{17}
$$

GT hard의 기본 계수는 모든 위치에서 1이다. Teacher가 크게 틀린 곳은 추가 GT 강조를 받고, Teacher가 비교적 정확하며 현재 Student보다 유효한 위치에서는 output soft를 참고한다. Hard와 soft를 양자택일하는 binary switch가 아니다.

β는 이미 ℓ_K에 포함돼 있으므로 최종 loss에서 다시 곱하지 않는다. 평균의 분모는 고정 N이며 soft 활성 pixel 수나 가중치 합으로 재정규화하지 않는다. band 평균 error로 weight를 만들기 때문에 각 band에서 Teacher가 모두 더 정확하다는 뜻은 아니다. [H §2.5]

### 8.7 Teacher consistency q와 고정 연속 가중치

고정 Teacher aligner에 native PAN과 알려진 이동 probe를 넣는다.

$$
\mathbf c_{T,i,0}=A_{\phi_T}(P_i,M_i),\qquad
\mathbf c_{T,i,j}=A_{\phi_T}
(\mathcal W(P_i,\boldsymbol\epsilon_j),M_i),
\tag{18}
$$

$$
q_{T,i}=\frac1{2N_{\mathrm{pr}}}
\sum_{j=1}^{N_{\mathrm{pr}}}
\|\mathbf c_{T,i,j}+\boldsymbol\epsilon_j-\mathbf c_{T,i,0}\|_1,
\qquad N_{\mathrm{pr}}=16.
\tag{19}
$$

Probe bank는 반경 0.25/0.5/1/2 HR pixel × 네 축 방향이다. 1/(2N_pr)은 probe와 두 좌표 성분의 평균이다. 학습 calibration에서 q_ref를 고정하고 다음 연속 weight를 사용한다.

$$
q_{\mathrm{ref}}=\operatorname{median}_{i\in\mathcal D_{\mathrm{cal},q}}q_{T,i}
=0.3276133416220546,
\tag{20}
$$

$$
\boxed{s_i=\operatorname{sg}\!\left(
\frac{q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_{T,i}}\right)}.
\tag{21}
$$

분자에는 q_ref 하나만 있고 2는 없다. q=q_ref이면 0.5, q=0이면 1이다. q_ref는 threshold가 아니라 scale이며 cutoff·temperature·δ·batch 재정규화·추가 clip을 넣지 않는다.

q는 Student 학습 중 재추정되는 값이 아니라 T0 raw q cache에서 실제 sample/augmentation view에 맞추어 읽는 고정 cue다. 과거 binary gate 값을 이 식에 대입하지 않는다. q가 낮다는 것은 알려진 추가 이동에 대한 Teacher aligner 반응이 일관적이라는 뜻이지 native absolute alignment가 정확하다는 뜻이 아니다. [H §2.6, §4]

### 8.8 GT signed-edge 감독

각 band에 고정 signed Scharr 연산자를 적용한다.

$$
S_x=\frac1{32}
\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},
\qquad S_y=S_x^\top.
\tag{22}
$$

D_x,D_y를 각 방향의 filtering 연산으로 두면 edge 오차는 다음과 같다.

$$
\begin{aligned}
\ell_{E,i}
={}&\frac12\operatorname{mean}_{b,p\in\Omega_E}
|D_xZ_{S,i,b}(p)-D_xY_{i,b}(p)|\\
&+\frac12\operatorname{mean}_{b,p\in\Omega_E}
|D_yZ_{S,i,b}(p)-D_yY_{i,b}(p)|.
\end{aligned}
\tag{23}
$$

Ω_E는 함수 입력의 상하좌우 1 pixel을 제외한 고정 interior다. reflect padding1을 유지한다. GT와 같은 band의 **부호 있는 방향 미분 응답**을 맞추며, 이진 edge나 gradient magnitude만 비교하는 방법이 아니다. Target은 GT이고 PAN/Teacher edge 모방이 아니다.

$$
L_{\mathrm{qEdge}}=\frac{\lambda_E}{B}
\sum_i s_i\ell_{E,i}.
\tag{24}
$$

λ_E는 독립된 절대 계수이고, edge에 d_T나 a_T를 다시 곱하지 않는다. 옛 config의 `stat.window=5`가 남아 있어도 현재 EDGE를 5×5 variance loss로 해석하지 않는다. [H §2.7]

### 8.9 Student의 두 목적함수와 gradient 분리

$$
\boxed{L_U=\frac1B\sum_i
[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}]},
\tag{25}
$$

$$
\boxed{L_A=\frac1B\sum_i s_i\ell_{H,i}},
\tag{26}
$$

$$
\boxed{G_U=\nabla_{\theta_S}L_U,\qquad
G_A=\nabla_{\phi_S}L_A}.
\tag{27}
$$

| 감독 항 | Student U-Net | Student aligner |
|---|---|---|
| GT hard | ℓ_H | s·ℓ_H |
| Teacher output soft | ℓ_K | 직접 gradient 없음 |
| GT edge | λ_E·s·ℓ_E | 직접 gradient 없음 |
| Student offset consistency | 없음 | 없음 |

같은 forward·같은 pre-step 파라미터에서 두 gradient를 별도로 계산하고 AdamW를 한 번 step한다. `L_U+L_A`를 더해 전체 모델에 단일 backward하면 의도와 다르다.

L_A를 미분할 때 U-Net의 입력 Jacobian을 거쳐 PAN warp와 aligner로 gradient가 전달되지만, 그 gradient를 U-Net 파라미터에 누적하지 않는다. A의 hard도 같은 α·d_T를 쓰므로 patch/pixel별 계수는 s_i(1+αd_T)다. β와 λ_E는 같은 상태의 A gradient에 직접 들어가지 않지만, U-Net을 변화시켜 이후 학습에는 간접 영향을 줄 수 있다. [H §2.8]

$$
\eta_A(t)=r_A\eta_U(t).
\tag{28}
$$

r_A는 optimizer LR 비율이며 shift 값에 곱하는 계수가 아니다. q는 sample의 기여를, r_A는 optimizer의 학습 속도를 조절한다. loss scale·AdamW LR·실제 shift 변화량을 자동 정비례로 취급하지 않는다.

### 8.10 학습·추론의 고정 조건

[R2]의 좁힌 본 실험에서는 λ_E=0.002, α=1, U peak LR=1e−4, r_A=0.03을 고정하고 β=.1(G23)과 β=.2(B20A03)의 마지막 비교만 남긴다. 이것은 기존 전수 평가의 모든 Student를 이 값으로 재해석한다는 뜻이 아니다. 전수 평가에서는 각 run의 실제 학습 config를 보존한다.

Fresh50K, batch48, AdamW, WD=.01, warmup100과 기존 cosine을 유지한다. optimizer β/ε·minimum LR·precision·clipping·accumulation·weight-decay exclusion은 실제 검증된 resolved config를 계승하며 문서에 없는 값을 라이브러리 default로 채우지 않는다. [H §5; R2 §3]

WV3 intensity scale은 `2x/2047−1`이며 τ_R도 그 scale의 값이다. crop=False와 실제 augmentation/index/rotation metadata를 보존한다. [H]에 기록된 feeder에서는 hflip/vflip=True가 해당 flip을 항상 적용하는 동작이므로, 이름만 보고 임의로 50% 확률 처리로 수정하지 않는다. 이번 평가 phase는 training augmentation을 변경하지 않는다. [H §10.4]

정상 추론에서는 같은 Student checkpoint의 A/U만 사용하며 Teacher·GT·q probe·loss 계산이 없다. NOA와 crop-consensus는 이번 사후 평가 모드이고 정상 A_ON을 자동 대체하지 않는다.

---

## 9. [P] 논문형 초안과의 정오·보완을 명시적으로 보존

아래는 **[H] §12.3에 기록된 차이**를 옮긴 것이다. 이번 문서 작성자가 확보하지 못한 [P] 원문을 독립적으로 대조했다는 주장이 아니다. 원문에 없다고 기록된 요소는 출처를 밝히고 §8에 반영했다.

| [H]가 기록한 [P] 상태 | 이번 통합본에서 계승한 정확한 정의 |
|---|---|
| Teacher L_off가 1/B Σ vector L1로 표현됨 | vector L1이 두 성분 합이면 1/(2B) component mean |
| Teacher offset 활성 주기 생략 | native recon 매 update, 홀수 update에만 1e−4·L_off |
| d_T/a_T에 SG 표시 불명확 | weight는 detach, 특히 a_T의 Student error 복사 경로만 SG |
| edge 평균 영역을 공간 전체로만 설명 | reflect1·Scharr/32·고정 1px interior·band별 signed 비교 |
| low-q 구조/high-q 식생 설명이 넓음 | 특정 분석 집단 관측/가설로 제한, semantic class나 절대 정합 신뢰도로 일반화하지 않음 |
| SGD 형태의 단순 θ←θ−η∇L | gradient 경로 설명으로만 읽고 실제 optimizer는 AdamW로 명시 |
| H_i와 영상 H, K_i와 probe 수 중복 | ℓ_H/ℓ_K/ℓ_E, N_pr 등 [H]의 분리 표기 |
| Teacher 구축 단계에서 donor 축약 | N2 donor → L1E4 T0 → Student 계보 유지 |
| 설명 그림의 Teacher error−Student error | 실제 advantage는 Student error−Teacher error |

사용자가 제거한 **q weight 분자의 2**와 Teacher component mean·q probe component mean·edge 두 방향 평균의 1/2는 다른 상수다. 한꺼번에 없애지 않는다.

전체 방법은 학습된 PAN 정합 재사용, Teacher-error 기반 adaptive output KD, q 기반 GT 구조 감독·정합 gradient 배분으로 설명한다. 직접 출력 증류는 soft이고 hard 재가중·q-edge는 Teacher 정보로 조절된 GT 감독이다. 현재 자료만으로 모든 성분 효용, 모든 센서 일반화, q 함수 최적성, absolute alignment 정확성, SOTA·최초성을 확정하지 않는다. [H §12.4]

---

## 10. 본 실험 복귀: 기존 QRECON24와 Narrow R2를 이어간다

### 10.1 복귀 대상은 새 학습법이 아니라 저장해 둔 각 서버의 본 실험

각 서버는 자기 평가 완료 후 저장된 `resume_queue`로 복귀한다. s1만 분석이 그 앞에 추가된다. 이 문서가 [R2]의 수학적 profile·seed·보류 결정을 새로 바꾸거나 원격에 자동 적용하는 것은 아니다.

아래는 [R2]에 정의된 **승인 계획의 연결표**다. 실제 완료/진행 상태를 확인한 뒤 이미 끝난 항목은 건너뛴다. 원격 적용이 안 된 과거 queue와 충돌하면 실행 책임자가 승인 revision으로 정리하고 그 근거를 남긴다. 무조건 저장소 main을 pull하거나 과거 넓은 grid 전체를 다시 넣지 않는다.

| 서버 | 이번 선행 phase | [R2] lock 전 본 실험 | lock 이후 |
|---|---|---|---|
| s1 | 자기 전수 평가 → 대표 분석 | 기존 G23 목표값 연결·보존, B20A03·1234 → B20A03·3407 | C* 지정 seed 4개 |
| s2 | 자기 전수 평가 | release/초기값 유효성 확인 후 B20A03·777 | C* 지정 seed 4개 |
| s3 | 자기 전수 평가 | 기존 4321 pair 연결, 초기값 확인 후 B20A03·2026 | C* 지정 seed 4개 |
| s4 | 자기 전수 평가 | G23·1234 기존 v2 ID의 host bridge | C* 지정 seed 4개 |
| s5 | 자기 전수 평가 | 기존 selector·자산 준비; [R2]의 넓은 grid 미시작분은 보류 유지 | C* 지정 seed 4개 |

s2–s5가 본 실험에 돌아간 뒤 s1 분석이 진행되어도 정상이다. 이미 시작한 본 실험의 정의를 s1 분석 결과로 도중에 바꾸지 않는다.

### 10.2 ERGAS 우선 checkpoint 선택

복귀할 주 방법의 후보 checkpoint 집합 K_r에서

$$
\mathcal F_r=\{k\in\mathcal K_r:H_{\mathrm{ON}}(r,k)\ge0.9585\}
$$

를 만든다. 그 집합에서 공식 RR ERGAS 최소 → SCC 최대 → PSNR 최대 → SAM 최소 → Q8 최대 → SSIM 최대 → HQNR 최대 → step의 결정적 tie-break 순서를 적용한다. 별도 epsilon tie band는 넣지 않는다. [R2 §2]

제안 selector ID는 `HQNR9585_ERGAS2040_v2`다. 이전 확인본의 `HQNR9585_RR_v1`은 SCC가 먼저였으므로 설치본의 실제 ORDER와 완료 판정 ID를 확인한다. 이번 문서는 이 selector의 원격 구현 여부를 새로 인증하지 않는다.

HQNR 우선은 하한을 지키는 것이다. 둘 다 하한을 넘으면 H=.9586/E=2.039가 H=.9600/E=2.070보다 선택상 우선이다. H=.9579/E=2.030은 하한 미달이다. 공식 RR 미완료 후보를 proxy와 섞어 최종 순위를 확정하지 않는다.

이번 선행 ON/NOA 평가는 기존 본 행 checkpoint 고정 비교이므로, 새 selector 적용과 섞어 모드 효과를 계산하지 않는다. 새로운 target step은 별도 paired record로 만들고 그 identity를 기록한다.

### 10.3 마지막 β 비교와 공통 lock

현재 잠정 기준은 G23(β=.1)이다. B20A03(β=.2)과 비교하는 기존 공통 block은 s1·1234/3407, s2·777, s3·2026/4321이며, 새 학습이 필요한 것은 [R2] 시점 기준 최대 4개 B20A03이다. s4 G23·1234는 host bridge 하나이지 독립 seed 추가가 아니다.

공통 유효 block에서 다음 순서로 고정한다. [R2 §4]

1. HQNR 하한과 ERGAS<2.040을 동시에 통과한 seed 수.
2. 동률이면 HQNR 하한을 통과한 seed 수.
3. 동률이면 HQNR 통과 seed의 선택 ERGAS 중앙값.
4. 여전히 같거나 판단 근거가 부족하면 G23 유지.

이것은 제한된 개발 표본의 운영 선택 규칙이며 통계적 우월성 증명이 아니다. β=.15, r_A=.02 같은 세 번째 후보를 몰래 추가하지 않는다.

`recipe_lock`에는 profile·모든 loss 계수·U/A gradient routing·optimizer/scheduler·50K·Teacher/cue/data/evaluator identity·정상 추론 A_ON·selector·공통 seed 배정·선택 근거를 고정한다. s1 분석에서 NOA가 좋았다는 이유로 이 lock의 정상 추론을 자동 NOA로 바꾸지 않는다.

### 10.4 lock 후 seed만 바꾸는 계획

| 서버 | [R2]의 지정 신규 Student seed | 공통 profile |
|---|---|---|
| s1 | 41001 / 41006 / 41011 / 41016 | C* |
| s2 | 41002 / 41007 / 41012 / 41017 | C* |
| s3 | 41003 / 41008 / 41013 / 41018 | C* |
| s4 | 41004 / 41009 / 41014 / 41019 | C* |
| s5 | 41005 / 41010 / 41015 / 41020 | C* |

C*는 lock 후 G23 또는 B20A03으로 해소할 표기이지 현재 실행 가능한 새 profile 이름이 아니다. 기존에 동일 계획 run이 완료/진행 중이면 그 ID를 재사용·연결하고 중복 시작하지 않는다.

동일 recipe에서 Student seed와 허용된 식별/경로 metadata만 달라진다. 중간 결과로 서버별 β·LR·loss를 바꾸지 않는다. 20개 결과 후 자동으로 무한 seed 확장하지 않고 전체 분포·HQNR 통과 수·공동목표 통과 수를 보고한다. 이 신규 seed는 T0부터 독립적으로 재학습한 반복이나 새로운 미사용 test set을 뜻하지 않는다.

### 10.5 선행 평가가 본 방법을 자동 교체하지 않는 이유

| 선행 평가·분석 결과 | 이번 운영의 기본 처리 |
|---|---|
| NOA 열·paired metric이 정상이고 코드 오류 없음 | 각 서버 예정 순서에 따라 본 실험 복귀 |
| NOA RR 개선·FR 악화 | 두 목표의 trade-off로 기록; 본 방법 자동 변경 없음 |
| NOA/consensus가 반복적으로 두 목표에서 유리 | 근거 보고 후 별도 inference-policy 개정 결정; 현재 실행 중 학습 유지 |
| 명확한 forward/evaluator 구현 오류 | 영향받는 기록을 blocked/invalidate하고 수정·재평가; seed로 오류 보상 금지 |

“분석이 끝났다”와 “새 방법을 채택했다”는 다르다. s1은 지정 분석을 끝내고 결과를 보존한 뒤 복귀하며, 가설이 맞을 때까지 분석을 무한 연장하지 않는다.

---

## 11. 구현 인계와 필수 회귀 검사

### 11.1 구현 범위

변경은 평가 분기·평가 작업 목록·로컬 phase orchestration·gspread 확장에 한정한다. trainer loss·Teacher·q cache·기존 Student checkpoint·본 학습 optimizer를 변경하지 않는다.

기존 확인 경로는 아래와 같으며 실제 설치본과 training-start revision을 연결한다.

```text
pa/model.py                         # A_ON / 기존 zero warp 분기
pa/aligner.py, pa/warp.py, pa/offset.py
kdv/teacher_assets.py                # strict skeleton/load 자산 연결
model/pancrafter_paper.py            # 기존 backbone API
feeders/feeder.py                    # 실제 데이터/전처리
tools/metrics/                     # 기존 공식 RR/FR metric 구현
tools/eval_fr_paperset.py
tools/qrecon24_select.py
tools/qrecon24_postrun.py
tools/gen_pakd50_configs.py
tools/qrecon24_switch.sh
tools/qrecon24_waiter.sh
gspread/gspread_upload.py           # NOA 컬럼 registry와 부분 업데이트
```

새 도구가 필요하면 기능별로 나누되, `--noa`, `--all-server-eval`, `--resume-main` 같은 옵션이 이미 있다고 가정하지 않는다. 이 문서의 새 state/manifest/path는 구현할 계약이며 원격 명령 실행 완료 보고가 아니다.

### 11.2 단계별 필수 검사

| ID | 검사 | 통과 기준 |
|---|---|---|
| V01 | Student strict load | full checkpoint key·shape 일치, 원 자산 불변 |
| V02 | NOA 호출 추적 | aligner·warp 호출 각각 0회 |
| V03 | ZERO 경로 | A 0회, sampler 사용; NOA와 별도 mode |
| V04 | residual/API | M 한 번 추가, 원 LRMS API 유지, PAN intensity 불변 |
| V05 | paired identity | RR/FR·ON/NOA 동일 checkpoint 및 metric 정의 |
| V06 | 무학습 보장 | optimizer/backward 없음, 전후 state hash 동일 |
| V07 | cache 분리 | ON/NOA/consensus를 mode identity로 구분 |
| V08 | subset 금지 | 지정 RR/FR scene 전체와 기존 metric crop 유지 |
| V09 | 목표 판정 | 동일 mode·checkpoint의 H≥.9585 및 E<2.040, 원값으로 판정 |
| V10 | 로컬 소유권 | s_i가 자기 탭/자기 source run만 갱신 |
| V11 | Sheet 보존 | 기존 값·Date·Train(h)·NOA 열 재업로드 시 소실 없음 |
| V12 | restart/멱등 | 중간 실패 후 성공한 RR/FR·업로드를 중복 생성하지 않음 |
| V13 | s1 순서 | 자기 전수 평가 완료 뒤에만 전체 size/probe 분석 시작 |
| V14 | s2–s5 독립 복귀 | 다른 서버 평가·s1 분석 미완료라도 자기 완료 시 MAIN_RESUME |
| V15 | 큐 보존 | 완료 run·held run 부활 없음, 원 resume 위치·revision 유지 |
| V16 | 학습 경로 복원 | NOA flag/sampler 우회가 본 학습에 남지 않음 |
| V17 | recipe lock | 평가 완료가 profile lock을 자동 생성/변경하지 않음 |
| V18 | consensus 정의 | 64 grid·성분별 중앙값·짝수 평균·전체 PAN 단일 warp |

V13/V14는 runner 모의 실행으로 확인한다. 예를 들어 s1이 분석 중이고 s2가 업로드까지 완료한 상황에서 s2의 다음 승인된 학습 case가 정상 선택되어야 한다. s2의 main task가 recipe_lock을 요구하면 `MAIN_WAIT_RECIPE_LOCK`으로 가되, `WAIT_S1_AUDIT`로 잘못 묶이지 않아야 한다.

### 11.3 평가 산출물

각 서버는 `cohort_manifest`, `phase_state`, `resume_manifest`, run별 paired metrics/identity, `upload_receipts`, `local_eval_summary.md`를 남긴다. 경로와 파일명은 프로젝트 규약에 맞출 수 있지만 의미와 key는 전 서버에서 같아야 한다.

s1에는 추가로 size/probe/consensus raw CSV·JSON 및 `s1_aligner_analysis.md`가 필요하다. 전 서버 통합 요약은 각 서버의 작은 summary/metrics만 모아 만들 수 있으며, **그 통합 보고서의 완료는 s2–s5 복귀 전제조건이 아니다.**

---

## 12. 시간·완료 건수·운영 보고

### 12.1 평가시간은 로컬 실측으로 산정

각 서버의 처음 2–3개 paired 평가에서 A_ON RR/FR, NOA RR/FR, export·metric·upload 시간을 구분해 잰다. 검증된 cache가 있으면 추론 재실행 시간과 cache validation 시간을 분리한다. 50K Train(h)를 평가시간의 대용으로 사용하지 않는다. [E1 §10]

서버 i의 평가시간을 E_i, s1 추가 분석을 D_1로 두면 안전 경계 이후 로컬 복귀 시점의 대용은 다음과 같다.

```text
s1: E_1 + D_1
s2: E_2
s3: E_3
s4: E_4
s5: E_5
```

모든 진단이 끝나는 시간의 대용은 max(E_1+D_1,E_2,E_3,E_4,E_5)이지 서버 시간을 합한 값이 아니다. 그동안 먼저 끝난 서버들은 이미 본 실험을 진행할 수 있다. 현재 실행 중 run의 잔여, setup, 장애, 미완 업로드 비용은 실제 상태에 따라 별도 반영한다.

평가·분석 비용은 진단 ledger에 남긴다. 기존 QRC24의 서버당 누적 신규 학습 24시간 정책을 이번 평가시간으로 채웠다고 계산하지 않으며, 새 문서 때문에 학습시간을 0부터 다시 세지도 않는다. [R2 §7]

### 12.2 로컬 완료 보고 표

각 서버는 다음 숫자를 별도로 보고한다.

| 항목 | 정의 |
|---|---|
| N_identified | snapshot에서 찾은 고유 평가 레코드 |
| N_required | 정의/자산 확인 후 필수 평가 대상 |
| N_on_rr, N_on_fr | paired ON의 RR·FR 완료 |
| N_noa_rr, N_noa_fr | NOA의 RR·FR 완료 |
| N_pair_verified | 동일 identity 비교 검증 완료 |
| N_uploaded | 자기 탭 업로드 확인 완료 |
| N_blocked, N_failed | 미완료 사유가 있는 대상 |
| N_waived | 명시적 사유·승인으로 제외한 대상 |
| phase / next_main_case | 현재 phase와 복귀할 실제 미완 작업 |

완전 완료는 N_required 전부의 RR/FR·pair·upload가 끝난 상태다. N_identified와 N_required의 차이를 숨기지 않는다. s1은 추가 분석 panel 완료 상태를 함께 보고한다. s2–s5에는 분석 panel 완료 칸을 의존조건으로 두지 않는다.

---

## 부록 A. 서버별 적용용 manifest 예시

아래 YAML은 **스키마 예시**이며 실제 서버 상태나 실행 가능한 최종 config가 아니다. `null`은 적용 시 채울 값이고, 존재하지 않는 checkpoint/lock/queue를 가리키는 가상 값으로 바꾸면 안 된다.

```yaml
protocol_id: PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918
method_id: qrecon_continuous_v1
source_train_server: null          # s1, s2, s3, s4, s5 중 실제 소유 서버
eval_server: null                  # 기본: source_train_server와 같음
phase: pending
cohort:
  id: null
  captured_at: null
  completed_run_keys: []
  pending_current_run_id: null
  records: []
  include_future_main_runs: false
student_eval:
  modes: [A_ON, A_BYPASS_RAW]
  checkpoint_policy: source_sheet_same_checkpoint
  strict_load: true
  full_frame_unet: true
  gt_used_for_shift_selection: false
  cache_identity_required: true
s1_analysis:
  enabled: null                    # s1만 true; s2-s5는 false
  after_local_student_eval_upload: true
  modes: [A_ON, A_ZERO_WARP, A_BYPASS_RAW, A_CROP64_MED]
  representative_asset_keys: []
sheet:
  spreadsheet_id: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
  source_tab: null
  schema_revision: noa_columns_v2
  write_only_owned_tab: true
  preserve_legacy_columns: true
  transport: existing_gspread_uploader
resume:
  approved_main_queue_revision: null
  captured_resume_queue_path: null
  captured_resume_queue_sha256: null
  next_unfinished_case: null
  main_worktree: null
  main_recipe_lock_id: null
  wait_for_other_servers_eval: false
  wait_for_s1_analysis: false       # s2-s5용: false; s1 자체 순서는 phase로 강제
  preserve_existing_main_lock_requirement: true
execution:
  allow_new_main_training_during_local_eval: false
  preempt_running_training: false
  restore_main_forward_mode: A_ON
```

`resume.wait_for_s1_analysis=false`는 s1이 자기 분석을 건너뛴다는 뜻이 아니다. s1은 자체 phase graph에서 분석 완료 후 복귀한다. 필드 이름은 **타 서버 의존성**을 나타낸다.

---

## 부록 B. source 자료와 원문 보존

| source | 참조 위치 | 계승/변경 |
|---|---|---|
| [E1] §2–4, §7–9 | 정확한 우회·checkpoint·metric·Sheet | 계승. writer 소유권과 paired host는 로컬 서버별로 변경 |
| [E1] §5–6, §10 | s1 분석·중앙 평가·실행 순서 | §1–2, §7의 최신 서버별 순서로 교체 |
| [H] §2–4 | Method·Teacher 계보·q 해석 | §8의 수식·설명으로 통합 |
| [H] §12.1–12.4 | 논문형 순서·문서 정오·주장 범위 | §8–9에 출처를 밝혀 반영 |
| [R2] §2–6 | 새 selector·하이퍼파라미터 축소·lock·seed20 | 본 실험 복귀 규칙으로 유지 |
| [R2] §7–8 | 운영시간·큐·uploader/검사 | 평가시간과 학습시간을 분리하여 연결 |
| [P] 원문 | 별도 파일 미확보 | 원문 문장·그림·절 전체의 완전 병합은 미확인 |

기존 source 파일은 수정하지 않는다. 이후 [P] 원문을 확보하면 그 문장·절과 §8–9를 **명시적으로 대조**할 수 있으나, 현재 문서를 원문 그대로라고 소급 표시하지 않는다.

저장소·Sheet의 논리 식별자:

```text
Repository: https://github.com/hojunking/PAN-Crafter-repro
Spreadsheet title: pan-cvpr27
Spreadsheet ID: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
```


### 이번 통합에 사용한 로컬 source bytes

다음은 이번 작업에서 읽은 파일 자체의 SHA256이다. 서버 실행 commit·checkpoint hash가 아니다. 원본 파일은 수정하지 않았다.

| 파일 | SHA256 |
|---|---|
| `PAN_S1_Aligner_Validation_NOA_RR_FR_Protocol_v1.md` | `127aded721be7472d422ea89a34030517c80e6be47965ca273f59aa74edb1e48` |
| `PAN_New_Session_Handoff_2026-09-17.md` | `5ecfd766490a1590d1f84355d2554401d0d9c70e88fff6af82c7339713846103` |
| `PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md` | `ff1737dca9d7c2eadc22a8e13fce161cde222b4c6b905ef13c161e7ea4600996` |


---

## 부록 C. 각 서버 실행 담당자에게 전달할 요약

```text
이 문서의 학습 Method는 QRECON24 / qrecon_continuous_v1이다.
수식·가중치·Teacher·기존 Student checkpoint를 바꾸지 않는다.

s1:
  현재 run을 안전한 경계까지 완료한다.
  s1 소유 기존 Student 전체를 같은 checkpoint로 A_ON/NOA RR·FR 평가한다.
  s1 원본 탭의 새 NOA 열에 gspread로 기록하고 완료를 검증한다.
  그 뒤 T0 및 대표 s1 Student의 크기/probe/consensus 분석을 수행한다.
  분석 기록을 남긴 뒤 s1의 승인된 본 실험 큐로 복귀한다.

s2–s5:
  각자 현재 run을 안전한 경계까지 완료한다.
  각자 자기 Student 전체를 같은 checkpoint로 A_ON/NOA RR·FR 평가한다.
  각자 자기 탭의 새 NOA 열에 gspread로 기록하고 완료를 검증한다.
  자기 평가가 끝나면 즉시 자기 본 실험 큐로 복귀한다.
  다른 서버 평가나 s1 분석 완료를 복귀 조건으로 걸지 않는다.

NOA = A_BYPASS_RAW:
  aligner 호출 없음, warp/sampler 호출 없음.
  같은 U-Net 가중치에 원 PAN·원 LRMS를 주고 기존 MS base를 한 번 더한다.
  aligner_enabled=False 또는 delta_override=0만 준 zero-warp와 구분한다.

같은 checkpoint·같은 모드의 HQNR/ERGAS로 판정한다.
기존 RR/FR·Date·Train(h)는 보존한다.
NOA 개선이 나와도 정상 학습/추론 정책을 자동 교체하지 않는다.

본 실험은 기존 승인 큐 및 Narrow R2 의도를 이어간다.
최종 seed-only의 공통 recipe_lock은 원래 조건대로 유지하되,
새 전 서버 평가 장벽이나 s1 분석 대기 조건은 만들지 않는다.
```

**인계 종료 상태:** 운영은 전 서버 로컬 Student 평가로 분산했고, s1에만 전수 평가 후 분석 단계를 둔다. 각 서버의 독립 복귀와 기존 본 실험의 공통 설정 보호를 구분했다. Method 핵심 수식은 handoff에서 통합했으며 [P] 원문 전체 병합은 미확인이다. 실제 코드 배포·서버 실행·Sheet 새 열 반영은 이 문서 작성의 수행 범위가 아니다.
