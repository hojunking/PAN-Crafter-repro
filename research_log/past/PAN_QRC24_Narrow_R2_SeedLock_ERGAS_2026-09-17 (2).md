# PAN QRECON24 — Narrow R2: 마지막 β 비교 후 단일 설정·seed 반복

작성일: 2026-09-17  
제안 revision: `QRC24_NARROW_R2_20260917`  
제안 selector: `HQNR9585_ERGAS2040_v2`  
상태: **계획 문서. 저장소 수정, 서버 큐 적용, 학습 실행, Sheet 수정은 수행하지 않았다.**

## 0. 결정 요약

**넓은 grid를 종료하고, G23을 잠정 기준으로 마지막 β 비교만 닫은 뒤 한 설정으로 고정한다.**

- 목표: 같은 checkpoint의 **raw-original HQNR ≥0.9585**, 이어서 **공식 RR ERGAS <2.040**.
- 최종 checkpoint 선택은 HQNR 하한 통과 집합 안에서 **ERGAS → SCC → PSNR → SAM → Q8 → SSIM** 순이다. 기존 SCC 우선 규칙을 이번 사용자 결정으로 명시적으로 변경한다.
- 당장 고정할 기본 축: Teacher T0, Student W104D121, q 정의, gradient routing, λE=.002, α=1, U peak LR=1e−4, rA=.03, 50K fresh training.
- 마지막 활성 하이퍼파라미터 축: **β=.1(G23) / β=.2(B20A03)** 두 값뿐.
- 기존 G23을 재사용하고 **B20A03 최대 4개만 추가**하면 기존 s3·4321을 포함해 최대 5개 seed의 같은 서버 내 대응 비교가 생긴다.
- s4 G23·1234는 **서버 간 재현 확인 1개**로 남긴다. 독립 seed로 중복 집계하지 않는다.
- 이후 **C* 하나를 동결하고 신규 seed 20개**를 실행한다. 서버마다 서로 다른 설정을 선택하지 않는다.
- 정상 기준 편성: **수렴 비교·서버 재현 최대 5run + 단일 설정 seed 20run = 최대 25run**. 기존 계획에서 이미 시작된 보류 대상 run의 원 정의 완료 비용은 별도다.
- 이 문서는 **목표 checkpoint 확보를 위한 제한된 탐색 정책**이다. rA=.03의 평균적 최적성, β=.2의 보편적 우위, q의 모든 성분 효용을 확정한 것이 아니다.

## 1. 근거와 이번 재조회 범위

### 1.1 출처 구분

이전 상세 보고서 `PAN_QRC24_Date17_Review_2026-09-17.md`와 현재 대화의 handoff를 읽고, 설계 전에 연결된 Google Drive에서 Sheet를 다시 내보냈다.

- Spreadsheet: `pan-cvpr27`
- ID: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`
- 이번 export SHA256: `2c8b0959ef473b8f9750a3aa4a202031112b8fb352fda17f3aa20730b08f3f5e`
- **이번 export의 누적 QRC24는 고유 66run**(s1 11 / s2 10 / s3 18 / s4 17 / s5 10).
- 이 숫자는 이전 보고서의 62run snapshot보다 4개 많다. 물리 행 82개를 원본 run ID로 묶었으며, 중복 행의 품질 지표 일치를 확인했다.
- export는 실시간 조회로 확보한 한 시점의 snapshot이다. 문서 작성 이후 결과가 더 생길 수 있다.
- 행의 Date·Train(h)를 실제 서버 프로세스의 시작/종료 시각·잔여시간으로 해석하지 않는다.
- 원자료: `PAN_QRC24_Narrow_R2_Source66_2026-09-17.csv`. 각 레코드에 원본 Sheet 행 링크를 남겼다.
- 실제 서버의 running/queued/postprocessing/started_interrupted 상태와 start manifest 원본을 이번 작업에서 직접 읽은 것은 아니다.

추가 4개:

| 서버/행 | Profile | Seed | Raw HQNR | ERGAS |
|---|---|---:|---:|---:|
| s1/83 | A_UNIF | 3407 | 0.9541 | 2.0709 |
| s2/96 | E_UNIF | 777 | 0.9553 | 2.0881 |
| s3/127 | G11 | 4321 | 0.9552 | 2.0667 |
| s4/55 | H33 | 3407 | 0.9550 | 2.0717 |

이 네 선택 행에는 공동목표 달성 결과가 없다. 다만 별도 checkpoint 전체를 재선택한 결과는 아직 아니다.

### 1.2 왜 후보를 이렇게 줄이는가

1. s1 G23·1234는 고정밀 HQNR **0.959148**, 공식 ERGAS 약 **2.0712**, step **31310**인 현재 확인된 하한 통과 자산이다. 단일 seed의 좋은 자산이지 안정적인 전 seed 개선은 아니다.
2. s1·1234의 rA=.03에서 G23은 A03_UNIF/A03_SHUF보다 HQNR +.0013/+.0022이고 RR 여섯 지표가 모두 좋았다. 따라서 q를 그대로 둔 경로에서 출발할 근거는 있다.
3. G23의 다른 seed 반복은 엇갈렸다. rA=.03 고정은 모든 후보 중 평균 성능이 가장 좋다는 결론이 아니라 **이미 하한을 통과한 조합을 중심으로 탐색을 끝내기 위한 운영 선택**이다.
4. β=.2의 큰 grid를 더 돌릴 근거는 부족하다. 다만 rA=.03에서 β=.1/.2를 직접 비교한 기존 block은 s3·4321 하나다. 여기서 B20A03은 G23보다 ERGAS −.0045, HQNR −.0010이었다. β=.2를 무조건 승격하거나 바로 폐기하기보다 이미 있는 G23의 짝만 최소 비용으로 채운다.
5. 이번 66개 **Sheet 선택 행**의 최저 ERGAS는 **2.0544**(s3 G12·2026, HQNR .9501)다. 이것을 “HQNR을 지키면서 2.04 아래로 갈 수 있음”의 근거로 사용하지 않는다.
6. G23 목표 자산에서 2.040까지 필요한 감소는 약 **.0312, 약 1.5%**다. 현재 기록만으로 seed 변경이 이 감소와 HQNR 하한을 동시에 달성할 것이라고 보장할 수 없다.

출처:
- 이전 분석 보고서 §3–8 및 부록 A.
- 이번 Sheet export의 원자료.
- [후속 구현·감사 기록 §4](https://github.com/hojunking/PAN-Crafter-repro/blob/main/research_log/2026-09-17_qrecon24-adjustment-r1-implementation.md): G23 official selector, 자산 보존, s1 factor1 감사.
- 이 설계는 외부 논문 연구를 새로 수행한 것이 아니라 제공된 실험 자료에 대한 **새로운 편성·의사결정 제안**이다.

## 2. 새 목표와 selector — 학습 loss와 구분

### 2.1 목표 정의

각 run r의 기존 후보 격자를 K_r라 하고, checkpoint k의 raw HQNR와 공식 RR ERGAS를 H(r,k), E(r,k)로 둔다.

\[
\mathcal F_r=\{k\in\mathcal K_r:H(r,k)\ge0.9585\}.
\]

\[
k_r^*=\arg\min_{k\in\mathcal F_r}
(E(r,k),-\mathrm{SCC}(r,k),-\mathrm{PSNR}(r,k),
 \mathrm{SAM}(r,k),-\mathrm{Q8}(r,k),-\mathrm{SSIM}(r,k),-H(r,k),k).
\]

마지막 step k는 모든 수치가 정확히 같은 경우의 결정적 tie-breaker다. 별도의 epsilon tie band는 도입하지 않는다.

\[
\mathrm{joint\_pass}(r)=
[H(r,k_r^*)\ge0.9585]\land[E(r,k_r^*)<2.040].
\]

**HQNR 우선은 하한을 지키는 것이다.** 하한 통과 후 HQNR을 끝없이 더 높이는 순서가 아니다.

| 예시(가상 수치) | 판단 |
|---|---|
| H=.9586, E=2.039 vs H=.9600, E=2.070 | 첫 checkpoint 선택: 둘 다 H 하한을 통과하므로 E 우선 |
| H=.9579, E=2.030 | 공동목표 불합격: 낮은 E가 H 미달을 보상하지 못함 |
| 표시 E=2.040, 고정밀 E 미확인 | strict pass 판정 보류 |
| 고정밀 H=.958499 | 하한 미달. 반올림으로 통과 처리하지 않음 |

E<2.040이 아직 없더라도 H 통과 집합의 최저 E를 개선 후보로 보존한다. “joint_pass=false”는 해당 run을 삭제하거나 실험 실패로 숨길 이유가 아니다.

### 2.2 실제 코드와 필요한 변경

현재 조회한 `tools/qrecon24_select.py`는 `HQNR9585_RR_v1`이고 ORDER의 첫 항이 SCC다. ERGAS 우선 실행 옵션이 이미 있다고 가정하지 않는다.

이번 제안은 다음과 같다.

- 기존 v1 selector 결과와 legacy best_hqnr/raw-max를 보존한다.
- 새 selector ID `HQNR9585_ERGAS2040_v2`를 추가한다. ORDER만 새 목적에 맞게 정의하며 trainer/criterion에는 이 변경을 전달하지 않는다.
- 새 결과 파일:
  `results/qrecon24_target_selection_HQNR9585_ERGAS2040_v2.json`
- 공식 RR가 필요한 모든 H-eligible checkpoint에서 완료되어야 해당 run의 최종 k*를 확정한다. 일부 누락이면 `incomplete_official_rr`, joint_pass/최종 선택을 확정하지 않는다.
- H-eligible가 없으면 `no_eligible`를 기록하고 legacy raw-max를 진단 값으로 따로 보존한다.
- v1이 official이라는 이유로 v2 평가·재정렬을 건너뛰면 안 된다. postrun/backlog의 완료 판정은 **run ID + selector ID + candidate-grid identity** 기준으로 바꾼다.
- 기존 공식 RR 캐시는 checkpoint/config/data/evaluator hash가 일치할 때 재사용한다. 선택 순서만 바뀌었다는 이유로 GPU 추론을 반복하지 않는다.
- 원래 GRID1010_50K_v1(기존 50개 후보)을 유지한다. 특정 run만 더 촘촘한 후보를 추가해 선택 기회를 늘리지 않는다.
- 비적격 run의 RR 요약이 필요하면 exact50K 등 공통 checkpoint를 별도 평가·표시한다. proxy RR와 공식 RR를 섞어 candidate 순위를 만들지 않는다.
- checkpoint가 유실된 경우 불완전으로 남긴다. 재학습을 자동으로 요구하거나 없는 값을 채우지 않는다.

확인한 코드:
[tools/qrecon24_select.py](https://github.com/hojunking/PAN-Crafter-repro/blob/main/tools/qrecon24_select.py)

### 2.3 새 학습보다 먼저 할 일

기존 66run을 원본 ID로 중복 제거하고 새 selector로 다시 읽는다. FR 로그로 H eligibility를 먼저 거르고, 적격 checkpoint의 공식 RR만 재사용/보완한다.

특히 G23·1234는 기존 적격 후보가 2개였다. ERGAS 우선으로 바꾸면 선택 step이 달라질 수 있지만, 실제 후보별 수치를 읽기 전 더 좋아진다고 단정하지 않는다. s4 H31의 고정밀 하한 판정도 이때 처리한다.

**재선택으로 얻은 개선은 새 학습의 효과가 아니다.** seed 수도 증가하지 않는다.

두 후보 이외의 기존 설정에서 공동목표 자산이 발견되면 그것도 보존한다. 그 발견을 이유로 이 문서의 고정 설정을 조용히 바꾸지 않고 lock 결정에 명시한다. 아래 2후보 종료안은 현재 확인된 결과를 기준으로 한 기본 경로다.

## 3. 학습 방법과 고정할 축

| 항목 | 고정값/정책 |
|---|---|
| Method | qrecon_continuous_v1 |
| Teacher | T0 W112D123, 동일 selected checkpoint의 A/U, frozen |
| Student | W104D121 + PAN global aligner |
| A 초기화 | T0 A 전체 state의 별도 clone |
| U 초기화 | fresh; 비교 단계는 대응 G23의 초기값 identity를 확인 |
| q | qref/(qref+q_T), 분자1; raw q cache/실제 augmentation view 유지 |
| q_ref / τ_R | .3276133416220546 / .012463942170143127 |
| U gradient | H + K + λE·s_q·E |
| A gradient | s_q·H만; U/A를 각각 미분 |
| Edge | GT signed Scharr, 기존 고정 영역·평균·계수 해석 |
| λE | .002 |
| α | 1.0 |
| rA | .03 |
| U peak LR / A peak LR | 1e−4 / 3e−6 |
| β | 마지막 비교 .1 또는 .2 → 이후 하나 |
| 학습 | fresh50K optimizer updates, batch48, 기존 AdamW·warmup100·cosine |
| 제외 | Student offset/jitter, soft/edge→A 직접 gradient, 새로운 loss/정규화/head |
| 추론 | 동일 Student checkpoint A/U; Teacher·GT·q probe 불필요 |

AdamW β1/β2·ε·minimum LR·precision·clipping·accumulation·augmentation·norm·crop·sampler·seed 적용 순서는 검증된 resolved config를 그대로 계승한다. 이 문서가 명시하지 않은 값을 라이브러리 default로 채우지 않는다.

rA=.03을 사용한다고 shift 값에 .03을 곱하지 않는다. U-Net을 이어 학습하는 FT/tail은 이번 계획에 없다.

G22/H22/L100은 **기존 비교 기준으로 보존**하되 새 active 후보에는 넣지 않는다. λE grid, α grid, U LR grid, rA grid, H13/H33의 확장도 중단한다.

## 4. 마지막 β 비교 — G23을 다시 학습하지 않고 짝만 채운다

### 4.1 비교 대상

| 대응 block | 이미 등록된 G23 | B20A03 처리 | 독립 Student seed |
|---|---|---|---|
| s1·1234 | 재사용 | 신규 1run | 1234 |
| s1·3407 | 재사용 | 신규 1run | 3407 |
| s2·777 | 재사용, **버전/초기값 감사 선행** | 신규 1run | 777 |
| s3·2026 | 재사용, 초기값 identity 확인 | 신규 1run | 2026 |
| s3·4321 | 재사용 | 이미 등록, 재사용 | 4321 |

따라서 **최대 4개 신규 학습으로 최대 5-seed의 G23↔B20A03 비교**를 만든다. 두 후보가 같은 λE/rA/α/U LR을 사용하므로 남은 차이는 β다.

다만 비교의 유효성은 seed 숫자만 같다는 것으로 성립하지 않는다. T0/cue/data/augmentation/초기 U tensor/학습 정의/evaluator가 연결되어야 한다. s2의 672ccf541ea7 실행은 과거 보고서에서 완전한 동등성 감사가 남아 있었다.

**대응의 정의·초기값을 검증하지 못한 block은 자동 재실행으로 예산을 늘리지 않고 짝 비교에서 보류한다.** 해당 block을 제외할 때 두 후보 모두 같은 비교집합에서 제외한다. 실제로 완료된 결과 자체는 남긴다. 최대 4개라는 수는 유효성을 검증한 block에만 해당한다.

기존 G23와 새 B20A03이 동일한 full config hash를 가져야 하는 것은 아니다(β와 run metadata가 다름). 차이 집합이 허용된 β 및 식별 metadata로 제한되는지 확인한다.

### 4.2 s4 서버 간 재현 확인

기존 계획의 **s4 G23·1234 v2 1개**를 유지한다. s1 G23·1234와 비교해 서버 환경·재현성을 확인한다.

- 새로운 독립 seed로 세지 않는다.
- 기본 β 후보 순위에는 s1·1234 block을 사용한다. 동일 seed를 두 번 가중하지 않는다.
- 결과가 다르다는 이유만으로 방법이 달라졌다고 단정하지 않는다. 실제 start config·초기값·data order·환경 identity를 확인한다.
- s4 B20A03·1234까지 추가로 학습해 같은 seed를 중복 비교할 필요는 없다.

### 4.3 동결 결정을 미루지 않는 규칙

비교에 유효한 공통 seed 집합에서 각 후보 c에 대해 다음을 기록한다.

- n_joint(c): H≥.9585 및 E<2.040을 같은 checkpoint에서 달성한 seed 수.
- n_H(c): H-eligible checkpoint가 존재하는 seed 수.
- med_E_H(c): 각 H 통과 seed에서 새 selector로 선택한 E의 중앙값.
- H raw-max, exact50K, late6의 seed별 분포는 보조 보고. 선택 목표를 late6로 사후 교체하지 않는다.

**공동목표를 충족하는 결과는 애초에 H 하한을 통과한 결과만 센다.** 동결 정책은
1) n_joint 큰 후보 → 2) 동률이면 n_H 큰 후보 → 3) 동률이고 적격 값이 있으면 med_E_H 낮은 후보 → 4) 여전히 같거나 둘 다 적격 결과가 없으면 G23이다.

이는 작은 개발 표본에서 탐색을 종료하기 위한 사전 의사결정 정책이지 통계적 유의성 판정이 아니다. 설정 선택과 별도로 “목표 자산 확보 여부”와 “반복 안정성”을 보고한다.

현재 잠정 동결값은 G23이다. 마지막 비교가 B20A03을 지지하면 β만 .2로 고정한다. **β=.15, rA=.02 등 중간값이나 세 번째 후보를 추가하지 않는다.**

공식 RR가 미완료면 미완료 상태를 해소해야 하며, 이를 음성 결과로 대신 넣지 않는다. 유실로 비교 불가한 block은 명시적으로 제외한 뒤 남은 공통집합과 G23 fallback을 사용한다.

## 5. 기존 ADJ-R1 큐 정리

이번 export 기준 81개 기존 편성 중 66개가 등록되어 **Sheet 미등록 15개**다. 아래는 서버 상태 확인 전의 분류다.

| 서버 | 남기는 것 | 미시작일 때만 보류 |
|---|---|---|
| s1 | 없음 | A_FREEZE/A_SHUF/G21·3407: 3개 |
| s2 | 없음 | E_SHUF/ALL_UNIF·777: 2개 |
| s3 | 없음 | G31/G33·4321: 2개 |
| s4 | G23·1234: 서버 간 재현 1개 | B20A03·1234: 1개 |
| s5 | 없음 | G23/H23/B20A03·9091 및 1103: 6개 |
| 합계 | 1개 | 14개 |

s1 A_UNIF·3407, s2 E_UNIF·777, s3 G11·4321, s4 H33·3407은 이미 등록된 신규 4개이므로 다시 편성하지 않는다.

여기서 “보류”는 실패 판정이 아니라 **성능 설정 고정에 자원을 집중하기 위한 우선순위 변경**이다. q-edge의 독립 기여를 논문에서 주장하는 데 필요한 E_SHUF/ALL_UNIF 등 증거가 완결된 것은 아니다. 논문용 ablation은 성능 수렴 큐와 분리한 후속 승인 예산으로 둔다. 최종 seed 단계에 몰래 섞지 않는다.

### 시작된 run 보호

미등록 run은 실제로 학습 중이거나 후처리 중일 수 있다. 실제 상태를 읽고 다음을 적용한다.

- 이미 완료: 원래 ID로 보존하고 중복 실행하지 않는다.
- 실행 중: 기존 정의로 끝낸다. 좋은/나쁜 초기 점수로 조기 중단하지 않는다.
- started_interrupted/checkpoint 존재: 기존 안전 정책에 따라 원 정의로 완료한다.
- 완전 미시작: 위 표의 보류 적용.
- 대조 규칙에 맞지 않는 carry-over 결과는 참고 자료이며, seed-only 코호트의 일부로 바꾸어 세지 않는다.

기준 편성은 81−14+4+20=91개다. 현재 66개 이후 **최대 25개**를 계획한다. 이는 “기존 15개에 새 25개를 전부 더한다”는 뜻이 아니다. 보류 대상 중 이미 시작된 run의 잔여 완료가 있으면 그 비용·건수는 별도로 표시한다.

## 6. 최종 단계 — C* 하나, 신규 seed 20개

### 6.1 recipe_lock

최종 학습을 시작하기 전에 `QRC24_LOCK_V1_20260917`에 다음을 기록하고 불변으로 둔다.

- 선택 profile G23 또는 B20A03, λE/rA/α/β/U LR.
- 전체 학습 정의·U/A gradient routing·50K·batch·optimizer/scheduler 상세.
- Teacher/cue/data/evaluator identity 및 검증 release.
- seed 적용 항목과 초기화/augmentation 정책.
- candidate grid와 selector `HQNR9585_ERGAS2040_v2`.
- 20개 신규 seed 및 server 배정.
- 선택 근거, 비교에 포함/제외한 block과 사유.

full run config hash는 seed·run path 때문에 서로 다를 수 있다. 별도의 recipe hash는 seed와 허용된 식별/물리 경로 metadata만 제거한 학습 정의에 대해 계산한다. 모델·loss·학습률·데이터 전처리를 hash 제외 목록에 넣지 않는다.

**lock 파일이 없으면 seed-stage config가 실행 가능 상태가 되어서는 안 된다.** 아래 C*는 runnable profile 이름이 아니라 선택 후 해소할 표기다.

### 6.2 사전 지정 seed

| 서버 | 신규 seed | 설정 | run 수 |
|---|---|---|---:|
| s1 | 41001 / 41006 / 41011 / 41016 | C* 하나 | 4 |
| s2 | 41002 / 41007 / 41012 / 41017 | C* 하나 | 4 |
| s3 | 41003 / 41008 / 41013 / 41018 | C* 하나 | 4 |
| s4 | 41004 / 41009 / 41014 / 41019 | C* 하나 | 4 |
| s5 | 41005 / 41010 / 41015 / 41020 | C* 하나 | 4 |

같은 seed 숫자를 서버만 바꾸어 20개로 채우지 않는다. 다섯 서버의 계산 환경 차이는 별도 metadata로 남기고 서버별 결과도 보고한다.

두 묶음으로 실행할 수 있다: 각 서버 앞의 2개 seed(총10) → 뒤의 2개(총10). 그러나 **10개 중간 결과로 β나 loss를 다시 바꾸지 않는다.** 기본 계획은 지정한 20개 전체 완료다. 성공 seed가 빨리 나와도 나머지 기록을 숨기지 않는다.

### 6.3 보고와 종료

목표 checkpoint는 공식 v2 selector로 보존한다. 아래를 동시에 보고한다.

| 보고 항목 | 의미 |
|---|---|
| Best target asset | H 하한을 지킨 가장 낮은 공식 E의 checkpoint |
| n_H / 20 | 고정 설정의 H 통과 seed 수 |
| n_joint / 20 | 두 목표를 같은 checkpoint에서 달성한 seed 수 |
| seed별 H/E 및 평균·중앙값·표준편차 | 가장 좋은 seed만 제시하지 않는 반복 결과 |
| 조건부 E 집계 | H 통과 seed에서만 집계했음을 분명히 함 |
| exact50K / late6 | best와 분리한 안정성 진단 |
| 서버별 요약 | host와 seed 변동을 무조건 하나로 귀속하지 않음 |

H 미달 seed를 통계에서 삭제하지 않는다. 같은 seed의 50개 checkpoint를 50번의 독립 성공으로 세지 않는다.

20개 후 공동목표가 없으면 자동 seed 무한 확장을 하지 않는다. 결과를 종결해 “고정 설정의 이번 seed 코호트에서 미달”로 보고하고 다음 예산·접근은 별도 결정한다. 이 단계 안에서 loss/Teacher/architecture/FT를 자동 변경하지 않는다.

FR20과 RR 평가 세트가 개발·선택에 반복 사용되었다는 점을 명시한다. 신규 Student seed는 새로운 미사용 test set이나 독립 Teacher 학습 반복을 의미하지 않는다.

## 7. 서버별 실행 순서와 시간 대용

### 7.1 순서

| 서버 | lock 전 | lock 후 |
|---|---|---|
| s1 | 기존 G23의 v2 재선택·보존 → B20A03·1234 → B20A03·3407 | C*의 지정 seed 4개 |
| s2 | release/초기값 감사 → 유효할 때 B20A03·777 | C*의 지정 seed 4개 |
| s3 | 기존 s3·4321 pair 재선택 → 초기값 확인 → B20A03·2026 | C*의 지정 seed 4개 |
| s4 | G23·1234 기존 v2 ID로 서버 간 재현 | C*의 지정 seed 4개 |
| s5 | 기존 run v2 selector·자산/환경 준비; 남은 넓은 grid 미시작분 보류 | C*의 지정 seed 4개 |

각 서버가 자신의 비교가 끝났다는 이유로 다른 C*를 정하면 안 된다. 전 서버 공통 lock을 기다린다. 대기 시간에는 후처리·평가·metadata 검사를 수행하되 새 하이퍼파라미터 탐색을 넣지 않는다.

### 7.2 시간

이번 export의 최근 Train(h)를 보고 잡은 **계획 대용**이다. 실행 중 run 잔여와 ledger를 새로 읽어 실제 예약을 갱신해야 한다.

- s1 2.40h/run
- s2 2.60h/run
- s3 1.50h/run
- s4 1.55h/run
- s5 2.70–3.40h/run

s5의 최근 세 등록 값은 1.64 / 2.69 / 3.40h이므로 과거 1.46h를 현재 고정 처리시간으로 사용하지 않았다. 2.70–3.40은 중앙값 부근과 최근 큰 값의 두 계획 시나리오이지 보장 범위·신뢰구간이 아니다. 이 변화의 원인을 방법 효율에 귀속하지 않는다.

| 서버 | lock 전 run | lock 후 run | 합계 | 학습 대용(h) |
|---|---:|---:|---:|---:|
| s1 | 2 | 4 | 6 | 14.40 |
| s2 | 1 | 4 | 5 | 13.00 |
| s3 | 1 | 4 | 5 | 7.50 |
| s4 | 1 | 4 | 5 | 7.75 |
| s5 | 0 | 4 | 4 | 10.80–13.60 |
| 합계 | 5 | 20 | 25 | **53.45–56.25** |

예약 대용으로 1.20×학습 대용 + run당10분을 한 번 적용하면 합산 **68.31–71.67 server-hours**다. lock 전후 두 stage를 순차로 진행하고 stage 안에서는 서버가 병렬로 동작한다는 가정의 경과시간 대용은 약 **20–24시간**이다.

현재 실행 중인 carry-over의 잔여, selector 구현/감사, 추가 공식 RR 후보 재평가, 장애/대기 비용은 별도다. 이미 Train(h)에 포함된 정기 평가를 다시 독립 시간으로 더하지 않는다. 위 여유치는 예약 목적이며 실제 지출 ledger에서 평가를 중복 계산하지 않는다.

원 QRC24의 **서버당 누적 신규 학습 24시간 이상** 정책은 유지한다. 이번 재편으로 24시간을 0부터 다시 세지 않는다. Sheet 중복행을 합산하지 말고 실제 ledger로 확인한다. 추가 시간이 필요한 경우에도 새로운 grid가 아니라 locked profile의 seed run으로 충족한다.

## 8. 구현 인계 요구사항

### 8.1 generator와 큐

기존 `tools/gen_pakd50_configs.py`에 신규 server/seed 조합 4개와 locked-seed 계획을 등록한다. B20A03이라는 새 수학적 profile을 만들 필요는 없다. 이미 있는 정의를 재사용한다.

새 비교 ID의 v3는 편성 metadata 이력이지 factor3 또는 수식 변경이 아니다. 원 v1/v2 config를 덮어쓰지 않는다. reference_run_id는 위 대응 G23의 실제 ID를 사용하고 가상 G22를 생성하지 않는다.

새 seed ID는 C* 선택 후 원 profile 이름을 넣어 확정한다. 모든 seed에 대해 기존 G22를 추가로 학습해야 한다고 해석하지 않는다.

동일 revision으로 일치시킬 대상:
- generator priority/registered/held/mandatory
- 저장소 queue 파일
- 실제 active/effective/persistent queue
- reservations와 ledger
- handover와 waiter
- selector/postrun 및 uploader
- immutable recipe_lock

경로는 현재 구현을 확인해 연결한다. `--narrow`나 `--lock` 같은 실행 옵션이 이미 있다고 가정하지 않는다. 안전한 case 경계 전환을 사용한다.

### 8.2 Sheet 기록

Sheet는 직접 수정하지 않는다. gspread uploader/manifest를 통해 반영한다.

기존 legacy quality columns를 조용히 새 selector 값으로 부분 교체하지 않는다. v2 목표값은 새 식별 컬럼 또는 기존 시스템의 명시적 target 영역에 다음을 한 묶음으로 업로드한다.

`target_selector`, `target_step`, `target_HQNR_raw`, `target_ERGAS`, `target_SCC`, `target_PSNR`,
`target_joint_pass`, `target_official_complete`, `recipe_lock_id`, `queue_revision`.

SAM/Q8/SSIM을 포함한 전체 같은-checkpoint 수치와 identity는 sidecar에 보존한다. target HQNR과 다른 step의 ERGAS를 같은 결과처럼 표시하면 안 된다. 열 위치는 header로 결정하며 기존 Y열 등을 고정 가정하지 않는다.

같은 run의 metadata 갱신은 새 학습 결과 행을 추가하는 것이 아니라 원 run의 target metadata 갱신으로 다룬다. 동일 run ID 중복행을 새로운 seed로 집계하지 않도록 uploader/분석기에 원본 ID key를 유지한다.

### 8.3 최소 회귀 검사

1. q 함수·uniform .5·λE 환산값·Student gradient routing이 기존 방법과 동일하다.
2. 대응 G23와 B20A03의 허용 config 차이는 β와 식별 metadata뿐이다.
3. 같은 pre-step 상태에서 β 변경은 A의 직접 gradient를 바꾸지 않는다.
4. v2 selector가 H 미달·반올림 경계·공식 RR 누락을 정확히 처리한다.
5. 가상 예시 H=.9586/E=2.039 vs H=.9600/E=2.070에서 앞 후보를 선택한다.
6. v1 official 완료가 v2 완료로 오인되지 않고, 유효한 RR cache는 재사용된다.
7. lock 후 학습 정의 diff는 seed 및 허용 metadata/환경 경로뿐이다.
8. 보류 ID가 mandatory/waiter/재시작 queue에 남아 자동 재편성되지 않는다.
9. 기존 완료·중단 checkpoint를 덮어쓰거나 잘못 미시작으로 판정하지 않는다.
10. 과거 81개 또는 39개 등록을 영구적인 live count로 하드코딩하지 않는다.

## 부록 A. lock 전 계획 ID

아래 ID는 신규 등록이 필요한 **계획용**이다. 기존 s4 G23만 원 ID를 유지한다. 완료/진행 중인 동등 run이 별도로 생겼으면 중복을 피하고 실제 ID로 연결한다.

| 서버 | Profile | Seed | 용도 | 계획 ID |
|---|---|---:|---|---|
| s1 | B20A03 | 1234 | beta_close | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S1234_FRESH50_v3` |
| s1 | B20A03 | 3407 | beta_close | `PAKD50_QRC24_S1_B20A03_W104_D121_WV3_T0_S3407_FRESH50_v3` |
| s2 | B20A03 | 777 | beta_close | `PAKD50_QRC24_S2_B20A03_W104_D121_WV3_T0_S777_FRESH50_v3` |
| s3 | B20A03 | 2026 | beta_close | `PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S2026_FRESH50_v3` |
| s4 | G23 | 1234 | host_bridge | `PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2` |

## 부록 B. 미시작일 때만 보류할 기존 14개 ID

| 서버 | Profile | Seed | 기존 ID |
|---|---|---:|---|
| s1 | A_FREEZE | 3407 | `PAKD50_QRC24_S1_A_FREEZE_W104_D121_WV3_T0_S3407_FRESH50_v1` |
| s1 | A_SHUF | 3407 | `PAKD50_QRC24_S1_A_SHUF_W104_D121_WV3_T0_S3407_FRESH50_v1` |
| s1 | G21 | 3407 | `PAKD50_QRC24_S1_G21_W104_D121_WV3_T0_S3407_FRESH50_v1` |
| s2 | E_SHUF | 777 | `PAKD50_QRC24_S2_E_SHUF_W104_D121_WV3_T0_S777_FRESH50_v1` |
| s2 | ALL_UNIF | 777 | `PAKD50_QRC24_S2_ALL_UNIF_W104_D121_WV3_T0_S777_FRESH50_v1` |
| s3 | G31 | 4321 | `PAKD50_QRC24_S3_G31_W104_D121_WV3_T0_S4321_FRESH50_v1` |
| s3 | G33 | 4321 | `PAKD50_QRC24_S3_G33_W104_D121_WV3_T0_S4321_FRESH50_v1` |
| s4 | B20A03 | 1234 | `PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S1234_FRESH50_v2` |
| s5 | G23 | 9091 | `PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S9091_FRESH50_v2` |
| s5 | H23 | 9091 | `PAKD50_QRC24_S5_H23_W104_D121_WV3_T0_S9091_FRESH50_v2` |
| s5 | B20A03 | 9091 | `PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S9091_FRESH50_v2` |
| s5 | G23 | 1103 | `PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S1103_FRESH50_v2` |
| s5 | H23 | 1103 | `PAKD50_QRC24_S5_H23_W104_D121_WV3_T0_S1103_FRESH50_v2` |
| s5 | B20A03 | 1103 | `PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S1103_FRESH50_v2` |

## 부록 C. 동봉 파일

- `PAN_QRC24_Narrow_R2_CasePlan_2026-09-17.csv`: 5개 lock 전 계획, 20개 조건부 seed 계획, 14개 미시작 보류의 총 39개 **편성 의사결정 행**. 39개 새 실험이라는 뜻이 아니다.
- `PAN_QRC24_Narrow_R2_Source66_2026-09-17.csv`: 이번 export에서 원본 ID로 중복 제거한 66개 결과, 실제 Sheet 행 링크.
- seed 행의 `C_STAR`, `{LOCKED_PROFILE}`, 빈 planned_run_id는 **미확정 계획 표기**다. 지금 실행할 수 있는 큐로 가장하지 않는다.
- 모든 `enabled_now`는 false다. 본 문서는 승인·구현 전 계획이며 원격 작업을 수행하지 않았다.

**종료 상태:** 목표의 2순위를 ERGAS로 변경했다. 기존 q/e 방법은 유지하고, 마지막 β 대응 비교 후에는 공통 profile 하나만 남긴다. 이후 변하는 학습 요소는 사전 지정 Student seed뿐이다.
