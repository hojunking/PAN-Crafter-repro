# s3 실행 인계 — QEGX / W104–D121
**2026-09-15 · 신규 15run · 학습 기준 20.64h · 예약 25시간 12분 14초 · 시간 상한 없음**

## 이번 역할
같은 작은 backbone에서 **q-edge 선택과 soft의 상호작용**을 분리한다. 기존 W112 12개는 끝난 근거로만 사용한다. 새 W104의 로컬 J0/JQ/XJ를 확보한다.

## 공통
- Student W104/D121, Teacher T0 W112/D123 frozen, Student A는 T0 A 복사.
- fresh50K, batch48, U LR1e-4/A LR1e-5, warmup100+cosine, WD0.01.
- α1, β0.1(명시한 B005만0.05), τR0.012463942170143127, λE0.09075170336956798.
- λoff1e-4 홀수 update, b2 HR px. GT hard 항상 유지.
- q gate는 frozen T0 AXIS16 q < 0.3276133416220546. cache asset_id `526dbd718af54c4a0c3eb5fd81778caa`.
- Selected raw-original HQNR를 주로 보고 exact50K·고정6점 plateau·fSCC·ERGAS도 저장.

## 명시 큐
| 순서 | 단계 | Seed | Case | U / A의 edge 계수 | 시간 기준(h) | 예약(h) | 질문 |
|---:|---|---:|---|---|---:|---:|---|
| 1 | A | 2026 | `J0` | `0` / `0` | 1.16 | 1.4427 | 동일 구조·seed의 joint GT-only 기준 |
| 2 | A | 2026 | `JQ` | `1` / `1` | 1.34 | 1.6407 | 기존 Q12 기준 |
| 3 | A | 2026 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 4 | A | 2026 | `XJ` | `1` / `1` | 1.33 | 1.6297 | all-edge에서 output soft 제거 대조 |
| 5 | A | 2026 | `QX50` | `g` / `g` | 1.50 | 1.8167 | 같은 q-edge gate에서 output soft 제거 |
| 6 | A | 2026 | `J_R3_NOEDGE` | `0` / `0` | 1.30 | 1.5967 | edge 없는 reconstruction KD 기준 |
| 7 | A | 2026 | `QEC3` | `c_E3` / `c_E3` | 1.34 | 1.6407 | s3 고정 pilot의 평균 edge loss 강도에 맞춘 상수 대조 |
| 8 | A | 2026 | `QES` | `g_shuffle` / `g_shuffle` | 1.50 | 1.8167 | e-decile/augmentation 조건부 q-label shuffle |
| 9 | B | 2026 | `J_QB005` | `1` / `1` | 1.34 | 1.6407 | all-edge에서 beta 0.05 |
| 10 | B | 2026 | `QE50_B005` | `g` / `g` | 1.50 | 1.8167 | 같은 beta 0.05에서 q-edge 선택 |
| 11 | C | 4321 | `J0` | `0` / `0` | 1.16 | 1.4427 | 동일 구조·seed의 joint GT-only 기준 |
| 12 | C | 4321 | `JQ` | `1` / `1` | 1.34 | 1.6407 | 기존 Q12 기준 |
| 13 | C | 4321 | `XJ` | `1` / `1` | 1.33 | 1.6297 | all-edge에서 output soft 제거 대조 |
| 14 | C | 4321 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 15 | C | 4321 | `QX50` | `g` / `g` | 1.50 | 1.8167 | 같은 q-edge gate에서 output soft 제거 |

## 가장 중요한 비교
`QE50−JQ`, `QX50−XJ`, `QE50−QX50` 대 `JQ−XJ`.
QX50는 QE50에서 output soft만 제거한 조건이다. **QX50 ≠ XJ**.
QEC3는 s3 J0S2026 exact50K pilot의 c_E3를 사용한다. s1 QEC의 파일을 덮어쓰지 않는다.
β0.05는 ungated/gated 두 run을 함께 비교하며 한 후보의 결과만 보고 β 효과라고 하지 않는다.

## 기동 전
Cue/data/Teacher/init hash·gradient·resume 검사. 신규 QX50/QEC3/QE50_B005 binding 및 QEGX branch가 필요하다.
QEC3는 local J0가 끝난 뒤 pilot을 계산한다. QES는 기존 deterministic shuffle cache를 그대로 쓴다.
QEDGE9 s5 seed2026 결과는 s3 control을 대체하지 않는다.
현재 실행 중인 run이 있다면 run 경계에서 인계하며 중복 실행하지 않는다.
기존 50h/deadline/9h 중단을 상속하지 않는다. s1/s5 큐는 유지한다.

## 예약
1.10×동일 서버 시간 대용 + run당10분, 정기평가 중복 가산 없음.
W104 및 QE50의 s3 시간은 새 실측 전 대용/임시값이다. 첫 실제 run으로 갱신한다.
구현·cue verify·pilot·현재 작업 잔여시간은 별도.
같은 기본 정의의 local 완료 run이 검증되면 재사용할 수 있지만, 다른 architecture/server/seed를 대신 넣지 않는다.

## 추가
β pair가 유력하면 S4321 J_QB005/QE50_B005 약3h27m 추가.
e cutoff를 시험하면 ES50/ESQE50를 별도 pair로 사용하고 GT hard는 유지한다.
전체 수식·판정·구현 규약은 `Experiment_Plan.md`를 따른다.
`case_registry.json`, `s3_queue.csv`는 계획용 명세이며 그대로 실행 가능한 config/queue라고 보장하지 않는다.
