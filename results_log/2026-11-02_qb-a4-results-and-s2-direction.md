# QB A4/E.03 완료 결과와 s2 후속 방향 — 2026-11-02

11월 2일 09:18–09:28 KST에 s1 원본 평가 JSON, donor 선택 receipt, 학습 로그와
[QB-ablations](https://docs.google.com/spreadsheets/d/1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE/edit#gid=261001011)을 확인했다.
이번 작업은 CPU 기반 분석이며 추론·학습·실행 큐를 변경하지 않았다.
사용자의 향후 서버 역할은 **s2: QB 후속 실험, s1/s3: ablation**이다.

재계산: [코드](assets/2026-11-02_qb-a4-results-review/recompute.py),
[원자료 요약](assets/2026-11-02_qb-a4-results-review/analysis.json),
[Sheet 스냅샷](assets/2026-11-02_qb-a4-results-review/sheet_snapshot.json),
[PNG](assets/2026-11-02_qb-a4-results-review/qb_a4_review.png),
[PDF](assets/2026-11-02_qb-a4-results-review/qb_a4_review.pdf).

**결론**

λ_abs를 2에서 4로 올린 실험은 검증 RRE50과 Student RR ERGAS를 일부 개선했다.
하지만 s1에서는 Student 세 seed 모두 HQNR 기준에 미달했고, s3에서는 세 seed 모두
HQNR을 유지했다. 따라서 A4/E.03을 일반적인 개선으로 확정하거나 λ_abs를 계속 높일
근거는 부족하다. RRE50은 RR 성능의 유용한 보조 지표지만 FR HQNR까지 보장하는 단일
선택 지표가 아니다. 기존 validation donor 선택 규칙을 test 성능으로 대체하지 않는다.

**완료 여부와 공식 결과**

s1은 A4/E.03 Stage 1 50K와 Student 30K R1/R2/R3가 모두 완료됐다.
launcher는 08:49:50 KST에 전체 완료, 08:51:10 KST에 마지막 Sheet readback 성공을 기록했다.
이전 handoff 설정에 따라 현재 s1은 WV3 TUNE30으로 자동 복귀했다.
s3도 Sheet에 Stage 1과 Student 세 개가 모두 COMPLETED로 올라왔다.
s3 원격 프로세스의 현재 상태나 전체 30K 평가 원본까지 직접 확인한 것은 아니다.
s2용 실행 환경이나 큐 전환은 이번 분석에서 구성하지 않았다.

목표는 **동일 checkpoint에서 HQNR ≥ .9315 AND ERGAS ≤ 3.5**다.
새 Student 여섯 개에서 동시 통과는 없다.

| 서버 | Student | 대표 step | HQNR | ERGAS | 선택 규칙 |
|---|---|---:|---:|---:|---|
| s1 | A4/E.03 R1 | 23K | .920770 | 3.518307 | HQNR 최고 fallback |
| s1 | A4/E.03 R2 | 30K | .927337 | 3.514089 | HQNR 최고 fallback |
| s1 | A4/E.03 R3 | 24K | .923498 | 3.514421 | HQNR 최고 fallback |
| s3 | A4/E.03 R1 | 21K | .950180 | 3.519644 | HQNR 충족 중 ERGAS 최소 |
| s3 | A4/E.03 R2 | 29K | .944009 | 3.516729 | HQNR 충족 중 ERGAS 최소 |
| s3 | A4/E.03 R3 | 25K | .938317 | 3.514210 | HQNR 충족 중 ERGAS 최소 |

s1 A4의 1K 단위 평가 30개 × 3 seed 중 HQNR 충족 checkpoint는 없다.
s1의 fallback ERGAS와 s3의 HQNR 조건부 최소 ERGAS를 동일 선택 기준처럼 비교하면 안 된다.
Teacher의 ERGAS 3.488–3.492도 HQNR이 약 .75이므로 목표 달성 결과가 아니다.

**동일 step·동일 Student seed 비교**

s1에서 A2/E.03과 A4/E.03을 고정 30K, 동일 Student seed 세 개로 비교했다.

| s1, 30K 평균 | HQNR | ERGAS | D_s | D_lambda |
|---|---:|---:|---:|---:|
| A2/E.03 | .941471 | 3.515895 | .029180 | .030283 |
| A4/E.03 | .922587 | 3.510889 | .043223 | .035642 |
| 차이 | −.018884 | −.005005 | +.014043 | +.005359 |

R1/R2/R3 ERGAS 변화는 각각 −.003810 / +.004066 / −.015272다.
ERGAS 개선 자체도 모든 Student seed에서 공통으로 발생하지 않았다.
반면 HQNR은 각각 −.023997 / −.018415 / −.014239로 일관되게 악화했다.
FR 20장 기준으로 각 seed에서 20/20, 18/20, 19/20장이 악화했다.
이 60개 비교는 동일 장면과 donor를 공유하므로 독립 표본 60개로 해석하지 않는다.

장면별 H=(1−D_lambda)(1−D_s)의 변화량을 두 항에 대칭적으로 분해하면
HQNR 하락의 72.6%는 D_s 항, 나머지는 D_lambda 항이다. 지표의 대수적 분해이며
물리적 원인의 72.6%를 입증한 것은 아니다.

HQNR 격차는 학습 초기부터 존재하며 후반까지 유지된다.
단순히 cosine 후반에만 발생한 손상이 아니므로 학습 기간 연장이 우선 해결책은 아니다.

s3는 전체 고정30K 원본이 없어 공식 대표값으로만 비교했다.
A2→A4의 평균 HQNR은 .943004→.944169, 평균 ERGAS는 3.524165→3.516861이다.
평균 ERGAS 개선은 −.007305이나 seed별 변화는 +.004664 / −.020147 / −.006431로,
R2의 기여가 가장 크다. s3 A4를 버릴 근거도 없지만, 세 seed 모두 개선했다고 말할 수 없다.

**RRE50을 낮추면 충분한가**

| 서버 | donor | 검증 선택 step | RRE50 | RRE90 |
|---|---|---:|---:|---:|
| s1 | A2/E.03 | 44K | .235295 | .687462 |
| s1 | A4/E.03 | 44K | .209725 | .643745 |
| s3 | A2/E.03 | 39K | .270547 | .727176 |
| s3 | A4/E.03 | 42K | .228485 | .652866 |

모두 Stage 1 전체 50K를 실행한 뒤 기존 validation selector로 선택한 checkpoint다.
s1의 A4 RRE50은 A2 대비 10.9% 감소했다. s3 A4보다도 낮지만 Student HQNR은 훨씬 낮다.
따라서 기존의 “RRE50 개선은 RR 개선 가능성을 높인다”는 방향은 남아 있으나,
“λ_epsilon=.03이면 HQNR은 유지된다”거나 “RRE50이 낮은 donor가 최종적으로 우수하다”는
강한 해석은 이번 결과와 맞지 않는다.

검증 epsilon EPE50도 s1 A2 .138686 → A4 .139416으로 거의 같고,
s3 A4는 .170434로 더 나쁘지만 Student HQNR은 높다.
RR validation에서 측정한 이동 응답과 FR 복원 성능은 같은 문제가 아니다.
기존 선형 기울기의 약 −.003 ERGAS/.01px를 목표 달성 예측식처럼 사용하지 않는다.

**실제 이동량이 커져서 생긴 문제인가**

s1 선택 donor의 공식 FR20에 대한 원본 예측 이동량을 비교했다. 단위는 현재 PAN pixel이다.

| 통계 | A2/E.03 | A4/E.03 |
|---|---:|---:|
| 평균 dy | 1.995414 | 2.019058 |
| 평균 dx | −1.909223 | −1.947780 |
| 이동 norm 중앙값 | 2.821986 | 2.860772 |
| 이동 norm p90 | 2.969319 | 2.977874 |

같은 장면에서 두 이동 벡터의 차이 norm 중앙값은 .060172px, p90은 .092963px,
최댓값은 .132865px다. 큰 이동량 증가가 HQNR −.019를 설명한다는 증거는 없다.
작은 소수 픽셀 차이의 효과가 없다는 뜻은 아니며, FR20 이외의 훈련 분포 전체를
이 수치로 대표할 수도 없다.

R1의 25–30K 기록 배치 51개에서 Hard weight 평균은 A2/A4 모두 약 1.21736이고,
d_e 평균은 .49977/.49967, support 비율은 .793213으로 같다.
전반적인 weight 크기나 support 감소가 원인이라고 주장할 근거도 약하다.
다만 같은 평균이어도 강조하는 픽셀 위치는 달라질 수 있다.

현재 남는 설명 후보는 (a) 학습 전체의 미세한 보정 패턴 차이,
(b) Teacher difficulty와 q의 공간적 분포 차이,
(c) RR에서 학습한 Student 함수의 FR 전달 차이와 이들의 상호작용이다.
현재 자료로 어느 하나를 원인으로 확정하지 않는다.
서버별 Stage 1 seed와 실행 환경이 함께 바뀌었으므로 서버 효과와 seed 효과도 분리되지 않았다.
한 donor 아래 Student 세 seed는 Stage 1 독립 반복 세 번이 아니다.

**기존 후보와의 위치**

HQNR을 충족한 기존 최저 ERGAS는 A2 기반 QB_BASE R2 @26K의
HQNR .945384 / ERGAS 3.509656으로, 목표까지 ERGAS .009656이 남는다.
새 A4 결과는 이 최저 ERGAS를 갱신하지 않았다.

W128 R1/R2는 대표 ERGAS 3.511541/3.511738이며 H050 R1/R2는 3.511504/3.512173이다.
작은 후속 후보로는 남지만 각각 동일 Student seed의 A2 기준을 모두 이긴 것은 아니다.
특히 Sheet H050 R1의 ERGAS 셀은 공백으로 확인되어 로컬 원본 30K 값으로 보완했다.
Sheet 자체는 이번 작업에서 수정하지 않았다.

C60LR050 R1은 완료됐다. 대표값은 42K의 .943603/3.520594,
60K 최종값은 .941518/3.525746으로 기존 R1 대표 ERGAS 3.514885보다 나쁘다.
s3 B96의 LR1e-4/2e-4 대표 평균 ERGAS도 3.571403/3.557144로 기준 R1/R2 평균
3.525927보다 나쁘다. 길이·batch·여러 Student 손잡이를 넓게 다시 탐색할 우선순위는 낮다.

**s2에서 권하는 순서 — 제안이며 아직 등록·실행하지 않음**

1. **A2/E.03과 A4/E.03의 Stage 1 반복 재현을 먼저 한다.** s2에서 자체 Stage 1을
   학습하며 각 설정에 독립 Stage 1 seed 최소 두 개를 배정한다. 각 donor의 첫 Student는
   동일 seed로 비교하고 이후 R2/R3로 확인한다. 계수 효과와 donor 학습 변동을 구분하는
   것이 목적이다. test가 잘 나온 donor만 남겨 전체 재현성이 좋아 보이게 보고하지 않는다.
2. **중간값 A3/E.03과 균형 확인 A4/E.06을 후속 후보로 둔다.** A3은 abs 증량의 중간점이다.
   A4/E.06은 A2/E.03의 epsilon/abs 계수비 .015를 유지한다.
   A4/E.03의 비는 .0075로 절반이다. 다만 계수비가 실제 loss/gradient 비는 아니며,
   E.06이 HQNR을 회복한다는 증거는 아직 없다. 반복 결과를 보고 순서를 확정한다.
3. Student는 우선 LR1e-4/C30/B48/W112D123/DIFF/UBIL, α=.25/γ=.5/β=0,
   full scale1로 고정한다. Stage 1 균형이 안정된 뒤 H050·W128 같은 작은 후속 비교를 한다.
   회전 loss, TTA, SWA, 새 밴드 가중 loss를 추가하지 않는다.
4. donor는 전체50K 이후 기존 validation RRE50→RRE90→ValERGAS→earlystep 규칙을 유지한다.
   RRE .20/.23을 새로운 자동 합격·중단 문턱으로 삼지 않는다.
   Student는 기존 8/9/10K HQNR 모두 <.90 중단 규칙을 유지하며,
   Teacher HQNR로 Student를 중단하지 않는다.

**s1/s3 ablation에서 분리할 질문**

전체 ablation 데이터셋과 최종 표 구성은 아직 지정되지 않았다.
이번 QB 현상과 관련해서는 아래 비교가 원인 규명에 도움이 된다.
각 서버는 자체 학습한 donor를 사용하되, 서버 내부의 해당 비교에서는 donor를 고정한다.

- A2/A4 donor × 기존 Hard / 균일 GT(α=γ=0)의 2×2 비교:
  donor 차이가 가중 손실을 꺼도 유지되는지 확인한다.
  균일 GT에서도 지지영역 등 donor 의존성이 남으므로 순수한 기하만의 완전 분리는 아니다.
- 같은 donor에서 corrected PAN만 사용 / corrected PAN+residual 비교:
  residual 경로가 보정 입력의 이득을 회수하는지, 원본 PAN 의존성을 높이는지 확인한다.
- 필요한 경우 α만/γ만/둘 다의 기존 loss component 비교로 좁힌다.
  첫 단계부터 전체 조합을 곱하지 않는다.

이들은 주 실험의 scale1 조건을 유지하면서 기여를 분리할 수 있는 ablation 제안이다.
주 방법의 새 구성요소로 추가하는 것이 아니다.

**가중치 보관**

s1 A4 Student 세 개는 각각 1K–30K 평가 checkpoint 30개의 .pt 파일 존재와
0보다 큰 파일 크기를 확인했다. 목표 통과 여부와 무관하게 보관돼 있다.
s3는 Sheet metadata의 weights_saved=true 및 checkpoint SHA까지 확인했으며,
원격 파일 실재 여부를 이 세션에서 직접 재검사하지 않았다.
