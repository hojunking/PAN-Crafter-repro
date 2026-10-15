# GF2 L100 종료 전 결과 검토와 다음 방향 — 2026-09-21 23:09 KST

## 0. 범위·결론

- 조회: live `pan-cvpr27`, spreadsheet ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`.
- 원본 export: `pan-cvpr27_20260921_2309KST.xlsx`.
- SHA256: `0e315b9bf8426a6c904b9115dbe5ecebf9ba75eb4fcb6a6c165aa8b2bd937d74`.
- 직전 기준: `GF2_L100_Results_and_Next_Direction_20260921_2119KST.md`.
- 이번에는 결과 분석과 후속 가설 설계만 수행했다. 학습·queue·Sheet·평가 규약은 변경하지 않았다.
- 원격 프로세스, 미업로드 결과, 100K의 실제 correction tensor와 gradient를 직접 측정하지 않았다.
- XLSX importer 실패 후 원본 ZIP/XML의 저장된 cell 값을 표준 라이브러리로 추출했다. 핵심 endpoint는 별도 native Sheets API readback과 대조했다. 워크북을 수정·재계산하지 않았다.

**새 결과는 H010의 종합 개선을 지지하지 않는다. H010은 두 seed 모두 RR를 개선하지만 두 seed 모두 D_s를 악화시키며, 두 번째 seed에서는 큰 HQNR 손실이 발생했다. E1은 새 R100/S100 조건의 두 seed 모두 BASE보다 HQNR·ERGAS·D_s에서 불리하다. 반면 s3 R100/S100의 두 번째 seed는 H=0.955960/E=0.551502로, 같은 100K 구조에서도 좋은 RR와 상대적으로 좋은 FR이 함께 가능하다는 관측을 제공한다.**

**11개 100K Student 모두 자기 run의 50K 중간점에서 100K endpoint로 가는 동안 HQNR·ERGAS·D_s가 함께 개선됐다. 따라서 현재의 핵심을 단순한 후반 붕괴로 표현하면 잘못이다. fresh50K와 fresh100K endpoint 차이, 같은 100K run의 후반 개선, 서로 다른 seed의 차이는 분리해야 한다.**

## 1. 완료 등록

| 서버 | Student 공식 완료 | Teacher 전체평가 공식 완료 | 남은 주요 등록 |
|---|---:|---:|---|
| s3 | 4/6 | 0/2 | SS95002의 R50/S100 및 R50/S50; Teacher 후보 전체평가 |
| s4 | 4/4 | 1/1 | 계획된 Student 비교 완료 |
| s5 | 4/4 | 1/1 | 계획된 Student 비교 완료 |

L100 공식 완료는 Student12 + Teacher2 = 14행이다. Student 중 11개는100K, 하나는 의도된50K 대조다. 모든 등록행은 `OFFICIAL_EVAL_COMPLETE`, 후보50개, `READBACK_VERIFIED`다. 아직 등록되지 않은 항목은 미실행·실패·idle과 동의어가 아니다. s3 Student metadata에는 R50/R100 local reference가 연결돼 있으나 이번 검토에서 해당 원격 bytes는 다시 읽지 않았다.

직전 검토 이후 GF2 신규 Student3개(S04/S10/S14), Teacher2개(T03/T04)가 추가됐다.
현행 QG40/G20/L100 Student 총69개(QG40 23/G20 34/L100 12)의 저장 RAW_MAX는 모두0.964 미만이다. 동일 seed의 다양한 profile이 포함돼 있어69개 독립 seed 실험은 아니다.
11개100K Student 중8개는 endpoint ERGAS<0.552다. 기본 공동목표는 같은 정상 A_ON checkpoint에서 HQNR>0.964 및 ERGAS<0.552이며, 달성한 결과는 없다.

## 2. L100 Student 전체 endpoint

| Case | 서버 | Student seed | 길이 | Profile | HQNR | ERGAS | D_lambda | D_s | 출처행 |
|---|---|---|---|---|---|---|---|---|---|
| L100I1-S01 | s3 | 95001 | T50/S50 | BASE | 0.952923 | 0.560808 | 0.019998 | 0.027607 | GF2-s3(5090)!34 |
| L100I1-S02 | s3 | 95001 | T50/S100 | BASE | 0.950305 | 0.553379 | 0.020451 | 0.029830 | GF2-s3(5090)!35 |
| L100I1-S03 | s3 | 95001 | T100/S100 | BASE | 0.946663 | 0.551329 | 0.020473 | 0.033533 | GF2-s3(5090)!36 |
| L100I1-S04 | s3 | 95002 | T100/S100 | BASE | 0.955960 | 0.551502 | 0.020636 | 0.023873 | GF2-s3(5090)!37 |
| L100I1-S07 | s4 | 95001 | T100/S100 | BASE | 0.953483 | 0.552704 | 0.020265 | 0.026773 | GF2-s4!14 |
| L100I1-S08 | s4 | 95001 | T100/S100 | H010 | 0.953660 | 0.544284 | 0.019450 | 0.027401 | GF2-s4!15 |
| L100I1-S09 | s4 | 95002 | T100/S100 | H010 | 0.946669 | 0.546269 | 0.020063 | 0.033919 | GF2-s4!16 |
| L100I1-S10 | s4 | 95002 | T100/S100 | BASE | 0.954274 | 0.548244 | 0.020168 | 0.026062 | GF2-s4!17 |
| L100I1-S11 | s5 | 95001 | T100/S100 | BASE | 0.947716 | 0.550621 | 0.020819 | 0.032101 | GF2-s5!21 |
| L100I1-S12 | s5 | 95001 | T100/S100 | E1 | 0.947044 | 0.552141 | 0.020243 | 0.033359 | GF2-s5!22 |
| L100I1-S13 | s5 | 95002 | T100/S100 | E1 | 0.951072 | 0.550514 | 0.020509 | 0.028998 | GF2-s5!23 |
| L100I1-S14 | s5 | 95002 | T100/S100 | BASE | 0.951309 | 0.547760 | 0.020804 | 0.028465 | GF2-s5!24 |

비교는 모두 동일 row의 EXACT_FINAL이며, 다른 checkpoint의 H/E를 합치지 않았다. 원시 정밀도와 각 선택점의 checkpoint SHA는 동봉 CSV에 보존한다.

## 3. 대응 효과

| 대조 ALT−BASE | ΔHQNR | ERGAS 상대변화 | ΔD_lambda | ΔD_s |
|---|---|---|---|---|
| Student50to100 | -0.002618 | -1.325% | +0.000453 | +0.002223 |
| Teacher50to100 | -0.003642 | -0.371% | +0.000022 | +0.003703 |
| H010_SS95001 | +0.000176 | -1.523% | -0.000815 | +0.000628 |
| H010_SS95002 | -0.007604 | -0.360% | -0.000105 | +0.007857 |
| E1_SS95001 | -0.000672 | +0.276% | -0.000576 | +0.001257 |
| E1_SS95002 | -0.000237 | +0.503% | -0.000295 | +0.000533 |

### 3.1 H010: RR 개선 후보이지 공간 개선 후보가 아니다

- SS95001: E -1.523%, H +0.000176, D_s +0.000628.
- SS95002: E -0.360%, H -0.007604, D_s +0.007857.
- 두 pair 평균: E 상대변화 -0.942%, H -0.003714, D_s +0.004243.
- 두 seed 모두 D_lambda는 개선됐다. 첫 seed의 거의 유지된 H는 D_lambda 개선이 D_s 비용을 상쇄한 결과다.
- RR_VAL_SELECTED에서도 부호가 같으며, SS95002 H 차이는 약 -0.007626이다.
- SS95002에서는 100K recipe의 중간50K부터 H010-BASE H 차이가 -0.007883이다. endpoint -0.007604와 비슷하므로 마지막 몇 update의 붕괴로 설명할 수 없다.
- 따라서 H010을 다음 모든 fresh run의 기본값으로 승격하지 않는다. RR 개선 자산으로 보존하고, 후기 도입 대조에 사용할 수 있다. 초기부터 적용했을 때의 공간 비용 원인은 미확정이다.

### 3.2 E1: R100/S100에서는 두 seed 모두 불리하다

- SS95001: H -0.000672, E +0.276%, D_s +0.001257.
- SS95002: H -0.000237, E +0.503%, D_s +0.000533.
- 두 seed 모두 D_lambda는 조금 좋아지지만, H/E/D_s의 공동 개선은 없다.
- TA/50K에서의 과거 이득을 이 reference/horizon으로 일반화하지 않는다.
- E1과 그보다 더 약한 E02의 추가 자동 탐색은 우선순위를 낮춘다. edge 제거를 권고하는 결과도 아니다.

### 3.3 학습 길이: 첫 seed의 대조만 완성

T50/S50→T50/S100은 E -1.325%, H -0.002618, D_s +0.002223.
T50/S100→T100/S100은 E -0.371%, H -0.003642, D_s +0.003703.

이 결과는 SS95001의 대응 효과로 유지하되, T100/S100 SS95002의 H=0.955960을 고려하면 ‘T100은 공간 품질이 나쁜 reference’라고 일반화할 수 없다. SS95002의 R50 대조 두 개가 아직 필요하다.
fresh50K와 fresh100K는 처음부터 cosine horizon이 다르므로 단순 추가50K의 효과라고 부르지 않는다.

## 4. 더 중요한 새 관측: 같은 RR 정확도에서 큰 FR 차이

동일 s3/TS94001/R100/BASE/S100:
- SS95001: H0.946663, E0.551329, D_s0.033533.
- SS95002: H0.955960, E0.551502, D_s0.023873.

E 차이는 +0.000173(+0.0314%)에 불과하지만 H +0.009297, D_s -0.009660 차이가 난다. 50K 중간점부터 H 차이가 약0.007385로 존재한다.

이것은 현재 RR 수준만으로 FR 품질을 구분하기 어렵다는 관측이다. U의 초기화·학습 경로 또는 학습분포 일반화가 중요한 후보라는 해석을 지지하지만, 이 두100K run의 실제 c를 직접 비교하지 않았으므로 U만의 인과효과라고 확정하지 않는다. 앞서 사용자가 제공한50K 진단은 U-side 가설의 별도 근거이며100K 직접 측정으로 바꾸어 인용하지 않는다.

같은100K run 내50K→100K 비교:

| Case | H@50K of100 | H@100K | ΔHQNR | ERGAS 상대변화 | ΔD_s |
|---|---|---|---|---|---|
| L100I1-S02 | 0.946326 | 0.950305 | +0.003979 | -3.608% | -0.003207 |
| L100I1-S03 | 0.945388 | 0.946663 | +0.001275 | -4.040% | -0.001096 |
| L100I1-S04 | 0.952772 | 0.955960 | +0.003188 | -2.071% | -0.003319 |
| L100I1-S07 | 0.952704 | 0.953483 | +0.000779 | -3.441% | -0.000427 |
| L100I1-S08 | 0.949360 | 0.953660 | +0.004299 | -2.799% | -0.004009 |
| L100I1-S09 | 0.941053 | 0.946669 | +0.005616 | -2.264% | -0.004535 |
| L100I1-S10 | 0.948936 | 0.954274 | +0.005338 | -2.478% | -0.004586 |
| L100I1-S11 | 0.946923 | 0.947716 | +0.000793 | -3.932% | -0.000751 |
| L100I1-S12 | 0.944231 | 0.947044 | +0.002813 | -3.991% | -0.002642 |
| L100I1-S13 | 0.946576 | 0.951072 | +0.004495 | -1.916% | -0.004274 |
| L100I1-S14 | 0.945465 | 0.951309 | +0.005844 | -1.330% | -0.005186 |

11/11에서 H↑, E↓, D_s↓다. ‘처음부터100K로 학습한 모델이 fresh50K보다 낮다’와 ‘추가학습 구간이 나쁘다’는 서로 다른 명제다. 이번 관측은 후자를 지지하지 않는다.
최고RAW H와 endpoint H의 차이도100K Student에서 최대0.001586이다. selector 조정만으로0.964에 도달할 후보는 없다.

## 5. Teacher 자체의 결과

| 서버 | Teacher seed | 업데이트 | HQNR | ERGAS | D_lambda | D_s | 출처행 |
|---|---|---|---|---|---|---|---|
| s4 | 94002 | 100000 | 0.937017 | 0.558937 | 0.027099 | 0.036872 | GF2-s4!18 |
| s5 | 94003 | 100000 | 0.942321 | 0.557435 | 0.025815 | 0.032710 | GF2-s5!25 |

s4 BASE Student는 Teacher보다 H +0.016466/+0.017257, E -1.115%/-1.913%.
s5 BASE Student는 Teacher보다 H +0.005395/+0.008987, E -1.222%/-1.736%.

Student가 reference를 넘는 현상은 유지된다. 따라서 ‘KD가 전혀 작동하지 않는다’고 할 수 없다. 하지만 Teacher 자체가 높은 FR 공간 품질을 가진 reference라는 근거도 없다.
서로 다른 서버의 Teacher absolute score를 계수효과로 읽지 않는다. s3 Teacher 전체평가는 미등록이므로 해당 비교는 완료되지 않았다.

## 6. 보존할 현재 세 자산

1. **기존 공간 우수 자산:** QGBASE s3/TA/SS92001 exact50K, H0.957673/E0.556494/D_s0.023503.
2. **신규 RR 제약 내 최고 endpoint-H:** s3 R100/SS95002 BASE100K, H0.955960/E0.551502/D_s0.023873.
3. **RR 최저 자산:** s4 R100/SS95001 H010100K, H0.953660/E0.544284/D_s0.027401.

두 지표를 한꺼번에 한 줄의 최고점으로 합치지 않는다. 이들은 RR–FR 상충의 서로 다른 위치다. s4 BASE/SS95002 H0.954274/E0.548244 역시 RR 여유가 있는 로컬 continuation 출발점으로 유용하다.

현재 최고100K의 D_lambda0.020636을 유지한다는 평균곱 근사에서는 H0.964를 위해 D_s≈0.015688이 필요하다. D_s0.023873에서 약34.3% 감소다. 장면별 평균의 공분산 때문에 정확한 보장선은 아니다. H010 최저RR 자산에서는 D_s≈0.016879, 약38.4% 감소가 필요하다.

## 7. 다음은 더 큰 학습 개입을 하되, 구조 변경과 구분한다

아래는 **새 제안**이며 이미 검증된 성능 개선이나 실행 승인·queue 변경이 아니다. Teacher P0/W112D123, Student PLH/W104D122, PANGlobalAligner, MS 좌표계, 기존 hard/soft/q-edge/q-A routing은 유지한다.

### A. 공간 품질이 좋은 가중치에서 짧은 저LR 재학습 + 후기 H010

**질문:** fresh100K 경로를 다시 만드는 대신, 이미 얻은 spatial mapping을 출발점으로 RR를 더 fitting할 수 있는가? H010을 처음부터 주는 비용과 후기 도입의 효과는 다른가?

- 출발점은 사전에 지정한 정확한 checkpoint. 예: 기존 s3 S92001 exact50K와 새 s3 S95002 exact100K.
- 원 Teacher A/U와 해당 tau/q/cache 전체를 계속 사용한다. parent의 reference를 새 Teacher로 중간 교체하지 않는다.
- 각 parent에서 동일한 두 continuation fork를 만든다: FT-BASE(alpha1)와 FT-H010(alpha0.1).
- 검정용 시작값: 추가20K, U peak1e-5, A peak3e-7, beta0.1, edge0.002. A는 trainable이고 기존 routing을 유지한다.
- 두 fork에서 같은 사전 정의 LR schedule, optimizer-state 처리, sampler/view stream을 사용한다. gradient/optimizer state를 어느 쪽에서만 초기화하지 않는다.
- LR를 다시 양수로 만드는 것은 의도된 fine-tuning revision이다. exact-resume이나 기존cosine100K와 동일한 실험으로 부르지 않는다.
- parent step+추가 update를 둘 다 기록하고, parent Teacher/calibration 비용도 별도로 남긴다.

현재 H010의 나쁜 seed가50K 이전부터 공간 차이를 갖는다는 관측 때문에, 초기 학습 경로는 BASE로 만들고 hard 강조 약화는 후기에 도입하는 질문의 정보 가치가 높다. 효과는 미확정이다.

**선택 편향 주의:** 위 자산을 선택한 이유에 test HQNR 관측이 있으므로 이 continuation은 test-aware 개발 실험이다. 논문 최종 검증에서는 결정한 스케줄 전체를 새 초기화·사전고정 seed들로 재실행하며, 이미 반복 조회한 test를 untouched test라고 부르지 않는다. 같은 parent의 fork 여러 개를 독립학습 반복으로 세지 않는다.

### B. 작은 계수 sweep 대신 실제 gradient 비중을 바꾸는 큰 대조

앞서 사용자50K 진단은 soft/hard 약0.009, edge/hard 약3e-4를 보고했다. 이는 현재100K 측정치가 아니다. 따라서 먼저 고정 train/validation probe에서 raw·weighted gradient와 soft gate 비활성 비율을 측정한다.

기존 정의를 유지하는 검정값:
- **K1:** beta0.1→1.0, edge0.002 유지.
- **E100:** edge0.002→0.1, beta0.1 유지.
- 나머지 alpha, reference, initial checkpoint, LR, data stream 고정. 두 계수를 동시에 바꾸지 않는다.

동일 forward state의 단순 선형 추정으로는 위50K 비율에서 soft/hard≈0.09, edge/hard≈0.015가 된다. 학습중 실제 비율이나 parameter update가 이 숫자가 된다고 보장하지 않는다.
이 설계는 ‘점수가 안 나오니 아무 계수나 크게’가 아니라, 기존0.5–2배 변화가 사실상 hard-only 최적화의 큰 비중을 바꾸지 못했을 가능성을 검정한다.

Soft의 핵심 한계:
a_T=0인 위치에서는 beta를 아무리 키워도 soft gradient는0이다. q 역시 relative-shift response이며 FR 공간 fidelity의 정답이 아니다. 새100K에서 활성 soft mass가 거의 없다면 beta를 무제한 올리지 않고 이 축을 종료한다.

Edge 정답은 HRMS GT의 signed Scharr이지 PAN edge가 아니다. 큰edge가 D_s를 낮춘다는 보장은 없다. 반대로 양의signedD_s만으로 GT edge를 무조건 줄여야 한다고 말할 수도 없다.
Strong edge는 기존 q-edge를 사용하며 A에 직접 역전파하지 않는다.

### C. 학습에서만 PAN frequency/contrast 변화를 주는 augmentation

**새 가설:** RR 분포에서 얻은 PAN–MS 결합이 PAN detail amplitude 변화에 지나치게 민감하여 FR에서 불안정한 spatial mapping을 만들 수 있다. 현재 signed D_s와 과거 RR–FR blur 관측은 이 가설의 단서이지 원인 증명이 아니다.

예시:
P_gamma = G(P) + gamma [P-G(P)],
gamma∈{0.75,1.0,1.25},
native gamma1을50%, 양쪽 변화를각25%로 사용한다.

G는 native PAN 격자의 중심정렬 대칭 blur로 정의한다. MS와 HRMS GT는 이동·변형하지 않는다. γ<1만 사용해서 단순히 blur를 주입하는 것과 구분한다.
- 수정한PAN에서 기존 LP 생성규약으로 LPAN을 다시 만들고, 표준 synchronized warp 및 signedHP 차분을 유지한다.
- Teacher와 Student, q probe 모두 동일한 augmented pair를 봐야 한다.
- 예전 native q-cache를 새PAN에 붙여 사용하지 않는다. 고정view ID에augmentation을 포함하고 새로운 q-cache/calibration protocol을 별도 revision으로 검증한다.
- 추론은 원래 native PAN(gamma1), 원래LP, 원래full-frame 평가다. FR test PAN이나 GT를 바꾸지 않는다.
- 새로운 train-view 범위와 calibration 변경 효과를 기록하고, 이전native-training실험과 동일프로토콜이라고 부르지 않는다.
- 같은 perturbation의 frozen-c replay를 별도진단으로 사용하면 A반응과 U의 intensity/detail 반응을 분리할 수 있다. 이diagnostic을 공식inference로 채택하지 않는다.

이 축은 backbone/추론architecture를 바꾸지 않지만 단순scalar tuning보다 큰 **학습입력분포 revision**이다. 실제코드·cache무결성·추가비용을 검증한 후 실행해야 한다.

## 8. 자원 방향과 중단 기준

기존L100의 남은s3대응실험과Teacher평가는 마무리·회수하되, 추가 Teacher를 같은 recipe로 반복 생산하지 않는다. 기존reference를 활용하는Student-only실험의 정보가치가 더 높다.

| 서버 제안 | 다음 질문 | 주의 |
|---|---|---|
| s3 | 우수공간endpoint에서 FT-BASE/후기H010 | original reference 유지; test-aware 개발표시 |
| s4 | K1와E100의실효gradient 대조 | 로컬같은parent/BASE에대응; 동시에계수변경금지 |
| s5 | 학습PAN frequency-contrast augmentation | 새data-view/q-cache revision; 공식평가불변 |

이는 서버배치 제안이며 현재queue를변경한명령이아니다. s1/s2 ablation은 건드리지 않는다.

먼저한계수·한입력분포의짧은paired진단을완결하고, 적어도한번의사전정의독립조건에서도같은방향인지본뒤확장한다. 좋은seed가나올때까지단독ALT를재추첨하지않는다.
상수PANwarp·평가reference교체·출력blur로D_s만낮추는것은본방향에포함하지않는다. RR 정확도·추정정합·nativeFR지표를같이본다.

## 9. PAN-Crafter의 과거 높은 성능을 해석하는 방식

첨부 PAN-Crafter PDF p14 Table12의GF2:
- CM3A/MARs 모두없음: H0.959/E0.632/D_s0.021.
- MARs만있음: H0.945/E0.574/D_s0.032.
- 두요소모두: H0.964/E0.552/D_s0.017.

즉 PAN재구성만추가하면공간지표가항상좋아진다는결과가아니다. 논문의이baseline에는localattention이남으며현재attention-freePANDA와동일architecture도아니다. 기존s1W168DUAL도폭·입력·task·modulation이함께달랐다.
그러므로 PAN task를 GF2에만 다시넣는것을단순hyperparametertuning이라고부르지않는다. 이런변경은원한다면별도methodrevision이며현재기본추천은아니다.

## 10. Ablation 짧은 상태

WV3/s1에 TPLUS·TZERO와C00–C03까지6행이등록돼있다. C03의validation-selected H0.954402/E2.050050이고, C02 대비H+0.002555/E+0.000095다. 첫seed의Teacher A초기화효과단서지만C07FULL은아직미등록이다. QB/s2에는ABLR2완료행이없다. 전과같이시트미등록을미실행으로해석하지않는다.

## 11. 출처와 판정 범위

- live Sheet export 및 GF2-s3(5090)!B34:N42, GF2-s4!B14:N22, GF2-s5!B21:N28 native readback.
- 기존 메인 method: `PAN_Final_Method_qe_AlignmentAware_Fitting_2026-09-20_KR_v4_notation.md`, §4.3, §5.1–5.4.
- 사용자 제공50K정합·hard/gradient진단(Q1–Q3): 관측범위·추정기한계를유지한다.
- `2026-09-20_gf2-hqnr-gap-audit.md`: 과거B01/DUAL비교및RR–FRblur관측. 그문서의‘현mainline’은과거B01이지현재L100이아니다.
- 첨부 `pancrafter.pdf`, p14 Table12 및p10§C.3: 표이미지의checkmark까지확인했다.
- 저장소HEAD조회: `6bb78e6d61183a7ed1cd1da992527e7207d48ba3`. 실제원격실행중checkout을인증한것은아니다.
- `.552`는PAN-Crafter보충Table7/12의GF2 ERGAS기준이다. 본문Table2의`.522`를동일값으로고쳐합치지않는다.

**종합: H010은RR전용후보로내리고E1약화는중단한다. 100K와현재architecture의가능성은유지하되, 다음예산은우수공간해에서의후기fitting·실효supervision변경·학습입력분포강건성으로이동한다.**
