# KD·Mutual Learning 프레임워크 구현 보고서 (2026-08-31)

명세: [`s1_mutual_and_kd_implementation_spec.md`](s1_mutual_and_kd_implementation_spec.md)
· 검토에서 합의된 조정을 반영한 구현이다. **상태: 구현·검증 완료, 캠페인 미기동.**
서버 배정(사용자 확정): **s1 = KD(Part B), s2 = mutual(Part A — 2-peer 라 VRAM 큰 쪽)**.

## 0. 명세 대비 확정 변경 2가지

1. **checkpoint 선택 = 공식 HQNR 불변** (사용자 확정 — 명세 §24 의 val-ERGAS·last.pt
   정책은 채택하지 않는다). 모든 신규 trainer 가 기존 Trainer 의 평가·best 선택·export
   경로를 **상속**하므로 HQNR 선택·`best_state.json`·mat export·시트 업로드가 그대로 돈다.
2. **MTF 는 clean-room 가우시안 근사** — DLPan(GPL-3.0) 커널을 학습 경로에 넣지 않는다.
   저장소 검증된 lpan 레시피 형태(σ=1.98 기본, 밴드별 σ config 개방, `requires_grad=False`).

## 1. 파일 구성 (기존 구조 불변 — 모듈 추가만)

| 파일 | 내용 |
|---|---|
| `kd/ops.py` | MTFDownsampler · AbsoluteGradient(Sobel) · LocalVarianceMap · shift 후보 · robust/mean 정규화 · ramp_then_decay |
| `kd/losses.py` | SiS(bandwise/shared_vector + 진단) · edge · uncertainty NLL · U-Know hard/soft weight · spectral KD(L1+SAM) · mutual 3종(residual/spectral/edge) |
| `kd/features.py` | FeatureTap(hook — **모델 코드 무수정**) · FeatureProj(1×1 공통 64ch) · UncertaintyHead · WithUncertainty 래퍼 |
| `train_kd.py` | TeacherTrainer(T1·T2) · KDTrainer(K0~K5) — `Trainer` 서브클래스, `train()` 만 오버라이드 |
| `train_mutual.py` | MutualTrainer(M0~M3) — 2-peer, 같은 accelerator 에 prepare |
| `main.py` | `--trainer {default,teacher,kd,mutual}` + `teacher_config/checkpoint` + `{kd,teacher,mutual}_args`(YamlAction dict) — 미지 키 assert 통과용 선언 포함 |
| `tools/test_kd.py` | unit test 10건 (명세 §26) |
| config 12벌 | s1: T1·T2·K0~K4 (8) / s2: M0~M3 (4) — 전부 `expect_params_m` 기입 |

## 2. 핵심 설계 결정

- **서브클래스 상속으로 규약 보존**: validate/test_reduced/test_full/save_best_hqnr/
  export 를 전부 상속 → work_dir 레이아웃·`reduced_best_hqnr.mat` 완료 판정·워치독·
  러너 v2·시트 자동 업로드가 **무수정으로 동작**한다 (통합 smoke 로 관통 확인).
- **MARs dual anchor 유지** (§0.4): KD/aux/mutual 은 전부 MS half 에만, PAN half 는
  기존 hard L1. teacher 는 MS half 만 no_grad forward (연산 절약).
- **WithUncertainty 래퍼**: forward 계약(residual 반환)을 유지해 평가 경로 호환.
  θ 는 forward 직후 `theta()` (마지막 decoder feature hook). head 는 래퍼에 속해
  accelerate checkpoint 에 자동 포함 → KD 로더가 `head.` 접두로 T1/T2 를 자동 인식.
- **teacher 로딩**: `load_teacher()` 가 config+checkpoint 에서 frozen teacher 구성.
  K2+ 에 uncertainty 없는 teacher 를 주면 명시적 에러 (조용한 오구성 방지).
- **MutualTrainer**: peer_b 는 다른 init seed 로 생성, 같은 accelerator 에 prepare →
  `save_state/load_state` 가 **두 peer + 두 optimizer 를 함께** 저장·복원. 같은
  augmented batch 공유(§3.1), mutual target 은 detach(§14). best 선택은 **두 peer
  평균 HQNR**, per-peer 수치는 `[peer_a]/[peer_b]` 로그로 분리. export 는 peer_a
  기본 + `*_peerB` mat 병기. 학습 로그에 disagreement(§16 핵심 신호) 상시 기록.
- **스케줄**: KD 는 §22 (soft 5-15-40K, SiS 10-20-45K, feat 15-25-40K), mutual 은
  §13 (5K warm-up → 15K ramp → 40K plateau → decay).
- **k5 의 FeatureProj** 는 별도 param group 으로 optimizer 에 추가 (student 체크포인트
  키 레이아웃은 k0~k4 에서 기본 실행과 동일 — 가중치 비교 가능).

## 3. 검증

**unit test 10/10** (`tools/test_kd.py`):
SiS shift 회복·radius0≡L1·shared/bandwise 구분·target detach / uncertainty 양수·정규화
·고오차 영역 θ 증가(200-step 학습 검증) / **mutual gradient isolation**(peer_b grad 0)
/ feature KD 사영·teacher detach / 연산자·스케줄 / 래퍼 forward 계약·state_dict.

**통합 smoke 3종** — 실데이터 120 iter 미니 학습 → 공식 HQNR 평가 → best 저장 →
mat export 까지 전 구간 관통 (teacher/kd(k1b, c6 teacher 실로딩)/mutual(m1, 2-peer)).
config 12벌은 `smoke_cases.py`(expect_params 대조) 통과.

**gspread**: KD/mutual/teacher 실행의 run 명 suffix(`KD:k1b(T=c6_c4d124)` ·
`MUT:m1` · `+unc.head`)와 Notes(trainer·variant·teacher 경로) 확장.

## 4. 실행 안내 (캠페인 기동은 별도 승인)

```bash
# s1 (KD): T1 → T2 → K0 → K1A → K1B → [T calibration 통과 시] K2 → K3 → K4
# s2 (mutual): M0 → M1 → M2 → M3   (pull 후 cases_queue.txt + campaign_start.sh)
```
- K2~K4 의 `teacher_checkpoint` 는 기본 T1 을 가리킨다 — T1 calibration
  (θ-오차 상관, §8.3) 확인 후 필요 시 T2 로 교체.
- 예상 시간: s1 ≈ 22~25h, s2 ≈ 16~20h (명세 §30 기준) — 20h 마감이면 K4/M3 는
  게이트·잔여시간 판단.

## 5. 남은 것 / 알려진 한계

- uncertainty calibration 의 정식 통계(Spearman·quintile)는 학습 로그의 배치 수준
  진단으로 우선 대체 — 정밀 분석은 T1 완료 후 체크포인트 사후 분석 스크립트로.
- mutual 의 peer_b 는 best 선택 시점이 peer_a 와 공유된다(두 peer 평균 HQNR 의
  최적 epoch). per-peer 독립 best 는 v2 과제.
- shift-robustness 곡선 평가(§25.5)는 캠페인 결과 분석 단계에서 도구로 추가 예정.
