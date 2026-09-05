# UVS-KD 30h 계획 — 검토와 구현 보고 (2026-09-06, s2 실행용)

대상: [2026-09-06_uvs-kd_30h_experiment-plan.md](2026-09-06_uvs-kd_30h_experiment-plan.md)
구현: `uvs/{shift,losses}.py` · `feeders/feeder_uvs.py` · `train_uvs.py` · `tools/{uvs_prepare_teacher,uvs_build_cache,uvs_controlled_shift,test_uvs}.py`
· `tools/uvs_prepare.sh`(s2 원샷) · `config/UVS_*.yaml` 10벌 · `config/queues/s2_uvs_kd.txt` · `tools/campaign_gate.py::gate_uvs`
판정: best checkpoint HQNR(12-19, 장면별 평균) → fSCC (main.py 공통) + controlled-shift AUC(§13.2 도구).

---

## 0. 요약

- 구현 완료. 단위 검사 12/12. teacher 준비(T_shift·T_unc·gate)·cache·smoke·통합은 s1 의 `c0_hqnr`/`d122` 로 로컬 검증 (§5).
- **s2 에서 할 일은 한 줄**: `git pull && ./tools/uvs_prepare.sh c0_hqnr 30` — teacher 준비 → gate → cache → 큐 기동(B0·K0·K1·K2),
  이후 `gate_uvs` 가 teacher shift gate PASS 면 S0·M1·M2 를, M2 결과에 따라 R1(seed)·C1(w96) 또는 M3 를 연다.
- 계획과 다르게 한 것 4가지 (§2): PAN 채널 warp 방식, shift 입력 격자, teacher 입력 경로, warp 보간. 전부 지난 두 캠페인에서
  실측한 phase/aliasing 문제 때문이고, 각각 수치 근거를 적었다.
- 계획의 전제 중 저장소 데이터와 어긋나는 것 1가지 (§1.2): train patch 의 shift 는 우리 audit 로는 추정 노이즈다. pseudo-label 항은
  q_A 로 약화되고, T_shift 의 실질 신호는 합성 translation 이다. gate(§5.4)가 그 결과를 판정한다.

---

## 1. 전제 확인

| 전제 | 확인 |
|---|---|
| teacher `c0_hqnr` (w128 d[2,2,4] CM3A 3, 7.173M, 5.21h, HQNR 0.9542) | s1 work_dir 에 있음 (best_hqnr 0.9542@115). s2 도 같은 이름·config 로 학습돼 있음(시트) |
| student `d122` (w128 d[1,2,2] attn 0, 3.1795M, 2.29h, HQNR 0.9539), `d122_w96`(1.79M, 1.94h) | 있음 |
| §1.1 "train patch shift P99 2.3–2.75 LR px" (s2 audit) | **s1 audit(`outputs/global_shift_cache`)와 어긋남**: 16² patch 에서 추정기 오차 0.27 px > 추정 shift(p50 0.09), 부호 55:45. 즉 train patch 의 Δ 는 대부분 노이즈. 계획 §5.1 의 pseudo-label 항은 q_A(accepted·margin)로 가중하되 T_shift 의 신호는 합성 translation(§5.1 B)에서 온다. 참조된 `2026-09-03_alignment-audit-s2-detail.md` 는 이 저장소에 없다 |
| §1.2 bicubic phase (−0.5,−0.5) | 확인됨(2026-09-04). 단 W152 에서 lms/interp23tap 입력 파이프라인(P0)은 anchor 대비 HQNR 이득이 없었고 후반 plateau 가 −0.0024 낮았다 → **이 캠페인의 절대 HQNR 은 시트의 d122(0.9539, bicubic) 와 직접 비교하지 말고 B0 기준으로만** 비교 |
| 좌표 규약 | W(I,δ)(y,x)=I(y+δy,x+δx), δ_{MS←PAN}, LR 1px = PAN 4px — T-U01/U02/U04 로 검증. audit cache Δ(P←M) 와는 δ=−Δ |

## 2. 계획과 다르게 한 것 (이유)

| 항목 | 계획 | 구현 | 근거 |
|---|---|---|---|
| shift 적용 후 PAN 채널 | P^a=W(P,4δ) 뒤 LP(P^a)=↑MTF↓(P^a) 재계산 | **[P, LP, P−LP] 세 채널을 같은 4δ 로 강체 warp** | FR 512² 실측: LP(W(P)) vs W(LP(P)) 차 MAD ≈1e-2(HF 채널 sd 0.105 의 10%), **정수 4px 이동에서도** 남음 → MTF↓ 데시메이션 aliasing. 또 증강 HR PAN 을 다시 MTF↓ 하면 [2::4] phase 가 flip 비대칭이라 표본 75% 에서 1px 오정렬(2026-09-05 버그 재발). 강체 warp 는 translation 에 정확·phase 무관 |
| shift 모듈의 P_LR 입력 | MTFDown(P) | **feeder 가 주는 lpan** (데이터셋 MTF↓ 레시피, LR 증강과 정합) | 같은 이유(phase). T-U12 로 flip 표본에서 δ 변환 정확 확인 |
| teacher 입력 | teacher 도 lms 사용 | **teacher 는 native(내부 bicubic) 경로**, student 만 lms | teacher 는 bicubic 입력으로 학습됐다. 입력 분포를 바꾸면 OOD(anchor 추론 sweep 에서 확인). Y_T 자체는 GT 정렬 출력이므로 R_T=Y_T−LMS 는 그대로 유효. 계획 §5.3-5 의 "필요 시 fine-tune" 은 하지 않음 |
| warp 보간 | bilinear | **bicubic** (config `shift.warp_mode` 로 전환) | bilinear 은 PAN 고주파를 저역통과한다 — PAN 의 존재 이유가 HF. 저장소의 다른 warp 와도 일치 |
| T_unc head | softplus θ | logvar head s (θ=exp s), warm start 전역 상수 | 기존 `calibrate_head.py` 와 동일(수치 안정). U_T 정규화는 계획대로 θ 의 Q10/Q90 |
| cache 증강 | 영상과 같이 flip/rot | 동일 (feeder 내부). **근사**: cache 는 원본 patch 의 teacher 출력이고 CNN 은 flip/rot 에 정확히 등변이 아니다 | 계획이 수용한 근사. 벡터 규칙은 T-U05 impulse 로 검증 |
| controlled-shift 목표 | "Student shift MAE" | RR 고유 shift≈0 가정 → 목표 δ=−a, MAE=‖δ_S+a‖ | 부호 규약 명시 |
| 산출물 이름 | `best_hqnr.pt` 등 | 저장소 규약(`best_hqnr/`, `metrics.csv`, `best_hqnr_meta.json`) | 기존 도구 호환 |

## 3. 구현 요지

```
feeder_uvs   : 원 feeder 증강 + cache(R_T,U_T) 를 같은 flip/rot, δ_T 는 벡터 규칙, c_T 불변. b0 는 0 채움
train_uvs    : x_ms = [W(P),W(LP),W(P−LP), LMS] (δ_use), x_pan = [P,LP,P−LP,LMS] raw; y = LMS + R_S; p̂ = LP·rep + R_PAN
   b0/k0/k1/k2 : δ=0.  L = hard + λ_s·soft + λ_pan·L_PAN  (k1/k2 는 (1±U_T) routing, k2 는 w_V)
   s0/m1/m2/m3 : δ_use = η·sg(ĉ_T δ_T) + (1−η)·ĉ_S δ_S (s0/m1 은 η=0), L += λ_δ(L_vec + 0.1 L_conf) [+ λ_w L_warp]
   추론        : δ = ĉ_S δ_S, c_S<0.35 → 0.  FR 12-19 에서 δ_S vs audit(−Δ) corr/medErr 를 metrics.csv(g1_* 열 재사용)에 기록
uvs_prepare_teacher : T_shift(16,32,32) 8K — q_A·SmoothL1(δ,δ_A) + SmoothL1(δ(aug), δ_A+a), a ~ 0.2·0/0.6·U1/0.2·U3
                      T_unc head-only 5K (aligned PAN 입력, NLL).  gate §5.4 → gate.json, θ/V_GT 분위수 → norm.json
uvs_build_cache     : train 9714 표본 R_T(fp16) U_T(fp16) δ_T c_T (+applied) → outputs/uvs_cache/<teacher>_wv3_train.npz (~0.7 GB)
gate_uvs            : teacher shift gate → S0/M1/M2 개방; M2 ≥ max(K2,B0)−1e-4 & ≥ M1 → R1(seed) → C1(w96); 아니면 M3
```

## 4. 실행 전 알게 된 것 / 리스크
- audit pseudo-label 이 노이즈라 T_shift 의 FR-audit 일치 gate(≤0.20 LR px)가 빡빡할 수 있다. 실패 시 계획 §10.1 대로 shift 계열은 닫히고
  K 계열만 돈다 (gate 가 자동 집행).
- G1(s1)과 같은 soft-argmax 수축: 49 후보·τ0.07 에서 매끈한 patch 는 conf 가 낮다 → 추론 gate(0.35)가 identity 로 보호.
- teacher forcing 초기(η=1)에는 student 가 정렬된 입력만 본다 — 20K 이후 student δ 로 넘어갈 때 곡선이 꺾이는지 로그 `eta/|use|/MAE(S,T)` 로 확인.

## 5. 검증 (아래 갱신)
- `tools/test_uvs.py` 12/12 (부호·4× scale·identity center·cost-volume argmax·cache 벡터 변환 impulse·강체 warp·w_V 평균 1·routing stop-grad·shift KD·η 스케줄·gate·flip 격자).

## 6. teacher 준비에서 알게 된 것 (s1 `c0_hqnr`, 2026-09-06 03:xx)

**T_shift 는 정확하다.** 합성 gate: identity MAE 0.000 · |a|≤1 MAE 0.052 (≤0.12) · 전체 0.074 (≤0.25) · **FR 12-19 예측 vs audit(−Δ) medErr 0.075 LR px (≤0.20), conf 0.65.**
RR 20장에서도 예측 δ_T ≈ (−0.07, +0.03)(0–13) / (+0.03, −0.02)(14–19) 가 독립 audit 과 ~0.02 px 안에서 일치한다.

**그 정확한 δ 로 PAN 을 정렬해 frozen teacher 에 넣으면 RR 이 20장면 전부 나빠진다** — ERGAS 2.0586 → 2.1862 (**+6.2%**), audit δ 로는 +12.1%,
장면별 +1.2~+9.6%. 계획 §5.4 no-harm gate(≤0.5%)는 통과 불가다. 해석: raw 쌍으로 학습된 모델은 그 오정렬을 이미 내부에서 보상하고 있어
입력 정렬이 OOD 가 된다 — GA 캠페인의 anchor 추론 sweep(α↑ → HQNR↓)·P0 에 추론 shift 를 준 결과와 같은 현상이고, 이번엔 GT 기준 RR 로도 확인됐다.

조치: `--teacher-input raw`(기본) — **cache 의 Y_T 는 raw PAN 으로** 만든다(soft target 품질 = 원 teacher). δ_T/c_T 는 T_shift 에서 그대로 온다.
`pass_shift` 는 shift 정확도 4항목으로 판정하고 rr_noharm 은 정보로만 기록(gate.json). 계획 §5.3 의 "MS mode 는 aligned PAN" 은 채택하지 않는다.
학생 쪽 가설(S0/M1/M2: student 가 자기 δ 로 PAN 을 warp 하며 **처음부터** 학습)은 그대로 검증한다 — frozen 모델에 해로웠던 것이
warp-in-the-loop 학습에서도 해로운지가 이 캠페인의 핵심 질문이 됐다.

### 5.1 로컬 검증 결과 (s1, `c0_hqnr` → `d122`)
- teacher 준비: T_shift 8K — identity MAE 0.000 · |a|≤1 0.052 · 전체 0.074 · FR 12-19 audit 대비 medErr **0.075** LR px (conf 0.65) → shift gate PASS.
  T_unc 5K(raw 입력) — val Spearman(θ, err) **0.92** → PASS. θ Q10/Q90 = 7.4e-5 / 2.4e-3, V_GT Q10/Q90 = 2.8e-4 / 2.5e-2 (norm.json).
- cache: 9,714 표본, 683 MB, c_T 평균 0.67, |δ_T| p50 0.062 LR px, U_T 평균 0.29, gate 0 비율 1.6%. (npz 는 git 제외 — s2 는 `tools/uvs_prepare.sh` 로 생성)
- smoke 10/10 (d122 peak 6.5 GB, w96 4.9 GB, FR 512² 추론).
- 통합 210 iter b0·k2·s0·m2 완주(rc=0, mat 2종, `best_hqnr_meta.json`). 로그: k2 `U 0.29 wVmax 1.9`, s0/m2 `shift/eta/MAE(S,T)/cS/cT/|dS|/|use|/pB`,
  FR `[shift] |δS| conf gated corr medErr`. m2 는 η 스케줄(통합용 60→150)이 1→0 으로 내려가는 것을 확인. 210 iter 시점 student conf 0.04 →
  추론 gate 가 identity 로 보호(가 1.00) — 50K 에서 conf 가 오르는지가 §14.1 판정 항목.

## 7. s2 인계
```
git pull
./tools/uvs_prepare.sh c0_hqnr 30
```
순서: 단위검사 → T_shift/T_unc + gate(gate.json) → cache(npz, 0.7 GB) → smoke → 큐 기동(B0·K0·K1·K2). 본 큐 뒤 `gate_uvs` 가 shift gate PASS 면
S0·M1·M2 를, M2 가 max(K2,B0)−1e-4 이상이고 M1 이상이면 R1(seed 1234) → (재현 시) C1(w96), 아니면 M3 를 연다.
완주 후 `python tools/uvs_controlled_shift.py --run work_dir/<run>` (controlled-shift 곡선·AUC·FR audit·confidence bin).
`gspread/server.txt` 가 `s2` 인지 확인. 시트에는 UVS 범주(⑭) 구분행과 case 별 질문·세팅·판정 Notes 가 자동으로 들어간다.
