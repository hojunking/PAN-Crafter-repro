# 경량화 case 명세 — 기존 구조 대비 변경점 (2026-08-26)

## 0. 기준 구조와 이번 변경의 요약

**기준(C0) = `s1_A1`** — 논문 충실 재구성본 + 11ch 입력 + nocrop. WV3 50K 단일 시드에서
ERGAS **2.0351**(논문 2.040 초과), HQNR 0.9493.

```
입력 11ch: [ PAN(1) | ↑LPAN(1) | PAN−↑LPAN(1) | ↑MS(8) ]
    Conv 3×3 → C=128
full-res : ResBlock ×2                          ← LayerNorm(Eq 5)·γβ 직접학습(Eq 6)
    ↓ DownConv
H/2      : ResBlock ×2 → AttnBlock(cond2_e)     ← CM3A: MS K/V + PAN K/V, k=3
    ↓ DownConv
H/4      : ResBlock ×4 → AttnBlock(cond_bot)    (bottleneck)
    ↑ UpConv (+skip)
H/2      : ResBlock ×2 → AttnBlock(cond2_d)
    ↑ UpConv (+skip)
full-res : ResBlock ×2 → LN→SiLU→Conv(zero-init)
    + 잔차 bicubic(ms,×4)
학습: MARs dual-mode (batch ×2 복제: MS mode + PAN mode), L1, 50K, seed 2025
```

이번 캠페인에서 **함께 바뀌는 공통 사항** (모든 case 에 적용):

| 항목 | 이전 | 이후 |
|---|---|---|
| **best 선택 기준** | 검증셋 ERGAS 최소 (`select_on: val`) | **HQNR 최대 (`select_on: hqnr`)** |
| 학습 로그 | 지표 나열 | 매 eval epoch `[핵심] HQNR / SCC / ERGAS` 한 줄 보장 |
| 체크포인트/산출물 | `best_val`, `reduced_best_val.mat` | `best_hqnr`, `reduced_best_hqnr.mat` |

> **선택에 쓰는 HQNR 은 공식 DLPan 프로토콜이다** — `tools/metrics/eval_fr.py` 의
> MTF/Q2n 기반 D_λ^K 와 block-UQI D_s 를 **index 12–19** 에 대해 학습 중 직접 계산한다.
> utils 의 QNR(global-UIQI, 전체 20장)은 순위를 뒤집는 것이 실측으로 확인되어
> (paper_ln 0.9265/0.9360 vs paper_ln_mlp1 0.9170/0.9388) **선택에 쓰지 않는다**
> (CSV 곡선 기록으로만 남는다). 공식 구현 로드에 실패하면 조용히 근사로 떨어지지 않고
> 즉시 중단한다.
>
> **알려진 한계.** FR 검증 split 이 없어 FR 테스트셋(12–19)으로 고른다 — no-reference 라
> GT 누출은 없지만 선택 편향은 존재한다. 사후 확정 수치는 언제나 같은 프로토콜로 다시 잰다.

---

## 1. Tier A — Teacher 한계선 (무손실 기대)

### C0 `c0_hqnr` — 비교선 재수립 · 7.1730 M (구조 변경 없음)

`s1_A1` 과 **완전히 같은 구조**를 select_on=hqnr 로 다시 돌린다. 기존 s1_A1 은
검증셋 ERGAS 로 best 를 골랐으므로, C1~C4 를 그와 비교하면 **선택 기준이 섞인다.**
같은 기준으로 뽑힌 C0 가 있어야 "무손실" 판정이 성립한다.


### C1 `c1_nopan` — CM3A 의 PAN K/V 브랜치 제거 · 6.2249 M / 150.0 G

**무엇을 지우나.** AttnBlock 3개 각각에서 다음 모듈이 사라진다 (블록당 316,032 params):

```
cond*.attn.k_pan     Conv( [x | ↑lpan] )                      148,608
cond*.attn.v_pan     Conv( [x | ↑lpan | pan | pan−↑lpan] )    150,912
cond*.attn.proj_pan  Conv2d 1×1                                16,512
```

남는 것: `Q = Conv([cond|x])`, `[K_ms|V_ms] = Conv([ms|x])` 의 **MS-only local self-attention**.
Eq (8)의 α 결합은 `x_pan` 항이 없어져 `x_attn = α1 ⊙ x_ms` 로 줄어든다.

**왜 여기부터.** ① 배포 구조에서 두 번 확인된 중립~유의 개선(−0.44%, −0.37% p=0.049),
② 11ch stem 이 `↑LPAN`·`PAN−↑LPAN` 을 이미 공급하므로 k_pan/v_pan 입력이 **shared
feature 에 이미 있는 정보의 재주입**이다.

### C2 `c2_encbtl` — C1 + decoder AttnBlock 제거 · 5.6047 M / 139.6 G

C1 에서 `cond2_d`(H/2 decoder 쪽 AttnBlock 전체: attn+MLP+α, 620,192 params)를 통째로
제거한다. forward 에서 decoder H/2 는 ResBlock ×2 만 지난다.
근거: 배포 구조에서 decoder 쪽 제거가 encoder 쪽보다 일관되게 쌌다.
**주의: 그 근거는 H/4 위치에서 나왔고 재구성본 AttnBlock 은 H/2 에 있다.**

### C3b `c3b_btl` / C3e `c3e_enc` — AttnBlock 1개만 유지 · 각 4.9845 M

C1 에서 AttnBlock 을 하나만 남긴다. **위치 대결**이다:

- **C3b**: bottleneck(H/4)만 — coarse/global 문맥 담당. 배포 구조의 `["4"]`-only(6.041M)가
  +0.42~0.74% 였는데 당시 2σ=1.32% 아래라 **판정 미확정**이었다 → 재검
- **C3e**: encoder(H/2)만 — skip 과 bottleneck 양쪽에 출력이 전달되는 위치

### C4 `c4_noattn` — AttnBlock 전부 제거 · 4.3643 M / 126.5 G

`cond2_e`·`cond_bot`·`cond2_d` 셋 다 없다. **순수 conv U-Net + MARs + 11ch.**
forward 는 ResBlock·Down/Up·skip 만 지난다. 지금까지 **한 번도 학습해본 적 없는** 극단
기준점. 논문 보충 C.3 이 U-Net 골격의 강함을 스스로 인정하고, 11ch 가 cross-modality
신호를 입력단에서 선처리한다는 것이 근거다.

---

## 2. Tier B — KD Student 후보 (손실 감수)

| case | config | 변경 (C-계열 기반) | Params | FLOPs | 예상 손실 근거 |
|---|---|---|---:|---:|---|
| C5 | `c5_c2d124` | C2 + full-res ResBlock 2→1 (enc1·dec1 각 1블록 삭제) | 5.0123 M | **100.8 G** | d124 +1.60% |
| C6 | `c6_c4d124` | C4 + 같은 depth 축소 | 3.7719 M | 87.7 G | 미지 |
| C7 | `c7_c1w96` | C1 + 전 채널 128→96 (모든 Conv/Norm/attn 폭 축소) | 3.5259 M | 85.1 G | w96 +2.53% |
| C8 | `c8_c4w96` | C4 + 폭 96. **FLOPs 71.5 G < 논문 주장 79 G** | **2.4622 M** | **71.5 G** | 최대 축소 |

Tier B 는 무손실을 기대하지 않는다 — **손실이 나야 KD 가 메울 대상이 생긴다.**
Teacher 확정 후 s2 의 KD 파이프라인과 맞물려 돌린다. **자동 체인(`_run_cases.sh`)에는
포함되지 않으며** 의도된 보류다.

- **C5**: C2 구조에서 `encoder1`·`decoder1` 의 ResBlock 을 각 2→1 로 줄인다
  (full-res 활성이 가장 비싸다 — FLOPs −38.8G). 삭제되는 모듈: `encoder1.1`, `decoder1.1`.
- **C6**: C4(AttnBlock 전무)에 같은 depth 축소를 겹친다. 남는 것은 순수 conv 몸통
  (1,2,4)-U-Net 뿐이다.
- **C7**: C1 구조의 모든 `hidden_size` 128→96. Conv·LN·CMAAA·MLP 전 층의 폭이 3/4 로
  줄고 head_dim 도 16→12 가 된다.
- **C8**: C4 + 폭 96. AttnBlock 도 PAN 브랜치도 없는 최소 구성으로,
  FLOPs 71.5 G 는 논문이 주장한 79.03 G 보다 작다.

## 3. Tier 실험 — M1 `m1_single` · 구조 동일 7.1730 M

**MARs 의 PAN back-reconstruction mode 를 학습에서 제거**한다 (`mars: ms`).
구조는 그대로이고 학습 루프가 바뀐다:

```
dual (배포본):  batch 를 2배 복제 → 앞 절반 PAN mode(switch=0), 뒤 절반 MS mode(1)
                loss = L1(PAN 복원) + L1(HRMS)
ms   (M1)   :  복제 없음, 전부 MS mode
                loss = L1(HRMS)
```

step 당 연산이 절반 → **학습 2배 가속** (Student 다벌 학습에 직접 이득).
논문 Table 16 은 w/o MARs 를 2.212 vs 2.040 으로 보고한다 — 저자의 최대 기여 주장이며,
nocrop·11ch 파이프라인에서의 재검증이다.

---

## 4. 실행 계획과 판정

- 전 case 50K · seed 2025 · 11ch · nocrop · LN. 시드 1벌 (재구성본 ERGAS 2σ=0.11%).
- **판정: HQNR 로 선택하되, HQNR 의 시드 2σ 가 1.18% 로 크다.** 후보 간 HQNR 차이가
  그 아래면 ERGAS(2σ=0.11%)로 보조 판정한다.
- 순서(이분): **C1 → C4** (양 끝) → C4 무손실이면 C3/C2 생략, C4 손실이면 C3b → C3e → C2.
  M1 은 마지막 (2.8h).

## 5. s2 재현 프로토콜 (환경 변경 검증)

s2 는 **동일 config·동일 seed(2025)** 로 같은 실험을 돌린다. 목적은 서버(GPU·드라이버)
변경이 결과를 바꾸는지의 검증이다.

- config 는 이 커밋의 `config/c*_*.yaml`, `config/m1_single.yaml` 을 그대로 쓴다. 경로는
  `./tools/setup_paths.sh --apply` 로 치환된다.
- 업로드는 `echo s2 > gspread/server.txt` 후 자동으로 `WV3-s2` 시트에 붙는다.
  s1 수치와 **같은 표에 놓지 않는다** — 비교는 시트 간 대조로 한다.
- s1 참조값 (단일 시드): A0 2.0436 / A1 2.0351 (ERGAS), A1 HQNR 0.9493.
  서버 간 차이가 재구성본 시드 2σ(0.11%) 수준이면 "환경 무관" 으로 판정한다.
