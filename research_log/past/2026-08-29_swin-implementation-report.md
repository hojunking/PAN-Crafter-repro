# Swin hybrid 구현 보고서 (2026-08-29)

계획: [`2026-08-29_swin-24h-plan-v2.md`](2026-08-29_swin-24h-plan-v2.md) (§6 추기 포함).
**상태: 코드 구현·검증 완료, 실험 미기동** (case 확정 대기 — config·게이트 임계값만
남았고 둘 다 config/1줄 수준 작업이다).

## 1. 무엇을 만들었나

| 파일 | 내용 | 규모 |
|---|---|---|
| `model/swin.py` | 표준 SwinBlock (W/SW-MSA·rel-pos bias·mask) 신규 | ~170줄 |
| `model/pancrafter_paper.py` | `swin_depth`(btl)·`swin_mid`(H/2 enc) 옵션 배선 | +20줄 |
| `model/lr_tinyswin.py` | LR-TinySwin (초소형, mode 주입 포함) 신규 | ~80줄 |
| `tools/verify_swin.py` | 위상·mask·국소성·grad·params·호환 검사 23건 신규 | ~190줄 |
| `tools/_run_cases.sh` | 게이트 **다중 패스** 루프, config 부재(rc=2) 분리 | 수정 |
| `tools/smoke_cases.py` | `ConfigMissing`(exit 2), `expect_params_m` 대조 | 수정 |
| `tools/campaign_gate.py` | 실패 원장 인지 (원장 case 를 다시 열지 않음) | 수정 |
| `tools/campaign_start.sh` | 캠페인 부트스트랩 (상태 리셋 함정 일괄 처리) 신규 | ~60줄 |
| `gspread/gspread_upload.py` | `sw{n}@btl/H2` 서술자·Notes, LR-TinySwin 계열 | 수정 |

외부 의존성(timm 등) 없음 — Docker 이미지 재빌드 불필요.

## 2. SwinBlock 설계 결정 (`model/swin.py`)

- **표준형 유지, mode 조건화 없음.** MARs mode 주입은 ResBlock 의 ModeModulation
  (Eq 6)이 이미 담당한다. Swin 을 순정으로 두면 "표준 Swin Transformer 를 붙였다"는
  비교 서사가 성립하고, 원안 §1.2 의 명세와도 일치한다.
- **입출력은 conv 형식 (B,C,H,W).** U-Net 사이에 끼우기 위해 내부에서만 (B,H,W,C)로
  변환한다. `swin_blocks(n, ...)` 헬퍼가 W→SW 교대 배열(짝수 index shift 0, 홀수
  window/2)을 만든다 — 원안의 "W-MSA→SW-MSA 쌍 기본 단위" 규칙.
- **SW-MSA attention mask 는 (H,W,device) 별 캐시.** 학습 16²·검증 64²·FR 128² 가
  번갈아 와도 재계산하지 않는다. 캐시는 일반 dict 라 state_dict 에 들어가지 않는다.
- **비배수 해상도는 우하단 zero-pad 후 절단.** 이 저장소의 모든 격자는 window 8 의
  배수라(아래 §6 표) 실전에서는 타지 않는 안전장치다.
- **초기화**: rel-pos bias 는 trunc_normal(0.02). `PANCrafterPaper.initialize_weights`
  의 `_basic` 은 nn.Linear 만 만지므로(qkv/proj/MLP 는 xavier — 저장소 관행과 동일)
  bias 테이블을 덮어쓰지 않음을 확인했다.
- **파라미터 산식** (dim 128·heads 4·window 8·mlp 2, 블록당):
  qkv 49,536 + proj 16,512 + MLP 65,920 + LN 512 + rel-pos (2·8−1)²·4=900
  = **133,380 (0.1334M)** — 원안 §1.3 의 "Swin 쌍 ≈0.2668M" 과 정확히 일치(실측 검증).

## 3. PANCrafterPaper 배선

- `swin_depth`: H/4 bottleneck 의 **ResBlock(+CM3A cond_bot 이 있으면 그 뒤)** 에 삽입
  — 원안 "기존 ResBlock 뒤" 명세.
- `swin_mid`: H/2 encoder 끝, **cond2_e 뒤·skip2 캡처 앞** — decoder 의 skip 연결도
  Swin 처리된 feature 를 받도록 했다 (기존 CM3A enc 위치와 동일한 관례).
- `swin_heads=4, swin_window=8, swin_mlp_ratio=2.0` 기본값 = 원안 §1.2 명세.
- **체크포인트 호환**: 기본값(0)이면 빈 ModuleList 라 파라미터·state_dict 키가 하나도
  안 생긴다. c6 3.7719M·c0 7.1730M 불변을 기계 검사로 확인 (§5-9).

## 4. LRTinySwin 설계 결정

- 구조는 원안 §5.2 그대로: unshuffle(4) → conv → **residual Swin group**(SwinBlock×n
  + 3×3 conv, 잔차 합) → conv → PixelShuffle(4) → 8ch 잔차. 잔차 기준선(↑MS)은
  trainer 가 밖에서 더하는 기존 계약 유지 — train.py 무수정.
- **원안에 없던 결정 하나: mode 주입.** SwinBlock 이 표준형이라 mode 신호가 어디에도
  없으면 dual MARs 에서 한 함수로 두 목표(MS/PAN 재구성)를 배워야 해 학습이 성립하지
  않는다. 입력 conv 직후와 Swin group 직후에 ModeModulation(각 2×C=128 params)을
  넣어 해결했다 — LR-Fuse 가 ResBlock 경유로 mode 를 받았던 것과 등가.
- 실측 params: 9ch 계열(w64·sw2) **0.1939M**, 11ch 계열 **0.2037M** — 원안 추정
  0.20~0.30M 범위. (계획 §6-8: 실전 투입 시엔 L1 과 매칭한 w128 판 ≈0.59M 권장.)

## 5. 검증 — `tools/verify_swin.py` 23건 전부 통과

shape smoke 로는 못 잡는 오류(window 위상, mask, 국소성)를 겨냥한 검사들이다.

1. `window_partition/reverse` 왕복 항등, `PixelShuffle(PixelUnshuffle(x))==x`
2. shape 보존: 16²/32²/128² + **비배수 20×20**(padding 경로) × shift 0/4
3. **impulse 국소성** — Δ출력(impulse 유무 차)이 창 안에만 있는가. 절대 출력이 아니라
   **차분**을 보는 이유: MLP bias 가 전 위치에 상수를 더해 절대 출력으로는 검사가
   공허해진다. W-MSA 는 정확히 자기 창(64픽셀), SW-MSA 는 shift 창(32픽셀) 안 확인.
4. **SW-MSA mask 양방향 검증** — 모서리 impulse 가 순환 shift 로 반대편 구획과 같은
   창에 놓이는 배치에서, mask 켜면 누설 0 · **mask 강제 해제 시엔 실제로 새는 것**까지
   확인 (한쪽만 보면 mask 가 무의미하게 통과할 수 있다).
5. PANCrafterPaper(btl/mid/둘 다) forward+backward — Swin 전 파라미터에 비영 grad.
   zero_module 함정 회피(0 파라미터 난수화 후 검사).
6. LRTinySwin — pan/ms **입력별** 비영 grad(둘 다 기여 확인), dual switch, FR 512².
7. params 검산 — SwinBlock 133,380 · c6+swin2 = 4.0387M(계획 표와 일치).
8. 체크포인트 호환 — c6/c0 params 불변, swin 키 부재.

검사 과정에서 잡힌 것은 구현 결함이 아니라 **검사 자체의 버그 2건**이었다
(mask 누설 프로브가 잘못된 사분면을 봄, c0 재구성 depth 오기) — 수정 후 전부 통과.
구현 본체는 첫 실행에서 통과했다.

## 6. 해상도·window 정합 (실전에서 padding 을 타지 않는 근거)

| 경로 | full-res | H/2 | H/4 (btl) | LR grid (TinySwin) |
|---|---:|---:|---:|---:|
| 학습 patch | 64² | 32² | 16² | 16² |
| RR 테스트 | 256² | 128² | 64² | 64² |
| FR 테스트 | 512² | 256² | 128² | 128² |

전부 window 8 의 배수. shift 4 mask 는 (H,W)별 1회 생성 후 캐시.

## 7. 운영 결함 수정 (계획 검증 패스가 잡은 것)

1. **게이트 다중 패스** — 종전 러너는 본 큐 후 게이트를 1회만 평가해, 전제가 게이트
   실행분인 2단 체인(SW4→SW6→SW8)은 구조적으로 도달 불가였다. 이제 "새로 열리는
   것이 없거나 마감 도달까지"(최대 4패스) 반복하며, 같은 목록이 반복되면 진전 없음으로
   종료한다. `campaign_gate.py` 는 실패 원장에 오른 case 를 다시 열지 않는다.
2. **config 부재 ≠ 실패** — smoke 가 config 부재를 `ConfigMissing`(exit 2)으로 구분,
   체인은 원장에 남기지 않고 그 패스만 건너뛴 뒤 DONE 줄에 `config없음(미실행)` 으로
   표시한다. s2 가 pull 전에 기동해도 캠페인이 영구 무산되지 않는다.
3. **`tools/campaign_start.sh`** — 새 캠페인 기동 시 잔존 상태 3함정(지난 마감 파일로
   전 case 즉시 스킵 / 이전 DONE 로그로 감시자 영구 정지 / 구 큐 ORDER)을 일괄
   처리한다: 큐 config 존재 선검증 → 큐 복사 → 마감 = now+N시간 → 로그 회전 → 기동
   확인. 실패 원장은 자동 삭제하지 않는다(재도전 여부는 사람이 결정).
4. **`expect_params_m`** — config 에 기대 params 를 적으면 smoke 가 실측과 0.5% 이내
   대조한다. 옵션 하나(예: `cm3a_pan_branch`) 빠뜨려 딴 모델을 학습하는 사고를
   학습 전에 잡는다.

## 8. gspread 확장 (업로드 무장애 보장)

- 서술자: `sw2@btl`, `sw2@H2` bit 추가. LR-TinySwin 은 `LR-TinySwin w64 sw2 9ch계열`.
- Notes: `swin=2@btl (표준 Swin, W→SW 교대 · h4·w8·mlp2)` 식 델타 표기,
  LR-TinySwin 계열 arch 줄. dry-run 으로 4개 조합 확인 완료.
- FLOPs 는 기존 캐시-미스 자동 측정이 새 계열에도 그대로 적용된다.

## 9. 남은 것 (case 확정 대기 — 전부 config/1줄 수준)

1. config 파일 생성 (계획 v2 §2·§3 + §6 수정: CM3A 는 `cm3a_pan_branch: false` +
   `expect_params_m: 4.3921`, 용량 대조군 d[1,2,6], LR 은 w128 판)
2. `campaign_gate.py` 에 SW 사다리·w112 게이트 항목 추가 (임계값은 §6-6 의
   0.23%/운영 게이트 구분 확정 후)
3. 양 서버 큐 파일 작성 → **`tools/campaign_start.sh --queue <파일>` 로 기동**
