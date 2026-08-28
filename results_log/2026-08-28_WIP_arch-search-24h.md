# [WIP] 진행 중인 실험 — 24h 아키텍처 탐색 (2026-08-28)

> **진행 중 — 0/9.** 결과가 나오는 대로 시트에 자동 업로드된다.
> 전부 끝나면 정식 문서로 대체하고 이 파일은 지운다 (`CONVENTION.md` §2).

계획: [`research_log/2026-08-28_architecture-search-24h-plan.md`](../research_log/2026-08-28_architecture-search-24h-plan.md)
(검토·확정 사항은 계획서 §10 추기 참고). 직전 캠페인 결과:
[2026-08-28_lightweight-case-results.md](2026-08-28_lightweight-case-results.md).

**목표.** 재현 충실이 아니라 Pareto 탐색이다 — full-res stage 제거, decoder 비대칭,
d124 위에서의 width 재판정, 그리고 전 연산을 1/16 면적에서 수행하는 새 구조(LR-Fuse).

**판정.** best 선택은 기존과 동일하게 **HQNR(공식 12-19)** 이다 (계획서 §2.2 의
val-ERGAS 항은 검토에서 폐기 — 기존 11개 캠페인 수치와의 비교 가능성 유지).
HQNR·D_s 단독 개선은 채택 근거로 쓰지 않는다(축소 시 자동 개선 기전).
해석 축은 ERGAS·SAM + 실측 추론시간.

### s1 실행 큐 (9건, 전부 50K · 11ch 기본 · nocrop · LN, params 는 build 실측)

| # | 실행 | Params | 무엇이 다른가 | 확인 질문 |
|--:|---|---:|---|---|
| 1 | `N3_9_d124_noattn` | 3.770M | c6 을 9ch 입력으로 | 현 최상 Pareto(c6)가 9ch 에서도 유지되는가 |
| 2 | `R1_w128_d024_noattn` | 2.999M | depth(0,2,4) — full-res 블록 양쪽 모두 0 | full-res stage 를 완전히 제거해도 되는가 |
| 3 | `R3_w112_d124_noattn` | 2.892M | c6 폭 128→112 | 중간 폭 |
| 4 | `R4_w96_d124_noattn` | 2.129M | c6 폭 128→96 | c8(w96·d224)의 depth 교란을 걷어낸 재판정 |
| 5 | `A1_asym_114_10` | 2.703M | enc(1,1)·btl4·dec(H/2 1, full 0) | decoder 비대칭 (보수) |
| 6 | `L1_11_lr_fuse_w64` | 0.544M | 새 구조: PixelUnshuffle→저해상도 backbone | 초경량 Pareto 지점 성립 여부 (11ch 계열) |
| 7 | `L1_9_lr_fuse_w64` | 0.534M | 〃 9ch 계열 | 〃 |
| 8 | `A2_asym_014_10` | 2.407M | enc(0,1)·btl4·dec(1,0) — full-res body 전무 | 비대칭 과감판 |
| 9 | `R6_w96_d024_noattn` | 1.693M | stage 제거+폭 96 동시 | 초경량 U-Net |

계획서의 R2(d014)는 A2 와 사실상 동일 구조라 삭제하고, 그 여유로 조건부였던
A2·R6 을 꼬리에 무조건 편성했다. **R5(w80)·A3(dec ResBlock 없음)·L2(w96) 는
`tools/campaign_gate.py` 가 본 큐 결과의 ERGAS(DLPan 프로토콜)로 자동 판정해
이어 돌린다** — R5 는 R4 가 c6 와 동급(≤2.1034)일 때, A3 는 A2 ≤2.12,
L2 는 L1 최저 ≤2.25. `N2_9_d224_noattn` 은 s2 몫이다
(s2 는 `work_dir/cases_queue.txt` 에 자기 큐를 적고 같은 러너를 쓴다).

**해석 주의.**
- A1 은 c6 대비 enc H/2·dec H/2·dec full 을 동시에 줄인다 — 효과를 "decoder
  비대칭"에 단독 귀속할 수 없다. 깨끗한 대조는 **A1 vs A2 짝**(enc full 1 vs 0)이다.
- A2 는 이름과 달리 **사실상 대칭 d014** 다(enc 0,1 / dec 1,0). asym 은 계열 라벨.
- "full-res 제거"(R1)·"dec 없음"(A3) 은 ResBlock·skip 기준이다 — 입출력 conv,
  Down/Up, 출력 head 는 남는다.

학습 합계 ≈ 16.5h (case 당 1.3~2.4h) + 평가·업로드 오버헤드. LR-Fuse 는 dual MARs
그대로 학습한다(출력이 8ch 라 성립) — m1 에서 확인된 단일모드 ERGAS 열화 교란 배제.

### 구현 변경 (이 캠페인용)

- `model/pancrafter_paper.py` — `dec_depth` 옵션: decoder (full-res, H/2) 블록 수 분리,
  0 이면 skip concat 째 생략. 미지정 시 encoder 미러(기존 체크포인트 호환, c0·c6 params 불변 확인).
- `model/lr_fuse.py` — 신규. 인터페이스·잔차 계약은 기존과 동일해 train.py 무수정.
- `tools/smoke_cases.py` — 학습 전 build·forward·backward·FR 형상 + **실배치
  (batch×MARs 복제) OOM 검사**(AdamW step 포함). 체인이 case 시작 전에 실행해
  config 오류·OOM 으로 2h 슬롯을 태우는 것을 막는다. 15/15 통과 확인.
- gspread — `dec_depth`(dd01 표기)·LR-Fuse 계열 서술자/Notes 지원. FLOPs 는
  캐시 미스(신규 구조)면 자동 업로드에서도 측정해 채운다.
- 러너 v2 — 재기동 시 **resume 우선**(epoch-*/checkpoint-* mtime 최신), 완료 판정은
  reduced+full mat **둘 다**, 최종 실패는 `cases_failed.txt` 에 기록해 재기동 시
  재소모 방지, `cases_deadline.txt`(24h 마감) 지나면 새 case 시작 안 함,
  `cases_queue.txt` 로 서버별 큐 분리, 본 큐 후 `campaign_gate.py` 조건부 실행.

### 진행 기록

- 2026-08-28: 체인 기동. 이전 캠페인 로그는 `work_dir/cases_chain_lightweight-11.log` 로 보관.
- 2026-08-28 16:0x: 러너 v2 핸드오버 — N3 학습은 건드리지 않고 체인 셸만 교체.
  마감 2026-08-29T15:34:43 설정.
- 2026-08-28 18:01: **`N3_9_d124_noattn` 완료 (1/9).** DLPan: ERGAS **2.0810** ·
  SCC 0.9908 · Q2n 0.9208, HQNR(공식) **0.9475** @epoch 230.
  c6(11ch 쌍둥이: 2.0826 / 0.9908 / 0.9536) 대비 ERGAS −0.08%·SCC 동률,
  HQNR −0.64%(판정선 1.18% 미만) → **전 지표 동급. 질문 1 답: 9ch 로도 c6 급이
  유지된다** — attention 없는 최소 구조에서 11ch 의 추가 채널은 필수(load-bearing)가
  아니다. best epoch 이 230(후반)인 점은 직전 캠페인(65–150)과 다른 패턴 — 추가 관찰.
  R1 은 18:01 개시(smoke 실배치 peak 2,990MB).
- 2026-08-28 19:25: **`R1_w128_d024_noattn` 완료 (2/9, 학습 1.39h — c6 의 2.40h 대비
  42% 단축).** DLPan: ERGAS **2.1301** · SCC 0.9901 · Q2n 0.9178, HQNR(공식)
  **0.9540** @epoch 205. c6 대비 HQNR 동급(명목 +0.04%)·SCC 포화권이지만
  **ERGAS +2.28% 실손실** — 주력 게이트(≤2.12) 탈락. **질문 2 답(전반부):
  full-res ResBlock 을 완전히 빼면(d024) HQNR 는 무손실이나 ERGAS 를 d124 대비
  +2.3% 낸다.** 손실 크기가 c4 의 어텐션 제거(+2.38%)와 비슷하다 — enc/dec
  full-res 블록 1+1 이 어텐션 3개 몫의 ERGAS 를 지키고 있었다.
  best epoch 205 — 후반 패턴 지속(N3 230). R3(w112) 는 19:25 개시.
