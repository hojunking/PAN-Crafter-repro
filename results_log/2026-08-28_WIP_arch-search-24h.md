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
A2·R6 을 꼬리에 무조건 편성했다. R5(w80)·A3(dec 전무)·L2(w96) 는 config 만
만들어 두었고 이번 큐 결과를 보고 결정한다. `N2_9_d224_noattn` 은 s2 몫이다.

학습 합계 ≈ 16.5h (case 당 1.3~2.4h) + 평가·업로드 오버헤드. LR-Fuse 는 dual MARs
그대로 학습한다(출력이 8ch 라 성립) — m1 에서 확인된 단일모드 ERGAS 열화 교란 배제.

### 구현 변경 (이 캠페인용)

- `model/pancrafter_paper.py` — `dec_depth` 옵션: decoder (full-res, H/2) 블록 수 분리,
  0 이면 skip concat 째 생략. 미지정 시 encoder 미러(기존 체크포인트 호환, c0·c6 params 불변 확인).
- `model/lr_fuse.py` — 신규. 인터페이스·잔차 계약은 기존과 동일해 train.py 무수정.
- `tools/smoke_cases.py` — 학습 전 build·forward·backward·FR 형상 검증. 체인이 case
  시작 전에 실행해 config 오류로 2h 슬롯을 태우는 것을 막는다. 15/15 통과 확인.
- gspread — `dec_depth`(dd01 표기)·LR-Fuse 계열 서술자/Notes 지원.

### 진행 기록

- 2026-08-28: 체인 기동. 이전 캠페인 로그는 `work_dir/cases_chain_lightweight-11.log` 로 보관.
