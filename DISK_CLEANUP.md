# work_dir 정리 — 디스크 회수 지침

실험이 쌓이면 `work_dir/` 가 디스크를 채운다. **run 을 지우지 않고** 재생성 가능한
산출물만 걷어내는 절차다. 도구는 `tools/prune_workdir.py` 하나이고, 모든 서버가
같은 것을 돌린다.

s1 실측 (2026-09-14): **work_dir 329G → 145G**, 디스크 여유 56G → 240G.

---

## 1. 먼저 — 이게 왜 급한가

한 run 이 남기는 양이 크다. PAKD50 은 한 벌(2시간)에 **5.9G** 를 쓴다.
50h 캠페인이면 서버당 130G 다. s1 은 정리 전 여유가 56G 였고, 그대로 뒀으면
**마감 하루 전에 디스크가 차서 캠페인이 죽었을 것**이다.

캠페인을 시작하기 전에 `df -h` 로 확인하고, 여유가 캠페인 예상 산출량보다
작으면 먼저 정리한다.

---

## 2. 무엇을 지우고 무엇을 남기는가

정리 축은 **run 이 아니라 산출물 종류**다. 범주(ARCHIVED)로 run 을 통째 지우는 방식은
쓰지 않는다 — 과거 run 재평가가 실제로 일어나기 때문이다(2026-09-13 PALS24 가
09-12 결론을 뒤집은 것이 그 예). 종류로 자르면 **어느 run 이든 재평가 가능한 상태**로 남는다.

| | 대상 | s1 회수 | 정체 |
|---|---|---:|---|
| **T2** | 완료 run 의 `epoch-*` / `checkpoint-*` | 83.7G | accelerate 재개 전용. run 이 끝나면 쓸모없다 |
| **T1** | `results/*.mat` 안의 `lms`·`gt`·`pan`·`ms` | 101.1G | 데이터셋 복사본 |

**남는 것**

- `best_*/` · `last/` 체크포인트 (벌당 32M) — 재평가·donor 재사용의 원천
- `.mat` 의 `sr` · `pan_aligned` · `pan_aligned_forward_fp32` · `delta` — **run 이 실제로 만든 것 전부**
- `results/*.json`(`fr_mat20.json` 등) · `metrics.csv` · `meta/` — **판정 수치는 여기 있다. 손대지 않는다**
- `candidates/` — PAKD50 의 후보 격자는 계획상 전수 보존이라 건드리지 않는다
- `results/full|reduced/*.png` (T3) — 6.1G 뿐이라 **유지하기로 결정**(2026-09-14 사용자 결정)

### T2 의 근거

`epoch-*`/`checkpoint-*` 는 `main.py --resume` 이 읽는 재개용 스냅샷이다.
run 당 9+5 개만 남는 **rolling window** 라 임의 epoch 의 가중치를 주지도 못한다
(`tools/best_on_grid.py` 는 `metrics.csv` 만 읽으므로 영향 없다).
완료된 run 에는 죽은 무게다.

### T1 의 근거 — "정보 손실 없음" 의 정확한 뜻

`.mat` 에 존재하는 키는 **전부 8종**이고, 그중 넷이 데이터셋을 그대로 복사해 둔 것이다.
806 개 파일이 같은 테스트셋을 806 벌 복제하고 있었다.

지워도 되는 이유는 "입력이라서" 가 아니라 **"원본이 `data/PanCollection/` 의 h5 에
따로 있고 `.mat` 안의 건 그 복사본이라서"** 다. 성격이 아니라 **중복**이 근거다.
(`gt` 는 입력이 아니라 정답이지만 같은 이유로 h5 에 있다.)

세 가지가 각각 참이라 손실이 없다 — s1 에서 전부 실측 확인했다.

1. **판정 수치는 `.mat` 에 없다.** HQNR·SCC 는 `fr_mat20.json`(보유 run 118/118)·
   `metrics.csv`·시트에 있다.
2. **네 키는 h5 에서 복원된다.**

   | run | gt | lms | pan | ms |
   |---|---|---|---|---|
   | 현 프로토콜(mat20) | 비트 동일 | 비트 동일 | 비트 동일 | 비트 동일 |
   | 구 H5 세트 run | 3.05e-05 | 9.15e-05 | 9.01e-05 | 6.10e-05 |

   구 run 은 float32 저장 반올림 때문에 2047 스케일에서 1e-4 미만 차이가 난다.
   **비트 단위로는 복원되지 않는다.** 다만 h5 쪽이 clip 되지 않은 float64 원본이라
   정보량이 더 많고, 그 값으로 계산된 지표는 이미 json·csv·시트에 있다.
   즉 지우는 바이트에 h5 에 없는 내용은 없다.
3. **그 키를 읽는 코드가 없다.** `.mat` 을 읽는 18 곳이 전부 `["sr"]` 만 꺼낸다.
   시트의 RR 지표조차 `gt` 를 h5(`eval_dlpan.GT_H5`)에서 읽는다.
   유일한 예외는 `tools/make_report_figures.py` 가 정성 비교 그림을 그리려고 읽는
   `wv3_baseline`·`wv3_fixed` 의 `reduced_best_reduced.mat` **2 개**이고,
   도구의 `EXCLUDE` 에 박아 두었다.

> **T1 이후 `data/PanCollection/` 을 지우면 안 된다.** 그전에는 중복이었지만
> 이제는 유일본이다. `.mat` 하나만 들고 다른 머신에서 독립 평가하던 방식도
> h5 가 같이 있어야 한다.

---

## 3. 실행

```bash
cd <저장소>
python tools/prune_workdir.py                  # 계획만 출력 (기본값, 아무것도 지우지 않는다)
python tools/prune_workdir.py --apply          # T2 + T1 실행
python tools/prune_workdir.py --tier t2 --apply  # 단계만 따로
```

**반드시 dry-run 을 먼저 보고 제외 목록을 확인한다.** 지금 돌고 있는 run 이
제외되었는지 눈으로 본 뒤에 `--apply` 한다.

T2 는 단순 삭제라 즉시 끝난다. T1 은 `.mat` 전량을 다시 쓰므로 수십 분 걸린다
(s1: 546 파일 약 25분). 캠페인과 병행해도 되지만 I/O 가 늘어난다.

결과는 `work_dir/_prune/prune_<타임스탬프>.json` 에 남는다 — 파일별 삭제 키,
before/after 바이트, run 별 출처 h5, 제외 사유.

### 안전장치

- **기본이 dry-run.** 실제 삭제는 `--apply` 를 붙였을 때만.
- 다음 run 은 자동 제외한다 — 실행 중(`ps` 명령줄에 이름이 있음) · 1시간 내 갱신 ·
  완료 표식(`results/reduced_best_val.mat` 또는 `reduced_best_hqnr.mat`) 없음 · `_` 접두.
- T1 은 파일마다 **임시파일 작성 → 보존 키 비트 단위 검증 → `os.replace` 원자적 교체**.
  검증에 실패하면 임시파일을 지우고 **원본을 그대로 둔다**. s1 실행 546 파일 오류 0.

### 실행 후 확인

```bash
du -sh work_dir && df -h .
python - <<'PY'
from scipy.io import whosmat
import glob
print(sorted({k for p in glob.glob('work_dir/*/results/*.mat') for k,_,_ in whosmat(p)}))
PY
```

`sr`·`pan_aligned`·`pan_aligned_forward_fp32`·`delta` 만 나와야 한다
(제외된 run 에는 입력 키가 남는다 — 정상).

s1 사후 검증: 정리된 `.mat` 으로 RR 지표를 다시 계산해 정상 동작을 확인했다
(PALS24 L1E4 S1234 — ERGAS 2.0373 · SCC 0.98818, `fr_mat20.json` 의 HQNR 0.95552 는 무손상).

---

## 4. 복원이 필요하면

`.mat` 의 입력 키가 필요해지면 h5 에서 읽는다. run 이 어느 h5 를 썼는지는
`work_dir/<run>/meta/` 에 기록돼 있다(s1 기준 123/123 run 보유).

| `.mat` | 출처 h5 |
|---|---|
| `reduced_*.mat` | `data/PanCollection/<DS>/reduced_examples_h5/test_<ds>_multiExm1.h5` (`gt`·`lms`·`ms`·`pan`) |
| `full_*_mat20.mat` | `data/PanCollection/<DS>/full_examples_mat20/test_<ds>_OrigScale_mat20.h5` (`lms`·`ms`·`pan`) |
| `full_*.mat` (구 run) | `data/PanCollection/<DS>/full_examples_h5/test_<ds>_OrigScale_multiExm1.h5` |

FR 세트에는 `gt` 가 없다 — 그 해상도의 정답 영상은 존재하지 않는다.
FR 판정에 no-reference 지표(HQNR)를 쓰는 이유다.

`sr` 자체가 필요하면 체크포인트에서 다시 만든다:

```bash
python tools/eval_fr_paperset.py <run>     # fr_mat20.json + full_<ckpt>_mat20.mat 재생성
```

---

## 5. 평상시 습관

- 캠페인 **시작 전** `df -h` 확인. 여유 < 예상 산출량이면 먼저 정리한다.
- 캠페인이 끝나면 그 서버에서 `--tier t2 --apply` 를 한 번 돌린다(즉시 끝나고 회수량이 크다).
- T1 은 여유가 부족할 때만 돌려도 된다. 다만 한 번 돌려두면 이후 run 은 여전히
  입력 키를 갖고 저장되므로 **주기적으로 반복**해야 한다.
- `data/PanCollection/` · `assets/` · `results_log/` 는 정리 대상이 아니다.

관련: `SETUP.md`(데이터 배치) · `KNOWN_ISSUES.md` F-1·F-2·F-3(데이터 결함) ·
`results_log/CONVENTION.md`(보고서 규약).
