#!/usr/bin/env bash
# 실행 하나가 끝나면 그 결과만 구글시트로 올린다. 체인 스크립트에서 부른다.
#
#   ./tools/_upload.sh <실행명> [실행명 ...]
#
# 실패해도 0 을 돌려준다 — 업로드 실패로 학습 체인이 멈추면 안 된다.
# 전체를 다시 정렬하고 싶으면 gspread_upload.py 를 --replace 로 직접 부를 것.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 체인 밖에서 단독 실행될 수도 있으므로 환경을 직접 잡는다
if [ "${CONDA_DEFAULT_ENV:-}" != "pancrafter" ]; then
  # shellcheck disable=SC1091
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/knuvi/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate pancrafter
fi
[ $# -ge 1 ] || { echo "usage: $0 <실행명> [...]" >&2; exit 0; }
# Explicit M20 adapter: no v1 selector, whole legacy backlog, or optional GPU diagnostics.
LEGACY_UPLOAD=()
for M20_TAG in "$@"; do
  case "$M20_TAG" in PAKD50_QRC24_*_S52[0-9][0-9][0-9]_FRESH50_v4)
    python tools/mix20h_postrun.py "$M20_TAG" --device cuda --upload || echo "[upload] M20 evaluation/upload pending: $M20_TAG";;
    *) LEGACY_UPLOAD+=("$M20_TAG");;
  esac
done
[ ${#LEGACY_UPLOAD[@]} -gt 0 ] || exit 0
set -- "${LEGACY_UPLOAD[@]}"
LOG="$REPO/work_dir/gspread_upload.log"
{
  echo "--- $(date -Iseconds)  $* ---"
  # 논문 세트(.mat FR 20장) 평가 — 시트의 FR 열. run 의 센서에 맞는 h5 가 없으면 스크립트가 건너뛴다 (KNOWN_ISSUES F-2).
  FR_T0=$(date +%s)
  python tools/eval_fr_paperset.py "$@" 2>&1 | grep -v Warning || true
  FR_SEC=$(( $(date +%s) - FR_T0 )); NF16_N=$(printf '%s\n' "$@" | grep -c '^NF16_\|^PALS24_\|^PALSV18_' || true); [ "${NF16_N:-0}" -gt 0 ] || NF16_N=1
  # QRECON24 §7.2 / ADJ-R1 §7.2·§11: 공식 RR selector(--official; 같은 checkpoint 의 raw HQNR ≥ .9585 후보만 재추론) 는 run 뒤처리(case 경계, 다음 학습 시작 전) 에서 GPU 로 —
  # 이 run + 이 서버의 완료 QRC24 run 중 공식 선택이 안 끝난 backlog(예: s1 G23 S1234) 를 함께. 버전 감사(§8.1) 도 같은 자리. 시간은 QRC24 ledger 에 select_<run>(kind diag) 으로 따로 적는다 — Train(h) 에 섞지 않는다.
  for t in "$@"; do
    case "$t" in PAKD50_QRC24_*)
      set +e; python tools/qrecon24_postrun.py "$t" --backlog > "$REPO/work_dir/$t/results/qrecon24_postrun.log" 2>&1; rc=$?; set -e
      grep -v Warning "$REPO/work_dir/$t/results/qrecon24_postrun.log" | tail -6
      [ $rc -eq 0 ] || echo "[upload] !! qrecon24_postrun 실패 (rc=$rc): $t — work_dir/$t/results/qrecon24_postrun.log";;
    esac
  done   # 예산 ledger 가 있는 캠페인(NF16·PALS24) 의 FR 평가 시간 분담
  # 아키텍처 고정 다중 데이터셋 캠페인: WV3 학습 run 이 끝나면 WV2 zero-shot run 도 만들어 함께 올린다
  ZS=()
  for t in "$@"; do
    case "$t" in ARCH_W168_D123_DUAL_WV3_*|S1_T05_W168_D123_DUAL)
      if [ -f "$REPO/data/PanCollection/WV2/reduced_examples_h5/test_wv2_multiExm1_pan.h5" ]; then
        python tools/make_zeroshot_run.py --run "$t" --sensor wv2 2>&1 | grep -v Warning && ZS+=("${t}_zs_wv2") || echo "[upload] zero-shot 실패: $t"
      else
        echo "[upload] WV2 데이터 없음 — zero-shot 생략 (tools/setup_wv2.py)"
      fi;;
    esac
  done
  set -- "$@" "${ZS[@]}"
  # PA(A1–A3) run: 명세 §10.10·§11 진단 (교차 평가 · learned/zero/wrong-sign · known-shift 반응 · tile vs full) → results/pa_diag.json
  for t in "$@"; do
    T0=$(date +%s)                       # 이 run 의 GPU 진단 시간 (pa_diag 부터; NF16 은 ledger 에 기록한다)
    case "$t" in PA_A*|PO10_*|S2W112*|NF16_*|NA104_*|PALS24_*|PALSV18_*|PAKD50_*|B01_*|B03_*|B04_*)
      set +e; python tools/pa_diag.py --run "$t" > "$REPO/work_dir/$t/results/pa_diag.log" 2>&1; rc=$?; set -e
      grep -v Warning "$REPO/work_dir/$t/results/pa_diag.log" | tail -25
      [ $rc -eq 0 ] || echo "[upload] !! pa_diag 실패 (rc=$rc): $t — work_dir/$t/results/pa_diag.log";;
    esac
    case "$t" in PO10_*)
      # PO10 §10.3–10.4: 추가 변위 반응(64/256/512)·보간 대조 → offset_response_*.csv, interpolation_controls.json
      set +e; python tools/po10_diag.py --run "$t" > "$REPO/work_dir/$t/results/po10_diag.log" 2>&1; rc=$?; set -e
      grep -v Warning "$REPO/work_dir/$t/results/po10_diag.log" | tail -12
      [ $rc -eq 0 ] || echo "[upload] !! po10_diag 실패 (rc=$rc): $t — work_dir/$t/results/po10_diag.log";;
    NA104_*)
      # NA104 §14.3: artifact 진단(과도한 smoothing·링잉·band bias·평탄/어두운 영역 악화) 을 **주 selector(best_hqnr)** 와 보조(best_rr_val)·last 에서. 선택·판정에는 쓰지 않는다
      for CK in best_hqnr best_rr_val last; do
        [ -d "$REPO/work_dir/$t/$CK" ] || continue
        set +e; python tools/na104_diag.py --run "$t" --ckpt "$CK" --out "na104_diag_$CK" > "$REPO/work_dir/$t/results/na104_diag_$CK.log" 2>&1; rc=$?; set -e
        grep -v Warning "$REPO/work_dir/$t/results/na104_diag_$CK.log" | tail -2
        [ $rc -eq 0 ] || echo "[upload] !! na104_diag($CK) 실패 (rc=$rc): $t"
      done;;
    PALS24_*|PALSV18_*)
      # PALS24 §9 / PALSV18 V1·V4: last(정확한 50K) 에서 반응(고정 probe, 64²/256²/512²)·drift·보간 대조·native-reference stress, best_raw 와 10K/≈25K checkpoint 는 반응만. 참조 = 원 P / 고정 donor(N2 last). GPU 시간은 ledger diag_<run>
      case "$t" in PALSV18_*) PSET=palsv18; PLED=work_dir/_palsv18_budget/ledger.json;; *) PSET=pals24; PLED=work_dir/_pals24_budget/ledger.json;; esac
      set +e; python tools/po10_diag.py --run "$t" --ckpt last --out "po10_diag_last_$PSET" --probe-set "$PSET" --native-reference --ref-run PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT --ref-ckpt last > "$REPO/work_dir/$t/results/po10_diag_last_$PSET.log" 2>&1; rc=$?; set -e
      grep -v Warning "$REPO/work_dir/$t/results/po10_diag_last_$PSET.log" | tail -12
      [ $rc -eq 0 ] || echo "[upload] !! po10_diag(last, $PSET) 실패 (rc=$rc): $t — work_dir/$t/results/po10_diag_last_$PSET.log"
      for CK in best_hqnr checkpoint-10000 epoch-125; do
        [ -f "$REPO/work_dir/$t/$CK/model.safetensors" ] || continue
        set +e; python tools/po10_diag.py --run "$t" --ckpt "$CK" --out "po10_diag_${CK}_$PSET" --probe-set "$PSET" --response-only --ref-run PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT --ref-ckpt last > "$REPO/work_dir/$t/results/po10_diag_${CK}_$PSET.log" 2>&1; rc=$?; set -e
        [ $rc -eq 0 ] || echo "[upload] !! po10_diag($CK, response-only) 실패 (rc=$rc): $t"
      done
      [ -f "$PLED" ] && python tools/_ledger_update.py "$PLED" add "diag_$t" "$(python -c "print(($(( $(date +%s) - T0 )) + $(( FR_SEC / NF16_N ))) / 3600.0)")" diag "eval_fr_paperset(분담) + pa_diag + po10_diag(last: probes + native-reference; best/10K/ep125: response-only)"   # 잠금 갱신 (리뷰 P2-6)
      ;;
    NF16_*)
      # NF16 §8: 반응·closure·shortcut 대조·stress 를 **last(정확한 50K)** 에서, 참조는 native P / 고정 donor(N2 last) 로 (§8.4). 진단 GPU 시간은 ledger 에 diag_<run> 으로 더한다 (§10)
      set +e; python tools/po10_diag.py --run "$t" --ckpt last --out po10_diag_last --native-reference --ref-run PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT --ref-ckpt last > "$REPO/work_dir/$t/results/po10_diag_last.log" 2>&1; rc=$?; set -e
      grep -v Warning "$REPO/work_dir/$t/results/po10_diag_last.log" | tail -12
      [ $rc -eq 0 ] || echo "[upload] !! po10_diag(last) 실패 (rc=$rc): $t — work_dir/$t/results/po10_diag_last.log"
      python - "$t" "$(( $(date +%s) - T0 ))" "$(( FR_SEC / NF16_N ))" <<'PYEOF'
import json, os, sys, time
t, sec, fr = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); lp = "work_dir/_nf16_budget/ledger.json"
if os.path.exists(lp):
    d = json.load(open(lp)); prev = d["entries"].get(f"diag_{t}", {}); h = (sec + fr) / 3600.0 + float(prev.get("hours") or 0.0)   # 재기동 시 같은 run 을 다시 진단하면 누적한다 (검토 지적)
    d["entries"][f"diag_{t}"] = dict(kind="diag", hours=h, runs=int(prev.get("runs", 0)) + 1, note="eval_fr_paperset(분담) + pa_diag + po10_diag(last, native-reference)", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(d, open(lp, "w"), indent=1)
PYEOF
      ;;
    esac
  done
  # 구글 API 가 간헐적으로 503 을 낸다. 몇 번 다시 시도한다.
  for k in 1 2 3; do
    python gspread/gspread_upload.py "$@" 2>&1 && break
    echo "[upload] 시도 $k 실패 — 60초 후 재시도"; sleep 60
  done
} >> "$LOG" || echo "[upload] 실패 — $LOG 참고"
tail -2 "$LOG" | grep -E "업로드 완료|실패" || true
exit 0
