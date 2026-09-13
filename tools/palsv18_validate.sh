#!/usr/bin/env bash
# PALSV18 V-pre / V-post 실행기 — 대조군·신규 run 의 best_raw(best_hqnr)·last 에 V1(po10_diag palsv18 probe) + V2–V4(palsv18_validate.py). GPU 시간은 ledger 에 vpre/vpost 로 기록.
#   ./tools/palsv18_validate.sh pre            # 기존 9 대조군 + N2 last + L1E2 (학습과 겹쳐 돌린다: aligner 전용 추론 + FR 20장 재추론, 가볍다)
#   ./tools/palsv18_validate.sh post           # 신규 6 run (완료된 것만) — 체인 DONE 뒤
#   ./tools/palsv18_validate.sh run <run> [ckpt...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python; DONOR=PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT; LEDGER=work_dir/_palsv18_budget/ledger.json
MODE="${1:-pre}"; shift || true
case "$MODE" in
  pre)  RUNS="NF16_P0_W112_D123_WV3_S1234_N2LAST_v1 NF16_P2_W112_D123_WV3_S1234_N2LAST_v1 PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1
             PALS24_CTRLP0_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1
             PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
             NF16_P3_W112_D123_WV3_S1234_N2LAST_v1 $DONOR"; CKPTS="best_hqnr last"; KEY=vpre;;
  post) RUNS="$(ls -d work_dir/PALSV18_*_v1 2>/dev/null | xargs -n1 basename | while read -r r; do [ -f "work_dir/$r/results/reduced_best_hqnr.mat" ] && echo "$r"; done | tr '\n' ' ')
             NF16_P0_W112_D123_WV3_S1234_N2LAST_v1 NF16_P2_W112_D123_WV3_S1234_N2LAST_v1 PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1
             PALS24_CTRLP0_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1
             PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1 NF16_P3_W112_D123_WV3_S1234_N2LAST_v1 $DONOR"; CKPTS="best_hqnr last"; KEY=vpost;;
  run)  RUNS="$1"; shift; CKPTS="${*:-best_hqnr last}"; KEY=vpost;;
  *) echo "usage: $0 pre|post|run <run> [ckpt...]" >&2; exit 2;;
esac
T0=$(date +%s); FAILS=""
for r in $RUNS; do
  for ck in $CKPTS; do
    [ -f "work_dir/$r/$ck/model.safetensors" ] || { echo "[v] $r/$ck 없음 — 생략"; continue; }
    [ "$r" = "$DONOR" ] && [ "$ck" = "best_hqnr" ] && continue                        # N2 는 exact50K last 만 (계획 §6.1)
    mkdir -p "work_dir/$r/palsv18"; L="work_dir/$r/palsv18/${ck}.log"; echo "[v] $r / $ck  $(date +%H:%M:%S)"
    # V1: signed 2D 반응 (palsv18 probe, 장면별 fit, donor 대비 drift) — 기존 pals24 산출물(po10_diag_<ck>_pals24.json) 은 그대로 두고 별도 이름
    if [ ! -f "work_dir/$r/results/po10_diag_${ck}_palsv18.json" ]; then
      "$PY" tools/po10_diag.py --run "$r" --ckpt "$ck" --out "po10_diag_${ck}_palsv18" --probe-set palsv18 --response-only --ref-run "$DONOR" --ref-ckpt last > "$L" 2>&1 || { echo "  !! V1 실패 ($r/$ck) — $L"; FAILS="$FAILS $r/$ck:V1"; }
      grep "fr512\|aligner 없음" "$L" | head -1 | cut -c1-200
    fi
    # V2–V4: pre 는 best_raw 만 (약 12 min/ckpt; 예약 2.0h 안) — last 의 V2–V4 는 post 에서 (§9.4/§10.2 우선순위: 세 seed best_raw 전수 → exact50K)
    PARTS="v2,v3,v4"; [ "$MODE" = "pre" ] && [ "$ck" = "last" ] && PARTS=""
    if [ -n "$PARTS" ] && [ ! -f "work_dir/$r/results/palsv18_${ck}.json" ]; then
      "$PY" tools/palsv18_validate.py --run "$r" --ckpt "$ck" --parts "$PARTS" --stress-dirs 4 >> "$L" 2>&1 || { echo "  !! V2–V4 실패 ($r/$ck) — $L"; FAILS="$FAILS $r/$ck:V234"; }
      grep "^  V[234]" "$L" | cut -c1-220
    fi
  done
done
SEC=$(( $(date +%s) - T0 ))
"$PY" - "$KEY" "$SEC" "$RUNS" <<'PYEOF'
import json, os, sys, time
key, sec, runs = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(); lp = "work_dir/_palsv18_budget/ledger.json"
if os.path.exists(lp):
    d = json.load(open(lp)); prev = d["entries"].get(key, {}); h = sec / 3600.0 + float(prev.get("hours") or 0.0)
    d["entries"][key] = dict(kind="diag", hours=h, runs=int(prev.get("runs", 0)) + 1, n_checkpoints_note=f"{len(runs)} run(s) this call", note=("V-pre: 기존 대조군 V1/V2/V3/V4 (학습과 겹쳐 돌린 벽시계 시간 전부 계상; 예약 2.0h)" if key == "vpre" else "V-post: 신규 run V1–V4 (예약 3.5h)"), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(d, open(lp, "w"), indent=1); print(f"[v] ledger {key}: +{sec / 3600:.2f} h (누적 {h:.2f} h)")
PYEOF
[ -z "$FAILS" ] && echo "[v] 전부 완료 ($SEC s)" || { echo "[v] 실패:$FAILS"; exit 1; }
