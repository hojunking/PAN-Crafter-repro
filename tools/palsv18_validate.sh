#!/usr/bin/env bash
# PALSV18 V-pre / V-post 실행기 — best_raw(best_hqnr)·last 에 V1(po10_diag palsv18 probe) + V2–V4(palsv18_validate.py). GPU 시간은 잠금 ledger 에 vpre/vpost 로, 시작 전 예산 guard (리뷰 P2-6).
#   ./tools/palsv18_validate.sh pre    # 계획 §4.2 V0/V1/V2: 기존 11 대조군 best·last 의 V1+V2(+shortcut) — 학습 시작 **전** (예약 2.0h; vpre_reserved 로 미리 잡는다)
#   ./tools/palsv18_validate.sh post   # 신규 run(완료분) + 대조군의 V3·V4 (예약 3.5h)
#   ./tools/palsv18_validate.sh run <run> [ckpt...] [--parts v2,v3,v4]
# 재검증 생략은 results/palsv18_<ck>.json 의 checkpoint sha·tool_version·완료 parts 를 모두 확인할 때만 (리뷰 P2-3).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python; DONOR=PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT; LEDGER=work_dir/_palsv18_budget/ledger.json; LU="$PY tools/_ledger_update.py $LEDGER"
CONTROLS="NF16_P0_W112_D123_WV3_S1234_N2LAST_v1 NF16_P2_W112_D123_WV3_S1234_N2LAST_v1 PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1
          PALS24_CTRLP0_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S7777_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1
          PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L000_W112_D123_WV3_S2025_N2LAST_R200_v1 PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1 NF16_P3_W112_D123_WV3_S1234_N2LAST_v1 $DONOR"
MODE="${1:-pre}"; shift || true; PARTS_OVERRIDE=""
case "$MODE" in
  pre)  RUNS="$CONTROLS"; CKPTS="best_hqnr last"; KEY=vpre; PARTS="v2"; EST=0.07; RESERVE_PLAN=2.0;;
  post) NEW=$(ls -d work_dir/PALSV18_*_v1 2>/dev/null | xargs -n1 basename | while read -r r; do [ -f "work_dir/$r/results/reduced_best_hqnr.mat" ] && echo "$r"; done | tr '\n' ' '); RUNS="$NEW $CONTROLS"; CKPTS="best_hqnr last"; KEY=vpost; PARTS="v2,v3,v4"; EST=0.35; RESERVE_PLAN=3.5;;
  run)  RUNS="$1"; shift; CKPTS=""; while [ $# -gt 0 ]; do case "$1" in --parts) PARTS_OVERRIDE="$2"; shift 2;; *) CKPTS="$CKPTS $1"; shift;; esac; done; CKPTS="${CKPTS:-best_hqnr last}"; KEY=vpost; PARTS="${PARTS_OVERRIDE:-v2,v3,v4}"; EST=0.35; RESERVE_PLAN=0;;
  *) echo "usage: $0 pre|post|run <run> [ckpt...] [--parts …]" >&2; exit 2;;
esac
# 예약 등록: 아직 쓰지 않은 V 예산을 ledger 에 reserved 로 잡아 학습 gate 가 보게 한다 (리뷰 P2-6)
if [ "$RESERVE_PLAN" != "0" ]; then
  DONE_H=$($LU get | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['entries'].get('$KEY',{}).get('hours',0.0))")
  $LU set "${KEY}_reserved" "{\"kind\": \"reserved\", \"hours\": $($PY -c "print(max(0.0, $RESERVE_PLAN - $DONE_H))"), \"note\": \"미수행 $KEY 예약 (계획 §4.1 $RESERVE_PLAN h) — 실측이 쌓이면 줄인다\"}"
fi
guard() {   # used(예약 포함) + 이 checkpoint 예상 + 남은 학습 예약(1.7 × 미완 run) + report 0.5 + buffer 1.0 ≤ 18 — 아니면 시작하지 않는다 (계획 §4.3)
  $PY - "$1" <<'PYEOF'
import json, os, subprocess, sys
est = float(sys.argv[1]); d = json.loads(subprocess.run([sys.executable, "tools/_ledger_update.py", "work_dir/_palsv18_budget/ledger.json", "get"], capture_output=True, text=True).stdout)
q = [l.strip() for l in open("config/queues/palsv18_s1.txt") if l.strip() and not l.startswith("#")]
unfinished = [r for r in q if not (os.path.exists(f"work_dir/{r}/last_meta.json") and json.load(open(f"work_dir/{r}/last_meta.json")).get("step") == 50000)]
need = d["used"] + est + 1.7 * len(unfinished) + 1.5
print(f"   guard: used {d['used']:.2f} + est {est} + train {1.7 * len(unfinished):.1f} ({len(unfinished)} 미완) + report/buffer 1.5 = {need:.2f} ≤ 18 → {'OK' if need <= 18.0 else 'STOP'}")
sys.exit(0 if need <= 18.0 else 3)
PYEOF
}
need_run() {  # 0 = 돌려야 함, 1 = 완료(같은 sha·같은 도구·parts 포함) — 리뷰 P2-3
  $PY - "$1" "$2" "$3" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.getcwd()); from tools.palsv18_validate import TOOL_VERSION
from kdv.teacher_assets import sha256_file
run, ck, parts = sys.argv[1], sys.argv[2], set(sys.argv[3].split(",")); p = f"work_dir/{run}/results/palsv18_{ck}.json"
if not os.path.exists(p): sys.exit(0)
j = json.load(open(p)); sha = sha256_file(f"work_dir/{run}/{ck}/model.safetensors")
sys.exit(1 if (j.get("ckpt_sha256") == sha and j.get("tool_version") == TOOL_VERSION and parts <= set(j.get("parts", []))) else 0)
PYEOF
}
need_v1() {
  $PY - "$1" "$2" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.getcwd()); from kdv.teacher_assets import sha256_file
run, ck = sys.argv[1], sys.argv[2]; p = f"work_dir/{run}/results/po10_diag_{ck}_palsv18.json"
if not os.path.exists(p): sys.exit(0)
j = json.load(open(p)); sys.exit(1 if j.get("ckpt_sha256") == sha256_file(f"work_dir/{run}/{ck}/model.safetensors") and j.get("probe_set") == "palsv18" else 0)
PYEOF
}
T0=$(date +%s); FAILS=""; STOPPED=""
for r in $RUNS; do
  for ck in $CKPTS; do
    [ -f "work_dir/$r/$ck/model.safetensors" ] || { echo "[v] $r/$ck 없음 — 생략"; continue; }
    [ "$r" = "$DONOR" ] && [ "$ck" = "best_hqnr" ] && continue                        # N2 는 exact50K last 만 (계획 §6.1)
    P="$PARTS"; [ "$r" = "$DONOR" ] && P=$(echo "$P" | sed 's/v4//; s/,,/,/; s/,$//; s/^,//')   # donor 는 stress 참조 자체라 V4 생략
    mkdir -p "work_dir/$r/palsv18"; L="work_dir/$r/palsv18/${ck}.log"
    NV1=0; need_v1 "$r" "$ck" || NV1=1; NV=0; [ -n "$P" ] && { need_run "$r" "$ck" "$P" || NV=1; }
    [ $NV1 -eq 1 ] && [ $NV -eq 1 ] && { echo "[v] $r / $ck 완료(같은 sha·도구·parts) — 생략"; continue; }
    guard "$EST" || { echo "[v] 예산 guard STOP — $r/$ck 부터 미수행 (18h 상한)"; STOPPED="$r/$ck"; break 2; }
    echo "[v] $r / $ck  $(date +%H:%M:%S)  parts=${P:-V1만}"; TC=$(date +%s)
    if [ $NV1 -eq 0 ]; then
      "$PY" tools/po10_diag.py --run "$r" --ckpt "$ck" --out "po10_diag_${ck}_palsv18" --probe-set palsv18 --response-only --ref-run "$DONOR" --ref-ckpt last > "$L" 2>&1 || { echo "  !! V1 실패 ($r/$ck) — $L"; FAILS="$FAILS $r/$ck:V1"; }
      grep "fr512\|aligner 없음" "$L" | head -1 | cut -c1-200
    fi
    if [ $NV -eq 0 ]; then
      "$PY" tools/palsv18_validate.py --run "$r" --ckpt "$ck" --parts "$P" --stress-dirs 8 >> "$L" 2>&1 || { echo "  !! V2–V4 실패 ($r/$ck) — $L"; FAILS="$FAILS $r/$ck:V234"; }
      grep "^  V[234]" "$L" | cut -c1-220
    fi
    H=$($PY -c "print(($(date +%s) - $TC) / 3600.0)"); $LU add "$KEY" "$H" diag "$MODE: V1 + $P (best_raw·last; 잠금 갱신)" > /dev/null; EST=$H
    if [ "$RESERVE_PLAN" != "0" ]; then DONE_H=$($LU get | $PY -c "import json,sys; print(json.load(sys.stdin)['entries'].get('$KEY',{}).get('hours',0.0))"); $LU set "${KEY}_reserved" "{\"hours\": $($PY -c "print(max(0.0, $RESERVE_PLAN - $DONE_H))")}" > /dev/null; fi
  done
done
[ "$RESERVE_PLAN" != "0" ] && [ -z "$STOPPED" ] && $LU set "${KEY}_reserved" '{"hours": 0.0, "note": "완료 — 실측은 '"$KEY"' 항목"}' > /dev/null
SEC=$(( $(date +%s) - T0 )); echo "[v] ledger: $($LU get | cut -c1-160)"
[ -n "$STOPPED" ] && { echo "[v] 예산 상한으로 중단: $STOPPED 이후 미수행"; exit 4; }
[ -z "$FAILS" ] && echo "[v] 전부 완료 ($SEC s)" || { echo "[v] 실패:$FAILS"; exit 1; }
