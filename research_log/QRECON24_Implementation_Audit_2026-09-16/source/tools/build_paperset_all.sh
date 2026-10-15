#!/usr/bin/env bash
# 센서별 논문 FR 세트(PanCollection .mat 형식 20장)를 받아 입력 h5 로 만든다 (KNOWN_ISSUES F-2).
#
#   ./tools/build_paperset_all.sh                 # wv3 qb gf2 wv2 전부 (이미 있으면 건너뜀)
#   ./tools/build_paperset_all.sh qb gf2          # 일부만
#   ./tools/build_paperset_all.sh --no-download   # Drive 시도 없이 이미 받은 .mat 만 (속도제한 걸렸을 때)
#
# 산출: data/PanCollection/<DS>/full_examples_mat20/test_<s>_OrigScale_mat20{,_pan}.h5 + provenance.json
# .mat 원본은 data/PanCollection/<DS>/full_examples_mat/ 에 둔다. Drive 가 막히면 브라우저로 받아 거기 두면 된다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
declare -A DRIVE=(   # PanCollection README "Testing Dataset (FullData, mat Format)"
  [wv3]=16pGIqvwWfyQVvkk3s1xrwLpavqQd0Bv7
  [qb]=1bTmEpPksnQjcyjKHJ-nORQwJkIqazml1
  [gf2]=1GMePYYHhQARDOGfeTv6cHqfxCck4XU5U
  [wv2]=1wDBTv8ZBDEkcN1rDuPOwTqTE3rwmtsAd )
NODL="${PAPERSET_NO_DOWNLOAD:-0}"; ARGS=()
for x in "$@"; do [ "$x" = "--no-download" ] && NODL=1 || ARGS+=("$x"); done
SENSORS=("${ARGS[@]}"); [ ${#SENSORS[@]} -gt 0 ] || SENSORS=(wv3 qb gf2 wv2)
for s in "${SENSORS[@]}"; do
  D="${s^^}"; H5="data/PanCollection/$D/full_examples_mat20/test_${s}_OrigScale_mat20.h5"; SRC="data/PanCollection/$D/full_examples_mat"
  if [ -f "$H5" ]; then echo "[paperset] $s: 있음 — 건너뜀"; continue; fi
  [ -d "data/PanCollection/$D" ] || { echo "[paperset] $s: data/PanCollection/$D 없음 — 건너뜀"; continue; }
  mkdir -p "$SRC"
  # ls 가 실패하면 pipefail+set -e 로 스크립트가 죽는다 — find 로 센다
  n=$(find "$SRC" -maxdepth 1 -name "Test(HxWxC)_${s}_data_fr*.mat" | wc -l)
  if [ "$n" != 20 ] && [ "$NODL" = 1 ]; then echo "!! $s: .mat $n 장, 다운로드 생략(--no-download) — 브라우저로 받아 $SRC 에 둘 것"; continue; fi
  if [ "$n" != 20 ]; then
    echo "[paperset] $s: .mat 20장 다운로드 (Drive ${DRIVE[$s]})"
    "$PY" -c "import gdown" 2>/dev/null || "$PY" -m pip install -q gdown
    mkdir -p "$SRC"; ( cd "$SRC" && "$PY" -m gdown --folder "https://drive.google.com/drive/folders/${DRIVE[$s]}" ) || true
    find "$SRC" -mindepth 2 -name 'Test(HxWxC)_*_fr*.mat' -exec mv -n {} "$SRC"/ \; 2>/dev/null || true
    n=$(find "$SRC" -maxdepth 1 -name "Test(HxWxC)_${s}_data_fr*.mat" | wc -l)
    [ "$n" = 20 ] || { echo "!! $s: .mat 가 $n 장 — 브라우저로 받아 $SRC 에 둘 것 (https://drive.google.com/drive/folders/${DRIVE[$s]})"; continue; }
  fi
  "$PY" tools/build_fr_paperset.py --src "$SRC" --sensor "$s"
done
