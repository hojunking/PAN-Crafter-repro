#!/usr/bin/env bash
# PanCollection 배포 zip + 저자 pan_h5.zip 을 REPRODUCTION.md 1절의 배치로 만든다.
# 새 서버에서 한 번만 실행한다. 이미 만들어져 있으면 멱등하게 다시 건다.
#
#   ./tools/setup_data_layout.sh
#
# 실제 h5 는 형제 저장소가 공유하는 한 곳에만 두고, 두 저장소는 심볼릭 링크로 본다.
#   <song>/datasets/PanCollection/            ← 실제 파일 (20 GB)
#   <song>/CANConv/data/datasets/<sensor>/    ← CANConv 가 기대하는 이름
#   <song>/PAN-Crafter/data/PanCollection/    ← feeder 가 기대하는 이름 + 저자 *_pan.h5
#
# feeder 는 경로 문자열에서 split/센서를 추론하므로 (KNOWN_ISSUES B-2/B-3)
# 디렉터리와 파일 이름을 바꾸면 안 된다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONG="$(dirname "$REPO")"
SHARED="$SONG/datasets/PanCollection"
CANCONV="${PANCRAFTER_CANCONV:-$SONG/CANConv}"
PANH5="$REPO/pan_h5/extracted"

# --- 0. 원본 압축 해제본을 공유 위치로 옮긴다 (같은 파일시스템이라 즉시) ---
if [ -d "$REPO/data/_extracted/PanCollection" ] && [ ! -d "$SHARED" ]; then
  mkdir -p "$SONG/datasets"
  mv "$REPO/data/_extracted/PanCollection" "$SHARED"
  rmdir "$REPO/data/_extracted" 2>/dev/null || true
fi
[ -d "$SHARED" ] || { echo "!! 공유 데이터 없음: $SHARED" >&2; exit 1; }
[ -d "$PANH5" ] || { echo "!! pan_h5 미해제: unzip -o pan_h5/pan_h5.zip -d $PANH5" >&2; exit 1; }

ln_s() {  # ln_s <target> <linkpath>
  [ -e "$1" ] || { echo "  !! 원본 없음: $1" >&2; return 1; }
  mkdir -p "$(dirname "$2")"; rm -f "$2"; ln -s "$1" "$2"
}

# --- 1. CANConv 쪽: presets.json 이 기대하는 이름 ---
echo "[1/3] CANConv data/datasets/"
T="$SHARED/training_data"; E="$SHARED/test_data"
ln_s "$T/train_wv3_9714.h5"   "$CANCONV/data/datasets/wv3/train_wv3.h5"
ln_s "$T/valid_wv3_9714.h5"   "$CANCONV/data/datasets/wv3/valid_wv3.h5"
ln_s "$T/train_qb_17139.h5"   "$CANCONV/data/datasets/qb/train_qb.h5"
ln_s "$T/valid_qb_17139.h5"   "$CANCONV/data/datasets/qb/valid_qb.h5"
ln_s "$T/train_gf2_19809.h5"  "$CANCONV/data/datasets/gf2/train_gf2.h5"
ln_s "$T/valid_gf2_19809.h5"  "$CANCONV/data/datasets/gf2/valid_gf2.h5"
for s in wv3 qb gf2 wv2; do
  ln_s "$E/test_${s}_multiExm1.h5"           "$CANCONV/data/datasets/$s/test_${s}_multiExm1.h5"
  ln_s "$E/test_${s}_OrigScale_multiExm1.h5" "$CANCONV/data/datasets/$s/test_${s}_OrigScale_multiExm1.h5"
done

# --- 2. PAN-Crafter 쪽: 본체 h5 는 링크, 저자 *_pan.h5(lpan) 는 실제 파일 ---
echo "[2/3] PAN-Crafter data/PanCollection/"
D="$REPO/data/PanCollection"
for s in wv3 qb gf2; do
  S=$(echo "$s" | tr a-z A-Z)
  # 공유 위치를 직접 가리킨다. CANConv 를 경유하지 않는다 —
  # 지표 구현이 tools/metrics/ 로 들어온 뒤 CANConv 는 선택 의존이 되었다.
  case $s in
    wv3) TR=train_wv3_9714;  VA=valid_wv3_9714  ;;
    qb)  TR=train_qb_17139;  VA=valid_qb_17139  ;;
    gf2) TR=train_gf2_19809; VA=valid_gf2_19809 ;;
  esac
  ln_s "$T/$TR.h5" "$D/$S/train_${s}.h5"
  ln_s "$T/$VA.h5" "$D/$S/valid_${s}.h5"
  ln_s "$E/test_${s}_multiExm1.h5" \
       "$D/$S/reduced_examples_h5/test_${s}_multiExm1.h5"
  ln_s "$E/test_${s}_OrigScale_multiExm1.h5" \
       "$D/$S/full_examples_h5/test_${s}_OrigScale_multiExm1.h5"
  # lpan (저자 배포본). 링크가 아니라 복사 — 원본 zip 을 건드리지 않기 위해서다.
  cp -f "$PANH5/train_${s}_pan.h5" "$D/$S/train_${s}_pan.h5"
  cp -f "$PANH5/valid_${s}_pan.h5" "$D/$S/valid_${s}_pan.h5"
  cp -f "$PANH5/test_${s}_multiExm1_pan.h5" \
        "$D/$S/reduced_examples_h5/test_${s}_multiExm1_pan.h5"
  cp -f "$PANH5/test_${s}_OrigScale_multiExm1_pan.h5" \
        "$D/$S/full_examples_h5/test_${s}_OrigScale_multiExm1_pan.h5"
done

echo "[3/3] 배치 결과"
find "$D" -maxdepth 3 | sort | sed "s|$REPO/||"
echo
echo "다음: python tools/repair_lpan.py --sensor wv3   (KNOWN_ISSUES F-1, full-res 평가 전 필수)"
