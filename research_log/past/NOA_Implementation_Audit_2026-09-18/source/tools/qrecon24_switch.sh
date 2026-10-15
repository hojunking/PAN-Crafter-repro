#!/usr/bin/env bash
# QRECON24 전환 (s1–s5; 계획 research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §9–§10, 노트 research_log/2026-09-16_qrecon24-implementation.md) — pull 뒤 한 번.
#   ./tools/qrecon24_switch.sh --dry-run   # 검사만: gate(K01–K48) · cue verify(이 서버 T0) · 명시 큐 config == 생성기 · 예약 표 — 운영 파일은 **쓰지 않는다**
#   ./tools/qrecon24_switch.sh             # 적용: 위 검사 + 서버 로컬 mandatory(PAKD50 + QRECON24)·reservations · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(상한 없음)
#                                          #      · chain 이 살아 있으면 **그대로 두고** work_dir/cases_queue_handover.txt 를 둔다 → runner 가 **현재 run 이 끝난 case 경계에서** 남은 옛 큐를 버리고 QRECON24 큐로 바꾼다 (감사 F02; 옛 pending 을 더 돌리지 않는다)
#                                          #      · chain 이 없으면 큐 config/queues/qrecon24_<srv>.txt 로 campaign_start · 대기자 tools/qrecon24_waiter.sh
#   ./tools/qrecon24_switch.sh --extend    # §8.3: 기본 큐를 다 마쳤는데 신규 완료 **학습** 시간(train_hours) 합 < 24h 이면 지정 3 run(추가 seed) 을 편성: config 생성 + extra_priority + QRECON24 mandatory + 활성 큐(work_dir/_qrecon24/queue_active.txt)
#                                          #   + 예약 파일 갱신, chain 이 살아 있으면 case 경계 인계 파일, 없으면 campaign_start(활성 큐) + 대기자 (감사 F03/F05). ≥ 24h 면 rc 0 으로 '불필요' 만 출력
# ADJ-R1 (2026-09-17; research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md §5·§9·§11): 같은 스크립트로 적용한다 — 큐 파일 = 생성기 활성 편성(등록 완료 + §5 순서; 보류 3 run 제외), v2 config 16 벌,
#   서버 로컬 mandatory/reservations/queue_active/handover 를 같은 revision 으로. 보류 run 이 이미 실행 중/완료면 원 정의로 끝난다(보고만). 완료 QRC24 run 의 버전 감사(§8.1) 와 공식 RR selector backlog(§7.2) 는
#   GPU 가 비어 있으면 지금(⑦), 아니면 다음 case 경계(tools/_upload.sh → qrecon24_postrun.py) 에서.
# 원칙(§9.1·§10.1): Sheet 만 보고 프로세스를 kill 하지 않는다 · 완료 ID 덮어쓰기 없음 · 실행 중 코드에 pull 하지 않는다(현재 run 은 그 코드로 끝난다) · 실패/NaN 자동 반복 없음
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; DRY=0; EXTEND=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --extend) EXTEND=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s1|s2|s3|s4|s5) ;; *) fail "QRECON24 switch 는 s1–s5 용 (이 서버: $SERVER)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
[ -f work_dir/_eval_phase/hold.json ] && fail "평가 phase hold 중이다 (계획 2026-09-18 §2) — 새 학습을 기동하지 않는다. 확인: python tools/eval_phase.py status · 해제: python tools/eval_phase.py release"
CAMP=work_dir/_qrecon24; QUEUE="config/queues/qrecon24_${SERVER}.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음 (git pull)"
mkdir -p "$CAMP" work_dir/_qrecon24_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
GATE_ON=0; grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null && GATE_ON=1
if [ "$EXTEND" = 1 ]; then
  [ "$DRY" = 1 ] && fail "--extend 는 --dry-run 과 같이 쓰지 않는다"
  "$PY" - "$SERVER" <<'PYEOF' || fail "--extend 실패"
import json, os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv = sys.argv[1]; base = G.priority_for(srv); pend = [r for r in base if not terminal(r)]
if pend: sys.exit(f"!! 기본 큐 미완 {len(pend)} run — 확장은 기본 큐를 다 마친 뒤 (§8.3)")
led = G._load_json(G.QRC24_LEDGER); ent = [(rid, e) for rid, e in (led.get("entries") or {}).items() if e.get("kind") == "run" and str(e.get("status", "")).startswith("FINISHED") and "QRC24" in rid and "#" not in rid]
# 감사 F05: 24h 판정은 **학습 시간**(train_hours = 학습 종료까지; hours) 만 — hours_total(평가/export 포함)·setup 은 따로 보인다
tr_h = [float(e.get("train_hours") or e.get("hours") or 0.0) for _, e in ent]; tot_h = [float(e.get("hours_total") or e.get("hours") or 0.0) for _, e in ent]; setup_h = [float(e.get("setup_hours") or 0.0) for _, e in ent]
tot = sum(tr_h); print(f"   신규 완료 {len(ent)} run: train_h 합 {tot:.2f} h · hours_total(평가/export 포함) 합 {sum(tot_h):.2f} h · postprocess 합 {sum(tot_h) - tot:.2f} h · setup 합 {sum(setup_h):.2f} h (ledger {G.QRC24_LEDGER})")
if tot >= 24.0:
    print("   train_h ≥ 24h — 확장 불필요 (§8.3)"); sys.exit(0)
sd0, profs = G.QRC24_EXTRA[srv]; items = None
for sd in [sd0] + list(G.QRC24_RESERVE_SEEDS):
    cand = [G.qrc24_run_name(srv, p, sd) for p in profs]
    if not any(complete(r) or terminal(r) for r in cand): items = cand; break
if items is None: sys.exit("!! 지정 seed 와 reserve seed 전부 이미 결과가 있다 — 사람이 결정")
G.generate(srv, items, os.path.join(G.ROOT, "config"), projected=None)
xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); cur = G.extra_priority(); add = [r for r in items if r not in cur]        # 감사 F04: 기존 항목을 읽은 뒤 덧붙인다
with open(xp, "a") as fh: fh.write("# QRECON24 §8.3 확장 (신규 완료 train_h 합 < 24h; %s)\n" % time.strftime("%Y-%m-%dT%H:%M") + "\n".join(add) + "\n")
mp = os.path.join(G.ROOT, G.QRC24_MANDATORY_FILE); mand = [l.strip() for l in open(mp) if l.strip() and not l.startswith("#")] if os.path.exists(mp) else list(base)
with open(mp, "a") as fh: fh.write("# §8.3 확장\n" + "\n".join(r for r in items if r not in mand) + "\n")
qa = os.path.join(G.ROOT, "work_dir", "_qrecon24", "queue_active.txt"); active = list(base) + [r for r in G.extra_priority() if r not in base]
open(qa, "w").write("# QRECON24 활성 큐 = 기본 큐 + §8.3 확장 (tools/qrecon24_switch.sh --extend; 대기자·재기동이 이 파일을 쓴다)\n" + "\n".join(active) + "\n")
G.write_reservation_file(srv, active, G.measured_hours_all(srv))
open(os.path.join(G.ROOT, "work_dir", "cases_queue_handover.txt"), "w").write("# QRECON24 --extend 인계 (runner 가 case 경계에서 적용; chain 이 없으면 campaign_start 가 지운다)\n" + "\n".join(items) + "\n")
open(os.path.join(G.ROOT, "work_dir", "_qrecon24", "extend_added.txt"), "w").write("\n".join(items) + "\n")
print("   추가:", items, "→ extra_priority + mandatory + queue_active + reservations + handover (config 생성)")
PYEOF
  rc=$?; [ $rc -eq 0 ] || exit $rc
  [ -f work_dir/_qrecon24/extend_added.txt ] || { echo "[qrecon24] 확장 없음 — 끝"; exit 0; }; rm -f work_dir/_qrecon24/extend_added.txt
  if ps -eo args | grep -q '[_]run_cases\.sh'; then
    { echo "# QRECON24 $SERVER 확장 반영 $(date -Iseconds) — tools/qrecon24_switch.sh --extend"; grep -vE '^[[:space:]]*(#|$)' work_dir/_qrecon24/queue_active.txt; } > "$TMP/cq" && mv "$TMP/cq" work_dir/cases_queue.txt
    echo "   chain 살아 있음 — 확장 3 run 은 case 경계 인계 파일(work_dir/cases_queue_handover.txt) 로 이어 붙는다; gate 가 켜진 서버는 다음 pass 에도 편성"
  else
    rm -f work_dir/cases_queue_handover.txt
    ./tools/campaign_start.sh --queue work_dir/_qrecon24/queue_active.txt --hours 24 --label "qrecon24-$SERVER-extend-$(date +%m%d-%H%M)" || fail "확장 기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동(활성 큐) · 마감 파일 제거 · 감시자 cron 등록"
  fi
  if ! ps -eo args | grep -q '[q]recon24_waiter\.sh'; then setsid nohup ./tools/qrecon24_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $!"; fi
  exit 0
fi
echo "[qrecon24] ① $SERVER — unit gate (K01–K50; 임시 fixture — 운영 파일 기록 없음)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep "FAIL" | head -5; fail "unit gate 실패 — $CAMP/gate_$SERVER.log"; }; tail -1 "$CAMP/gate_$SERVER.log"
echo "[qrecon24] ② cue 자산 verify (raw q 의 출처: 이 서버 T0 aligner 로 96 base × 4 state 재계산)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 — work_dir/_qedge9/verify_$SERVER.json"
echo "[qrecon24] ③ 명시 큐 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략; 서버 토큰 == 이 서버)"
mapfile -t ITEMS < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")
"$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$(IFS=,; echo "${ITEMS[*]}")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP/cfg" "$DRY" "${ITEMS[@]}" <<'PYEOF' || fail "config 검사 실패"
import filecmp, json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv, tmp, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"; items = sys.argv[4:]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []
assert items == G.priority_for(srv), "큐 파일 != 생성기 명시 순서 (git pull)"
for run in items + G.extra_priority():
    f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, run) or "PAKD50"
    if br == "QRECON24" and G.qrc24_parse(G.case_of(run))[0] != srv: bad.append(f"{run}: 서버 토큰이 {srv} 가 아니다"); continue
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(run) else "대기(cue 미준비)"))
    if st.startswith("완료") or st.startswith("실행 중"):
        print(f"   {run:<62} {br:<9} {st}"); continue
    if not os.path.exists(f): bad.append(f"{run}: config 없음 (git pull)"); continue
    if not os.path.exists(t): print(f"   {run:<62} {br:<9} {st} — (extra; 생성기 비교 생략)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {run:<62} {br:<9} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{run}: config != 생성기")
alive = any("tools/_run_cases.sh" in l for l in ps.splitlines()); cur = [l for l in ps.splitlines() if "main.py" in l and "--config" in l]
print("   chain:", "살아 있음" if alive else "없음", "· 학습 중:", (cur[0][:120] if cur else "없음"), "— 진행 중 run 은 마지막 update 까지 둔다 (§9.1)")
if bad: print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
echo "[qrecon24] ③-ADJ 보류 run(§9; 미시작일 때만 보류 — 편성·mandatory 에서 제외) · 등록/남은 편성 · 버전 감사(§8.1; 완료 run 의 시작 manifest 기준)"
"$PY" - "$SERVER" "$DRY" "$TMP" <<'PYEOF' || fail "ADJ-R1 상태 검사 실패"
import json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
srv, dry, tmp = sys.argv[1], sys.argv[2] == "1", sys.argv[3]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
print(f"   revision {G.QRC24_ADJ_REVISION} · 활성 {len(G.qrc24_items(srv))} = 등록 {len(G.QRC24_REGISTERED[srv])} + 남은 {len(G.QRC24_ADJ_ORDER[srv])}(v2 추가 {len(G.qrc24_adj_runs(srv))}) · 보류 {len(G.qrc24_held_runs(srv))}")
_HST = {"terminal": "완료/실패 — 원 정의 결과 그대로(편성 밖)", "running_original_definition": "실행 중 — 끝까지 원 정의로(§9; 이 switch 가 활성 큐 맨 앞에 두어 인계 뒤에도 이어진다)",
        "started_interrupted": "시작됨(체크포인트 있음)·미완 — 보류하지 않고 원 정의로 끝낸다(§9; 활성 큐 맨 앞)", G.QRC24_HELD_STATUS: "미시작 — 보류(편성·mandatory 밖)"}
for run, (st, why) in G.qrc24_held_states(srv, ps).items():
    print(f"   보류 {run}: {_HST[st]} [{st}] — {why}")
r = subprocess.run([sys.executable, "tools/qrecon24_version_audit.py", "--all"] + (["--no-write"] if dry else ["--out", os.path.join("work_dir", "_qrecon24", "version_audit.json")]), capture_output=True, text=True)     # dry-run 은 아무 파일도 쓰지 않는다
print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l) or "   (완료 QRC24 run 없음)")
PYEOF
echo "[qrecon24] ④ 예약 표 (§8.2 1.20×R_s + 10/60; 실측은 PAKD50+QEDGE9+QEGX+EDGEBAL+QRECON24 ledger 통합)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then echo "[qrecon24] --dry-run — 운영 파일·chain·cron 을 바꾸지 않았다 (gate 'pakd50' $( [ "$GATE_ON" = 1 ] && echo 켜짐 || echo 꺼짐 ))"; exit 0; fi
echo "[qrecon24] ⑤ 서버 로컬 기본 묶음·예약 파일 · 옛 extra_priority 보존 분리"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); cur = G.extra_priority(); old = [x for x in cur if G.branch_for(srv, x) != "QRECON24"]; keep = [x for x in cur if G.branch_for(srv, x) == "QRECON24"]   # 감사 F04: 옮기기 전에 읽는다
if old:
    bak = xp.replace("extra_priority.txt", f"extra_priority.pre_qrecon24_{time.strftime('%m%d-%H%M')}.txt"); os.replace(xp, bak)
    open(xp, "w").write(f"# QRECON24 전환({time.strftime('%Y-%m-%dT%H:%M')}): 이전 추가 편성 {len(old)} 항목은 {os.path.basename(bak)} 에 보존 — QRECON24 의 추가(§8.3 --extend; run 이름) 만 여기에\n" + "\n".join(keep) + "\n")
    print(f"   extra_priority {len(old)} 항목(비 QRECON24) → {os.path.basename(bak)} (보존; 편성에서 제외 — 이전 QEDGE9/QEGX/EDGEBAL 미완 항목은 superseded, 필요하면 run 이름으로 다시 적는다)")
runs = G.write_mandatory_file(srv); m = G.measured_hours_all(srv)
import subprocess; _ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
_hs = G.qrc24_held_states(srv, _ps); _keepfirst = [r for r, (s_, _) in _hs.items() if s_ in ("running_original_definition", "started_interrupted")]      # §9: 시작된 보류 run 은 원 정의로 끝낸다 — 활성 큐·mandatory 맨 앞
q = _keepfirst + [it for it in G.priority_for(srv) + G.extra_priority() if G.branch_for(srv, it) == "QRECON24" and it not in _keepfirst]; p = os.path.join(G.ROOT, G.QRC24_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} QRECON24 run (자체 ledger {G.QRC24_LEDGER}; 상한 없음, required 경고만; 대기자 tools/qrecon24_waiter.sh 의 pending 목록) — tools/qrecon24_switch.sh\n" + "\n".join(q) + "\n")
res = G.write_reservation_file(srv, q + [x for x in G.priority_for(srv) + G.extra_priority() if x not in q], m)          # 시작된 보류 run 도 예약에 포함
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.QRC24_MANDATORY_FILE}: {len(q)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
# ADJ-R1 §11.1-4: revision 기록 · 보류 상태 기록 · 활성 큐 파일(--extend 뒤 대기자가 우선 읽는 파일) 을 같은 revision 으로
import json
held = {r: dict(status=s_, reason=w, kept_in_queue=(r in _keepfirst), checkpoint_present=G.run_started(r)) for r, (s_, w) in _hs.items()}
from tools.campaign_gate import terminal as _term
_pend = [it for it in q if not _term(it)]; _done_n = len(q) - len(_pend)      # 완료·실패 run 은 실행 큐에서 뺀다 — runner 의 "완료됨 — 업로드만 확인" 이 매 전환마다 FR 평가·pa_diag(GPU)·시트 업로드를 되풀이하던 것(검토 지적). 이력은 config/queues 와 queue_revision.json 에 남는다
_qe = os.path.join(G.ROOT, "work_dir", "_qrecon24", "queue_effective.txt")
open(_qe, "w").write(f"# QRECON24 실제 실행 순서 ({G.QRC24_ADJ_REVISION}; 시작된 보류 run 을 맨 앞에 + 미완 편성 — 완료·실패 {_done_n} run 은 뺐다(이력은 config/queues/qrecon24_{srv}.txt·queue_revision.json)). tools/qrecon24_switch.sh ⑥ 이 인계/기동에 쓴다\n" + "\n".join(_pend) + "\n")
print(f"   queue_effective.txt: 미완 {len(_pend)} run (완료·실패 {_done_n} 제외)" + (f" · 시작된 보류 {len(_keepfirst)} run 을 맨 앞에 유지: {_keepfirst}" if _keepfirst else ""))
json.dump(dict(queue_revision=G.QRC24_ADJ_REVISION, plan=G.QRC24_ADJ_PLAN, server=srv, applied_at=time.strftime("%Y-%m-%dT%H:%M:%S"), items=G.qrc24_items(srv), effective_queue=q, pending_queue=_pend, completed_omitted=_done_n, added_v2=G.qrc24_adj_runs(srv), held=held), open(os.path.join(G.ROOT, "work_dir", "_qrecon24", "queue_revision.json"), "w"), indent=1, ensure_ascii=False)
json.dump(held, open(os.path.join(G.ROOT, "work_dir", "_qrecon24", "held_runs.json"), "w"), indent=1, ensure_ascii=False)
qa = os.path.join(G.ROOT, "work_dir", "_qrecon24", "queue_active.txt")
if os.path.exists(qa):
    active = q + [x for x in G.extra_priority() if G.branch_for(srv, x) == "QRECON24" and x not in q]
    open(qa, "w").write(f"# QRECON24 활성 큐 ({G.QRC24_ADJ_REVISION}; 기본 큐 + §8.3 확장) — tools/qrecon24_switch.sh\n" + "\n".join(active) + "\n"); print(f"   queue_active.txt 갱신: {len(active)} run")
print(f"   queue_revision.json / held_runs.json 기록 ({G.QRC24_ADJ_REVISION}; 보류 {len(held)})")
PYEOF
echo "[qrecon24] ⑥ chain: 상한 없음 — 마감 파일 제거; runner 는 그대로(현재 run·후처리 유지), 없으면 기동; 대기자 기동"
if [ -f work_dir/cases_deadline.txt ]; then cp work_dir/cases_deadline.txt "$CAMP/cases_deadline.before_qrecon24.txt"; rm -f work_dir/cases_deadline.txt; echo "   cases_deadline.txt 제거 (사본 $CAMP/cases_deadline.before_qrecon24.txt)"; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  QEFF="$CAMP/queue_effective.txt"; grep -qvE '^[[:space:]]*(#|$)' "$QEFF" 2>/dev/null || QEFF="$QUEUE"
  { echo "# QRECON24 인계 $(date -Iseconds) — runner 가 현재 run 이 끝난 case 경계에서 남은 옛 큐를 버리고 이 큐로 바꾼다 (tools/_run_cases.sh HANDOVER)"; grep -vE '^[[:space:]]*(#|$)' "$QEFF"; } > "$TMP/handover" && mv "$TMP/handover" work_dir/cases_queue_handover.txt
  # 영속 큐도 같은 순서로 (감시자·재부팅 재기동이 옛 편성/보류 run 을 돌리지 않게; runner 는 이 파일을 기동 시에만 읽는다) — ADJ-R1 검토 지적
  { echo "# QRECON24 $SERVER ($(date -Iseconds); tools/qrecon24_switch.sh — 인계와 같은 순서)"; grep -vE '^[[:space:]]*(#|$)' "$QEFF"; } > "$TMP/cq" && mv "$TMP/cq" work_dir/cases_queue.txt
  echo "   chain 살아 있음 — 죽이지 않는다. 현재 run 은 마지막 update 까지 돌고, case 경계에서 work_dir/cases_queue_handover.txt 의 QRECON24 큐로 인계된다 (옛 pending 은 더 돌지 않는다; 옛 runner 코드면 인계 파일을 못 읽으므로 다음 gate pass/대기자가 이어받는다)"
else
  rm -f work_dir/cases_queue_handover.txt
  ./tools/campaign_start.sh --queue "$( grep -qvE '^[[:space:]]*(#|$)' "$CAMP/queue_effective.txt" 2>/dev/null && echo "$CAMP/queue_effective.txt" || echo "$QUEUE" )" --hours 24 --label "qrecon24-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동($QUEUE) · 마감 파일 제거 · 감시자 cron 등록"
fi
if ! ps -eo args | grep -q '[q]recon24_waiter\.sh'; then setsid nohup ./tools/qrecon24_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $! ($CAMP/waiter.log · 상태 $CAMP/status.json)"; else echo "   대기자 이미 실행 중 (파일을 매 loop 다시 읽는다)"; fi
echo "[qrecon24] ⑦ 공식 RR selector backlog(§7.2) · 버전 감사(§8.1): GPU 가 비어 있으면 지금, 아니면 다음 case 경계(tools/_upload.sh → tools/qrecon24_postrun.py)"
if ps -eo args | grep -q '[m]ain\.py --config'; then echo "   학습 중 — 다음 case 경계에서 (backlog: $("$PY" tools/qrecon24_postrun.py --backlog --dry-run 2>&1 | grep -v Warning | tail -1))"; else "$PY" tools/qrecon24_postrun.py --backlog 2>&1 | grep -v Warning | tail -12; fi
echo "[qrecon24] 완료 — $SERVER 명시 큐 $(grep -cvE '^[[:space:]]*(#|$)' "$QUEUE") run (마감 admission 제외). 확인: tail -f work_dir/cases_chain.log · $CAMP/status.json · 결과 selector: python tools/qrecon24_select.py <run>"
