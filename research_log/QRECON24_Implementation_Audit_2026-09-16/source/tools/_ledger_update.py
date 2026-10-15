#!/usr/bin/env python
"""예산 ledger 잠금 갱신 (fcntl.flock) — trainer·_upload.sh·palsv18_validate.sh 가 같은 json 을 만진다 (PALSV18 리뷰 P2-6).
    python tools/_ledger_update.py <ledger.json> set <entry> '<json dict>'      # entry 를 dict 로 교체/병합 (기존 hours 누적은 add 사용)
    python tools/_ledger_update.py <ledger.json> add <entry> <hours> [kind] [note]   # hours 누적 (없으면 생성)
    python tools/_ledger_update.py <ledger.json> get                             # used(전체 entries 합, reserved 포함)·entries 상태 요약 출력
"""
import fcntl, json, os, sys, time


def locked(path):
    class L:
        def __enter__(self):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True); self.f = open(path + ".lock", "w"); fcntl.flock(self.f, fcntl.LOCK_EX); return self
        def __exit__(self, *a):
            fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()
    return L()


def load(path, total=18.0):
    return json.load(open(path)) if os.path.exists(path) else dict(total_gpu_hours=total, entries={})


def used_hours(d):
    return sum(float(e.get("hours_total") or e.get("hours") or 0.0) for e in d["entries"].values())


def main():
    path, op = sys.argv[1], sys.argv[2]
    with locked(path):
        d = load(path)
        if op == "set":
            name, val = sys.argv[3], json.loads(sys.argv[4]); e = d["entries"].get(name, {}); e.update(val); e.setdefault("updated", time.strftime("%Y-%m-%dT%H:%M:%S")); d["entries"][name] = e
        elif op == "add":
            name, h = sys.argv[3], float(sys.argv[4]); e = d["entries"].get(name, {}); e["hours"] = float(e.get("hours") or 0.0) + h; e["kind"] = (sys.argv[5] if len(sys.argv) > 5 else e.get("kind", "diag"))
            if len(sys.argv) > 6:
                e["note"] = sys.argv[6]
            e["runs"] = int(e.get("runs", 0)) + 1; e["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); d["entries"][name] = e
        elif op == "get":
            print(json.dumps(dict(total=d.get("total_gpu_hours"), used=used_hours(d), entries={k: dict(kind=v.get("kind"), hours=round(float(v.get("hours_total") or v.get("hours") or 0.0), 3), status=v.get("status")) for k, v in d["entries"].items()}))); return
        json.dump(d, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
