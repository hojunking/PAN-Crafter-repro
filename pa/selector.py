"""§10.8 best 선택기 — stream(raw / aligned) 마다 독립.

1. 적격 checkpoint HQNR 의 running maximum 을 보존한다 (anchor).
2. maximum − tol_hqnr 이상인 checkpoint 만 동률 후보.
3. 후보 중 fSCC maximum − tol_fscc 이상인 집합.
4. 그 집합에서 가장 늦은 optimizer step.
running max 가 오르면 과거 후보가 best 가 될 수 있으므로 band 안 후보의 가중치를 보존한다(candidates/). band 밖으로 영구히 나간 후보만 정리.
"""
import json
import math
import os


class BestSelector:
    def __init__(self, name, tol_hqnr=1e-4, tol_fscc=1e-4):
        self.name = name; self.tol_h = tol_hqnr; self.tol_f = tol_fscc
        self.max_hqnr = -math.inf; self.cands = []; self.best = None; self.history = []

    def update(self, step, epoch, hqnr, fscc, eligible, path, reason=""):
        """반환 dict(changed, best, pruned:[path...]). eligible=False 면 후보에 넣지 않고 이력만 남긴다."""
        rec = dict(step=int(step), epoch=int(epoch), hqnr=float(hqnr), fscc=float(fscc), path=path, eligible=bool(eligible), reason=reason)
        self.history.append({k: v for k, v in rec.items()})
        if not eligible or not (math.isfinite(hqnr) and math.isfinite(fscc)):
            return dict(changed=False, best=self.best, pruned=[])
        self.max_hqnr = max(self.max_hqnr, rec["hqnr"])
        self.cands.append(rec)
        band = [c for c in self.cands if c["hqnr"] >= self.max_hqnr - self.tol_h]
        pruned = [c["path"] for c in self.cands if c not in band]
        self.cands = band
        fmax = max(c["fscc"] for c in band)
        tie = [c for c in band if c["fscc"] >= fmax - self.tol_f]
        best = max(tie, key=lambda c: c["step"])
        changed = self.best is None or best["step"] != self.best["step"]
        self.best = best
        return dict(changed=changed, best=best, pruned=pruned)

    def state(self):
        return dict(name=self.name, tol_hqnr=self.tol_h, tol_fscc=self.tol_f, max_hqnr=self.max_hqnr, cands=self.cands, best=self.best, history=self.history)

    def save(self, path, extra=None):
        s = self.state(); s.update(extra or {})
        json.dump(s, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path, expect_protocol=None):
        s = json.load(open(path)); o = cls(s["name"], s["tol_hqnr"], s["tol_fscc"])
        if expect_protocol is not None and s.get("protocol_id") != expect_protocol:
            raise RuntimeError(f"selector state protocol {s.get('protocol_id')} != {expect_protocol} — 이어 쓰지 않는다 (§13)")
        o.max_hqnr = s["max_hqnr"]; o.cands = s["cands"]; o.best = s["best"]; o.history = s["history"]
        return o
