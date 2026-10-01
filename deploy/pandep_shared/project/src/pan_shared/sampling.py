"""Committed-only deterministic sensor/sample/four-view streams (no prefetch)."""
from __future__ import annotations

import copy
import hashlib
import json
import random

from .common import seed_for


def stream_seed(master, *parts):
    return seed_for(master, *parts)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BalancedSampler:
    """peek is speculative; commit alone advances RNG, epochs and exposures.

    Each full-data sensor has the same sample/augmentation sequence in Shared,
    Wide and Single regardless of interleaving. State is plain Python and can be
    saved in the full resume. No DataLoader worker/prefetch is permitted here.
    """

    def __init__(self, seed, sample_ids, sensors=None, effective_batch=48):
        self.sensors = tuple(sensors or sample_ids)
        if len(self.sensors) not in (1, 3) or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("one sensor or three distinct sensors required")
        self.ids = {s: list(map(int, sample_ids[s])) for s in self.sensors}
        if any(not ids or len(ids) != len(set(ids)) for ids in self.ids.values()):
            raise ValueError("sample IDs must be nonempty and unique")
        self.batch_size = int(effective_batch)
        if self.batch_size <= 0:
            raise ValueError("positive effective batch required")
        self.identity = _digest([int(seed), self.sensors, self.ids, self.batch_size])
        self._state = {
            "schema": "PANDEP_SAMPLER_v1", "identity": self.identity,
            "step": 0, "sensor_rng": random.Random(stream_seed(seed, "sensor-order")).getstate(),
            "triplet": [], "triplet_cursor": 0, "streams": {},
        }
        for s in self.sensors:
            self._state["streams"][s] = {
                "sample_rng": random.Random(stream_seed(seed, "sample-order", s)).getstate(),
                "augmentation_rng": random.Random(stream_seed(seed, "augmentation", s)).getstate(),
                "order": [], "cursor": 0, "epochs_started": 0,
                "visits": 0, "samples": 0, "counts": [0] * len(self.ids[s]),
            }
        self._positions = {s: {v: i for i, v in enumerate(ids)} for s, ids in self.ids.items()}
        self._pending = None

    def peek(self):
        if self._pending is not None:
            return copy.deepcopy(self._pending[0])
        # Copy only the selected mutable stream. Copying all full epoch arrays
        # recursively per optimizer update would add avoidable Python overhead.
        state = dict(self._state, streams=dict(self._state["streams"]))
        if state["triplet_cursor"] == len(state["triplet"]):
            rng = random.Random(); rng.setstate(state["sensor_rng"])
            order = list(self.sensors)
            rng.shuffle(order)
            state.update(triplet=order, triplet_cursor=0, sensor_rng=rng.getstate())
        sensor = state["triplet"][state["triplet_cursor"]]
        committed = state["streams"][sensor]
        stream = dict(committed, order=list(committed["order"]), counts=list(committed["counts"]))
        state["streams"][sensor] = stream
        sample_rng = random.Random(); sample_rng.setstate(stream["sample_rng"])
        aug_rng = random.Random(); aug_rng.setstate(stream["augmentation_rng"])
        ids = []
        while len(ids) < self.batch_size:
            if stream["cursor"] == len(stream["order"]):
                stream["order"] = list(self.ids[sensor]); sample_rng.shuffle(stream["order"])
                stream["cursor"] = 0; stream["epochs_started"] += 1
            take = min(self.batch_size - len(ids), len(stream["order"]) - stream["cursor"])
            ids.extend(stream["order"][stream["cursor"]:stream["cursor"] + take])
            stream["cursor"] += take
        rotations = [aug_rng.randrange(4) for _ in ids]
        stream["sample_rng"] = sample_rng.getstate()
        stream["augmentation_rng"] = aug_rng.getstate()
        stream["visits"] += 1; stream["samples"] += len(ids)
        for sample_id in ids:
            stream["counts"][self._positions[sensor][sample_id]] += 1
        state["triplet_cursor"] += 1; state["step"] += 1
        batch = {"step": state["step"], "sensor": sensor, "sample_ids": ids, "rot_ids": rotations}
        batch["token"] = _digest([self.identity, batch])
        self._pending = batch, state
        return copy.deepcopy(batch)

    def commit(self, token):
        if self._pending is None or self._pending[0]["token"] != token:
            raise ValueError("batch commit token mismatch")
        self._state = self._pending[1]; self._pending = None

    def state_dict(self):
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        if state.get("identity") != self.identity or state.get("schema") != "PANDEP_SAMPLER_v1":
            raise ValueError("BLOCKED_IDENTITY: sampler stream identity changed")
        self._state = copy.deepcopy(state); self._pending = None

    def exposures(self):
        return {
            s: {"optimizer_visits": v["visits"], "committed_samples": v["samples"],
                "distinct_ids_seen": sum(n > 0 for n in v["counts"]),
                "train_subset_size": len(self.ids[s]),
                "effective_epochs": v["samples"] / len(self.ids[s])}
            for s, v in self._state["streams"].items()
        }
