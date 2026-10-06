"""TA2 native LMS sources, independent view RNG, and immutable shift assets.

Only the existing hash-pinned WV3 / QB-msfix / GF2 sources are accepted in
production. Stored LMS is read before any HR geometry augmentation. FR access
has an explicit key allowlist and never checks/loads a GT dataset.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from align.resample import _CDF23, interp23tap
from ta2.registration import ESTIMATOR, estimate_shift, fixed_band_weights, register_band, transform_shifts


SENSOR = {"WV3": (8, 2047., 9714, 1080), "QB": (4, 2047., 17139, 1905),
          "GF2": (4, 1023., 19809, 2201)}
SPLITS = ("train", "val", "rr", "fr")
AUGMENTATION = {"crop": False, "hflip_probability": 1.0, "vflip_probability": 1.0,
                "rotation": "isolated Python Random.randint(0,3), uniform CCW",
                "order": ["read_stored_LMS", "horizontal_flip", "vertical_flip", "rot90"],
                "B_order": "bicubic(MS) after corresponding LR augmentation; align_corners=False",
                "source": "feeders/feeder.py:PanFeeder.augment; no constructor global seed reset"}
CAL_COUNT = 512


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable TA2 asset differs: " + str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ta2-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def operator_contract():
    import align.resample as module
    return dict(name="align.resample.interp23tap", source_sha256=sha256(module.__file__),
                coefficients_half=_CDF23, ratio=4, padding="circular per 2x stage",
                zero_stuff_phase=["stage0 odd/odd", "stage1 even/even"],
                native_sample_phase_yx=[2, 2], calculation_dtype="float64", scale="original DN",
                torch_version=torch.__version__, numpy_version=np.__version__,
                crop_order="stored dataset patch is authoritative; no full-scene lineage claim",
                pan_or_gt_dependency=False, bicubic_fallback=False)


def _band_ncc(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x, y = x-x.mean(), y-y.mean()
    norm = float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if norm <= 1e-20 else float(np.sum(x*y)/norm)


def _stat_stamp(path):
    st = Path(path).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def audit_source(path, split, *, bands, max_dn, expected_count=None, expected_size=None,
                 phase_samples=16, chunk=32):
    """Full finite/range/LMS parity scan; phase sample is deterministic, not a gate.

    float64 tolerance 1e-7 DN exceeds observed roundoff (~1e-12 DN), not
    sensor/noise error. float32 tolerance is 32 ULP at nominal max DN. NCC alone
    never overrides the max-absolute and RMSE tests. No source files are edited.
    """
    if split not in SPLITS:
        raise ValueError("Unknown source split")
    keys = ("pan", "ms", "lms") + (() if split == "fr" else ("gt",))
    before = _stat_stamp(path)
    with h5py.File(path, "r") as f:
        if not set(keys).issubset(f.keys()):
            raise ValueError("BLOCKED_LMS_LINEAGE: required stored native LMS/source keys absent")
        n, c, h, w = f["pan"].shape
        expected = {"pan": (n, 1, h, w), "ms": (n, bands, h//4, w//4),
                    "lms": (n, bands, h, w), "gt": (n, bands, h, w)}
        if not n or c != 1 or h != w or h % 4 or any(f[k].shape != expected[k] for k in keys):
            raise ValueError("Native PAN/MS/LMS/GT geometry mismatch")
        if (expected_count is not None and n != expected_count) or (expected_size is not None and h != expected_size):
            raise ValueError("Native split count/size differs from frozen source profile")
        stats = {k: dict(minimum=float("inf"), maximum=-float("inf"), zero_count=0,
                         saturation_count=0) for k in keys}
        dtype = np.dtype(f["lms"].dtype)
        tolerance = 1e-7 if dtype.itemsize >= 8 else 32*np.finfo(np.float32).eps*max_dn
        max_abs, squared, pixels = 0., 0., 0
        ncc_sum, ncc_count = np.zeros(bands), np.zeros(bands, dtype=np.int64)
        phase_ids = sorted(set(np.linspace(0, n-1, min(n, phase_samples), dtype=int).tolist()))
        phase_rows = []
        for start in range(0, n, chunk):
            arrays = {k: np.asarray(f[k][start:start+chunk], np.float64) for k in keys}
            for k, x in arrays.items():
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite native observation: " + k)
                s = stats[k]
                s["minimum"] = min(s["minimum"], float(x.min()))
                s["maximum"] = max(s["maximum"], float(x.max()))
                s["zero_count"] += int(np.count_nonzero(x == 0))
                s["saturation_count"] += int(np.count_nonzero(x == max_dn))
            regenerated = interp23tap(torch.from_numpy(arrays["ms"]), ratio=4).numpy()
            difference = regenerated-arrays["lms"]
            max_abs = max(max_abs, float(np.abs(difference).max()))
            squared += float(np.square(difference).sum()); pixels += difference.size
            for local in range(len(regenerated)):
                for b in range(bands):
                    value = _band_ncc(regenerated[local, b], arrays["lms"][local, b])
                    if value is not None:
                        ncc_sum[b] += value; ncc_count[b] += 1
                index = start+local
                if index in phase_ids:
                    row = {"sample_id": index, "stored_vs_regenerated": [], "gt_available": split != "fr"}
                    # Same-band operator phase, no multispectral registration assertion.
                    pairs = {"stored_vs_regenerated": (arrays["lms"][local], regenerated[local])}
                    if split != "fr":
                        bicubic = F.interpolate(torch.from_numpy(arrays["ms"][local:local+1]),
                                                scale_factor=4, mode="bicubic", align_corners=False)[0].numpy()
                        pairs.update(lms_vs_gt=(arrays["lms"][local], arrays["gt"][local]),
                                     bicubic_vs_gt=(bicubic, arrays["gt"][local]))
                    for name, (moving, reference) in pairs.items():
                        row[name] = [register_band(m/max_dn, r/max_dn, sigma_p=.8, sigma_r=.8)
                                     for m, r in zip(moving, reference)]
                    phase_rows.append(row)
        for key in ("pan", "gt"):
            if key in stats and (stats[key]["minimum"] < 0 or stats[key]["maximum"] > max_dn):
                raise ValueError("Native PAN/GT outside declared DN range")
        rmse = float(np.sqrt(squared/pixels))
        if max_abs > tolerance or rmse > tolerance:
            raise ValueError(f"BLOCKED_LMS_LINEAGE: {split} max_abs={max_abs}, RMSE={rmse}, tolerance={tolerance}")
        phase_summary = {}
        for name in ("stored_vs_regenerated",) + (() if split == "fr" else ("lms_vs_gt", "bicubic_vs_gt")):
            records = [b for row in phase_rows for b in row[name]]
            valid = np.asarray([b["shift"] for b in records if b["valid"]])
            phase_summary[name] = dict(measurements=len(records), valid_count=len(valid),
                fallback_count=sum(not b["valid"] for b in records),
                mean_dy_dx=valid.mean(axis=0).tolist() if len(valid) else None,
                median_dy_dx=np.median(valid, axis=0).tolist() if len(valid) else None,
                mean_absolute_component_pixels=float(np.abs(valid).mean()) if len(valid) else None)
        result = dict(count=n, shapes={k: list(f[k].shape) for k in keys},
                      dtypes={k: str(f[k].dtype) for k in keys}, statistics=stats,
                      lms_parity=dict(status="PASS", scan="ALL_STORED_SAMPLES_ALL_PIXELS", max_abs_dn=max_abs,
                          rmse_dn=rmse, tolerance_dn=tolerance,
                          band_ncc=[float(ncc_sum[b]/ncc_count[b]) if ncc_count[b] else None for b in range(bands)],
                          band_ncc_valid_samples=ncc_count.tolist()),
                      phase_diagnostics=dict(sample_ids=phase_ids, rows=phase_rows, summary=phase_summary,
                          scope="same-band structural proxy; not physical displacement GT",
                          fr_gt_alignment_certified=False), source_gt_accessed=split != "fr")
    if before != _stat_stamp(path):
        raise ValueError("Source mutated during native LMS audit")
    return result


def preflight_sources(repo_root, sensor, work_root):
    root, work = Path(repo_root).resolve(), Path(work_root).resolve()
    sensor = sensor.upper()
    if sensor not in SENSOR:
        raise ValueError("TA2 supports WV3, QB and GF2 only")
    bands, max_dn, train_n, val_n = SENSOR[sensor]
    catalog_path = root/"ablr2/sensor_sources.json"
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema") != "ABLR2_SENSOR_SOURCES_v1":
        raise ValueError("Unregistered source catalog")
    entry = catalog["sensors"][sensor]
    source_paths = {s: (root/entry["splits"][s]["path"]).resolve(strict=True) for s in SPLITS}
    identity = dict(sensor=sensor, catalog_sha256=sha256(catalog_path), operator=operator_contract(),
                    verifier_sha256=sha256(__file__), estimator_sha256=sha256(inspect.getfile(estimate_shift)),
                    source_sha256={s: sha256(p) for s, p in source_paths.items()})
    for s in SPLITS:
        if identity["source_sha256"][s] != entry["splits"][s]["sha256"]:
            raise ValueError("Pinned source bytes changed: " + s)
    target = work/"dataset_manifest.json"
    if target.is_file():
        previous = json.loads(target.read_text())
        if previous.get("identity") != identity:
            raise ValueError("Immutable data preflight identity changed; use an explicit revision")
        return previous
    audits = {}
    for split, path in source_paths.items():
        audits[split] = dict(path=str(path), sha256=identity["source_sha256"][split],
                            source_identity=entry["splits"][split]["source_identity"],
                            **audit_source(path, split, bands=bands, max_dn=max_dn,
                                expected_count=dict(train=train_n, val=val_n, rr=20, fr=20)[split],
                                expected_size=dict(train=64, val=64, rr=256, fr=512)[split]))
    # QB fixed-source identity and direct stored LMS reproduction bind L to the
    # exact msfix MS in this file; no old raw-QB LMS is mixed into the new source.
    result = dict(schema="TA2_NATIVE_LMS_DATA_v1", status="PASS", identity=identity,
                  sensor=sensor, bands=bands, max_dn=max_dn, band_order=entry["band_order"],
                  source_provenance=entry["source_provenance"], augmentation=AUGMENTATION,
                  augmentation_source_sha256=sha256(root/"feeders/feeder.py"), splits=audits,
                  fr_gt_accessed=False, regenerated_missing_lms=False,
                  missing_lms_policy="BLOCKED_LMS_LINEAGE; explicit independently verified operator asset required")
    immutable_json(target, result)
    return result


class NativeDataset:
    """Canonical read-only DN arrays; GT absence is split policy, not key discovery."""
    def __init__(self, path, split, *, bands, max_dn, expected_sha=None):
        if split not in SPLITS:
            raise ValueError("Unknown split")
        self.path, self.split, self.bands, self.max_dn = str(path), split, bands, float(max_dn)
        self.max_pixel = self.max_dn
        if expected_sha and sha256(path) != expected_sha:
            raise ValueError("Native dataset bytes changed after preflight")
        self.source_sha256 = expected_sha or sha256(path)
        keys = ("pan", "ms", "lms") + (() if split == "fr" else ("gt",))
        with h5py.File(path, "r") as f:
            # Official RR/FR references retain their original stored dtype;
            # only the training/validation RAM cache is reduced to FP32. Input
            # model normalization below is always performed exactly once.
            self.arrays = {k: np.asarray(f[k][:], np.float32 if split in ("train", "val") else None)
                           for k in keys}
        for x in self.arrays.values():
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite source array")
            x.flags.writeable = False
        self.base_count = len(self.arrays["pan"])
        self.has_gt = split != "fr"

    def __len__(self):
        return self.base_count

    def get_view(self, index, rot=None):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if rot is not None and int(rot) not in (0, 1, 2, 3):
            raise ValueError("Rotation must be 0..3")
        result = {}
        for k, array in self.arrays.items():
            x = array[index]
            if rot is not None:
                x = np.rot90(x[:, ::-1, ::-1], int(rot), axes=(-2, -1))
            result[k] = torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).mul_(2/self.max_dn).sub_(1)
        result.update(sample_id=index, rot=int(rot or 0), hflip=rot is not None, vflip=rot is not None)
        return result

    def base(self, index):
        return self.get_view(index)

    def __getitem__(self, index):
        return self.get_view(*index) if isinstance(index, (tuple, list)) else self.base(index)


class NativeStream:
    """Resumable sample permutation plus independent legacy augmentation RNG."""
    def __init__(self, count, seed):
        if count < 1:
            raise ValueError("Empty native sample stream")
        self.count, self.seed = int(count), int(seed)
        self.samples = np.random.default_rng(seed)
        self.augmentation = random.Random(seed)
        self.permutation = self.samples.permutation(count)
        self.cursor = 0

    def next(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Native batch size must be a positive integer")
        ids = []
        while len(ids) < batch_size:
            available = min(batch_size-len(ids), self.count-self.cursor)
            ids.extend(self.permutation[self.cursor:self.cursor+available].tolist())
            self.cursor += available
            if self.cursor == self.count:
                self.permutation = self.samples.permutation(self.count)
                self.cursor = 0
        return ids, [self.augmentation.randint(0, 3) for _ in ids]

    def state_dict(self):
        return dict(count=self.count, seed=self.seed, samples=self.samples.bit_generator.state,
                    augmentation=self.augmentation.getstate(), permutation=self.permutation.tolist(), cursor=self.cursor)

    def load_state_dict(self, state):
        if state["count"] != self.count or state["seed"] != self.seed:
            raise ValueError("Native stream seed/count changed on resume")
        self.samples.bit_generator.state = state["samples"]
        def tuples(x):
            return tuple(tuples(i) for i in x) if isinstance(x, (tuple, list)) else x
        self.augmentation.setstate(tuples(state["augmentation"]))
        self.permutation = np.asarray(state["permutation"], np.int64)
        self.cursor = int(state["cursor"])
        if sorted(self.permutation.tolist()) != list(range(self.count)) or not 0 <= self.cursor < self.count:
            raise ValueError("Invalid saved native sample stream")


class DatasetBundle:
    def __init__(self, repo_root, sensor, work_root, *, verify=True):
        if not verify:
            raise ValueError("Production native-LMS verification cannot be bypassed")
        self.root, self.work_root = Path(repo_root).resolve(), Path(work_root).resolve()
        self.manifest = preflight_sources(self.root, sensor, self.work_root)
        self.sensor, self.max_dn, self.bands = sensor.upper(), self.manifest["max_dn"], self.manifest["bands"]
        self.datasets = {split: NativeDataset(item["path"], split, bands=self.bands,
                         max_dn=self.max_dn, expected_sha=item["sha256"])
                         for split, item in self.manifest["splits"].items()}
        self._registration = {}
        self._case_shift_cache = {}

    def get(self, split, index, rot=None):
        return self.datasets[split].get_view(index, rot)

    def batch(self, split, ids, rots=None, device="cpu"):
        if rots is not None and len(rots) != len(ids):
            raise ValueError("Native augmentation metadata length mismatch")
        samples = [self.get(split, i, None if rots is None else r) for i, r in
                   zip(ids, [None]*len(ids) if rots is None else rots)]
        return {k: (torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k])
                    else torch.tensor([s[k] for s in samples])).to(device) for k in samples[0]}

    def offline_shifts(self, split, reference="NATIVE_LMS", descriptor="gradient", *, gt_train_only=False):
        reference = reference.upper()
        if reference == "GT" and (split != "train" or not gt_train_only):
            raise ValueError("GT cache is train supervision only; official inference must not access GT")
        if reference not in {"NATIVE_LMS", "BICUBIC_MS", "GT"}:
            raise ValueError("Unknown offline reference")
        key = (split, reference, descriptor)
        if key in self._registration:
            return self._registration[key]
        dataset = self.datasets[split]
        identity = dict(source_sha256=dataset.source_sha256, split=split, reference=reference,
                        descriptor=descriptor, estimator=ESTIMATOR,
                        source_code_sha256=sha256(inspect.getfile(estimate_shift)), max_dn=self.max_dn)
        target = self.work_root/"offline_shifts"/("_".join(key)+"_"+canonical_sha(identity)[:16]+".json")
        if target.exists():
            result = json.loads(target.read_text())
            if result.get("identity") != identity or len(result.get("rows", [])) != len(dataset):
                raise ValueError("Offline shift cache identity/count mismatch")
            cached_shifts = np.asarray(result.get("shifts"), dtype=np.float64)
            if (cached_shifts.shape != (len(dataset), 2) or not np.isfinite(cached_shifts).all()
                    or any(r.get("sample_id") != i or r.get("shift") != result["shifts"][i]
                           for i, r in enumerate(result["rows"]))):
                raise ValueError("Offline shift cache row/coordinate corruption")
        else:
            rows = []
            for i in range(len(dataset)):
                if reference == "BICUBIC_MS":
                    # Reproduce the actual model reference's FP32 normalization
                    # and interpolate order, not an almost-equivalent DN path.
                    normalized_ms = torch.tensor(dataset.arrays["ms"][i:i+1], dtype=torch.float32).mul_(2/self.max_dn).sub_(1)
                    base = F.interpolate(normalized_ms, scale_factor=4, mode="bicubic", align_corners=False)
                    ref = ((base[0]+1)*(self.max_dn/2)).numpy()
                else:
                    ref = dataset.arrays["gt" if reference == "GT" else "lms"][i]
                rows.append(dict(sample_id=i, **estimate_shift(dataset.arrays["pan"][i], ref,
                    descriptor=descriptor, target=reference, max_dn=self.max_dn, split=split)))
            result = dict(identity=identity, rows=rows, shifts=[r["shift"] for r in rows],
                          fallback_count=sum(not r["valid"] for r in rows),
                          registration_seconds=sum(r["registration_seconds"] for r in rows))
            immutable_json(target, result)
        result = dict(result, cache_path=str(target), cache_sha256=sha256(target))
        self._registration[key] = result
        return result

    def case_shifts(self, case_id, split):
        """Borrow a canonical no-grad tensor; callers index/transform, never edit.

        Training indexes with an ID list (advanced-index copy). Augmentation
        clones coordinate components, so canonical cached values never change.
        Cache keys include the split; a train GT proxy cannot enter inference.
        """
        case_id = case_id["case_id"] if isinstance(case_id, dict) else case_id
        short = case_id.removeprefix("TA2-")
        aliases = {"U01": "B03", "D01": "B03"}
        short = aliases.get(short, short)
        if split not in SPLITS:
            raise ValueError("Unknown fixed correction split")
        if short == "S07" and split != "train":
            raise ValueError("S07 pseudo shifts are training targets, never inference corrections")
        cache = getattr(self, "_case_shift_cache", None)
        if cache is None:
            self._case_shift_cache = cache = {}
        key = (short, split)
        if key in cache:
            return cache[key]
        def frozen(value):
            cache[key] = value.detach().contiguous()
            return cache[key]
        if short == "S07":
            item = self.offline_shifts(split, "GT", gt_train_only=True)
        elif short == "B02":
            shifts = self.offline_shifts("train")["shifts"]
            return frozen(torch.tensor(np.median(shifts, axis=0), dtype=torch.float32).repeat(len(self.datasets[split]), 1))
        elif short in {"B03", "B04", "B05", "B06", "B08"}:
            reference = "GT" if short == "B06" and split == "train" else "BICUBIC_MS" if short == "B05" else "NATIVE_LMS"
            item = self.offline_shifts(split, reference, "intensity" if short == "B04" else "gradient",
                                       gt_train_only=reference == "GT")
        elif short in {"B00", "B01", "U00", "D00"}:
            return frozen(torch.zeros(len(self.datasets[split]), 2))
        else:
            raise ValueError("Case has no fixed offline correction: " + str(case_id))
        shifts = torch.tensor(item["shifts"], dtype=torch.float32)
        if short == "B08":
            if len(shifts) < 2:
                raise ValueError("Cyclic derangement requires at least two images")
            shifts = shifts.roll(-1, dims=0)
        return frozen(shifts)

    def band_weights(self, cal_ids=None):
        dataset = self.datasets["train"]
        ids = list(range(min(CAL_COUNT, len(dataset)))) if cal_ids is None else [int(i) for i in cal_ids]
        if not ids or len(set(ids)) != len(ids) or any(not 0 <= i < len(dataset) for i in ids):
            raise ValueError("Invalid train CAL IDs")
        identity = dict(source_sha256=dataset.source_sha256, sample_ids=ids, split="train", uses_gt=False,
                        selection="first512 canonical train IDs frozen before any model/quality result" if cal_ids is None else "explicit frozen train CAL IDs",
                        code_sha256=sha256(inspect.getfile(fixed_band_weights)))
        path = self.work_root/"cal_band_weights.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != identity:
                raise ValueError("Frozen CAL set changed")
        else:
            result = dict(identity=identity, **fixed_band_weights(dataset.arrays["pan"][ids],
                          dataset.arrays["lms"][ids], max_dn=self.max_dn))
            immutable_json(path, result)
        return result


def transform_shift(delta, rot, hflip=True, vflip=True):
    """Singular-name compatibility; preserves gradient to global B07 parameters."""
    return transform_shifts(delta, rot, hflip=hflip, vflip=vflip)
