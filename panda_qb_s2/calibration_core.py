"""SensorFocus-local exact RGB/Lanczos-4 RR-train calibration.

Copied from panda_stage2_condq.calibration to preserve frozen WV3 dependencies.
The ECDF, serialization schema and mask reduction token remain compatible;
operator binding and the actual valid-pixel population explicitly use Lanczos-4.

This module never imports the old MASK2 calibration.  Accumulation is deliberately
in-memory and exact: callers must supply a peak-memory budget, and an over-budget
inventory fails before allocating population arrays.  There is no sampled/LUT
fallback.  The budget covers this module's population/sort workspace, not models,
input batches, the Python interpreter, or an accelerator allocator's cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import torch

from donor30.masking import validate_mask
from panda_stage2_condq.spec import MASK_POLICY, MASK2_POLICY, MASK2_CALIBRATION_SCHEMA
from .contract import CAMPAIGN_ID


SCHEMA = "PANDA_S2_CONDQ_TRAIN_GLOBAL_V1"
ROI_MARGIN = 8
WARP_CONTRACT = "lanczos4_reflect_FP32_current_pan_px"
ERROR_REDUCTION = "all_bands_mean_all_train_pixels_unmasked"
MASK2_ERROR_REDUCTION = "all_bands_mean_all_train_lanczos4_valid_pixels"
RGB_INDICES = {"GF2": [2, 1, 0], "QB": [2, 1, 0]}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("sample IDs must be a sequence, not a string")
    result = list(values)
    if not result or any(not isinstance(v, str) or not v for v in result):
        raise ValueError("sample IDs must be nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError("duplicate sample IDs")
    return result


def sample_ids_sha256(values: Sequence[str]) -> str:
    """Inventory identity is independent of the order of calibration batches."""
    # Match the repository's existing canonical_sha256 inventory contract,
    # including ASCII-escaped JSON for possible non-ASCII sample identifiers.
    encoded = json.dumps(sorted(_ids(values)), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete source/donor/operator contract, preserving extras."""
    value = json.loads(_canonical(dict(binding)))
    required = {"sensor", "source_split", "sample_ids_sha256", "source_manifest_sha256",
                "aligner_state_sha256", "teacher_state_sha256", "donor_checkpoint_id",
                "reconstruction_spec", "scharr", "warp"}
    if required - value.keys():
        raise ValueError(f"calibration binding missing {sorted(required - value.keys())}")
    sensor = value["sensor"]
    if sensor not in RGB_INDICES or value["source_split"] != "RR_train_only":
        raise ValueError("calibration requires one supported sensor and RR_train_only")
    for key in ("sample_ids_sha256", "source_manifest_sha256", "aligner_state_sha256",
                "teacher_state_sha256"):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA256 binding: {key}")
    if not isinstance(value["donor_checkpoint_id"], str) or not value["donor_checkpoint_id"]:
        raise ValueError("donor_checkpoint_id must identify the single frozen A/T checkpoint")
    expected_model = {"bands": 8 if sensor == "WV3" else 4, "width": 112,
                      "depth": [1, 2, 3], "input_layout": "P0", "base_reference": "BICUBIC_MS"}
    expected_scharr = {"kernel": "absolute_xy_/32", "padding": "reflect1",
                       "rgb_indices": RGB_INDICES[sensor], "roi_margin": ROI_MARGIN}
    if value["reconstruction_spec"] != expected_model:
        raise ValueError("calibration requires W112-D123/P0/BICUBIC_MS reconstruction")
    if value["scharr"] != expected_scharr or value["warp"] != WARP_CONTRACT:
        raise ValueError("calibration Scharr/warp operator binding mismatch")
    from .contract import validate_source_teacher
    validate_source_teacher(value)
    return value


def _array_hash(metadata: Mapping[str, Any], cuts: np.ndarray,
                q_arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    clean = {k: v for k, v in metadata.items() if k != "calibration_artifact_sha256"}
    digest.update(_canonical(clean))
    for name, array in [("g_cutpoints_f64", cuts), *[(f"q_{j:04d}", q) for j, q in enumerate(q_arrays)]]:
        array = np.ascontiguousarray(array)
        digest.update(_canonical({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _positive_budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("memory_budget_bytes must be an explicit positive integer")
    return value


def _archive_headers(path: os.PathLike) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Check NPY dimensions before np.load can allocate arrays from a bad file."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != len({entry.filename for entry in infos}):
            raise ValueError("duplicate calibration archive members")
        for entry in infos:
            if not entry.filename.endswith(".npy") or "/" in entry.filename:
                raise ValueError("unexpected calibration archive member")
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("unsupported calibration NPY version")
            if fortran or dtype.hasobject or any(not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError("invalid calibration array storage layout")
            headers[entry.filename[:-4]] = (shape, dtype)
    return headers


class CalibrationStats:
    """Fixed exact tables. Device placement is explicit and never changes in lookup.

    Public metadata/binding accessors return copies. The persisted artifact is
    immutable (exclusive publication; no overwrite), content-hashed and schema
    checked. Its hash is a canonical content hash, distinct from the ZIP file hash.
    """

    def __init__(self, metadata: Mapping[str, Any], cuts: np.ndarray,
                 q_arrays: Sequence[np.ndarray]):
        self._metadata = copy.deepcopy(dict(metadata))
        self._binding = validate_binding(self._metadata.get("binding", {}))
        policy = self._metadata.get("mask_policy", MASK_POLICY)
        expected_schema = SCHEMA if policy == MASK_POLICY else MASK2_CALIBRATION_SCHEMA
        expected_reduction = ERROR_REDUCTION if policy == MASK_POLICY else MASK2_ERROR_REDUCTION
        if policy not in (MASK_POLICY, MASK2_POLICY) or self._metadata.get("schema") != expected_schema:
            raise ValueError("incompatible calibration schema/policy; old donor30 artifacts are forbidden")
        if self._metadata.get("teacher_error_reduction") != expected_reduction:
            raise ValueError("calibration tau_e population differs from the declared mask policy")
        tau = self._metadata.get("tau_e")
        if not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau < 0:
            raise ValueError("invalid calibrated tau_e")
        if self._metadata.get("median_rule") != "midpoint_even_global_pixels":
            raise ValueError("incompatible global median rule")
        self._validate_arrays(cuts, q_arrays)
        actual = _array_hash(self._metadata, cuts, q_arrays)
        if actual != self._metadata.get("calibration_artifact_sha256"):
            raise ValueError("calibration artifact content SHA256 mismatch")
        self._cuts = torch.from_numpy(np.array(cuts, dtype=np.float64, copy=True))
        self._q = tuple(torch.from_numpy(np.array(q, dtype=np.float32, copy=True)) for q in q_arrays)
        self._table_signature = self._current_table_signature()

    def _current_table_signature(self) -> tuple:
        return tuple((id(value), value._version, value.data_ptr(), str(value.device),
                      value.dtype, tuple(value.shape)) for value in (self._cuts, *self._q))

    def assert_integrity(self, *, full_hash: bool = False) -> None:
        """Cheap mutation guard per lookup; optional full content audit at lifecycle boundaries."""
        if self._current_table_signature() != self._table_signature:
            raise ValueError("immutable calibration tensor changed after finalization")
        if full_hash:
            cuts, q_arrays = self._cpu_arrays()
            if _array_hash(self._metadata, cuts, q_arrays) != self.artifact_sha256:
                raise ValueError("immutable calibration content SHA256 changed")

    def _validate_arrays(self, cuts: np.ndarray, q_arrays: Sequence[np.ndarray]) -> None:
        if cuts.dtype != np.dtype("float64") or cuts.ndim != 1 or not np.isfinite(cuts).all():
            raise ValueError("g cutpoints must be finite FP64 vector")
        if cuts.size and (cuts[0] < 0 or not np.all(cuts[1:] > cuts[:-1])):
            raise ValueError("g cutpoints must be nonnegative and strictly increasing")
        kind = self._metadata.get("calibration_kind")
        condq = kind == "exact_condq"
        if kind not in {"exact_condq", "tau_e_only"}:
            raise ValueError("unknown calibration kind")
        counts = []
        for array in q_arrays:
            if array.dtype != np.dtype("float32") or array.ndim != 1 or not array.size:
                raise ValueError("each q bin must be a nonempty FP32 vector")
            if not np.isfinite(array).all() or array[0] < 0 or not np.all(array[1:] >= array[:-1]):
                raise ValueError("q ECDF samples must be finite, nonnegative, sorted")
            counts.append(int(array.size))
        if condq and len(q_arrays) != cuts.size + 1:
            raise ValueError("every retained g bin needs a nonempty q ECDF")
        if not condq and (cuts.size or q_arrays):
            raise ValueError("tau_e_only calibration cannot contain q tables")
        grouping = self._metadata.get("g_grouping", {})
        if condq and (grouping.get("quantile_method") != "linear"
                      or grouping.get("boundary_side") != "right"
                      or grouping.get("edge_dtype") != "float64"
                      or grouping.get("k_effective") != len(q_arrays)):
            raise ValueError("incompatible g bin semantics")
        if condq and self._metadata.get("q_ecdf") != {
            "operator": "right_continuous_leq", "dtype": "float32", "storage": "exact_sorted_arrays"
        }:
            raise ValueError("incompatible q ECDF semantics")
        if self._metadata.get("bin_counts") != counts:
            raise ValueError("q bin count mismatch")
        n_samples = self._metadata.get("sample_count", 0)
        shape = self._metadata.get("patch_shape", [])
        if not isinstance(n_samples, int) or n_samples <= 0 or len(shape) != 2:
            raise ValueError("missing complete calibration sample/shape counts")
        if any(not isinstance(n, int) or n <= 2 * ROI_MARGIN for n in shape):
            raise ValueError("invalid calibration patch shape")
        h, w = shape
        if self.mask_policy == MASK_POLICY:
            if self._metadata.get("n_e") != n_samples * h * w:
                raise ValueError("full-image e population count mismatch")
        else:
            counts_by_id = self._metadata.get("valid_e_counts_by_sample", [])
            if (not isinstance(counts_by_id, list) or len(counts_by_id) != n_samples
                    or any(not isinstance(row, list) or len(row) != 2
                           or not isinstance(row[0], str) or type(row[1]) is not int
                           or not (h-14)*(w-14) <= row[1] <= h*w for row in counts_by_id)):
                raise ValueError("MASK2 requires complete per-sample valid e population counts")
            ids = [row[0] for row in counts_by_id]
            if ids != sorted(ids) or sample_ids_sha256(ids) != self._binding["sample_ids_sha256"]:
                raise ValueError("MASK2 valid e count sample inventory mismatch")
            if self._metadata.get("n_e") != sum(row[1] for row in counts_by_id):
                raise ValueError("MASK2 valid e population count mismatch")
        expected_q = n_samples * (h - 2 * ROI_MARGIN) * (w - 2 * ROI_MARGIN) if condq else 0
        if self._metadata.get("n_q") != expected_q or sum(counts) != expected_q:
            raise ValueError("ROI8 q population count mismatch")
        if self._metadata.get("unaugmented") is not True:
            raise ValueError("calibration must be unaugmented")
        if "sample_count" in self._binding and self._binding["sample_count"] != n_samples:
            raise ValueError("calibration population count disagrees with bound inventory")

    @property
    def tau_e(self) -> float:
        return float(self._metadata["tau_e"])

    @property
    def schema(self) -> str:
        return self._metadata["schema"]

    @property
    def mask_policy(self) -> str:
        return self._metadata.get("mask_policy", MASK_POLICY)

    @property
    def calibration_kind(self) -> str:
        return self._metadata["calibration_kind"]

    @property
    def binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._binding)

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def artifact_sha256(self) -> str:
        return self._metadata["calibration_artifact_sha256"]

    @property
    def device(self) -> torch.device:
        return self._cuts.device

    @property
    def memory_bytes(self) -> int:
        return self._cuts.numel() * 8 + sum(q.numel() * 4 for q in self._q)

    @property
    def bin_counts(self) -> tuple[int, ...]:
        return tuple(int(q.numel()) for q in self._q)

    def validate_binding(self, expected: Mapping[str, Any]) -> None:
        value = validate_binding(expected)
        if value != self._binding:
            differing = sorted(k for k in self._binding.keys() | value.keys()
                               if self._binding.get(k) != value.get(k))
            raise ValueError(f"calibration donor/data/operator binding mismatch: {differing}")

    def to(self, device: Any, *, memory_budget_bytes: int | None = None) -> "CalibrationStats":
        """Explicit allocation; returning self on the existing device is intentional."""
        self.assert_integrity()
        device = torch.device(device)
        if memory_budget_bytes is not None and self.memory_bytes > _positive_budget(memory_budget_bytes):
            raise MemoryError("exact ECDF device tables exceed the explicit memory budget")
        if self.device == device:
            return self
        if memory_budget_bytes is None:
            raise ValueError("moving exact ECDF tables requires an explicit memory_budget_bytes")
        result = object.__new__(type(self))
        result._metadata = copy.deepcopy(self._metadata)
        result._binding = copy.deepcopy(self._binding)
        result._cuts = self._cuts.to(device=device)
        result._q = tuple(q.to(device=device) for q in self._q)
        result._table_signature = result._current_table_signature()
        return result

    @torch.no_grad()
    def group_counts(self, g_roi: torch.Tensor) -> torch.Tensor:
        """Query population per bin (distinct from stored train bin_counts)."""
        self.assert_integrity()
        if self.calibration_kind != "exact_condq":
            raise ValueError("tau_e_only calibration has no conditional-q bins")
        if g_roi.ndim != 4 or g_roi.shape[1] != 1 or not g_roi.numel() or g_roi.dtype != torch.float32:
            raise ValueError("g query requires nonempty FP32 [B,1,H,W]")
        if g_roi.device != self.device:
            raise ValueError("place calibration and g explicitly on the same device")
        if not bool(torch.isfinite(g_roi).all()) or bool((g_roi < 0).any()):
            raise ValueError("g query must be finite and nonnegative")
        groups = torch.bucketize(g_roi.detach().reshape(-1).double(), self._cuts, right=True)
        return torch.bincount(groups, minlength=len(self._q))

    @torch.no_grad()
    def lookup_exact(self, g_roi: torch.Tensor, q_roi: torch.Tensor) -> torch.Tensor:
        self.assert_integrity()
        if self._metadata["calibration_kind"] != "exact_condq":
            raise ValueError("tau_e_only calibration cannot perform a conditional-q lookup")
        if g_roi.shape != q_roi.shape or g_roi.ndim != 4 or g_roi.shape[1] != 1 or not g_roi.numel():
            raise ValueError("g/q lookup requires matching nonempty [B,1,H,W] maps")
        if g_roi.dtype != torch.float32 or q_roi.dtype != torch.float32:
            raise ValueError("g/q queries must be FP32 Scharr values")
        if g_roi.device != self.device or q_roi.device != self.device:
            raise ValueError("place calibration and g/q explicitly on the same device before lookup")
        if not bool(torch.isfinite(g_roi).all()) or not bool(torch.isfinite(q_roi).all()):
            raise ValueError("nonfinite g/q query")
        if bool((g_roi < 0).any()) or bool((q_roi < 0).any()):
            raise ValueError("negative Scharr g/q query")
        groups = torch.bucketize(g_roi.detach().reshape(-1).double(), self._cuts, right=True)
        queries = q_roi.detach().reshape(-1).contiguous()
        ranks = torch.empty_like(queries)
        for j, sorted_q in enumerate(self._q):
            chosen = groups == j
            if bool(chosen.any()):
                count = torch.searchsorted(sorted_q, queries[chosen], right=True)
                ranks[chosen] = count.to(torch.float32) / sorted_q.numel()
        return ranks.reshape_as(q_roi)

    def _cpu_arrays(self) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        cuts = self._cuts.detach().cpu().numpy()
        q = tuple(v.detach().cpu().numpy() for v in self._q)
        return cuts, q

    def save(self, path: os.PathLike) -> dict[str, Any]:
        """Atomically publish one verified artifact; never replace an existing path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable calibration already exists: {path}")
        self.assert_integrity()
        cuts, q_arrays = self._cpu_arrays()
        # Detect accidental in-process table mutation before publishing.
        if _array_hash(self._metadata, cuts, q_arrays) != self.artifact_sha256:
            raise ValueError("calibration tables changed after finalization")
        arrays = {"metadata_json": np.frombuffer(_canonical(self._metadata), dtype=np.uint8),
                  "g_cutpoints_f64": cuts}
        arrays.update({f"q_{j:04d}": q for j, q in enumerate(q_arrays)})
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            loaded = type(self).load(temporary, expected_binding=self.binding)
            if loaded.artifact_sha256 != self.artifact_sha256:
                raise ValueError("calibration serialization read-back mismatch")
            archive_hash = file_sha256(temporary)
            # link() is atomic and exclusive, unlike replace(); a racing writer loses safely.
            os.link(temporary, path)
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            return {"path": str(path), "artifact_sha256": self.artifact_sha256,
                    "file_sha256": archive_hash, "memory_bytes": self.memory_bytes}
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: os.PathLike, *, expected_binding: Mapping[str, Any],
             expected_file_sha256: str | None = None,
             memory_budget_bytes: int | None = None,
             expected_mask_policy: str | None = None) -> "CalibrationStats":
        if expected_file_sha256 is not None and file_sha256(path) != expected_file_sha256:
            raise ValueError("calibration archive file SHA256 mismatch")
        headers = _archive_headers(path)
        meta_shape, meta_dtype = headers.get("metadata_json", ((), None))
        if len(meta_shape) != 1 or meta_dtype != np.dtype("uint8") or meta_shape[0] > 16 * 1024 * 1024:
            raise ValueError("invalid calibration metadata encoding")
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files or "g_cutpoints_f64" not in archive.files:
                raise ValueError("not a Stage 2 calibration archive")
            encoded = archive["metadata_json"]
            if encoded.dtype != np.uint8 or encoded.ndim != 1 or encoded.size > 16 * 1024 * 1024:
                raise ValueError("invalid calibration metadata encoding")
            metadata = json.loads(encoded.tobytes())
            if (expected_mask_policy is not None
                    and metadata.get("mask_policy", MASK_POLICY) != expected_mask_policy):
                raise ValueError("calibration loss mask policy mismatch")
            if validate_binding(metadata.get("binding", {})) != validate_binding(expected_binding):
                raise ValueError("calibration donor/data/operator binding mismatch")
            counts = metadata.get("bin_counts", [])
            if not isinstance(counts, list) or any(not isinstance(n, int) or n <= 0 for n in counts):
                raise ValueError("invalid calibration bin counts")
            expected_keys = {"metadata_json", "g_cutpoints_f64", *[f"q_{j:04d}" for j in range(len(counts))]}
            if set(archive.files) != expected_keys or len(archive.files) != len(expected_keys):
                raise ValueError("missing, duplicate, or unexpected calibration arrays")
            if headers["g_cutpoints_f64"] != ((max(0, len(counts)-1),), np.dtype("float64")):
                raise ValueError("g cutpoint archive size/dtype differs from metadata")
            for j, count in enumerate(counts):
                if headers[f"q_{j:04d}"] != ((count,), np.dtype("float32")):
                    raise ValueError("q archive size/dtype differs from population counts")
            # Loading plus validation/copy can hold three copies transiently.
            estimate = 3 * (8 * max(0, len(counts) - 1) + 4 * sum(counts)) + int(encoded.size)
            if memory_budget_bytes is not None and estimate > _positive_budget(memory_budget_bytes):
                raise MemoryError("exact calibration load exceeds the explicit memory budget")
            cuts = archive["g_cutpoints_f64"]
            q_arrays = [archive[f"q_{j:04d}"] for j in range(len(counts))]
        result = cls(metadata, cuts, q_arrays)
        result.validate_binding(expected_binding)
        return result


class CalibrationBuilder:
    """One pass over the exact, fixed train inventory, followed by exact sorting.

    Any failed add/finalize poisons this builder. Callers cannot catch a failed
    batch and accidentally publish a partial population as a successful artifact.
    e_t/g/q are full-grid FP32 maps; only this class slices g/q to fixed ROI8.
    """

    def __init__(self, expected_sample_ids: Sequence[str], binding: Mapping[str, Any], *,
                 k: int = 5, memory_budget_bytes: int, include_condq: bool = True,
                 mask_policy: str = MASK_POLICY):
        self._expected = _ids(expected_sample_ids)
        self._binding = validate_binding(binding)
        if sample_ids_sha256(self._expected) != self._binding["sample_ids_sha256"]:
            raise ValueError("sample inventory hash differs from calibration binding")
        if "sample_count" in self._binding and self._binding["sample_count"] != len(self._expected):
            raise ValueError("sample inventory count differs from calibration binding")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be an explicit positive integer")
        if not isinstance(include_condq, bool):
            raise ValueError("include_condq must be boolean")
        if mask_policy not in (MASK_POLICY, MASK2_POLICY):
            raise ValueError("Explicit full-image or MASK2 calibration policy required")
        self.mask_policy = mask_policy
        self._e_cursor = 0
        self._valid_e_counts = {}
        self.k = k
        self.memory_budget_bytes = _positive_budget(memory_budget_bytes)
        self.include_condq = include_condq
        self._expected_set = set(self._expected)
        self._seen: set[str] = set()
        self._shape: tuple[int, int] | None = None
        self._e: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._state = "collecting"
        self.estimated_peak_bytes = 0

    @staticmethod
    def _check_map(value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{name} must be an FP32 tensor")
        if value.ndim != 4 or value.shape[1] != 1 or not value.numel():
            raise ValueError(f"{name} must have shape [B,1,H,W]")
        if value.requires_grad or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"{name} must be detached, finite and nonnegative")

    def _allocate(self, shape: tuple[int, int]) -> None:
        h, w = shape
        n = len(self._expected)
        ne = n * h * w
        nq = n * (h - 16) * (w - 16) if self.include_condq else 0
        # Conservative peak includes original populations, FP64 g quantiles,
        # int64 membership, masks, sorted bins and stats' own immutable copies.
        estimate = ne * 8 + nq * 64 + n * 1024 + self.k * 1024 + 65536
        self.estimated_peak_bytes = estimate
        if estimate > self.memory_budget_bytes:
            raise MemoryError(f"exact calibration estimated peak {estimate} exceeds budget {self.memory_budget_bytes}; no approximation fallback")
        self._e = np.empty(ne, dtype=np.float32)
        if self.include_condq:
            self._g = np.empty(nq, dtype=np.float32)
            self._q = np.empty(nq, dtype=np.float32)
        self._shape = shape

    def add(self, sample_ids: Sequence[str], e_t: torch.Tensor,
            g: torch.Tensor | None = None, q: torch.Tensor | None = None, *,
            sensor: str, split: str = "train", resolution: str = "RR",
            unaugmented: bool = True, support_ok: bool = True,
            loss_mask: torch.Tensor | None = None) -> None:
        if self._state != "collecting":
            raise RuntimeError(f"calibration builder is {self._state}")
        try:
            ids = _ids(sample_ids)
            if sensor != self._binding["sensor"] or split != "train" or resolution != "RR":
                raise ValueError("calibration forbids wrong sensor, validation/test or FR samples")
            if unaugmented is not True or support_ok is not True:
                raise ValueError("calibration requires unaugmented samples and valid ROI8 Scharr support")
            if set(ids) - self._expected_set or set(ids) & self._seen:
                raise ValueError("unknown or duplicate calibration sample ID")
            self._check_map(e_t, "e_t")
            if e_t.shape[0] != len(ids):
                raise ValueError("sample ID count and map batch size differ")
            shape = (int(e_t.shape[-2]), int(e_t.shape[-1]))
            if min(shape) <= 2 * ROI_MARGIN:
                raise ValueError("calibration patches must have nonempty fixed ROI8")
            if self.mask_policy == MASK2_POLICY:
                if loss_mask is None:
                    raise ValueError("MASK2 calibration requires original PAN Lanczos-4 support mask")
                loss_mask, valid_counts = validate_mask(e_t, loss_mask)
                if not bool(loss_mask[..., 7:-7, 7:-7].all()):
                    raise ValueError("MASK2 calibration requires full ROI8 Scharr stencil support")
            elif loss_mask is not None:
                raise ValueError("Full-image calibration forbids a loss mask")
            if self.include_condq:
                self._check_map(g, "g")
                self._check_map(q, "q")
                if g.shape != e_t.shape or q.shape != e_t.shape or g.device != e_t.device or q.device != e_t.device:
                    raise ValueError("calibration e/g/q require matching full-grid shapes and devices")
            elif g is not None or q is not None:
                raise ValueError("tau_e_only collection must not execute/collect q or g")
            if self._shape is None:
                self._allocate(shape)
            elif self._shape != shape:
                raise ValueError("calibration inventory patch shape changed")
            h, w = shape
            offset = len(self._seen)
            end = offset + len(ids)
            values = e_t.detach() if loss_mask is None else e_t.detach()[loss_mask]
            values = values.cpu().numpy().reshape(-1)
            self._e[self._e_cursor:self._e_cursor+values.size] = values
            self._e_cursor += values.size
            if loss_mask is not None:
                self._valid_e_counts.update(zip(ids, map(int, valid_counts.cpu().tolist())))
            if self.include_condq:
                roi_pixels = (h-16)*(w-16)
                target = slice(offset*roi_pixels, end*roi_pixels)
                self._g[target] = g[..., 8:-8, 8:-8].detach().cpu().numpy().reshape(-1)
                self._q[target] = q[..., 8:-8, 8:-8].detach().cpu().numpy().reshape(-1)
            self._seen.update(ids)
        except BaseException:
            self._state = "failed"
            raise

    def finalize(self) -> CalibrationStats:
        if self._state != "collecting":
            raise RuntimeError(f"calibration builder is {self._state}")
        try:
            if self._seen != self._expected_set or self._e is None:
                raise ValueError(f"incomplete train inventory: {len(self._expected_set - self._seen)} missing sample IDs")
            # Midpoint arithmetic is explicitly FP64, unlike numpy.median(float32).
            self._e = self._e[:self._e_cursor]
            n_e = int(self._e.size)
            mids = ((n_e - 1) // 2, n_e // 2)
            self._e.partition(mids)
            tau_e = (float(self._e[mids[0]]) + float(self._e[mids[1]])) / 2.0
            cuts = np.empty(0, dtype=np.float64)
            q_arrays: list[np.ndarray] = []
            grouping: dict[str, Any] = {}
            ties: list[dict[str, Any]] = []
            n_q = 0
            if self.include_condq:
                n_q = int(self._q.size)
                g64 = self._g.astype(np.float64)
                probs = np.arange(1, self.k, dtype=np.float64) / self.k
                raw_cuts = np.quantile(g64, probs, method="linear") if probs.size else np.empty(0, dtype=np.float64)
                cuts = np.unique(raw_cuts).astype(np.float64)
                duplicate_count = int(raw_cuts.size - cuts.size)
                merges = []
                while True:
                    groups = np.searchsorted(cuts, g64, side="right")
                    counts = np.bincount(groups, minlength=cuts.size + 1)
                    empty = np.flatnonzero(counts == 0)
                    if not empty.size:
                        break
                    j = int(empty[0])
                    boundary = 0 if j == 0 else j - 1
                    merges.append({"empty_bin": j, "removed_cutpoint": float(cuts[boundary])})
                    cuts = np.delete(cuts, boundary)
                for j in range(cuts.size + 1):
                    values = self._q[groups == j]
                    values.sort()
                    q_arrays.append(values)
                    adjacent_equal = values[1:] == values[:-1]
                    tied = np.zeros(values.size, dtype=np.bool_)
                    tied[:-1] |= adjacent_equal
                    tied[1:] |= adjacent_equal
                    ties.append({"n": int(values.size), "zero_count": int(np.searchsorted(values, np.float32(0), side="right")),
                                 "tied_sample_count": int(np.count_nonzero(tied)),
                                 "duplicate_excess_count": int(np.count_nonzero(adjacent_equal))})
                grouping = {"k_requested": self.k, "k_effective": len(q_arrays),
                            "quantile_method": "linear", "quantile_api": "numpy.quantile", "numpy_version": np.__version__,
                            "boundary_side": "right", "edge_dtype": "float64", "scharr_dtype": "float32",
                            "raw_quantile_cutpoints": raw_cuts.tolist(), "duplicate_boundaries_removed": duplicate_count,
                            "empty_bin_merge_rule": "remove_lower_boundary_first_empty; first_bin_remove_upper",
                            "empty_bin_merges": merges}
            metadata = {
                "schema": SCHEMA, "binding": self._binding,
                "calibration_kind": "exact_condq" if self.include_condq else "tau_e_only",
                "teacher_error_reduction": ERROR_REDUCTION, "median_rule": "midpoint_even_global_pixels",
                "tau_e": tau_e, "n_e": n_e, "n_q": n_q,
                "sample_count": len(self._expected), "patch_shape": list(self._shape),
                "unaugmented": True, "g_grouping": grouping,
                "q_ecdf": {"operator": "right_continuous_leq", "dtype": "float32", "storage": "exact_sorted_arrays"} if self.include_condq else None,
                "bin_counts": [int(v.size) for v in q_arrays], "q_ties_by_bin": ties,
                "estimated_peak_bytes": self.estimated_peak_bytes,
                "memory_budget_bytes": self.memory_budget_bytes,
                "algorithm_version": "exact_numpy_linear_f64_right_torch_leq_v1",
            }
            if self.mask_policy == MASK2_POLICY:
                metadata.update(schema=MASK2_CALIBRATION_SCHEMA,
                    mask_policy=MASK2_POLICY, teacher_error_reduction=MASK2_ERROR_REDUCTION,
                    valid_e_counts_by_sample=[[key, self._valid_e_counts[key]]
                                             for key in sorted(self._valid_e_counts)],
                    spatial_population="full_fixed_ROI8_guarded_not_adaptively_cropped",
                    algorithm_version="exact_numpy_linear_f64_right_torch_leq_MASK2_v1")
            metadata["calibration_artifact_sha256"] = _array_hash(metadata, cuts, q_arrays)
            result = CalibrationStats(metadata, cuts, q_arrays)
            self._state = "finalized"
            self._e = self._g = self._q = None
            return result
        except BaseException:
            self._state = "failed"
            raise
