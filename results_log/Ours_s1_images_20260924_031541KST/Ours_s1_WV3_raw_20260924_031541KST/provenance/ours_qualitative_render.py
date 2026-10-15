"""Direct, non-generative DN-to-PNG previews; not CANConv paper panels.

The shared 1--99% RGB stretch follows tools/qrc24_qualitative.py.  Every
standalone image retains the input H/W, including borders.  The all-band raw
float32 predictions belong in a separate export; PNGs are display derivatives.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin


STATUS = "PREVIEW_ONLY_CANCONV_RENDERER_UNVERIFIED"
WV3_BANDS = ("Coastal", "Blue", "Green", "Yellow", "Red", "RedEdge", "NIR1", "NIR2")


def _cube(value, name, bands=None):
    array = np.asarray(value)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(f"{name} must be nonempty CHW, got {array.shape}")
    if bands is not None and array.shape[0] != bands:
        raise ValueError(f"{name} band count {array.shape[0]} != {bands}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric DN values")
    return array


def _rgb_indices(sensor, band_order, channels):
    sensor = str(sensor).upper()
    if sensor == "WV3" and channels == 8:
        labels = list(WV3_BANDS if band_order is None else band_order)
        expected = (4, 2, 1)
    elif sensor == "QB" and channels == 4:
        if band_order is None:
            raise ValueError("QB requires the verified source manifest's band_order")
        labels = list(band_order)
        expected = None
    else:
        raise ValueError(f"Unsupported sensor/channel pair: {sensor}/{channels}")
    if len(labels) != channels:
        raise ValueError("band_order does not match the prediction's band count")
    names = [str(b).strip().lower().replace("_", "").replace(" ", "") for b in labels]
    aliases = (("r", "red"), ("g", "green"), ("b", "blue"))
    indices = []
    for aliases_one in aliases:
        found = [i for i, name in enumerate(names) if name in aliases_one]
        if len(found) != 1:
            raise ValueError(f"Cannot uniquely locate RGB in band_order={labels}")
        indices.append(found[0])
    if expected is not None and tuple(indices) != expected:
        raise ValueError("WV3 source band order disagrees with the repository's RGB convention")
    return indices, labels


def shared_rgb(cube, rgb_indices, bounds):
    """Apply already-fixed reference bounds; never independently stretch output."""
    image = np.empty(cube.shape[1:] + (3,), dtype=np.float64)
    for channel, band in enumerate(rgb_indices):
        low, high = bounds[channel]
        image[..., channel] = (np.asarray(cube[band], dtype=np.float64) - low) / (high - low + 1e-12)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace different preview: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)


def _save_png(path, array, metadata):
    image = Image.fromarray(array)
    info = PngImagePlugin.PngInfo()
    info.add_text("Status", STATUS)
    info.add_text("Rendering", json.dumps(metadata, sort_keys=True, allow_nan=False))
    content = io.BytesIO()
    image.save(content, format="PNG", pnginfo=info, optimize=False)
    data = content.getvalue()
    _write_once(path, data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "size_wh": list(image.size), "mode": image.mode}


def render_scene(pred, raw, sensor, output_dir, index, band_order, max_dn, split):
    """Render one scene, returning ``{'files': [...], 'configuration': {...}}``.

    ``pred`` and raw ``lms``, ``pan``, and (RR only) ``gt`` are CHW DN arrays,
    not normalized tensors. ``output_dir`` is this sensor/split's preview root.
    Reference bounds use RR GT or FR native LMS, with linear gamma=1. The error
    preview is RR mean(abs(pred-GT)) across ALL bands, displayed in grayscale
    over a fixed [0, 0.05*max_dn] DN scale for every scene of this sensor.
    """
    if isinstance(index, bool) or int(index) != index or int(index) < 0:
        raise ValueError("index must be a nonnegative integer")
    split_aliases = {"rr": "reduced", "reduced": "reduced", "fr": "full", "full": "full"}
    if str(split).lower() not in split_aliases:
        raise ValueError("split must be reduced/rr or full/fr")
    split = split_aliases[str(split).lower()]
    max_dn = float(max_dn)
    if not np.isfinite(max_dn) or max_dn <= 0:
        raise ValueError("max_dn must be finite and positive")
    pred = _cube(pred, "prediction")
    channels, height, width = pred.shape
    lms = _cube(raw["lms"], "lms", channels)
    pan = _cube(raw["pan"], "pan", 1)
    reference_name = "gt" if split == "reduced" else "lms"
    ref = _cube(raw[reference_name], reference_name, channels)
    if lms.shape != pred.shape or ref.shape != pred.shape or pan.shape[1:] != (height, width):
        raise ValueError("Reference, LMS, PAN and prediction must have the same native H/W")
    indices, labels = _rgb_indices(sensor, band_order, channels)
    bounds = [list(map(float, np.percentile(ref[b], (1.0, 99.0)))) for b in indices]
    pan_bounds = list(map(float, np.percentile(pan[0], (1.0, 99.0))))
    configuration = {
        "status": STATUS, "paper_comparison_ready": False,
        "sensor": str(sensor).upper(), "split": split, "scene_index_zero_based": int(index),
        "band_order": labels, "rgb_indices_zero_based": indices,
        "input_axis": "CHW", "input_unit": "DN", "max_dn": max_dn,
        "source_shape_chw": list(pred.shape), "standalone_size_wh": [width, height],
        "reference": "RR GT" if split == "reduced" else "FR native LMS (not ground truth)",
        "stretch": "per-scene reference per-band percentiles, shared across all RGB views",
        "percentiles": [1.0, 99.0], "rgb_bounds_dn": bounds, "gamma": 1.0,
        "rgb_quantization": "clip display [0,1], round-to-nearest uint8 [0,255]",
        "stretch_denominator_epsilon": 1e-12,
        "pan_bounds_dn": pan_bounds, "crop": None, "resize": None, "roi": None,
        "error_formula": "mean_C(abs(pred_DN-GT_DN))" if split == "reduced" else None,
        "error_colormap": "linear grayscale" if split == "reduced" else None,
        "error_limits_dn": [0.0, 0.05 * max_dn] if split == "reduced" else None,
        "convention_source": "tools/qrc24_qualitative.py shared GT/LMS 1-99% convention; not CANConv comparison renderer",
        "canconv_renderer_verified": False, "canconv_scene_correspondence_verified": False,
        "standalone_pngs_unannotated_to_preserve_all_pixels": True,
        "rr_fr_scene_correspondence": "not asserted; scene indices belong to separate split orders",
        "source_arrays_modified": False,
    }
    prediction_rgb = shared_rgb(pred, indices, bounds)
    reference_rgb = shared_rgb(ref, indices, bounds)
    lms_rgb = shared_rgb(lms, indices, bounds)
    pan_gray = np.rint(np.clip((pan[0].astype(np.float64) - pan_bounds[0]) /
                              (pan_bounds[1] - pan_bounds[0] + 1e-12), 0, 1) * 255).astype(np.uint8)
    pan_rgb = np.repeat(pan_gray[..., None], 3, axis=2)
    views = [("output", prediction_rgb), ("reference", reference_rgb),
             ("input_lms", lms_rgb), ("input_pan", pan_rgb)]
    panels = [("PAN", pan_rgb), ("Native LMS", lms_rgb), ("Ours prediction", prediction_rgb)]
    if split == "reduced":
        error = np.mean(np.abs(pred.astype(np.float64) - ref.astype(np.float64)), axis=0)
        error_gray = np.rint(np.clip(error / (0.05 * max_dn), 0, 1) * 255).astype(np.uint8)
        error_rgb = np.repeat(error_gray[..., None], 3, axis=2)
        views.append(("errormap", error_rgb))
        panels += [("GT", reference_rgb), ("Mean abs error (DN)", error_rgb)]
        configuration["error_summary_dn"] = {"mean": float(error.mean()), "max": float(error.max())}
    header_h, footer_h = 25, 32
    canvas = Image.new("RGB", (width * len(panels), height + header_h + footer_h), color="white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, rgb) in enumerate(panels):
        canvas.paste(Image.fromarray(rgb), (column * width, header_h))
        draw.text((column * width + 5, 5), label, fill="black")
    caption = (f"PREVIEW ONLY | {str(sensor).upper()} {split} scene {int(index)} | "
               "shared reference 1-99% RGB stretch; no crop; gamma=1")
    draw.text((5, height + header_h + 3), caption, fill="black")
    draw.text((5, height + header_h + 16), "CANConv renderer/ROI/scene correspondence NOT verified.", fill="black")
    views.append(("compare_output", np.asarray(canvas)))
    output_dir = Path(output_dir)
    files = []
    for folder, array in views:
        file_info = _save_png(output_dir / folder / f"testimg{int(index)}.png", array, configuration)
        file_info["path"] = str(Path(file_info["path"]).relative_to(output_dir))
        files.append(file_info)
    result = {"files": files, "configuration": configuration}
    _write_once(output_dir / "render_metadata" / f"testimg{int(index)}.json",
                (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return result
