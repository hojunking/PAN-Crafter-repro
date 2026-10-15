"""GDAL's actual eight-band equal-weight WeightedBrovey baseline.

The function accepts verified nonnegative product units, NOT normalized [-1,1]
network inputs. It does not infer NoData, alter geometry, estimate weights, clip
output to an assumed bit depth, or substitute an approximate numpy algorithm.

API and options follow GDAL's own VRT documentation and anonymous MEM test:
https://gdal.org/en/stable/drivers/raster/vrt.html#pansharpened-vrt
https://github.com/OSGeo/gdal/blob/v3.4.1/autotest/gdrivers/vrtpansharpen.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = {
    "id": "PANPV_GDAL_WEIGHTED_BROVEY8_EQUAL_CUBIC_F32_v1",
    "algorithm": "GDAL WeightedBrovey",
    "baseline_name": "eight-band equal-weight baseline (not optimized WV2 SRF)",
    "weights": [0.125] * 8,
    "input_layout": {"pan": "HW", "ms": "8hw"},
    "input_units": "verified nonnegative product units; same PAN/MS encoding",
    "output_layout": "8HW",
    "input_and_output_dtype": "Float32",
    "resampling": "Cubic",
    "num_threads": 1,
    "nodata": "None",
    "finite_zero_is_valid": True,
    "bit_depth": None,
    "nbits_metadata": "absent on anonymous Float32 MEM inputs; never inherited",
    "spatial_extent_adjustment": "None",
    "pan_geotransform": [0, 1, 0, 0, 0, -1],
    "ms_geotransform": [0, 4, 0, 0, 0, -4],
    "projection": "none; pixel grid only, not claimed georeferencing",
    "ratio": 4,
    "output_clipping": "none",
    "zero_pseudo_pan": "GDAL ComputeFactor uses zero ratio; output all eight bands zero",
    "mask_policy": "No inferred mask; caller owns declared common evaluation support",
}


def _gdal():
    try:
        from osgeo import gdal
    except ImportError as error:
        raise RuntimeError("GDAL runtime unavailable; use the isolated panpv/Dockerfile image") from error
    if not hasattr(gdal, "CreatePansharpenedVRT"):
        raise RuntimeError("GDAL has no CreatePansharpenedVRT binding")
    return gdal


def config_metadata():
    """JSON-compatible frozen configuration plus actual runtime/source identity."""
    gdal = _gdal()
    config = json.loads(json.dumps(CONFIG))
    return {"config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "software": {"gdal_release": gdal.VersionInfo("RELEASE_NAME"),
                         "gdal_version_num": gdal.VersionInfo("VERSION_NUM"),
                         "gdal_build_info": gdal.VersionInfo("BUILD_INFO"),
                         "numpy": np.__version__},
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _inputs(pan_hw, ms_chw):
    for name, source in (("PAN", pan_hw), ("MS", ms_chw)):
        if np.ma.isMaskedArray(source):
            raise ValueError(name + " masked arrays are not accepted; provide data and manage common mask separately")
        array = np.asarray(source)
        if array.dtype.kind not in "uif":
            raise ValueError(name + " must contain real product values")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(name + " must be finite and nonnegative, not normalized network values")
    pan, ms = np.asarray(pan_hw), np.asarray(ms_chw)
    if pan.ndim != 2 or ms.ndim != 3 or ms.shape[0] != 8:
        raise ValueError("expected PAN HW and MS 8hw (all eight bands)")
    if not all(pan.shape) or pan.shape != (ms.shape[1] * 4, ms.shape[2] * 4):
        raise ValueError("PAN/MS must have exact ratio-four corresponding pixel-grid extent")
    if max(float(pan.max()), float(ms.max())) > np.finfo(np.float32).max:
        raise ValueError("input cannot be represented in Float32")
    return np.ascontiguousarray(pan, dtype=np.float32), np.ascontiguousarray(ms, dtype=np.float32)


def _xml(height, width):
    bands = "".join(f'<VRTRasterBand dataType="Float32" band="{i}" subClass="VRTPansharpenedRasterBand"/>'
                    for i in range(1, 9))
    spectral = "".join(f'<SpectralBand dstBand="{i}"/>' for i in range(1, 9))
    return (f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}" subClass="VRTPansharpenedDataset">'
            '<GeoTransform>0,1,0,0,0,-1</GeoTransform>' + bands +
            '<PansharpeningOptions><Algorithm>WeightedBrovey</Algorithm>'
            '<AlgorithmOptions><Weights>0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125</Weights></AlgorithmOptions>'
            '<Resampling>Cubic</Resampling><NumThreads>1</NumThreads>'
            '<NoData>None</NoData><SpatialExtentAdjustment>None</SpatialExtentAdjustment>' +
            spectral + '</PansharpeningOptions></VRTDataset>')


def brovey8(pan_hw, ms_chw):
    """Return GDAL WeightedBrovey's unclipped Float32 8HW prediction.

No file is read/written. Source datasets/bands stay alive until the virtual
prediction is completely materialized. The synthetic affines encode only the
already verified matching grids; they do not perform scene registration.
"""
    pan, ms = _inputs(pan_hw, ms_chw)
    gdal = _gdal()
    height, width = pan.shape
    driver = gdal.GetDriverByName("MEM")
    pan_ds = driver.Create("", width, height, 1, gdal.GDT_Float32)
    ms_ds = driver.Create("", width // 4, height // 4, 8, gdal.GDT_Float32)
    if pan_ds is None or ms_ds is None:
        raise RuntimeError("GDAL MEM dataset creation failed")
    vrt = None
    try:
        pan_ds.SetGeoTransform(CONFIG["pan_geotransform"])
        ms_ds.SetGeoTransform(CONFIG["ms_geotransform"])
        pan_band = pan_ds.GetRasterBand(1)
        if pan_band.WriteArray(pan) != gdal.CE_None:
            raise RuntimeError("GDAL PAN memory write failed")
        spectral = [ms_ds.GetRasterBand(i) for i in range(1, 9)]
        for index, band in enumerate(spectral):
            if band.WriteArray(ms[index]) != gdal.CE_None:
                raise RuntimeError("GDAL spectral memory write failed")
        for band in [pan_band, *spectral]:
            if band.GetNoDataValue() is not None or band.GetMetadataItem("NBITS", "IMAGE_STRUCTURE") is not None:
                raise RuntimeError("Unexpected NoData/NBITS on new Float32 MEM input")
        vrt = gdal.CreatePansharpenedVRT(_xml(height, width), pan_band, spectral)
        if vrt is None:
            raise RuntimeError("GDAL CreatePansharpenedVRT failed: " + gdal.GetLastErrorMsg())
        if (vrt.RasterCount, vrt.RasterYSize, vrt.RasterXSize) != (8, height, width):
            raise RuntimeError("GDAL changed output bands/extent")
        if tuple(vrt.GetGeoTransform()) != tuple(CONFIG["pan_geotransform"]):
            raise RuntimeError("GDAL changed PAN pixel-grid geometry")
        if any(vrt.GetRasterBand(i).DataType != gdal.GDT_Float32 or
               vrt.GetRasterBand(i).GetNoDataValue() is not None for i in range(1, 9)):
            raise RuntimeError("Unexpected output dtype/NoData")
        result = vrt.ReadAsArray()
        if result is None or result.shape != (8, height, width) or result.dtype != np.float32:
            raise RuntimeError("GDAL returned invalid Float32 8-band result")
        if not np.isfinite(result).all():
            raise FloatingPointError("GDAL Brovey produced nonfinite values")
        return np.ascontiguousarray(result)
    finally:
        # Never close the source MEM datasets before the VRT consumer.
        vrt = None
        ms_ds = None
        pan_ds = None
