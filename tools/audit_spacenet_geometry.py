"""Bounded read-only public SpaceNet Paris geometry audit (never training).

Only TIFF metadata and small raw pixel samples are read via HTTP ranges.
The paired full downloads, when finished, remain the authoritative local assets.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import requests
import tifffile


BASE = 'https://spacenet-dataset.s3.amazonaws.com/AOIs/AOI_3_Paris/'
STEM = '16FEB29111913-{}-055649178040_01_P001'
SIZES = {'MS': 268438248, 'PAN': 536875617}


class BoundedRangeFile(io.RawIOBase):
    def __init__(self, url, size, budget=2_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.pos, self.fetched = 0, 0
        self.cache = {}
        self.etag = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError('negative seek')
        self.pos = pos
        return pos

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        if count <= 0:
            return b''
        start, end = self.pos, self.pos + count
        for (lo, hi), data in self.cache.items():
            if lo <= start and end <= hi:
                self.pos = end
                return data[start-lo:end-lo]
        lo, hi = start, min(self.size, max(end, start + 65536))
        if self.fetched + hi - lo > self.budget:
            raise ValueError('bounded metadata/sample budget exceeded')
        with requests.get(self.url, headers={'Range': f'bytes={lo}-{hi-1}'},
                          timeout=(15, 45), stream=True) as r:
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {lo}-{hi-1}/{self.size}':
                raise ValueError('server did not honor bounded range')
            etag = r.headers.get('ETag')
            if self.etag is not None and self.etag != etag:
                raise ValueError('object changed during audit')
            self.etag = etag
            data = r.raw.read(hi-lo+1)
            if len(data) != hi-lo:
                raise ValueError('range size mismatch')
        self.cache[(lo, hi)] = data
        self.fetched += len(data)
        self.pos = end
        return data[start-lo:end-lo]


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf8', errors='replace')
    return value


def leaves(element):
    return {n.tag: n.text.strip() if n.text else '' for n in element if not len(n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-root', type=Path, help='Downloaded AOIs/AOI_3_Paris directory; no network when supplied')
    args = parser.parse_args()
    result = {'purpose': 'READ_ONLY_GEOMETRY_AUDIT_NOT_ADMITTED_FOR_TRAINING', 'images': {}}
    for group, code in [('MS', 'M2AS'), ('PAN', 'P2AS')]:
        stem = STEM.format(code)
        xml_url = BASE + 'metadata/' + stem + '.XML'
        if args.local_root:
            xml_bytes = (args.local_root / 'metadata' / (stem + '.XML')).read_bytes()
        else:
            response = requests.get(xml_url, timeout=(15, 45))
            response.raise_for_status()
            xml_bytes = response.content
        root = ET.fromstring(xml_bytes)
        tile_name = STEM.format(code + '_R08C3') + '.TIF'
        tile = next(e for e in root.findall('./TIL/TILE') if e.findtext('FILENAME') == tile_name)
        image = {'xml_url': xml_url, 'xml_sha256': hashlib.sha256(xml_bytes).hexdigest(),
                 'product': leaves(root.find('IMD')),
                 'image': leaves(root.find('IMD/IMAGE')),
                 'map_product': leaves(root.find('IMD/MAP_PROJECTED_PRODUCT')),
                 'tile_index': leaves(root.find('TIL')), 'selected_tile': leaves(tile),
                 'band_xml_order': [e.tag for e in root.find('IMD') if e.tag.startswith('BAND_')]}
        image_url = BASE + group + '/' + tile_name
        if args.local_root:
            image_path = args.local_root / group / tile_name
            if image_path.stat().st_size != SIZES[group]:
                raise ValueError('Local TIFF incomplete/wrong size')
            reader = image_path.open('rb')
            image['local_path'] = str(image_path)
        else:
            reader = BoundedRangeFile(image_url, SIZES[group])
        with tifffile.TiffFile(reader) as tf:
            page = tf.pages[0]
            image['tiff'] = {'url': image_url, 'shape': list(page.shape), 'dtype': str(page.dtype),
                             'byteorder': tf.byteorder, 'geotiff': serial(tf.geotiff_metadata),
                             'tags': {t.name: serial(t.value) for t in page.tags.values()
                                      if t.name not in ('StripOffsets', 'StripByteCounts',
                                                       'TileOffsets', 'TileByteCounts')}}
            image['pixel_samples'] = []
            if int(page.compression) == 1:
                offsets = page.dataoffsets
                for index in sorted(set((0, len(offsets)//2, len(offsets)-1))):
                    reader.seek(offsets[index])
                    data = reader.read(min(page.databytecounts[index], 4096))
                    a = np.frombuffer(data, dtype=np.dtype(tf.byteorder + 'u2'))
                    image['pixel_samples'].append({'strip_or_tile': index,
                                                  'values': int(a.size), 'min': int(a.min()),
                                                  'max': int(a.max()), 'zero_fraction': float((a == 0).mean())})
            else:
                image['pixel_sample_note'] = 'compressed; samples not decoded'
        image['range_bytes_fetched'] = getattr(reader, 'fetched', 0)
        image['tiff_etag'] = getattr(reader, 'etag', None)
        reader.close()
        if args.local_root:
            a = tifffile.memmap(image_path, mode='r')
            bands = a[None] if a.ndim == 2 else a
            image['full_local_pixel_stats'] = []
            for band in bands:
                stats = {'min': 65535, 'max': 0, 'zeros': 0, 'ones': 0, 'above_2047': 0, 'pixels': 0}
                for y in range(0, band.shape[0], 128):
                    strip = band[y:y+128]
                    stats['min'] = min(stats['min'], int(strip.min()))
                    stats['max'] = max(stats['max'], int(strip.max()))
                    stats['zeros'] += int((strip == 0).sum())
                    stats['ones'] += int((strip == 1).sum())
                    stats['above_2047'] += int((strip > 2047).sum())
                    stats['pixels'] += int(strip.size)
                image['full_local_pixel_stats'].append(stats)
            image['nodata_note'] = ('No explicit GDAL_NODATA TIFF tag; zero/one frequencies are observations, '
                                    'not proof of a valid-data mask. XML/TIFF data mask semantics need separate validation.')
        result['images'][group] = image
    ms = result['images']['MS']['tiff']['tags']
    pan = result['images']['PAN']['tiff']['tags']
    ms_scale, pan_scale = ms['ModelPixelScaleTag'], pan['ModelPixelScaleTag']
    ms_tie, pan_tie = ms['ModelTiepointTag'], pan['ModelTiepointTag']
    def bounds(tags, shape):
        scale, tie = tags['ModelPixelScaleTag'], tags['ModelTiepointTag']
        left, top = tie[3] - tie[0]*scale[0], tie[4] + tie[1]*scale[1]
        return [left, top-shape[-2]*scale[1], left+shape[-1]*scale[0], top]
    ms_bounds = bounds(ms, result['images']['MS']['tiff']['shape'])
    pan_bounds = bounds(pan, result['images']['PAN']['tiff']['shape'])
    result['pair'] = {
        'pixel_scale_ratio_x_y': [ms_scale[i] / pan_scale[i] for i in (0, 1)],
        'tiepoint_ms_minus_pan_in_pan_pixels_x_north_y': [
            (ms_tie[3+i] - pan_tie[3+i]) / pan_scale[i] for i in (0, 1)],
        'canonical_acquisition_id': result['images']['MS']['image'].get('CATID'),
        'ms_bounds_wgs84_left_bottom_right_top': ms_bounds,
        'pan_bounds_wgs84_left_bottom_right_top': pan_bounds,
        'max_boundary_difference_in_pan_pixels': max(
            abs(a-b)/pan_scale[i % 2] for i, (a, b) in enumerate(zip(ms_bounds, pan_bounds))),
        'grid_interpretation': ('Both RasterPixelIsArea and tiepoint=(0.5,0.5). The first MS center is '
                                '(1.5,1.5) PAN pixels down/right from the first PAN center: NORMAL 4x '
                                'pixel-center geometry, not image misregistration. Outer footprints coincide '
                                'within floating-point metadata tolerance. Residual feature misalignment '
                                'requires a separate image-based measurement.'),
        'warning': 'Tiles from the same acquisition are not independent source scenes; '
                   'same nominal 4x scale alone does not prove PAN/MS pixel-center alignment. '
                   'This corrected LV2A product is not unprocessed raw sensor telemetry.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'output': str(output), 'pair': result['pair'],
                      'range_bytes': {k: v['range_bytes_fetched'] for k, v in result['images'].items()}}, indent=2))


if __name__ == '__main__':
    main()
