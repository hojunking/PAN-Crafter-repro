"""Explicit metric gates; ranking never mixes corrected-PAN and official HQNR."""
import math


def choose(rows, *, ergas_max, psnr_min=None, order='HQNR'):
    if order not in ('HQNR','PSNR_HQNR'):
        raise ValueError('Explicit HQNR or PSNR_HQNR order required')
    if not math.isfinite(ergas_max) or ergas_max<0:
        raise ValueError('Finite nonnegative ERGAS ceiling required')
    if psnr_min is not None and not math.isfinite(psnr_min):
        raise ValueError('Finite PSNR floor required')
    qualified=[]
    for row in rows:
        e,p,h=(float(row['rr']['ERGAS']),float(row['rr']['PSNR']),float(row['fr']['HQNR']))
        if not all(math.isfinite(v) for v in (e,p,h)):
            raise ValueError('Nonfinite candidate metric; no silent omission')
        if e<=ergas_max and (psnr_min is None or p>=psnr_min):qualified.append(row)
    if not qualified:
        raise ValueError('NO_QUALIFIED_DONOR: explicit ERGAS/PSNR gates; no cross-server or weaker fallback')
    def key(row):
        e,p,h=row['rr']['ERGAS'],row['rr']['PSNR'],row['fr']['HQNR']
        return (-h,e,-p,row['step']) if order=='HQNR' else (-p,-h,e,row['step'])
    return min(qualified,key=key)
