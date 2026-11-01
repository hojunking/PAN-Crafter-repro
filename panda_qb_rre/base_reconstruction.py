"""Preserve the authentic original Stage1 Teacher class and its forward path."""
from panda_s1loss9.base_reconstruction import (BandReconstructionSpec, BandReconstructor,
    build_reconstructor, base_from_batch, reconstruct_with_base, upsampler_manifest)
TEACHER_TYPES=(BandReconstructor,)
def teacher_spec(sensor):
    if sensor!='QB':raise ValueError('QB required')
    return BandReconstructionSpec(4,112,(1,2,3),'P0','BICUBIC_MS',1)
