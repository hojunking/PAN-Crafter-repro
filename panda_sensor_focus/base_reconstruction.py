"""Shared Student constructor and source-preserving Teacher dispatch."""
from panda_s1loss9.base_reconstruction import (BandReconstructionSpec,BandReconstructor,
    build_reconstructor,base_from_batch,upsampler_manifest,reconstruct_with_base as _qb)
from panda_sensor5.base_reconstruction import (BandReconstructionSpec as GF2TeacherSpec,
    BandReconstructor as GF2Teacher, reconstruct_with_base as _gf2)
TEACHER_TYPES=(BandReconstructor,GF2Teacher)

def teacher_spec(sensor):
    cls=BandReconstructionSpec if sensor=='QB' else GF2TeacherSpec
    return cls(4,112,(1,2,3),'P0','BICUBIC_MS',1)

def reconstruct_with_base(reconstructor,pan,base,ms):
    if type(reconstructor) is GF2Teacher:
        return _gf2(reconstructor,pan,base,ms)
    return _qb(reconstructor,pan,base,ms)
