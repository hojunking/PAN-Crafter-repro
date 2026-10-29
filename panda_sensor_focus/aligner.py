"""Exact original source classes; do not relabel imported model identities."""
from panda_s1loss9.aligner import BandPANAligner
from panda_sensor5.aligner import BandPANAligner as GF2PANAligner
ALIGNER_TYPES=(BandPANAligner,GF2PANAligner)
def aligner_class(sensor):
    return BandPANAligner if sensor=='QB' else GF2PANAligner
