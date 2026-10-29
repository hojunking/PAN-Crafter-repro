"""Unchanged registered Lanczos-4 implementation, with truthful consumer metadata."""
from copy import deepcopy
from panda_sensor_refine.warp import warp_pan,warp_support_mask,mask_manifest,boundary_manifest
from .contract import WARP_MANIFEST
def warp_manifest(): return deepcopy(WARP_MANIFEST)
