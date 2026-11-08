"""Shared bounds for the visible print diameter relative to the detected iris."""
import math

MIN_LENS_SCALE=.90
MAX_LENS_SCALE=1.40
ENLARGED_LENS_SCALE=1.30

def clamp_lens_scale(value):
    value=float(value)
    if not math.isfinite(value):return 1.
    return max(MIN_LENS_SCALE,min(MAX_LENS_SCALE,value))
