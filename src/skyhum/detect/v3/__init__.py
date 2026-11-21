"""Detector v3: a learned fly-over detector that finds the closest-approach (CPA) time and frequency of aircraft hum.

Inference only, ported from the research code (front end and CNN of detector v0/v1, v3 checkpoint loaded as in
v2det, sliding detection with multi-peak NMS). Audio is resampled to 1 kHz and cut into 240 s windows every 30 s;
each window gives a heatmap over (log frequency 30-400 Hz, time) plus a speed and a slant-distance head. A detection
is a local maximum of the heatmap: t (CPA time, s from the first sample), f (Hz at CPA), p_raw (sigmoid of the
logit), v (m/s), d (m) and p = p_raw * gate(v, d), which suppresses slow or close sources (cars, boats, mowers).

The round-5b locator used these detections as hints (`--hints`, `{t, f, p}`, `--hint-pmin 0.8`). Needs PyTorch;
runs on the CPU.
"""
from skyhum.detect.v3.detect import (  # noqa: F401
                                     HINT_PMIN,
                                     NAME,
                                     Detector,
                                     available,
                                     load_1k,
                                     plot,
                                     shared,
                                     sliding_detect,
                                     to_1k,
)
