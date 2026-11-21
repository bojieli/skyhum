"""Run the round-5b locator (skyhum.match.r5b) on audio in memory.

The frozen round-5b configuration (eval/r5b/final_r5b_args.txt) is used unchanged except for paths. Detector hints
are passed as in the round-5b test: a JSON list of detector v3 detections ({t, f, p, ...}, skyhum.detect.v3) given
with --hints; the frozen --hint-pmin 0.8 keeps only the confident ones.
"""
import json
import math
import os
import tempfile
from importlib import resources

import numpy as np

from skyhum.io.audio import resample

NAME = "r5b"
FS_SPEC = 4000


MODEL_FILES = ("k3_full.json", "env_full.json")


def _model(name):
    d = os.environ.get("SKYHUM_R5B_MODELS")
    if d:
        return os.path.join(d, name)
    return str(resources.files("skyhum.models").joinpath("r5b", name))


def models_available():
    """The fitted round-5b tables (per-pass LLR, f0 prior, envelope), shipped in skyhum/models/r5b.
    SKYHUM_R5B_MODELS points at another directory holding the same files."""
    return all(os.path.isfile(_model(n)) for n in MODEL_FILES)


def frozen_args():
    return ["--llr", _model("k3_full.json"), "--f0-prior", "3", "--hex-tol", "0.06", "--top", "20", "--top2", "6",
            "--excl-s", "20", "--rescore", "1", "--same-dt", "20", "--same-dlogf", "9", "--hint-pmin", "0.8",
            "--unexpl-cost", "1", "--env-llr", _model("env_full.json"), "--env-w", "1", "--env-clip", "2",
            "--pass-feats", "1", "--pass-keep", "80", "--llr-null", "0"]


def write_spec(x16k, path, fs=16000):
    from skyhum._vendor.afg import audio as afg_audio
    y = resample(np.asarray(x16k, np.float32), fs, FS_SPEC)
    sp = afg_audio.compute_spectrogram(y.astype(np.float64), FS_SPEC, win_s=1.0, hop_s=0.2, fmax=620.0)
    np.savez(path, W=sp["W"].astype(np.float16), f=sp["f"], t=sp["t"], hop_s=sp["hop_s"], duration=sp["duration"])


def write_hints(dets, path):
    """Locator hints file: the detector's detections, all fields, as the research runs wrote them."""
    with open(path, "w") as f:
        json.dump([{k: float(v) for k, v in d.items()} for d in dets], f, indent=0)


def run(spec_path, tracks_path, bbox, s_min, s_max, *, spacing_m=500.0, ground_m=0.0, dem_dir=None, extra=(),
        workdir=None, hints_path=None):
    """bbox: (west, south, east, north). s_min, s_max: earliest and latest recording start (epoch s).
    hints_path: detector hints file (write_hints), or None to run without hints."""
    from skyhum.match import r5b
    w, s, e, n = bbox
    with tempfile.TemporaryDirectory(prefix="skyhum_r5b_", dir=workdir) as td:
        argv = ["--spec", spec_path, "--tracks", tracks_path, "--bbox", f"{s},{n},{w},{e}",
                "--spacing", str(spacing_m), "--ground", str(ground_m), "--start", str(s_min), "--end", str(s_max),
                "--plots", "0", "--out", os.path.join(td, "loc")] + frozen_args() + list(extra)
        if hints_path:
            argv += ["--hints", hints_path]
        if dem_dir:
            argv += ["--dem-dir", dem_dir]
        out = r5b.main(argv)
        if out is None:
            out = json.load(open(os.path.join(td, "loc.json")))
    out.pop("params", None)
    return out


def mirror(h):
    """Single-event answer: the hypothesis mirrored across the ground track of its strongest voter (round 5b)."""
    vs = [p for p in h.get("passes", []) if p.get("vote", 0) >= 1.0 and not p.get("dropped")
          and p.get("ac_lat") is not None]
    if not vs:
        return None
    p = max(vs, key=lambda q: q["vote"])
    la0, lo0 = p["ac_lat"], p["ac_lon"]
    kx = 111320.0 * math.cos(math.radians(la0))
    E, N = (h["lon"] - lo0) * kx, (h["lat"] - la0) * 111132.0
    u = (math.sin(math.radians(p["ac_hdg"])), math.cos(math.radians(p["ac_hdg"])))
    al = E * u[0] + N * u[1]
    pe, pn = E - al * u[0], N - al * u[1]
    me, mn = E - 2 * pe, N - 2 * pn
    return la0 + mn / 111132.0, lo0 + me / kx
