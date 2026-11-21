"""selftest: synthetic propeller passes heard at a synthetic site, with synthetic ADS-B tracks.

Checks, all on CPU:
  1. the detector finds Doppler-like passes in the synthetic audio
  2. reference matcher: verify SUPPORTS the true claim (a 1.5 km circle around the site, start known to +-30 s),
     and rejects the claim moved 20 km and the true place at a clock 25 min off (a few seconds)
  3. r5b locator, --fast settings, 5 decoys: the same three checks (minutes on a CPU; skipped with --quick or when
     PyTorch is not installed)
  4. the tone scrubber removes most tonal ridge cells
"""
import os
import tempfile
import time

import numpy as np

from skyhum import synth
from skyhum.detect import tonal
from skyhum.detect.spectrogram import logspec
from skyhum.io.adsb import Tracks, write_tracks
from skyhum.io.regions import displaced, parse_place
from skyhum.score import verify as V

FS = 16000


def build(dirpath, seed=7):
    """Write the synthetic tracks file and return (world, audio, tracks_path)."""
    w = synth.make_world(seed=seed)
    x = synth.render(w, fs=FS)
    p = os.path.join(dirpath, "tracks.npz")
    write_tracks(p, w["flights"], w["t0"], w["t1"], meta=dict(synthetic=True, seed=seed))
    return w, x, p


def true_claim(w):
    lat, lon = w["site"]
    return parse_place(f"{lat + 0.004:.4f},{lon - 0.003:.4f},r=1.5km")


def write_fixtures(dirpath, seed=7):
    """Demo files for the README: clip.wav (planted passes), quiet.wav (no aircraft), tracks.npz, area.geojson."""
    from skyhum.io.audio import save
    os.makedirs(dirpath, exist_ok=True)
    w, x, p = build(dirpath, seed)
    save(os.path.join(dirpath, "clip.wav"), x, FS)
    quiet = synth.render(dict(w, flights=[]), fs=FS)
    save(os.path.join(dirpath, "quiet.wav"), quiet, FS)
    sp = synth.speechlike(x.size, FS)
    save(os.path.join(dirpath, "call.wav"), np.clip(x + sp, -1, 1), FS)
    lat, lon = w["site"]
    import json
    d = 0.03
    with open(os.path.join(dirpath, "area.geojson"), "w") as f:
        json.dump(dict(type="Feature", properties=dict(name="synthetic demo area"),
                       geometry=dict(type="Polygon", coordinates=[[[lon - d, lat - d], [lon + d, lat - d],
                                                                   [lon + d, lat + d], [lon - d, lat + d],
                                                                   [lon - d, lat - d]]])), f)
    with open(os.path.join(dirpath, "README.txt"), "w") as f:
        f.write("Synthetic demo data written by `skyhum selftest --write-fixtures`. Nothing here is real.\n"
                f"Site: {lat:.4f},{lon:.4f}  start: 2026-09-08T14:03:00Z  seed: {seed}\n"
                "clip.wav: three planted propeller passes. call.wav: the same with a speech-like buzz.\n"
                "quiet.wav: noise only. tracks.npz: synthetic ADS-B tracks in the afg tracks.npz layout.\n")
    return dict(dir=os.path.abspath(dirpath), site=[lat, lon], start="2026-09-08T14:03:00Z", seed=seed)


def run(seed=7, quick_scrub=True, r5b=True):
    import importlib.util
    t0 = time.time()
    checks = []

    def add(name, ok, detail, skipped=False):
        checks.append(dict(check=name, ok=bool(ok), detail=detail, skipped=skipped))

    with tempfile.TemporaryDirectory(prefix="skyhum_selftest_") as td:
        w, x, p = build(td, seed)
        tracks = Tracks(p)
        spec = logspec(x, FS)
        det = tonal.detect(spec)
        n_dop = sum(d["kind"] == "doppler-like" for d in det)
        add("detect finds the planted passes", n_dop >= 2, f"{n_dop} Doppler-like pass(es) found, 3 planted")
        claim = true_claim(w)
        lo, hi = w["start"] - 30, w["start"] + 30
        cases = [("supports the true claim", claim, 0, True), ("rejects the claim moved 20 km", displaced(claim, 20.0),
                                                               0, False),
                 ("rejects a clock 25 min off", claim, 1500, False)]
        for name, area, shift, want in cases:
            r = V.run(spec, tracks, area, lo + shift, hi + shift, n_decoys=100, seed=seed)
            add(f"reference verify {name}", (r["verdict"] == "SUPPORTED") == want,
                f"{r['verdict']} (margin {r.get('margin')})")
        if not r5b:
            add("r5b locator", True, "skipped (--quick)", skipped=True)
        elif importlib.util.find_spec("torch") is None:
            add("r5b locator", True, "skipped: PyTorch not installed (pip install torch)", skipped=True)
        elif not __import__("skyhum.match.locator", fromlist=["x"]).models_available():
            add("r5b locator", True, "skipped: fitted r5b tables not installed (not in the public repository yet)",
                skipped=True)
        else:
            from skyhum.score import r5b_rule as R
            for name, area, shift, want in cases:
                r = R.verify(x, FS, tracks, area, lo + shift, hi + shift, n_decoys=5, seed=seed, fast=True)
                add(f"r5b verify {name}", (r["verdict"] == "SUPPORTED") == want,
                    f"{r['verdict']} (claim score {r.get('claim_score')}, decoy level {r.get('decoy_level')})")
        if quick_scrub:
            from skyhum.scrub import tones
            seg = x[: 150 * FS]
            y, _ = tones.scrub(seg)
            before, after = _ridge_cells(seg), _ridge_cells(y)
            add("tone scrubber removes most tonal ridge cells", after <= 0.5 * before,
                f"{before} -> {after} ridge cells in the first 150 s")
    ok = all(c["ok"] for c in checks)
    return dict(result="PASS" if ok else "FAIL", seconds=round(time.time() - t0, 1), checks=checks, seed=seed,
                note="synthetic data only; this checks the install, not real-world accuracy")


def _ridge_cells(x):
    """Spectrogram cells that stand out as tonal ridges (whitened level above the detector threshold)."""
    return int((logspec(x, FS).W > tonal.THR).sum())
