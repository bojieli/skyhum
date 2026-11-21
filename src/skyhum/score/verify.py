"""verify: is this audio consistent with a claimed place and clock?

Rule (from the research verify rule, applied to this build's matcher):
  claim score  = best cell score inside the claimed area over the stated clock window
  decoy level  = max(0, best score of the same area at every decoy clock, and at the --versus area)
  margin       = claim score - decoy level
  SUPPORTED    margin > THETA[operating_point]
  INCONCLUSIVE no pass of a known aircraft falls inside the recording at any cell of the claim, too few decoys
               can be built from the ADS-B coverage, or the audio is too short
  otherwise    UNCORROBORATED (not evidence that the claim is false)
"""
import numpy as np

from skyhum import __version__
from skyhum.errors import MissingData, NotImplementedInBuild
from skyhum.match import reference as M
from skyhum.score import decoys as D

# Default: beats every decoy (the benchmarked rule). Strict: margin above THETA_STRICT score units. In the research
# tree the strict threshold was fitted to <= 1 % false accepts for the GPU scorer; for the reference CPU matcher in
# this build it is a fixed, uncalibrated value.
THETA = {"default": 0.0, "strict": 2.0}
MIN_AUDIO_S = 60.0
MAX_TOL_S = 1800.0
MAX_CELLS = 2500

LIMITS = [
    "UNCORROBORATED is not evidence that the claim is false: the benchmarked default rule accepted only 18 of 110 "
    "true US webcam claims.",
    "A neighbouring place can pass too: listeners on both sides of a boundary hear much the same aircraft "
    "(2 of 110 wrong neighbours passed in the benchmark).",
    "Not tested: indoor attenuation, speech overlap, platform processing (calls, uploads), clock drift.",
    "Only aircraft that broadcast ADS-B and are in the track file can be matched.",
]


FINE_KM = 0.5


def cell_spacing(area, spacing_km=None):
    """0.5 km unless that gives more than MAX_CELLS cells. The reference matcher needs about 0.5 km: at 1 km it
    misses true matches on the synthetic tests, so coarser grids are flagged in the result."""
    if spacing_km:
        return float(spacing_km)
    s = FINE_KM
    while area.grid(s)[0].size > MAX_CELLS:
        s *= 1.25
    return round(s, 3)


def run(spec, tracks, claim, win_lo, win_hi, *, versus=None, operating_point="default", n_decoys=200, seed=0,
        spacing_km=None, ground_m=0.0, progress=None):
    dur = spec.n_cols * spec.col_s
    if win_hi - win_lo > 2 * MAX_TOL_S:
        raise NotImplementedInBuild("clock windows wider than +-30 min",
                                    "the CPU reference matcher in this build is limited to --start-tol <= 30m")
    if not tracks.covers(win_lo - 60, win_hi + dur + 60):
        raise MissingData(f"the ADS-B file does not cover the recording: tracks span {tracks.t0:.0f}..{tracks.t1:.0f}"
                          f", the recording needs {win_lo - 60:.0f}..{win_hi + dur + 60:.0f} (epoch s)")
    base = dict(version=__version__, matcher=M.NAME, benchmarked=False, operating_point=operating_point,
                claim=claim.to_json(), seed=seed, limits=LIMITS, llr=None)
    if dur < MIN_AUDIO_S:
        return dict(base, verdict="INCONCLUSIVE", reason=f"audio too short ({dur:.0f} s < {MIN_AUDIO_S:.0f} s)",
                    passes=[], decoys_used=0)
    offsets = D.draw_offsets(n_decoys, [(win_lo, win_hi, dur)], tracks, seed=seed)
    Wn = D.windows(win_lo, win_hi, offsets)
    sp = cell_spacing(claim, spacing_km)
    lats, lons = claim.grid(sp)
    res = M.score_cells(spec, tracks, lats, lons, Wn, ground_m=ground_m, progress=progress)
    claim_score = float(res["scores"][:, 0].max())
    dec = res["scores"][:, 1:].max(axis=0) if offsets.size else np.zeros(0)
    rival = None
    if versus is not None:
        vl, vo = versus.grid(cell_spacing(versus, spacing_km))
        rv = M.score_cells(spec, tracks, vl, vo, Wn[:1], ground_m=ground_m)
        rival = float(rv["scores"][:, 0].max())
    in_range = int(res["in_range"].max())
    level = max([0.0] + dec.tolist() + ([rival] if rival is not None else []))
    margin = claim_score - level
    rank = 1 + int((dec >= claim_score).sum()) if claim_score > 0 else 1 + dec.size
    out = dict(base, cells=int(lats.size), cell_spacing_km=sp, claim_score=round(claim_score, 3),
               decoy_level=round(level, 3), margin=round(margin, 3), decoys_used=int(offsets.size),
               decoy_rank=f"{rank} of {offsets.size + 1}",
               confidence=round(float((dec < claim_score).mean()) if dec.size and claim_score > 0 else 0.0, 3),
               versus=None if versus is None else dict(area=versus.to_json(), score=round(rival, 3)),
               passes_in_range=in_range, passes=res["passes"] if claim_score > 0 else [],
               tested_clock_window=dict(lo=win_lo, hi=win_hi), best_clock=float(res["best_S"][res["best_cell"], 0]),
               theta=THETA[operating_point], operating_points={},
               coarse_grid=bool(sp > FINE_KM + 1e-9))
    for op, th in THETA.items():
        out["operating_points"][op] = bool(claim_score > 0 and margin > th)
    if in_range == 0:
        out.update(verdict="INCONCLUSIVE", reason="no ADS-B aircraft passes within 9 km of the claim while the "
                                                  "recording runs")
    elif offsets.size < D.MIN_DECOYS:
        out.update(verdict="INCONCLUSIVE", reason=f"only {offsets.size} decoy clocks fit inside the ADS-B file "
                                                  f"(need {D.MIN_DECOYS}); supply tracks covering about +-1 h more")
    elif out["operating_points"][operating_point]:
        out.update(verdict="SUPPORTED", reason="the claim beats every decoy" +
                   ("" if operating_point == "default" else f" by more than {THETA['strict']}"))
    else:
        out.update(verdict="UNCORROBORATED",
                   reason="no pass fits the claim" if claim_score <= 0 else "decoys fit as well or better")
    return out
