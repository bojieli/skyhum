"""find: rank places inside a supplied area.

Every cell of the area is scored at the stated clock window and at the same decoy clocks verify uses. Sessions
given together (--sessions) are assumed to come from one spot: their cell scores are summed. The top cells are
reported as coarse tiles (0.1 degree, about 10 km). A point estimate is printed only when the best cell beats
every decoy by more than the strict threshold.
"""
import numpy as np

from skyhum import __version__
from skyhum.errors import MissingData
from skyhum.io.geo import haversine_km
from skyhum.match import reference as M
from skyhum.score import decoys as D
from skyhum.score.verify import FINE_KM, LIMITS, THETA, cell_spacing

TILE_DEG = 0.1
NMS_KM = 3.0


def tile(lat, lon):
    a, b = np.floor(lat / TILE_DEG) * TILE_DEG, np.floor(lon / TILE_DEG) * TILE_DEG
    return [round(float(b), 1), round(float(a), 1), round(float(b + TILE_DEG), 1), round(float(a + TILE_DEG), 1)]


def run(sessions, tracks, area, *, top=5, n_decoys=200, seed=0, spacing_km=None, ground_m=0.0, progress=None):
    """sessions: list of (spec, win_lo, win_hi)."""
    sp = cell_spacing(area, spacing_km)
    lats, lons = area.grid(sp)
    joint = None
    in_range = 0
    for spec, lo, hi in sessions:
        if not tracks.covers(lo - 60, hi + spec.n_cols * spec.col_s + 60):
            raise MissingData("the ADS-B file does not cover every session")
    offsets = D.draw_offsets(n_decoys, [(lo, hi, spec.n_cols * spec.col_s) for spec, lo, hi in sessions], tracks,
                             seed=seed)
    for spec, lo, hi in sessions:
        res = M.score_cells(spec, tracks, lats, lons, D.windows(lo, hi, offsets), ground_m=ground_m,
                            progress=progress)
        joint = res["scores"] if joint is None else joint + res["scores"]
        in_range = max(in_range, int(res["in_range"].max()))
    real = joint[:, 0]
    dec = joint[:, 1:].max(axis=0) if offsets.size else np.zeros(0)
    order = np.argsort(-real)
    picked = []
    for i in order:
        if real[i] <= 0 or len(picked) >= top:
            break
        if all(haversine_km(lats[i], lons[i], lats[j], lons[j]) > NMS_KM for j in picked):
            picked.append(int(i))
    best = float(real[order[0]]) if real.size else 0.0
    level = max([0.0] + dec.tolist())
    margin = best - level
    strict = bool(best > 0 and margin > THETA["strict"])
    default = bool(best > 0 and margin > THETA["default"])
    out = dict(version=__version__, matcher=M.NAME, benchmarked=False, area=area.to_json(), sessions=len(sessions),
               cells=int(lats.size), cell_spacing_km=sp, coarse_grid=bool(sp > FINE_KM + 1e-9),
               decoys_used=int(offsets.size), best_score=round(best, 3),
               decoy_level=round(level, 3), margin=round(margin, 3),
               confidence=round(float((dec < best).mean()) if dec.size and best > 0 else 0.0, 3),
               beats_all_decoys=default, strict=strict, passes_in_range=in_range, limits=LIMITS, seed=seed,
               ranked=[dict(rank=k + 1, tile_bbox=tile(lats[i], lons[i]), score=round(float(real[i]), 3))
                       for k, i in enumerate(picked)],
               point_estimate=None)
    if not default:
        out["ranked"] = []                  # nothing beats the decoys: a ranking would be chance
    if strict:
        i = picked[0]
        out["point_estimate"] = dict(lat=round(float(lats[i]), 4), lon=round(float(lons[i]), 4),
                                     cell_spacing_km=sp)
    if in_range == 0 or offsets.size < D.MIN_DECOYS:
        out["result"] = "INCONCLUSIVE"
    elif strict:
        out["result"] = "POINT_ESTIMATE"
    else:
        out["result"] = "NO_POINT_ESTIMATE"
    return out
