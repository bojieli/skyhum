"""verify and find on the round-5b locator.

Each scoring is one locator run over the area's bounding box and a clock window (skyhum.match.locator). The
area's score is the best hypothesis that lies inside the area (0 when none). Decoys are the same run at clocks
shifted by tens of minutes or whole days (skyhum.score.decoys); a decoy is skipped, never scored as zero, when the
ADS-B file does not cover it. The verdict rule is the same as for the reference matcher.

Not calibrated: in round 5b's decoy check (clock known to 30 s) about 2 in 3 confident answers were within 2 km,
on a 7-cluster test, and no margin threshold gave usable precision. THETA_STRICT below is a fixed margin in
log-likelihood units, not a calibrated value.
"""
import os
import tempfile

import numpy as np

from skyhum import __version__
from skyhum.errors import MissingData, NotImplementedInBuild
from skyhum.match import locator as LOC
from skyhum.match.reference import hex_hash
from skyhum.score import decoys as D
from skyhum.score.verify import LIMITS, MAX_TOL_S, MIN_AUDIO_S

THETA = {"default": 0.0, "strict": 3.0}
DEFAULT_DECOYS = 20
FAST_ARGS = ["--llr-calib-n", "3000", "--calib-n", "10000", "--top", "2", "--top2", "1", "--refine-radius", "400",
             "--refine2-radius", "150", "--refine2-spacing", "50"]
SINGLE_EVENT_ARGS = ["--min-voters", "1", "--nms-km", "1"]
CONFIDENCE = "not calibrated: about 2 in 3 confident answers within 2 km on a 7-cluster test"
NOT_VALIDATED = ("confidence " + CONFIDENCE + " (clock known to 30 s: 34 of 50); no margin threshold gives "
                 "usable precision; blind and wide runs have no decoy check")
R5B_LIMITS = LIMITS + [
    "Matcher r5b: 43 positive hours on 7 site clusters; confident answers within 2 km about 2 in 3 (not calibrated). "
    "Run here without detector hints, which the test used.",
]


class Session:
    """One recording prepared for the locator: spectrogram file and duration."""

    def __init__(self, x, fs, tmpdir):
        self.path = os.path.join(tmpdir, f"spec{id(self)}.npz")
        LOC.write_spec(x, self.path, fs)
        self.dur = x.size / fs

    @classmethod
    def from_features(cls, path, tmpdir):
        """A session from a released tonal feature file (skyhum.bench.features) instead of audio."""
        from skyhum.bench import features as F
        item = F.load(path)
        if F.SETS[item["header"]["set"]]["kind"] != "tonal":
            raise ValueError(f"{path}: not a tonal feature file")
        self = cls.__new__(cls)
        self.path = F.locator_spec(item, os.path.join(tmpdir, f"spec{id(self)}.npz"))
        self.dur = float(item["header"]["duration_s"])
        return self


def _passes(h):
    out = []
    for p in h.get("passes", []):
        if p.get("vote", 0) <= 0 or p.get("dropped"):
            continue
        out.append(dict(t_cpa_utc=p.get("t_cpa_utc"), offset_s=p.get("t_cpa_rec"), aircraft=hex_hash(p.get("hex")),
                        cls=p.get("cls"), d_cpa_km=round(p.get("d_cpa_m", 0) / 1000, 2), z=p.get("z"),
                        f0_hz=p.get("f0"), vote=p.get("vote")))
    return out


def area_score(sess, tracks, area, lo, hi, *, extra=(), ground_m=0.0, dem_dir=None):
    """(score, best hypothesis or None, run summary)."""
    out = LOC.run(sess.path, tracks.path, area.bbox, lo, hi, ground_m=ground_m, dem_dir=dem_dir, extra=extra)
    hyps = [h for h in out.get("hypotheses", []) if bool(area.contains(h["lat"], h["lon"]))]
    best = max(hyps, key=lambda h: h["score"]) if hyps else None
    return (max(0.0, float(best["score"])) if best else 0.0), best, dict(n_passes=int(out.get("n_passes", 0)),
                                                                         n_sites=int(out.get("n_sites", 0)))


def _check(tracks, spans):
    for lo, hi, dur in spans:
        if hi - lo > 2 * MAX_TOL_S:
            raise NotImplementedInBuild("clock windows wider than +-30 min", "use --start-tol 30m or less")
        if not tracks.covers(lo - 60, hi + dur + 60):
            raise MissingData("the ADS-B file does not cover the recording and its clock window")


def verify(x, fs, tracks, claim, lo, hi, *, versus=None, operating_point="default", n_decoys=DEFAULT_DECOYS, seed=0,
           fast=False, ground_m=0.0, dem_dir=None, progress=None):
    extra = FAST_ARGS if fast else []
    kw = dict(extra=extra, ground_m=ground_m, dem_dir=dem_dir)
    with tempfile.TemporaryDirectory(prefix="skyhum_v_") as td:
        sess = Session(x, fs, td)
        _check(tracks, [(lo, hi, sess.dur)])
        base = dict(version=__version__, matcher=LOC.NAME, benchmarked="partial", fast=bool(fast),
                    operating_point=operating_point, claim=claim.to_json(), seed=seed, limits=R5B_LIMITS, llr=None,
                    note=NOT_VALIDATED, theta=THETA[operating_point], tested_clock_window=dict(lo=lo, hi=hi))
        if sess.dur < MIN_AUDIO_S:
            return dict(base, verdict="INCONCLUSIVE", reason=f"audio too short ({sess.dur:.0f} s)", passes=[],
                        decoys_used=0, operating_points={})
        score, best, info = area_score(sess, tracks, claim, lo, hi, **kw)
        if progress:
            progress(1, 1 + n_decoys)
        rival = None
        if versus is not None:
            rival = area_score(sess, tracks, versus, lo, hi, **kw)[0]
        offsets = D.draw_offsets(n_decoys, [(lo, hi, sess.dur)], tracks, seed=seed)
        dec = []
        if score > 0:                      # with no claim score the margin cannot be positive: skip the decoy runs
            for k, off in enumerate(offsets):
                dec.append(area_score(sess, tracks, claim, lo + off, hi + off, **kw)[0])
                if progress:
                    progress(k + 2, 1 + offsets.size)
        dec = np.array(dec)
    level = max([0.0] + dec.tolist() + ([rival] if rival is not None else []))
    margin = score - level
    out = dict(base, claim_score=round(score, 3), decoy_level=round(level, 3), margin=round(margin, 3),
               decoys_used=int(dec.size), decoys_available=int(offsets.size),
               decoy_rank=f"{1 + int((dec >= score).sum())} of {dec.size + 1}" if score > 0 else "n/a (no claim score)",
               confidence=round(float((dec < score).mean()), 3) if dec.size and score > 0 else 0.0,
               versus=None if versus is None else dict(area=versus.to_json(), score=round(rival, 3)),
               passes_in_range=info["n_passes"], passes=_passes(best) if best else [],
               best=None if not best else dict(start_iso=best.get("start_iso"), n_voting=best.get("n_voting")),
               operating_points={op: bool(score > 0 and margin > th and dec.size >= D.MIN_DECOYS)
                                 for op, th in THETA.items()}, coarse_grid=False)
    if info["n_passes"] == 0:
        out.update(verdict="INCONCLUSIVE", reason="no ADS-B aircraft passes within 9 km of the claim while the "
                                                  "recording runs")
    elif score > 0 and offsets.size < D.MIN_DECOYS:
        out.update(verdict="INCONCLUSIVE", reason=f"only {offsets.size} decoy clocks fit inside the ADS-B file")
    elif out["operating_points"][operating_point]:
        out.update(verdict="SUPPORTED", reason=f"the claim beats every decoy ({dec.size} decoy clocks)")
    else:
        out.update(verdict="UNCORROBORATED", reason="no hypothesis inside the claim" if score <= 0 else
                   "decoys fit as well or better")
    return out


def find(sessions, tracks, area, *, top=5, n_decoys=10, seed=0, fast=False, single_event=False, ground_m=0.0,
         dem_dir=None, progress=None):
    """sessions: list of (x, fs, lo, hi). Multi-session: per-cell scores are not available from the locator, so
    sessions are scored separately and only hypotheses that every session places within 3 km of each other count."""
    from skyhum.io.geo import haversine_km
    from skyhum.score.find import tile
    extra = (FAST_ARGS if fast else []) + (SINGLE_EVENT_ARGS if single_event else [])
    kw = dict(extra=extra, ground_m=ground_m, dem_dir=dem_dir)
    with tempfile.TemporaryDirectory(prefix="skyhum_f_") as td:
        S = [(Session(x, fs, td), lo, hi) for x, fs, lo, hi in sessions]
        _check(tracks, [(lo, hi, s.dur) for s, lo, hi in S])
        offsets = D.draw_offsets(n_decoys, [(lo, hi, s.dur) for s, lo, hi in S], tracks, seed=seed)
        runs = []
        n_pass = 0
        for s, lo, hi in S:
            out = LOC.run(s.path, tracks.path, area.bbox, lo, hi, **kw)
            n_pass = max(n_pass, int(out.get("n_passes", 0)))
            runs.append([h for h in out.get("hypotheses", []) if bool(area.contains(h["lat"], h["lon"]))])
        cands = []
        for h in runs[0]:
            tot, ok = max(0.0, h["score"]), True
            for other in runs[1:]:
                near = [g for g in other if haversine_km(h["lat"], h["lon"], g["lat"], g["lon"]) <= 3.0]
                if not near:
                    ok = False
                    break
                tot += max(0.0, max(g["score"] for g in near))
            if ok and tot > 0:
                cands.append((tot, h))
        cands.sort(key=lambda c: -c[0])
        best = cands[0][0] if cands else 0.0
        dec = []
        if best > 0:
            for k, off in enumerate(offsets):
                tot = 0.0
                for s, lo, hi in S:
                    tot += area_score(s, tracks, area, lo + off, hi + off, **kw)[0]
                dec.append(tot)
                if progress:
                    progress(k + 1, offsets.size)
        dec = np.array(dec)
    level = max([0.0] + dec.tolist())
    margin = best - level
    enough = dec.size >= D.MIN_DECOYS
    default = bool(best > 0 and margin > THETA["default"] and enough)
    strict = bool(best > 0 and margin > THETA["strict"] and enough)
    picked = []
    for tot, h in cands:
        if len(picked) >= top:
            break
        if all(haversine_km(h["lat"], h["lon"], g["lat"], g["lon"]) > 3.0 for _, g in picked):
            picked.append((tot, h))
    out = dict(version=__version__, matcher=LOC.NAME, benchmarked="partial", fast=bool(fast),
               single_event=bool(single_event), area=area.to_json(), sessions=len(S), decoys_used=int(dec.size),
               best_score=round(best, 3), decoy_level=round(level, 3), margin=round(margin, 3),
               confidence=round(float((dec < best).mean()), 3) if dec.size and best > 0 else 0.0,
               beats_all_decoys=default, strict=strict, theta_strict=THETA["strict"], passes_in_range=n_pass,
               note=NOT_VALIDATED, limits=R5B_LIMITS, seed=seed,
               ranked=[dict(rank=k + 1, tile_bbox=tile(h["lat"], h["lon"]), score=round(t, 3))
                       for k, (t, h) in enumerate(picked)] if default else [],
               point_estimate=None, cells=None, cell_spacing_km=0.5, coarse_grid=False)
    if strict:
        h = picked[0][1]
        pe = dict(lat=round(float(h["lat"]), 4), lon=round(float(h["lon"]), 4))
        if single_event:
            m = LOC.mirror(h)
            pe["mirror"] = None if m is None else dict(lat=round(m[0], 4), lon=round(m[1], 4))
        out["point_estimate"] = pe
    if n_pass == 0 or (best > 0 and not enough):
        out["result"] = "INCONCLUSIVE"
    elif strict:
        out["result"] = "POINT_ESTIMATE"
    else:
        out["result"] = "NO_POINT_ESTIMATE"
    return out
