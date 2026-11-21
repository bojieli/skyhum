"""Reference CPU matcher (numpy only).

For a site and a flight, the exact Doppler curve log(f/f0) = -log(1 + rdot/c) is computed from the track at
reception time, sampled every 2 s over +-60 s of the closest point of approach (CPA) and weighted (d_cpa/r)^2.
The template is slid over every column (CPA time) and every f0 row of the whitened spectrogram; its score is the
weighted mean level on the curve minus the mean of the same curve 4 rows above and below (ridge contrast). The
best row per column gives a profile, which is turned into a robust z-score against all columns of the recording.

A pass votes sqrt(clip(z - Z0, 0, VCAP)) at the column its CPA falls on for a given clock. A site's score for a
clock is the sum of votes when at least two separate sounds vote (passes with CPAs < 8 s apart count once).

This follows the structure of the benchmarked GPU scorer (locate/gpu_locate.py in the research tree) but is a
simplified re-implementation: no candidate-spot pruning, no reversed-template null, no silence penalty, no DEM
(flat ground at --ground-alt), no refinement stage. Results from it are labelled `matcher: reference-cpu` and the
benchmark numbers in the docs do not apply to it.
"""
import hashlib

import numpy as np

from skyhum.io.geo import geodetic_to_ecef

C_SOUND = 343.0
HALF_W, STEP = 60, 2
DMAX_M = 9000.0
CONTRAST_ROWS = 4
F0_LO, F0_HI = 35.0, 450.0
Z0, VCAP = 5.5, 9.0     # Z0: on noise-only synthetic audio the per-column z stays below ~5 (tests)
DIL = 2
EXCL_S = 8.0
MIN_VOTERS = 2
NAME = "reference-cpu"


def hex_hash(h):
    return hashlib.sha256(str(h).encode()).hexdigest()[:10]


class Pass:
    __slots__ = ("t_cpa", "d_cpa", "hex", "cls", "dr", "dc", "w", "z", "vote")

    def __init__(self, t_cpa, d_cpa, hexid, cls, dr, dc, w):
        self.t_cpa, self.d_cpa, self.hex, self.cls = t_cpa, d_cpa, hexid, cls
        self.dr, self.dc, self.w = dr, dc, w
        self.z = self.vote = None


def site_passes(flights, site_ecef, log_step, dmax_m=DMAX_M):
    """Every local minimum of slant range <= dmax for each flight -> Pass with its template."""
    out = []
    offs = np.arange(-HALF_W, HALF_W + 1, STEP, dtype=float)
    for fl in flights:
        D = fl["ecef"] - site_ecef
        r = np.linalg.norm(D, axis=1)
        if r.min() > dmax_m or r.size < 5:
            continue
        rdot = np.einsum("ij,ij->i", D, fl["vel"]) / np.maximum(r, 1.0)
        t_rx = fl["t"] + r / C_SOUND
        lr = -np.log1p(rdot / C_SOUND)
        loc = np.flatnonzero((r[1:-1] <= r[:-2]) & (r[1:-1] < r[2:]) & (r[1:-1] <= dmax_m)) + 1
        for k in loc:
            t_cpa = float(t_rx[k])
            tq = t_cpa + offs
            ok = (tq >= t_rx[0]) & (tq <= t_rx[-1])
            if ok.sum() < 8:
                continue
            lq = np.interp(tq[ok], t_rx, lr)
            rq = np.interp(tq[ok], t_rx, r)
            lr0 = np.interp(t_cpa, t_rx, lr)
            dr = np.round((lq - lr0) / log_step).astype(int)       # rows relative to the f(CPA) row
            w = (r[k] / np.maximum(rq, 1.0)) ** 2
            out.append(Pass(t_cpa, float(r[k]), fl["hex"], fl["cls"], dr, offs[ok].astype(int), w / w.sum()))
    return out


class Scorer:
    def __init__(self, spec):
        self.spec = spec
        W = spec.W.astype(np.float32)
        self.R, self.T = W.shape
        self.pr = 120                                    # row padding: |dr| + contrast rows stay inside
        self.pc = HALF_W + 1
        self.Wp = np.zeros((self.R + 2 * self.pr, self.T + 2 * self.pc), np.float32)
        self.Wp[self.pr:self.pr + self.R, self.pc:self.pc + self.T] = W
        lo = max(0, spec.row_of(F0_LO))
        hi = min(self.R, spec.row_of(F0_HI) + 1)
        self.rows = (lo, hi)

    def profile(self, p):
        """z-score over columns of the template's best-row ridge contrast."""
        lo, hi = self.rows
        a = lo - CONTRAST_ROWS
        n = hi - lo + 2 * CONTRAST_ROWS
        S = np.zeros((n, self.T), np.float32)
        for dr, dc, w in zip(p.dr, p.dc, p.w):
            r0 = self.pr + a + int(dr)
            c0 = self.pc + int(dc)
            if r0 < 0 or r0 + n > self.Wp.shape[0]:
                continue
            S += np.float32(w) * self.Wp[r0:r0 + n, c0:c0 + self.T]
        C = S[CONTRAST_ROWS:-CONTRAST_ROWS] - 0.5 * (S[:-2 * CONTRAST_ROWS] + S[2 * CONTRAST_ROWS:])
        P = C.max(axis=0)
        med = np.median(P)
        mad = 1.4826 * np.median(np.abs(P - med)) + 1e-6
        return (P - med) / mad

    def votes(self, p):
        if p.vote is None:
            z = self.profile(p)
            zd = z.copy()
            for d in range(1, DIL + 1):
                zd[d:] = np.maximum(zd[d:], z[:-d])
                zd[:-d] = np.maximum(zd[:-d], z[d:])
            p.z = zd
            p.vote = np.sqrt(np.clip(zd - Z0, 0, VCAP)).astype(np.float32)
        return p.vote


def score_clocks(scorer, passes, S):
    """Score of one site for each recording start S (epoch seconds, array (m,)). Returns (score (m,), n_in (m,))."""
    S = np.asarray(S, float)
    m = S.size
    if not passes:
        return np.zeros(m), np.zeros(m, int)
    T = scorer.T
    tc = np.array([p.t_cpa for p in passes])
    order = np.argsort(tc)
    tc = tc[order]
    grp = np.r_[0, np.cumsum(np.diff(tc) >= EXCL_S)]              # same-sound groups
    ng = int(grp[-1]) + 1
    G = np.zeros((ng, m), np.float32)
    n_in = np.zeros(m, int)
    for j, i in enumerate(order):
        col = np.round(tc[j] - S).astype(int)
        inside = (col >= 0) & (col < T)
        n_in += inside
        if not inside.any():
            continue
        v = np.zeros(m, np.float32)
        v[inside] = scorer.votes(passes[i])[col[inside]]
        G[grp[j]] = np.maximum(G[grp[j]], v)
    nv = (G > 0).sum(axis=0)
    return np.where(nv >= MIN_VOTERS, G.sum(axis=0), 0.0), n_in


def score_cells(spec, tracks, lats, lons, windows, ground_m=0.0, progress=None):
    """windows: (nw, ns) recording starts; row 0 is the real clock window, the rest are decoys.

    Returns dict(scores (ncell, nw), best_S (ncell, nw), in_range (ncell,) passes with CPA inside the recording for
    the real window, details for the best real cell)."""
    windows = np.atleast_2d(np.asarray(windows, float))
    T = spec.n_cols
    t_lo = windows.min() - HALF_W - 30
    t_hi = windows.max() + T + HALF_W + 30
    flights = tracks.resampled(t_lo - 120, t_hi + 120)
    scorer = Scorer(spec)
    sites = geodetic_to_ecef(lats, lons, np.full(np.size(lats), ground_m + 1.5))
    nc, nw = sites.shape[0], windows.shape[0]
    scores = np.zeros((nc, nw))
    best_S = np.zeros((nc, nw))
    in_range = np.zeros(nc, int)
    all_passes = []
    flat = windows.ravel()
    for ci in range(nc):
        ps = [p for p in site_passes(flights, sites[ci], spec.log_step) if t_lo <= p.t_cpa <= t_hi]
        sc, n_in = score_clocks(scorer, ps, flat)
        sc = sc.reshape(nw, -1)
        n_in = n_in.reshape(nw, -1)
        k = sc.argmax(axis=1)
        scores[ci] = sc[np.arange(nw), k]
        best_S[ci] = windows[np.arange(nw), k]
        in_range[ci] = int(n_in[0].max())
        all_passes.append(ps)
        if progress:
            progress(ci + 1, nc)
    bi = int(np.argmax(scores[:, 0])) if nc else 0
    det = pass_details(scorer, all_passes[bi], best_S[bi, 0]) if nc else []
    return dict(scores=scores, best_S=best_S, in_range=in_range, best_cell=bi, passes=det, scorer=scorer)


def pass_details(scorer, passes, S):
    out = []
    for p in passes:
        col = int(round(p.t_cpa - S))
        if 0 <= col < scorer.T:
            v = float(scorer.votes(p)[col])
            out.append(dict(t_cpa=p.t_cpa, offset_s=col, aircraft=hex_hash(p.hex), cls=p.cls,
                            d_cpa_km=round(p.d_cpa / 1000, 2), z=round(float(p.z[col]), 2), vote=round(v, 3)))
    out.sort(key=lambda d: d["t_cpa"])
    return out
