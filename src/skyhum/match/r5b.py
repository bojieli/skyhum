"""r5b.py - the round-5b locator, vendored from locate/gpu_locate.py (branch locator-r5b, fdd7129).

Package changes (behaviour of the scoring is unchanged): imports afg from skyhum._vendor; the device comes from
SKYHUM_DEVICE (cpu never touches CUDA, and Triton is imported only on CUDA); main() takes argv and returns the
result; progress goes to stderr only when SKYHUM_VERBOSE is set; plots and the grid dump are skipped with
--plots 0. The CPU path is the torch fallback that was always in this file (no Triton kernels): slower, same maths.

Original header follows.

gpu_locate.py - direct Doppler-curve matching geolocation on the GPU.

README
======
Locate where a recording was made by matching the *exact* Doppler curves that historical ADS-B tracks predict at
every site of a grid against the whitened spectrogram, with an unknown (full day) or loosely known recording clock.

Upstream (AlanBacker/audio-flyover-geolocation) traces a curve, compresses it into (v, d, tCPA) with a
constant-velocity fit and filters ADS-B by those numbers. Speed changes, turns and climbs bias v and d, the silence
filter deletes true answers, and it is slow. Here nothing is traced or fitted: for every (site, flight pass) the full
emission-time curve r(t) = f/f0 is computed from the real trajectory and scored directly on the spectrogram.

Pipeline
  1. Log-frequency spectrogram (unknown f0 = vertical shift): 1 s columns, 0.4 % rows, max-pooled +-1 row.
  2. Site grid (DEM ground + 1.5 m) x flights (prop/rotor/unknown; --jets adds jets): every local minimum of slant
     range <= --dmax (9 km) is a pass. Template = the exact emission-time log-ratio curve at 1 s steps over +-60 s of
     CPA reception, points weighted (d_cpa/R)^2 (louder near CPA).
  3. Score of a template at (CPA column, f0 row) = weighted mean level on the curve minus the same curve shifted
     +-4 rows (ridge contrast: a loud event that the curve only crosses scores ~0, a steady tone too).
     --mode cand (default): a bank of 90 generic straight-pass curves finds the 400 most curve-like spots of the
     recording (and, for the null, 400 rising spots); every exact template is evaluated only within +-4 s x +-3 rows
     of those spots by a Triton gather kernel (~25 us/pass incl. null). z = robust z against 96 random spots.
     --mode full: FFT cross-correlation over every column x f0 (exhaustive, ~20x slower).
  4. Shared clock: recording start S = t_cpa_utc - tau. Vote of a pass = clip(z - z0, 0, vcap) ** 0.5 (concave:
     three explained passes beat one loud one), dilated +-2 s, summed per site over S (1 s bins, whole clock
     window). Passes at one site whose CPAs are < 8 s apart explain the same sound -> max, not sum. Site score =
     best S with >= 2 voting passes (the upstream "locked two-event" constraint, generalised to n passes).
  5. Null: everything repeated with time-reversed (rising) templates at rising spots. z0 = 99th percentile of the
     best reversed z of 100k random passes; null level = best reversed site score anywhere in the search.
  6. Refine: top 60 coarse hypotheses (3 km NMS) -> 100 m grid +-1.5 km, clock +-15 s; top 12 -> 25 m grid
     +-400 m, clock +-6 s. Final ranking by refined score.

  7. Options added with the evaluation harness (eval/run_eval.py; numbers in eval/RESULTS.md):
     --hints DETS --hint-prior W  soft detector prior: a pass's vote is multiplied by W + (1 - W) * p of a detection
                                  within --hint-dt s of its CPA (W for passes no detector supports). Recommended:
                                  v1 detections, W = 0.5 (dev blind 15 -> 23 of 36 within 2 km). Without
                                  --hint-prior, hints only add candidate spots (old behaviour).
     --silence-pen L              every close (< --silence-d) prop/rotor pass whose CPA lies inside the recording at
                                  clock S but that does not vote costs L (same for the null). 2x faster, accuracy
                                  neutral, better calibrated.
     --excl-s 20 + --rescore 1 --same-dt 20 --same-dlogf 9 --hex-tol 9   one sound = one vote: passes whose CPAs are
                                  within 20 s explain the same sound (often different harmonics of it) in the grid,
                                  and the final hypotheses are re-scored the same way.
     Tested and off by default: --vote idf (per-spot chance weighting: hurt), --calib-win (window-restricted z0:
     hurt), --pre-top (finer coarse re-score: no gain), --win-max (clock-consistent spots only: same scores),
     larger --n-cand (no gain, 2.5-4x slower).
     Round 3 (eval/RESULTS.md, "Round 3"):
     --silence-hint P             a close pass is not counted silent where a detector hint with p >= P lies within
                                  --silence-hint-dt (20) s of its CPA (the curve can miss a pass the detector hears:
                                  ADS-B timing offsets, xc1085643's PA34). Dev neutral, kept with P = 0.5.
     --silence-cover 1            penalise only passes the curve model covers (straight / steady / sampled over
                                  +-cover_w s; explain() reports `covered`): off, does not catch the failure above.
     --tie-z W, --spots K         vote above the cap / several refined spots per NMS cluster: both hurt on dev, off.
     Recommended (round 3): --hints v3_dets.json (model/v3) --hint-pmin 0.8 --hint-prior 0.5 --silence-pen 1
     --silence-hint 0.5 --excl-s 20 --rescore 1 --same-dt 20 --same-dlogf 9 --hex-tol 9
     (test blind 12 -> 21 of 55 within 2 km vs the defaults; round-2 recommendation below gives 20 of 55)
     Round 2 recommendation: --hints v2_dets.json --hint-pmin 0.8 --hint-prior 0.5 --silence-pen 1 --excl-s 20 --rescore 1
     --same-dt 20 --same-dlogf 9 --hex-tol 9 (detector v2 r3, model/v2det.py; test blind 5 -> 9 of 18 within 2 km
     vs the defaults; with v1 detections drop --hint-pmin)

Usage
  python locate/gpu_locate.py --spec rec.spec.npz --tracks tracks.npz --dem-dir dem \
      --bbox 40.75,41.25,-74.0,-72.45 --spacing 500 [--start 2026-09-08T11:00Z --end 2026-09-08T23:00Z] \
      [--hints v0_dets.json] [--truth 40.87,-73.5] --out runs/colsto_blind
Outputs: <out>.json (ranked hypotheses: site, start UTC, explaining passes with z/f0/CPA/d, score, null margin),
<out>.heat.png (coarse heat map), <out>.refine<k>.png (fine heat maps), <out>.rank<k>.png (curves on spectrogram),
<out>.grid.npz (coarse score arrays). --truth/--compare are evaluation only (reported, never used for ranking).
Related: diag_kingair.py (event-B model check), diag_trace.py (trace-vs-ADS-B rms maps).
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

from skyhum._vendor.afg.dem import ElevationModel, fetch_copernicus  # noqa: E402
from skyhum._vendor.afg.geo import enu_vectors_to_ecef, geodetic_to_ecef  # noqa: E402
from skyhum._vendor.afg.tracks import Flights, Resampled  # noqa: E402

def _pick_device():
    want = os.environ.get("SKYHUM_DEVICE", "auto")
    if want == "cpu" or os.environ.get("SKYHUM_NO_GPU_PROBE"):
        return torch.device("cpu")             # never initialise CUDA
    if want == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEV = _pick_device()
F_LO, F_HI = 30.0, 600.0
F0_LO, F0_HI = 35.0, 450.0
CLIP = (0.0, 15.0)
HALF_W = 60            # s of curve either side of CPA (reception time)
DIL = 2                # s; votes of different passes agree within +-DIL
EXCL_S = 8.0           # s; passes at one site with CPAs closer than this explain the same sound -> max, not sum
T_LOG = time.time()
NEG = -1e9             # score of a site / clock without enough voters (round-5b LLR scores can be negative)


def group_of(cls, desc):
    """piston / turboprop / heli / other from the ADS-B class and the Doc 8643 description code (eval/r5b)."""
    if cls == "rotor" or desc[:1] in ("H", "G"):
        return "heli"
    if cls == "prop" and len(desc) == 3 and desc[2] in "PT":
        return "piston" if desc[2] == "P" else "turboprop"
    return "other"


def log(*a):
    if os.environ.get("SKYHUM_VERBOSE"):
        print(f"[{time.time() - T_LOG:7.1f}s]", *a, file=sys.stderr, flush=True)


def parse_utc(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        s = s.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-5] + "Z"


# ----------------------------------------------------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------------------------------------------------
class LogSpec:
    """Whitened spectrogram resampled to log frequency (rows) and pooled to col_s columns, on the GPU."""

    def __init__(self, W, f, t, hop, col_s, log_step, fpool=0, harm=1, clip=CLIP):
        Wt = torch.as_tensor(np.clip(W, *clip), device=DEV)
        n = max(1, int(round(col_s / hop)))
        nt = Wt.shape[1] // n
        Wt = Wt[:, :nt * n].reshape(Wt.shape[0], nt, n).mean(2)
        self.t0 = float(np.mean(t[:n]))            # recording time (s) of column 0 centre
        self.col_s = n * hop
        self.nt = nt
        nb = int(np.floor(np.log(F_HI / F_LO) / log_step)) + 1
        self.lf = F_LO * np.exp(log_step * np.arange(nb))
        fi = np.interp(self.lf, f, np.arange(f.size))
        i0 = np.floor(fi).astype(int)
        i1 = np.minimum(i0 + 1, f.size - 1)
        w = torch.as_tensor((fi - i0)[:, None], device=DEV, dtype=torch.float32)
        L = (1 - w) * Wt[torch.as_tensor(i0, device=DEV)] + w * Wt[torch.as_tensor(i1, device=DEV)]
        if fpool:
            L = torch.nn.functional.max_pool1d(L.T[None], 2 * fpool + 1, 1, fpool)[0].T
        if harm > 1:
            # harmonic sum: row b holds the mean of rows b + log(h)/step, h = 1..harm (zero above F_HI)
            H = L.clone()
            for h in range(2, harm + 1):
                sh = int(round(np.log(h) / log_step))
                H[:nb - sh] += L[sh:]
            L = H / harm
        self.L = L.contiguous()
        self.nb = nb
        self.log_step = log_step
        self.b0lo = int(np.ceil(np.log(F0_LO / F_LO) / log_step))
        self.b0hi = int(np.floor(np.log(F0_HI / F_LO) / log_step)) + 1

    def f0_of_row(self, b):
        return F_LO * np.exp(self.log_step * np.asarray(b, float))


def load_spec(path):
    z = np.load(path, allow_pickle=False)
    return dict(W=z["W"].astype(np.float32), f=z["f"], t=z["t"], hop=float(z["hop_s"]), duration=float(z["duration"]))


# ----------------------------------------------------------------------------------------------------------------
# Sites and flights
# ----------------------------------------------------------------------------------------------------------------
def make_grid(bbox, spacing, dem):
    la0, la1, lo0, lo1 = bbox
    latc = 0.5 * (la0 + la1)
    dlat = spacing / 111132.0
    dlon = spacing / (111320.0 * math.cos(math.radians(latc)))
    lats = np.arange(la0, la1 + 1e-9, dlat)
    lons = np.arange(lo0, lo1 + 1e-9, dlon)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    g = dem.sample(LA.ravel(), LO.ravel())
    return lats, lons, LA.ravel(), LO.ravel(), g


class Frame:
    """Local ENU frame at a reference point (float32-friendly coordinates for culling)."""

    def __init__(self, lat0, lon0):
        self.o = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([0.0]))[0]
        la, lo = np.radians(lat0), np.radians(lon0)
        self.R = np.array([[-np.sin(lo), np.cos(lo), 0],
                           [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                           [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])

    def enu(self, ecef):
        return (ecef - self.o) @ self.R.T


def ecef_to_geodetic_t(X):
    """(..., 3) float64 ECEF -> lat, lon (deg), ellipsoidal height (m); WGS84, Bowring (one step: < 1 mm here)."""
    A, F = 6378137.0, 1 / 298.257223563
    B = A * (1 - F)
    E2 = F * (2 - F)
    EP2 = E2 / (1 - E2)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    lon = torch.atan2(y, x)
    p = torch.sqrt(x * x + y * y)
    th = torch.atan2(z * A, p * B)
    lat = torch.atan2(z + EP2 * B * torch.sin(th) ** 3, p - E2 * A * torch.cos(th) ** 3)
    N = A / torch.sqrt(1 - E2 * torch.sin(lat) ** 2)
    h = p / torch.cos(lat) - N
    return torch.rad2deg(lat), torch.rad2deg(lon), h


class Traffic:
    def __init__(self, tracks, t0, t1, classes, bbox, dmax, dec=4):
        fl = Flights(tracks)
        self.fl = fl
        la0, la1, lo0, lo1 = bbox
        m_lat = dmax / 111132.0 + 0.02
        m_lon = dmax / (111320.0 * math.cos(math.radians(0.5 * (la0 + la1)))) + 0.02
        ok = ((fl.t >= t0) & (fl.t <= t1) & (fl.h < dmax + 1500) & (fl.lat > la0 - m_lat) & (fl.lat < la1 + m_lat)
              & (fl.lon > lo0 - m_lon) & (fl.lon < lo1 + m_lon))
        fids = np.unique(fl.fid[ok])
        fids = np.array([f for f in fids if str(fl.cls[f]) in classes], int)
        log(f"traffic: {fl.nflights} segments in file, {fids.size} low enough / near the box / in time ({classes})")
        R = Resampled(fl, dt=1.0, t0=t0, t1=t1, fids=fids)
        self.R = R
        self.fids = np.unique(R.fid)
        # per-flight contiguous range in R
        self.first = np.array([R._first[int(f)] for f in self.fids])
        self.last = np.array([R._last[int(f)] for f in self.fids]) + 1
        self.ecef = torch.as_tensor(R.ecef, device=DEV, dtype=torch.float64)
        self.vel = torch.as_tensor(R.vel, device=DEV, dtype=torch.float64)
        self.tau = torch.as_tensor(R.t, device=DEV, dtype=torch.float64)
        self.dec = dec
        maxf = int(R.fid.max()) + 1
        F0 = np.zeros(maxf, np.int64)
        F1 = np.zeros(maxf, np.int64)
        for f, a in R._first.items():
            F0[f] = a
            F1[f] = R._last[f]
        self.F0_t = torch.as_tensor(F0, device=DEV)
        self.F1_t = torch.as_tensor(F1, device=DEV)
        self.fid_t = torch.as_tensor(R.fid, device=DEV, dtype=torch.long)
        # flights that should be audible when close (silence penalty): prop / rotor (not unknown, not jets)
        self.loud_t = torch.as_tensor(np.array([str(fl.cls[f]) in ("prop", "rotor") if f < len(fl.cls) else False
                                                for f in range(maxf)]), device=DEV)
        # seconds from each resampled sample to the nearest raw ADS-B point of its flight (silence coverage check)
        gap = np.full(R.n, 1e3, np.float32)
        o = np.argsort(fl.fid, kind="stable")
        fs_, ts_ = fl.fid[o], fl.t[o]
        b0 = np.searchsorted(fs_, self.fids, "left")
        b1 = np.searchsorted(fs_, self.fids, "right")
        for f, a0, a1, i0, i1 in zip(self.fids, b0, b1, self.first, self.last):
            rt = np.sort(ts_[a0:a1])
            if rt.size == 0:
                continue
            tt = R.t[i0:i1]
            j = np.clip(np.searchsorted(rt, tt), 1, max(1, rt.size - 1))
            gap[i0:i1] = np.minimum(np.abs(tt - rt[j - 1]), np.abs(rt[np.minimum(j, rt.size - 1)] - tt))
        self.rawgap_t = torch.as_tensor(gap, device=DEV)
        log(f"traffic: {self.fids.size} flights resampled, {R.n} samples at 1 s")


def enumerate_passes(tr, frame, site_enu, grid_shape, lats, lons, dmax, t_c_lo, t_c_hi, site_mask=None):
    """All (site, flight, approx CPA sample) with a local slant-range minimum <= dmax (+ culling margin).
    Returns int64 arrays (site, sample index into Resampled)."""
    R = tr.R
    enu_all = frame.enu(R.ecef).astype(np.float32)
    S = torch.as_tensor(site_enu, device=DEV, dtype=torch.float32)
    nla, nlo = grid_shape
    dlat = lats[1] - lats[0] if lats.size > 1 else 1.0
    dlon = lons[1] - lons[0] if lons.size > 1 else 1.0
    out_s, out_i = [], []
    margin = 400.0
    for f, a, b in zip(tr.fids, tr.first, tr.last):
        idx = np.arange(a, b, tr.dec)
        if idx[-1] != b - 1:
            idx = np.r_[idx, b - 1]
        if idx.size < 3:
            continue
        # cheap time cull: reception time ~ emission time + < 40 s
        tt = R.t[idx]
        if tt[-1] < t_c_lo - 200 or tt[0] > t_c_hi + 200:
            continue
        P = enu_all[idx]
        low = P[:, 2] < dmax + 600
        if not low.any():
            continue
        la, lo = R.lat[idx][low], R.lon[idx][low]
        i0 = max(0, int(np.floor((la.min() - lats[0] - dmax / 111132.0) / dlat)))
        i1 = min(nla, int(np.ceil((la.max() - lats[0] + dmax / 111132.0) / dlat)) + 1)
        mlon = dmax / (111320.0 * math.cos(math.radians(la.mean())))
        j0 = max(0, int(np.floor((lo.min() - lons[0] - mlon) / dlon)))
        j1 = min(nlo, int(np.ceil((lo.max() - lons[0] + mlon) / dlon)) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        sid = (np.arange(i0, i1)[:, None] * nlo + np.arange(j0, j1)[None, :]).ravel()
        sid_t = torch.as_tensor(sid, device=DEV)
        if site_mask is not None:
            sid_t = sid_t[site_mask[sid_t]]
            if sid_t.numel() == 0:
                continue
        D = torch.cdist(S[sid_t], torch.as_tensor(P, device=DEV))           # (ns, m)
        Dp = torch.nn.functional.pad(D, (1, 1), value=float("inf"))
        loc = (D <= Dp[:, :-2]) & (D < Dp[:, 2:]) & (D < dmax + margin)
        loc[:, 0] = False                                                    # truncated at the flight start/end
        loc[:, -1] = False
        ss, kk = torch.nonzero(loc, as_tuple=True)
        if ss.numel():
            out_s.append(sid_t[ss])
            out_i.append(torch.as_tensor(idx, device=DEV)[kk])
    if not out_s:
        return torch.zeros(0, dtype=torch.long, device=DEV), torch.zeros(0, dtype=torch.long, device=DEV)
    s = torch.cat(out_s)
    i = torch.cat(out_i)
    o = torch.argsort(s * (R.n + 1) + i)
    return s[o], i[o]


class Templates:
    """Exact emission-time Doppler templates for a batch of passes."""

    def __init__(self, tr, obs_ecef, site, samp, dmax, c, log_step, col_s, half_w=HALF_W, wind_ecef=None,
                 max_db=None, t_c_lo=-np.inf, t_c_hi=np.inf, wpow=0.0):
        R = tr.R
        Wn = int(round(half_w)) + 55
        P = site.numel()
        fl_of = tr.fid_t[samp]
        a = tr.F0_t[fl_of]
        b = tr.F1_t[fl_of]
        off = torch.arange(-Wn, Wn + 1, device=DEV)
        idx = samp[:, None] + off[None, :]
        inwin = (idx >= a[:, None]) & (idx <= b[:, None])
        idx = torch.minimum(torch.maximum(idx, a[:, None]), b[:, None])
        obs = obs_ecef[site]                                                   # (P,3) float64
        D = tr.ecef[idx] - obs[:, None, :]
        rng = torch.linalg.norm(D, dim=2)
        rdot = (D * tr.vel[idx]).sum(2) / rng.clamp(min=1e-6)
        if wind_ecef is None:
            ceff = torch.full_like(rng, c)
        else:
            n = -D / rng.clamp(min=1e-6)[..., None]
            ceff = c + (n * wind_ecef).sum(2)
        trx = tr.tau[idx] + rng / ceff
        lr = -torch.log1p(rdot / ceff)                                         # log(f/f0)
        rk = torch.where(inwin, rng, torch.full_like(rng, float("inf")))
        k = rk.argmin(1)
        ar = torch.arange(P, device=DEV)
        d_cpa = rng[ar, k]
        # first / last valid window index
        kv0 = inwin.float().argmax(1)
        kv1 = 2 * Wn - inwin.flip(1).float().argmax(1)
        ok = (d_cpa <= dmax) & (k > kv0) & (k < kv1)
        t_c = trx[ar, k]
        ok &= (t_c >= t_c_lo) & (t_c <= t_c_hi)
        J = int(round(half_w / col_s))
        jj = torch.arange(-J, J + 1, device=DEV, dtype=torch.float64) * col_s
        tq = t_c[:, None] + jj[None, :]
        # trx is increasing (c > v); invalid window entries are clipped copies at the ends
        trx_v = torch.where(inwin, trx, torch.where(off[None, :] < 0, trx[ar, kv0][:, None] - 1e3 + off[None, :] * 1e-3,
                                                    trx[ar, kv1][:, None] + 1e3 + off[None, :] * 1e-3))
        pos = torch.searchsorted(trx_v.contiguous(), tq.contiguous()).clamp(1, 2 * Wn)
        x0 = trx_v.gather(1, pos - 1)
        x1 = trx_v.gather(1, pos)
        y0 = lr.gather(1, pos - 1)
        y1 = lr.gather(1, pos)
        wgt = ((tq - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0, 1)
        lq = y0 + wgt * (y1 - y0)
        rq = rng.gather(1, pos - 1) + wgt * (rng.gather(1, pos) - rng.gather(1, pos - 1))
        self.w = (d_cpa[:, None] / rq.clamp(min=1.0)).float() ** wpow if wpow else torch.ones_like(rq, dtype=torch.float32)
        valid = (tq >= trx[ar, kv0][:, None]) & (tq <= trx[ar, kv1][:, None])
        db = torch.round(lq / log_step).long()
        if max_db is not None:
            valid &= db.abs() <= max_db
        ok &= valid.sum(1) >= int(0.6 * (2 * J + 1))
        self.ok = ok
        self.db = db
        self.valid = valid
        self.t_c = t_c
        self.d_cpa = d_cpa
        self.site = site
        self.fid = fl_of
        self.samp = samp
        self.v_cpa = torch.linalg.norm(tr.vel[idx][ar, k], dim=1)
        up = obs / torch.linalg.norm(obs, dim=1, keepdim=True)
        self.elev = torch.rad2deg(torch.asin(((D[ar, k] * up).sum(1) / rng[ar, k].clamp(min=1.0)).clamp(-1, 1))).float()
        self.cpa_vec = D[ar, k]                     # observer -> aircraft at CPA (ECEF, m): terrain line of sight
        self.vel_cpa = tr.vel[idx][ar, k]            # aircraft velocity at CPA (ECEF, m/s): single-event mirror point
        self.obs = obs
        self.J = J
        self.lr = lq

    def select(self, m):
        for k in ("db", "valid", "t_c", "d_cpa", "site", "fid", "v_cpa", "ok", "lr", "w", "samp", "elev", "cpa_vec", "obs", "vel_cpa"):
            setattr(self, k, getattr(self, k)[m])
        return self


# ----------------------------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------------------------
def fast_len(n):
    m = n
    while True:
        x = m
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


class FFTScorer:
    """Profiles P(tau) = max_f0 [mean level along the template - flat line at f0] for all CPA columns tau."""

    def __init__(self, ls, J, max_db, dr=0):
        self.ls = ls
        self.dr = dr
        max_db = max_db + dr
        nb, nt = ls.L.shape
        self.NB = fast_len(max(nb + max_db - ls.b0lo + 1, ls.b0hi + max_db + 1))
        self.NT = fast_len(nt + 2 * J + 2)
        Lp = torch.zeros(self.NB, self.NT, device=DEV)
        Lp[:nb, :nt] = ls.L
        self.FL = torch.fft.rfft2(Lp)
        self.J = J
        log(f"FFT scorer: L {nb}x{nt} padded to {self.NB}x{self.NT}")

    def _fill(self, T, bi, db, cols, w):
        if self.dr:   # ridge: curve minus the curve shifted +-dr rows
            T.index_put_((bi, db % self.NB, cols), w, accumulate=True)
            T.index_put_((bi, (db + self.dr) % self.NB, cols), -0.5 * w, accumulate=True)
            T.index_put_((bi, (db - self.dr) % self.NB, cols), -0.5 * w, accumulate=True)
        else:         # flat: curve minus the flat line at f0
            T.index_put_((bi, db % self.NB, cols), w, accumulate=True)
            T.index_put_((bi, torch.zeros_like(db), cols), -w, accumulate=True)

    def full_maps(self, db, w):
        """(B, f0 rows, nt) weighted mean level along the curve minus the flat line."""
        B, K = db.shape
        J = self.J
        T = torch.zeros(B, self.NB, self.NT, device=DEV)
        cols = ((torch.arange(K, device=DEV) - J) % self.NT)[None, :].expand(B, K)
        bi = torch.arange(B, device=DEV)[:, None].expand(B, K)
        self._fill(T, bi, db, cols, w)
        C = torch.fft.irfft2(self.FL[None] * torch.fft.rfft2(T).conj(), s=(self.NB, self.NT))
        return C[:, self.ls.b0lo:self.ls.b0hi, :self.ls.nt] / w.sum(1).clamp(min=1e-6)[:, None, None]

    def zmap(self, db, valid, w, rev=False):
        if rev:
            db, valid, w = db.flip(1), valid.flip(1), w.flip(1)
        P, arg = self.profiles(db, valid, w)
        return robust_z(P), arg

    def profiles(self, db, valid, w=None):
        """Weighted mean level along the curve minus the same weights on the flat line at f0 (one template image:
        +w at (db_k, k), -w at (0, k)), max over f0 rows."""
        B, K = db.shape
        J = self.J
        wv = valid.float() if w is None else w * valid.float()
        T = torch.zeros(B, self.NB, self.NT, device=DEV)
        cols = ((torch.arange(K, device=DEV) - J) % self.NT)[None, :].expand(B, K)
        bi = torch.arange(B, device=DEV)[:, None].expand(B, K)
        self._fill(T, bi, db, cols, wv)
        FT = torch.fft.rfft2(T)
        del T
        C = torch.fft.irfft2(self.FL[None] * FT.conj(), s=(self.NB, self.NT))
        del FT
        C = C[:, self.ls.b0lo:self.ls.b0hi, :self.ls.nt]
        n = wv.sum(1).clamp(min=1e-6)
        P, arg = C.max(1)
        return P / n[:, None], arg + self.ls.b0lo


def robust_z(P):
    med = P.median(1, keepdim=True).values
    mad = (P - med).abs().median(1, keepdim=True).values * 1.4826 + 1e-6
    return (P - med) / mad


# ----------------------------------------------------------------------------------------------------------------
# Candidate scorer: exact templates evaluated only around curve-like spots of the spectrogram (Triton gather kernel)
# ----------------------------------------------------------------------------------------------------------------
try:
    if DEV.type != "cuda":
        raise ImportError("Triton kernels are used on CUDA only")
    import triton
    import triton.language as tl

    @triton.jit
    def _curve_kernel(L_ptr, nb, nt, ld, db_ptr, w_ptr, K, J, qb_ptr, qt_ptr, Q, out_ptr, DR, BLOCK: tl.constexpr):
        p = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        qm = offs < Q
        qb = tl.load(qb_ptr + offs, mask=qm, other=0)
        qt = tl.load(qt_ptr + offs, mask=qm, other=0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, K):
            d = tl.load(db_ptr + p * K + k)
            w = tl.load(w_ptr + p * K + k)
            r = qb + d
            c = qt + (k - J)
            cm = qm & (c >= 0) & (c < nt)
            v = tl.load(L_ptr + r * ld + c, mask=cm & (r >= 0) & (r < nb), other=0.0)
            vu = tl.load(L_ptr + (r + DR) * ld + c, mask=cm & (r + DR >= 0) & (r + DR < nb), other=0.0)
            vd = tl.load(L_ptr + (r - DR) * ld + c, mask=cm & (r - DR >= 0) & (r - DR < nb), other=0.0)
            acc += w * (v - 0.5 * (vu + vd))
        tl.store(out_ptr + p * Q + offs, acc, mask=qm)

    @triton.jit
    def _curve_kernel_pp(L_ptr, nb, nt, ld, db_ptr, w_ptr, K, J, qb_ptr, qt_ptr, Q, out_ptr, DR, BLOCK: tl.constexpr):
        # same as _curve_kernel, but every pass p has its own Q queries (qb/qt of shape (P, Q); qb < 0 = padding)
        p = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        qm = offs < Q
        qb = tl.load(qb_ptr + p * Q + offs, mask=qm, other=-1)
        qt = tl.load(qt_ptr + p * Q + offs, mask=qm, other=0)
        qm = qm & (qb >= 0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, K):
            d = tl.load(db_ptr + p * K + k)
            w = tl.load(w_ptr + p * K + k)
            r = qb + d
            c = qt + (k - J)
            cm = qm & (c >= 0) & (c < nt)
            v = tl.load(L_ptr + r * ld + c, mask=cm & (r >= 0) & (r < nb), other=0.0)
            vu = tl.load(L_ptr + (r + DR) * ld + c, mask=cm & (r + DR >= 0) & (r + DR < nb), other=0.0)
            vd = tl.load(L_ptr + (r - DR) * ld + c, mask=cm & (r - DR >= 0) & (r - DR < nb), other=0.0)
            acc += w * (v - 0.5 * (vu + vd))
        tl.store(out_ptr + p * Q + offs, acc, mask=qm)

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def curve_scores(L, db, w, qb, qt, J, dr):
    """out[p, q] = sum_k w[p,k] * (L[r, c] - (L[r + dr, c] + L[r - dr, c]) / 2), r = qb[q] + db[p,k],
    c = qt[q] + k - J (zero outside L): level along the curve minus the same curve shifted +-dr rows."""
    P, K = db.shape
    Q = qb.numel()
    out = torch.empty(P, Q, device=DEV, dtype=torch.float32)
    if HAVE_TRITON and DEV.type == "cuda":
        BLOCK = 256
        grid = (P, triton.cdiv(Q, BLOCK))
        _curve_kernel[grid](L, L.shape[0], L.shape[1], L.stride(0), db.to(torch.int32).contiguous(),
                            w.contiguous(), K, J, qb.to(torch.int32).contiguous(), qt.to(torch.int32).contiguous(),
                            Q, out, dr, BLOCK=BLOCK)
        return out
    nb, nt = L.shape
    Lp = torch.nn.functional.pad(L, (0, 1, 0, 1)).reshape(-1)
    acc = torch.zeros(P, Q, device=DEV)
    for k in range(K):
        r = qb[None, :] + db[:, k:k + 1]
        c = (qt + k - J)[None, :].expand(P, -1)
        def g(rr):
            ok = (rr >= 0) & (rr < nb) & (c >= 0) & (c < nt)
            return Lp[torch.where(ok, rr * (nt + 1) + c, torch.full_like(rr, nb * (nt + 1)))]
        acc += w[:, k:k + 1] * (g(r) - 0.5 * (g(r + dr) + g(r - dr)))
    return out.copy_(acc)


def linear_pass_bank(J, col_s, log_step, c, wpow, speeds, dists):
    """Templates of straight constant-speed passes (only used to find curve-like spots, never to score sites)."""
    dbs, ws = [], []
    tq = np.arange(-J, J + 1) * col_s
    for v in speeds:
        for d in dists:
            tau = np.arange(-400, 400.01, 0.25)
            R = np.hypot(v * tau, d)
            trx = tau + R / c
            lr = -np.log1p(v * v * tau / R / c)
            t_c = d / c
            l = np.interp(t_c + tq, trx, lr)
            r = np.interp(t_c + tq, trx, R)
            dbs.append(np.round(l / log_step).astype(np.int64))
            ws.append((d / r) ** wpow)
    return torch.as_tensor(np.array(dbs), device=DEV), torch.as_tensor(np.array(ws), device=DEV, dtype=torch.float32)


class CandScorer:
    """Score passes only at candidate spots (CPA column, f0 row) where a bank of generic falling (or, for the null,
    rising) curves responds, +-rt columns x +-rb rows around each; noise from random spots of the same shape."""

    def __init__(self, ls, J, max_db, c, wpow, n_cand, rt, rb, n_noise=96, hints=None, hint_dt=8.0, hint_df=0.03,
                 hint_pmin=0.2, z_bank=3.0, hints_only=False, dr=4):
        self.ls, self.J = ls, J
        self.rt, self.rb = rt, rb
        self.dr = dr
        fft = FFTScorer(ls, J, max_db, dr)
        speeds = np.arange(20, 161, 15.0)
        dists = 300 * 1.5 ** np.arange(9)
        db, w = linear_pass_bank(J, ls.col_s, ls.log_step, c, wpow, speeds, dists)
        self.cand = {}
        for rev in (False, True):
            dbb, wb = (db.flip(1), w.flip(1)) if rev else (db, w)
            Z = None
            for s0 in range(0, db.shape[0], 12):
                Cm = fft.full_maps(dbb[s0:s0 + 12], wb[s0:s0 + 12])          # (B, rows, nt)
                flat = Cm.reshape(Cm.shape[0], -1)
                g = torch.Generator(device="cpu").manual_seed(0)
                sub = torch.randint(0, flat.shape[1], (60000,), generator=g).to(DEV)
                smp = flat[:, sub]
                med = smp.median(1).values
                mad = (smp - med[:, None]).abs().median(1).values * 1.4826 + 1e-6
                Zb = ((Cm - med[:, None, None]) / mad[:, None, None]).amax(0)
                Z = Zb if Z is None else torch.maximum(Z, Zb)
                del Cm
            pk = torch.nn.functional.max_pool2d(Z[None, None], (2 * rb + 1, 2 * rt + 1), 1, (rb, rt))[0, 0]
            m = (Z == pk) & (Z > z_bank)
            # ignore spots whose +-J curve would stick out of the recording by more than half
            rows, cols = torch.nonzero(m, as_tuple=True)
            vals = Z[rows, cols]
            o = torch.argsort(vals, descending=True)[:n_cand if not (hints_only and hints and not rev) else 0]
            cb = rows[o] + ls.b0lo
            ct = cols[o]
            if hints and not rev:
                hb, ht = [], []
                for h in hints:
                    if h.get("p", 1.0) < hint_pmin:
                        continue
                    c0 = int(round((h["t"] - ls.t0) / ls.col_s))
                    b0 = int(round(np.log(h["f"] / F_LO) / ls.log_step))
                    nt_ = int(np.ceil(hint_dt / ls.col_s / (2 * rt + 1)))
                    nb_ = int(np.ceil(hint_df / ls.log_step / (2 * rb + 1)))
                    for i in range(-nt_, nt_ + 1):
                        for j in range(-nb_, nb_ + 1):
                            ht.append(c0 + i * (2 * rt + 1))
                            hb.append(b0 + j * (2 * rb + 1))
                cb = torch.cat([cb, torch.as_tensor(hb, device=DEV, dtype=torch.long)])
                ct = torch.cat([ct, torch.as_tensor(ht, device=DEV, dtype=torch.long)])
            self.cand[rev] = (cb, ct, vals[o])
            log(f"bank {'rising (null)' if rev else 'falling'}: {int(m.sum())} spots > z {z_bank}, "
                f"kept {cb.numel()} (z_bank >= {float(vals[o][-1]) if o.numel() else 0:.1f})")
        del fft
        g = torch.Generator(device="cpu").manual_seed(2)
        self.noise_b = torch.randint(ls.b0lo, ls.b0hi, (n_noise,), generator=g).to(DEV)
        self.noise_t = torch.randint(J // 2, ls.nt - J // 2, (n_noise,), generator=g).to(DEV)
        ob, ot = torch.meshgrid(torch.arange(-rb, rb + 1, device=DEV), torch.arange(-rt, rt + 1, device=DEV),
                                indexing="ij")
        self.ob, self.ot = ob.reshape(-1), ot.reshape(-1)
        self.E = self.ob.numel()
        self.L = ls.L.contiguous()
        self.idf = {False: None, True: None}

    def zcand(self, db, valid, w, rev=False, grp=None):
        """z of every pass at every candidate spot (P, n_cand), with the CPA column and f0 row of the best fit.
        grp (P,) long: with a harmonic comb (--comb, self.Lg), pass p is scored on the comb image of its group."""
        cb, ct, _ = self.cand[rev]
        if rev:
            db, valid, w = db.flip(1), valid.flip(1), w.flip(1)
        wv = (w * valid.float()).contiguous()
        cb_all = torch.cat([cb, self.noise_b])
        ct_all = torch.cat([ct, self.noise_t])
        qb = (cb_all[:, None] + self.ob[None, :]).reshape(-1)
        qt = (ct_all[:, None] + self.ot[None, :]).reshape(-1)
        P = db.shape[0]
        if getattr(self, "Lg", None) and grp is not None:
            out = torch.empty(P, qb.numel(), device=DEV)
            for g in torch.unique(grp).tolist():
                m = grp == g
                Li = self.Lg.get(int(g), self.L)
                out[m] = curve_scores(Li, db[m].contiguous(), wv[m].contiguous(), qb, qt, self.J, self.dr)
            out = out / wv.sum(1).clamp(min=1e-6)[:, None]
        else:
            out = curve_scores(self.L, db, wv, qb, qt, self.J, self.dr) / wv.sum(1).clamp(min=1e-6)[:, None]
        out = out.view(P, -1, self.E)
        val, arg = out.max(2)
        nc = cb.numel()
        noise = val[:, nc:]
        med = noise.median(1, keepdim=True).values
        mad = (noise - med).abs().median(1, keepdim=True).values * 1.4826 + 1e-6
        z = (val[:, :nc] - med) / mad
        tcol = (ct[None, :] + self.ot[arg[:, :nc]]).clamp(0, self.ls.nt - 1)
        brow = cb[None, :] + self.ob[arg[:, :nc]]
        return z, tcol, brow

    def zcand_win(self, db, valid, w, rev, t_c, S_lo, S_hi):
        """zcand restricted to the spots a pass can use when the recording start S is within [S_lo, S_hi]: its CPA
        column (t_c - t0 - S) / col_s must be within +-(rt + DIL + 1) of the spot. Same z values as zcand for those
        spots, -1e9 for the others (never vote)."""
        cb, ct, _ = self.cand[rev]
        if rev:
            db, valid, w = db.flip(1), valid.flip(1), w.flip(1)
        if not hasattr(self, "_ord"):
            self._ord = {}
        if rev not in self._ord:
            o = torch.argsort(ct)
            self._ord[rev] = (o, ct[o].contiguous(), cb[o])
        o, cts, cbs = self._ord[rev]
        wv = (w * valid.float()).contiguous()
        P, nc = db.shape[0], ct.numel()
        nrm = wv.sum(1).clamp(min=1e-6)[:, None]
        # noise (shared queries)
        qb = (self.noise_b[:, None] + self.ob[None, :]).reshape(-1)
        qt = (self.noise_t[:, None] + self.ot[None, :]).reshape(-1)
        noise = (curve_scores(self.L, db, wv, qb, qt, self.J, self.dr) / nrm).view(P, -1, self.E).amax(2)
        med = noise.median(1, keepdim=True).values
        mad = (noise - med).abs().median(1, keepdim=True).values * 1.4826 + 1e-6
        pad = self.rt + DIL + 1
        c_hi = (t_c - self.ls.t0 - S_lo) / self.ls.col_s + pad
        c_lo = (t_c - self.ls.t0 - S_hi) / self.ls.col_s - pad
        a = torch.searchsorted(cts.double(), c_lo.contiguous())
        b = torch.searchsorted(cts.double(), c_hi.contiguous(), right=True)
        cnt = (b - a).clamp(min=0)
        M = int(cnt.max()) if P else 0
        z = torch.full((P, nc), -1e9, device=DEV)
        tcol = torch.zeros(P, nc, dtype=torch.long, device=DEV)
        brow = torch.zeros(P, nc, dtype=torch.long, device=DEV)
        if M == 0:
            return z, tcol, brow
        m = torch.arange(M, device=DEV)
        ok = m[None, :] < cnt[:, None]                                      # (P, M)
        si = (a[:, None] + m[None, :]).clamp(max=nc - 1)                     # index into sorted spots
        qb = torch.where(ok[:, :, None], cbs[si][:, :, None] + self.ob[None, None, :], -1).reshape(P, -1)
        qt = (cts[si][:, :, None] + self.ot[None, None, :]).reshape(P, -1)
        Q = qb.shape[1]
        out = torch.empty(P, Q, device=DEV, dtype=torch.float32)
        BLOCK = 128
        _curve_kernel_pp[(P, triton.cdiv(Q, BLOCK))](
            self.L, self.L.shape[0], self.L.shape[1], self.L.stride(0), db.to(torch.int32).contiguous(), wv, db.shape[1],
            self.J, qb.to(torch.int32).contiguous(), qt.to(torch.int32).contiguous(), Q, out, self.dr, BLOCK=BLOCK)
        val, arg = (out / nrm).view(P, M, self.E).max(2)
        zz = (val - med) / mad
        orig = o[si]                                                         # original spot index
        pp = torch.arange(P, device=DEV)[:, None].expand(P, M)
        z[pp[ok], orig[ok]] = zz[ok]
        tcol[pp[ok], orig[ok]] = (cts[si] + self.ot[arg]).clamp(0, self.ls.nt - 1)[ok]
        brow[pp[ok], orig[ok]] = (cbs[si] + self.ob[arg])[ok]
        return z, tcol, brow

    def zmap(self, db, valid, w, rev=False, vote=None, win=None, stat=None, grp=None):
        """Dense (P, nt) z map (0 where nothing was evaluated) and the f0 row of each value. With vote(z, rev),
        the map holds per-spot votes instead of z. win = (t_c, S_lo, S_hi) restricts to clock-consistent spots.
        stat(z, brow) -> per-spot statistic used instead of z (round 5b: z + kappa * f0 prior)."""
        if win is not None and HAVE_TRITON and DEV.type == "cuda" and not getattr(self, "Lg", None):
            z, tcol, brow = self.zcand_win(db, valid, w, rev, *win)
        else:
            z, tcol, brow = self.zcand(db, valid, w, rev, grp)
        if stat is not None:
            z = stat(z, brow)
        if vote is not None:
            z = vote(z, rev)
        P = z.shape[0]
        Z = torch.zeros(P, self.ls.nt, device=DEV)
        Z.scatter_reduce_(1, tcol, z, "amax")
        # f0 row of the winning candidate per column
        R = torch.zeros(P, self.ls.nt, device=DEV, dtype=torch.long)
        win = z >= Z.gather(1, tcol)
        R.scatter_(1, torch.where(win, tcol, torch.zeros_like(tcol)), torch.where(win, brow, torch.zeros_like(brow)))
        return Z, R


# ----------------------------------------------------------------------------------------------------------------
# Main search
# ----------------------------------------------------------------------------------------------------------------
class Search:
    def __init__(self, a):
        self.a = a
        sp = load_spec(a.spec)
        self.sp = sp
        self.dur = sp["duration"]
        self.ls = LogSpec(sp["W"], sp["f"], sp["t"], sp["hop"], a.col_s, a.log_step, a.fpool, a.harm, (0.0, a.clip))
        log(f"spectrogram: {sp['W'].shape} -> log {self.ls.L.shape} (col {self.ls.col_s:.2f} s, step {a.log_step})")
        self.bbox = [float(x) for x in a.bbox.split(",")]
        if a.dem_dir and glob.glob(os.path.join(a.dem_dir, "*.tif")):
            paths = sorted(glob.glob(os.path.join(a.dem_dir, "*.tif")))
        elif a.dem_dir:
            paths = fetch_copernicus(self.bbox, a.dem_dir)
        else:
            paths = None
        self.dem = ElevationModel(paths) if paths else ElevationModel(constant=a.ground)
        self.c = 20.0468 * math.sqrt(a.temp + 273.15)
        meta = json.loads(str(np.load(a.tracks)["meta"]))
        self.S_min = parse_utc(a.start) if a.start else float(meta["t0"])
        self.S_max = parse_utc(a.end) if a.end else float(meta["t1"]) - self.dur
        self.nS = int(np.ceil(self.S_max - self.S_min)) + 1
        log(f"clock window: start {iso(self.S_min)} .. {iso(self.S_max)} ({self.nS} s), c={self.c:.1f} m/s")
        classes = set(a.classes.split(","))
        if a.jets:
            classes.add("jet")
        t_c_lo = self.S_min + self.ls.t0 - a.half_w
        t_c_hi = self.S_max + self.ls.t0 + self.dur + a.half_w
        self.t_c_lo, self.t_c_hi = t_c_lo, t_c_hi
        self.tr = Traffic(a.tracks, t_c_lo - 400, t_c_hi + 100, classes, self.bbox, a.dmax)
        self.max_db = int(np.ceil(a.max_logspan / a.log_step))
        self.wind = None
        if a.wind:
            we, wn = (float(x) for x in a.wind.split(","))
            la, lo = 0.5 * (self.bbox[0] + self.bbox[1]), 0.5 * (self.bbox[2] + self.bbox[3])
            self.wind = torch.as_tensor(enu_vectors_to_ecef(np.array([la]), np.array([lo]), np.array([we]),
                                                            np.array([wn]), np.array([0.0]))[0], device=DEV)
        self.hints = json.load(open(a.hints)) if a.hints else None
        self.hint_map = None
        self.heard = None
        if self.hints is not None and a.silence_hint > 0:
            # --silence-hint P: columns within --silence-hint-dt s of a detection with p >= P are "heard"; a close
            # pass whose CPA falls there is never counted as silent (the curve model may miss it: turns, ADS-B
            # timing offsets, partial passes - xc1085643's PA34)
            tt = self.ls.t0 + self.ls.col_s * np.arange(self.ls.nt)
            hd = np.zeros(self.ls.nt, bool)
            for h in self.hints:
                if float(h.get("p", 0.0)) >= a.silence_hint:
                    hd |= np.abs(tt - float(h["t"])) <= a.silence_hint_dt
            self.heard = torch.as_tensor(hd, device=DEV)
            log(f"silence penalty: {hd.mean():.3f} of the recording counts as heard (detector p >= {a.silence_hint})")

    # -- one grid ------------------------------------------------------------------------------------------------
    def setup_grid(self, bbox, spacing):
        lats, lons, LA, LO, g = make_grid(bbox, spacing, self.dem)
        frame = Frame(0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3]))
        obs = geodetic_to_ecef(LA, LO, g + 1.5)
        return dict(lats=lats, lons=lons, LA=LA, LO=LO, g=g, obs=torch.as_tensor(obs, device=DEV),
                    enu=frame.enu(obs), frame=frame)

    def make_templates(self, G, site, samp, col_s=None, log_step=None, max_db=None, t_lo=None, t_hi=None):
        return Templates(self.tr, G["obs"], site, samp, self.a.dmax, self.c, log_step or self.a.log_step,
                         col_s or self.ls.col_s, half_w=self.a.half_w, wpow=self.a.wpow, wind_ecef=self.wind, max_db=self.max_db if max_db is None else max_db,
                         t_c_lo=self.t_c_lo if t_lo is None else t_lo, t_c_hi=self.t_c_hi if t_hi is None else t_hi)

    def hint_mask(self, P_shape, t_c):
        """(B, nt) bool: CPA column within --hint-dt of a detector hint."""
        tt = self.ls.t0 + self.ls.col_s * torch.arange(P_shape[1], device=DEV, dtype=torch.float64)
        m = torch.zeros(P_shape, dtype=torch.bool, device=DEV)
        for h in self.hints:
            if h.get("p", 1.0) < self.a.hint_pmin:
                continue
            m |= (tt - h["t"]).abs()[None, :] <= self.a.hint_dt
        return m

    # -- passes of a grid ----------------------------------------------------------------------------------------
    def passes_for_grid(self, G, S_lo, S_hi):
        a = self.a
        t_lo = S_lo + self.ls.t0 - a.half_w
        t_hi = S_hi + self.ls.t0 + self.dur + a.half_w
        site, samp = enumerate_passes(self.tr, G["frame"], G["enu"], (G["lats"].size, G["lons"].size), G["lats"],
                                      G["lons"], a.dmax, t_lo, t_hi, G.get("mask"))
        if site.numel() == 0:
            return site, samp
        keep, tcs, fids = [], [], []
        ch = self.a.tmpl_chunk
        for s0 in range(0, site.numel(), ch):
            T = self.make_templates(G, site[s0:s0 + ch], samp[s0:s0 + ch], t_lo=t_lo, t_hi=t_hi)
            keep.append(T.ok)
            tcs.append(T.t_c)
            fids.append(T.fid)
        keep = torch.cat(keep)
        # the same CPA found from two neighbouring local minima -> keep one
        key = torch.stack([site, torch.cat(fids), torch.round(torch.cat(tcs) / 5).long()], 1)[keep]
        site, samp = site[keep], samp[keep]
        if site.numel() == 0:
            return site, samp
        tcs = torch.cat(tcs)[keep]
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), site.numel(), dtype=torch.long, device=DEV)
        first.scatter_reduce_(0, inv, torch.arange(site.numel(), device=DEV), "amin")
        site, samp, tcs = site[first], samp[first], tcs[first]
        # order by site, then CPA time (needed to group passes that would explain the same sound)
        o = torch.argsort((tcs - tcs.min()) + site.double() * 1e7)
        return site[o], samp[o]

    def build_scorer(self):
        a = self.a
        J = int(round(a.half_w / self.ls.col_s))
        if a.mode == "full":
            self.scorer = FFTScorer(self.ls, J, self.max_db, a.ridge)
            self.bs = a.batch
        else:
            self.scorer = CandScorer(self.ls, J, self.max_db, self.c, a.wpow, a.n_cand, a.cand_rt, a.cand_rb,
                                     hints=self.hints if a.hint_prior is None else None, hint_dt=a.hint_dt,
                                     hint_pmin=a.hint_pmin,
                                     hints_only=a.hints_only, dr=a.ridge)
            self.bs = a.cand_batch
        self.build_hint_map()
        self.setup_r5b()
        self.build_comb()
        self.llr_hcol = None
        if self.a.llr and self.hints is not None and self.a.llr_hint:
            tt = self.ls.t0 + self.ls.col_s * np.arange(self.ls.nt)
            hd = np.zeros(self.ls.nt, bool)
            for h in self.hints:
                if float(h.get("p", 0.0)) >= self.a.hint_pmin:
                    hd |= np.abs(tt - float(h["t"])) <= self.a.hint_dt
            self.llr_hcol = torch.as_tensor(hd, device=DEV)

    def calibrate(self, G, site, samp):
        """z0 = percentile of the best time-reversed z of random passes (chance level of one pass)."""
        a = self.a
        if a.llr:
            return self.llr_calibrate(G, site, samp)
        g = torch.Generator(device="cpu").manual_seed(0)
        sub = torch.randperm(site.numel(), generator=g)[:min(site.numel(), a.calib_n)].to(DEV)
        W = self.S_max - self.S_min
        self.calib_win = W if (a.calib_win and a.mode == "cand" and HAVE_TRITON and DEV.type == "cuda"
                               and W < a.calib_win_max) else 0.0
        self._gcal = torch.Generator(device=DEV).manual_seed(1)
        if self.calib_win:
            log(f"window-aware calibration: clock window {W:.0f} s -> chance level of a pass within it")
        zr_best, zf_best = [], []
        cand = a.mode == "cand"
        zc = {False: [], True: []}
        for s0 in range(0, sub.numel(), self.bs):
            T = self.make_templates(G, site[sub[s0:s0 + self.bs]], samp[sub[s0:s0 + self.bs]], t_lo=-np.inf,
                                    t_hi=np.inf)
            for rev, lst in ((True, zr_best), (False, zf_best)):
                if cand:
                    z, _, br = self.scorer.zcand(T.db, T.valid, T.w, rev, self.grp_of(T))
                    st = self.stat_fn(T)
                    if st is not None:      # round 5b: chance level of the prior-adjusted statistic
                        z = st(z, br)
                    zc[rev].append(z)
                    if self.calib_win:
                        # clock known to +-W/2: a pass can only use spots near its CPA column at the true clock; emulate
                        # with a random clock that puts the CPA inside the recording
                        u = torch.rand(T.t_c.numel(), device=DEV, dtype=torch.float64, generator=self._gcal)
                        S_mid = T.t_c - self.ls.t0 - u * self.dur
                        hw = 0.5 * self.calib_win
                        z = self.scorer.zcand_win(T.db, T.valid, T.w, rev, T.t_c, S_mid - hw, S_mid + hw)[0]
                    lst.append(z.max(1).values)
                else:
                    lst.append(self.scorer.zmap(T.db, T.valid, T.w, rev)[0].max(1).values)
        zr_best = torch.cat(zr_best).cpu().numpy()
        zf_best = torch.cat(zf_best).cpu().numpy()
        self.z0 = float(a.z0) if a.z0 else float(np.percentile(zr_best, a.z0_pct))
        if cand and a.vote == "idf":
            # information of an explained spot: -log10(fraction of random passes that explain it)
            for rev in (False, True):
                n = (torch.cat(zc[rev]) >= self.z0).sum(0).float()
                N = float(sub.numel())
                self.scorer.idf[rev] = torch.log10((N + 1) / (n + 1))
            i_f = self.scorer.idf[False].cpu().numpy()
            log(f"spot information (-log10 frac of random passes explaining it): falling min {i_f.min():.2f} "
                f"median {np.median(i_f):.2f}; rising min {self.scorer.idf[True].min().item():.2f}")
        self.calib = dict(n=int(zr_best.size), rev_pct_50_90_99_999=np.percentile(zr_best, [50, 90, 99, 99.9]).round(2).tolist(),
                          fwd_pct_50_90_99_999=np.percentile(zf_best, [50, 90, 99, 99.9]).round(2).tolist())
        log(f"calibration on {zr_best.size} passes: best-z percentiles 50/90/99/99.9 reversed "
            f"{self.calib['rev_pct_50_90_99_999']} forward {self.calib['fwd_pct_50_90_99_999']} -> z0={self.z0:.2f}")

    def vote_idf(self, z, rev):
        """vote of a pass at a spot = information of the spot x soft step of (z - z0) over --zsoft."""
        return self.scorer.idf[rev][None, :] * ((z - self.z0) / self.a.zsoft).clamp(0, 1)

    # -- round 5b: per-type f0 prior, harmonic comb, calibrated per-pass log-likelihood ratio -------------------
    def setup_r5b(self):
        """Load the round-5b model (--llr / --f0-model JSON from eval/r5b/fit_llr.py) and build per-flight tables."""
        a = self.a
        self.r5 = None
        path = a.llr or a.f0_model
        if not path:
            return
        M = json.load(open(path))
        self.r5 = M
        fl = self.tr.fl
        nf = len(fl.cls)
        F = M["f0"]
        lf_tab = math.log(F["rows_f_lo"]) + F["step"] * np.arange(F["n_rows"])
        lf_ls = np.log(self.ls.lf)
        names, tabs = [], []
        for kind in ("type", "family", "group"):
            for k, v in F[kind].items():
                names.append((kind, k))
                tabs.append(np.interp(lf_ls, lf_tab, np.asarray(v, float)))
        idx = {n: i for i, n in enumerate(names)}
        groups = M["audibility"]["groups"]
        ti = np.zeros(nf, np.int64)
        gi = np.zeros(nf, np.int64)
        for f in range(nf):
            g = group_of(str(fl.cls[f]), str(fl.desc[f]))
            gi[f] = groups.index(g)
            t, d = str(fl.type[f]), str(fl.desc[f])
            ti[f] = idx.get(("type", t), idx.get(("family", d), idx[("group", g)]))
        self.ftab_t = torch.as_tensor(ti, device=DEV)
        self.fgrp_t = torch.as_tensor(gi, device=DEV)
        self.f0tab_t = torch.as_tensor(np.array(tabs), device=DEV, dtype=torch.float32).clamp(-a.f0_clip, a.f0_clip)
        A = M["audibility"]
        self.pi_t = torch.as_tensor(np.array([A["pi"][g] for g in groups]), device=DEV, dtype=torch.float32)
        self.pi_ld = (A["log10d"][0], A["log10d"][-1], len(A["log10d"]))
        self.pi_el = (A["elev"][0], A["elev"][-1], len(A["elev"]))
        log(f"r5b model {os.path.basename(path)}: {len(names)} f0 tables (kappa {a.f0_prior}), audibility groups "
            f"{groups}, llr scoring {'on' if a.llr else 'off'}")

    def stat_fn(self, T):
        """Per-spot statistic s = z + kappa * llr_f0(type of the pass, f0 row of the spot)."""
        if self.r5 is None or self.a.f0_prior <= 0:
            return None
        tab = self.ftab_t[T.fid]
        k = self.a.f0_prior
        nb = self.f0tab_t.shape[1]

        def f(z, brow):
            pr = self.f0tab_t[tab[:, None].expand_as(brow), brow.clamp(0, nb - 1)]
            return torch.where(z > -1e8, z + k * pr, z)
        return f

    def grp_of(self, T):
        return self.fgrp_t[T.fid] if (self.r5 is not None and getattr(self.scorer, "Lg", None)) else None

    def pi_of(self, T):
        """P(audible | group, d_cpa, elevation) by bilinear lookup in the fitted table, x --pi-scale."""
        l0, l1, nl = self.pi_ld
        e0, e1, ne = self.pi_el
        x = ((torch.log10(T.d_cpa.float().clamp(min=1.0)) - l0) / (l1 - l0) * (nl - 1)).clamp(0, nl - 1 - 1e-4)
        y = ((T.elev - e0) / (e1 - e0) * (ne - 1)).clamp(0, ne - 1 - 1e-4)
        i, j = x.floor().long(), y.floor().long()
        u, v = x - i, y - j
        g = self.fgrp_t[T.fid]
        P = self.pi_t
        p = ((1 - u) * (1 - v) * P[g, i, j] + u * (1 - v) * P[g, i + 1, j] + (1 - u) * v * P[g, i, j + 1]
             + u * v * P[g, i + 1, j + 1])
        if self.a.terrain_rho < 1.0:
            p = p * (1.0 - (1.0 - self.a.terrain_rho) * self.blocked(T))
        return (p * self.a.pi_scale).clamp(0.005, 0.98)

    # -- round 5b item 12: terrain line of sight ------------------------------------------------------------------
    def setup_terrain(self):
        """DEM heights on a regular lat/lon grid covering every CPA position the search can see (box + dmax), on
        the GPU, for the line-of-sight test (--terrain-rho < 1 or --pass-feats)."""
        a = self.a
        la0, la1, lo0, lo1 = self.bbox
        mla = a.dmax / 111132.0 + 0.02
        mlo = a.dmax / (111320.0 * math.cos(math.radians(0.5 * (la0 + la1)))) + 0.02
        la0, la1, lo0, lo1 = la0 - mla, la1 + mla, lo0 - mlo, lo1 + mlo
        step = a.terrain_step / 111132.0
        n_la = int(np.ceil((la1 - la0) / step)) + 1
        n_lo = int(np.ceil((lo1 - lo0) / (step / math.cos(math.radians(0.5 * (la0 + la1)))))) + 1
        lats = np.linspace(la0, la1, n_la)
        lons = np.linspace(lo0, lo1, n_lo)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        H = np.asarray(self.dem.sample(LA.ravel(), LO.ravel()), np.float32).reshape(n_la, n_lo)
        H = np.nan_to_num(H, nan=0.0)
        self.terr = dict(H=torch.as_tensor(H, device=DEV)[None, None], la0=la0, la1=la1, lo0=lo0, lo1=lo1)
        log(f"terrain grid {n_la}x{n_lo} at {a.terrain_step:.0f} m, heights {H.min():.0f}..{H.max():.0f} m")

    def clearance(self, T):
        """(P,) minimum height (m) of the observer -> aircraft-at-CPA line above the terrain, over the part of the
        line more than --terrain-skip m from the observer (the DEM cannot resolve the microphone's surroundings)."""
        if getattr(self, "terr", None) is None:
            self.setup_terrain()
        tr = self.terr
        n = 24
        L = torch.linalg.norm(T.cpa_vec, dim=1).clamp(min=1.0)
        s0 = (self.a.terrain_skip / L).clamp(max=0.5)
        s = s0[:, None] + (0.98 - s0[:, None]) * torch.linspace(0, 1, n, device=DEV, dtype=torch.float64)[None, :]
        X = T.obs[:, None, :] + s[..., None] * T.cpa_vec[:, None, :]                       # (P, n, 3) ECEF
        lat, lon, h = ecef_to_geodetic_t(X)
        gx = ((lon - tr["lo0"]) / (tr["lo1"] - tr["lo0"]) * 2 - 1).float()
        gy = ((lat - tr["la0"]) / (tr["la1"] - tr["la0"]) * 2 - 1).float()
        g = torch.stack([gx, gy], -1)[None]                                                    # (1, P, n, 2)
        Ht = torch.nn.functional.grid_sample(tr["H"], g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
        return (h.float() - Ht).min(1).values

    def blocked(self, T):
        """(P,) soft terrain blocking in [0, 1]: sigmoid(-clearance / --terrain-soft)."""
        return torch.sigmoid(-self.clearance(T) / self.a.terrain_soft)

    def build_comb(self):
        """--comb 'piston:1,1.5,2;heli:1,1.333,1.667,2': per-group harmonic-comb images (mean of the log spectrogram
        at f0 x each multiplier; zero above F_HI / below F_LO), scored with the same ridge contrast."""
        a = self.a
        if not a.comb or self.r5 is None:
            return
        groups = self.r5["audibility"]["groups"]
        L = self.ls.L
        nb = L.shape[0]
        Lg = {}
        for part in a.comb.split(";"):
            g, mults = part.split(":")
            acc = torch.zeros_like(L)
            cnt = torch.zeros(nb, 1, device=DEV)
            for m in (float(x) for x in mults.split(",")):
                sh = int(round(math.log(m) / self.ls.log_step))
                if sh >= 0:
                    acc[:nb - sh] += L[sh:]
                    cnt[:nb - sh] += 1
                else:
                    acc[-sh:] += L[:nb + sh]
                    cnt[-sh:] += 1
            Lg[groups.index(g)] = (acc / cnt.clamp(min=1)).contiguous()
        self.scorer.Lg = Lg
        log(f"harmonic comb images for groups {[groups[k] for k in Lg]}")

    def llr_calibrate(self, G, site, samp):
        """Per-recording chance distribution of the per-column statistic (random passes at random sites, every
        column, dilated +-DIL), forward and time-reversed; -> log LR tables against the fitted audible
        distribution p_a(s) (model JSON), the miss floor and z0 (99th pct of the best reversed statistic)."""
        a = self.a
        M = self.r5["llr"]
        edges = torch.as_tensor(M["s_edges"], device=DEV, dtype=torch.float32)      # bin k: [e_k, e_k+1)
        s_floor = float(M["s_floor"])
        nbin = edges.numel() - 1
        g = torch.Generator(device="cpu").manual_seed(0)
        sub = torch.randperm(site.numel(), generator=g)[:min(site.numel(), a.llr_calib_n)].to(DEV)
        H = {False: torch.zeros(nbin + 2, device=DEV, dtype=torch.float64),
             True: torch.zeros(nbin + 2, device=DEV, dtype=torch.float64)}
        best_r = []
        for s0 in range(0, sub.numel(), self.bs):
            q = sub[s0:s0 + self.bs]
            T = self.make_templates(G, site[q], samp[q], t_lo=-np.inf, t_hi=np.inf)
            st = self.stat_fn(T)
            for rev in ((False, True) if a.llr_null else (False,)):
                Z, _ = self.scorer.zmap(T.db, T.valid, T.w, rev, stat=st, grp=self.grp_of(T))
                if rev:
                    best_r.append(Z.max(1).values)
                Z = torch.nn.functional.max_pool1d(Z[:, None], 2 * DIL + 1, 1, DIL)[:, 0]
                # bin 0: <= s_floor (incl. no spot); 1..nbin: edges; nbin+1: above the last edge
                k = torch.bucketize(Z.reshape(-1).clamp(min=s_floor), edges, right=True)
                H[rev] += torch.bincount(k, minlength=nbin + 2)[:nbin + 2].double()
        self.z0 = float(np.percentile(torch.cat(best_r).cpu().numpy(), a.z0_pct)) if best_r else 0.0
        pa = np.asarray(M["pa_bins"], float)                # audible: [P(<= floor), density mass per bin ..., tail]
        self.llr_tab = {}
        for rev in ((False, True) if a.llr_null else (False,)):
            c = H[rev].cpu().numpy()
            N = c.sum()
            # survival of the chance statistic on the bin edges, log-linear tail extrapolation where counts are thin
            sv = np.cumsum(c[::-1])[::-1] / N                 # sv[k] = P(bin >= k)
            lsv = np.log(np.maximum(sv, 1e-300))
            ok = np.flatnonzero(np.cumsum(c[::-1])[::-1] >= 30)
            last = int(ok.max()) if ok.size else 1
            lo = max(1, last - 8)
            slope = min(-0.02, np.polyfit(np.arange(lo, last + 1), lsv[lo:last + 1], 1)[0]) if last > lo else -0.3
            kk = np.arange(c.size)
            lsv = np.where(kk > last, lsv[last] + slope * (kk - last), lsv)
            lsv = np.minimum.accumulate(lsv)
            pc = np.exp(lsv[:-1]) - np.exp(lsv[1:])               # mass per bin 0..nbin
            pc = np.r_[pc, np.exp(lsv[-1])]
            pc = np.maximum(pc, np.exp(lsv) * (1 - np.exp(slope)) * 1e-3)
            lr = np.log(np.maximum(pa, 1e-12)) - np.log(np.maximum(pc, 1e-300))
            lr = np.maximum.accumulate(np.minimum(lr, a.llr_cap))
            self.llr_tab[rev] = torch.as_tensor(lr, device=DEV, dtype=torch.float32)
        self.llr_edges, self.llr_floor = edges, s_floor
        self.hint_r = float(M.get("hint_r", 0.0))
        lt = self.llr_tab[False].cpu().numpy()
        log(f"llr calibration on {sub.numel()} passes: z0 {self.z0:.2f}; log LR at s = floor {lt[0]:.2f}, "
            + ", ".join(f"{float(e):.0f}:{lt[i + 1]:.1f}" for i, e in enumerate(M['s_edges'][:-1]) if i % 8 == 0))
        self.calib = dict(n=int(sub.numel()), llr=lt.round(2).tolist())

    def hint_terms(self):
        """(log H_present, log H_absent) of a detector hint at a pass's CPA column: P(hint | audible) = r against the
        recording's hint coverage h0 (the chance of a hint at a random column)."""
        if self.llr_hcol is None or self.hint_r <= 0:
            return None
        h0 = float(self.llr_hcol.float().mean().clamp(1e-3, 0.5))
        r = min(max(self.hint_r, h0 + 1e-3), 0.995)
        return math.log(r / h0), math.log((1 - r) / (1 - h0))

    def zvotes_llr(self, T, rev):
        """Dense (P, nt) LLR votes (dilated +-DIL) and the per-pass miss cost:
        term(c) = log(1 - pi + pi * LR(s_c) * H(c)); base = term at the floor without a hint (<= 0, charged to every
        pass inside the recording at clock S); vote(c) = term(c) - base >= 0."""
        Z, _ = self.scorer.zmap(T.db, T.valid, T.w, rev, stat=self.stat_fn(T), grp=self.grp_of(T))
        Z = torch.nn.functional.max_pool1d(Z[:, None], 2 * DIL + 1, 1, DIL)[:, 0]
        k = torch.bucketize(Z.clamp(min=self.llr_floor), self.llr_edges, right=True)
        lr = self.llr_tab[rev][k.clamp(max=self.llr_tab[rev].numel() - 1)]
        pi = self.pi_of(T)[:, None]
        ht = None if rev else self.hint_terms()
        if ht is not None:
            lr = lr + torch.where(self.llr_hcol[None, :], ht[0], ht[1])
            lfl = self.llr_tab[rev][0] + ht[1]
        else:
            lfl = self.llr_tab[rev][0]
        term = torch.log1p(-pi + pi * torch.exp(lr))
        base = torch.log1p(-pi + pi * torch.exp(lfl))                             # (P, 1)
        v = (term - base).clamp(min=0)
        if self.a.slack_s > DIL:
            # round 5b item 11: per-aircraft time slack - a pass may explain a ridge up to --slack-s s from its
            # predicted CPA (ADS-B timestamp offsets, e.g. the 7 s PA34), at --slack-cost nats per second beyond DIL
            K = int(round(self.a.slack_s / self.ls.col_s))
            k0 = int(round(DIL / self.ls.col_s))
            vs = v
            for k in range(k0 + 1, K + 1):
                pen = self.a.slack_cost * (k * self.ls.col_s - DIL)
                vs = torch.maximum(vs, torch.nn.functional.pad(v[:, k:], (0, k)) - pen)
                vs = torch.maximum(vs, torch.nn.functional.pad(v[:, :-k], (k, 0)) - pen)
            v = vs.clamp(min=0)
        v = torch.where(v >= self.a.llr_eps, v, torch.zeros_like(v))
        if not self.a.miss_cost:
            base = torch.zeros_like(base)
        return v, base[:, 0]

    def score_grid_llr(self, G, site, samp, S_lo, S_hi, progress=False):
        """score(site, S) = sum of LLR votes of the passes at S (one sound = one vote: groups of passes whose CPAs
        are within --excl-s take the max) + sum of miss costs of the passes inside the recording at S. score2 needs
        >= --min-voters passes with vote >= --llr-vmin at S, score1 >= 1; --clock-marg: log-sum-exp over S."""
        a = self.a
        Ns = G["LA"].size
        nS = int(np.ceil(S_hi - S_lo)) + 1
        nsb = max(1, min(Ns, int(a.site_batch_bytes // (4 * nS))))
        counts = torch.bincount(site, minlength=Ns)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), torch.cumsum(counts, 0)])
        res = {k: np.full(Ns, NEG, np.float32) for k in ("score2", "score1", "null2", "null1")}
        res["n2"] = np.zeros(Ns, np.float32)
        res["score0"] = np.zeros(Ns, np.float32)     # best LLR over S without any voter requirement (multi-session)
        res["n0"] = np.zeros(Ns, np.float32)
        resS = {k: np.full(Ns, S_lo, np.float64) for k in ("S2", "S1", "nS2")}
        for b_i, sb in enumerate(range(0, Ns, nsb)):
            se = min(Ns, sb + nsb)
            p0, p1 = int(starts[sb]), int(starts[se])
            if p1 == p0:
                continue
            acc = {k: torch.zeros(se - sb, nS + (1 if k in ("fe", "re") else 0), device=DEV)
                   for k in ("f", "fm", "fn", "r", "rm", "rn", "fe", "re")}
            for q in range(p0, p1, self.bs):
                q1 = min(p1, q + self.bs)
                T = self.make_templates(G, site[q:q1], samp[q:q1], t_lo=-np.inf, t_hi=np.inf)
                lead, shift = self.groups(T)
                for rev in ((False, True) if a.llr_null else (False,)):
                    v, base = self.zvotes_llr(T, rev)
                    w, Tl, gbase = self.exclusive(v, T, lead, shift, base)
                    kk = "r" if rev else "f"
                    self.scatter(acc[kk], acc[kk + "m"], acc[kk + "n"], Tl, w, sb, S_lo, nS, vmin=a.llr_vmin)
                    self.cost(acc[kk + "e"], Tl, gbase, sb, S_lo, nS)
            for key, out2, out1, s2, s1 in (("f", "score2", "score1", "S2", "S1"), ("r", "null2", "null1", "nS2", None)):
                A = acc[key] + torch.cumsum(acc[key + "e"], 1)[:, :nS]
                Nv = acc[key + "n"]
                if key == "f":
                    v0, i0 = A.max(1)
                    res["score0"][sb:se] = v0.cpu().numpy()
                    res["n0"][sb:se] = Nv.gather(1, i0[:, None])[:, 0].cpu().numpy()
                for out, need, sk in ((out2, a.min_voters, s2), (out1, 1, s1)):
                    X = torch.where(Nv >= need, A, torch.full_like(A, NEG))
                    vb, ib = X.max(1)
                    if a.clock_marg:
                        vm = torch.logsumexp(X, 1)
                        vb = torch.where(vb > NEG / 2, vm, vb)
                    res[out][sb:se] = vb.cpu().numpy()
                    if sk:
                        resS[sk][sb:se] = S_lo + ib.cpu().numpy()
                    if key == "f" and out == out2:
                        res["n2"][sb:se] = Nv.gather(1, ib[:, None])[:, 0].cpu().numpy()
            if progress and b_i % 10 == 0:
                log(f"  sites {se}/{Ns}, passes {p1}/{site.numel()}")
        return res, resS

    def cost(self, E, T, base, sb, S_lo, nS):
        """E (difference array over S) += base for every pass (group) whose CPA lies inside the recording
        (+-margin) at clock S."""
        nt = self.ls.nt
        mg = self.a.silence_margin
        tc = T.t_c - self.ls.t0 - S_lo
        lo = torch.ceil(tc - (nt - 1) * self.ls.col_s + mg).long().clamp(0, nS)
        hi = (torch.floor(tc - mg).long() + 1).clamp(0, nS)
        ok = (hi > lo) & (base < 0)
        rows = (T.site - sb)[ok]
        E.index_put_((rows, lo[ok]), base[ok].to(E.dtype), accumulate=True)
        E.index_put_((rows, hi[ok]), -base[ok].to(E.dtype), accumulate=True)

    def zvotes(self, T, rev, S_win=None):
        """Dense (P, nt) votes, dilated +-DIL columns. S_win = (S_lo, S_hi): only clock-consistent spots (cand mode,
        windows <= --win-max s; identical votes inside the window, faster)."""
        kw = {}
        if (S_win is not None and self.a.mode == "cand" and self.a.win_max > 0
                and S_win[1] - S_win[0] <= self.a.win_max):
            kw["win"] = (T.t_c, S_win[0], S_win[1])
        if self.a.mode == "cand" and self.a.vote == "idf":
            w, R = self.scorer.zmap(T.db, T.valid, T.w, rev, vote=self.vote_idf, **kw)
        else:
            z, R = self.scorer.zmap(T.db, T.valid, T.w, rev, stat=self.stat_fn(T), grp=self.grp_of(T), **kw)
            if self.hints is not None and self.a.mode == "full" and not rev and self.a.hint_prior is None:
                z = torch.where(self.hint_mask(z.shape, None), z, torch.zeros_like(z))
            vp = self.a.vpow if self.a.vote == "pow" else 1.0
            w = (z - self.z0).clamp(0, self.a.vcap) ** vp
            if self.a.tie_z > 0:
                # above the cap, a better curve fit still counts a little (breaks ties between sites whose votes
                # all saturate; the cap still stops one loud pass from outvoting several)
                w = w + self.a.tie_z * (z - self.z0 - self.a.vcap).clamp(min=0) ** vp
        if self.hint_map is not None and not rev:
            # soft detector prior: vote x (w0 + (1 - w0) * p of a detection at this CPA column (and f0 row))
            cols = torch.arange(w.shape[1], device=DEV)[None, :].expand_as(R)
            h = self.hint_map[R.clamp(0, self.ls.nb - 1), cols]
            w = w * (self.a.hint_prior + (1 - self.a.hint_prior) * h)
        return torch.nn.functional.max_pool1d(w[:, None], 2 * DIL + 1, 1, DIL)[:, 0]

    def build_hint_map(self):
        """(nb, nt) max detector p around each hint: +-hint_dt s; rows within +-hint_df (log) of the hint frequency or
        of its 1/3, 1/2, 2, 3 multiples (the detector and the curve match may pick different harmonics); hint_df <= 0
        -> time only."""
        a = self.a
        self.hint_map = None
        if self.hints is None or a.hint_prior is None:
            return
        ls = self.ls
        H = torch.zeros(ls.nb, ls.nt, device=DEV)
        tt = ls.t0 + ls.col_s * torch.arange(ls.nt, device=DEV, dtype=torch.float64)
        lf = torch.as_tensor(np.log(ls.lf), device=DEV)
        n = 0
        for h in self.hints:
            p = float(h.get(a.hint_key, h.get("p", 1.0)))
            if p < a.hint_pmin:
                continue
            n += 1
            mt = (tt - h["t"]).abs() <= a.hint_dt
            if a.hint_df > 0:
                mb = torch.zeros(ls.nb, dtype=torch.bool, device=DEV)
                for k in (1 / 3, 0.5, 1.0, 2.0, 3.0):
                    mb |= (lf - math.log(h["f"] * k)).abs() <= a.hint_df
            else:
                mb = torch.ones(ls.nb, dtype=torch.bool, device=DEV)
            H = torch.maximum(H, (mb[:, None] & mt[None, :]).float() * p)
        self.hint_map = H
        log(f"hint prior: {n} detections with {a.hint_key} >= {a.hint_pmin}, unsupported votes x {a.hint_prior}, "
            f"{float((H > 0).float().mean()):.3f} of the (f0, t) plane supported")



    def score_grid(self, G, site, samp, S_lo, S_hi, progress=False):
        """Per site: best summed vote over recording starts S in [S_lo, S_hi] with >= 2 voting passes (score2), the
        same with a single pass allowed (score1), and both for time-reversed templates (null2, null1)."""
        a = self.a
        if a.llr:
            return self.score_grid_llr(G, site, samp, S_lo, S_hi, progress)
        Ns = G["LA"].size
        nS = int(np.ceil(S_hi - S_lo)) + 1
        nsb = max(1, min(Ns, int(a.site_batch_bytes // (4 * nS))))
        counts = torch.bincount(site, minlength=Ns)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), torch.cumsum(counts, 0)])
        res = {k: np.zeros(Ns, np.float32) for k in ("score2", "score1", "null2", "null1", "n2")}
        resS = {k: np.full(Ns, S_lo, np.float64) for k in ("S2", "S1", "nS2")}
        for b_i, sb in enumerate(range(0, Ns, nsb)):
            se = min(Ns, sb + nsb)
            p0, p1 = int(starts[sb]), int(starts[se])
            if p1 == p0:
                continue
            keys = ["f", "fm", "r", "rm", "fn"] + (["fe", "fv", "re", "rv"] if a.silence_pen > 0 else [])
            acc = {k: torch.zeros(se - sb, nS + (1 if k in ("fe", "re") else 0), device=DEV) for k in keys}
            for q in range(p0, p1, self.bs):
                q1 = min(p1, q + self.bs)
                T = self.make_templates(G, site[q:q1], samp[q:q1], t_lo=-np.inf, t_hi=np.inf)
                lead, shift = self.groups(T)
                for rev in (False, True):
                    v = self.zvotes(T, rev, (S_lo, S_hi))
                    w, Tl = self.exclusive(v, T, lead, shift)
                    self.scatter(acc["r" if rev else "f"], acc["rm" if rev else "fm"],
                                 None if rev else acc["fn"], Tl, w, sb, S_lo, nS)
                    if a.silence_pen > 0:
                        self.silence(acc["re" if rev else "fe"], acc["rv" if rev else "fv"], T, v, sb, S_lo, nS)
            for key, mk, out2, out1, s2, s1 in (("f", "fm", "score2", "score1", "S2", "S1"),
                                                 ("r", "rm", "null2", "null1", "nS2", None)):
                A, M = acc[key], acc[mk]
                two = torch.where(A - M > 1e-6, A, torch.zeros_like(A))
                if a.silence_pen > 0:
                    # expected-audible close passes inside the recording at clock S that do not vote
                    E = torch.cumsum(acc[key + "e"], 1)[:, :nS]
                    silent = (E - acc[key + "v"]).clamp(min=0)
                    two = torch.where(two > 0, (two - a.silence_pen * silent).clamp(min=1e-3), two)
                    A = torch.where(A > 0, (A - a.silence_pen * silent).clamp(min=1e-3), A)
                v2, i2 = two.max(1)
                v1, i1 = A.max(1)
                res[out2][sb:se] = v2.cpu().numpy()
                res[out1][sb:se] = v1.cpu().numpy()
                resS[s2][sb:se] = S_lo + i2.cpu().numpy()
                if s1:
                    resS[s1][sb:se] = S_lo + i1.cpu().numpy()
                if key == "f":
                    res["n2"][sb:se] = acc["fn"].gather(1, i2[:, None])[:, 0].cpu().numpy()
            if progress and b_i % 10 == 0:
                log(f"  sites {se}/{Ns}, passes {p1}/{site.numel()}")
        return res, resS

    def silence(self, E, V, T, v, sb, S_lo, nS):
        """Silence evidence: E (difference array over S) += 1 for every close (d_cpa < --silence-d) prop/rotor pass
        whose CPA lies inside the recording (+-margin) at clock S; V += 1 where that pass votes (v > 0) at S."""
        m = (T.d_cpa < self.a.silence_d) & self.tr.loud_t[T.fid]
        if self.a.silence_cover:
            m &= self.covered(T)
        if not bool(m.any()):
            return
        nt = self.ls.nt
        mg = self.a.silence_margin
        tc = T.t_c[m] - self.ls.t0 - S_lo
        lo = torch.ceil(tc - (nt - 1) * self.ls.col_s + mg).long().clamp(0, nS)
        hi = (torch.floor(tc - mg).long() + 1).clamp(0, nS)
        ok = hi > lo
        rows = (T.site[m] - sb)[ok]
        E.index_put_((rows, lo[ok]), torch.ones_like(rows, dtype=E.dtype), accumulate=True)
        E.index_put_((rows, hi[ok]), -torch.ones_like(rows, dtype=E.dtype), accumulate=True)
        vv = v[m] > 0
        if self.heard is not None:
            vv = vv | self.heard[None, :]          # the detector heard something near this CPA: not silent
        jt = torch.arange(nt, device=DEV, dtype=torch.float64) * self.ls.col_s
        idx = torch.round(tc[:, None] - jt[None, :]).long()
        okv = vv & (idx >= lo[:, None]) & (idx < hi[:, None])
        flat = ((T.site[m] - sb)[:, None] * nS + idx.clamp(0, nS - 1))[okv]
        V.view(-1).index_put_((flat,), torch.ones_like(flat, dtype=V.dtype), accumulate=True)

    def covered(self, T):
        """Passes the straight-pass curve model can be trusted on (--silence-cover): the flight is tracked over
        +-cover_w s of the CPA (not a partial pass), raw ADS-B points are never more than --cover-gap s apart there
        (no interpolated stretch), heading changes by < --cover-turn deg and speed by < --cover-dv (relative)."""
        a, tr = self.a, self.tr
        cw = int(round(a.cover_w))
        f0, f1 = tr.F0_t[tr.fid_t[T.samp]], tr.F1_t[tr.fid_t[T.samp]]
        ok = (T.samp - cw >= f0) & (T.samp + cw <= f1)
        i0 = (T.samp - cw).clamp(min=0)
        i1 = (T.samp + cw).clamp(max=tr.vel.shape[0] - 1)
        v0, v1 = tr.vel[i0], tr.vel[i1]
        n0, n1 = v0.norm(dim=1).clamp(min=1e-3), v1.norm(dim=1).clamp(min=1e-3)
        ang = torch.rad2deg(torch.arccos(((v0 * v1).sum(1) / (n0 * n1)).clamp(-1, 1)))
        ok &= (ang < a.cover_turn) & ((n1 - n0).abs() / (0.5 * (n0 + n1)) < a.cover_dv)
        jj = torch.arange(-cw, cw + 1, device=DEV)
        ix = (T.samp[:, None] + jj[None, :]).clamp(0, tr.rawgap_t.numel() - 1)
        ok &= tr.rawgap_t[ix].amax(1) < a.cover_gap
        return ok

    def groups(self, T):
        """Passes at the same site whose CPAs are within EXCL_S of each other (chained) would explain the same
        sound at the same clock: they form one group whose vote is the max, not the sum."""
        P = T.t_c.numel()
        new = torch.ones(P, dtype=torch.bool, device=DEV)
        if P > 1:
            new[1:] = (T.site[1:] != T.site[:-1]) | ((T.t_c[1:] - T.t_c[:-1]).abs() > self.a.excl_s)
        gid = torch.cumsum(new.long(), 0) - 1
        lead = torch.nonzero(new, as_tuple=True)[0]
        shift = torch.round((T.t_c - T.t_c[lead[gid]]) / self.ls.col_s).long()
        return (gid, lead), shift

    def exclusive(self, w, T, lead, shift, base=None):
        gid, lead_idx = lead
        if lead_idx.numel() == w.shape[0]:
            return (w, T) if base is None else (w, T, base)
        nt = w.shape[1]
        j = torch.arange(nt, device=DEV)[None, :] + shift[:, None]
        ok = (j >= 0) & (j < nt)
        wa = torch.where(ok, w.gather(1, j.clamp(0, nt - 1)), torch.zeros_like(w))
        out = torch.zeros(lead_idx.numel(), nt, device=DEV)
        out.scatter_reduce_(0, gid[:, None].expand(-1, nt), wa, "amax")

        class _L:
            pass
        Tl = _L()
        Tl.t_c, Tl.site = T.t_c[lead_idx], T.site[lead_idx]
        if base is None:
            return out, Tl
        # one sound: the group misses only as one pass (its least costly member)
        gb = torch.full((lead_idx.numel(),), -1e30, device=DEV)
        gb.scatter_reduce_(0, gid, base, "amax")
        return out, Tl, gb

    def scatter(self, A, M, N, T, w, sb, S_lo, nS, vmin=None):
        nt = w.shape[1]
        jt = torch.arange(nt, device=DEV, dtype=torch.float64) * self.ls.col_s
        idx = torch.round(T.t_c[:, None] - self.ls.t0 - S_lo - jt[None, :]).long()
        ok = (idx >= 0) & (idx < nS) & (w > 0)
        flat = ((T.site - sb)[:, None] * nS + idx.clamp(0, nS - 1))[ok]
        A.view(-1).index_add_(0, flat, w[ok])
        M.view(-1).scatter_reduce_(0, flat, w[ok], "amax")
        if N is not None:
            N.view(-1).index_add_(0, flat, torch.ones_like(w[ok]) if vmin is None else (w[ok] >= vmin).float())

    # -- explain one hypothesis ----------------------------------------------------------------------------------
    def explain(self, G, sidx, S, site_all, samp_all):
        """Per-pass z and vote at the hypothesis clock S (+-DIL) for every pass at site index sidx."""
        m = site_all == sidx
        if not m.any():
            return []
        T = self.make_templates(G, site_all[m], samp_all[m], t_lo=-np.inf, t_hi=np.inf)
        z, arg = self.scorer.zmap(T.db, T.valid, T.w, False)
        zr, _ = self.scorer.zmap(T.db, T.valid, T.w, True)
        if self.a.llr:
            V, base = self.zvotes_llr(T, False)
        else:
            V = self.zvotes(T, False)
            base = torch.zeros(T.fid.numel(), device=DEV)
        cov = self.covered(T)
        feats = self.a.pass_feats or self.a.env_llr
        if feats and self.r5 is not None:
            pi_all = self.pi_of(T).cpu().numpy()
            clr = self.clearance(T).cpu().numpy() if self.a.pass_feats else None
        else:
            pi_all, clr = None, None
        if self.a.pass_feats:
            # aircraft position and ground track at CPA (single-event mode: the mirror point across the track)
            ac = T.obs + T.cpa_vec
            alat, alon, _ = ecef_to_geodetic_t(ac)
            la_r, lo_r = torch.deg2rad(alat), torch.deg2rad(alon)
            ve = (-torch.sin(lo_r) * T.vel_cpa[:, 0] + torch.cos(lo_r) * T.vel_cpa[:, 1])
            vn = (-torch.sin(la_r) * torch.cos(lo_r) * T.vel_cpa[:, 0] - torch.sin(la_r) * torch.sin(lo_r) * T.vel_cpa[:, 1]
                  + torch.cos(la_r) * T.vel_cpa[:, 2])
            hdg = torch.rad2deg(torch.atan2(ve, vn)) % 360
            alat, alon, hdg = alat.cpu().numpy(), alon.cpu().numpy(), hdg.cpu().numpy()
        out = []
        fl = self.tr.fl
        for i in range(z.shape[0]):
            c0 = int(round((T.t_c[i].item() - S - self.ls.t0) / self.ls.col_s))
            lo, hi = max(0, c0 - DIL), min(z.shape[1], c0 + DIL + 1)
            if hi <= lo:
                continue
            j = lo + int(torch.argmax(z[i, lo:hi]))
            f = int(T.fid[i])
            zz = float(z[i, j])
            out.append(dict(fid=f, hex=str(fl.hex[f]), type=str(fl.type[f]), reg=str(fl.reg[f]), cls=str(fl.cls[f]),
                            d_cpa_m=round(float(T.d_cpa[i]), 0), v_cpa_ms=round(float(T.v_cpa[i]), 1),
                            t_cpa_rec=round(self.ls.t0 + c0 * self.ls.col_s, 1), t_cpa_utc=iso(T.t_c[i].item()),
                            z=round(zz, 2), z_rev_best=round(float(zr[i].max()), 2),
                            vote=round(float(V[i, c0] if 0 <= c0 < V.shape[1] else 0.0), 2),
                            miss_cost=round(float(base[i]), 3),
                            covered=bool(cov[i]),
                            f0=round(float(self.ls.f0_of_row(int(arg[i, j]))), 1) if zz != 0 else None))
            if feats:
                d = out[-1]
                if pi_all is not None:
                    d["pi"] = round(float(pi_all[i]), 3)
                if clr is not None:
                    d["clear_m"] = round(float(clr[i]), 0)
                if self.a.pass_feats:
                    d.update(ac_lat=round(float(alat[i]), 6), ac_lon=round(float(alon[i]), 6), ac_hdg=round(float(hdg[i]), 1))
                if zz != 0:
                    d.update(self.envelope(T, i, j, int(arg[i, j])))
        out.sort(key=lambda d: -d["z"])
        return out

    def envelope(self, T, i, j, row0):
        """Round 5b item 10: amplitude envelope of pass i whose ridge is at CPA column j, f0 row row0.
        Observed: ridge contrast along the template (L at the curve minus the mean of +-dr rows) per column.
        Expected for a point source: level(t) = -10 log10(1 + (v t / d)^2) dB (spherical spreading, no directivity),
        half-power width 2 d / v. Returns env_r (Pearson correlation of observed and expected over the valid
        columns), env_sd (std of the expected shape, dB: how much the shape can say; ~0 for distant slow passes),
        env_pk (mean contrast within +-max(4 s, d / 2v) of CPA) and env_n (valid columns)."""
        L = self.ls.L
        nb, nt = L.shape
        J = T.J
        K = 2 * J + 1
        dr = max(2, int(self.a.ridge or 4))
        k = torch.arange(K, device=DEV)
        cols = j - J + k
        rows = row0 + T.db[i]
        ok = T.valid[i] & (cols >= 0) & (cols < nt) & (rows - dr >= 0) & (rows + dr < nb)
        if int(ok.sum()) < 8:
            return dict(env_r=None, env_sd=0.0, env_pk=None, env_n=int(ok.sum()))
        r_, c_ = rows[ok], cols[ok]
        obs = (L[r_, c_] - 0.5 * (L[r_ + dr, c_] + L[r_ - dr, c_])).double()
        tau = (k[ok].double() - J) * self.ls.col_s
        d = float(T.d_cpa[i])
        v = max(float(T.v_cpa[i]), 1.0)
        e = -10 * torch.log10(1 + (v * tau / d) ** 2)
        es, os_ = e.std(), obs.std()
        rr = float(((e - e.mean()) * (obs - obs.mean())).mean() / (es * os_)) if es > 1e-6 and os_ > 1e-9 else 0.0
        near = tau.abs() <= max(4.0, 0.5 * d / v)
        return dict(env_r=round(rr, 3), env_sd=round(float(es), 2),
                    env_pk=round(float(obs[near].mean()), 4) if bool(near.any()) else None, env_n=int(ok.sum()))

    # -- refinement: fine grid, clock locked near the hypothesis ---------------------------------------------------
    def refine(self, lat0, lon0, S0, r, spacing, dt_s):
        dlat = r / 111132.0
        dlon = r / (111320.0 * math.cos(math.radians(lat0)))
        G = self.setup_grid((lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon), spacing)
        S_lo, S_hi = S0 - dt_s, S0 + dt_s
        site, samp = self.passes_for_grid(G, S_lo, S_hi)
        if site.numel() == 0:
            return None
        res, resS = self.score_grid(G, site, samp, S_lo, S_hi)
        j = int(np.argmax(res["score2"]))
        S = float(resS["S2"][j])
        passes = self.explain(G, j, S, site, samp)
        return dict(lat=float(G["LA"][j]), lon=float(G["LO"][j]), start_utc=S, start_iso=iso(S),
                    score=float(res["score2"][j]), null_max=float(res["null2"].max()),
                    n_voting=int(sum(p["vote"] >= vmin_of(self.a) for p in passes)), passes=passes[:12 if not self.a.llr else self.a.pass_keep],
                    _heat=(res["score2"].reshape(G["lats"].size, G["lons"].size), G["lats"], G["lons"]),
                    _plot=(G, j, site, samp))


HARM = np.log([1.0, 2.0, 0.5, 1.5, 1 / 1.5, 3.0, 1 / 3.0])


def vmin_of(a):
    return a.llr_vmin if getattr(a, "llr", None) else 1e-9


def rescore(h, a, hints=None):
    """Final consistency check of one hypothesis' explanation (applied identically to every hypothesis):
    one sound = one vote (voters within --same-dt s and --same-dlogf of a stronger voter are dropped), and one
    aircraft = one propeller frequency (a weaker voter of an aircraft already used must have f0 within --hex-tol
    (log) of the same or a simple harmonic ratio). Removed votes are subtracted from the score; < 2 voters left -> 0."""
    vmin = vmin_of(a)
    voters = sorted((p for p in h.get("passes", []) if p["vote"] >= vmin and p.get("f0")), key=lambda p: -p["vote"])
    kept, removed = [], 0.0
    for p in voters:
        lf = math.log(p["f0"])
        same_sound = any(abs(p["t_cpa_rec"] - q["t_cpa_rec"]) < a.same_dt and abs(lf - math.log(q["f0"])) < a.same_dlogf
                         for q in kept)
        bad_hex = any(q["hex"] == p["hex"] and np.abs(lf - math.log(q["f0"]) - HARM).min() > a.hex_tol for q in kept)
        if same_sound or bad_hex:
            removed += p["vote"]
            p["dropped"] = "same_sound" if same_sound else "hex_f0"
        else:
            kept.append(p)
    h["score_raw"] = h["score"]
    h["n_voting"] = len(kept)
    if a.llr:
        if h["score"] <= NEG / 2:
            return
        h["score"] = h["score"] - removed if len(kept) >= a.min_voters else NEG
        if a.env_llr and h["score"] > NEG / 2:
            h["env_llr"] = round(sum(env_term(p, a) for p in kept), 3)
            h["score"] += a.env_w * h["env_llr"]
        if a.unexpl_cost and hints:
            un = sum(1 for t in hints if not any(abs(t - q["t_cpa_rec"]) <= a.hint_dt + DIL for q in kept))
            h["n_unexplained"] = un
            if h["score"] > NEG / 2:
                h["score"] -= a.unexpl_cost * un
        return
    h["score"] = max(0.0, h["score"] - removed) if len(kept) >= 2 else 0.0


_ENV = {}


def env_term(p, a):
    """Round 5b item 10: log LR of a voter's envelope correlation env_r, from the table fitted on dev
    (eval/r5b/fit_env.py: true voters at the true site/clock against voters of wrong-site hypotheses), binned by
    env_sd (how informative the shape is); clipped to +-env_clip; 0 without features."""
    if p.get("env_r") is None:
        return 0.0
    if "M" not in _ENV:
        _ENV["M"] = json.load(open(a.env_llr))
    M = _ENV["M"]
    sd_e, r_e = M["sd_edges"], M["r_edges"]
    i = int(np.clip(np.searchsorted(sd_e, p.get("env_sd", 0.0), side="right") - 1, 0, len(sd_e) - 2))
    k = int(np.clip(np.searchsorted(r_e, p["env_r"], side="right") - 1, 0, len(r_e) - 2))
    return float(np.clip(M["llr"][i][k], -a.env_clip, a.env_clip))


NEG_MODE = [False]


def nms_hypotheses(res, resS, G, n, min_km, key="score2", skey="S2"):
    sc = res[key]
    order = np.argsort(-sc)
    picked = []
    for i in order:
        if sc[i] <= (NEG / 2 if NEG_MODE[0] else 0) or len(picked) >= n:
            break
        ok = True
        for j in picked:
            dkm = haversine_km(G["LA"][i], G["LO"][i], G["LA"][j], G["LO"][j])
            if dkm < min_km and abs(resS[skey][i] - resS[skey][j]) < 30:
                ok = False
                break
        if ok:
            picked.append(i)
    return picked


def extra_spots(res, resS, G, picked, a, key="score2", skey="S2"):
    """--spots K: besides the best coarse site of each NMS cluster, refine up to K - 1 more coarse sites of the same
    cluster (within --nms-km and 30 s of its clock) that score >= --spot-frac of the cluster best and lie >=
    --spot-sep-km from every other refined spot. The 500 m grid can lose much of a close pass's evidence, so the
    cluster's best coarse site is not always the one nearest the refined optimum."""
    sc = res[key]
    taken = list(picked)
    extra = []
    for i in picked:
        d = haversine_km(G["LA"], G["LO"], G["LA"][i], G["LO"][i])
        cand = np.flatnonzero((d < a.nms_km) & (np.abs(resS[skey] - resS[skey][i]) < 30) & (sc >= a.spot_frac * sc[i])
                              & (sc > 0))
        cand = cand[np.argsort(-sc[cand])]
        n = 0
        for j in cand:
            if n >= a.spots - 1:
                break
            if all(haversine_km(G["LA"][j], G["LO"][j], G["LA"][k], G["LO"][k]) >= a.spot_sep_km for k in taken):
                taken.append(int(j))
                extra.append(int(j))
                n += 1
    return extra


def haversine_km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = p2 - p1, np.radians(lo2 - lo1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def plot_heat(path, G, res, hyps, truth, null_level, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = res["score2"].reshape(G["lats"].size, G["lons"].size)
    fig, ax = plt.subplots(figsize=(14, 7))
    ext = [G["lons"][0], G["lons"][-1], G["lats"][0], G["lats"][-1]]
    im = ax.imshow(H, origin="lower", extent=ext, aspect=1 / math.cos(math.radians(G["lats"].mean())), cmap="magma",
                   vmin=0, vmax=max(1e-3, float(H.max())))
    plt.colorbar(im, ax=ax, label="best two-pass score (sum of votes)")
    if null_level > 0:
        ax.contour(G["lons"], G["lats"], H, levels=[null_level], colors="cyan", linewidths=0.6)
    for k, h in enumerate(hyps[:10]):
        ax.plot(h["lon"], h["lat"], "o", mfc="none", mec="lime", ms=10)
        ax.text(h["lon"], h["lat"], f" {k + 1}", color="lime", fontsize=9)
    if truth:
        ax.plot(truth[1], truth[0], "x", color="cyan", ms=12, mew=2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_refine(path, heat, best, truth, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, lats, lons = heat
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [lons[0], lons[-1], lats[0], lats[-1]]
    im = ax.imshow(H, origin="lower", extent=ext, aspect=1 / math.cos(math.radians(lats.mean())), cmap="magma")
    plt.colorbar(im, ax=ax)
    ax.plot(best[1], best[0], "o", mfc="none", mec="lime", ms=12)
    if truth and lats[0] <= truth[0] <= lats[-1] and lons[0] <= truth[1] <= lons[-1]:
        ax.plot(truth[1], truth[0], "x", color="cyan", ms=12, mew=2)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_hyp(path, srch, G, sidx, site, samp, hyp):
    """Predicted curves of the voting passes over the spectrogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    voters = [p for p in hyp["passes"] if p["vote"] > 0][:6]
    if not voters:
        return
    sp = srch.sp
    fig, axs = plt.subplots(1, len(voters), figsize=(5 * len(voters), 4.5), squeeze=False)
    m = site == sidx
    T = srch.make_templates(G, site[m], samp[m], col_s=0.5, t_lo=-np.inf, t_hi=np.inf)
    fids = T.fid.cpu().numpy()
    for ax, p in zip(axs[0], voters):
        i = int(np.flatnonzero(fids == p["fid"])[0])
        tc = p["t_cpa_rec"]
        fm = (sp["f"] >= 0.6 * p["f0"]) & (sp["f"] <= 1.6 * p["f0"])
        tm = (sp["t"] >= tc - 90) & (sp["t"] <= tc + 90)
        ax.imshow(np.clip(sp["W"][np.ix_(fm, tm)], 0, 12), origin="lower", aspect="auto", cmap="magma",
                  extent=[sp["t"][tm][0], sp["t"][tm][-1], sp["f"][fm][0], sp["f"][fm][-1]])
        tt = tc + 0.5 * np.arange(-T.J, T.J + 1)
        ff = p["f0"] * np.exp(T.lr[i].cpu().numpy())
        v = T.valid[i].cpu().numpy()
        ax.plot(tt[v], ff[v] * 1.03, "c:", lw=0.8)
        ax.plot(tt[v], ff[v] / 1.03, "c:", lw=0.8)
        ax.set_title(f"{p['reg'] or p['hex']} {p['type']} d={p['d_cpa_m'] / 1000:.1f}km z={p['z']:.1f} "
                     f"f0={p['f0']}", fontsize=8)
    fig.suptitle(f"{hyp['lat']:.4f},{hyp['lon']:.4f} start {hyp['start_iso']} (curve +-3 % dotted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--dem-dir")
    ap.add_argument("--ground", type=float, default=0.0, help="constant ground height if no DEM")
    ap.add_argument("--bbox", required=True, help="lat_min,lat_max,lon_min,lon_max")
    ap.add_argument("--spacing", type=float, default=500.0)
    ap.add_argument("--start", help="earliest recording start (UTC ISO or epoch); default tracks t0")
    ap.add_argument("--end", help="latest recording start; default tracks t1 - duration")
    ap.add_argument("--classes", default="prop,rotor,unknown")
    ap.add_argument("--jets", action="store_true")
    ap.add_argument("--dmax", type=float, default=9000.0, help="max CPA slant range (m)")
    ap.add_argument("--temp", type=float, default=20.0, help="air temperature (C) for the speed of sound")
    ap.add_argument("--wind", help="uniform wind east,north (m/s, direction it blows towards)")
    ap.add_argument("--col-s", type=float, default=1.0)
    ap.add_argument("--log-step", type=float, default=0.004)
    ap.add_argument("--fpool", type=int, default=1, help="max-pool the log spectrogram over +-n rows (model tolerance)")
    ap.add_argument("--clip", type=float, default=15.0, help="whitened dB are clipped to [0, clip]")
    ap.add_argument("--ridge", type=int, default=4,
                    help="score = level on the curve minus the curve shifted +-ridge log-f rows (0: minus flat line)")
    ap.add_argument("--harm", type=int, default=1, help="harmonic sum h=1..n in the log spectrogram")
    ap.add_argument("--wpow", type=float, default=2.0, help="template point weight (d_cpa/R)^wpow")
    ap.add_argument("--half-w", type=float, default=60.0, help="s of curve either side of CPA")
    ap.add_argument("--max-logspan", type=float, default=0.45, help="template points beyond |log ratio| are dropped")
    ap.add_argument("--mode", choices=("cand", "full"), default="cand",
                    help="cand: exact templates scored around candidate spots (fast); full: every CPA column x f0 (FFT)")
    ap.add_argument("--n-cand", type=int, default=400, help="candidate spots per recording (cand mode)")
    ap.add_argument("--cand-rt", type=int, default=4, help="+- columns searched around a candidate spot")
    ap.add_argument("--cand-rb", type=int, default=3, help="+- log-f rows searched around a candidate spot")
    ap.add_argument("--cand-batch", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=96, help="templates per FFT batch (full mode)")
    ap.add_argument("--site-batch-bytes", type=float, default=4e8)
    ap.add_argument("--calib-n", type=int, default=100000)
    ap.add_argument("--z0", type=float, default=None)
    ap.add_argument("--z0-pct", type=float, default=99.0)
    ap.add_argument("--vpow", type=float, default=0.5, help="pow votes: clip(z - z0, 0, vcap) ** vpow")
    ap.add_argument("--vote", choices=("pow", "idf", "linear"), default="pow",
                    help="idf: vote = information of the explained spot (-log10 fraction of random passes explaining "
                         "it) x soft step of z; linear: clip(z - z0, 0, vcap); pow: linear ** vpow (concave: "
                         "rewards the number of explained passes over the strength of one)")
    ap.add_argument("--zsoft", type=float, default=4.0, help="idf vote ramps from z0 to z0 + zsoft")
    ap.add_argument("--excl-s", type=float, default=EXCL_S,
                    help="passes at one site with CPAs closer than this (chained) explain the same sound: max, not sum")
    ap.add_argument("--calib-win", type=int, default=0,
                    help="1: when the clock window is < --calib-win-max s, z0 is the chance level of a random pass "
                         "restricted to the spots its CPA can reach within such a window (lower z0 for known clocks)")
    ap.add_argument("--calib-win-max", type=float, default=7200.0)
    ap.add_argument("--tmpl-chunk", type=int, default=200000, help="passes per template batch when culling (GPU memory)")
    ap.add_argument("--pre-top", type=int, default=0, help="pre-refine around this many coarse sites (0 = off)")
    ap.add_argument("--pre-nms-km", type=float, default=1.0)
    ap.add_argument("--pre-spacing", type=float, default=250.0)
    ap.add_argument("--pre-radius", type=float, default=400.0, help="m around each pre-refined coarse site")
    ap.add_argument("--rescore", type=int, default=0, help="1: final one-sound-one-vote / one-aircraft-one-f0 check")
    ap.add_argument("--same-dt", type=float, default=25.0)
    ap.add_argument("--same-dlogf", type=float, default=0.1)
    ap.add_argument("--hex-tol", type=float, default=0.06)
    ap.add_argument("--win-max", type=float, default=0.0,
                    help="clock windows up to this (s) only evaluate clock-consistent candidate spots (0 = never; identical scores, no speed-up measured under GPU sharing)")
    ap.add_argument("--silence-pen", type=float, default=0.0,
                    help="score penalty per close prop/rotor pass inside the recording that does not vote (0 = off)")
    ap.add_argument("--silence-d", type=float, default=2500.0, help="silence penalty: max CPA slant range (m)")
    ap.add_argument("--silence-margin", type=float, default=40.0, help="silence penalty: CPA >= this far inside (s)")
    ap.add_argument("--silence-cover", type=int, default=0,
                    help="1: the silence penalty counts only passes the curve model covers (see Search.covered)")
    ap.add_argument("--silence-hint", type=float, default=0.0,
                    help="> 0: a close pass is not silent where a detector hint with p >= this lies within "
                         "--silence-hint-dt s of its CPA (needs --hints)")
    ap.add_argument("--silence-hint-dt", type=float, default=20.0)
    ap.add_argument("--cover-w", type=float, default=30.0, help="coverage check: +- s around the CPA")
    ap.add_argument("--cover-turn", type=float, default=20.0, help="coverage check: max heading/climb change (deg)")
    ap.add_argument("--cover-dv", type=float, default=0.2, help="coverage check: max relative speed change")
    ap.add_argument("--cover-gap", type=float, default=15.0, help="coverage check: max s to the nearest raw ADS-B point")
    ap.add_argument("--tie-z", type=float, default=0.0,
                    help="vote above the cap: + tie_z * (z - z0 - vcap) ** vpow (0 = hard cap, old behaviour)")
    ap.add_argument("--spots", type=int, default=1, help="coarse sites refined per NMS cluster (see extra_spots)")
    ap.add_argument("--spot-frac", type=float, default=0.7)
    ap.add_argument("--spot-sep-km", type=float, default=1.0)
    ap.add_argument("--vcap", type=float, default=20.0, help="max vote of one pass (linear votes)")
    ap.add_argument("--hints", help="detector events json: list of {t, f, p} (t = CPA s, f = Hz)")
    ap.add_argument("--hints-only", action="store_true", help="cand mode: only the detector hints, no bank spots")
    ap.add_argument("--hint-dt", type=float, default=8.0)
    ap.add_argument("--hint-pmin", type=float, default=0.2)
    ap.add_argument("--hint-prior", type=float, default=None,
                    help="soft prior instead of extra spots: votes of passes without a detection near their CPA are "
                         "multiplied by this (0..1); with one, by prior + (1 - prior) * p")
    ap.add_argument("--hint-df", type=float, default=0.0, help="hint prior: log-f tolerance (0 = time only)")
    ap.add_argument("--hint-key", default="p", help="hint field used as p (p or p_raw)")
    ap.add_argument("--top", type=int, default=60, help="coarse hypotheses (after NMS) refined at level 1")
    ap.add_argument("--top2", type=int, default=12, help="level-1 hypotheses refined again at level 2")
    ap.add_argument("--nms-km", type=float, default=3.0)
    ap.add_argument("--refine-radius", type=float, default=1500.0)
    ap.add_argument("--refine-spacing", type=float, default=100.0)
    ap.add_argument("--refine-dt", type=float, default=15.0)
    ap.add_argument("--refine2-radius", type=float, default=400.0)
    ap.add_argument("--refine2-spacing", type=float, default=25.0)
    ap.add_argument("--refine2-dt", type=float, default=6.0)
    ap.add_argument("--plots", type=int, default=5)
    ap.add_argument("--truth", help="lat,lon (evaluation only, never used by the search)")
    ap.add_argument("--compare", action="append", default=[], help="lat,lon of another hypothesis to report")
    ap.add_argument("--out", required=True)
    add_r5b_args(ap)
    return ap


def get_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t_start = time.time()
    NEG_MODE[0] = bool(a.llr)
    srch = Search(a)
    G = srch.setup_grid(srch.bbox, a.spacing)
    Ns = G["LA"].size
    log(f"grid: {G['lats'].size} x {G['lons'].size} = {Ns} sites at {a.spacing:.0f} m")
    site, samp = srch.passes_for_grid(G, srch.S_min, srch.S_max)
    log(f"passes within {a.dmax:.0f} m and the clock window: {site.numel()} ({site.numel() / Ns:.1f} per site)")
    if site.numel() == 0:     # no prop/rotor traffic in the box and clock window: nothing to match -> no answer
        json.dump(dict(params=vars(a), hypotheses=[], n_sites=int(Ns), n_passes=0, null_level=0.0, z0=None,
                       clock=[iso(srch.S_min), iso(srch.S_max)], runtime_total_s=round(time.time() - t_start, 1)),
                  open(a.out + ".json", "w"), indent=1)
        log("no passes: empty result")
        return json.load(open(a.out + ".json"))
    srch.build_scorer()
    srch.calibrate(G, site, samp)
    res, resS = srch.score_grid(G, site, samp, srch.S_min, srch.S_max, progress=True)
    t_coarse = time.time() - t_start
    null_level = float(res["null2"].max())
    if null_level <= NEG / 2:       # --llr-null 0 (or no reversed two-pass anywhere)
        null_level = 0.0
    log(f"coarse done in {t_coarse:.0f} s; null level (best reversed two-pass score anywhere) {null_level:.2f}")
    C = dict(G=G, res=res, resS=resS, site=site, samp=samp)          # coarse level used to pick hypotheses
    t_pre = 0.0
    if a.pre_top > 0:
        # pre-refine: the 500 m grid loses much of a close pass's evidence (curves are very site-sensitive), so the
        # truth can sit at coarse rank > 100. Re-score a finer grid only around the best coarse sites, full clock.
        t0p = time.time()
        pre = nms_hypotheses(res, resS, G, a.pre_top, a.pre_nms_km)
        G2 = srch.setup_grid(srch.bbox, a.pre_spacing)
        PLA, PLO = G["LA"][pre], G["LO"][pre]
        keep = np.zeros(G2["LA"].size, bool)
        for k0 in range(0, len(pre), 64):
            dkm = haversine_km(G2["LA"][:, None], G2["LO"][:, None], PLA[None, k0:k0 + 64], PLO[None, k0:k0 + 64])
            keep |= (dkm <= a.pre_radius / 1000.0).any(1)
        G2["mask"] = torch.as_tensor(keep, device=DEV)
        site2, samp2 = srch.passes_for_grid(G2, srch.S_min, srch.S_max)
        res2, resS2 = srch.score_grid(G2, site2, samp2, srch.S_min, srch.S_max)
        C = dict(G=G2, res=res2, resS=resS2, site=site2, samp=samp2)
        t_pre = time.time() - t0p
        log(f"pre-refine: {len(pre)} coarse sites -> {int(keep.sum())} sites at {a.pre_spacing:.0f} m, "
            f"{site2.numel()} passes, best {res2['score2'].max():.2f} (coarse best {res['score2'].max():.2f}), "
            f"{t_pre:.0f} s")
    picked = nms_hypotheses(C["res"], C["resS"], C["G"], a.top, a.nms_km)
    if a.spots > 1:
        picked = picked + extra_spots(C["res"], C["resS"], C["G"], picked, a)
        log(f"refining {len(picked)} coarse sites ({a.spots} spots per cluster at most)")
    truth = [float(x) for x in a.truth.split(",")] if a.truth else None

    def coarse_hyp(i, L=None):
        L = L or C
        Gx, rx = L["G"], L["res"]
        S = float(L["resS"]["S2"][i])
        passes = srch.explain(Gx, i, S, L["site"], L["samp"])
        return dict(lat=float(Gx["LA"][i]), lon=float(Gx["LO"][i]), ground_m=float(Gx["g"][i]), start_utc=S,
                    start_iso=iso(S), score=float(rx["score2"][i]), score_single=float(rx["score1"][i]),
                    coarse_site_rank=int((rx["score2"] > rx["score2"][i]).sum()) + 1,
                    n_voting=int(sum(p["vote"] >= vmin_of(a) for p in passes)), site_null=float(rx["null2"][i]),
                    passes=passes[:12 if not a.llr else a.pass_keep])

    def near_best(la, lo):
        d = haversine_km(G["LA"], G["LO"], la, lo)
        near = d < 1.5
        return int(np.flatnonzero(near)[np.argmax(res["score2"][near])]) if near.any() else int(np.argmin(d))

    t_ref = time.time()

    def refine_into(h, level):
        r, sp_, dt_s = ((a.refine_radius, a.refine_spacing, a.refine_dt) if level == 1 else
                        (a.refine2_radius, a.refine2_spacing, a.refine2_dt) if level == 2 else
                        (a.refine3_radius, a.refine3_spacing, a.refine3_dt))
        ref = srch.refine(h["lat"], h["lon"], h["start_utc"], r, sp_, dt_s)
        if ref is not None and ref["score"] >= h["score"]:
            h["_heat"], h["_plot"] = ref.pop("_heat"), ref.pop("_plot")
            h.update(ref)
        h["refine_level"] = level

    hyps = []
    for i in picked:
        h = coarse_hyp(i)
        h["coarse"] = {k: h[k] for k in ("lat", "lon", "start_iso", "score")}
        h["_plot"] = (C["G"], i, C["site"], C["samp"])
        refine_into(h, 1)
        hyps.append(h)
    hyps.sort(key=lambda h: -h["score"])
    for h in hyps[:a.top2]:
        refine_into(h, 2)
    extra = {}
    for name, pt in ([("truth", truth)] if truth else []) + [(f"compare{k}", [float(x) for x in c.split(",")[:2]])
                                                               for k, c in enumerate(a.compare)]:
        i = near_best(*pt)
        h = coarse_hyp(i, dict(G=G, res=res, resS=resS, site=site, samp=samp))
        h["coarse"] = {k: h[k] for k in ("lat", "lon", "start_iso", "score")}
        if h["score"] > (NEG / 2 if a.llr else 0):
            refine_into(h, 1)
            refine_into(h, 2)
        h["query"] = pt
        h["dist_query_km"] = float(haversine_km(h["lat"], h["lon"], pt[0], pt[1]))
        extra[name] = h
    t_ref = time.time() - t_ref
    hint_t = [float(x["t"]) for x in (srch.hints or []) if float(x.get("p", 0.0)) >= a.hint_pmin]
    if a.rescore:
        for h in hyps + list(extra.values()):
            rescore(h, a, hint_t)
    hyps.sort(key=lambda h: -h["score"])
    if a.refine3_top > 0:
        # round 5b: final sub-grid refinement (10 m, +-150 m, clock +-4 s) of the best hypotheses after the rescore;
        # the refined spot is re-scored the same way and replaces the old one only if it still scores at least as high
        for h in hyps[:a.refine3_top]:
            if h["score"] <= (NEG / 2 if a.llr else 0):
                continue
            old = {k: h[k] for k in ("lat", "lon", "start_utc", "start_iso", "score", "passes", "n_voting") if k in h}
            ref = srch.refine(h["lat"], h["lon"], h["start_utc"], a.refine3_radius, a.refine3_spacing, a.refine3_dt)
            if ref is None:
                continue
            ref.pop("_heat", None)
            ref.pop("_plot", None)
            cand = dict(h, **ref)
            if a.rescore:
                rescore(cand, a, hint_t)
            if cand["score"] >= old["score"]:
                h.update(cand)
                h["refine_level"] = 3
        hyps.sort(key=lambda h: -h["score"])
    for r, h in enumerate(hyps):
        h["rank"] = r + 1
        h["null_margin"] = h["score"] - null_level
        if truth:
            h["dist_truth_km"] = float(haversine_km(h["lat"], h["lon"], truth[0], truth[1]))
    for name, h in extra.items():
        h["would_rank"] = int(sum(x["score"] > h["score"] for x in hyps)) + 1
        h["null_margin"] = h["score"] - null_level
    out = dict(params=vars(a), runtime_coarse_s=round(t_coarse, 1), runtime_refine_s=round(t_ref, 1), z0=srch.z0,
               calibration=srch.calib, null_level=null_level, n_sites=int(Ns), n_passes=int(site.numel()),
               clock=[iso(srch.S_min), iso(srch.S_max)])
    if truth:
        tr_rank = next((h["rank"] for h in hyps if h["dist_truth_km"] < 3.0), None)
        out["truth_summary"] = dict(first_hypothesis_within_3km=tr_rank,
                                    top1_dist_km=hyps[0]["dist_truth_km"] if hyps else None,
                                    coarse_site_rank_near_truth=extra["truth"]["coarse_site_rank"])
    if a.plots > 0:
        np.savez_compressed(a.out + ".grid.npz", lats=G["lats"], lons=G["lons"], **res, **resS)
        plot_heat(a.out + ".heat.png", G, res, hyps, truth, null_level,
              f"{os.path.basename(a.out)}: coarse two-pass score, cyan contour = null level {null_level:.1f}, "
              f"x = truth (evaluation only), circles = refined top 10")
    for k, h in enumerate(hyps[:a.plots]):
        try:
            if "_heat" in h:
                plot_refine(a.out + f".refine{k + 1}.png", h["_heat"], (h["lat"], h["lon"]), truth,
                            f"rank {k + 1}: {h['lat']:.5f},{h['lon']:.5f} start {h['start_iso']} score {h['score']:.1f}")
            Gp, j, sp_, sm_ = h["_plot"]
            plot_hyp(a.out + f".rank{k + 1}.png", srch, Gp, j, sp_, sm_, h)
        except Exception as e:  # plotting must not kill the run
            log("plot failed", repr(e))
    for name, h in extra.items():
        if "_heat" in h:
            plot_refine(a.out + f".{name}.png", h["_heat"], (h["lat"], h["lon"]), truth,
                        f"{name}: {h['lat']:.5f},{h['lon']:.5f} score {h['score']:.1f}")
    for h in hyps + list(extra.values()):
        h.pop("_heat", None)
        h.pop("_plot", None)
    if a.llr:     # hypotheses left without enough voters are no answer
        hyps = [h for h in hyps if h["score"] > NEG / 2]
        for r, h in enumerate(hyps):
            h["rank"] = r + 1
    out["hypotheses"] = hyps
    out.update(extra)
    out["runtime_total_s"] = round(time.time() - t_start, 1)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    log(f"done: coarse {t_coarse:.0f} s, refine {t_ref:.0f} s, total {out['runtime_total_s']:.0f} s; "
        f"z0 {srch.z0:.2f}, null level {null_level:.2f}")

    def line(h):
        return (f"{h['lat']:.4f},{h['lon']:.4f} start {h['start_iso']} score {h['score']:.1f} (coarse "
                f"{h['coarse']['score']:.1f}, margin {h['null_margin']:+.1f}) voters " +
                ", ".join(f"{p['reg'] or p['hex']}/{p['type']} d{p['d_cpa_m'] / 1000:.1f}km z{p['z']:.0f} "
                          f"t{p['t_cpa_rec']:.0f} f{p['f0']}" for p in h["passes"] if p["vote"] > 0))
    for h in hyps[:10]:
        log(f"#{h['rank']:2d} " + line(h) + (f"  [{h['dist_truth_km']:.1f} km from truth]" if truth else ""))
    for name, h in extra.items():
        log(f"{name} (would rank {h['would_rank']}, {h['dist_query_km']:.2f} km from query): " + line(h))
    return out



def add_r5b_args(ap):
    """Round 5b (eval/RESULTS.md, "Round 5b: locator"); everything off by default, so earlier tags reproduce."""
    ap.add_argument("--llr", help="model JSON (eval/r5b/fit_llr.py): score = sum of calibrated per-pass log-likelihood "
                                  "ratios (replaces votes, cap, hint multiplier and flat silence penalty)")
    ap.add_argument("--f0-model", help="model JSON for the f0 prior / comb only (old votes)")
    ap.add_argument("--f0-prior", type=float, default=0.0, help="kappa: spot statistic s = z + kappa * llr_f0(type, f0)")
    ap.add_argument("--f0-clip", type=float, default=3.0, help="|llr_f0| clip (nats)")
    ap.add_argument("--comb", default="", help="per-group harmonic comb images, e.g. 'piston:1,1.5,2;heli:1,1.5,2'")
    ap.add_argument("--min-voters", type=int, default=2, help="llr: voters (vote >= --llr-vmin) needed for an answer")
    ap.add_argument("--llr-vmin", type=float, default=1.0, help="llr: a pass is a voter when its vote >= this (nats)")
    ap.add_argument("--llr-eps", type=float, default=0.05, help="llr: votes below this are dropped (sparsity)")
    ap.add_argument("--llr-cap", type=float, default=12.0, help="llr: max log LR of one pass's statistic")
    ap.add_argument("--llr-calib-n", type=int, default=30000, help="llr: random passes for the chance distribution")
    ap.add_argument("--llr-hint", type=int, default=1, help="llr: detector hints enter as P(hint | audible) / coverage")
    ap.add_argument("--miss-cost", type=int, default=1, help="llr: 0 drops the log(1 - pi) missing-pass term (ablation)")
    ap.add_argument("--pi-scale", type=float, default=1.0, help="llr: audibility pi x this")
    ap.add_argument("--unexpl-cost", type=float, default=0.0,
                    help="llr rescore: cost (nats) per detector hint (p >= --hint-pmin) that no kept voter explains")
    ap.add_argument("--clock-marg", type=int, default=0, help="llr: site score = log-sum-exp over the clock")
    ap.add_argument("--pass-feats", type=int, default=0,
                    help="per-pass features in the output (pi, terrain clearance, envelope); no effect on scores")
    ap.add_argument("--pass-keep", type=int, default=40, help="llr: passes kept per hypothesis in the output")
    ap.add_argument("--env-llr", help="envelope log-LR table JSON (eval/r5b/fit_env.py); adds env_w x sum over voters")
    ap.add_argument("--env-w", type=float, default=1.0)
    ap.add_argument("--env-clip", type=float, default=2.0)
    ap.add_argument("--terrain-rho", type=float, default=1.0,
                    help="audibility x this when the CPA line of sight is blocked by terrain (1 = off)")
    ap.add_argument("--terrain-soft", type=float, default=15.0, help="m: blocking = sigmoid(-clearance / this)")
    ap.add_argument("--terrain-skip", type=float, default=150.0, help="m of the line next to the observer ignored")
    ap.add_argument("--terrain-step", type=float, default=60.0, help="m: terrain grid spacing")
    ap.add_argument("--slack-s", type=float, default=0.0,
                    help="llr: per-aircraft time slack (s); votes may shift up to this at --slack-cost nats/s beyond DIL")
    ap.add_argument("--slack-cost", type=float, default=0.3)
    ap.add_argument("--refine3-top", type=int, default=0, help="final 10 m sub-grid refinement of this many top hypotheses")
    ap.add_argument("--refine3-radius", type=float, default=150.0)
    ap.add_argument("--refine3-spacing", type=float, default=10.0)
    ap.add_argument("--refine3-dt", type=float, default=4.0)
    ap.add_argument("--llr-null", type=int, default=1,
                    help="llr: 0 skips the time-reversed null pass (null_level = no answer; the decoy null replaces it); "
                         "scores and rankings are unchanged, coarse time roughly halves")


if __name__ == "__main__":
    main()
