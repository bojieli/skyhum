"""The whole-country search: "which region?" from a recording whose start time is known to a few seconds.

Ported from the research country search (tonal channel, round-5b per-pass LLR). For a group of sessions known to come
from one place (one recording = a group of one), per traffic variant (the real clock, K clock decoys at +-10..25 min,
other-day decoys; a decoy shifts every session):
  A. candidates per session: every detector-v3 detection (p >= ev_pmin) at recording time t_e. Every low prop / rotor /
     unknown aircraft of the country flying within [S0 + t_e - tau - 35, S0 + t_e + tau + 8] s marks the 2 km lattice
     cells within 9 km of its track; exact emission-time templates (skyhum.match.r5b.Templates) keep the (cell,
     aircraft) passes whose CPA reception time is within tau + 8 s of the detection, and those whose round-5b
     statistic gives a vote >= surv_vote at that time survive (the detector term excluded).
  B. scoring: the union of the sessions' surviving cells is scored for every session with ALL passes of the window
     (calibrated per-pass log-likelihood ratio, llr.py: f0 prior, LLR of the curve statistic, hint term, miss cost
     log(1 - pi) for silent passes; one sound one vote; one aircraft one vote).
  C. per subset of sessions x window kind (60 / 30 / 10 min) x clock tolerance (+-5 / +-30 s): session score = max
     over its clock; joint = sum over the subset's sessions (cells must come from the subset's own candidates, >= 2
     voters in total, from >= 2 different aircraft for a single session); ranked cells -> regions (CBSA / county, SA4).
     The top cells are refined (500 m, then 125 m) and re-scored the same way.
Output: per variant, subset, window, tau: the ranked regions (score, best point, voters). The scorer (score.py) alone
opens the truth. No answer is validated as correct; see docs/locate.md for the base rates and limits.

Runs on CUDA when available, else on the CPU (slow: hours per group).
"""
import datetime as dt
import math
import os
import random
import time
import types

import numpy as np

from skyhum.country import guard
from skyhum.country.common import Lattice, Raster, Regions, dump_json, load_json

T_LOG = time.time()
VERBOSE = [True]


def log(*a):
    if VERBOSE[0]:
        print(f"[{time.time() - T_LOG:7.1f}s]", *a, flush=True)


NEG = -1e9
CLOCK_OFFS_MIN = [10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 35, -35, 40, -40, 45, -45, 50, -50, 55, -55, 60, -60]

# research defaults (every published country-search number used these)
DEFAULTS = dict(tau=30.0, taus="5,30", k_clock=8, k_day=2, ev_pmin=0.8, ev_sig=8.0, surv_vote=2.5, vmin=1.0,
                spacing=2000.0, dmax=9000.0, dec=6, kappa=3.0, excl_s=20.0, edge_s=30.0, miss_margin=40.0,
                top_regions=40, refine_top=8, tmpl_chunk=6000, calib_n=20000, variants="", seed=0, loud_w="1",
                loud_cap=10.0)


class MissingDay(Exception):
    def __init__(self, day, path, why="missing"):
        super().__init__(f"ADS-B day {day} has an adsb.lol release but its file is {why}: {path} "
                         f"(run: skyhum adsb fetch --date {day})")
        self.day = day


def _empty_marker(d, day, region):
    p = os.path.join(d, f"{day}.json")
    if not os.path.exists(p):
        return False
    c = load_json(p).get(f"{region}_low")
    return c is not None and c.get("segments", 0) == 0


def available_days(adsb_dir):
    """UTC days with an adsb.lol release (from releases.json, written by skyhum adsb fetch)."""
    p = os.path.join(adsb_dir, "releases.json")
    return {d for d, u in load_json(p).items() if u} if os.path.exists(p) else set()


class DayCache:
    """Region-wide low-altitude day files (stream_low.py) -> raw point tables, a few days kept in memory."""

    KEYS = ("fid", "t", "lat", "lon", "h", "gs", "trk", "vr")

    def __init__(self, d, region, avail=()):
        self.d, self.region, self.mem, self.avail = d, region, {}, set(avail)

    def day(self, day):
        if day not in self.mem:
            p = os.path.join(self.d, f"{day}_{self.region}_low.npz")
            if not os.path.exists(p):
                if day in self.avail and not _empty_marker(self.d, day, self.region):
                    # the day has an adsb.lol release: its file must be here (a silent "no traffic" would bias)
                    raise MissingDay(day, p)
                self.mem[day] = None
            else:
                try:
                    z = np.load(p)
                    X = {k: z[k] for k in self.KEYS}
                except (OSError, ValueError, KeyError) as e:
                    raise MissingDay(day, p, f"unreadable: {e!r}") from e
                if "t_base" in z.files:          # time-slimmed day file: float32 offsets from t_base
                    X["t"] = float(z["t_base"]) + X["t"].astype(np.float64)
                for k in ("lat", "lon", "h", "gs", "trk", "vr"):
                    X[k] = X[k].astype(np.float64)
                X["fid"] = X["fid"].astype(np.int64)
                X["meta"] = {k: z[k] for k in ("f_hex", "f_type", "f_reg", "f_cls", "f_desc")}
                self.mem[day] = X
            while len(self.mem) > 4:
                self.mem.pop(next(iter(self.mem)))
        return self.mem[day]

    def block(self, t0, t1):
        """Raw points in [t0, t1] of every day file that overlaps (flights renumbered globally)."""
        # every UTC day that overlaps [t0, t1] (not the day after t1: arange(t0, t1 + 3600) used to add it when
        # t1 was within an hour of midnight, reported as a "missing" day)
        days = sorted({dt.datetime.fromtimestamp(min(t, t1), dt.timezone.utc).strftime("%Y-%m-%d")
                       for t in np.arange(t0, t1 + 3600, 3600.0)})
        parts, meta, off, missing = [], {k: [] for k in ("f_hex", "f_type", "f_reg", "f_cls", "f_desc")}, 0, []
        for d in days:
            X = self.day(d)
            if X is None:
                missing.append(d)
                continue
            m = (X["t"] >= t0) & (X["t"] <= t1)
            P = {k: X[k][m] for k in self.KEYS}
            P["fid"] = P["fid"].astype(np.int64) + off
            parts.append(P)
            for k in meta:
                meta[k].append(X["meta"][k])
            off += len(X["meta"]["f_hex"])
        if not parts:
            return None, missing
        B = {k: np.concatenate([p[k] for p in parts]) for k in self.KEYS}
        for k in meta:
            B[k] = np.concatenate(meta[k])
        o = np.lexsort((B["t"], B["fid"]))
        for k in self.KEYS:
            B[k] = B[k][o]
        return B, missing


def resample(B, fids, t0, t1):
    """1 s resampling of a subset of flights of a raw block, vectorised over all flights (afg.tracks.Resampled is
    a per-flight PCHIP loop, ~1 ms per flight, too slow for 20k CONUS flights per pass): positions by cubic Hermite
    interpolation with the broadcast ground velocity as the derivative at each raw point (linear where it is
    missing), altitude / velocity / vertical rate linear - as Resampled, which also interpolates velocity
    linearly. Returns an object with Resampled's attributes (fid, t, lat, lon, h, ve, vn, vu, gs, ecef, vel, n,
    _first, _last)."""
    from skyhum._vendor.afg.geo import enu_vectors_to_ecef, geodetic_to_ecef
    fids = np.unique(np.asarray(fids, np.int64))
    if fids.size == 0:
        return None
    a = np.searchsorted(B["fid"], fids, "left")
    b = np.searchsorted(B["fid"], fids, "right")
    m = b - a >= 3
    fids, a, b = fids[m], a[m], b[m]
    if fids.size == 0:
        return None
    t = B["t"]
    ta = np.maximum(np.ceil(np.maximum(t[a], t0)), -1e18)
    tb = np.floor(np.minimum(t[b - 1], t1))
    n = (tb - ta).astype(np.int64) + 1
    ok = n >= 3
    fids, a, b, ta, n = fids[ok], a[ok], b[ok], ta[ok], n[ok]
    if fids.size == 0:
        return None
    tot = int(n.sum())
    rep = np.repeat(np.arange(fids.size), n)
    start = np.cumsum(n) - n
    tg = ta[rep] + (np.arange(tot) - start[rep])
    fg = fids[rep]
    # bracketing raw points: global search on (fid, t) with t offset per flight
    key_raw = B["fid"].astype(np.float64) * 1e10 + (t - 1.6e9)
    key_q = fg.astype(np.float64) * 1e10 + (tg - 1.6e9)
    j = np.searchsorted(key_raw, key_q, "right") - 1
    j = np.clip(j, a[rep], b[rep] - 2)
    t_0, t_1 = t[j], t[j + 1]
    h_ = np.maximum(t_1 - t_0, 1e-3)
    u = np.clip((tg - t_0) / h_, 0, 1)
    lat0, lat1, lon0, lon1 = B["lat"][j], B["lat"][j + 1], B["lon"][j], B["lon"][j + 1]
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    gs0, gs1 = B["gs"][j].astype(float), B["gs"][j + 1].astype(float)
    tr0, tr1 = np.radians(B["trk"][j].astype(float)), np.radians(B["trk"][j + 1].astype(float))
    ve0, vn0, ve1, vn1 = gs0 * np.sin(tr0), gs0 * np.cos(tr0), gs1 * np.sin(tr1), gs1 * np.cos(tr1)
    mlat = 111132.954 - 559.822 * np.cos(2 * np.radians(lat0))
    mlon = np.maximum(111412.84 * np.cos(np.radians(lat0)), 1.0)
    cy, cx = (lat1 - lat0) * mlat, dlon * mlon                               # chord (m)
    good = np.isfinite(ve0) & np.isfinite(ve1) & np.isfinite(vn0) & np.isfinite(vn1)
    # derivative at the ends (m per unit u); chord slope where the velocity is missing
    dx0 = np.where(good, ve0 * h_, cx)
    dx1 = np.where(good, ve1 * h_, cx)
    dy0 = np.where(good, vn0 * h_, cy)
    dy1 = np.where(good, vn1 * h_, cy)
    h10, h01, h11 = u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2     # Hermite basis (h00 unused)
    x = h10 * dx0 + h01 * cx + h11 * dx1
    y = h10 * dy0 + h01 * cy + h11 * dy1
    lat = lat0 + y / mlat
    lon = ((lon0 + x / mlon + 180.0) % 360.0) - 180.0
    hh = B["h"].astype(float)
    h = hh[j] + u * (hh[j + 1] - hh[j])
    h = np.where(np.isfinite(h), h, np.where(np.isfinite(hh[j]), hh[j], hh[j + 1]))
    ve = np.where(good, ve0 + u * (ve1 - ve0), cx / h_)
    vn = np.where(good, vn0 + u * (vn1 - vn0), cy / h_)
    vr = B["vr"].astype(float)
    vu = vr[j] + u * (vr[j + 1] - vr[j])
    vu = np.where(np.isfinite(vu), vu, (hh[j + 1] - hh[j]) / h_)
    vu = np.where(np.isfinite(vu), vu, 0.0)
    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h)
    R = types.SimpleNamespace()
    if not keep.all():
        # drop flights with any bad sample (rare: altitude missing everywhere)
        badf = np.unique(fg[~keep])
        keep = ~np.isin(fg, badf)
    R.fid, R.t, R.lat, R.lon, R.h = fg[keep], tg[keep], lat[keep], lon[keep], h[keep]
    R.ve, R.vn, R.vu = ve[keep], vn[keep], vu[keep]
    if R.t.size == 0:
        return None
    R.gs = np.hypot(R.ve, R.vn)
    R.ecef = geodetic_to_ecef(R.lat, R.lon, R.h)
    R.vel = enu_vectors_to_ecef(R.lat, R.lon, R.ve, R.vn, R.vu)
    R.n = R.t.size
    uf, first, cnt = np.unique(R.fid, return_index=True, return_counts=True)
    R._first = dict(zip(uf.tolist(), first.tolist()))
    R._last = dict(zip(uf.tolist(), (first + cnt - 1).tolist()))
    return R


class Sub:
    """Device view of resampled flights in the layout skyhum.match.r5b.Templates expects (fid_t, F0_t, F1_t, ecef,
    vel, tau); fid_t holds GLOBAL block flight ids (metadata index)."""

    def __init__(self, R, torch, DEV):
        self.R = R
        self.n = R.n
        maxf = int(R.fid.max()) + 1
        F0 = np.zeros(maxf, np.int64)
        F1 = np.zeros(maxf, np.int64)
        for f, a in R._first.items():
            F0[f] = a
            F1[f] = R._last[f]
        self.F0_t = torch.as_tensor(F0, device=DEV)
        self.F1_t = torch.as_tensor(F1, device=DEV)
        self.fid_t = torch.as_tensor(R.fid.astype(np.int64), device=DEV)
        self.ecef = torch.as_tensor(R.ecef, device=DEV, dtype=torch.float64)
        self.vel = torch.as_tensor(R.vel, device=DEV, dtype=torch.float64)
        self.tau = torch.as_tensor(R.t, device=DEV, dtype=torch.float64)


# ----------------------------------------------------------------------------------------------------------------
# main search object
# ----------------------------------------------------------------------------------------------------------------
class Session:
    pass


class Locator:
    def __init__(self, a, region, sessions_oids, loud=None):
        import torch
        self.torch = torch
        from skyhum.match import r5b as G
        self.G = G
        self.DEV = G.DEV
        self.a = a
        self.region = region
        self.lat = Lattice(region, a.spacing)
        st = a.static
        self.dem = Raster(os.path.join(st, f"dem_{region}.npz"), "ground")
        self.regions = Regions(region, st)
        self.cache = DayCache(a.adsb, region, a.avail)
        self.c = 20.0468 * math.sqrt(20.0 + 273.15)
        from skyhum.country import llr
        self.M = llr.Model(a.model, torch, self.DEV, a.kappa)
        # loud-pass channel: an object with the research loud_llr.Loud interface (t_of, nearest, sig_of, p_cand,
        # event_terms, site_terms), or None for the tonal channel only (this build)
        self.L = loud
        self.sessions = [self.load_session(o) for o in sessions_oids]

    # -- sessions ------------------------------------------------------------------------------------------------
    def load_session(self, oid):
        from skyhum._vendor.afg import audio
        G, torch = self.G, self.torch
        a = self.a
        bd = os.path.join(a.root, "blind", oid)
        inp = load_json(os.path.join(bd, "input.json"))
        s = Session()
        s.oid, s.clock, s.dur, s.windows = oid, float(inp["clock_utc"]), float(inp["dur_s"]), inp["windows"]
        t = time.time()
        if os.path.exists(os.path.join(bd, "spec_q.npz")):
            # quantised spectrogram: the same W, clipped to [0, 15] as LogSpec does, 1/17 dB steps
            z = np.load(os.path.join(bd, "spec_q.npz"))
            sp = dict(W=z["Wq"].astype(np.float32) / 17.0, f=z["f"], t=z["t"], hop_s=float(z["hop_s"]))
        elif os.path.exists(os.path.join(bd, "spec.npz")):
            z = np.load(os.path.join(bd, "spec.npz"))
            sp = dict(W=z["W"].astype(np.float32), f=z["f"], t=z["t"], hop_s=float(z["hop_s"]))
        else:
            aud = [f for f in os.listdir(bd) if f.startswith("audio.")][0]
            x, fs, _ = audio.load_audio(os.path.join(bd, aud), fs=4000)
            sp = audio.compute_spectrogram(x, fs, win_s=1.0, hop_s=0.2, fmax=620.0)
            del x
        s.ls = G.LogSpec(sp["W"].astype(np.float32), sp["f"], sp["t"], sp["hop_s"], 1.0, 0.004, 1, 1, (0.0, 15.0))
        del sp
        hp = os.path.join(bd, "hints_v3.json")
        s.hints = load_json(hp) if os.path.exists(hp) else []
        J = int(round(60.0 / s.ls.col_s))
        s.J = J
        s.max_db = int(np.ceil(0.45 / 0.004))
        s.scorer = G.CandScorer(s.ls, J, s.max_db, self.c, 2.0, 400, 4, 3, hints=s.hints, hint_dt=8.0,
                                hint_pmin=a.ev_pmin, dr=4)
        tt = s.ls.t0 + s.ls.col_s * np.arange(s.ls.nt)
        hd = np.zeros(s.ls.nt, bool)
        for h in s.hints:
            if float(h.get("p", 0)) >= a.ev_pmin:
                hd |= np.abs(tt - float(h["t"])) <= 8.0
        s.hcol = torch.as_tensor(hd, device=self.DEV)
        s.events = sorted(float(h["t"]) for h in s.hints if float(h.get("p", 0)) >= a.ev_pmin)
        lp = os.path.join(bd, "loud.json")
        s.loud = load_json(lp)["events"] if (self.L is not None and os.path.exists(lp)) else []
        s.loud_rho0 = max(len(s.loud) / max(s.dur, 1.0), 1.0 / 900.0)
        s.loud_cand = sorted(self.L.t_of(e) for e in s.loud if e["p_air"] >= self.L.p_cand) if self.L else []
        s.llr = None
        log(f"session {oid}: {s.dur:.0f} s, {len(s.events)} events (v3 p >= {a.ev_pmin}), spectrogram "
            f"{time.time() - t:.0f} s")
        return s

    # -- sites ---------------------------------------------------------------------------------------------------
    def site_arrays(self, lat, lon):
        from skyhum._vendor.afg.geo import geodetic_to_ecef
        g = self.dem.bilinear(lat, lon)
        obs = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), g + 1.5)
        return obs

    # -- pass enumeration by lattice join ------------------------------------------------------------------------
    def find_passes(self, sub, site_key, site_obs, t_lo=None, t_hi=None, gen_cells=False):
        """(site index, sample index) of every local minimum of slant range <= dmax between the flights in sub and
        the sites (site_key: coarse lattice key per site, sorted ascending). gen_cells: the sites are generated
        from the samples (every lattice cell within reach); returns the new site keys too. t_lo/t_hi restrict the
        minimum's sample time (emission)."""
        torch, DEV = self.torch, self.DEV
        a = self.a
        R = sub.R
        L = self.lat
        dec = a.dec
        # decimated sample positions per flight, plus each flight's last sample
        idx = []
        for f, s0 in R._first.items():
            e = R._last[f]
            ii = np.arange(s0, e + 1, dec)
            if ii[-1] != e:
                ii = np.r_[ii, e]
            idx.append(ii)
        if not idx:
            return None
        idx = np.concatenate(idx)
        flt = R.fid[idx]
        reach = a.dmax + 600 + dec * 80.0
        si, sj = L.ij(R.lat[idx], R.lon[idx])
        di, dj = L.offsets(reach)
        if gen_cells:
            keys_all = []
            for c0 in range(0, idx.size, 20000):
                ki = (si[c0:c0 + 20000, None] + di[None, :])
                kj = (sj[c0:c0 + 20000, None] + dj[None, :])
                la, lo = L.la0 + ki * L.dlat, L.lo0 + kj * L.dlon
                ok = (np.abs(la - R.lat[idx][c0:c0 + 20000, None]) * 111.132 <= reach / 1000) & \
                     (np.hypot((la - R.lat[idx][c0:c0 + 20000, None]) * 111.132,
                               (lo - R.lon[idx][c0:c0 + 20000, None]) * 111.32 * np.cos(np.radians(la)))
                      <= (a.dmax + 400) / 1000)
                ok &= (ki >= 0) & (ki < L.ni) & (kj >= 0) & (kj < L.nj)
                keys_all.append(L.key(ki[ok], kj[ok]))
            site_key = np.unique(np.concatenate(keys_all)) if keys_all else np.zeros(0, np.int64)
            if site_key.size == 0:
                return None
            la, lo = L.latlon_of_key(site_key)
            site_obs = self.site_arrays(la, lo)
        keys_t = torch.as_tensor(site_key, device=DEV)
        obs_t = torch.as_tensor(site_obs, device=DEV, dtype=torch.float64)
        ecef = sub.ecef[torch.as_tensor(idx, device=DEV)]
        pairs_s, pairs_p, pairs_d = [], [], []
        dit = torch.as_tensor(di, device=DEV)
        djt = torch.as_tensor(dj, device=DEV)
        sit = torch.as_tensor(si, device=DEV)
        sjt = torch.as_tensor(sj, device=DEV)
        CH = max(1000, int(4e6 // max(1, di.size)))
        for c0 in range(0, idx.size, CH):
            c1 = min(idx.size, c0 + CH)
            k = (sit[c0:c1, None] + dit[None, :]) * L.nj + (sjt[c0:c1, None] + djt[None, :])
            k = k.reshape(-1)
            lo_ = torch.searchsorted(keys_t, k, right=False)
            hi_ = torch.searchsorted(keys_t, k, right=True)
            cnt = hi_ - lo_
            m = cnt > 0
            if not bool(m.any()):
                continue
            rows = torch.arange(c0, c1, device=DEV).repeat_interleave(di.size)[m]
            lo_, cnt = lo_[m], cnt[m]
            rr = rows.repeat_interleave(cnt)
            st = torch.repeat_interleave(lo_, cnt)
            offs = torch.arange(int(cnt.sum()), device=DEV) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            ss = st + offs
            d = torch.linalg.norm(ecef[rr] - obs_t[ss], dim=1)
            ok = d < a.dmax + 400 + dec * 70.0
            pairs_s.append(ss[ok])
            pairs_p.append(rr[ok])
            pairs_d.append(d[ok].float())
        if not pairs_s:
            return None
        ps = torch.cat(pairs_s)
        pp = torch.cat(pairs_p)
        pd = torch.cat(pairs_d)
        o = torch.argsort(ps * (idx.size + 1) + pp)
        ps, pp, pd = ps[o], pp[o], pd[o]
        flt_t = torch.as_tensor(flt, device=DEV)
        same_prev = torch.zeros_like(ps, dtype=torch.bool)
        same_next = torch.zeros_like(ps, dtype=torch.bool)
        if ps.numel() > 1:
            sp_ = (ps[1:] == ps[:-1]) & (pp[1:] == pp[:-1] + 1) & (flt_t[pp[1:]] == flt_t[pp[:-1]])
            same_prev[1:] = sp_
            same_next[:-1] = sp_
        dprev = torch.full_like(pd, float("inf"))
        dnext = torch.full_like(pd, float("inf"))
        dprev[1:] = pd[:-1]
        dnext[:-1] = pd[1:]
        loc = same_prev & same_next & (pd <= dprev) & (pd < dnext) & (pd < a.dmax + 400)
        samp = torch.as_tensor(idx, device=DEV)[pp[loc]]
        site = ps[loc]
        if t_lo is not None:
            tt = sub.tau[samp]
            m = (tt >= t_lo - dec) & (tt <= t_hi + dec)
            samp, site = samp[m], site[m]
        return site, samp, site_key, site_obs

    # -- templates and per-pass compact votes ----------------------------------------------------------------------
    def pass_votes(self, s, sub, obs_t, site, samp, S0, tau, tc_lo=-np.inf, tc_hi=np.inf, hint=True):
        """Templates for (site, samp); keep passes with CPA reception in [tc_lo, tc_hi]; vote of each pass at every
        clock S in [S0 - tau, S0 + tau] (1 s): V (P, 2 tau + 1), base (P,), and pass info."""
        torch, G = self.torch, self.G
        a = self.a
        nS = int(2 * tau) + 1
        outs = []
        ch = a.tmpl_chunk if self.DEV.type == "cuda" else min(a.tmpl_chunk, 400)   # same results, less memory
        for q in range(0, site.numel(), ch):
            T = G.Templates(sub, obs_t, site[q:q + ch], samp[q:q + ch], a.dmax, self.c, 0.004,
                            s.ls.col_s, half_w=60.0, wpow=2.0, max_db=s.max_db, t_c_lo=tc_lo, t_c_hi=tc_hi)
            T.select(T.ok)
            if T.site.numel() == 0:
                continue
            # elevation of the aircraft at the approximate CPA sample
            Pp = sub.ecef[T.samp]
            O = obs_t[T.site]
            up = O / torch.linalg.norm(O, dim=1, keepdim=True)
            D = Pp - O
            elev = torch.rad2deg(torch.asin(((D * up).sum(1) / torch.linalg.norm(D, dim=1).clamp(min=1.0))
                                            .clamp(-1, 1)))
            V, base = self.M.votes(s, T, elev, S0, tau, nS, G, hint=hint)
            outs.append(dict(site=T.site, fid=T.fid, t_c=T.t_c, d=T.d_cpa.float(), V=V, base=base,
                             v=T.v_cpa.float(), elev=elev.float()))
            del T
        if not outs:
            return None
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    # -- per-site aggregation over the clock ------------------------------------------------------------------------
    def aggregate(self, s, PV, nsites, S0, tau, w0, w1, need_distinct=True):
        """For sites 0..nsites-1: per clock S (S0 - tau .. S0 + tau): score = sum of group-max votes + sum of miss
        costs of passes inside [w0, w1]; voters = groups with vote >= vmin. Returns dict of (nsites, nS) tensors."""
        torch, DEV = self.torch, self.DEV
        a = self.a
        nS = int(2 * tau) + 1
        Sv = S0 - tau + torch.arange(nS, device=DEV, dtype=torch.float64)
        rel = PV["t_c"][:, None] - Sv[None, :]                       # CPA reception in recording seconds
        inv = (rel >= w0 + a.edge_s) & (rel <= w1 - a.edge_s)
        inb = (rel >= w0 + a.miss_margin) & (rel <= w1 - a.miss_margin)
        V = torch.where(inv, PV["V"], torch.zeros_like(PV["V"]))
        base = torch.where(inb, PV["base"][:, None].expand_as(V), torch.zeros_like(V))
        site, fid, tc = PV["site"], PV["fid"], PV["t_c"]
        # one aircraft = one vote per site
        key = site * (int(fid.max()) + 1) + fid
        uk, inv1 = torch.unique(key, return_inverse=True)
        Va = torch.zeros(uk.numel(), nS, device=DEV).scatter_reduce_(0, inv1[:, None].expand(-1, nS), V, "amax")
        tca = torch.zeros(uk.numel(), device=DEV, dtype=torch.float64).scatter_reduce_(0, inv1, tc, "amin",
                                                                                       include_self=False)
        sa = uk // (int(fid.max()) + 1)
        # one sound = one vote: aircraft groups at a site with CPAs within excl_s chain into one group (max)
        o = torch.argsort(sa.double() * 1e10 + (tca - tca.min()))
        sa, tca, Va = sa[o], tca[o], Va[o]
        new = torch.ones(sa.numel(), dtype=torch.bool, device=DEV)
        if sa.numel() > 1:
            new[1:] = (sa[1:] != sa[:-1]) | ((tca[1:] - tca[:-1]) > a.excl_s)
        gid = torch.cumsum(new.long(), 0) - 1
        ng = int(gid.max()) + 1 if gid.numel() else 0
        Vg = torch.zeros(ng, nS, device=DEV).scatter_reduce_(0, gid[:, None].expand(-1, nS), Va, "amax")
        gsite = torch.zeros(ng, dtype=torch.long, device=DEV).scatter_(0, gid, sa)
        tot = torch.zeros(nsites, nS, device=DEV).index_add_(0, gsite, Vg)
        tot.index_add_(0, site, base)
        nv = torch.zeros(nsites, nS, device=DEV).index_add_(0, gsite, (Vg >= a.vmin).float())
        return tot, nv

    # -- one variant ---------------------------------------------------------------------------------------------
    def run_variant(self, name, shifts, subsets):
        """shifts[i]: seconds added to session i's clock (0 = real). Returns the per-subset results."""
        a = self.a
        t_var = time.time()
        blocks, surv, info = [], [], []
        for i, s in enumerate(self.sessions):
            S0 = s.clock + shifts[i]
            B, missing = self.cache.block(S0 - 900, S0 + s.dur + 400)
            blocks.append(B)
            keys = self.stage_a(s, B, S0) if B is not None else np.zeros(0, np.int64)
            surv.append(keys)
            info.append(dict(missing_days=missing, n_surv=int(keys.size)))
        U = np.unique(np.concatenate(surv)) if surv else np.zeros(0, np.int64)
        res = dict(variant=name, shifts=shifts, sessions=info, n_union=int(U.size), subsets={})
        if U.size == 0:
            res["runtime_s"] = round(time.time() - t_var, 1)
            return res
        la, lo = self.lat.latlon_of_key(U)
        sites = dict(key=U.copy(), lat=la, lon=lo, origin=[None] * U.size)
        member = np.stack([np.isin(U, k) for k in surv], 0)            # (n_sessions, n_sites)
        tabs = [self.stage_b(s, B, s.clock + shifts[i], sites) if B is not None else None
                for i, (s, B) in enumerate(zip(self.sessions, blocks))]
        n_coarse = U.size
        out = self.rank_all(sites, member, tabs, subsets, None)
        # refinement around each subset's top cells (two levels)
        for lvl, (sp, rad) in enumerate(((500.0, 2000.0), (125.0, 500.0))):
            new = self.refine_sites(out, sp, rad, top=a.refine_top if lvl == 0 else max(2, a.refine_top // 3))
            if new is None:
                break
            for k in ("key", "lat", "lon"):
                sites[k] = np.r_[sites[k], new[k]]
            sites["origin"] += new["origin"]
            member = np.concatenate([member, np.zeros((member.shape[0], new["key"].size), bool)], 1)
            add = [self.stage_b(s, B, s.clock + shifts[i], new) if B is not None else None
                   for i, (s, B) in enumerate(zip(self.sessions, blocks))]
            for i in range(len(tabs)):
                if tabs[i] is None or add[i] is None:
                    continue
                for kk in tabs[i]:
                    tabs[i][kk] = np.concatenate([tabs[i][kk], add[i][kk]], -1)
            out = self.rank_all(sites, member, tabs, subsets, n_coarse)
        res["subsets"] = out
        res["runtime_s"] = round(time.time() - t_var, 1)
        return res

    def set_block(self, B):
        self.M.set_block(B)
        ci = {"jet": 0, "prop": 1, "rotor": 2}
        self.blk_cls = self.torch.as_tensor(np.array([ci.get(str(c), 3) for c in B["f_cls"]], np.int64),
                                            device=self.DEV)

    def stage_a(self, s, B, S0):
        """Candidate cells of one session at clock S0 (see module doc)."""
        torch, a = self.torch, self.a
        tau = a.tau
        self.set_block(B)
        keys = []
        npass = 0
        for te in s.events:
            tr0 = S0 + te - tau - a.ev_sig
            tr1 = S0 + te + tau + a.ev_sig
            m = (B["t"] >= tr0 - 35 - 90) & (B["t"] <= tr1 + 90)
            fids = np.unique(B["fid"][m])
            if fids.size == 0:
                continue
            R = resample(B, fids, tr0 - 35 - 90, tr1 + 90)
            if R is None:
                continue
            sub = Sub(R, torch, self.DEV)
            fp = self.find_passes(sub, None, None, t_lo=tr0 - 35, t_hi=tr1, gen_cells=True)
            if fp is None:
                continue
            site, samp, skey, sobs = fp
            obs_t = torch.as_tensor(sobs, device=self.DEV, dtype=torch.float64)
            # survival on the audio evidence alone (the hint term would pass every pass at a detection)
            PV = self.pass_votes(s, sub, obs_t, site, samp, S0, tau, tc_lo=tr0, tc_hi=tr1, hint=False)
            if PV is None:
                continue
            npass += PV["site"].numel()
            # vote at the event: the pass's best vote over the clock window, CPA near the event
            v = PV["V"].amax(1)
            keep = v >= a.surv_vote
            keys.append(skey[PV["site"][keep].cpu().numpy()])
            del sub, PV
        nloud = 0
        for te in (s.loud_cand if self.L is not None else []):
            # loud event (p_air >= the loud channel's 1 false alarm / clean hour point): cells near the mirror points
            # of every low aircraft whose CPA reception can match it; survival on the loud LLR of that event alone
            ev = self.L.nearest(s.loud, te)
            sig = self.L.sig_of(ev)
            tr0 = S0 + te - tau - 3 * sig
            tr1 = S0 + te + tau + 3 * sig
            m = (B["t"] >= tr0 - 35 - 90) & (B["t"] <= tr1 + 90)
            fids = np.unique(B["fid"][m])
            if fids.size == 0:
                continue
            R = resample(B, fids, tr0 - 35 - 90, tr1 + 90)
            if R is None:
                continue
            sub = Sub(R, torch, self.DEV)
            fp = self.find_passes(sub, None, None, t_lo=tr0 - 35, t_hi=tr1, gen_cells=True)
            if fp is None:
                continue
            site, samp, skey, sobs = fp
            obs_t = torch.as_tensor(sobs, device=self.DEV, dtype=torch.float64)
            PV = self.pass_votes(s, sub, obs_t, site, samp, S0, tau, tc_lo=tr0, tc_hi=tr1, hint=False)
            if PV is None:
                continue
            Tl = self.L.event_terms(ev, PV, S0, tau, s.loud_rho0, self.blk_cls)        # (P, nS)
            keep = Tl.amax(1) >= a.surv_vote
            nloud += int(keep.sum())
            keys.append(skey[PV["site"][keep].cpu().numpy()])
            del sub, PV
        k = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
        log(f"  stage A {s.oid} @ {S0 - s.clock:+.0f} s: {len(s.events)} events, {npass} event passes, "
            f"{len(s.loud_cand) if self.L is not None else 0} loud events ({nloud} loud passes), {k.size} cells")
        return k

    def stage_b(self, s, B, S0, sites):
        """All passes of the session window at the given sites -> per window x tau: best score over the clock,
        voters there, best >=2-voter score and best >=1-voter score (numpy arrays)."""
        torch, a = self.torch, self.a
        tau = a.tau
        self.set_block(B)
        key = np.asarray(sites["key"])
        o = np.argsort(key, kind="stable")
        skey = key[o]
        obs = self.site_arrays(np.asarray(sites["lat"])[o], np.asarray(sites["lon"])[o])
        # flights that come near any site during the recording (+ margins)
        t0, t1 = S0 - tau - 120, S0 + s.dur + tau + 60
        m = (B["t"] >= t0) & (B["t"] <= t1)
        ki, kj = self.lat.ij(B["lat"][m], B["lon"][m])
        di, dj = self.lat.offsets(a.dmax + 1500)
        near = np.unique((np.unique(skey)[:, None] // self.lat.nj + di[None, :]) * self.lat.nj
                         + (np.unique(skey)[:, None] % self.lat.nj + dj[None, :]))
        hit = np.isin(self.lat.key(ki, kj), near)
        fids = np.unique(B["fid"][m][hit])
        n = key.size
        W = s.windows
        taus = [float(x) for x in a.taus.split(",")]
        chans = self.channels()
        out = {f"{p}{k}": np.full((len(W), len(taus), n), NEG if k != "nv" else 0, np.float32)
               for p in chans for k in ("best", "nv", "s2", "s1")}
        if fids.size == 0:
            # no traffic: score 0 everywhere, no voters
            for p in chans:
                out[p + "best"][:] = 0.0
            return out
        R = resample(B, fids, t0 - 60, t1 + 30)
        if R is None:
            for p in chans:
                out[p + "best"][:] = 0.0
            return out
        sub = Sub(R, torch, self.DEV)
        tb = time.time()
        fp = self.find_passes(sub, skey, obs)
        if fp is None:
            for p in chans:
                out[p + "best"][:] = 0.0
            return out
        site, samp, _, _ = fp
        tb1 = time.time()
        obs_t = torch.as_tensor(obs, device=self.DEV, dtype=torch.float64)
        PV = self.pass_votes(s, sub, obs_t, site, samp, S0, tau)
        if PV is None:
            for p in chans:
                out[p + "best"][:] = 0.0
            return out
        tb2 = time.time()
        inv = np.empty(n, np.int64)
        inv[o] = np.arange(n)
        for wi, w in enumerate(W):
            tot, nv = self.aggregate(s, PV, n, S0, tau, w["t0"], w["t1"])
            ch = {"": (tot, nv)}
            if self.L is not None:
                lt, nl, extra, nc, miss = self.L.site_terms(s, PV, n, S0, tau, w["t0"], w["t1"], a, self.blk_cls)
                ch["l_"] = (lt + miss, nl)
                for wl in self.loud_ws():
                    ch[f"c{wl:g}_"] = (tot + wl * (extra + miss), nv + nc)
            for p, (TT, NN) in ch.items():
                for ti, tq in enumerate(taus):
                    c0, c1 = int(tau - tq), int(tau + tq) + 1
                    T_, N_ = TT[:, c0:c1], NN[:, c0:c1]
                    b, ib = T_.max(1)
                    s2 = torch.where(N_ >= 2, T_, torch.full_like(T_, NEG)).max(1).values
                    s1 = torch.where(N_ >= 1, T_, torch.full_like(T_, NEG)).max(1).values
                    out[p + "best"][wi, ti] = b.cpu().numpy()[inv]
                    out[p + "nv"][wi, ti] = N_.gather(1, ib[:, None])[:, 0].cpu().numpy()[inv]
                    out[p + "s2"][wi, ti] = s2.cpu().numpy()[inv]
                    out[p + "s1"][wi, ti] = s1.cpu().numpy()[inv]
        log(f"  stage B {s.oid}: {n} sites, {fids.size} flights, {PV['site'].numel()} passes; join {tb1 - tb:.1f} s, "
            f"templates+votes {tb2 - tb1:.1f} s, aggregate {time.time() - tb2:.1f} s")
        del sub, PV
        return out

    def loud_ws(self):
        return [float(x) for x in self.a.loud_w.split(",")] if self.L is not None else []

    def channels(self):
        """score channels: "" tonal (round-5b LLR), "l_" loud only, "c<w>_" tonal + w * loud"""
        return [""] + (["l_"] + [f"c{w:g}_" for w in self.loud_ws()] if self.L is not None else [])

    # -- ranking ---------------------------------------------------------------------------------------------------
    def rank_all(self, sites, member, tabs, subsets, refine_off):
        """Per subset x window kind x tau: joint score per site -> best site and ranked regions."""
        a = self.a
        taus = [float(x) for x in a.taus.split(",")]
        rid = np.atleast_1d(self.regions.of(sites["lat"], sites["lon"]))
        out = {}
        nsite = sites["key"].size
        for si_, X in enumerate(subsets):
            # candidate sites: coarse cells from the subset's own sessions + refined sites spawned by this subset
            cand = member[X].any(0)
            if refine_off is not None:
                org = sites["origin"]
                cand[refine_off:] = np.array([o is not None and si_ in o for o in org[refine_off:]], bool)
            for kind in ("60", "30", "10"):
                # window index per session: the session's first window of this kind
                wsel = []
                for i in X:
                    ws = [j for j, w in enumerate(self.sessions[i].windows) if w["kind"] == kind]
                    wsel.append(ws[0] if ws else None)
                if any(w is None for w in wsel) or any(tabs[i] is None for i in X):
                    continue
                for ti, tq in enumerate(taus):
                    res = {}
                    for p in self.channels():
                        if len(X) == 1:
                            sc = tabs[X[0]][p + "s2"][wsel[0], ti].copy()
                            s1 = tabs[X[0]][p + "s1"][wsel[0], ti].copy() if p == "" else None
                        else:
                            sc = np.zeros(nsite, np.float64)
                            nv = np.zeros(nsite)
                            for i, wj in zip(X, wsel):
                                sc += tabs[i][p + "best"][wj, ti]
                                nv += tabs[i][p + "nv"][wj, ti]
                            sc = np.where(nv >= 2, sc, NEG)
                            s1 = None
                        sc = np.where(cand, sc, NEG)
                        r = self.regions_ranked(sc, sites, rid, s1=np.where(cand, s1, NEG) if s1 is not None else None)
                        if p == "":
                            res.update(r)                     # "s2" / "s1": tonal (unchanged keys)
                        else:
                            res[p + "s2"] = r["s2"]
                    out[f"{si_}|{kind}|{int(tq)}"] = res
        return out

    def regions_ranked(self, sc, sites, rid, s1=None):
        a = self.a
        rid = np.atleast_1d(rid)
        sc = np.atleast_1d(sc)
        s1 = None if s1 is None else np.atleast_1d(s1)
        res = {}
        for name, x in (("s2", sc), ("s1", s1)):
            if x is None:
                continue
            ok = x > NEG / 2
            if not ok.any():
                res[name] = dict(top=[], n_scored=0)
                continue
            idx = np.flatnonzero(ok)
            o = idx[np.argsort(-x[idx])]
            seen, top = set(), []
            for j in o:
                r = int(rid[j])
                if r in seen:
                    continue
                seen.add(r)
                top.append([r, round(float(x[j]), 3), round(float(sites["lat"][j]), 5),
                            round(float(sites["lon"][j]), 5), int(j)])
                if len(top) >= a.top_regions:
                    break
            res[name] = dict(top=top, n_scored=int(ok.sum()), n_regions=len(set(rid[idx].tolist())))
        return res

    def refine_sites(self, out, spacing, radius, top):
        lat, lon, origin = [], [], []
        seen = {}
        for k, v in out.items():
            si_ = int(k.split("|")[0])
            for r in [x for ch in v for x in v[ch].get("top", [])[:top] if ch.endswith("s2")]:
                la0, lo0 = r[2], r[3]
                n = int(radius // spacing)
                dla = spacing / 111132.0
                dlo = spacing / (111320.0 * math.cos(math.radians(la0)))
                for i in range(-n, n + 1):
                    for j in range(-n, n + 1):
                        la, lo = round(la0 + i * dla, 6), round(lo0 + j * dlo, 6)
                        kk = (la, lo)
                        if kk in seen:
                            origin[seen[kk]].add(si_)
                            continue
                        seen[kk] = len(lat)
                        lat.append(la)
                        lon.append(lo)
                        origin.append({si_})
        if not lat:
            return None
        lat, lon = np.array(lat), np.array(lon)
        ki, kj = self.lat.ij(lat, lon)
        return dict(key=self.lat.key(ki, kj), lat=lat, lon=lon, origin=origin)

    # -- per-recording chance calibration of the statistic (r5b llr_calibrate analogue) ----------------------------
    def calibrate(self, s):
        torch, a = self.torch, self.a
        S0 = s.clock
        B, _ = self.cache.block(S0 - 300, S0 + s.dur + 300)
        rng = np.random.default_rng(a.seed)
        stats = None
        if B is not None and B["t"].size:
            m = np.flatnonzero((B["t"] >= S0) & (B["t"] <= S0 + s.dur))
            if m.size:
                pick = rng.choice(m, min(400, m.size), replace=False)
                ki, kj = self.lat.ij(B["lat"][pick] + rng.normal(0, 0.03, pick.size),
                                     B["lon"][pick] + rng.normal(0, 0.04, pick.size))
                keys = np.unique(self.lat.key(ki, kj))
                la, lo = self.lat.latlon_of_key(keys)
                fl = np.unique(B["fid"][(B["t"] >= S0 - 200) & (B["t"] <= S0 + s.dur + 100)])
                self.M.set_block(B)
                R = resample(B, fl, S0 - 200, S0 + s.dur + 100)
                if R is not None:
                    sub = Sub(R, torch, self.DEV)
                    fp = self.find_passes(sub, keys, self.site_arrays(la, lo))
                    if fp is not None:
                        site, samp, _, sobs = fp
                        g = torch.Generator(device="cpu").manual_seed(0)
                        sel = torch.randperm(site.numel(), generator=g)[:a.calib_n].to(self.DEV)
                        obs_t = torch.as_tensor(sobs, device=self.DEV, dtype=torch.float64)
                        stats = self.M.calib_stats(s, sub, obs_t, site[sel], samp[sel], self.G, self.c, a)
        self.M.calibrate(s, stats)
        log(f"  calibration {s.oid}: {0 if stats is None else stats[1]} random passes; LLR(floor) "
            f"{float(s.llr[0]):.2f}, LLR(s=20) {float(s.llr[min(15, s.llr.numel() - 1)]):.2f}")


def params(**kw):
    a = types.SimpleNamespace(**DEFAULTS)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def run_group(group_path, out_path, root, adsb, static, model, use_guard=True, loud=None, verbose=True, **kw):
    """Search one group (groups/<gid>.json: {gid, sessions: [oid], subsets: [[i, ...]]}); resumable: finished
    variants of an interrupted run are kept. root holds blind/<oid>/ and decoy_days.json."""
    VERBOSE[0] = verbose
    a = params(root=root, adsb=adsb, static=static, model=model, avail=available_days(adsb), **kw)
    grp = load_json(group_path)
    oids = grp["sessions"]
    if use_guard:
        allowed = [os.path.join(root, "blind", o) for o in oids] + [
            adsb, static, os.path.join(root, "decoy_days.json"), group_path, model]
        guard.install(root, [os.path.join(root, "truth")], allowed, [os.path.dirname(os.path.abspath(out_path))])
    regions = {load_json(os.path.join(root, "blind", o, "input.json"))["region_prior"] for o in oids}
    if len(regions) != 1:
        raise ValueError("every session of a group must have the same country")
    region = regions.pop()
    random.seed(a.seed)
    t0 = time.time()
    L = Locator(a, region, oids, loud=loud)
    for s in L.sessions:
        L.calibrate(s)
    subsets = grp.get("subsets") or [[i] for i in range(len(oids))]
    dp = os.path.join(root, "decoy_days.json")
    dd = load_json(dp) if os.path.exists(dp) else {}
    variants = [("real", [0.0] * len(oids))]
    for m in CLOCK_OFFS_MIN[:a.k_clock]:
        variants.append((f"clk{m:+d}m", [60.0 * m] * len(oids)))
    for k in range(a.k_day):
        sh = []
        for o in oids:
            offs = dd.get(o, [])
            sh.append(86400.0 * offs[k] if k < len(offs) else None)
        if all(x is not None for x in sh):
            variants.append((f"day{k}", sh))
    if a.variants:
        keep = set(a.variants.split(","))
        variants = [v for v in variants if v[0] in keep]
    pv = {k: v for k, v in vars(a).items() if k not in ("avail",)}
    out = dict(gid=grp.get("gid"), sessions=oids, subsets=subsets, region=region, params=pv,
               session_info=[dict(oid=s.oid, n_events=len(s.events), dur=s.dur, windows=s.windows)
                             for s in L.sessions], variants=[])
    if os.path.exists(out_path):                     # resume: keep finished variants of an interrupted run
        try:
            old = load_json(out_path)
            if old.get("subsets") == subsets and old.get("sessions") == oids:
                out["variants"] = [v for v in old["variants"] if v["variant"] in {n for n, _ in variants}]
                log(f"resuming: {len(out['variants'])} variants already done")
        except (OSError, ValueError):
            pass
    done_v = {v["variant"] for v in out["variants"]}
    for name, sh in variants:
        if name in done_v:
            continue
        r = L.run_variant(name, sh, subsets)
        out["variants"].append(r)
        log(f"variant {name}: union {r['n_union']} cells, {r['runtime_s']} s")
        out["runtime_s"] = round(time.time() - t0, 1)
        out["touched"] = sorted(guard.TOUCHED)
        dump_json(out, out_path)
    out["done"] = True
    out["touched"] = sorted(guard.TOUCHED)
    dump_json(out, out_path)
    log(f"done in {time.time() - t0:.0f} s")
    return out
