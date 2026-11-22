"""Known-clock localisation from loud events and ADS-B tracks (single event and several events).

Geometry. A loud event is the CPA of one low aircraft. For a track sample at emission time t (position p, velocity
v, height h above ground), the sites whose CPA to that aircraft falls on this sample lie on the line through p
perpendicular to the ground track: lateral offset l on either side (the two mirror points), slant range
d = sqrt(l^2 + a^2 + h^2), a = h * v_up / v_ground (a climbing aircraft is closest to a point ahead of it).
The event is received at t + d / c. Every (sample, l) is one "entry": (grid cell, t_rx, d, v, class, aircraft).

Score of a site x and clock offset delta (recording time = UTC - start - delta):
  per event k:  s_k(x, delta) = max over entries j at x of
        log N(t_k ; t_rx_j - delta, sig_t) - log rho0                      timing against the event rate rho0
      + log N(ltau_k ; log(d_j / v_j), sig) - log U                        envelope width  -> range
      + [Doppler: log N(log beta ; log(v_j / c)) + log N(ltau_dop ; log(d_j / v_j))] - log U   when the HF fit is usable
      + log q_k[class_j] / prior[class_j]                                  class posterior
      - log(1 - pi(class_j, d_j))                                          this pass is not "missed"
  event term = max(0, log(1 - p_k + p_k exp(s_k))), capped; site score S(x) = max_delta sum_k event term
      + sum over all loud passes predicted at x in the window of log(1 - pi)   (missing-pass cost).
Several events at one site share one delta: repeated departures give a timing fingerprint, and aircraft with
different speeds and directions break the along-track / clock degeneracy. With one event the result is the pair of
mirror ridges of every low aircraft in the air at that time; the answer is then the best aircraft's two points.

Ported from the research loud-event channel (locate.py): the channel's own, minimal scorer. The research
evaluation driver (run_locate.py: registry cases, jittered boxes, decoy hours) is not part of the package.
"""
import math

import numpy as np
import torch

from skyhum.loud.features import C

CLS_IDX = {"jet": 0, "prop": 1, "rotor": 2}
LOG_U_LTAU = math.log(6.0)       # log tau uniform over [0.3, 120] s
LOG_U_LBETA = math.log(1.9)      # log beta uniform over [0.05, 0.33]
R_EARTH = 6371000.0


class Box:
    def __init__(self, lat0, lon0, half_deg, spacing=100.0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.cos(math.radians(lat0)) * R_EARTH * math.pi / 180.0
        self.ky = R_EARTH * math.pi / 180.0
        self.hx, self.hy = half_deg * self.kx, half_deg * self.ky
        self.sp = spacing
        self.nx, self.ny = int(2 * self.hx / spacing) + 1, int(2 * self.hy / spacing) + 1

    def en(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def latlon(self, e, n):
        return self.lat0 + n / self.ky, self.lon0 + e / self.kx

    def cell_en(self, cell):
        iy, ix = np.divmod(cell, self.nx)
        return ix * self.sp - self.hx, iy * self.sp - self.hy

    def cell_of(self, e, n):
        ix = np.rint((e + self.hx) / self.sp).astype(np.int64)
        iy = np.rint((n + self.hy) / self.sp).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy * self.nx + ix, ok


class Traffic:
    """Low, moving track samples (1 s) of one tracks.npz in the local frame of a box, expanded into entries."""

    def __init__(self, tracks, box, ground, t_lo, t_hi, shift=0.0, d_max=2000.0, v_min=20.0, lat_step=50.0):
        from skyhum._vendor.afg.tracks import Flights, Resampled
        fl = Flights(tracks)
        self.box, self.d_max = box, d_max
        self.empty = True
        try:
            R = Resampled(fl, dt=1.0, t0=t_lo - shift - 90, t1=t_hi - shift + 90)
        except RuntimeError:
            return
        h = R.h - ground
        spd = np.sqrt(R.gs ** 2 + R.vu ** 2)
        e, n = box.en(R.lat, R.lon)
        m = ((h <= d_max) & (spd >= v_min) & (R.gs >= 5.0) & (np.abs(e) <= box.hx + d_max)
             & (np.abs(n) <= box.hy + d_max))
        if not m.any():
            return
        t, e, n, h, spd = R.t[m] + shift, e[m], n[m], np.maximum(h[m], 2.0), spd[m]
        ux, uy = R.ve[m] / R.gs[m], R.vn[m] / R.gs[m]
        a = np.clip(h * R.vu[m] / R.gs[m], -400, 400)
        fid = R.fid[m]
        cls = np.array([CLS_IDX.get(str(c), 3) for c in fl.cls])[fid]
        L = np.arange(-d_max, d_max + 1, lat_step)
        px = (e + a * ux)[:, None] - L[None, :] * uy[:, None]
        py = (n + a * uy)[:, None] + L[None, :] * ux[:, None]
        d = np.sqrt(a[:, None] ** 2 + L[None, :] ** 2 + h[:, None] ** 2)
        cell, ok = box.cell_of(px, py)
        ok &= d <= d_max
        if not ok.any():
            return
        rep = lambda v: np.broadcast_to(v[:, None], d.shape)[ok]  # noqa: E731
        self.cell = cell[ok]
        self.px, self.py = px[ok].astype(np.float32), py[ok].astype(np.float32)     # exact entry positions (m)
        self.d = d[ok].astype(np.float32)
        self.t_rx = rep(t) + self.d / C
        self.v = rep(spd).astype(np.float32)
        self.fid = rep(fid).astype(np.int32)
        self.cls = rep(cls).astype(np.int8)
        self.side = np.sign(np.broadcast_to(L[None, :], d.shape)[ok]).astype(np.int8)
        self.air = rep((h > 100.0) & (spd > 30.0))
        self.hex = fl.hex
        self.type = fl.type
        self.fcls = fl.cls
        o = np.argsort(self.t_rx, kind="stable")
        for k in ("cell", "d", "t_rx", "v", "fid", "cls", "side", "air", "px", "py"):
            setattr(self, k, getattr(self, k)[o])
        self.empty = False
        self.n_aircraft = int(np.unique(self.fid).size)

    def window(self, t0, t1):
        a, b = np.searchsorted(self.t_rx, [t0, t1])
        return slice(int(a), int(b))


def pi_of(cal, cls, d, air):
    """Detection probability of a pass (dev table, shrunk); class index 3 (unknown type) uses the prop row and
    passes that are not airborne (ground rolls, very low) the ground row."""
    out = np.zeros(d.shape, np.float32)
    bins = np.asarray(cal["d_bins"], float)
    for ci, name in ((0, "jet"), (1, "prop"), (2, "rotor"), (3, "prop"), (-1, "ground")):
        tab = np.array([(r["k"] + 0.5) / (r["n"] + 2.0) for r in cal["pi"][name]], np.float32)
        m = (cls == ci) & air if ci >= 0 else ~air
        if m.any():
            b = np.clip(np.searchsorted(bins, d[m], side="right") - 1, 0, tab.size - 1)
            out[m] = np.where(d[m] < bins[-1], tab[b], 0.0)
    return np.minimum(out, 0.9)


def pass_loglik(ev, t_rx, d, v, cls, cal, rho0, pi=None):
    """Log-likelihood ratio of one event against candidate passes, WITHOUT the timing term.

    ev: evidence record (model.evidence_of). t_rx, d, v, cls: arrays of candidate passes (reception time of the CPA
    in recording seconds for the assumed clock, slant range m, speed m/s, class index jet 0 / prop 1 / rotor 2 /
    other 3). Returns (a, t_ev, sig_t): the caller adds -0.5 ((t_ev - t_rx) / sig_t)^2 and maximises over passes.
    """
    ltau_pred = np.log(np.maximum(d, 1.0) / np.maximum(v, 1.0))
    z = (ev["ltau"] - ltau_pred) / ev["sig_ltau"]
    a = -0.5 * z * z - math.log(ev["sig_ltau"] * math.sqrt(2 * math.pi)) + LOG_U_LTAU
    t_ev, sig_t = ev["t"], ev["sig_t"]
    dp = ev.get("dop")
    if dp:
        t_ev, sig_t = dp["t"], dp["sig_t"]
        zb = (math.log(dp["beta"]) - np.log(np.maximum(v, 1.0) / C)) / dp["sig_lbeta"]
        zt = (dp["ltau"] - ltau_pred) / dp["sig_ltau"]
        a = a - 0.5 * zb * zb - math.log(dp["sig_lbeta"] * math.sqrt(2 * math.pi)) + LOG_U_LBETA
        a = a - 0.5 * zt * zt - math.log(dp["sig_ltau"] * math.sqrt(2 * math.pi)) + LOG_U_LTAU
    q = np.array([ev["q"]["jet"], ev["q"]["prop"], ev["q"]["rotor"], 0.0])
    pr = np.array([cal["class_prior"]["jet"], cal["class_prior"]["prop"], cal["class_prior"]["rotor"], 1.0])
    lc = np.clip(np.log(np.maximum(q, 1e-3) / np.maximum(pr, 1e-3)), -3.0, 3.0)
    lc[3] = 0.0
    a = a + lc[cls]
    a = a - math.log(rho0 * sig_t * math.sqrt(2 * math.pi))
    if pi is not None:
        a = a - np.log1p(-pi)
    return a, t_ev, sig_t


def score(evs, tr, cal, centre, W, win=None, cap=10.0, miss=True, rho0=None, dev="cpu", keep_event_maps=False):
    """Site scores for a set of events. centre: clock offset at the middle of the search window (s), W: half width.
    win: (t0, t1) recording interval considered observed (for the missing-pass cost). Returns a dict."""
    box = tr.box
    ncell = box.nx * box.ny
    nb = int(2 * W) + 1
    d_lo = centre - W
    S = torch.zeros((ncell, nb), device=dev)
    if rho0 is None:
        rho0 = max(len(evs) / max(1.0, (win[1] - win[0]) if win else 3600.0), 1.0 / 900.0)
    maps = []
    for ev in evs:
        if tr.empty:
            break
        sig_t = ev["dop"]["sig_t"] if ev.get("dop") else ev["sig_t"]
        t_ev = ev["dop"]["t"] if ev.get("dop") else ev["t"]
        pad = W + 3 * sig_t
        sl = tr.window(t_ev + tr.t0 + centre - pad, t_ev + tr.t0 + centre + pad)
        if sl.stop <= sl.start:
            continue
        d, v, cls = tr.d[sl], tr.v[sl], tr.cls[sl]
        pi = pi_of(cal, cls, d, tr.air[sl]) if miss else None
        a, t_ev, sig_t = pass_loglik(ev, None, d, v, cls, cal, rho0, pi)
        dimp = (tr.t_rx[sl] - tr.t0) - t_ev                      # delta that puts this pass on the event
        cell = torch.as_tensor(tr.cell[sl], device=dev)
        a_t = torch.as_tensor(a.astype(np.float32), device=dev)
        b0 = torch.as_tensor(dimp - d_lo, device=dev, dtype=torch.float32)
        A = torch.full((ncell * nb,), -30.0, device=dev)
        M = int(math.ceil(3 * sig_t))
        bc = torch.round(b0)
        for m in range(-M, M + 1):
            b = bc + m
            ok = (b >= 0) & (b < nb)
            val = a_t - 0.5 * ((b - b0) / sig_t) ** 2
            idx = (cell * nb + b.long())[ok]
            A.scatter_reduce_(0, idx, val[ok], reduce="amax")
        p = float(min(max(ev["p_air"], 1e-4), 0.999))
        term = torch.clamp(torch.log1p(-p + p * torch.exp(torch.clamp(A, max=30.0)) - 0.0), min=0.0, max=cap)
        term = term.reshape(ncell, nb)
        S += term
        if keep_event_maps:
            maps.append(term.max(1).values.cpu().numpy())
    # mild preference for the centre of the clock window (ties only)
    bb = torch.arange(nb, device=dev, dtype=torch.float32)
    S = S - 0.25 * ((bb - W) / max(W, 1.0)) ** 2
    best, bi = S.max(1)
    best = best.cpu().numpy()
    delta = (bi.cpu().numpy() + d_lo).astype(np.float32)
    mc = np.zeros(ncell, np.float32)
    if miss and win is not None and not tr.empty:
        sl = tr.window(win[0] + tr.t0 + centre + 10, win[1] + tr.t0 + centre - 10)
        if sl.stop > sl.start:
            key = ((tr.fid[sl].astype(np.int64) * ncell + tr.cell[sl]) * 64
                   + ((tr.t_rx[sl] - tr.t0) // 120 % 64).astype(np.int64))
            o = np.lexsort((tr.d[sl], key))
            first = np.r_[True, key[o][1:] != key[o][:-1]]
            ii = o[first]
            pi = pi_of(cal, tr.cls[sl][ii], tr.d[sl][ii], tr.air[sl][ii])
            np.add.at(mc, tr.cell[sl][ii], np.log1p(-pi))
    return dict(S=best + mc, delta=delta, miss=mc, raw=best, maps=maps)


def hypotheses(res, box, n=5, nms_m=800.0):
    S = res["S"].copy()
    out = []
    r = int(nms_m / box.sp)
    S2 = S.reshape(box.ny, box.nx)
    for _ in range(n):
        k = int(np.argmax(S2))
        iy, ix = divmod(k, box.nx)
        if S2[iy, ix] <= 0:
            break
        e, nn = ix * box.sp - box.hx, iy * box.sp - box.hy
        # sub-grid refinement: parabola through the score at the cell and its two neighbours, per axis
        if 0 < ix < box.nx - 1 and 0 < iy < box.ny - 1:
            for axis in (0, 1):
                lo = S2[iy, ix - 1] if axis == 0 else S2[iy - 1, ix]
                hi = S2[iy, ix + 1] if axis == 0 else S2[iy + 1, ix]
                den = lo - 2 * S2[iy, ix] + hi
                if lo > -1e8 and hi > -1e8 and den < -1e-6:
                    off = float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5)) * box.sp
                    if axis == 0:
                        e += off
                    else:
                        nn += off
        lat, lon = box.latlon(e, nn)
        out.append(dict(score=float(S2[iy, ix]), e=float(e), n=float(nn), lat=float(lat), lon=float(lon),
                        delta=float(res["delta"][k]), miss=float(res["miss"][k])))
        S2[max(0, iy - r): iy + r + 1, max(0, ix - r): ix + r + 1] = -1e9
    return out


def single_event(ev, tr, cal, centre, W, rho0=1.0 / 600.0, top=6):
    """One event, known clock: rank the aircraft; for each give the two mirror points at the best clock.

    Returns a list of candidates {hex, type, cls, score, points: [(e, n), (e, n)], d, v, delta}."""
    if tr.empty:
        return []
    sig_t = ev["dop"]["sig_t"] if ev.get("dop") else ev["sig_t"]
    t_ev = ev["dop"]["t"] if ev.get("dop") else ev["t"]
    sl = tr.window(t_ev + tr.t0 + centre - W, t_ev + tr.t0 + centre + W)
    if sl.stop <= sl.start:
        return []
    d, v, cls, fid, side = tr.d[sl], tr.v[sl], tr.cls[sl], tr.fid[sl], tr.side[sl]
    a, t_ev, sig_t = pass_loglik(ev, None, d, v, cls, cal, rho0, None)
    dimp = (tr.t_rx[sl] - tr.t0) - t_ev
    a = a - 0.25 * ((dimp - centre) / max(W, 1.0)) ** 2            # centre preference (ties only)
    out = []
    for f in np.unique(fid):
        m = np.flatnonzero(fid == f)
        j = m[int(np.argmax(a[m]))]
        pts = []
        for sgn in (-1, 1):
            mm = m[side[m] == sgn] if sgn != 0 else m
            if mm.size == 0:
                mm = m[side[m] == 0]
            if mm.size == 0:
                continue
            jj = mm[int(np.argmax(a[mm]))]
            pts.append((float(tr.px[sl][jj]), float(tr.py[sl][jj]), float(a[jj])))   # exact, not the grid cell
        out.append(dict(hex=str(tr.hex[f]), type=str(tr.type[f]), cls=str(tr.fcls[f]), score=float(a[j]),
                        d=float(d[j]), v=float(v[j]), delta=float(dimp[j]), points=pts))
    out.sort(key=lambda r: -r["score"])
    return out[:top]
