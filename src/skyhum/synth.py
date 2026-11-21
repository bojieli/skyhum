"""Synthetic propeller fly-overs and matching synthetic ADS-B tracks (for the selftest, tests and demos).

Nothing here is real data. Flights fly straight and level; the audio at a chosen site is the sum of each flight's
blade-pass tone and two harmonics, Doppler-shifted at reception time, attenuated as 1/r, in low-frequency noise.
"""
import datetime as dt
import math

import numpy as np

from skyhum.io.geo import enu_to_ecef_vectors, geodetic_to_ecef, offset

C = 343.0
DEMO_SITE = (34.0500, -118.3000)           # a synthetic site inside the bundled "Los Angeles basin" box
DEMO_START = dt.datetime(2026, 9, 8, 14, 3, 0, tzinfo=dt.timezone.utc).timestamp()


def _flight(rng, site, t_cpa, d_h_km, alt_m, speed, heading_deg, half_s=700.0, dt_s=2.0):
    """Straight flight whose closest horizontal approach to `site` is d_h_km at time t_cpa (emission time)."""
    hd = math.radians(heading_deg)
    ux, uy = math.sin(hd), math.cos(hd)                       # east, north unit vector of travel
    px, py = -uy * d_h_km, ux * d_h_km                         # perpendicular offset of the CPA point (km)
    t = np.arange(t_cpa - half_s, t_cpa + half_s + dt_s / 2, dt_s)
    s_km = speed * (t - t_cpa) / 1000.0
    lat, lon = offset(site[0], site[1], py + uy * s_km, px + ux * s_km)
    n = t.size
    return dict(t=t, lat=lat, lon=lon, h=np.full(n, float(alt_m)), gs=np.full(n, float(speed)),
                trk=np.full(n, heading_deg % 360), vr=np.zeros(n), cls="prop", type="SYN",
                hex=f"syn{rng.integers(0, 16 ** 4):04x}", f0=float(rng.uniform(70, 130)))


def make_world(seed=0, site=DEMO_SITE, start=DEMO_START, dur_s=240.0, span_s=4800.0, n_bg=24,
               planted_at=(50.0, 120.0, 190.0)):
    """Flights: `planted_at` passes close to the site while the recording runs, plus n_bg flights at random times
    and places within 15 km. Track coverage: start - span_s .. start + dur_s + span_s."""
    rng = np.random.default_rng(seed)
    flights = []
    for off in planted_at:
        alt = rng.uniform(400, 700)
        dh = rng.uniform(0.6, 1.6)
        t_rx = start + off
        flights.append(_flight(rng, site, t_rx - math.hypot(dh * 1000, alt) / C, dh, alt, rng.uniform(55, 70),
                               rng.uniform(0, 360)))
    for _ in range(n_bg):
        t_c = rng.uniform(start - span_s + 800, start + dur_s + span_s - 800)
        if start - 120 < t_c < start + dur_s + 120:
            t_c += rng.choice([-1, 1]) * 900
        flights.append(_flight(rng, site, t_c, rng.uniform(0.3, 15.0), rng.uniform(300, 1200), rng.uniform(50, 75),
                               rng.uniform(0, 360)))
    t0, t1 = start - span_s, start + dur_s + span_s
    for f in flights:
        m = (f["t"] >= t0) & (f["t"] <= t1)
        for k in ("t", "lat", "lon", "h", "gs", "trk", "vr"):
            f[k] = f[k][m]
    flights = [f for f in flights if f["t"].size > 10]
    return dict(flights=flights, site=site, start=start, dur_s=dur_s, t0=t0, t1=t1, seed=seed)


def render(world, site=None, start=None, dur_s=None, fs=16000, amp_1km=0.01, noise=0.02, seed=1):
    """Audio heard at `site` (default: the world's site) from `start` for dur_s seconds."""
    site = site or world["site"]
    start = world["start"] if start is None else start
    dur_s = world["dur_s"] if dur_s is None else dur_s
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    tr = start + np.arange(n) / fs
    x = np.zeros(n)
    S = geodetic_to_ecef(site[0], site[1], 1.5)
    for f in world["flights"]:
        te = np.arange(f["t"][0], f["t"][-1], 0.05)
        lat = np.interp(te, f["t"], f["lat"])
        lon = np.interp(te, f["t"], f["lon"])
        P = geodetic_to_ecef(lat, lon, np.interp(te, f["t"], f["h"]))
        V = enu_to_ecef_vectors(lat, lon, f["gs"][0] * math.sin(math.radians(f["trk"][0])),
                                f["gs"][0] * math.cos(math.radians(f["trk"][0])), 0.0)
        D = P - S
        r = np.linalg.norm(D, axis=1)
        if r.min() > 12000:
            continue
        rdot = np.einsum("ij,ij->i", D, V) / r
        t_rx = te + r / C
        if t_rx[-1] < tr[0] or t_rx[0] > tr[-1]:
            continue
        fo = f["f0"] / (1 + rdot / C)
        a = amp_1km * (1000.0 / r)
        fi = np.interp(tr, t_rx, fo, left=0.0, right=0.0)
        ai = np.interp(tr, t_rx, a, left=0.0, right=0.0)
        ph = 2 * np.pi * np.cumsum(fi) / fs + rng.uniform(0, 2 * np.pi)
        for h, g in ((1, 1.0), (2, 0.6), (3, 0.35)):
            x += g * ai * np.sin(h * ph)
    from scipy.signal import lfilter
    wn = rng.standard_normal(n)
    brown = lfilter([1.0], [1.0, -0.995], wn)
    brown *= noise / (np.std(brown) + 1e-12)
    x += brown + 0.3 * noise * wn
    return (x / max(1.0, np.abs(x).max() * 1.05)).astype(np.float32)


def speechlike(n, fs=16000, seed=3, duty=0.5):
    """A crude voiced-speech stand-in: bursts of a 120-220 Hz harmonic buzz with syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    f0 = 160 + 40 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    v = sum(np.sin(k * ph) / k for k in range(1, 12))
    syl = (np.sin(2 * np.pi * 4 * t) > 0).astype(float)
    on = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(1.0, 3.0) * fs)
        if rng.random() < duty:
            on[i:i + L] = 1
        i += L
    return (0.1 * v * syl * on).astype(np.float32)
