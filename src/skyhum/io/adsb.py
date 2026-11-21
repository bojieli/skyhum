"""ADS-B track files.

This build reads one format: the per-day `tracks.npz` written by the upstream afg pipeline
(third_party/afg, `afg.adsb`), in SI units:

  fid (int, per sample)  flight index          t (float, epoch s)   lat, lon (deg)   h (m, altitude)
  gs (m/s)  trk (deg)  vr (m/s)                NaN where not broadcast
  f_hex, f_cls (per flight)                    ICAO hex and class: prop / rotor / jet / unknown
  meta (JSON string, optional)                 may hold t0 / t1 coverage bounds

Synthetic tracks from `skyhum selftest --write-fixtures` use the same layout. Automatic download
(`--adsb auto`, `skyhum adsb fetch`) is not implemented in this build.
"""
import json
import os

import numpy as np

from skyhum.errors import MissingData, NotImplementedInBuild, UsageError
from skyhum.io.geo import enu_to_ecef_vectors, geodetic_to_ecef

LOUD = ("prop", "rotor", "unknown", "")


class Tracks:
    def __init__(self, path):
        if path is None:
            raise UsageError("--adsb PATH is required: a tracks.npz day file covering the recording (docs/data.md)")
        if path == "auto":
            raise NotImplementedInBuild("--adsb auto (automatic ADS-B download)",
                                        "pass --adsb PATH to a tracks.npz day file")
        if not os.path.isfile(path):
            raise MissingData(f"ADS-B track file not found: {path}")
        try:
            z = np.load(path, allow_pickle=False)
            self.fid = z["fid"].astype(np.int64)
            self.t = z["t"].astype(np.float64)
            self.lat, self.lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
            self.h = z["h"].astype(np.float64)
            n = self.t.size
            self.gs = z["gs"].astype(np.float64) if "gs" in z else np.full(n, np.nan)
            self.trk = z["trk"].astype(np.float64) if "trk" in z else np.full(n, np.nan)
            self.vr = z["vr"].astype(np.float64) if "vr" in z else np.full(n, np.nan)
            nf = int(self.fid.max()) + 1 if n else 0
            self.hex = z["f_hex"].astype(str) if "f_hex" in z else np.array([f"f{i}" for i in range(nf)])
            self.cls = z["f_cls"].astype(str) if "f_cls" in z else np.array(["unknown"] * nf)
            self.meta = json.loads(str(z["meta"])) if "meta" in z else {}
        except (KeyError, ValueError, OSError) as e:
            raise MissingData(f"{path} is not a tracks.npz day file: {e}") from e
        self.path = path
        if self.t.size == 0:
            raise MissingData(f"{path} holds no samples")
        self.t0 = float(self.meta.get("t0", self.t.min()))
        self.t1 = float(self.meta.get("t1", self.t.max()))

    def covers(self, a, b):
        return self.t0 <= a and b <= self.t1

    def resampled(self, t_lo, t_hi, classes=LOUD, dt=1.0):
        """Uniform 1 s samples of every flight of the given classes in [t_lo, t_hi]: list of dicts with
        ecef (n,3), vel (n,3), t (n,), hex, cls."""
        out = []
        order = np.argsort(self.fid, kind="stable")
        f_sorted = self.fid[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(f_sorted)) + 1, f_sorted.size]
        for a, b in zip(cuts[:-1], cuts[1:]):
            ii = order[a:b]
            f = int(self.fid[ii[0]])
            cls = str(self.cls[f]) if f < len(self.cls) else "unknown"
            if cls not in classes:
                continue
            ii = ii[np.argsort(self.t[ii])]
            t = self.t[ii]
            t, u = np.unique(t, return_index=True)
            ii = ii[u]
            if t.size < 2 or t[-1] < t_lo or t[0] > t_hi:
                continue
            ga, gb = np.ceil(max(t[0], t_lo) / dt) * dt, np.floor(min(t[-1], t_hi) / dt) * dt
            if gb - ga < 2 * dt:
                continue
            tg = np.arange(ga, gb + dt / 2, dt)
            lat = np.interp(tg, t, self.lat[ii])
            lon = np.interp(tg, t, np.degrees(np.unwrap(np.radians(self.lon[ii]))))
            okh = np.isfinite(self.h[ii])
            if okh.sum() < 2:
                continue
            h = np.interp(tg, t[okh], self.h[ii][okh])
            gs, trk, vr = self.gs[ii], self.trk[ii], self.vr[ii]
            okv = np.isfinite(gs) & np.isfinite(trk)
            if okv.sum() >= 2:
                ve = np.interp(tg, t[okv], gs[okv] * np.sin(np.radians(trk[okv])))
                vn = np.interp(tg, t[okv], gs[okv] * np.cos(np.radians(trk[okv])))
            else:
                ve = np.gradient(lon, dt) * 111320.0 * np.cos(np.radians(lat))
                vn = np.gradient(lat, dt) * 111132.0
            okr = np.isfinite(vr)
            vu = np.interp(tg, t[okr], vr[okr]) if okr.sum() >= 2 else np.gradient(h, dt)
            out.append(dict(t=tg, ecef=geodetic_to_ecef(lat, lon, h), vel=enu_to_ecef_vectors(lat, lon, ve, vn, vu),
                            hex=str(self.hex[f]) if f < len(self.hex) else f"f{f}", cls=cls))
        return out


def write_tracks(path, flights, t0=None, t1=None, meta=None):
    """flights: list of dicts with t, lat, lon, h, gs, trk, vr (arrays), hex, cls. Writes the afg layout."""
    cols = {k: [] for k in ("fid", "t", "lat", "lon", "h", "gs", "trk", "vr")}
    for i, fl in enumerate(flights):
        n = len(fl["t"])
        cols["fid"].append(np.full(n, i, np.int32))
        for k in ("t", "lat", "lon", "h", "gs", "trk", "vr"):
            cols[k].append(np.asarray(fl[k], np.float64))
    m = dict(meta or {})
    if t0 is not None:
        m["t0"], m["t1"] = float(t0), float(t1)
    np.savez_compressed(path, **{k: np.concatenate(v) for k, v in cols.items()},
                        f_hex=np.array([f["hex"] for f in flights]), f_cls=np.array([f["cls"] for f in flights]),
                        f_type=np.array([f.get("type", "") for f in flights]),
                        f_reg=np.array([""] * len(flights)), f_desc=np.array([""] * len(flights)),
                        meta=json.dumps(m))
