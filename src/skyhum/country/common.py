"""Country boxes, the coarse search lattice, 30 arc-second rasters, region lookup and the work directory layout.

Work directory (``$SKYHUM_WORKDIR/country`` unless a path is given):

  static/      region rasters, DEM and population tables (``skyhum country static``)
  static/raw/  the public boundary and population files they are built from (downloaded at run time)
  adsb/        region-wide low-altitude ADS-B day files ``<day>_<REGION>_low.npz`` (``skyhum adsb fetch``)
  adsb/raw/    partial adsb.lol day archives while a download resumes
  blind/<oid>/ one recording: input.json (clock, duration, windows), spectrogram, detector hints
  groups/      groups of recordings known to come from one place
  truth/       truth.json for benchmark scoring (never opened by the search)
  runs/<tag>/  search outputs, one JSON per group
"""
import json
import math
import os

import numpy as np

BOX = {"CONUS": (24.0, 50.0, -125.5, -66.5), "AU": (-44.5, -9.5, 112.5, 154.5)}
LATC = {"CONUS": 37.0, "AU": -27.0}
COUNTRY = {"US": "CONUS", "AU": "AU", "CONUS": "CONUS"}
REGION_KIND = {"CONUS": "US Census CBSA 2023 (metropolitan / micropolitan area), else the county",
               "AU": "ABS Statistical Area Level 4 (SA4) 2021"}


def region_of_country(c):
    try:
        return COUNTRY[str(c).upper()]
    except KeyError as e:
        raise ValueError(f"country must be US or AU, not {c!r}") from e


def workdir(path=None):
    """The country-search work directory (created)."""
    d = path or os.path.join(os.environ.get("SKYHUM_WORKDIR") or os.path.expanduser("~/.local/share/skyhum"),
                             "country")
    d = os.path.abspath(os.path.expanduser(d))
    os.makedirs(d, exist_ok=True)
    return d


class Lattice:
    """Regular lat/lon lattice with ~spacing m cells (dlon fixed at the region's centre latitude)."""

    def __init__(self, region, spacing=2000.0):
        self.region = region
        self.la0, self.la1, self.lo0, self.lo1 = BOX[region]
        self.spacing = spacing
        self.dlat = spacing / 111132.0
        self.dlon = spacing / (111320.0 * math.cos(math.radians(LATC[region])))
        self.ni = int(np.ceil((self.la1 - self.la0) / self.dlat)) + 1
        self.nj = int(np.ceil((self.lo1 - self.lo0) / self.dlon)) + 1

    def ij(self, lat, lon):
        return (np.floor((np.asarray(lat) - self.la0) / self.dlat + 0.5).astype(np.int64),
                np.floor((np.asarray(lon) - self.lo0) / self.dlon + 0.5).astype(np.int64))

    def key(self, i, j):
        return np.asarray(i, np.int64) * self.nj + np.asarray(j, np.int64)

    def latlon_of_key(self, k):
        k = np.asarray(k, np.int64)
        return self.la0 + (k // self.nj) * self.dlat, self.lo0 + (k % self.nj) * self.dlon

    def offsets(self, radius_m):
        """(di, dj) lattice offsets that can hold a point within radius_m (conservative over the region)."""
        cmin = min(math.cos(math.radians(self.la0)), math.cos(math.radians(self.la1)))
        ri = int(math.ceil(radius_m / self.spacing)) + 1
        rj = int(math.ceil(radius_m / (111320.0 * cmin * self.dlon))) + 1
        di, dj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1), indexing="ij")
        return di.ravel(), dj.ravel()


class Raster:
    """30 arc-second raster (static.py): nearest for ids, bilinear for the DEM."""

    def __init__(self, path, key):
        z = np.load(path)
        self.a = z[key]
        self.lat0, self.lon0, self.step = float(z["lat0"]), float(z["lon0"]), float(z["step"])

    def nearest(self, lat, lon):
        i = np.clip(np.floor((np.asarray(lat) - self.lat0) / self.step).astype(int), 0, self.a.shape[0] - 1)
        j = np.clip(np.floor((np.asarray(lon) - self.lon0) / self.step).astype(int), 0, self.a.shape[1] - 1)
        return self.a[i, j]

    def bilinear(self, lat, lon):
        y = (np.asarray(lat, float) - self.lat0) / self.step - 0.5
        x = (np.asarray(lon, float) - self.lon0) / self.step - 0.5
        i = np.clip(np.floor(y).astype(int), 0, self.a.shape[0] - 2)
        j = np.clip(np.floor(x).astype(int), 0, self.a.shape[1] - 2)
        u = np.clip(y - i, 0, 1)
        v = np.clip(x - j, 0, 1)
        A = self.a.astype(np.float32)
        return ((1 - u) * (1 - v) * A[i, j] + u * (1 - v) * A[i + 1, j] + (1 - u) * v * A[i, j + 1]
                + u * v * A[i + 1, j + 1])


class Regions:
    """Region lookup: CONUS = Census CBSA (else the county, "rural"), AU = ABS SA4."""

    def __init__(self, region, static):
        self.region = region
        self.r = Raster(os.path.join(static, f"regions_{region}.npz"), "rid")
        with open(os.path.join(static, f"regions_{region}.json")) as f:
            self.table = json.load(f)

    def of(self, lat, lon, search=6):
        """Region id; points on water or just outside a polygon take the nearest region within ~search cells
        (30" each, ~5 km), else -1."""
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        rid = np.asarray(self.r.nearest(lat, lon)).copy()
        bad = np.flatnonzero(rid < 0)
        if bad.size:
            st = self.r.step
            for k in bad:
                best = None
                for rad in range(1, search + 1):
                    oi, oj = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1), indexing="ij")
                    v = self.r.nearest(lat[k] + oi.ravel() * st, lon[k] + oj.ravel() * st)
                    ok = v >= 0
                    if ok.any():
                        d = (oi.ravel()[ok]) ** 2 + (oj.ravel()[ok] * np.cos(np.radians(lat[k]))) ** 2
                        best = int(v[ok][np.argmin(d)])
                        break
                rid[k] = -1 if best is None else best
        return rid if rid.size > 1 else rid.reshape(())

    def name(self, rid):
        return self.table[int(rid)]["name"] if rid >= 0 else "none (water / outside)"


def haversine_km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = p2 - p1, np.radians(np.asarray(lo2) - np.asarray(lo1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def md_table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def frac(k, n):
    return f"{k}/{n} ({100 * k / n:.0f} %)" if n else "-"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)
