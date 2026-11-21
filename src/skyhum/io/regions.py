"""Place grammar shared by --claim, --where, --area and --versus.

  Named region   "Los Angeles basin"          bundled table, `skyhum regions list`
  Circle         37.77,-122.42,r=5km          r in km or m
  Box            bbox:-122.6,37.2,-121.7,38.0 west,south,east,north
  File           @area.geojson                Polygon / MultiPolygon / Feature / FeatureCollection
"""
import json
import math
import re
from importlib import resources

import numpy as np

from skyhum.errors import UsageError
from skyhum.io.geo import KM_PER_DEG_LAT, haversine_km, km_per_deg_lon

_CIRCLE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*r\s*=\s*(\d+(?:\.\d+)?)\s*(km|m)?)?\s*$",
                     re.I)


def load_regions():
    with resources.files("skyhum.data").joinpath("regions.json").open("r", encoding="utf-8") as f:
        return json.load(f)["regions"]


class Area:
    """A place: membership test, bounding box, size and a cell grid."""

    def __init__(self, kind, label, bbox, circle=None, polygons=None):
        self.kind, self.label = kind, label
        self.bbox = tuple(float(v) for v in bbox)          # west, south, east, north
        self.circle = circle                                # (lat, lon, r_km)
        self.polygons = polygons                            # list of list of rings [(lon, lat), ...]

    # ---------------------------------------------------------------- geometry
    @property
    def center(self):
        if self.circle:
            return self.circle[0], self.circle[1]
        w, s, e, n = self.bbox
        return (s + n) / 2, (w + e) / 2

    def extent_km(self):
        """Largest side of the bounding box (circle: its diameter)."""
        if self.circle:
            return 2 * self.circle[2]
        w, s, e, n = self.bbox
        lat = (s + n) / 2
        return max((n - s) * KM_PER_DEG_LAT, (e - w) * km_per_deg_lon(lat))

    def contains(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if self.circle:
            return haversine_km(self.circle[0], self.circle[1], lat, lon) <= self.circle[2] + 1e-9
        w, s, e, n = self.bbox
        inbox = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        if not self.polygons:
            return inbox
        out = np.zeros(lat.shape, bool)
        for rings in self.polygons:
            inside = _in_ring(lon, lat, rings[0])
            for hole in rings[1:]:
                inside &= ~_in_ring(lon, lat, hole)
            out |= inside
        return out & inbox

    def grid(self, spacing_km):
        """Cell centres inside the area on a regular grid (lat, lon arrays)."""
        w, s, e, n = self.bbox
        lat0 = (s + n) / 2
        dlat = spacing_km / KM_PER_DEG_LAT
        dlon = spacing_km / km_per_deg_lon(lat0)
        lats = np.arange(s + dlat / 2, n, dlat) if n - s > dlat else np.array([lat0])
        lons = np.arange(w + dlon / 2, e, dlon) if e - w > dlon else np.array([(w + e) / 2])
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        LA, LO = LA.ravel(), LO.ravel()
        m = self.contains(LA, LO)
        if not m.any():                                    # tiny area: use its centre
            c = self.center
            return np.array([c[0]]), np.array([c[1]])
        return LA[m], LO[m]

    def to_json(self):
        d = dict(kind=self.kind, label=self.label, bbox=list(self.bbox), extent_km=round(self.extent_km(), 2))
        if self.circle:
            d["circle"] = dict(lat=self.circle[0], lon=self.circle[1], r_km=self.circle[2])
        return d

    def __repr__(self):
        return f"Area({self.kind}, {self.label!r})"


def _in_ring(x, y, ring):
    ring = np.asarray(ring, float)
    xs, ys = ring[:, 0], ring[:, 1]
    inside = np.zeros(np.shape(x), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi)
        inside ^= cross
        j = i
    return inside


def _check_latlon(lat, lon, text):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise UsageError(f"{text!r}: latitude must be within -90..90 and longitude within -180..180 (lat first)")


def parse_place(text, default_radius_km=None):
    """Parse the place grammar into an Area. Raises UsageError with a readable message."""
    if text is None or not str(text).strip():
        raise UsageError("empty place")
    s = str(text).strip()
    if s.startswith("@"):
        return _from_geojson(s[1:])
    if s.lower().startswith("bbox:"):
        try:
            w, so, e, n = (float(v) for v in s[5:].split(","))
        except ValueError as ex:
            raise UsageError(f"{text!r}: use bbox:west,south,east,north") from ex
        _check_latlon(so, w, text)
        _check_latlon(n, e, text)
        if not (w < e and so < n):
            raise UsageError(f"{text!r}: need west < east and south < north (boxes across the 180th meridian are "
                             "not supported)")
        return Area("bbox", s, (w, so, e, n))
    m = _CIRCLE.match(s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        _check_latlon(lat, lon, text)
        if m.group(3) is None:
            if default_radius_km is None:
                raise UsageError(f"{text!r}: add a radius, for example {lat},{lon},r=2km")
            r = float(default_radius_km)
        else:
            r = float(m.group(3)) / (1000.0 if (m.group(4) or "km").lower() == "m" else 1.0)
        if r <= 0:
            raise UsageError(f"{text!r}: radius must be positive")
        dlat = r / KM_PER_DEG_LAT
        dlon = r / max(km_per_deg_lon(lat), 1e-6)
        return Area("circle", s, (lon - dlon, lat - dlat, lon + dlon, lat + dlat), circle=(lat, lon, r))
    key = s.lower()
    for reg in load_regions():
        if key == reg["name"].lower() or key in [a.lower() for a in reg.get("aliases", [])]:
            return Area("named", reg["name"], reg["bbox"])
    raise UsageError(f"unknown place {text!r}: use a named region (skyhum regions list), lat,lon,r=5km, "
                     "bbox:w,s,e,n or @file.geojson")


def _from_geojson(path):
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
    except OSError as e:
        raise UsageError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise UsageError(f"{path} is not valid JSON: {e}") from e
    geoms = []

    def walk(o):
        t = o.get("type") if isinstance(o, dict) else None
        if t == "FeatureCollection":
            for ft in o.get("features", []):
                walk(ft)
        elif t == "Feature":
            walk(o.get("geometry") or {})
        elif t == "Polygon":
            geoms.append(o["coordinates"])
        elif t == "MultiPolygon":
            geoms.extend(o["coordinates"])
        else:
            raise UsageError(f"{path}: only Polygon and MultiPolygon geometries are supported (got {t})")

    walk(g)
    if not geoms:
        raise UsageError(f"{path}: no polygon found")
    pts = np.array([p[:2] for poly in geoms for ring in poly for p in ring], float)
    for lon, lat in pts:
        _check_latlon(lat, lon, path)
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    if bbox[2] - bbox[0] > 180:
        raise UsageError(f"{path}: polygons across the 180th meridian are not supported")
    return Area("geojson", "@" + path, bbox, polygons=[[[tuple(p[:2]) for p in ring] for ring in poly]
                                                        for poly in geoms])


def displaced(area, distance_km, bearing_deg=90.0):
    """The same shape moved by distance_km (used for the 'nearby wrong claim' in audit and the selftest)."""
    lat, lon = area.center
    dn = distance_km * math.cos(math.radians(bearing_deg))
    de = distance_km * math.sin(math.radians(bearing_deg))
    nlat = lat + dn / KM_PER_DEG_LAT
    nlon = lon + de / km_per_deg_lon(lat)
    if area.circle:
        r = area.circle[2]
        return parse_place(f"{nlat:.5f},{nlon:.5f},r={r}km")
    w, s, e, n = area.bbox
    dy, dx = nlat - lat, nlon - lon
    return Area("bbox", f"{area.label} moved {distance_km:g} km", (w + dx, s + dy, e + dx, n + dy))
