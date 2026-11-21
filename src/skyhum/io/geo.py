"""Small geodesy helpers (WGS84), numpy only."""
import numpy as np

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
KM_PER_DEG_LAT = 111.132


def geodetic_to_ecef(lat, lon, h):
    lat, lon, h = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)), np.asarray(h, float)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def enu_to_ecef_vectors(lat, lon, e, n, u):
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    x = -so * e - sl * co * n + cl * co * u
    y = co * e - sl * so * n + cl * so * u
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def km_per_deg_lon(lat):
    return 111.320 * np.cos(np.radians(lat))


def offset(lat, lon, north_km, east_km):
    return lat + north_km / KM_PER_DEG_LAT, lon + east_km / km_per_deg_lon(lat)
