"""Static, recording-independent tables of the country search, built from public files downloaded at run time.

  <static>/dem_<R>.npz       ground elevation (m) on a 30 arc-second grid from AWS Terrain Tiles (terrarium, zoom 8,
                             ~0.5 km pixels), bilinear; R = CONUS, AU
  <static>/regions_<R>.npz   region id per 30" cell: CONUS = US Census CBSA 2023 (metropolitan and micropolitan
                             statistical areas), cells outside every CBSA get their county ("rural"); AU = ABS SA4 2021
  <static>/regions_<R>.json  [{id, key, name, kind, pop, cells, lat, lon}] (pop: Census 2023 CBSA estimates / 2020
                             county population; AU: 0 here, the ABS ERP table ships with the package)
  <static>/county_pop.npz    US county centres of population (2020) for the population baseline

Public inputs (downloaded into <static>/raw unless already there; about 50 MB plus ~2,000 terrain tiles):
  US Census  cb_2023_us_cbsa_500k.zip, cb_2023_us_county_500k.zip (cartographic boundary files),
             cbsa-est2023-alldata.csv (population estimates), CenPop2020_Mean_CO.txt (centres of population)
  ABS        SA4_2021_AUST_SHP_GDA2020.zip (ASGS Edition 3 digital boundary files, CC BY 4.0)
  AWS        Terrain Tiles, terrarium PNG, zoom 8 (open data on AWS; sources listed by the tile provider)
Needs pyshp, rasterio and Pillow (pip install "skyhum[country]").
"""
import csv
import io
import json
import math
import os
import urllib.request
import zipfile

import numpy as np

from skyhum.country.common import BOX

STEP = 1.0 / 120.0
UA = "skyhum-static-build"
RAW_FILES = {
    "cb_2023_us_cbsa_500k.zip": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cbsa_500k.zip",
    "cb_2023_us_county_500k.zip": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip",
    "cbsa-est2023-alldata.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/metro/totals/"
                                "cbsa-est2023-alldata.csv",
    "CenPop2020_Mean_CO.txt": "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt",
    "SA4_2021_AUST_SHP_GDA2020.zip": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-"
                                     "standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-"
                                     "files/SA4_2021_AUST_SHP_GDA2020.zip",
}
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _get(url, path):
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)


def fetch_raw(raw, log=print, regions=("CONUS", "AU")):
    os.makedirs(raw, exist_ok=True)
    need = [k for k in RAW_FILES if ("AU" in regions if k.startswith("SA4") else "CONUS" in regions)]
    for name in need:
        p = os.path.join(raw, name)
        if not os.path.exists(p):
            log(f"downloading {name}")
            _get(RAW_FILES[name], p)


def grid(R):
    la0, la1, lo0, lo1 = BOX[R]
    lats = la0 + STEP * (np.arange(int(round((la1 - la0) / STEP))) + 0.5)
    lons = lo0 + STEP * (np.arange(int(round((lo1 - lo0) / STEP))) + 0.5)
    return lats, lons


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def dem(R, st, raw, z=8, log=print):
    from PIL import Image
    from scipy.ndimage import map_coordinates
    out = os.path.join(st, f"dem_{R}.npz")
    if os.path.exists(out):
        return
    la0, la1, lo0, lo1 = BOX[R]
    x0, y1 = tile_xy(la0, lo0, z)
    x1, y0 = tile_xy(la1, lo1, z)
    X = range(int(x0), int(x1) + 1)
    Y = range(int(y0), int(y1) + 1)
    mos = np.zeros((len(Y) * 256, len(X) * 256), np.float32)
    cache = os.path.join(raw, "terrarium")
    os.makedirs(cache, exist_ok=True)
    log(f"DEM {R}: {len(X) * len(Y)} terrain tiles")
    for j, y in enumerate(Y):
        for i, x in enumerate(X):
            p = os.path.join(cache, f"{z}_{x}_{y}.png")
            for _ in range(4):
                if os.path.exists(p):
                    break
                try:
                    _get(TILE_URL.format(z=z, x=x, y=y), p)
                except OSError:
                    pass
            a = np.asarray(Image.open(p).convert("RGB"), np.float32)
            mos[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    lats, lons = grid(R)
    n = 2 ** z
    px = ((lons + 180.0) / 360.0 * n - X[0]) * 256 - 0.5
    py = ((1.0 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2.0 * n - Y[0]) * 256 - 0.5
    PY, PX = np.meshgrid(py, px, indexing="ij")
    g = map_coordinates(mos, [PY.ravel(), PX.ravel()], order=1, mode="nearest").reshape(PY.shape)
    g = np.maximum(g, -100.0)   # sea: terrarium bathymetry -> sea level (receivers on boats / piers ~0 m)
    g = np.where(g < 0, 0.0, g)
    np.savez_compressed(out, lat0=BOX[R][0], lon0=BOX[R][2], step=STEP, ground=g.astype(np.int16))
    log(f"DEM {R}: {g.shape}, {float(g.min()):.0f} to {float(g.max()):.0f} m")


def rasterize(shapes_ids, R):
    from rasterio import features
    from rasterio.transform import from_origin
    la0, la1, lo0, lo1 = BOX[R]
    lats, lons = grid(R)
    tr = from_origin(lo0, la1, STEP, STEP)
    img = features.rasterize(shapes_ids, out_shape=(lats.size, lons.size), transform=tr, fill=-1, dtype="int32")
    return img[::-1]          # row 0 = southernmost (lat0)


def read_shp(zpath):
    import shapefile
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    base = [n for n in names if n.endswith(".shp")][0][:-4]
    r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                         shx=io.BytesIO(z.read(base + ".shx")))
    flds = [f[0] for f in r.fields[1:]]
    for sr in r.iterShapeRecords():
        if sr.shape.shapeType == 0:
            continue
        yield dict(zip(flds, sr.record)), sr.shape.__geo_interface__


def regions_conus(st, raw, log=print):
    out = os.path.join(st, "regions_CONUS.npz")
    if os.path.exists(out):
        return
    pop = {}
    with open(os.path.join(raw, "cbsa-est2023-alldata.csv"), encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["LSAD"] in ("Metropolitan Statistical Area", "Micropolitan Statistical Area"):
                pop[row["CBSA"]] = int(row["POPESTIMATE2023"])
    cpop = {}
    cc = []
    with open(os.path.join(raw, "CenPop2020_Mean_CO.txt"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fips = row["STATEFP"] + row["COUNTYFP"]
            cpop[fips] = int(row["POPULATION"])
            cc.append((float(row["LATITUDE"]), float(row["LONGITUDE"]), int(row["POPULATION"])))
    cc = np.array(cc)
    np.savez_compressed(os.path.join(st, "county_pop.npz"), lat=cc[:, 0], lon=cc[:, 1], pop=cc[:, 2])
    table, shp = [], []
    for rec, geo in read_shp(os.path.join(raw, "cb_2023_us_cbsa_500k.zip")):
        i = len(table)
        kind = "metro" if rec["LSAD"] == "M1" else "micro"
        table.append(dict(id=i, key="cbsa:" + rec["CBSAFP"], name=rec["NAME"], kind=kind,
                          pop=pop.get(rec["CBSAFP"], 0)))
        shp.append((geo, i))
    img = rasterize(shp, "CONUS")
    shp2 = []
    for rec, geo in read_shp(os.path.join(raw, "cb_2023_us_county_500k.zip")):
        fips = rec["STATEFP"] + rec["COUNTYFP"]
        if rec["STATEFP"] in ("02", "15", "60", "66", "69", "72", "78"):
            continue
        i = len(table)
        table.append(dict(id=i, key="county:" + fips, name=f"{rec['NAME']} County, {rec['STUSPS']} (rural)",
                          kind="rural", pop=cpop.get(fips, 0)))
        shp2.append((geo, i))
    img2 = rasterize(shp2, "CONUS")
    img = np.where(img >= 0, img, img2)
    finish(st, img, table, "CONUS", log)


def regions_au(st, raw, log=print):
    out = os.path.join(st, "regions_AU.npz")
    if os.path.exists(out):
        return
    table, shp = [], []
    for rec, geo in read_shp(os.path.join(raw, "SA4_2021_AUST_SHP_GDA2020.zip")):
        if geo is None or not geo.get("coordinates"):
            continue
        code = rec.get("SA4_CODE21")
        if not code or not code[:1].isdigit() or int(code) >= 900 or code.endswith("97") or code.endswith("99"):
            continue
        i = len(table)
        table.append(dict(id=i, key="sa4:" + code, name=rec.get("SA4_NAME21"), kind="sa4", pop=0))
        shp.append((geo, i))
    img = rasterize(shp, "AU")
    finish(st, img, table, "AU", log)


def finish(st, img, table, R, log=print):
    lats, lons = grid(R)
    for t in table:
        m = img == t["id"]
        n = int(m.sum())
        t["cells"] = n
        if n:
            ii, jj = np.nonzero(m)
            t["lat"], t["lon"] = float(lats[ii].mean()), float(lons[jj].mean())
            # the area centroid can fall outside; use the cell nearest to it inside the region
            k = int(np.argmin((lats[ii] - t["lat"]) ** 2 + (lons[jj] - t["lon"]) ** 2))
            t["lat"], t["lon"] = float(lats[ii[k]]), float(lons[jj[k]])
    np.savez_compressed(os.path.join(st, f"regions_{R}.npz"), lat0=BOX[R][0], lon0=BOX[R][2], step=STEP,
                        rid=img.astype(np.int32))
    with open(os.path.join(st, f"regions_{R}.json"), "w") as f:
        json.dump(table, f, indent=0)
    log(f"regions {R}: {len(table)} regions, {float((img >= 0).mean()):.3f} of cells covered")


def build(static, raw=None, regions=("CONUS", "AU"), dem_too=True, log=print):
    """Download what is missing and build every table that is not there yet. raw: an existing directory holding the
    public files (default <static>/raw)."""
    os.makedirs(static, exist_ok=True)
    raw = raw or os.path.join(static, "raw")
    fetch_raw(raw, log, regions)
    for R in regions:
        if dem_too:
            dem(R, static, raw, log=log)
    if "CONUS" in regions:
        regions_conus(static, raw, log)
    if "AU" in regions:
        regions_au(static, raw, log)


def ready(static, region):
    return all(os.path.exists(os.path.join(static, f)) for f in
               (f"dem_{region}.npz", f"regions_{region}.npz", f"regions_{region}.json")
               + (("county_pop.npz",) if region == "CONUS" else ()))
