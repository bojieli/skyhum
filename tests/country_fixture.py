"""Synthetic country-search outputs: two tiny region rasters, a truth file and saved search outputs. No real data."""
import json
import os

import numpy as np

# AU: 2 x 2 raster, step 0.5 deg, west half SA4 318, east half SA4 103 (codes from the shipped ABS table)
AU = dict(lat0=-30.0, lon0=140.0, rid=[[0, 1], [0, 1]],
          table=[dict(id=0, key="sa4:318", name="West", kind="sa4", pop=0, cells=2, lat=-29.5, lon=140.25),
                 dict(id=1, key="sa4:103", name="East", kind="sa4", pop=0, cells=2, lat=-29.5, lon=140.75)])
US = dict(lat0=40.0, lon0=-100.0, rid=[[0, 1], [0, 1]],
          table=[dict(id=0, key="cbsa:1", name="Alpha, XX", kind="metro", pop=2000000, cells=2, lat=40.5, lon=-99.75),
                 dict(id=1, key="cbsa:2", name="Beta, XX", kind="metro", pop=500000, cells=2, lat=40.5, lon=-99.25)])
CLOCK = 1767268800.0          # 2026-01-01T12:00:00Z
WIN = [dict(kind="60", t0=0.0, t1=3600.0, w=0), dict(kind="30", t0=0.0, t1=1800.0, w=1)]


def write_static(d):
    os.makedirs(d, exist_ok=True)
    for R, spec in (("AU", AU), ("CONUS", US)):
        np.savez(os.path.join(d, f"regions_{R}.npz"), lat0=spec["lat0"], lon0=spec["lon0"], step=0.5,
                 rid=np.array(spec["rid"], np.int32))
        with open(os.path.join(d, f"regions_{R}.json"), "w") as f:
            json.dump(spec["table"], f)
    np.savez(os.path.join(d, "county_pop.npz"), lat=np.array([40.5, 40.6, 40.5]), lon=np.array([-99.7, -99.8, -99.2]),
             pop=np.array([1500000, 500000, 500000]))
    return d


def top(rid, score, lat, lon):
    return [rid, score, lat, lon, 0]


def variant(name, tops, shift=0.0, n=1):
    return dict(variant=name, shifts=[shift] * n, sessions=[dict(missing_days=[])] * n, n_union=len(tops),
                subsets={f"{i}|{k}|{t}": dict(s2=dict(top=tops), s1=dict(top=tops))
                         for i in range(n) for k in ("60", "30") for t in (5, 30)})


def write_runs(root):
    """Four single-session AU groups at site A (west) and two US groups; returns (runs dir, truth path)."""
    runs = os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    os.makedirs(os.path.join(root, "truth"), exist_ok=True)
    truth = {}
    groups = [
        # oid, set, site, lat, lon, region, real top list, decoy top score
        ("a1", "a2o", "s1", -29.6, 140.2, "AU", [top(0, 12.0, -29.61, 140.21), top(1, 3.0, -29.5, 140.7)], 2.0),
        ("a2", "a2o", "s1", -29.6, 140.2, "AU", [top(1, 4.0, -29.5, 140.7), top(0, 3.5, -29.6, 140.3)], 6.0),
        ("a3", "a2o", "s2", -29.7, 140.8, "AU", [top(1, 9.0, -29.7, 140.81)], 1.0),
        ("a4", "a2o", "s3", -29.7, 140.3, "AU", [], 1.0),
        ("w1", "webcam", "c1", 40.6, -99.8, "CONUS", [top(0, 7.0, 40.61, -99.79)], 3.0),
        ("w2", "webcam", "c2", 40.6, -99.2, "CONUS", [top(0, 2.0, 40.61, -99.79)], 5.0),
    ]
    for oid, st, site, la, lo, reg, tops, dtop in groups:
        truth[oid] = dict(case_id="case_" + oid, set=st, site=site, lat=la, lon=lo, clock=CLOCK, dur=3600.0,
                          clock_verified=True, region=reg, windows=WIN,
                          pos=[dict(t=900.0, cls="prop"), dict(t=2000.0, cls="rotor")])
        dec = [variant(f"clk{m:+d}m", [top(0, dtop, la, lo)], 60.0 * m) for m in (10, -10, 15, -15)]
        out = dict(gid="s" + oid, sessions=[oid], subsets=[[0]], region=reg, done=True, touched=[],
                   variants=[variant("real", tops)] + dec)
        with open(os.path.join(runs, f"s{oid}.json"), "w") as f:
            json.dump(out, f)
    tp = os.path.join(root, "truth", "truth.json")
    with open(tp, "w") as f:
        json.dump(truth, f)
    return runs, tp
