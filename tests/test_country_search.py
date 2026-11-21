"""Country search, ADS-B day cut, staging and the guard on synthetic data (no real recording, no download)."""
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest


def trace(icao, typ, t0, pts, cat="A1"):
    """A readsb trace_full JSON: rows [dt, lat, lon, alt_ft|'ground', gs_kt, trk, flags, vr_fpm, {category}]."""
    tr = []
    for k, (dt, la, lo, alt) in enumerate(pts):
        tr.append([dt, la, lo, alt, 100.0, 90.0, 0, 0.0, {"category": cat} if k == 0 else None])
    return gzip.compress(json.dumps(dict(icao=icao, t=typ, r="N1", timestamp=t0, trace=tr)).encode())


def test_cut_trace_low_box_ground_and_classes():
    from skyhum.country.adsb_days import cut_trace
    types = {"C172": "L1P", "B738": "L2J", "R44": "H1P"}
    pts = [(i * 5.0, 34.0, -118.0 + i * 0.001, 3000) for i in range(6)] + [(30.0, 34.0, -117.99, "ground")] + \
          [(35.0 + i * 5, 34.0, -117.98 + i * 0.001, 12000) for i in range(3)]
    grp, meta, out = cut_trace(trace("a1", "C172", 1.7e9, pts), use=["CONUS", "AU"], iv={}, types=types)
    assert grp == "low" and meta["cls"] == "prop" and list(out) == ["CONUS"]
    assert len(out["CONUS"]) == 1 and out["CONUS"][0].shape[0] == 6           # ground and > 10,000 ft dropped
    assert cut_trace(trace("b1", "B738", 1.7e9, pts), ["CONUS"], {}, types)[0] == "jet"
    assert cut_trace(trace("c1", "R44", 1.7e9, pts), ["AU"], {}, types) is None            # outside the AU box
    iv = {"CONUS": [[1.7e9 + 100, 1.7e9 + 200]]}
    assert cut_trace(trace("a1", "C172", 1.7e9, pts), ["CONUS"], iv, types) is None        # outside the intervals


def test_cut_stream_writes_day_files(tmp_path):
    from skyhum.country.adsb_days import Chain, cut_stream, day_done
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for k in range(3):
            raw = trace(f"a{k}", "C172", 1.7e9, [(i * 5.0, 34.0 + k * 0.01, -118.0 + i * 0.001, 2000)
                                                  for i in range(5)])
            ti = tarfile.TarInfo(f"traces/{k:02x}/trace_full_a{k}.json")
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
    data = buf.getvalue()
    a, b = tmp_path / "part.aa", tmp_path / "part.ab"
    a.write_bytes(data[:1000])
    b.write_bytes(data[1000:])
    d = str(tmp_path / "adsb")
    c = cut_stream(Chain([str(a), str(b)]), d, "2023-11-14", regions=("CONUS", "AU"), workers=1,
                   types={"C172": "L1P"})
    assert c["CONUS_low"]["segments"] == 3 and c["AU_low"]["segments"] == 0 and day_done(d, "2023-11-14")
    z = np.load(os.path.join(d, "2023-11-14_CONUS_low.npz"))
    assert set(z["f_cls"]) == {"prop"} and z["t"].size == 15


def test_plan_merges_windows_and_decoy_days():
    from skyhum.country.adsb_days import nearest_other_days, plan
    t = 1.7e9                                                  # 2023-11-14T22:13:20Z: the window spills into 11-15
    inp = [dict(oid="a", region_prior="AU", clock_utc=t, dur_s=3600.0),
           dict(oid="b", region_prior="AU", clock_utc=t + 3 * 86400, dur_s=3600.0)]
    dd = nearest_other_days(inp)
    assert dd == {"a": [3], "b": [-3]}
    p = plan(inp, dd)
    assert sorted(p) == ["2023-11-14", "2023-11-15", "2023-11-17", "2023-11-18"]
    assert all(r == ["AU"] for r in (list(v) for v in p.values()))


def test_stage_windows():
    from skyhum.country.stage import windows
    w = windows(7200.0)
    assert [x["kind"] for x in w].count("60") == 2 and [x["kind"] for x in w].count("30") == 4
    assert windows(2400.0)[0]["note"] == "full<60" and windows(1200.0) == []


def test_guard_denies_truth_in_a_subprocess(tmp_path):
    root = tmp_path / "root"
    (root / "truth").mkdir(parents=True)
    (root / "truth" / "truth.json").write_text("{}")
    (root / "blind" / "x").mkdir(parents=True)
    (root / "blind" / "x" / "input.json").write_text("{}")
    (root / "blind" / "y").mkdir(parents=True)
    code = f"""
import os, sys
from skyhum.country import guard
r = {str(root)!r}
guard.install(r, [os.path.join(r, "truth")], [os.path.join(r, "blind", "x")], [])
open(os.path.join(r, "blind", "x", "input.json")).read()
out = []
for p in ("truth/truth.json", "blind/y"):
    try:
        os.listdir(os.path.join(r, p)) if p.endswith("y") else open(os.path.join(r, p))
        out.append("allowed")
    except PermissionError:
        out.append("denied")
print(",".join(out), len(guard.TOUCHED))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.stdout.strip() == "denied,denied 1", p.stderr


@pytest.mark.slow
def test_search_finds_the_synthetic_site(tmp_path):
    """Synthetic flights over a synthetic site: the country search ranks the site's region first (CPU, ~1 min)."""
    pytest.importorskip("torch")
    from country_fixture import write_static
    from skyhum.country import search, stage
    from skyhum.io.audio import save as write_audio
    from skyhum.synth import DEMO_SITE, make_world, render
    world = make_world(seed=3, dur_s=600.0, planted_at=(80.0, 250.0, 420.0), n_bg=10)
    root = str(tmp_path / "root")
    static = write_static(os.path.join(root, "static"))
    # one CONUS region around the synthetic site (the fixture raster sits elsewhere): 1 x 1 cell, step 1 deg
    np.savez(os.path.join(static, "regions_CONUS.npz"), lat0=33.5, lon0=-118.8, step=1.0,
             rid=np.zeros((1, 1), np.int32))
    np.savez(os.path.join(static, "dem_CONUS.npz"), lat0=33.5, lon0=-118.8, step=0.5,
             ground=np.zeros((2, 2), np.int16))
    adsb = os.path.join(root, "adsb")
    os.makedirs(adsb)
    import datetime as dt
    day = dt.datetime.fromtimestamp(world["start"], dt.timezone.utc).strftime("%Y-%m-%d")
    fl = world["flights"]
    np.savez(os.path.join(adsb, f"{day}_CONUS_low.npz"),
             fid=np.concatenate([np.full(f["t"].size, i) for i, f in enumerate(fl)]),
             **{k: np.concatenate([f[k] for f in fl]) for k in ("t", "lat", "lon", "h", "gs", "trk", "vr")},
             f_hex=np.array([f["hex"] for f in fl]), f_type=np.array(["C172"] * len(fl)),
             f_reg=np.array([""] * len(fl)), f_cls=np.array(["prop"] * len(fl)), f_desc=np.array(["L1P"] * len(fl)))
    x = render(world)
    wav = str(tmp_path / "rec.wav")
    write_audio(wav, x, 16000)
    hints = [dict(t=t + 8.0, f=f["f0"], p=0.95) for t, f in zip((80.0, 250.0, 420.0), fl[:3])]
    oid = stage.stage_session(root, "CONUS", world["start"], world["dur_s"], audio=wav, hints=hints,
                              wins=[dict(kind="60", t0=0.0, t1=600.0, w=0)])
    gp = stage.write_group(root, [oid])
    from skyhum.country.runner import paths
    out = os.path.join(root, "runs", "t", "g.json")
    os.makedirs(os.path.dirname(out))
    res = search.run_group(gp, out, root, adsb, static, paths(root)["model"], use_guard=False, verbose=False,
                           variants="real", calib_n=2000, k_day=0)
    top = res["variants"][0]["subsets"]["0|60|5"]["s2"]["top"]
    assert top and top[0][0] == 0
    from skyhum.country.common import haversine_km
    assert float(haversine_km(top[0][2], top[0][3], *DEMO_SITE)) < 25.0     # the region tier of the benchmark
