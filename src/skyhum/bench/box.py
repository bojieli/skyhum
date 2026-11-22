"""Local runner of the box benchmark (round-5b locator in a jittered box), with the clock-decoy null.

Per case and clock mode: the search box is the truth +- U(0.25 deg) per axis (seeded per case, seed 1; the box never
sits on the truth), half-width 0.35 deg; the clock window is the start +- 30 min (clk30m), +- 30 s (clk30s), +- 5 s
(clk5s, single event: one voter is enough), or the daylight window of the local date (blind). Clock decoys (clk
modes): the same recording, box and arguments with the window moved by D (K = 20 offsets nearest 0 whose assumed audio
span never overlaps the true one, grid 120 s / 900 s), run in a fixed per-case order and stopped after the first decoy
that reaches the real top score (p is exact below 2 / 21).

Inputs per case: <audio_dir>/<id>.<ext>, <hints_dir>/<id>.json (detector v3), and the low-altitude ADS-B day files
(skyhum adsb fetch) cut here to the box + 0.1 deg. Differences from the research inputs: the research cut its tracks
from per-site trace dirs without the 10,000 ft ceiling and used Copernicus GLO-30 terrain (pass --dem-dir); both can
move individual answers. Outputs: <root>/runs/<tag>/<id>.<mode>.json (+ <tag>__decoy/).
"""
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

BOX = 0.35
JITTER, SEED = 0.25, 1
CLK = {"clk30m": 1800.0, "clk30s": 30.0, "clk5s": 5.0}
CLK_GRID = {"clk30s": (60.0, 120.0), "clk30m": (300.0, 900.0), "clk5s": (60.0, 120.0)}
SE_ARGS = ["--min-voters", "1", "--top", "10", "--top2", "4", "--nms-km", "1"]


def jitter_offset(cid, J=JITTER, seed=SEED):
    h = hashlib.sha1(f"jitter|{seed}|{cid}".encode()).digest()
    u = [int.from_bytes(h[k:k + 8], "big") / 2.0 ** 64 for k in (0, 8)]
    return (2 * u[0] - 1) * J, (2 * u[1] - 1) * J


def sun_times(lat, lon, day):
    """(sunrise, sunset) UTC epoch for local solar date `day` (NOAA approximation)."""
    d0 = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    la = math.radians(lat)
    x = math.cos(math.radians(90.833)) / (math.cos(la) * math.cos(decl)) - math.tan(la) * math.tan(decl)
    noon = d0 + (720 - 4 * lon - eqt) * 60
    if x <= -1:
        return noon - 43200, noon + 43200
    if x >= 1:
        return noon, noon
    ha = math.degrees(math.acos(x))
    return noon - 4 * ha * 60, noon + 4 * ha * 60


def day_window(c):
    """Blind clock window: daylight of the local solar date (+-30 min), or the whole date at night."""
    solar = dt.datetime.fromtimestamp(c["start_true"] + c["lon"] / 15 * 3600, dt.timezone.utc).date()
    sr, ss = sun_times(c["lat"], c["lon"], solar)
    w0, w1 = sr - 1800, ss + 1800 - c["dur"]
    if not (w0 <= c["start_true"] <= w1):
        d0 = dt.datetime(solar.year, solar.month, solar.day, tzinfo=dt.timezone.utc).timestamp() - c["lon"] / 15 * 3600
        w0, w1 = d0, d0 + 86400 - c["dur"]
    return w0, w1


def clock_window(c, mode):
    if mode in CLK:
        return c["start_true"] - CLK[mode], c["start_true"] + CLK[mode]
    return day_window(c)


def clock_offsets(c, mode, K=20):
    W = CLK[mode]
    margin, step = CLK_GRID[mode]
    lo, hi = day_window(c)
    lo, hi = lo + W, hi - W
    dmin = c["dur"] + 2 * W + margin
    cand = []
    for j in range(400):
        for s in (1, -1):
            d = s * (dmin + j * step)
            if lo <= c["start_true"] + d <= hi:
                cand.append(d)
        if len(cand) >= K:
            break
    return sorted(cand[:K], key=lambda d: (abs(d), d))


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tracks_from_days(adsb_dir, region, bbox, t0, t1, out):
    """Cut region-wide low day files to a box and time span; writes the tracks.npz layout skyhum.match.r5b reads."""
    la0, la1, lo0, lo1 = bbox
    parts, meta, off = [], {k: [] for k in ("f_hex", "f_type", "f_reg", "f_cls", "f_desc")}, 0
    days = sorted({dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
                   for t in np.arange(t0, t1 + 3600, 3600.0)})
    for d in days:
        p = os.path.join(adsb_dir, f"{d}_{region}_low.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"ADS-B day file missing: {p} (skyhum adsb fetch --date {d})")
        z = np.load(p)
        m = (z["t"] >= t0) & (z["t"] <= t1) & (z["lat"] >= la0) & (z["lat"] <= la1) & (z["lon"] >= lo0) & \
            (z["lon"] <= lo1)
        parts.append({k: z[k][m] for k in ("fid", "t", "lat", "lon", "h", "gs", "trk", "vr")})
        parts[-1]["fid"] = parts[-1]["fid"].astype(np.int64) + off
        for k in meta:
            meta[k].append(z[k])
        off += len(z["f_hex"])
    X = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    used, inv = np.unique(X["fid"], return_inverse=True)
    M = {k: np.concatenate(v)[used] for k, v in meta.items()}
    np.savez_compressed(out, fid=inv.astype(np.int32), t=X["t"], lat=X["lat"], lon=X["lon"], h=X["h"].astype(float),
                        gs=X["gs"].astype(float), trk=X["trk"].astype(float), vr=X["vr"].astype(float), **M,
                        meta=json.dumps(dict(t0=float(t0), t1=float(t1), bbox=list(bbox), src="adsb.lol (ODbL 1.0)")))
    return out


def write_spec(audio, out):
    from skyhum._vendor.afg import audio as A
    x, fs, _ = A.load_audio(audio, fs=4000)
    sp = A.compute_spectrogram(x, fs, win_s=1.0, hop_s=0.2, fmax=620.0)
    np.savez(out, W=sp["W"].astype(np.float16), f=sp["f"], t=sp["t"], hop_s=sp["hop_s"], duration=sp["duration"])
    return float(sp["duration"])


def _locate(argv, log_path):
    with open(log_path, "w") as lf:
        r = subprocess.run([sys.executable, "-m", "skyhum.match.r5b", *argv], stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def _top(out):
    try:
        with open(out + ".json") as f:
            H = json.load(f).get("hypotheses", [])
    except (OSError, ValueError):
        return None
    return float(H[0]["score"]) if H else -1e9


def run_cases(cases, root, tag, modes, audio_dir, hints_dir, adsb_dir, region="AU", jobs=1, decoys=True, K=20,
              dem_dir=None, log=print):
    """Run every case x mode (resumable: finished outputs are kept). Returns the runs directory."""
    from skyhum.match.locator import frozen_args
    if not audio_dir or not hints_dir:
        raise SystemExit("the box benchmark needs --audio-dir and --hints-dir (or --runs to rescore outputs)")
    runs = os.path.join(root, "runs")
    rd, dd, prep = os.path.join(runs, tag), os.path.join(runs, tag + "__decoy"), os.path.join(root, "prep")
    for d in (rd, dd, prep):
        os.makedirs(d, exist_ok=True)
    from skyhum.country.stage import find_audio
    for c in cases:
        aud = find_audio(audio_dir, c)
        hp = os.path.join(hints_dir, c["id"] + ".json")
        if aud is None or not os.path.exists(hp):
            log(f"  {c['id']}: no audio or hints, skipped")
            continue
        sp = os.path.join(prep, c["id"] + ".spec.npz")
        if not os.path.exists(sp):
            c["dur"] = write_spec(aud, sp)
        else:
            c["dur"] = float(np.load(sp)["duration"])
        dla, dlo = jitter_offset(c["id"])
        la, lo = c["lat"] + dla, c["lon"] + dlo
        bbox = [la - BOX, la + BOX, lo - BOX, lo + BOX]
        trk = os.path.join(prep, c["id"] + ".tracks.npz")
        w0d, w1d = day_window(c)
        if not os.path.exists(trk):
            p = BOX + 0.1
            tracks_from_days(adsb_dir, region, [la - p, la + p, lo - p, lo + p], w0d - 1200, w1d + c["dur"] + 1200,
                             trk)
        for mode in modes:
            w0, w1 = clock_window(c, mode)
            base = ["--spec", sp, "--tracks", trk, "--bbox=" + ",".join(f"{v:.5f}" for v in bbox), "--plots", "0",
                    "--hints", hp] + frozen_args() + (SE_ARGS if mode == "clk5s" else []) + (
                        ["--dem-dir", dem_dir] if dem_dir else [])
            out = os.path.join(rd, f"{c['id']}.{mode}")
            if not os.path.exists(out + ".json"):
                rc = _locate(base + ["--start", iso(w0), "--end", iso(w1), "--out", out], out + ".log")
                with open(out + ".info.json", "w") as f:
                    json.dump(dict(w0=w0, w1=w1, window_h=(w1 - w0) / 3600, bbox=bbox, rc=rc), f)
            if not decoys or mode not in CLK:
                continue
            s_real = _top(out)
            specs = clock_offsets(c, mode, K)
            specs.sort(key=lambda v: hashlib.sha1(f"{c['id']}|{mode}|('clock', {v})".encode()).hexdigest())
            for v in specs:
                do = os.path.join(dd, f"{c['id']}.{mode}.c{int(v):+d}")
                if not os.path.exists(do + ".json"):
                    _locate(base + ["--start", iso(w0 + v), "--end", iso(w1 + v), "--out", do], do + ".log")
                s = _top(do)
                if s_real is not None and s is not None and s >= s_real:
                    break                                    # stop after the first exceedance
        log(f"  {c['id']}: done")
    return runs
