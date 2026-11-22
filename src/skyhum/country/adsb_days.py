"""Region-wide low-altitude ADS-B day files from the adsb.lol public daily releases (ODbL 1.0).

adsb.lol publishes one global archive per UTC day (github.com/adsblol/globe_history_<year>, release
``v<YYYY.MM.DD>-planes-readsb-prod-0``; 2023 days are one ~0.9 GB tar, recent days several GB in split parts).
Nothing is downloaded unless you run ``skyhum adsb fetch``. A download goes to ``<workdir>/adsb/raw/<day>/`` and
resumes after an interruption (HTTP range requests); once complete, the archive is cut to

  <workdir>/adsb/<day>_<REGION>_low.npz   prop / rotor / unknown aircraft, airborne points at or below 10,000 ft
                                          (pressure altitude, else geometric) inside the country box, split into
                                          segments (breaks at readsb new-leg / stale flags, gaps > 45 s, dropped
                                          points), >= 3 points each; h = geometric altitude if present, else baro
  <workdir>/adsb/<day>_<REGION>_jet.npz   jets, the same way (reference only; the search never reads it)
  <workdir>/adsb/<day>.json               counts

and the raw archive is deleted (``--keep-raw`` keeps it). The cut is by a fixed country box and, optionally, by time
intervals (a recording's window plus the clock-decoy margins); never by any location inside the country.
Aircraft classes follow ICAO Doc 8643 description codes (type table from the tar1090-db project, downloaded to
``<workdir>/adsb/icao_aircraft_types.js`` on first use) and the ADS-B emitter category.
Derived files are a database derived from adsb.lol: share them under ODbL 1.0 and attribute adsb.lol.
"""
import datetime as dt
import gzip
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from multiprocessing import Pool

import numpy as np

from skyhum.country.common import BOX

FT, KT, FPM = 0.3048, 0.514444, 0.3048 / 60.0
ALT_MAX_FT = 10000.0
MAX_GAP = 45.0
UA = "skyhum-adsb-fetch"
TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types.js"

# ---------------------------------------------------------------------------------------------------------------
# aircraft classes (from afg.adsb, audio-flyover-geolocation by AlanBacker, MIT; see THIRD_PARTY_NOTICES.md)
# ---------------------------------------------------------------------------------------------------------------
_FALLBACK_TYPES = {
    "C150": "L1P", "C152": "L1P", "C172": "L1P", "C182": "L1P", "C206": "L1P", "C210": "L1P",
    "P28A": "L1P", "P28R": "L1P", "PA32": "L1P", "PA34": "L2P", "PA44": "L2P", "PA46": "L1P",
    "SR20": "L1P", "SR22": "L1P", "BE35": "L1P", "BE36": "L1P", "BE58": "L2P", "M20P": "L1P",
    "DA40": "L1P", "DA42": "L2P", "C208": "L1T", "PC12": "L1T", "TBM7": "L1T", "TBM9": "L1T",
    "BE20": "L2T", "BE9L": "L2T", "B350": "L2T", "DH8A": "L2T", "DH8B": "L2T", "DH8C": "L2T",
    "DH8D": "L2T", "AT43": "L2T", "AT45": "L2T", "AT72": "L2T", "AT75": "L2T", "AT76": "L2T",
    "SF34": "L2T", "C130": "L4T", "DHC6": "L2T", "E120": "L2T",
    "R22": "H1P", "R44": "H1P", "R66": "H1T", "EC35": "H2T", "EC45": "H2T", "AS50": "H1T",
    "B06": "H1T", "B407": "H1T", "S76": "H2T", "A139": "H2T", "H60": "H2T",
    "A319": "L2J", "A320": "L2J", "A321": "L2J", "B737": "L2J", "B738": "L2J", "B739": "L2J",
    "B38M": "L2J", "B39M": "L2J", "E170": "L2J", "E75L": "L2J", "CRJ2": "L2J", "CRJ7": "L2J",
    "CRJ9": "L2J", "B752": "L2J", "B763": "L2J", "B772": "L2J", "B77W": "L2J", "B788": "L2J",
    "B789": "L2J", "A332": "L2J", "A333": "L2J", "A359": "L2J", "C25A": "L2J", "C56X": "L2J",
    "CL30": "L2J", "GLF4": "L2J", "GLF5": "L2J", "GLEX": "L2J", "E55P": "L2J", "PC24": "L2J",
}


def parse_types(obj):
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k.upper()] = v.get("desc", "")
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k.upper()] = v[1]
    return out


def load_type_table(adsb_dir, allow_download=True):
    """ICAO type designator -> Doc 8643 description code (C172 -> L1P). Falls back to a small built-in table."""
    p = os.path.join(adsb_dir, "icao_aircraft_types.js")
    if not os.path.exists(p) and allow_download:
        try:
            os.makedirs(adsb_dir, exist_ok=True)
            req = urllib.request.Request(TYPES_URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r, open(p + ".tmp", "wb") as f:
                f.write(r.read())
            os.replace(p + ".tmp", p)
        except (OSError, urllib.error.URLError):
            pass
    if os.path.exists(p):
        with open(p, "rb") as f:
            b = f.read()
        if b[:2] == b"\x1f\x8b":
            b = gzip.decompress(b)
        try:
            return parse_types(json.loads(b))
        except ValueError:
            pass
    return dict(_FALLBACK_TYPES)


def classify(type_code, types, category=None):
    """-> 'prop' | 'rotor' | 'jet' | 'glider' | 'balloon' | 'unknown'"""
    desc = types.get((type_code or "").upper(), "") if type_code else ""
    if desc and len(desc) >= 3:
        c0, eng = desc[0], desc[2]
        if c0 in "HG":
            return "rotor"
        if eng == "J":
            return "jet"
        if eng in "PTE":
            return "prop"
    cat = (category or "").upper()
    if cat == "A7":
        return "rotor"
    if cat in ("A3", "A4", "A5", "A6"):
        return "jet"
    if cat == "B1":
        return "glider"
    if cat == "B2":
        return "balloon"
    if (type_code or "").upper() in ("GLID",):
        return "glider"
    if (type_code or "").upper() in ("BALL",):
        return "balloon"
    return "unknown"


# ---------------------------------------------------------------------------------------------------------------
# releases
# ---------------------------------------------------------------------------------------------------------------
def release_tag(day):
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def release_urls(day, adsb_dir, refresh=False):
    """Download URLs of the adsb.lol release for a UTC day (sorted; split parts concatenate in this order), or []
    when there is no release. Cached in <adsb_dir>/releases.json."""
    cp = os.path.join(adsb_dir, "releases.json")
    cache = {}
    if os.path.exists(cp):
        with open(cp) as f:
            cache = json.load(f)
    if day in cache and not refresh:
        return cache[day]
    y = day[:4]
    url = f"https://api.github.com/repos/adsblol/globe_history_{y}/releases/tags/{release_tag(day)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        urls = sorted(a["browser_download_url"] for a in d.get("assets", []))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        urls = []
    cache[day] = urls
    os.makedirs(adsb_dir, exist_ok=True)
    with open(cp + ".tmp", "w") as f:
        json.dump(cache, f, indent=0)
    os.replace(cp + ".tmp", cp)
    return urls


def download(url, path, chunk=1 << 20, retries=6, progress=None):
    """Resumable download: bytes go to path.part, continued with a range request; renamed when complete."""
    part = path + ".part"
    for k in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0                       # server ignored the range: start over
                total = r.headers.get("Content-Range", "").split("/")[-1] if have else r.headers.get("Content-Length")
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        if progress:
                            progress(f.tell(), int(total) if total and total.isdigit() else None)
            if total and total.isdigit() and os.path.getsize(part) != int(total):
                raise OSError(f"short download {os.path.getsize(part)} of {total}")
            os.replace(part, path)
            return path
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:             # already complete
                os.replace(part, path)
                return path
            if k == retries - 1:
                raise
        except OSError:
            if k == retries - 1:
                raise
        time.sleep(min(60, 5 * 2 ** k))
    return path


class Chain:
    """Read several files as one stream (split tar parts)."""

    def __init__(self, paths):
        self.paths, self.f = list(paths), None

    def read(self, n=-1):
        out = b""
        while n < 0 or len(out) < n:
            if self.f is None:
                if not self.paths:
                    break
                self.f = open(self.paths.pop(0), "rb")
            b = self.f.read(-1 if n < 0 else n - len(out))
            if not b:
                self.f.close()
                self.f = None
                continue
            out += b
        return out


# ---------------------------------------------------------------------------------------------------------------
# low-altitude cut (one trace file -> segments per region)
# ---------------------------------------------------------------------------------------------------------------
_W = {}


def _init(use, iv, types):
    _W.update(use=use, iv=iv, types=types)


def _f(v):
    return np.nan if v is None or isinstance(v, str) else float(v)


def cut_trace(raw, use=None, iv=None, types=None):
    """One readsb trace_full JSON (gzip or plain) -> (group, meta, {region: [segments]}) or None."""
    use = use if use is not None else _W["use"]
    iv = iv if iv is not None else _W["iv"]
    types = types if types is not None else _W["types"]
    try:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        d = json.loads(raw)
    except ValueError:
        return None
    except OSError:
        return None
    tr = d.get("trace") or []
    if len(tr) < 3:
        return None
    lat = np.array([_f(e[1]) for e in tr])
    lon = np.array([_f(e[2]) for e in tr])
    inreg = {}
    for r in use:
        a, b, c, e = BOX[r]
        m = (lat >= a) & (lat <= b) & (lon >= c) & (lon <= e)
        if m.sum() >= 3:
            inreg[r] = m
    if not inreg:
        return None
    cat = None
    for e in tr:
        if len(e) > 8 and isinstance(e[8], dict) and e[8].get("category"):
            cat = e[8]["category"]
            break
    typ = d.get("t") or ""
    cls = classify(typ, types, cat)
    if cls in ("glider", "balloon"):
        return None
    grp = "jet" if cls == "jet" else "low"
    base = float(d.get("timestamp", 0))
    tt = base + np.array([e[0] for e in tr], float)
    alt_raw = [e[3] for e in tr]
    ground = np.array([a == "ground" for a in alt_raw])
    alt_b = np.array([_f(a) for a in alt_raw]) * FT
    alt_g = np.array([_f(e[10]) if len(e) > 10 else np.nan for e in tr]) * FT
    gs = np.array([_f(e[4]) for e in tr]) * KT
    trk = np.array([_f(e[5]) for e in tr])
    flags = np.array([int(e[6] or 0) if len(e) > 6 and isinstance(e[6], int) else 0 for e in tr])
    vr = np.array([_f(e[7]) if len(e) > 7 else np.nan for e in tr]) * FPM
    gr = np.array([_f(e[11]) if len(e) > 11 else np.nan for e in tr]) * FPM
    vr = np.where(np.isfinite(gr), gr, vr)
    alt_sel = np.where(np.isfinite(alt_b), alt_b, alt_g)
    h = np.where(np.isfinite(alt_g), alt_g, alt_b)
    low = np.isfinite(alt_sel) & (alt_sel <= ALT_MAX_FT * FT) & np.isfinite(h) & ~ground & np.isfinite(lat)
    brk = ((flags & 2) > 0) | ((flags & 1) > 0)
    meta = dict(hex=d.get("icao", ""), type=typ, reg=d.get("r") or "", category=cat or "",
                desc=types.get(typ.upper(), "") if typ else "", cls=cls)
    out = {}
    for r, m in inreg.items():
        keep = m & low
        if iv.get(r):
            tin = np.zeros(tt.size, bool)
            for a, b in iv[r]:
                tin |= (tt >= a) & (tt <= b)
            keep &= tin
        if keep.sum() < 3:
            continue
        idx = np.flatnonzero(keep)
        cut = np.zeros(idx.size, bool)
        cut[1:] = (np.diff(idx) > 1) | (np.diff(tt[idx]) > MAX_GAP) | brk[idx][1:]
        starts = np.flatnonzero(cut | (np.arange(idx.size) == 0))
        ends = np.append(starts[1:], idx.size)
        segs = []
        for s, e in zip(starts, ends):
            if e - s >= 3:
                ii = idx[s:e]
                segs.append(np.stack([tt[ii], lat[ii], lon[ii], h[ii], alt_b[ii], alt_g[ii], gs[ii], trk[ii], vr[ii]],
                                     1))
        if segs:
            out[r] = segs
    if not out:
        return None
    return grp, meta, out


def _members(fobj, limit=None):
    try:
        with tarfile.open(fileobj=fobj, mode="r|") as tf:
            for k, m in enumerate(tf):
                if limit is not None and k >= limit:
                    return
                if m.isfile() and "trace_full_" in m.name:
                    yield tf.extractfile(m).read()
    except (tarfile.ReadError, EOFError):
        return                                   # truncated archive (only with --max-members style tests)


def _cut_one(raw):
    return cut_trace(raw)


def cut_stream(fobj, outdir, day, regions=("CONUS", "AU"), intervals=None, workers=4, types=None, limit=None):
    """Stream a day archive (file-like) into <outdir>/<day>_<R>_{low,jet}.npz and <day>.json. Returns the counts."""
    os.makedirs(outdir, exist_ok=True)
    iv = intervals or {}
    acc = {}
    n_files = n_kept = 0
    types = types or dict(_FALLBACK_TYPES)
    with Pool(workers, initializer=_init, initargs=(list(regions), iv, types)) as pool:
        for res in pool.imap_unordered(_cut_one, _members(fobj, limit), chunksize=64):
            n_files += 1
            if res is None:
                continue
            grp, meta, out = res
            n_kept += 1
            for r, segs in out.items():
                A = acc.setdefault((r, grp), dict(rows=[], fid=[], flights=[]))
                for s in segs:
                    A["fid"].append(np.full(s.shape[0], len(A["flights"]), np.int32))
                    A["rows"].append(s)
                    A["flights"].append(meta)
    counts = dict(day=day, files=n_files, kept_aircraft=n_kept)
    for (r, grp), A in acc.items():
        X = np.concatenate(A["rows"])
        fl = A["flights"]
        tmp = os.path.join(outdir, f".{day}_{r}_{grp}.tmp.npz")
        np.savez_compressed(
            tmp, fid=np.concatenate(A["fid"]), t=X[:, 0], lat=X[:, 1], lon=X[:, 2], h=X[:, 3].astype(np.float32),
            alt_baro=X[:, 4].astype(np.float32), alt_geom=X[:, 5].astype(np.float32), gs=X[:, 6].astype(np.float32),
            trk=X[:, 7].astype(np.float32), vr=X[:, 8].astype(np.float32),
            f_hex=np.array([f["hex"] for f in fl]), f_type=np.array([f["type"] for f in fl]),
            f_reg=np.array([f["reg"] for f in fl]), f_cls=np.array([f["cls"] for f in fl]),
            f_desc=np.array([f["desc"] for f in fl]), f_cat=np.array([f["category"] for f in fl]),
            meta=json.dumps(dict(t0=float(X[:, 0].min()), t1=float(X[:, 0].max()), region=r, group=grp, day=day,
                                 bbox=list(BOX[r]), alt_max_ft=ALT_MAX_FT, src="adsb.lol global day (ODbL 1.0)")))
        os.replace(tmp, os.path.join(outdir, f"{day}_{r}_{grp}.npz"))
        cls = {}
        for f in fl:
            cls[f["cls"]] = cls.get(f["cls"], 0) + 1
        counts[f"{r}_{grp}"] = dict(segments=len(fl), points=int(X.shape[0]), by_cls=cls,
                                    aircraft=len({f["hex"] for f in fl}))
    for r in regions:                     # a day with no low traffic in a region still gets a (empty) marker
        counts.setdefault(f"{r}_low", dict(segments=0, points=0))
    with open(os.path.join(outdir, f"{day}.json.tmp"), "w") as f:
        json.dump(counts, f, indent=1)
    os.replace(os.path.join(outdir, f"{day}.json.tmp"), os.path.join(outdir, f"{day}.json"))
    return counts


def day_done(adsb_dir, day):
    return os.path.exists(os.path.join(adsb_dir, f"{day}.json"))


def fetch_day(day, adsb_dir, regions=("CONUS", "AU"), intervals=None, workers=4, keep_raw=False, progress=None,
              log=print):
    """Download (resumable) and cut one UTC day. Returns the counts, or None when adsb.lol has no release."""
    if day_done(adsb_dir, day):
        with open(os.path.join(adsb_dir, f"{day}.json")) as f:
            return json.load(f)
    urls = release_urls(day, adsb_dir)
    if not urls:
        log(f"{day}: no adsb.lol release")
        return None
    raw = os.path.join(adsb_dir, "raw", day)
    os.makedirs(raw, exist_ok=True)
    paths = []
    for u in urls:
        p = os.path.join(raw, os.path.basename(u))
        if not os.path.exists(p):
            log(f"{day}: downloading {os.path.basename(u)}")
            download(u, p, progress=progress)
        paths.append(p)
    types = load_type_table(adsb_dir)
    log(f"{day}: cutting {', '.join(regions)} below {ALT_MAX_FT:.0f} ft")
    counts = cut_stream(Chain(paths), adsb_dir, day, regions, intervals, workers, types)
    if not keep_raw:
        for p in paths:
            os.remove(p)
        try:
            os.rmdir(raw)
        except OSError:
            pass
    return counts


# ---------------------------------------------------------------------------------------------------------------
# planning: which days and intervals a set of recordings needs
# ---------------------------------------------------------------------------------------------------------------
MARG_CLOCK, MARG = 3600 + 900, 900


def day_of(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).date()


def plan(inputs, decoy_days=None):
    """inputs: [{oid, region_prior, clock_utc, dur_s}]. decoy_days: {oid: [day offsets]} (other-day decoys).
    Returns {day: {region: [[t0, t1], ...]}}: each recording's window +- (60 min clock decoys + 15 min), and the same
    UTC time of day on each decoy day (+- 15 min)."""
    iv = {}
    for c in inputs:
        r = c["region_prior"]
        spans = [(c["clock_utc"] - MARG_CLOCK, c["clock_utc"] + c["dur_s"] + MARG_CLOCK)]
        for off in (decoy_days or {}).get(c["oid"], []):
            spans.append((c["clock_utc"] + off * 86400 - MARG, c["clock_utc"] + c["dur_s"] + off * 86400 + MARG))
        for a, b in spans:
            for t in (a, b):
                iv.setdefault((day_of(t).isoformat(), r), []).append((a, b))
    days = {}
    for (d, r), L in iv.items():
        days.setdefault(d, {}).setdefault(r, []).extend(L)
    out = {}
    for d in sorted(days):
        merged = {}
        for r, L in days[d].items():
            L = sorted(L)
            m = [list(L[0])]
            for a, b in L[1:]:
                if a <= m[-1][1]:
                    m[-1][1] = max(m[-1][1], b)
                else:
                    m.append([a, b])
            d0 = dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()
            merged[r] = [[max(a, d0), min(b, d0 + 86400)] for a, b in m if min(b, d0 + 86400) > max(a, d0)]
        out[d] = merged
    return out


def nearest_other_days(inputs, k=3):
    """Benchmark decoy days: per recording, the k nearest *other* recording days of the same country (offsets)."""
    case_days = {}
    for c in inputs:
        case_days.setdefault(c["region_prior"], set()).add(day_of(c["clock_utc"]))
    out = {}
    for c in inputs:
        d0 = day_of(c["clock_utc"])
        others = sorted((d for d in case_days[c["region_prior"]] if d != d0), key=lambda d: (abs((d - d0).days), d))[:k]
        out[c["oid"]] = [(d - d0).days for d in others]
    return out
