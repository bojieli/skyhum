"""Fuse the country search's region scores with free side information (population, UTC offset), the value of the
sound given a prior (metro known, 40 km box, metro-size box), decoy confidence on the fused score, the population
baselines, and the first verify-a-claim table. Post-processing of saved search outputs only; CPU, minutes.

Model (per unit = group subset x window x clock tolerance; the real variant and each decoy variant the same way):
  logit(r) = lam * log pop(r)          population prior (CONUS: Census 2023 CBSA / 2020 county; AU: ABS ERP 2021 SA4)
           + log 1[offset(r) = offset(truth site) at the recording instant]   time-zone prior (UTC offset incl. DST)
           + beta * A(r)               A = region score of the search (best >=2-voter cell, s2); optionally the
                                       >=1-voter s1 score for regions without an s2 score (single sessions only);
                                       regions with no score get c (min(c, lowest listed) when the 40-region list is
                                       full)
  (lam, beta, c, use_s1) are fitted by leave-one-SITE-out cross-validation (maximum mean log posterior of the true
  region over the training sites' units, all windows / clocks / N pooled, separately for CONUS and AU and per prior
  set); each unit is reported with parameters that never saw its site.

Needs timezonefinder (pip install "skyhum[country]").
"""
import datetime as dt
import glob
import hashlib
import json
import math
import os
from importlib import resources
from zoneinfo import ZoneInfo

import numpy as np

from skyhum.country.common import Regions, haversine_km, load_json

INF = float("inf")
NEG = -1e9
KINDS = ("60", "30")
TAUS = (5, 30)
FULL = 40
TIERS = (0.1, 0.3, 1, 2, 5, 10, 25, 50, 100)
GRID_LAM = (0.5, 0.75, 1.0, 1.25)
GRID_BETA = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
GRID_C = (-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
GRID_S1 = (False, True)
EPS = 1e-6          # floor mixed into the posterior (a truth outside the UTC-offset mask is not -inf)
FAM = {"webcam": "CONUS", "events": "CONUS", "speech": "CONUS", "a2o": "AU"}


# --------------------------------------------------------------------------------------------------- region tables
def sa4_population():
    """ABS ERP 2021 by SA4 (ABS Data API ABS_ANNUAL_ERP_ASGS2021, CC BY 4.0), shipped with the package."""
    import csv
    sa4 = {}
    with resources.files("skyhum.country").joinpath("data", "abs_erp_sa4_2021.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["REGION_TYPE"] == "SA4":
                sa4[row["ASGS_2021"]] = float(row["OBS_VALUE"])
    return sa4


class RegionInfo:
    def __init__(self, R, tf, static):
        self.R = R
        self.reg = Regions(R, static)
        tab = self.reg.table
        n = len(tab)
        self.name = [t["name"] for t in tab]
        self.cells = np.array([t.get("cells", 0) for t in tab])
        if R == "CONUS":
            pop = np.array([t.get("pop", 0) for t in tab], float)
        else:
            sa4 = sa4_population()
            pop = np.array([sa4.get(t["key"].split(":")[1], 0.0) for t in tab])
            miss = [t["name"] for t, p in zip(tab, pop) if p <= 0 and t.get("cells", 0)]
            if miss:
                print("AU SA4 without population:", miss)
        self.valid = (self.cells > 0) & (pop > 0)
        self.pop = np.where(self.valid, pop, 0.0)
        self.logpop = np.where(self.valid, np.log(np.maximum(pop, 1.0)), NEG)
        lat = np.array([t.get("lat", np.nan) for t in tab], float)
        lon = np.array([t.get("lon", np.nan) for t in tab], float)
        # guess point of a region from the prior alone: population centroid (CONUS: county centres of population
        # 2020 falling inside the region; AU: area centroid snapped inside the SA4, no finer population used)
        self.plat, self.plon = lat.copy(), lon.copy()
        if R == "CONUS":
            cp = np.load(os.path.join(static, "county_pop.npz"))
            m = (cp["lat"] > 24.0) & (cp["lat"] < 50.0) & (cp["lon"] > -125.5) & (cp["lon"] < -66.5)
            cl, co, cw = cp["lat"][m], cp["lon"][m], cp["pop"][m].astype(float)
            rid = np.atleast_1d(self.reg.of(cl, co))
            for r in np.unique(rid[rid >= 0]):
                k = rid == r
                if cw[k].sum() > 0:
                    self.plat[r] = float((cl[k] * cw[k]).sum() / cw[k].sum())
                    self.plon[r] = float((co[k] * cw[k]).sum() / cw[k].sum())
        self.tz = [tf.timezone_at(lng=float(self.plon[i]), lat=float(self.plat[i])) if self.valid[i] else None
                   for i in range(n)]
        if any(self.valid[i] and self.tz[i] is None for i in range(n)):
            for i in range(n):
                if self.valid[i] and self.tz[i] is None:
                    self.tz[i] = tf.closest_timezone_at(lng=float(self.plon[i]), lat=float(self.plat[i]))
        self.tznames = sorted({z for z in self.tz if z})
        self.tzidx = np.array([self.tznames.index(z) if z else -1 for z in self.tz])
        self.order = np.argsort(-self.pop)      # population order (largest first)
        self._mask = {}

    def offsets(self, t):
        """UTC offset (minutes) of every region at instant t (epoch s)."""
        T = dt.datetime.fromtimestamp(t, dt.timezone.utc)
        off = np.array([T.astimezone(ZoneInfo(z)).utcoffset().total_seconds() / 60 for z in self.tznames])
        o = np.full(len(self.tz), -99999.0)
        ok = self.tzidx >= 0
        o[ok] = off[self.tzidx[ok]]
        return o

    def mask(self, t, off_truth):
        k = (int(t // 3600), off_truth)
        if k not in self._mask:
            self._mask[k] = self.valid & (self.offsets(t) == off_truth)
        return self._mask[k]


def site_offset(tf, lat, lon, t):
    z = tf.timezone_at(lng=lon, lat=lat) or tf.closest_timezone_at(lng=lon, lat=lat)
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).astimezone(ZoneInfo(z)).utcoffset().total_seconds() / 60, z


# ---------------------------------------------------------------------------------------------------- units
def parse_top(lst):
    """[[rid, score, lat, lon, j], ...] -> (rid int array, score array, lat, lon, full)"""
    if not lst:
        return (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), False)
    a = np.array([x[:4] for x in lst], float)
    return (a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], len(lst) >= FULL)


def load_units(runs, truth, truth_dir, ch="s2"):
    tdir = os.path.realpath(truth_dir)
    units = []
    for f in sorted(glob.glob(os.path.join(runs, "*.json"))):
        r = load_json(f)
        if not r.get("done"):
            continue
        if any(p.startswith(tdir) for p in r.get("touched", [])):
            raise SystemExit(f"BLINDNESS VIOLATION: {f}")
        sess = r["sessions"]

        def lacks(v, sess=sess):
            for i, s_ in enumerate(v.get("sessions", [])):
                t = truth[sess[i]]
                sh = v["shifts"][i] or 0.0
                a, b = float(t["clock"]) + sh - 900, float(t["clock"]) + sh + float(t["dur"]) + 400
                for d in s_.get("missing_days", []):
                    d0 = dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()
                    if d0 <= b and d0 + 86400 >= a:
                        return True
            return False
        real = next(v for v in r["variants"] if v["variant"] == "real")
        if lacks(real):
            continue                                    # excluded exactly as score.py does
        decs = [v for v in r["variants"] if v["variant"] != "real" and not lacks(v)]
        t0 = truth[sess[0]]
        for si, X in enumerate(r["subsets"]):
            if not (r["gid"].startswith("g") or len(X) == 1):
                continue
            for kind in KINDS:
                for tau in TAUS:
                    key = f"{si}|{kind}|{tau}"
                    rr = real["subsets"].get(key)
                    if rr is None:
                        if real["n_union"] == 0 and all(len([w for w in truth[sess[i]]["windows"] if w["kind"] == kind])
                                                        for i in X):
                            rr = {ch: {"top": []}}
                        else:
                            continue
                    if ch not in rr and rr.get("s2", {}).get("top"):
                        continue                    # output without this channel (as score.py --channel)
                    heard = []
                    for i in X:
                        t = truth[sess[i]]
                        w = [w for w in t["windows"] if w["kind"] == kind][0]
                        heard.append(sum(1 for q in (t["pos"] or []) if w["t0"] <= q["t"] <= w["t1"]
                                         and q["cls"] in ("prop", "rotor", "unknown")))
                    d2 = [parse_top(v["subsets"].get(key, {}).get(ch, {}).get("top")) for v in decs]
                    d1 = [parse_top(v["subsets"].get(key, {}).get("s1", {}).get("top")) for v in decs]
                    units.append(dict(gid=r["gid"], set=t0["set"], site=t0["site"], N=len(X), sess0=sess[0],
                                      sessions=[sess[i] for i in X], kind=kind, tau=tau, region=r["region"],
                                      lat=float(t0["lat"]), lon=float(t0["lon"]), clock=float(t0["clock"]),
                                      heard_total=int(sum(heard)), heard_max=int(max(heard)),
                                      s2=parse_top(rr.get(ch, {}).get("top")),
                                      s1=parse_top((rr.get("s1") or {}).get("top")),
                                      d2=d2, d1=d1))
    return units


# ---------------------------------------------------------------------------------------------------- fusion
def acoustic_vec(n, s2, s1, beta, c, use_s1):
    """beta * A(r) over all n regions (vectorised over beta, c: returns (G, n))"""
    beta = np.asarray(beta, float)[:, None]
    c = np.asarray(c, float)[:, None]
    rid, sc, _, _, full = s2
    lo = sc.min() if full and sc.size else INF
    cc = np.minimum(c, lo)
    A = np.repeat(cc, n, axis=1).astype(float)
    if use_s1 and s1[0].size:
        r1, sc1, _, _, full1 = s1
        if full1:
            A = np.minimum(A, sc1.min())
        keep = ~np.isin(r1, rid)
        A[:, r1[keep]] = sc1[keep][None, :]
    if rid.size:
        A[:, rid] = sc[None, :]
    return beta * A


def logits(RI, u, mask, lam, beta, c, use_s1, s2=None, s1=None, use_pop=True):
    s2 = u["s2"] if s2 is None else s2
    s1 = u["s1"] if s1 is None else s1
    G = len(np.atleast_1d(beta))
    lp = (np.asarray(lam, float)[:, None] * np.where(RI.valid, RI.logpop, 0.0)[None, :]) if use_pop \
        else np.zeros((G, len(RI.pop)))
    L = lp + acoustic_vec(len(RI.pop), s2, s1, np.atleast_1d(beta), np.atleast_1d(c), use_s1)
    return np.where(mask[None, :], L, -np.inf)


def logpost_truth(L, tr):
    m = L.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(L - m).sum(1))
    lp = L[:, tr] - lse if tr >= 0 else np.full(L.shape[0], -np.inf)
    return np.log((1 - EPS) * np.exp(lp) + EPS / L.shape[1])


def grid():
    g = [(l, b, c, s) for l in GRID_LAM for b in GRID_BETA for c in GRID_C for s in GRID_S1]
    return g


# ---------------------------------------------------------------------------------------------------- helpers
def frac(k, n):
    return f"{k}/{n} ({100 * k / n:.0f} %)" if n else "-"


def pct(x):
    return f"{100 * x:.1f} %"


def boot_ci(U, f, B=2000, seed=0):
    """cluster (site) bootstrap 95 % CI of mean(f(u))"""
    if not U:
        return (0, 0)
    sites = sorted({u["site"] for u in U})
    by = {s: [f(u) for u in U if u["site"] == s] for s in sites}
    S = np.array([sum(by[s]) for s in sites], float)
    C = np.array([len(by[s]) for s in sites], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sites), (B, len(sites)))
    v = S[idx].sum(1) / C[idx].sum(1)
    return (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5))


def fci(U, f):
    k = sum(bool(f(u)) for u in U)
    n = len(U)
    if not n:
        return "-"
    lo, hi = boot_ci(U, lambda u: float(bool(f(u))))
    return f"{k}/{n} ({100 * k / n:.0f} %; {lo:.0f}-{hi:.0f})"


def table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def jitter(key, J):
    h = hashlib.sha1(f"prior-box|{key}".encode()).digest()
    a = int.from_bytes(h[:4], "big") / 2 ** 32
    b = int.from_bytes(h[4:8], "big") / 2 ** 32
    return (2 * a - 1) * J, (2 * b - 1) * J


# ---------------------------------------------------------------------------------------------------- main
def timezone_finder():
    try:
        from timezonefinder import TimezoneFinder
    except ImportError as e:
        raise SystemExit("the prior and claim tables need timezonefinder: pip install 'skyhum[country]'") from e
    return TimezoneFinder()


def run(runs, truth_path, static, out, print_md=False):
    """Writes OUT/RESULTS_prior.md and OUT/units_prior.json. Returns the units."""
    os.makedirs(out, exist_ok=True)
    truth = load_json(truth_path)
    tf = timezone_finder()
    RI = {R: RegionInfo(R, tf, static) for R in ("CONUS", "AU")}
    units = load_units(runs, truth, os.path.dirname(os.path.abspath(truth_path)))
    for u in units:
        R = RI[u["region"]]
        u["tr"] = int(R.reg.of(u["lat"], u["lon"]))
        u["off"], u["tzname"] = site_offset(tf, u["lat"], u["lon"], u["clock"])
        u["mask_tz"] = R.mask(u["clock"], u["off"])
        u["mask_all"] = R.valid
        u["tr_in_tz"] = bool(u["tr"] >= 0 and u["mask_tz"][u["tr"]])
    G = grid()
    gl = np.array([g[0] for g in G])
    gb = np.array([g[1] for g in G])
    gc = np.array([g[2] for g in G])
    gs = np.array([g[3] for g in G])
    MODES = {"pop": ("mask_all", True), "pop+tz": ("mask_tz", True), "tz": ("mask_tz", False)}
    # per-unit log posterior of the truth for every grid point and mode
    for u in units:
        R = RI[u["region"]]
        u["ll"] = {}
        for md, (mk, usepop) in MODES.items():
            ll = np.empty(len(G))
            for s1 in (False, True):
                k = gs == s1
                L = logits(R, u, u[mk], gl[k], gb[k], gc[k], s1 if u["N"] == 1 else False, use_pop=usepop)
                ll[k] = logpost_truth(L, u["tr"])
            u["ll"][md] = ll
    # leave-one-site-out fit, per family and mode
    fits = {}
    for fam in ("CONUS", "AU"):
        U = [u for u in units if u["region"] == fam]
        sites = sorted({u["site"] for u in U})
        for md in MODES:
            tot = np.sum([u["ll"][md] for u in U], 0)
            persite = {s: np.sum([u["ll"][md] for u in U if u["site"] == s], 0) for s in sites}
            full = int(np.argmax(tot))
            fits[(fam, md, "all")] = G[full]
            for s in sites:
                gi = int(np.argmax(tot - persite[s]))
                for u in U:
                    if u["site"] == s:
                        u.setdefault("par", {})[md] = G[gi]
    # predictions
    for u in units:
        R = RI[u["region"]]
        u["pred"] = {}
        # acoustic alone (the locator's own answer: s2 top)
        rid, sc, la, lo, _ = u["s2"]
        u["pred"]["acoustic"] = dict(top=list(rid[:3]), lat=la[0] if rid.size else None,
                                     lon=lo[0] if rid.size else None)
        # prior alone
        for md, mk in (("prior pop", "mask_all"), ("prior pop+tz", "mask_tz")):
            o = [int(r) for r in R.order if u[mk][r]][:3]
            u["pred"][md] = dict(top=o, lat=R.plat[o[0]], lon=R.plon[o[0]])
        # fused
        for md, (mk, usepop) in MODES.items():
            lam, beta, c, s1 = u["par"][md]
            s1 = s1 and u["N"] == 1
            L = logits(R, u, u[mk], [lam], [beta], [c], s1, use_pop=usepop)[0]
            o = np.argsort(-L)[:3]
            P = np.exp(L - L.max())
            P /= P.sum()
            r0 = int(o[0])
            # best point of the chosen region: its s2 point, else (s1 used) its s1 point, else the prior point
            pt = (R.plat[r0], R.plon[r0])
            cand = [u["s2"]] + ([u["s1"]] if s1 else [])
            for (rr, _, la_, lo_, _) in cand:
                hit = np.flatnonzero(rr == r0)
                if hit.size:
                    pt = (la_[hit[0]], lo_[hit[0]])
                    break
            # decoys: same parameters, top posterior
            dtop = []
            for d2, d1 in zip(u["d2"], u["d1"]):
                Ld = logits(R, u, u[mk], [lam], [beta], [c], s1, s2=d2, s1=d1, use_pop=usepop)[0]
                Pd = np.exp(Ld - Ld.max())
                dtop.append(float(np.log(1.0 / Pd.sum())))     # log top posterior
            ltop = float(np.log(P[r0]))
            K = len(dtop)
            u["pred"]["fused " + md] = dict(top=[int(x) for x in o], lat=pt[0], lon=pt[1], ptop=float(P[r0]),
                                            ptrue=float(P[u["tr"]]) if u["tr"] >= 0 else 0.0,
                                            margin=ltop - (max(dtop) if dtop else -INF),
                                            confident=K > 0 and all(ltop > d + 1e-9 for d in dtop), K=K,
                                            acoustic_used=bool(np.isin(r0, u["s2"][0]).any() or
                                                               (s1 and np.isin(r0, u["s1"][0]).any())))
        for p in u["pred"].values():
            p["top1"] = bool(p["top"]) and p["top"][0] == u["tr"]
            p["top3"] = u["tr"] in p["top"][:3]
            p["dist"] = float(haversine_km(p["lat"], p["lon"], u["lat"], u["lon"])) if p["lat"] is not None else INF
    # acoustic-alone confidence = score.py's own-decoy rule (for reference in the confidence table)
    for u in units:
        real_top = u["s2"][1][0] if u["s2"][0].size else -INF
        dt_ = [d[1][0] if d[0].size else -INF for d in u["d2"]]
        K = len(dt_)
        u["pred"]["acoustic"]["confident"] = bool(u["s2"][0].size) and K > 0 and all(real_top > d for d in dt_)
        u["pred"]["acoustic"]["margin"] = (real_top - max(dt_)) if (u["s2"][0].size and dt_) else -INF
        # gated: the prior's answer, replaced by the locator's own answer when that beats every decoy (no fitting)
        for base in ("pop", "pop+tz"):
            src = u["pred"]["acoustic"] if u["pred"]["acoustic"]["confident"] else u["pred"]["prior " + base]
            g = {k: src[k] for k in ("top", "lat", "lon", "top1", "top3", "dist")}
            g["confident"] = u["pred"]["acoustic"]["confident"]
            u["pred"]["gated " + base] = g
    md = report(units, RI, fits, G)
    with open(os.path.join(out, "RESULTS_prior.md"), "w") as f:
        f.write(md)
    slim = []
    for u in units:
        s = {k: v for k, v in u.items() if k in ("gid", "set", "site", "N", "kind", "tau", "region", "heard_total",
                                                 "tr", "tzname", "off", "tr_in_tz")}
        s["par"] = u["par"]
        s["pred"] = {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else vv.item())
                         for kk, vv in v.items()} for k, v in u["pred"].items()}
        s["prior_cond"] = u.get("prior_cond")
        slim.append(s)
    with open(os.path.join(out, "units_prior.json"), "w") as f:
        json.dump(slim, f, default=float)
    if print_md:
        print(md)
    return units


# ---------------------------------------------------------------------------------------------------- report
METHODS = (("prior alone: population", "prior pop"), ("prior alone: population + UTC offset", "prior pop+tz"),
           ("acoustic alone", "acoustic"), ("acoustic + population", "fused pop"),
           ("acoustic + UTC offset (uniform within)", "fused tz"),
           ("acoustic + population + UTC offset", "fused pop+tz"),
           ("gated: population prior, sound's answer when it beats every decoy", "gated pop"),
           ("gated: population + UTC offset, sound's answer when it beats every decoy", "gated pop+tz"))


def row_tiers(U, k):
    n = len(U)
    return ([fci(U, lambda u: u["pred"][k]["top1"]), frac(sum(u["pred"][k]["top3"] for u in U), n)]
            + [frac(sum(u["pred"][k]["dist"] <= x for u in U), n) for x in TIERS])


def prior_cond(u, RI):
    """Prior-conditioned value (item 2): metro known, 40 km box, r5a-like box. Returns dict of outcomes."""
    R = RI[u["region"]]
    out = {}
    tr = u["tr"]
    use1 = u["N"] == 1
    lists = [("s2", u["s2"], u["d2"])] + ([("s1", u["s1"], u["d1"])] if use1 else [])

    def dist(la, lo):
        return float(haversine_km(la, lo, u["lat"], u["lon"]))
    # metro known
    prior_pt = (R.plat[tr], R.plon[tr]) if tr >= 0 else (np.nan, np.nan)
    out["metro_prior"] = dist(*prior_pt)
    for nm, L, D in lists:
        rid, sc, la, lo, _ = L
        h = np.flatnonzero(rid == tr)
        if h.size:
            real = sc[h[0]]
            dsc = []
            for d in D:
                hh = np.flatnonzero(d[0] == tr)
                dsc.append(d[1][hh[0]] if hh.size else -INF)
            out[f"metro_{nm}_has"] = True
            out[f"metro_{nm}_dist"] = dist(la[h[0]], lo[h[0]])
            out[f"metro_{nm}_beats"] = bool(D) and all(real > x for x in dsc)
        else:
            out[f"metro_{nm}_has"] = False
            out[f"metro_{nm}_dist"] = out["metro_prior"]
            out[f"metro_{nm}_beats"] = False
    # boxes: (name, half-width km or deg, jitter, unit)
    for bn, half, J, deg in (("box40", 20.0, 10.0, False), ("boxr5a", 0.35, 0.25, True)):
        if deg:
            dla, dlo = jitter(u["sess0"] + bn, J)
            cla, clo = u["lat"] + dla, u["lon"] + dlo
            inbox = lambda la, lo, cla=cla, clo=clo, half=half: (  # noqa: E731
                (np.abs(la - cla) <= half) & (np.abs(lo - clo) <= half))
        else:
            dx, dy = jitter(u["sess0"] + bn, J)
            cla = u["lat"] + dy / 111.132
            clo = u["lon"] + dx / (111.32 * math.cos(math.radians(u["lat"])))
            inbox = lambda la, lo, cla=cla, clo=clo, half=half: ((np.abs(la - cla) * 111.132 <= half)  # noqa: E731
                                    & (np.abs(lo - clo) * 111.32 * math.cos(math.radians(cla)) <= half))
        out[f"{bn}_prior"] = dist(cla, clo)
        for nm, L, D in lists:
            rid, sc, la, lo, _ = L
            ok = np.flatnonzero(inbox(la, lo)) if rid.size else np.zeros(0, int)
            if ok.size:
                j = ok[np.argmax(sc[ok])]
                real = sc[j]
                dsc = []
                for d in D:
                    okd = np.flatnonzero(inbox(d[2], d[3])) if d[0].size else np.zeros(0, int)
                    dsc.append(d[1][okd].max() if okd.size else -INF)
                out[f"{bn}_{nm}_has"] = True
                out[f"{bn}_{nm}_dist"] = dist(la[j], lo[j])
                out[f"{bn}_{nm}_beats"] = bool(D) and all(real > x for x in dsc)
            else:
                out[f"{bn}_{nm}_has"] = False
                out[f"{bn}_{nm}_dist"] = out[f"{bn}_prior"]
                out[f"{bn}_{nm}_beats"] = False
    # verify a claimed region: true claim and false claims (same UTC offset; population-weighted; nearest neighbour)
    m = u["mask_tz"].copy()
    if tr >= 0:
        m[tr] = False
    w = np.where(m, R.pop, 0.0)
    dd = haversine_km(R.plat, R.plon, R.plat[tr], R.plon[tr]) if tr >= 0 else None
    big = R.pop >= (250000 if R.R == "CONUS" else 0)
    cand = np.flatnonzero(m & big & R.valid)
    nn = int(cand[np.argmin(dd[cand])]) if cand.size and dd is not None else -1
    out["nn_claim"] = nn
    for nm, L, D in lists:
        rid, sc, la, lo, _ = L
        # acceptance per region: listed / beats all own decoys at that region
        acc_list = np.zeros(len(R.pop), bool)
        acc_beat = np.zeros(len(R.pop), bool)
        if rid.size:
            acc_list[rid] = True
            dmax = np.full(len(R.pop), -INF)
            for d in D:
                if d[0].size:
                    np.maximum.at(dmax, d[0], d[1])
            acc_beat[rid] = (sc > dmax[rid]) & (len(D) > 0)
        for an, acc in (("listed", acc_list), ("beats", acc_beat)):
            out[f"ver_{nm}_{an}_true"] = bool(tr >= 0 and acc[tr])
            out[f"ver_{nm}_{an}_falsepop"] = float((w * acc).sum() / w.sum()) if w.sum() > 0 else 0.0
            out[f"ver_{nm}_{an}_falsenn"] = bool(nn >= 0 and acc[nn])
    return out


def report(units, RI, fits, G):
    md = ("# Region score x free side information (population, UTC offset); value of the sound given a prior\n\n"
          "_Generated by `skyhum country prior` from saved search outputs (same units and exclusions as `skyhum "
          "country score`). Post-processing only. Top-1 cells: count (%, 95 % site-cluster bootstrap CI)._\n\n")
    # parameters
    md += "### Fitted weights (all sites; each reported unit uses its own leave-one-site-out fit)\n\n"
    rows = []
    for (fam, mdn, _), g in sorted(fits.items()):
        nsplit = len({tuple(u["par"][mdn]) for u in units if u["region"] == fam})
        rows.append([fam, mdn, g[0], g[1], g[2], g[3], nsplit])
    md += table(rows, ["family", "prior", "lam (population exponent)", "beta (per score unit)",
                       "c (score of an unscored region)", "use s1 (single sessions)", "distinct LOSO fits"]) + "\n"
    tzout = [u for u in units if not u["tr_in_tz"]]
    md += (f"Truth region outside its own UTC-offset mask (region tz taken at the population centroid): "
           f"{len(tzout)}/{len(units)} units"
           + (f" ({', '.join(sorted({u['site'] for u in tzout}))})" if tzout else "") + ".\n\n")
    # item 1 tables
    md += "## 1. Prior + acoustic: top-1 / top-3 region and distance tiers\n\n"
    head = ["method", "top-1 (95 % CI)", "top-3"] + [f"<={x:g} km" if x >= 1 else f"<={int(x * 1000)} m"
                                                     for x in TIERS]
    for st in ("webcam", "events", "speech", "a2o"):
        note = " (Australia: SA4 regions, ABS ERP 2021, AEST/ACST/AWST incl. DST)" if st == "a2o" else ""
        md += f"### {st}{note}\n\n"
        for kind in KINDS:
            for tau in TAUS:
                for N in (1, 2, 4, 8):
                    U = [u for u in units if u["set"] == st and u["kind"] == kind and u["tau"] == tau and u["N"] == N]
                    if not U:
                        continue
                    if N > 1 and st in ("events", "speech"):
                        pass
                    rows = [[nm] + row_tiers(U, k) for nm, k in METHODS]
                    md += (f"**{st}, {kind} min, +-{tau} s, N = {N}: {len(U)} units, "
                           f"{len({u['site'] for u in U})} sites**\n\n") + table(rows, head) + "\n"
    # pooled US
    md += "### US pooled (webcam + events + speech), N = 1\n\n"
    for kind in KINDS:
        for tau in TAUS:
            U = [u for u in units if u["region"] == "CONUS" and u["kind"] == kind and u["tau"] == tau and u["N"] == 1]
            rows = [[nm] + row_tiers(U, k) for nm, k in METHODS]
            md += f"**{kind} min, +-{tau} s: {len(U)} units**\n\n" + table(rows, head) + "\n"
    # acoustic contribution
    md += "### How much of the fused answer is acoustic (60 min, +-5 s, N = 1)\n\n"
    rows = []
    for st in ("webcam", "events", "speech", "a2o"):
        U = [u for u in units if u["set"] == st and u["kind"] == "60" and u["tau"] == 5 and u["N"] == 1]
        for k in ("fused pop", "fused pop+tz", "gated pop", "gated pop+tz"):
            pk = "prior pop+tz" if k.endswith("pop+tz") else "prior pop"
            a = sum(u["pred"][k]["top1"] for u in U)
            b = sum(u["pred"][pk]["top1"] for u in U)
            won = sum(u["pred"][k]["top1"] and not u["pred"][pk]["top1"] for u in U)
            lost = sum(u["pred"][pk]["top1"] and not u["pred"][k]["top1"] for u in U)
            used = sum(bool(u["pred"][k].get("acoustic_used", u["pred"][k].get("confident"))) for u in U)
            rows.append([st, k, len(U), b, a, won, lost, used])
    md += table(rows, ["set", "fused", "units", "prior-alone top-1", "fused top-1", "gained by sound",
                       "lost by sound", "answer is an acoustically scored region"]) + "\n"
    # item 4 baselines
    md += "## 4. Population baselines (N = 1, 60 min units; expected values for random draws)\n\n"
    rows = []
    for fam, sets in (("CONUS", ("webcam", "events", "speech")), ("AU", ("a2o",))):
        R = RI[fam]
        U = [u for u in units if u["set"] in sets and u["kind"] == "60" and u["tau"] == 5 and u["N"] == 1]
        n = len(U)
        big = int(R.order[0])
        for lab, mk in (("", "mask_all"), (" within the UTC offset", "mask_tz")):
            # always the largest region (within mask)
            t1 = t3 = 0
            dists = []
            ew = e3 = 0.0
            ek = np.zeros(len(TIERS))
            for u in U:
                o = [int(r) for r in R.order if u[mk][r]]
                t1 += o[0] == u["tr"]
                t3 += u["tr"] in o[:3]
                dists.append(float(haversine_km(R.plat[o[0]], R.plon[o[0]], u["lat"], u["lon"])))
                w = np.where(u[mk], R.pop, 0.0)
                pt = w[u["tr"]] / w.sum() if u["tr"] >= 0 else 0.0
                ew += pt
                e3 += 1 - (1 - pt) ** 3
                d = haversine_km(R.plat, R.plon, u["lat"], u["lon"])
                ek += np.array([(w * (d <= x)).sum() / w.sum() for x in TIERS])
            nm0 = R.name[big] if mk == "mask_all" else "largest region in the truth's UTC offset"
            rows.append([fam, f"always the largest region{lab} ({nm0})", n, frac(t1, n), frac(t3, n)]
                        + [frac(sum(d <= x for d in dists), n) for x in TIERS[5:]])
            rows.append([fam, f"population-weighted random region{lab}", n, pct(ew / n), pct(e3 / n) + " (3 draws)"]
                        + [pct(v / n) for v in ek[5:]])
    md += table(rows, ["family", "baseline", "units", "top-1", "top-3"] + [f"<={x:g} km" for x in TIERS[5:]])
    md += ("\nAU random-region distance tiers place each SA4's population at its area centroid (no finer AU population "
           "grid used), so they are approximate; the CONUS ones use county centres of population. The score.py "
           "random baseline (1.64 %) is a population-weighted random region over all of CONUS; it is reproduced in the "
           "first CONUS random row.\n\n")
    # item 3 confidence
    md += "## 3. Confidence on the fused score (decoy-based)\n\n"
    md += ("Each decoy variant (clock shifted +-10..60 min or another day; same UTC-offset mask, same fitted weights) "
           "is fused the same way. Confident = the real top posterior beats every decoy's top posterior (the scorer's "
           "own-decoy rule applied to the fused score). Posterior thresholds use the LOSO-fitted posterior itself.\n\n")
    rows = []
    for grp, sets in (("US (webcam+events+speech)", ("webcam", "events", "speech")), ("A2O", ("a2o",))):
        for kind in KINDS:
            for tau in TAUS:
                U = [u for u in units if u["set"] in sets and u["kind"] == kind and u["tau"] == tau and u["N"] == 1]
                for k in ("acoustic", "fused pop", "fused pop+tz"):
                    C = [u for u in U if u["pred"][k].get("confident")]
                    r = [grp, f"{kind} min +-{tau} s", k, len(U), f"{sum(u['pred'][k]['top1'] for u in C)} / "
                         f"{sum(not u['pred'][k]['top1'] for u in C)}"]
                    for thr in ((2, 5) if k == "acoustic" else (0.5, 1.0)):
                        A = [u for u in U if u["pred"][k].get("margin", -INF) > thr]
                        r.append(f"{sum(u['pred'][k]['top1'] for u in A)} / {sum(not u['pred'][k]['top1'] for u in A)}")
                    for pt in (0.5, 0.9):
                        A = [u for u in U if u["pred"][k].get("ptop", 0) >= pt] if k != "acoustic" else None
                        r.append(f"{sum(u['pred'][k]['top1'] for u in A)} / {sum(not u['pred'][k]['top1'] for u in A)}"
                                 if A is not None else "-")
                    rows.append(r)
    md += table(rows, ["set", "window", "score", "units", "beats all decoys c / w", "margin > 2 (fused: 0.5) c / w",
                       "margin > 5 (fused: 1.0) c / w", "posterior >= 0.5 c / w", "posterior >= 0.9 c / w"])
    md += ("\nMargin = log top posterior minus the best decoy's (fused), or raw score minus the best decoy top "
           "(acoustic, as in RESULTS_conus.md); the two scales differ.\n\n")
    rows = []
    for grp, sets in (("US", ("webcam", "events", "speech")), ("A2O", ("a2o",))):
        for N in (2, 4, 8):
            for kind in KINDS:
                U = [u for u in units if u["set"] in sets and u["kind"] == kind and u["tau"] == 5 and u["N"] == N]
                if not U:
                    continue
                for k in ("acoustic", "fused pop+tz"):
                    C = [u for u in U if u["pred"][k].get("confident")]
                    A = [u for u in U if u["pred"][k].get("ptop", 0) >= 0.9] if k != "acoustic" else None
                    rows.append([grp, N, f"{kind} min +-5 s", k, len(U),
                                 f"{sum(u['pred'][k]['top1'] for u in C)} / {sum(not u['pred'][k]['top1'] for u in C)}",
                                 (f"{sum(u['pred'][k]['top1'] for u in A)} / {sum(not u['pred'][k]['top1'] for u in A)}"
                                  if A is not None else "-")])
    md += "Multi-session:\n\n" + table(rows, ["set", "N", "window", "score", "units", "beats all decoys c / w",
                                             "posterior >= 0.9 c / w"]) + "\n"
    # item 2 prior-conditioned
    for u in units:
        u["prior_cond"] = prior_cond(u, RI)
    md += "## 2. Value of the sound given a prior\n\n" + PRIORCOND_TEXT
    tiers2 = (0.1, 0.3, 1, 2, 5, 10)
    h2 = ["set", "window", "N", "units", "rule", "answers"] + [f"<={x:g} km" if x >= 1 else f"<={int(x * 1000)} m"
                                                               for x in tiers2] + ["closer than prior / farther"]
    for cond, lab in (("metro", "metro known (the true region)"), ("box40", "40 km box, centre jittered +-10 km"),
                      ("boxr5a", "r5a-like box +-0.35 deg, centre jittered +-0.25 deg")):
        rows = []
        for st in ("webcam", "events", "speech", "a2o"):
            for kind in KINDS:
                for tau in TAUS:
                    for N in (1, 2, 4, 8):
                        U = [u for u in units if u["set"] == st and u["kind"] == kind and u["tau"] == tau
                             and u["N"] == N]
                        if not U or (N > 1 and kind == "30" and st != "a2o"):
                            continue
                        P = [u["prior_cond"] for u in U]
                        n = len(U)
                        rows.append([st, f"{kind} min +-{tau} s", N, n, "prior alone", "-"]
                                    + [frac(sum(p[f"{cond}_prior"] <= x for p in P), n) for x in tiers2] + ["-"])
                        for nm in (("s2", "s1") if N == 1 else ("s2",)):
                            for rule in ("has", "beats"):
                                def dd(p, cond=cond, nm=nm, rule=rule):
                                    return p[f"{cond}_{nm}_dist"] if p[f"{cond}_{nm}_{rule}"] else p[f"{cond}_prior"]
                                rl = (f"sound ({'>=2' if nm == 's2' else '>=1'} voters)"
                                      + (", only if it beats every decoy there" if rule == "beats" else ""))
                                rows.append([st, f"{kind} min +-{tau} s", N, n, rl,
                                             frac(sum(p[f"{cond}_{nm}_{rule}"] for p in P), n)]
                                            + [frac(sum(dd(p) <= x for p in P), n) for x in tiers2]
                                            + [f"{sum(dd(p) < p[f'{cond}_prior'] for p in P)} / "
                                               f"{sum(dd(p) > p[f'{cond}_prior'] for p in P)}"])
        md += f"### {lab}\n\n" + table(rows, h2) + "\n"
    # r5a comparison: heard >= 1 (labelled positives), box r5a, +-30 s and +-5 s
    md += "### Comparison with round 5a in-box numbers (positives only: >= 1 labelled pass in the window)\n\n"
    rows = []
    for grp, sets in (("A2O", ("a2o",)), ("US webcams", ("webcam",)), ("US all", ("webcam", "events", "speech"))):
        for kind in KINDS:
            for tau in TAUS:
                U = [u for u in units if u["set"] in sets and u["kind"] == kind and u["tau"] == tau and u["N"] == 1
                     and u["heard_total"] >= 1]
                if not U:
                    continue
                P = [u["prior_cond"] for u in U]
                n = len(U)
                for nm in ("s2", "s1"):
                    def dd(p, nm=nm):
                        return p[f"boxr5a_{nm}_dist"] if p[f"boxr5a_{nm}_has"] else INF
                    rows.append([grp, f"{kind} min +-{tau} s", nm, n,
                                 fci(U, lambda u, nm=nm: (u["prior_cond"][f"boxr5a_{nm}_dist"] <= 2
                                                          and u["prior_cond"][f"boxr5a_{nm}_has"]))]
                                + [frac(sum(dd(p) <= x for p in P), n) for x in (0.1, 0.3, 1, 5, 10)])
    md += table(rows, ["set", "window", "list", "positives", "<=2 km (95 % CI)", "<=100 m", "<=300 m", "<=1 km",
                       "<=5 km", "<=10 km"])
    md += "\n"
    # verify vs find
    md += "### Verify a claimed region vs find the region (N = 1 unless stated)\n\n" + VERIFY_TEXT
    rows = []
    for st in ("webcam", "events", "speech", "a2o"):
        for kind in KINDS:
            for tau in TAUS:
                for N in (1, 2, 4, 8):
                    U = [u for u in units if u["set"] == st and u["kind"] == kind and u["tau"] == tau and u["N"] == N]
                    if not U or (N > 1 and (kind == "30" or tau == 30)):
                        continue
                    P = [u["prior_cond"] for u in U]
                    n = len(U)
                    findA = sum(u["pred"]["acoustic"]["top1"] for u in U)
                    findF = sum(u["pred"]["fused pop+tz"]["top1"] for u in U)
                    for nm in (("s2", "s1") if N == 1 else ("s2",)):
                        for rule in ("listed", "beats"):
                            nnv = [p for p in P if p["nn_claim"] >= 0]
                            rows.append([st, f"{kind} min +-{tau} s", N, n, f"{nm} {rule}",
                                         frac(sum(p[f"ver_{nm}_{rule}_true"] for p in P), n),
                                         pct(np.mean([p[f"ver_{nm}_{rule}_falsepop"] for p in P])),
                                         frac(sum(p[f"ver_{nm}_{rule}_falsenn"] for p in nnv), len(nnv)),
                                         frac(findA, n), frac(findF, n)])
    md += table(rows, ["set", "window", "N", "units", "accept rule", "true claim accepted",
                       "false claim accepted (population-weighted, same UTC offset)",
                       "false claim accepted (nearest other big region)", "find: acoustic top-1",
                       "find: acoustic + pop + tz top-1"]) + "\n"
    # by heard
    md += "### Verify by labelled passes heard (60 min, +-5 s, N = 1; rule: s1 beats decoys)\n\n"
    rows = []
    for st in ("webcam", "events", "speech", "a2o"):
        for lab, f in (("0", lambda h: h == 0), ("1", lambda h: h == 1), ("2+", lambda h: h >= 2)):
            U = [u for u in units if u["set"] == st and u["kind"] == "60" and u["tau"] == 5 and u["N"] == 1
                 and f(u["heard_total"])]
            if not U:
                continue
            P = [u["prior_cond"] for u in U]
            n = len(U)
            nnv = [p for p in P if p["nn_claim"] >= 0]
            rows.append([st, lab, n, frac(sum(p["ver_s1_beats_true"] for p in P), n),
                         pct(np.mean([p["ver_s1_beats_falsepop"] for p in P])),
                         frac(sum(p["ver_s1_beats_falsenn"] for p in nnv), len(nnv)),
                         frac(sum(u["pred"]["fused pop+tz"]["top1"] for u in U), n),
                         frac(sum(p["metro_s1_beats"] and p["metro_s1_dist"] <= 2 for p in P), n)])
    md += table(rows, ["set", "heard", "units", "true claim accepted", "false (pop-weighted)", "false (nearest big)",
                       "find (fused top-1)", "metro known: verified and <= 2 km"]) + "\n"
    return md




PRIORCOND_TEXT = """Computed from the same saved outputs, restricted afterwards. The saved output holds only the
best cell of each of the top 40 regions (refined for the top 8), so these in-prior numbers are a lower bound on what
the sound adds inside a prior. Prior alone = the region's population centroid (metro known) or the box centre (box).
Sound = the best saved cell inside the region / box, else the prior guess; "beats every decoy there" keeps it only
when its score beats every decoy variant inside the same region / box.

"""

VERIFY_TEXT = """A claim names a region. "listed" accepts it when the sound produced any scored cell in it; "beats" also
needs that region's real score to beat every decoy variant at the same region. False claims: the other regions in
the same UTC offset weighted by population, or the nearest other region with >= 250 k people (CONUS) / the nearest
other SA4 (AU). "find" = top-1 region from the whole country.

"""
