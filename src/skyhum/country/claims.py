"""Verify a claimed place from saved country-search outputs (post-processing, CPU).

A claim is an area: the metro (CBSA / rural county; AU: SA4), a 40 km x 40 km box, or a 5 km x 5 km neighbourhood.
Claim score of a variant = best saved cell score inside the area (s2 list: cells with >= 2 voters, the best cell of
each of the top 40 regions; -inf if none). Decoy margin = real claim score - max(0, max over the decoy variants (wrong
clock / other day) of their claim score). Accept if margin > theta (theta = 0: "beats every decoy and scores above 0").
True claim: the true metro; a box / neighbourhood containing the true site (centre jittered +-10 / +-1.5 km).
False claims: (tz) every other region in the same UTC offset, population-weighted (box / neighbourhood centred on its
population centroid); (nn) the nearest other region (CONUS >= 250 k people; AU any SA4); (d10-25) 8 claims per unit
centred 10-25 km from the true site in random directions. A false claim whose area contains the true site (or, at
metro level, whose region is the true one) is dropped. The truth is opened on the scoring side only.

These are operating points of this saved-output verifier (only the best cell of each of the top 40 regions is kept):
lower bounds on true accept; its false accept can move either way with a verifier that scores every cell.
"""
import math
import os

import numpy as np

from skyhum.country import prior as PF
from skyhum.country.common import haversine_km, load_json

INF = float("inf")
THETAS = [-INF, 0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
LEVELS = (("metro", None, None), ("box40", 20.0, 10.0), ("nbhd5", 2.5, 1.5))
FALSE = ("tz", "nn", "d10-25")
NSAMP = 8


def hrand(key, k=2):
    import hashlib
    h = hashlib.sha1(key.encode()).digest()
    return [int.from_bytes(h[4 * i:4 * i + 4], "big") / 2 ** 32 for i in range(k)]


def box_scores(cells, cla, clo, half):
    """cells = (rid, sc, la, lo, full); claims (M,) centres -> (M,) best score inside each half-km box"""
    rid, sc, la, lo, _ = cells
    M = len(cla)
    if not rid.size:
        return np.full(M, -INF)
    inside = ((np.abs(la[None, :] - cla[:, None]) * 111.132 <= half)
              & (np.abs(lo[None, :] - clo[:, None]) * 111.32 * np.cos(np.radians(cla))[:, None] <= half))
    return np.where(inside, sc[None, :], -INF).max(1)


def region_scores(cells, rr):
    rid, sc, _, _, _ = cells
    out = np.full(len(rr), -INF)
    if rid.size:
        d = dict(zip(rid.tolist(), sc.tolist()))
        out = np.array([d.get(int(r), -INF) for r in rr])
    return out


def margins(u, level, half, claims):
    """claims: dict(cla, clo) or rids -> margin array"""
    if level == "metro":
        f = lambda c: region_scores(c, claims)  # noqa: E731
    else:
        f = lambda c: box_scores(c, claims[0], claims[1], half)  # noqa: E731
    real = f(u["s2"])
    if not u["d2"]:
        return np.full(len(real), -INF)
    # decoy level of the claim: the best decoy score inside the area, floored at 0 (an area no decoy scores counts as
    # 0 = no aircraft evidence, so a real score must also be positive); keeps every margin finite
    dec = np.maximum(np.max([f(d) for d in u["d2"]], 0), 0.0)
    m = real - dec
    m = np.where(np.isneginf(real), -INF, m)
    return np.where(np.isnan(m), -INF, m)


def contains(cla, clo, half, lat, lon):
    return ((np.abs(lat - cla) * 111.132 <= half) & (np.abs(lon - clo) * 111.32 * np.cos(np.radians(cla)) <= half))


def unit_claims(u, RI):
    R = RI[u["region"]]
    tr = u["tr"]
    out = {}
    key = u["sess0"] + "|" + "|".join(u["sessions"])
    # candidate false regions: same offset, valid, not the truth region
    m = u["mask_tz"].copy()
    if tr >= 0:
        m[tr] = False
    cand = np.flatnonzero(m)
    w = R.pop[cand]
    big = R.pop >= (250000 if R.R == "CONUS" else 0)
    dd = haversine_km(R.plat, R.plon, u["lat"], u["lon"])
    nnc = np.flatnonzero(m & big)
    nn = int(nnc[np.argmin(dd[nnc])]) if nnc.size else -1
    # 10-25 km displaced points
    pts = []
    for k in range(64):
        a, b = hrand(key + f"|d{k}")
        dkm = 10 + 15 * a
        th = 2 * math.pi * b
        pts.append((u["lat"] + dkm * math.cos(th) / 111.132,
                    u["lon"] + dkm * math.sin(th) / (111.32 * math.cos(math.radians(u["lat"])))))
    pts = np.array(pts)
    for level, half, J in LEVELS:
        o = {}
        if level == "metro":
            o["true"] = margins(u, level, half, np.array([tr]))[0] if tr >= 0 else -INF
            o["tz"] = (margins(u, level, half, cand), w)
            o["nn"] = margins(u, level, half, np.array([nn]))[0] if nn >= 0 else None
            rp = np.atleast_1d(R.reg.of(pts[:, 0], pts[:, 1]))
            ok = [i for i in range(len(pts)) if rp[i] >= 0 and rp[i] != tr and R.valid[rp[i]]][:NSAMP]
            o["d10-25"] = margins(u, level, half, rp[ok]) if ok else np.zeros(0)
        else:
            a, b = hrand(key + f"|t{level}")
            dy, dx = (2 * a - 1) * J, (2 * b - 1) * J
            cla = u["lat"] + dy / 111.132
            clo = u["lon"] + dx / (111.32 * math.cos(math.radians(u["lat"])))
            o["true"] = margins(u, level, half, (np.array([cla]), np.array([clo])))[0]
            cl, co = R.plat[cand], R.plon[cand]
            keep = ~contains(cl, co, half, u["lat"], u["lon"])
            o["tz"] = (margins(u, level, half, (cl[keep], co[keep])), w[keep])
            if nn >= 0 and not contains(R.plat[nn], R.plon[nn], half, u["lat"], u["lon"]):
                o["nn"] = margins(u, level, half, (np.array([R.plat[nn]]), np.array([R.plon[nn]])))[0]
            else:
                o["nn"] = None
            keep = ~contains(pts[:, 0], pts[:, 1], half, u["lat"], u["lon"])
            P = pts[keep][:NSAMP]
            o["d10-25"] = margins(u, level, half, (P[:, 0], P[:, 1])) if len(P) else np.zeros(0)
        out[level] = o
    return out


# --------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def unit_rates(u, level, th):
    """per-unit (true accept 0/1, {false type: (accepted, count) or weighted rate})"""
    o = u["claims"][level]
    t = float(o["true"] > th)
    f = {}
    m, w = o["tz"]
    f["tz"] = (float((w * (m > th)).sum() / w.sum()), 1.0) if w.sum() > 0 else None
    f["nn"] = (float(o["nn"] > th), 1.0) if o["nn"] is not None else None
    d = o["d10-25"]
    f["d10-25"] = (float((d > th).sum()), float(d.size)) if d.size else None
    return t, f


def boot(sites, num, den, B=2000, seed=0):
    sites = np.asarray(sites)
    us = sorted(set(sites.tolist()))
    S = np.array([num[sites == s].sum() for s in us])
    C = np.array([den[sites == s].sum() for s in us])
    if C.sum() == 0:
        return (0.0, 0.0)
    idx = np.random.default_rng(seed).integers(0, len(us), (B, len(us)))
    v = S[idx].sum(1) / np.maximum(C[idx].sum(1), 1e-12)
    return (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5))


def stats(U, level, th):
    """dict: true (k, n, ci), false type -> (rate %, ci, k, n, wilson)"""
    if not U:
        return None
    sites = [u["site"] for u in U]
    T = np.array([unit_rates(u, level, th)[0] for u in U])
    res = {"true": (int(T.sum()), len(U), boot(sites, T, np.ones(len(U))))}
    for ft in FALSE:
        rows = [(u["site"], unit_rates(u, level, th)[1][ft]) for u in U]
        rows = [(s, r) for s, r in rows if r is not None]
        if not rows:
            res[ft] = None
            continue
        s_ = [s for s, _ in rows]
        num = np.array([r[0] for _, r in rows])
        den = np.array([r[1] for _, r in rows])
        k, n = num.sum(), den.sum()
        res[ft] = (100 * k / n, boot(s_, num, den), k, n, wilson(int(round(k)), int(round(n))) if ft != "tz" else None)
    return res


def ftxt(r, ft):
    if r is None:
        return "-"
    if ft == "tz":
        return f"{r[0]:.1f} % ({r[1][0]:.1f}-{r[1][1]:.1f})"
    lo, hi = r[4]
    return (f"{int(r[2])}/{int(r[3])} = {r[0]:.1f} % (boot {r[1][0]:.1f}-{r[1][1]:.1f}; "
            f"Wilson {100 * lo:.1f}-{100 * hi:.1f})")


def ttxt(r):
    k, n, ci = r
    return f"{k}/{n} ({100 * k / n:.0f} %; {ci[0]:.0f}-{ci[1]:.0f})" if n else "-"


def table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def thname(th):
    return "any scored cell" if th == -INF else f"> {th:g}"


def worst_fa(res):
    """max point false-accept over the false-claim types (%)"""
    return max((res[ft][0] for ft in FALSE if res.get(ft)), default=0.0)


def pick_theta(U, level, cap=1.0):
    for th in THETAS:
        r = stats_fast(U, level, th)
        if r <= cap:
            return th
    return INF


def stats_fast(U, level, th):
    """max over false types of pooled point false-accept (%), no bootstrap"""
    best = 0.0
    for ft in FALSE:
        num = den = 0.0
        for u in U:
            f = unit_rates(u, level, th)[1][ft]
            if f is not None:
                num += f[0]
                den += f[1]
        if den:
            best = max(best, 100 * num / den)
    return best


# --------------------------------------------------------------------------------------------- report
def report(units, calls, tag):
    md = f"## Verifying a claimed place ({tag} score)\n\n" + INTRO
    LEVN = {"metro": "metro (region)", "box40": "40 km box", "nbhd5": "5 km neighbourhood"}
    groups = (("webcam", ("webcam",)), ("A2O", ("a2o",)), ("events", ("events",)), ("speech", ("speech",)))
    # 1. verify vs N at theta 0
    md += "### Verify vs N (accept = margin > 0: beats every decoy and scores above 0)\n\n"
    head = ["set", "window", "N", "units", "claim", "true accepted (95 % CI)", "false: same UTC offset, pop-weighted",
            "false: next region over", "false: 10-25 km away"]
    for kind, tau in (("60", 5), ("60", 30), ("30", 5), ("30", 30)):
        rows = []
        for gname, sets in groups:
            for N in (1, 2, 4, 8):
                U = [u for u in units if u["set"] in sets and u["kind"] == kind and u["tau"] == tau and u["N"] == N]
                if not U or (gname in ("events", "speech") and N > 1):
                    continue
                for level, _, _ in LEVELS:
                    r = stats(U, level, 0.0)
                    rows.append([gname, f"{kind} min +-{tau} s", N, len(U), LEVN[level], ttxt(r["true"])]
                                + [ftxt(r[ft], ft) for ft in FALSE])
        md += f"**{kind} min, +-{tau} s**\n\n" + table(rows, head) + "\n"
    # 2. operating curves
    md += "### Operating curve: true accept vs false accept as the decoy-margin threshold rises (60 min, +-5 s)\n\n"
    md += ("Rows: threshold on (real claim score - best decoy claim score). The false-accept columns are point "
           "estimates (%); the next-region and 10-25 km columns are proportions of sampled false claims.\n\n")
    for gname, sets in (("webcam", ("webcam",)), ("A2O", ("a2o",)), ("US events+speech", ("events", "speech"))):
        for N in (1, 2, 4, 8):
            U = [u for u in units if u["set"] in sets and u["kind"] == "60" and u["tau"] == 5 and u["N"] == N]
            if not U or (gname.startswith("US ev") and N > 1):
                continue
            rows = []
            for level, _, _ in LEVELS:
                for th in THETAS:
                    r = stats_cheap(U, level, th)
                    rows.append([LEVN[level], thname(th), f"{r['true']:.1f} %"]
                                + [f"{r[ft]:.2f} %" if r[ft] is not None else "-" for ft in FALSE])
            md += (f"**{gname}, N = {N} ({len(U)} units, {len({u['site'] for u in U})} sites)**\n\n"
                   + table(rows, ["claim", "threshold", "true accept", "false: same offset", "false: next region",
                                  "false: 10-25 km"]) + "\n")
    # 3. FA <= 1 %
    md += "### Threshold that keeps false accept <= 1 % (all three false-claim types; 60 min, +-5 s)\n\n" + FA1_TEXT
    rows = []
    for gname, sets in (("webcam", ("webcam",)), ("A2O", ("a2o",)), ("US events+speech", ("events", "speech"))):
        for N in (1, 2, 4, 8):
            U = [u for u in units if u["set"] in sets and u["kind"] == "60" and u["tau"] == 5 and u["N"] == N]
            if not U or (gname.startswith("US ev") and N > 1):
                continue
            for level, _, _ in LEVELS:
                th = pick_theta(U, level)
                r = stats(U, level, th) if th < INF else None
                # leave-one-site-out threshold choice
                lt = lf = 0.0
                ln = 0
                fnum = {ft: [0.0, 0.0] for ft in FALSE}
                for s in sorted({u["site"] for u in U}):
                    tr_ = [u for u in U if u["site"] != s]
                    te = [u for u in U if u["site"] == s]
                    ths = pick_theta(tr_, level)
                    for u in te:
                        t, f = unit_rates(u, level, ths)
                        lt += t
                        ln += 1
                        for ft in FALSE:
                            if f[ft] is not None:
                                fnum[ft][0] += f[ft][0]
                                fnum[ft][1] += f[ft][1]
                lf = " / ".join(f"{100 * a / b:.2f} %" if b else "-" for a, b in fnum.values())
                rows.append([gname, N, len(U), LEVN[level], thname(th) if th < INF else "none",
                             ttxt(r["true"]) if r else "0",
                             " / ".join(ftxt(r[ft], ft) if r and r[ft] else "-" for ft in FALSE) if r else "-",
                             f"{int(lt)}/{ln} ({100 * lt / ln:.0f} %)", lf])
    md += table(rows, ["set", "N", "units", "claim", "threshold (in-sample)", "true accept at it",
                       "false accept at it: offset / next region / 10-25 km", "LOSO: true accept",
                       "LOSO: false accept offset / next / 10-25 km"]) + "\n"
    # 4. real calls
    md += "### Verify through real calls (call-test v3: 4 webcams x 4 hours)\n\n"
    if not calls:
        md += "_Call-test v3 outputs were not complete when this was generated; see the watcher note._\n\n"
    else:
        md += CALL_TEXT
        Wth = {}
        for N in (1, 2, 4):
            Uw = [u for u in units if u["set"] == "webcam" and u["kind"] == "60" and u["tau"] == 5 and u["N"] == N]
            for level, _, _ in LEVELS:
                Wth[(N, level)] = pick_theta(Uw, level) if Uw else INF
        names = {"call_unproc": "unprocessed reference", "call_meet_out": "Google Meet received (outdoor)",
                 "call_tg_out": "Telegram received (outdoor)"}
        for kind, tau in (("60", 5), ("60", 30), ("30", 5)):
            rows = []
            for cond, cn in names.items():
                for N in (1, 2, 4):
                    U = [u for u in calls if u["set"] == cond and u["kind"] == kind and u["tau"] == tau and u["N"] == N]
                    if not U:
                        continue
                    for level, _, _ in LEVELS:
                        r0 = stats(U, level, 0.0)
                        th = Wth[(N, level)]
                        r1 = stats(U, level, th) if th < INF else None
                        rows.append([cn, N, len(U), LEVN[level], ttxt(r0["true"])]
                                    + [ftxt(r0[ft], ft) for ft in FALSE]
                                    + [thname(th) if th < INF else "none",
                                       ttxt(r1["true"]) if r1 else "-",
                                       " / ".join(f"{r1[ft][0]:.1f} %" if r1 and r1[ft] else "-" for ft in FALSE)])
            md += (f"**{kind} min, +-{tau} s**\n\n"
                   + table(rows, ["condition", "N", "units", "claim", "true accepted (margin > 0)",
                                  "false: same offset", "false: next region", "false: 10-25 km",
                                  "webcam threshold for <= 1 % (main run)", "true accepted at it",
                                  "false at it: offset / next / 10-25 km"]) + "\n")
    return md


def stats_cheap(U, level, th):
    out = {"true": 100 * np.mean([unit_rates(u, level, th)[0] for u in U])}
    for ft in FALSE:
        num = den = 0.0
        for u in U:
            f = unit_rates(u, level, th)[1][ft]
            if f is not None:
                num += f[0]
                den += f[1]
        out[ft] = 100 * num / den if den else None
    return out


INTRO = """Accept a claim when its area's best saved cell beats the same area in every decoy variant by more than a
threshold (an area no decoy scores counts as 0 for the decoys). Claims: metro, 40 km box, 5 km neighbourhood (centre
jittered). False claims: same UTC offset (population-weighted), next region over, 10-25 km away. CIs: true accept
95 % site-cluster bootstrap; sampled false claims count, bootstrap and Wilson.

"""

FA1_TEXT = """The threshold is the smallest one in the grid at which the pooled point false accept is <= 1 % for every
false-claim type. "In-sample" picks it on the same units it reports, so it is optimistic. "LOSO" picks it on the other
sites and applies it to the held-out site's units: this is the honest number. "none" = no threshold in the grid
reaches 1 %. With a few dozen sampled false claims per cell, 0 observed does not mean <= 1 %: read the Wilson and
bootstrap upper bounds in the verify-vs-N tables.

"""

CALL_TEXT = """The same country search on the received audio of real calls (desktop Google Meet and Telegram, plus the
same hours unprocessed). The last three columns apply the threshold that kept false accept <= 1 % on the main-run
webcams (same N, in-sample) to the call audio: a transfer, not refitted.

"""


def build(units, RI, tf):
    for u in units:
        R = RI[u["region"]]
        u["tr"] = int(R.reg.of(u["lat"], u["lon"]))
        u["off"], u["tzname"] = PF.site_offset(tf, u["lat"], u["lon"], u["clock"])
        u["mask_tz"] = R.mask(u["clock"], u["off"])
        u["claims"] = unit_claims(u, RI)
        for k in ("s1", "d1"):
            u.pop(k, None)
    return units


def run(runs, truth_path, static, out, channel="s2", tag="tonal", calls_runs=None, calls_truth=None,
        print_md=False):
    """Writes OUT/VERIFY_<tag>.md. calls_runs / calls_truth: optional outputs of the same search on call audio."""
    os.makedirs(out, exist_ok=True)
    tf = PF.timezone_finder()
    RI = {R: PF.RegionInfo(R, tf, static) for R in ("CONUS", "AU")}
    truth = load_json(truth_path)
    units = build(PF.load_units(runs, truth, os.path.dirname(os.path.abspath(truth_path)), channel), RI, tf)
    calls = []
    if calls_runs and calls_truth and os.path.isdir(calls_runs):
        t3 = load_json(calls_truth)
        calls = build(PF.load_units(calls_runs, t3, os.path.dirname(os.path.abspath(calls_truth)), channel), RI, tf)
    md = report(units, calls, tag)
    with open(os.path.join(out, f"VERIFY_{tag}.md"), "w") as f:
        f.write(md)
    if print_md:
        print(md)
    return units, calls
