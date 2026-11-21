"""Score country-search outputs against a truth file: region top-1 / top-3, distance tiers, decoy confidence.

The only part of the country search that opens the truth. It refuses outputs whose search touched the truth
directory (the guard records every work-directory path a search opened).

Definitions
  top-1 / top-3     the true region (CBSA or rural county; AU: SA4 2021) is the first / among the first three ranked
                    regions. No answer = not correct.
  distance tiers    great-circle distance from the best (refined) point to the true site.
  decoy p           (1 + #decoys whose top score >= the real top score) / (1 + K) with the unit's own decoys (clock
                    shifted +-10..25 min, other days); confident = p at its minimum (the real answer beats every
                    decoy). Pooled p: against every decoy top of the same set / window / clock / N; confident at
                    p_pool <= 0.01.
  plausible regions regions whose best score >= the best decoy top; "all" when the real top does not beat every decoy.
  heard events      labelled prop / rotor / unknown passes (truth side) with CPA inside the window.

Truth file: {oid: {set, site, lat, lon, clock, dur, windows, pos: [{t, cls}], clock_verified}}. Units whose real
search lacked an ADS-B day (no adsb.lol release) are excluded; decoys that lacked one are dropped from K.
"""
import datetime as dt
import glob
import json
import os

import numpy as np

from skyhum.country.common import Regions, frac, haversine_km, load_json
from skyhum.country.common import md_table as table

INF = float("inf")
SETS = ("webcam", "events", "speech", "a2o")


def lacks_day(v, sess, truth):
    """A session's traffic block [S0 - 900, S0 + dur + 400] (S0 = clock + shift) overlapped a day it lacked."""
    for i, s_ in enumerate(v.get("sessions", [])):
        t = truth[sess[i]]
        sh = v["shifts"][i] or 0.0
        a, b = float(t["clock"]) + sh - 900, float(t["clock"]) + sh + float(t["dur"]) + 400
        for d in s_.get("missing_days", []):
            d0 = dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp()
            if d0 <= b and d0 + 86400 >= a:
                return True
    return False


def touched_truth(r, truth_dir):
    return any(p.startswith(truth_dir) for p in r.get("touched", []))


def score_units(runs, truth, truth_dir, regions, channel="s2"):
    """Every scored unit (subset x window kind x clock tolerance) of every finished output in runs."""
    units, bad = [], []
    tdir = os.path.realpath(truth_dir)
    for f in sorted(glob.glob(os.path.join(runs, "*.json"))):
        r = load_json(f)
        if not r.get("done"):
            continue
        if touched_truth(r, tdir):
            bad.append(f)
            continue
        reg = regions[r["region"]]
        real = next(v for v in r["variants"] if v["variant"] == "real")
        sess = r["sessions"]
        decs = [v for v in r["variants"] if v["variant"] != "real" and not lacks_day(v, sess, truth)]
        real_missing = lacks_day(real, sess, truth)
        t0 = truth[sess[0]]
        tr_rid = int(reg.of(t0["lat"], t0["lon"]))
        for si, X in enumerate(r["subsets"]):
            for kind in ("60", "30", "10"):
                for tau in (5, 30):
                    key = f"{si}|{kind}|{tau}"
                    rr = real["subsets"].get(key)
                    if rr is None:
                        if real["n_union"] == 0 and all(len([w for w in truth[sess[i]]["windows"] if w["kind"] == kind])
                                                        for i in X):
                            rr = {channel: {"top": []}}
                        else:
                            continue
                    if channel not in rr and rr.get("s2", {}).get("top"):
                        continue                       # output without this channel (tonal-only run)
                    top = rr.get(channel, {}).get("top", [])
                    dtops = [(v["subsets"].get(key, {}).get(channel, {}).get("top") or [[None, -INF]])[0][1]
                             for v in decs]
                    real_top = top[0][1] if top else -INF
                    K = len(dtops)
                    nbeat = sum(1 for d in dtops if d >= real_top)
                    p = (1 + nbeat) / (1 + K) if top else 1.0
                    thr = max(dtops) if dtops else -INF
                    nplaus = sum(1 for t in top if t[1] >= thr) if top and real_top > thr else INF
                    heard = []
                    for i in X:
                        t = truth[sess[i]]
                        w = [w for w in t["windows"] if w["kind"] == kind][0]
                        heard.append(sum(1 for q in (t.get("pos") or []) if w["t0"] <= q["t"] <= w["t1"]
                                         and q["cls"] in ("prop", "rotor", "unknown")))
                    u = dict(gid=r["gid"], set=t0["set"], site=t0["site"], N=len(X), sessions=[sess[i] for i in X],
                             kind=kind, tau=tau, region=r["region"], truth_rid=tr_rid, answered=bool(top), p=p, K=K,
                             margin=(real_top - thr) if top else -INF, n_plausible=nplaus, heard=heard,
                             heard_total=int(sum(heard)), heard_max=int(max(heard)),
                             clock_verified=all(truth[sess[i]].get("clock_verified", True) for i in X), dtops=dtops,
                             real_missing_days=real_missing, real_top=real_top)
                    if top:
                        u["pred_rid"] = [int(t[0]) for t in top[:3]]
                        u["top1"] = top[0][0] == tr_rid
                        u["top3"] = tr_rid in [t[0] for t in top[:3]]
                        u["dist_km"] = float(haversine_km(top[0][2], top[0][3], t0["lat"], t0["lon"]))
                        u["confident"] = p <= 1.0 / (1 + K) + 1e-9 and K > 0
                    else:
                        u.update(top1=False, top3=False, dist_km=INF, confident=False)
                    units.append(u)
    if bad:
        raise SystemExit(f"BLINDNESS VIOLATION: these outputs touched the truth directory: {bad}")
    pool = {}
    for u in units:
        pool.setdefault((u["set"], u["kind"], u["tau"], u["N"]), []).extend(u["dtops"])
    for u in units:
        P = np.sort(np.asarray(pool[(u["set"], u["kind"], u["tau"], u["N"])], float))
        if u["answered"] and P.size:
            u["p_pool"] = float((P.size - np.searchsorted(P, u["real_top"], "left") + 1) / (P.size + 1))
        else:
            u["p_pool"] = 1.0
        u["conf_pool"] = u["answered"] and u["p_pool"] <= 0.01
    return units


def run(runs, truth_path, static, out_prefix, channel="s2", region=None, print_md=True):
    """Score runs/*.json; writes OUT.json (every unit) and OUT.md (tables). Returns the scored units."""
    truth = load_json(truth_path)
    R = {r: Regions(r, static) for r in ("CONUS", "AU") if os.path.exists(os.path.join(static, f"regions_{r}.npz"))}
    units = score_units(runs, truth, os.path.dirname(os.path.abspath(truth_path)), R, channel)
    if region:
        units = [u for u in units if u["region"] == region]
    with open(out_prefix + ".json", "w") as f:
        json.dump(units, f)
    nmiss = sum(1 for u in units if u.get("real_missing_days"))
    units = [u for u in units if not u.get("real_missing_days")]
    md = report(units, R, truth, static)
    md = (f"_Excluded: {nmiss} units whose real search lacked an ADS-B day (no adsb.lol release). Decoy runs that "
          f"lacked a day are dropped from K._\n\n") + md
    with open(out_prefix + ".md", "w") as f:
        f.write(md)
    if print_md:
        print(md)
    return units


def med(x):
    x = sorted(x)
    if not x:
        return "-"
    m = x[len(x) // 2] if len(x) % 2 else 0.5 * (x[len(x) // 2 - 1] + x[len(x) // 2])
    return "all" if m == INF else f"{m:.0f}"


def row(U):
    n = len(U)
    conf = [u for u in U if u["confident"]]
    return [str(n), frac(sum(u["top1"] for u in U), n), frac(sum(u["top3"] for u in U), n),
            frac(sum(u["dist_km"] <= 25 for u in U), n), frac(sum(u["dist_km"] <= 50 for u in U), n),
            frac(sum(u["dist_km"] <= 100 for u in U), n), med([u["n_plausible"] for u in U]),
            f"{sum(u['top1'] for u in conf)} / {sum(not u['top1'] for u in conf)}",
            f"{sum(u['top1'] for u in U if u['conf_pool'])} / {sum(not u['top1'] for u in U if u['conf_pool'])}"]


def baselines(R, static):
    """Population prior (CONUS): always the largest metro; population-weighted random region / point."""
    tab = R["CONUS"].table
    pops = np.array([t.get("pop", 0) if t.get("cells", 0) else 0 for t in tab], float)
    big = int(np.argmax(pops))
    top3 = list(np.argsort(-pops)[:3])
    cp = np.load(os.path.join(static, "county_pop.npz"))
    m = (cp["lat"] > 24.5) & (cp["lat"] < 49.5) & (cp["lon"] > -125) & (cp["lon"] < -66.9)
    return dict(big=big, top3=top3, pops=pops, tot=pops.sum(), clat=cp["lat"][m], clon=cp["lon"][m],
                cpop=cp["pop"][m].astype(float))


def base_rows(U, B, R, truth):
    tab = R["CONUS"].table
    n = len(U)
    if not n:
        return []
    big = B["big"]
    t_big = sum(u["truth_rid"] == big for u in U)
    t3 = sum(u["truth_rid"] in B["top3"] for u in U)
    out = []
    tr = [(truth[u["sessions"][0]]["lat"], truth[u["sessions"][0]]["lon"]) for u in U]
    d_big = [float(haversine_km(tab[big]["lat"], tab[big]["lon"], la, lo)) for la, lo in tr]
    out.append(["largest metro always (" + tab[big]["name"].split(",")[0] + ")", str(n), frac(t_big, n), frac(t3, n)]
               + [frac(sum(d <= x for d in d_big), n) for x in (25, 50, 100)])
    pw = [B["pops"][u["truth_rid"]] / B["tot"] if u["truth_rid"] >= 0 else 0 for u in U]
    pk = []
    for x in (25, 50, 100):
        s = 0.0
        for la, lo in tr:
            d = haversine_km(B["clat"], B["clon"], la, lo)
            s += B["cpop"][d <= x].sum() / B["cpop"].sum()
        pk.append(f"{100 * s / n:.1f} % (expected)")
    out.append(["population-weighted random region / person", str(n), f"{100 * np.mean(pw):.2f} % (expected)",
                f"~{100 * min(1, 3 * np.mean(pw)):.2f} % (expected, 3 draws)"] + pk)
    return out


def grp_multi(u):
    return u["gid"].startswith("g") or u["N"] == 1


def report(units, R, truth, static):
    head = ["units", "top-1 metro", "top-3 metro", "<=25 km", "<=50 km", "<=100 km", "median plausible",
            "confident (own decoys) correct / wrong", "confident (pooled p<=0.01) correct / wrong"]
    md = "## Country search: region identification (single sessions)\n\n"
    S = [u for u in units if u["N"] == 1]
    for st in SETS:
        rows = []
        for kind in ("60", "30", "10"):
            for tau in (5, 30):
                U = [u for u in S if u["set"] == st and u["kind"] == kind and u["tau"] == tau]
                if U:
                    rows.append([f"{kind} min", f"+-{tau} s"] + row(U))
        if rows:
            md += (f"### {st}{' (Australia-wide)' if st == 'a2o' else ''}\n\n"
                   + table(rows, ["window", "clock"] + head) + "\n")
    us = [u for u in S if u["region"] == "CONUS" and u["kind"] == "60" and u["tau"] == 5]
    if us and "CONUS" in R and os.path.exists(os.path.join(static, "county_pop.npz")):
        B = baselines(R, static)
        md += "### Population-prior baselines (CONUS sessions, 60 min windows)\n\n"
        md += table(base_rows(us, B, R, truth), ["baseline", "units", "top-1", "top-3", "<=25 km", "<=50 km",
                                                 "<=100 km"]) + "\n"
    md += "### By heard events in the window (60 min, +-5 s)\n\n"
    rows = []
    for st in SETS:
        for lab, f in (("0", lambda h: h == 0), ("1", lambda h: h == 1), ("2", lambda h: h == 2),
                       ("3+", lambda h: h >= 3)):
            U = [u for u in S if u["set"] == st and u["kind"] == "60" and u["tau"] == 5 and f(u["heard_total"])]
            if U:
                rows.append([st, lab] + row(U))
    md += table(rows, ["set", "heard"] + head) + "\n"
    md += "### Coverage vs confident-wrong (60 min, +-5 s; answers ranked by margin over the best decoy)\n\n"
    rows = []
    for grp, sets in (("US (webcam+events+speech)", ("webcam", "events", "speech")), ("A2O", ("a2o",))):
        U = [u for u in S if u["set"] in sets and u["kind"] == "60" and u["tau"] == 5]
        if not U:
            continue
        for thr in (-INF, 0, 2, 5, 10, 20):
            A = [u for u in U if u["answered"] and u["margin"] > thr]
            rows.append([grp, "any answer" if thr == -INF else f"margin > {thr}", frac(len(A), len(U)),
                         frac(sum(u["top1"] for u in A), len(A)), frac(sum(not u["top1"] for u in A), len(A))])
    md += table(rows, ["set", "threshold", "coverage", "correct metro", "wrong metro"]) + "\n"
    md += "## Multi-session, country only\n\n"
    M = [u for u in units if u["kind"] in ("60", "30")]
    headm = ["units", "top-1", "top-3", "<=100 m", "<=300 m", "<=1 km", "<=2 km", "<=25 km", "<=50 km", "<=100 km",
             "median plausible", "confident (own) c / w", "confident (pooled) c / w", ">=2 heard (cum.)"]
    for st in ("webcam", "a2o", "events", "speech"):
        rows = []
        for kind in ("60", "30"):
            for tau in (5, 30):
                for N in (1, 2, 4, 8, 16):
                    U = [u for u in M if u["set"] == st and u["kind"] == kind and u["tau"] == tau and u["N"] == N
                         and grp_multi(u)]
                    if not U:
                        continue
                    conf = [u for u in U if u["confident"]]
                    n = len(U)
                    rows.append([f"{kind} min", f"+-{tau} s", str(N), str(n), frac(sum(u["top1"] for u in U), n),
                                 frac(sum(u["top3"] for u in U), n)]
                                + [frac(sum(u["dist_km"] <= x for u in U), n) for x in (0.1, 0.3, 1, 2, 25, 50, 100)]
                                + [med([u["n_plausible"] for u in U]),
                                   f"{sum(u['top1'] for u in conf)} / {sum(not u['top1'] for u in conf)}",
                                   f"{sum(u['top1'] for u in U if u['conf_pool'])} / "
                                   f"{sum(not u['top1'] for u in U if u['conf_pool'])}",
                                   frac(sum(u["heard_total"] >= 2 for u in U), n)])
        if rows:
            md += f"### {st}\n\n" + table(rows, ["session", "clock", "N"] + headm) + "\n"
    md += "### Accumulation of individually weak sessions (every session <= 1 heard pass; 60 min, +-5 s)\n\n"
    rows = []
    for st in ("webcam", "a2o"):
        for N in (1, 2, 4, 8, 16):
            U = [u for u in M if u["set"] == st and u["kind"] == "60" and u["tau"] == 5 and u["N"] == N
                 and grp_multi(u) and u["heard_max"] <= 1]
            if U:
                conf = [u for u in U if u["confident"]]
                n = len(U)
                rows.append([st, str(N), str(n), frac(sum(u["top1"] for u in U), n),
                             frac(sum(u["dist_km"] <= 25 for u in U), n),
                             f"{sum(u['top1'] for u in conf)} / {sum(not u['top1'] for u in conf)}",
                             f"{sum(u['top1'] for u in U if u['conf_pool'])} / "
                             f"{sum(not u['top1'] for u in U if u['conf_pool'])}",
                             frac(sum(u["heard_total"] >= 2 for u in U), n)])
    md += table(rows, ["set", "N", "units", "top-1", "<=25 km", "confident (own) c / w", "confident (pooled) c / w",
                       ">=2 heard (cum.)"])
    return md
