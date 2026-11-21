"""Scorers of the local (box) locator benchmark: round-5b tables and the single-event two-point table.

Inputs: a cases file (JSON list: id, lat, lon, cat, split, cluster, start_true, npos, source) and a runs directory
holding one locator output per case and clock mode, ``<runs>/<tag>/<id>.<mode>.json`` (``--plots 0`` outputs of
``skyhum.match.r5b``), with decoy outputs of the same tag in ``<runs>/<tag>__decoy/<id>.<mode>.<shift>.json``.

r5b tables  within 2 km with cluster-bootstrap CIs, accuracy tiers (<= 100 m ... <= 10 km), decoy p-value confidence
            and confident-wrong at fixed coverage, paired against a reference tag.
single event (clock +-5 s, one pass is enough): the answer is a pair, the top-1 site and its mirror image across the
            ground track of its strongest voter; tiers per category (hard = exactly one labelled positive, pos >= 2,
            neg none) and confident answers at decoy p <= pthr (correct = two-point error <= 2 km).
"""
import glob
import json
import os

import numpy as np

from skyhum.match.locator import mirror

TIERS = [(0.1, "<=100 m"), (0.3, "<=300 m"), (1.0, "<=1 km"), (2.0, "<=2 km"), (5.0, "<=5 km"), (10.0, "<=10 km")]


def hkm(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(h)))


def score_run(c, out):
    """One locator output (path without .json) against its case. A run that ended without output = no answer."""
    d = _load(out + ".json")
    info = _load(out + ".info.json")
    H = d.get("hypotheses", [])
    r = dict(id=c["id"], source=c.get("source"), cat=c["cat"], split=c.get("split"), npos=c.get("npos"),
             runtime_s=d.get("runtime_total_s"), window_h=info.get("window_h"), n_hyp=len(H),
             null_level=d.get("null_level"), n_passes=d.get("n_passes"))
    if not H:
        return dict(r, err_km=None, rank=None, rank500=None, gap=0.0, ratio=1.0, null_margin=0.0, score1=0.0)
    for h in H:
        h["_d"] = hkm(h["lat"], h["lon"], c["lat"], c["lon"])
    top = H[0]
    ru = next((h for h in H[1:] if hkm(h["lat"], h["lon"], top["lat"], top["lon"]) > 3.0), None)
    s2 = ru["score"] if ru else 0.0
    r.update(err_km=top["_d"], score1=top["score"], score2=s2, gap=top["score"] - s2,
             ratio=top["score"] / max(s2, 1e-3), null_margin=top["score"] - d.get("null_level", 0.0),
             rank=next((k + 1 for k, h in enumerate(H) if h["_d"] < 2.0), None),
             rank500=next((k + 1 for k, h in enumerate(H) if h["_d"] < 0.5), None),
             clock_err_s=top["start_utc"] - c["start_true"] if "start_true" in c else None,
             n_voting=top.get("n_voting"))
    return r


def _load(p):
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def decoy_p(runs, tag, cid, mode, real_score):
    sc = []
    for f in glob.glob(os.path.join(runs, tag + "__decoy", f"{cid}.{mode}.*.json")):
        if f.endswith(".info.json"):
            continue
        try:
            H = _load(f).get("hypotheses", [])
        except (ValueError, OSError):
            continue
        sc.append(float(H[0]["score"]) if H else -1e9)
    if not sc or real_score is None:
        return None, len(sc)
    return (1 + sum(1 for s in sc if s >= real_score)) / (len(sc) + 1), len(sc)


def rows_for(runs, tag, cases, mode, use_decoy):
    R = {}
    for c in cases:
        f = os.path.join(runs, tag, f"{c['id']}.{mode}")
        if not (os.path.exists(f + ".json") or os.path.exists(f + ".info.json")):
            continue
        if os.path.exists(f + ".log") and not os.path.exists(f + ".json"):
            with open(f + ".log", errors="ignore") as fh:
                t = fh.read()
            if "OutOfMemory" in t or "OUT_OF_MEMORY" in t or "CUDA error" in t:
                continue                 # crashed: not a result
        r = score_run(c, f)
        H = _load(f + ".json").get("hypotheses", [])
        r["top2"] = any(hkm(h["lat"], h["lon"], c["lat"], c["lon"]) < 2 for h in H[:2])
        r["err2_km"] = min((hkm(h["lat"], h["lon"], c["lat"], c["lon"]) for h in H[:2]), default=None)
        r["ok"] = r["err_km"] is not None and r["err_km"] < 2
        r["cluster"] = c.get("cluster", c["id"])
        r["answered"] = bool(H)
        if use_decoy:
            p, k = decoy_p(runs, tag, c["id"], mode, H[0]["score"] if H else -1e9)
            r["p"], r["K"] = (p if H else 1.0), k
            r["conf"] = -np.log(r["p"]) if r["p"] is not None else None
        else:
            r["conf"] = r["gap"] if H else -1e9
        R[c["id"]] = r
    return R


def boot_ci(rows, key, B, seed=0):
    cl = sorted({r["cluster"] for r in rows})
    by = {c: [r for r in rows if r["cluster"] == c] for c in cl}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        pick = rng.choice(len(cl), len(cl))
        rr = [r for i in pick for r in by[cl[i]]]
        v.append(np.mean([r[key] for r in rr]) if rr else np.nan)
    return np.nanpercentile(v, [2.5, 97.5])


def tier_cells(rows, key, B):
    out = []
    for t, _ in TIERS:
        for r in rows:
            r["_t"] = r[key] is not None and r[key] <= t
        lo, hi = boot_ci(rows, "_t", B)
        out.append(f"{sum(r['_t'] for r in rows)} ({lo:.0%}-{hi:.0%})")
    med = np.median([r[key] if r[key] is not None else 1e4 for r in rows])
    return out, (f"{med:.2f}" if med < 1e3 else "no answer")


def at_coverage(pos, ctl, cov):
    cs = sorted([r["conf"] for r in pos if r["conf"] is not None], reverse=True)
    if not cs:
        return None
    k = max(1, int(round(cov * len(pos))))
    thr = cs[min(k, len(cs)) - 1]
    pa = [r for r in pos if r["conf"] is not None and r["conf"] >= thr]
    ca = [r for r in ctl if r["conf"] is not None and r["conf"] >= thr]
    return dict(thr=thr, pos_answered=len(pa), pos_correct=sum(r["ok"] for r in pa),
                ctl_answered=len(ca), ctl_wrong=sum(not r["ok"] for r in ca))


def r5b_tables(cases, runs, tags, modes=("blind", "clk30m", "clk30s", "wide"), pos_cat="pos", decoy=False,
               boot=2000, cov=(0.2, 0.3), pthr=0.05, tiers=True, top2=False):
    """Markdown tables and a dict {mode: {tag: {n, ok, ci, top2, tiers}}}."""
    out, lines, tiers_out = {}, [], []
    lines.append("| mode | tag | n pos | <2 km (cluster 95 % CI) | top-2 <2 km | median km | vs " + tags[0]
                 + " better/worse rank | controls n | "
                 + " | ".join(f"cov {c}: pos ans/correct, ctl wrong" for c in cov)
                 + (f" | p<={pthr}: pos ans/correct, ctl ans/wrong" if decoy else "") + " |")
    lines.append("|---" * (9 + len(cov) + (1 if decoy else 0)) + "|")
    for mode in modes:
        R = {t: rows_for(runs, t, cases, mode, decoy) for t in tags}
        common = sorted(set.intersection(*[set(r) for r in R.values()])) if R else []
        if not common:
            continue
        for t in tags:
            rows = [R[t][i] for i in common]
            pos = [r for r in rows if r["cat"] == pos_cat]
            ctl = [r for r in rows if r["cat"] != pos_cat and r["cat"] != "pos"]
            if not pos:
                continue
            ok = np.mean([r["ok"] for r in pos])
            lo, hi = boot_ci(pos, "ok", boot)
            t2 = sum(r["top2"] for r in pos)

            def rk(r):
                return r["rank"] or 999
            bw = "-" if t == tags[0] else (
                f"{sum(rk(R[t][i]) < rk(R[tags[0]][i]) for i in common if R[t][i]['cat'] == pos_cat)} / "
                f"{sum(rk(R[t][i]) > rk(R[tags[0]][i]) for i in common if R[t][i]['cat'] == pos_cat)}")
            med = np.median([r["err_km"] if r["err_km"] is not None else 1e4 for r in pos])
            covs = []
            for cv in cov:
                x = at_coverage(pos, ctl, float(cv))
                covs.append("-" if x is None
                            else f"{x['pos_answered']}/{x['pos_correct']}, {x['ctl_wrong']}/{len(ctl)}")
            extra = ""
            if decoy:
                pa = [r for r in pos if r.get("p") is not None and r["p"] <= pthr and r["answered"]]
                ca = [r for r in ctl if r.get("p") is not None and r["p"] <= pthr and r["answered"]]
                extra = f" | {len(pa)}/{sum(r['ok'] for r in pa)}, {len(ca)}/{sum(not r['ok'] for r in ca)}"
            lines.append(f"| {mode} | {t} | {len(pos)} | {int(round(ok * len(pos)))} ({ok:.0%}, {lo:.0%}-{hi:.0%}) | "
                         f"{t2} | {med:.1f} | {bw} | {len(ctl)} | " + " | ".join(covs) + extra + " |")
            out.setdefault(mode, {})[t] = dict(n=len(pos), ok=int(round(ok * len(pos))), ci=[float(lo), float(hi)],
                                               top2=int(t2),
                                               tiers=[sum(1 for r in pos if r["err_km"] is not None
                                                          and r["err_km"] <= x) for x, _ in TIERS])
        if tiers:
            tiers_out.append((mode, R, common))
    if tiers:
        keys = [("err_km", "top-1")] + ([("err2_km", "better of top 2")] if top2 else [])
        lines += ["", "Accuracy tiers (positives; count and cluster-bootstrap 95 % CI; median error km):", "",
                  "| mode | tag | point | n | " + " | ".join(n for _, n in TIERS) + " | median km |",
                  "|---" * (5 + len(TIERS)) + "|"]
        for mode, R, common in tiers_out:
            for t in tags:
                pos = [R[t][i] for i in common if R[t][i]["cat"] == pos_cat]
                if not pos:
                    continue
                for key, nm in keys:
                    cells, med = tier_cells(pos, key, boot)
                    lines.append(f"| {mode} | {t} | {nm} | {len(pos)} | " + " | ".join(cells) + f" | {med} |")
    return "\n".join(lines) + "\n", out


SE_TIERS = [0.1, 0.3, 1.0, 2.0, 5.0, 10.0]


def single_event(cases, runs, tag, mode="clk5s", pthr=0.05):
    """Markdown tables and per-case rows of the single-event (two-point) benchmark."""
    rows = []
    for c in cases:
        f = os.path.join(runs, tag, f"{c['id']}.{mode}.json")
        if not os.path.exists(f):
            continue
        H = _load(f).get("hypotheses", [])
        r = dict(id=c["id"], cat=c["cat"], cluster=c.get("cluster", c["id"]), e1=None, e2=None, eh2=None, p=None)
        if H:
            h = H[0]
            r["e1"] = hkm(h["lat"], h["lon"], c["lat"], c["lon"])
            m = mirror(h)
            r["em"] = hkm(m[0], m[1], c["lat"], c["lon"]) if m else None
            r["e2"] = min(x for x in (r["e1"], r["em"]) if x is not None)
            r["eh2"] = min(hkm(x["lat"], x["lon"], c["lat"], c["lon"]) for x in H[:2])
            r["score"] = h["score"]
            r["p"], r["K"] = decoy_p(runs, tag, c["id"], mode, h["score"])
        rows.append(r)
    md = [f"single-event mode {mode}, tag {tag}", "",
          "| category | n | point | answered | " + " | ".join(f"<={t:g} km" for t in SE_TIERS) + " |",
          "|---" * (4 + len(SE_TIERS)) + "|"]
    summary = {}
    for cat in ("hard", "pos", "neg"):
        R = [r for r in rows if r["cat"] == cat]
        if not R:
            continue
        for key, nm in (("e1", "top-1"), ("e2", "2-point (top-1 + mirror)"), ("eh2", "better of top-2 hyps")):
            cells = [sum(1 for r in R if r[key] is not None and r[key] <= t) for t in SE_TIERS]
            summary.setdefault(cat, {})[key] = cells
            md.append(f"| {cat} | {len(R)} | {nm} | {sum(r['e1'] is not None for r in R)} | "
                      + " | ".join(map(str, cells)) + " |")
    md += ["", f"Decoy null (clock decoys, p <= {pthr}); correct = 2-point error <= 2 km:", "",
           "| category | n | with decoys | confident | confident correct | confident wrong |",
           "|---|---|---|---|---|---|"]
    for cat in ("hard", "pos", "neg"):
        R = [r for r in rows if r["cat"] == cat]
        if not R:
            continue
        C = [r for r in R if r["p"] is not None and r["p"] <= pthr]
        ok = [r for r in C if cat != "neg" and r["e2"] is not None and r["e2"] <= 2]
        summary.setdefault(cat, {}).update(n=len(R), confident=len(C), correct=len(ok), wrong=len(C) - len(ok))
        md.append(f"| {cat} | {len(R)} | {sum(r['p'] is not None for r in R)} | {len(C)} | {len(ok)} | "
                  f"{len(C) - len(ok)} |")
    return "\n".join(md) + "\n", summary, rows


# ------------------------------------------------------------------------------------------------- gate calibration
M0 = [None, 0.0, 1.0, 2.0, 4.0, 8.0]


def _gate_rows(cases, runs, tag, mode, se):
    R = []
    for c in cases:
        f = os.path.join(runs, tag, f"{c['id']}.{mode}")
        if not os.path.exists(f + ".info.json") and not os.path.exists(f + ".json"):
            continue
        H = _load(f + ".json").get("hypotheses", [])
        if not H:
            R.append(dict(c=c, ans=False, p=1.0, m=-1e9, ok=False, K=0))
            continue
        h = H[0]
        s = float(h["score"])
        e = hkm(h["lat"], h["lon"], c["lat"], c["lon"])
        if se:
            mp = mirror(h)
            if mp is not None:
                e = min(e, hkm(mp[0], mp[1], c["lat"], c["lon"]))
        ds = []
        for g in glob.glob(os.path.join(runs, tag + "__decoy", f"{c['id']}.{mode}.*.json")):
            if g.endswith(".info.json"):
                continue
            hh = _load(g).get("hypotheses", [])
            ds.append(float(hh[0]["score"]) if hh else -1e9)
        K = len(ds)
        p = (1 + sum(d >= s for d in ds)) / (K + 1) if K else None
        R.append(dict(c=c, ans=True, p=p, m=(s - max(ds)) if ds else None, ok=e < 2.0, K=K))
    return R


def _conf(r, m0):
    if not r["ans"] or r["p"] is None or r["p"] > 0.05:
        return False
    return m0 is None or (r["m"] is not None and r["m"] >= m0)


def _boot(R, fn, B=2000, seed=0):
    cl = sorted({r["c"].get("cluster", r["c"]["id"]) for r in R})
    by = {k: [r for r in R if r["c"].get("cluster", r["c"]["id"]) == k] for k in cl}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        rr = [r for i in rng.choice(len(cl), len(cl)) for r in by[cl[i]]]
        x = fn(rr)
        if x is not None:
            v.append(x)
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)


def gate_table(cases, runs, tag, modes, se=False):
    """Decoy-gate calibration: per case p = (1 + #{decoy top >= real top}) / (K + 1), margin = real - best decoy;
    gate = p <= 0.05 (and margin >= m0). Precision = correct / confident over pos + hard + neg (correct: top-1 within
    2 km; single event: the two-point answer); cluster-bootstrap CIs. Returns (markdown, {mode: {gate: counts}})."""
    lines = ["| mode | gate | n pos / hard / neg (with full decoys) | confident pos (correct) "
             "| confident hard (correct) | confident neg | precision, all confident (95 % CI) "
             "| false-fix rate neg (95 % CI) | false-fix rate hard+neg wrong (95 % CI) |", "|---" * 9 + "|"]
    out = {}
    for mode in modes:
        R = _gate_rows(cases, runs, tag, mode, se)

        def cat(k, R=R):
            return [r for r in R if r["c"]["cat"] == k]

        def full(L):
            return sum(1 for r in L if not r["ans"] or (r["p"] is not None and (r["K"] >= 20 or r["p"] > 0.05)))
        for m0 in M0:
            C = [r for r in R if _conf(r, m0)]
            cp = [r for r in C if r["c"]["cat"] == "pos"]
            ch = [r for r in C if r["c"]["cat"] == "hard"]
            cn = [r for r in C if r["c"]["cat"] == "neg"]
            prec = sum(r["ok"] for r in C) / len(C) if C else None

            def fprec(rr, m0=m0):
                c = [r for r in rr if _conf(r, m0)]
                return sum(r["ok"] for r in c) / len(c) if c else None

            def fneg(rr, m0=m0):
                n = [r for r in rr if r["c"]["cat"] == "neg"]
                return sum(_conf(r, m0) for r in n) / len(n) if n else None

            def fctl(rr, m0=m0):
                n = [r for r in rr if r["c"]["cat"] != "pos"]
                return sum(_conf(r, m0) and not r["ok"] for r in n) / len(n) if n else None
            lo, hi = _boot(R, fprec)
            nlo, nhi = _boot(R, fneg)
            clo, chi = _boot(R, fctl)
            neg, ctl = cat("neg"), [r for r in R if r["c"]["cat"] != "pos"]
            g = "p <= 0.05" if m0 is None else f"p <= 0.05 & margin >= {m0:g}"
            pr = f"{sum(r['ok'] for r in C)}/{len(C)} = {prec:.2f} ({lo:.2f}-{hi:.2f})" if C else "-"
            wrong_ctl = sum(not r["ok"] for r in C if r["c"]["cat"] != "pos")
            lines.append(f"| {mode} | {g} | {len(cat('pos'))} / {len(cat('hard'))} / {len(neg)} ({full(cat('pos'))} / "
                         f"{full(cat('hard'))} / {full(neg)}) | {len(cp)} ({sum(r['ok'] for r in cp)}) | {len(ch)} "
                         f"({sum(r['ok'] for r in ch)}) | {len(cn)} | {pr} | {len(cn)}/{len(neg)} "
                         f"({nlo:.2f}-{nhi:.2f}) | {wrong_ctl}/{len(ctl)} ({clo:.2f}-{chi:.2f}) |")
            out.setdefault(mode, {})[g] = dict(confident=len(C), correct=sum(r["ok"] for r in C), pos=len(cp),
                                               pos_correct=sum(r["ok"] for r in cp), hard=len(ch), neg=len(cn),
                                               ctl_wrong=wrong_ctl)
    return "\n".join(lines) + "\n", out


# ------------------------------------------------------------------------------------------------- multi-session
def _load_map(prefix):
    f = prefix + ".grid.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    if "score0" not in z.files:
        return None
    return z["score0"].astype(np.float64), z["n0"].astype(np.float64), z["lats"], z["lons"]


def multi_session(plan, runs, tag, modes=("blind", "clk30s"), pthr=0.05, refined=True):
    """Same-sensor sets of N hours located on one shared box: the hours' LLR maps (<id>.<mode>.grid.npz) are summed,
    the joint top-1 must have >= 2 voters; fix = within 2 km of the sensor. Confident = pooled decoy p <= pthr (decoy
    maps <id>.<mode>.<k>.grid.npz). Sensor-weighted rates, sensor-bootstrap CIs. Returns (markdown, rows)."""
    tag_dir = os.path.join(runs, tag)
    rows = []
    for mode in modes:
        for s in plan["sets"]:
            maps = [_load_map(os.path.join(tag_dir, f"{h}.{mode}")) for h in s["hours"]]
            if any(m is None for m in maps):
                continue
            LA, LO = np.meshgrid(maps[0][2], maps[0][3], indexing="ij")
            LA, LO = LA.ravel(), LO.ravel()
            tl, tn = plan["hours"][s["hours"][0]]["lat"], plan["hours"][s["hours"][0]]["lon"]

            def joint(ms):
                Jm = sum(m[0] for m in ms)
                Nv = sum(m[1] for m in ms)
                Jv = np.where(Nv >= 2, Jm, -np.inf)
                i = int(np.argmax(Jv))
                return (float(Jv[i]) if np.isfinite(Jv[i]) else None), i
            top, i = joint(maps)
            dec = []
            ks = sorted({os.path.basename(f).split(f".{mode}.")[1][:-9] for h in s["hours"][:1]
                         for f in glob.glob(os.path.join(tag_dir, f"{h}.{mode}.*.grid.npz"))})
            for k in ks:
                dm = [_load_map(os.path.join(tag_dir, f"{h}.{mode}.{k}")) for h in s["hours"]]
                if any(m is None for m in dm):
                    continue
                dec.append(joint(dm)[0])
            err = None if top is None else hkm(LA[i], LO[i], tl, tn)
            if top is not None and refined:                # the joint 50 m refinement, where it was run
                fm = [_load_map(os.path.join(tag_dir, f"{h}.{mode}.f{i}")) for h in s["hours"]]
                if not any(m is None for m in fm):
                    fv, fi = joint(fm)
                    if fv is not None:
                        FA, FO = np.meshgrid(fm[0][2], fm[0][3], indexing="ij")
                        err = hkm(FA.ravel()[fi], FO.ravel()[fi], tl, tn)
            dk = [d if d is not None else -np.inf for d in dec]
            rows.append(dict(mode=mode, site=s["site"], N=s["N"], top=top, err_km=err,
                             ok=bool(err is not None and err < 2), decoys=dk))
    for r in rows:
        key = (r["mode"], r["site"], r["N"])
        pool = [d for q in rows if (q["mode"], q["site"], q["N"]) == key for d in q["decoys"]]
        r["p_pool"] = ((1 + sum(1 for d in pool if r["top"] is not None and d >= r["top"])) / (len(pool) + 1)
                       if (pool and r["top"] is not None) else 1.0)
    lines = [f"| mode | N | sets | sensors | fix <2 km | confident (p_pool <= {pthr}) | confident & wrong "
             "| median km | <=100 m | <=300 m | <=1 km | <=2 km | <=5 km | <=10 km |", "|---" * 14 + "|"]
    for mode in modes:
        for N in sorted({r["N"] for r in rows}):
            R = [r for r in rows if r["mode"] == mode and r["N"] == N]
            if not R:
                continue
            sites = sorted({r["site"] for r in R})
            w = np.array([1.0 / sum(q["site"] == r["site"] for q in R) for r in R])
            w = w / w.sum()
            fix = float(sum(wi for wi, r in zip(w, R) if r["ok"]))
            conf = [r["p_pool"] <= float(pthr) for r in R]
            cw = float(sum(wi for wi, r, c in zip(w, R, conf) if c and not r["ok"]))
            cf = float(sum(wi for wi, c in zip(w, conf) if c))
            med = np.median([r["err_km"] if r["err_km"] is not None else 1e4 for r in R])
            tiers = []
            for t in (0.1, 0.3, 1.0, 2.0, 5.0, 10.0):
                ok_t = np.array([r["err_km"] is not None and r["err_km"] <= t for r in R], float)
                v = float((w * ok_t).sum())
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(1000):
                    pick = rng.choice(len(sites), len(sites))
                    vals = [np.mean([o for o, r in zip(ok_t, R) if r["site"] == sites[k]]) for k in pick]
                    bs.append(np.mean(vals))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                tiers.append(f"{v:.0%} ({lo:.0%}-{hi:.0%})")
            lines.append(f"| {mode} | {N} | {len(R)} | {len(sites)} | {fix:.0%} | {cf:.0%} | {cw:.0%} | "
                         f"{(f'{med:.2f}' if med < 1e3 else 'no answer')} | " + " | ".join(tiers) + " |")
    return "\n".join(lines) + "\n", rows
