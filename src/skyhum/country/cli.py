"""Command-line parts of the country search and the benchmark reproduction: `skyhum locate --country`, `skyhum
country ...`, `skyhum adsb fetch --country` and `skyhum bench run --reproduce`. skyhum/cli.py only registers these."""
import json
import os
import sys

from skyhum import errors as E

BASE_RATE = ("Base rate: in the published benchmark the correct US region was ranked first for about 5 % of one-hour "
             "recordings (15/327, sound only, clock known to +-5 s; 7/110 for webcams); for Australian rural sensors "
             "37/139 hours.")
LIMITS = ("Limits: no answer is validated as correct. The decoy rule below is the research confidence rule; its "
          "confidence is not calibrated (round-5b test: about 2 in 3 confident answers within 2 km on a 7-cluster "
          "test). Outdoor public recordings only; ordinary indoor or "
          "phone callers were not measured. For research, validation and defence; not for locating private "
          "individuals.")


def add_locate(sub, common, Formatter, epilog):
    p = sub.add_parser("locate", parents=[common], formatter_class=Formatter, epilog=epilog,
                       help="whole-country search: which US metro / Australian SA4? (research; explicit command)",
                       description="Search a whole country (--country US: contiguous US, regions = Census CBSAs or "
                                   "counties; --country AU: Australia, regions = ABS SA4s) for the region whose low "
                                   "aircraft traffic explains the passes in the audio, with the start time known to a "
                                   "few seconds. Prints the ranked regions, the decoy comparison, the base rate and "
                                   "the limits. " + BASE_RATE + " Needs the static tables (skyhum country static), "
                                   "the ADS-B days (skyhum adsb fetch) and detector-v3 detections (--hints).")
    p.add_argument("audio", metavar="AUDIO", help="the recording (wav, flac, mp3 ...)")
    p.add_argument("--country", required=True, choices=["US", "AU"], help="search area")
    p.add_argument("--start", required=True, metavar="TIME", help="recording start, UTC ISO-8601 or epoch seconds")
    p.add_argument("--clock-tol", type=int, choices=[5, 30], default=5,
                   help="clock tolerance reported (s); both are searched (default: 5)")
    p.add_argument("--hints", metavar="FILE",
                   help="detector-v3 detections of this audio, JSON [{\"t\": s, \"f\": Hz, \"p\": prob}, ...] "
                        "(required until the detector ships in the package)")
    p.add_argument("--sessions", metavar="A,B", help="more recordings you state come from the same place")
    p.add_argument("--session-starts", metavar="T1,T2", help="start times of --sessions, same order")
    p.add_argument("--session-hints", metavar="H1,H2", help="detections of --sessions, same order")
    p.add_argument("--decoy-days", default="-7,7", metavar="D1,D2",
                   help="other-day decoys as day offsets (default: -7,7; 'none' for clock decoys only). Each needs "
                        "its ADS-B day")
    p.add_argument("--top", type=int, default=5, help="regions to print (default: 5)")
    p.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")
    p.add_argument("--report", metavar="FILE", help="write the full result as JSON")
    return p


def add_country(sub, common, Formatter, epilog):
    p = sub.add_parser("country", parents=[common], formatter_class=Formatter, epilog=epilog,
                       help="country-search data and scorers: static tables, staging, runner, score, prior, claims")
    cs = p.add_subparsers(dest="country_cmd", metavar="ACTION")

    def q(name, help_):
        x = cs.add_parser(name, help=help_, description=help_, parents=[common], formatter_class=Formatter)
        x.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")
        return x
    x = q("static", "download the public boundary / population / terrain files and build the region rasters")
    x.add_argument("--country", choices=["US", "AU", "both"], default="both")
    x.add_argument("--raw", metavar="DIR", help="use already downloaded public files from DIR")
    x = q("stage", "stage a benchmark manifest (blind inputs, groups, truth, decoy days) from your audio")
    x.add_argument("manifest", help="manifest JSON (e.g. the shipped a2o_country_v1, or a webcam feature release)")
    x.add_argument("--audio-dir", required=True, metavar="DIR", help="recordings named <item id>.<ext>")
    x.add_argument("--hints-dir", required=True, metavar="DIR", help="detector-v3 detections <item id>.json")
    x.add_argument("--spec-dir", metavar="DIR", help="spectrograms <item id>.npz instead of audio (feature releases)")
    x = q("run", "search every staged group (resumable; one process per group)")
    x.add_argument("tag", help="name of the run (outputs in ROOT/runs/TAG)")
    x.add_argument("--jobs", type=int, default=1, help="groups in parallel (default: 1)")
    x.add_argument("--groups", metavar="GID,...", help="only these groups")
    for name, help_ in (("score", "score search outputs: region top-1, tiers, decoy confidence, multi-session"),
                        ("prior", "fuse with population / UTC-offset priors; value of the sound given a prior"),
                        ("claims", "verify claimed places (true and false claims) from search outputs")):
        x = q(name, help_)
        x.add_argument("runs", help="directory of search outputs (*.json)")
        x.add_argument("--truth", required=True, help="truth.json the runs were staged with")
        x.add_argument("--static", metavar="DIR", help="static tables (default: ROOT/static)")
        x.add_argument("--out", required=True, metavar="PATH", help="output prefix (score) or directory")
        if name != "prior":
            x.add_argument("--channel", default="s2", help="score channel: s2 tonal (default), l_s2, c1_s2")
        if name == "claims":
            x.add_argument("--calls-runs", metavar="DIR", help="outputs of the same search on call audio")
            x.add_argument("--calls-truth", metavar="FILE")
    return p


def add_bench_run_args(q):
    """Arguments of `skyhum bench run` (the parser object is created by skyhum/cli.py)."""
    from skyhum.bench.reproduce import FAMILIES
    q.add_argument("--reproduce", choices=list(FAMILIES), metavar="FAMILY",
                   help="row family to rerun or rescore: " + ", ".join(FAMILIES) + " (REPRODUCING.md)")
    q.add_argument("--set", choices=["a2o", "webcam"], default="a2o",
                   help="benchmark set (default: a2o; webcam needs the webcam feature release)")
    q.add_argument("--jobs", type=int, default=1, help="searches in parallel (default: 1)")
    q.add_argument("--manifest", metavar="FILE", help="a manifest other than the shipped one (feature releases)")
    q.add_argument("--audio-dir", metavar="DIR", help="rerun: recordings named <item id>.<ext>")
    q.add_argument("--hints-dir", metavar="DIR", help="rerun: detector-v3 detections <item id>.json")
    q.add_argument("--spec-dir", metavar="DIR", help="rerun: spectrograms <item id>.npz instead of audio")
    q.add_argument("--outputs", "--runs", dest="runs", metavar="DIR",
                   help="rescore: existing locator outputs (yours or the published ones) instead of searching")
    q.add_argument("--truth", metavar="FILE", help="rescore (country families): truth.json of --outputs")
    q.add_argument("--cases", metavar="FILE",
                   help="rescore (box families): cases JSON of --outputs (default: the shipped A2O manifest)")
    q.add_argument("--static", metavar="DIR", help="static tables (default: WORKDIR/country/static)")
    q.add_argument("--channel", default="s2", help="country score channel (default: s2, tonal)")
    q.add_argument("--calls-runs", metavar="DIR", help="verify: outputs of the same search on call audio")
    q.add_argument("--calls-truth", metavar="FILE", help="verify: truth of --calls-runs")
    q.add_argument("--tag", help="box families: tag of the runs (default: r5b_test / r5b_se / r5b_ms)")
    q.add_argument("--plan", metavar="FILE", help="multi-session: the set plan of the outputs")
    q.add_argument("--ref-tag", help="box: a second tag scored on the same cases (e.g. the round-5a locator)")
    q.add_argument("--decoy", action="store_true", help="box: use the clock-decoy runs (<runs>/<tag>__decoy)")
    q.add_argument("--adsb-dir", metavar="DIR", help="box rerun: low-altitude ADS-B day files")
    q.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")


def add_adsb_fetch_args(q):
    q.add_argument("--country", choices=["US", "AU", "both"],
                   help="download whole adsb.lol days and cut them to the country's low traffic (resumable)")
    q.add_argument("--keep-raw", action="store_true", help="keep the downloaded day archive")
    q.add_argument("--workers", type=int, default=4, help="processes cutting the archive (default: 4)")
    q.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")


# ------------------------------------------------------------------------------------------------- commands
def _root(args, workdir):
    from skyhum.country.common import workdir as cwd
    return cwd(getattr(args, "country_root", None) or os.path.join(workdir, "country"))


def cmd_adsb_fetch(args, out, workdir):
    from skyhum.country import adsb_days
    if not args.country:
        raise E.NotImplementedInBuild("adsb fetch --area", "use --country US|AU (whole-country days)")
    root = _root(args, workdir)
    regs = {"US": ("CONUS",), "AU": ("AU",), "both": ("CONUS", "AU")}[args.country]
    res = {}
    for d in args.date:
        c = adsb_days.fetch_day(d, os.path.join(root, "adsb"), regs, workers=args.workers, keep_raw=args.keep_raw,
                                log=out.line)
        res[d] = "no release" if c is None else {k: v for k, v in c.items() if k.endswith("_low")}
        out.line(f"{d}: {res[d]}")
    out.line("ADS-B data: adsb.lol, Open Database License 1.0; derived files stay under ODbL.")
    return dict(days=res)


def cmd_country(args, out, workdir):
    from skyhum.country import claims, prior, runner, score, stage, static
    root = _root(args, workdir)
    c = args.country_cmd
    if c == "static":
        regs = {"US": ("CONUS",), "AU": ("AU",), "both": ("CONUS", "AU")}[args.country]
        static.build(os.path.join(root, "static"), raw=args.raw, regions=regs, log=out.line)
        return dict(static=os.path.join(root, "static"))
    if c == "stage":
        st, miss = stage.stage_manifest(root, args.manifest, args.audio_dir, args.hints_dir, args.spec_dir,
                                        log=out.line)
        return dict(staged=len(st), missing=miss)
    if c == "run":
        st = runner.run_all(root, args.tag, args.jobs, args.groups.split(",") if args.groups else None, log=out.line)
        return dict(groups=st)
    st_dir = args.static or os.path.join(root, "static")
    if c == "score":
        U = score.run(args.runs, args.truth, st_dir, args.out, channel=args.channel, print_md=not out.json)
        return dict(units=len(U), tables=args.out + ".md")
    if c == "prior":
        U = prior.run(args.runs, args.truth, st_dir, args.out, print_md=not out.json)
        return dict(units=len(U), tables=os.path.join(args.out, "RESULTS_prior.md"))
    if c == "claims":
        U, C = claims.run(args.runs, args.truth, st_dir, args.out, channel=args.channel, calls_runs=args.calls_runs,
                          calls_truth=args.calls_truth, print_md=not out.json)
        return dict(units=len(U), call_units=len(C), tables=os.path.join(args.out, "VERIFY_tonal.md"))
    raise E.UsageError("skyhum country needs an action: static, stage, run, score, prior or claims")


def _duration(path):
    import soundfile as sf
    try:
        return float(sf.info(path).duration)
    except RuntimeError:
        from skyhum._vendor.afg import audio as A
        x, fs, _ = A.load_audio(path, fs=4000)
        return len(x) / fs


def cmd_locate(args, out, workdir):
    from skyhum.country import adsb_days, search, stage, static
    from skyhum.country.common import Regions, region_of_country
    from skyhum.io.clock import parse_start
    region = region_of_country(args.country)
    root = _root(args, workdir)
    st_dir = os.path.join(root, "static")
    if not static.ready(st_dir, region):
        raise E.MissingData(f"static tables for {args.country} are missing: run `skyhum country static --country "
                            f"{args.country}` (downloads public boundary, population and terrain files)")
    audios = [args.audio] + (args.sessions.split(",") if args.sessions else [])
    starts = [args.start] + (args.session_starts.split(",") if args.session_starts else [])
    hints = [args.hints] + (args.session_hints.split(",") if args.session_hints else [])
    if not (len(audios) == len(starts) == len(hints)):
        raise E.UsageError("--sessions, --session-starts and --session-hints need one entry per extra recording")
    if any(h is None for h in hints):
        raise E.MissingData("--hints FILE is required: the detector-v3 detections of the audio ([{\"t\": s, "
                            "\"f\": Hz, \"p\": prob}]). Without detections the search proposes no candidate places")
    run_dir = os.path.join(root, "locate")
    oids, inputs = [], []
    for a, s, h in zip(audios, starts, hints):
        t0 = parse_start(s, None)
        dur = _duration(a)
        oid = stage.stage_session(run_dir, region, t0, dur, audio=a, hints=h)
        oids.append(oid)
        inputs.append(dict(oid=oid, region_prior=region, clock_utc=t0, dur_s=dur))
    offs = [] if args.decoy_days == "none" else [int(x) for x in args.decoy_days.split(",")]
    with open(os.path.join(run_dir, "decoy_days.json.tmp"), "w") as f:
        json.dump({o: offs for o in oids}, f)
    os.replace(os.path.join(run_dir, "decoy_days.json.tmp"), os.path.join(run_dir, "decoy_days.json"))
    adsb = os.path.join(root, "adsb")
    need = adsb_days.plan(inputs, {o: offs for o in oids})
    lack = [d for d in need if not adsb_days.day_done(adsb, d)]
    if lack:
        raise E.MissingData("ADS-B days missing (whole adsb.lol days, about 1-4 GB download each, cut to a few MB): "
                            f"skyhum adsb fetch --country {args.country} " + " ".join(f"--date {d}" for d in lack))
    n = len(oids)
    subsets = [[i] for i in range(n)] + ([list(range(n))] if n > 1 else [])
    gp = stage.write_group(run_dir, oids, subsets)
    outp = os.path.join(run_dir, "runs", os.path.basename(gp))
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    from skyhum.country.runner import paths
    try:
        res = search.run_group(gp, outp, run_dir, adsb, st_dir, paths(root)["model"], use_guard=False,
                               verbose=out.verbose, k_day=len(offs))
    except search.MissingDay as e:
        raise E.MissingData(str(e)) from e
    R = Regions(region, st_dir)
    summary = summarize(res, R, len(subsets) - 1, args.clock_tol, args.top)
    _print(summary, out, args)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(dict(summary=summary, search=res), f, default=float)
    return summary


def summarize(res, R, si, tau, top):
    real = next(v for v in res["variants"] if v["variant"] == "real")
    decs = [v for v in res["variants"] if v["variant"] != "real"]
    key = f"{si}|60|{tau}"
    rr = real["subsets"].get(key) or real["subsets"].get(f"{si}|30|{tau}") or {}
    T = (rr.get("s2") or {}).get("top", [])
    dt = [((v["subsets"].get(key) or {}).get("s2") or {}).get("top") or [[None, -1e9]] for v in decs]
    dtops = [x[0][1] for x in dt]
    best_decoy = max(dtops) if dtops else None
    beats = bool(T) and bool(dtops) and all(T[0][1] > d for d in dtops)
    return dict(country=R.region, window=key, n_decoys=len(dtops), best_decoy=best_decoy,
                beats_every_decoy=beats,
                regions=[dict(rank=k + 1, region=R.name(r[0]), score=r[1], lat=r[2], lon=r[3]) for k, r in
                         enumerate(T[:top])],
                base_rate=BASE_RATE, limits=LIMITS)


def _print(s, out, args):
    out.line(f"country search ({args.country}), window {s['window']}, clock +-{args.clock_tol} s")
    if not s["regions"]:
        out.result("result           NO ANSWER (no region with two explaining passes)")
    else:
        for r in s["regions"]:
            out.line(f"  {r['rank']:>2}. {r['region']:<48} score {r['score']:8.2f}  best point "
                     f"{r['lat']:.3f},{r['lon']:.3f}")
        bd = "none" if s["best_decoy"] is None else f"{s['best_decoy']:.2f}"
        out.result(f"result           top region {'beats' if s['beats_every_decoy'] else 'does NOT beat'} every decoy "
                   f"({s['n_decoys']} decoys, best decoy score {bd})")
    print(s["base_rate"], file=sys.stdout if not out.json else sys.stderr)
    print(s["limits"], file=sys.stdout if not out.json else sys.stderr)


def cmd_bench_run(args, out, workdir):
    from skyhum.bench import reproduce
    if not args.reproduce:
        raise E.UsageError("bench run needs --reproduce FAMILY (" + ", ".join(reproduce.FAMILIES) + ")")
    return reproduce.run(args, out, workdir)
