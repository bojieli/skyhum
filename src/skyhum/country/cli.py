"""Command-line parts of the country search and the benchmark reproduction (`skyhum country ...`, `skyhum bench run
--reproduce`). skyhum/cli.py only registers these."""
import os

from skyhum import errors as E

BASE_RATE = ("Base rate: in the published benchmark the correct US region was ranked first for about 5 % of one-hour "
             "recordings (15/327, sound only, clock known to +-5 s; 7/110 for webcams); for Australian rural sensors "
             "37/139 hours.")
LIMITS = ("Limits: no answer is validated as correct. The decoy rule below is the research confidence rule; the "
          "confidence of the round-5b locator is not validated. Outdoor public recordings only; ordinary indoor or "
          "phone callers were not measured. For research, validation and defence; not for locating private "
          "individuals.")


def add_country(sub, common, Formatter, epilog):
    p = sub.add_parser("country", parents=[common], formatter_class=Formatter, epilog=epilog,
                       help="country search: static tables and the scorers (score, prior, claims) of search outputs")
    cs = p.add_subparsers(dest="country_cmd", metavar="ACTION")

    def q(name, help_):
        x = cs.add_parser(name, help=help_, description=help_, parents=[common], formatter_class=Formatter)
        x.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")
        return x
    x = q("static", "download the public boundary / population / terrain files and build the region rasters")
    x.add_argument("--country", choices=["US", "AU", "both"], default="both")
    x.add_argument("--raw", metavar="DIR", help="use already downloaded public files from DIR")
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
    q.add_argument("--tag", help="box families: tag of the runs (default: r5b_test / r5b_se)")
    q.add_argument("--ref-tag", help="box: a second tag scored on the same cases (e.g. the round-5a locator)")
    q.add_argument("--decoy", action="store_true", help="box: use the clock-decoy runs (<runs>/<tag>__decoy)")
    q.add_argument("--adsb-dir", metavar="DIR", help="box rerun: low-altitude ADS-B day files")
    q.add_argument("--country-root", metavar="DIR", help="country work directory (default: WORKDIR/country)")


# ------------------------------------------------------------------------------------------------- commands
def _root(args, workdir):
    from skyhum.country.common import workdir as cwd
    return cwd(getattr(args, "country_root", None) or os.path.join(workdir, "country"))


def cmd_country(args, out, workdir):
    from skyhum.country import claims, prior, score, static
    root = _root(args, workdir)
    c = args.country_cmd
    if c == "static":
        regs = {"US": ("CONUS",), "AU": ("AU",), "both": ("CONUS", "AU")}[args.country]
        static.build(os.path.join(root, "static"), raw=args.raw, regions=regs, log=out.line)
        return dict(static=os.path.join(root, "static"))
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
    raise E.UsageError("skyhum country needs an action: static, score, prior or claims")


def cmd_bench_run(args, out, workdir):
    from skyhum.bench import reproduce
    if not args.reproduce:
        raise E.UsageError("bench run needs --reproduce FAMILY (" + ", ".join(reproduce.FAMILIES) + ")")
    return reproduce.run(args, out, workdir)
