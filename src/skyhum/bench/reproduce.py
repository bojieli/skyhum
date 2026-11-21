"""`skyhum bench run --reproduce FAMILY`: rerun or rescore a family of published rows. REPRODUCING.md maps every
row to a family, its inputs and the number to compare against.

Families
  country       whole-country search, one table per set (rows A2-A6w; B2-B6w with --set webcam)
  prior         region score x population / UTC-offset prior, the sound given a prior (A3t, A10, A11; B3p, B8-B9)
  verify        verifying a claimed place (section V; A12, B10)
  box           round-5b locator in a jittered 0.25 deg box, date / +-30 min / +-30 s clocks (A17, A17b, A17c)
  single-event  round-5b single-event two-point mode at +-5 s (A17d; A17f needs the decoy runs, see below)

Two ways to run each family:
  rescore  --outputs DIR (and --truth FILE / --cases FILE): score search outputs you already have (yours, or saved
           research outputs); CPU, minutes.
  rerun    no --outputs: stage the manifest's items from --audio-dir (you download the recordings) with detector hints
           from --hints-dir, search every group with the local runner (resumable, --jobs N), then score. Needs the
           ADS-B days (skyhum adsb fetch) and the static tables (skyhum country static). GPU-hours; CPU works but is
           slow (see REPRODUCING.md).
Sets: a2o (the shipped manifests: links and metadata only) or webcam (needs the webcam feature release: pass its
manifest with --manifest). Event and speech livestream items are not published (see REPRODUCING.md).
"""
import json
import os
from importlib import resources

from skyhum import errors as E

FAMILIES = {
    "country": "whole-country search: region top-1, distance tiers, decoy confidence, multi-session",
    "prior": "population / UTC-offset prior fusion and the value of the sound given a prior",
    "verify": "verify a claimed place (metro / 40 km box / 5 km), true and false claims",
    "box": "round-5b locator in a jittered box (date / +-30 min / +-30 s)",
    "single-event": "round-5b single-event two-point mode (+-5 s)",
}
COUNTRY_FAMILIES = ("country", "prior", "verify")
MANIFEST = {("country", "a2o"): "a2o_country_v1.json", ("box", "a2o"): "a2o_box_v1.json"}


def manifest_path(name):
    return str(resources.files("skyhum.bench").joinpath("manifests", name))


def load_manifest(family, set_, path=None):
    if path:
        with open(path) as f:
            return json.load(f)
    kind = "country" if family in COUNTRY_FAMILIES else "box"
    if set_ == "webcam":
        raise E.MissingData("webcam rows need the webcam feature release (webcam audio is not redistributable); "
                            "pass its manifest with --manifest FILE once it is published")
    if set_ in ("events", "speech"):
        raise E.UsageError("event and speech livestream items are not published (not reproducible by design); "
                           "see REPRODUCING.md")
    with open(manifest_path(MANIFEST[(kind, set_)])) as f:
        return json.load(f)


def run(args, out, workdir):
    fam = args.reproduce
    if fam not in FAMILIES:
        raise E.UsageError(f"--reproduce must be one of {', '.join(FAMILIES)}")
    res_dir = os.path.join(workdir, "reproduce", f"{fam}_{args.set}")
    os.makedirs(res_dir, exist_ok=True)
    if fam in COUNTRY_FAMILIES:
        return _country(fam, args, out, workdir, res_dir)
    return _box(fam, args, out, workdir, res_dir)


# ------------------------------------------------------------------------------------------------- country
def _country(fam, args, out, workdir, res_dir):
    from skyhum.country import claims, prior, score
    from skyhum.country.common import workdir as cwd
    root = args.country_root or cwd(os.path.join(workdir, "country"))
    static = os.path.join(root, "static")
    runs, truth = args.runs, args.truth
    if runs:
        if not truth:
            raise E.UsageError("--outputs needs --truth FILE (the truth.json the outputs were staged with)")
        static = args.static or static
    else:
        runs, truth = _country_rerun(args, out, root)
    _need_static(static)
    if fam == "country":
        prefix = os.path.join(res_dir, "score")
        units = score.run(runs, truth, static, prefix, channel=args.channel, print_md=False)
        out.line(f"wrote {prefix}.md ({len(units)} units)")
        return dict(family=fam, tables=prefix + ".md", units=len(units))
    if fam == "prior":
        units = prior.run(runs, truth, static, res_dir)
        out.line(f"wrote {os.path.join(res_dir, 'RESULTS_prior.md')} ({len(units)} units)")
        return dict(family=fam, tables=os.path.join(res_dir, "RESULTS_prior.md"), units=len(units))
    units, calls = claims.run(runs, truth, static, res_dir, channel=args.channel, calls_runs=args.calls_runs,
                              calls_truth=args.calls_truth)
    out.line(f"wrote {os.path.join(res_dir, 'VERIFY_tonal.md')} ({len(units)} units, {len(calls)} call units)")
    return dict(family=fam, tables=os.path.join(res_dir, "VERIFY_tonal.md"), units=len(units))


def _need_static(static):
    if not any(os.path.exists(os.path.join(static, f"regions_{R}.npz")) for R in ("AU", "CONUS")):
        raise E.MissingData(f"static tables missing in {static}: run `skyhum country static` first")


def _country_rerun(args, out, root):
    M = load_manifest("country", args.set, args.manifest)
    try:
        from skyhum.country import runner, stage
        from skyhum.country.adsb_days import day_done, plan
    except ImportError as e:
        raise E.NotImplementedInBuild("rerunning the country search", "pass --outputs DIR --truth FILE to rescore "
                                      "locator outputs") from e
    if not args.audio_dir or not args.hints_dir:
        raise E.UsageError("a rerun needs --audio-dir (the recordings, named <item id>.<ext>; download them from the "
                           "manifest urls) and --hints-dir (detector-v3 detections <item id>.json), or --outputs to "
                           "rescore saved outputs")
    staged, missing = stage.stage_manifest(root, M, args.audio_dir, args.hints_dir, args.spec_dir, log=out.line)
    if missing:
        out.warn(f"{len(missing)} items lack audio or hints; their groups are skipped")
    inputs = [dict(oid=it["session"], region_prior=M.get("region", "AU"), clock_utc=it["clock_utc"],
                   dur_s=it["duration_s"]) for it in M["items"] if it["session"] in set(staged)]
    need = plan(inputs, {it["session"]: it.get("decoy_day_offsets", []) for it in M["items"]})
    lack = [d for d in need if not day_done(os.path.join(root, "adsb"), d)]
    if lack:
        raise E.MissingData(f"{len(lack)} ADS-B days are not fetched yet (about 1-4 GB download each, cut to a few "
                            f"MB): skyhum adsb fetch --country {'AU' if M.get('region', 'AU') == 'AU' else 'US'} "
                            + " ".join(f"--date {d}" for d in lack[:5]) + (" ..." if len(lack) > 5 else ""))
    tag = f"reproduce_{args.set}"
    st = runner.run_all(root, tag, jobs=args.jobs, log=out.line)
    bad = {g: s for g, s in st.items() if s != "done"}
    if bad:
        out.warn(f"{len(bad)} groups did not finish: {bad}; rerun the same command to resume")
    return os.path.join(root, "runs", tag), os.path.join(root, "truth", "truth.json")


# ------------------------------------------------------------------------------------------------- box
def _box(fam, args, out, workdir, res_dir):
    from skyhum.bench import r5b_eval
    if args.cases:
        with open(args.cases) as f:
            cases = json.load(f)
    else:
        M = load_manifest(fam, args.set, args.manifest)
        cases = [dict(id=it["id"], lat=it["lat"], lon=it["lon"], cat=it["cat"], cluster=it["cluster"],
                      start_true=it["start_true_utc"], npos=it.get("npos"), source="a2o", split="test")
                 for it in M["items"]]
    runs = args.runs
    tag = args.tag or ("r5b_se" if fam == "single-event" else "r5b_test")
    if not runs:
        try:
            from skyhum.bench import box
        except ImportError as e:
            raise E.NotImplementedInBuild("rerunning the box benchmark", "pass --outputs DIR to rescore locator "
                                          "outputs") from e
        root = os.path.join(workdir, "box")
        modes = ("clk5s",) if fam == "single-event" else ("blind", "clk30m", "clk30s")
        runs = box.run_cases(cases, root, tag, modes, audio_dir=args.audio_dir, hints_dir=args.hints_dir,
                             adsb_dir=args.adsb_dir or os.path.join(workdir, "country", "adsb"), jobs=args.jobs,
                             log=out.line)
    if fam == "box":
        tags = [tag] + ([args.ref_tag] if args.ref_tag else [])
        md, summ = r5b_eval.r5b_tables(cases, runs, tags[::-1] if args.ref_tag else tags,
                                       modes=("blind", "clk30m", "clk30s", "wide"), decoy=args.decoy)
    else:
        md, summ, _ = r5b_eval.single_event(cases, runs, tag)
    p = os.path.join(res_dir, f"{fam}.md")
    with open(p, "w") as f:
        f.write(md)
    out.line(md)
    out.line(f"wrote {p}")
    return dict(family=fam, tables=p, summary=summ)
