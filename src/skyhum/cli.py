"""skyhum command line.

Exit codes: 0 success (verify: SUPPORTED), 10 UNCORROBORATED, 11 INCONCLUSIVE, 2 usage error or a feature not in
this build, 3 missing data, 1 selftest failure or internal error.
"""
import argparse
import os
import sys

from skyhum import __version__
from skyhum import errors as E
from skyhum.report import document, dumps, write

EPILOG = """exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use."""

PLACE_HELP = ('a place: a named region ("Los Angeles basin", see `skyhum regions list`), a circle '
              '"lat,lon,r=5km", a box "bbox:west,south,east,north", or "@file.geojson"')


class Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=30, width=100)


def _global_args(p, top):
    """Global options. On subcommands they default to SUPPRESS so a value given before the command survives."""
    d = (lambda v: v) if top else (lambda v: argparse.SUPPRESS)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=d("auto"),
                   help="device for the r5b locator: auto, cpu or cuda (default: auto; cpu never touches a GPU)")
    p.add_argument("--workdir", default=d(None), metavar="DIR",
                   help="working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)")
    p.add_argument("--json", action="store_true", default=d(False), help="print one JSON document to stdout")
    p.add_argument("-q", "--quiet", action="store_true", default=d(False), help="print only the result line")
    p.add_argument("-v", "--verbose", action="store_true", default=d(False), help="print progress to stderr")


def _audio_args(p, start_required=True):
    p.add_argument("audio", metavar="AUDIO", help="wav, flac, ogg; mp3, m4a, webm need ffmpeg")
    p.add_argument("--channel", type=int, default=0, help="channel to use, 0-based (default: 0)")
    g = p.add_argument_group("clock")
    g.add_argument("--start", metavar="TIME",
                   help="when the recording began: UTC ISO-8601 (2026-09-08T14:03:00Z) or epoch seconds"
                        + (" (required unless --start-from-file)" if start_required else ""))
    g.add_argument("--tz", metavar="ZONE", help="time zone for a --start without Z or offset, e.g. Europe/London")
    g.add_argument("--start-from-file", action="store_true",
                   help="take the start from container metadata or file mtime (prints a warning)")
    g.add_argument("--start-tol", default="30s", metavar="DUR",
                   help="clock uncertainty: 5s, 30s (default), 30m; 2h and 'date' are not in this build's CPU "
                        "matcher. Wider means more chance matches")
    g.add_argument("--duration", metavar="DUR", help="use only the first DUR of the audio, e.g. 10m")
    d = p.add_argument_group("data and output")
    d.add_argument("--adsb", metavar="PATH|auto", default=None,
                   help="ADS-B tracks.npz day file (afg layout); 'auto' download is not in this build")
    d.add_argument("--out", metavar="DIR", help="directory for extra outputs (spectrogram PNG)")
    d.add_argument("--report", metavar="FILE", help="write a report: FILE.json or a self-contained FILE.html")
    a = p.add_argument_group("advanced")
    a.add_argument("--ground-alt", type=float, default=0.0, metavar="M",
                   help="ground altitude in metres (no terrain model in this build; default 0)")
    a.add_argument("--spacing", type=float, default=None, metavar="KM", help="grid spacing (default: automatic)")
    a.add_argument("--seed", type=int, default=0, help="seed for decoy clocks (default: 0)")


def _matcher_args(p, decoy_default_text):
    m = p.add_argument_group("matcher")
    m.add_argument("--matcher", choices=["auto", "r5b", "reference"], default="auto",
                   help="r5b: the round-5b locator (test partial; needs PyTorch and its fitted tables, slow on CPU); "
                        "reference: the simplified numpy matcher (fast, not benchmarked); auto (default): r5b when "
                        "available, otherwise reference with a notice")
    m.add_argument("--decoys", type=int, default=None, metavar="N",
                   help=f"decoy clocks to try (default: {decoy_default_text})")
    m.add_argument("--fast", action="store_true",
                   help="r5b: fewer calibration passes and refinements; for checking the install, not for results")
    m.add_argument("--dem-dir", metavar="DIR",
                   help="r5b: directory of GeoTIFF terrain tiles (otherwise flat ground at --ground-alt); nothing is "
                        "downloaded")
    m.add_argument("--hints", default="auto", metavar="auto|none|FILE",
                   help="r5b: detector hints, as in the round-5b test. auto (default): run detector v3 on the audio; "
                        "FILE: a JSON list of {t, f, p} (seconds from the start of the audio, Hz, probability), e.g. "
                        "from `skyhum detect --hints-out`; none: no hints")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="skyhum", formatter_class=Formatter, epilog=EPILOG,
        description="Test whether audio carries an aircraft-Doppler location clue. Verify a claimed place, audit "
                    "your own recordings, scrub the clue. Runs locally.")
    ap.add_argument("--version", action="version", version=f"skyhum {__version__}")
    _global_args(ap, top=True)
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    common = argparse.ArgumentParser(add_help=False)
    _global_args(common, top=False)

    def cmd(name, help_, desc=None):
        return sub.add_parser(name, help=help_, description=desc or help_, parents=[common],
                              formatter_class=Formatter, epilog=EPILOG)

    p = cmd("selftest", "synthetic plane + synthetic track: checks the install and prints PASS or FAIL",
            "Synthetic propeller passes and synthetic ADS-B tracks. Checks that the detector finds the passes, that "
            "verify (reference matcher, then the r5b locator) supports the true place and rejects a displaced place "
            "and a wrong clock, and that the tone scrubber works. CPU only: seconds with --quick, minutes without.")
    p.add_argument("--seed", type=int, default=7, help="seed of the synthetic world (default: 7)")
    p.add_argument("--quick", action="store_true", help="skip the r5b locator checks (they take minutes on a CPU)")
    p.add_argument("--write-fixtures", metavar="DIR",
                   help="also write the synthetic demo files (clip.wav, call.wav, quiet.wav, tracks.npz, "
                        "area.geojson) to DIR and exit")

    cmd("doctor", "check Python packages, ffmpeg, GPU, ADS-B cache and detector weights")

    p = cmd("regions", "named regions accepted by --claim/--where/--area")
    p.add_argument("action", choices=["list"], help="list the bundled named regions")

    p = cmd("detect", "list aircraft fly-overs in the audio (no location involved)",
            "List aircraft fly-overs found in the audio: closest-approach time, frequency and probability from "
            "detector v3 (the learned detector of the research runs; needs PyTorch, CPU is enough), or tonal passes "
            "from the classic ridge detector. Optional spectrogram PNG.")
    _audio_args(p, start_required=False)
    p.add_argument("--detector", choices=["auto", "v3", "classic"], default="auto",
                   help="v3: the learned detector; classic: the ridge detector; auto (default): v3 when PyTorch is "
                        "installed, otherwise classic")
    p.add_argument("--min-p", type=float, default=0.5, metavar="P",
                   help="v3: list detections with p >= P (default: 0.5; the locator uses p >= 0.8)")
    p.add_argument("--hints-out", metavar="FILE",
                   help="v3: write every detection as a locator hints file (JSON), for --hints FILE")
    p.add_argument("--png", metavar="FILE", help="write the spectrogram with detections to FILE (needs matplotlib)")

    p = cmd("verify", "is the audio consistent with a claimed place and time?",
            "Score the claimed place at the stated clock against decoys (the same place at wrong clocks and other "
            "days, and optionally a rival place). SUPPORTED only if the claim beats every decoy. UNCORROBORATED is "
            "not evidence that the claim is false.")
    _audio_args(p)
    p.add_argument("--claim", required=True, metavar="PLACE", help=PLACE_HELP)
    p.add_argument("--versus", metavar="PLACE", help="a rival place, scored as one more decoy")
    p.add_argument("--operating-point", choices=["default", "strict"], default="default",
                   help="default: beats every decoy; strict: beats them by a fixed margin (not calibrated)")
    _matcher_args(p, "20 with r5b, 200 with reference")

    p = cmd("audit", "you say where you were: how much did the audio give away?",
            "You supply the truth. Reports detected passes, ADS-B passes in range, whether verify accepts the true "
            "place, whether a place 15 km away also passes, and fixed-rule advice. With --scrub, repeats the audit "
            "on scrubbed audio.")
    _audio_args(p)
    p.add_argument("--where", required=True, metavar="PLACE", help="where you were. " + PLACE_HELP)
    p.add_argument("--scrub", metavar="OUT", help="also scrub the audio to OUT and audit the result")
    p.add_argument("--method", choices=["gate", "tones"], default="gate", help="scrub method for --scrub")
    _matcher_args(p, "20 with r5b, 200 with reference")

    p = cmd("scrub", "remove the clue: low-band gate in pauses, or tone scrubbing",
            "gate: replace the band below --cutoff with comfort noise while nobody speaks (WebRTC VAD). tones: track "
            "long tonal lines (aircraft, machinery) and replace them with noise at the local floor, under speech "
            "too. Tested in offline simulation only; real platform pipelines may behave differently.")
    p.add_argument("input", metavar="IN", help="input audio")
    p.add_argument("output", metavar="OUT", help="output audio (.wav or .flac; others need ffmpeg)")
    p.add_argument("--method", choices=["gate", "tones"], default="gate", help="default: gate")
    p.add_argument("--cutoff", type=float, default=400.0, metavar="HZ", help="gate cutoff in Hz (default: 400)")
    p.add_argument("--channel", type=int, default=0, help="input channel, 0-based")
    p.add_argument("--seed", type=int, default=0, help="comfort-noise seed")

    p = cmd("find", "rank places inside a supplied area (research and validation)",
            "Rank candidate places inside an area you supply. A point estimate is printed only above the strict "
            "threshold. The confidence of the r5b locator is NOT validated (its test did not run the clock decoys), "
            "so the strict threshold is an uncalibrated margin. For research, validation and defence; not for "
            "locating private individuals.")
    _audio_args(p)
    p.add_argument("--area", required=True, metavar="PLACE", help="search area. " + PLACE_HELP)
    p.add_argument("--top", type=int, default=5, help="ranked tiles to show (default: 5)")
    p.add_argument("--sessions", metavar="A,B", help="more recordings you state come from the same spot")
    p.add_argument("--session-starts", metavar="T1,T2", help="start times of --sessions, same order")
    p.add_argument("--single-event", action="store_true",
                   help="r5b: allow answers from one pass (clock within 5 s); prints the mirror point too")
    _matcher_args(p, "10 with r5b, 200 with reference")

    p = cmd("bench", "public benchmark: sources (links and metadata), fetch, scoring")
    bs = p.add_subparsers(dest="bench_cmd", metavar="ACTION")
    bs.add_parser("list", help="list sources with licences", parents=[common], formatter_class=Formatter)
    q = bs.add_parser("fetch", help="download recordings from the original host (not in this build)",
                      parents=[common], formatter_class=Formatter)
    q.add_argument("--source", required=True)
    q.add_argument("--n", type=int, default=20)
    q = bs.add_parser("run", help="score saved results and print the tier table (not in this build)",
                      parents=[common], formatter_class=Formatter)
    q.add_argument("--tier", default="all")
    q.add_argument("--reproduce", choices=["paper"])

    p = cmd("adsb", "download and index ADS-B history for an area and dates")
    as_ = p.add_subparsers(dest="adsb_cmd", metavar="ACTION")
    q = as_.add_parser("fetch", help="download ADS-B days (not in this build)", parents=[common],
                       formatter_class=Formatter)
    q.add_argument("--area", required=True, metavar="PLACE")
    q.add_argument("--date", required=True, action="append", metavar="YYYY-MM-DD")
    return ap


# ---------------------------------------------------------------------------------------------------------- helpers
class Out:
    def __init__(self, args):
        self.json = getattr(args, "json", False)
        self.quiet = getattr(args, "quiet", False)
        self.verbose = getattr(args, "verbose", False)

    def line(self, text=""):
        if not self.json and not self.quiet:
            print(text)

    def result(self, text):
        if not self.json:
            print(text)

    def warn(self, text):
        if not self.quiet or self.verbose:
            print(f"skyhum: {text}", file=sys.stderr)

    def progress(self, label):
        if not self.verbose:
            return None

        def f(i, n):
            if i == n or i % max(1, n // 10) == 0:
                print(f"  {label}: {i}/{n} cells", file=sys.stderr)
        return f


def workdir(args):
    return (getattr(args, "workdir", None) or os.environ.get("SKYHUM_WORKDIR")
            or os.path.expanduser("~/.local/share/skyhum"))


def check_device(args, out):
    dev = getattr(args, "device", "auto")
    os.environ["SKYHUM_DEVICE"] = dev
    if getattr(args, "matcher", "auto") == "reference" and dev == "cuda":
        raise E.NotImplementedInBuild("--device cuda with --matcher reference", "the reference matcher is CPU only")


def need_r5b(args):
    import importlib.util

    from skyhum.match import locator as LOC
    if getattr(args, "matcher", "reference") == "auto":
        missing = [w for w, ok in (("PyTorch", importlib.util.find_spec("torch") is not None),
                                   ("the fitted r5b tables", LOC.models_available())) if not ok]
        args.matcher = "reference" if missing else "r5b"
        if missing:
            print(f"skyhum: {' and '.join(missing)} not available: using --matcher reference (not benchmarked). "
                  "See docs/locator.md.", file=sys.stderr)
    if args.matcher != "r5b":
        return False
    if not LOC.models_available():
        raise E.MissingData("the fitted round-5b tables are not installed (they are not in the public repository "
                            "yet); set SKYHUM_R5B_MODELS to a directory holding them, or use --matcher reference")
    if importlib.util.find_spec("torch") is None:
        raise E.MissingData("the r5b locator needs PyTorch (the CPU build is enough): pip install torch, "
                            "or use --matcher reference")
    if args.dem_dir:
        import glob
        if not glob.glob(os.path.join(args.dem_dir, "*.tif")):
            raise E.UsageError(f"--dem-dir {args.dem_dir} holds no GeoTIFF (*.tif); skyhum does not download terrain")
    return True


def n_decoys(args, r5b_default):
    need_r5b(args)
    n = args.decoys if args.decoys is not None else (r5b_default if args.matcher == "r5b" else 200)
    if n < 1:
        raise E.UsageError("--decoys must be at least 1")
    return n


def load_clock_audio(args, out, need_start=True):
    from skyhum.io import audio as A
    from skyhum.io.clock import parse_duration, parse_start, parse_tolerance
    dur = parse_duration(args.duration) if args.duration else None
    x, info = A.load(args.audio, channel=args.channel, duration_s=dur)
    start = parse_start(args.start, args.tz) if args.start else None
    if start is None and args.start_from_file:
        start, how = A.file_start_time(args.audio, info["duration_s"])
        out.warn(f"start taken from {how}: {start:.0f}. File times are often wrong; pass --start if you know it.")
    if start is None and need_start:
        raise E.UsageError("--start is required (UTC, e.g. 2026-09-08T14:03:00Z), or pass --start-from-file")
    win = parse_tolerance(args.start_tol, start) if start is not None else None
    return x, info, start, win


def finish(args, out, command, result, code):
    if getattr(args, "report", None):
        write(args.report, document(command, result, code))
        out.line(f"report           {args.report}")
    if out.json:
        print(dumps(document(command, result, code)))
    return code


def matcher_label(r):
    if r.get("matcher") == "r5b":
        h = r.get("hints")
        h = h[0] if isinstance(h, list) and h else h
        hs = f", {h['n_used']} detector v3 hint(s) at p >= {h['hint_pmin']:g}" if h else ", no detector hints"
        return ("r5b round-5b locator (test partial; confidence not validated" + hs + ")"
                + (" [--fast]" if r.get("fast") else ""))
    return f"{r.get('matcher')} (simplified, not benchmarked)"


def fmt_kv(rows):
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{max(w, 16)}} {v}" for k, v in rows)


# ---------------------------------------------------------------------------------------------------------- commands
def cmd_selftest(args, out):
    from skyhum import selftest
    if args.write_fixtures:
        r = selftest.write_fixtures(args.write_fixtures, seed=args.seed)
        out.line(f"wrote synthetic demo files to {r['dir']}")
        out.line(f"site {r['site'][0]:.4f},{r['site'][1]:.4f}  start {r['start']}  (synthetic, not real data)")
        return finish(args, out, "selftest", r, E.OK)
    os.environ["SKYHUM_DEVICE"] = getattr(args, "device", "auto")
    r = selftest.run(seed=args.seed, r5b=not args.quick)
    for c in r["checks"]:
        tag = "skip" if c.get("skipped") else ("ok" if c["ok"] else "FAIL")
        out.line(f"  [{tag}] {c['check']}: {c['detail']}")
    out.result(f"selftest {r['result']} ({r['seconds']} s, synthetic data only)")
    return finish(args, out, "selftest", r, E.OK if r["result"] == "PASS" else E.FAILED)


def cmd_doctor(args, out):
    from skyhum import doctor
    r = doctor.run(device=getattr(args, "device", "auto"), workdir=workdir(args))
    for c in r["checks"]:
        out.line(f"  {c['status']:<5} {c['check']:<17} {c['detail']}")
    out.result(f"doctor {r['status']}")
    return finish(args, out, "doctor", r, E.OK if r["status"] == "ok" else E.FAILED)


def cmd_regions(args, out):
    from skyhum.io.regions import load_regions, parse_place
    rows = []
    for reg in load_regions():
        a = parse_place(reg["name"])
        rows.append(dict(name=reg["name"], country=reg["country"], bbox=reg["bbox"],
                         extent_km=round(a.extent_km(), 1), aliases=reg.get("aliases", [])))
        out.line(f"  {reg['name']:<24} {reg['country']}  {a.extent_km():5.0f} km  bbox:"
                 + ",".join(f"{v:g}" for v in reg["bbox"]))
    out.line("Approximate boxes, not official boundaries. Use @file.geojson for an exact area.")
    return finish(args, out, "regions", dict(regions=rows), E.OK)


def pick_detector(args, out):
    from skyhum.detect import v3
    if args.detector == "auto":
        if v3.available():
            return "v3"
        out.warn("PyTorch or the detector v3 weights are not available: using the classic ridge detector "
                 "(fallback). pip install torch for detector v3.")
        return "classic"
    if args.detector == "v3" and not v3.available():
        raise E.MissingData("detector v3 needs PyTorch (the CPU build is enough) and its weights; "
                            "or use --detector classic")
    return args.detector


def _dur(args):
    from skyhum.io.clock import parse_duration
    return parse_duration(args.duration) if getattr(args, "duration", None) else None


def run_v3(path=None, x=None, fs=None, channel=0, duration_s=None):
    """Detector v3 on a file (decoded at its own rate, straight to 1 kHz, as in the research runs) or on audio in
    memory. CPU only."""
    from skyhum.detect import v3
    d = v3.shared("cpu")
    if path is not None:
        return d.detect_1k(v3.load_1k(path, channel=channel, duration_s=duration_s))
    return d.detect(x, fs)


def load_hints(path):
    import json
    try:
        H = json.load(open(path))
    except (OSError, ValueError) as e:
        raise E.UsageError(f"--hints {path}: {e}") from None
    if not isinstance(H, list) or not all(isinstance(h, dict) and "t" in h and "f" in h for h in H):
        raise E.UsageError(f"--hints {path}: expected a JSON list of {{t, f, p}} objects")
    return H


def hints_for(args, out, path=None, x=None, fs=None):
    """Detector hints for one recording, or None (--hints none, or detector v3 unavailable)."""
    from skyhum.detect import v3
    h = getattr(args, "hints", "none")
    if h == "none":
        return None
    if h != "auto":
        return load_hints(h)
    if not v3.available():
        out.warn("detector v3 is not available (PyTorch or weights missing): r5b runs without hints")
        args.hints = "none"
        return None
    return run_v3(path=path, x=x, fs=fs, channel=args.channel, duration_s=_dur(args) if path else None)


def cmd_detect(args, out):
    check_device(args, out)
    if pick_detector(args, out) == "v3":
        return _detect_v3(args, out)
    return _detect_classic(args, out)


def _detect_v3(args, out):
    import json

    from skyhum.detect import v3
    from skyhum.detect.spectrogram import logspec
    from skyhum.io.clock import hms
    x, info, start, _ = load_clock_audio(args, out, need_start=False)
    dets = run_v3(path=args.audio, channel=args.channel, duration_s=_dur(args))
    if start is not None:
        for d in dets:
            d["t_utc"] = start + d["t"]
    if args.hints_out:
        with open(args.hints_out, "w") as f:
            json.dump([{k: v for k, v in d.items() if k != "t_utc"} for d in dets], f, indent=0)
    png = args.png or (os.path.join(args.out, "spectrogram.png") if args.out else None)
    if png:
        try:
            if args.out:
                os.makedirs(args.out, exist_ok=True)
            v3.plot(logspec(x, 16000), dets, png, title=os.path.basename(args.audio), pmin=args.min_p)
        except ImportError:
            out.warn("matplotlib is not installed: no PNG written (pip install 'skyhum[plot]')")
            png = None
    out.line(f"detect  {os.path.basename(args.audio)}  ({info['duration_s']:.0f} s, detector v3)")
    shown = sorted([d for d in dets if d["p"] >= args.min_p], key=lambda d: d["t"])
    for d in shown:
        when = f"{d['t']:7.0f} s" + (f"  {hms(d['t_utc'])}" if "t_utc" in d else "")
        out.line(f"  {when}  {d['f']:6.1f} Hz  p {d['p']:.2f}  v {d['v']:4.0f} m/s  d {d['d']:5.0f} m")
    n_hint = sum(d["p"] >= v3.HINT_PMIN for d in dets)
    out.result(f"{n_hint} fly-over(s) at p >= {v3.HINT_PMIN:g}, {len(shown)} at p >= {args.min_p:g} (detector v3)"
               + (f"; hints {args.hints_out}" if args.hints_out else "") + (f"; spectrogram {png}" if png else ""))
    r = dict(audio=info, detector="v3", hint_pmin=v3.HINT_PMIN, min_p=args.min_p, n_hint=n_hint, detections=dets,
             hints_out=args.hints_out, png=png, start=start)
    return finish(args, out, "detect", r, E.OK)


def _detect_classic(args, out):
    from skyhum.detect import tonal
    from skyhum.detect.spectrogram import logspec
    from skyhum.io.clock import hms
    if args.hints_out:
        raise E.UsageError("--hints-out needs detector v3")
    x, info, start, _ = load_clock_audio(args, out, need_start=False)
    spec = logspec(x, 16000)
    passes = tonal.detect(spec, t0=start)
    png = args.png or (os.path.join(args.out, "spectrogram.png") if args.out else None)
    if png:
        try:
            if args.out:
                os.makedirs(args.out, exist_ok=True)
            tonal.plot(spec, passes, png, title=os.path.basename(args.audio))
        except ImportError:
            out.warn("matplotlib is not installed: no PNG written (pip install 'skyhum[plot]')")
            png = None
    out.line(f"detect  {os.path.basename(args.audio)}  ({info['duration_s']:.0f} s, detector {tonal.NAME}"
             + (", fallback without PyTorch" if args.detector == "auto" else "") + ")")
    for p in passes:
        when = f"{p['cpa_s']:7.0f} s" + (f"  {hms(p['cpa_utc'])}" if "cpa_utc" in p else "")
        out.line(f"  {when}  {p['kind']:<12} {p['f_start_hz']:6.1f} -> {p['f_end_hz']:6.1f} Hz  "
                 f"slide {p['slide_pct']:4.1f}%  level {p['level']:.1f}  ridges {p['n_ridges']}")
    n = sum(p["kind"] == "doppler-like" for p in passes)
    out.result(f"{n} Doppler-like pass(es), {len(passes) - n} steady tone(s)" + (f"; spectrogram {png}" if png else ""))
    r = dict(audio=info, detector=tonal.NAME, passes=passes, png=png, start=start)
    return finish(args, out, "detect", r, E.OK)


def _verify_human(out, args, r, claim_label):
    from skyhum.io.clock import hms
    w = r.get("tested_clock_window")
    rows = [("verify", os.path.basename(args.audio)), ("claim", claim_label)]
    if w:
        rows.append(("clock window", f"{hms(w['lo'])}-{hms(w['hi'])}"))
    rows.append(("matcher", matcher_label(r)))
    if r.get("coarse_grid"):
        rows.append(("grid", f"{r['cell_spacing_km']} km cells: coarse, true matches can be missed; use a smaller "
                             "place or --spacing 0.5"))
    if "passes" in r:
        rows.append(("passes used", str(sum(1 for p in r["passes"] if p["vote"] > 0))))
    if "decoy_rank" in r:
        rows.append(("decoy rank", f"{r['decoy_rank']}  (margin {r['margin']})"))
    if r.get("versus"):
        rows.append(("versus", f"{r['versus']['area']['label']}: score {r['versus']['score']}"))
    rows.append(("verdict", f"{r['verdict']}  ({r['operating_point']} operating point)"))
    rows.append(("why", r.get("reason", "")))
    rows.append(("not tested", "indoor attenuation, speech overlap, platform processing"))
    out.line(fmt_kv(rows))
    if r["verdict"] == "UNCORROBORATED":
        out.line("UNCORROBORATED is not evidence that the claim is false.")


def _verifier(args, out, x, tracks, lo, hi, versus=None, audio_path=None):
    """Return verify_fn(area) for the chosen matcher. audio_path: the file x came from (detector hints are computed
    from the file, as in the research runs); None computes them from x."""
    if need_r5b(args):
        from skyhum.score import r5b_rule as R
        nd = n_decoys(args, R.DEFAULT_DECOYS)
        if not args.fast:
            out.warn(f"r5b on {os.environ.get('SKYHUM_DEVICE', 'auto')}: one locator run per decoy ({nd}); on a CPU "
                     "this can take a long time")
        hints = hints_for(args, out, path=audio_path, x=x, fs=16000)
        return lambda area: R.verify(x, 16000, tracks, area, lo, hi, versus=versus,
                                     operating_point=getattr(args, "operating_point", "default"), n_decoys=nd,
                                     seed=args.seed, fast=args.fast, ground_m=args.ground_alt, dem_dir=args.dem_dir,
                                     progress=out.progress("decoy runs"), hints=hints)
    if getattr(args, "hints", "auto") not in ("auto", "none"):
        raise E.UsageError("--hints FILE is for the r5b locator; the reference matcher takes no hints")
    from skyhum.detect.spectrogram import logspec
    from skyhum.score import verify as V
    nd = n_decoys(args, 200)
    spec = logspec(x, 16000)
    return lambda area: V.run(spec, tracks, area, lo, hi, versus=versus,
                              operating_point=getattr(args, "operating_point", "default"), n_decoys=nd,
                              seed=args.seed, spacing_km=args.spacing, ground_m=args.ground_alt,
                              progress=out.progress("verify"))


def cmd_verify(args, out):
    from skyhum.io.adsb import Tracks
    from skyhum.io.regions import parse_place
    check_device(args, out)
    claim = parse_place(args.claim)
    versus = parse_place(args.versus) if args.versus else None
    tracks = Tracks(args.adsb)
    x, info, start, (lo, hi) = load_clock_audio(args, out)
    r = _verifier(args, out, x, tracks, lo, hi, versus, audio_path=args.audio)(claim)
    r["audio"] = info
    _verify_human(out, args, r, claim.label)
    if out.quiet:
        out.result(r["verdict"])
    return finish(args, out, "verify", r, E.VERDICT_EXIT[r["verdict"]])


def cmd_audit(args, out):
    from skyhum import audit
    from skyhum.detect import tonal
    from skyhum.detect.spectrogram import logspec
    from skyhum.io.adsb import Tracks
    from skyhum.io.regions import parse_place
    check_device(args, out)
    where = parse_place(args.where)
    tracks = Tracks(args.adsb)
    x, info, start, (lo, hi) = load_clock_audio(args, out)
    r = audit.run(_verifier(args, out, x, tracks, lo, hi, audio_path=args.audio),
                  tonal.detect(logspec(x, 16000), t0=start), where)
    r["audio"] = info
    if args.scrub:
        y, sinfo = _scrub(x, args.method, 400.0, args.seed)
        from skyhum.io.audio import save
        save(args.scrub, y)
        if args.hints not in ("auto", "none"):
            raise E.UsageError("--hints FILE describes the original audio; use --hints auto with --scrub")
        after = audit.run(_verifier(args, out, y, tracks, lo, hi), tonal.detect(logspec(y, 16000), t0=start), where)
        r["scrubbed"] = dict(method=args.method, output=args.scrub, scrub_info=sinfo,
                             detected=after["detected"], true_claim=after["true_claim"],
                             wrong_claim=after["wrong_claim"],
                             speech_quality="skipped: no speech-quality metric in this build")
    _audit_human(out, args, r)
    code = E.VERDICT_EXIT[r["true_claim"]["verdict"]]
    return finish(args, out, "audit", r, code)


def _audit_human(out, args, r):
    rows = [("audit", os.path.basename(args.audio)), ("where", r["where"]["label"]),
            ("matcher", matcher_label(r)),
            ("detected", f"{r['detected']['doppler_like']} Doppler-like pass(es) ({r['detected']['detector']})"),
            ("ADS-B in range", f"{r['in_range']['passes']} pass(es) within 9 km while recording"),
            ("true place", f"{r['true_claim']['verdict']}  default={r['true_claim']['operating_points'].get('default')}"
                           f" strict={r['true_claim']['operating_points'].get('strict')}"),
            (f"{r['wrong_claim']['distance_km']:g} km away", r["wrong_claim"]["verdict"]),
            ("exposure tier", "not computed in this build")]
    if "scrubbed" in r:
        s = r["scrubbed"]
        rows += [("after scrub", f"{s['method']} -> {s['output']}"),
                 ("  detected", f"{s['detected']['doppler_like']} Doppler-like pass(es)"),
                 ("  true place", s["true_claim"]["verdict"]),
                 ("  speech quality", s["speech_quality"])]
    out.line(fmt_kv(rows))
    out.line("advice:")
    for a in r["advice"]:
        out.line(f"  - {a}")


def _scrub(x, method, cutoff, seed):
    if method == "gate":
        from skyhum.scrub import lowband_gate
        y, info = lowband_gate.gate(x, fc=cutoff, seed=seed)
    else:
        from skyhum.scrub import tones
        y, info = tones.scrub(x, seed=seed)
    return y, {k: v for k, v in info.items() if k != "gain"}


def cmd_scrub(args, out):
    from skyhum.io import audio as A
    x, info = A.load(args.input, channel=args.channel)
    y, sinfo = _scrub(x, args.method, args.cutoff, args.seed)
    A.save(args.output, y)
    out.line(f"scrub  {args.input} -> {args.output}  method {args.method}"
             + (f" cutoff {args.cutoff:g} Hz" if args.method == "gate" else ""))
    for k, v in sinfo.items():
        out.line(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
    out.line("Tested in offline simulation only. Real call and upload pipelines may behave differently;"
             " run `skyhum audit` on the output.")
    out.result(f"wrote {args.output}")
    r = dict(input=info, output=args.output, method=args.method, cutoff_hz=args.cutoff, info=sinfo)
    return finish(args, out, "scrub", r, E.OK)


def cmd_find(args, out):
    from skyhum.io.regions import parse_place
    area = parse_place(args.area)
    check_device(args, out)
    from skyhum.io import audio as A
    from skyhum.io.adsb import Tracks
    from skyhum.io.clock import parse_duration, parse_start, parse_tolerance
    tracks = Tracks(args.adsb)
    x, info, start, (lo, hi) = load_clock_audio(args, out)
    sessions = [(x, lo, hi)]
    paths = [args.audio]
    if args.sessions:
        extra = [s for s in args.sessions.split(",") if s]
        starts = [s for s in (args.session_starts or "").split(",") if s]
        if len(starts) != len(extra):
            raise E.UsageError("--session-starts needs one start time per file in --sessions")
        dur = parse_duration(args.duration) if args.duration else None
        for pth, st in zip(extra, starts):
            xs, _ = A.load(pth, channel=args.channel, duration_s=dur)
            sessions.append((xs, *parse_tolerance(args.start_tol, parse_start(st, args.tz))))
            paths.append(pth)
    need_r5b(args)
    if args.single_event:
        if args.matcher != "r5b":
            raise E.UsageError("--single-event needs --matcher r5b")
        if max(h - l for _, l, h in sessions) > 10.0:
            raise E.UsageError("--single-event needs a clock known within 5 s (--start-tol 5s)")
    if need_r5b(args):
        from skyhum.score import r5b_rule as R
        out.warn("find: " + R.NOT_VALIDATED)
        if args.hints not in ("auto", "none") and len(sessions) > 1:
            raise E.UsageError("--hints FILE covers one recording; use --hints auto with --sessions")
        hints = [hints_for(args, out, path=p) for p in paths]
        r = R.find([(xs, 16000, l, h) for xs, l, h in sessions], tracks, area, top=args.top,
                   n_decoys=n_decoys(args, 10), seed=args.seed, fast=args.fast, single_event=args.single_event,
                   ground_m=args.ground_alt, dem_dir=args.dem_dir, progress=out.progress("decoy runs"),
                   hints=None if all(h is None for h in hints) else hints)
    else:
        if args.hints not in ("auto", "none"):
            raise E.UsageError("--hints FILE is for the r5b locator; the reference matcher takes no hints")
        from skyhum.detect.spectrogram import logspec
        from skyhum.score import find as F
        r = F.run([(logspec(xs, 16000), l, h) for xs, l, h in sessions], tracks, area, top=args.top,
                  n_decoys=n_decoys(args, 200), seed=args.seed, spacing_km=args.spacing, ground_m=args.ground_alt,
                  progress=out.progress("find"))
    rows = [("find", os.path.basename(args.audio) + (f" + {len(sessions) - 1} session(s)" if len(sessions) > 1
                                                      else "")),
            ("area", f"{area.label} ({area.extent_km():.0f} km across)"),
            ("matcher", matcher_label(r)),
            ("decoys", f"{r['decoys_used']} clocks; best {r['best_score']} vs decoy level {r['decoy_level']}")]
    for t in r["ranked"]:
        rows.append((f"tile {t['rank']}", "bbox:" + ",".join(f"{v:.1f}" for v in t["tile_bbox"])
                     + f"  score {t['score']}"))
    if not r["ranked"]:
        rows.append(("ranking", "nothing beats the decoys; no ranking shown"))
    pe = r["point_estimate"]
    rows.append(("point estimate", f"{pe['lat']:.4f},{pe['lon']:.4f}" if pe else "no point estimate "
                                                                              "(below the strict threshold)"))
    if pe and pe.get("mirror"):
        rows.append(("mirror point", f"{pe['mirror']['lat']:.4f},{pe['mirror']['lon']:.4f} (single pass: either side "
                                     "of the aircraft track fits)"))
    rows.append(("confidence", "research use, not validated" if r["matcher"] == "r5b" else "not calibrated"))
    rows.append(("result", r["result"]))
    out.line(fmt_kv(rows))
    code = {"POINT_ESTIMATE": E.OK, "NO_POINT_ESTIMATE": E.UNCORROBORATED, "INCONCLUSIVE": E.INCONCLUSIVE}[r["result"]]
    return finish(args, out, "find", r, code)


def cmd_bench(args, out):
    from skyhum.bench import SOURCES
    if args.bench_cmd in (None, "list"):
        for s in SOURCES:
            out.line(f"  {s['id']:<11} {s['what']}\n              {s['url']}  licence: {s['licence']}")
        out.line("The repository holds links and metadata only, never audio.")
        return finish(args, out, "bench list", dict(sources=SOURCES), E.OK)
    if args.bench_cmd == "fetch":
        raise E.NotImplementedInBuild("bench fetch", "catalogue licence review pending (docs/bench.md)")
    raise E.NotImplementedInBuild("bench run", "scoring and table reproduction are planned for phase P4")


def cmd_adsb(args, out):
    raise E.NotImplementedInBuild("adsb fetch", "download ADS-B days yourself (docs/data.md) and pass --adsb PATH")


COMMANDS = dict(selftest=cmd_selftest, doctor=cmd_doctor, regions=cmd_regions, detect=cmd_detect, verify=cmd_verify,
                audit=cmd_audit, scrub=cmd_scrub, find=cmd_find, bench=cmd_bench, adsb=cmd_adsb)


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    out = Out(args)
    if not args.command:
        ap.print_help()
        return E.USAGE
    try:
        return COMMANDS[args.command](args, out)
    except E.SkyhumError as e:
        if out.json:
            print(dumps(document(args.command, dict(error=str(e)), e.exit_code)))
        print(f"skyhum {args.command}: {e}", file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
