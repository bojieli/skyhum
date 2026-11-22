"""Resumable local runner: search every group of a work directory, one process per group, N at a time.

Each group runs in its own process (the truth guard is installed once per process and allows only that group's
inputs). Finished outputs (done=true) are skipped; an interrupted group resumes from its finished variants. A group
that stops on a missing ADS-B day is reported with the fetch command and left for a later run.

  python -m skyhum.country.runner all ROOT TAG [--jobs N] [--groups GID,...]
  python -m skyhum.country.runner one ROOT GROUP_JSON OUT_JSON
"""
import argparse
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import time

from skyhum.country.common import load_json

MISSING_DAY_EXIT = 3


def paths(root):
    from importlib import resources
    model = os.environ.get("SKYHUM_R5B_MODELS")
    model = os.path.join(model, "k3_full.json") if model else str(
        resources.files("skyhum.models").joinpath("r5b", "k3_full.json"))
    return dict(adsb=os.path.join(root, "adsb"), static=os.path.join(root, "static"), model=model)


def done(out):
    try:
        return bool(load_json(out).get("done"))
    except (OSError, ValueError):
        return False


def run_all(root, tag, jobs=1, groups=None, log=print, extra_env=None):
    """Returns {gid: status} with status done / failed / missing-day."""
    files = sorted(glob.glob(os.path.join(root, "groups", "*.json")))
    if groups:
        keep = set(groups)
        files = [f for f in files if os.path.basename(f)[:-5] in keep]
    od = os.path.join(root, "runs", tag)
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(od, "_log"), exist_ok=True)
    todo = [(f, os.path.join(od, os.path.basename(f))) for f in files]
    status = {os.path.basename(f)[:-5]: "done" for f, o in todo if done(o)}
    todo = [(f, o) for f, o in todo if not done(o)]
    log(f"{len(files)} groups, {len(status)} already done, {len(todo)} to run with {jobs} job(s)")
    env = dict(os.environ, **(extra_env or {}))

    def one(fo):
        f, o = fo
        gid = os.path.basename(f)[:-5]
        t = time.time()
        with open(os.path.join(od, "_log", gid + ".log"), "a") as lf:
            r = subprocess.run([sys.executable, "-m", "skyhum.country.runner", "one", root, f, o], stdout=lf,
                               stderr=subprocess.STDOUT, env=env)
        st = "done" if r.returncode == 0 and done(o) else (
            "missing-day" if r.returncode == MISSING_DAY_EXIT else "failed")
        log(f"  {gid}: {st} ({time.time() - t:.0f} s)")
        return gid, st
    with cf.ThreadPoolExecutor(max(1, jobs)) as ex:
        for gid, st in ex.map(one, todo):
            status[gid] = st
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m skyhum.country.runner")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("all")
    a1.add_argument("root")
    a1.add_argument("tag")
    a1.add_argument("--jobs", type=int, default=1)
    a1.add_argument("--groups")
    a2 = sp.add_parser("one")
    a2.add_argument("root")
    a2.add_argument("group")
    a2.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "all":
        st = run_all(a.root, a.tag, a.jobs, a.groups.split(",") if a.groups else None)
        return 0 if all(v == "done" for v in st.values()) else 1
    from skyhum.country import search
    p = paths(a.root)
    try:
        search.run_group(a.group, a.out, a.root, p["adsb"], p["static"], p["model"])
    except search.MissingDay as e:
        print(str(e), flush=True)
        return MISSING_DAY_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
