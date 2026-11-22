"""Stage recordings for the country search.

A staged session is <root>/blind/<oid>/ with
  audio.<ext> or spec.npz   the recording (a link or copy) or its search spectrogram (4 kHz, 1 s windows, 0.2 s hop)
  input.json                {oid, region_prior ("CONUS" | "AU"), clock_utc, dur_s, windows: [{w, t0, t1, kind}]}
  hints_v3.json             detector-v3 detections [{t, p}, ...] (recording seconds); required: without detections
                            the search proposes no candidate cells
A group <root>/groups/<gid>.json lists sessions known to come from one place and the subsets to score.
Benchmark staging also writes <root>/truth/truth.json (opened only by the scorers) and <root>/decoy_days.json.
"""
import os
import secrets
import shutil

from skyhum.country.common import dump_json, load_json

AUDIO_EXT = (".mp3", ".flac", ".wav", ".m4a", ".opus", ".webm", ".ogg")


def windows(dur):
    """Each full hour ("60"; 30-55 min recordings give one window of their own length), the half hours ("30"),
    and minutes 0-10 and 30-40 of each hour ("10")."""
    out = []
    nh = int((dur + 300) // 3600)
    if nh == 0 and dur >= 1800:
        out.append(dict(kind="60", t0=0.0, t1=float(dur), note="full<60"))
    for k in range(nh):
        a = 3600.0 * k
        out.append(dict(kind="60", t0=a, t1=min(a + 3600.0, dur)))
    for k in range(int(dur // 1800)):
        out.append(dict(kind="30", t0=1800.0 * k, t1=1800.0 * k + 1800))
    for k in range(int(dur // 1800)):
        out.append(dict(kind="10", t0=1800.0 * k, t1=1800.0 * k + 600))
    for i, w in enumerate(out):
        w["w"] = i
    return out


def _place_audio(src, bd):
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(bd, "audio" + ext)
    if os.path.exists(dst):
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return dst


def stage_session(root, region, clock_utc, dur_s, audio=None, spec=None, hints=None, oid=None, wins=None):
    """One recording -> blind/<oid>/. hints: a list of detections or a path to a JSON list."""
    oid = oid or secrets.token_hex(5)
    bd = os.path.join(root, "blind", oid)
    os.makedirs(bd, exist_ok=True)
    if audio:
        _place_audio(audio, bd)
    if spec:
        shutil.copyfile(spec, os.path.join(bd, "spec.npz"))
    if hints is not None:
        h = load_json(hints) if isinstance(hints, str) else hints
        dump_json(h, os.path.join(bd, "hints_v3.json"))
    dump_json(dict(oid=oid, region_prior=region, clock_utc=float(clock_utc), dur_s=float(dur_s),
                   windows=wins or windows(dur_s)), os.path.join(bd, "input.json"), indent=1)
    return oid


def write_group(root, sessions, subsets=None, gid=None):
    gid = gid or ("g" + secrets.token_hex(4) if len(sessions) > 1 else "s" + sessions[0])
    os.makedirs(os.path.join(root, "groups"), exist_ok=True)
    p = os.path.join(root, "groups", gid + ".json")
    dump_json(dict(gid=gid, sessions=list(sessions), subsets=subsets or [[i] for i in range(len(sessions))]), p)
    return p


def find_audio(audio_dir, item):
    """<audio_dir>/<item id>.<ext> or <audio_dir>/<item id>/audio.<ext>."""
    for ext in AUDIO_EXT:
        for p in (os.path.join(audio_dir, item["id"] + ext), os.path.join(audio_dir, item["id"], "audio" + ext)):
            if os.path.exists(p):
                return p
    return None


def stage_manifest(root, manifest, audio_dir=None, hints_dir=None, spec_dir=None, log=print):
    """Stage a country benchmark manifest (e.g. skyhum/bench/manifests/a2o_country_v1.json, or a webcam feature
    release in the same layout). Per item: audio from audio_dir, or a spectrogram <spec_dir>/<id>.npz; detector hints
    <hints_dir>/<id>.json. Writes blind/, groups/, truth/truth.json and decoy_days.json. Returns (staged, missing)."""
    M = load_json(manifest) if isinstance(manifest, str) else manifest
    region = M.get("region", "AU")
    truth_p = os.path.join(root, "truth", "truth.json")
    os.makedirs(os.path.dirname(truth_p), exist_ok=True)
    truth = load_json(truth_p) if os.path.exists(truth_p) else {}
    decoy = load_json(os.path.join(root, "decoy_days.json")) if os.path.exists(
        os.path.join(root, "decoy_days.json")) else {}
    staged, missing = [], []
    for it in M["items"]:
        oid = it["session"]
        audio = find_audio(audio_dir, it) if audio_dir else None
        spec = os.path.join(spec_dir, it["id"] + ".npz") if spec_dir else None
        spec = spec if spec and os.path.exists(spec) else None
        hp = os.path.join(hints_dir, it["id"] + ".json") if hints_dir else None
        hp = hp if hp and os.path.exists(hp) else None
        if not (audio or spec) or not hp:
            missing.append(dict(id=it["id"], audio=bool(audio or spec), hints=bool(hp)))
            continue
        stage_session(root, region, it["clock_utc"], it["duration_s"], audio=audio, spec=spec, hints=hp, oid=oid,
                      wins=it["windows"])
        truth[oid] = dict(case_id=it["id"], set=M.get("set", "a2o"), lat=it["lat"], lon=it["lon"],
                          site=it.get("site") or f"a2o{it['a2o_site_id']}", clock=it["clock_utc"],
                          clock_verified=it.get("clock_verified", True), dur=it["duration_s"], region=region,
                          windows=it["windows"], pos=it.get("labelled_passes", []))
        decoy[oid] = it.get("decoy_day_offsets", [])
        staged.append(oid)
    dump_json(truth, truth_p)
    os.chmod(truth_p, 0o600)
    dump_json(decoy, os.path.join(root, "decoy_days.json"))
    ok = set(staged) | {o for o in truth}
    for k, g in enumerate(M.get("groups", [])):
        if set(g["sessions"]) <= ok:
            gid = ("g" if len(g["sessions"]) > 1 else "s") + f"{M['name'].replace('_', '')}{k:03d}"
            write_group(root, g["sessions"], g["subsets"], gid=gid)
    log(f"staged {len(staged)} of {len(M['items'])} items; {len(missing)} lack audio or detector hints")
    return staged, missing
