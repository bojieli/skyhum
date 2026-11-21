"""Safety gate for `skyhum find`.

These checks are speed bumps, not security: anyone can edit the code. They exist so the tool's default behaviour
matches its purpose (checking your own or consented audio) and so that every find run leaves a local record.
The real limits are the low blind success rate, the area cap and the absence of any country-wide search.

  1. --research must be given.
  2. --attest must be one of ATTEST.
  3. The area must be at most MAX_EXTENT_KM across (largest side of its bounding box). Larger asks are refused,
     not truncated.
  4. Every attempt, allowed or refused, is appended to <workdir>/skyhum-find.log with the attestation and a sha256
     of each audio file. Nothing is sent anywhere.
"""
import json
import os
import time

from skyhum import __version__
from skyhum.errors import Refused

ATTEST = ("own", "consented", "public-benchmark")
MAX_EXTENT_KM = 100.0
LOG_NAME = "skyhum-find.log"


def log_path(workdir):
    return os.path.join(workdir, LOG_NAME)


def _log(workdir, entry):
    os.makedirs(workdir, exist_ok=True)
    with open(log_path(workdir), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def check(*, research, attest, area, audio_paths, workdir):
    """Raise Refused (exit 4) unless every condition holds. Logs the attempt either way."""
    from skyhum.io.audio import sha256
    hashes = {os.path.basename(p): (sha256(p) if os.path.isfile(p) else None) for p in audio_paths}
    entry = dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), version=__version__, attest=attest,
                 research=bool(research), area=area.to_json() if area is not None else None, audio_sha256=hashes)
    reason = None
    if not research:
        reason = ("find is a research mode: pass --research and --attest own|consented|public-benchmark. "
                  "To check a place you already have in mind, use `skyhum verify` or `skyhum audit`.")
    elif attest not in ATTEST:
        reason = f"--attest must be one of {', '.join(ATTEST)} (got {attest!r})"
    elif area is None:
        reason = "find needs --area (one metro, or at most 100 km across)"
    elif area.extent_km() > MAX_EXTENT_KM:
        reason = (f"the area is {area.extent_km():.0f} km across; find is limited to one metro or at most "
                  f"{MAX_EXTENT_KM:.0f} km across. Larger areas are refused, not truncated.")
    entry["outcome"] = "refused" if reason else "allowed"
    if reason:
        entry["reason"] = reason
    _log(workdir, entry)
    if reason:
        raise Refused(reason)
    return log_path(workdir)
