"""Start time and clock tolerance parsing (`--start`, `--tz`, `--start-tol`, `--duration`)."""
import datetime as dt
import re

from skyhum.errors import UsageError

_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|min|h)?\s*$", re.I)
_UNIT = {None: 1.0, "ms": 1e-3, "s": 1.0, "m": 60.0, "min": 60.0, "h": 3600.0}


def parse_duration(text):
    """'5s', '30s', '30m', '2h', '90' (seconds) -> seconds (float)."""
    m = _DUR.match(str(text))
    if not m:
        raise UsageError(f"cannot read duration {text!r}: use forms like 5s, 30s, 30m, 2h")
    return float(m.group(1)) * _UNIT[(m.group(2) or "").lower() or None]


def parse_start(text, tz=None):
    """UTC ISO-8601 with Z or an offset, epoch seconds, or a naive local time with --tz. Returns epoch seconds."""
    if text is None:
        return None
    s = str(text).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as e:
        raise UsageError(f"cannot read start time {text!r}: use 2026-09-08T14:03:00Z or epoch seconds") from e
    if d.tzinfo is None:
        if not tz:
            raise UsageError(f"start time {text!r} has no time zone: add Z for UTC, an offset such as +01:00, "
                             "or pass --tz (for example --tz Europe/London)")
        try:
            from zoneinfo import ZoneInfo
            d = d.replace(tzinfo=ZoneInfo(tz))
        except Exception as e:  # ZoneInfoNotFoundError, ValueError
            raise UsageError(f"unknown time zone {tz!r}") from e
    return d.timestamp()


def parse_tolerance(text, start):
    """Returns (window_lo, window_hi) for the recording start, in epoch seconds.

    'date' means the whole UTC day that contains `start` (the weakest test)."""
    if str(text).strip().lower() == "date":
        day = dt.datetime.fromtimestamp(start, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        lo = day.timestamp()
        return lo, lo + 86400.0 - 1.0
    tol = parse_duration(text)
    if tol > 12 * 3600:
        raise UsageError("--start-tol above 12h: use 'date' for a whole-day window")
    return start - tol, start + tol


def iso(t):
    return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hms(t):
    return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%H:%M:%SZ")
