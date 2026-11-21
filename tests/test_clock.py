import datetime as dt

import pytest

from skyhum.errors import UsageError
from skyhum.io.clock import parse_duration, parse_start, parse_tolerance

T = dt.datetime(2026, 9, 8, 14, 3, tzinfo=dt.timezone.utc).timestamp()


def test_start_forms():
    assert parse_start("2026-09-08T14:03:00Z") == T
    assert parse_start("2026-09-08T15:03:00+01:00") == T
    assert parse_start(str(int(T))) == T
    assert parse_start("2026-09-08T15:03:00", tz="Europe/London") == T      # BST in September


def test_naive_start_needs_tz():
    with pytest.raises(UsageError, match="time zone"):
        parse_start("2026-09-08T14:03:00")


def test_bad_start_and_tz():
    with pytest.raises(UsageError):
        parse_start("yesterday")
    with pytest.raises(UsageError):
        parse_start("2026-09-08T14:03:00", tz="Mars/Olympus")


@pytest.mark.parametrize("text,sec", [("5s", 5), ("30s", 30), ("30m", 1800), ("2h", 7200), ("90", 90)])
def test_durations(text, sec):
    assert parse_duration(text) == sec


def test_tolerance():
    assert parse_tolerance("30s", T) == (T - 30, T + 30)
    lo, hi = parse_tolerance("date", T)
    assert lo <= T <= hi and hi - lo == 86399
    with pytest.raises(UsageError):
        parse_tolerance("13h", T)
    with pytest.raises(UsageError):
        parse_duration("soon")
