"""The find safety gate: refusals (exit 4), the local log, and no point estimate below the strict threshold."""
import json
import re

import pytest

from conftest import SITE, START, fx


def find(run_cli, demo, *extra, audio="clip.wav", area=f"{SITE},r=4km"):
    return run_cli("find", fx(demo, audio), "--adsb", fx(demo, "tracks.npz"), "--start", START, "--area", area,
                   "--matcher", "reference", *extra)


def read_log(tmp_path):
    p = tmp_path / "workdir" / "skyhum-find.log"
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_refuses_without_research(run_cli, demo, tmp_path):
    code, out, err = find(run_cli, demo, "--attest", "own")
    assert code == 4 and "--research" in err and "PASS" not in out
    log = read_log(tmp_path)
    assert log[-1]["outcome"] == "refused" and log[-1]["audio_sha256"]["clip.wav"]


def test_refuses_without_attest(run_cli, demo):
    assert find(run_cli, demo, "--research")[0] == 4


@pytest.mark.parametrize("bad", ["mine", "OWN", "", "public"])
def test_refuses_bad_attest(run_cli, demo, bad):
    code, _, err = find(run_cli, demo, "--research", "--attest", bad)
    assert code == 4 and "--attest" in err


@pytest.mark.parametrize("area", ["bbox:-125,30,-110,45", "34.05,-118.30,r=60km", "bbox:-119.5,33.5,-118.0,34.5"])
def test_refuses_large_area(run_cli, demo, area):
    code, _, err = find(run_cli, demo, "--research", "--attest", "own", area=area)
    assert code == 4 and "refused, not truncated" in err


def test_refuses_large_geojson(run_cli, demo, tmp_path):
    p = tmp_path / "big.geojson"
    p.write_text(json.dumps({"type": "Polygon", "coordinates": [[[-120, 33], [-117, 33], [-117, 35], [-120, 35],
                                                                 [-120, 33]]]}))
    assert find(run_cli, demo, "--research", "--attest", "own", area=f"@{p}")[0] == 4


def test_gate_runs_before_anything_else(run_cli, tmp_path):
    """A refused request is refused even when the audio and ADS-B do not exist."""
    code, _, err = run_cli("find", tmp_path / "none.wav", "--adsb", tmp_path / "none.npz", "--start", START,
                           "--area", "bbox:-125,30,-110,45", "--research", "--attest", "own")
    assert code == 4


def test_allowed_run_logs_and_gives_point_estimate_only_when_strict(run_cli, demo, tmp_path):
    code, out, _ = find(run_cli, demo, "--research", "--attest", "own", "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["strict"] and r["point_estimate"] is not None
    log = read_log(tmp_path)
    assert log[-1]["outcome"] == "allowed" and log[-1]["attest"] == "own"
    assert len(log[-1]["audio_sha256"]["clip.wav"]) == 64


def test_no_point_estimate_below_strict(run_cli, demo, monkeypatch):
    from skyhum.score import verify as V
    monkeypatch.setitem(V.THETA, "strict", 1e9)                  # nothing can pass strict
    code, out, _ = find(run_cli, demo, "--research", "--attest", "own")
    assert code == 10 and "no point estimate" in out
    # apart from the echo of the area the user supplied, only coarse 0.1-degree tiles may appear
    body = "\n".join(line for line in out.splitlines() if not line.startswith("area "))
    assert not re.search(r"-?\d+\.\d{2,}\s*,\s*-?\d+\.\d{2,}", body)
    assert "tile 1" in body
    code, out, _ = find(run_cli, demo, "--research", "--attest", "own", "--json")
    r = json.loads(out)["result"]
    assert r["point_estimate"] is None and r["result"] == "NO_POINT_ESTIMATE"


def test_quiet_audio_inconclusive_no_estimate(run_cli, demo):
    code, out, _ = find(run_cli, demo, "--research", "--attest", "public-benchmark", "--json", audio="quiet.wav")
    r = json.loads(out)["result"]
    assert code in (10, 11) and r["point_estimate"] is None and r["ranked"] == []
    assert r["result"] in ("NO_POINT_ESTIMATE", "INCONCLUSIVE")


def test_sessions_need_starts(run_cli, demo):
    code, _, err = find(run_cli, demo, "--research", "--attest", "own", "--sessions", fx(demo, "call.wav"))
    assert code == 2 and "--session-starts" in err


def test_multi_session_joint(run_cli, demo):
    code, out, _ = find(run_cli, demo, "--research", "--attest", "own", "--sessions", fx(demo, "call.wav"),
                        "--session-starts", START, "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["sessions"] == 2
