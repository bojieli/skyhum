"""find without a gate: any area, no attestation, and still no point estimate below the strict threshold."""
import json
import re

from conftest import SITE, START, fx


def find(run_cli, demo, *extra, audio="clip.wav", area=f"{SITE},r=4km"):
    return run_cli("find", fx(demo, audio), "--adsb", fx(demo, "tracks.npz"), "--start", START, "--area", area,
                   "--matcher", "reference", *extra)


def test_no_attestation_needed(run_cli, demo):
    code, out, _ = find(run_cli, demo, "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["strict"] and r["point_estimate"] is not None
    assert "attest" not in r and "log" not in r


def test_research_flags_are_gone(run_cli, demo):
    assert find(run_cli, demo, "--research")[0] == 2           # argparse: unknown option


def test_no_area_cap(run_cli, demo):
    code, _, err = find(run_cli, demo, "--spacing", "5", area="bbox:-119.2,33.6,-117.4,34.6")
    assert code in (0, 10, 11) and "refused" not in err


def test_no_point_estimate_below_strict(run_cli, demo, monkeypatch):
    from skyhum.score import verify as V
    monkeypatch.setitem(V.THETA, "strict", 1e9)                  # nothing can pass strict
    code, out, _ = find(run_cli, demo)
    assert code == 10 and "no point estimate" in out
    body = "\n".join(line for line in out.splitlines() if not line.startswith("area "))
    assert not re.search(r"-?\d+\.\d{2,}\s*,\s*-?\d+\.\d{2,}", body)
    code, out, _ = find(run_cli, demo, "--json")
    r = json.loads(out)["result"]
    assert r["point_estimate"] is None and r["result"] == "NO_POINT_ESTIMATE"


def test_quiet_audio_no_estimate(run_cli, demo):
    code, out, _ = find(run_cli, demo, "--json", audio="quiet.wav")
    r = json.loads(out)["result"]
    assert code in (10, 11) and r["point_estimate"] is None and r["ranked"] == []


def test_sessions_need_starts(run_cli, demo):
    code, _, err = find(run_cli, demo, "--sessions", fx(demo, "call.wav"))
    assert code == 2 and "--session-starts" in err


def test_multi_session_joint(run_cli, demo):
    code, out, _ = find(run_cli, demo, "--sessions", fx(demo, "call.wav"), "--session-starts", START, "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["sessions"] == 2
