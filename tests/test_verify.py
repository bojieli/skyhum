"""Golden tests on the synthetic fixtures: verdicts and exit codes."""
import json

import pytest

from conftest import SITE, START, fx

ARGS = ["--start", START, "--matcher", "reference"]


def verify(run_cli, demo, audio, claim, *extra):
    return run_cli("verify", fx(demo, audio), "--adsb", fx(demo, "tracks.npz"), "--claim", claim, *ARGS, *extra)


def test_true_claim_supported(run_cli, demo):
    code, out, _ = verify(run_cli, demo, "clip.wav", f"{SITE},r=2km")
    assert code == 0 and "SUPPORTED" in out


def test_true_claim_strict(run_cli, demo):
    code, out, _ = verify(run_cli, demo, "clip.wav", f"{SITE},r=2km", "--operating-point", "strict", "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["verdict"] == "SUPPORTED" and r["operating_points"] == {"default": True, "strict": True}


def test_geojson_claim(run_cli, demo):
    code, _, _ = verify(run_cli, demo, "clip.wav", "@" + fx(demo, "area.geojson"))
    assert code == 0


def test_displaced_claim_not_supported(run_cli, demo):
    code, out, _ = verify(run_cli, demo, "clip.wav", "34.05,-118.10,r=2km")
    assert code in (10, 11) and "SUPPORTED" not in out.replace("UNCORROBORATED", "")


def test_quiet_audio_uncorroborated(run_cli, demo):
    code, out, _ = verify(run_cli, demo, "quiet.wav", f"{SITE},r=2km")
    assert code == 10 and "not evidence that the claim is false" in out


def test_wrong_clock_uncorroborated(run_cli, demo):
    code, _, _ = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--claim",
                         f"{SITE},r=2km", "--start", "2026-09-08T14:28:00Z", "--matcher", "reference")
    assert code == 10


def test_versus_rival_counts_as_decoy(run_cli, demo):
    code, out, _ = verify(run_cli, demo, "clip.wav", "34.05,-118.10,r=2km", "--versus", f"{SITE},r=2km", "--json")
    r = json.loads(out)["result"]
    assert code in (10, 11) and r["versus"]["score"] > 0


def test_missing_coverage_is_missing_data(run_cli, demo):
    code, _, err = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--claim",
                           f"{SITE},r=2km", "--start", "2026-09-09T14:03:00Z")
    assert code == 3 and "does not cover" in err


def test_deterministic(run_cli, demo):
    a = json.loads(verify(run_cli, demo, "clip.wav", f"{SITE},r=2km", "--json")[1])["result"]
    b = json.loads(verify(run_cli, demo, "clip.wav", f"{SITE},r=2km", "--json")[1])["result"]
    for k in ("verdict", "claim_score", "decoy_level", "decoy_rank", "passes"):
        assert a[k] == b[k]


def test_claim_score_does_not_depend_on_decoys(run_cli, demo):
    """Blindness: the claim's own score is computed the same way whatever decoys are drawn."""
    a = json.loads(verify(run_cli, demo, "clip.wav", f"{SITE},r=2km", "--json", "--decoys", "30")[1])["result"]
    b = json.loads(verify(run_cli, demo, "clip.wav", f"{SITE},r=2km", "--json", "--seed", "5")[1])["result"]
    assert a["claim_score"] == b["claim_score"]


@pytest.mark.parametrize("argv,code", [
    (["--adsb", "auto"], 2),                       # automatic download not in this build
    (["--device", "cuda", "--matcher", "reference"], 2),   # the reference matcher is CPU only
    (["--start-tol", "2h", "--matcher", "reference"], 2),  # clock window limit
])
def test_not_in_build(run_cli, demo, argv, code):
    base = ["verify", fx(demo, "clip.wav"), "--claim", f"{SITE},r=2km", "--start", START]
    if "--adsb" not in argv:
        base += ["--adsb", fx(demo, "tracks.npz")]
    got, _, err = run_cli(*base, *argv)
    assert got == code and "not yet implemented in this build" in err


def test_usage_errors(run_cli, demo):
    assert run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--claim", "x")[0] == 2
    assert run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--claim",
                   f"{SITE},r=2km")[0] == 2                                   # no --start
    assert run_cli("verify", fx(demo, "nope.wav"), "--adsb", fx(demo, "tracks.npz"), "--claim",
                   f"{SITE},r=2km", "--start", START)[0] == 2
    assert run_cli("verify", fx(demo, "clip.wav"), "--claim", f"{SITE},r=2km")[0] == 2       # no --adsb, no --start
    assert run_cli("verify")[0] == 2                                          # argparse
    assert run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "missing.npz"), "--claim",
                   f"{SITE},r=2km", "--start", START)[0] == 3
