import json

from conftest import SITE, START, fx

VERIFY_KEYS = {"verdict", "operating_point", "llr", "confidence", "decoy_rank", "passes", "claim",
               "tested_clock_window", "limits", "version", "seed", "matcher", "benchmarked"}


def test_verify_json_schema(run_cli, demo, tmp_path):
    rep = tmp_path / "r.json"
    code, out, _ = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--claim", f"{SITE},r=2km", "--matcher", "reference", "--report", rep, "--json")
    doc = json.loads(out)
    assert doc["schema"] == 1 and doc["command"] == "verify" and doc["exit_code"] == code == 0
    r = doc["result"]
    assert VERIFY_KEYS <= set(r)
    assert r["benchmarked"] is False and r["llr"] is None
    assert r["decoy_rank"].endswith(f"of {r['decoys_used'] + 1}")
    for p in r["passes"]:
        assert set(p) >= {"t_cpa", "aircraft", "z", "vote", "d_cpa_km"} and len(p["aircraft"]) == 10
    assert json.loads(rep.read_text()) == doc


def test_html_report_is_self_contained(run_cli, demo, tmp_path):
    rep = tmp_path / "r.html"
    code, _, _ = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                         "--claim", f"{SITE},r=2km", "--matcher", "reference", "--report", rep)
    html = rep.read_text()
    assert code == 0 and "SUPPORTED" in html and "<title>" in html
    assert "http://" not in html and "https://" not in html and "<script" not in html


def test_report_rejects_other_extensions(run_cli, demo, tmp_path):
    code, _, _ = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                         "--claim", f"{SITE},r=2km", "--matcher", "reference", "--report", tmp_path / "r.txt")
    assert code == 2
