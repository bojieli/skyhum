import json

from conftest import SITE, START, fx


def test_no_command_prints_help(run_cli):
    code, out, _ = run_cli()
    assert code == 2 and "verify" in out and "find" in out


def test_help_lists_exit_codes(run_cli):
    code, out, _ = run_cli("verify", "--help")
    assert code == 0 and "exit codes" in out and "--start-tol" in out and "--claim" in out


def test_doctor(run_cli):
    code, out, _ = run_cli("--device", "cpu", "doctor")
    assert code == 0 and "numpy" in out and "r5b tables" in out


def test_regions_json(run_cli):
    code, out, _ = run_cli("regions", "list", "--json")
    regs = json.loads(out)["result"]["regions"]
    assert code == 0 and any(r["name"] == "Los Angeles basin" for r in regs)


def test_global_flags_before_or_after(run_cli):
    a = run_cli("--json", "regions", "list")
    b = run_cli("regions", "list", "--json")
    assert json.loads(a[1]) == json.loads(b[1])


def test_detect(run_cli, demo, tmp_path):
    code, out, _ = run_cli("detect", fx(demo, "clip.wav"), "--start", START, "--json", "--detector", "classic")
    r = json.loads(out)["result"]
    assert code == 0 and r["detector"] == "classic-ridge" and sum(p["kind"] == "doppler-like" for p in r["passes"]) >= 2
    code, out, _ = run_cli("detect", fx(demo, "quiet.wav"), "--json", "--detector", "classic")
    assert code == 0 and not [p for p in json.loads(out)["result"]["passes"] if p["kind"] == "doppler-like"]


def test_detect_png(run_cli, demo, tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    png = tmp_path / "s.png"
    code, out, _ = run_cli("detect", fx(demo, "clip.wav"), "--png", png)
    assert code == 0 and png.stat().st_size > 1000


def test_audit_with_scrub(run_cli, demo, tmp_path):
    out_wav = tmp_path / "clean.wav"
    code, out, _ = run_cli("audit", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--where", f"{SITE},r=1km", "--scrub", out_wav, "--json", "--matcher", "reference")
    r = json.loads(out)["result"]
    assert code == 0 and r["true_claim"]["verdict"] == "SUPPORTED"
    assert r["scrubbed"]["true_claim"]["verdict"] != "SUPPORTED" and out_wav.exists()
    assert r["exposure_tier"]["modelled"] is None and r["advice"]


def test_bench(run_cli):
    code, out, _ = run_cli("bench", "list")
    assert code == 0 and "ODbL" in out and "xeno-canto" in out
    assert run_cli("bench", "fetch", "--source", "xeno-canto")[0] == 2
    assert run_cli("bench", "run", "--reproduce", "paper")[0] == 2


def test_adsb_fetch_not_in_build(run_cli):
    code, _, err = run_cli("adsb", "fetch", "--area", "Los Angeles basin", "--date", "2026-09-08")
    assert code == 2 and "not yet implemented in this build" in err


def test_error_as_json(run_cli, demo):
    code, out, _ = run_cli("--json", "verify", fx(demo, "clip.wav"), "--adsb", "auto", "--claim", f"{SITE},r=2km",
                           "--start", START)
    d = json.loads(out)
    assert code == 2 and d["exit_code"] == 2 and "error" in d["result"]
