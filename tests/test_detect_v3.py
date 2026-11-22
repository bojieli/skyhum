"""Detector v3 (learned fly-over detector), CPU. Synthetic audio only; the reference detections in
tests/data/v3_selftest_research.json were produced by the research code on the same synthetic clips."""
import json
import os

import numpy as np
import pytest

from conftest import START, fx

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def det():
    from skyhum.detect import v3
    return v3.Detector(seed=0, threads=4)


def test_checkpoint_is_a_plain_state_dict():
    from skyhum.detect.v3.net import build_net, weights_path
    ck = torch.load(weights_path(), map_location="cpu", weights_only=True)
    assert ck["arch"]["kind"] == "v1"
    assert set(ck["state"]) == set(build_net().state_dict())


@pytest.mark.parametrize("name", ["clip.wav", "call.wav", "quiet.wav"])
def test_matches_research_detections(det, demo, name):
    from skyhum.detect import v3
    want = json.load(open(os.path.join(HERE, "data", "v3_selftest_research.json")))["detections"][name]
    got = det.detect_1k(v3.load_1k(fx(demo, name)))
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert set(g) == set(w)
        assert g["t"] == w["t"] and g["start"] == w["start"] and abs(g["f"] - w["f"]) < 1e-9
        for k in ("p", "p_raw"):
            assert abs(g[k] - w[k]) < 1e-4
        for k in ("v", "d"):
            assert abs(g[k] / w[k] - 1) < 1e-4


def test_deterministic_and_rate_independent(det, demo):
    from skyhum.io import audio as A
    x, _ = A.load(fx(demo, "clip.wav"))
    a, b = det.detect(x, 16000), det.detect(x, 16000)
    assert a == b
    hi = [(d["t"], round(d["f"])) for d in a if d["p"] >= 0.8]
    assert len(hi) >= 2 and {120.0, 190.4} <= {t for t, _ in hi}


def test_short_audio_is_padded_to_one_window(det):
    x = np.random.default_rng(0).standard_normal(30 * 16000).astype(np.float32)
    got = det.detect(x, 16000)
    assert got and all(d["start"] == 0.0 and {"t", "f", "p", "p_raw", "v", "d"} <= set(d) for d in got)


def test_gate_and_harmonics():
    from skyhum.detect.v3.detect import _harmonic, gate
    assert gate(60, 1000) > 0.99 and gate(15, 1000) < 1e-3 and gate(60, 50) < 1e-3
    assert gate(1e-30, 1000) == 0.0                                          # no overflow
    assert _harmonic(100, 201) and _harmonic(300, 100) and not _harmonic(100, 150)


def test_cli_detect_v3_and_hints_out(run_cli, demo, tmp_path):
    h = tmp_path / "h.json"
    code, out, _ = run_cli("detect", fx(demo, "clip.wav"), "--start", START, "--hints-out", h, "--json")
    r = json.loads(out)["result"]
    assert code == 0 and r["detector"] == "v3" and r["n_hint"] >= 2
    assert all("t_utc" in d for d in r["detections"])
    H = json.load(open(h))
    assert len(H) == len(r["detections"]) and {"t", "f", "p"} <= set(H[0])
    code, out, _ = run_cli("detect", fx(demo, "quiet.wav"), "--json")
    assert code == 0 and json.loads(out)["result"]["n_hint"] == 0


def test_cli_detect_classic_has_no_hints_out(run_cli, demo):
    code, _, err = run_cli("detect", fx(demo, "clip.wav"), "--detector", "classic", "--hints-out", "x.json")
    assert code == 2 and "v3" in err


@pytest.fixture
def fake_r5b(monkeypatch):
    """Replace the locator run by a stub that records its argv and the hints file it was given."""
    from skyhum.match import r5b
    calls = []

    def main(argv):
        h = argv[argv.index("--hints") + 1] if "--hints" in argv else None
        calls.append(dict(argv=list(argv), hints=json.load(open(h)) if h else None))
        return dict(hypotheses=[], n_passes=0, n_sites=0)
    monkeypatch.setattr(r5b, "main", main)
    return calls


@pytest.mark.parametrize("mode", ["auto", "file", "none"])
def test_verify_passes_hints_to_the_locator(run_cli, demo, tmp_path, fake_r5b, mode):
    from skyhum.match.locator import frozen_args
    hf = tmp_path / "h.json"
    run_cli("detect", fx(demo, "clip.wav"), "--hints-out", hf)
    arg = {"auto": "auto", "file": hf, "none": "none"}[mode]
    code, out, _ = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--claim", "34.05,-118.30,r=2km", "--matcher", "r5b", "--fast", "--decoys", "5",
                           "--hints", arg, "--json")
    r = json.loads(out)["result"]
    assert code == 11 and fake_r5b                                     # stub: no passes -> INCONCLUSIVE
    argv = fake_r5b[0]["argv"]
    assert argv[argv.index("--hint-pmin") + 1] == "0.8" and "--hint-pmin" in frozen_args()
    if mode == "none":
        assert fake_r5b[0]["hints"] is None and r["hints"] is None
    else:
        H = fake_r5b[0]["hints"]
        assert H == json.load(open(hf)) and r["hints"]["n_used"] == sum(h["p"] >= 0.8 for h in H) >= 2
        assert "Detector v3 hints used" in r["limits"][-1]


def test_hints_file_is_checked(run_cli, demo, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"t": 1}')
    code, _, err = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--claim", "34.05,-118.30,r=2km", "--matcher", "r5b", "--hints", bad)
    assert code == 2 and "--hints" in err
    code, _, err = run_cli("verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--claim", "34.05,-118.30,r=2km", "--matcher", "reference", "--hints", bad)
    assert code == 2 and "reference" in err


def test_find_and_audit_scrub_get_hints(run_cli, demo, tmp_path, fake_r5b):
    code, out, _ = run_cli("find", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--area", "34.05,-118.30,r=3km", "--matcher", "r5b", "--fast", "--decoys", "5", "--json")
    r = json.loads(out)["result"]
    assert code == 11 and fake_r5b[0]["hints"] and r["hints"][0]["n_used"] >= 2
    fake_r5b.clear()
    code, out, _ = run_cli("audit", fx(demo, "call.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                           "--where", "34.05,-118.30,r=1km", "--matcher", "r5b", "--fast", "--decoys", "5",
                           "--scrub", tmp_path / "clean.wav", "--json")
    assert code == 11 and all(c["hints"] is not None for c in fake_r5b)
    assert len({json.dumps(c["hints"]) for c in fake_r5b}) == 2              # original and scrubbed audio
