"""Loud-event channel. The classifier test runs from raw A2O event dicts (numbers only, CC BY 4.0, no audio); the
detector and locator tests use synthetic audio and synthetic tracks."""
import json
import math
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def raw_events():
    return json.load(open(os.path.join(HERE, "data", "loud_events_a2o.json")))["events"]


def test_featurize_and_classifier_reproduce_original(raw_events):
    from skyhum.detect.loud_model import LoudClassifier
    from skyhum.loud.model import FEAT_NAMES, featurize
    z = np.load(os.path.join(HERE, "data", "loud_equiv.npz"))
    X = np.stack([featurize(e) for e in raw_events])
    assert X.shape == (400, 96) and len(FEAT_NAMES) == 96
    assert np.array_equal(np.isnan(X), np.isnan(z["X"]))
    assert np.array_equal(np.nan_to_num(X), np.nan_to_num(z["X"]))
    m = LoudClassifier()
    assert m.feat_names == FEAT_NAMES
    assert np.abs(m.predict_proba(X) - z["P"]).max() < 1e-9


def test_model_proba_and_evidence(raw_events):
    from skyhum.loud.model import CLASSES, Model, featurize, prep_event
    M = Model()
    ev = raw_events[:80]
    P = M.proba(ev)
    assert P.shape == (80, 4) and np.allclose(P.sum(1), 1.0)
    keep = [i for i, e in enumerate(ev) if prep_event(e) is not None]
    want = M.clf.predict_proba(np.stack([featurize(prep_event(ev[i])) for i in keep]))
    assert np.abs(P[keep] - want).max() == 0.0
    E = M.evidence(ev, P)
    assert len(E) == len(keep)
    for e in E:
        assert 0.0 <= e["p_air"] <= 1.0 and e["sig_t"] > 0 and e["sig_ltau"] > 0
        assert abs(sum(e["q"].values()) - 1) < 1e-6 and set(e["q"]) == set(CLASSES[1:])
    assert M.cal["thr"]["1.0"] > 0


def _pass_audio(fs=8000, dur=600.0, tc=300.0, tau=8.0, gain_db=18.0, seed=0):
    """White-noise background plus a broadband source whose power follows a Lorentzian around tc."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    bg = rng.standard_normal(t.size)
    src = rng.standard_normal(t.size) * np.sqrt(10 ** (gain_db / 10) / (1 + ((t - tc) / tau) ** 2))
    return (0.01 * (bg + src)).astype(np.float32)


def test_detector_finds_a_synthetic_pass():
    pytest.importorskip("torch")
    from skyhum.loud import features as F
    x = F.dither(_pass_audio())
    _, events = F.analyze(x, 8000, "cpu")
    best = max(events, key=lambda e: e["x_det"])
    assert abs(best["tc"] - 300.0) < 3.0 and 4.0 < best["tau"] < 16.0 and best["x_det"] > 6.0
    assert best["dop"] is not None and set(best["dop"]) >= {"zs", "zs_rev", "beta", "tau", "tc", "f0"}


def test_doppler_linear_symmetry():
    from skyhum.loud.features import C, doppler_linear
    f = doppler_linear(np.array([-1000.0, 1000.0]), 100.0, 60.0, 500.0, 0.0)
    assert f[0] > 100 > f[1] and abs(f[0] - 100 / (1 - 60 / C)) < 0.5 and abs(f[1] - 100 / (1 + 60 / C)) < 0.5


def test_single_event_points_at_the_site(tmp_path):
    pytest.importorskip("torch")
    from skyhum import synth
    from skyhum.io.adsb import write_tracks
    from skyhum.loud.locate import Box, Traffic, score, single_event
    from skyhum.loud.model import Model
    w = synth.make_world(seed=7)
    p = str(tmp_path / "tracks.npz")
    write_tracks(p, w["flights"], w["t0"], w["t1"])
    lat, lon = w["site"]
    box = Box(lat + 0.03, lon - 0.02, 0.06)
    tr = Traffic(p, box, 0.0, w["start"] - 700, w["start"] + w["dur_s"] + 700)
    tr.t0 = w["start"]
    assert not tr.empty
    se, sn = box.en(lat, lon)
    # the entry nearest the site with the shortest slant range: a pass the site really hears
    near = np.hypot(tr.px - se, tr.py - sn) < 60
    j = np.flatnonzero(near)[np.argmin(tr.d[near])]
    t_ev = float(tr.t_rx[j] - tr.t0)
    ev = dict(t=t_ev, sig_t=2.0, ltau=math.log(tr.d[j] / tr.v[j]), sig_ltau=0.4, p_air=0.95,
              q=dict(jet=0.05, prop=0.9, rotor=0.05), dop=None)
    cal = Model().cal
    cands = single_event(ev, tr, cal, 0.0, 5.0)
    assert cands
    err = min(math.hypot(e - se, n - sn) for e, n, _ in cands[0]["points"])
    assert err < 150.0
    res = score([ev], tr, cal, 0.0, 5.0, win=(0.0, w["dur_s"]))
    assert res["S"].shape == (box.nx * box.ny,) and res["S"].max() > 0
