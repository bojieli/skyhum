"""Webcam feature release format (skyhum.bench.features), on synthetic audio only.

Each set must be magnitude-only (no complex values, no phase, no waveform), band-limited to its stated band, on the
documented window / hop, and loadable by the locator and loud-channel loaders.
"""
import io
import json

import numpy as np
import pytest

from skyhum.bench import features as F
from skyhum.io.audio import resample

FS = 16000
DUR = 150.0


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(3)
    t = np.arange(int(DUR * FS)) / FS
    f0 = 110.0 * (1 + 0.02 * np.tanh((t - 70) / 15))               # Doppler-like falling tone
    x = 0.05 * np.sin(2 * np.pi * np.cumsum(f0) / FS)
    x += 0.02 * np.sin(2 * np.pi * 3 * np.cumsum(f0) / FS)
    syl = (np.sin(2 * np.pi * 4.0 * t) > 0.3) * ((t > 20) & (t < 40))  # 4 Hz bursty voice-band component
    x += 0.05 * syl * np.sin(2 * np.pi * 700 * t) * np.sin(2 * np.pi * 150 * t)
    x += 0.01 * rng.standard_normal(t.size)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def files(synth):
    x4k = resample(synth, FS, F.FS_TONAL)
    ts = F.tonal_sets(x4k)
    fr = F.loud_frames(synth.astype(np.float64), FS)
    meta = dict(id="synthetic", camera="synthetic", condition="original", start_utc="2026-01-01T00:00:00Z")
    out = {s: F.to_bytes(s, ts[s], dict(meta, fs=F.FS_TONAL)) for s in ("tonal_620", "tonal_400")}
    out["loud_full"] = F.to_bytes("loud_full", fr, dict(meta, fs=FS))
    out["loud_env"] = F.to_bytes("loud_env", F.loud_envelope(fr), dict(meta, fs=FS))
    return out


def _raw(b):
    return np.load(io.BytesIO(b), allow_pickle=False)


@pytest.mark.parametrize("name", list(F.SETS))
def test_no_complex_no_phase_float16(files, name):
    z = _raw(files[name])
    spec = F.SETS[name]
    for k in z.files:
        a = z[k]
        assert not np.iscomplexobj(a), k
        assert "phase" not in k.lower() and "angle" not in k.lower()
        if k == "header":
            assert a.dtype == np.uint8
        elif k in spec["arrays"]:
            assert a.dtype == np.float16, (k, a.dtype)
        else:
            assert k in ("f", "t") and a.dtype == np.float32
    assert set(spec["arrays"]) <= set(z.files)
    # nothing beyond the documented arrays and axes
    assert set(z.files) <= set(spec["arrays"]) | {"f", "t", "header"}


@pytest.mark.parametrize("name,fmax", [("tonal_620", 621.0), ("tonal_400", 400.0)])
def test_tonal_band_window_hop(files, name, fmax):
    it = F.load(files[name])
    hdr = it["header"]
    assert hdr["band_hz"][1] == fmax and hdr["window_s"] == 1.0 and hdr["hop_s"] == 0.2
    f, t, W = it["f"], it["t"], it["W"]
    assert f.max() <= fmax and f.min() >= 29.0
    assert W.shape == (f.size, t.size)
    assert np.allclose(np.diff(t), 0.2, atol=1e-3)
    assert np.allclose(np.diff(f), F.FS_TONAL / 16384, atol=1e-4)          # 1 s Hann, zero-padded x4
    # magnitude only: dB above background, clipped to the locator's range; nothing negative
    assert W.min() >= 0.0 and W.max() <= 15.0
    # time resolution is the 1 s window: far coarser than a syllable; no sample-rate data
    assert t.size <= DUR / 0.2 + 1


def test_tonal_400_is_rows_of_620(files):
    a, b = F.load(files["tonal_620"]), F.load(files["tonal_400"])
    n = b["f"].size
    assert np.array_equal(a["f"][:n], b["f"]) and np.array_equal(a["W"][:n], b["W"])
    assert a["f"][n:].min() > 400.0


def test_tonal_sees_the_tone(files):
    it = F.load(files["tonal_620"])
    row = int(np.argmin(np.abs(it["f"] - 110.0)))
    assert it["W"][row - 3: row + 4].max(0).mean() > 3.0     # the 110 Hz line stands above background


def test_loud_full_bands_and_frames(files):
    it = F.load(files["loud_full"])
    hdr = it["header"]
    assert hdr["hop_s"] == F.LOUD_FRAME_S and hdr["window_s"] == 0.128
    fc = np.array(hdr["fc_hz"])
    assert fc.size == 25 and fc.min() >= 24.0 and fc.max() <= 6400.0
    lo, hi = hdr["band_hz"]
    assert fc.min() / 2 ** (1 / 6) >= lo and fc.max() * 2 ** (1 / 6) <= hi
    nT = it["L"].shape[1]
    assert it["L"].shape == it["Lf"].shape == (25, nT)
    assert abs(nT - DUR / F.LOUD_FRAME_S) <= 2
    for k in F.SETS["loud_full"]["arrays"][2:]:
        assert it[k].shape == (nT,), k
    # per second: (2 x 25 band levels + 14 scalars) x 3.9 frames = about 250 numbers, 64x fewer than 16 kHz samples
    per_s = sum(it[k].size for k in F.SETS["loud_full"]["arrays"]) / DUR
    assert per_s < 260


def test_loud_env_is_smooth_and_sparse(files):
    it = F.load(files["loud_env"])
    hdr = it["header"]
    assert hdr["hop_s"] == F.ENV_HOP_S and hdr["window_s"] == F.ENV_SMOOTH_S
    for k in ("env_all", "env_low"):
        e = it[k]
        assert e.ndim == 1 and abs(e.size - (DUR - F.ENV_SMOOTH_S) / F.ENV_HOP_S) <= 2
    assert (it["env_all"].size + it["env_low"].size) / DUR < 4.0      # 2 curves at ~2 values per second


def test_load_rejects_complex_and_foreign(files):
    z = dict(_raw(files["tonal_620"]))
    z["W"] = z["W"].astype(np.complex64)
    b = io.BytesIO()
    np.savez(b, **z)
    with pytest.raises(ValueError):
        F.load(b.getvalue())
    hdr = json.loads(bytes(_raw(files["tonal_620"])["header"]).decode())
    z = dict(_raw(files["tonal_620"]))
    z["header"] = np.frombuffer(json.dumps(dict(hdr, format="other")).encode(), np.uint8)
    b = io.BytesIO()
    np.savez(b, **z)
    with pytest.raises(ValueError):
        F.load(b.getvalue())


def test_locator_spec_from_features(files, tmp_path):
    from skyhum.match import r5b
    full = F.load(files["tonal_620"])
    part = F.load(files["tonal_400"])
    a = r5b.load_spec(F.locator_spec(full, str(tmp_path / "a.npz")))
    b = r5b.load_spec(F.locator_spec(part, str(tmp_path / "b.npz")))
    assert a["W"].shape == b["W"].shape == full["W"].shape
    n = part["f"].size
    assert np.array_equal(a["W"][:n], b["W"][:n])
    assert not b["W"][n:].any()                       # 400-620 Hz: background (0) in the restricted set
    assert a["hop"] == 0.2 and a["f"].max() <= 621.0


def test_loud_frames_from(files):
    fr = F.loud_frames_from(F.load(files["loud_full"]))
    assert set(fr) == set(F.SETS["loud_full"]["arrays"])
    env = F.loud_envelope(fr)
    direct = F.load(files["loud_env"])
    assert np.allclose(env["env_all"], direct["env_all"], atol=0.05)
    with pytest.raises(ValueError):
        F.loud_frames_from(F.load(files["loud_env"]))


def test_r5b_session_from_features(files, tmp_path):
    from skyhum.score.r5b_rule import Session
    p = tmp_path / "t.npz"
    p.write_bytes(files["tonal_400"])
    s = Session.from_features(str(p), str(tmp_path))
    assert abs(s.dur - DUR) < 0.01
    z = np.load(s.path)
    assert z["W"].dtype == np.float16 and z["f"].max() <= 621.0
    q = tmp_path / "l.npz"
    q.write_bytes(files["loud_env"])
    with pytest.raises(ValueError):
        Session.from_features(str(q), str(tmp_path))
