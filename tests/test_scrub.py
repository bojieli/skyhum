import numpy as np
import pytest

from skyhum import synth
from skyhum.detect import tonal
from skyhum.detect.spectrogram import logspec

FS = 16000


@pytest.fixture(scope="module")
def clip():
    w = synth.make_world(seed=7)
    return synth.render(w, dur_s=150.0)


def ridge_cells(x):
    return int((logspec(x, FS).W > tonal.THR).sum())


def test_tone_scrub_removes_ridges(clip):
    from skyhum.scrub import tones
    y, info = tones.scrub(clip)
    assert y.shape == clip.shape and info["n_tracks"] >= 1
    assert ridge_cells(y) <= 0.5 * ridge_cells(clip)


def test_gate_leaves_speech_and_high_band(clip):
    pytest.importorskip("webrtcvad")
    from scipy.signal import butter, sosfiltfilt

    from skyhum.scrub import lowband_gate
    sp = synth.speechlike(clip.size, FS, duty=0.5)
    x = np.clip(clip + sp, -1, 1).astype(np.float32)
    y, info = lowband_gate.gate(x, fc=400.0, return_gain=True)
    assert 0.1 < info["open_frac"] < 0.9
    hp = butter(8, 600, "highpass", fs=FS, output="sos")
    assert np.allclose(sosfiltfilt(hp, y)[FS:-FS], sosfiltfilt(hp, x)[FS:-FS], atol=2e-3)   # above 1.25 fc: unchanged
    g = info["gain"]                                    # per 256-sample frame; frame t covers samples t*256-512 .. +512
    open_run = np.convolve(g >= 1.0, np.ones(5), "same") >= 5
    idx = np.flatnonzero(open_run[2:-2]) + 2
    samples = (idx[:, None] * 256 + np.arange(-128, 128)[None, :]).ravel()
    samples = samples[(samples >= 0) & (samples < x.size)]
    assert samples.size > FS * 10
    assert np.allclose(y[samples], x[samples], atol=1e-6)                          # speech passes untouched


def test_gate_removes_lines_in_pauses(clip):
    pytest.importorskip("webrtcvad")
    from skyhum.scrub import lowband_gate
    y, info = lowband_gate.gate(clip, fc=400.0)
    assert info["open_frac"] < 0.2
    assert ridge_cells(y) <= 0.5 * ridge_cells(clip)


def test_scrub_cli(run_cli, demo, tmp_path):
    from conftest import fx
    out = tmp_path / "o.wav"
    code, text, _ = run_cli("scrub", fx(demo, "call.wav"), out, "--method", "tones")
    assert code == 0 and out.exists() and "offline simulation" in text
    import soundfile as sf
    y, sr = sf.read(out)
    assert sr == FS and y.size == 240 * FS
