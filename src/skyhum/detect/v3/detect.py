"""Sliding detection over a recording of any length (research `sliding_detect`, NMS mode "multi").

240 s windows every 30 s (plus one window flush with the end). In each window, up to 4 local maxima of the logit
heatmap (3 x 3 neighbourhood, p_raw >= 0.05; the global maximum when none qualifies), ranked by p. Across windows a
detection is dropped when a stronger one lies within 10 s (any frequency) or within 30 s at a harmonically related
frequency (ratio within 4 % of k or 1/k, k <= 8).
"""
import importlib.util
import math
import os

import numpy as np

from skyhum.detect.v3.frontend import FS, HT, LOGF, NF, SEG_S, Frontend
from skyhum.detect.v3.frontend import N as N_WIN
from skyhum.detect.v3.net import load_net, weights_path

NAME = "v3"
DT_ANY, DT_HARM = 10.0, 30.0
HINT_PMIN = 0.8                 # the round-5b locator's --hint-pmin


def available():
    """PyTorch importable and the v3 weights present."""
    return importlib.util.find_spec("torch") is not None and os.path.isfile(weights_path())


def gate(v, d):
    """Soft aircraft-likeness: ~1 for v >= 35 m/s and d >= 150 m, ~0 for v < 25 m/s or d < 90 m. Same arithmetic
    as the research code; returns 0 where it would overflow."""
    a, b = -(math.log(v) - math.log(30)) / 0.06, -(math.log(d) - math.log(120)) / 0.1
    if a > 700 or b > 700:
        return 0.0
    return 1 / (1 + math.exp(a)) / (1 + math.exp(b))


def _freq(fi):
    return float(np.exp(np.interp((fi + 0.5) * 4, np.arange(NF), np.log(LOGF))))


def _harmonic(f1, f2, tol=0.04, kmax=8):
    r = max(f1, f2) / min(f1, f2)
    k = round(r)
    return 1 <= k <= kmax and abs(r / k - 1) <= tol


def window_outputs(net, fe, x, hop_s=30.0, batch=16):
    """Yields (start_s, logits (3, HF, HT)) for 240 s windows over x (1 kHz)."""
    import torch
    starts = np.arange(0, max(1, x.size - N_WIN + 1), int(hop_s * FS))
    if x.size > N_WIN and starts[-1] + N_WIN < x.size:          # cover the tail too
        starts = np.append(starts, x.size - N_WIN)
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            seg = np.stack([x[s:s + N_WIN] if s + N_WIN <= x.size else np.pad(x[s:], (0, s + N_WIN - x.size))
                            for s in starts[i:i + batch]])
            seg = seg / (seg.std(axis=1, keepdims=True) + 1e-9)
            out = net(fe(torch.as_tensor(seg, dtype=torch.float32, device=fe.dev))).float().cpu().numpy()
            for s, o in zip(starts[i:i + batch], out):
                yield s / FS, o


def _det(s0, o, fi, ti):
    p = float(1 / (1 + np.exp(-o[0, fi, ti])))
    v, dd = 50 * math.exp(o[1, fi, ti]), 1000 * math.exp(o[2, fi, ti])
    return dict(t=s0 + (ti + 0.5) / HT * SEG_S, f=_freq(fi), p_raw=p, start=s0, v=v, d=dd, p=p * gate(v, dd))


def sliding_detect(net, fe, x, hop_s=30.0, k_per_win=4, floor=0.05):
    """x: 1 kHz float32 audio. Returns detections (dicts t, f, p, p_raw, v, d, start), strongest p first."""
    dets = []
    lg = math.log(floor / (1 - floor))
    for s0, o in window_outputs(net, fe, x, hop_s):
        h = o[0]
        hp = np.pad(h, 1, constant_values=-np.inf)
        loc = np.ones_like(h, bool)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if a or b:
                    loc &= h >= hp[1 + a:1 + a + h.shape[0], 1 + b:1 + b + h.shape[1]]
        cand = np.argwhere(loc & (h >= lg))
        if not len(cand):
            fi, ti = np.unravel_index(np.argmax(h), h.shape)
            cand = np.array([[fi, ti]])
        ds = [_det(s0, o, fi, ti) for fi, ti in cand]
        ds.sort(key=lambda d: -d["p"])
        dets += ds[:k_per_win]
    dets.sort(key=lambda d: -d["p"])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            dt = abs(d["t"] - k["t"])
            if dt <= DT_ANY or (dt <= DT_HARM and _harmonic(d["f"], k["f"])):
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


class Detector:
    """detect(x, fs) -> detections. The network loads once; each call reseeds the front-end dither (seed), so the
    same audio always gives the same detections."""

    name = NAME

    def __init__(self, weights=None, device="cpu", seed=0, threads=None):
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        self.dev = device
        self.seed = seed
        self.net = load_net(weights, device)
        self.fe = Frontend(device, seed)

    def detect_1k(self, x1k):
        self.fe.reseed(self.seed)
        x1k = np.asarray(x1k, np.float32)
        return [{k: float(v) for k, v in d.items()} for d in sliding_detect(self.net, self.fe, x1k)]

    def detect(self, x, fs):
        """x: mono audio at fs (any rate). Resampled to 1 kHz with a polyphase filter, as in the research runs."""
        return self.detect_1k(to_1k(x, fs))


def to_1k(x, fs):
    from math import gcd

    from scipy.signal import resample_poly
    fs = int(fs)
    x = np.asarray(x)
    if fs == FS:
        return x.astype(np.float32)
    g = gcd(fs, FS)
    return resample_poly(x, FS // g, fs // g).astype(np.float32)




def load_1k(path, channel=0, offset_s=0.0, duration_s=None):
    """Decode a file at its own rate and resample straight to 1 kHz, as the research runs did (not through the
    16 kHz copy the other commands use). wav/flac/ogg/aiff through soundfile, anything else through ffmpeg."""

    from skyhum.errors import MissingData, UsageError
    from skyhum.io.audio import NATIVE, ffmpeg_path
    if not os.path.isfile(path):
        raise UsageError(f"audio file not found: {path}")
    x = sr = None
    if os.path.splitext(path)[1].lower() in NATIVE:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float64", always_2d=True)
            if channel >= data.shape[1]:
                raise UsageError(f"{path} has {data.shape[1]} channel(s); --channel {channel} does not exist")
            x = data[:, channel]
        except ImportError:
            x = None
    if x is None:
        if not ffmpeg_path():
            raise MissingData(f"decoding {path} needs ffmpeg on PATH")
        from skyhum._vendor.afg.audio import load_audio
        x, sr, nch = load_audio(path, channel=channel)
        if channel >= nch:
            raise UsageError(f"{path} has {nch} channel(s); --channel {channel} does not exist")
    x = to_1k(x, sr)
    a = int(round(offset_s * FS))
    b = x.size if duration_s is None else min(x.size, a + int(round(duration_s * FS)))
    return x[a:b]


_CACHE = {}


def shared(device="cpu", seed=0):
    """One Detector per process (the network loads once)."""
    key = (str(device), int(seed))
    if key not in _CACHE:
        _CACHE[key] = Detector(device=device, seed=seed)
    return _CACHE[key]


def plot(spec, dets, path, title="", pmin=0.5):
    """Spectrogram PNG (skyhum.detect.spectrogram.LogSpec) with v3 detections at (CPA time, frequency)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    ext = [0, spec.n_cols * spec.col_s, 0, spec.W.shape[0]]
    ax.imshow(spec.W, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=10)
    ticks = [f for f in (30, 50, 100, 200, 400, 600) if f <= spec.freqs[-1]]
    ax.set_yticks([np.log(f / spec.freqs[0]) / spec.log_step for f in ticks], [str(f) for f in ticks])
    for d in dets:
        if d["p"] < pmin:
            continue
        y = np.log(d["f"] / spec.freqs[0]) / spec.log_step
        ax.plot(d["t"], y, "o", mfc="none", ms=10, mew=1.2, color="cyan" if d["p"] >= HINT_PMIN else "grey")
    ax.set_xlabel("seconds from start")
    ax.set_ylabel("Hz")
    ax.set_title(title or f"whitened spectrogram (circles: detector v3, p >= {pmin:g}; cyan p >= {HINT_PMIN:g})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
