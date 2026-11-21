"""lowband_gate.py - a speech-presence-gated low-band suppressor with comfort-noise substitution.

Ported unchanged in behaviour from the research branch mitig-v0 (mitig/lowband_gate.py, commit 3597d35). It is a
reference design tested in offline simulation, not any platform's code.

What it does, on the 16 kHz microphone (send) signal, frame by frame:
  1. Speech presence: WebRTC's GMM VAD (the `webrtcvad` binding of libwebrtc's VAD, 20 ms frames, aggressiveness
     VAD_MODE), with a 40 ms look-ahead and a 200 ms hangover (typical of codec DTX / VAD front ends).
  2. STFT (Hann 64 ms, hop 16 ms, 15.6 Hz bins, 75 % overlap).  Below FC (default 400 Hz, linear cross-fade to
     1.25 FC) and only while no speech is present, every bin is replaced by comfort noise: random phase, magnitude =
     a slow noise-floor estimate (minimum statistics of the 0.1 s-smoothed power over the last 10 s, upward rate
     limited to RISE_DB_S dB/s, then the median over 1/2 octave and at least +-3 bins).  The comfort noise therefore
     sounds like the room's steady low-frequency floor but carries no narrow-band line and no fast envelope: a Doppler
     line, a harmonic series or a pass's 20-60 s swell is not reproduced.
  3. While speech is present the band passes untouched (gain opens within one frame, closes with a 64 ms ramp after
     the hangover), so voiced speech, including its fundamental, is not filtered.
Only the modified low band is re-synthesised (y = x + ISTFT(changes)), so everything above 1.25 FC is bit-exact.

Latency of a streaming version: one 64 ms frame + 40 ms VAD look-ahead.  Cost: one 1024-point FFT per 16 ms.

usage: gate(x16k, fc=400.0) -> (y, info)
"""
import numpy as np

FS = 16000
N = 1024
HOP = 256
VAD_MODE = 2
HANG_S = 0.20
LOOKAHEAD_S = 0.04
RELEASE_S = 0.064
TAU_S = 0.10
MIN_WIN_S = 10.0
RISE_DB_S = 0.3
BIAS_DB = 2.5            # minimum statistics underestimates the mean power of a smoothed periodogram
_WIN = np.hanning(N + 1)[:N]
_NORM = (_WIN ** 2).sum() / HOP          # = 1.5 at 75 % overlap


def vad_frames(x, mode=VAD_MODE):
    """WebRTC VAD decision per 20 ms frame (bool)."""
    try:
        import webrtcvad
    except ImportError as e:
        from skyhum.errors import MissingData
        raise MissingData("the pause gate needs the WebRTC VAD: pip install webrtcvad-wheels") from e
    v = webrtcvad.Vad(mode)
    fl = FS // 50
    n = x.size // fl
    pcm = (np.clip(x[:n * fl], -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    out = np.zeros(n, bool)
    for i in range(n):
        out[i] = v.is_speech(pcm[2 * i * fl:2 * (i + 1) * fl], FS)
    return out


def speech_gain(vad20, n_frames):
    """Per-STFT-frame gate gain in [0, 1]: 1 = speech present (band passes), 0 = comfort noise."""
    t20 = (np.arange(vad20.size) + 0.5) * 0.02
    hang = int(round(HANG_S / 0.02))
    la = int(round(LOOKAHEAD_S / 0.02))
    v = vad20.astype(np.int32)
    k = np.convolve(v, np.ones(hang + la + 1, np.int32))[la:la + v.size] > 0  # hangover after, look-ahead before
    tc = (np.arange(n_frames) * HOP) / FS                                         # frame centre (x padded by N/2)
    lo = np.clip(np.searchsorted(t20, tc - N / 2 / FS), 0, v.size)
    hi = np.clip(np.searchsorted(t20, tc + N / 2 / FS), 0, v.size)
    cs = np.r_[0, np.cumsum(k)]
    sp = (cs[hi] - cs[lo]) > 0
    g = sp.astype(np.float64)
    rel = max(1, int(round(RELEASE_S * FS / HOP)))
    for i in range(1, g.size):                    # instant attack, linear release
        if g[i] < g[i - 1]:
            g[i] = max(g[i], g[i - 1] - 1.0 / rel)
    return g, sp


def _smooth_windows(freqs, octs=0.5, min_bins=3):
    out = []
    for i, f in enumerate(freqs):
        lo, hi = f * 2 ** (-octs / 2), f * 2 ** (octs / 2)
        sel = (freqs >= lo) & (freqs <= hi)
        sel[max(0, i - min_bins):i + min_bins + 1] = True
        out.append(np.flatnonzero(sel))
    return out


def _smooth(floor, wins):
    """Median over each bin's 1/2-octave (>= +-3 bin) window: a line narrower than half the window cannot raise it."""
    return np.stack([np.median(floor[:, w], axis=1) for w in wins], axis=1)


_CN_CAL = {}


def _cn_gain():
    """Power gain of random-phase overlap-add relative to a coherent signal with the same STFT magnitude."""
    if "g" not in _CN_CAL:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(FS * 8)
        X = _stft(x)
        Y = np.abs(X) * np.exp(2j * np.pi * rng.random(X.shape))
        y = _istft(Y, x.size)
        _CN_CAL["g"] = float(np.mean(x[N:-N] ** 2) / np.mean(y[N:-N] ** 2))
    return _CN_CAL["g"]


def _stft(x):
    xp = np.pad(x, (N // 2, N))
    nf = 1 + (xp.size - N) // HOP
    idx = np.arange(N)[None, :] + HOP * np.arange(nf)[:, None]
    return np.fft.rfft(xp[idx] * _WIN, axis=1)


def _istft(X, n):
    nf = X.shape[0]
    fr = np.fft.irfft(X, n=N, axis=1) * _WIN
    y = np.zeros(nf * HOP + N)
    for j in range(N // HOP):                                 # overlap-add in N/HOP interleaved strides
        blk = fr[j::N // HOP].reshape(-1)
        s = j * HOP
        y[s:s + blk.size] += blk
    return y[N // 2:N // 2 + n] / _NORM


def gate(x, fc=400.0, seed=0, vad=None, return_gain=False):
    """Apply the gate to x (float, 16 kHz; full scale 1.0).  Returns (y float32, info)."""
    x = np.asarray(x, np.float64)
    n = x.size
    vad20 = vad_frames(x) if vad is None else vad
    freqs = np.fft.rfftfreq(N, 1 / FS)
    K = int(np.searchsorted(freqs, 1.25 * fc)) + 1
    f = freqs[:K]
    c = np.clip((1.25 * fc - f) / (0.25 * fc), 0, 1)
    wins = _smooth_windows(np.maximum(f, 1.0))
    cn_amp = np.sqrt(_cn_gain())
    xp = np.pad(x, (N // 2, N))
    total = 1 + (xp.size - N) // HOP
    g, sp = speech_gain(vad20, total)
    idx0 = np.arange(N)
    # 1) low-bin power for the whole signal (K bins only: small)
    P = np.empty((total, K))
    for c0 in range(0, total, 16384):
        c1 = min(total, c0 + 16384)
        idx = idx0[None, :] + HOP * np.arange(c0, c1)[:, None]
        P[c0:c1] = np.abs(np.fft.rfft(xp[idx] * _WIN, axis=1)[:, :K]) ** 2
    # 2) smoothed power, causal running minimum over 1 s blocks x 10, rate-limited rise
    from scipy.signal import lfilter
    a = np.exp(-HOP / FS / TAU_S)
    ps = lfilter([1 - a], [1, -a], P, axis=0, zi=(a * P[:1]))[0]
    blk = int(round(FS / HOP))
    nb = -(-total // blk)
    pad = nb * blk - total
    bm = np.concatenate([ps, np.repeat(ps[-1:], pad, 0)]).reshape(nb, blk, K)
    run = np.minimum.accumulate(bm, axis=1).reshape(nb * blk, K)[:total]          # min within current block so far
    bmin = bm.min(axis=1)                                                          # full-block minima
    nblk = int(round(MIN_WIN_S))
    prev = np.full_like(bmin, np.inf)
    for b in range(1, nb):                                  # min over the previous nblk-1 full blocks
        prev[b] = bmin[max(0, b - nblk + 1):b].min(axis=0)
    m = np.minimum(run, np.repeat(prev, blk, 0)[:total])
    rise_lim = RISE_DB_S * HOP / FS
    lm = 10 * np.log10(m + 1e-30)
    fl = np.empty_like(lm)
    fl[0] = lm[0]
    for t in range(1, total):                               # rate-limited rise, instant fall
        fl[t] = np.minimum(lm[t], fl[t - 1] + rise_lim)
    floor = 10 ** (fl / 10)
    lvl = np.sqrt(_smooth(floor, wins) * 10 ** (BIAS_DB / 10)) * cn_amp
    # 3) changes only in frames with g < 1
    rng = np.random.default_rng(seed)
    y = x.copy()
    act = np.flatnonzero(g < 1.0)
    for c0 in range(0, act.size, 16384):
        fr_i = act[c0:c0 + 16384]
        idx = idx0[None, :] + HOP * fr_i[:, None]
        X = np.fft.rfft(xp[idx] * _WIN, axis=1)[:, :K]
        G = 1 - c[None, :] * (1 - g[fr_i, None])
        cn = lvl[fr_i] * np.exp(2j * np.pi * rng.random((fr_i.size, K)))
        D = np.zeros((fr_i.size, N // 2 + 1), complex)
        D[:, :K] = (G - 1) * X + (1 - G) * cn
        fr = np.fft.irfft(D, n=N, axis=1) * _WIN / _NORM
        for i, t in enumerate(fr_i):
            s = t * HOP - N // 2
            a0, b0 = max(0, s), min(n, s + N)
            if b0 > a0:
                y[a0:b0] += fr[i, a0 - s:b0 - s]
    info = dict(fc=fc, vad_frac=float(vad20.mean()), open_frac=float(np.mean(g > 0.5)), vad_mode=VAD_MODE,
                cn_band_db_median=float(np.median(10 * np.log10(np.sum(floor[:, (f >= 30) & (f <= fc)], 1) + 1e-30))))
    if return_gain:
        info["gain"] = g                    # per STFT frame (hop 16 ms, frame t centred at t * HOP / FS)
    return y.astype(np.float32), info
