"""Whitened log-frequency spectrogram used by the detector and the matcher.

Audio is reduced to 2 kHz, cut into 2 s Hann frames every 1 s (column c is centred on second c of the
recording), mapped to log-spaced rows (1 % apart, 30-600 Hz), whitened per row against a running median,
scaled by a robust noise level and clipped to 0..15. A propeller's Doppler slide is a ridge in this image.
"""
import numpy as np

from skyhum.io.audio import resample

FS_SPEC = 2000
WIN_S, HOP_S = 2.0, 1.0
F_LO, F_HI = 30.0, 600.0
LOG_STEP = 0.01
BG_BLOCK_S = 300
CLIP = (0.0, 15.0)


class LogSpec:
    def __init__(self, W, freqs, raw_db):
        self.W = W                  # (rows, cols) whitened, clipped
        self.freqs = freqs          # (rows,) Hz
        self.raw_db = raw_db        # (rows, cols) un-whitened level, for plots
        self.log_step = LOG_STEP
        self.col_s = HOP_S

    @property
    def n_cols(self):
        return self.W.shape[1]

    def row_of(self, f_hz):
        return int(round(np.log(f_hz / F_LO) / LOG_STEP))


def logspec(x, fs):
    y = resample(np.asarray(x, np.float32), fs, FS_SPEC).astype(np.float64)
    n, hop = int(WIN_S * FS_SPEC), int(HOP_S * FS_SPEC)
    yp = np.pad(y, (n // 2, n // 2))
    nf = max(1, 1 + (yp.size - n) // hop)
    win = np.hanning(n + 1)[:n]
    P = np.empty((nf, n // 2 + 1))
    for a in range(0, nf, 1024):
        b = min(nf, a + 1024)
        idx = np.arange(n)[None, :] + hop * np.arange(a, b)[:, None]
        P[a:b] = np.abs(np.fft.rfft(yp[idx] * win, axis=1)) ** 2
    L = 10 * np.log10(P + 1e-12)
    lin = np.fft.rfftfreq(n, 1 / FS_SPEC)
    k = np.arange(int(np.floor(np.log(F_HI / F_LO) / LOG_STEP)) + 1)
    freqs = F_LO * np.exp(LOG_STEP * k)
    pos = np.interp(freqs, lin, np.arange(lin.size))
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    R = (1 - w)[:, None] * L[:, i0].T + w[:, None] * L[:, np.minimum(i0 + 1, lin.size - 1)].T   # (rows, cols)
    E = R - _running_median(R, BG_BLOCK_S)
    sig = 1.4826 * np.median(np.abs(E - np.median(E))) + 1e-9
    W = np.clip(E / sig, *CLIP)
    W = np.maximum(W, np.maximum(np.roll(W, 1, 0), np.roll(W, -1, 0)))         # max-pool +-1 row
    return LogSpec(W.astype(np.float32), freqs, R.astype(np.float32))


def _running_median(R, blk):
    n = R.shape[1]
    if n <= blk:
        return np.median(R, axis=1, keepdims=True)
    starts = list(range(0, max(1, n - blk // 2), blk // 2))
    cen = np.array([0.5 * (s + min(n, s + blk) - 1) for s in starts])
    med = np.stack([np.median(R[:, s:min(n, s + blk)], axis=1) for s in starts], axis=1)
    out = np.empty_like(R)
    cols = np.arange(n)
    for r in range(R.shape[0]):
        out[r] = np.interp(cols, cen, med[r])
    return out
