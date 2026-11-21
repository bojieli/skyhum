"""tones.py - Doppler-tone tracking scrubber, plus the re-encode and high-pass baselines.

Ported unchanged in behaviour from the research branch mitig-v0 (mitig/scrub.py, commit 3597d35), where it was
tested in offline simulation (docs/limits.md).

  scrub      Doppler-tone tracker + notch.  STFT 0.512 s Hann, hop 0.128 s (1.95 Hz bins), 20-2000 Hz.  Per-bin
             background = 30 s block median (as afg's whitening); a bin is a line candidate where it stands
             THR_DB above that background AND above the +-15-bin spectral median, and is a local spectral maximum;
             or (steady lines: hovering rotor, mains) the raw level is THR_DB + 3 above its spectral median.
             Candidates are linked into tracks (8-connected after +-1 bin dilation); tracks lasting >= MIN_DUR_S
             are kept (fly-over lines last tens of seconds; voiced speech harmonics glide and break within ~1 s).
             Kept track cells, widened by +-NOTCH_BINS, are replaced by random-phase noise at the local floor (the lower
             of the time background and the spectral median).
             Applied everywhere, under speech too.  Nothing outside the tracked lines changes.
  reenc      re-encoding alone: AAC-LC 96 kbit/s at 44.1 kHz (a typical upload transcode), decoded back.
  hpf400     the naive "cut the low band" edit: 8th-order Butterworth high-pass at 400 Hz.

usage: scrub(x16k) -> (y, info); reencode(x16k) -> y; hpf(x16k, 400) -> y   (x16k: float, 16 kHz mono)
"""
import os
import subprocess

import numpy as np

FS = 16000
N = 8192
HOP = 2048
FMIN, FMAX = 20.0, 2000.0
THR_DB = 5.0
SPEC_MED_BINS = 15
MIN_DUR_S = 2.0
NOTCH_BINS = 2
BG_BLOCK_S = 30.0
_WIN = np.hanning(N + 1)[:N]
_NORM = (_WIN ** 2).sum() / HOP
FFMPEG = os.environ.get("SKYHUM_FFMPEG", "ffmpeg")


def _frames(xp, t0, t1, K0, K1):
    idx = np.arange(N)[None, :] + HOP * np.arange(t0, t1)[:, None]
    return np.fft.rfft(xp[idx] * _WIN, axis=1)[:, K0:K1].astype(np.complex64)


def _bg_db(L, blk):
    """Per-bin running median background (blocks of blk frames, half overlap, linear interpolation)."""
    n = L.shape[0]
    starts = list(range(0, max(1, n - blk // 2), blk // 2)) or [0]
    cen, med = [], []
    for s in starts:
        e = min(n, s + blk)
        cen.append(0.5 * (s + e - 1))
        med.append(np.median(L[s:e], axis=0))
    cen, med = np.array(cen), np.array(med)
    fr = np.arange(n)
    B = np.empty_like(L)
    for k in range(L.shape[1]):
        B[:, k] = np.interp(fr, cen, med[:, k])
    return B


def scrub(x, thr_db=THR_DB, min_dur_s=MIN_DUR_S, notch_bins=NOTCH_BINS, seed=0):
    from scipy import ndimage
    x = np.asarray(x, np.float64)
    n = x.size
    freqs = np.fft.rfftfreq(N, 1 / FS)
    K0, K1 = int(np.searchsorted(freqs, FMIN)), int(np.searchsorted(freqs, FMAX)) + 1
    xp = np.pad(x, (N // 2, N))
    total = 1 + (xp.size - N) // HOP
    X = np.concatenate([_frames(xp, a, min(total, a + 4096), K0, K1) for a in range(0, total, 4096)])
    L = (10 * np.log10(np.abs(X).astype(np.float32) ** 2 + 1e-20)).astype(np.float32)
    B = _bg_db(L, int(round(BG_BLOCK_S * FS / HOP)))
    E = L - B
    Es = E - ndimage.median_filter(E, size=(1, 2 * SPEC_MED_BINS + 1), mode="nearest")
    mx = ndimage.maximum_filter(E, size=(1, 3), mode="nearest")
    Lmed = ndimage.median_filter(L, size=(1, 2 * SPEC_MED_BINS + 1), mode="nearest")
    Ls = L - Lmed
    lmx = ndimage.maximum_filter(L, size=(1, 3), mode="nearest")
    # moving line: above its own 30 s background and above its spectral neighbourhood; steady line (hover, mains):
    # a spectral peak of the raw level
    cand = ((E > thr_db) & (Es > thr_db) & (E >= mx)) | ((Ls > thr_db + 3) & (L >= lmx))
    lab, nl = ndimage.label(ndimage.binary_dilation(cand, structure=np.ones((1, 3), bool)),
                            structure=np.ones((3, 3), bool))
    keep = np.zeros(nl + 1, bool)
    if nl:
        sl = ndimage.find_objects(lab)
        min_fr = int(round(min_dur_s * FS / HOP))
        for i, s in enumerate(sl):
            if s is not None and (s[0].stop - s[0].start) >= min_fr:
                keep[i + 1] = True
    M = keep[lab] & cand
    if notch_bins:
        M = ndimage.binary_dilation(M, structure=np.ones((1, 2 * notch_bins + 1), bool))
    # random-phase noise at the background level in the masked cells; synthesise only the change
    rng = np.random.default_rng(seed)
    cn_gain = _cn_gain()
    rows = np.flatnonzero(M.any(axis=1))
    y = x.copy()
    for r0 in range(0, rows.size, 4096):
        rr = rows[r0:r0 + 4096]
        D = np.zeros((rr.size, N // 2 + 1), complex)
        m = M[rr]
        amp = np.sqrt(10 ** (np.minimum(B[rr], Lmed[rr]) / 10) * cn_gain)   # local floor, line excluded
        new = amp * np.exp(2j * np.pi * rng.random(m.shape))
        D[:, K0:K1] = np.where(m, new - X[rr], 0)
        fr = np.fft.irfft(D, n=N, axis=1) * _WIN / _NORM
        for i, t in enumerate(rr):
            s = t * HOP - N // 2
            a0, b0 = max(0, s), min(n, s + N)
            if b0 > a0:
                y[a0:b0] += fr[i, a0 - s:b0 - s]
    info = dict(n_tracks=int(keep.sum()), masked_cell_frac=float(M.mean()),
                masked_frame_frac=float(M.any(axis=1).mean()),
                masked_frac_below_620=float(M[:, freqs[K0:K1] <= 620].mean()))
    return y.astype(np.float32), info


_CAL = {}


def _cn_gain():
    if "g" not in _CAL:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(FS * 30)
        xp = np.pad(x, (N // 2, N))
        total = 1 + (xp.size - N) // HOP
        X = _frames(xp, 0, total, 0, N // 2 + 1)
        Y = np.abs(X) * np.exp(2j * np.pi * rng.random(X.shape))
        fr = np.fft.irfft(Y, n=N, axis=1) * _WIN / _NORM
        y = np.zeros(x.size)
        for t in range(total):
            s = t * HOP - N // 2
            a0, b0 = max(0, s), min(x.size, s + N)
            if b0 > a0:
                y[a0:b0] += fr[t, a0 - s:b0 - s]
        _CAL["g"] = float(np.mean(x[N:-N] ** 2) / np.mean(y[N:-N] ** 2))
    return _CAL["g"]


def reencode(x, codec="aac", br="96k", sr=44100):
    """Upload-style transcode: f32 16 kHz -> AAC-LC (ffmpeg native) at sr -> decoded back to 16 kHz."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mitig_re_") as td:
        enc = os.path.join(td, "a.m4a")
        subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(FS), "-ac", "1",
                        "-i", "pipe:0", "-ar", str(sr), "-c:a", codec, "-b:a", br, enc],
                       input=np.ascontiguousarray(x, np.float32).tobytes(), check=True, capture_output=True)
        out = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", enc, "-ac", "1", "-ar", str(FS), "-f", "f32le",
                              "pipe:1"], check=True, capture_output=True).stdout
    y = np.frombuffer(out, np.float32).copy()
    # AAC adds encoder priming delay: align by cross-correlation of the first 60 s (|lag| <= 0.2 s)
    m = min(x.size, y.size, 60 * FS)
    L = int(0.2 * FS)
    a = np.asarray(x[L:m - L], np.float64)
    from scipy.signal import fftconvolve
    cc = fftconvolve(np.asarray(y[:m], np.float64), a[::-1], "valid")   # lags 0..2L, d = k - L
    lag = int(np.argmax(np.abs(cc))) - L
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.r_[np.zeros(-lag, np.float32), y]
    y = y[:x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))
    return y.astype(np.float32)


def hpf(x, fc=400.0, order=8):
    from scipy.signal import butter, sosfilt
    sos = butter(order, fc, "highpass", fs=FS, output="sos")
    return sosfilt(sos, np.asarray(x, np.float64)).astype(np.float32)
