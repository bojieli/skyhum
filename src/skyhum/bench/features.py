"""Derived, magnitude-only features of webcam audio: the release format for the public webcam hours.

The webcam audio behind the published webcam rows cannot be redistributed. Instead the project releases, per hour,
the numbers the research pipeline actually consumed, computed by this module from the audio. Anyone holding the
audio can recompute them with ``python -m skyhum.bench.features extract``; anyone holding the released files can
feed them to the locator and the loud channel with ``load`` / ``locator_spec`` / ``loud_frames``.

Four feature sets (DATA_CARD.md describes each):

  tonal_620   the locator's spectrogram exactly as it consumes it: afg ``compute_spectrogram`` at 4 kHz, 1 s Hann
              window, 0.2 s hop, whitened (dB above a 120 s per-bin background, minus the per-frame median), with
              transient frames gated to 0, clipped to [0, 15] dB (the locator's own clip), rows 29.8-620.1 Hz.
  tonal_400   the same array, rows 29.8-400 Hz only (owner-restricted set). ``locator_spec`` fills 400-620 Hz
              with 0 (= background) so the locator runs unchanged.
  loud_full   the loud channel's frame features exactly as the research detector stores them: third-octave band
              levels 25 Hz-6.3 kHz on 0.256 s frames, floor-tracked levels, band-group envelopes, spectral flatness,
              tonal crest below / above 400 Hz, amplitude-modulation prominence / frequency / harmonic.
  loud_env    owner-restricted loud set: two 2 s smoothed level curves (all bands 25 Hz-6.3 kHz, and 50-180 Hz),
              sampled every 0.512 s.

Every array is real-valued (float16 on disk). No phase, no complex values and no waveform samples are stored.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np

FORMAT = "skyhum-webcam-features/1"
FS_TONAL = 4000
TONAL = dict(win_s=1.0, hop_s=0.2, fmax=620.0, clip=(0.0, 15.0), f_lo=30.0)
LOUD_FRAME_S = 0.256
ENV_SMOOTH_S = 2.048          # 8 frames of 0.256 s
ENV_HOP_S = 0.512
K_BANDS = np.arange(-16, 9)   # third-octave index: fc = 1000 * 2^(k/3); 25 Hz ... 6.3 kHz
FC = 1000.0 * 2.0 ** (K_BANDS / 3.0)
GROUPS = {"low": (50.0, 180.0), "midlow": (180.0, 900.0), "mid": (180.0, 3600.0), "hi": (900.0, 3600.0)}

SETS = {
    "tonal_620": dict(kind="tonal", band_hz=(29.0, 621.0), window_s=1.0, hop_s=0.2, arrays=("W",),
                      what="locator spectrogram as consumed (whitened, gated, clipped to 0-15 dB)"),
    "tonal_400": dict(kind="tonal", band_hz=(29.0, 400.0), window_s=1.0, hop_s=0.2, arrays=("W",),
                      what="locator spectrogram rows up to 400 Hz (owner-restricted)"),
    "loud_full": dict(kind="loud", band_hz=(22.0, 7130.0), window_s=0.128, hop_s=LOUD_FRAME_S,
                      arrays=("L", "Lf", "gf_low", "gm_low", "gf_midlow", "gm_midlow", "gf_mid", "gm_mid",
                              "gf_hi", "gm_hi", "flat", "crest_lf", "crest_hf", "modp", "modf", "modh"),
                      what="loud-channel frame features as stored by the research detector"),
    "loud_env": dict(kind="loud", band_hz=(22.0, 7130.0), window_s=ENV_SMOOTH_S, hop_s=ENV_HOP_S,
                     arrays=("env_all", "env_low"),
                     what="2 s smoothed level curves, all bands and 50-180 Hz (owner-restricted)"),
}


# ----------------------------------------------------------------------------------------------------------------
# decode
# ----------------------------------------------------------------------------------------------------------------
def _ffmpeg():
    from skyhum.io.audio import ffmpeg_path
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is needed to decode audio")
    return ff


def source_rate(path):
    p = subprocess.run(["ffprobe" if not os.environ.get("SKYHUM_FFPROBE") else os.environ["SKYHUM_FFPROBE"],
                        "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int(p.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return 16000


def decode(path, fs):
    """Mono float32 at fs, decoded and resampled by ffmpeg (the research pipeline's decode)."""
    p = subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(int(fs)),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


# ----------------------------------------------------------------------------------------------------------------
# tonal channel
# ----------------------------------------------------------------------------------------------------------------
def tonal_spectrogram(x4k):
    """The locator's spectrogram of 4 kHz audio: W (dB above background, gated, clipped to 0-15), f, t."""
    from skyhum._vendor.afg import audio as afg_audio
    sp = afg_audio.compute_spectrogram(np.asarray(x4k, np.float64), FS_TONAL, win_s=TONAL["win_s"],
                                       hop_s=TONAL["hop_s"], fmax=TONAL["fmax"])
    f = sp["f"]
    df = f[1] - f[0]
    keep = f >= TONAL["f_lo"] - df          # one bin below 30 Hz: the locator interpolates at 30 Hz
    W = np.clip(sp["W"][keep], *TONAL["clip"]).astype(np.float32)
    return dict(W=W, f=f[keep], t=sp["t"], hop_s=float(sp["hop_s"]), win_s=float(sp["win_s"]),
                duration=float(sp["duration"]))


def tonal_sets(x4k):
    sp = tonal_spectrogram(x4k)
    out = {"tonal_620": sp}
    sel = sp["f"] <= SETS["tonal_400"]["band_hz"][1]
    out["tonal_400"] = dict(sp, W=sp["W"][sel], f=sp["f"][sel])
    return out


# ----------------------------------------------------------------------------------------------------------------
# loud channel (numpy port of the research detector's frame features; same chunking, windows and statistics)
# ----------------------------------------------------------------------------------------------------------------
def _hann(n):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float64)


def _power_stft(seg, n_fft, hop):
    """|STFT|^2 with a periodic Hann window and centred, reflect-padded frames (torch.stft(center=True))."""
    xp = np.pad(seg.astype(np.float64), n_fft // 2, mode="reflect")
    nfr = 1 + (xp.size - n_fft) // hop
    view = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:nfr]
    w = _hann(n_fft)
    out = np.empty((n_fft // 2 + 1, nfr), np.float64)
    for s in range(0, nfr, 4096):
        F = np.fft.rfft(view[s:s + 4096] * w, axis=1)
        out[:, s:s + 4096] = (F.real ** 2 + F.imag ** 2).T
    return out


def band_matrix(fs, n_fft):
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    df = f[1] - f[0]
    M = np.zeros((FC.size, f.size), np.float64)
    ok = np.zeros(FC.size, bool)
    for i, fc in enumerate(FC):
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        if hi > 0.45 * fs:
            continue
        ok[i] = True
        M[i] = np.clip(np.minimum(f + df / 2, hi) - np.maximum(f - df / 2, lo), 0, None) / df
    return M, ok


def _lower_median(a, axis=0):
    n = a.shape[axis]
    return np.take(np.partition(a, (n - 1) // 2, axis=axis), (n - 1) // 2, axis=axis)


def loud_frames(x, fs):
    """Third-octave levels and frame features on a 0.256 s grid (fs 16000 or 8000)."""
    n_fft = int(round(0.128 * fs))
    hop = n_fft // 4
    M, ok = band_matrix(fs, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    sel_hf = (f >= 400) & (f <= min(3400, 0.43 * fs))
    sel_lf = (f >= 40) & (f <= 400)
    L, flat, crl, crh, Bh = [], [], [], [], []
    chunk = hop * 8 * 400
    pad = n_fft
    for a in range(0, x.size, chunk):
        seg = x[max(0, a - pad): a + chunk + pad]
        S = _power_stft(seg, n_fft, hop)
        off = (a - max(0, a - pad)) // hop
        S = S[:, off: off + chunk // hop]
        nT = S.shape[1] // 8
        if nT == 0:
            break
        P = S[:, : nT * 8].reshape(S.shape[0], nT, 8).mean(2) + 1e-14
        L.append(10 * np.log10(M @ P + 1e-14))
        Bh.append(M @ S[:, : nT * 8])
        ph = P[sel_hf]
        flat.append(np.exp(np.log(ph).mean(0)) / ph.mean(0))
        for sel, dst in ((sel_lf, crl), (sel_hf, crh)):
            q = 10 * np.log10(P[sel])
            dst.append(q.max(0) - _lower_median(q, 0))
    L = np.concatenate(L, 1)
    L[~ok] = np.nan
    Bh = np.concatenate(Bh, 1)
    kq, half = 47, 23
    centres = np.arange(L.shape[1]) * 8 + 4

    def floor_of(v):
        vp = np.pad(v, half, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(vp, kq)[centres]
        return np.partition(win, 4, axis=1)[:, 4]
    Lf = np.full_like(L, np.nan)
    for b in range(L.shape[0]):
        if ok[b]:
            Lf[b] = 10 * np.log10(floor_of(Bh[b]) + 1e-14)
    out = dict(L=L, Lf=Lf)
    for g, (lo, hi) in GROUPS.items():
        sel = (FC >= lo) & (FC <= hi) & ok
        gsum = Bh[sel].sum(0)
        out["gf_" + g] = 10 * np.log10(floor_of(gsum) + 1e-14)
        out["gm_" + g] = 10 * np.log10(gsum[: L.shape[1] * 8].reshape(-1, 8).mean(1) + 1e-14)
    out.update(flat=np.concatenate(flat), crest_lf=np.concatenate(crl), crest_hf=np.concatenate(crh))
    out.update(_am_features(x, fs, L.shape[1]))
    return out


def _am_features(x, fs, nT):
    """Amplitude modulation of the 300-3400 Hz band at 9-40 Hz (rotor blade slap), per 0.256 s frame."""
    n1 = int(round(0.016 * fs))
    h1 = n1 // 2
    f = np.fft.rfftfreq(n1, 1.0 / fs)
    sel = (f >= 300) & (f <= min(3400, 0.43 * fs))
    env = []
    chunk = h1 * 20000
    for a in range(0, x.size, chunk):
        seg = x[max(0, a - n1): a + chunk + n1]
        S = _power_stft(seg, n1, h1)
        off = (a - max(0, a - n1)) // h1
        env.append(S[sel][:, off: off + chunk // h1].sum(0))
    env = np.concatenate(env)
    k = 251
    cs = np.concatenate([[0.0], np.cumsum(env)])
    i = np.arange(env.size)
    lo, hi = np.maximum(0, i - k // 2), np.minimum(env.size, i + k // 2 + 1)
    mean = (cs[hi] - cs[lo]) / (hi - lo)
    m = env / (mean + 1e-14) - 1.0
    Mod = _power_stft(m, 256, 32)
    fm = np.fft.rfftfreq(256, 0.008)
    a_, b_ = np.searchsorted(fm, 9.0), np.searchsorted(fm, 40.0)
    c_, d_ = np.searchsorted(fm, 5.0), np.searchsorted(fm, 60.0)
    band = Mod[a_:b_]
    ix = band.argmax(0)
    pk = band[ix, np.arange(band.shape[1])]
    med = _lower_median(Mod[c_:d_], 0) + 1e-12
    i2 = np.minimum(2 * (ix + a_), Mod.shape[0] - 2)
    cols = np.arange(Mod.shape[1])
    h2v = np.maximum(Mod[i2, cols], Mod[i2 + 1, cols])

    def fit(v):
        v = v[:nT]
        return np.pad(v, (0, nT - v.size), mode="edge") if v.size < nT else v
    return dict(modp=fit(10 * np.log10(pk / med)), modf=fit(fm[a_ + ix]), modh=fit(10 * np.log10(h2v / med + 1e-12)))


def loud_envelope(fr):
    """Owner-restricted loud set from the band levels: 2.048 s power-mean level curves every 0.512 s."""
    L = np.asarray(fr["L"], np.float64)
    ok = np.isfinite(L).all(1)
    p = 10 ** (L / 10)
    n = int(round(ENV_SMOOTH_S / LOUD_FRAME_S))
    step = int(round(ENV_HOP_S / LOUD_FRAME_S))
    low = ok & (FC >= GROUPS["low"][0]) & (FC <= GROUPS["low"][1])
    out = {}
    for name, sel in (("env_all", ok), ("env_low", low)):
        s = p[sel].sum(0)
        if s.size < n:
            out[name] = np.zeros(0)
            continue
        sm = np.convolve(s, np.ones(n) / n, mode="valid")
        out[name] = 10 * np.log10(sm[::step] + 1e-14)
    out["t0_s"] = (n * LOUD_FRAME_S) / 2.0
    return out


# ----------------------------------------------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------------------------------------------
def to_bytes(set_name, arrays, meta):
    """One feature file (npz, compressed): float16 arrays named in SETS plus axes and a JSON header."""
    spec = SETS[set_name]
    store = {}
    for k in spec["arrays"]:
        store[k] = np.asarray(arrays[k], np.float32).astype(np.float16)
    if spec["kind"] == "tonal":
        store["f"] = np.asarray(arrays["f"], np.float32)
        store["t"] = np.asarray(arrays["t"], np.float32)
    hdr = dict(format=FORMAT, set=set_name, band_hz=list(spec["band_hz"]), window_s=spec["window_s"],
               hop_s=spec["hop_s"], what=spec["what"], **meta)
    if set_name == "loud_full":
        hdr["fc_hz"] = [round(float(c), 3) for c in FC]
    if set_name == "loud_env":
        hdr["t0_s"] = float(arrays["t0_s"])
    if spec["kind"] == "tonal":
        hdr["duration_s"] = float(arrays["duration"])
    store["header"] = np.frombuffer(json.dumps(hdr, sort_keys=True).encode(), np.uint8)
    b = io.BytesIO()
    np.savez_compressed(b, **store)
    return b.getvalue()


def load(path_or_bytes):
    """Read and check one feature file. Returns dict(header=..., <arrays> as float32)."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    z = np.load(src, allow_pickle=False)
    hdr = json.loads(bytes(z["header"]).decode())
    if hdr.get("format") != FORMAT or hdr.get("set") not in SETS:
        raise ValueError("not a skyhum webcam feature file")
    spec = SETS[hdr["set"]]
    out = dict(header=hdr)
    for k in z.files:
        if k == "header":
            continue
        a = z[k]
        if np.iscomplexobj(a) or a.dtype.kind not in "f":
            raise ValueError(f"{k}: unexpected dtype {a.dtype}")
        out[k] = a.astype(np.float32)
    missing = [k for k in spec["arrays"] if k not in out]
    if missing:
        raise ValueError(f"missing arrays {missing}")
    if spec["kind"] == "tonal":
        lo, hi = spec["band_hz"]
        if out["f"].min() < lo or out["f"].max() > hi:
            raise ValueError("frequency axis outside the stated band")
    return out


def locator_spec(item, path):
    """Write the locator input (skyhum.match.locator / r5b ``--spec``) from a tonal feature file. Rows that the set
    does not carry (400-620 Hz in tonal_400) are filled with 0, the background level after the locator's clip."""
    hdr = item["header"]
    W, f = item["W"], item["f"]
    df = float(f[1] - f[0])
    n_full = int(round((TONAL["fmax"] - f[0]) / df)) + 1
    f_full = f[0] + df * np.arange(n_full)
    f_full = f_full[f_full <= TONAL["fmax"] + df]
    Wf = np.zeros((f_full.size, W.shape[1]), np.float16)
    Wf[: W.shape[0]] = W.astype(np.float16)
    np.savez(path, W=Wf, f=f_full.astype(np.float64), t=item["t"].astype(np.float64), hop_s=float(hdr["hop_s"]),
             duration=float(hdr["duration_s"]))
    return path


def loud_frames_from(item):
    """loud_full file -> the frame-feature dict the loud detector reads (float32; NaN where a band is absent)."""
    if item["header"]["set"] != "loud_full":
        raise ValueError("loud_frames_from needs a loud_full file")
    return {k: item[k] for k in SETS["loud_full"]["arrays"]}


def extract(audio_path, item_id, sets=tuple(SETS), meta=None):
    """Compute the requested sets for one audio file. Returns {set: npz bytes}."""
    meta = dict(meta or {}, id=item_id)
    out = {}
    if any(SETS[s]["kind"] == "tonal" for s in sets):
        ts = tonal_sets(decode(audio_path, FS_TONAL))
        for s in sets:
            if s in ts:
                out[s] = to_bytes(s, ts[s], dict(meta, fs=FS_TONAL))
    if any(SETS[s]["kind"] == "loud" for s in sets):
        fs = 16000 if source_rate(audio_path) >= 16000 else 8000
        fr = loud_frames(decode(audio_path, fs), fs)
        if "loud_full" in sets:
            out["loud_full"] = to_bytes("loud_full", fr, dict(meta, fs=fs))
        if "loud_env" in sets:
            out["loud_env"] = to_bytes("loud_env", loud_envelope(fr), dict(meta, fs=fs))
    return out


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m skyhum.bench.features", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="audio -> feature files <out>/<set>/<id>.npz")
    e.add_argument("--audio", required=True)
    e.add_argument("--id", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--sets", default=",".join(SETS))
    c = sub.add_parser("check", help="verify feature files (format, dtype, band)")
    c.add_argument("files", nargs="+")
    s = sub.add_parser("spec", help="tonal feature file -> locator --spec npz")
    s.add_argument("file")
    s.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        for name, b in extract(a.audio, a.id, tuple(a.sets.split(","))).items():
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            p = os.path.join(a.out, name, a.id + ".npz")
            with open(p, "wb") as fh:
                fh.write(b)
            print(f"{p}  {len(b)} bytes  sha256 {sha256(b)}")
    elif a.cmd == "check":
        for p in a.files:
            it = load(p)
            print(p, it["header"]["set"], {k: v.shape for k, v in it.items() if k != "header"})
    else:
        print(locator_spec(load(a.file), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
