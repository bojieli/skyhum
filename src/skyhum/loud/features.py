"""Blind loud-event detector and per-event measurements at full bandwidth (8 or 16 kHz audio). No ADS-B here.

Ported from the research loud-event channel (features.py), inference only. analyze(x, fs) returns the frame
features and the candidate events; events_from_file(path) decodes a file the way the research runs did (ffmpeg,
mono, 16 kHz when the file has it, else 8 kHz, a -120 dB seeded dither against digital silence) and analyses it.
PyTorch (CPU) is used for the STFTs and the de-Doppler correlation.

Envelope model. A source of constant power passing at speed v and slant CPA distance d gives received power
proportional to 1 / (d^2 + v^2 t^2): a Lorentzian in time with half-width tau = d / v (half-power full width 2 d / v).

High-frequency de-Doppler. All narrow lines of one aircraft (fan tones, buzz-saw shaft orders, upper prop / rotor
harmonics) follow the same curve in log frequency. For each (beta = v / c, tau = d / v) the curve is one template;
its 2-D cross-correlation with the whitened log-frequency spectrogram gives the mean brightness along the curve for
every (source frequency, CPA time) at once. The fit therefore returns the CPA reception time, v and d from audio alone.
Nulls: the time-reversed curve (a rising tone no passing aircraft can make) and a flat line (steady tone).
"""
import shutil
import subprocess

import numpy as np
import torch
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

C = 343.0             # speed of sound (m/s)
DT = 0.256            # s, frame step of the band-level series (8 STFT hops of 32 ms)

K_BANDS = np.arange(-16, 9)                     # third-octave index: fc = 1000 * 2^(k/3); 25 Hz ... 6.3 kHz
FC = 1000.0 * 2.0 ** (K_BANDS / 3.0)
GROUPS = {"low": (50.0, 180.0), "midlow": (180.0, 900.0), "mid": (180.0, 3600.0), "hi": (900.0, 3600.0)}
DET = dict(smooth_s=4.0, height=4.0, prom=3.0, dist_s=6.0, low_height=6.0, low_prom=4.0, max_per_h=100, dop_min=5.0)


def doppler_linear(t, f0, v, d, tc, c=C):
    """Received frequency of a source f0 on a straight line at speed v, CPA distance d, CPA reception time tc
    (afg.physics.doppler_linear, MIT, AlanBacker/audio-flyover-geolocation)."""
    t = np.asarray(t, dtype=float)
    tau_c = tc - d / c
    T = t - tau_c
    c2, v2 = c * c, v * v
    s = (c2 * T - np.sqrt(c2 * v2 * T * T + d * d * (c2 - v2))) / (c2 - v2)
    R = c * (T - s)
    rdot = v2 * s / np.maximum(R, 1e-9)
    return f0 / (1.0 + rdot / c)


def band_matrix(fs, n_fft):
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    df = f[1] - f[0]
    M = np.zeros((FC.size, f.size), np.float32)
    ok = np.zeros(FC.size, bool)
    for i, fc in enumerate(FC):
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        if hi > 0.45 * fs:
            continue
        ok[i] = True
        # fractional overlap of each FFT bin [f - df/2, f + df/2] with the band
        M[i] = np.clip(np.minimum(f + df / 2, hi) - np.maximum(f - df / 2, lo), 0, None) / df
    return M, ok


def frame_features(x, fs, dev):
    """Band levels and frame features on a 0.256 s grid."""
    n_fft = int(round(0.128 * fs))
    hop = n_fft // 4
    M, ok = band_matrix(fs, n_fft)
    Mt = torch.as_tensor(M, device=dev)
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    win = torch.hann_window(n_fft, device=dev)
    sel_hf = torch.as_tensor((f >= 400) & (f <= min(3400, 0.43 * fs)), device=dev)
    sel_lf = torch.as_tensor((f >= 40) & (f <= 400), device=dev)
    xt = torch.as_tensor(x, device=dev)
    L, flat, crl, crh, Bh = [], [], [], [], []
    chunk = hop * 8 * 400                        # ~102 s
    pad = n_fft
    for a in range(0, xt.numel(), chunk):
        seg = xt[max(0, a - pad): a + chunk + pad]
        S = torch.stft(seg, n_fft, hop, window=win, center=True, return_complex=True).abs() ** 2
        off = (a - max(0, a - pad)) // hop
        S = S[:, off: off + chunk // hop]
        nT = S.shape[1] // 8
        if nT == 0:
            break
        P = S[:, : nT * 8].reshape(S.shape[0], nT, 8).mean(2) + 1e-14
        L.append(10 * torch.log10(Mt @ P + 1e-14))
        Bh.append(Mt @ S[:, : nT * 8])
        ph = P[sel_hf]
        flat.append(torch.exp(torch.log(ph).mean(0)) / ph.mean(0))
        for sel, dst in ((sel_lf, crl), (sel_hf, crh)):
            q = 10 * torch.log10(P[sel])
            dst.append(q.max(0).values - q.median(0).values)
    L = torch.cat(L, 1).cpu().numpy()
    L[~ok] = np.nan
    # Floor-tracked levels: 10th percentile of the 32 ms band powers over +-0.75 s. A continuous source (aircraft)
    # keeps its level; speech, radio chatter and other gated sounds drop to the level between their bursts.
    Bh = torch.cat(Bh, 1)
    kq, half = 47, 23
    Lf = np.full_like(L, np.nan)
    gf = {}
    centres = torch.arange(L.shape[1], device=dev) * 8 + 4

    def floor_of(v):
        vp = torch.nn.functional.pad(v[None, None], (half, half), mode="replicate")[0, 0]
        return vp.unfold(0, kq, 1).kthvalue(5, dim=1).values[centres]
    for b in range(L.shape[0]):
        if ok[b]:
            Lf[b] = (10 * torch.log10(floor_of(Bh[b]) + 1e-14)).cpu().numpy()
    for g, (lo, hi) in GROUPS.items():
        sel = torch.as_tensor((FC >= lo) & (FC <= hi) & ok, device=dev)
        gsum = Bh[sel].sum(0)
        gf["gf_" + g] = (10 * torch.log10(floor_of(gsum) + 1e-14)).cpu().numpy()
        gf["gm_" + g] = (10 * torch.log10(gsum[: L.shape[1] * 8].reshape(-1, 8).mean(1) + 1e-14)).cpu().numpy()
    out = dict(L=L, Lf=Lf, **gf, flat=torch.cat(flat).cpu().numpy(), crest_lf=torch.cat(crl).cpu().numpy(),
               crest_hf=torch.cat(crh).cpu().numpy())
    out.update(am_features(xt, fs, dev, L.shape[1]))
    return out


def am_features(xt, fs, dev, nT):
    """Amplitude modulation of the 300-3400 Hz band (blade slap shows up at the main-rotor blade-pass rate even when
    the fundamental itself, 11-27 Hz, is below the microphone / codec band)."""
    n1 = int(round(0.016 * fs))
    h1 = n1 // 2                                 # 8 ms -> 125 Hz envelope rate
    f = np.fft.rfftfreq(n1, 1.0 / fs)
    sel = torch.as_tensor((f >= 300) & (f <= min(3400, 0.43 * fs)), device=dev)
    win = torch.hann_window(n1, device=dev)
    env = []
    chunk = h1 * 20000
    for a in range(0, xt.numel(), chunk):
        seg = xt[max(0, a - n1): a + chunk + n1]
        S = torch.stft(seg, n1, h1, window=win, center=True, return_complex=True).abs() ** 2
        off = (a - max(0, a - n1)) // h1
        env.append(S[sel][:, off: off + chunk // h1].sum(0))
    env = torch.cat(env)
    k = 251
    mean = torch.nn.functional.avg_pool1d(env[None, None], k, 1, k // 2, count_include_pad=False)[0, 0]
    m = env / (mean + 1e-14) - 1.0
    n2, h2 = 256, 32                             # 2.05 s windows, 0.256 s hop
    Mod = torch.stft(m, n2, h2, window=torch.hann_window(n2, device=dev), center=True, return_complex=True).abs() ** 2
    fm = np.fft.rfftfreq(n2, 0.008)
    a_, b_ = np.searchsorted(fm, 9.0), np.searchsorted(fm, 40.0)
    c_, d_ = np.searchsorted(fm, 5.0), np.searchsorted(fm, 60.0)
    band = Mod[a_:b_]
    pk, ix = band.max(0)
    med = Mod[c_:d_].median(0).values + 1e-12
    i2 = torch.clamp(2 * (ix + a_), max=Mod.shape[0] - 2)
    h2v = torch.maximum(Mod.gather(0, i2[None])[0], Mod.gather(0, (i2 + 1)[None])[0])
    modp = (10 * torch.log10(pk / med)).cpu().numpy()
    modh = (10 * torch.log10(h2v / med + 1e-12)).cpu().numpy()
    modf = fm[a_ + ix.cpu().numpy()]

    def fit(v):
        v = v[:nT]
        return np.pad(v, (0, nT - v.size), mode="edge") if v.size < nT else v
    return dict(modp=fit(modp), modf=fit(modf), modh=fit(modh))


def group_level(L, lo, hi):
    sel = (FC >= lo) & (FC <= hi) & np.isfinite(L[:, 0])
    return 10 * np.log10(np.sum(10 ** (L[sel] / 10), 0) + 1e-14)


def smooth_db(E, n):
    return 10 * np.log10(uniform_filter1d(10 ** (E / 10), n, mode="nearest") + 1e-14)


def background(E, q=20, win_s=600.0):
    step = 4
    d = E[::step]
    b = percentile_filter(d, q, size=max(3, int(win_s / (DT * step))), mode="nearest")
    return np.interp(np.arange(E.size), np.arange(d.size) * step, b)


def lorentz_fit(t, e_db, t0, floor_db):
    """Fit level(t) = 10 log10(A / (1 + ((t - tc) / tau)^2) + c) to a dB series.
    Returns tc, tau, peak-over-floor, rms."""
    pk = float(e_db.max())

    def model(p, tt):
        return 10 * np.log10(10 ** (p[2] / 10) / (1 + ((tt - p[0]) / np.exp(p[1])) ** 2) + 10 ** (p[3] / 10))

    best = None
    # start from the -3 dB width of the series; a second start guards against a poor first guess
    above = np.flatnonzero(e_db >= pk - 3.0)
    w0 = float(np.clip(0.5 * (t[above[-1]] - t[above[0]]), 0.6, 60.0)) if above.size else 5.0
    for tau0 in (w0, 3.0 * w0):
        p0 = [t0, np.log(min(tau0, 100.0)), pk, floor_db]
        try:
            r = least_squares(lambda p: model(p, t) - e_db, p0, loss="soft_l1", f_scale=1.5, max_nfev=30,
                              bounds=([t[0], np.log(0.3), pk - 6, floor_db - 15], [t[-1], np.log(120.0), pk + 6, pk]))
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    p = best.x
    res = model(p, t) - e_db
    return dict(tc=float(p[0]), tau=float(np.exp(p[1])), pk_db=float(p[2] - p[3]),
                rms=float(np.sqrt(np.mean(res ** 2))))


def split_fit(t, e_db, tc, floor_db):
    """Left / right half-widths around a fixed centre (directivity asymmetry: >0 means the decay is slower)."""
    pk = float(e_db.max())

    def model(p, tt):
        tau = np.where(tt < tc, np.exp(p[0]), np.exp(p[1]))
        return 10 * np.log10(10 ** (p[2] / 10) / (1 + ((tt - tc) / tau) ** 2) + 10 ** (p[3] / 10))
    try:
        r = least_squares(lambda p: model(p, t) - e_db, [np.log(5.0), np.log(5.0), pk, floor_db], loss="soft_l1",
                          f_scale=1.5, max_nfev=30, bounds=([np.log(0.3), np.log(0.3), pk - 6, floor_db - 15],
                                               [np.log(120.0), np.log(120.0), pk + 6, pk]))
    except Exception:  # noqa: BLE001
        return None
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def crossing_width(t, x_db, ip, drop):
    """Width of the peak at `drop` dB below its top (linear interpolation); None when a side never gets there."""
    thr = x_db[ip] - drop
    i = ip
    while i > 0 and x_db[i] > thr:
        i -= 1
    j = ip
    while j < x_db.size - 1 and x_db[j] > thr:
        j += 1
    if x_db[i] > thr or x_db[j] > thr:
        return None, None
    tl = t[i] + (thr - x_db[i]) / max(1e-6, x_db[i + 1] - x_db[i]) * (t[i + 1] - t[i])
    tr = t[j - 1] + (x_db[j - 1] - thr) / max(1e-6, x_db[j - 1] - x_db[j]) * (t[j] - t[j - 1])
    return float(t[ip] - tl), float(tr - t[ip])


def detect(F, det=DET):
    """Candidate events from the band levels. Returns the events (without the de-Doppler part) and the envelopes."""
    L = F["Lf"].astype(np.float64)              # floor-tracked band levels
    nT = L.shape[1]
    t = (np.arange(nT) + 0.5) * DT
    env, bg, exc = {}, {}, {}
    for g in GROUPS:
        env[g] = F["gf_" + g].astype(np.float64)
        s1 = smooth_db(env[g], 4)
        bg[g] = background(s1)
        exc[g] = s1 - bg[g]
    n4 = int(round(det["smooth_s"] / DT))
    xs = smooth_db(env["mid"], n4) - bg["mid"]
    pk, pr = find_peaks(xs, height=det["height"], prominence=det["prom"], distance=int(det["dist_s"] / DT))
    cand = [(int(i), float(p), "mid") for i, p in zip(pk, pr["prominences"])]
    xl = smooth_db(env["low"], n4) - bg["low"]
    pl, prl = find_peaks(xl, height=det["low_height"], prominence=det["low_prom"], distance=int(det["dist_s"] / DT))
    for i, p in zip(pl, prl["prominences"]):
        if not any(abs(i - j) * DT < 8.0 for j, _, _ in cand):
            cand.append((int(i), float(p), "low"))
    cap = int(det["max_per_h"] * nT * DT / 3600.0) + 5
    cand = sorted(sorted(cand, key=lambda c: -c[1])[:cap])
    Lb = np.where(np.isfinite(L), L, -140.0)
    bgb = np.stack([background(smooth_db(Lb[b], 4)) for b in range(L.shape[0])])
    events = []
    for ip, prom, src in cand:
        g = "mid" if src == "mid" else "low"
        x1 = exc[g]
        lo_i, hi_i = max(0, ip - int(4 / DT)), min(nT, ip + int(4 / DT) + 1)
        ip1 = lo_i + int(np.argmax(x1[lo_i:hi_i]))
        top = x1[ip1]
        # fitting window: out to -12 dB or +-60 s, cut at a valley that rises again by 3 dB
        a = ip1
        mn = top
        while a > 0 and (ip1 - a) * DT < 60 and x1[a] > top - 12:
            mn = min(mn, x1[a])
            if x1[a] > mn + 3 and x1[a] < top - 3:
                break
            a -= 1
        b = ip1
        mn = top
        while b < nT - 1 and (b - ip1) * DT < 60 and x1[b] > top - 12:
            mn = min(mn, x1[b])
            if x1[b] > mn + 3 and x1[b] < top - 3:
                break
            b += 1
        a, b = min(a, max(0, ip1 - int(3 / DT))), max(b, min(nT - 1, ip1 + int(3 / DT)))
        tt = t[a:b + 1]
        s1 = smooth_db(env[g], 4)
        fit = lorentz_fit(tt, s1[a:b + 1], t[ip1], float(bg[g][ip1]))
        if fit is None:
            continue
        ev = dict(t_pk=float(t[ip1]), src=src, prom=prom, x_det=float((xs if src == "mid" else xl)[ip]),
                  x_mid=float(exc["mid"][ip1]), x_low=float(exc["low"][ip1]), x_hi=float(exc["hi"][ip1]),
                  x_midlow=float(exc["midlow"][ip1]), win=[float(tt[0]), float(tt[-1])],
                  tc=fit["tc"], tau=fit["tau"], fit_pk=fit["pk_db"], fit_rms=fit["rms"],
                  edge=bool(t[ip1] < 15 or t[ip1] > t[-1] - 15))
        sp = split_fit(tt, s1[a:b + 1], fit["tc"], float(bg[g][ip1]))
        ev["tau_l"], ev["tau_r"] = sp if sp else (None, None)
        for drop in (3, 6, 10):
            wl, wr = crossing_width(t, x1, ip1, drop)
            ev["wl%d" % drop], ev["wr%d" % drop] = wl, wr
        for g2 in ("low", "midlow", "hi"):
            s2 = smooth_db(env[g2], 4)
            if exc[g2][a:b + 1].max() >= 3.0:
                f2 = lorentz_fit(tt, s2[a:b + 1], t[a + int(np.argmax(s2[a:b + 1]))], float(bg[g2][ip1]))
            else:
                f2 = None
            ev["tc_" + g2] = f2["tc"] if f2 else None
            ev["tau_" + g2] = f2["tau"] if f2 else None
        h = max(2, int(min(max(fit["tau"], 1.0), 6.0) / DT))
        s_, e_ = max(0, ip1 - h), min(nT, ip1 + h + 1)
        ev["band_exc"] = [None if not np.isfinite(L[k, 0]) else float(np.mean(Lb[k, s_:e_]) - bgb[k, ip1])
                          for k in range(L.shape[0])]
        tot = 10 * np.log10(np.sum(10 ** (Lb[:, s_:e_].mean(1) / 10)))
        ev["band_rel"] = [None if not np.isfinite(L[k, 0]) else float(np.mean(Lb[k, s_:e_]) - tot)
                          for k in range(L.shape[0])]
        ev["rough"] = float(np.std(env[g][s_:e_] - s1[s_:e_]))
        for g2 in GROUPS:                       # burstiness: mean level minus floor-tracked level
            ev["burst_" + g2] = float(np.mean(F["gm_" + g2][s_:e_] - F["gf_" + g2][s_:e_]))
        dx = np.diff(x1[a:b + 1]) / DT          # steepest rise / fall of the 1 s envelope (dB/s): gated sources jump
        ev["rise_max"] = float(dx.max()) if dx.size else 0.0
        ev["fall_max"] = float(-dx.min()) if dx.size else 0.0
        ev["modp"] = float(np.max(F["modp"][s_:e_]))
        ev["modp_med"] = float(np.median(F["modp"][s_:e_]))
        ev["modf"] = float(F["modf"][s_ + int(np.argmax(F["modp"][s_:e_]))])
        ev["modh"] = float(np.max(F["modh"][s_:e_]))
        ev["flat"] = float(np.mean(F["flat"][s_:e_]))
        ev["flat_bg"] = float(np.median(F["flat"][max(0, ip1 - 600): ip1 + 600]))
        ev["crest_lf"] = float(np.median(F["crest_lf"][s_:e_]))
        ev["crest_hf"] = float(np.median(F["crest_hf"][s_:e_]))
        ev["lev_abs"] = float(s1[ip1])
        ev["lev_med"] = float(np.median(s1))
        events.append(ev)
    return events, dict(t=t, exc=exc, env=env, bg=bg)


class DeDoppler:
    HALF = 20.0                                  # s of audio either side of the envelope peak
    TC_SPAN = 12.0                               # s searched around the envelope peak
    STEP = 0.006                                 # log-frequency row (0.6 %)
    AVG = 4                                      # STFT hops (32 ms) averaged per column -> 0.128 s
    BETAS = np.geomspace(0.05, 0.33, 17)         # v / c, ratio 1.125
    TAUS = np.geomspace(0.7, 40.0, 16)           # d / v (s), ratio 1.31
    WEIGHTED = False
    DILATE = 2                                   # +- rows (1.2 %) of tolerance for the hypothesis grid (self-test)

    def __init__(self, fs, dev):
        self.fs, self.dev = fs, dev
        self.n_fft = int(round(0.064 * fs))
        self.hop = self.n_fft // 2
        self.col = self.AVG * self.hop / fs      # 0.128 s
        self.fmin, self.fmax = 400.0, min(0.45 * fs, 7000.0)
        self.nb = int(np.log(self.fmax / self.fmin) / self.STEP) + 1
        self.lf = self.fmin * np.exp(self.STEP * np.arange(self.nb))
        self.nt = int(round(2 * self.HALF / self.col))
        f = np.fft.rfftfreq(self.n_fft, 1.0 / fs)
        fi = np.interp(self.lf, f, np.arange(f.size))
        self.i0 = torch.as_tensor(np.floor(fi).astype(np.int64), device=dev)
        self.w = torch.as_tensor((fi - np.floor(fi)).astype(np.float32), device=dev)[:, None]
        self.win = torch.hann_window(self.n_fft, device=dev)
        hyp, curves = [], []
        for b in self.BETAS:
            for tau in self.TAUS:
                U = float(np.clip(5 * tau, 5.0, 14.0))
                nu = int(U / self.col)
                u = np.arange(-nu, nu + 1) * self.col
                s = np.log(doppler_linear(u, 1.0, b * C, tau * b * C, 0.0, C))
                hyp.append((b, tau, U))
                # points weigh as the expected amplitude 1 / sqrt(1 + (u / tau)^2): the bright, steep middle of
                # the curve decides, so a short flat piece of somebody else's asymptote cannot win (self-test)
                wgt = np.ones(u.size) if not self.WEIGHTED else 1.0 / np.sqrt(1.0 + (u / tau) ** 2)
                curves.append((np.arange(-nu, nu + 1), np.rint(s / self.STEP).astype(int), wgt / wgt.sum()))
        self.hyp = np.array(hyp)
        rmax = max(int(np.abs(c[1]).max()) for c in curves)
        umax = max(int(c[0].max()) for c in curves)
        self.R = self._fast(self.nb + rmax + 2)
        self.T = self._fast(self.nt + umax + 2)
        nh = len(curves)
        # forward, time-reversed, and (per distinct length) flat templates
        self.lens = sorted({len(c[0]) for c in curves})
        # forward curves and flat lines; the time-reversed templates come from the forward FFTs (see _kf)
        specs = [(rr, uu, ww) for uu, rr, ww in curves]
        for n in self.lens:
            uu = np.arange(-(n // 2), n // 2 + 1)
            specs.append((np.zeros(uu.size, int), uu, np.full(uu.size, 1.0 / uu.size)))
        parts = []
        for s0 in range(0, len(specs), 40):       # small batches (memory)
            K = torch.zeros((len(specs[s0:s0 + 40]), self.R, self.T), device=dev)
            for i, (rr, uu, ww) in enumerate(specs[s0:s0 + 40]):
                K[i, rr % self.R, uu % self.T] = torch.as_tensor(ww, dtype=torch.float32, device=dev)
            parts.append(torch.conj(torch.fft.rfft2(K)))
            del K
        self.Kf = torch.cat(parts)
        del parts
        self.flat_of = np.array([self.lens.index(len(c[0])) for c in curves])
        self.span = np.array([int(np.abs(c[1]).max()) for c in curves])
        self.nh = nh

    def _kf(self, sl):
        """conj(FFT) of templates sl in the virtual order [forward (nh), reversed (nh), flat]. For a real template,
        reversing time gives FFT_rev[kr, kt] = conj(FFT[-kr, kt])."""
        nh = self.nh
        idx = np.arange(sl.start, sl.stop)
        out = []
        for a, b, kind in ((0, nh, "f"), (nh, 2 * nh, "r"), (2 * nh, 2 * nh + len(self.lens), "x")):
            ii = idx[(idx >= a) & (idx < b)]
            if ii.size == 0:
                continue
            if kind == "f":
                out.append(self.Kf[ii[0]: ii[-1] + 1])
            elif kind == "x":
                out.append(self.Kf[ii[0] - nh: ii[-1] + 1 - nh])
            else:
                k = self.Kf[ii[0] - nh: ii[-1] + 1 - nh]
                out.append(torch.conj(torch.roll(torch.flip(k, dims=[1]), 1, dims=1)))
        return torch.cat(out) if len(out) > 1 else out[0]

    @staticmethod
    def _fast(n):
        while True:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1

    def whiten(self, x):
        S = torch.stft(x, self.n_fft, self.hop, window=self.win, center=True, return_complex=True).abs() ** 2
        P = 10 * torch.log10(S + 1e-14)                                    # (F, T)
        k = 31
        Pp = torch.nn.functional.pad(P.T[:, None, :], (k // 2, k // 2), mode="replicate")[:, 0]
        P = P - Pp.unfold(1, k, 1).median(2).values.T                      # remove the broadband level per column
        P = P - P.median(1, keepdim=True).values                           # remove steady lines per bin
        P = P.clamp(0.0, 12.0)
        n = (P.shape[1] // self.AVG) * self.AVG
        P = P[:, :n].reshape(P.shape[0], n // self.AVG, self.AVG).mean(2)
        W = (1 - self.w) * P[self.i0] + self.w * P[torch.clamp(self.i0 + 1, max=P.shape[0] - 1)]
        if self.DILATE:
            k = 2 * self.DILATE + 1
            W = torch.nn.functional.max_pool2d(W[None, None], (k, 1), stride=1, padding=(self.DILATE, 0))[0, 0]
        return W                                                           # (nb, cols)

    def fit(self, x, fs, t_pk):
        """x: full recording (torch, on device). Returns the best curve near t_pk (s) or None."""
        a = int(round((t_pk - self.HALF) * fs))
        b = a + int(round(2 * self.HALF * fs))
        if a < 0 or b > x.numel():
            pad_l, pad_r = max(0, -a), max(0, b - x.numel())
            seg = torch.nn.functional.pad(x[max(0, a): min(x.numel(), b)], (pad_l, pad_r))
        else:
            seg = x[a:b]
        W = self.whiten(seg)[:, : self.nt]
        Wp = torch.zeros((self.R, self.T), device=self.dev)
        Wp[: W.shape[0], : W.shape[1]] = W
        Wf = torch.fft.rfft2(Wp)
        c0 = int((self.HALF - self.TC_SPAN) / self.col)
        c1 = int((self.HALF + self.TC_SPAN) / self.col)
        nh = self.nh

        def corr(sl):
            Cc = torch.fft.irfft2(Wf[None] * self._kf(sl), s=(self.R, self.T))[:, : self.nb, : self.nt]
            # z against the map's own mean and standard deviation. (Median / MAD is wrong here: the whitened
            # spectrogram is clipped at 0, so short templates have a near-zero MAD and inflated z - self-test.)
            sub = Cc[:, ::3, ::3].reshape(Cc.shape[0], -1)
            mu = sub.mean(1)
            sd = sub.std(1) + 1e-4
            return Cc, (Cc[:, :, c0:c1] - mu[:, None, None]) / sd[:, None, None]

        # per hypothesis: robust z over (source frequency, CPA time); keep only the pooled maxima (small memory)
        zs_l, z1_l = [], []
        for s in range(0, 2 * nh, 40):
            _, Zc = corr(slice(s, min(s + 40, 2 * nh)))
            # multi-line statistic: the three brightest source frequencies (9-row pools) at one hypothesis and time
            top = torch.nn.functional.max_pool2d(Zc, (9, 1), (9, 1)).topk(3, dim=1).values
            zs_l.append(top.sum(1) / np.sqrt(3.0))
            z1_l.append(top[:, 0])
            del Zc, top
        zs, z1 = torch.cat(zs_l), torch.cat(z1_l)
        best = {}
        for name, stat in (("s", zs), ("1", z1)):
            fwd, rev = stat[:nh], stat[nh:2 * nh]
            v, idx = fwd.reshape(-1).max(0)
            h, ic = divmod(int(idx), fwd.shape[1])
            best[name] = (float(v), h, ic, float(rev.max()))
        v, h, ic, vrev = best["s"]
        Cb, Zb = corr(slice(h, h + 1))
        col = Zb[0, :, ic]
        row = int(col.argmax())
        raw = float(Cb[0, row, c0 + ic])
        sp = int(self.span[h])
        jf = 2 * nh + int(self.flat_of[h])
        Cf, _ = corr(slice(jf, jf + 1))
        fl = Cf[0, max(0, row - sp): row + sp + 1, c0 + ic]
        # sharpness: spread of the hypotheses within 85 % of the best
        fwd = zs[:nh]
        near = (fwd >= 0.85 * v).nonzero()
        hh = self.hyp[near[:, 0].cpu().numpy()]
        tcs = near[:, 1].cpu().numpy() * self.col
        pool_h = torch.nn.functional.max_pool2d(Zb, (9, 1), (9, 1))[0, :, ic]
        nlines = int((pool_h >= max(6.0, 0.6 * float(pool_h.max()))).sum())
        z1_at = float(Zb.max())
        b, tau, _ = self.hyp[h]
        v1, h1, ic1, v1rev = best["1"]
        return dict(zs=v, zs_rev=vrev, z1=z1_at, z1_best=v1, z1_rev=v1rev,
                    beta=float(b), tau=float(tau), tc=float(t_pk - self.TC_SPAN + ic * self.col),
                    f0=float(self.lf[row]), raw=raw, flat_margin=float(raw - fl.max()), nlines=nlines,
                    beta_lo=float(hh[:, 0].min()), beta_hi=float(hh[:, 0].max()),
                    tau_lo=float(hh[:, 1].min()), tau_hi=float(hh[:, 1].max()),
                    tc_spread=float(tcs.max() - tcs.min()),
                    beta1=float(self.hyp[h1][0]), tau1=float(self.hyp[h1][1]),
                    tc1=float(t_pk - self.TC_SPAN + ic1 * self.col))


_DD = {}


def analyze(x, fs, dev="cpu", dedoppler=True):
    """Full blind analysis of one recording (mono float32 at 8 or 16 kHz): frame features and the event list."""
    dev = torch.device(dev)
    F = frame_features(x, fs, dev)
    events, _envs = detect(F)
    if dedoppler and events:
        if (fs, dev.type) not in _DD:
            _DD[(fs, dev.type)] = DeDoppler(fs, dev)
        dd = _DD[(fs, dev.type)]
        xt = torch.as_tensor(x, device=dev)
        for ev in events:                       # the de-Doppler fit only for events that stand out
            ev["dop"] = dd.fit(xt, fs, ev["t_pk"]) if ev["x_det"] >= DET["dop_min"] else None
    return F, events


# de-Doppler version of the research events: 1 median / MAD z, 12 x 12 grid (failed the self-test);
# 2 mean / std z, 17 x 16 grid, +-2 rows (this code)
DOP_VERSION = 2


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        from skyhum.errors import MissingData
        raise MissingData("the loud-event channel decodes audio with ffmpeg; install it")
    return ff


def audio_rate(path):
    ff = shutil.which("ffprobe")
    if not ff:
        from skyhum.errors import MissingData
        raise MissingData("the loud-event channel needs ffprobe (part of ffmpeg)")
    r = subprocess.run([ff, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return int(r.stdout.strip().split("\n")[0])


def load_audio(path, fs=None):
    """Decode to mono float32 with ffmpeg. fs None: 16 kHz when the file has >= 16 kHz, else 8 kHz."""
    if fs is None:
        fs = 16000 if audio_rate(path) >= 16000 else 8000
    r = subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(fs), "-f",
                        "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy(), fs


def dither(x):
    """-120 dB seeded dither (digital silence breaks the floor tracking), as in the research runs."""
    return x + 1e-6 * np.random.default_rng(0).standard_normal(x.size).astype(np.float32)


def events_from_file(path, dev="cpu", min_s=60.0):
    """Blind events of one audio file: {fs, dur, events, dop_version}; None when shorter than min_s."""
    x, fs = load_audio(path)
    if x.size < fs * min_s:
        return None
    _F, events = analyze(dither(x), fs, dev)
    return dict(fs=fs, dur=x.size / fs, events=events, dop_version=DOP_VERSION)
