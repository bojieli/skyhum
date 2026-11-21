"""Classic tonal-pass detector (no learned weights).

Finds ridges in the whitened log-frequency spectrogram that last at least MIN_DUR_S seconds, groups ridges whose
steepest descent falls within GROUP_S of each other (harmonics of one sound), and labels each group:

  doppler-like  the ridge falls in pitch by at least MIN_SLIDE (what a passing propeller or rotor does)
  steady        a tone that does not slide (machinery, mains hum, a hovering rotor)

This is a transparent baseline. The learned detector from the research tree (model/v3) is not shipped in this
build: its weights are not public yet, and it would be downloaded with a pinned sha256, never stored in git.
"""
import numpy as np

THR = 4.0
MIN_DUR_S = 10.0
MIN_SLIDE = 0.02
GROUP_S = 8.0
NAME = "classic-ridge"


def detect(spec, t0=None):
    from scipy import ndimage
    W = spec.W
    cand = (W > THR) & (W >= ndimage.maximum_filter(W, size=(3, 1), mode="nearest"))
    lab, n = ndimage.label(ndimage.binary_dilation(cand, structure=np.ones((3, 3), bool)), structure=np.ones((3, 3)))
    ridges = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        rows, cols = np.nonzero((lab[sl] == i) & cand[sl])
        if rows.size == 0:
            continue
        rows, cols = rows + sl[0].start, cols + sl[1].start
        c0, c1 = cols.min(), cols.max()
        if (c1 - c0 + 1) * spec.col_s < MIN_DUR_S:
            continue
        uc = np.unique(cols)
        # one frequency per column: level-weighted mean row
        fr = np.array([np.average(rows[cols == c], weights=W[rows[cols == c], c]) for c in uc])
        f = spec.freqs[0] * np.exp(spec.log_step * fr)
        k = max(1, uc.size // 6)
        f_hi, f_lo = float(np.median(f[:k])), float(np.median(f[-k:]))
        slide = f_hi / f_lo - 1.0
        sm = np.convolve(np.log(f), np.ones(5) / 5, mode="same")
        d = np.diff(sm[2:-2]) if uc.size > 8 else np.diff(np.log(f))
        j = int(np.argmin(d)) + (2 if uc.size > 8 else 0) if d.size else uc.size // 2
        snr = float(np.mean(W[rows, cols]))
        ridges.append(dict(start_s=float(c0 * spec.col_s), end_s=float(c1 * spec.col_s),
                           cpa_s=float(uc[min(j, uc.size - 1)] * spec.col_s), f_start_hz=round(f_hi, 1),
                           f_end_hz=round(f_lo, 1), slide=round(slide, 4), level=round(snr, 2)))
    ridges.sort(key=lambda r: r["cpa_s"])
    passes = []
    for r in ridges:
        g = passes[-1] if passes else None
        if g and abs(r["cpa_s"] - g["cpa_s"]) <= GROUP_S:
            g["ridges"].append(r)
            continue
        passes.append(dict(cpa_s=r["cpa_s"], ridges=[r]))
    out = []
    for g in passes:
        rs = g["ridges"]
        best = max(rs, key=lambda r: r["level"] * (r["end_s"] - r["start_s"]))
        dop = sum(r["slide"] >= MIN_SLIDE for r in rs) >= max(1, len(rs) // 2)
        p = dict(kind="doppler-like" if dop else "steady", cpa_s=best["cpa_s"], start_s=min(r["start_s"] for r in rs),
                 end_s=max(r["end_s"] for r in rs), f_start_hz=best["f_start_hz"], f_end_hz=best["f_end_hz"],
                 slide_pct=round(100 * best["slide"], 1), level=best["level"], n_ridges=len(rs))
        if t0 is not None:
            p["cpa_utc"] = t0 + p["cpa_s"]
        out.append(p)
    return out


def plot(spec, passes, path, title=""):
    """Spectrogram PNG with detected passes marked. Needs matplotlib (pip install 'skyhum[plot]')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    ext = [0, spec.n_cols * spec.col_s, 0, spec.W.shape[0]]
    ax.imshow(spec.W, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=10)
    ticks = [f for f in (30, 50, 100, 200, 400, 600) if f <= spec.freqs[-1]]
    ax.set_yticks([np.log(f / spec.freqs[0]) / spec.log_step for f in ticks], [str(f) for f in ticks])
    for p in passes:
        ax.axvline(p["cpa_s"], color="cyan" if p["kind"] == "doppler-like" else "grey", lw=0.8, ls="--")
    ax.set_xlabel("seconds from start")
    ax.set_ylabel("Hz")
    ax.set_title(title or "whitened spectrogram (dashed: detected passes)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
