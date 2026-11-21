"""Decoy clocks: the same audio scored as if it had been recorded at a wrong time.

A real location clue should fit the claimed place at the stated clock better than at clocks shifted by tens of
minutes, or on other days. Each decoy is the whole clock window moved by one offset, so a decoy gets exactly as
many chances as the real window. Offsets are drawn from a seeded generator and must stay inside the ADS-B file's
coverage; a decoy that would need data we do not have is not used (never scored as zero).
"""
import numpy as np

MIN_DECOYS = 5           # fewer usable decoys than this: INCONCLUSIVE
DAY = 86400.0


def draw_offsets(n, spans, tracks, seed=0, margin_s=180.0):
    """Up to n offsets (seconds) valid for every span (win_lo, win_hi, dur_s): clock offsets of +-(gap .. gap+50 min)
    and other days, kept only where the ADS-B file covers the shifted window of every span."""
    rng = np.random.default_rng(seed)
    width = max(hi - lo for lo, hi, _ in spans)
    gap = max(600.0, width + 300.0)
    cand = [k * DAY for k in (-3, -2, -1, 1, 2, 3)]          # other days at the same clock window
    cand.extend((rng.uniform(gap, gap + 3000.0, 8 * n) * rng.choice([-1.0, 1.0], 8 * n)).tolist())
    out = []
    for off in cand:
        if all(tracks.covers(lo + off - margin_s, hi + off + dur + margin_s) for lo, hi, dur in spans):
            out.append(float(off))
        if len(out) >= n:
            break
    return np.array(out)


def windows(win_lo, win_hi, offsets, step_s=1.0):
    S = np.arange(win_lo, win_hi + step_s / 2, step_s)
    return np.vstack([S] + [S + o for o in offsets])
