# The locator (round 5b)

`verify`, `audit` and `find` use the round-5b locator by default (`--matcher r5b`). It is vendored from the
research code (round 5b) as `src/skyhum/match/r5b.py`, with the round's frozen configuration.

## What it does

For every grid site and every ADS-B pass within 9 km, it computes the exact Doppler curve the pass would leave at
that site and scores it against the whitened spectrogram, then combines passes that agree on one recording clock.
Round 5b replaced the earlier vote counting with calibrated evidence:

- **Per-pass log-likelihood ratios.** Each pass's match statistic is turned into a log-likelihood ratio fitted on
  development data (`k3_full.json`), including a cost for passes that should have been audible but were not.
- **f0 prior.** A prior over blade-pass frequency per aircraft type (kappa 3).
- **Amplitude-envelope LLR.** How the level rises and falls around the closest approach (`env_full.json`).
- **Terrain line of sight** (`--dem-dir`, off in the frozen configuration) and **time slack** (off in the frozen
  configuration): both are in the code and were tested as ablations; the package exposes neither by default.
- **Single-event mirror point.** One pass fixes a line across the aircraft's track and a distance, so two points
  fit equally well. `find --single-event` prints both.

## Fitted tables

`k3_full.json` (per-pass LLR and f0 prior) and `env_full.json` (envelope) ship in `src/skyhum/models/r5b/`.
Their fitting set and licences are in the [model card](../src/skyhum/models/MODEL_CARD.md).

## How the package uses it

- `verify`: one run over the claimed area's bounding box at the stated clock window; the claim's score is the best
  hypothesis inside the area. The same run at decoy clocks (tens of minutes off, or other days) gives the decoy
  level. `SUPPORTED` needs the claim to beat every decoy (default) or beat them by a fixed margin of 3 (strict).
  The default is 20 decoys; at least 5 must fit inside the ADS-B file.
- `find`: one run over the area, hypotheses inside the area, 10 decoy runs by default. A point estimate only above
  the strict margin. Several sessions (`--sessions`) are scored separately and combined where they agree within
  3 km; the research multi-session scorer was not ported (and was not tested in round 5b).
- `--fast` lowers the calibration sample and the refinement for demos and tests. Do not use it for results.
- Detector hints, as in the test: before the locator runs, detector v3 (`skyhum.detect.v3`) scans the audio and
  its detections go to the locator as `--hints` (a JSON list of `{t, f, p, p_raw, v, d, start}`, t in seconds from
  the start of the audio). The frozen `--hint-pmin 0.8` keeps the confident ones: they add candidate spots (within
  8 s and 3 % of each hint) to the curve search, and a hint at a pass's closest approach enters its log-likelihood
  ratio as P(hint | audible pass) against the recording's hint coverage (fitted in `k3_full.json`).
  `--hints auto` (default) computes them from the file, decoded at its own rate and resampled straight to 1 kHz as
  the research runs did; `--hints FILE` takes a file written by `skyhum detect --hints-out FILE`; `--hints none`
  runs without (lower recall than the test). With `--scrub`, the scrubbed audio gets its own hints.
- Detector v3 on a CPU takes a few seconds per hour of audio; the port gives the research code's detections exactly
  (same seed; checked on two A2O hours and on the synthetic clips, `tests/test_detect_v3.py`).

## CPU and GPU

The locator uses Triton kernels on CUDA and an equivalent PyTorch path on the CPU. The CPU path in the package
precomputes the ridge-contrast image once per call (bit-identical sums, checked by a test), which makes it about
three times faster than the original fallback. It is still slow: on the synthetic demo (four minutes of audio, a
small box), one run takes on the order of a minute with `--fast`. An hour of audio over a metro area needs a GPU.
`--device cpu` never initialises CUDA.

## What the round-5b test measured, and what it did not

The test ran on the split_v2 test set with a jittered box (0.25 degree) and detector v3 hints, 43 selected
positive hours from 7 site clusters. Top-1 within 2 km:

| Clock | Round 5a | Round 5b |
|---|---|---|
| Blind (whole day) | 15 | 22 |
| Known to 30 min | 19 | 23 |
| Known to 30 s | 23 | 31 |

- The cluster-bootstrap confidence intervals overlap (only 7 site clusters).
- The no-box (wide) mode reached only 9 positives before the run was stopped: not a result.
- **Not run:** the clock decoys of the main test, the decoys of the single-event hard and negative hours, and the
  multi-session test. So the false-fix rate (how often a decoy-beating answer is wrong) and the calibration of the
  strict threshold are unmeasured. `verify` and `find` say so in their output.

## Not in the package

The queue, rented-host and vast scripts used to run the research evaluations. Every evaluation has a local runner
instead (see REPRODUCING.md).
