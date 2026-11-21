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
  level. `SUPPORTED` needs the claim to beat every decoy (default) or beat them by a fixed margin of 3 (strict);
  neither is calibrated.
  The default is 20 decoys; at least 5 must fit inside the ADS-B file.
- `find`: one run over the area, hypotheses inside the area, 10 decoy runs by default. A point estimate only above
  the strict margin. Several sessions (`--sessions`) are scored separately and combined where they agree within
  3 km; the research multi-session scorer was not ported (and was not tested in round 5b).
- `--fast` lowers the calibration sample and the refinement for demos and tests. Do not use it for results.
- Detector hints: the test used detector v3 hints. The package has no learned detector yet, so runs here have none.

## CPU and GPU

The locator uses Triton kernels on CUDA and an equivalent PyTorch path on the CPU. The CPU path in the package
precomputes the ridge-contrast image once per call (bit-identical sums, checked by a test), which makes it about
three times faster than the original fallback. It is still slow: on the synthetic demo (four minutes of audio, a
small box), one run takes on the order of a minute with `--fast`. An hour of audio over a metro area needs a GPU.
`--device cpu` never initialises CUDA.

## What the round-5b test measured

Round-5b locator, 43 positive hours (split_v2 test, 7 site clusters, with detector v3 hints), top-1 within 2 km:
blind 22 (51%, CI 26-67%), clock known to 30 min 23 (53%), clock known to 30 s 31 (72%, CI 48-87%), no box (wide)
19 (44%). Decoy check, 20 clock decoys per hour, p <= 0.05: with the clock known to 30 s, 50 confident answers of
which 34 were within 2 km (68%, CI 36-87%); 9 of the 43 positive hours were confidently wrong; 1 of 12 empty hours
was still answered. Clock known to 30 min: 25 of 49 confident answers within 2 km (51%). Single event: 22 confident
answers, 14 correct and 8 wrong (21% false fixes on 38 hard hours). No margin threshold gives usable precision
(68-73% at 30 s, on 7 site clusters). The blind and wide runs have no decoy check. Several hours from one spot
(1 / 4 / 8 hours): blind 10% -> 24% -> 52%, clock known to 30 s 15% -> 47% -> 67%; the pooled confidence of these
is unmeasured. A real-world false-fix run on 634 recordings is still running.

Round 5a, same 43 hours, within 2 km: blind 15, clock to 30 min 19, clock to 30 s 23.

## Not in the package

The queue, rented-host and vast scripts used to run the research evaluations. Every evaluation has a local runner
instead (see REPRODUCING.md).
