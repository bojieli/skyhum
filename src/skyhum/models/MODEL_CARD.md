# Model card: skyhum shipped models

Three models ship in `src/skyhum/models/`, unchanged from the runs behind the published numbers (except the loud
classifier's storage format, see below). Checksums: `SHA256SUMS`, verified by `tests/test_models.py` and
`scripts/prepub_check.sh`.

**Read this first.** The weights were trained and fitted on the sources listed below. Some of those sources carry
NonCommercial, NoDerivatives, academic-only or YouTube terms. The maintainers treat training a model as
non-derivative use of the recordings; that is a position, not legal advice. Users are responsible for their own
use. No source is omitted from this list. No event or speech livestream (rallies, protests, amateur radio, anything
that could identify or locate a private individual) is in any training or fitting set: checked against every
case id of the three sets.

## 1. Detector v3 (`v3/ckpt.pt`, 3,063,073 bytes; `v3/config.json`, training configuration with paths reduced to file names)

- **What it is.** A CNN aircraft fly-over detector. Input: 240 s of audio at 1 kHz, as a whitened log-frequency
  spectrogram, 128 bins from 30 to 400 Hz by 300 time steps. Output per frequency-time cell: a detection score
  and a speed / slant-distance head; peaks give pass candidates with a closest-approach time and frequency.
- **How the published numbers used it.** Its detections (p >= 0.8) are the "hints" given to the locator in the
  round-3 to round-5b evaluations and in the country-wide search (`--hint-pmin 0.8`). The blind-detection figures
  (for example 34 of 86 relabel-audible passes at three test cameras) are this model.
- **Training data (30,000 steps, batch 48).**

  | Source | Role | Uses | Licence / terms |
  |---|---|---|---|
  | Australian Acoustic Observatory (A2O), train sites | real positives and negatives (ADS-B labelled) | train split | CC BY 4.0 |
  | xeno-canto background pool | backgrounds | 163 recordings: 135 CC BY-NC-SA 4.0, **28 CC BY-NC-ND** | per recording |
  | Codec-augmented copies of the xeno-canto pool | backgrounds through call codecs | same 163 recordings | as above |
  | AudioSet (balanced, eval and an unbalanced subset) | backgrounds | 70,613 segment uses | labels CC BY 4.0; **audio is YouTube video under YouTube's terms** |
  | Freesound (via LAION-Audio-630K) | backgrounds | 13,508 uses | per sound: CC0, CC BY, **CC BY-NC, Sampling+** |
  | FSD50K | backgrounds | 8,899 uses | per clip: CC0, CC BY, CC BY-NC |
  | TAU Urban Acoustic Scenes 2019 (development) | backgrounds | 554 uses | **academic-use licence** |
  | WHAM! noise | backgrounds | 247 uses | **CC BY-NC 4.0** |
  | ESC-50 | listed in the source catalogue, 0 uses in the final pool | – | CC BY-NC 3.0 |
  | SONYC-UST v2 | backgrounds | 300 uses | CC BY 4.0 |
  | MUSAN noise | backgrounds | 27 uses | CC BY 4.0 |
  | Synthetic fly-overs | positives | generated | – |

- **Evaluation.** split_v2 dev and test (site-clustered); detector scores in the research tables.
- **Known failure modes.** Continuous speech over a pass (passes with speech within 10 s of closest approach were
  rarely heard); urban noise and music; jets and distant passes; call codecs with discontinuous transmission;
  machinery tones that slide. At the study operating point it caught 34 of 86 relabel-audible passes at three test
  cameras.
- **In this package.** Inference is ported (`skyhum.detect.v3`: front end, network, sliding detection; no training
  code). The checkpoint loads with `torch.load(weights_only=True)`. `skyhum detect` uses it by default when PyTorch
  is installed (`--detector classic` otherwise), and the r5b runs of `verify`, `audit` and `find` pass its
  detections as hints (`--hints auto`, frozen `--hint-pmin 0.8`), as the test did. On the CPU with the same dither
  seed the port reproduces the research code's detections exactly (maximum difference 0.0 on two A2O test hours,
  67 and 87 detections, and on the synthetic selftest clips). The research hints were made on a GPU with an
  unseeded dither, so a CPU rerun differs slightly (32 of 34 detections at p >= 0.8 matched on one hour, 5 of 5
  on another).

## 2. Round-5b fitted tables (`r5b/k3_full.json`, `r5b/env_full.json`)

- **What they are.** `k3_full.json`: per-pass log-likelihood ratios of the match statistic by aircraft group,
  audibility by distance, and the f0 prior by aircraft type. `env_full.json`: the amplitude-envelope log-likelihood
  table. Used by the round-5b locator (`--llr`, `--f0-prior 3`, `--env-llr`), the package's default matcher.
- **Fitting set:** split_v2 dev hours. Composition: A2O 280 positive, 142 hard, 354 negative hours; xeno-canto 1
  positive, 3 hard, 96 negative hours (per-recording licences, which may include CC BY-NC-ND); colsto's YouTube
  video (1 positive); 1 public-webcam negative hour (YouTube livestream). Detector v3 hints were inputs.
- **Evaluation.** split_v2 test, partial (README, "Limits"; docs/locator.md).
- **Failure modes.** As the locator: running without hints (`--hints none`, or no PyTorch) means lower recall than the test; the strict threshold is not
  calibrated (about 2 in 3 confident answers within 2 km on the 7-cluster test; docs/locator.md).

## 3. Loud-event classifier (`loud/classifier.npz`, `loud/calibration.json`)

- **What it is.** Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier, 33 iterations x 4 classes:
  none / jet / prop / rotor) on 96 features of a loud transient event (third-octave levels, envelope fit, AM
  modulation, high-frequency de-Doppler). `calibration.json` holds the detection thresholds, time / width /
  Doppler calibration and the detection-probability table.
- **Storage.** The research model was a pickle. It ships as its tree arrays and is evaluated in numpy
  (`skyhum.detect.loud_model`), so no pickle is loaded. On 400 A2O dev events the exported model reproduces the
  pickle's probabilities exactly (maximum absolute difference 0.0; fixture `tests/data/loud_equiv.npz`).
- **Training set:** dev events, 35,275 of them: A2O 19,452; public airport and city webcams 15,823 (Midway,
  Lorain, Saginaw, Duluth, LaGuardia, Kearney, Chehalis, Indio, Big Bear, San Francisco, Honolulu, Keokuk, Belen,
  Burlington, Arvada, Elkhart, Marysville, Montauk, Moosehead, Shinnecock, Tucson, Cheyenne, Vancouver, Anchorage,
  Seaside, St Paul, Mackinaw, Olympia, Laramie, Kingman, Milwaukee, Cocoa). Webcam audio comes from YouTube
  livestreams under YouTube's terms. Camera-grouped cross-validation.
- **Failure modes.** Helicopters at the Red Hook test camera: 0 of 103 detected, and 0 of 37 props; speech and
  radio chatter are gated out, which also drops events under talk.
- **In this package.** `skyhum.loud` ports the blind event detector (`features`), the speech / ATC gating,
  `featurize`, the calibrated evidence (`model`) and the known-clock scorer (`locate`), inference only and not yet
  in the command line. The research evaluation driver (`run_locate.py`: registry, jittered boxes, decoy hours) is not
  ported. On one A2O hour the port's events, features, probabilities and evidence equal the research code's on the
  CPU (maximum difference 0.0). `tests/test_loud.py` rebuilds `loud_equiv.npz` from the raw event dicts in
  `tests/data/loud_events_a2o.json` (numbers only, from A2O audio, CC BY 4.0).
