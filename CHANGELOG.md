# Changelog

## Unreleased

- Removed the `find` gate (`--research`, `--attest`, the 100 km area cap and the local log); the README and ETHICS
  state the intended use instead.
- r5b numbers updated to the final round-5b stage 1 (decoy check included); confidence is described as "not calibrated: about 2 in 3 confident answers within 2 km on a 7-cluster test".
- Ship the research models: detector v3 weights, the round-5b fitted tables, and the loud-event classifier as
  numpy tree arrays (no pickle), with SHA256SUMS and a model card listing every training source and licence.
- Whole-country search scorers ported from the research code: `skyhum country score / prior / claims`, and the
  region tables (`skyhum country static`, public Census / ABS / terrain files downloaded at run time).
- Box-benchmark scorers (round-5b tables, single-event two-point mode with decoys).
- `skyhum bench run --reproduce FAMILY --outputs DIR` rescores locator outputs for the country, prior, verify, box
  and single-event row families. Public A2O manifests (links and metadata only). REPRODUCING.md.

## 0.1.0-alpha (first public release)

- Licence MIT. Third-party notices added. The fitted round-5b tables are not published yet; `--matcher auto`
  (default) falls back to the reference matcher without them.

- The round-5b locator (from the research branch `locator-r5b`) is the default matcher for `verify`, `audit` and
  `find`: f0 prior, calibrated per-pass log-likelihood ratios, amplitude-envelope LLR, optional terrain line of
  sight (`--dem-dir`), and the single-event mirror point (`find --single-event`). It runs on CPU (slower) when no
  GPU is available. Its test is partial; the `find` confidence threshold is not validated (docs/limits.md).
- The earlier simplified matcher stays available as `--matcher reference` (fast, numpy only, not benchmarked).
- First package: CLI with `selftest`, `doctor`, `regions`, `detect`, `verify`, `audit`, `scrub`, gated `find`,
  `bench list`; JSON and HTML reports; synthetic fixtures; tests; docs.
