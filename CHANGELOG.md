# Changelog

## Unreleased

- Removed the `find` gate (`--research`, `--attest`, the 100 km area cap and the local log); the README and ETHICS
  state the intended use instead.
- Ship the research models: detector v3 weights, the round-5b fitted tables, and the loud-event classifier as
  numpy tree arrays (no pickle), with SHA256SUMS and a model card listing every training source and licence.

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
