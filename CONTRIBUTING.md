# Contributing

Thanks for your interest. Bug reports, test cases and documentation fixes are the most useful contributions.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,plot]"
pre-commit install            # optional: runs the checks below on each commit
```

## Before you open a pull request

```
pytest -q                     # synthetic data only, a couple of minutes on a laptop CPU
ruff check src tests scripts
skyhum selftest
scripts/prepub_check.sh       # no audio, no large files, no key-like strings
python scripts/gen_cli_docs.py   # if you changed any command or flag
```

## Rules

- **Never commit audio, ADS-B data, model weights, or recordings of anyone.** Tests generate synthetic audio and
  synthetic tracks at run time. Benchmark catalogues hold links and metadata only.
- **No secrets.** API keys belong in your environment, never in the repository.
- **Numbers in docs need a source.** Do not add accuracy figures without the evaluation that produced them.
- **Scope.** Contributions that widen `find` (larger areas, country-wide search, multi-region joint scoring) or
  weaken the gate will not be merged. See [docs/ETHICS.md](docs/ETHICS.md).
- Keep output honest: a new result type must state what was not tested.

## Style

Python 3.10+, `ruff` with the settings in `pyproject.toml`, line length 120. Plain English in docs and messages.
