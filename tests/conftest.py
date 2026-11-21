"""Fixtures: synthetic audio and synthetic ADS-B tracks, generated at test time. No real data is used."""
import os

import pytest

START = "2026-09-08T14:03:00Z"
SITE = "34.05,-118.30"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("SKYHUM_NO_GPU_PROBE", "1")          # never touch a GPU from the tests
    monkeypatch.setenv("SKYHUM_DEVICE", "cpu")
    monkeypatch.setenv("SKYHUM_WORKDIR", str(tmp_path / "workdir"))


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    from skyhum.selftest import write_fixtures
    d = tmp_path_factory.mktemp("demo")
    write_fixtures(str(d), seed=7)
    return d


@pytest.fixture
def run_cli(capsys):
    """Call skyhum.cli.main in-process; returns (exit_code, stdout, stderr)."""
    from skyhum.cli import main

    def run(*argv):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as e:                               # argparse usage errors
            code = e.code
        cap = capsys.readouterr()
        return code, cap.out, cap.err
    return run


def fx(demo, name):
    return os.path.join(str(demo), name)
