import pytest

from skyhum import selftest


@pytest.mark.parametrize("seed", [7, 3])
def test_selftest_quick_passes(seed):
    r = selftest.run(seed=seed, r5b=False)
    assert r["result"] == "PASS", r["checks"]
    names = [c["check"] for c in r["checks"]]
    assert any("true claim" in n for n in names) and any("20 km" in n for n in names)


def test_selftest_cli_quick(run_cli):
    code, out, _ = run_cli("--device", "cpu", "selftest", "--quick")
    assert code == 0 and "selftest PASS" in out and "[skip] r5b" in out
