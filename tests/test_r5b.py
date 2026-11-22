"""The round-5b locator on CPU (slow: minutes). Synthetic data only."""
import json

import pytest

from conftest import SITE, START, fx

pytest.importorskip("torch")
pytestmark = pytest.mark.slow


def _models():
    from skyhum.match import locator
    return locator.models_available()


needs_models = pytest.mark.skipif(not _models(), reason="fitted r5b tables not installed")


def test_curve_scores_cpu_matches_three_gather_form():
    import torch

    from skyhum.match import r5b
    torch.manual_seed(0)
    L = torch.rand(120, 90)
    P, K, Q, J, dr = 7, 21, 500, 10, 4
    db = torch.randint(-30, 30, (P, K))
    w = torch.rand(P, K)
    qb = torch.randint(-10, 130, (Q,))
    qt = torch.randint(-5, 95, (Q,))
    got = r5b.curve_scores(L, db, w, qb, qt, J, dr)
    nb, nt = L.shape
    Lp = torch.nn.functional.pad(L, (0, 1, 0, 1)).reshape(-1)
    want = torch.zeros(P, Q)
    for k in range(K):
        r = qb[None, :] + db[:, k:k + 1]
        c = (qt + k - J)[None, :].expand(P, -1)

        def g(rr, c=c):
            ok = (rr >= 0) & (rr < nb) & (c >= 0) & (c < nt)
            return Lp[torch.where(ok, rr * (nt + 1) + c, torch.full_like(rr, nb * (nt + 1)))]
        want += w[:, k:k + 1] * (g(r) - 0.5 * (g(r + dr) + g(r - dr)))
    assert torch.equal(got, want)


@needs_models
def test_r5b_verify_true_and_displaced(run_cli, demo):
    base = ["verify", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START, "--fast",
            "--decoys", "5", "--json"]
    code, out, _ = run_cli(*base, "--claim", f"{SITE},r=2km")
    r = json.loads(out)["result"]
    assert code == 0 and r["matcher"] == "r5b" and r["verdict"] == "SUPPORTED" and r["passes"]
    assert r["hints"]["detector"] == "v3" and r["hints"]["n_used"] >= 2          # detector v3 hints, as in the test
    assert "not calibrated" in r["note"]
    code, out, _ = run_cli(*base, "--claim", "34.05,-118.10,r=2km")
    assert code in (10, 11)


@needs_models
def test_r5b_find_gate_and_output(run_cli, demo):
    code, out, err = run_cli("find", fx(demo, "clip.wav"), "--adsb", fx(demo, "tracks.npz"), "--start", START,
                             "--area", f"{SITE},r=3km", "--fast", "--decoys", "5")
    assert code in (0, 10) and "not calibrated" in out + err
    if code == 10:
        assert "no point estimate" in out
