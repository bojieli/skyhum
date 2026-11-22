"""Box-benchmark scorers (round-5b tables, single-event two-point mode) on synthetic locator outputs."""
import json
import os

import numpy as np
import pytest


def hyp(lat, lon, score, passes=()):
    return dict(lat=lat, lon=lon, score=score, start_utc=1e9, passes=list(passes))


def write(d, name, H):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(dict(hypotheses=H), f)


@pytest.fixture
def box(tmp_path):
    runs = str(tmp_path / "runs")
    cases = [dict(id="p1", lat=-30.0, lon=150.0, cat="pos", cluster="c1", start_true=1e9),
             dict(id="p2", lat=-30.0, lon=150.0, cat="pos", cluster="c2", start_true=1e9),
             dict(id="n1", lat=-31.0, lon=151.0, cat="neg", cluster="c3", start_true=1e9),
             dict(id="h1", lat=-30.0, lon=150.01144, cat="hard", cluster="c1", start_true=1e9)]
    for mode in ("blind", "clk30s"):
        write(f"{runs}/T", f"p1.{mode}", [hyp(-30.005, 150.0, 9.0)])                   # 0.56 km
        write(f"{runs}/T", f"p2.{mode}", [hyp(-30.1, 150.0, 5.0), hyp(-30.0, 150.001, 4.0)])   # 11 km, top-2 hit
        write(f"{runs}/T", f"n1.{mode}", [hyp(-31.2, 151.0, 3.0)])
    write(f"{runs}/T__decoy", "p1.clk30s.c+1865", [hyp(-30.3, 150.3, 2.0)])
    write(f"{runs}/T__decoy", "p2.clk30s.c+1865", [hyp(-30.3, 150.3, 7.0)])
    # single event: the voter flies north along lon 150; the top point is 1.1 km west of that track and the truth
    # 1.1 km east, so the mirror point (across the track) lands on the truth
    p = dict(vote=3.0, ac_lat=-30.0, ac_lon=150.0, ac_hdg=0.0)
    write(f"{runs}/S", "h1.clk5s", [hyp(-30.0, 149.98856, 6.0, [p])])
    write(f"{runs}/S__decoy", "h1.clk5s.c+1865", [hyp(-30.3, 150.3, 1.0)])
    return cases, runs


def test_r5b_tables(box):
    from skyhum.bench import r5b_eval
    cases, runs = box
    md, out = r5b_eval.r5b_tables(cases, runs, ["T"], modes=("blind", "clk30s"), decoy=True, boot=200)
    assert out["blind"]["T"]["n"] == 2 and out["blind"]["T"]["ok"] == 1 and out["blind"]["T"]["top2"] == 2
    assert out["clk30s"]["T"]["tiers"] == [0, 0, 1, 1, 1, 1]
    # decoys: p1 beats its decoy (p = 1/2 > 0.05: never confident with K = 1), p2 does not
    assert "| clk30s | T | 2 | 1 (50%" in md and "Accuracy tiers" in md


def test_single_event_mirror_and_decoys(box):
    from skyhum.bench import r5b_eval
    cases, runs = box
    md, summ, rows = r5b_eval.single_event(cases, runs, "S", pthr=0.5)
    r = rows[0]
    assert r["e1"] == pytest.approx(2.2, abs=0.05) and r["e2"] < 0.05
    assert summ["hard"]["e2"][:3] == [1, 1, 1] and summ["hard"]["e1"][2] == 0
    assert summ["hard"]["confident"] == 1 and summ["hard"]["correct"] == 1


def test_gate_table_counts_confident_answers(box):
    from skyhum.bench import r5b_eval
    cases, runs = box
    for k in range(20):                       # 20 low decoys: p1 reaches p = 1/21 <= 0.05
        write(f"{runs}/T__decoy", f"p1.clk30s.c{k + 2000:+d}", [hyp(-30.3, 150.3, 1.0)])
    md, out = r5b_eval.gate_table(cases, runs, "T", ("clk30s",))
    g = out["clk30s"]["p <= 0.05"]
    assert g["confident"] == 1 and g["correct"] == 1 and g["pos"] == 1
    assert out["clk30s"]["p <= 0.05 & margin >= 8"]["confident"] == 0          # margin 9 - 2 = 7
    assert "1/1 = 1.00" in md


def test_multi_session_sums_maps(tmp_path):
    from skyhum.bench import r5b_eval
    runs = tmp_path / "runs"
    d = runs / "MS"
    d.mkdir(parents=True)
    lats, lons = np.array([-30.0, -30.1]), np.array([150.0, 150.1])
    # hour a votes for the true cell (0, 0) weakly; hour b strongly for a wrong cell alone (1 voter); summed, the true
    # cell wins because only it has >= 2 voters
    for h, sc, nv in (("a", [[3.0, 0.0], [0.0, 0.0]], [[1, 0], [0, 0]]),
                      ("b", [[2.0, 0.0], [0.0, 9.0]], [[1, 0], [0, 1]])):
        np.savez(d / f"{h}.clk30s.grid.npz", score0=np.ravel(sc), n0=np.ravel(nv), lats=lats, lons=lons)
    plan = dict(hours={"a": dict(lat=-30.0, lon=150.0), "b": dict(lat=-30.0, lon=150.0)},
                sets=[dict(site=1, N=2, hours=["a", "b"])])
    md, rows = r5b_eval.multi_session(plan, str(runs), "MS", modes=("clk30s",))
    assert rows[0]["ok"] and rows[0]["top"] == 5.0 and "| clk30s | 2 | 1 | 1 | 100% |" in md
