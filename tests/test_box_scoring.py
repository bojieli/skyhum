"""Box-benchmark scorers (round-5b tables, single-event two-point mode) on synthetic locator outputs."""
import json
import os

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
