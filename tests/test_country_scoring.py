"""Country-search scorers on synthetic saved outputs (tests/country_fixture.py)."""
import json
import os

import pytest

from country_fixture import write_runs, write_static


@pytest.fixture
def world(tmp_path):
    st = write_static(str(tmp_path / "static"))
    runs, truth = write_runs(str(tmp_path / "root"))
    return dict(static=st, runs=runs, truth=truth, out=str(tmp_path / "out"))


def units_of(U, st, kind="60", tau=5, N=1):
    return {u["sessions"][0]: u for u in U if u["set"] == st and u["kind"] == kind and u["tau"] == tau and u["N"] == N}


def test_score_top1_confidence_and_pooled(world):
    from skyhum.country import score
    U = score.run(world["runs"], world["truth"], world["static"], world["out"], print_md=False)
    a = units_of(U, "a2o")
    assert [a[k]["top1"] for k in ("a1", "a2", "a3", "a4")] == [True, False, True, False]
    assert a["a1"]["confident"] and a["a1"]["p"] == pytest.approx(1 / 5) and a["a1"]["K"] == 4
    assert not a["a2"]["confident"] and a["a2"]["p"] == pytest.approx(1.0)
    assert not a["a4"]["answered"] and a["a4"]["dist_km"] == float("inf")
    assert a["a1"]["dist_km"] < 2 and a["a1"]["heard_total"] == 2
    w = units_of(U, "webcam")
    assert w["w1"]["top1"] and not w["w2"]["top1"]
    md = open(world["out"] + ".md").read()
    assert "| 60 min | +-5 s | 4 | 2/4 (50 %) |" in md
    assert "largest metro always (Alpha)" in md and "1/2 (50 %)" in md
    saved = json.load(open(world["out"] + ".json"))
    assert len(saved) == len(U)


def test_score_refuses_outputs_that_touched_the_truth(world):
    from skyhum.country import score
    f = os.path.join(world["runs"], "sa1.json")
    r = json.load(open(f))
    r["touched"] = [os.path.realpath(os.path.dirname(world["truth"])) + "/truth.json"]
    json.dump(r, open(f, "w"))
    with pytest.raises(SystemExit, match="BLINDNESS"):
        score.run(world["runs"], world["truth"], world["static"], world["out"], print_md=False)


def test_missing_day_units_are_excluded(world):
    from skyhum.country import score
    f = os.path.join(world["runs"], "sa3.json")
    r = json.load(open(f))
    r["variants"][0]["sessions"] = [dict(missing_days=["2026-01-01"])]
    json.dump(r, open(f, "w"))
    score.run(world["runs"], world["truth"], world["static"], world["out"], print_md=False)
    md = open(world["out"] + ".md").read()
    assert "| 60 min | +-5 s | 3 | 1/3 (33 %) |" in md


def test_prior_and_claims(world):
    pytest.importorskip("timezonefinder")
    from skyhum.country import claims, prior
    U = prior.run(world["runs"], world["truth"], world["static"], world["out"])
    assert os.path.exists(os.path.join(world["out"], "RESULTS_prior.md"))
    u = [x for x in U if x["sess0"] == "a1" and x["kind"] == "60" and x["tau"] == 5][0]
    assert u["pred"]["acoustic"]["top1"] and u["pred"]["acoustic"]["confident"]
    assert all(x["par"]["pop"] for x in U)                      # every unit has leave-one-site-out weights
    V, calls = claims.run(world["runs"], world["truth"], world["static"], world["out"])
    v = [x for x in V if x["sess0"] == "a1" and x["kind"] == "60" and x["tau"] == 5][0]
    assert v["claims"]["metro"]["true"] == pytest.approx(10.0)  # 12 real - max(0, 2 decoy)
    v2 = [x for x in V if x["sess0"] == "a2" and x["kind"] == "60" and x["tau"] == 5][0]
    assert v2["claims"]["metro"]["true"] < 0                    # 3.5 real < 6 decoy at the true region
    assert calls == [] and "Verify vs N" in open(os.path.join(world["out"], "VERIFY_tonal.md")).read()


def test_bench_run_reproduce_country_cli(world, run_cli):
    code, out, err = run_cli("bench", "run", "--reproduce", "country", "--outputs", world["runs"], "--truth",
                             world["truth"], "--static", world["static"], "--json")
    assert code == 0, err
    res = json.loads(out)["result"]
    assert res["family"] == "country" and os.path.exists(res["tables"])


def test_bench_run_webcam_needs_feature_release(run_cli):
    code, _, err = run_cli("bench", "run", "--reproduce", "country", "--set", "webcam")
    assert code == 3 and "webcam feature release" in err


def test_bench_run_needs_truth_with_outputs(world, run_cli):
    code, _, err = run_cli("bench", "run", "--reproduce", "verify", "--outputs", world["runs"])
    assert code == 2 and "--truth" in err


def test_shipped_manifests_are_a2o_links_only():
    from skyhum.bench.reproduce import manifest_path
    for name in ("a2o_country_v1.json", "a2o_box_v1.json"):
        M = json.load(open(manifest_path(name)))
        assert M["items"]
        for it in M["items"]:
            assert it["url"].startswith("https://data.acousticobservatory.org/")
            assert it["licence"] == "CC BY 4.0"
            assert not any(k in it for k in ("audio", "dir", "path"))
    C = json.load(open(manifest_path("a2o_country_v1.json")))
    sess = {it["session"] for it in C["items"]}
    assert len(C["items"]) == 139 and all(set(g["sessions"]) <= sess for g in C["groups"])
