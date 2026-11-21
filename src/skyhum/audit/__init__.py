"""audit: you supply where you were; the report shows what the audio gave away.

Steps in this build:
  1. passes detected blind (classic detector) and ADS-B passes in range of your site while the recording ran
  2. whether verify accepts the true claim, at both operating points
  3. whether a nearby wrong claim (the same shape moved WRONG_KM) also passes
  4. exposure tier: not computed in this build (it needs find-grade ranking around the true site)
  5. with --scrub: the same report on the scrubbed audio
Advice comes from fixed rules below, not from a model.
"""
from skyhum.detect import tonal
from skyhum.io.regions import displaced

WRONG_KM = 15.0


def advice(true_v, wrong_v, n_detected, in_range):
    out = []
    if true_v["verdict"] == "SUPPORTED":
        out.append("The audio fits where you were better than every decoy clock. Treat this place as exposed for "
                   "this recording.")
        out.append("Mute your microphone while you are only listening; gaps with no speech carry the clue most "
                   "clearly.")
        out.append("Before sharing, try `skyhum scrub --method gate` and run this audit again on the output.")
        if wrong_v["verdict"] == "SUPPORTED":
            out.append(f"A claim {WRONG_KM:g} km away also passes, so the clue here is coarse rather than exact.")
    elif true_v["verdict"] == "INCONCLUSIVE":
        out.append("There was too little to test. This says nothing about other recordings from the same place.")
    else:
        out.append("The true place was not corroborated. That is not proof the audio is safe: the default rule "
                   "misses most true claims, and other sessions may add up.")
    if n_detected:
        out.append(f"{n_detected} aircraft-like tonal pass(es) are audible in this recording.")
    if in_range:
        out.append("Repeated sessions from one spot add up: avoid streaming from the same place at a recurring "
                   "time, or scrub every session.")
    return out


def run(verify_fn, detected, where):
    """verify_fn(area) -> verify result dict (any matcher); detected: tonal.detect output."""
    true_v = verify_fn(where)
    wrong = displaced(where, WRONG_KM)
    wrong_v = verify_fn(wrong)
    n_dop = sum(p["kind"] == "doppler-like" for p in detected)
    return dict(
        version=true_v["version"], matcher=true_v["matcher"], benchmarked=true_v.get("benchmarked"),
        where=where.to_json(),
        detected=dict(detector=tonal.NAME, doppler_like=n_dop, steady=len(detected) - n_dop, passes=detected),
        in_range=dict(passes=true_v.get("passes_in_range", 0),
                      note="ADS-B passes within 9 km while the recording ran (not an audibility model)"),
        true_claim=dict(verdict=true_v["verdict"], operating_points=true_v.get("operating_points", {}),
                        margin=true_v.get("margin"), decoy_rank=true_v.get("decoy_rank"),
                        passes=true_v.get("passes", [])),
        wrong_claim=dict(area=wrong.to_json(), distance_km=WRONG_KM, verdict=wrong_v["verdict"],
                         operating_points=wrong_v.get("operating_points", {})),
        exposure_tier=dict(modelled=None, measured=None, note="not computed in this build"),
        advice=advice(true_v, wrong_v, n_dop, true_v.get("passes_in_range", 0)),
        limits=true_v["limits"])
