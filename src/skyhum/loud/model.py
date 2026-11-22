"""Event features, the classifier wrapper and the calibrated per-event evidence (what a locator consumes).

Ported from the research loud-event channel (model.py). The classifier is the shipped numpy export
(skyhum.detect.loud_model.LoudClassifier, no pickle) and the calibration is models/loud/calibration.json.

Evidence of one event:
  p_air            P(aircraft pass within ~1.5 km | audio)
  q                class posterior over (jet, prop, rotor), given aircraft
  t, sig_t         CPA reception time in the recording (s) and its 1-sigma (envelope, bias-corrected per class)
  ltau, sig_ltau   log of d / v (s) from the envelope width, bias-corrected, and its 1-sigma
  dop              optional high-frequency Doppler evidence: t, sig_t, beta = v / c, sig_beta, ltau, sig_ltau
"""
import json
import math
from importlib import resources

import numpy as np

from skyhum.detect.loud_model import LoudClassifier

CLASSES = ["none", "jet", "prop", "rotor"]
NB = 22                                           # third-octave bands 25 Hz - 3.15 kHz (present at 8 and 16 kHz)


def _g(d, k, default=np.nan):
    v = d.get(k) if d else None
    return default if v is None else v


GATE_DB = 6.0        # mean level exceeds the floor-tracked level by this much in 0.9-3.6 kHz: speech / radio burst


def prep_event(e):
    """Speech / ATC-radio handling (review 2: airport streams carry ATC, band envelopes there are speech).
    An event whose 0.9-3.6 kHz band is gated (bursty) is not trusted in the mid band: its time and width are
    taken from the 50-180 Hz envelope, where radio and speech have no energy; without a low-band peak it is
    dropped (returns None)."""
    if e is None:
        return None
    if e.get("burst_hi", 0.0) < GATE_DB or e.get("src") == "low":
        return e
    if e.get("tc_low") is None or e.get("tau_low") is None or e["x_low"] < 4.0:
        return None
    return dict(e, tc=e["tc_low"], tau=e["tau_low"], gated=True)


def featurize(e):
    """Feature vector of one event. Only bands <= 3.15 kHz are used so that 8 kHz and 16 kHz audio look alike."""
    d = e.get("dop") or {}
    lt = lambda v: np.log(v) if (v is not None and v == v and v > 0) else np.nan  # noqa: E731
    f = [e["x_mid"], e["x_low"], e["x_hi"], e["x_midlow"], e["prom"], lt(e["tau"]), e["fit_pk"], e["fit_rms"],
         lt(_g(e, "tau_r")) - lt(_g(e, "tau_l")),
         _g(e, "wl3"), _g(e, "wr3"), _g(e, "wl6"), _g(e, "wr6"), _g(e, "wl10"), _g(e, "wr10"),
         _g(e, "tc_hi") - _g(e, "tc_low"), _g(e, "tc_hi") - e["tc"], _g(e, "tc_low") - e["tc"],
         lt(_g(e, "tau_hi")) - lt(_g(e, "tau_low")), lt(_g(e, "tau_low")), lt(_g(e, "tau_midlow")), lt(_g(e, "tau_hi")),
         e["rough"], e["burst_low"], e["burst_midlow"], e["burst_mid"], e["burst_hi"], e["rise_max"], e["fall_max"],
         e["modp"], e["modp_med"], e["modf"], e["modh"], e["flat"], e["flat"] - e["flat_bg"], e["crest_lf"],
         e["crest_hf"], e["lev_abs"] - e["lev_med"], 1.0 if e["src"] == "low" else 0.0,
         _g(d, "zs"), _g(d, "zs") - _g(d, "zs_rev"), _g(d, "z1"), _g(d, "z1_best") - _g(d, "z1_rev"),
         _g(d, "flat_margin"), _g(d, "beta"), lt(_g(d, "tau")), _g(d, "nlines"), _g(d, "tc") - e["tc"],
         lt(_g(d, "f0")), _g(d, "tc_spread"), lt(_g(d, "tau_hi")) - lt(_g(d, "tau_lo")),
         lt(_g(d, "tau")) - lt(e["tau"])]
    br = e["band_rel"][:NB]
    bx = e["band_exc"][:NB]
    f += [np.nan if v is None else v for v in br]
    f += [np.nan if v is None else v for v in bx]
    return np.array(f, np.float32)


FEAT_NAMES = (["x_mid", "x_low", "x_hi", "x_midlow", "prom", "ltau", "fit_pk", "fit_rms", "asym", "wl3", "wr3", "wl6",
               "wr6", "wl10", "wr10", "tc_hi-lo", "tc_hi-mid", "tc_lo-mid", "ltau_hi-lo", "ltau_low", "ltau_midlow",
               "ltau_hi", "rough", "burst_low", "burst_midlow", "burst_mid", "burst_hi", "rise_max", "fall_max",
               "modp", "modp_med", "modf", "modh", "flat", "flat-bg", "crest_lf", "crest_hf", "lev_rel", "src_low",
               "dop_zs", "dop_zs-rev", "dop_z1", "dop_z1-rev", "dop_flat_margin", "dop_beta", "dop_ltau", "dop_nlines",
               "dop_tc-env", "dop_lf0", "dop_tc_spread", "dop_ltau_spread", "dop_ltau-env"]
              + ["rel%d" % i for i in range(NB)] + ["exc%d" % i for i in range(NB)])


def calibration_path():
    return str(resources.files("skyhum.models").joinpath("loud/calibration.json"))


class Model:
    """Classifier + calibration. proba(events) -> (n, 4) class probabilities; evidence(events) -> evidence records."""

    def __init__(self, classifier=None, calibration=None):
        self.cal = json.load(open(calibration or calibration_path()))
        self.clf = LoudClassifier(classifier)
        assert self.clf.classes == CLASSES and self.cal["classes"] == CLASSES

    def proba(self, events):
        """Class probabilities (none, jet, prop, rotor); raw events are passed through prep_event, a dropped
        (gated) event gets p(none) = 1."""
        out = np.zeros((len(events), 4))
        out[:, 0] = 1.0
        prepped = [prep_event(e) for e in events]
        idx = [i for i, e in enumerate(prepped) if e is not None]
        if idx:
            pr = self.clf.predict_proba(np.stack([featurize(prepped[i]) for i in idx]))
            out[idx] = 0.0
            for j, k in enumerate(self.clf.classes):
                out[idx, CLASSES.index(k)] = pr[:, j]
        return out

    def evidence(self, events, proba=None):
        """Calibrated evidence records for a list of raw events (same order; gated events are dropped)."""
        P = self.proba(events) if proba is None else proba
        out = []
        for e, p in zip(events, P):
            pe = prep_event(e)
            if pe is not None:
                out.append(evidence_of(pe, p, self.cal))
        return out


def dop_usable(e, cal):
    d = e.get("dop")
    g = cal["dop_gate"]
    if g.get("disabled"):
        return False
    return bool(d and d["zs"] >= g["zs"] and d["zs"] - d["zs_rev"] >= g["margin_rev"]
                and d["flat_margin"] >= g["flat_margin"])


def evidence_of(e, p, cal):
    p_air = float(1.0 - p[0])
    q = np.asarray(p[1:], float)
    q = q / max(q.sum(), 1e-9)
    # class-mixture corrections (the class is uncertain, so the bias is the posterior-weighted mean)
    bt = sum(q[i] * cal["env_t"][c]["bias"] for i, c in enumerate(CLASSES[1:]))
    st = math.sqrt(sum(q[i] * cal["env_t"][c]["sigma"] ** 2 for i, c in enumerate(CLASSES[1:])))
    bl = sum(q[i] * cal["env_ltau"][c]["bias"] for i, c in enumerate(CLASSES[1:]))
    sl = math.sqrt(sum(q[i] * cal["env_ltau"][c]["sigma"] ** 2 for i, c in enumerate(CLASSES[1:])))
    ev = dict(t_pk=e["t_pk"], p_air=p_air, q=dict(zip(CLASSES[1:], [float(v) for v in q])),
              t=float(e["tc"] - bt), sig_t=float(st), ltau=float(np.log(e["tau"]) - bl), sig_ltau=float(sl),
              x_db=e["x_det"], dop=None)
    if dop_usable(e, cal):
        d, c = e["dop"], cal["dop"]
        ev["dop"] = dict(t=float(d["tc"] - c["t_bias"]), sig_t=c["t_sigma"],
                         beta=float(d["beta"] * math.exp(-c["lbeta_bias"])),
                         sig_lbeta=c["lbeta_sigma"], ltau=float(np.log(d["tau"]) - c["ltau_bias"]),
                         sig_ltau=c["ltau_sigma"], zs=d["zs"], f0=d["f0"])
    return ev
