"""Shipped models: checksums match SHA256SUMS, and the exported loud classifier reproduces the original model."""
import hashlib
import os
from importlib import resources

import numpy as np

MODELS = str(resources.files("skyhum.models"))
HERE = os.path.dirname(os.path.abspath(__file__))


def test_checksums():
    lines = open(os.path.join(MODELS, "SHA256SUMS")).read().split("\n")
    entries = [ln.split("  ", 1) for ln in lines if ln.strip()]
    assert len(entries) >= 6
    for want, rel in entries:
        with open(os.path.join(MODELS, rel), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == want, rel


def test_no_pickles_shipped():
    for _root, _, files in os.walk(MODELS):
        assert not [f for f in files if f.endswith((".pkl", ".pickle", ".joblib"))]


def test_loud_classifier_matches_original():
    from skyhum.detect.loud_model import LoudClassifier
    z = np.load(os.path.join(HERE, "data", "loud_equiv.npz"))
    m = LoudClassifier()
    assert m.classes == ["none", "jet", "prop", "rotor"] and len(m.feat_names) == 96
    assert np.abs(m.predict_proba(z["X"]) - z["P"]).max() < 1e-9


def test_r5b_tables_available():
    from skyhum.match import locator
    assert locator.models_available()
