"""Loud-event classifier (none / jet / prop / rotor), loaded from plain arrays.

The research model is a scikit-learn HistGradientBoostingClassifier. It ships here as its tree arrays
(`models/loud/classifier.npz`: per tree node value, feature, threshold, missing-goes-left, children, leaf flag) and
is evaluated in numpy, so no pickle is ever loaded. tests/test_models.py checks that the outputs match the original
model on 400 A2O dev events to 1e-9.
"""
from importlib import resources

import numpy as np


class LoudClassifier:
    def __init__(self, path=None):
        path = path or str(resources.files("skyhum.models").joinpath("loud/classifier.npz"))
        z = np.load(path, allow_pickle=False)
        self.n_iter, self.n_class = int(z["n_iter"]), int(z["n_class"])
        self.baseline = z["baseline"].astype(np.float64)
        self.classes = [str(c) for c in z["classes"]]
        self.feat_names = [str(c) for c in z["feat_names"]]
        self.trees = [[{k: z[f"t{i}_{c}_{k}"] for k in ("value", "feature_idx", "num_threshold",
                                                          "missing_go_to_left", "left", "right", "is_leaf")}
                       for c in range(self.n_class)] for i in range(self.n_iter)]

    @staticmethod
    def _tree(t, X):
        node = np.zeros(X.shape[0], np.int64)
        while True:
            leaf = t["is_leaf"][node].astype(bool)
            if leaf.all():
                return t["value"][node]
            f = t["feature_idx"][node]
            x = X[np.arange(X.shape[0]), f]
            go_left = np.where(np.isnan(x), t["missing_go_to_left"][node].astype(bool), x <= t["num_threshold"][node])
            nxt = np.where(go_left, t["left"][node], t["right"][node])
            node = np.where(leaf, node, nxt)

    def decision_function(self, X):
        X = np.atleast_2d(np.asarray(X, np.float64))
        raw = np.tile(self.baseline, (X.shape[0], 1))
        for trees in self.trees:
            for c, t in enumerate(trees):
                raw[:, c] += self._tree(t, X)
        return raw

    def predict_proba(self, X):
        raw = self.decision_function(X)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
