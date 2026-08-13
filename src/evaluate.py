"""Cluster-quality metrics, kept together because they disagree with each other."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src.utils import SEED


def k_sweep(X: np.ndarray, ks) -> dict:
    """Run every cluster-quality test across a range of K.

    Each test asks its own question, which is exactly why they disagree. Returns
    lists keyed by metric name, in the order of `ks`.
    """
    out = {"inertia": [], "silhouette": [], "calinski_harabasz": [], "davies_bouldin": []}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        out["inertia"].append(km.inertia_)
        out["silhouette"].append(silhouette_score(X, km.labels_))
        out["calinski_harabasz"].append(calinski_harabasz_score(X, km.labels_))
        out["davies_bouldin"].append(davies_bouldin_score(X, km.labels_))
    return out


def picks(ks, m: dict) -> dict:
    """What each test would choose if you followed it to the letter.

    They do not agree, which is the point. The elbow is the sharpest bend in
    inertia, found by second difference.
    """
    d2 = np.diff(m["inertia"], 2)
    return {
        "silhouette": ks[int(np.argmax(m["silhouette"]))],
        "calinski_harabasz": ks[int(np.argmax(m["calinski_harabasz"]))],
        "davies_bouldin": ks[int(np.argmin(m["davies_bouldin"]))],
        "elbow": ks[int(np.argmax(d2)) + 1],
    }
