"""Clustering, relabelling and the nearest-neighbour search."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import FEATS, SEED


def standardize(df: pd.DataFrame, cols: list) -> np.ndarray:
    """K-Means measures straight-line distance, so raw percentages would let
    whichever column has the widest range quietly run the whole show."""
    return StandardScaler().fit_transform(df[cols])


def cluster(df: pd.DataFrame, cols: list, k: int):
    """Fit K-Means and renumber the clusters by mean upward mobility.

    K-Means assigns label integers arbitrarily, so without this every figure,
    colour and name would shuffle between runs. Type 0 is always the lowest
    mobility group. Returns (labels, centroids, X).
    """
    X = standardize(df, cols)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    cent = pd.DataFrame(km.cluster_centers_, columns=[FEATS[c] for c in cols])
    key = "Upward mobility" if "MOBILITY" in cols else cent.columns[0]
    order = cent[key].sort_values().index.tolist()
    labels = pd.Series(km.labels_).map({o: i for i, o in enumerate(order)}).values
    return labels, cent.iloc[order].reset_index(drop=True), X


def nearest(X: np.ndarray, i: int, n: int = 5) -> np.ndarray:
    """Row indices of the n counties closest to row i.

    Every feature counts equally, which is what standardizing already implies:
    a county's minority share weighs the same as its unemployment rate. That is
    a choice, not a law. See the bias probe.
    """
    d = np.linalg.norm(X - X[i], axis=1)
    return np.argsort(d)[1:n + 1]
