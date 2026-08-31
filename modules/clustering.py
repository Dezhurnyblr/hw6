"""Методы кластеризации: K-Means → Spectral."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    silhouette: float | None
    n_clusters: int
    n_noise: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ClusterBench:
    """K-Means, Agglomerative, DBSCAN, HDBSCAN, GMM, Spectral."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.results: dict[str, ClusterResult] = {}

    def run_all(self, x: np.ndarray, *, k: int = 4) -> dict[str, ClusterResult]:
        self.results = {
            "K-Means": self._kmeans(x, k),
            "Agglomerative": self._agglomerative(x, k),
            "DBSCAN": self._dbscan(x),
            "HDBSCAN": self._hdbscan(x),
            "GMM": self._gmm(x, k),
            "Spectral": self._spectral(x, k),
        }
        return self.results

    def _silhouette(self, x: np.ndarray, labels: np.ndarray) -> float | None:
        mask = labels != -1
        unique = set(labels[mask]) if mask.any() else set(labels)
        if len(unique) < 2:
            return None
        if mask.any() and not mask.all():
            return float(silhouette_score(x[mask], labels[mask]))
        return float(silhouette_score(x, labels))

    def _kmeans(self, x: np.ndarray, k: int) -> ClusterResult:
        model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(x)
        return ClusterResult(
            name="K-Means",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=k,
            extra={"inertia": float(model.inertia_)},
        )

    def _agglomerative(self, x: np.ndarray, k: int) -> ClusterResult:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x)
        return ClusterResult(
            name="Agglomerative",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=k,
        )

    def _dbscan(self, x: np.ndarray) -> ClusterResult:
        labels = DBSCAN(eps=2.5, min_samples=10).fit_predict(x)
        n_noise = int((labels == -1).sum())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return ClusterResult(
            name="DBSCAN",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=n_clusters,
            n_noise=n_noise,
        )

    def _hdbscan(self, x: np.ndarray) -> ClusterResult:
        labels = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5).fit_predict(x)
        n_noise = int((labels == -1).sum())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return ClusterResult(
            name="HDBSCAN",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=n_clusters,
            n_noise=n_noise,
        )

    def _gmm(self, x: np.ndarray, k: int) -> ClusterResult:
        model = GaussianMixture(n_components=k, random_state=self.random_state)
        labels = model.fit_predict(x)
        proba = model.predict_proba(x)
        return ClusterResult(
            name="GMM",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=k,
            extra={"mean_max_proba": float(proba.max(axis=1).mean())},
        )

    def _spectral(self, x: np.ndarray, k: int) -> ClusterResult:
        model = SpectralClustering(
            n_clusters=k,
            random_state=self.random_state,
            affinity="nearest_neighbors",
            n_neighbors=10,
        )
        labels = model.fit_predict(x)
        return ClusterResult(
            name="Spectral",
            labels=labels,
            silhouette=self._silhouette(x, labels),
            n_clusters=k,
        )

    def summary_table(self) -> "pd.DataFrame":
        import pandas as pd

        rows = []
        for result in self.results.values():
            rows.append(
                {
                    "method": result.name,
                    "silhouette": result.silhouette,
                    "n_clusters": result.n_clusters,
                    "n_noise": result.n_noise,
                }
            )
        return pd.DataFrame(rows)
