"""Метрики, выбор k, профили кластеров, аномалии."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class KSearchResult:
    k_range: list[int]
    inertias: list[float]
    silhouettes: list[float]


@dataclass
class AnomalyReport:
    counts: dict[str, int]
    overlap: pd.DataFrame


class ClusterEvaluator:
    """Elbow, Silhouette, профили, сравнение методов поиска аномалий."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state

    def search_k(
        self,
        x: np.ndarray,
        *,
        k_min: int = 2,
        k_max: int = 10,
    ) -> KSearchResult:
        k_range = list(range(k_min, k_max + 1))
        inertias: list[float] = []
        silhouettes: list[float] = []
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(x)
            inertias.append(float(model.inertia_))
            silhouettes.append(float(silhouette_score(x, labels)))
        return KSearchResult(k_range=k_range, inertias=inertias, silhouettes=silhouettes)

    def pick_k(self, search: KSearchResult, *, preferred: int = 4) -> int:
        best_sil = search.k_range[int(np.argmax(search.silhouettes))]
        print(
            f"[Evaluator] max silhouette при k={best_sil}, "
            f"беру k={preferred} (интерпретируемость)"
        )
        return preferred

    def cluster_profile(
        self,
        processed: pd.DataFrame,
        labels: np.ndarray,
        profile_cols: tuple[str, ...],
    ) -> pd.DataFrame:
        profile = processed.copy()
        profile["Cluster"] = labels
        means = profile.groupby("Cluster")[list(profile_cols)].mean().round(1)
        means["size"] = profile.groupby("Cluster").size()
        return means

    def silhouette_samples(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return silhouette_samples(x, labels)

    def detect_anomalies(
        self,
        x: np.ndarray,
        labels_kmeans: np.ndarray,
        labels_dbscan: np.ndarray,
        labels_hdbscan: np.ndarray,
        *,
        contamination: float = 0.05,
    ) -> AnomalyReport:
        sil = silhouette_samples(x, labels_kmeans)
        low_threshold = float(np.percentile(sil, 5))
        flags = {
            "low_silhouette": sil < low_threshold,
            "dbscan_noise": labels_dbscan == -1,
            "hdbscan_noise": labels_hdbscan == -1,
            "isolation_forest": IsolationForest(
                contamination=contamination,
                random_state=self.random_state,
            ).fit_predict(x)
            == -1,
            "lof": LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
            ).fit_predict(x)
            == -1,
        }
        overlap = pd.DataFrame(flags)
        overlap["n_methods"] = overlap.iloc[:, :5].sum(axis=1)
        counts = {name: int(flag.sum()) for name, flag in flags.items()}
        counts["all_5_methods"] = int((overlap["n_methods"] == 5).sum())
        return AnomalyReport(counts=counts, overlap=overlap)
