"""Графики кластеризации в output/."""

from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False


class ClusterVisualizer:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        display: bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.display = _running_in_notebook() if display is None else display
        sns.set_theme(style="whitegrid", context="notebook")

    def _show_image(self, path: Path) -> None:
        if not self.display:
            return
        from IPython.display import Image, display

        display(Image(filename=str(path)))

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Сохранено: {path}")
        self._show_image(path)
        return path

    def plot_numeric_distributions(
        self,
        frame: pd.DataFrame,
        num_cols: list[str],
        *,
        filename: str = "eda_distributions.png",
    ) -> Path:
        n_cols_plot = 5
        n_rows = int(np.ceil(len(num_cols) / n_cols_plot))
        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(18, 3 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for i, col in enumerate(num_cols):
            axes[i].hist(frame[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            axes[i].set_title(col, fontsize=9)
        for j in range(len(num_cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Распределения числовых признаков", fontsize=14, y=1.01)
        return self._save(fig, filename)

    def plot_spending_boxplots(
        self,
        frame: pd.DataFrame,
        spending_cols: list[str],
        *,
        filename: str = "eda_spending_boxplots.png",
    ) -> Path:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, col in zip(axes.ravel(), spending_cols):
            sns.boxplot(y=frame[col], ax=ax, color="coral")
            ax.set_title(col)
        fig.suptitle("Выбросы в тратах по категориям", fontsize=14)
        return self._save(fig, filename)

    def plot_correlation_heatmap(
        self,
        frame: pd.DataFrame,
        num_cols: list[str],
        *,
        filename: str = "eda_correlation.png",
    ) -> Path:
        corr = frame[num_cols].corr()
        fig, ax = plt.subplots(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, linewidths=0.5, ax=ax)
        ax.set_title("Корреляционная матрица")
        return self._save(fig, filename)

    def plot_dendrogram(
        self,
        x: np.ndarray,
        *,
        sample_size: int = 200,
        random_state: int = 42,
        filename: str = "hierarchical_dendrogram.png",
    ) -> Path:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
        linked = linkage(x[idx], method="ward")
        fig, ax = plt.subplots(figsize=(14, 5))
        dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
        ax.set_title("Дендрограмма (Ward)")
        ax.set_xlabel("Объект / кластер")
        ax.set_ylabel("Расстояние")
        return self._save(fig, filename)

    def plot_k_search(
        self,
        k_range: list[int],
        inertias: list[float],
        silhouettes: list[float],
        *,
        k_best: int,
        filename: str = "k_selection.png",
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(k_range, inertias, "o-", color="steelblue")
        axes[0].axvline(k_best, color="red", linestyle="--", label=f"k={k_best}")
        axes[0].set_title("Метод локтя (Elbow)")
        axes[0].set_xlabel("k")
        axes[0].legend()

        axes[1].plot(k_range, silhouettes, "o-", color="coral")
        axes[1].axvline(k_best, color="red", linestyle="--", label=f"k={k_best}")
        axes[1].set_title("Silhouette по k")
        axes[1].set_xlabel("k")
        axes[1].legend()
        return self._save(fig, filename)

    def plot_methods_comparison(
        self,
        table: pd.DataFrame,
        *,
        filename: str = "methods_silhouette.png",
    ) -> Path:
        data = table.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(data["method"], data["silhouette"], color="steelblue")
        ax.set_title("Silhouette по методам кластеризации")
        ax.set_ylabel("Silhouette")
        ax.tick_params(axis="x", rotation=15)
        return self._save(fig, filename)

    def plot_anomalies_pca(
        self,
        x: np.ndarray,
        flags: dict[str, np.ndarray],
        *,
        random_state: int = 42,
        filename: str = "anomalies_pca.png",
    ) -> Path:
        coords = PCA(n_components=2, random_state=random_state).fit_transform(x)
        fig, axes = plt.subplots(1, len(flags), figsize=(5 * len(flags), 4.5))
        if len(flags) == 1:
            axes = [axes]
        for ax, (title, flag) in zip(axes, flags.items()):
            ax.scatter(coords[~flag, 0], coords[~flag, 1], s=10, alpha=0.4, c="steelblue", label="норма")
            ax.scatter(coords[flag, 0], coords[flag, 1], s=30, alpha=0.8, c="red", marker="x", label="аномалия")
            ax.set_title(title)
            ax.legend(fontsize=8)
        fig.suptitle("Аномалии в PCA-пространстве (только визуализация)", fontsize=13)
        return self._save(fig, filename)

    def plot_projections(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        k: int,
        random_state: int = 42,
        filename: str = "projections_kmeans.png",
    ) -> Path:
        pca = PCA(n_components=2, random_state=random_state).fit_transform(x)
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(x)
        projections: list[tuple[str, np.ndarray]] = [("PCA", pca), ("t-SNE", tsne)]
        if HAS_UMAP:
            umap_coords = umap.UMAP(n_components=2, random_state=random_state).fit_transform(x)
            projections.append(("UMAP", umap_coords))

        fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 5))
        if len(projections) == 1:
            axes = [axes]
        for ax, (name, coords) in zip(axes, projections):
            ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set2", s=15, alpha=0.7)
            ax.set_title(f"{name} — K-Means (k={k})")
        fig.suptitle("Проекции (только визуализация)", fontsize=13)
        return self._save(fig, filename)

    def plot_silhouette(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        k: int,
        filename: str = "silhouette_plot.png",
    ) -> Path:
        samples = silhouette_samples(x, labels)
        fig, ax = plt.subplots(figsize=(8, 5))
        y_lower = 10
        for cluster in range(k):
            cluster_sil = np.sort(samples[labels == cluster])
            size = cluster_sil.shape[0]
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(cluster), fontsize=12)
            y_lower = y_upper + 10
        ax.axvline(samples.mean(), color="red", linestyle="--", label=f"mean={samples.mean():.3f}")
        ax.set_xlabel("Silhouette")
        ax.set_title("Silhouette по кластерам K-Means")
        ax.legend()
        return self._save(fig, filename)

    def plot_radar_profiles(
        self,
        profile: pd.DataFrame,
        radar_cols: list[str],
        *,
        filename: str = "cluster_radar.png",
    ) -> Path:
        radar = profile[radar_cols].copy()
        radar_norm = (radar - radar.min()) / (radar.max() - radar.min())
        categories = radar_cols
        n = len(categories)
        angles = [i / float(n) * 2 * pi for i in range(n)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(radar_norm)))
        for idx, (cluster, row) in enumerate(radar_norm.iterrows()):
            values = row.tolist() + [row.tolist()[0]]
            ax.plot(angles, values, "o-", linewidth=2, label=f"Кластер {cluster}", color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_title("Профили кластеров (радар)", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        return self._save(fig, filename)

    def plot_parallel_coordinates(
        self,
        processed: pd.DataFrame,
        labels: np.ndarray,
        profile_cols: list[str],
        *,
        filename: str = "cluster_parallel_coords.png",
    ) -> Path:
        pc_df = processed[profile_cols].copy()
        pc_df["Cluster"] = labels.astype(str)
        for col in profile_cols:
            col_min, col_max = pc_df[col].min(), pc_df[col].max()
            pc_df[col] = (pc_df[col] - col_min) / (col_max - col_min)

        fig, ax = plt.subplots(figsize=(12, 5))
        parallel_coordinates(pc_df, "Cluster", colormap=plt.cm.Set2, alpha=0.3, ax=ax)
        ax.set_title("Параллельные координаты (нормализованные признаки)")
        ax.set_ylabel("Нормализованное значение")
        return self._save(fig, filename)

    def plot_3d_pca(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        *,
        k: int,
        random_state: int = 42,
        filename: str = "projections_3d_pca.png",
    ) -> Path:
        pca = PCA(n_components=3, random_state=random_state)
        coords = pca.fit_transform(x)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=labels,
            cmap="Set2",
            s=15,
            alpha=0.7,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title(f"3D PCA — K-Means (k={k})")
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Кластер")
        return self._save(fig, filename)

    def plot_3d_features(
        self,
        processed: pd.DataFrame,
        labels: np.ndarray,
        *,
        x_col: str = "Income",
        y_col: str = "Total_Spending",
        z_col: str = "Age",
        filename: str = "projections_3d_features.png",
    ) -> Path:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            processed[x_col],
            processed[y_col],
            processed[z_col],
            c=labels,
            cmap="Set2",
            s=20,
            alpha=0.7,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D: {x_col} × {y_col} × {z_col} по кластерам")
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Кластер")
        return self._save(fig, filename)

    def plot_k_distance(
        self,
        x: np.ndarray,
        *,
        min_samples: int = 10,
        eps: float | None = None,
        filename: str = "dbscan_k_distance.png",
    ) -> Path:
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(x)
        distances, _ = nbrs.kneighbors(x)
        k_dist = np.sort(distances[:, -1])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(k_dist, color="steelblue", linewidth=1)
        if eps is not None:
            ax.axhline(eps, color="red", linestyle="--", label=f"eps={eps:.3f}")
            ax.legend()
        ax.set_title(f"k-distance graph (k={min_samples}) — выбор eps для DBSCAN")
        ax.set_xlabel("Точки (отсортированы)")
        ax.set_ylabel(f"{min_samples}-е расстояние")
        return self._save(fig, filename)

    def plot_hypothesis_income_deals(
        self,
        frame: pd.DataFrame,
        *,
        filename: str = "hypothesis_income_deals.png",
    ) -> Path:
        clean = frame[["Income", "NumDealsPurchases"]].dropna()
        clean = clean.copy()
        clean["Income_Q"] = pd.qcut(
            clean["Income"], 4, labels=["Q1 (низкий)", "Q2", "Q3", "Q4 (высокий)"]
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.scatterplot(
            data=clean, x="Income", y="NumDealsPurchases", alpha=0.4, ax=axes[0]
        )
        axes[0].set_title("H1/H6: Доход vs покупки по скидкам")
        sns.boxplot(
            data=clean, x="Income_Q", y="NumDealsPurchases", hue="Income_Q", palette="Set2", legend=False, ax=axes[1]
        )
        axes[1].set_title("NumDealsPurchases по квартилям дохода")
        axes[1].tick_params(axis="x", rotation=15)
        return self._save(fig, filename)

    def plot_hypothesis_channels(
        self,
        frame: pd.DataFrame,
        *,
        filename: str = "hypothesis_income_channels.png",
    ) -> Path:
        clean = frame[["Income", "NumWebPurchases", "NumStorePurchases"]].dropna()
        clean = clean.copy()
        clean["Income_Q"] = pd.qcut(clean["Income"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        melted = clean.melt(
            id_vars="Income_Q",
            value_vars=["NumWebPurchases", "NumStorePurchases"],
            var_name="Канал",
            value_name="Покупки",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=melted,
            x="Income_Q",
            y="Покупки",
            hue="Канал",
            palette="muted",
            ax=ax,
        )
        ax.set_title("H2: Канал покупок по квартилям дохода")
        return self._save(fig, filename)

    def plot_hypothesis_family_meat(
        self,
        frame: pd.DataFrame,
        *,
        filename: str = "hypothesis_family_meat.png",
    ) -> Path:
        clean = frame[["Family_Size", "MntMeatProducts", "Kidhome", "Teenhome"]].dropna()
        clean = clean.copy()
        clean["Has_Children"] = (clean["Kidhome"] + clean["Teenhome"] > 0).map(
            {True: "С детьми", False: "Без детей"}
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(
            data=clean, x="Has_Children", y="MntMeatProducts", hue="Has_Children", palette="Set2", legend=False, ax=axes[0]
        )
        axes[0].set_title("H3: Траты на мясо — семьи с/без детей")
        sns.scatterplot(
            data=clean, x="Family_Size", y="MntMeatProducts", hue="Has_Children", ax=axes[1]
        )
        axes[1].set_title("Family_Size vs MntMeatProducts")
        return self._save(fig, filename)

    def plot_outlier_impact(
        self,
        k_range: list[int],
        sil_with: list[float],
        sil_without: list[float],
        *,
        filename: str = "outlier_impact_silhouette.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(k_range, sil_with, "o-", label="С обработкой выбросов (IQR cap)", color="steelblue")
        ax.plot(k_range, sil_without, "s--", label="Без обработки", color="coral")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        ax.set_title("Влияние выбросов на K-Means (StandardScaler, без cap)")
        ax.legend()
        return self._save(fig, filename)
