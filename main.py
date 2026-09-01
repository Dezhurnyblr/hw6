"""Домашняя работа №6 — кластеризация (Customer Personality Analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules import (  # noqa: E402
    ClusterBench,
    ClusterEvaluator,
    ClusterVisualizer,
    CustomerDataset,
    CustomerPreprocessor,
    HypothesisTester,
)
from modules.preprocessing import SPENDING_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
K_PREFERRED = 4
RANDOM_STATE = 42


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = CustomerDataset(DATA)
    preprocessor = CustomerPreprocessor()
    evaluator = ClusterEvaluator(random_state=RANDOM_STATE)
    bench = ClusterBench(random_state=RANDOM_STATE)
    viz = ClusterVisualizer(OUT)
    hyp_tester = HypothesisTester()

    section("1. Датасет Customer Personality Analysis")
    print(
        "Сегментация клиентов: демография + траты + каналы покупок.\n"
        "Вопрос: можно ли назвать каждую группу одним словом? "
        "Да — «экономные», «премиум», «семейные», «онлайн-активные»."
    )
    raw = dataset.load()
    print(raw.head(3).to_string())

    section("2. EDA — распределения, масштабы, выбросы, корреляции")
    eda = preprocessor.describe_eda(raw)
    print("Пропуски:\n", eda["missing"])
    print("Константные столбцы:", eda["const_cols"])
    print("\nТоп признаков по размаху (range):")
    print(eda["scale"].head(8).to_string())
    if eda["high_corr"]:
        print("\nСильные корреляции (|r| > 0.8):")
        for left, right, value in eda["high_corr"][:10]:
            print(f"  {left} ↔ {right}: {value}")

    num_cols = list(eda["num_cols"])
    viz.plot_numeric_distributions(raw, num_cols)
    viz.plot_spending_boxplots(raw, SPENDING_COLS)
    viz.plot_correlation_heatmap(raw, num_cols)

    section("3–6. Предобработка")
    print(
        "Выбросы: IQR-winsorization (не удаляем строки — K-Means чувствителен).\n"
        "Мультиколлинеарность: убираем избыточные признаки (|r|>0.8).\n"
        "Family_Size = Kidhome + Teenhome + Adults (2 при Married/Together, иначе 1).\n"
        "Масштабирование: RobustScaler (медиана/IQR, не усиливает выбросы как StandardScaler)."
    )
    prepared = preprocessor.prepare(raw)
    print(f"Признаков: {len(prepared.feature_names)}")
    if prepared.dropped_cols:
        print(f"Удалено из-за мультиколлинеарности: {list(prepared.dropped_cols)}")
    if prepared.outlier_stats:
        capped_total = sum(prepared.outlier_stats.values())
        print(f"Обрезано выбросов (IQR cap): {capped_total} значений")

    section("Влияние выбросов на K-Means")
    print("Сравниваю silhouette: с IQR-cap + RobustScaler vs без обработки + StandardScaler.")
    prepared_raw = preprocessor.prepare(
        raw, handle_outliers=False, drop_multicollinear=True, scaler="standard"
    )
    k_search_robust = evaluator.search_k(prepared.x, k_min=2, k_max=10)
    k_search_raw = evaluator.search_k(prepared_raw.x, k_min=2, k_max=10)
    viz.plot_outlier_impact(
        k_search_robust.k_range,
        k_search_robust.silhouettes,
        k_search_raw.silhouettes,
    )

    section("Гипотезы и проверки")
    hyp_results = hyp_tester.run_all(prepared.processed)
    hyp_table = hyp_tester.summary_table(hyp_results)
    print(hyp_table[["id", "hypothesis", "test", "p_value", "significant", "conclusion"]].to_string(index=False))
    viz.plot_hypothesis_income_deals(prepared.processed)
    viz.plot_hypothesis_channels(prepared.processed)
    viz.plot_hypothesis_family_meat(prepared.processed)

    section("8. Выбор числа кластеров (Elbow + Silhouette)")
    k_pick = evaluator.pick_k(k_search_robust, preferred=K_PREFERRED, silhouette_threshold=0.90)
    k_best = k_pick.k
    viz.plot_k_search(
        k_search_robust.k_range,
        k_search_robust.inertias,
        k_search_robust.silhouettes,
        k_best=k_best,
    )

    section("7. Кластеризация: K-Means → Spectral")
    results = bench.run_all(prepared.x, k=k_best)
    summary = bench.summary_table()
    print(summary.to_string(index=False))

    gmm_info = results["GMM"].extra
    print(
        f"\nGMM: признаков={gmm_info['n_features']}, "
        f"параметров≈{gmm_info['n_params']}, "
        f"params/sample={gmm_info['params_per_sample']}, "
        f"BIC={gmm_info['bic']:.0f}, AIC={gmm_info['aic']:.0f}"
    )
    print(
        "GMM чувствителен к выбросам (гауссовы компоненты «тянутся» к хвостам). "
        "reg_covar=1e-3 стабилизирует ковариации."
    )

    dbscan_eps = results["DBSCAN"].extra.get("eps")
    if dbscan_eps:
        print(f"\nDBSCAN eps={dbscan_eps:.3f} — автоматически с k-distance graph (не фиксированный 2.5).")
        viz.plot_k_distance(prepared.x, min_samples=10, eps=dbscan_eps)

    viz.plot_dendrogram(prepared.x, random_state=RANDOM_STATE)
    viz.plot_methods_comparison(summary)

    labels_kmeans = results["K-Means"].labels
    profile = evaluator.cluster_profile(
        prepared.processed,
        labels_kmeans,
        prepared.profile_cols,
    )
    print("\nПрофили кластеров K-Means:")
    print(profile.to_string())

    section("9. Поиск аномалий")
    anomalies = evaluator.detect_anomalies(
        prepared.x,
        labels_kmeans,
        results["DBSCAN"].labels,
        results["HDBSCAN"].labels,
    )
    for name, count in anomalies.counts.items():
        print(f"  {name}: {count}")
    print("\nСколько методов согласны:")
    print(anomalies.overlap["n_methods"].value_counts().sort_index().to_string())

    viz.plot_anomalies_pca(
        prepared.x,
        {
            "DBSCAN шум": results["DBSCAN"].labels == -1,
            "Isolation Forest": anomalies.overlap["isolation_forest"].to_numpy(),
            "LOF": anomalies.overlap["lof"].to_numpy(),
        },
        random_state=RANDOM_STATE,
    )

    section("10. Визуализация")
    print("PCA / t-SNE / UMAP / 3D — только для картинок, кластеризация в полном пространстве.")
    paths = [
        viz.plot_projections(prepared.x, labels_kmeans, k=k_best, random_state=RANDOM_STATE),
        viz.plot_3d_pca(prepared.x, labels_kmeans, k=k_best, random_state=RANDOM_STATE),
        viz.plot_3d_features(prepared.processed, labels_kmeans),
        viz.plot_silhouette(prepared.x, labels_kmeans, k=k_best),
        viz.plot_radar_profiles(profile, list(prepared.profile_cols)),
        viz.plot_parallel_coordinates(
            prepared.processed,
            labels_kmeans,
            list(prepared.profile_cols),
        ),
    ]

    section("Итог")
    print("ДЗ №6: кластеризация, датасет Customer Personality Analysis")
    print(f"Клиентов: {len(raw)}, признаков после prepare: {len(prepared.feature_names)}")
    print(f"Выбранное k: {k_best} ({k_pick.reason})")
    best = results["K-Means"]
    print(f"K-Means silhouette: {best.silhouette:.3f}")
    confirmed = sum(1 for h in hyp_results if h.significant)
    print(f"Гипотез подтверждено: {confirmed}/{len(hyp_results)}")
    print("\nПрофили (кратко):")
    for cluster_id, row in profile.iterrows():
        print(
            f"  Кластер {cluster_id}: n={int(row['size'])}, "
            f"Income={row['Income']:.0f}, "
            f"Family_Size={row['Family_Size']:.1f}, "
            f"Total_Spending={row['Total_Spending']:.0f}"
        )
    print("\nГрафики:")
    for path in sorted(OUT.glob("*.png")):
        print(f"  • {path.name}")


if __name__ == "__main__":
    main()
