"""Предобработка: производные признаки, One-Hot, StandardScaler."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SPENDING_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
PURCHASE_COLS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]
PROFILE_COLS = [
    "Age",
    "Income",
    "Total_Spending",
    "Total_Purchases",
    "Recency",
    "Family_Size",
    "NumWebVisitsMonth",
]
DROP_COLS = ["ID", "Z_CostContact", "Z_Revenue", "Year_Birth", "Dt_Customer"]
CAT_COLS = ["Education", "Marital_Status"]


class PreparedData(NamedTuple):
    """Сырые и подготовленные данные для кластеризации."""

    raw: pd.DataFrame
    processed: pd.DataFrame
    feature_names: tuple[str, ...]
    x: np.ndarray
    scaler: StandardScaler
    profile_cols: tuple[str, ...]


class CustomerPreprocessor:
    """EDA-вспомогательные методы + prepare для моделей."""

    def describe_eda(self, frame: pd.DataFrame) -> dict[str, object]:
        num_cols = frame.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in {"ID", "Z_CostContact", "Z_Revenue"}]
        missing = frame.isnull().sum()
        missing = missing[missing > 0]
        const_cols = [c for c in frame.columns if frame[c].nunique() <= 1]
        scale = frame[num_cols].agg(["min", "max", "mean", "std"]).T
        scale["range"] = scale["max"] - scale["min"]
        corr = frame[num_cols].corr()
        high_corr: list[tuple[str, str, float]] = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if abs(value) > 0.8:
                    high_corr.append(
                        (corr.columns[i], corr.columns[j], round(float(value), 3))
                    )
        return {
            "num_cols": num_cols,
            "missing": missing,
            "const_cols": const_cols,
            "scale": scale.sort_values("range", ascending=False),
            "high_corr": high_corr,
        }

    def prepare(self, frame: pd.DataFrame) -> PreparedData:
        """Пропуски, производные признаки, One-Hot, StandardScaler."""
        df = frame.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        df["Income_missing"] = df["Income"].isna().astype(int)
        df["Income"] = df["Income"].fillna(df["Income"].median())

        df["Age"] = 2024 - frame["Year_Birth"]
        customer_dates = pd.to_datetime(frame["Dt_Customer"], dayfirst=True)
        df["Customer_For"] = (customer_dates - customer_dates.min()).dt.days
        df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
        df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
        df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + 1

        cat_encoded = pd.get_dummies(df[CAT_COLS], drop_first=True)
        features = df.drop(columns=CAT_COLS + (["Response"] if "Response" in df.columns else []))
        x_df = pd.concat([features, cat_encoded], axis=1)
        names = tuple(str(col) for col in x_df.columns)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_df)

        print(
            f"[Preprocessor] признаков={len(names)}, "
            f"строк={len(x_df)}, StandardScaler применён"
        )
        return PreparedData(
            raw=frame,
            processed=df,
            feature_names=names,
            x=x_scaled,
            scaler=scaler,
            profile_cols=tuple(PROFILE_COLS),
        )
