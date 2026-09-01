"""Предобработка: производные признаки, One-Hot, масштабирование, выбросы, мультиколлинеарность."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

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
# Пары с |r| > 0.8 — оставляем один признак из пары (мультиколлинеарность)
MULTICOLLINEAR_DROP = [
    "MntMeatProducts",  # ↔ Total_Spending, MntWines
    "NumStorePurchases",  # ↔ Total_Purchases
    "NumWebPurchases",  # ↔ Total_Purchases
]
PARTNER_STATUSES = {"Married", "Together"}


class PreparedData(NamedTuple):
    """Сырые и подготовленные данные для кластеризации."""

    raw: pd.DataFrame
    processed: pd.DataFrame
    feature_names: tuple[str, ...]
    x: np.ndarray
    scaler: StandardScaler | RobustScaler
    profile_cols: tuple[str, ...]
    dropped_cols: tuple[str, ...]
    outlier_stats: dict[str, int]
    scaler_name: str


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

    @staticmethod
    def _adults_in_household(marital_status: pd.Series) -> pd.Series:
        """Взрослых в домохозяйстве: 2 при партнёре (Married/Together), иначе 1."""
        return marital_status.apply(
            lambda status: 2 if status in PARTNER_STATUSES else 1
        )

    @staticmethod
    def cap_outliers_iqr(
        frame: pd.DataFrame,
        cols: list[str],
        *,
        factor: float = 1.5,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Winsorization по IQR: обрезаем хвосты, не удаляем строки."""
        df = frame.copy()
        capped: dict[str, int] = {}
        for col in cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            before = df[col].copy()
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped[col] = int((before != df[col]).sum())
        return df, capped

    def prepare(
        self,
        frame: pd.DataFrame,
        *,
        handle_outliers: bool = True,
        drop_multicollinear: bool = True,
        scaler: Literal["standard", "robust"] = "robust",
    ) -> PreparedData:
        """Пропуски, производные признаки, One-Hot, масштабирование.

        Выбросы: IQR-winsorization на тратах и Income (K-Means чувствителен).
        Мультиколлинеарность: удаляем избыточные признаки (|r|>0.8).
        Масштабирование: RobustScaler по умолчанию (медиана/IQR, не усиливает выбросы).
        """
        df = frame.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        df["Income_missing"] = df["Income"].isna().astype(int)
        df["Income"] = df["Income"].fillna(df["Income"].median())

        df["Age"] = 2024 - frame["Year_Birth"]
        customer_dates = pd.to_datetime(frame["Dt_Customer"], dayfirst=True)
        df["Customer_For"] = (customer_dates - customer_dates.min()).dt.days
        df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
        df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)

        adults = self._adults_in_household(frame["Marital_Status"])
        df["Adults_In_Household"] = adults
        df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + adults

        outlier_stats: dict[str, int] = {}
        if handle_outliers:
            outlier_cols = SPENDING_COLS + ["Income", "NumWebVisitsMonth"]
            df, outlier_stats = self.cap_outliers_iqr(df, outlier_cols)

        cat_encoded = pd.get_dummies(df[CAT_COLS], drop_first=True)
        features = df.drop(columns=CAT_COLS + (["Response"] if "Response" in df.columns else []))
        x_df = pd.concat([features, cat_encoded], axis=1)

        dropped: list[str] = []
        if drop_multicollinear:
            for col in MULTICOLLINEAR_DROP:
                if col in x_df.columns:
                    x_df = x_df.drop(columns=[col])
                    dropped.append(col)

        names = tuple(str(col) for col in x_df.columns)

        if scaler == "robust":
            scaler_obj: StandardScaler | RobustScaler = RobustScaler()
            scaler_name = "RobustScaler"
        else:
            scaler_obj = StandardScaler()
            scaler_name = "StandardScaler"

        x_scaled = scaler_obj.fit_transform(x_df)

        print(
            f"[Preprocessor] признаков={len(names)}, строк={len(x_df)}, "
            f"{scaler_name}, outliers_capped={sum(outlier_stats.values())}, "
            f"dropped_multicollinear={dropped}"
        )
        return PreparedData(
            raw=frame,
            processed=df,
            feature_names=names,
            x=x_scaled,
            scaler=scaler_obj,
            profile_cols=tuple(PROFILE_COLS),
            dropped_cols=tuple(dropped),
            outlier_stats=outlier_stats,
            scaler_name=scaler_name,
        )
