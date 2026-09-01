"""Гипотезы о поведении клиентов и их статистическая проверка."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HypothesisResult:
    id: str
    statement: str
    test: str
    statistic: float
    p_value: float
    effect_size: float | None
    conclusion: str
    significant: bool


class HypothesisTester:
    """Проверка маркетинговых гипотез на подготовленных данных."""

    ALPHA = 0.05

    def run_all(self, processed: pd.DataFrame) -> list[HypothesisResult]:
        return [
            self._h1_income_vs_deals(processed),
            self._h2_income_quartile_channel(processed),
            self._h3_family_vs_meat(processed),
            self._h4_age_vs_recency(processed),
            self._h5_education_wine_spending(processed),
            self._h6_high_income_less_deals(processed),
        ]

    def summary_table(self, results: list[HypothesisResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            rows.append(
                {
                    "id": result.id,
                    "hypothesis": result.statement,
                    "test": result.test,
                    "statistic": round(result.statistic, 4),
                    "p_value": round(result.p_value, 6),
                    "effect": round(result.effect_size, 4) if result.effect_size is not None else None,
                    "significant": result.significant,
                    "conclusion": result.conclusion,
                }
            )
        return pd.DataFrame(rows)

    def _h1_income_vs_deals(self, df: pd.DataFrame) -> HypothesisResult:
        """H1: Доход отрицательно коррелирует с покупками по скидкам."""
        clean = df[["Income", "NumDealsPurchases"]].dropna()
        r, p = stats.pearsonr(clean["Income"], clean["NumDealsPurchases"])
        sig = p < self.ALPHA
        direction = "отрицательная" if r < 0 else "положительная"
        return HypothesisResult(
            id="H1",
            statement="Чем выше доход, тем реже покупают по скидкам (NumDealsPurchases)",
            test="Pearson r",
            statistic=float(r),
            p_value=float(p),
            effect_size=float(r),
            significant=sig,
            conclusion=(
                f"Подтверждена: {direction} связь (r={r:.3f}, p={p:.4f})"
                if sig
                else f"Не подтверждена (r={r:.3f}, p={p:.4f})"
            ),
        )

    def _h2_income_quartile_channel(self, df: pd.DataFrame) -> HypothesisResult:
        """H2: Канал покупок зависит от дохода (веб vs магазин)."""
        clean = df[["Income", "NumWebPurchases", "NumStorePurchases"]].dropna()
        clean = clean.copy()
        clean["Income_Q"] = pd.qcut(clean["Income"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        web_share = clean.groupby("Income_Q", observed=True)["NumWebPurchases"].mean()
        store_share = clean.groupby("Income_Q", observed=True)["NumStorePurchases"].mean()
        groups_web = [
            clean.loc[clean["Income_Q"] == q, "NumWebPurchases"].values for q in web_share.index
        ]
        stat, p = stats.kruskal(*groups_web)
        q1_web, q4_web = web_share.iloc[0], web_share.iloc[-1]
        sig = p < self.ALPHA
        return HypothesisResult(
            id="H2",
            statement="Доход влияет на предпочтение веб-канала (NumWebPurchases по квартилям)",
            test="Kruskal-Wallis",
            statistic=float(stat),
            p_value=float(p),
            effect_size=None,
            significant=sig,
            conclusion=(
                f"Подтверждена: Q1 веб={q1_web:.1f}, Q4 веб={q4_web:.1f} (p={p:.4f})"
                if sig
                else f"Не подтверждена (p={p:.4f})"
            ),
        )

    def _h3_family_vs_meat(self, df: pd.DataFrame) -> HypothesisResult:
        """H3: Семьи с детьми тратят больше на мясо."""
        clean = df[["Kidhome", "Teenhome", "MntMeatProducts"]].dropna()
        with_children = clean[clean["Kidhome"] + clean["Teenhome"] > 0]["MntMeatProducts"]
        no_children = clean[clean["Kidhome"] + clean["Teenhome"] == 0]["MntMeatProducts"]
        stat, p = stats.mannwhitneyu(with_children, no_children, alternative="greater")
        effect = (with_children.mean() - no_children.mean()) / clean["MntMeatProducts"].std()
        sig = p < self.ALPHA
        return HypothesisResult(
            id="H3",
            statement="Семьи с детьми тратят больше на мясо (MntMeatProducts)",
            test="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            effect_size=float(effect),
            significant=sig,
            conclusion=(
                f"Подтверждена: с детьми={with_children.mean():.0f}, "
                f"без={no_children.mean():.0f} (d={effect:.2f})"
                if sig
                else f"Не подтверждена (p={p:.4f})"
            ),
        )

    def _h4_age_vs_recency(self, df: pd.DataFrame) -> HypothesisResult:
        """H4: Возраст коррелирует с Recency (давность покупки)."""
        clean = df[["Age", "Recency"]].dropna()
        r, p = stats.spearmanr(clean["Age"], clean["Recency"])
        sig = p < self.ALPHA
        return HypothesisResult(
            id="H4",
            statement="Возраст связан с давностью последней покупки (Recency)",
            test="Spearman ρ",
            statistic=float(r),
            p_value=float(p),
            effect_size=float(r),
            significant=sig,
            conclusion=(
                f"Подтверждена: ρ={r:.3f} (p={p:.4f})"
                if sig
                else f"Слабая/нет связи (ρ={r:.3f}, p={p:.4f})"
            ),
        )

    def _h5_education_wine_spending(self, df: pd.DataFrame) -> HypothesisResult:
        """H5: Уровень образования влияет на траты на вино."""
        if "Education" not in df.columns:
            return HypothesisResult(
                id="H5",
                statement="Образование влияет на MntWines",
                test="—",
                statistic=0.0,
                p_value=1.0,
                effect_size=None,
                significant=False,
                conclusion="Нет столбца Education",
            )
        clean = df[["Education", "MntWines"]].dropna()
        groups = [
            clean.loc[clean["Education"] == edu, "MntWines"].values
            for edu in clean["Education"].unique()
        ]
        stat, p = stats.kruskal(*groups)
        sig = p < self.ALPHA
        top_edu = clean.groupby("Education")["MntWines"].mean().idxmax()
        return HypothesisResult(
            id="H5",
            statement="Уровень образования влияет на траты на вино (MntWines)",
            test="Kruskal-Wallis",
            statistic=float(stat),
            p_value=float(p),
            effect_size=None,
            significant=sig,
            conclusion=(
                f"Подтверждена: макс. MntWines у {top_edu} (p={p:.4f})"
                if sig
                else f"Не подтверждена (p={p:.4f})"
            ),
        )

    def _h6_high_income_less_deals(self, df: pd.DataFrame) -> HypothesisResult:
        """H6: Верхний квартиль дохода реже использует скидки, чем нижний."""
        clean = df[["Income", "NumDealsPurchases"]].dropna()
        q25, q75 = clean["Income"].quantile([0.25, 0.75])
        low = clean[clean["Income"] <= q25]["NumDealsPurchases"]
        high = clean[clean["Income"] >= q75]["NumDealsPurchases"]
        stat, p = stats.mannwhitneyu(high, low, alternative="less")
        effect = (high.mean() - low.mean()) / clean["NumDealsPurchases"].std()
        sig = p < self.ALPHA
        return HypothesisResult(
            id="H6",
            statement="Верхний квартиль дохода реже покупает по скидкам, чем нижний",
            test="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            effect_size=float(effect),
            significant=sig,
            conclusion=(
                f"Подтверждена: high={high.mean():.1f} deals, "
                f"low={low.mean():.1f} deals (p={p:.4f})"
                if sig
                else f"Не подтверждена (p={p:.4f})"
            ),
        )
