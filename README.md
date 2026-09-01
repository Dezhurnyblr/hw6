# Домашняя работа 6

Кластеризация: сегментация клиентов на датасете **Customer Personality Analysis**.

Будет использовать `data/marketing_campaign.csv`. Если файла нет — `main.py` сам скачает.

## Как запускать

Python 3.10+.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Windows (cmd): `.\.venv\Scripts\activate.bat`

Linux/mac: `source .venv/bin/activate`

Графики падают в `output/`. Можно так же пройтись по ноутбуку `hw6.ipynb`.

Крутится 1–2 минуты: EDA, гипотезы, 6 методов кластеризации, аномалии, 2D/3D графики.

## Что за данные

Около 2240 клиентов. Признаки: возраст, доход, образование, семейное положение, траты по категориям (`MntWines`, `MntMeatProducts`…), каналы покупок, отклик на кампании.

**Зачем кластеризация:** разные сегменты клиентов требуют разных маркетинговых предложений. Каждый кластер можно назвать одним словом: «экономные», «премиум», «семейные».

## Что делал (пункты 1–10 + доработки)

1. Выбор датасета — Customer Personality Analysis
2. EDA — распределения, масштабы, boxplot-выбросы, корреляции
3. Предобработка — производные признаки, удаление констант
4. **RobustScaler** (вместо StandardScaler — не усиливает выбросы)
5. One-Hot для `Education` и `Marital_Status` (не LabelEncoder)
6. Пропуски `Income` — флаг `Income_missing` + медиана
7. K-Means, Agglomerative, DBSCAN, HDBSCAN, GMM, Spectral
8. Elbow + Silhouette → **pick_k с порогом 90%**: preferred k, если silhouette ≥ 90% от max
9. Аномалии: DBSCAN/HDBSCAN шум, низкий силуэт, Isolation Forest, LOF
10. PCA, t-SNE, UMAP, **3D PCA и 3D scatter**, силуэт-плот, радар, параллельные координаты

### Доработки (финал)

| Тема | Решение |
|------|---------|
| **Выбросы** | IQR-winsorization на тратах и Income; сравнение silhouette с/без обработки |
| **StandardScaler** | Заменён на RobustScaler (медиана/IQR) |
| **Мультиколлинеарность** | Удаляем `MntMeatProducts`, `NumStorePurchases`, `NumWebPurchases` (|r|>0.8 с агрегатами) |
| **pick_k()** | Если preferred даёт silhouette ≥ 90% от max — берём его, иначе best_sil |
| **DBSCAN eps** | Автовыбор через k-distance graph (не фиксированный 2.5) |
| **GMM** | Отчёт: n_features, n_params, BIC/AIC; `reg_covar=1e-3` для стабильности |
| **Family_Size** | `Kidhome + Teenhome + Adults` (2 при Married/Together, иначе 1) |
| **Гипотезы** | 6 маркетинговых гипотез с тестами (Pearson, Mann-Whitney, Kruskal-Wallis) |

### Гипотезы (H1–H6)

- **H1:** Чем выше доход, тем реже покупают по скидкам
- **H2:** Доход влияет на предпочтение веб-канала
- **H3:** Семьи с детьми тратят больше на мясо
- **H4:** Возраст связан с давностью покупки (Recency)
- **H5:** Образование влияет на траты на вино
- **H6:** Верхний квартиль дохода реже использует скидки, чем нижний

## Файлы

```
main.py
hw6.ipynb
requirements.txt
data/marketing_campaign.csv   — скачивается автоматически
modules/
  datasets.py       — загрузка csv
  preprocessing.py  — EDA + prepare (выбросы, мультиколлинеарность)
  clustering.py     — методы кластеризации (DBSCAN eps auto, GMM info)
  evaluation.py     — выбор k (90% порог), аномалии, профили
  hypotheses.py     — гипотезы и статистические тесты
  visualization.py  — графики в output/ (включая 3D)
output/             — картинки
```

## About

Домашняя работа 6 — кластеризация.
