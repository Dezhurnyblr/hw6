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

Крутится 1–2 минуты: EDA, 6 методов кластеризации, аномалии, графики.

## Что за данные

Около 2240 клиентов. Признаки: возраст, доход, образование, семейное положение, траты по категориям (`MntWines`, `MntMeatProducts`…), каналы покупок, отклик на кампании.

**Зачем кластеризация:** разные сегменты клиентов требуют разных маркетинговых предложений. Каждый кластер можно назвать одним словом: «экономные», «премиум», «семейные».

## Что делал (пункты 1–10)

1. Выбор датасета — Customer Personality Analysis
2. EDA — распределения, масштабы, boxplot-выбросы, корреляции
3. Предобработка — производные признаки, удаление констант
4. StandardScaler
5. One-Hot для `Education` и `Marital_Status` (не LabelEncoder)
6. Пропуски `Income` — флаг `Income_missing` + медиана
7. K-Means, Agglomerative, DBSCAN, HDBSCAN, GMM, Spectral
8. Elbow + Silhouette → k=4 (интерпретируемость важнее max silhouette)
9. Аномалии: DBSCAN/HDBSCAN шум, низкий силуэт, Isolation Forest, LOF
10. PCA, t-SNE, UMAP, силуэт-плот, радар, параллельные координаты

## Файлы

```
main.py
hw6.ipynb
requirements.txt
data/marketing_campaign.csv   — скачивается автоматически
modules/
  datasets.py       — загрузка csv
  preprocessing.py  — EDA + prepare
  clustering.py     — методы кластеризации
  evaluation.py     — выбор k, аномалии, профили
  visualization.py  — графики в output/
output/             — картинки
```

## About

Домашняя работа 6 — кластеризация.
