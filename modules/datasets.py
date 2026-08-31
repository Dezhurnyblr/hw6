"""Загрузка marketing_campaign.csv (Customer Personality Analysis)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

MARKETING_CAMPAIGN_URL = (
    "https://raw.githubusercontent.com/johng034/"
    "customer-segmentation/master/marketing_campaign.csv"
)


class CustomerDataset:
    """Customer Personality Analysis — сегментация клиентов."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "marketing_campaign.csv"

    def load(self) -> pd.DataFrame:
        """Читаю csv. Нет файла — качаю."""
        if self._needs_download():
            self._download()
        frame = pd.read_csv(self.path)
        print(
            f"[CustomerDataset] rows={len(frame)}, "
            f"cols={frame.shape[1]}, "
            f"Income mean={frame['Income'].mean(skipna=True):.0f}, "
            f"пропусков Income={frame['Income'].isna().sum()}"
        )
        return frame

    def _needs_download(self) -> bool:
        if not self.path.exists():
            return True
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            first = handle.readline()
        return first.startswith("version https://git-lfs")

    def _download(self) -> None:
        print(f"[CustomerDataset] Скачиваю {MARKETING_CAMPAIGN_URL}")
        try:
            urllib.request.urlretrieve(MARKETING_CAMPAIGN_URL, self.path)
        except OSError as exc:
            if self.path.exists():
                self.path.unlink()
            raise RuntimeError(
                "Не удалось скачать датасет. Положите marketing_campaign.csv "
                f"в {self.data_dir} вручную (Kaggle: imakash3011/"
                "customer-personality-analysis)."
            ) from exc
        print(f"[CustomerDataset] Сохранено: {self.path}")
