from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd

class BaseDataLoader(ABC):
    """
    Абстрактный базовый класс для загрузчиков данных.
    Гарантирует, что данные всегда возвращаются в формате (X_train, y_train, X_test, y_test).
    """

    def __init__(self, target_column: str):
        """
        Args:
            target_column (str): Название целевой колонки (label).
        """
        self.target_column = target_column

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Загружает данные, делает предобработку и разбиение на train/test.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            X_train, y_train, X_test, y_test
        """
        pass

    def _validate_data(self, df: pd.DataFrame):
        """Вспомогательный метод для проверки наличия целевой колонки."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
