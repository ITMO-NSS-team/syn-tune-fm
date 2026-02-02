from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np

class BaseDataGenerator(ABC):
    """
    Интерфейс для всех методов генерации синтетических данных.
    Обеспечивает взаимозаменяемость методов в пайплайне.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Параметры генератора (epochs, batch_size и т.д.),
                     передаваемые через Hydra.
        """
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseDataGenerator':
        """
        Обучение генератора на реальных данных.
        
        Args:
            X (pd.DataFrame): Матрица признаков.
            y (pd.Series): Целевая переменная.
        """
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Генерация синтетических данных.

        Args:
            n_samples (int): Количество сэмплов для генерации.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_synthetic, y_synthetic)
        """
        pass

    def save(self, path: str):
        """Метод для сохранения состояния генератора (опционально)."""
        pass
    
    def load(self, path: str):
        """Метод для загрузки состояния генератора (опционально)."""
        pass
