from abc import ABC, abstractmethod
import numpy as np

class BaseMetric(ABC):
    """
    Интерфейс для метрик оценки качества модели.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя метрики для логирования (например, 'accuracy')."""
        pass

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> float:
        """
        Расчет значения метрики.

        Args:
            y_true: Истинные метки классов.
            y_pred: Предсказанные метки классов (hard predictions).
            y_probs: Вероятности классов (soft predictions). Обязательно для LogLoss/AUC.

        Returns:
            float: Значение метрики.
        """
        pass
